"""Optimizer-priced evidence acquisition inside the cross-fitted planner (PostgreSQL).

For every candidate, PARSE/BIND, EXPLAIN and VALUE_PROBE always run (sub-millisecond to a few ms).
BOUNDED EXEC -- the expensive, heavy-tailed evidence -- runs only if the optimizer's price
C_hat (EXPLAIN total cost) is <= theta. A skipped candidate has no execution evidence and no
result fingerprint; acceptor, selector, V and Q are trained with exactly this masking, so the
planner learns to act on partial evidence.

Usage: python -m experiments.crossfit.priced_planner_pg
Writes results/crossfit/priced_planner_pg.json (stdout: priced_planner_pg.txt).
"""
from __future__ import annotations
import copy, json

import numpy as np

from plangate.paths import CROSSFIT_DIR
from plangate.planning import crossfit as D


def triggers_partial(ev):
    """DB flag for candidates whose bounded execution may have been skipped (no join-condition check)."""
    pb, ex, vp, eb = (ev.get(k) or {} for k in ("PARSE_BIND", "EXPLAIN", "VALUE_PROBE", "EXEC_BOUNDED"))
    return bool((not pb.get("valid")) or vp.get("n_missing")
                or (ev.get("EXEC_BOUNDED") and pb.get("valid") and not eb.get("ok"))
                or (eb.get("ok") and (eb.get("empty") or eb.get("all_null"))))


def price(e):
    ex = e["ev"].get("EXPLAIN") or {}
    return ex.get("total_cost", float("inf")) if ex.get("valid") else float("inf")


def masked_rows(rows, theta):
    """Copy of the action table where EXEC_BOUNDED is skipped for candidates priced above theta."""
    out, n_exec, n_all, saved_ms = [], 0, 0, 0.0
    for r in rows:
        r2 = copy.deepcopy(r)
        for k, e in r2.items():
            if not isinstance(e, dict) or "ev" not in e:
                continue
            n_all += 1
            if price(e) <= theta:
                n_exec += 1
            else:
                saved_ms += e["db_ms"]["EXEC_BOUNDED"]
                e["ev"]["EXEC_BOUNDED"] = None
                e["db_ms"]["EXEC_BOUNDED"] = 0.0
        out.append(r2)
    return out, n_exec / max(n_all, 1), saved_ms


def summarize(res):
    return (float(np.mean([x["correct"] for x in res])), float(np.mean([x["cost"] for x in res])),
            float(np.percentile([x["lat"] for x in res], 95)))


def main():
    rows, items = D.load_action_table("pg")
    n = len(rows)
    fold_of, train_folds = D.question_folds(n)

    prices = np.array([price(e) for r in rows for k, e in r.items() if isinstance(e, dict) and "ev" in e])
    finite = prices[np.isfinite(prices)]
    thetas = [("all (theta=inf)", float("inf"))] + [(f"p{q}", float(np.percentile(finite, q))) for q in (95, 90, 75, 50)]
    exec_ms_all = [e["db_ms"]["EXEC_BOUNDED"] for r in rows for k, e in r.items() if isinstance(e, dict) and "ev" in e]

    report = {}
    print(f"PostgreSQL, n={n}; candidate EXEC time: total {sum(exec_ms_all)/1000:.1f}s, "
          f"p95 {np.percentile(exec_ms_all, 95):.0f} ms, max {max(exec_ms_all):.0f} ms")
    print(f"{'theta':16s} {'exec%':>6} {'DB s saved':>10} | {'acc@1.25x':>9} {'acc@2x':>7} {'acc@3x':>7} {'best acc':>9} "
          f"{'cost@best':>9} | {'DB ms/q':>8} {'p95 DB ms':>9} {'p95 lat s':>9}")
    flash_cost = None
    for name, theta in thetas:
        R, exec_frac, saved = masked_rows(rows, theta)
        P = D.build_planner(R, items, "full", fold_of, train_folds, "pairwise", trigger_fn=triggers_partial)
        pts = []
        for lam in (0.02, 0.05, 0.1, 0.25):
            for tau in np.linspace(0.2, 0.9, 15):
                pts.append(D.run_hybrid(R, P, "full", lam, tau))
        for lam in (0.0, 0.01, 0.02, 0.05, 0.1, 0.25):
            pts.append(D.run_planner(R, P, "full", lam))
        if flash_cost is None:
            flash_cost = float(np.mean([r["c0"]["cost"] for r in rows]))
        summ = [(summarize(res), res) for res in pts]

        def best_under(b):
            c = [s for s in summ if b is None or s[0][1] <= b + 1e-9]
            return max(c, key=lambda s: (s[0][0], -s[0][1])) if c else None

        b125, b2, b3, bb = best_under(1.25 * flash_cost), best_under(2 * flash_cost), best_under(3 * flash_cost), best_under(None)
        # DB time actually spent per query at the best point (all evidence operators, every acquired candidate)
        dbms = []
        for i, r in enumerate(R):
            plan = bb[1][i]["plan"]
            acquired = ["c0"] + ([a for a in plan.split("->")] if plan != "ACCEPT" else [])
            dbms.append(sum(sum(R[i][a]["db_ms"].values()) for a in acquired if R[i].get(a)))
        line = (f"{name:16s} {exec_frac:6.1%} {saved/1000:10.1f} | {b125[0][0]:9.3f} {b2[0][0]:7.3f} {b3[0][0]:7.3f} "
                f"{bb[0][0]:9.3f} {bb[0][1]:9.2f} | {np.mean(dbms):8.1f} {np.percentile(dbms, 95):9.1f} {bb[0][2]/1000:9.1f}")
        print(line, flush=True)
        report[name] = {"theta": theta, "exec_frac": exec_frac, "db_s_saved": saved / 1000,
                        "acc_1.25x": b125[0][0], "acc_2x": b2[0][0], "acc_3x": b3[0][0], "best": bb[0],
                        "db_ms_per_q": float(np.mean(dbms)), "db_ms_p95": float(np.percentile(dbms, 95))}
    (CROSSFIT_DIR / "priced_planner_pg.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
