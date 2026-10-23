"""Action-level analysis of the PostgreSQL action table (BIRD mini-dev, 500 entries).

  P1  analysis A of analyze_actions.py on PostgreSQL (incl. real optimizer features)
  P2  Can the optimizer price evidence actions? EXPLAIN estimate vs measured execution time
  P3  Price-aware evidence acquisition: execute a candidate only when the optimizer says it is cheap
Usage: python -m experiments.crossfit.analyze_actions_pg
"""
from __future__ import annotations
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.crossfit import analyze_actions as AA
from plangate.paths import CROSSFIT_DIR
from plangate.planning.crossfit import load_action_table


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows, items = load_action_table("pg")
    n = len(rows)
    print(f"PostgreSQL arm: {n} mini-dev questions; c0=flash acc={np.mean([r['c0']['correct'] for r in rows]):.3f}; "
          + ", ".join(f"{a}={np.mean([r[a]['correct'] for r in rows]):.3f}" for a in ("SWITCH_PRO", "SWITCH_QWEN")))
    rep = {"oracle": AA.oracle_actions(rows)}
    table, y, Pb, P_all = AA.exp_a(rows, items)
    rep["exp_a"] = table

    # ---- P2: does the optimizer price execution?
    est, ms = [], []
    expl_ms = []
    for r in rows:
        for k, e in r.items():
            if not isinstance(e, dict) or "ev" not in e:
                continue
            ex, eb = e["ev"]["EXPLAIN"], e["ev"]["EXEC_BOUNDED"]
            expl_ms.append(e["db_ms"]["EXPLAIN"])
            if ex.get("valid") and (eb.get("ok") or eb.get("timed_out")):
                est.append(ex["total_cost"]); ms.append(max(e["db_ms"]["EXEC_BOUNDED"], 0.01))
    est, ms = np.array(est), np.array(ms)
    print(f"\n== P2: optimizer as the price of evidence  ({len(ms)} executed candidates)")
    print(f"  EXPLAIN time: median {np.median(expl_ms):.2f} ms, p95 {np.percentile(expl_ms, 95):.2f} ms")
    print(f"  execution time: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms, max {ms.max():.0f} ms; "
          f"top 5% of candidates = {np.sort(ms)[-max(1, len(ms)//20):].sum()/ms.sum():.0%} of all execution time")
    print(f"  Spearman(est. cost, exec time) = {spearman(np.log1p(est), np.log(ms)):.3f}")
    p2 = {"spearman": spearman(np.log1p(est), np.log(ms))}
    for th in (100, 500, 1000):
        lab = (ms > th).astype(int)
        if 0 < lab.sum() < len(lab):
            a = roc_auc_score(lab, est)
            print(f"  AUROC(est. cost -> exec > {th} ms) = {a:.3f}   (positives {lab.mean():.1%})")
            p2[f"auroc_{th}"] = a
    rep["p2"] = p2

    # ---- P3: price-aware evidence acquisition for the first candidate
    # Acceptor with all probes vs without execution; execute only if est. cost <= theta.
    base = [AA.feats_base(r, items) for r in rows]
    with_exec = [{**b, **AA.feats_probes(r, AA.PROBES)} for b, r in zip(base, rows)]
    no_exec = [{**b, **AA.feats_probes(r, ["PARSE_BIND", "VALUE_PROBE", "EXPLAIN"])} for b, r in zip(base, rows)]
    P_with = AA.oof(AA.mat(with_exec), y); P_without = AA.oof(AA.mat(no_exec), y)
    c_est = np.array([r["c0"]["ev"]["EXPLAIN"].get("total_cost", 0.0) if r["c0"]["ev"]["EXPLAIN"].get("valid") else 0.0
                      for r in rows])
    cheap_ms = np.array([sum(r["c0"]["db_ms"][o] for o in ("PARSE_BIND", "VALUE_PROBE", "EXPLAIN")) for r in rows])
    exec_ms = np.array([r["c0"]["db_ms"]["EXEC_BOUNDED"] for r in rows])
    print("\n== P3: price-aware evidence acquisition on the first candidate")
    print(f"  {'policy':34s} {'AUROC':>6} {'exec%':>6} {'DB ms/q':>8} {'p95 DB ms':>10} {'max DB ms':>10}")
    rep["p3"] = {}
    pols = [("no execution (cheap probes only)", np.zeros(n, bool)), ("always execute", np.ones(n, bool))]
    for q in (50, 75, 90, 95):
        th = np.percentile(c_est, q)
        pols.append((f"execute if est.cost <= p{q} ({th:.0f})", c_est <= th))
    rng = np.random.default_rng(0)
    for q in (75, 90):
        m = np.zeros(n, bool); m[rng.choice(n, int(n * q / 100), replace=False)] = True
        pols.append((f"random {q}% execute (control)", m))
    for name, m in pols:
        score = np.where(m, P_with, P_without)
        dbms = cheap_ms + np.where(m, exec_ms, 0)
        auc = roc_auc_score(y, score)
        print(f"  {name:34s} {auc:6.3f} {m.mean():6.1%} {dbms.mean():8.1f} {np.percentile(dbms,95):10.1f} {dbms.max():10.0f}")
        rep["p3"][name] = {"auroc": auc, "exec_frac": float(m.mean()), "db_ms": float(dbms.mean()),
                           "p95": float(np.percentile(dbms, 95)), "max": float(dbms.max())}
    (CROSSFIT_DIR / "action_report_pg.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
