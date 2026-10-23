"""Run the SQLens re-implementation on top of the first candidate (flash) for the 300 SQLite questions.

Usage: python -m experiments.baselines.run_sqlens [model ...] [--separate]
       (default models: deepseek-v4-flash deepseek-v4-pro; --separate: one prompt per LLM signal)
Merges policies "SQLens (<model>)" into results/baselines/published_methods_sqlite.json.
"""
from __future__ import annotations
import concurrent.futures as cf, json, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from plangate.baselines import sqlens as SL
from plangate.engines.sqlite import ex_correct
from plangate.paths import BASELINES_DIR
from plangate.planning.crossfit import load_action_table


def main():
    separate = "--separate" in sys.argv
    models = [a for a in sys.argv[1:] if not a.startswith("--")] or ["deepseek-v4-flash", "deepseek-v4-pro"]
    rows, items = load_action_table("sqlite")
    y_err = np.array([int(not r["c0"]["correct"]) for r in rows])

    # DB signals on the first candidate (shared by both variants)
    def dbs(r):
        return SL.db_signals(r["db_id"], r["c0"]["sql"])
    with cf.ThreadPoolExecutor(8) as ex:
        DB = list(ex.map(dbs, rows))
    fire = {s: np.mean([d[0][s] for d in DB]) for s in SL.DB_SIGNALS}
    print("DB signal firing rates:", {k: round(v, 3) for k, v in fire.items()})
    print(f"DB signal time: median {np.median([d[2] for d in DB]):.0f} ms, p95 {np.percentile([d[2] for d in DB], 95):.0f} ms")

    pb = BASELINES_DIR / "published_methods_sqlite.json"
    store = json.loads(pb.read_text(encoding="utf8"))
    pos = {r["question_id"]: i for i, r in enumerate(rows)}
    report = {}
    for model in models:
        def llm(r):
            fn = SL.llm_signals_separate if separate else SL.llm_signals
            return fn(items[r["question_id"]], r["c0"]["sql"], model)
        with cf.ThreadPoolExecutor(16) as ex:
            LS = list(ex.map(llm, rows))
        L = np.array([[DB[i][0][s] for s in SL.DB_SIGNALS] + [LS[i][0][s] for s in SL.LLM_SIGNALS]
                      for i in range(len(rows))], dtype=float)
        lm = SL.LabelModel().fit(L)
        soft = lm.posterior(L)
        # downstream classifier trained on the label model's soft labels (no gold labels)
        X = np.vstack([L, L]); yy = np.r_[np.ones(len(L)), np.zeros(len(L))]; w = np.r_[soft, 1 - soft]
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, yy, sample_weight=w)
        p_err = clf.predict_proba(L)[:, 1]
        auc = roc_auc_score(y_err, p_err)
        flag = p_err >= 0.5
        prec = (y_err[flag].mean() if flag.any() else float("nan"))
        print(f"\n[{model}] detector AUROC={auc:.3f} | flagged {flag.mean():.1%} | precision {prec:.3f} | "
              f"recall {(flag & (y_err == 1)).sum() / y_err.sum():.3f}")
        print("  per-signal AUROC vs gold error:",
              {s: round(roc_auc_score(y_err, L[:, j]), 3) for j, s in enumerate(SL.SIGNALS) if 0 < L[:, j].sum() < len(L)})

        def fix(i):
            r = rows[i]; it = items[r["question_id"]]
            base_cost = r["c0"]["cost"] + LS[i][2] + DB[i][2] / 1000
            base_lat = r["c0"]["llm_ms"] + DB[i][2] + LS[i][3]
            if not flag[i]:
                return {"correct": int(r["c0"]["correct"]), "cost": base_cost, "lat": base_lat, "n_llm": 2,
                        "changed": 0}
            sig = {**DB[i][0], **LS[i][0]}
            final, c, l, n, dbms = SL.correct(it, r["c0"]["sql"], sig, DB[i][1], LS[i][1], model, SL.db_signals)
            ok = int(ex_correct(it, final)) if final != r["c0"]["sql"] else int(r["c0"]["correct"])
            return {"correct": ok, "cost": base_cost + c + dbms / 1000, "lat": base_lat + l, "n_llm": 2 + n,
                    "changed": int(final != r["c0"]["sql"])}
        with cf.ThreadPoolExecutor(16) as ex:
            res = list(ex.map(fix, range(len(rows))))
        acc = np.mean([x["correct"] for x in res])
        print(f"  SQLens final EX={acc:.3f} (base flash {1 - y_err.mean():.3f}) | cost {np.mean([x['cost'] for x in res]):.2f} "
              f"| p95 {np.percentile([x['lat'] for x in res], 95)/1000:.1f}s | changed {np.mean([x['changed'] for x in res]):.1%}")
        name = f"SQLens{' separate-prompts' if separate else ''} ({model.replace('deepseek-v4-', '')})"
        store["policies"][name] = [res[pos[q]] for q in store["question_ids"]]
        report[name] = {"auroc": auc, "flag_rate": float(flag.mean()), "precision": float(prec), "acc": acc}
    pb.write_text(json.dumps(store, default=float), encoding="utf8")
    (BASELINES_DIR / "sqlens_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
