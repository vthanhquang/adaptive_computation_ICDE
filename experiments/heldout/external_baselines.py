"""Held-out comparison of PlanGate with external (non-PlanGate) baselines on the TEST splits.

External baselines need no training, so they are computed directly from the replay tables
(plangate.evaluation.external). PlanGate points are the frozen, calibration-chosen operating points
of each run's summary.json (optimizer_gated = the original PlanGate design; adaptive = adaptive
evidence acquisition). Costs: LLM units + 1 unit per DB-second, measured at the scale of the run.
Paired bootstrap of PlanGate vs. the best external baseline that fits each budget.

Usage:  python -m experiments.heldout.external_baselines pg pg_x100 sqlite
Output: results/heldout/external_baselines.json
"""
import json, sys

import numpy as np

from plangate.evaluation.external import baselines
from plangate.evaluation.heldout import chosen_traces, load, run_meta, run_scale, run_summary
from plangate.evaluation.stats import paired_bootstrap
from plangate.paths import HELDOUT_DIR

BUDGETS = ("1.25", "1.5", "2.0", "3.0", "4.0")


def main():
    report = {}
    for d in sys.argv[1:]:
        names = list(run_meta(d)["splits"])
        rows, _, _ = load(names[-1], run_scale(d))
        order = {r["question_id"]: i for i, r in enumerate(rows)}
        B = baselines(rows)
        S = run_summary(d)
        cheap = S["chosen"]["1.0"]["cheap_only"]["cost"]
        traces = {}
        for x in chosen_traces(d):
            if x["config"].startswith("p1.0_"):
                traces.setdefault(x["config"], {})[x["q"]] = x
        rep = {"n": len(rows), "baselines": {k: {"acc": float(np.mean([v[0] for v in vs])),
                                                  "cost": float(np.mean([v[1] for v in vs])),
                                                  "db_ms": float(np.mean([v[2] for v in vs]))} for k, vs in B.items()},
               "budgets": {}}
        print(f"== {d} (n={len(rows)}, cheap cost {cheap:.2f})")
        for k, v in rep["baselines"].items():
            print(f"   {k:26s} EX {v['acc']:.3f} cost {v['cost']:6.2f} db {v['db_ms']:7.0f}ms")
        for b in BUDGETS:
            lim = float(b) * cheap
            fit = [(k, v) for k, v in rep["baselines"].items() if v["cost"] <= lim + 1e-9]
            if not fit:
                continue
            kb, vb = max(fit, key=lambda kv: (kv[1]["acc"], -kv[1]["cost"]))
            row = {"best_external": [kb, vb]}
            for pol in ("optimizer_gated", "adaptive", "fixed_bundle"):
                key = f"p1.0_B{float(b)}_{pol}"
                if key not in traces:
                    continue
                pg = traces[key]
                a = [pg[r["question_id"]]["correct"] for r in rows]
                e = [B[kb][order[r["question_id"]]][0] for r in rows]
                t = S["chosen"]["1.0"][str(float(b))][pol]["test"]
                row[pol] = {"acc": t["acc"], "cost": t["cost"], "db_ms": t["db_ms"],
                            "vs_best_external": paired_bootstrap(a, e)}
            rep["budgets"][b] = row
            line = f"  {b}x best external: {kb} {vb['acc']:.3f}@{vb['cost']:.2f}"
            for pol in ("optimizer_gated", "adaptive"):
                if pol in row:
                    x = row[pol]; dlt = x["vs_best_external"]
                    line += f" | {pol} {x['acc']:.3f}@{x['cost']:.2f} {100*dlt[0]:+.1f} [{100*dlt[1]:+.1f},{100*dlt[2]:+.1f}]"
            print(line)
        report[d] = rep
    (HELDOUT_DIR / "external_baselines.json").write_text(json.dumps(report, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
