"""Aggregate the 3 leave-databases-out folds into one held-out comparison over all 498 questions:
each question is answered by the fold whose training set excludes its database. Compares PlanGate
(optimizer-gated, original design) and adaptive with no-DB and with external baselines (which need
no training) at each budget, with paired bootstrap CIs.

Usage:  python -m experiments.heldout.leave_db_out_summary
Output: results/heldout/leave_db_out_summary.json, results/tables/leave_db_out.tex
"""
import json

import numpy as np

from plangate.evaluation.external import baselines
from plangate.evaluation.heldout import chosen_traces, load
from plangate.evaluation.stats import paired_bootstrap as boot
from plangate.paths import HELDOUT_DIR, TABLES_DIR

rows, _, _ = load("pg_test")
order = {r["question_id"]: i for i, r in enumerate(rows)}
B = baselines(rows)
ext = {k: {"acc": float(np.mean([v[0] for v in vs])), "cost": float(np.mean([v[1] for v in vs]))} for k, vs in B.items()}
cheap = ext["flash"]["cost"]
NICE = {"flash": "cheap model", "rule cascade (DB trigger)": "rule cascade", "agreement cascade (MoT)": "MoT cascade",
        "qwen": "strong model", "pro": "mid model", "self-consistency@2": "self-cons.@2",
        "ensemble-3 exec vote": "ensemble-3"}
out = {"n": len(rows), "budgets": {}}
pols = ("no_db", "fixed_bundle", "optimizer_gated", "adaptive")
lines = [r"\begin{tabular}{@{}lrrrrll@{}}", r"\toprule",
         r"Budget & no-DB & fixed & gated & adaptive & best external & gated $-$ best ext. \\", r"\midrule"]
for b in ("1.25", "1.5", "2.0", "3.0", "4.0"):
    per = {p: {} for p in pols}
    dbms = {p: [] for p in pols}
    costs = {p: [] for p in pols}
    for k in range(3):
        for x in chosen_traces(f"pg_leave_db_out/fold{k}"):
            for p in pols:
                if x["config"] == f"p1.0_B{float(b)}_{p}":
                    per[p][x["q"]] = x["correct"]; dbms[p].append(x["db_ms"]); costs[p].append(x["cost"])
    row = {p: {"acc": float(np.mean(list(per[p].values()))), "n": len(per[p]), "db_ms": float(np.mean(dbms[p])),
               "cost": float(np.mean(costs[p]))} for p in pols if len(per[p]) == len(rows)}
    fit = [(k, v) for k, v in ext.items() if v["cost"] <= float(b) * cheap + 1e-9]
    kb, vb = max(fit, key=lambda kv: (kv[1]["acc"], -kv[1]["cost"]))
    e = [B[kb][order[r["question_id"]]][0] for r in rows]
    for p in row:
        a = [per[p][r["question_id"]] for r in rows]
        row[p]["vs_best_external"] = boot(a, e)
        if p != "no_db" and "no_db" in row:
            row[p]["vs_no_db"] = boot(a, [per["no_db"][r["question_id"]] for r in rows])
    row["best_external"] = [kb, vb]
    out["budgets"][b] = row
    g = row.get("optimizer_gated")
    d = g["vs_best_external"] if g else None
    fmt = lambda x: f"{x:+.1f}".replace("-", "$-$")
    lines.append(f"{b}$\\times$ & " + " & ".join(f"{row[p]['acc']:.3f}" if p in row else "--" for p in pols)
                 + f" & {NICE.get(kb, kb)} {vb['acc']:.3f} & "
                 + (f"{fmt(100*d[0])} [{fmt(100*d[1])}, {fmt(100*d[2])}]" if d else "--") + r" \\")
    print(b, {p: round(row[p]["acc"], 3) for p in pols if p in row}, "best ext", kb, round(vb["acc"], 3),
          "gated-ext", [round(100 * x, 1) for x in d] if d else None,
          "adaptive-noDB", [round(100 * x, 1) for x in row["adaptive"]["vs_no_db"]] if "adaptive" in row else None)
lines += [r"\bottomrule", r"\end{tabular}"]
(HELDOUT_DIR / "leave_db_out_summary.json").write_text(json.dumps(out, indent=1), encoding="utf8")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
(TABLES_DIR / "leave_db_out.tex").write_text("\n".join(lines), encoding="utf8")
