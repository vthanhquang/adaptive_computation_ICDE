"""How good is selection between disagreeing candidates? (the bottleneck of verified replacement)

Usage: python -m experiments.crossfit.selection_diagnostics
"""
import numpy as np

from plangate.jsonl import read_jsonl
from plangate.paths import CROSSFIT_DIR
from plangate.planning.crossfit import fp, vote

rows = read_jsonl(CROSSFIT_DIR / "action_table_sqlite.jsonl")
for pair in (("c0", "SWITCH_QWEN"), ("c0", "SWITCH_PRO"), ("c0", "RESAMPLE_1"), ("c0", "REPAIR_DB")):
    a, b = pair
    dis = [r for r in rows if fp(r[a]) != fp(r[b])]
    ca = sum(r[a]["correct"] for r in dis); cb = sum(r[b]["correct"] for r in dis)
    both = sum(r[a]["correct"] and r[b]["correct"] for r in dis)
    print(f"{a} vs {b}: disagree on {len(dis)}/{len(rows)}; within disagreements {a} right {ca}, {b} right {cb}, "
          f"neither {len(dis)-ca-cb+both}")
sets = {"c0+qwen": ["c0", "SWITCH_QWEN"], "c0+R1+R2": ["c0", "RESAMPLE_1", "RESAMPLE_2"],
        "c0+pro+qwen": ["c0", "SWITCH_PRO", "SWITCH_QWEN"],
        "all5": ["c0", "RESAMPLE_1", "RESAMPLE_2", "SWITCH_PRO", "SWITCH_QWEN"],
        "all7": ["c0", "VALUE_FIX", "REPAIR_DB", "RESAMPLE_1", "RESAMPLE_2", "SWITCH_PRO", "SWITCH_QWEN"]}
print()
for k, S in sets.items():
    anyc = np.mean([any(r[s]["correct"] for s in S if r.get(s)) for r in rows])
    v = np.mean([r[vote(r, [s for s in S if r.get(s)])]["correct"] for r in rows])
    print(f"{k:12s} oracle-select={anyc:.3f}  exec-vote={v:.3f}")
