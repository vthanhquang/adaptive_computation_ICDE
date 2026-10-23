"""How well does the pre-generation EllieSQL router decide, compared to post-generation DB evidence?

Usage: python -m experiments.baselines.elliesql_router_diagnostics
"""
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from plangate.jsonl import read_jsonl
from plangate.paths import BASELINES_DIR

rows = read_jsonl(BASELINES_DIR / "elliesql.jsonl")
L = ["G_B", "G_M", "G_A"]
label = [next((l for l in L if r[l]["correct"]), "none") for r in rows]
print("oracle (waterfall) label distribution:", dict(Counter(label)))
print("router distribution:", dict(Counter(r["route"] for r in rows)))
exact = np.mean([r["route"] == l for r, l in zip(rows, label) if l != "none"])
print(f"router agrees with oracle label (solvable questions): {exact:.3f}")
for lvl in L:
    sub = [r for r in rows if r["route"] == lvl]
    print(f"  routed to {lvl}: n={len(sub):3d}  acc of chosen pipeline={np.mean([r[lvl]['correct'] for r in sub]):.3f}  "
          f"acc if G_A instead={np.mean([r['G_A']['correct'] for r in sub]):.3f}")
# ranking quality: does P(advanced) predict that the basic pipeline fails?
y = np.array([int(not r["G_B"]["correct"]) for r in rows])
p_hard = np.array([r["route_probs"][1] + r["route_probs"][2] for r in rows])
print(f"AUROC of router P(not basic) for 'basic pipeline fails': {roc_auc_score(y, p_hard):.3f}")
