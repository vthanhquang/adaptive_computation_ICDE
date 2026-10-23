"""Calibration of the state-value estimator V(s) at the moment the adaptive policy accepts (TEST traces).
The value-of-information argument assumes calibrated V and Q; this measures how close the runs are.

Usage:  python -m experiments.heldout.calibration_check pg pg_x100 sqlite
Output: results/heldout/calibration.json
"""
import json, sys

import numpy as np

from plangate.evaluation.heldout import chosen_traces
from plangate.paths import HELDOUT_DIR

out = {}
for d in sys.argv[1:]:
    v, y = [], []
    for x in chosen_traces(d):
        if not (x["config"].startswith("p1.0_") and x["config"].endswith("_adaptive")):
            continue
        acc = [t for t in x["trace"] if t["action"] == "ACCEPT"]
        if acc:
            v.append(acc[-1]["V"]); y.append(x["correct"])
    v, y = np.array(v), np.array(y)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(v, bins) - 1, 0, 9)
    ece = sum(abs(v[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(10) if (idx == b).any()) / len(v)
    rel = [(float(v[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())) for b in range(10) if (idx == b).any()]
    out[d] = {"n": int(len(v)), "ece": float(ece), "mean_V": float(v.mean()), "accuracy": float(y.mean()),
              "brier": float(np.mean((v - y) ** 2)), "reliability": rel}
    print(f"{d}: n={len(v)} ECE={ece:.3f} mean V={v.mean():.3f} acc={y.mean():.3f} Brier={np.mean((v-y)**2):.3f}")
(HELDOUT_DIR / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf8")
