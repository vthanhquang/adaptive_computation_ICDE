"""Statistics shared by the experiments."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a, b, n=5000, seed=0):
    """Mean difference a - b of paired per-question outcomes and its 95% percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
