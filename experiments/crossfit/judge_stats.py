"""Paired bootstrap CIs for judge accuracy by evidence condition (judge study).

Usage: python -m experiments.crossfit.judge_stats   (stdout: results/crossfit/judge_stats.txt)
"""
import numpy as np

from plangate.jsonl import read_jsonl
from plangate.paths import CROSSFIT_DIR

rng = np.random.default_rng(0)
for tag in ("deepseek-v4-flash", "deepseek-v4-pro"):
    res = read_jsonl(CROSSFIT_DIR / f"judge_{tag.replace('.', '_')}.jsonl")
    dec = [o for o in res if o["c0_correct"] != o["x_correct"]]
    truth = ["c0" if o["c0_correct"] else o["other"] for o in dec]
    # average of both orders: each order is an independent single-order judgement
    acc = {m: np.array([((o[m]["pick1"] == t) + (o[m]["pick2"] == t)) / 2 for o, t in zip(dec, truth)])
           for m in ("sql", "results", "diff")}
    print(f"judge={tag}  n={len(dec)} decisive pairs (accuracy averaged over both presentation orders)")
    for m, a in acc.items():
        b = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(4000)]
        print(f"  {m:8s} {a.mean():.3f}  95% CI [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
    for x, y in (("results", "sql"), ("diff", "sql"), ("diff", "results")):
        d = acc[x] - acc[y]
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print(f"  {x}-{y}: {d.mean():+.3f}  CI [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]")
