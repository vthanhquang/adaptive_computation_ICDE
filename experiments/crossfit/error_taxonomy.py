"""Taxonomy of wrong candidates: how does the predicted result differ from gold?

Usage: python -m experiments.crossfit.error_taxonomy results/crossfit/generation_matrix.jsonl flash_direct
"""
import sys
from collections import Counter

from plangate.benchmark import load_dev
from plangate.engines.sqlite import connect, run_bounded
from plangate.jsonl import iter_jsonl

matrix, config = sys.argv[1], sys.argv[2]
items = {x["question_id"]: x for x in load_dev()}
rows = [r for r in iter_jsonl(matrix) if r["config"] == config]
cats = Counter(); examples = {}
for r in rows:
    if r["correct"]:
        continue
    it = items[r["question_id"]]
    conn = connect(it["db_id"])
    g, ge, _, _ = run_bounded(conn, it["SQL"], 30)
    p, pe, _, _ = run_bounded(conn, r["sql"], 30)
    conn.close()
    if pe:
        c = "exec_error"
    elif not p:
        c = "empty_result"
    elif len(p[0]) != len(g[0]) if g else False:
        c = "column_count_mismatch" if set(map(lambda x: x[:len(g[0])], p)) != set(g) else "extra_columns_only"
        if len(p[0]) > len(g[0]) and all(any(gv in pr for gv in gr) for gr, pr in zip(sorted(g, key=repr), sorted(p, key=repr))):
            c = "extra_columns_only"
    elif len(set(p)) != len(set(g)):
        c = "row_count_mismatch"
    else:
        c = "same_shape_different_values"
    cats[c] += 1
    examples.setdefault(c, []).append((it["question"], r["sql"][:160], it["SQL"][:160]))
n = len(rows)
print(f"{config}: {n} answers, {sum(cats.values())} wrong")
for c, k in cats.most_common():
    print(f"  {c:30s} {k:4d}  ({k/n:.1%} of all)")
for c in cats:
    print(f"\n--- {c}")
    for q, s, g in examples[c][:2]:
        print(f"Q: {q}\n  pred: {s}\n  gold: {g}")
