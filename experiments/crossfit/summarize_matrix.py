"""Per-config accuracy / cost summary of a generation matrix.

Usage: python -m experiments.crossfit.summarize_matrix results/crossfit/generation_matrix.jsonl
"""
import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf8")]
by = collections.defaultdict(list)
for r in rows:
    by[r["config"]].append(r)
print(f"{'config':32s} {'n':>4} {'acc':>6} {'cost':>6} {'lat_s':>6} {'p_tok':>6} {'c_tok':>6} err empty")
for k, v in by.items():
    n = len(v)
    print(f"{k:32s} {n:4d} {sum(r['correct'] for r in v)/n:6.3f} {sum(r['cost'] for r in v)/n:6.2f} "
          f"{sum(r['latency_ms'] for r in v)/n/1000:6.1f} {sum(r['usage']['prompt_tokens'] for r in v)/n:6.0f} "
          f"{sum(r['usage']['completion_tokens'] for r in v)/n:6.0f} {sum(1 for r in v if r.get('error')):3d} "
          f"{sum(1 for r in v if not r['sql']):5d}")

q = collections.defaultdict(dict)
for r in rows:
    q[r["question_id"]][r["config"]] = r
complete = {k: v for k, v in q.items() if len(v) == len(by)}
n = len(complete)
print(f"\nquestions with all configs: {n}")
print("any-config correct (oracle acc):", round(sum(any(c['correct'] for c in v.values()) for v in complete.values()) / n, 3))
print("all-config correct:", round(sum(all(c['correct'] for c in v.values()) for v in complete.values()) / n, 3))
