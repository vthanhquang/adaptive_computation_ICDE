"""LLM judge study: selecting between disagreeing candidates with and without execution evidence.

For every question and pair (c0, X) whose execution results differ, a cheap judge (flash)
picks A or B under three evidence conditions:
  sql      : question + schema + the two SQL strings
  results  : + execution previews of both results
  diff     : + differential execution probe (A EXCEPT B, B EXCEPT A, counts, examples)
Each pair is judged in both orders (position-bias control). A pick counts only if both
orders agree; otherwise the pair is a tie.

Usage:  python -m experiments.crossfit.judge [judge_model]
Output: results/crossfit/judge_<model>.jsonl
"""
from __future__ import annotations
import concurrent.futures as cf, json, re, sys

import numpy as np

from plangate.benchmark import load_dev
from plangate.engines.sqlite import schema_with_samples
from plangate.evidence.diff_probe import diff_probe, render
from plangate.jsonl import read_jsonl
from plangate.llm.generation import cached_chat, cost_units, SYSTEM
from plangate.paths import CROSSFIT_DIR
from plangate.planning.crossfit import fp

PAIRS = ["SWITCH_QWEN", "SWITCH_PRO", "RESAMPLE_1", "REPAIR_DB"]
MODES = ["sql", "results", "diff"]

JUDGE = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

Two candidate SQL queries were written for this question. They return DIFFERENT results.

SQL A:
```sql
{a}
```

SQL B:
```sql
{b}
```
{ev}
Which query answers the question correctly? Consider the exact entities, filters, aggregation and the columns the question asks for.
Think briefly, then end with a final line of the form "Answer: A" or "Answer: B"."""


def ask(model, item, a, b, ev_text):
    ev = f"\nExecution evidence from the database:\n{ev_text}\n" if ev_text else ""
    body = JUDGE.format(schema=schema_with_samples(item["db_id"]), question=item["question"],
                        evidence=item.get("evidence") or "none", a=a, b=b, ev=ev)
    d = cached_chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}])
    m = re.findall(r"Answer:\s*\**\s*([AB])\b", d["text"] or "")
    return (m[-1] if m else None), cost_units(model, d["usage"]), d["latency_ms"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"
    items = {x["question_id"]: x for x in load_dev()}
    rows = read_jsonl(CROSSFIT_DIR / "action_table_sqlite.jsonl")
    jobs = []
    for r in rows:
        for x in PAIRS:
            if fp(r["c0"]) != fp(r[x]):
                jobs.append((r, x))

    def run(job):
        r, x = job
        it = items[r["question_id"]]
        a, b = r["c0"]["sql"], r[x]["sql"]
        p_ab, ms_ab = diff_probe(it["db_id"], a, b)
        p_ba, ms_ba = diff_probe(it["db_id"], b, a)
        out = {"question_id": r["question_id"], "other": x, "c0_correct": r["c0"]["correct"],
               "x_correct": r[x]["correct"], "diff_ms": ms_ab, "diff_payload": p_ab}
        for mode in MODES:
            ev1 = "" if mode == "sql" else render(p_ab, mode)
            ev2 = "" if mode == "sql" else render(p_ba, mode)
            j1, c1, l1 = ask(model, it, a, b, ev1)          # A = c0
            j2, c2, l2 = ask(model, it, b, a, ev2)          # A = x
            pick1 = {"A": "c0", "B": x}.get(j1); pick2 = {"A": x, "B": "c0"}.get(j2)
            out[mode] = {"pick1": pick1, "pick2": pick2, "cost": c1 + c2, "lat": max(l1, l2),
                         "cost_single": c1, "lat_single": l1}
        return out

    res = []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, o in enumerate(ex.map(run, jobs)):
            res.append(o)
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(jobs)}", flush=True)
    tag = model.replace(".", "_")
    (CROSSFIT_DIR / f"judge_{tag}.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False, default=str)
                                                                   for o in res), encoding="utf8")
    analyze(res, model)


def analyze(res, model):
    print(f"\n== LLM judge: pairwise selection with judge={model}; {len(res)} disagreeing pairs")
    decisive = [o for o in res if o["c0_correct"] != o["x_correct"]]
    print(f"   pairs where exactly one is correct: {len(decisive)} (only these can be judged right/wrong)")
    print(f"  {'evidence':8s} {'single-order acc':>17} {'consistent':>11} {'acc|consistent':>15} {'acc (tie->prior)':>17}")
    for mode in MODES:
        single = np.mean([o[mode]["pick1"] == ("c0" if o["c0_correct"] else o["other"]) for o in decisive])
        cons = [o for o in decisive if o[mode]["pick1"] and o[mode]["pick1"] == o[mode]["pick2"]]
        acc_c = np.mean([o[mode]["pick1"] == ("c0" if o["c0_correct"] else o["other"]) for o in cons]) if cons else 0
        # tie -> prior: keep stronger model if switching to qwen/pro, else keep c0
        def final(o):
            p = o[mode]["pick1"] if o[mode]["pick1"] == o[mode]["pick2"] else None
            if p is None:
                p = o["other"] if o["other"] == "SWITCH_QWEN" else "c0"
            return p
        acc_t = np.mean([final(o) == ("c0" if o["c0_correct"] else o["other"]) for o in decisive])
        print(f"  {mode:8s} {single:17.3f} {len(cons)/len(decisive):11.1%} {acc_c:15.3f} {acc_t:17.3f}")
    print("\n  by pair type (single-order accuracy on decisive pairs):")
    for x in PAIRS:
        sub = [o for o in decisive if o["other"] == x]
        if sub:
            print(f"    c0 vs {x:12s} n={len(sub):3d}  " + "  ".join(
                f"{m}={np.mean([o[m]['pick1'] == ('c0' if o['c0_correct'] else o['other']) for o in sub]):.3f}"
                for m in MODES))
    ms = [o["diff_ms"] for o in res]
    print(f"\n  differential probe DB time: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")
    print(f"  diff applicable (same #columns, both executed): "
          f"{np.mean(['only_A' in o['diff_payload'] for o in res]):.1%}")


if __name__ == "__main__":
    main()
