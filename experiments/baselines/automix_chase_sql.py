"""AutoMix and CHASE-SQL selection, re-implemented in the PlanGate setting (same cache, same costs).

AutoMix (Aggarwal et al., NeurIPS 2024) -- threshold meta-verifier variant
  small model (flash) answers; the SAME small model self-verifies its answer k=3 times
  (sampled, T=0.7, no execution); confidence = fraction of "Yes"; if confidence < theta the
  query is routed to the large model (qwen), whose answer replaces the small one.
  Verification prompt: zero-shot by default; --fewshot uses few-shot context-grounded prompts as in the paper.

CHASE-SQL selection (Pourreza et al., ICLR 2025)
  multiple generators produce candidates; candidates are grouped by execution result;
  a pairwise selection agent compares cluster representatives given question, schema and
  both execution results; the candidate with most pairwise wins is returned.
  Deviations: generators = our pool (flash, pro, qwen [+ flash resample]) instead of CHASE's
  three CoT generators; selector = off-the-shelf deepseek-v4-pro (CHASE fine-tunes it).

Usage:  python -m experiments.baselines.automix_chase_sql sqlite|pg [--fewshot]
Output: policies merged into results/baselines/published_methods_<engine>.json
"""
from __future__ import annotations
import concurrent.futures as cf, json, re, sys
from collections import Counter
from itertools import combinations

import numpy as np

from plangate.baselines.fewshot_exemplars import EXEMPLARS, result_text as ex_result_text
from plangate.llm.generation import PG_SYSTEM, SYSTEM, cached_chat, cost_units
from plangate.paths import BASELINES_DIR
from plangate.planning.crossfit import fp, load_action_table

FLASH, PRO = "deepseek-v4-flash", "deepseek-v4-pro"
K_VERIFY = 3

VERIFY = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

Proposed SQL query:
```sql
{sql}
```

Does the proposed SQL query correctly answer the question? Check the tables, columns, joins, filter values, aggregation and the returned columns.
Answer "Yes" or "No" on the first line, then give a one-sentence reason."""

SELECT = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

Two candidate SQL queries were written for this question and return different results.

Candidate A:
```sql
{a}
```
Execution result of A: {ra}

Candidate B:
```sql
{b}
```
Execution result of B: {rb}

Which candidate answers the question correctly? Think briefly, then end with a final line "Answer: A" or "Answer: B"."""


# ---- few-shot variants: exemplars from BIRD dev questions outside both evaluation sets
FEWSHOT = "--fewshot" in sys.argv


def _verify_examples():
    out, k = [], 1
    for e in EXEMPLARS:
        for sql, ok in ((e["correct"], True), (e["wrong"], False)):
            reason = ("The SQL uses the right tables, columns, filters and aggregation for the question."
                      if ok else e["why_wrong"])
            out.append(f"Example {k}:\nQuestion: {e['question']}\nHint: {e['evidence']}\n"
                       f"AI Generated SQL: {sql}\nEvaluation: {reason} So it is {'Correct' if ok else 'Incorrect'}.")
            k += 1
    return "\n\n".join(out)


VERIFY_FEWSHOT = """Instruction: Your task is to evaluate if the AI Generated SQL is correct, based on the database schema and the question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

""" + _verify_examples().replace("{", "{{").replace("}", "}}") + """

Now evaluate the following case.
Context (database schema):
{schema}

Question: {question}
Hint: {evidence}
AI Generated SQL: {sql}
Evaluation:"""


def _select_examples():
    out = []
    for i, e in enumerate(EXEMPLARS):
        a_ok = i % 2 == 0   # alternate the position of the correct candidate
        a, b = (e["correct"], e["wrong"]) if a_ok else (e["wrong"], e["correct"])
        out.append(f"Example {i+1}:\nQuestion: {e['question']}\nHint: {e['evidence']}\n"
                   f"Candidate A: {a}\nExecution result of A: {ex_result_text(e['db_id'], a)}\n"
                   f"Candidate B: {b}\nExecution result of B: {ex_result_text(e['db_id'], b)}\n"
                   f"Reasoning: {e['why_wrong']} Candidate {'B' if a_ok else 'A'} has this error.\n"
                   f"Answer: {'A' if a_ok else 'B'}")
    return "\n\n".join(out)


SELECT_FEWSHOT = """Instruction: Given the database info and the question, there are two candidate queries that return different results. One is correct and one is incorrect. Compare the two candidates, analyse the differences of the queries and of their execution results, and choose the correct one.

""" + _select_examples().replace("{", "{{").replace("}", "}}") + """

Now solve the following case.
Database schema:
{schema}

Question: {question}
Hint: {evidence}

Candidate A:
```sql
{a}
```
Execution result of A: {ra}

Candidate B:
```sql
{b}
```
Execution result of B: {rb}

Think briefly, then end with a final line "Answer: A" or "Answer: B"."""

if FEWSHOT:
    VERIFY_TMPL, SELECT_TMPL, TAG = VERIFY_FEWSHOT, SELECT_FEWSHOT, " few-shot"
else:
    VERIFY_TMPL = SELECT_TMPL = TAG = None   # set below to the zero-shot templates


def setup(engine):
    rows, items = load_action_table(engine)
    if engine == "sqlite":
        from plangate.engines.sqlite import schema_with_samples
        return rows, items, schema_with_samples, SYSTEM
    from plangate.engines.postgres import pg_schema
    return rows, items, pg_schema, PG_SYSTEM


def result_text(e):
    eb = e["ev"]["EXEC_BOUNDED"]
    if not eb.get("ok"):
        return f"execution error ({eb.get('error', '')[:120]})"
    return f"{eb['n_rows']}{'+' if eb.get('truncated') else ''} rows x {eb['n_cols']} columns; first rows: {eb['preview']}"


def autoMix_verify(r, item, schema_fn, system):
    body = (VERIFY_TMPL or VERIFY).format(schema=schema_fn(item["db_id"]), question=item["question"],
                                          evidence=item.get("evidence") or "none", sql=r["c0"]["sql"])
    yes, cost, lats = 0, 0.0, []
    for s in range(1, K_VERIFY + 1):
        d = cached_chat(FLASH, [{"role": "system", "content": system}, {"role": "user", "content": body}],
                        temperature=0.7, sample=s, max_tokens=300)
        txt = (d["text"] or "").strip()
        if FEWSHOT:   # AutoMix-style verdict: last "Correct"/"Incorrect" in the evaluation
            v = re.findall(r"\b(Incorrect|Correct)\b", txt)
            yes += bool(v) and v[-1] == "Correct"
        else:
            first = txt.lower()
            yes += first.startswith("yes") or first.startswith("**yes")
        cost += cost_units(FLASH, d["usage"]); lats.append(d["latency_ms"])
    return yes / K_VERIFY, cost, max(lats)   # k verifications run in parallel


def chase_select(r, item, names, schema_fn, system):
    """Pairwise tournament over execution clusters. Returns (final name, judge cost, judge latency, n_calls)."""
    clusters = {}
    for n in names:
        f = fp(r[n]); key = ("fp:" + f) if f else ("fail:" + n)
        clusters.setdefault(key, []).append(n)
    reps = [v[0] for v in clusters.values()]           # first generator in each cluster
    ok_reps = [n for n in reps if fp(r[n])] or reps
    if len(ok_reps) == 1:
        return ok_reps[0], 0.0, 0.0, 0
    wins, cost, lats, calls = Counter(), 0.0, [], 0
    for a, b in combinations(ok_reps, 2):
        body = (SELECT_TMPL or SELECT).format(schema=schema_fn(item["db_id"]), question=item["question"],
                             evidence=item.get("evidence") or "none", a=r[a]["sql"], ra=result_text(r[a]),
                             b=r[b]["sql"], rb=result_text(r[b]))
        d = cached_chat(PRO, [{"role": "system", "content": system}, {"role": "user", "content": body}])
        m = re.findall(r"Answer:\s*\**\s*([AB])\b", d["text"] or "")
        if m:
            wins[a if m[-1] == "A" else b] += 1
        cost += cost_units(PRO, d["usage"]); lats.append(d["latency_ms"]); calls += 1
    size = {n: len(clusters[("fp:" + fp(r[n])) if fp(r[n]) else ("fail:" + n)]) for n in ok_reps}
    best = max(ok_reps, key=lambda n: (wins[n], size[n], -names.index(n)))
    return best, cost, max(lats), calls   # pairwise comparisons run in parallel


def gen_cost(r, names, parallel=True):
    db = lambda n: r[n]["db_ms"]["EXEC_BOUNDED"]
    cost = sum(r[n]["cost"] + db(n) / 1000 for n in names)
    lat = max(r[n]["llm_ms"] + db(n) for n in names) if parallel else sum(r[n]["llm_ms"] + db(n) for n in names)
    return cost, lat


def main():
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    engine = pos_args[0] if pos_args else "sqlite"
    tagname = TAG or ""
    rows, items, schema_fn, system = setup(engine)
    rows.sort(key=lambda r: r["question_id"])
    pools = {f"CHASE-SQL selection{tagname} (flash, pro, qwen)": ["c0", "SWITCH_PRO", "SWITCH_QWEN"],
             f"CHASE-SQL selection{tagname} (flash x2, pro, qwen)": ["c0", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"]}

    def per_q(r):
        it = items[r["question_id"]]
        conf, vcost, vlat = autoMix_verify(r, it, schema_fn, system)
        out = {"question_id": r["question_id"], "automix_conf": conf, "automix_vcost": vcost, "automix_vlat": vlat}
        for name, names in pools.items():
            final, jc, jl, calls = chase_select(r, it, names, schema_fn, system)
            gc, gl = gen_cost(r, names)
            out[name] = {"correct": int(r[final]["correct"]), "cost": gc + jc, "lat": gl + jl,
                         "n_llm": len(names) + calls, "final": final}
        return out

    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(per_q, rows))

    policies = {}
    # confidence takes values in {0, 1/3, 2/3, 1}: route if conf < 0.3 / 0.6 / 0.9
    for th in (0.3, 0.6, 0.9):
        out = []
        for r, x in zip(rows, res):
            c0c, c0l = r["c0"]["cost"], r["c0"]["llm_ms"]
            if x["automix_conf"] >= th:
                out.append({"correct": int(r["c0"]["correct"]), "cost": c0c + x["automix_vcost"],
                            "lat": c0l + x["automix_vlat"], "n_llm": 1 + K_VERIFY})
            else:
                q = r["SWITCH_QWEN"]
                out.append({"correct": int(q["correct"]), "cost": c0c + x["automix_vcost"] + q["cost"],
                            "lat": c0l + x["automix_vlat"] + q["llm_ms"], "n_llm": 2 + K_VERIFY})
        policies[f"AutoMix{tagname} (theta={th:.2f})"] = out
    for name in pools:
        policies[name] = [x[name] for x in res]

    y = np.array([int(r["c0"]["correct"]) for r in rows]); conf = np.array([x["automix_conf"] for x in res])
    from sklearn.metrics import roc_auc_score
    print(f"[{engine}] n={len(rows)}; AutoMix self-verification AUROC on c0 = {roc_auc_score(y, conf):.3f}; "
          f"mean confidence {conf.mean():.2f}")
    print(f"  {'policy':44s} {'acc':>6} {'cost':>6} {'p95 lat s':>9} {'LLM/q':>6}")
    for k, v in policies.items():
        lat = [p["lat"] for p in v]
        print(f"  {k:44s} {np.mean([p['correct'] for p in v]):6.3f} {np.mean([p['cost'] for p in v]):6.2f} "
              f"{np.percentile(lat, 95)/1000:9.1f} {np.mean([p['n_llm'] for p in v]):6.2f}")
    path = BASELINES_DIR / f"published_methods_{engine}.json"
    # merge into the existing file (other scripts append EllieSQL / SQLens / DIN / MAC policies)
    store = json.loads(path.read_text(encoding="utf8")) if path.exists() else \
        {"question_ids": [r["question_id"] for r in rows], "policies": {}}
    pos = {r["question_id"]: i for i, r in enumerate(rows)}
    for k, v in policies.items():
        store["policies"][k] = [v[pos[q]] for q in store["question_ids"]]
    store[f"automix{tagname.strip().replace(' ', '_') or ''}_auroc"] = roc_auc_score(y, conf)
    path.write_text(json.dumps(store, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
