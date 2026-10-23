"""SQLite action table: first candidate + every DB probe + every follow-up action, per question.

For each question, starting from the first candidate c0 (= flash_direct):
  DB probes      PARSE_BIND, EXPLAIN, VALUE_PROBE, EXEC_BOUNDED on c0   (payload + ms)
  VALUE_FIX      DB-only literal rewrite from VALUE_PROBE                 (no LLM)
  REPAIR_DB      same model, conditioned on c0's DB evidence
  REPAIR_BLIND   same model, generic feedback (control)
  RESAMPLE_1/2   same model, temperature 0.8, independent samples
  SWITCH_PRO     deepseek-v4-pro direct     (from the matrix)
  SWITCH_QWEN    qwen3.8-max direct         (from the matrix)
Every produced SQL gets EX correctness and its own DB evidence (for agreement features).

Usage:  python -m experiments.crossfit.build_action_table [generation_matrix.jsonl]
Output: results/crossfit/action_table_sqlite.jsonl (LLM calls are cached; uncached calls need API keys)
"""
from __future__ import annotations
import concurrent.futures as cf, json, sys, time

from plangate.benchmark import load_dev
from plangate.engines.sqlite import ex_correct
from plangate.evidence.cache import get_evidence
from plangate.evidence.operators import DB_OPS, evidence_to_feedback, value_fix
from plangate.jsonl import iter_jsonl
from plangate.llm.generation import generate, repair
from plangate.paths import CONFIG_DIR, CROSSFIT_DIR

CATALOG = json.loads((CONFIG_DIR / "generation_configs.json").read_text())
FIRST, PRO, QWEN = "flash_direct", "pro_direct", "qwen_max_direct"
GENERIC = ("- No specific diagnostics are available. Re-check table and column names, join conditions, "
           "filter values, and that only the requested columns are returned.")
RESAMPLE_T = 0.8


def evidence(db_id, sql):
    ev, ms = {}, {}
    for op in DB_OPS:
        ev[op], ms[op] = get_evidence(op, db_id, sql)
    return ev, ms


def entry(it, sql, cost, llm_ms):
    ev, ms = evidence(it["db_id"], sql)
    return {"sql": sql, "correct": ex_correct(it, sql), "cost": cost, "llm_ms": llm_ms, "ev": ev, "db_ms": ms}


def build(it, M):
    q = it["question_id"]
    c0 = M[q][FIRST]
    row = {"question_id": q, "db_id": it["db_id"], "difficulty": it["difficulty"]}
    row["c0"] = entry(it, c0["sql"], c0["cost"], c0["latency_ms"])
    ev0 = row["c0"]["ev"]

    fixed = value_fix(c0["sql"], ev0["VALUE_PROBE"])
    row["VALUE_FIX"] = entry(it, fixed, 0.0, 0.0) if fixed else None

    model = CATALOG[FIRST]["model"]
    r = repair(it, model, c0["sql"], evidence_to_feedback(ev0))
    row["REPAIR_DB"] = entry(it, r["sql"], r["cost"], r["latency_ms"])
    r = repair(it, model, c0["sql"], GENERIC)
    row["REPAIR_BLIND"] = entry(it, r["sql"], r["cost"], r["latency_ms"])
    for s in (1, 2):
        g = generate(it, CATALOG[FIRST], temperature=RESAMPLE_T, sample=s)
        row[f"RESAMPLE_{s}"] = entry(it, g["sql"], g["cost"], g["latency_ms"])
    for name, cfg in (("SWITCH_PRO", PRO), ("SWITCH_QWEN", QWEN)):
        m = M[q][cfg]
        row[name] = entry(it, m["sql"], m["cost"], m["latency_ms"])
    return row


def main():
    matrix = sys.argv[1] if len(sys.argv) > 1 else CROSSFIT_DIR / "generation_matrix.jsonl"
    items = {x["question_id"]: x for x in load_dev()}
    M = {}
    for r in iter_jsonl(matrix):
        M.setdefault(r["question_id"], {})[r["config"]] = r
    Q = sorted(q for q in M if all(c in M[q] for c in (FIRST, PRO, QWEN)))
    t0 = time.time()
    out = CROSSFIT_DIR / "action_table_sqlite.jsonl"
    rows = []
    with cf.ThreadPoolExecutor(16) as ex, out.open("w", encoding="utf8") as f:
        futs = [ex.submit(build, items[q], M) for q in Q]
        for i, fut in enumerate(cf.as_completed(futs)):
            row = fut.result(); rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(Q)}  {time.time()-t0:.0f}s", flush=True)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
