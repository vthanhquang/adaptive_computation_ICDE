"""PostgreSQL action table on BIRD mini-dev (500 entries, PostgreSQL dialect).

Same action set as the SQLite table (c0=flash, RESAMPLE_1, REPAIR_DB, VALUE_FIX, SWITCH_PRO,
SWITCH_QWEN), PostgreSQL prompts, gold and evidence on the local PG 16 server.
Usage:  python -m experiments.crossfit.build_action_table_pg [n]
Output: results/crossfit/action_table_pg.jsonl
"""
from __future__ import annotations
import concurrent.futures as cf, json, sys, time

from plangate.benchmark import load_minidev
from plangate.engines import postgres as pg
from plangate.evidence.operators import evidence_to_feedback, value_fix
from plangate.llm.generation import ROLE_MODELS as MODELS, generate_pg, pg_repair_body
from plangate.paths import CROSSFIT_DIR


def gen(item, model, temperature=0.0, sample=0, body=None):
    sql, cost, latency, _ = generate_pg(item, model, temperature, sample, body)
    return sql, cost, latency


def evidence(db_id, sql):
    ev, ms = {}, {}
    for op in pg.OPS:
        ev[op], ms[op] = pg.get_evidence(op, db_id, sql)
    return ev, ms


def entry(it, sql, cost, llm_ms):
    ev, ms = evidence(it["db_id"], sql)
    return {"sql": sql, "correct": pg.ex_correct(it, sql), "cost": cost, "llm_ms": llm_ms, "ev": ev, "db_ms": ms}


def build(it):
    row = {"question_id": it["question_id"], "db_id": it["db_id"], "difficulty": it["difficulty"]}
    sql, c, l = gen(it, MODELS["flash"])
    row["c0"] = entry(it, sql, c, l)
    ev0 = row["c0"]["ev"]
    fixed = value_fix(sql, ev0["VALUE_PROBE"])
    row["VALUE_FIX"] = entry(it, fixed, 0.0, 0.0) if fixed else None
    body = pg_repair_body(it, sql, evidence_to_feedback(ev0))
    s, c, l = gen(it, MODELS["flash"], body=body)
    row["REPAIR_DB"] = entry(it, s, c, l)
    s, c, l = gen(it, MODELS["flash"], temperature=0.8, sample=1)
    row["RESAMPLE_1"] = entry(it, s, c, l)
    for name, m in (("SWITCH_PRO", "pro"), ("SWITCH_QWEN", "qwen")):
        s, c, l = gen(it, MODELS[m])
        row[name] = entry(it, s, c, l)
    return row


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    items = load_minidev()[:n]
    for it in items:
        pg.gold_result(it)
    out = CROSSFIT_DIR / "action_table_pg.jsonl"
    t0 = time.time(); k = 0
    with cf.ThreadPoolExecutor(16) as ex, out.open("w", encoding="utf8") as f:
        for fut in cf.as_completed([ex.submit(build, it) for it in items]):
            try:
                row = fut.result()
            except Exception as e:
                print("ERROR", e, flush=True); continue
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); f.flush(); k += 1
            if k % 25 == 0:
                print(f"{k}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    print("done", k)


if __name__ == "__main__":
    main()
