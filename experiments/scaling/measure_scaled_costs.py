"""Measure DB operator costs of every replay-table candidate on the k-scaled schema x<k>.

Observations (results, fingerprints, value-probe outcomes) are taken from the original database;
only the *costs* (EXPLAIN estimate, VALUE_PROBE / EXEC_BOUNDED / mandatory-probe milliseconds) are
re-measured on the scaled copy. Output: results/heldout/replay/<split>.x<k>_costs.jsonl with
{question_id, cand, op_ms:{op: ms}, explain_total_cost}.

Usage: python -m experiments.scaling.measure_scaled_costs <k> <split> [--workers 4]
       (k in {10, 100}; split in {pg_train, pg_test})
"""
import argparse, concurrent.futures as cf, json, threading, time

import psycopg

from plangate.engines import postgres as pg
from plangate.jsonl import iter_jsonl, resolve
from plangate.paths import CACHE_DIR, REPLAY_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int); ap.add_argument("split"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sch = f"x{a.k}"
    # scaled sessions: search_path to the scaled schema, separate evidence cache
    local = threading.local()

    def conn():
        c = getattr(local, "c", None)
        if c is None or c.closed:
            c = psycopg.connect(pg.DSN, autocommit=True)
            c.execute("SET default_transaction_read_only = on")
            c.execute(f"SET search_path TO {sch}")
            local.c = c
        return c

    pg.conn = conn                                   # route evidence operators to the scaled schema
    pg.CACHE = CACHE_DIR / f"pg_evidence_{sch}"
    pg.CACHE.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(REPLAY_DIR / f"{a.split}.jsonl"))
    out = resolve(REPLAY_DIR / f"{a.split}.{sch}_costs.jsonl")
    done = {(x["question_id"], x["cand"]) for x in iter_jsonl(out)} if out.exists() else set()
    out = out.with_suffix("") if out.suffix == ".gz" else out      # new measurements go to a plain file
    jobs = []
    for r in rows:
        for k in ("c0", "VALUE_FIX", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"):
            if r.get(k):
                jobs.append((r["question_id"], r["db_id"], k, r[k]["sql"]))
        for m, e in r["REPAIR"].items():
            jobs.append((r["question_id"], r["db_id"], f"REPAIR:{m}", e["sql"]))
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    print(f"{a.split} on {sch}: {len(jobs)} candidate measurements", flush=True)

    def measure(j):
        q, db, cand, sql = j
        ms, tc = {}, None
        for op in pg.OPS:
            ev, t = pg.get_evidence(op, db, sql)
            ms[op] = t
            if op == "EXPLAIN" and ev.get("valid"):
                tc = ev.get("total_cost")
        return {"question_id": q, "cand": cand, "op_ms": ms, "explain_total_cost": tc}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf8") as f:
        for i, res in enumerate(ex.map(measure, jobs)):
            f.write(json.dumps(res) + "\n")
            if (i + 1) % 200 == 0:
                f.flush(); print(f"  {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
