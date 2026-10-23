"""Run DIN-SQL and MAC-SQL re-implementations on the 300 SQLite questions (backbone flash).

Writes results/baselines/din_mac_sql.jsonl and merges "DIN-SQL" / "MAC-SQL" policies into
results/baselines/published_methods_sqlite.json for the matched-budget comparison of
experiments/crossfit/planner.py.
Usage: python -m experiments.baselines.run_din_mac_sql [model]
"""
from __future__ import annotations
import concurrent.futures as cf, json, sys, time
from collections import Counter

import numpy as np

from plangate.baselines import classic as C
from plangate.benchmark import load_dev
from plangate.engines.sqlite import ex_correct
from plangate.jsonl import iter_jsonl
from plangate.paths import BASELINES_DIR, CROSSFIT_DIR


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"
    items = {x["question_id"]: x for x in load_dev()}
    qids = sorted(r["question_id"] for r in iter_jsonl(CROSSFIT_DIR / "action_table_sqlite.jsonl"))
    for it in {items[q]["db_id"] for q in qids}:      # warm per-DB schema caches once
        C.din_table_info(it); C.din_column_descriptions(it); C.mac_db_info(it)

    def run(q):
        it = items[q]; out = {"question_id": q}
        for name, fn in (("DIN-SQL", C.din_sql), ("MAC-SQL", C.mac_sql)):
            try:
                sql, meta = fn(it, model)
            except Exception as e:
                sql, meta = "", {"cost": 0.0, "ms": 0.0, "calls": 0, "tokens": 0, "db_ms": 0.0, "error": str(e)[:200]}
            out[name] = {"sql": sql, "correct": ex_correct(it, sql), **meta}
        return out

    t0 = time.time(); rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(cf.as_completed([ex.submit(run, q) for q in qids])):
            rows.append(r.result())
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(qids)} {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["question_id"])
    (BASELINES_DIR / "din_mac_sql.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows), encoding="utf8")

    pol = {}
    for name in ("DIN-SQL", "MAC-SQL"):
        pol[name] = [{"correct": int(r[name]["correct"]), "cost": r[name]["cost"] + r[name]["db_ms"] / 1000,
                      "lat": r[name]["ms"] + r[name]["db_ms"], "n_llm": r[name]["calls"]} for r in rows]
        v = pol[name]
        print(f"{name:8s} EX={np.mean([x['correct'] for x in v]):.3f} cost={np.mean([x['cost'] for x in v]):.2f} "
              f"tokens={np.mean([r[name]['tokens'] for r in rows]):.0f} p95={np.percentile([x['lat'] for x in v], 95)/1000:.1f}s "
              f"calls={np.mean([x['n_llm'] for x in v]):.2f} errors={sum(1 for r in rows if r[name].get('error'))} "
              f"api_retries={sum(r[name].get('api_errors', 0) for r in rows)} "
              f"empty_sql={sum(1 for r in rows if not r[name]['sql'] or r[name]['sql'].startswith('error'))}")
    print("DIN-SQL class distribution:", dict(Counter(r["DIN-SQL"].get("label") for r in rows)))
    print("MAC-SQL pruned share:", np.mean([bool(r["MAC-SQL"].get("pruned")) for r in rows]))

    pb = BASELINES_DIR / "published_methods_sqlite.json"
    d = json.loads(pb.read_text(encoding="utf8"))
    order = {r["question_id"]: i for i, r in enumerate(rows)}
    for name, v in pol.items():
        d["policies"][name] = [v[order[q]] for q in d["question_ids"]]
    pb.write_text(json.dumps(d, default=float), encoding="utf8")
    print("appended DIN-SQL / MAC-SQL to", pb)


if __name__ == "__main__":
    main()
