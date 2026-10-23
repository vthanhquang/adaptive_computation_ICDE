"""Run the EllieSQL re-implementation on the 300-question SQLite sample and compare with PlanGate.

Step 1 (LLM, cached): schema linking + all three pipelines (G_B, G_M, G_A) per question.
Step 2 (CPU):        released Qwen2.5-0.5B DPO router -> pipeline choice per question.
Writes results/baselines/elliesql.jsonl and merges EllieSQL policies into
results/baselines/published_methods_sqlite.json, which experiments/crossfit/planner.py compares at
matched budgets.

Usage: python -m experiments.baselines.run_elliesql
"""
from __future__ import annotations
import concurrent.futures as cf, json, time

import numpy as np

from plangate.baselines import elliesql as E
from plangate.benchmark import load_dev
from plangate.engines.sqlite import ex_correct
from plangate.jsonl import iter_jsonl
from plangate.paths import BASELINES_DIR, CROSSFIT_DIR

LEVELS = ["G_B", "G_M", "G_A"]


def run_item(item):
    out = {"question_id": item["question_id"], "db_id": item["db_id"], "difficulty": item["difficulty"]}
    link_acc = E.Acc()
    linked_str, keys_only = E.schema_linking(item, link_acc)
    out["link"] = link_acc.snap(); out["keys_only"] = keys_only
    for lvl in LEVELS:
        acc, db_ms = E.Acc(), [0.0]
        if lvl == "G_B":
            sql = E.gen_basic(item, linked_str, acc)
        elif lvl == "G_M":
            sql = E.gen_intermediate(item, linked_str, acc, db_ms)
        else:
            sql = E.gen_advanced(item, linked_str, acc, db_ms)
        out[lvl] = {"sql": sql, "correct": ex_correct(item, sql), **acc.snap(), "db_ms": db_ms[0]}
    return out


def main():
    items = {x["question_id"]: x for x in load_dev()}
    qids = [r["question_id"] for r in iter_jsonl(CROSSFIT_DIR / "action_table_sqlite.jsonl")]
    path = BASELINES_DIR / "elliesql.jsonl"
    t0 = time.time(); rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(run_item, items[q]) for q in qids]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(qids)} {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["question_id"])

    print("loading DPO router ...", flush=True)
    router = E.DPORouter()
    for r in rows:
        lab, probs, ms = router.predict(E.router_input(items[r["question_id"]]["question"], r["keys_only"]))
        r["route"] = LEVELS[lab]; r["route_probs"] = probs; r["router_ms"] = ms
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf8")

    # ---- summary
    def outcome(r, lvl):
        link, p = r["link"], r[lvl]
        return {"correct": int(p["correct"]), "cost": link["cost"] + p["cost"] + p["db_ms"] / 1000,
                "lat": link["ms"] + p["ms"] + p["db_ms"], "n_llm": link["calls"] + p["calls"],
                "tokens": link["tokens"] + p["tokens"]}

    pol = {f"EllieSQL {lvl} (always)": [outcome(r, lvl) for r in rows] for lvl in LEVELS}
    pol["EllieSQL (DPO router)"] = [dict(outcome(r, r["route"]), lat=outcome(r, r["route"])["lat"] + r["router_ms"])
                                    for r in rows]
    orc = []
    for r in rows:   # oracle router: cheapest pipeline that is correct (waterfall label of the paper)
        lvl = next((l for l in LEVELS if r[l]["correct"]), "G_B")
        orc.append(outcome(r, lvl))
    pol["EllieSQL (oracle router)"] = orc

    from collections import Counter
    print(f"\nEllieSQL on {len(rows)} BIRD dev questions (backbone {E.BACKBONE})")
    print("route distribution:", dict(Counter(r["route"] for r in rows)))
    print(f"  {'policy':30s} {'EX':>6} {'cost':>6} {'tokens':>7} {'p95 lat s':>9} {'LLM/q':>6}")
    for k, v in pol.items():
        print(f"  {k:30s} {np.mean([x['correct'] for x in v]):6.3f} {np.mean([x['cost'] for x in v]):6.2f} "
              f"{np.mean([x['tokens'] for x in v]):7.0f} {np.percentile([x['lat'] for x in v], 95)/1000:9.1f} "
              f"{np.mean([x['n_llm'] for x in v]):6.2f}")

    pb = BASELINES_DIR / "published_methods_sqlite.json"
    d = json.loads(pb.read_text(encoding="utf8"))
    order = {r["question_id"]: i for i, r in enumerate(rows)}
    for k, v in pol.items():
        if "oracle" in k:
            continue
        d["policies"][k] = [v[order[q]] for q in d["question_ids"]]
    pb.write_text(json.dumps(d, default=float), encoding="utf8")
    print("appended EllieSQL policies to", pb)


if __name__ == "__main__":
    main()
