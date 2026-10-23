"""Counterfactual generation matrix: run every generation config on the same questions.

Usage:
  python -m experiments.crossfit.generation_matrix --n 40 --configs pilot --out results/crossfit/generation_pilot.jsonl
  python -m experiments.crossfit.generation_matrix --n 300 --configs catalog --out results/crossfit/generation_matrix.jsonl
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, time
from pathlib import Path

from plangate.benchmark import stratified_sample
from plangate.engines.sqlite import ex_correct, gold_result
from plangate.llm.generation import generate
from plangate.paths import CONFIG_DIR

PILOT = {m.replace(".", "_") + "__direct": {"model": m, "profile": "direct", "schema": "samples"}
         for m in ["hy3", "deepseek-v4-flash", "deepseek-v4.1-flash", "glm-5.3-flash",
                   "deepseek-v4-pro", "qwen3.8-max", "kimi-k3"]}

CATALOG = json.loads((CONFIG_DIR / "generation_configs.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    configs = PILOT if a.configs == "pilot" else CATALOG
    items = [x for x in stratified_sample(a.n, a.seed) if gold_result(x) is not None]
    print(f"{len(items)} questions with executable gold, {len(configs)} configs", flush=True)

    jobs = [(it, name, cfg) for it in items for name, cfg in configs.items()]

    def run(job):
        it, name, cfg = job
        g = generate(it, cfg)
        g["correct"] = ex_correct(it, g["sql"])
        return {"question_id": it["question_id"], "db_id": it["db_id"], "difficulty": it["difficulty"],
                "config": name, **g}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex, out.open("w", encoding="utf8") as f:
        futs = [ex.submit(run, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs)):
            r = fut.result()
            rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)

    print("\nconfig                         acc    cost/q  lat_ms  errors")
    for name in configs:
        rs = [r for r in rows if r["config"] == name]
        acc = sum(r["correct"] for r in rs) / len(rs)
        cost = sum(r["cost"] for r in rs) / len(rs)
        lat = sum(r["latency_ms"] for r in rs) / len(rs)
        errs = sum(1 for r in rs if r.get("error"))
        print(f"{name:30s} {acc:.3f}  {cost:6.2f}  {lat:7.0f}  {errs}")


if __name__ == "__main__":
    main()
