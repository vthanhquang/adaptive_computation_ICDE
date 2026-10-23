"""Held-out evaluation harness: split manifests, replay tables, policy grids and parallel serving.

A split manifest (configs/splits/<name>.json) lists question ids, their source benchmark and the
replay table that holds their pre-computed outcomes (results/heldout/replay/<table>.jsonl[.gz]).
Every policy is served from the replay table, so a run makes no LLM call and no DB access.
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .. import benchmark
from ..jsonl import iter_jsonl
from ..paths import HELDOUT_DIR, REPLAY_DIR, SCALING_DIR, SPLITS_DIR
from ..planning import adaptive as AD

BUDGETS = (1.25, 1.5, 2.0, 3.0, 4.0)
TAUS = (None, 0.5, 0.7)
GRID_LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35)
MARGINS = (-0.01, -0.003, 0.0, 0.005)
POLICIES = ("no_db", "fixed_bundle", "optimizer_gated", "rule_selective", "adaptive")


_SCALED_RUN = re.compile(r"pg_x(\d+)")


def run_dir(name):
    """Result directory of an evaluation run. Held-out runs ("pg", "sqlite", "pg_leave_db_out/fold0")
    live in results/heldout/, resource-scaling runs ("pg_x10", "pg_x100", ...) in results/scaling/."""
    return (SCALING_DIR if _SCALED_RUN.match(name) else HELDOUT_DIR) / name


def run_scale(name):
    """Data scale factor of a run (0 = original database)."""
    m = _SCALED_RUN.match(name)
    return int(m.group(1)) if m else 0


def run_meta(name):
    return json.loads((run_dir(name) / "run_meta.json").read_text(encoding="utf8"))


def run_summary(name):
    return json.loads((run_dir(name) / "summary.json").read_text(encoding="utf8"))


def chosen_traces(name):
    """Per-question traces of the calibrated operating points of a run (config key, question, trace)."""
    return iter_jsonl(run_dir(name) / "chosen_traces.jsonl.gz")


def load_spec(split):
    return json.loads((SPLITS_DIR / f"{split}.json").read_text(encoding="utf8"))


def load_items(spec):
    """Benchmark items of a split manifest, in manifest order."""
    src = spec["source"]
    if src == "bird_dev":
        pool = {x["question_id"]: x for x in benchmark.load_dev()}
    elif src == "bird_train":
        pool = {x["question_id"]: x for x in benchmark.load_train()}
    elif src == "minidev_pg":
        pool = {x["question_id"]: x for x in benchmark.load_minidev()}
    else:
        raise ValueError(src)
    return [pool[q] for q in spec["question_ids"]]


def load(split, scale=0):
    """(replay rows, items by question id, manifest) of a split; `scale` > 0 charges the DB costs
    measured on the scale-times larger PostgreSQL database."""
    spec = load_spec(split)
    table = spec.get("table", split)
    want = set(spec["question_ids"])
    uniq = {}
    for r in iter_jsonl(REPLAY_DIR / f"{table}.jsonl"):   # one row per question
        if r["question_id"] in want:
            uniq.setdefault(r["question_id"], r)
    rows = list(uniq.values())
    if scale:
        rows = apply_scaled_costs(rows, REPLAY_DIR / f"{split}.x{scale}_costs.jsonl")
    items = {x["question_id"]: x for x in load_items(spec)}
    return rows, items, spec


def apply_scaled_costs(rows, path):
    """Resource-scaling replay: keep the observations of the original database, charge the DB
    times and expose the optimizer estimate measured on the scaled schema."""
    sc = {}
    for x in iter_jsonl(path):
        sc[(x["question_id"], x["cand"])] = x
    miss = 0

    def patch(q, cand, e):
        nonlocal miss
        x = sc.get((q, cand))
        if x is None:
            miss += 1
            return
        e["db_ms"] = dict(x["op_ms"])
        exv = e["ev"].get("EXPLAIN") or {}
        if exv.get("valid") and x.get("explain_total_cost") is not None:
            e["ev"]["EXPLAIN"] = dict(exv, total_cost=x["explain_total_cost"])

    for r in rows:
        for k in ("c0", "VALUE_FIX", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"):
            if r.get(k):
                patch(r["question_id"], k, r[k])
        for m, e in r["REPAIR"].items():
            patch(r["question_id"], f"REPAIR:{m}", e)
    if miss:
        print(f"  WARNING: {miss} candidates without scaled costs in {Path(path).name}", flush=True)
    return rows


def summarize(out, price):
    acc = np.array([o["correct"] for o in out], float)
    db = np.array([o["db_ms"] for o in out], float)
    llm = np.array([o["llm_cost"] for o in out], float)
    wall = np.array([o["wall"] for o in out], float)
    return {"acc": float(acc.mean()), "cost": float((llm + price * db / 1000).mean()), "llm_cost": float(llm.mean()),
            "db_ms": float(db.mean()), "db_p95": float(np.percentile(db, 95)), "wall_p95": float(np.percentile(wall, 95)),
            "n_vp": float(np.mean([o.get("n_vp", 0) for o in out])), "n_ex": float(np.mean([o.get("n_ex", 0) for o in out])),
            "n": len(out)}


def grid(engine, train_rows):
    """Parameter grid of every policy. The optimizer gate is PostgreSQL-only (SQLite has no cost estimate)."""
    G = [("cheap_only", {})]
    for lam in GRID_LAMBDAS:
        for tau in TAUS:
            G += [("no_db", {"lam": lam, "tau": tau}), ("fixed_bundle", {"lam": lam, "tau": tau}),
                  ("rule_selective", {"lam": lam, "tau": tau})]
            G += [("adaptive", {"lam": lam, "tau": tau, "voi_margin": m}) for m in MARGINS]
    if engine == "pg":
        cs = [(r["c0"]["ev"].get("EXPLAIN") or {}).get("total_cost") for r in train_rows]
        cs = np.array([c for c in cs if c is not None], float)
        for pct in (50, 75, 90):
            th = float(np.percentile(cs, pct))
            for lam in GRID_LAMBDAS:
                for tau in TAUS:
                    G.append(("optimizer_gated", {"lam": lam, "theta": th, "pct": pct, "tau": tau}))
    G.append(("oracle", {}))
    return G


_W = {}


def _init(pl_path, data_path, oof=False):
    _W["pl"] = pickle.loads(Path(pl_path).read_bytes())
    rows, items = pickle.loads(Path(data_path).read_bytes())
    _W["R"] = AD.Runner(_W["pl"], rows, items, oof=oof)
    _W["qs"] = [r["question_id"] for r in rows]


def _job(args):
    pol, kw, price = args
    R = _W["R"]
    R.db_price = price
    kws = {k: v for k, v in kw.items() if k != "pct"}
    return pol, json.dumps(kw, sort_keys=True), price, [R.run(q, pol, **kws) for q in _W["qs"]]


def run_all(pl_path, rows, items, G, prices, workers, tmp, oof=False):
    """runs[(policy, params_json, price)] = per-question outputs. Non-adaptive policies never
    observe the DB price, so they run once and are re-priced; adaptive runs once per price."""
    data_path = Path(tmp) / ("serve_data_oof.pkl" if oof else "serve_data.pkl")
    data_path.write_bytes(pickle.dumps((rows, items)))
    jobs = []
    for pol, kw in G:
        jobs += [(pol, kw, p) for p in prices] if pol == "adaptive" else [(pol, kw, 1.0)]
    runs = {}
    with cf.ProcessPoolExecutor(workers, initializer=_init, initargs=(str(pl_path), str(data_path), oof)) as ex:
        for pol, kwj, p, out in ex.map(_job, jobs, chunksize=2):
            if pol == "adaptive":
                runs[(pol, kwj, p)] = out
            else:
                for pp in prices:
                    runs[(pol, kwj, pp)] = out
    return runs
