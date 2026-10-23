"""Validate cross-engine scoring: on mini-dev, compare PG-gold correctness (official) with
SQLite-gold-on-SQLite vs candidate-on-PostgreSQL correctness, for every cached candidate.

Usage:  python -m experiments.heldout.validate_cross_engine_gold   (needs the PostgreSQL server)
Output: results/heldout/cross_engine_gold_validation.json
"""
import json

from plangate.benchmark import load_dev
from plangate.engines.cross_engine import ex_correct_cross
from plangate.jsonl import iter_jsonl
from plangate.paths import CROSSFIT_DIR, HELDOUT_DIR

HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
dev = {x["question_id"]: x for x in load_dev()}
agree = n = tp = tn = fp = fn = 0
dis = []
for r in iter_jsonl(CROSSFIT_DIR / "action_table_pg.jsonl"):
    it = dev[r["question_id"]]
    for k in ("c0", "RESAMPLE_1", "REPAIR_DB", "SWITCH_PRO", "SWITCH_QWEN", "VALUE_FIX"):
        e = r.get(k)
        if not e:
            continue
        a, b = bool(e["correct"]), ex_correct_cross(it, e["sql"])
        n += 1; agree += a == b
        tp += a and b; tn += (not a) and (not b); fp += (not a) and b; fn += a and (not b)
        if a != b and len(dis) < 40:
            dis.append({"qid": r["question_id"], "cand": k, "pg_gold": a, "cross": b})
res = {"n_candidates": n, "agreement": agree / n, "both_correct": tp, "both_wrong": tn,
       "cross_only_correct": fp, "pg_only_correct": fn, "examples": dis}
(HELDOUT_DIR / "cross_engine_gold_validation.json").write_text(json.dumps(res, indent=1), encoding="utf8")
print({k: v for k, v in res.items() if k != "examples"})
