"""Immutable split manifests for the held-out evaluation (configs/splits/*.json) with leakage checks.

  sqlite_train / sqlite_cal : BIRD *train* questions; train and calibration use DISJOINT databases,
                              and both are database-disjoint from BIRD dev (by BIRD construction)
  sqlite_test               : the 300 BIRD dev questions of the cross-fitted study (the questions of
                              results/crossfit/action_table_sqlite.jsonl, in dev order)
  pg_train / pg_cal         : BIRD dev questions NOT in mini-dev, run on PostgreSQL (question-disjoint
                              from pg_test; same 11 databases), scored with SQLite gold (cross-engine)
  pg_test                   : the 500 official PostgreSQL mini-dev entries
A manifest lists the candidate question ids; evaluation uses the questions that have a row in the
replay table named by the manifest ("table", default: the split name).
The reported runs calibrate operating points by cross-fitting on the training split (--cal oof), so
the *_cal manifests are only used with an explicit --cal <split>.

Leakage checks: no shared question id, no shared normalized question text, no shared
(db, normalized gold SQL) between any train/cal split and the test split of the same engine.
pg_test and sqlite_test share questions (same benchmark, two engines); they are never pooled.

Usage: python -m experiments.heldout.make_splits [--sqlite-train N] [--sqlite-cal N]
"""
import argparse, hashlib, json, random, re
from collections import defaultdict

from plangate import benchmark
from plangate.jsonl import iter_jsonl
from plangate.paths import CROSSFIT_DIR, SPLITS_DIR

SEED = 20261019


def nq(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def nsql(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip().rstrip(";")


def write(name, engine, source, items, **extra):
    """extra may include table=<replay table name> when the split reads another split's table."""
    ids = [x["question_id"] for x in items]
    spec = {"name": name, "engine": engine, "source": source, "n": len(ids), "seed": SEED,
            "databases": sorted({x["db_id"] for x in items}), "question_ids": ids, **extra}
    spec["sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    (SPLITS_DIR / f"{name}.json").write_text(json.dumps(spec, indent=1), encoding="utf8")
    return spec


def check(train, test, label):
    issues = []
    tid = {x["question_id"] for x in test}
    tq = {nq(x["question"]) for x in test}
    ts = {(x["db_id"], nsql(x["SQL"])) for x in test}
    for x in train:
        if x["question_id"] in tid:
            issues.append(("id", x["question_id"]))
        if nq(x["question"]) in tq:
            issues.append(("question_text", x["question_id"]))
        if (x["db_id"], nsql(x["SQL"])) in ts:
            issues.append(("db_sql", x["question_id"]))
    return {"pair": label, "n_train": len(train), "n_test": len(test), "violations": len(issues),
            "examples": issues[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-train", type=int, default=800)
    ap.add_argument("--sqlite-cal", type=int, default=300)
    ap.add_argument("--pg-cal-frac", type=float, default=0.3)
    a = ap.parse_args()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = {"seed": SEED, "splits": {}, "leakage": []}

    dev = benchmark.load_dev()
    mini = benchmark.load_minidev()
    mini_ids = {x["question_id"] for x in mini}
    devmap = {x["question_id"]: x for x in dev}

    # ---------------- PostgreSQL
    pool = [x for x in dev if x["question_id"] not in mini_ids]
    mini_q = {nq(x["question"]) for x in mini}
    mini_sql = {(devmap[x["question_id"]]["db_id"], nsql(devmap[x["question_id"]]["SQL"])) for x in mini}
    pool = [x for x in pool if nq(x["question"]) not in mini_q          # drop textual duplicates of test
            and (x["db_id"], nsql(x["SQL"])) not in mini_sql]           # and same-gold-SQL duplicates
    seen, dedup = set(), []
    for x in pool:                                                       # one copy per (db, gold SQL)
        k = (x["db_id"], nsql(x["SQL"]))
        if k not in seen and nq(x["question"]) not in seen:
            seen.add(k); seen.add(nq(x["question"])); dedup.append(x)
    pool = dedup
    by_db = defaultdict(list)
    for x in pool:
        by_db[x["db_id"]].append(x)
    pg_train, pg_cal = [], []
    for db in sorted(by_db):
        g = by_db[db][:]; rng.shuffle(g)
        k = round(len(g) * a.pg_cal_frac)
        pg_cal += g[:k]; pg_train += g[k:]
    report["splits"]["pg_train"] = write("pg_train", "pg", "bird_dev", pg_train, gold="cross")["n"]
    report["splits"]["pg_cal"] = write("pg_cal", "pg", "bird_dev", pg_cal, gold="cross")["n"]
    report["splits"]["pg_test"] = write("pg_test", "pg", "minidev_pg", mini, gold="pg")["n"]
    mini_as_dev = [dict(devmap[x["question_id"]], question=x["question"]) for x in mini]
    report["leakage"] += [check(pg_train, mini_as_dev, "pg_train vs pg_test"),
                          check(pg_cal, mini_as_dev, "pg_cal vs pg_test"),
                          check(pg_train, pg_cal, "pg_train vs pg_cal")]

    # ---------------- SQLite
    sample = {r["question_id"] for r in iter_jsonl(CROSSFIT_DIR / "action_table_sqlite.jsonl")}
    test = [x for x in dev if x["question_id"] in sample]
    report["splits"]["sqlite_test"] = write("sqlite_test", "sqlite", "bird_dev", test, table="sqlite_test")["n"]
    if (benchmark.BIRD_TRAIN_DIR / "train.json").exists():
        train = benchmark.load_train()
        dbs = sorted({x["db_id"] for x in train}); rng.shuffle(dbs)
        cal_dbs = set(dbs[: max(1, len(dbs) // 4)])                     # ~25% of databases for calibration
        tr_pool = [x for x in train if x["db_id"] not in cal_dbs]
        ca_pool = [x for x in train if x["db_id"] in cal_dbs]
        tr = rng.sample(tr_pool, min(a.sqlite_train, len(tr_pool)))
        ca = rng.sample(ca_pool, min(a.sqlite_cal, len(ca_pool)))
        report["splits"]["sqlite_train"] = write("sqlite_train", "sqlite", "bird_train", tr)["n"]
        report["splits"]["sqlite_cal"] = write("sqlite_cal", "sqlite", "bird_train", ca)["n"]
        report["leakage"] += [check(tr, test, "sqlite_train vs sqlite_test"), check(ca, test, "sqlite_cal vs sqlite_test"),
                              check(tr, ca, "sqlite_train vs sqlite_cal")]
        report["sqlite_db_overlap_train_test"] = sorted({x["db_id"] for x in tr} & {x["db_id"] for x in test})
        report["sqlite_db_overlap_train_cal"] = sorted({x["db_id"] for x in tr} & {x["db_id"] for x in ca})
    else:
        report["note"] = "BIRD train not found (data/bird_train): sqlite_train / sqlite_cal not written"

    (SPLITS_DIR / "MANIFEST.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "leakage"}, indent=1))
    for l in report["leakage"]:
        print(f"  {l['pair']:32s} train={l['n_train']:5d} test={l['n_test']:5d} violations={l['violations']}")


if __name__ == "__main__":
    main()
