"""Leave-databases-out splits for PostgreSQL: the 11 BIRD databases are split into 3 groups; fold k
trains on pg_train questions of the other groups and tests on pg_test questions of group k. Every
test question is then answered by a policy that never saw its database.

Usage:  python -m experiments.heldout.make_leave_db_out_splits
Output: configs/splits/pg_lodo{0,1,2}_{train,test}.json
"""
import random

from experiments.heldout.make_splits import SEED, write
from plangate.evaluation.heldout import load_items, load_spec
from plangate.jsonl import iter_jsonl
from plangate.paths import REPLAY_DIR

built_tr = {r["question_id"] for r in iter_jsonl(REPLAY_DIR / "pg_train.jsonl")}
tr_items = [x for x in load_items(load_spec("pg_train")) if x["question_id"] in built_tr]
te_items = load_items(load_spec("pg_test"))
seen, te_u = set(), []
for x in te_items:
    if x["question_id"] not in seen:
        seen.add(x["question_id"]); te_u.append(x)
dbs = sorted({x["db_id"] for x in te_u})
rng = random.Random(SEED); rng.shuffle(dbs)
groups = [dbs[k::3] for k in range(3)]
for k, g in enumerate(groups):
    tr = [x for x in tr_items if x["db_id"] not in g]
    te = [x for x in te_u if x["db_id"] in g]
    write(f"pg_lodo{k}_train", "pg", "bird_dev", tr, gold="cross", table="pg_train", held_out_dbs=g)
    write(f"pg_lodo{k}_test", "pg", "minidev_pg", te, gold="pg", table="pg_test", held_out_dbs=g)
    print(k, g, "train", len(tr), "test", len(te))
