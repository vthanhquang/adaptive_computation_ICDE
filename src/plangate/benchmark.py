"""BIRD benchmark loaders: dev (SQLite), train (SQLite) and mini-dev (PostgreSQL)."""
from __future__ import annotations

import json
import random
from collections import defaultdict

from .paths import DATA_DIR

BIRD_DEV_DIR = DATA_DIR / "bird" / "dev_20240627"
BIRD_TRAIN_DIR = DATA_DIR / "bird_train" / "train"
MINIDEV_DIR = DATA_DIR / "pg" / "minidev" / "MINIDEV"


def load_dev():
    return json.loads((BIRD_DEV_DIR / "dev.json").read_text(encoding="utf8"))


def load_train():
    """BIRD training split. Question ids are prefixed with 't' so that they never collide with dev
    ids in any cache or result table (BIRD train and dev use overlapping integer ids)."""
    data = json.loads((BIRD_TRAIN_DIR / "train.json").read_text(encoding="utf8"))
    for i, x in enumerate(data):
        x["question_id"] = f"t{i}"
        x.setdefault("difficulty", "unknown")
    return data


def load_minidev():
    """The 500 official BIRD mini-dev questions with PostgreSQL gold SQL."""
    return json.loads((MINIDEV_DIR / "mini_dev_postgresql.json").read_text(encoding="utf8"))


def stratified_sample(n, seed=0):
    """Sample n dev questions, stratified by (db_id, difficulty) proportionally."""
    data = load_dev()
    groups = defaultdict(list)
    for x in data:
        groups[(x["db_id"], x["difficulty"])].append(x)
    rng = random.Random(seed)
    picked = []
    total = len(data)
    for key in sorted(groups):
        g = groups[key]
        k = max(1, round(n * len(g) / total))
        picked += rng.sample(g, min(k, len(g)))
    rng.shuffle(picked)
    return picked[:n]
