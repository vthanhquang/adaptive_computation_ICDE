"""Disk cache for SQLite DB evidence keyed by (op, db_id, sql). Evidence is deterministic."""
from __future__ import annotations
import hashlib, json, os, threading

from ..paths import CACHE_DIR
from .operators import run_db_op

CACHE = CACHE_DIR / "evidence"
_lock = threading.Lock()
_key_locks = {}


def get_evidence(op, db_id, sql):
    """Returns (payload, db_ms). db_ms is the measured cost of the first real run."""
    k = hashlib.sha256(json.dumps([op, db_id, sql]).encode("utf8")).hexdigest()
    path = CACHE / f"{k}.json"
    with _lock:
        key_lock = _key_locks.setdefault(k, threading.Lock())
    with key_lock:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf8"))
            return d["payload"], d["db_ms"]
        payload, ms = run_db_op(op, db_id, sql)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"payload": payload, "db_ms": ms}, ensure_ascii=False, default=str), encoding="utf8")
        os.replace(tmp, path)
        return payload, ms
