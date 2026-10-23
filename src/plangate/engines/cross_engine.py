"""Cross-engine execution accuracy for PostgreSQL candidates whose gold SQL exists only in SQLite.

BIRD dev questions outside mini-dev have SQLite gold queries only. The PostgreSQL dump holds the same
data, so a PostgreSQL candidate is scored by comparing its (normalized) result set with the result of
the SQLite gold query on the SQLite copy. `validate()` measures agreement with the official
PostgreSQL gold on mini-dev, where both are available.
"""
from __future__ import annotations
import datetime as _dt
import math
from decimal import Decimal

from . import postgres, sqlite


def nv(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
        if math.isnan(f):
            return None
        # 6 significant digits: REAL columns (float4) in the PostgreSQL dump accumulate differently
        # from SQLite doubles, e.g. 402524570.17 vs 402524570.0228 for the same SUM
        # postgres.run_bounded already rounds to 4 decimals, so apply the same rounding to both sides first
        f = float(f"{round(f, 4):.6g}")
        return 0.0 if f == 0 else f
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode("utf8", errors="replace")
    return str(v)


def norm_rows(rows):
    return {tuple(nv(v) for v in r) for r in rows}


_gold = {}


def gold_sets(item):
    """Normalized SQLite gold result set plus official tied alternatives (dev only)."""
    q = item["question_id"]
    if q not in _gold:
        g = sqlite.gold_result(item)
        _gold[q] = None if g is None else [norm_rows(g)] + [norm_rows(a) for a in sqlite.gold_alternatives(item)]
    return _gold[q]


def ex_correct_cross(item, sql):
    gs = gold_sets(item)
    if not gs or not sql:
        return False
    rows, err, _, _ = postgres.run_bounded(sql, timeout_ms=30000)
    if err is not None:
        return False
    got = norm_rows(rows)
    return any(got == g for g in gs)
