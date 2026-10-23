"""SQLite engine: read-only sandbox, schema rendering, bounded execution and BIRD EX scoring."""
from __future__ import annotations
import json, sqlite3, time
from collections import defaultdict
from functools import lru_cache

from ..benchmark import BIRD_DEV_DIR, BIRD_TRAIN_DIR


def db_path(db_id):
    p = BIRD_DEV_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"
    if p.exists():
        return p
    # BIRD train databases (disjoint database names from dev)
    return BIRD_TRAIN_DIR / "train_databases" / db_id / f"{db_id}.sqlite"


def connect(db_id):
    # Read-only URI connection: generated SQL can never mutate the benchmark DB.
    uri = "file:" + db_path(db_id).as_posix() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf8", errors="replace")
    return conn


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=None)
def schema_ddl(db_id):
    conn = connect(db_id)
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    conn.close()
    return "\n\n".join(sql.strip() + ";" for _, sql in rows if sql)


@lru_cache(maxsize=None)
def schema_with_samples(db_id, k=3):
    """DDL plus k example rows per table (schema-aware prompt profile)."""
    conn = connect(db_id)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    parts = []
    for t in tables:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0]
        cur = conn.execute(f"SELECT * FROM {quote_ident(t)} LIMIT {k}")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        sample = "\n".join(
            "  " + ", ".join(f"{c}={_short(v)}" for c, v in zip(cols, r)) for r in rows)
        parts.append(f"{ddl.strip()};\n/* {k} example rows from {t}:\n{sample}\n*/")
    conn.close()
    return "\n\n".join(parts)


def _short(v):
    s = repr(v)
    return s if len(s) <= 40 else s[:37] + "..."


@lru_cache(maxsize=None)
def table_columns(db_id):
    conn = connect(db_id)
    out = {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        out[t] = [r[1] for r in conn.execute(f"PRAGMA table_info({quote_ident(t)})")]
    conn.close()
    return out


def is_read_only_select(sql):
    s = sql.strip().rstrip(";").strip()
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    if ";" in s:
        return False
    return True


def run_bounded(conn, sql, timeout_s=10.0, max_rows=None):
    """Execute with a wall-clock timeout. Returns (rows, error, elapsed_ms, timed_out)."""
    if not is_read_only_select(sql):
        return None, "rejected: not a single read-only SELECT", 0.0, False
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    t0 = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        return rows, None, (time.perf_counter() - t0) * 1000, False
    except sqlite3.OperationalError as e:
        timed_out = "interrupted" in str(e)
        return None, str(e), (time.perf_counter() - t0) * 1000, timed_out
    except Exception as e:
        return None, str(e), (time.perf_counter() - t0) * 1000, False
    finally:
        conn.set_progress_handler(None, 0)


_gold_cache = {}
_alt_cache = {}


@lru_cache(maxsize=None)
def _tied_gold():
    """Official BIRD alternative gold SQL for questions with tied answers."""
    p = BIRD_DEV_DIR / "dev_tied_append.json"
    out = defaultdict(list)
    if p.exists():
        for x in json.loads(p.read_text(encoding="utf8")):
            out[x["question_id"]].append(x["SQL"])
    return out


def gold_result(item):
    qid = item["question_id"]
    if qid not in _gold_cache:
        conn = connect(item["db_id"])
        rows, err, _, _ = run_bounded(conn, item["SQL"], timeout_s=60)
        conn.close()
        _gold_cache[qid] = None if err else set(rows)
    return _gold_cache[qid]


def gold_alternatives(item):
    qid = item["question_id"]
    if qid not in _alt_cache:
        alts = []
        conn = connect(item["db_id"])
        for sql in _tied_gold().get(qid, []):
            rows, err, _, _ = run_bounded(conn, sql, timeout_s=60)
            if not err:
                alts.append(set(rows))
        conn.close()
        _alt_cache[qid] = alts
    return _alt_cache[qid]


def ex_correct(item, sql, timeout_s=30):
    """BIRD execution accuracy: set equality of result rows (with official tied alternatives)."""
    gold = gold_result(item)
    if gold is None or not sql:
        return False
    conn = connect(item["db_id"])
    rows, err, _, _ = run_bounded(conn, sql, timeout_s=timeout_s)
    conn.close()
    if err is not None:
        return False
    got = set(rows)
    return got == gold or any(got == a for a in gold_alternatives(item))
