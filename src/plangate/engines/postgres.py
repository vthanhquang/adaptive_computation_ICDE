"""PostgreSQL engine: BIRD dev databases loaded from the official mini-dev PostgreSQL dump.

Every session is read-only and every statement is bounded by statement_timeout.
Evidence operators mirror plangate.evidence.operators (SQLite), plus optimizer-based pricing:
EXPLAIN (FORMAT JSON) gives estimated total cost / rows *before* anything executes.
The connection string defaults to a local server on port 5544 and can be set with PLANGATE_PG_DSN.
"""
from __future__ import annotations
import hashlib, json, math, os, re, threading, time
from collections import Counter
from decimal import Decimal
from functools import lru_cache

import psycopg
import sqlglot
from sqlglot import exp

from ..benchmark import load_minidev  # noqa: F401  (re-exported for engine-generic callers)
from ..paths import CACHE_DIR
from .sqlite import table_columns

DSN = os.environ.get("PLANGATE_PG_DSN", "host=localhost port=5544 user=postgres dbname=bird")
_local = threading.local()


def conn():
    c = getattr(_local, "conn", None)
    if c is None or c.closed:
        c = psycopg.connect(DSN, autocommit=True)
        c.execute("SET default_transaction_read_only = on")
        _local.conn = c
    return c


def _qi(name):
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=None)
def tables_of(db_id):
    return list(table_columns(db_id))   # same table set as the SQLite copy


@lru_cache(maxsize=None)
def pg_schema(db_id, k=3):
    """CREATE TABLE text (quoted identifiers, PG types) + k example rows per table."""
    c = psycopg.connect(DSN, autocommit=True)
    parts = []
    for t in tables_of(db_id):
        cols = c.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND lower(table_name)=lower(%s) ORDER BY ordinal_position", (t,)).fetchall()
        real = c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
                         "AND lower(table_name)=lower(%s)", (t,)).fetchone()[0]
        ddl = f"CREATE TABLE {_qi(real)} (\n" + ",\n".join(f"  {_qi(n)} {ty}" for n, ty in cols) + "\n);"
        rows = c.execute(f"SELECT * FROM {_qi(real)} LIMIT {k}").fetchall()
        sample = "\n".join("  " + ", ".join(f"{n}={_short(v)}" for (n, _), v in zip(cols, r)) for r in rows)
        parts.append(f"{ddl}\n/* {k} example rows from {real}:\n{sample}\n*/")
    c.close()
    return "\n\n".join(parts)


def _short(v):
    s = repr(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _norm(v):
    if isinstance(v, (Decimal, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    return v


def is_read_only_select(sql):
    s = (sql or "").strip().rstrip(";").strip()
    low = s.lower()
    return (low.startswith("select") or low.startswith("with")) and ";" not in s


def run_bounded(sql, timeout_ms=5000, max_rows=None, params=None):
    """(rows, error, elapsed_ms, timed_out). Rows are normalized tuples."""
    if not is_read_only_select(sql) and params is None:
        return None, "rejected: not a single read-only SELECT", 0.0, False
    c = conn()
    t0 = time.perf_counter()
    try:
        c.execute(f"SET statement_timeout = {int(timeout_ms)}")
        cur = c.execute(sql.strip().rstrip(";"), params)
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        rows = [tuple(_norm(v) for v in r) for r in rows]
        return rows, None, (time.perf_counter() - t0) * 1000, False
    except psycopg.errors.QueryCanceled:
        return None, "timeout", (time.perf_counter() - t0) * 1000, True
    except Exception as e:
        return None, str(e).split("\n")[0][:200], (time.perf_counter() - t0) * 1000, False


_gold = {}


def gold_result(item):
    q = item["question_id"]
    if q not in _gold:
        rows, err, _, _ = run_bounded(item["SQL"], timeout_ms=60000)
        _gold[q] = None if err else set(rows)
    return _gold[q]


def ex_correct(item, sql):
    g = gold_result(item)
    if g is None or not sql:
        return False
    rows, err, _, _ = run_bounded(sql, timeout_ms=30000)
    return err is None and set(rows) == g


# ------------------------------------------------------------------ evidence
def _plan_stats(node, acc):
    acc["nodes"] += 1
    t = node.get("Node Type", "")
    acc["types"][t] += 1
    if node.get("Join Type") and not node.get("Hash Cond") and not node.get("Merge Cond") \
            and not node.get("Join Filter") and t == "Nested Loop":
        acc["unconditioned_nl"] += 1
    for ch in node.get("Plans", []) or []:
        _plan_stats(ch, acc)


def explain(sql):
    """Optimizer evidence + price. Never executes the query."""
    if not is_read_only_select(sql):
        return {"valid": False, "error": "not a single read-only SELECT"}, 0.0
    c = conn()
    t0 = time.perf_counter()
    try:
        c.execute("SET statement_timeout = 5000")
        plan = c.execute("EXPLAIN (FORMAT JSON) " + sql.strip().rstrip(";")).fetchone()[0][0]["Plan"]
    except Exception as e:
        return {"valid": False, "error": str(e).split("\n")[0][:200]}, (time.perf_counter() - t0) * 1000
    acc = {"nodes": 0, "types": Counter(), "unconditioned_nl": 0}
    _plan_stats(plan, acc)
    p = {
        "valid": True, "total_cost": plan.get("Total Cost", 0.0), "plan_rows": plan.get("Plan Rows", 0),
        "nodes": acc["nodes"], "seq_scans": acc["types"]["Seq Scan"],
        "index_scans": acc["types"]["Index Scan"] + acc["types"]["Index Only Scan"] + acc["types"]["Bitmap Heap Scan"],
        "hash_joins": acc["types"]["Hash Join"], "nested_loops": acc["types"]["Nested Loop"],
        "merge_joins": acc["types"]["Merge Join"], "sorts": acc["types"]["Sort"],
        "aggregates": acc["types"]["Aggregate"] + acc["types"]["HashAggregate"] + acc["types"]["GroupAggregate"],
        "unconditioned_nl": acc["unconditioned_nl"],
    }
    return p, (time.perf_counter() - t0) * 1000


def parse_bind(sql):
    p, ms = explain(sql)
    return {"valid": p.get("valid", False), "error": p.get("error")}, ms


def _literal_predicates(sql, db_id):
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return []
    cols_by_table = {t.lower(): [c.lower() for c in cs] for t, cs in table_columns(db_id).items()}
    alias = {}
    for t in tree.find_all(exp.Table):
        alias[(t.alias_or_name or t.name).lower()] = t.name.lower()
        alias[t.name.lower()] = t.name.lower()
    out = []
    for node in tree.find_all(exp.EQ, exp.Like, exp.ILike):
        l, r = node.left, node.right
        if isinstance(l, exp.Literal) and isinstance(r, exp.Column):
            l, r = r, l
        if not (isinstance(l, exp.Column) and isinstance(r, exp.Literal) and r.is_string):
            continue
        col, tq = l.name.lower(), (l.table or "").lower()
        tables = [alias[tq]] if tq in alias else [t for t in set(alias.values()) if col in cols_by_table.get(t, [])]
        for t in tables[:1]:
            if col in cols_by_table.get(t, []):
                op = "=" if isinstance(node, exp.EQ) else ("ILIKE" if isinstance(node, exp.ILike) else "LIKE")
                out.append((t, l.name, r.this, op))
    return out[:6]


def _real_name(table, column):
    c = conn()
    t = c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
                  "AND lower(table_name)=lower(%s)", (table,)).fetchone()
    col = c.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' "
                    "AND lower(table_name)=lower(%s) AND lower(column_name)=lower(%s)", (table, column)).fetchone()
    return (t[0] if t else table), (col[0] if col else column)


def value_probe(sql, db_id):
    t0 = time.perf_counter()
    preds = _literal_predicates(sql, db_id)
    if not preds:
        return {"supported": False, "n_predicates": 0}, (time.perf_counter() - t0) * 1000
    res = []
    for t, cname, lit, op in preds:
        T, C = _real_name(t, cname)
        q = f"SELECT EXISTS(SELECT 1 FROM {_qi(T)} WHERE {_qi(C)}::text {op} %s)"
        rows, err, _, _ = run_bounded(q, 2000, params=(lit,))
        exists = bool(rows[0][0]) if rows else None
        r = {"table": T, "column": C, "literal": lit, "op": op, "exists": exists}
        if exists is False:
            near = []
            for pat in [f"%{lit.strip('%')}%"] + [f"%{w}%" for w in sorted(set(
                    re.findall(r"[A-Za-z0-9]{4,}", lit)), key=len, reverse=True)[:3]]:
                rows, _, _, _ = run_bounded(f"SELECT DISTINCT {_qi(C)}::text FROM {_qi(T)} "
                                            f"WHERE {_qi(C)}::text ILIKE %s LIMIT 5", 2000, params=(pat,))
                for v in rows or []:
                    if v[0] not in near:
                        near.append(v[0])
                if len(near) >= 5:
                    break
            r["near_values"] = near[:5]
        res.append(r)
    return {"supported": True, "n_predicates": len(res), "n_missing": sum(r["exists"] is False for r in res),
            "probes": res}, (time.perf_counter() - t0) * 1000


def exec_bounded(sql, timeout_ms=5000, max_rows=1000):
    rows, err, ms, to = run_bounded(sql, timeout_ms, max_rows + 1)
    if err:
        return {"ok": False, "error": err[:200], "timed_out": to}, ms
    n = len(rows)
    cells = [v for r in rows[:max_rows] for v in r]
    ns = (sum(v is None for v in cells) / len(cells)) if cells else 0.0
    return {"ok": True, "n_rows": min(n, max_rows), "truncated": n > max_rows, "n_cols": len(rows[0]) if rows else 0,
            "empty": n == 0, "all_null": bool(cells) and ns == 1.0, "null_share": round(ns, 3),
            "dup_rows": n - len(set(rows)) if n <= max_rows else None,
            "preview": [[str(v)[:30] for v in r] for r in rows[:3]],
            "fingerprint": hashlib.md5(repr(sorted(set(rows[:max_rows]), key=repr)).encode()).hexdigest()}, ms


def diff_probe(sql_a, sql_b, k=5, timeout_ms=5000):
    if not (is_read_only_select(sql_a) and is_read_only_select(sql_b)):
        return {"supported": False}, 0.0
    a, b = sql_a.strip().rstrip(";"), sql_b.strip().rstrip(";")
    total, p = 0.0, {"supported": True}
    for side, s in (("A", a), ("B", b)):
        c = conn()
        t0 = time.perf_counter()
        try:
            c.execute(f"SET statement_timeout = {timeout_ms}")
            cur = c.execute(s)
            cols = [d.name for d in cur.description]
            rows = cur.fetchmany(5001)
            p[side] = {"ok": True, "cols": cols, "n_rows": len(rows), "more": len(rows) > 5000,
                       "head": [[str(v)[:40] for v in r] for r in rows[:k]]}
        except Exception as e:
            p[side] = {"ok": False, "error": str(e).split("\n")[0][:150]}
        total += (time.perf_counter() - t0) * 1000
    if p["A"].get("ok") and p["B"].get("ok") and len(p["A"]["cols"]) == len(p["B"]["cols"]):
        for name, x, y in (("only_A", a, b), ("only_B", b, a)):
            rows, err, ms, _ = run_bounded(f"SELECT * FROM (SELECT * FROM ({x}) _a EXCEPT SELECT * FROM ({y}) _b) _d "
                                           f"LIMIT 1001", timeout_ms, params=None)
            total += ms
            p[name] = {"ok": False, "error": err[:120]} if err else {
                "ok": True, "n": len(rows), "more": len(rows) > 1000, "rows": [[str(v)[:40] for v in r] for r in rows[:k]]}
    return p, total


OPS = ("PARSE_BIND", "EXPLAIN", "VALUE_PROBE", "EXEC_BOUNDED")
CACHE = CACHE_DIR / "pg_evidence"
_locks, _g = {}, threading.Lock()


def get_evidence(op, db_id, sql):
    """Returns (payload, db_ms), cached on disk by (op, db_id, sql). Evidence is deterministic."""
    k = hashlib.sha256(json.dumps([op, db_id, sql]).encode()).hexdigest()
    path = CACHE / f"{k}.json"
    with _g:
        lock = _locks.setdefault(k, threading.Lock())
    with lock:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf8"))
            return d["payload"], d["db_ms"]
        if op == "PARSE_BIND":
            p, ms = parse_bind(sql)
        elif op == "EXPLAIN":
            p, ms = explain(sql)
        elif op == "VALUE_PROBE":
            p, ms = value_probe(sql, db_id)
        else:
            p, ms = exec_bounded(sql)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"payload": p, "db_ms": ms}, default=str), encoding="utf8")
        os.replace(tmp, path)
        return p, ms
