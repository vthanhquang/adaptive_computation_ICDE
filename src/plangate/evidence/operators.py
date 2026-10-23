"""DBMS evidence operators over the BIRD SQLite databases.

Each operator inspects ONE candidate and returns (payload, db_ms). The planner decides
which (if any) to run; nothing here chains operators.

  PARSE_BIND   : prepare via EXPLAIN QUERY PLAN -> syntax / unknown table / column errors
  EXPLAIN      : plan structure features (full scans, temp b-trees, cartesian-style joins)
  VALUE_PROBE  : micro-probes on literal predicates: does `col = 'lit'` have any match?
                 if not, which near values exist (case/LIKE variants)?
  EXEC_BOUNDED : time-bounded execution, returns only result *shape* (rows, null share,
                 duplicates) -- never compared to gold.
"""
from __future__ import annotations
import hashlib, re, time
import sqlglot
from sqlglot import exp

from ..engines.sqlite import connect, run_bounded, table_columns, quote_ident as _q


def parse_bind(conn, sql):
    t0 = time.perf_counter()
    if not sql.strip():
        return {"valid": False, "error": "empty SQL"}, 0.0
    try:
        conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        p = {"valid": True, "error": None}
    except Exception as e:
        p = {"valid": False, "error": str(e)[:200]}
    return p, (time.perf_counter() - t0) * 1000


def explain(conn, sql):
    t0 = time.perf_counter()
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    except Exception as e:
        return {"valid": False, "error": str(e)[:200]}, (time.perf_counter() - t0) * 1000
    details = [str(r[3]).upper() for r in rows]
    full_scans = sum(d.startswith("SCAN") and "USING" not in d for d in details)
    searches = sum(d.startswith("SEARCH") for d in details)
    temps = sum("TEMP B-TREE" in d for d in details)
    n_tables_plan = sum(d.startswith(("SCAN", "SEARCH")) for d in details)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        joins = list(tree.find_all(exp.Join))
        joins_without_on = sum(1 for j in joins if not j.args.get("on") and not j.args.get("using"))
        n_from_tables = len(list(tree.find_all(exp.Table)))
    except Exception:
        joins_without_on, n_from_tables = 0, n_tables_plan
    p = {
        "valid": True,
        "n_nodes": len(details),
        "full_scans": full_scans,
        "searches": searches,
        "temp_btrees": temps,
        "joins_without_on": joins_without_on,
        "multi_full_scan": full_scans >= 2,
        "suspicious_plan": joins_without_on > 0 or (full_scans >= 2 and searches == 0 and n_from_tables >= 2),
        "details": details[:12],
    }
    return p, (time.perf_counter() - t0) * 1000


def _literal_predicates(sql, db_id):
    """(table, column, literal, op) for `col = 'text'` / `col LIKE 'text'` predicates."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    cols_by_table = table_columns(db_id)
    alias = {}
    for t in tree.find_all(exp.Table):
        alias[(t.alias_or_name or t.name).lower()] = t.name
        alias[t.name.lower()] = t.name
    out = []
    for node in tree.find_all(exp.EQ, exp.Like):
        l, r = node.left, node.right
        if isinstance(l, exp.Literal) and isinstance(r, exp.Column):
            l, r = r, l
        if not (isinstance(l, exp.Column) and isinstance(r, exp.Literal) and r.is_string):
            continue
        col, tq = l.name, (l.table or "").lower()
        if tq and tq in alias:
            tables = [alias[tq]]
        else:
            tables = [t for t in set(alias.values()) if t in cols_by_table and
                      any(c.lower() == col.lower() for c in cols_by_table[t])]
        for t in tables[:1]:
            if t in cols_by_table and any(c.lower() == col.lower() for c in cols_by_table[t]):
                out.append((t, col, r.this, "LIKE" if isinstance(node, exp.Like) else "="))
    return out[:6]


def value_probe(conn, sql, db_id, timeout_s=2.0):
    t0 = time.perf_counter()
    preds = _literal_predicates(sql, db_id)
    if not preds:
        return {"supported": False, "n_predicates": 0}, (time.perf_counter() - t0) * 1000
    results = []
    for t, c, lit, op in preds:
        probe = f"SELECT EXISTS(SELECT 1 FROM {_q(t)} WHERE {_q(c)} {op} ?)"
        conn.set_progress_handler(None, 0)
        rows, err, _, _ = _run_param(conn, probe, (lit,), timeout_s)
        exists = bool(rows[0][0]) if rows else None
        r = {"table": t, "column": c, "literal": lit, "op": op, "exists": exists}
        if exists is False:
            like = f"%{lit.strip('%')}%"
            near, _, _, _ = _run_param(
                conn, f"SELECT DISTINCT {_q(c)} FROM {_q(t)} WHERE {_q(c)} LIKE ? LIMIT 5", (like,), timeout_s)
            r["near_values"] = [str(v[0]) for v in (near or [])][:5]
            # fall back to per-word matches, longest words first
            for w in sorted(set(re.findall(r"[A-Za-z0-9]{4,}", lit)), key=len, reverse=True)[:3]:
                if len(r["near_values"]) >= 5:
                    break
                near, _, _, _ = _run_param(
                    conn, f"SELECT DISTINCT {_q(c)} FROM {_q(t)} WHERE {_q(c)} LIKE ? LIMIT 5", (f"%{w}%",), timeout_s)
                for v in near or []:
                    if str(v[0]) not in r["near_values"]:
                        r["near_values"].append(str(v[0]))
            r["near_values"] = r["near_values"][:5]
            if not r["near_values"]:
                top, _, _, _ = _run_param(
                    conn, f"SELECT {_q(c)} FROM {_q(t)} GROUP BY {_q(c)} ORDER BY COUNT(*) DESC LIMIT 5", (), timeout_s)
                r["frequent_values"] = [str(v[0]) for v in (top or [])][:5]
        results.append(r)
    p = {"supported": True, "n_predicates": len(results),
         "n_missing": sum(1 for r in results if r["exists"] is False), "probes": results}
    return p, (time.perf_counter() - t0) * 1000


def _run_param(conn, sql, params, timeout_s):
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    try:
        return conn.execute(sql, params).fetchall(), None, 0, False
    except Exception as e:
        return None, str(e), 0, False
    finally:
        conn.set_progress_handler(None, 0)


def exec_bounded(conn, sql, timeout_s=5.0, max_rows=1000):
    rows, err, ms, timed_out = run_bounded(conn, sql, timeout_s=timeout_s, max_rows=max_rows + 1)
    if err:
        return {"ok": False, "error": err[:200], "timed_out": timed_out}, ms
    n = len(rows)
    ncols = len(rows[0]) if rows else 0
    cells = [v for r in rows[:max_rows] for v in r]
    null_share = (sum(v is None for v in cells) / len(cells)) if cells else 0.0
    p = {
        "ok": True, "n_rows": min(n, max_rows), "truncated": n > max_rows, "n_cols": ncols,
        "empty": n == 0, "all_null": bool(cells) and null_share == 1.0, "null_share": round(null_share, 3),
        "dup_rows": n - len(set(rows)) if n <= max_rows else None,
        "preview": [list(map(lambda v: str(v)[:30], r)) for r in rows[:3]],
        # order-insensitive result fingerprint, used for cross-candidate agreement
        "fingerprint": hashlib.md5(repr(sorted(set(rows[:max_rows]), key=repr)).encode()).hexdigest(),
    }
    return p, ms


def value_fix(sql, vp):
    """DB-only rewrite (no LLM): replace a non-existent literal with the DB value it most
    plausibly meant -- a case-insensitive exact match, or the single near value found.
    Returns the rewritten SQL, or None if no safe rewrite exists."""
    if not vp or not vp.get("supported"):
        return None
    out, changed = sql, False
    for r in vp["probes"]:
        if r["exists"] is not False:
            continue
        lit, near = r["literal"], r.get("near_values") or []
        core = lit.strip("%").lower()
        exact = [v for v in near if v.lower() == core]
        if exact:
            new = exact[0] if r["op"] == "=" else lit.replace(lit.strip("%"), exact[0])
        elif len(near) == 1 and r["op"] == "=":
            new = near[0]
        else:
            continue
        for quote_old in (f"'{lit}'", f'"{lit}"'):
            if quote_old in out:
                out = out.replace(quote_old, "'" + new.replace("'", "''") + "'")
                changed = True
                break
    return out if changed else None


DB_OPS = ("PARSE_BIND", "EXPLAIN", "VALUE_PROBE", "EXEC_BOUNDED")


def run_db_op(kind, db_id, sql):
    conn = connect(db_id)
    try:
        if kind == "PARSE_BIND":
            return parse_bind(conn, sql)
        if kind == "EXPLAIN":
            return explain(conn, sql)
        if kind == "VALUE_PROBE":
            return value_probe(conn, sql, db_id)
        if kind == "EXEC_BOUNDED":
            return exec_bounded(conn, sql)
        raise ValueError(kind)
    finally:
        conn.close()


def evidence_to_feedback(evidence):
    """Render collected evidence as text for a REPAIR prompt."""
    lines = []
    for kind, p in evidence.items():
        if kind == "PARSE_BIND" and not p.get("valid"):
            lines.append(f"- Parse/bind error: {p.get('error')}")
        elif kind == "EXPLAIN" and p.get("valid"):
            if p.get("joins_without_on"):
                lines.append("- Query plan: a JOIN has no join condition (cartesian product).")
            elif p.get("suspicious_plan"):
                lines.append("- Query plan: multiple full table scans with no index lookups; check join conditions.")
        elif kind == "VALUE_PROBE" and p.get("supported"):
            for r in p["probes"]:
                if r["exists"] is False:
                    alt = r.get("near_values") or r.get("frequent_values") or []
                    lines.append(f"- Value probe: no row has {r['table']}.{r['column']} {r['op']} '{r['literal']}'."
                                 + (f" Existing values include: {alt}." if alt else ""))
        elif kind == "EXEC_BOUNDED":
            if not p.get("ok"):
                lines.append(f"- Execution failed: {p.get('error')}")
            elif p.get("empty"):
                lines.append("- Execution returned 0 rows.")
            elif p.get("all_null"):
                lines.append("- Execution returned only NULL values.")
            else:
                lines.append(f"- Execution returned {p['n_rows']} rows x {p['n_cols']} columns; first rows: {p['preview']}")
    return "\n".join(lines) if lines else "- No errors were detected."
