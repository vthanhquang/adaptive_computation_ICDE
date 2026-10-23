"""Differential execution probe: let the DBMS compute what distinguishes two candidates.

For candidates A and B (both read-only SELECTs) it returns
  - shape of each result (columns, row count, first rows)
  - if column counts match: |A EXCEPT B|, |B EXCEPT A| and up to k rows from each difference
All statements are time-bounded and executed on a read-only connection.
"""
from __future__ import annotations
import time

from ..engines.sqlite import connect, run_bounded, is_read_only_select


def _shape(conn, sql, k, timeout_s):
    t0 = time.perf_counter()
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(5001)
        return {"ok": True, "cols": cols, "n_rows": len(rows), "more": len(rows) > 5000,
                "head": [list(map(_s, r)) for r in rows[:k]]}, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}, (time.perf_counter() - t0) * 1000
    finally:
        conn.set_progress_handler(None, 0)


def _s(v):
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def diff_probe(db_id, sql_a, sql_b, k=5, timeout_s=5.0):
    """Returns (payload, db_ms)."""
    if not (is_read_only_select(sql_a) and is_read_only_select(sql_b)):
        return {"supported": False}, 0.0
    a, b = sql_a.strip().rstrip(";"), sql_b.strip().rstrip(";")
    conn = connect(db_id)
    total = 0.0
    try:
        sa, ms = _shape(conn, a, k, timeout_s); total += ms
        sb, ms = _shape(conn, b, k, timeout_s); total += ms
        p = {"supported": True, "A": sa, "B": sb}
        if sa.get("ok") and sb.get("ok") and len(sa["cols"]) == len(sb["cols"]):
            for name, x, y in (("only_A", a, b), ("only_B", b, a)):
                q = f"SELECT * FROM ({x}) EXCEPT SELECT * FROM ({y})"
                rows, err, ms, _ = run_bounded(conn, f"SELECT * FROM ({q}) LIMIT 1001", timeout_s=timeout_s)
                total += ms
                if err:
                    p[name] = {"ok": False, "error": err[:120]}
                else:
                    p[name] = {"ok": True, "n": len(rows), "more": len(rows) > 1000,
                               "rows": [list(map(_s, r)) for r in rows[:k]]}
        return p, total
    finally:
        conn.close()


def render(p, mode):
    """Text evidence for the judge. mode: 'results' (previews) or 'diff' (previews + differences)."""
    if not p.get("supported"):
        return "(no execution evidence)"
    lines = []
    for side in ("A", "B"):
        s = p[side]
        if not s.get("ok"):
            lines.append(f"Result of SQL {side}: execution error: {s['error']}")
        else:
            n = f"{s['n_rows']}{'+' if s['more'] else ''}"
            lines.append(f"Result of SQL {side}: columns {s['cols']}, {n} rows; first rows: {s['head']}")
    if mode == "diff":
        if "only_A" in p:
            for side, key in (("A", "only_A"), ("B", "only_B")):
                d = p[key]
                if d.get("ok"):
                    n = f"{d['n']}{'+' if d['more'] else ''}"
                    lines.append(f"Rows returned by SQL {side} but NOT by the other: {n}"
                                 + (f"; examples: {d['rows']}" if d["rows"] else ""))
        elif p["A"].get("ok") and p["B"].get("ok"):
            lines.append(f"The two results have different numbers of columns "
                         f"({len(p['A']['cols'])} vs {len(p['B']['cols'])}).")
    return "\n".join(lines)
