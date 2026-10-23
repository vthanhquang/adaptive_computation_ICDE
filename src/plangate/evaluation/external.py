"""External (non-PlanGate) baselines computed directly from a replay table; they need no training.

  single models (flash, pro, qwen), self-consistency@2 with execution vote (flash x2),
  MoT-style agreement cascade (flash x2 agree -> accept, else qwen), 3-model execution-vote ensemble
  (flash, pro, qwen), rule cascade (DB trigger on flash -> qwen).
Costs: LLM units + 1 unit per DB-second (mandatory checks, and execution where a baseline needs it).
"""
from __future__ import annotations
from collections import Counter

from ..planning.adaptive import fp, visible_trigger

MAND = ("PARSE_BIND", "EXPLAIN")


def c(e, ex=False):
    ms = sum(e["db_ms"].get(op, 0) for op in MAND) + (e["db_ms"].get("EXEC_BOUNDED", 0) if ex else 0)
    return e["cost"] + ms / 1000.0, ms


def vote(entries):
    fps = [fp(e) for e in entries]
    cnt = Counter(x for x in fps if x)
    if not cnt:
        return entries[0]
    top = max(cnt.values())
    for e, x in zip(entries, fps):
        if x and cnt[x] == top:
            return e


def baselines(rows):
    """{name: [(correct, cost, db_ms) per row]}."""
    B = {}
    for name, key in (("flash", "c0"), ("pro", "SWITCH_PRO"), ("qwen", "SWITCH_QWEN")):
        B[name] = [(int(r[key]["correct"]), *c(r[key])) for r in rows]
    out = []
    for r in rows:
        a, b = r["c0"], r["RESAMPLE_1"]
        (ca, ma), (cb, mb) = c(a, True), c(b, True)
        out.append((int(vote([a, b])["correct"]), ca + cb, ma + mb))
    B["self-consistency@2"] = out
    out = []
    for r in rows:
        a, b, q = r["c0"], r["RESAMPLE_1"], r["SWITCH_QWEN"]
        (ca, ma), (cb, mb) = c(a, True), c(b, True)
        if fp(a) and fp(a) == fp(b):
            out.append((int(a["correct"]), ca + cb, ma + mb))
        else:
            cq, mq = c(q)
            out.append((int(q["correct"]), ca + cb + cq, ma + mb + mq))
    B["agreement cascade (MoT)"] = out
    out = []
    for r in rows:
        es = [r["c0"], r["SWITCH_PRO"], r["SWITCH_QWEN"]]
        cs = [c(e, True) for e in es]
        out.append((int(vote(es)["correct"]), sum(x for x, _ in cs), sum(m for _, m in cs)))
    B["ensemble-3 exec vote"] = out
    out = []
    for r in rows:
        a, q = r["c0"], r["SWITCH_QWEN"]
        ca, ma = c(a, True)
        ca += a["db_ms"].get("VALUE_PROBE", 0) / 1000.0
        ma += a["db_ms"].get("VALUE_PROBE", 0)
        if visible_trigger(a, frozenset({"VALUE_PROBE", "EXEC_BOUNDED"})):
            cq, mq = c(q)
            out.append((int(q["correct"]), ca + cq, ma + mq))
        else:
            out.append((int(a["correct"]), ca, ma))
    B["rule cascade (DB trigger)"] = out
    return B
