"""Re-implementation of SQLens (Gong et al., NeurIPS 2025; arXiv 2506.04494) on BIRD / SQLite.

Error detection: 9 database-based signals (Appendix B definitions) + 5 LLM-based signals,
aggregated by an unsupervised generative label model (weak supervision, fit on unlabeled dev
queries) and a downstream classifier trained on its soft labels.
Correction (Algorithm 1): Error Selector -> Error Fixer (with parser feedback) for up to
MAX_ITER iterations, then SQL Auditor chooses between the original and revised query.

The 5 LLM signals are elicited either in one structured call (`llm_signals`) or with one prompt
per signal as in the paper (`llm_signals_separate`). Value ambiguity uses an exact-match search over
text columns. The implementation follows the paper's description; the authors' code is not used.
"""
from __future__ import annotations
import json, re, time
from functools import lru_cache

import networkx as nx
import numpy as np
import sqlglot
from sqlglot import exp

from ..engines.sqlite import connect, run_bounded, schema_with_samples
from ..llm.generation import cached_chat, cost_units

MAX_ITER = 2
SUBQUERY_LIMIT = 3

DB_SIGNALS = ["suboptimal_join_tree", "incorrect_join_predicate", "empty_predicate", "abnormal_result",
              "incorrect_filter_in_subquery", "incorrect_group_by", "unnecessary_subquery",
              "value_ambiguity", "table_similarity"]
LLM_SIGNALS = ["evidence_violation", "insufficient_evidence", "low_confidence_link", "column_ambiguity",
               "llm_self_check"]
SIGNALS = DB_SIGNALS + LLM_SIGNALS


# ------------------------------------------------------------------ schema graph
@lru_cache(maxsize=None)
def _schema(db_id):
    conn = connect(db_id)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    cols, pks, fks, types = {}, {}, [], {}
    for t in tables:
        info = list(conn.execute(f'PRAGMA table_info("{t}")'))
        cols[t] = [r[1] for r in info]
        types[t] = {r[1]: (r[2] or "").upper() for r in info}
        pks[t] = [r[1] for r in info if r[5]]
    for t in tables:
        for row in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
            ref, frm, to = row[2], row[3], row[4]
            if to is None:
                to = (pks.get(ref) or [frm])[0]
            fks.append((t, frm, ref, to))
    conn.close()
    g = nx.Graph()
    g.add_nodes_from(tables)
    for t, _, ref, _ in fks:
        if ref in cols:
            g.add_edge(t, ref)
    lower = {t.lower(): t for t in tables}
    return {"tables": tables, "cols": cols, "pks": pks, "fks": fks, "types": types, "graph": g, "lower": lower}


def _tables_and_aliases(tree, S):
    alias = {}
    for t in tree.find_all(exp.Table):
        real = S["lower"].get(t.name.lower())
        if real:
            alias[(t.alias_or_name or t.name).lower()] = real
            alias[t.name.lower()] = real
    return alias


def _resolve(col, alias, S, used_tables):
    tq = (col.table or "").lower()
    if tq in alias:
        return alias[tq]
    cands = [t for t in used_tables if col.name.lower() in (c.lower() for c in S["cols"][t])]
    return cands[0] if len(cands) == 1 else None


# ------------------------------------------------------------------ DB-based signals
def db_signals(db_id, sql):
    """Returns ({signal: 0/1}, details, db_ms). 1 = signal fires (potential error)."""
    S = _schema(db_id)
    t0 = time.perf_counter()
    out = {s: 0 for s in DB_SIGNALS}
    details = {}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        tree = None
    conn = connect(db_id)
    try:
        # abnormal result: empty, all-zero column, or all-NULL column
        rows, err, _, _ = run_bounded(conn, sql, timeout_s=10, max_rows=2000)
        if err is None:
            if not rows:
                out["abnormal_result"] = 1; details["abnormal_result"] = "empty result"
            else:
                for j in range(len(rows[0])):
                    colv = [r[j] for r in rows]
                    if all(v is None for v in colv) or all(v == 0 for v in colv):
                        out["abnormal_result"] = 1
                        details["abnormal_result"] = f"result column {j+1} is all NULL/zero"
                        break
        else:
            out["abnormal_result"] = 1; details["abnormal_result"] = f"execution error: {err[:120]}"
        if tree is None:
            return out, details, (time.perf_counter() - t0) * 1000

        alias = _tables_and_aliases(tree, S)
        used = sorted(set(alias.values()))
        n_select = len(list(tree.find_all(exp.Select)))
        if n_select - 1 > SUBQUERY_LIMIT:
            out["unnecessary_subquery"] = 1; details["unnecessary_subquery"] = f"{n_select-1} subqueries"

        # incorrect GROUP BY: GROUP BY without any aggregate function
        for sel in tree.find_all(exp.Select):
            if sel.args.get("group") and not list(sel.find_all(exp.AggFunc)):
                out["incorrect_group_by"] = 1; details["incorrect_group_by"] = "GROUP BY without aggregate"

        # incorrect join predicate: equality must be a PK-FK pair or two FKs to the same key
        fk_pairs = set()
        for t, c, rt, rc in S["fks"]:
            fk_pairs.add((t.lower(), c.lower(), rt.lower(), rc.lower()))
            fk_pairs.add((rt.lower(), rc.lower(), t.lower(), c.lower()))
        fk_target = {(t.lower(), c.lower()): (rt.lower(), rc.lower()) for t, c, rt, rc in S["fks"]}
        for j in tree.find_all(exp.Join):
            on = j.args.get("on")
            if on is None:
                continue
            for eq in on.find_all(exp.EQ):
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                    a = _resolve(eq.left, alias, S, used); b = _resolve(eq.right, alias, S, used)
                    if not a or not b:
                        continue
                    ka = (a.lower(), eq.left.name.lower()); kb = (b.lower(), eq.right.name.lower())
                    ok = (ka + kb) in fk_pairs or (fk_target.get(ka) and fk_target.get(ka) == fk_target.get(kb)) \
                        or fk_target.get(ka) == kb or fk_target.get(kb) == ka
                    if not ok:
                        out["incorrect_join_predicate"] = 1
                        details["incorrect_join_predicate"] = f"{a}.{eq.left.name} = {b}.{eq.right.name}"

        # suboptimal join tree: more tables than the minimum Steiner tree over required tables
        join_cols = set()
        for j in tree.find_all(exp.Join):
            on = j.args.get("on")
            if on is not None:
                for c in on.find_all(exp.Column):
                    join_cols.add(id(c))
        required = set()
        for c in tree.find_all(exp.Column):
            if id(c) not in join_cols:
                t = _resolve(c, alias, S, used)
                if t:
                    required.add(t)
        if len(used) > 1 and required and all(t in S["graph"] for t in required):
            req = sorted(required)
            try:
                if len(req) == 1:
                    min_n = 1
                else:
                    st = nx.algorithms.approximation.steiner_tree(S["graph"], req)
                    min_n = st.number_of_nodes()
                if len(used) > min_n:
                    out["suboptimal_join_tree"] = 1
                    details["suboptimal_join_tree"] = f"uses {len(used)} tables, {min_n} suffice ({req})"
            except Exception:
                pass

        # empty predicate + value ambiguity on column = 'literal' / LIKE predicates
        for node in tree.find_all(exp.EQ, exp.Like):
            l, r = node.left, node.right
            if isinstance(l, exp.Literal) and isinstance(r, exp.Column):
                l, r = r, l
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Literal)):
                continue
            t = _resolve(l, alias, S, used)
            if not t:
                continue
            op = "LIKE" if isinstance(node, exp.Like) else "="
            lit = r.this if r.is_string else r.this
            q = f'SELECT EXISTS(SELECT 1 FROM "{t}" WHERE "{l.name}" {op} ?)'
            res, e, _, _ = _param(conn, q, (lit,))
            if res is not None and not res[0][0]:
                out["empty_predicate"] = 1
                details["empty_predicate"] = f"{t}.{l.name} {op} {lit!r} matches no row"
            if r.is_string and op == "=":
                others = []
                near = set(used)
                for u in used:
                    near |= set(S["graph"].neighbors(u)) if u in S["graph"] else set()
                for t2 in sorted(near):
                    for c2 in S["cols"][t2]:
                        if (t2, c2.lower()) == (t, l.name.lower()) or "TEXT" not in S["types"][t2].get(c2, "TEXT"):
                            continue
                        res2, _, _, _ = _param(conn, f'SELECT EXISTS(SELECT 1 FROM "{t2}" WHERE "{c2}" = ?)',
                                               (lit,), timeout_s=0.5)
                        if res2 and res2[0][0]:
                            others.append(f"{t2}.{c2}")
                    if len(others) >= 3:
                        break
                if others:
                    out["value_ambiguity"] = 1
                    details["value_ambiguity"] = f"value {lit!r} also appears in {others[:3]}"

        # incorrect filter in subquery: column = (SELECT ...) returning more than one row
        for eq in tree.find_all(exp.EQ):
            for side in (eq.left, eq.right):
                sub = side.this if isinstance(side, exp.Subquery) else None
                if isinstance(sub, exp.Select):
                    r2, e2, _, _ = run_bounded(conn, sub.sql(dialect="sqlite"), timeout_s=5, max_rows=2)
                    if r2 is not None and len(r2) > 1:
                        out["incorrect_filter_in_subquery"] = 1
                        details["incorrect_filter_in_subquery"] = "scalar comparison with a multi-row subquery"

        # table similarity: another table contains all referenced columns of a used table
        ref_cols = {}
        for c in tree.find_all(exp.Column):
            t = _resolve(c, alias, S, used)
            if t:
                ref_cols.setdefault(t, set()).add(c.name.lower())
        for t, cs in ref_cols.items():
            for t2 in S["tables"]:
                if t2 != t and len(cs) >= 2 and cs <= {x.lower() for x in S["cols"][t2]}:
                    out["table_similarity"] = 1
                    details["table_similarity"] = f"{t2} has the same columns {sorted(cs)} as {t}"
    finally:
        conn.close()
    return out, details, (time.perf_counter() - t0) * 1000


def _param(conn, sql, params, timeout_s=2.0):
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    try:
        return conn.execute(sql, params).fetchall(), None, 0, False
    except Exception as e:
        return None, str(e), 0, False
    finally:
        conn.set_progress_handler(None, 0)


# ------------------------------------------------------------------ LLM-based signals (one structured call)
LLM_SIGNAL_PROMPT = """Database schema:
{schema}

Question: {question}
Evidence (external knowledge): {evidence}

Proposed SQL query:
```sql
{sql}
```

Answer the following checks about the proposed SQL query and return ONLY a JSON object:
1. "evidence_violation": does the SQL fail to reflect all the evidence specified in the question? (true/false)
2. "insufficient_evidence": do you lack sufficient evidence to determine whether the SQL answers the question correctly? (true/false)
3. "links": map each concept of the question to the SQL clause that implements it; for each link say whether you are confident in the generated clause: a list of {{"concept": ..., "clause": ..., "confident": true/false}}
4. "column_ambiguity": are there other similar columns in the schema that could also be used to answer the question instead of the ones used? (true/false)
5. "correct": is the SQL query correct for the question? (true/false)
6. "confidence": your confidence (0 to 1) that the SQL query is correct."""


def llm_signals(item, sql, model):
    body = LLM_SIGNAL_PROMPT.format(schema=schema_with_samples(item["db_id"]), question=item["question"],
                                    evidence=item.get("evidence") or "None", sql=sql)
    d = cached_chat(model, [{"role": "system", "content": "You are a careful SQL reviewer."},
                            {"role": "user", "content": body}], max_tokens=1500)
    txt = d["text"] or ""
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    links = j.get("links") or []
    low = sum(1 for x in links if isinstance(x, dict) and x.get("confident") is False)
    conf = j.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else (0.5 if j.get("correct") is None else float(bool(j.get("correct"))))
    sig = {"evidence_violation": int(bool(j.get("evidence_violation"))),
           "insufficient_evidence": int(bool(j.get("insufficient_evidence"))),
           "low_confidence_link": int(low > 0),
           "column_ambiguity": int(bool(j.get("column_ambiguity"))),
           "llm_self_check": int(j.get("correct") is False or conf < 0.5)}
    details = {"low_confidence_links": [x for x in links if isinstance(x, dict) and x.get("confident") is False][:3],
               "self_confidence": conf}
    return sig, details, cost_units(model, d["usage"]), d["latency_ms"]


# ------------------------------------------------------------------ LLM-based signals, one prompt each (paper setting)
_SEP_HEAD = """Database schema:
{schema}

Question: {question}
Evidence (external knowledge): {evidence}

Proposed SQL query:
```sql
{sql}
```

"""
SEPARATE_PROMPTS = {
    "evidence_violation": "Determine whether the proposed SQL query reflects all the evidence specified in the question. "
                          "Answer on the last line with exactly 'VIOLATION: yes' if some evidence is not reflected, otherwise 'VIOLATION: no'.",
    "insufficient_evidence": "Do you have sufficient evidence to determine whether the SQL answers the user question correctly? "
                             "Answer on the last line with exactly 'SUFFICIENT: yes' or 'SUFFICIENT: no'.",
    "low_confidence_link": "Map each concept in the question to the SQL clause that implements it. For each link identified, "
                           "indicate whether you are confident in the generated clause by answering yes or no, one line per link "
                           "in the form 'LINK: <concept> -> <clause> | CONFIDENT: yes/no'.",
    "column_ambiguity": "Are there other similar columns in the schema that could also be used to answer the question instead of "
                        "the columns used in the SQL? Answer on the last line with exactly 'AMBIGUOUS: yes' or 'AMBIGUOUS: no'.",
    "llm_self_check": "Is the proposed SQL query correct for the question? Give your confidence that it is correct as a number "
                      "between 0 and 1 on the last line in the form 'CONFIDENCE: <number>'.",
}


def llm_signals_separate(item, sql, model):
    """The 5 LLM-based signals elicited with one prompt each, as in the paper."""
    head = _SEP_HEAD.format(schema=schema_with_samples(item["db_id"]), question=item["question"],
                            evidence=item.get("evidence") or "None", sql=sql)
    sig, cost, lat, details = {}, 0.0, 0.0, {}
    for name, instr in SEPARATE_PROMPTS.items():
        d = cached_chat(model, [{"role": "system", "content": "You are a careful SQL reviewer."},
                                {"role": "user", "content": head + instr}], max_tokens=800)
        t = d["text"] or ""
        cost += cost_units(model, d["usage"]); lat = max(lat, d["latency_ms"])   # prompts run in parallel
        if name == "evidence_violation":
            sig[name] = int(bool(re.search(r"VIOLATION:\s*yes", t, re.I)))
        elif name == "insufficient_evidence":
            sig[name] = int(bool(re.search(r"SUFFICIENT:\s*no", t, re.I)))
        elif name == "low_confidence_link":
            low = re.findall(r"CONFIDENT:\s*no", t, re.I)
            sig[name] = int(len(low) > 0)
            details["low_confidence_links"] = re.findall(r"LINK:.*CONFIDENT:\s*no", t, re.I)[:3]
        elif name == "column_ambiguity":
            sig[name] = int(bool(re.search(r"AMBIGUOUS:\s*yes", t, re.I)))
        else:
            m = re.findall(r"CONFIDENCE:\s*([01](?:\.\d+)?)", t)
            conf = float(m[-1]) if m else 0.5
            details["self_confidence"] = conf
            sig[name] = int(conf < 0.5)
    return sig, details, cost, lat


# ------------------------------------------------------------------ weak supervision (unsupervised label model)
class LabelModel:
    """Binary generative label model with conditionally independent labeling functions (EM).

    Each signal is a labeling function voting 'error' (1) or abstaining (0); per-LF firing
    probabilities are estimated separately for erroneous and correct queries, plus a class prior.
    Fit on unlabeled queries only (as in SQLens)."""

    def fit(self, L, iters=200):
        n, m = L.shape
        self.pi = 0.4
        self.a = np.full(m, 0.6)    # P(fire | error)
        self.b = np.full(m, 0.2)    # P(fire | correct)
        for _ in range(iters):
            q = self.posterior(L)
            self.pi = np.clip(q.mean(), 0.05, 0.95)
            self.a = np.clip((q[:, None] * L).sum(0) / q.sum(), 1e-3, 1 - 1e-3)
            self.b = np.clip(((1 - q)[:, None] * L).sum(0) / (1 - q).sum(), 1e-3, 1 - 1e-3)
            # identifiability: an error signal must fire more often on errors than on correct queries
            swap = self.a < self.b
            self.a[swap], self.b[swap] = self.b[swap], self.a[swap]
        return self

    def posterior(self, L):
        la = np.log(self.pi) + (L * np.log(self.a) + (1 - L) * np.log(1 - self.a)).sum(1)
        lb = np.log(1 - self.pi) + (L * np.log(self.b) + (1 - L) * np.log(1 - self.b)).sum(1)
        return 1 / (1 + np.exp(lb - la))


# ------------------------------------------------------------------ correction loop (Algorithm 1)
SELECTOR = """Question: {question}
Evidence: {evidence}

SQL query:
```sql
{sql}
```

The following potential errors were detected in the SQL query:
{errors}

Rank these errors by how likely they are to be real and relevant to answering the question. Reply with ONLY the number of the single most important error to fix first."""

FIXER = """Database schema:
{schema}

Question: {question}
Evidence: {evidence}

SQL query:
```sql
{sql}
```

Error report:
{report}

Analyze the error report and decide whether the SQL query needs fixing. If it does, generate the corrected SQL query; otherwise return the original query.
Return only the final SQL inside one ```sql code block."""

AUDITOR = """Database schema:
{schema}

Question: {question}
Evidence: {evidence}

Original SQL:
```sql
{a}
```

Revised SQL:
```sql
{b}
```

Which query best aligns with the question? Reply with ONLY "original" or "revised"."""


def _sqlblock(t):
    m = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", t or "", re.S | re.I)
    return (m[-1] if m else (t or "")).strip().rstrip(";")


def describe(sig, det, dtl):
    msgs = {
        "suboptimal_join_tree": "The query joins more tables than necessary",
        "incorrect_join_predicate": "A join predicate does not follow a primary-key/foreign-key relationship",
        "empty_predicate": "A filter predicate matches no row in the database",
        "abnormal_result": "The query result looks abnormal",
        "incorrect_filter_in_subquery": "A scalar comparison uses a subquery that returns multiple rows",
        "incorrect_group_by": "GROUP BY is used without an aggregate function",
        "unnecessary_subquery": "The query uses an unusually large number of subqueries",
        "value_ambiguity": "A filter value also appears in other columns; the filtered column may be wrong",
        "table_similarity": "Another table has the same columns; the wrong table may be used",
        "evidence_violation": "The query may not reflect all the evidence",
        "insufficient_evidence": "There is insufficient evidence to verify the query",
        "low_confidence_link": "Some question concepts are mapped to clauses with low confidence",
        "column_ambiguity": "Other similar columns could answer the question",
        "llm_self_check": "The query may be incorrect according to a self-check",
    }
    out = []
    for s in SIGNALS:
        if sig.get(s):
            d = det.get(s) or ""
            if s == "low_confidence_link":
                d = str(dtl.get("low_confidence_links", ""))[:200]
            out.append((s, msgs[s] + (f" ({d})" if d else "")))
    return out


def correct(item, sql, sig, det, dtl, model, db_sig_fn):
    """Returns (final_sql, cost, latency_ms, n_calls, db_ms)."""
    schema = schema_with_samples(item["db_id"])
    ev = item.get("evidence") or "None"
    cost = lat = 0.0; calls = 0; db_ms = 0.0
    original, cur = sql, sql
    fixed = set()
    for _ in range(MAX_ITER):
        errs = [(s, m) for s, m in describe(sig, det, dtl) if s not in fixed]
        if not errs:
            break
        choice = errs[0]
        if len(errs) > 1:
            listing = "\n".join(f"{i+1}. {m}" for i, (_, m) in enumerate(errs))
            d = cached_chat(model, [{"role": "user", "content": SELECTOR.format(
                question=item["question"], evidence=ev, sql=cur, errors=listing)}], max_tokens=50)
            cost += cost_units(model, d["usage"]); lat += d["latency_ms"]; calls += 1
            k = re.findall(r"\d+", d["text"] or "")
            if k and 1 <= int(k[0]) <= len(errs):
                choice = errs[int(k[0]) - 1]
        d = cached_chat(model, [{"role": "user", "content": FIXER.format(
            schema=schema, question=item["question"], evidence=ev, sql=cur, report=choice[1])}])
        cost += cost_units(model, d["usage"]); lat += d["latency_ms"]; calls += 1
        new = _sqlblock(d["text"])
        # parser feedback loop
        conn = connect(item["db_id"])
        try:
            conn.execute("EXPLAIN QUERY PLAN " + new)
            perr = None
        except Exception as e:
            perr = str(e)
        finally:
            conn.close()
        if perr:
            d = cached_chat(model, [{"role": "user", "content": FIXER.format(
                schema=schema, question=item["question"], evidence=ev, sql=new,
                report=f"The SQL fails to parse/execute: {perr}")}])
            cost += cost_units(model, d["usage"]); lat += d["latency_ms"]; calls += 1
            new = _sqlblock(d["text"])
        fixed.add(choice[0])
        cur = new
        s2, det2, ms = db_sig_fn(item["db_id"], cur)
        db_ms += ms; lat += ms
        sig = {**sig, **s2}; det = {**det, **det2}
    if cur.strip() == original.strip():
        return original, cost, lat, calls, db_ms
    d = cached_chat(model, [{"role": "user", "content": AUDITOR.format(
        schema=schema, question=item["question"], evidence=ev, a=original, b=cur)}], max_tokens=20)
    cost += cost_units(model, d["usage"]); lat += d["latency_ms"]; calls += 1
    final = cur if "revised" in (d["text"] or "").lower() else original
    return final, cost, lat, calls, db_ms
