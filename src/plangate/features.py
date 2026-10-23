"""Planner state features, split by information availability (RQ1 ablation).

P0: question + schema only          (what a pre-generation router can see)
P1: P0 + candidate SQL / model signals + SQL-text agreement with earlier candidates
P2: P1 + DBMS evidence (parse/bind, plan, value probes, bounded-execution shape,
         execution-result agreement with earlier candidates)
"""
from __future__ import annotations
import math, re
from functools import lru_cache

from .engines.sqlite import table_columns, schema_with_samples

KEYWORDS = ["how many", "average", "percentage", "percent", "ratio", "most", "least", "highest",
            "lowest", "difference", "list", "times", "compare", "per ", "each", "between", "total",
            "rate", "earliest", "latest", "top", "more than", "less than", "name", "which", "when", "id"]

DB_IDS = ["california_schools", "card_games", "codebase_community", "debit_card_specializing",
          "european_football_2", "financial", "formula_1", "student_club", "superhero",
          "thrombosis_prediction", "toxicology"]


@lru_cache(maxsize=None)
def _schema_stats(db_id):
    tc = table_columns(db_id)
    return len(tc), sum(len(c) for c in tc.values()), len(schema_with_samples(db_id))


def p0_features(item):
    q = item["question"].lower()
    ev = (item.get("evidence") or "").lower()
    n_tab, n_col, schema_len = _schema_stats(item["db_id"])
    f = {
        "q_words": len(q.split()), "ev_words": len(ev.split()), "q_digits": len(re.findall(r"\d+", q)),
        "q_quotes": q.count("'") + q.count('"'), "ev_has_formula": int(any(c in ev for c in "/*-+=")),
        "ev_has_sql_like": int(any(k in ev for k in ["count(", "sum(", "divide", "subtract"])),
        "n_tables": n_tab, "n_columns": n_col, "schema_kchars": schema_len / 1000,
    }
    for k in KEYWORDS:
        f["kw_" + k.strip().replace(" ", "_")] = int(k in q)
    for d in DB_IDS:
        f["db_" + d] = int(item["db_id"] == d)
    return f


def normalize_sql(sql):
    return re.sub(r"\s+", " ", (sql or "").lower().replace("`", '"')).strip().rstrip(";")


def sql_features(sql, completion_tokens=0):
    s = (sql or "").lower()
    return {
        "sql_len": len(s) / 100, "sql_joins": len(re.findall(r"\bjoin\b", s)),
        "sql_aggs": len(re.findall(r"\b(count|avg|sum|min|max)\s*\(", s)),
        "sql_selects": len(re.findall(r"\bselect\b", s)), "sql_group": int("group by" in s),
        "sql_order": int("order by" in s), "sql_limit": int(bool(re.search(r"\blimit\b", s))),
        "sql_case": int(" case " in f" {s} "), "sql_cast": int("cast(" in s), "sql_div": s.count("/"),
        "sql_distinct": int("distinct" in s), "sql_literals": len(re.findall(r"'[^']*'", s)),
        "sql_having": int("having" in s), "sql_cte": int(s.strip().startswith("with")),
        "sql_empty": int(not s.strip()), "gen_ctok": math.log1p(completion_tokens),
    }


def db_features(ev):
    """ev: {op: payload}; missing ops -> neutral defaults with an 'acquired' flag."""
    pb, ex, vp, eb = (ev.get(k) for k in ("PARSE_BIND", "EXPLAIN", "VALUE_PROBE", "EXEC_BOUNDED"))
    f = {}
    f["has_pb"] = int(pb is not None)
    f["pb_invalid"] = int(pb is not None and not pb.get("valid"))
    f["has_ex"] = int(ex is not None)
    f["ex_full_scans"] = ex.get("full_scans", 0) if ex and ex.get("valid") else 0
    f["ex_temp"] = ex.get("temp_btrees", 0) if ex and ex.get("valid") else 0
    f["ex_nojoin_cond"] = ex.get("joins_without_on", 0) if ex and ex.get("valid") else 0
    f["ex_suspicious"] = int(bool(ex and ex.get("suspicious_plan")))
    # PostgreSQL optimizer estimates (absent on SQLite -> 0)
    ok = bool(ex and ex.get("valid") and "total_cost" in ex)
    f["ex_log_cost"] = math.log1p(ex["total_cost"]) if ok else 0
    f["ex_log_rows"] = math.log1p(ex["plan_rows"]) if ok else 0
    f["ex_est_one_row"] = int(ok and ex["plan_rows"] <= 1)
    for k in ("seq_scans", "index_scans", "nested_loops", "hash_joins", "sorts", "aggregates", "unconditioned_nl"):
        f["ex_" + k] = ex.get(k, 0) if ok else 0
    f["has_vp"] = int(vp is not None)
    f["vp_preds"] = vp.get("n_predicates", 0) if vp else 0
    f["vp_missing"] = vp.get("n_missing", 0) if vp else 0
    f["has_eb"] = int(eb is not None)
    if eb is not None:
        ok = eb.get("ok", False)
        f.update({"eb_fail": int(not ok), "eb_timeout": int(bool(eb.get("timed_out"))),
                  "eb_empty": int(ok and eb.get("empty", False)), "eb_all_null": int(ok and eb.get("all_null", False)),
                  "eb_log_rows": math.log1p(eb.get("n_rows", 0)) if ok else 0, "eb_cols": eb.get("n_cols", 0) if ok else 0,
                  "eb_null_share": eb.get("null_share", 0) if ok else 0, "eb_dups": int(bool(ok and eb.get("dup_rows"))),
                  "eb_single_row": int(ok and eb.get("n_rows") == 1)})
    else:
        f.update({k: 0 for k in ["eb_fail", "eb_timeout", "eb_empty", "eb_all_null", "eb_log_rows", "eb_cols",
                                 "eb_null_share", "eb_dups", "eb_single_row"]})
    return f


def state_vector(level, item, config_names, cur_config, cur, prev, configs_all):
    """Feature dict for 'is the current candidate correct?'.

    cur/prev: dicts with keys sql, ctok, ev (evidence dict), fp (result fingerprint or None).
    """
    f = dict(p0_features(item))
    for c in configs_all:
        f["cfg_" + c] = int(cur_config == c)
    f["n_prev"] = len(prev)
    if level >= 1:
        f.update(sql_features(cur["sql"], cur.get("ctok", 0)))
        ns = normalize_sql(cur["sql"])
        same = [normalize_sql(p["sql"]) == ns for p in prev]
        f["sql_agree_any"] = int(any(same))
        f["sql_agree_n"] = sum(same)
        f["repaired"] = int(cur.get("repaired", False))
    if level >= 2:
        f.update(db_features(cur["ev"]))
        fps = [p.get("fp") for p in prev]
        agree = [cur.get("fp") is not None and fp == cur.get("fp") for fp in fps]
        f["res_agree_any"] = int(any(agree))
        f["res_agree_n"] = sum(agree)
        f["res_disagree_all"] = int(bool(prev) and not any(agree))
    return f
