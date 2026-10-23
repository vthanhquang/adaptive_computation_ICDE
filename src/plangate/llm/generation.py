"""LLM generation operators (GEN / REPAIR) with a persistent response cache.

The cache makes every planner replay deterministic and free after the first run:
a (model, messages, temperature) triple is only ever paid for once.
"""
from __future__ import annotations
import hashlib, json, os, re, threading

from ..engines.sqlite import schema_ddl, schema_with_samples
from ..paths import CACHE_DIR
from . import providers

CACHE = CACHE_DIR / "llm"

# Model roles of the held-out evaluation: cheap first candidate, mid-tier switch, strong switch.
ROLE_MODELS = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro", "qwen": "qwen3.8-max"}

# Output tokens are priced higher than input tokens by most providers; to compare models on one
# scale we use a per-model rate multiplier and a fixed 1:4 input/output weighting.
OUTPUT_WEIGHT = 4.0


def cost_units(model, usage):
    rate = providers.RATE.get(model, 1.0)
    return rate * (usage["prompt_tokens"] + OUTPUT_WEIGHT * usage["completion_tokens"]) / 1000.0


def _key(model, messages, temperature, sample=0):
    parts = [model, messages, temperature] + ([sample] if sample else [])
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()


_locks = {}
_locks_guard = threading.Lock()


def cached_chat(model, messages, temperature=0.0, max_tokens=6000, sample=0):
    """`sample` distinguishes independent stochastic samples (temperature > 0) in the cache."""
    k = _key(model, messages, temperature, sample)
    path = CACHE / f"{k}.json"
    with _locks_guard:
        lock = _locks.setdefault(k, threading.Lock())
    with lock:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf8"))
            d["cached"] = True
            return d
        last = None
        for _ in range(3):
            try:
                text, usage, latency = providers.chat(model, messages, temperature, max_tokens)
                break
            except Exception as e:  # network / rate limit; retry a few times
                last = e
        else:
            return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "latency_ms": 0.0, "error": str(last)[:300], "cached": False}
        d = {"model": model, "text": text, "usage": usage, "latency_ms": latency}
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf8")
        os.replace(tmp, path)
        d["cached"] = False
        return d


# ---------------------------------------------------------------- prompts

SYSTEM = "You are an expert SQLite data analyst who writes correct SQLite queries."

DIRECT_TMPL = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

Write a single SQLite query that answers the question. Use the hint. Select only the columns the question asks for.
Return only the SQL inside one ```sql code block, no explanation."""

DECOMPOSE_TMPL = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

First reason step by step:
1. Identify the tables and columns needed (check exact column names, quote names with spaces using backticks).
2. Identify join paths and filter values (check the example rows for the exact value format).
3. Identify aggregation, ordering, and what exactly must be returned (only the columns the question asks for).
Then write a single SQLite query that answers the question.
End your answer with the final SQL inside one ```sql code block."""

REPAIR_TMPL = """Database schema:
{schema}

Question: {question}
Hint: {evidence}

A previous attempt produced this SQL:
```sql
{sql}
```

The database returned the following diagnostic evidence about that SQL:
{feedback}

Decide whether the SQL is wrong. If it is wrong, fix it; if it is right, return it unchanged.
Return only the final SQL inside one ```sql code block."""

PROFILES = {
    "direct": DIRECT_TMPL,
    "decompose": DECOMPOSE_TMPL,
}

# PostgreSQL dialect (BIRD mini-dev); the schema is rendered by engines.postgres.pg_schema.
PG_SYSTEM = "You are an expert PostgreSQL data analyst who writes correct PostgreSQL queries."
PG_DIRECT_TMPL = """Database schema (PostgreSQL):
{schema}

Question: {question}
Hint: {evidence}

Write a single PostgreSQL query that answers the question. Use the hint. Select only the columns the question asks for.
Return only the SQL inside one ```sql code block, no explanation."""
PG_REPAIR_TMPL = """Database schema (PostgreSQL):
{schema}

Question: {question}
Hint: {evidence}

A previous attempt produced this SQL:
```sql
{sql}
```

The database returned the following diagnostic evidence about that SQL:
{feedback}

Decide whether the SQL is wrong. If it is wrong, fix it; if it is right, return it unchanged.
Return only the final PostgreSQL query inside one ```sql code block."""


def build_messages(item, profile, schema_mode):
    schema = schema_ddl(item["db_id"]) if schema_mode == "ddl" else schema_with_samples(item["db_id"])
    body = PROFILES[profile].format(schema=schema, question=item["question"], evidence=item.get("evidence") or "none")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def build_repair_messages(item, sql, feedback):
    body = REPAIR_TMPL.format(schema=schema_with_samples(item["db_id"]), question=item["question"],
                              evidence=item.get("evidence") or "none", sql=sql, feedback=feedback)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_sql(text):
    blocks = _SQL_BLOCK.findall(text or "")
    sql = blocks[-1] if blocks else (text or "")
    sql = re.sub(r"^\s*(sql|sqlite)\s*\n", "", sql.strip(), flags=re.I)
    # keep only the first statement
    sql = sql.strip().rstrip(";").split(";\n")[0].strip()
    return sql


def generate(item, config, temperature=0.0, sample=0):
    """config: dict(model, profile, schema)."""
    msgs = build_messages(item, config["profile"], config["schema"])
    d = cached_chat(config["model"], msgs, temperature=temperature, sample=sample)
    return {
        "sql": extract_sql(d["text"]),
        "raw_len": len(d["text"] or ""),
        "usage": d["usage"],
        "latency_ms": d["latency_ms"],
        "cost": cost_units(config["model"], d["usage"]),
        "error": d.get("error"),
    }


def generate_pg(item, model, temperature=0.0, sample=0, body=None):
    """PostgreSQL generation (direct prompt, or a given user `body` such as a repair prompt).
    Returns (sql, cost units, latency_ms, error or None)."""
    from ..engines.postgres import pg_schema
    body = body or PG_DIRECT_TMPL.format(schema=pg_schema(item["db_id"]), question=item["question"],
                                         evidence=item.get("evidence") or "none")
    d = cached_chat(model, [{"role": "system", "content": PG_SYSTEM}, {"role": "user", "content": body}],
                    temperature=temperature, sample=sample)
    return extract_sql(d["text"]), cost_units(model, d["usage"]), d["latency_ms"], d.get("error")


def pg_repair_body(item, sql, feedback):
    from ..engines.postgres import pg_schema
    return PG_REPAIR_TMPL.format(schema=pg_schema(item["db_id"]), question=item["question"],
                                 evidence=item.get("evidence") or "none", sql=sql, feedback=feedback)


def repair(item, model, sql, feedback):
    d = cached_chat(model, build_repair_messages(item, sql, feedback))
    return {
        "sql": extract_sql(d["text"]),
        "usage": d["usage"],
        "latency_ms": d["latency_ms"],
        "cost": cost_units(model, d["usage"]),
        "error": d.get("error"),
    }
