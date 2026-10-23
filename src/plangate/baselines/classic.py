"""Re-implementations of two classic multi-stage Text-to-SQL pipelines on BIRD / SQLite.

DIN-SQL (Pourreza & Rafiei, NeurIPS 2023) -- BIRD version (DIN-SQL_BIRD.py):
  schema linking -> classification (EASY / NON-NESTED / NESTED + sub-questions)
  -> class-specific few-shot generation -> self-correction.
MAC-SQL (Wang et al., COLING 2025):
  Selector (LLM schema pruning for large DBs) -> Decomposer (sub-question CoT)
  -> Refiner (execution-guided fix), within MAX_ROUND = 3 agent turns.

Prompt templates are read verbatim from the vendored upstream files in third_party/ (string
constants only, via ast). Control flow, schema rendering and output parsing follow the
upstream code. Only the backbone LLM differs (deepseek-v4-flash instead of GPT-4).
"""
from __future__ import annotations
import ast, csv, json, re, sqlite3, time
from functools import lru_cache
from pathlib import Path

from ..benchmark import BIRD_DEV_DIR as BIRD
from ..engines.sqlite import connect, run_bounded
from ..paths import THIRD_PARTY_DIR
from ..llm.generation import cached_chat, cost_units

TP = THIRD_PARTY_DIR


def _string_constants(path):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


DIN = _string_constants(TP / "din_sql" / "DIN-SQL_BIRD.py")
MAC = _string_constants(TP / "mac_sql" / "const.py")


class Acc:
    def __init__(self, model):
        self.model = model; self.cost = 0.0; self.ms = 0.0; self.calls = 0; self.tokens = 0

    def call(self, messages, temperature=0.0, max_tokens=4000):
        for attempt in range(4):   # failed calls are never cached; retry with backoff
            d = cached_chat(self.model, messages, temperature=temperature, max_tokens=max_tokens)
            if not d.get("error") and (d["text"] or "").strip():
                break
            self.errors = getattr(self, "errors", 0) + 1
            time.sleep(10 * (attempt + 1))
        self.cost += cost_units(self.model, d["usage"]); self.ms += d["latency_ms"]; self.calls += 1
        self.tokens += d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"]
        return d["text"] or ""

    def snap(self):
        return {"cost": self.cost, "ms": self.ms, "calls": self.calls, "tokens": self.tokens,
                "api_errors": getattr(self, "errors", 0)}


def _fmt(template, **kw):
    """LangChain f-string templates: '{{' escapes and named variables."""
    try:
        return template.format(**kw)
    except (KeyError, IndexError, ValueError):
        return template


# =================================================================== DIN-SQL
@lru_cache(maxsize=None)
def din_table_info(db_id):
    """Mimics langchain SQLDatabase.get_table_info(): CREATE TABLE + 3 sample rows per table."""
    conn = connect(db_id)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    parts = []
    for t in tables:
        info = list(conn.execute(f'PRAGMA table_info("{t}")'))
        lines = [f"\t{c[1]} {c[2] or ''}{' NOT NULL' if c[3] else ''}" for c in info]
        pk = [c[1] for c in sorted(info, key=lambda c: c[5]) if c[5]]
        if pk:
            lines.append(f"\tPRIMARY KEY ({', '.join(pk)})")
        for fk in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
            lines.append(f"\tFOREIGN KEY({fk[3]}) REFERENCES {fk[2]} ({fk[4] or fk[3]})")
        ddl = f"\nCREATE TABLE {t} (\n" + ", \n".join(lines) + "\n)"
        try:
            rows = conn.execute(f'SELECT * FROM "{t}" LIMIT 3').fetchall()
        except sqlite3.Error:
            rows = []
        cols = [c[1] for c in info]
        body = "\n".join("\t".join(str(v)[:100] for v in r) for r in rows)
        parts.append(f"{ddl}\n\n/*\n{len(rows)} rows from {t} table:\n" + "\t".join(cols) + f"\n{body}\n*/")
    conn.close()
    return "\n\n".join(parts)


@lru_cache(maxsize=None)
def din_column_descriptions(db_id):
    """table_descriptions_parser(): positional CSV columns 0 (name), 2 (description), 4 (value description)."""
    d = BIRD / "dev_databases" / db_id / "database_description"
    out = ""
    for p in sorted(d.glob("*.csv")):
        out += f"Table: {p.stem}\n"
        with open(p, encoding="latin-1", newline="") as f:
            rows = list(csv.reader(f))[1:]
        for row in rows:
            row = row + [""] * (5 - len(row))
            if row[2].strip():
                cd = re.sub(r"\s+", " ", row[2]); vd = re.sub(r"\s+", " ", row[4])
                out += (f"Column {row[0]}: column description -> {cd}, value description -> {vd}\n" if row[4].strip()
                        else f"Column {row[0]}: column description -> {cd}\n")
        out += "\n"
    return out


def _din_links(t):
    m = re.search(r"Schema_links:\s*\[(.*?)\]", t)
    return [x.strip() for x in m.group(1).split(",")] if m else []


def _din_label(t):
    lm = re.search(r'Label:\s*"(.*?)"', t)
    sm = re.search(r"sub_questions:\s*\[(.*?)\]", t)
    return (lm.group(1) if lm else None), ([x.strip() for x in sm.group(1).split(",")] if sm else [])


def _din_sql(t, key="SQL"):
    m = re.search(rf"{key}:\s*(.*?)$", t, re.S)
    return m.group(1).strip() if m else None


def din_sql(item, model):
    acc = Acc(model)
    schema, cd = din_table_info(item["db_id"]), din_column_descriptions(item["db_id"])
    q, hint = item["question"], str(item.get("evidence"))

    def chat(sys_key, human_key, **kw):
        return acc.call([{"role": "system", "content": _fmt(DIN[sys_key])},
                         {"role": "user", "content": _fmt(DIN[human_key], **kw)}])

    links = _din_links(chat("SYSTEM_SCHEMA_LINKING_TEMPLATE", "HUMAN_SCHEMA_LINKING_TEMPLATE",
                            question=q, schema=schema, hint=hint, columns_descriptions=cd))
    label, subq = _din_label(chat("SYSTEM_CLASSIFICATION_TEMPLATE", "HUMAN_CLASSIFICATION_TEMPLATE",
                                  question=q, schema=schema, hint=hint, columns_descriptions=cd, schema_links=links))
    label = label or "NESTED"
    if "EASY" in label:
        gen = chat("SYSTEM_EASY_CLASS_TEMPLATE", "HUMAN_EASY_CLASS_TEMPLATE", question=q, schema=schema, hint=hint,
                   columns_descriptions=cd, schema_links=links)
    elif "NON-NESTED" in label:
        gen = chat("SYSTEM_NON_NESTED_CLASS_TEMPLATE", "HUMAN_NON_NESTED_CLASS_TEMPLATE", question=q, schema=schema,
                   hint=hint, columns_descriptions=cd, schema_links=links)
    else:
        gen = chat("SYSTEM_NESTED_CLASS_TEMPLATE", "HUMAN_NESTED_CLASS_TEMPLATE", question=q, schema=schema,
                   hint=hint, columns_descriptions=cd, schema_links=links, sub_questions=subq)
    sql = _din_sql(gen)
    corr = chat("SYSTEM_SELF_CORRECTION_PROMPT", "HUMAN_SELF_CORRECTION_PROMPT", question=q, schema=schema,
                columns_descriptions=cd, hint=hint, sql_query=sql)
    final = _din_sql(corr, "Revised_SQL")
    final = (final or sql or "SELECT * FROM table").replace("\n", " ").replace("\r", " ")
    final = re.sub(r"```(sql)?", "", final).strip().rstrip(";")
    return final, {**acc.snap(), "label": label, "db_ms": 0.0}


# =================================================================== MAC-SQL
@lru_cache(maxsize=None)
def _tables_json():
    return {d["db_id"]: d for d in json.loads((BIRD / "dev_tables.json").read_text(encoding="utf8"))}


def _is_email(s):
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", s))


def _is_valid_date(s):
    if not isinstance(s, str):
        return False
    s = s.split()[0] if s.split() else s
    if len(s) != 10 or not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return False
    y, m, d = map(int, s.split("-"))
    return y >= 1 and 1 <= m <= 12 and 1 <= d <= 31


def _value_examples(values, col_type):
    if not values:
        return ""
    if len(values) > 10 and col_type in ["INTEGER", "REAL", "NUMERIC", "FLOAT", "INT"]:
        return ""
    vals, has_null = [], False
    for v in values:
        if v is None:
            has_null = True
        elif str(v).strip() != "":
            vals.append(v)
    if not vals:
        return ""
    if col_type in ["TEXT", "VARCHAR"]:
        new = []
        for v in vals:
            if not isinstance(v, str):
                new.append(v)
            elif v == "":
                continue
            elif "https://" in v or "http://" in v or _is_email(v):
                return ""
            else:
                new.append(v)
        vals = new
        if not vals or max(len(str(a)) for a in vals) > 50:
            return ""
    vals = vals[:6]
    if all(_is_valid_date(v) for v in vals):
        vals = vals[:1]
    if has_null:
        vals.insert(0, None)
    return str(vals)


@lru_cache(maxsize=None)
def mac_db_info(db_id):
    """Selector._load_single_db_info: column full names, value examples, PKs, FKs."""
    dbj = _tables_json()[db_id]
    keys = []
    for c in dbj["primary_keys"] + dbj["foreign_keys"]:
        keys.extend(c if isinstance(c, list) else [c])
    conn = connect(db_id)
    desc, values, pks, fks = {}, {}, {}, {}
    cols_orig, cols_full = dbj["column_names_original"], dbj["column_names"]
    for ti, tb in enumerate(dbj["table_names_original"]):
        names, is_key, d = [], [], []
        for ci, (tix, name) in enumerate(cols_orig):
            if tix != ti:
                continue
            names.append(name); is_key.append(ci in keys)
            d.append((name, cols_full[ci][1].replace("_", " "), ""))
        desc[tb] = d
        types = {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info(`{tb}`)")}
        vals = []
        for name, k in zip(names, is_key):
            low = name.lower()
            if k or low.endswith("id") or low.endswith("email") or low.endswith("url"):
                vals.append((name, "")); continue
            try:   # upstream fetches all groups; capped here for very large tables
                v = [r[0] for r in conn.execute(
                    f"SELECT `{name}` FROM `{tb}` GROUP BY `{name}` ORDER BY COUNT(*) DESC LIMIT 5000")]
                vals.append((name, _value_examples(v, types.get(name, ""))))
            except Exception:
                vals.append((name, ""))
        values[tb] = vals; fks[tb] = []; pks[tb] = []
    for frm, to in dbj["foreign_keys"]:
        ft = dbj["table_names_original"][cols_orig[frm][0]]
        tt = dbj["table_names_original"][cols_orig[to][0]]
        fks[ft].append((cols_orig[frm][1], tt, cols_orig[to][1]))
    for pk in dbj["primary_keys"]:
        for c in (pk if isinstance(pk, list) else [pk]):
            pks[dbj["table_names_original"][cols_orig[c][0]]].append(cols_orig[c][1])
    conn.close()
    counts = [sum(1 for t, _ in cols_orig if t == i) for i in range(len(dbj["table_names_original"]))]
    return {"desc": desc, "values": values, "pk": pks, "fk": fks,
            "total": sum(counts), "avg": sum(counts) // len(counts)}


def mac_desc(db_id, extracted):
    """Selector._get_db_desc_str with the BIRD list-style table rendering."""
    info = mac_db_info(db_id)
    schema, fk_all = "", []
    for tb in info["desc"]:
        cd, cv, fk, pk = info["desc"][tb], info["values"][tb], info["fk"][tb], info["pk"][tb]
        decision = extracted.get(tb, "")
        allc = [n for n, _, _ in cd]
        important = pk + [f[0] for f in fk]
        if decision == "drop_all":
            nd, nv = cd[:6], cv[:6]
        elif decision in ("keep_all", ""):
            nd, nv = cd, cv
        else:
            nd, nv, chosen = [], [], []
            for i, c in enumerate(allc):
                if c in important or c in decision:
                    nd.append(cd[i]); nv.append(cv[i]); chosen.append(c)
            if len(allc) > 6 and len(nv) < 6:
                for i, c in enumerate(allc):
                    if len(chosen) >= 6:
                        break
                    if c not in chosen:
                        nd.append(cd[i]); nv.append(cv[i]); chosen.append(c)
        lines = []
        for (c, full, extra), (_, vs) in zip(nd, nv):
            s = f"  ({c},"
            if full:
                s += f" {full.strip()}."
            if vs:
                s += f" Value examples: {vs}."
            lines.append(s + "),")
        schema += f"# Table: {tb}\n" + "[\n" + "\n".join(lines).strip(",") + "\n]" + "\n"
        for c, tt, tc in fk:
            link = f"{tb}.`{c}` = {tt}.`{tc}`"
            if link not in fk_all:
                fk_all.append(link)
    return schema.strip(), "\n".join(fk_all).strip()


def _mac_parse_json(text):
    s = text.find("```json"); e = text.find("```", s + 7)
    if s == -1 or e == -1:
        return {}
    try:
        j = json.loads(text[s + 7:e])
    except Exception:
        return {}
    for k, v in j.items():
        if isinstance(v, str) and v not in ("keep_all", "drop_all"):
            return {}
        if not isinstance(v, (str, list)):
            return {}
    return j


def _mac_parse_sql(text):
    m = re.findall(r"```sql(.*?)```", text, re.S)
    return m[-1].strip() if m else "error: No SQL found in the input string"


def _mac_exec(db_id, sql):
    conn = connect(db_id)
    t0 = time.perf_counter()
    try:
        rows, err, ms, timed_out = run_bounded(conn, sql, timeout_s=120)
    finally:
        conn.close()
    if timed_out:
        return None, (time.perf_counter() - t0) * 1000
    if err:
        return {"sql": sql, "sqlite_error": err, "exception_class": "sqlite3.OperationalError"}, ms
    return {"sql": sql, "data": rows[:5], "sqlite_error": "", "exception_class": ""}, ms


def mac_sql(item, model):
    acc = Acc(model)
    db_id, q, ev = item["db_id"], item["question"], item.get("evidence") or ""
    info = mac_db_info(db_id)
    db_ms = 0.0
    desc, fk = mac_desc(db_id, {})
    need_prune = not (info["avg"] <= 6 and info["total"] <= 30)
    call = lambda prompt: acc.call([{"role": "user", "content": prompt}], temperature=0.1)
    rounds = 1                                   # round 1: Selector
    if need_prune:
        ext = _mac_parse_json(call(MAC["selector_template"].format(
            db_id=db_id, query=q, evidence=ev, desc_str=desc, fk_str=fk)))
        desc, fk = mac_desc(db_id, ext)
    rounds += 1                                  # round 2: Decomposer
    sql = _mac_parse_sql(call(MAC["decompose_template_bird"].format(query=q, desc_str=desc, fk_str=fk, evidence=ev)))
    while rounds < 3:                            # MAX_ROUND = 3 agent turns in total
        rounds += 1                              # Refiner turn
        if "error" in sql:
            break
        res, ms = _mac_exec(db_id, sql); db_ms += ms
        if res is None:
            break   # timeout -> keep SQL
        need = False
        if "data" not in res:
            need = True
        elif len(res["data"]) == 0:
            res["sqlite_error"] = "no data selected"; need = True
        elif any(v is None for r in res["data"] for v in r):
            res["sqlite_error"] = "exist None value, you can add `NOT NULL` in SQL"; need = True
        if not need:
            break
        sql_arg = sql if sql.lower().startswith("select") else "SELECT" + sql
        sql = _mac_parse_sql(call(MAC["refiner_template"].format(
            query=q, evidence=ev, desc_str=desc, fk_str=fk, sql=sql_arg,
            sqlite_error=res["sqlite_error"], exception_class=res["exception_class"])))
    return sql, {**acc.snap(), "pruned": need_prune, "db_ms": db_ms}
