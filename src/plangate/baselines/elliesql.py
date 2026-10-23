"""Re-implementation of EllieSQL (Zhu et al., 2025; arXiv 2503.22402) on BIRD dev / SQLite.

Follows the released code (github.com/HKUSTDial/EllieSQL): prompts are copied verbatim;
pipelines and router input format mirror src/pipeline_factory.py, src/modules/* and
src/sft/instruction_templates.py. Only the backbone LLM differs (deepseek-v4-flash here,
gpt-4o-mini in the paper).

  schema linking  : EnhancedSchemaLinker (enriched schema -> JSON tables/columns, + PK/FK)
  G_B  (basic)    : GPTSQLGenerator           -- direct generation
  G_M  (interm.)  : DirectDCRefineSQLGenerator -- divide-and-conquer CoT + execution refiner
  G_A  (advanced) : DirectDCOSRefineSQLGenerator -- online-synthesised few-shot + DC + refiner
  router          : released Qwen2.5-0.5B DPO classifier (label 0/1/2 -> G_B/G_M/G_A)
"""
from __future__ import annotations
import csv, json, re, sqlite3, time
from functools import lru_cache

from ..benchmark import BIRD_DEV_DIR as BIRD
from ..engines.sqlite import connect, run_bounded
from ..llm.generation import cached_chat, cost_units

BACKBONE = "deepseek-v4-flash"

# ------------------------------------------------------------------ prompts (verbatim)
ENHANCED_SCHEMA_SYSTEM = """
You are a smart and responsible SQLite SQL expert. Assist in identifying the database tables and columns involved in natural language queries.
### Instruction:
Your task is to analyze the provided database schema, comprehend the posed question, and leverage the hint to identify which tables are needed to generate a SQL query for answering the question.
The returned JSON format must strictly adhere to the following specifications:
{
    "tables": [
        {
            "table": "table name",
            "columns": ["relevant column 1", "relevant column 2", ...]
        },
        ...
    ]
}
Each relevant column must belong to its respective table, and the output JSON object must be wrapped in a code block using ```json```.

Please note that each table and column comes with detailed description information and example values for reference.
"""
BASE_SCHEMA_USER = """
### Database schema:
{schema_str}

### User question:
{query}

### Hint:
{evidence}
"""
SQL_GENERATION_SYSTEM = "You are a SQLite expert."
SQL_GENERATION_USER = """
You are a intelligent and responsible SQLite expert.

### Instruction:
You need to read the database schema to generate SQL query for the user question. The outputted SQL must be surrounded by ```sql``` code block.

### Database Schema:
{schema}

### Hint:
{evidence}

### User Question:
{query}

Generate the corresponding SQL query surrounded by ```sql``` code block.
"""
_DC_HEAD = """
You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Value Examples" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query.
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions.
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question.

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:**
    - Only select columns mentioned in the user's question.
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs).
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause.
11. **SQLite Functions Only:**
    - Use only functions available in SQLite.
12. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).
13. **Schema Syntax:**
    - Use `table_name`.`column_name` to refer to columns from different tables, when table name or column name contains whitespace.
14. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements. Do not use `CROSS JOIN` or `LEFT/RIGHT JOIN`.

Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer, and finally try to simplify the SQL query using `INNER JOIN` over nested `SELECT` statements IF POSSIBLE.
"""
_DC_TAIL = """
Please respond with xml tags structured as follows:
```xml
<think>
Your detailed reasoning for the SQL query generation, with Recursive Divide-and-Conquer approach.
</think>
<sql>
The final SQL query that answers the question.
</sql>
```

**************************
【Table creation statements】
{SCHEMA_CONTEXT}

**************************
【Question】
Question:
{QUESTION}

Hint:
{HINT}

**************************

Only output xml format (starting with ```xml and ending with ```) as your response.

"""
DIRECT_DC_PROMPT = _DC_HEAD + _DC_TAIL
DIRECT_DC_WITH_OS_PROMPT = _DC_HEAD + """
Here are some NL2SQL examples for your reference:
{OS_EXAMPLES}
""" + _DC_TAIL
OS_SYSTEM = "\nYou are a SQLite SQL expert. \n"
ONLINE_SYNTHESIS_PROMPT = """
### Instruction:
You are a SQLite SQL expert. Your job is to create {k} examples, where each example consists of a question and a SQL query to fetch the data for it.
I want each example to look like this, question input and SQL output pairs:

### Example:
"Question": "What's the description of the series code SM.POP.TOTL for Aruba?
(Hints: Aruba is the name of the country where ShortName = 'Aruba')"

"SQL": "SELECT T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2
ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = 'Aruba' AND T2.Seriescode = 'SM.POP.TOTL'"

### Task:
You should generate examples that examine and showcase different aspects and relationships of the following table schemas,
described in "Table creation statements". Understand the database tables and their relationships.
Understand the columns and their types and meanings to construct interesting examples.

Generate a mixture of SQL examples that include:
• some simple SQL query examples without JOIN
• some SQL query examples with aggregates, like COUNT
• some simple SQL query examples with JOIN
• some complex SQL query examples with nested JOIN

## Database Schema:
{TARGET_DATABASE_SCHEMA}

Generate total of {k} examples. Only outputs the examples ('question input' and 'SQL output' pairs), and each example can be separated by a new line.
"""
REFINER_SYSTEM = "\nYou are a SQLite expert and very good at fixing SQL errors.\n"
REFINER_USER = """
### Instruction:
There is a SQL query generated based on the following Database Schema description and the potential Evidence to respond to the Question.
However, executing this SQL has resulted in an error or empty result, and you need to fix it based on the error message.
Utilize your knowledge of SQLite to generate the correct SQL. SQL must be surrounded by ```sql``` code block.

### Database schema:
{db_schema}

### Question:
{question}

### Hint:
{evidence}

### SQL:
{sql} \n

### Execution result:
result_type: {result_type}
result: {result}
error_message: {error_message} \n

### Response:
"""


# ------------------------------------------------------------------ schema (enriched with BIRD descriptions)
def _read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, csv.Error):
            continue
    return []


@lru_cache(maxsize=None)
def db_schema(db_id):
    """Dict like EllieSQL's SchemaExtractor output: tables -> columns -> info, PKs, FKs."""
    conn = connect(db_id)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    desc_dir = BIRD / "dev_databases" / db_id / "database_description"
    out = {"database": db_id, "tables": [], "foreign_keys": []}
    for t in tables:
        desc = {}
        p = desc_dir / f"{t}.csv"
        if p.exists():
            for row in _read_csv(p):
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                name = row.get("original_column_name", "")
                if name:
                    desc[name.lower()] = row
        cols, pks = {}, []
        for cid, name, ctype, notnull, dflt, pk in conn.execute(f'PRAGMA table_info("{t}")'):
            d = desc.get(name.lower(), {})
            try:
                vals = [str(v[0])[:40] for v in conn.execute(
                    f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL LIMIT 3')]
            except sqlite3.Error:
                vals = []
            cols[name] = {"type": ctype, "expanded_name": d.get("column_name", ""),
                          "description": d.get("column_description", ""), "data_format": d.get("data_format", ""),
                          "value_description": d.get("value_description", ""), "value_examples": vals}
            if pk:
                pks.append(name)
        for row in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
            ref_t, frm, to = row[2], row[3], row[4]
            if to is None:   # FK to the referenced table's PK
                pk = [r[1] for r in conn.execute(f'PRAGMA table_info("{ref_t}")') if r[5]]
                to = pk[0] if pk else frm
            out["foreign_keys"].append({"table": [t, ref_t], "column": [frm, to]})
        out["tables"].append({"table": t, "columns": cols, "primary_keys": pks})
    conn.close()
    return out


def format_enriched_db_schema(schema):
    res = [f"Database: {schema['database']}\n"]
    for table in schema["tables"]:
        res.append(f"Table name: {table['table']}")
        res.append("Columns:")
        for c, ci in table["columns"].items():
            d = []
            if ci["expanded_name"]: d.append(f"Meaning: {ci['expanded_name']}")
            if ci["description"]: d.append(f"Description: {ci['description']}")
            if ci["data_format"]: d.append(f"Format: {ci['data_format']}")
            if ci["value_description"]: d.append(f"Value description: {ci['value_description']}")
            if ci["value_examples"]: d.append(f"Value examples: {', '.join(ci['value_examples'])}")
            res.append(f"  - {c} ({ci['type']}): {' | '.join(d)}" if d else f"  - {c} ({ci['type']})")
        if table["primary_keys"]:
            res.append(f"Primary key: {', '.join(table['primary_keys'])}")
        res.append("")
    if schema.get("foreign_keys"):
        res.append("Foreign keys:")
        for fk in schema["foreign_keys"]:
            res.append(f"  {fk['table'][0]}.{fk['column'][0]} = {fk['table'][1]}.{fk['column'][1]}")
    return "\n".join(res)


def _extract_schema_json(text):
    m = re.findall(r"```json\s*(.*?)\s*```", text or "", re.S)
    raw = m[0] if m else (text or "").strip()
    try:
        s = json.loads(raw)
    except Exception:
        return None
    if not isinstance(s, dict) or not isinstance(s.get("tables"), list):
        return None
    for t in s["tables"]:
        if not isinstance(t, dict) or not isinstance(t.get("table"), str) or not isinstance(t.get("columns"), list):
            return None
    return s


def _validate(linked, schema):
    db = {t["table"]: set(t["columns"]) for t in schema["tables"]}
    for t in linked["tables"]:
        if t["table"] not in db or any(c not in db[t["table"]] for c in t["columns"]):
            return False
    return True


def _enhance(linked, schema):
    """_enhance_linked_schema + enhance_schema_only_with_keys from the released linker."""
    dbt = {t["table"]: t for t in schema["tables"]}
    names = [t["table"] for t in linked["tables"]]
    for t in linked["tables"]:
        pks = dbt[t["table"]]["primary_keys"]
        if pks:
            t["primary_keys"] = pks
            for pk in pks:
                if pk not in t["columns"]:
                    t["columns"].append(pk)
    if len(names) > 1:
        for fk in schema.get("foreign_keys", []):
            s, d = fk["table"]; sc, dc = fk["column"]
            if s in names and d in names:
                so = next(t for t in linked["tables"] if t["table"] == s)
                do = next(t for t in linked["tables"] if t["table"] == d)
                if sc not in so["columns"]: so["columns"].append(sc)
                if dc not in do["columns"]: do["columns"].append(dc)
                so.setdefault("foreign_keys", [])
                if not any(x["column"] == sc and x["referenced_table"] == d for x in so["foreign_keys"]):
                    so["foreign_keys"].append({"column": sc, "referenced_table": d, "referenced_column": dc})
    for t in linked["tables"]:
        t["columns_info"] = {c: dbt[t["table"]]["columns"][c] for c in t["columns"] if c in dbt[t["table"]]["columns"]}
    return linked


def format_linked_schema(linked):
    res = []
    for t in linked["tables"]:
        res.append(f"Table: {t['table']}")
        if t["columns"]:
            res.append("Columns:")
            for c in t["columns"]:
                ci = t.get("columns_info", {}).get(c, {})
                d = []
                if ci.get("type"): d.append(f"Type: {ci['type']}")
                if ci.get("expanded_name"): d.append(f"Meaning: {ci['expanded_name']}")
                if ci.get("description"): d.append(f"Description: {ci['description']}")
                if ci.get("data_format"): d.append(f"Format: {ci['data_format']}")
                if ci.get("value_description"): d.append(f"Value description: {ci['value_description']}")
                if ci.get("value_examples"): d.append(f"Value examples: {', '.join(ci['value_examples'])}")
                res.append(f"  - {c}: {' | '.join(d)}" if d else f"  - {c}")
        if "primary_keys" in t:
            res.append(f"Primary key: {', '.join(t['primary_keys'])}")
        if "foreign_keys" in t:
            res.append("Foreign keys:")
            for fk in t["foreign_keys"]:
                res.append(f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
        res.append("")
    return "\n".join(res)


# ------------------------------------------------------------------ LLM wrapper with accounting
class Acc:
    def __init__(self):
        self.cost = 0.0; self.ms = 0.0; self.calls = 0; self.ptok = 0; self.ctok = 0

    def call(self, system, user, temperature=0.0, sample=0, max_tokens=10000):
        d = cached_chat(BACKBONE, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=temperature, sample=sample, max_tokens=max_tokens)
        self.cost += cost_units(BACKBONE, d["usage"]); self.ms += d["latency_ms"]; self.calls += 1
        self.ptok += d["usage"]["prompt_tokens"]; self.ctok += d["usage"]["completion_tokens"]
        return d["text"] or ""

    def snap(self):
        return {"cost": self.cost, "ms": self.ms, "calls": self.calls, "tokens": self.ptok + self.ctok}


def _extract_sql(text):
    m = re.findall(r"```sql\s*(.*?)\s*```", text or "", re.S)
    return m[-1].strip() if m else (text or "").strip()


def _extract_sql_xml(text):
    m = re.findall(r"<sql>\s*(.*?)\s*</sql>", text or "", re.S)
    return m[-1].strip() if m else (text or "").strip()


# ------------------------------------------------------------------ pipeline stages
def schema_linking(item, acc, retries=3):
    schema = db_schema(item["db_id"])
    user = BASE_SCHEMA_USER.format(schema_str=format_enriched_db_schema(schema), query=item["question"],
                                   evidence=item.get("evidence") or "None")
    for k in range(retries):   # released code retries until validation passes
        text = acc.call(ENHANCED_SCHEMA_SYSTEM, user, temperature=0.0 if k == 0 else 0.3, sample=k)
        linked = _extract_schema_json(text)
        if linked and _validate(linked, schema):
            break
    else:   # fallback after failed retries: keep only existing tables/columns (or full schema)
        db = {t["table"]: set(t["columns"]) for t in schema["tables"]}
        tabs = []
        for t in (linked or {}).get("tables", []):
            if t.get("table") in db:
                tabs.append({"table": t["table"], "columns": [c for c in t["columns"] if c in db[t["table"]]]})
        linked = {"tables": tabs or [{"table": t["table"], "columns": list(t["columns"])} for t in schema["tables"]]}
    linked = _enhance(linked, schema)
    keys_only = {"tables": [{"table": t["table"], "columns": list(t["columns"])} for t in linked["tables"]]}
    return format_linked_schema(linked), keys_only


def _exec(db_id, sql):
    conn = connect(db_id)
    rows, err, ms, timed_out = run_bounded(conn, sql, timeout_s=10)
    conn.close()
    if timed_out:
        return "timeout", None, "SQL execution timed out after 10 seconds", ms
    if err:
        return "error", None, err, ms
    return "success", rows[:50], None, ms


def refine(item, sql, acc, db_ms):
    """FeedbackBasedRefiner.process_sql: refine only on execution error / timeout, max 3 iterations."""
    full = format_enriched_db_schema(db_schema(item["db_id"]))
    for _ in range(3):
        rtype, res, err, ms = _exec(item["db_id"], sql); db_ms[0] += ms
        if rtype == "success":
            return sql
        text = acc.call(REFINER_SYSTEM, REFINER_USER.format(
            sql=sql, result_type=rtype, result=res, error_message=err, question=item["question"],
            evidence=item.get("evidence") or "None", db_schema=full), max_tokens=5000)
        sql = _extract_sql(text)
    return sql


def gen_basic(item, linked_str, acc):
    return _extract_sql(acc.call(SQL_GENERATION_SYSTEM, SQL_GENERATION_USER.format(
        schema=linked_str, evidence=item.get("evidence") or "None", query=item["question"])))


def gen_intermediate(item, linked_str, acc, db_ms):
    sql = _extract_sql_xml(acc.call(SQL_GENERATION_SYSTEM, DIRECT_DC_PROMPT.format(
        SCHEMA_CONTEXT=linked_str, HINT=item.get("evidence") or "None", QUESTION=item["question"])))
    return refine(item, sql, acc, db_ms)


def gen_advanced(item, linked_str, acc, db_ms):
    examples = acc.call(OS_SYSTEM, ONLINE_SYNTHESIS_PROMPT.format(TARGET_DATABASE_SCHEMA=linked_str, k=5),
                        max_tokens=5000)
    sql = _extract_sql_xml(acc.call(SQL_GENERATION_SYSTEM, DIRECT_DC_WITH_OS_PROMPT.format(
        OS_EXAMPLES=examples, SCHEMA_CONTEXT=linked_str, HINT=item.get("evidence") or "None",
        QUESTION=item["question"])))
    return refine(item, sql, acc, db_ms)


# ------------------------------------------------------------------ router (released checkpoint)
def router_input(question, keys_only_schema):
    """PipelineClassificationTemplates.create_classifier_prompt (src/sft/instruction_templates.py)."""
    tables = keys_only_schema["tables"]
    n_t = len(tables); n_c = sum(len(t["columns"]) for t in tables)
    analysis = (f"The database schema related to the user question contains {n_t} table{'s' if n_t > 1 else ''} "
                f"with a total of {n_c} columns.\n")
    schema_str = "".join(f"Table: {t['table']}; Columns: {', '.join(t['columns'])}\n" for t in tables)
    return f"""
Database Schema Related to the Question:
{schema_str}
Schema Analysis:
{analysis}
Question: {question}"""


class DPORouter:
    def __init__(self, repo="derrickzhu/EllieSQL_Router_Checkpoints", subfolder="qwen_dpo"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder).eval()
        torch.manual_seed(42)

    def predict(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()
        return int(max(range(3), key=lambda i: probs[i])), probs, (time.perf_counter() - t0) * 1000
