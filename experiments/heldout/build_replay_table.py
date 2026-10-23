"""Evidence-conditioned replay table of a split (results/heldout/replay/<split>.jsonl).

For every question of a split it records, with real LLM calls (cached by exact prompt):
  c0            cheap model (deepseek-v4-flash), T=0, plus ALL DB observations on it
  VALUE_FIX     DB-only literal rewrite (requires VALUE_PROBE on c0)
  REPAIR[m]     cheap-model repair conditioned ONLY on the evidence visible under mask m,
                m in {none, vp, ex, vpex}; PARSE_BIND and EXPLAIN are always visible.
                The prompt (hence the cache key) differs per mask, so counterfactual states never
                reuse a repair generated from evidence they did not acquire.
  RESAMPLE_1    cheap model, T=0.8
  SWITCH_PRO    deepseek-v4-pro, T=0
  SWITCH_QWEN   qwen3.8-max, T=0
Every produced SQL gets its own full DB observations and an EX label (labels are stored for
offline training and evaluation only; the policy never sees them at serving time).

Usage:
  python -m experiments.heldout.build_replay_table pg_test
  python -m experiments.heldout.build_replay_table pg_train
  python -m experiments.heldout.build_replay_table sqlite_train
  python -m experiments.heldout.build_replay_table sqlite_test --repair-masks vpex
Finished questions are skipped, so an interrupted build resumes where it stopped.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, time

from plangate.engines import sqlite
from plangate.evaluation.heldout import load_items, load_spec
from plangate.evidence.operators import evidence_to_feedback, value_fix
from plangate.jsonl import iter_jsonl, resolve, write_jsonl
from plangate.paths import CONFIG_DIR, REPLAY_DIR

MASKS = {"none": (), "vp": ("VALUE_PROBE",), "ex": ("EXEC_BOUNDED",), "vpex": ("VALUE_PROBE", "EXEC_BOUNDED")}
MANDATORY = ("PARSE_BIND", "EXPLAIN")


class LLMError(Exception):
    pass


# ------------------------------------------------------------------ engine adapters
class SQLiteEngine:
    name = "sqlite"

    def __init__(self):
        from plangate.evidence.cache import get_evidence
        from plangate.evidence.operators import DB_OPS
        from plangate.llm.generation import generate, repair
        self.get_evidence, self.ops, self._gen, self._rep = get_evidence, DB_OPS, generate, repair
        cat = json.loads((CONFIG_DIR / "generation_configs.json").read_text())
        self.cfg = {"flash": cat["flash_direct"], "pro": cat["pro_direct"], "qwen": cat["qwen_max_direct"]}

    def gen(self, it, m, temperature=0.0, sample=0):
        g = self._gen(it, self.cfg[m], temperature=temperature, sample=sample)
        if g.get("error"):
            raise LLMError(g["error"])
        return g["sql"], g["cost"], g["latency_ms"]

    def repair(self, it, sql, feedback):
        r = self._rep(it, self.cfg["flash"]["model"], sql, feedback)
        if r.get("error"):
            raise LLMError(r["error"])
        return r["sql"], r["cost"], r["latency_ms"]

    def correct(self, it, sql):
        return sqlite.ex_correct(it, sql)


class PGEngine:
    name = "pg"

    def __init__(self, gold):
        from plangate.engines import postgres
        from plangate.llm.generation import ROLE_MODELS, generate_pg, pg_repair_body
        self.pg, self.gold, self.models = postgres, gold, ROLE_MODELS
        self._gen, self._repair_body = generate_pg, pg_repair_body
        self.get_evidence, self.ops = postgres.get_evidence, postgres.OPS

    def _call(self, it, model, temperature=0.0, sample=0, body=None):
        sql, cost, latency, err = self._gen(it, model, temperature, sample, body)
        if err:
            raise LLMError(err)
        return sql, cost, latency

    def gen(self, it, m, temperature=0.0, sample=0):
        return self._call(it, self.models[m], temperature, sample)

    def repair(self, it, sql, feedback):
        return self._call(it, self.models["flash"], body=self._repair_body(it, sql, feedback))

    def correct(self, it, sql):
        if self.gold == "pg":                     # official PostgreSQL gold (mini-dev)
            return self.pg.ex_correct(it, sql)
        from plangate.engines.cross_engine import ex_correct_cross
        return ex_correct_cross(it, sql)          # SQLite gold, validated on mini-dev


# ------------------------------------------------------------------ builder
def observe(E, it, sql):
    ev, ms = {}, {}
    for op in E.ops:
        ev[op], ms[op] = E.get_evidence(op, it["db_id"], sql)
    return ev, ms


def entry(E, it, sql, cost, llm_ms):
    ev, ms = observe(E, it, sql)
    return {"sql": sql, "correct": bool(E.correct(it, sql)), "cost": cost, "llm_ms": llm_ms, "ev": ev, "db_ms": ms}


def feedback_for(ev, mask):
    vis = {op: ev[op] for op in ev if op in MANDATORY or op in MASKS[mask]}
    return evidence_to_feedback(vis)


def build(E, it, masks):
    row = {"question_id": it["question_id"], "db_id": it["db_id"], "difficulty": it.get("difficulty"),
           "engine": E.name}
    sql, c, l = E.gen(it, "flash")
    row["c0"] = entry(E, it, sql, c, l)
    ev0 = row["c0"]["ev"]
    fixed = value_fix(sql, ev0["VALUE_PROBE"])
    row["VALUE_FIX"] = entry(E, it, fixed, 0.0, 0.0) if fixed else None
    row["REPAIR"], seen = {}, {}
    for m in [x for x in MASKS if x in masks]:
        fb = feedback_for(ev0, m)
        if fb in seen:                            # identical prompt -> identical cached output
            row["REPAIR"][m] = dict(row["REPAIR"][seen[fb]], same_prompt_as=seen[fb])
            continue
        s, c, l = E.repair(it, sql, fb)
        row["REPAIR"][m] = dict(entry(E, it, s, c, l), feedback=fb)
        seen[fb] = m
    s, c, l = E.gen(it, "flash", temperature=0.8, sample=1)
    row["RESAMPLE_1"] = entry(E, it, s, c, l)
    for name, m in (("SWITCH_PRO", "pro"), ("SWITCH_QWEN", "qwen")):
        s, c, l = E.gen(it, m)
        row[name] = entry(E, it, s, c, l)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repair-masks", default="none,vp,ex,vpex")
    a = ap.parse_args()
    masks = set(a.repair_masks.split(","))
    spec = load_spec(a.split)
    E = SQLiteEngine() if spec["engine"] == "sqlite" else PGEngine(spec.get("gold", "pg"))
    items = load_items(spec)
    if a.limit:
        items = items[:a.limit]
    out = resolve(REPLAY_DIR / f"{spec.get('table', a.split)}.jsonl")
    done = {r["question_id"] for r in iter_jsonl(out)} if out.exists() else set()
    todo = [it for it in items if it["question_id"] not in done]
    print(f"{a.split}: {len(items)} questions, {len(done)} done, {len(todo)} to build", flush=True)
    t0, k, err = time.time(), 0, 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build, E, it, masks): it["question_id"] for it in todo}
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:                # LLM/API error: not written, retried next run
                err += 1
                print(f"  ERROR {futs[fut]}: {str(e)[:120]}", flush=True)
                continue
            write_jsonl(out, [row], append=True); k += 1
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s  errors={err}", flush=True)
    print(f"done {k}, errors {err}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
