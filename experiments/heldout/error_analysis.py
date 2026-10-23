"""Failure-mode analysis of the chosen operating points on TEST.

For each policy at a given DB price and budget, using the held-out traces and the replay table
(for hidden ground-truth properties used ONLY for this offline analysis):
  literal      c0 has a missing literal: how often VALUE_PROBE was bought on c0, VALUE_FIX used, fixed
  empty_ok     c0 returns an empty result yet is correct: harmful replacements (false-positive flag)
  join_agg     c0 joins or aggregates: EXEC acquisition rate and accuracy
  c0_correct   first candidate correct: accepted with no optional probe; regressions to a wrong answer
  expensive    c0 optimizer cost in the top decile (of TRAIN): EXEC acquisition rate, DB ms
  truncated    an acquired candidate result hit the 1,000-row cap (fingerprint may mislead selection)

Usage:  python -m experiments.heldout.error_analysis pg|sqlite [--price 1.0] [--budget 2.0]
Output: <run>/error_analysis_p<price>_B<budget>.json
"""
import argparse, json
from collections import defaultdict

import numpy as np

from plangate.evaluation.heldout import chosen_traces, load, run_dir, run_meta, run_scale
from plangate.features import sql_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--price", default="1.0"); ap.add_argument("--budget", default="2.0")
    a = ap.parse_args()
    d = run_dir(a.run)
    meta = run_meta(a.run)
    names = list(meta["splits"])          # insertion order: train, cal, test
    test, _, _ = load(names[-1], run_scale(a.run))
    train, _, _ = load(names[0], run_scale(a.run))
    T = {r["question_id"]: r for r in test}
    costs = [(r["c0"]["ev"].get("EXPLAIN") or {}).get("total_cost") for r in train]
    hi = np.percentile([c for c in costs if c is not None], 90) if any(c is not None for c in costs) else None
    traces = defaultdict(list)
    for x in chosen_traces(a.run):
        traces[x["config"]].append(x)
    pref = f"p{float(a.price)}_B{float(a.budget)}_"
    report = {}
    for cfg, outs in traces.items():
        if not cfg.startswith(pref):
            continue
        pol = cfg[len(pref):]
        S = defaultdict(lambda: defaultdict(float))
        for o in outs:
            r = T[o["q"]]; c0 = r["c0"]; ev = c0["ev"]
            acts = [(t["action"], t.get("cand")) for t in o["trace"]]
            vp_c0 = ("VALUE_PROBE", "c0") in acts
            ex_c0 = ("EXEC_BOUNDED", "c0") in acts
            bundle = pol in ("fixed_bundle", "optimizer_gated", "rule_selective")
            sf = sql_features(c0["sql"])
            eb = ev.get("EXEC_BOUNDED") or {}

            def add(cat, **kv):
                S[cat]["n"] += 1
                for k, v in kv.items():
                    S[cat][k] += float(v)

            if (ev.get("VALUE_PROBE") or {}).get("n_missing"):
                add("literal", vp_bought=vp_c0 or (bundle and o["n_vp"] > 0), value_fix_used=o.get("final") == "VALUE_FIX",
                    correct=o["correct"])
            if c0["correct"] and eb.get("ok") and eb.get("empty"):
                add("empty_ok", replaced=o.get("final") != "c0", regression=not o["correct"])
            if sf["sql_joins"] or sf["sql_aggs"]:
                add("join_agg", ex_c0=ex_c0 or (bundle and o["n_ex"] > 0), correct=o["correct"])
            if c0["correct"]:
                add("c0_correct", no_optional_probe=(o["n_vp"] + o["n_ex"]) == 0, regression=not o["correct"],
                    extra_llm=o["n_llm"] - 1)
            tc = (ev.get("EXPLAIN") or {}).get("total_cost")
            if hi is not None and tc is not None and tc >= hi:
                add("expensive", ex_c0=ex_c0 or (bundle and o["n_ex"] > 0), db_ms=o["db_ms"], correct=o["correct"])
            if eb.get("truncated"):
                add("truncated", correct=o["correct"])
        report[pol] = {cat: {k: (v if k == "n" else v / max(S[cat]["n"], 1)) for k, v in vals.items()} for cat, vals in S.items()}
    out = d / f"error_analysis_p{a.price}_B{a.budget}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf8")
    for pol, cats in report.items():
        print(f"== {pol}")
        for cat, v in cats.items():
            print(f"   {cat:11s} " + "  ".join(f"{k}={v[k]:.2f}" if k != "n" else f"n={int(v[k])}" for k in v))


if __name__ == "__main__":
    main()
