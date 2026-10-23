"""Behavioural checks of the adaptive evidence policy on a fixture planner.

  T1  skipped observations do not influence decisions: perturbing the payload of every optional
      observation that the policy did not acquire leaves the action trace unchanged
  T2  gold labels are invisible at serving time: flipping every EX label leaves the trace unchanged
  T3  an admissible query can be accepted without VALUE_PROBE or EXEC_BOUNDED
  T4  different states choose different optional probes (VALUE_PROBE vs EXEC_BOUNDED first)
  T5  traces log each action with predicted and observed cost; fixed_bundle still acquires everything
The fixture planner is trained on a slice of a TRAIN replay table (never on test data).

Usage:  python -m experiments.heldout.behaviour_checks [train_split] [serve_split]   (default: pg_train pg_train)
Output: results/heldout/behaviour_checks.json
"""
import copy, json, sys

from plangate.evaluation.heldout import load
from plangate.paths import HELDOUT_DIR
from plangate.planning import adaptive as AD


def main():
    tr_split = sys.argv[1] if len(sys.argv) > 1 else "pg_train"
    sv_split = sys.argv[2] if len(sys.argv) > 2 else tr_split
    tr, items_tr, _ = load(tr_split)
    if sv_split == tr_split:                 # disjoint halves of one split
        tr, sv, items_sv = tr[:-60], tr[-60:], items_tr
    else:
        sv, items_sv, _ = load(sv_split)
    tr, sv = tr[:150], sv[:80]
    pl = AD.Planner(tr, items_tr, n_masks=2, log=lambda *a: None).fit(lambdas=(0.0, 0.02, 0.1))
    R = AD.Runner(pl, sv, items_sv)
    lam = 0.02
    base = {r["question_id"]: R.run(r["question_id"], "adaptive", lam=lam) for r in sv}

    def acts(o):
        return [(t["action"], t.get("cand"), t.get("variant")) for t in o["trace"]]

    # T1: perturb unacquired optional observations
    sv2 = copy.deepcopy(sv)
    for r in sv2:
        tr_acq = {(t["cand"], t["action"]) for t in base[r["question_id"]]["trace"] if t["action"] in AD.OPT}
        ents = [("c0", r["c0"])] + [(k, r[k]) for k in ("VALUE_FIX", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN") if r.get(k)]
        ents += [("REPAIR", e) for e in r["REPAIR"].values()]
        for k, e in ents:
            for op in AD.OPT:
                if (k, op) not in tr_acq:
                    e["ev"][op] = {"ok": True, "empty": True, "n_rows": 0, "fingerprint": "PERTURBED",
                                   "supported": True, "n_missing": 7, "probes": []}
    R2 = AD.Runner(pl, sv2, items_sv)
    t1 = sum(acts(R2.run(r["question_id"], "adaptive", lam=lam)) == acts(base[r["question_id"]]) for r in sv)
    # T2: flip labels
    sv3 = copy.deepcopy(sv)
    for r in sv3:
        for k in ("c0", "VALUE_FIX", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"):
            if r.get(k):
                r[k]["correct"] = not r[k]["correct"]
        for e in r["REPAIR"].values():
            e["correct"] = not e["correct"]
    R3 = AD.Runner(pl, sv3, items_sv)
    t2 = sum(acts(R3.run(r["question_id"], "adaptive", lam=lam)) == acts(base[r["question_id"]]) for r in sv)
    # T3/T4/T5
    no_probe_accept = [q for q, o in base.items() if o["n_vp"] == 0 and o["n_ex"] == 0]
    first_probe = {}
    for q, o in base.items():
        for t in o["trace"]:
            if t["action"] in AD.OPT:
                first_probe[q] = t["action"]; break
    kinds = set(first_probe.values())
    logged = all(("pred_cost" in t and "db_ms" in t) for o in base.values() for t in o["trace"] if t["action"] in AD.OPT)
    fb = [R.run(r["question_id"], "fixed_bundle", lam=lam) for r in sv]
    fb_full = all(o["n_vp"] == o["n_llm"] and o["n_ex"] == o["n_llm"] for o in fb)
    res = {"T1_hidden_obs_invariance": f"{t1}/{len(sv)}", "T2_label_invariance": f"{t2}/{len(sv)}",
           "T3_accept_without_optional_probe": len(no_probe_accept), "T4_first_probe_kinds": sorted(kinds),
           "T5_trace_costs_logged": logged, "T5_fixed_bundle_acquires_all": fb_full}
    ok = t1 == len(sv) and t2 == len(sv) and no_probe_accept and len(kinds) == 2 and logged and fb_full
    print(json.dumps(res, indent=1)); print("ALL PASS" if ok else "SOME CHECKS FAILED")
    q = no_probe_accept[0] if no_probe_accept else None
    if q:
        print("example skip trace", q, base[q]["trace"])
    out = HELDOUT_DIR / "behaviour_checks.json"
    out.write_text(json.dumps({**res, "pass": bool(ok), "example_skip": base.get(q)}, indent=1, default=str), encoding="utf8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
