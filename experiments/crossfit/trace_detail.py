"""Step-by-step trace of PlanGate on one question: state values, action scores, acceptor and
pairwise-selector outputs, evidence and costs. Used for the walk-through figure.
Usage: python -m experiments.crossfit.trace_detail 650   (writes results/crossfit/trace_<qid>.json)
"""
import json, sys

from plangate.paths import CROSSFIT_DIR
from plangate.planning import crossfit as D

QID = int(sys.argv[1]) if len(sys.argv) > 1 else 650
rows, items = D.load_action_table("sqlite")
fold_of, train_folds = D.question_folds(len(rows))
P = D.build_planner(rows, items, "full", fold_of, train_folds, "pairwise")
i = {r["question_id"]: k for k, r in enumerate(rows)}[QID]
r = rows[i]
out = {"question_id": QID}


def state_report(S, lam):
    v = P["V"][(i, S)]
    acts = []
    for a in P["acts"]:
        if a in S or (i, S, a) not in P["Q"] or a == "SWITCH_QWEN":
            continue
        c, l = D.cand_cost(r, a, "full")
        q = P["Q"][(i, S, a)]
        acts.append({"action": a, "Q": q, "cost": c, "score": q - v - lam * c})
    return v, sorted(acts, key=lambda x: -x["score"])


for lam, tau in ((0.1, 0.2), (0.05, 0.5)):
    print(f"\n===== lambda={lam}, tau={tau}")
    S = ("c0",)
    trace = []
    while len(S) - 1 < 2:
        v, acts = state_report(S, lam)
        print(f"state {S}: V={v:.3f}")
        for x in acts:
            print(f"   {x['action']:12s} Q={x['Q']:.3f}  cost={x['cost']:.2f}  Q-V-lam*C={x['score']:+.3f}")
        trace.append({"S": list(S), "V": v, "acts": acts})
        if not acts or acts[0]["score"] <= 0:
            break
        S = P["canon"][(i, frozenset(S + (acts[0]["action"],)))]
    v = P["V"][(i, S)]
    esc = v < tau
    print(f"phase 2: V({S})={v:.3f} {'<' if esc else '>='} tau={tau} -> {'SWITCH_QWEN' if esc else 'accept'}")
    if esc:
        S = P["canon"][(i, frozenset(S + ("SWITCH_QWEN",)))]
    print(f"final set {S}")
    for g in S:
        print(f"   A({g}|S)={P['A'][(i, S, g)]:.3f}  fp={D.fp(r[g])}  correct={r[g]['correct']}")
    pairs = {f"{a}>{b}": P["P"][(i, S, a, b)] for a in S for b in S if (i, S, a, b) in P["P"]}
    for k, v2 in pairs.items():
        print(f"   P({k})={v2:.3f}")
    g, a, _ = P["select"](i, S)
    print(f"selected: {g} (correct={r[g]['correct']})")
    out[f"lam={lam},tau={tau}"] = {"trace": trace, "final_set": list(S), "selected": g,
                                  "A": {g2: P["A"][(i, S, g2)] for g2 in S}, "P": pairs}
(CROSSFIT_DIR / f"trace_{QID}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf8")
