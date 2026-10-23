"""Gain attribution: where does the planner's gain over the cheap model come from?
Per engine, at the 2x budget and at the most accurate point of the two-phase policy.

Usage: python -m experiments.crossfit.attribution   (stdout: results/crossfit/attribution.txt)
"""
from collections import Counter

import numpy as np

from plangate.planning import crossfit as D
from plangate.planning.crossfit import triggers

for engine in D.ENGINES:
    rows, items = D.load_action_table(engine)
    n = len(rows)
    fold_of, train_folds = D.question_folds(n)
    P = D.build_planner(rows, items, "full", fold_of, train_folds, "pairwise")
    flash = np.mean([r["c0"]["cost"] for r in rows])
    runs = []
    for lam in (0.02, 0.05, 0.1, 0.25):
        for tau in np.linspace(0.2, 0.9, 15):
            rr = D.run_hybrid(rows, P, "full", lam, tau)
            runs.append((np.mean([x["correct"] for x in rr]), np.mean([x["cost"] for x in rr]), lam, tau, rr))
    c0acc = np.mean([r["c0"]["correct"] for r in rows])
    print(f"\n===== {engine}  n={n}  flash acc={c0acc:.3f} cost={flash:.2f}")
    print(f"  VALUE_FIX available on {sum(D.available(r,'VALUE_FIX') for r in rows)} questions; "
          f"flagged c0: {sum(triggers(r['c0']['ev']) for r in rows)}")
    for name, budget in (("2x", 2 * flash), ("best", 1e9)):
        ok = [x for x in runs if x[1] <= budget]
        acc, cost, lam, tau, rr = max(ok, key=lambda x: (x[0], -x[1]))
        print(f"-- {name}: acc={acc:.3f} cost={cost:.2f} lam={lam} tau={tau:.2f}")
        win, loss = Counter(), Counter()
        planc = Counter()
        for r, x in zip(rows, rr):
            fl = "flagged" if triggers(r["c0"]["ev"]) else "unflagged"
            src = D.SRC[x["final"]]
            planc[(fl, "escalated" if "SWITCH_QWEN" in x["plan"] else ("local" if x["plan"] != "ACCEPT" else "accept"))] += 1
            if x["correct"] and not r["c0"]["correct"]:
                win[(fl, src)] += 1
            if not x["correct"] and r["c0"]["correct"]:
                loss[(fl, src)] += 1
        print(f"   net gain over c0: {sum(win.values())} wins - {sum(loss.values())} losses = "
              f"{sum(win.values())-sum(loss.values())} questions ({(sum(win.values())-sum(loss.values()))/n*100:+.1f} pts)")
        print("   wins by (state, final source):", dict(win.most_common()))
        print("   losses by (state, final source):", dict(loss.most_common()))
        print("   plans:", dict(planc))
