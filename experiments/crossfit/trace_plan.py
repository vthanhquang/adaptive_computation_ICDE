"""Print the plan the planner executes for given questions (SQLite, two-phase, pairwise selector).

Usage: python -m experiments.crossfit.trace_plan 650 [qid ...]
"""
import sys

from plangate.planning import crossfit as D

QIDS = [int(x) for x in sys.argv[1:]] or [650]
rows, items = D.load_action_table("sqlite")
fold_of, train_folds = D.question_folds(len(rows))
P = D.build_planner(rows, items, "full", fold_of, train_folds, "pairwise")
idx = {r["question_id"]: i for i, r in enumerate(rows)}
for lam, tau in ((0.1, 0.2), (0.05, 0.5), (0.1, 0.75)):
    res = D.run_hybrid(rows, P, "full", lam, tau)
    for q in QIDS:
        x = res[idx[q]]
        print(f"lam={lam} tau={tau} q={q}: plan={x['plan']} final={x['final']} correct={x['correct']} "
              f"cost={x['cost']:.2f} lat={x['lat']/1000:.1f}s | V(c0)={P['V'][(idx[q], ('c0',))]:.3f}")
