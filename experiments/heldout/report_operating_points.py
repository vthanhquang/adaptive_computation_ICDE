"""Print the calibrated operating points of every policy on TEST: accuracy, cost, DB time (mean / p95),
end-to-end p95 latency and optional probes per question.

Usage: python -m experiments.heldout.report_operating_points pg pg_x10 pg_x100 sqlite
"""
import sys

from plangate.evaluation.heldout import run_summary

for d in sys.argv[1:]:
    S = run_summary(d)
    for p in S["chosen"]:
        print(f"== {d}  DB price {p}  (cheap_only acc {S['chosen'][p]['cheap_only']['acc']:.3f}, n={S['n']['test']})")
        for B in ("1.25", "1.5", "2.0", "3.0", "4.0"):
            row = S["chosen"][p].get(B, {})
            for pol, v in row.items():
                t = v["test"]
                print(f"  {B:>4}x {pol:16s} acc {t['acc']:.3f} cost {t['cost']:6.2f} db {t['db_ms']:7.1f}ms "
                      f"dbp95 {t['db_p95']:7.0f}ms wallp95 {t['wall_p95']/1000:5.1f}s probes {t['n_vp']+t['n_ex']:.2f}")
