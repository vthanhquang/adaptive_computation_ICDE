"""Probe ablations on a frozen run: adaptive without VALUE_PROBE, adaptive without BOUNDED_EXEC, and the
matching single-probe bundles. Re-uses the frozen planner.pkl of the run; operating points are chosen
per budget on cross-fitted TRAINING runs, exactly as in evaluate.py.

Usage:  python -m experiments.heldout.ablate pg|pg_x100 [--workers 8]
Output: <run>/ablation.json and a printed table (adaptive and bundles from summary.json for reference)
"""
import argparse, json, time

from plangate.evaluation.heldout import (GRID_LAMBDAS, chosen_traces, load, run_all, run_dir, run_meta,
                                         run_scale, run_summary, summarize)
from plangate.evaluation.stats import paired_bootstrap

BUDGETS = ("1.25", "1.5", "2.0", "3.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = run_dir(a.run)
    names = list(run_meta(a.run)["splits"])
    scale = run_scale(a.run)
    tr, items_tr, _ = load(names[0], scale)
    te, items_te, _ = load(names[-1], scale)
    G = [("cheap_only", {})]
    for lam in GRID_LAMBDAS:
        for tau in (None, 0.5):
            G += [("fixed_vp_only", {"lam": lam, "tau": tau}), ("fixed_ex_only", {"lam": lam, "tau": tau})]
            G += [(p, {"lam": lam, "tau": tau, "voi_margin": m}) for p in ("adaptive_noVP", "adaptive_noEX")
                  for m in (-0.003, 0.0)]
    t0 = time.time()
    cal = run_all(d / "planner.pkl", tr, items_tr, G, [1.0], a.workers, d, oof=True)
    print(f"cal {time.time()-t0:.0f}s", flush=True)
    tst = run_all(d / "planner.pkl", te, items_te, G, [1.0], a.workers, d)
    print(f"test {time.time()-t0:.0f}s", flush=True)
    S = run_summary(a.run)
    main_tr = {}
    for x in chosen_traces(a.run):
        if x["config"].startswith("p1.0_B") and x["config"].endswith("_adaptive"):
            main_tr.setdefault(x["config"], []).append(x)
    cheap = summarize(cal[("cheap_only", "{}", 1.0)], 1.0)["cost"]
    out = {"eval_dir": a.run, "budgets": {}}
    for B in BUDGETS:
        row = {}
        ad = main_tr.get(f"p1.0_B{float(B)}_adaptive")
        order = [o["q"] for o in ad] if ad else None
        for pol in ("adaptive_noVP", "adaptive_noEX", "fixed_vp_only", "fixed_ex_only"):
            c = [(k, summarize(v, 1.0)) for k, v in cal.items() if k[0] == pol]
            c = [(k, s) for k, s in c if s["cost"] <= float(B) * cheap + 1e-9]
            if not c:
                continue
            k, s = max(c, key=lambda ks: (ks[1]["acc"], -ks[1]["cost"]))
            t = summarize(tst[k], 1.0)
            row[pol] = {"params": json.loads(k[1]), "test": t}
            if ad:
                by = {o["q"]: o["correct"] for o in tst[k]}
                row[pol]["adaptive_minus"] = paired_bootstrap([o["correct"] for o in ad], [by[q] for q in order])
        for pol in ("adaptive", "fixed_bundle", "optimizer_gated"):
            v = S["chosen"]["1.0"].get(str(float(B)), {}).get(pol)
            if v:
                row[pol] = {"test": v["test"]}
        out["budgets"][B] = row
        print(f"  {B}x " + " | ".join(f"{p} {v['test']['acc']:.3f}/{v['test']['db_ms']:.0f}ms"
                                     + (f" (full-{p}: {100*v['adaptive_minus'][0]:+.1f})" if 'adaptive_minus' in v else "")
                                     for p, v in row.items()), flush=True)
    (d / "ablation.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
