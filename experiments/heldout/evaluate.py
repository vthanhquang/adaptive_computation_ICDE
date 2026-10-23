"""Held-out evaluation of adaptive evidence acquisition.

  1. fit the planner on the training split only (frozen, pickled with a content hash)
  2. run every policy over its parameter grid on the calibration and test splits
  3. choose operating points per budget on CALIBRATION ONLY, then report them on TEST
  4. paired bootstrap of adaptive vs. every baseline at each budget; full test frontiers
With --cal oof (the default) the calibration runs serve the TRAINING questions with the fold models
that did not see them (cross-fitted calibration), so the test split is touched exactly once.
DB price p (cost units per DB-second) is varied to model cheap (BIRD, p=1) and expensive evidence;
non-adaptive policies do not observe p, so their runs are re-priced, while adaptive is re-run per p.

Usage (the reported runs):
  python -m experiments.heldout.evaluate pg --prices 1,30,100
  python -m experiments.heldout.evaluate sqlite --prices 1,30,100
  python -m experiments.heldout.evaluate pg --scale 10  --prices 1
  python -m experiments.heldout.evaluate pg --scale 100 --prices 1
  python -m experiments.heldout.evaluate pg --scale 100 --prices 1 --no-opt-features --policies adaptive \\
      --name pg_x100_no_optimizer_features
  python -m experiments.heldout.evaluate pg --train pg_lodo0_train --test pg_lodo0_test --prices 1 \\
      --name pg_leave_db_out/fold0                                   (folds 0, 1, 2)
Outputs: results/heldout/<name>/ or results/scaling/<name>/ with summary.json, chosen_traces.jsonl.gz,
         run_meta.json, log.txt and the frozen planner.pkl
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, subprocess, time
from pathlib import Path

from plangate.evaluation.heldout import BUDGETS, POLICIES, grid, load, run_all, run_dir, summarize
from plangate.evaluation.stats import paired_bootstrap
from plangate.jsonl import write_jsonl
from plangate.llm.generation import ROLE_MODELS
from plangate.paths import REPO_ROOT
from plangate.planning import adaptive as AD


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=["pg", "sqlite"])
    ap.add_argument("--prices", default="1,30,100", help="DB prices in cost units per DB-second")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--n-masks", type=int, default=4, help="random evidence masks per training state")
    ap.add_argument("--scale", type=int, default=0, help="charge DB costs measured on the k-times scaled database")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train", default="", help="training split (default: <engine>_train)")
    ap.add_argument("--cal", default="oof", help="calibration split, or 'oof' for cross-fitted training runs")
    ap.add_argument("--test", default="", help="test split (default: <engine>_test)")
    ap.add_argument("--repair-allowed", default="",
                    help="REPAIR variants in the replay table (default: all on pg, 'vpex' on sqlite)")
    ap.add_argument("--no-opt-features", action="store_true", help="ablation: hide optimizer-derived features")
    ap.add_argument("--calibrate", action="store_true", help="isotonic calibration of V and Q")
    ap.add_argument("--voi-mode", default="diff", choices=["diff", "decomp"])
    ap.add_argument("--policies", default="", help="comma-separated subset of policies to run")
    ap.add_argument("--planner", default="", help="reuse a fitted planner.pkl instead of fitting")
    ap.add_argument("--name", default="", help="run name (default: <engine>[_x<scale>])")
    a = ap.parse_args()
    prices = [float(x) for x in a.prices.split(",")]
    repair_allowed = a.repair_allowed or ("vpex" if a.engine == "sqlite" else "")
    name = a.name or f"{a.engine}{('_x' + str(a.scale)) if a.scale else ''}"
    out_dir = run_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "w", encoding="utf8")

    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    tr, items_tr, sp_tr = load(a.train or f"{a.engine}_train", a.scale)
    oof_cal = a.cal == "oof"
    if oof_cal:
        ca, items_ca, sp_ca = tr, items_tr, dict(sp_tr, name=sp_tr["name"] + ":oof")
    else:
        ca, items_ca, sp_ca = load(a.cal, a.scale)
    te, items_te, sp_te = load(a.test or f"{a.engine}_test", a.scale)
    say(f"== {a.engine}: train {len(tr)}  cal {len(ca)}  test {len(te)}")
    t0 = time.time()
    if a.planner:
        pl = pickle.loads(Path(a.planner).read_bytes())
    else:
        pl = AD.Planner(tr, items_tr, n_masks=a.n_masks, log=say,
                        repair_allowed=repair_allowed.split(",") if repair_allowed else None,
                        no_opt=a.no_opt_features, calibrate=a.calibrate, voi_mode=a.voi_mode).fit()
    pl.log = print                     # the file logger is not picklable
    blob = pickle.dumps(pl)
    (out_dir / "planner.pkl").write_bytes(blob)
    phash = hashlib.sha256(blob).hexdigest()[:16]
    say(f"  planner fitted in {time.time()-t0:.0f}s, checkpoint {phash}")

    G = grid(a.engine, tr)
    if a.policies:
        keep = set(a.policies.split(",")) | {"cheap_only"}
        G = [g for g in G if g[0] in keep]
    res = {}
    for split, rows, items in (("cal", ca, items_ca), ("test", te, items_te)):
        t1 = time.time()
        res[split] = run_all(out_dir / "planner.pkl", rows, items, G, prices, a.workers, out_dir,
                             oof=(split == "cal" and oof_cal))
        say(f"  ran {len(G)} configs on {split} in {time.time()-t1:.0f}s")

    summary = {"engine": a.engine, "prices": prices, "budgets": BUDGETS, "planner": phash,
               "n": {"train": len(tr), "cal": len(ca), "test": len(te)}, "frontier": {}, "chosen": {}, "boot": {}}
    for (pol, kw, p), out in res["test"].items():
        summary["frontier"].setdefault(str(p), []).append({"policy": pol, "params": json.loads(kw), **summarize(out, p)})
    trace_dump = {}
    for p in prices:
        cal_cheap = summarize(res["cal"][("cheap_only", "{}", p)], p)["cost"]
        test_cheap = summarize(res["test"][("cheap_only", "{}", p)], p)
        summary["chosen"][str(p)] = {"cheap_only": test_cheap}
        for B in BUDGETS:
            pick = {}
            for pol in POLICIES:
                cands = [(k, summarize(v, p)) for k, v in res["cal"].items() if k[0] == pol and k[2] == p]
                ok = [(k, s) for k, s in cands if s["cost"] <= B * cal_cheap + 1e-9]
                if not ok:
                    continue
                k, s = max(ok, key=lambda ks: (ks[1]["acc"], -ks[1]["cost"]))
                pick[pol] = k
                st = summarize(res["test"][k], p)
                summary["chosen"][str(p)].setdefault(str(B), {})[pol] = {"params": json.loads(k[1]), "cal": s, "test": st}
                trace_dump[f"p{p}_B{B}_{pol}"] = res["test"][k]
            if "adaptive" in pick:
                ad = [o["correct"] for o in res["test"][pick["adaptive"]]]
                for pol in POLICIES:
                    if pol != "adaptive" and pol in pick:
                        bl = [o["correct"] for o in res["test"][pick[pol]]]
                        summary["boot"].setdefault(str(p), {}).setdefault(str(B), {})[pol] = \
                            paired_bootstrap(ad, bl, a.boot)
    if ("oracle", "{}", prices[0]) in res["test"]:
        summary["oracle"] = summarize(res["test"][("oracle", "{}", prices[0])], prices[0])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    write_jsonl(out_dir / "chosen_traces.jsonl.gz",
                ({"config": k, **o} for k, out in trace_dump.items() for o in out))
    meta = {"git_commit": git_commit(), "engine": a.engine, "planner_checkpoint": phash,
            "splits": {s["name"]: s["sha256"] for s in (sp_tr, sp_ca, sp_te)}, "prices": prices,
            "lambdas": AD.LAMBDAS, "n_masks": a.n_masks, "budgets": BUDGETS,
            "models": {"cheap": ROLE_MODELS["flash"], "mid": ROLE_MODELS["pro"], "strong": ROLE_MODELS["qwen"]},
            "time_s": time.time() - t0}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    for p in prices:
        say(f"\n== DB price {p:g} u/s   (cheap_only test acc {summary['chosen'][str(p)]['cheap_only']['acc']:.3f})")
        for B in BUDGETS:
            row = summary["chosen"][str(p)].get(str(B), {})
            say(f"  {B:4.2f}x " + " | ".join(
                f"{pol[:10]} {row[pol]['test']['acc']:.3f}/{row[pol]['test']['cost']:.2f}/{row[pol]['test']['db_ms']:.0f}ms"
                for pol in POLICIES if pol in row))
            bt = summary["boot"].get(str(p), {}).get(str(B), {})
            if bt:
                say("        adaptive - " + ", ".join(f"{k[:10]} {v[0]*100:+.1f} [{v[1]*100:+.1f},{v[2]*100:+.1f}]"
                                                   for k, v in bt.items()))
    say(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
