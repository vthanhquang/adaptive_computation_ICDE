"""Cross-fitted sequential planner versus baselines on the action tables (SQLite and PostgreSQL).

Trains the planner variants of plangate.planning.crossfit (full DB evidence, execution only, no DB;
pairwise vs pointwise selection; greedy, two-phase and latency-aware policies), runs every policy
over its parameter grid and reports frontiers, matched-budget comparisons with paired bootstrap
CIs, head-to-head results against the re-implemented published methods and split-half validation.

Usage:   python -m experiments.crossfit.planner sqlite|pg
Inputs:  results/crossfit/action_table_<engine>.jsonl, results/baselines/published_methods_<engine>.json
Outputs: results/crossfit/planner_report_<engine>.json, planner_frontier_<engine>.png (stdout: planner_<engine>.txt)
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict

import numpy as np

from plangate.jsonl import read_jsonl
from plangate.paths import BASELINES_DIR, CROSSFIT_DIR
from plangate.planning.crossfit import (VARIANT_ACTIONS, available, baselines, build_planner, confidence_cascades,
                                        load_action_table, pareto, question_folds, res, run_hybrid, run_planner,
                                        summarize, triggers)

def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    rows, items = load_action_table(engine)
    M = defaultdict(dict)
    if engine == "sqlite":
        for x in read_jsonl(CROSSFIT_DIR / "generation_matrix.jsonl"):
            M[x["question_id"]][x["config"]] = x
    n = len(rows)
    fold_of, train_folds = question_folds(n)
    pb = BASELINES_DIR / f"published_methods_{engine}.json"
    published = json.loads(pb.read_text(encoding="utf8")) if pb.exists() else None

    curves = defaultdict(list)   # name -> list of (label, results)
    for k, v in baselines(rows, items, fold_of, train_folds, M, published).items():
        curves[k] += v

    planners = {}
    for variant, selector in (("full", "pairwise"), ("full", "pointwise"), ("exec_only", "pairwise"),
                              ("nodb", "pairwise")):
        print(f"training planner variant={variant} selector={selector} ...", flush=True)
        P = build_planner(rows, items, variant, fold_of, train_folds, selector)
        name = f"PlanGate-{variant}" + ("" if selector == "pairwise" else " (pointwise select)")
        if selector == "pairwise":
            planners[variant] = P
        for lam in [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25]:
            curves[name].append((f"lam={lam}", run_planner(rows, P, variant, lam)))
        if selector == "pairwise":   # two-phase: cheap DB-guided actions, then gated escalation
            hname = f"PlanGate-{variant} 2-phase"
            for lam in (0.02, 0.05, 0.1, 0.25):
                for tau in np.linspace(0.2, 0.9, 15):
                    curves[hname].append((f"lam={lam},tau={tau:.2f}", run_hybrid(rows, P, variant, lam, tau)))
        if variant == "full" and selector == "pairwise":   # latency-aware objective
            for lw in [0.002, 0.005, 0.01, 0.02, 0.04]:
                curves["PlanGate-full (latency-aware)"].append(
                    (f"lat_w={lw}", run_planner(rows, planners[variant], variant, 0.0, lat_w=lw)))
    for k, v in confidence_cascades(rows, planners["full"]).items():
        curves[k] += v

    # oracle: cheapest single follow-up action that yields a correct answer
    orc = []
    for r in rows:
        opts = [("c0", ["c0"])] + [(a, ["c0", a]) for a in VARIANT_ACTIONS["full"] if available(r, a)]
        ok = [res(r, a, names) for a, names in opts if r[a]["correct"]]
        orc.append(min(ok, key=lambda x: x["cost"]) if ok else res(r, "c0", ["c0"]))
    curves["oracle (1 action)"].append(("", orc))

    # ---- report
    table = {}
    print(f"\n== CROSS-FITTED PLANNER (n={n}); cost = token-weighted units incl. DB; latency in seconds")
    print(f"  {'policy':44s} {'acc':>6} {'cost':>6} {'lat':>6} {'p95lat':>7} {'LLM/q':>6}  setting")
    for name, pts in curves.items():
        summ = [(summarize(rr), lab) for lab, rr in pts]
        table[name] = [(s, lab) for s, lab in summ]
        fr = pareto([(s[0], s[1], s[2], s[3], s[4], lab) for s, lab in summ], 1)
        for p in fr if len(fr) <= 6 else [fr[j] for j in np.linspace(0, len(fr) - 1, 6).astype(int)]:
            print(f"  {name:44s} {p[0]:6.3f} {p[1]:6.2f} {p[2]/1000:6.1f} {p[3]/1000:7.1f} {p[4]:6.2f}  {p[5]}")

    # ---- matched-budget comparisons (paired bootstrap) against the best baseline at each budget
    flash_cost = summarize(curves["fixed flash"][0][1])[1]
    rng = np.random.default_rng(0)

    def best_under(name, budget, key=1):
        names = name if isinstance(name, (list, tuple)) else [name]
        cand = [(summarize(rr), lab, rr) for nm in names if nm in curves for lab, rr in curves[nm]]
        cand = [c for c in cand if c[0][key] <= budget + 1e-9]
        return max(cand, key=lambda c: (c[0][0], -c[0][key])) if cand else None

    FULL = ["PlanGate-full", "PlanGate-full 2-phase", "PlanGate-full (latency-aware)"]
    NODB = ["PlanGate-nodb", "PlanGate-nodb 2-phase"]
    # "confidence cascade" baselines reuse PlanGate's DB-aware acceptor/selector -> reported, but
    # excluded from the external-baseline set.
    baseline_names = [k for k in curves if not k.startswith("PlanGate") and not k.startswith("oracle")
                      and not k.startswith("confidence cascade")]
    report = {"budgets": {}}
    print("\n== Matched COST budget: PlanGate-full (best policy family) vs best external baseline")
    for mult in (1.25, 1.5, 2.0, 3.0, 4.0):
        B = flash_cost * mult
        pg = best_under(FULL, B)
        bl = [(k, best_under(k, B)) for k in baseline_names]
        bl = [(k, v) for k, v in bl if v]
        kb, vb = max(bl, key=lambda kv: (kv[1][0][0], -kv[1][0][1]))
        nd = best_under(NODB, B)
        line = f"  budget {mult:.2f}x ({B:5.2f}): PlanGate-full {pg[0][0]:.3f} (cost {pg[0][1]:.2f})"
        line += f" | best baseline {vb[0][0]:.3f} [{kb}] | PlanGate-nodb {nd[0][0]:.3f}" if nd else ""
        a = np.array([x["correct"] for x in pg[2]]); b = np.array([x["correct"] for x in vb[2]])
        idx = rng.integers(0, n, (5000, n)); d = a[idx].mean(1) - b[idx].mean(1)
        line += f" | diff {a.mean()-b.mean():+.3f} CI[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        print(line)
        report["budgets"][str(mult)] = {"plangate": pg[0], "best_baseline": [kb, vb[0]], "nodb": nd[0] if nd else None,
                                         "diff": float(a.mean() - b.mean()),
                                         "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    # ---- head-to-head vs published methods: best PlanGate point with cost <= the method's cost
    print("\n== Head-to-head vs published methods (PlanGate point with cost <= method cost; paired bootstrap)")
    report["head_to_head"] = {}
    for fam in [k for k in curves if k.startswith("[paper]")]:
        for lab, rr in curves[fam]:
            s = summarize(rr)
            pg = best_under(FULL, s[1])
            if pg is None:
                continue
            a = np.array([x["correct"] for x in pg[2]]); b = np.array([x["correct"] for x in rr])
            idx = rng.integers(0, n, (5000, n)); d = a[idx].mean(1) - b[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {lab:34s} acc={s[0]:.3f} cost={s[1]:5.2f} | PlanGate {pg[0][0]:.3f} @ {pg[0][1]:5.2f} "
                  f"| diff {a.mean()-b.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}]")
            report["head_to_head"][lab] = {"method": s, "plangate": pg[0], "diff": float(a.mean() - b.mean()),
                                           "ci": [float(lo), float(hi)]}

    # ---- split-half validation: choose operating points on one half, evaluate on the other
    def pick(names, budget, idx):
        names = names if isinstance(names, (list, tuple)) else [names]
        best = None
        for nm in names:
            for lab, rr in curves.get(nm, []):
                sub = [rr[j] for j in idx]
                c = np.mean([x["cost"] for x in sub]); a = np.mean([x["correct"] for x in sub])
                if c <= budget + 1e-9 and (best is None or (a, -c) > (best[0], -best[1])):
                    best = (a, c, rr)
        return best

    perm = np.random.default_rng(1).permutation(n)
    halves = [perm[: n // 2], perm[n // 2:]]
    print("\n== Split-half validation (operating point chosen on one half, scored on the other; averaged)")
    report["split_half"] = {}
    for mult in (1.25, 1.5, 2.0, 3.0):
        B = flash_cost * mult
        accs = defaultdict(list); costs = defaultdict(list); diffs = []
        for h in (0, 1):
            tr, te = halves[h], halves[1 - h]
            chosen = {"PlanGate-full": pick(FULL, B, tr), "PlanGate-nodb": pick(NODB, B, tr)}
            for k in baseline_names:
                chosen[k] = pick(k, B, tr)
            ext = {k: v for k, v in chosen.items() if v and k in baseline_names}
            kb = max(ext, key=lambda k: (ext[k][0], -ext[k][1]))   # best baseline as judged on the train half
            for nm, key in (("PlanGate-full", "PlanGate-full"), ("PlanGate-nodb", "PlanGate-nodb"),
                            ("best external baseline", kb)):
                v = chosen[key]
                if v:
                    accs[nm].append(np.mean([v[2][j]["correct"] for j in te]))
                    costs[nm].append(np.mean([v[2][j]["cost"] for j in te]))
            a = np.array([chosen["PlanGate-full"][2][j]["correct"] for j in te])
            b = np.array([chosen[kb][2][j]["correct"] for j in te])
            diffs.append(a - b)
        d = np.concatenate(diffs)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print(f"  budget {mult:.2f}x: " + " | ".join(f"{k} {np.mean(accs[k]):.3f} (cost {np.mean(costs[k]):.2f})"
                                                   for k in accs)
              + f" | full - best-baseline {d.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
        report["split_half"][str(mult)] = {k: (float(np.mean(accs[k])), float(np.mean(costs[k]))) for k in accs}
        report["split_half"][str(mult)]["diff"] = [float(d.mean()), float(np.percentile(bs, 2.5)),
                                                    float(np.percentile(bs, 97.5))]

    # ---- equal-accuracy savings: cost / p95 latency PlanGate needs to match each strong baseline
    print("\n== Cost / latency PlanGate-full needs to match each baseline's accuracy")
    pg_pts = [(summarize(rr), lab) for nm in FULL for lab, rr in curves[nm]]
    report["match"] = {}
    targets = [(k, curves[k][0][1]) for k in (
        "fixed qwen", "self-consistency@3 (flash)", "self-consistency@2 (flash)", "ensemble@5 exec-vote",
        "ensemble@4 exec-vote", "ensemble@3 models exec-vote", "agreement cascade (MoT-style)",
        "rule cascade (DB trigger->qwen)") if k in curves]
    targets += [(lab, rr) for k in curves if k.startswith("[paper]") for lab, rr in curves[k]]
    for k, rr_k in targets:
        s = summarize(rr_k)
        ok = [p for p in pg_pts if p[0][0] >= s[0] - 1e-9]
        if ok:
            c = min(ok, key=lambda p: p[0][1]); l = min(ok, key=lambda p: p[0][3])
            print(f"  {k:32s} acc={s[0]:.3f} cost={s[1]:5.2f} p95={s[3]/1000:5.1f}s  ->  PlanGate cost {c[0][1]:5.2f} "
                  f"({c[0][1]/s[1]-1:+.0%}), p95 {l[0][3]/1000:5.1f}s ({l[0][3]/s[3]-1:+.0%})")
            report["match"][k] = {"baseline": s, "pg_cost": c[0][1], "pg_p95": l[0][3]}
        else:
            print(f"  {k:32s} acc={s[0]:.3f}: PlanGate-full max acc {max(p[0][0] for p in pg_pts):.3f} (not matched)")

    # ---- plan diversity at a mid operating point
    for lab, rr in curves["PlanGate-full"]:
        if lab == "lam=0.02":
            cnt = Counter(x["plan"] for x in rr)
            print(f"\n== Plan diversity, PlanGate-full lam=0.02 ({len(cnt)} distinct plans)")
            for p, k in cnt.most_common(12):
                acc = np.mean([x["correct"] for x in rr if x["plan"] == p])
                print(f"  {k:4d}  acc={acc:.2f}  {p}")
            # plan choice vs DB state
            fl = [triggers(r["c0"]["ev"]) for r in rows]
            for nm, cond in (("DB-flagged", True), ("unflagged", False)):
                c2 = Counter(x["plan"].split("->")[0] for x, f in zip(rr, fl) if f == cond)
                print(f"  first action | {nm:10s}: " + ", ".join(f"{a}={k}" for a, k in c2.most_common(6)))
            report["plans"] = dict(cnt)

    report["curves"] = {k: v for k, v in table.items()}
    (CROSSFIT_DIR / f"planner_report_{engine}.json").write_text(json.dumps(report, indent=1, default=float),
                                                                encoding="utf8")
    plot(table, engine)


def plot(table, engine):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    lines = {"PlanGate-full": ("#d62728", "-"), "PlanGate-full 2-phase": ("#17becf", "-"),
             "PlanGate-nodb 2-phase": ("#bcbd22", ":"),
             "PlanGate-full (pointwise select)": ("#e377c2", "--"),
             "PlanGate-exec_only": ("#ff7f0e", "--"),
             "PlanGate-nodb": ("#7f7f7f", ":"), "one-shot router": ("#1f77b4", "-"),
             "confidence cascade, replace (AutoMix-like)": ("#9467bd", "-."),
             "confidence cascade, verified select": ("#8c564b", "-.")}
    for ax, xi, xl in ((axes[0], 1, "mean cost per query (units)"), (axes[1], 3, "p95 latency (s)")):
        for name, (col, ls) in lines.items():
            if name not in table:
                continue
            pts = pareto([(s[0], s[1], s[2], s[3], s[4], lab) for s, lab in table[name]], xi)
            xs = [p[xi] / (1000 if xi == 3 else 1) for p in pts]
            ax.plot(xs, [p[0] for p in pts], ls, color=col, lw=2, label=name, drawstyle="steps-post", marker=".")
        for name, mk in (("fixed flash", "o"), ("fixed pro", "o"), ("fixed qwen", "o"), ("self-consistency@3 (flash)", "^"),
                         ("self-consistency@2 (flash)", "v"), ("ensemble@5 exec-vote", "P"),
                         ("ensemble@4 exec-vote", "p"), ("ensemble@3 models exec-vote", "X"),
                         ("agreement cascade (MoT-style)", "D"), ("rule cascade (DB trigger->qwen)", "s")):
            if name not in table:
                continue
            s = table[name][0][0]
            ax.scatter(s[xi] / (1000 if xi == 3 else 1), s[0], marker=mk, s=55, zorder=5, label=name)
        for name in [k for k in table if k.startswith("[paper]")]:
            pts = sorted(table[name], key=lambda t: t[0][xi])
            ax.plot([p[0][xi] / (1000 if xi == 3 else 1) for p in pts], [p[0][0] for p in pts], "k--", lw=1,
                    marker="s" if "AutoMix" in name else "*", ms=8, mfc="white", zorder=6,
                    label=name.replace("[paper] ", "") + " (re-impl.)")
        o = table["oracle (1 action)"][0][0]
        ax.scatter(o[xi] / (1000 if xi == 3 else 1), o[0], marker="*", s=200, color="gold", edgecolor="k",
                   zorder=6, label="oracle (best single action)")
        ax.set_xlabel(xl); ax.set_ylabel("execution accuracy"); ax.grid(alpha=0.3)
    axes[0].set_xlim(2.5, 22); axes[1].set_xlim(0, 60)
    axes[1].legend(fontsize=7, loc="lower right")
    axes[0].set_title("Accuracy vs cost"); axes[1].set_title("Accuracy vs p95 latency")
    path = CROSSFIT_DIR / f"planner_frontier_{engine}.png"
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print("saved", path)


if __name__ == "__main__":
    main()
