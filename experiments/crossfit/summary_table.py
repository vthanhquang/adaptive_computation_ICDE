"""Summary table of all methods on both engines from the saved planner reports and published-method runs.

Usage:  python -m experiments.crossfit.summary_table
Output: results/crossfit/summary_table.md
"""
import json

import numpy as np

from plangate.paths import BASELINES_DIR, CROSSFIT_DIR


def stats(v):
    return (np.mean([x["correct"] for x in v]), np.mean([x["cost"] for x in v]),
            np.percentile([x["lat"] for x in v], 95) / 1000)


def curve_points(rep, name):
    return [tuple(s) for s, _ in rep["curves"].get(name, [])]


def best(points, budget=None):
    pts = [p for p in points if budget is None or p[1] <= budget + 1e-9]
    return max(pts, key=lambda p: (p[0], -p[1])) if pts else None


def engine_table(engine):
    rep = json.loads((CROSSFIT_DIR / f"planner_report_{engine}.json").read_text(encoding="utf8"))
    pb = json.loads((BASELINES_DIR / f"published_methods_{engine}.json").read_text(encoding="utf8"))
    tab = []
    for name in ("fixed flash", "fixed pro", "fixed qwen"):
        p = curve_points(rep, name)
        if p:
            tab.append(("Fixed model", name.replace("fixed ", ""), p[0][0], p[0][1], p[0][3] / 1000))
    for name, label in (("self-consistency@3 (flash)", "Self-consistency@3"), ("self-consistency@2 (flash)", "Self-consistency@2"),
                        ("ensemble@3 models exec-vote", "Ensemble 3 models (exec vote)"),
                        ("ensemble@5 exec-vote", "Ensemble 5 candidates (exec vote)"),
                        ("agreement cascade (MoT-style)", "MoT-style agreement cascade"),
                        ("rule cascade (DB trigger->qwen)", "Rule cascade (DB trigger)")):
        p = curve_points(rep, name)
        if p:
            tab.append(("Generic baseline", label, p[0][0], p[0][1], p[0][3] / 1000))
    p = best(curve_points(rep, "one-shot router"))
    tab.append(("Generic baseline", "One-shot router (best point)", p[0], p[1], p[3] / 1000))
    groups = {}
    for k, v in pb["policies"].items():
        a, c, l = stats(v)
        fam = k.split(" (theta")[0].split(", t=")[0]
        if fam not in groups or a > groups[fam][1][0]:
            groups[fam] = (k, (a, c, l))
    for fam, (k, (a, c, l)) in sorted(groups.items()):
        tab.append(("Published method (re-impl.)", k, a, c, l))
    full = []
    for nm in ("PlanGate-full", "PlanGate-full 2-phase", "PlanGate-full (latency-aware)"):
        full += curve_points(rep, nm)
    flash_cost = curve_points(rep, "fixed flash")[0][1]
    for mult in (1.25, 2.0, None):
        p = best(full, None if mult is None else flash_cost * mult)
        lab = f"PlanGate (budget {mult}x)" if mult else "PlanGate (best point)"
        tab.append(("PlanGate", lab, p[0], p[1], p[3] / 1000))
    nodb = []
    for nm in ("PlanGate-nodb", "PlanGate-nodb 2-phase"):
        nodb += curve_points(rep, nm)
    p = best(nodb)
    tab.append(("Ablation", "PlanGate without DB (best point)", p[0], p[1], p[3] / 1000))
    o = curve_points(rep, "oracle (1 action)")[0]
    tab.append(("Upper bound", "Oracle (best single action)", o[0], o[1], o[3] / 1000))
    return tab


def main():
    lines = []
    for engine, title in (("sqlite", "SQLite (n=300)"), ("pg", "PostgreSQL (n=500)")):
        lines += ["", f"### {title}", "", "| Group | Method | EX | Cost/query | p95 latency (s) |", "|---|---|---|---|---|"]
        lines += [f"| {g} | {m} | {a:.3f} | {c:.2f} | {l:.1f} |" for g, m, a, c, l in engine_table(engine)]
    text = "\n".join(lines) + "\n"
    print(text)
    (CROSSFIT_DIR / "summary_table.md").write_text(text, encoding="utf8", newline="\n")


if __name__ == "__main__":
    main()
