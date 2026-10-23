"""LaTeX tables for the paper, generated from the frozen held-out and scaling results.

Usage:  python -m experiments.heldout.make_tables
Output: results/tables/{heldout_main,heldout_ci,tail_x100,pricing,ablation,external_heldout}.tex
"""
import json

from plangate.evaluation.heldout import run_dir, run_summary
from plangate.paths import HELDOUT_DIR, TABLES_DIR

OUT = TABLES_DIR
OUT.mkdir(parents=True, exist_ok=True)
POL = [("no_db", "No DB evidence"), ("fixed_bundle", "Fixed bundle"), ("optimizer_gated", "Optimizer-gated"),
       ("rule_selective", "Rule-selective"), ("adaptive", "Adaptive (VOI)")]
RUNS = [("pg", "PostgreSQL, 1$\\times$"), ("pg_x10", "PostgreSQL, 10$\\times$ costs"),
        ("pg_x100", "PostgreSQL, 100$\\times$ costs"), ("sqlite", "SQLite")]


def S(d):
    return run_summary(d)


def num(x):
    return f"{x:+.1f}".replace("-", "$-$")


def ci(b):
    return f"{num(100*b[0])} [{num(100*b[1])}, {num(100*b[2])}]"


def main_table(budgets=("1.5", "2.0", "3.0")):
    lines = [r"\begin{tabular}{@{}ll" + "rr" * len(budgets) + r"@{}}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}$\times$}}" for b in budgets) + r" \\",
             "".join(rf"\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(budgets))),
             r"Setting & Policy & " + " & ".join(["EX & DB ms"] * len(budgets)) + r" \\", r"\midrule"]
    for d, name in RUNS:
        s = S(d)
        ch = s["chosen"]["1.0"]
        first = True
        for pol, lab in POL:
            cells = []
            for b in budgets:
                v = ch.get(b, {}).get(pol)
                cells += ([f"{v['test']['acc']:.3f}", f"{v['test']['db_ms']:.0f}"] if v else ["--", "--"])
            row = (rf"\multirow{{5}}{{*}}{{\shortstack[l]{{{name}\\($n{{=}}{s['n']['test']}$)}}}}" if first else "") + \
                  f" & {lab} & " + " & ".join(cells) + r" \\"
            lines.append(row if pol != "adaptive" else row.replace(f"& {lab} &", rf"& \textbf{{{lab}}} &"))
            first = False
        lines.append(r"\midrule" if d != RUNS[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "heldout_main.tex").write_text("\n".join(lines), encoding="utf8")


def ci_table(budgets=("1.25", "1.5", "2.0", "3.0")):
    lines = [r"\begin{tabular}{@{}l" + "r" * len(budgets) + r"@{}}", r"\toprule",
             r"Setting & " + " & ".join(f"{b}$\\times$" for b in budgets) + r" \\", r"\midrule"]
    for d, name in RUNS:
        s = S(d)
        bt = s["boot"].get("1.0", {})
        for ref, lab in (("optimizer_gated", "vs.\\ optimizer-gated"), ("fixed_bundle", "vs.\\ fixed bundle")):
            cells = [ci(bt[b][ref]) if b in bt and ref in bt[b] else "--" for b in budgets]
            if all(c == "--" for c in cells):
                continue
            lines.append(f"{name}, {lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "heldout_ci.tex").write_text("\n".join(lines), encoding="utf8")


def tail_table():
    """DB mean / p95 and wall p95 at 1.25x and 1.5x for 100x costs."""
    s = S("pg_x100")["chosen"]["1.0"]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Budget, policy & EX & Cost & DB ms & DB p95 ms & Probes/q \\", r"\midrule"]
    for b in ("1.25", "1.5", "2.0"):
        for pol, lab in POL[1:]:
            v = s.get(b, {}).get(pol)
            if not v:
                continue
            t = v["test"]
            lines.append(f"{b}$\\times$, {lab} & {t['acc']:.3f} & {t['cost']:.2f} & {t['db_ms']:.0f} & {t['db_p95']:.0f} & "
                         f"{t['n_vp']+t['n_ex']:.2f} \\\\")
        lines.append(r"\midrule" if b != "2.0" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "tail_x100.tex").write_text("\n".join(lines), encoding="utf8")


def pricing_table():
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Setting & Probe & $\rho$ & AUROC & $>$100\,ms \\", r"\midrule"]
    short = {"pg": "PG 1$\\times$", "pg_x10": "PG 10$\\times$", "pg_x100": "PG 100$\\times$", "sqlite": "SQLite"}
    for d, name in RUNS:
        name = short[d]
        p = run_dir(d) / "policy_analysis_p1.0_B2.0.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text(encoding="utf8"))["pricing"]
        for op, lab in (("EXEC_BOUNDED", "exec"), ("VALUE_PROBE", "value probe")):
            x = a[op]
            auc = "--" if x["auroc_gt100ms"] != x["auroc_gt100ms"] else f"{x['auroc_gt100ms']:.3f}"
            lines.append(f"{name} & {lab} & {x['spearman']:.3f} & {auc} & {100*x['share_gt100ms']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "pricing.tex").write_text("\n".join(lines), encoding="utf8")


def ablation_table():
    """Probe ablations from <eval_dir>/ablation.json (EX / mean DB ms at the calibrated points)."""
    pols = [("adaptive", "Adaptive (both probes)"), ("adaptive_noVP", "Adaptive, no \\op{value probe}"),
            ("adaptive_noEX", "Adaptive, no \\op{bounded exec}"), ("fixed_vp_only", "Bundle: \\op{value probe} only"),
            ("fixed_ex_only", "Bundle: \\op{bounded exec} only"), ("fixed_bundle", "Bundle: both"),
            ("optimizer_gated", "Optimizer-gated")]
    runs = [(d, n) for d, n in (("pg", "PG 1$\\times$"), ("pg_x100", "PG 100$\\times$ costs"))
            if (run_dir(d) / "ablation.json").exists()]
    if not runs:
        return
    budgets = ("1.5", "2.0", "3.0")
    lines = [r"\begin{tabular}{@{}l" + "r" * (len(budgets) * len(runs)) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(budgets)}}}{{c}}{{{n}}}" for _, n in runs) + r" \\",
             "".join(rf"\cmidrule(lr){{{2+len(budgets)*i}-{1+len(budgets)*(i+1)}}}" for i in range(len(runs))),
             "Policy & " + " & ".join(f"{b}$\\times$" for _ in runs for b in budgets) + r" \\", r"\midrule"]
    A = {d: json.loads((run_dir(d) / "ablation.json").read_text(encoding="utf8"))["budgets"] for d, _ in runs}
    for pol, lab in pols:
        cells = []
        for d, _ in runs:
            for b in budgets:
                v = A[d].get(b, {}).get(pol)
                cells.append(f"{v['test']['acc']:.3f}\\,/\\,{v['test']['db_ms']:.0f}" if v else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ablation.tex").write_text("\n".join(lines), encoding="utf8")


def external_table():
    """Held-out PlanGate vs. the best external baseline that fits each budget (external_baselines.json).
    PlanGate = the original design: optimizer-gated on PostgreSQL, full bundle on SQLite."""
    p = HELDOUT_DIR / "external_baselines.json"
    if not p.exists():
        return
    E = json.loads(p.read_text(encoding="utf8"))
    nice = {"flash": "cheap model", "rule cascade (DB trigger)": "rule cascade", "agreement cascade (MoT)": "MoT cascade",
            "qwen": "strong model", "pro": "mid model", "self-consistency@2": "self-cons.@2",
            "ensemble-3 exec vote": "ensemble-3"}
    runs = [("pg", "PostgreSQL ($n{=}498$)", "optimizer_gated"), ("sqlite", "SQLite ($n{=}300$)", "fixed_bundle")]
    lines = [r"\begin{tabular}{@{}lllrlll@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{PostgreSQL ($n{=}498$)} & \multicolumn{3}{c}{SQLite ($n{=}300$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Budget & best external & \sys{} & $\Delta$ [95\% CI] & best external & \sys{} & $\Delta$ [95\% CI] \\", r"\midrule"]
    for b in ("1.25", "1.5", "2.0", "3.0", "4.0"):
        cells = []
        for d, _, pol in runs:
            row = E.get(d, {}).get("budgets", {}).get(b, {})
            if not row or pol not in row:
                cells += ["--", "--", "--"]; continue
            kb, vb = row["best_external"]
            x = row[pol]
            cells += [f"{nice.get(kb, kb)} {vb['acc']:.3f}", f"{x['acc']:.3f}", ci(x["vs_best_external"])]
        lines.append(f"{b}$\\times$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "external_heldout.tex").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main_table(); ci_table(); tail_table(); pricing_table(); ablation_table(); external_table()
    for f in sorted(OUT.glob("*.tex")):
        print(f.name)
