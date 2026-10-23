"""Paper figures for the held-out adaptive-evidence evaluation (vector PDF, paper style).

  heldout_frontier.pdf : test accuracy vs. cost per policy (upper frontier over the parameter grid),
                         PostgreSQL 1x, x10 and x100 costs, and SQLite
  scaling_dbtime.pdf   : DB time per question of each policy's calibrated operating point at 2x budget,
                         at 1x / x10 / x100 data scale (log axis)
Reads results/{heldout,scaling}/<run>/summary.json.
Usage: python -m experiments.figures.heldout_results   (writes results/figures/{heldout_frontier,scaling_dbtime}.{pdf,png})
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plangate.evaluation.heldout import run_dir, run_summary
from plangate.paths import FIGURES_DIR

OUT = FIGURES_DIR
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.labelpad": 1.5, "pdf.fonttype": 42})
CON, MID = "#a3281e", "#7a7a7a"
STY = {"adaptive": dict(color=CON, lw=1.5, ls="-", label="Adaptive evidence (ours)"),
       "optimizer_gated": dict(color="#0072b2", lw=1.0, ls=(0, (4, 1.5)), label="Optimizer-gated bundle"),
       "fixed_bundle": dict(color="#222222", lw=1.0, ls="-", label="Fixed bundle"),
       "rule_selective": dict(color="#009e73", lw=0.9, ls=(0, (1, 1.2)), label="Rule-selective"),
       "no_db": dict(color=MID, lw=0.9, ls=(0, (3, 1.2)), label="No DB evidence")}
RUNS = [("pg", "1.0", "(a) PostgreSQL, 1$\\times$ data"), ("pg_x10", "1.0", "(b) PostgreSQL, 10$\\times$ costs"),
        ("pg_x100", "1.0", "(c) PostgreSQL, 100$\\times$ costs"), ("sqlite", "1.0", "(d) SQLite (BIRD dev)")]


def pareto(pts):
    pts = sorted(pts, key=lambda p: (p[0], -p[1]))
    out, best = [], -1
    for c, a in pts:
        if a > best + 1e-12:
            out.append((c, a)); best = a
    return out


def frontier_fig():
    runs = [(d, p, t) for d, p, t in RUNS if (run_dir(d) / "summary.json").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(7.16, 1.85), gridspec_kw=dict(wspace=0.3))
    axes = np.atleast_1d(axes)
    for ax, (d, p, title) in zip(axes, runs):
        S = run_summary(d)
        F = S["frontier"][p]
        cheap = S["chosen"][p]["cheap_only"]
        xmax = cheap["cost"] * 4.2
        for pol, st in STY.items():
            pts = [(x["cost"], x["acc"]) for x in F if x["policy"] == pol and x["cost"] <= xmax]
            fr = pareto(pts)
            if fr:
                xs = [c for c, _ in fr] + [xmax]; ys = [a for _, a in fr] + [fr[-1][1]]
                ax.step(xs, ys, where="post", **{k: v for k, v in st.items() if k != "label"})
        ax.scatter([cheap["cost"]], [cheap["acc"]], marker="o", s=14, color="#222222", zorder=6)
        ax.set_title(title, pad=3); ax.set_xlabel("mean cost per question")
        ax.set_xlim(cheap["cost"] * 0.9, xmax)
        ax.grid(axis="y", color="#e6e6e6", lw=0.4); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("execution accuracy (test)")
    h = [plt.Line2D([], [], **{k: v for k, v in st.items()}) for st in STY.values()]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"heldout_frontier.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved heldout_frontier.pdf")


def scaling_fig(budget="2.0"):
    scales = [("pg", "1$\\times$"), ("pg_x10", "10$\\times$"), ("pg_x100", "100$\\times$")]
    scales = [(d, l) for d, l in scales if (run_dir(d) / "summary.json").exists()]
    if not scales:
        return
    pols = ["fixed_bundle", "optimizer_gated", "rule_selective", "adaptive"]
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    w = 0.2
    for j, pol in enumerate(pols):
        vals, accs = [], []
        for d, _ in scales:
            S = run_summary(d)
            c = S["chosen"]["1.0"].get(budget, {}).get(pol)
            vals.append(c["test"]["db_ms"] if c else np.nan); accs.append(c["test"]["acc"] if c else np.nan)
        x = np.arange(len(scales)) + (j - 1.5) * w
        ax.bar(x, vals, w, color=STY[pol]["color"], label=STY[pol]["label"], zorder=3)
        for xi, v, a in zip(x, vals, accs):
            if not np.isnan(v):
                ax.text(xi, v * 1.15, f"{a:.3f}", ha="center", va="bottom", fontsize=4.6, rotation=90)
    ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 4); ax.set_xticks(range(len(scales))); ax.set_xticklabels([l for _, l in scales])
    ax.set_ylabel("DB time per question (ms)"); ax.set_xlabel("data scale (PostgreSQL); bar labels: test EX at the 2$\\times$ budget")
    ax.grid(axis="y", color="#e6e6e6", lw=0.4, zorder=0); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"scaling_dbtime.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved scaling_dbtime.pdf")


if __name__ == "__main__":
    frontier_fig()
    scaling_fig()
