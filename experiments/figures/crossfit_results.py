"""Paper-quality result figures (vector PDF), in the style of the teaser.

  frontier.pdf : accuracy vs. cost and vs. p95 latency on SQLite and PostgreSQL
                 (results/crossfit/planner_report_<engine>.json)
  feedback.pdf : cumulative accuracy gain of the feedback loop along the stream
                 (results/crossfit/online_learning_from_logs_<engine>.json)

Usage: python -m experiments.figures.crossfit_results   (writes results/figures/{frontier,feedback}.{pdf,png})
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plangate.paths import CROSSFIT_DIR, FIGURES_DIR

OUT = FIGURES_DIR
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# colours shared with the teaser: PlanGate = CON, DB-related = DB
CON, DB, DARK, MID, LIGHT = "#a3281e", "#1f4e9c", "#222222", "#7a7a7a", "#bdbdbd"
OK = {"orange": "#e69f00", "sky": "#56b4e9", "green": "#009e73", "purple": "#cc79a7", "blue": "#0072b2"}


def pareto(pts, x):
    """Upper-left frontier of (acc, cost, lat_mean, lat_p95, n_llm) tuples along column x."""
    pts = sorted(pts, key=lambda p: (p[x], -p[0]))
    out, best = [], -1.0
    for p in pts:
        if p[0] > best + 1e-12:
            out.append(p); best = p[0]
    return out


def family(C, names):
    return [tuple(s) for n in names for s, _ in C.get(n, [])]


# ---------------------------------------------------------------------------------- Fig. 4
FRONTIERS = [   # label, report families, style
    ("PlanGate", ["PlanGate-full", "PlanGate-full 2-phase", "PlanGate-full (latency-aware)"],
     dict(color=CON, lw=1.5, ls="-", zorder=6)),
    ("PlanGate, no DB access", ["PlanGate-nodb", "PlanGate-nodb 2-phase"],
     dict(color=MID, lw=1.0, ls=(0, (4, 1.5)), zorder=5)),
    ("One-shot router (P0)", ["one-shot router"],
     dict(color=OK["blue"], lw=0.9, ls=(0, (1, 1.2)), zorder=4)),
]
POINTS = [      # label, family, marker, colour
    ("deepseek-v4-flash", "fixed flash", "o", DARK),
    ("deepseek-v4-pro", "fixed pro", "o", MID),
    ("qwen3.8-max", "fixed qwen", "o", "white"),
    ("Rule cascade", "rule cascade (DB trigger->qwen)", "s", OK["green"]),
    ("Agreement cascade", "agreement cascade (MoT-style)", "h", OK["orange"]),
    ("Self-consistency", ["self-consistency@3 (flash)", "self-consistency@2 (flash)"], "^", OK["sky"]),
    ("Exec.-vote ensemble", ["ensemble@5 exec-vote", "ensemble@4 exec-vote", "ensemble@3 models exec-vote"],
     "D", OK["purple"]),
]


def frontier_panel(ax, C, x, flash_cost, ylim, xlim):
    for lab, fams, st in FRONTIERS:
        fr = pareto(family(C, fams), x)
        xs = [p[x] / (1000 if x == 3 else 1) for p in fr]
        ys = [p[0] for p in fr]
        if xs:   # extend the last step to the right edge, as a frontier does
            xs, ys = xs + [xlim[1]], ys + [ys[-1]]
        ax.step(xs, ys, where="post", **st)
    for lab, fam, mk, col in POINTS:
        fams = fam if isinstance(fam, list) else [fam]
        pts = family(C, fams)
        if not pts:
            continue
        for p in pts:
            ax.scatter(p[x] / (1000 if x == 3 else 1), p[0], marker=mk, s=17, zorder=7,
                       facecolor=col, edgecolor=DARK, lw=0.5)
    if x == 1:   # budget guides: multiples of the cheap model's cost
        for m in (2, 4):
            ax.axvline(m * flash_cost, color=LIGHT, lw=0.5, ls=(0, (2, 2)), zorder=1)
            ax.text(m * flash_cost + 0.4, ylim[0] + 0.002, f"{m:g}$\\times$", fontsize=5.4, color=MID,
                    ha="left", va="bottom")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e6e6e6", lw=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def fig_frontier():
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95), gridspec_kw=dict(wspace=0.32))
    panels = [("sqlite", 1, "(a) SQLite: cost", (0, 26)), ("sqlite", 3, "(b) SQLite: p95 latency", (25, 66)),
              ("pg", 1, "(c) PostgreSQL: cost", (0, 26)), ("pg", 3, "(d) PostgreSQL: p95 latency", (10, 50))]
    for ax, (tag, x, title, xlim) in zip(axes, panels):
        C = json.loads((CROSSFIT_DIR / f"planner_report_{tag}.json").read_text(encoding="utf8"))["curves"]
        flash = C["fixed flash"][0][0][1]
        ylim = (0.585, 0.69)
        frontier_panel(ax, C, x, flash, ylim, xlim)
        ax.set_title(title, pad=3)
        ax.set_xlabel("mean cost per question" if x == 1 else "p95 latency (s)")
        if ax is axes[0]:
            ax.set_ylabel("execution accuracy")
    handles = [Line2D([], [], **{k: v for k, v in st.items() if k != "zorder"}, label=lab)
               for lab, _, st in FRONTIERS]
    handles += [Line2D([], [], marker=mk, ls="none", ms=4, mfc=col, mec=DARK, mew=0.5, label=lab)
                for lab, _, mk, col in POINTS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.2),
               handlelength=2.2, columnspacing=1.2, handletextpad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"frontier.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved", OUT / "frontier.pdf")


# ---------------------------------------------------------------------------------- Fig. 5
FEEDBACK = [    # label, regime, style
    ("Full outcomes (bound)", "gold-full (ADV)", dict(color=DARK, lw=1.0, ls="-")),
    ("Logs, gold labels", "logs, gold 100% (ADV)", dict(color=OK["blue"], lw=0.9, ls=(0, (4, 1.5)))),
    ("Logs, DB + agreement", "logs, self-db (ADV)", dict(color=OK["green"], lw=0.9, ls=(0, (1, 1.2)))),
    ("  + 25% audit, uncertain", "logs, self-db + audit 25% unc (ADV)", dict(color=CON, lw=1.5, ls="-")),
    ("  + 25% audit, random", "logs, self-db + audit 25% rand (ADV)", dict(color=CON, lw=0.9, ls=(0, (3, 1.2)))),
]


def fig_feedback():
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, tag, title in ((axes[0], "sqlite", "(a) SQLite"), (axes[1], "pg", "(b) PostgreSQL")):
        rep = json.loads((CROSSFIT_DIR / f"online_learning_from_logs_{tag}.json").read_text(encoding="utf8"))
        Rg, n, B = rep["regimes"], rep["n"], rep["block"]
        nb = len(Rg["static"]["curve"])
        w = np.minimum(B, n - np.arange(nb) * B)
        x = np.cumsum(w)                                   # questions served at the end of each block
        st = np.array(Rg["static"]["curve"])[:, 0]
        # gain over the static policy on a trailing window of n/3 questions, so that the value at
        # the end of the stream equals the "last third" numbers of Table VIII
        W = n // 3
        start = int(np.searchsorted(x, W))               # first block end with a full window
        ax.axhline(0, color=MID, lw=0.6, zorder=1)

        def trailing(v, j):
            """Mean of per-question values over the W questions ending at block j (blocks are
            averages; the oldest block in the window enters with a fractional weight)."""
            tot, cnt = 0.0, 0.0
            for i in range(j, -1, -1):
                take = min(w[i], W - cnt)
                tot += v[i] * take; cnt += take
                if cnt >= W:
                    break
            return tot / cnt

        for lab, reg, sty in FEEDBACK:
            if reg not in Rg:
                continue
            d = np.array(Rg[reg]["curve"])[:, 0] - st
            gain = [100 * trailing(d, j) for j in range(start, nb)]
            ax.plot(x[start:], gain, label=lab, zorder=4, **sty)
            print(f"  {tag} {lab:28s} end={gain[-1]:+.2f}  table={Rg[reg].get('vs_static')}")
        ax.set_title(title, pad=3)
        ax.set_xlim(x[start], n); ax.set_xlabel("questions served")
        ax.grid(axis="y", color="#e6e6e6", lw=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("EX gain over static (points),\ntrailing $n/3$ questions")
    axes[0].set_ylim(-2, 4.5)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.53, 1.27), handlelength=2.2, columnspacing=1.0, handletextpad=0.4,
               fontsize=5.9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"feedback.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved", OUT / "feedback.pdf")


if __name__ == "__main__":
    fig_frontier()
    fig_feedback()
