"""Infographic figures for the paper (vector PDF, IEEE double-column widths).

  system_overview.pdf : Fig. 1, online plan-act-observe loop with optional priced evidence and the
                        offline partially-observed training pipeline, plus a cost-at-scale inset
  examples.pdf        : Fig. 2, (a) BIRD Q650 "ask the database first" (measured trace), and
                        (b) Q368 at 100x data "knowing when not to ask" (measured, frozen policy)
All numbers are read from results/ (crossfit/trace_650.json, scaling/example.json, scale costs).
Usage: python -m experiments.figures.infographics   (writes results/figures/{system_overview,examples}.{pdf,png})
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Rectangle, Wedge, FancyArrowPatch, Polygon
import numpy as np

from experiments.scaling.cost_summary import SCALES, operator_costs
from plangate.paths import CROSSFIT_DIR, FIGURES_DIR, SCALING_DIR

OUT = FIGURES_DIR
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 6.5})

# palette
INK = "#1d2433"; MUTE = "#6b7385"; LINE = "#c9cfdb"
RED = "#b8322a"; RED_L = "#fbe9e7"
BLUE = "#1f5aa6"; BLUE_L = "#e8f0fb"
GREEN = "#1f8a5b"; GREEN_L = "#e6f4ec"
AMBER = "#c77d0a"; AMBER_L = "#fdf3e1"
GRAY_L = "#f3f4f7"; PURPLE = "#6a4c93"; PURPLE_L = "#f0ebf7"


# ------------------------------------------------------------------ primitives
def card(ax, x, y, w, h, fc="white", ec=LINE, lw=0.7, r=0.9, shadow=True, z=2):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.35, y - 0.35), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc="#000000", ec="none", alpha=0.07, zorder=z - 0.1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def badge(ax, x, y, n, color=INK, r=1.25, fs=6.2, z=6):
    ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=0.8, zorder=z))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", color="white", fontsize=fs, weight="bold", zorder=z + 1)


def tag(ax, x, y, text, color, fs=5.2, z=7):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="white", weight="bold", zorder=z,
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc=color, ec="none"))


def arrow(ax, p, q, color=INK, lw=0.9, ls="-", rad=0.0, ms=7, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def db_icon(ax, x, y, s=1.0, color=BLUE, z=6):
    w, h = 3.0 * s, 3.4 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=color, ec="none", zorder=z))
    for yy in (y - h / 2, y + h / 2, y):
        ax.add_patch(Ellipse((x, yy), w, 1.1 * s, fc=color if yy != y + h / 2 else "#5b8fd6", ec="white",
                             lw=0.5 if yy == y else 0.6, zorder=z + 0.1))


def llm_icon(ax, x, y, s=1.0, color=INK, z=6):
    ax.add_patch(FancyBboxPatch((x - 1.6 * s, y - 1.6 * s), 3.2 * s, 3.2 * s, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=color, ec="none", zorder=z))
    for k in (-0.9, 0, 0.9):
        for side in (-1, 1):
            ax.plot([x + k * s, x + k * s], [y + side * 1.6 * s, y + side * 2.1 * s], color=color, lw=0.7, zorder=z)
    ax.text(x, y - 0.05, "LLM", ha="center", va="center", color="white", fontsize=4.6 * s, weight="bold", zorder=z + 1)


def gauge(ax, x, y, r=1.9, frac=0.75, z=6):
    ax.add_patch(Wedge((x, y), r, 0, 180, width=0.55, fc="#dfe4ee", ec="none", zorder=z))
    ax.add_patch(Wedge((x, y), r, 180 - 180 * frac, 180, width=0.55, fc=AMBER, ec="none", zorder=z + 0.1))
    ang = np.deg2rad(180 - 180 * frac)
    ax.plot([x, x + 0.85 * r * np.cos(ang)], [y, y + 0.85 * r * np.sin(ang)], color=INK, lw=0.9, zorder=z + 0.2)
    ax.add_patch(Circle((x, y), 0.25, fc=INK, zorder=z + 0.3))


def chip(ax, x, y, text, state, fs=4.9, w=None, z=7):
    """state: 'acq' (acquired, solid), 'unk' (unknown, dashed), 'neg' (acquired, flags an error)"""
    fc, ec, tc, ls = {"acq": (BLUE_L, BLUE, BLUE, "-"), "unk": ("white", "#9aa3b5", MUTE, (0, (2, 1.2))),
                      "neg": (RED_L, RED, RED, "-")}[state]
    w = w or 0.62 * len(text) + 1.2
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.75), w, 1.5, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=fc, ec=ec, lw=0.6, linestyle=ls, zorder=z))
    ax.text(x, y - 0.03, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def title(ax, x, y, text, fs=6.8, color=INK, ha="left", z=8):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, weight="bold", color=color, zorder=z)


def body(ax, x, y, text, fs=5.5, color=INK, ha="left", va="center", z=8, **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=z, linespacing=1.25, **kw)


# ------------------------------------------------------------------ Fig. 1
def system_overview():
    fig = plt.figure(figsize=(7.16, 3.45))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
    # bands
    ax.add_patch(FancyBboxPatch((0.6, 21.0), 98.8, 26.4, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f8fc",
                                ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 98.8, 19.4, boxstyle="round,pad=0,rounding_size=1.2", fc="#fbf8f2",
                                ec="none", zorder=0))
    ax.text(2.0, 46.0, "ONLINE  ·  per question", fontsize=6.3, weight="bold", color=BLUE, va="center")
    ax.text(2.0, 18.7, "OFFLINE  ·  periodic, on a training workload", fontsize=6.3, weight="bold", color=AMBER, va="center")

    # (1) question
    card(ax, 2.0, 28.0, 11.0, 14.5)
    badge(ax, 2.6, 42.0, 1)
    title(ax, 4.4, 40.6, "Question")
    ax.add_patch(FancyBboxPatch((3.3, 31.6), 8.4, 6.0, boxstyle="round,pad=0,rounding_size=0.8", fc=GRAY_L, ec="none", zorder=3))
    ax.add_patch(Polygon([[4.6, 31.6], [5.8, 31.6], [4.4, 30.2]], fc=GRAY_L, ec="none", zorder=3))
    body(ax, 7.5, 34.6, "“Last access\ndate of users\nwith badge\n‘outliers’?”", fs=4.9, ha="center", style="italic")
    body(ax, 7.5, 29.3, "+ schema $S$, hint", fs=4.7, ha="center", color=MUTE)

    # (2) cheap LLM
    card(ax, 15.0, 28.0, 11.0, 14.5)
    badge(ax, 15.6, 42.0, 2)
    title(ax, 17.4, 40.6, "Generate")
    llm_icon(ax, 20.5, 35.4, s=1.05)
    body(ax, 20.5, 31.0, "cheap model $\\rightarrow c_0$", fs=5.0, ha="center")
    body(ax, 20.5, 29.3, "≈ 3 units, 3–5 s", fs=4.7, ha="center", color=MUTE)

    # (3) mandatory checks + pricing
    card(ax, 28.0, 28.0, 12.5, 14.5)
    badge(ax, 28.6, 42.0, 3)
    title(ax, 30.4, 40.6, "Check & price")
    tag(ax, 38.9, 37.6, "C1", BLUE)
    db_icon(ax, 31.4, 35.4, s=0.95)
    gauge(ax, 36.6, 34.4, r=2.0, frac=0.8)
    body(ax, 34.25, 31.0, "PARSE/BIND, EXPLAIN", fs=4.8, ha="center")
    body(ax, 34.25, 29.3, "price $\\hat C(e)$ before buying", fs=4.7, ha="center", color=MUTE)

    # (4) planner
    card(ax, 42.5, 22.6, 33.5, 19.9, ec=RED, lw=0.9)
    badge(ax, 43.1, 42.0, 4, color=RED)
    title(ax, 44.9, 40.6, "Adaptive evidence planner", color=RED)
    tag(ax, 73.6, 40.6, "C2", RED)
    # state
    body(ax, 44.4, 37.7, "partially observed state $s$", fs=4.9, color=MUTE)
    for k, (cand, chips) in enumerate((("$c_0$", [("PB", "acq"), ("VP", "neg"), ("EX", "unk")]),
                                       ("$c_1$", [("PB", "acq"), ("VP", "unk"), ("EX", "acq")]))):
        yy = 35.6 - 2.3 * k
        body(ax, 44.6, yy, cand, fs=5.6)
        for j, (t, st) in enumerate(chips):
            chip(ax, 48.6 + 3.4 * j, yy, t, st, w=3.0)
    body(ax, 44.4, 30.6, "dashed = not observed (unknown)", fs=4.3, color=MUTE)
    # three option columns
    cols = [(60.0, "Accept", "$V(s)$", GREEN, GREEN_L), (66.0, "LLM action", "$Q-\\lambda C$", INK, GRAY_L),
            (72.2, "Observe", "$G_\\lambda>\\lambda\\hat C$", BLUE, BLUE_L)]
    for x, t, f, col, fc in cols:
        ax.add_patch(FancyBboxPatch((x - 2.8, 30.2), 5.6, 8.7, boxstyle="round,pad=0,rounding_size=0.6", fc=fc,
                                    ec="none", zorder=3))
        body(ax, x, 37.6, t, fs=4.9, ha="center", color=col, weight="bold")
        body(ax, x, 35.4, f, fs=4.6, ha="center", color=col)
    body(ax, 66.0, 32.8, "resample\nrepair(E)\nswitch model", fs=4.1, ha="center", color=MUTE)
    body(ax, 72.2, 32.8, "value probe\nbounded exec\non one cand.", fs=4.1, ha="center", color=MUTE)
    body(ax, 60.0, 32.8, "stop when\nnothing pays", fs=4.1, ha="center", color=MUTE)
    # VOI rule
    ax.add_patch(FancyBboxPatch((44.4, 23.6), 24.6, 4.6, boxstyle="round,pad=0,rounding_size=0.6", fc=RED_L,
                                ec="none", zorder=3))
    body(ax, 56.7, 25.9, "value of information  $G_\\lambda(s,e)=\\mathbb{E}_z[U_\\lambda(s\\oplus z)]-U_\\lambda(s)\\;\\geq 0$\n"
                         "worth buying only if it can change the next decision",
         fs=4.4, ha="center", color=RED)
    # loop arrow
    ax.add_patch(FancyArrowPatch((71.0, 24.4), (71.0, 27.4), arrowstyle="-|>", mutation_scale=5, color=RED, lw=0.8,
                                 connectionstyle="arc3,rad=-1.6", zorder=6))
    body(ax, 72.4, 25.9, "act,\nobserve,\nre-plan", fs=4.2, color=RED, ha="left")

    # (5) verified selection
    card(ax, 80.5, 28.0, 9.0, 14.5)
    badge(ax, 81.1, 42.0, 5)
    title(ax, 82.9, 40.6, "Select")
    tag(ax, 87.7, 37.9, "C3", PURPLE)
    for cx, cy, n, col in ((83.4, 34.6, 2, PURPLE), (86.9, 34.0, 1, "#b9a7d6")):
        ax.add_patch(Circle((cx, cy), 1.6, fc=PURPLE_L, ec=col, lw=0.6, zorder=3))
        for t in range(n):
            ax.add_patch(Circle((cx - 0.5 + t * 1.0, cy), 0.42, fc=col, ec="none", zorder=4))
    body(ax, 85.0, 31.0, "result clusters\n+ pairwise $P$", fs=4.6, ha="center")

    # (6) answer
    card(ax, 91.2, 28.0, 7.4, 14.5, fc=GREEN_L, ec=GREEN)
    badge(ax, 91.8, 42.0, 6, color=GREEN)
    title(ax, 93.4, 40.6, "Answer", color=GREEN)
    body(ax, 94.9, 35.0, "SQL", fs=7.5, ha="center", color=GREEN, weight="bold")
    body(ax, 94.9, 31.4, "$\\checkmark$", fs=11, ha="center", color=GREEN)

    for (x0, x1) in ((13.0, 15.0), (26.0, 28.0), (40.5, 42.5), (76.0, 80.5), (89.5, 91.2)):
        arrow(ax, (x0, 35.0), (x1, 35.0), color=INK, lw=0.8, ms=6)

    # sandbox strip
    ax.add_patch(FancyBboxPatch((15.0, 22.9), 25.5, 3.4, boxstyle="round,pad=0,rounding_size=0.6", fc="white",
                                ec=BLUE, lw=0.5, linestyle=(0, (2, 1)), zorder=2))
    body(ax, 27.75, 24.6, "DB sandbox: read-only · one SELECT · timeout · row cap", fs=4.6, ha="center", color=BLUE)

    # ---------------- offline pipeline
    steps = [(2.0, "Evidence-conditioned replay", "every action × every observation;\nREPAIR once per visible mask\n{ $\\emptyset$, VP, EX, VP+EX }", AMBER),
             (21.0, "Partial-state sampling", "candidate sets × random\nevidence masks; outcome\nonly in targets, never features", AMBER),
             (40.0, "Estimators", "", AMBER),
             (59.0, "Calibrate (cross-fitted)", "out-of-fold runs choose\n$(\\lambda,\\tau,m)$ per budget;\ntest never touched", AMBER)]
    for k, (x, t, b, col) in enumerate(steps):
        card(ax, x, 3.2, 17.0, 13.6, fc="white")
        badge(ax, x + 0.6, 16.2, chr(ord("A") + k), color=col, r=1.15, fs=5.6)
        title(ax, x + 2.3, 14.9, t, fs=5.9)
        if b:
            body(ax, x + 1.2, 9.8, b, fs=4.8, color=INK)
        if k < 3:
            arrow(ax, (x + 17.0, 10.0), (x + 19.0, 10.0), color=AMBER, lw=0.8, ms=6)
    # estimator chips
    for j, (t, c) in enumerate((("A", PURPLE), ("P", PURPLE), ("V", GREEN), ("Q", INK), ("G", RED), ("$\\hat C$", BLUE))):
        xx = 42.2 + (j % 3) * 4.9; yy = 10.9 - (j // 3) * 4.0
        ax.add_patch(FancyBboxPatch((xx - 1.8, yy - 1.3), 3.6, 2.6, boxstyle="round,pad=0,rounding_size=0.5", fc=c,
                                    ec="none", zorder=4))
        ax.text(xx, yy - 0.05, t, ha="center", va="center", color="white", fontsize=6.0, weight="bold", zorder=5)
    body(ax, 48.5, 4.5, "gradient-boosted, regularized", fs=4.4, ha="center", color=MUTE)
    # deploy arrow to planner
    arrow(ax, (76.0, 10.0), (78.6, 10.0), color=AMBER, lw=0.8, ms=6)
    ax.add_patch(FancyArrowPatch((67.5, 16.8), (60.0, 22.5), arrowstyle="-|>", mutation_scale=6, color=AMBER, lw=0.8,
                                 linestyle=(0, (3, 1.5)), connectionstyle="arc3,rad=-0.25", zorder=5))
    body(ax, 66.0, 20.4, "frozen policy", fs=4.6, color=AMBER, weight="bold")

    # inset: why price evidence
    card(ax, 78.6, 2.0, 20.4, 15.4, fc="white")
    title(ax, 79.8, 16.0, "Why price evidence?", fs=5.8)
    ia = fig.add_axes([0.815, 0.075, 0.165, 0.215])
    costs = operator_costs()                     # mean ms per operator at 1x / 10x / 100x data
    ex, vp, xp = ([float(np.mean(costs[k][op])) for k in SCALES] for op in ("EXEC_BOUNDED", "VALUE_PROBE", "EXPLAIN"))
    xs = np.arange(3)
    ia.bar(xs - 0.22, ex, 0.4, color=BLUE, label="bounded exec")
    ia.bar(xs + 0.22, vp, 0.4, color="#7fa8de", label="value probe")
    ia.plot(xs, xp, "-o", color=AMBER, ms=2.2, lw=0.9, label="EXPLAIN")
    ia.set_yscale("log"); ia.set_ylim(0.3, 60000)
    ia.set_xticks(xs); ia.set_xticklabels(["1×", "10×", "100×"], fontsize=4.8)
    ia.tick_params(axis="y", labelsize=4.3, length=1.5, pad=1); ia.tick_params(axis="x", length=0, pad=1)
    ia.set_ylabel("mean ms", fontsize=4.6, labelpad=1)
    ia.spines[["top", "right"]].set_visible(False)
    for s_ in ia.spines.values():
        s_.set_linewidth(0.4)
    ia.legend(fontsize=3.9, frameon=False, loc="upper left", ncol=3, handlelength=0.9, borderpad=0.1, columnspacing=0.6, handletextpad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"system_overview.{ext}", dpi=300)
    plt.close(fig)
    print("saved system_overview.pdf")


# ------------------------------------------------------------------ Fig. 2
def examples():
    t650 = json.loads((CROSSFIT_DIR / "trace_650.json").read_text(encoding="utf8"))["lam=0.1,tau=0.2"]
    ex = json.loads((SCALING_DIR / "example.json").read_text(encoding="utf8"))
    fig = plt.figure(figsize=(7.16, 3.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 42.6); ax.axis("off")
    # panel frames
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 59.4, 41.4, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f8fc", ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((61.0, 0.6), 38.4, 41.4, boxstyle="round,pad=0,rounding_size=1.2", fc="#fbf8f2", ec="none", zorder=0))
    title(ax, 2.0, 40.4, "(a) Ask the database first  —  BIRD Q650, original data", fs=6.8, color=BLUE)
    title(ax, 62.4, 40.4, "(b) Know when not to ask  —  Q368, 100× data", fs=6.8, color=AMBER)

    # ---- (a) storyboard
    # card 1: question + c0
    card(ax, 2.0, 22.4, 18.0, 16.2)
    badge(ax, 2.6, 38.0, 1)
    title(ax, 4.4, 36.7, "Cheap model, 3.3 s", fs=5.9)
    body(ax, 3.2, 33.6, "Hint: “Outliers is the\nname of the badge”", fs=4.8, style="italic", color=MUTE)
    ax.add_patch(FancyBboxPatch((3.0, 23.4), 16.0, 7.0, boxstyle="round,pad=0,rounding_size=0.5", fc="#1d2433", ec="none", zorder=3))
    body(ax, 3.6, 28.6, "SELECT LastAccessDate, ...", fs=4.2, color="#e8ecf4", family="monospace")
    body(ax, 3.6, 26.8, "FROM users JOIN badges", fs=4.2, color="#e8ecf4", family="monospace")
    body(ax, 3.6, 25.0, "WHERE Name =", fs=4.2, color="#e8ecf4", family="monospace")
    ax.text(10.6, 25.0, "'Outliers'", fontsize=4.2, color="white", family="monospace", va="center", zorder=9,
            bbox=dict(boxstyle="round,pad=0.12", fc=RED, ec="none"))
    # card 2: probes
    card(ax, 22.0, 22.4, 17.0, 16.2)
    badge(ax, 22.6, 38.0, 2, color=BLUE)
    title(ax, 24.4, 36.7, "Database evidence, 54 ms", fs=5.9, color=BLUE)
    db_icon(ax, 25.0, 31.6, s=0.9)
    rows = [("bounded exec", "0 rows", RED), ("value probe", "not found", RED), ("near value", "'outliers'", GREEN)]
    for k, (a_, b_, c_) in enumerate(rows):
        body(ax, 27.6, 33.6 - 2.6 * k, a_, fs=4.7, color=MUTE)
        body(ax, 38.4, 33.6 - 2.6 * k, b_, fs=4.8, color=c_, ha="right", weight="bold")
    body(ax, 30.5, 24.0, "the SQL text looked fine;\nthe data says otherwise", fs=4.5, ha="center", color=MUTE, style="italic")
    # card 3: decision bars
    card(ax, 41.0, 22.4, 17.4, 16.2)
    badge(ax, 41.6, 38.0, 3, color=RED)
    title(ax, 43.4, 36.7, "Plan: value − $\\lambda\\cdot$cost", fs=5.9, color=RED)
    acts = t650["trace"][0]["acts"]
    names = {"VALUE_FIX": "value fix", "RESAMPLE_1": "resample", "REPAIR_DB": "repair(E)", "SWITCH_PRO": "switch pro"}
    sc = [(names[a["action"]], a["score"], a["cost"]) for a in acts]
    x0 = 51.8
    for k, (n, s, cst) in enumerate(sc):
        yy = 33.6 - 2.5 * k
        body(ax, 42.4, yy, n, fs=4.7, ha="left")
        L = (min(1, s / 0.25) * 4.0) if s >= 0 else (max(-1, s / 0.7) * 4.2)
        ax.add_patch(Rectangle((x0 if L >= 0 else x0 + L, yy - 0.75), abs(L), 1.5, fc=GREEN if s > 0 else "#d98880", ec="none", zorder=4))
        body(ax, x0 + (L + 0.4 if L >= 0 else 0.4), yy, f"{s:+.2f}", fs=4.4, color=GREEN if s > 0 else RED, ha="left")
    ax.plot([x0, x0], [25.4, 35.0], color=INK, lw=0.5, zorder=5)
    body(ax, 49.7, 23.7, f"$V(s_0)={t650['trace'][0]['V']:.3f}$; the DB-only fix\n(cost {sc[0][2]:.2f}) beats every LLM call", fs=4.4, ha="center", color=MUTE)
    # bottom row: outcome
    card(ax, 2.0, 2.0, 56.4, 18.0)
    badge(ax, 2.6, 19.3, 4, color=GREEN)
    title(ax, 4.4, 18.0, "Re-plan and select", fs=5.9, color=GREEN)
    # cluster bubbles
    ax.add_patch(Circle((10.0, 10.2), 4.9, fc=RED_L, ec=RED, lw=0.6, zorder=3))
    body(ax, 10.0, 12.6, "$K_1$: 0 rows", fs=5.0, ha="center", color=RED, weight="bold")
    body(ax, 10.0, 9.6, "$c_0$ (cheap)\nstrong model\nrepeats ‘Outliers’", fs=4.4, ha="center", color=RED)
    ax.add_patch(Circle((24.5, 10.2), 4.9, fc=GREEN_L, ec=GREEN, lw=0.8, zorder=3))
    body(ax, 24.5, 12.6, "$K_2$: 1 row", fs=5.0, ha="center", color=GREEN, weight="bold")
    body(ax, 24.5, 9.4, "value fix\n‘outliers’\n$\\checkmark$ correct", fs=4.4, ha="center", color=GREEN)
    P = t650["P"]["VALUE_FIX>c0"]
    arrow(ax, (15.2, 10.2), (19.3, 10.2), color=GREEN, lw=1.0, ms=6)
    body(ax, 17.2, 12.6, f"$P(K_2\\succ K_1)$\n$={P:.3f}$", fs=4.5, ha="center", color=GREEN)
    # stat tiles
    tiles = [("1", "LLM call"), ("2.50", "cost units"), ("3.4 s", "end-to-end"), (f"{t650['trace'][1]['V']:.2f}", "$V(s_1)$ after fix")]
    for k, (v, lab) in enumerate(tiles):
        xx = 36.0 + (k % 2) * 11.2; yy = 12.9 - (k // 2) * 6.4
        ax.add_patch(FancyBboxPatch((xx - 4.8, yy - 2.6), 9.6, 5.2, boxstyle="round,pad=0,rounding_size=0.6", fc=GRAY_L, ec="none", zorder=3))
        body(ax, xx, yy + 0.8, v, fs=7.2, ha="center", weight="bold", color=INK)
        body(ax, xx, yy - 1.3, lab, fs=4.4, ha="center", color=MUTE)
    body(ax, 47.3, 3.2, "every LLM-only plan repeats the wrong literal", fs=4.4, ha="center", color=MUTE, style="italic")
    for (x0_, x1_) in ((20.0, 22.0), (39.0, 41.0)):
        arrow(ax, (x0_, 30.5), (x1_, 30.5), lw=0.8, ms=6)
    arrow(ax, (49.7, 22.4), (49.7, 20.0), lw=0.8, ms=6)

    # ---- (b) knowing when not to ask
    card(ax, 62.4, 27.6, 35.6, 11.0)
    badge(ax, 63.0, 38.0, 1, color=AMBER)
    title(ax, 64.8, 36.7, "“What is the percentage of borderless cards?”", fs=5.6)
    ax.add_patch(FancyBboxPatch((63.4, 28.6), 33.6, 5.6, boxstyle="round,pad=0,rounding_size=0.5", fc="#1d2433", ec="none", zorder=3))
    body(ax, 64.0, 32.6, "SELECT COUNT(CASE WHEN bordercolor='borderless'", fs=4.2, color="#e8ecf4", family="monospace")
    body(ax, 64.0, 30.4, "  THEN id END)*100.0/COUNT(id) FROM cards", fs=4.2, color="#e8ecf4", family="monospace")
    # price vs VOI
    card(ax, 62.4, 12.2, 17.2, 13.8)
    badge(ax, 63.0, 25.4, 2, color=AMBER)
    title(ax, 64.8, 24.1, "Price it, then decide", fs=5.8)
    opts = {o["op"]: o for o in ex["options"]}
    body(ax, 63.4, 21.6, f"EXPLAIN cost {ex['explain_cost']:.1e}", fs=4.6, color=MUTE)
    lines = [("bounded exec", opts["EXEC_BOUNDED"]), ("value probe", opts["VALUE_PROBE"])]
    for k, (n, o) in enumerate(lines):
        yy = 19.6 - 3.3 * k
        body(ax, 63.4, yy + 0.8, n, fs=4.8, weight="bold")
        body(ax, 63.4, yy - 0.9, f"pred. {o['pred_ms']/1000:.2f} s  ·  VOI {o['voi']:+.3f}", fs=4.5, color=MUTE)
        body(ax, 78.8, yy, "skip", fs=5.0, ha="right", color=AMBER, weight="bold")
    body(ax, 71.0, 13.2, "no outcome could change the decision", fs=4.3, ha="center", color=MUTE, style="italic")
    # timeline
    card(ax, 81.0, 12.2, 17.0, 13.8)
    badge(ax, 81.6, 25.4, 3, color=GREEN)
    title(ax, 83.4, 24.1, "DB time, same answer $\\checkmark$", fs=5.8)
    bars = [("fixed bundle", ex["bundle_db_ms"], INK), ("optim.-gated", ex["gated_db_ms"], BLUE), ("adaptive", ex["adaptive_db_ms"], RED)]
    for k, (n, ms, col) in enumerate(bars):
        yy = 20.6 - 3.2 * k
        body(ax, 82.0, yy + 1.1, n, fs=4.5, color=MUTE)
        L = 14.0 * ms / max(b[1] for b in bars)
        ax.add_patch(Rectangle((82.0, yy - 0.9), max(L, 0.25), 1.3, fc=col, ec="none", zorder=4))
        body(ax, 82.0 + max(L, 0.25) + 0.4 if L < 9 else 82.0 + L - 0.4, yy - 0.25,
             f"{ms/1000:.1f} s" if ms > 50 else f"{ms:.0f} ms", fs=4.5,
             color=col if L < 9 else "white", ha="left" if L < 9 else "right", weight="bold")
    # takeaway
    card(ax, 62.4, 2.0, 35.6, 8.6, fc=AMBER_L, ec="none")
    body(ax, 80.2, 6.3, f"Accept immediately: {(ex['bundle_db_ms']-ex['adaptive_db_ms'])/1000:.1f} s of database time saved.\n"
                        "On 100× data, at 1.25–1.5× budgets, this cuts mean database time by 82–89%\n"
                        "at equal accuracy (498 held-out questions).", fs=4.7, ha="center", color=INK)
    arrow(ax, (80.2, 27.6), (80.2, 26.0), lw=0.8, ms=6)
    arrow(ax, (79.6, 19.0), (81.0, 19.0), lw=0.8, ms=6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"examples.{ext}", dpi=300)
    plt.close(fig)
    print("saved examples.pdf")


if __name__ == "__main__":
    system_overview()
    examples()
