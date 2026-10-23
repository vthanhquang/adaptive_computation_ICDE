"""PlanGate architecture figure (double-column, optimizer framing).

Usage: python -m experiments.figures.architecture   (writes results/figures/architecture.{pdf,svg,png})
C1-C3 badges mark the paper's contributions:
  C1 priced DB evidence operators, C2 adaptive re-planning loop, C3 verified selection.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plangate.paths import FIGURES_DIR

OUT = FIGURES_DIR
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})
DB = "#1f4e9c"          # DBMS-side components
CON = "#a3281e"         # contribution badges
DARK, MID = "#222222", "#8a8a8a"
FS = 6.4

fig = plt.figure(figsize=(7.16, 3.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")


def box(x, y, w, h, title=None, lines=(), ec=DARK, lw=0.8, tc=DARK, ls="-", fc="white", fs=FS, center=False, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7", ec=ec, fc=fc, lw=lw,
                                linestyle=ls, zorder=z))
    top = y + h - 1.2
    if title:
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs + 0.6, weight="bold", color=tc, zorder=z + 1)
        top -= 2.7
    for i, ln in enumerate(lines):
        ax.text(x + w / 2 if center else x + 0.8, top - i * 2.15, ln, ha="center" if center else "left", va="top",
                fontsize=fs, color=DARK, zorder=z + 1)


def arrow(p, q, label=None, off=(0, 0), ls="-", color=DARK, fs=5.9, rad=0.0, both=False, italic=True):
    ax.annotate("", xy=q, xytext=p, zorder=6,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=0.8, color=color, linestyle=ls,
                                mutation_scale=7, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", va="center",
                fontsize=fs, style="italic" if italic else "normal", color=color, zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.3))


def path(points, label=None, at=0, off=(0, 0), ls="-", color=DARK, fs=5.9):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=0.8, linestyle=ls, zorder=2)
    arrow(points[-2], points[-1], ls=ls, color=color)
    if label:
        (x1, y1), (x2, y2) = points[at], points[at + 1]
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], label, ha="center", va="center", fontsize=fs,
                style="italic", color=color, zorder=7, bbox=dict(fc="white", ec="none", pad=0.5))


def badge(x, y, text):
    ax.add_patch(FancyBboxPatch((x - 1.6, y - 0.95), 3.2, 1.9, boxstyle="round,pad=0.05,rounding_size=0.5",
                                fc=CON, ec="white", lw=0.5, zorder=9))
    ax.text(x, y - 0.05, text, ha="center", va="center", fontsize=5.8, color="white", weight="bold", zorder=10)


# ------------------------------------------------------------------ bands
ax.add_patch(FancyBboxPatch((0.4, 20.8), 99.2, 42.8, boxstyle="round,pad=0,rounding_size=1.2", fc="#f1f1f1", ec="none", zorder=0))
ax.add_patch(FancyBboxPatch((0.4, 0.4), 99.2, 18.4, boxstyle="round,pad=0,rounding_size=1.2", fc="#fbfbfb", ec=MID,
                            lw=0.6, linestyle=(0, (3, 2)), zorder=0))
ax.text(1.8, 42.2, "ONLINE  (per query)", rotation=90, ha="center", va="center", fontsize=7, weight="bold")
ax.text(1.8, 9.6, "OFFLINE", rotation=90, ha="center", va="center", fontsize=7, weight="bold")

# ------------------------------------------------------------------ online: request, planner loop, selection
box(3.4, 46.0, 11.4, 12.0, "NL query", ["question $q$", "schema $S$, hint", "budget $\\lambda$ / SLO"])

ax.add_patch(FancyBboxPatch((18.0, 42.6), 48.4, 20.0, boxstyle="round,pad=0.15,rounding_size=0.9", ec=DARK, fc="#fcfcfc",
                            lw=1.1, zorder=2))
ax.text(21.2, 61.6, "Adaptive planner", ha="left", va="top", fontsize=FS + 0.9, weight="bold")
ax.text(66.0, 61.6, "re-optimizes after every observation", ha="right", va="top", fontsize=FS - 0.2, style="italic")
badge(19.2, 61.0, "C2")
box(20.0, 44.4, 12.6, 11.4, "PLAN", ["choose $a_t$ by", "value $-\\ \\lambda\\cdot$cost"], center=True)
box(36.4, 44.4, 12.0, 11.4, "ACT", ["run exactly", "one operator"], center=True)
box(52.2, 44.4, 12.6, 11.4, "OBSERVE", ["add $y_t$ / $o_t$", "to state $s_t$"], center=True)
arrow((32.6, 50.2), (36.4, 50.2))
arrow((48.4, 50.2), (52.2, 50.2))
path([(58.5, 55.8), (58.5, 58.0), (26.3, 58.0), (26.3, 55.8)], "re-plan until accept", at=1, off=(0, 0))
arrow((14.8, 52.0), (20.0, 52.0), "$s_0$", off=(0, 1.1))

box(70.2, 45.0, 13.6, 13.0, "Verified selection", ["cluster candidates", "by result fingerprint;", "replace only when", "evidence supports it"])
badge(83.0, 58.2, "C3")
box(87.0, 47.0, 11.4, 9.0, "Answer", ["SQL + result"])
arrow((66.4, 51.4), (70.2, 51.4), "accept", off=(0, 1.1))
arrow((83.8, 51.4), (87.0, 51.4))

# ------------------------------------------------------------------ online: operator catalog with cost model
ax.add_patch(FancyBboxPatch((18.0, 25.6), 48.4, 14.8, boxstyle="round,pad=0.15,rounding_size=0.9", ec=DARK, fc="white",
                            lw=0.8, zorder=2))
ax.text(19.0, 39.6, "Operator catalog + cost model", ha="left", va="top", fontsize=FS + 0.6, weight="bold")
box(19.2, 26.4, 21.6, 10.8, "LLM operators", ["GEN, RESAMPLE,", "REPAIR(evidence), SWITCH", "cost: tokens, latency"], fs=5.9)
box(43.0, 26.4, 22.4, 10.8, "DB evidence operators", ["PARSE/BIND, EXPLAIN, PROBE,", "VALUE FIX, EXEC, DIFF PROBE",
    "cost: optimizer price $\\hat{C}$"], ec=DB, tc=DB, lw=1.2, fs=5.9)
badge(64.8, 37.4, "C1")
arrow((26.3, 40.4), (26.3, 44.4), "$C(a), \\hat{C}(e)$", off=(3.6, 0))
arrow((42.4, 44.4), (42.4, 40.4), "invoke $a_t$", off=(3.6, 0))
arrow((58.5, 40.4), (58.5, 44.4), "$y_t$, $o_t$", off=(3.0, 0))

# execution substrate
box(19.2, 21.3, 21.6, 3.2, None, ["LLM serving endpoints"], center=True, fs=5.9)
box(43.0, 21.3, 22.4, 3.2, None, ["DBMS + sandbox (read-only, timeout)"], ec=DB, lw=1.2, center=True, fs=5.9)
ax.text(22.0, 21.9, "", fontsize=1)
arrow((30.0, 26.4), (30.0, 24.5), both=True)
arrow((54.2, 26.4), (54.2, 24.5), both=True)

box(70.2, 27.0, 13.6, 9.0, "Telemetry", ["$(s_t, a_t, o_t)$, cost,", "latency"], ls=(0, (3, 2)), ec=MID)
arrow((64.8, 47.0), (72.0, 36.0), ls=(0, (3, 2)), color=MID)

# legend
box(85.8, 21.6, 13.4, 22.6, "Contributions", [], fs=5.8, z=1)
for i, (b, t1, t2) in enumerate((("C1", "priced DB", "evidence ops"), ("C2", "adaptive", "re-planning"),
                                 ("C3", "verified", "selection"))):
    y = 39.4 - i * 5.0
    badge(88.2, y, b)
    ax.text(90.2, y + 0.9, t1, ha="left", va="center", fontsize=5.8); ax.text(90.2, y - 1.0, t2, ha="left", va="center", fontsize=5.8)
ax.plot([86.6, 88.8], [24.6, 24.6], color=MID, lw=0.8, ls=(0, (3, 2)), zorder=10); ax.text(89.2, 24.6, "periodic", va="center", fontsize=5.6)
ax.add_patch(FancyBboxPatch((93.6, 24.0), 1.6, 1.2, boxstyle="round,pad=0.1", ec=DB, fc="white", lw=1.1, zorder=10))
ax.text(95.6, 24.6, "DBMS", va="center", fontsize=5.6)

# ------------------------------------------------------------------ offline (compact)
box(3.4, 3.0, 11.4, 12.0, "Workload", ["training queries", "+ gold SQL"])
box(18.0, 3.0, 15.4, 12.0, "Counterfactual", ["replay: run every", "action, cache", "outcomes + evidence"])
box(36.4, 3.0, 15.4, 12.0, "Value models", ["$A$, $P(a \\succ b)$,", "$V(s)$, $Q(s,a)$;", "cross-fitted"])
box(54.8, 3.0, 13.0, 12.0, "Calibration", ["$(\\lambda, \\tau)$ per", "budget / SLO"])
box(72.0, 3.0, 11.8, 12.0, "Policy store", ["$\\pi_v$ + cost", "model $C$"])
box(87.0, 3.0, 11.4, 12.0, "Cost profiling", ["DB time vs. $\\hat{C}$;", "token prices"], ec=DB, tc=DB, lw=1.2, fs=6.0)
for x1, x2 in ((14.8, 18.0), (33.4, 36.4), (51.8, 54.8), (67.8, 72.0)):
    arrow((x1, 9.0), (x2, 9.0))
arrow((87.0, 9.0), (83.8, 9.0))

# online <-> offline
path([(76.0, 15.0), (76.0, 17.4), (68.0, 17.4), (68.0, 44.0), (66.4, 44.0)], "deploy $\\pi_v$", at=0, off=(-4.0, -0.6))
arrow((68.0, 33.0), (66.4, 33.0))
path([(81.0, 27.0), (81.0, 19.7), (9.1, 19.7), (9.1, 15.0)], "periodic retraining on logged queries / shadow runs",
     at=1, off=(-10, 0), ls=(0, (3, 2)), color=MID)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"architecture.{ext}", dpi=300 if ext == "png" else None)
print("saved", OUT / "architecture.pdf")
