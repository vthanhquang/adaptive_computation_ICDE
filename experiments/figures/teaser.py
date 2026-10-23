"""Figure 1 (teaser): (a) decide-once pipelines vs. PlanGate's plan-act-observe loop,
(b, c) accuracy vs. cost against published methods on SQLite and PostgreSQL.
Usage: python -m experiments.figures.teaser   (writes results/figures/teaser.{pdf,svg,png})
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plangate.paths import CROSSFIT_DIR, FIGURES_DIR

OUT = FIGURES_DIR
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "axes.linewidth": 0.6})
DB, CON, DARK, MID = "#1f4e9c", "#a3281e", "#222222", "#8a8a8a"

METHODS = {   # label -> report family prefixes (best point over the method's variants)
    "AutoMix": ["[paper] AutoMix"], "CHASE-SQL": ["[paper] CHASE-SQL"], "EllieSQL": ["[paper] EllieSQL ("],
    "SQLens": ["[paper] SQLens"], "DIN-SQL": ["[paper] DIN-SQL"], "MAC-SQL": ["[paper] MAC-SQL"],
    "MoT": ["agreement cascade"], "flash": ["fixed flash"], "qwen": ["fixed qwen"],
}
MARK = {"AutoMix": "s", "CHASE-SQL": "D", "EllieSQL": "^", "SQLens": "v", "DIN-SQL": "P", "MAC-SQL": "X",
        "MoT": "h", "flash": "o", "qwen": "o"}


def pareto(pts):
    pts = sorted(pts, key=lambda p: (p[1], -p[0])); out, best = [], -1
    for p in pts:
        if p[0] > best + 1e-12:
            out.append(p); best = p[0]
    return out


LABEL_OFF = {   # (dx, dy) in points, per panel, to avoid overlapping labels
    "sqlite": {"qwen": (5, -7), "MoT": (6, 5), "AutoMix": (-8, -10), "MAC-SQL": (4, 1), "EllieSQL": (4, 1),
         "SQLens": (4, -6), "CHASE-SQL": (-34, 4), "flash": (4, -6)},
    "pg": {"qwen": (4, -6), "MoT": (5, 5), "AutoMix": (-12, 5), "CHASE-SQL": (-36, 4), "flash": (4, 2)},
}
YLIM = {"sqlite": (0.585, 0.69), "pg": (0.585, 0.69)}


def panel(ax, tag, title):
    rep = json.loads((CROSSFIT_DIR / f"planner_report_{tag}.json").read_text(encoding="utf8"))
    C = rep["curves"]
    pg = []
    for k in ("PlanGate-full", "PlanGate-full 2-phase", "PlanGate-full (latency-aware)"):
        pg += [tuple(s) for s, _ in C.get(k, [])]
    fr = pareto(pg)
    ax.plot([p[1] for p in fr], [p[0] for p in fr], "-", color=CON, lw=1.6, drawstyle="steps-post", zorder=5,
            label="PlanGate")
    ax.scatter([fr[-1][1]], [fr[-1][0]], color=CON, s=14, zorder=6)
    for lab, prefs in METHODS.items():
        cand = [tuple(s) for k, v in C.items() if any(k.startswith(p) for p in prefs) for s, _ in v]
        if not cand:
            continue
        b = max(cand, key=lambda s: (s[0], -s[1]))
        col = MID if lab in ("flash", "qwen", "MoT") else DARK
        lo = YLIM[tag][0]
        if b[0] < lo:   # below the axis range: draw at the bottom edge with a downward arrow
            ax.annotate("", xy=(b[1], lo), xytext=(b[1], lo + 0.008), zorder=7,
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col, mutation_scale=6))
            ax.annotate(f"{lab} ({b[0]:.2f})", (b[1], lo + 0.008), xytext=(-46, -1), textcoords="offset points",
                        fontsize=5.6, color=col)
            continue
        ax.scatter(b[1], b[0], marker=MARK[lab], s=22, facecolor="white", edgecolor=col, lw=0.9, zorder=7)
        ax.annotate(lab, (b[1], b[0]), xytext=LABEL_OFF[tag].get(lab, (3, 2)), textcoords="offset points",
                    fontsize=5.8, color=col, zorder=8)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("mean cost per query", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(0, 28.5); ax.set_ylim(*YLIM[tag])


fig = plt.figure(figsize=(7.16, 2.15))
# (a) schematic
axa = fig.add_axes([0.0, 0.0, 0.40, 1.0]); axa.set_xlim(0, 40); axa.set_ylim(0, 22); axa.axis("off")


def b(x, y, w, h, t, ec=DARK, fc="white", fs=6.2, lw=0.8, tc=DARK):
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.5", ec=ec, fc=fc, lw=lw))
    axa.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc)


def a(p, q, c=DARK, ls="-", both=False):
    axa.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=0.7, color=c,
                                                     linestyle=ls, mutation_scale=6, shrinkA=0, shrinkB=0))


axa.text(0.6, 21.2, "(a)", fontsize=7, weight="bold", va="top")
axa.text(3.0, 21.2, "Decide once (routers, cascades, fixed pipelines)", fontsize=6.4, va="top", style="italic")
b(1.0, 14.6, 4.4, 3.4, "$q$")
b(7.4, 14.6, 8.2, 3.4, "choose model /\npipeline", fs=5.8)
b(17.6, 14.6, 8.4, 3.4, "LLM call(s)", fs=5.8)
b(28.0, 14.6, 5.6, 3.4, "SQL", fs=5.8)
b(35.0, 14.6, 4.4, 3.4, "DB", ec=DB, tc=DB, fs=5.8)
for x1, x2 in ((5.4, 7.4), (15.6, 17.6), (26.0, 28.0), (33.6, 35.0)):
    a((x1, 16.3), (x2, 16.3), c=MID)
axa.text(20.0, 12.9, "DB used only to execute the final answer", fontsize=5.6, color=MID, ha="center", style="italic")

axa.text(3.0, 10.6, "PlanGate: plan $\\rightarrow$ act $\\rightarrow$ observe, over LLM and DB operators",
         fontsize=6.4, va="top", style="italic")
b(1.0, 2.6, 4.4, 3.6, "$q$")
b(7.4, 1.6, 9.2, 5.6, "adaptive\nplanner", fs=6.0, lw=1.1, ec=CON, tc=CON)
b(19.6, 5.0, 8.4, 2.8, "LLM ops", fs=5.8)
b(19.6, 0.8, 8.4, 2.8, "DB evidence\n(priced by $\\hat{C}$)", fs=5.2, ec=DB, tc=DB, lw=1.1)
b(30.6, 2.6, 8.8, 3.6, "verified\nanswer", fs=5.8)
a((5.4, 4.4), (7.4, 4.4))
a((16.6, 6.4), (19.6, 6.4), both=True)
a((16.6, 2.2), (19.6, 2.2), both=True)
a((16.6, 4.4), (30.6, 4.4))
axa.text(24.0, 4.45, "accept", fontsize=5.4, ha="center", va="center", style="italic",
         bbox=dict(fc="white", ec="none", pad=0.2))

axb = fig.add_axes([0.455, 0.19, 0.25, 0.70]); panel(axb, "sqlite", "(b) BIRD dev, SQLite (n=300)")
axb.set_ylabel("execution accuracy", fontsize=6.5, labelpad=1)
axc = fig.add_axes([0.745, 0.19, 0.25, 0.70]); panel(axc, "pg", "(c) BIRD mini-dev, PostgreSQL (n=500)")
axb.legend(loc="lower right", fontsize=5.8, frameon=False, handlelength=1.6, bbox_to_anchor=(1.0, 0.16))

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"teaser.{ext}", dpi=300 if ext == "png" else None)
print("saved", OUT / "teaser.pdf")
