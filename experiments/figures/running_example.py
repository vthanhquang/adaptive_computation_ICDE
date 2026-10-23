"""Running example (BIRD dev QID 650): how PlanGate fixes a value-grounding error with a DB-only action.
All numbers are the measured values from results/crossfit/action_table_sqlite.jsonl and
experiments/crossfit/trace_plan.py.
Usage: python -m experiments.figures.running_example   (writes results/figures/running_example.{pdf,svg,png})
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plangate.paths import FIGURES_DIR

OUT = FIGURES_DIR
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})
DB, CON, DARK, MID, OK, BAD = "#1f4e9c", "#a3281e", "#222222", "#8a8a8a", "#1e7b34", "#a3281e"
MONO = {"family": "DejaVu Sans Mono", "size": 5.6}
CHECK, CROSS = chr(0x2713), chr(0x2717)
MARK = {"family": "DejaVu Sans", "size": 7.0, "weight": "bold"}

fig = plt.figure(figsize=(3.5, 4.05))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 50); ax.set_ylim(0, 58); ax.axis("off")


def box(x, y, w, h, ec=DARK, lw=0.8, fc="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.6", ec=ec, fc=fc, lw=lw,
                                linestyle=ls))


def tag(x, y, text, color=DARK):
    ax.text(x, y, text, fontsize=6.6, weight="bold", color=color, va="top")


def arrow(y1, y2, x=4.0):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK, mutation_scale=6, shrinkA=0, shrinkB=0))


def mark(x, y, ok, size=7.0):
    ax.text(x, y, CHECK if ok else CROSS, fontdict={**MARK, "size": size}, color=OK if ok else BAD, va="top")


# question
box(0.6, 50.6, 48.8, 6.8, fc="#f4f4f4", ec="none")
ax.text(1.4, 56.6, "Q: Describe the last accessed date and location of the users", fontsize=6.4, va="top")
ax.text(1.4, 54.5, "     who received the outliers badge.", fontsize=6.4, va="top")
ax.text(1.4, 52.4, "Hint: Outliers is the name of the badge.", fontsize=6.4, va="top", style="italic")

# step 1: cheap generation
box(1.0, 41.2, 48.0, 8.4)
tag(1.8, 49.0, "1  GEN (cheap LLM)")
ax.text(48.2, 49.0, "3.3 s, cost 2.43", fontsize=6.0, ha="right", va="top", color=MID)
ax.text(1.8, 46.8, "SELECT users.LastAccessDate, users.Location", fontdict=MONO, va="top")
ax.text(1.8, 44.8, "FROM users JOIN badges ON badges.UserId = users.Id", fontdict=MONO, va="top")
ax.text(1.8, 42.8, "WHERE badges.Name = 'Outliers'", fontdict=MONO, va="top")
arrow(41.2, 40.6)

# step 2: DB evidence
box(1.0, 31.8, 48.0, 8.6, ec=DB, lw=1.1)
tag(1.8, 39.8, "2  DB evidence (priced, read-only)", DB)
ax.text(48.2, 39.8, "54 ms, no LLM", fontsize=6.0, ha="right", va="top", color=MID)
ax.text(1.8, 37.4, "PARSE/BIND", fontsize=6.2, va="top"); mark(10.6, 37.6, True)
ax.text(15.6, 37.4, "EXEC: 0 rows", fontsize=6.2, va="top"); mark(24.0, 37.6, False)
ax.text(1.8, 35.0, "VALUE PROBE:", fontsize=6.2, va="top")
ax.text(11.8, 35.1, "'Outliers'", fontdict=MONO, va="top")
ax.text(20.0, 35.0, "does not occur in", fontsize=6.2, va="top")
ax.text(30.6, 35.1, "badges.Name", fontdict=MONO, va="top")
ax.text(1.8, 32.8, "nearest stored value:", fontsize=6.2, va="top")
ax.text(15.4, 32.9, "'outliers'", fontdict=MONO, va="top")
arrow(31.8, 30.0)

# step 3: plan
box(1.0, 25.0, 48.0, 5.0, ec=CON, lw=1.1)
tag(1.8, 29.4, "3  PLAN", CON)
ax.text(10.0, 29.4, "flagged state: the cheapest action with value is", fontsize=6.2, va="top")
ax.text(10.0, 27.3, "VALUE FIX (DB-only rewrite, no LLM call)", fontsize=6.2, va="top", weight="bold")
arrow(25.0, 22.6)

# step 4: verified answer
box(1.0, 15.6, 48.0, 7.0)
tag(1.8, 22.0, "4  EXEC + ACCEPT")
ax.text(48.2, 22.0, "+11 ms", fontsize=6.0, ha="right", va="top", color=MID)
ax.text(1.8, 19.6, "WHERE badges.Name = 'outliers'", fontdict=MONO, va="top")
ax.text(1.8, 17.5, "1 row: ('2014-09-12 17:06', 'Leuven, Belgium')", fontdict=MONO, va="top")
mark(45.2, 19.8, True, size=9.0)

ax.text(1.0, 14.2, "PlanGate total: 1 LLM call, cost 2.50, 3.4 s, correct.", fontsize=6.4, weight="bold", va="top")

# LLM-only alternatives
box(1.0, 0.8, 48.0, 11.4, ec=MID, ls=(0, (3, 2)))
ax.text(1.8, 11.6, "LLM-only alternatives for the same query (all keep 'Outliers'):", fontsize=6.2, va="top",
        style="italic")
alts = [("resample (cheap LLM)", "+2.43"), ("repair without DB evidence", "+2.47"),
        ("switch to deepseek-v4-pro", "+7.87"), ("switch to qwen3.8-max", "+8.08")]
for i, (name, c) in enumerate(alts):
    y = 9.2 - i * 2.05
    mark(3.0, y, False)
    ax.text(6.0, y - 0.1, name, fontsize=6.1, va="top")
    ax.text(40.0, y - 0.1, f"cost {c}", fontsize=6.1, va="top", color=MID)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"running_example.{ext}", dpi=300 if ext == "png" else None)
print("saved", OUT / "running_example.pdf")
