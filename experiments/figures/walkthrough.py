"""Walk-through figure: how one question (BIRD dev QID 650) is processed by each PlanGate component.
All values are measured: results/crossfit/action_table_sqlite.jsonl and results/crossfit/trace_650.json
(python -m experiments.crossfit.trace_detail 650).
Usage: python -m experiments.figures.walkthrough   (writes results/figures/walkthrough.{pdf,svg,png})
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from plangate.paths import FIGURES_DIR

OUT = FIGURES_DIR
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})
DB, CON, DARK, MID, OK, BAD = "#1f4e9c", "#a3281e", "#222222", "#8a8a8a", "#1e7b34", "#a3281e"
MONO = {"family": "DejaVu Sans Mono", "size": 4.9}
CHECK, CROSS = chr(0x2713), chr(0x2717)
FS = 5.9

fig = plt.figure(figsize=(7.16, 4.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")


def panel(x, y, w, h, n, title, ec=DARK, tc=DARK, lw=0.8, badge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7", ec=ec, fc="white", lw=lw,
                                zorder=2))
    ax.add_patch(Circle((x + 1.3, y + h - 1.3), 1.05, fc=DARK, ec="none", zorder=4))
    ax.text(x + 1.3, y + h - 1.33, str(n), ha="center", va="center", fontsize=5.6, color="white", weight="bold", zorder=5)
    ax.text(x + 2.9, y + h - 1.3, title, ha="left", va="center", fontsize=6.4, weight="bold", color=tc, zorder=4)
    if badge:
        ax.add_patch(FancyBboxPatch((x + w - 3.5, y + h - 2.1), 3.0, 1.6, boxstyle="round,pad=0.05,rounding_size=0.4",
                                    fc=CON, ec="none", zorder=4))
        ax.text(x + w - 2.0, y + h - 1.33, badge, ha="center", va="center", fontsize=5.4, color="white",
                weight="bold", zorder=5)


def t(x, y, s, **kw):
    kw.setdefault("fontsize", FS)
    ax.text(x, y, s, va="top", zorder=4, **kw)


def mono(x, y, s, **kw):
    ax.text(x, y, s, va="top", fontdict={**MONO, **kw}, zorder=4)


def mark(x, y, ok):
    ax.text(x, y, CHECK if ok else CROSS, va="top", color=OK if ok else BAD, zorder=4,
            fontdict={"family": "DejaVu Sans", "size": 6.2, "weight": "bold"})


def arrow(p, q, label=None, color=DARK, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=6, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, linestyle=ls,
                                                              mutation_scale=7, shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.9, label, ha="center", va="bottom", fontsize=5.2,
                style="italic", color=color, zorder=7)


# ------------------------------------------------------------------ row 1
Y1, H1 = 31.0, 28.0
panel(0.8, Y1, 22.6, H1, 1, "Input (NL query)")
t(2.0, 55.6, "Q: Describe the last accessed date and")
t(2.0, 53.8, "location of the users who received the")
t(2.0, 52.0, "outliers badge.")
t(2.0, 49.4, "Hint: Outliers is the name of the badge.", style="italic")
t(2.0, 46.6, "Schema (excerpt):")
mono(2.0, 44.6, "badges(Id, UserId, Name, Date)")
mono(2.0, 42.8, "users(Id, LastAccessDate,")
mono(2.0, 41.0, "      Location, Reputation, ...)")
t(2.0, 38.2, "Budget: $\\lambda = 0.1$, $\\tau = 0.2$")
t(2.0, 36.2, "(high budget: $\\lambda = 0.05$, $\\tau = 0.5$)", color=MID)
t(2.0, 33.6, "State $s_\\varnothing$: no candidate yet", color=MID)

panel(25.6, Y1, 22.6, H1, 2, "GEN: cheap LLM")
t(26.8, 55.6, "deepseek-v4-flash, schema + hint")
mono(26.8, 53.2, "SELECT users.LastAccessDate,")
mono(26.8, 51.4, "       users.Location")
mono(26.8, 49.6, "FROM users JOIN badges")
mono(26.8, 47.8, "  ON badges.UserId = users.Id")
mono(26.8, 46.0, "WHERE badges.Name = 'Outliers'")
t(26.8, 43.0, "Candidate $c_0$ (follows the hint's casing)")
t(26.8, 40.2, "cost 2.43, latency 3.3 s")
t(26.8, 37.6, "Every step from here on runs exactly", color=MID)
t(26.8, 35.8, "one operator and returns to the planner.", color=MID)

panel(50.4, Y1, 22.6, H1, 3, "DB evidence on $c_0$", ec=DB, tc=DB, lw=1.1, badge="C1")
rows = [("PARSE/BIND", "valid", True, "2.1 ms"), ("EXPLAIN", "scan+index", True, "2.5 ms"),
        ("BOUNDED EXEC", "0 rows", False, "13.2 ms")]
for k, (op, res, ok, ms) in enumerate(rows):
    y = 55.6 - k * 2.4
    t(51.6, y, op); t(61.4, y, res); mark(66.9, y + 0.15, ok); t(72.6, y, ms, ha="right", color=MID)
t(51.6, 48.2, "VALUE PROBE")
mono(51.6, 46.2, "'Outliers' not in badges.Name")
t(51.6, 44.2, "nearest stored value:"); mono(62.0, 44.2, "'outliers'")
t(71.8, 48.2, "36.1 ms", ha="right", color=MID)
t(51.6, 41.2, "State $s_0$ = {$c_0$ + evidence}; flagged")
t(51.6, 39.4, "(empty result, missing literal)")
t(51.6, 36.4, "Total 54 ms, read-only, no LLM call.", color=MID)
t(51.6, 34.6, "PostgreSQL: EXEC runs only if $\\hat{C} \\leq \\theta$", color=MID)

panel(75.2, Y1, 24.0, H1, 4, "PLAN at $s_0$", ec=CON, tc=CON, lw=1.1, badge="C2")
t(76.4, 55.6, "$V(s_0) = 0.033$  (answer now: 3% correct)")
t(76.4, 53.2, "score $= Q(s_0,a) - V(s_0) - \\lambda C(a)$")
hdr = ("action", "$Q$", "$C$", "score")
xs = (76.4, 87.0, 91.2, 98.6)
for x, h in zip(xs, hdr):
    t(x, 50.6, h, weight="bold", ha="right" if h == "score" else "left")
ax.plot([76.4, 98.8], [48.8, 48.8], color=DARK, lw=0.4, zorder=4)
cand = [("VALUE FIX", "0.199", "0.01", "+0.164", True), ("RESAMPLE", "0.204", "2.48", "$-$0.078", False),
        ("REPAIR(ev.)", "0.233", "3.50", "$-$0.151", False), ("SWITCH pro", "0.199", "7.93", "$-$0.628", False)]
for k, (a, q, c, s, best) in enumerate(cand):
    y = 48.2 - k * 2.3
    kw = {"weight": "bold", "color": CON} if best else {}
    t(76.4, y, a, **kw); t(87.0, y, q, **kw); t(91.2, y, c, **kw); t(98.6, y, s, ha="right", **kw)
t(76.4, 38.0, "$\\rightarrow$ choose VALUE FIX: highest score,")
t(76.4, 36.2, "  costs only DB time (no LLM call)")
t(76.4, 33.6, "(stronger models are not even needed)", color=MID)

# ------------------------------------------------------------------ row 2
Y2, H2 = 1.2, 26.4
panel(0.8, Y2, 22.6, H2, 5, "ACT: VALUE FIX", ec=DB, tc=DB, lw=1.1, badge="C1")
t(2.0, 25.0, "DB-only rewrite of the missing literal:")
mono(2.0, 22.8, "'Outliers'  $\\rightarrow$  'outliers'")
mono(2.0, 20.4, "WHERE badges.Name = 'outliers'")
t(2.0, 17.6, "re-run the cheap probes on the new SQL:")
t(2.0, 15.4, "BOUNDED EXEC: 1 row"); mark(17.4, 15.55, True)
mono(2.0, 13.2, "('2014-09-12 17:06',")
mono(2.0, 11.4, " 'Leuven, Belgium')")
t(2.0, 8.6, "Candidate $c_1$; +10.8 ms, no LLM call")
t(2.0, 5.8, "Cost so far: 2.50 (1 LLM call)", weight="bold")

panel(25.6, Y2, 22.6, H2, 6, "OBSERVE + re-PLAN at $s_1$", ec=CON, tc=CON, lw=1.1, badge="C2")
t(26.8, 25.0, "$s_1$ = {$c_0$, $c_1$}:  $V(s_1) = 0.305$")
t(26.8, 22.6, "Phase 1: every remaining cheap action")
t(26.8, 20.8, "has score $<$ 0 (best: RESAMPLE $-$0.207)")
t(26.8, 19.0, "$\\rightarrow$ stop local actions")
t(26.8, 16.2, "Phase 2: escalate iff $V(s_1) < \\tau$")
t(26.8, 13.8, "low budget ($\\tau$=0.2): 0.305 $\\geq$ 0.2", weight="bold")
t(28.0, 12.0, "$\\rightarrow$ ACCEPT, go to selection", weight="bold")
t(26.8, 9.4, "high budget ($\\tau$=0.5): 0.305 $<$ 0.5", color=MID)
t(28.0, 7.6, "$\\rightarrow$ SWITCH qwen3.8-max (+8.08)", color=MID)
mono(28.0, 5.6, "WHERE T1.Name = 'Outliers'", color=MID)
t(28.0, 3.6, "$c_2$: 0 rows (same result as $c_0$)", color=MID)

panel(50.4, Y2, 22.6, H2, 7, "Verified selection", badge="C3")
t(51.6, 25.0, "Cluster candidates by result fingerprint:")
t(51.6, 22.4, "$K_1$ = {$c_0$, $c_2$}: 0 rows")
t(53.0, 20.6, "acceptor $A$ = 0.012", color=MID)
t(51.6, 18.2, "$K_2$ = {$c_1$}: 1 row")
t(53.0, 16.4, "acceptor $A$ = 0.287", color=MID)
t(51.6, 13.6, "Pairwise preference between clusters:")
t(51.6, 11.4, "$P(K_2 \\succ K_1) = 0.994$", weight="bold")
t(51.6, 8.6, "$\\rightarrow$ keep $c_1$; the stronger model's")
t(51.6, 6.8, "  answer does not replace it, because")
t(51.6, 5.0, "  the evidence does not support it.")

panel(75.2, Y2, 24.0, H2, 8, "Answer")
mono(76.4, 25.0, "SELECT users.LastAccessDate,")
mono(76.4, 23.2, "       users.Location")
mono(76.4, 21.4, "FROM users JOIN badges")
mono(76.4, 19.6, "  ON badges.UserId = users.Id")
mono(76.4, 17.8, "WHERE badges.Name = 'outliers'")
t(76.4, 14.8, "correct"); mark(81.6, 14.95, True)
t(76.4, 12.0, "low budget: 1 LLM call, cost 2.50, 3.4 s", weight="bold")
t(76.4, 9.8, "high budget: 2 LLM calls, cost 10.64", color=MID)
t(76.4, 7.0, "LLM-only alternatives (resample, repair", color=MID)
t(76.4, 5.2, "w/o evidence, pro, qwen) all return", color=MID)
t(76.4, 3.4, "0 rows with 'Outliers'.", color=MID)

# ------------------------------------------------------------------ flow arrows
arrow((23.4, 45.0), (25.6, 45.0), "$q$")
arrow((48.2, 45.0), (50.4, 45.0), "$c_0$")
arrow((73.0, 45.0), (75.2, 45.0), "$s_0$")
ax.plot([87.2, 87.2, 12.1], [31.0, 29.3, 29.3], color=DARK, lw=0.8, zorder=5)
arrow((12.1, 29.3), (12.1, 27.6))
ax.text(50.0, 29.75, "invoke $a_1$ = VALUE FIX", ha="center", va="bottom", fontsize=5.2, style="italic")
arrow((23.4, 14.0), (25.6, 14.0), "$o_1$")
arrow((48.2, 12.4), (50.4, 12.4))
arrow((73.0, 12.4), (75.2, 12.4))

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"walkthrough.{ext}", dpi=300 if ext == "png" else None)
print("saved", OUT / "walkthrough.pdf")
