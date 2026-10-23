# PlanGate: the ICDE story (working draft, 2026-10-06)

## Working title

**Ask the Database First: Cost-Based Evidence Planning for LLM Text-to-SQL Serving**

## Thesis (one sentence)

> In LLM Text-to-SQL serving, the DBMS is the cheapest and most informative source of *decision evidence*. Its own optimizer can price that evidence before it is paid for. A planner that buys DB evidence first, and buys LLM compute only when that evidence says it is needed, dominates LLM-only test-time-compute strategies on cost, latency and accuracy.

## How the story changed (and why the new one is stronger)

| | old framing | new framing |
|---|---|---|
| object | route a query to a model / pipeline | plan **evidence acquisition** over LLM and DBMS operators |
| role of the DBMS | final executor + verifier | **sensor** (cheap probes), **pricer** (optimizer estimates), **discriminator** (differential probes), **repairer** (value fix) |
| main competitor | routers (EllieSQL-style) | LLM-only test-time compute: self-consistency, ensembles (PRISM-like budget), LLM verification, cascades |
| why it is not "just a router" | — | decisions happen *after* the candidate exists, and **without DB evidence the same planner loses its advantage** (ablation on both engines) |

## Experimental setting

| | SQLite arm | PostgreSQL arm |
|---|---|---|
| data | BIRD dev, 300 questions (stratified), SQLite | **BIRD mini-dev, 500 questions, official PostgreSQL gold**, PG 16 with the official dump (75 tables, 1 GB) |
| first candidate | deepseek-v4-flash | deepseek-v4-flash (PG prompt) |
| LLM actions | resample (T=0.8), evidence-conditioned repair, blind repair, switch→deepseek-v4-pro, switch→qwen3.8-max | resample, evidence repair, switch→pro, switch→qwen |
| DB actions | parse/bind, EXPLAIN, value micro-probe, bounded execution, VALUE_FIX, differential probe | same, plus real optimizer estimates (cost, rows, operators) |

Protocol details:

- Every (question, action) outcome is real and cached, so all policies replay over identical outcomes.
- Every learned component (acceptor, pairwise selector, V, Q) is cross-fitted by question.
- Operating points are additionally validated with a **split-half protocol**: the point is chosen on one half of the questions and scored on the other.

---

## Findings

### F1: DB probes carry more decision information than an extra LLM call, at 1/200–1/500 of the latency

The table shows ΔAUROC for predicting "is the first candidate correct?" over a P1 baseline (question + schema + SQL), with 95% bootstrap CIs.

| evidence | SQLite ΔAUROC | median latency | PostgreSQL ΔAUROC | median latency |
|---|---|---|---|---|
| all 4 DB probes | **+0.080** [+0.034, +0.131] | 20 ms | **+0.060** [+0.030, +0.090] | 8 ms |
| bounded execution alone | +0.066 [+0.024, +0.112] | 9 ms | +0.058 [+0.031, +0.086] | 2 ms |
| value micro-probe alone | +0.036 [+0.008, +0.067] | 3 ms | +0.016 [−0.001, +0.035] | 2 ms |
| extra LLM sample, compared as SQL text | **−0.004** | 4.4 s | **+0.004** | 4.7 s |
| extra LLM sample, compared by *execution result* | +0.046 | 4.4 s | +0.068 | 4.7 s |

**An extra LLM sample is uninformative unless the DBMS executes it.** Even "ask another LLM" is really a DB signal.

### F2: LLM verification without DB evidence is worse than a coin flip

This is pairwise selection between the first candidate and a disagreeing alternative, on 155 decisive pairs (SQLite). Accuracy is averaged over both presentation orders, with 95% CIs.

| judge | SQL text only | + result previews | + **differential execution probe** (A∖B, B∖A) |
|---|---|---|---|
| deepseek-v4-flash | 0.465 [0.40, 0.53] | 0.523 | **0.577** [0.51, 0.65] |
| deepseek-v4-pro | **0.429** [0.36, 0.50] | 0.577 | 0.590 |

- From SQL text alone, LLM judges systematically prefer the wrong candidate.
- DB evidence lifts both judges by 11–16 points (significant).
- For the cheap judge, the differential probe beats plain previews: +5.5 points, CI [+0.3, +10.6]. **The cheap judge with differential evidence equals the expensive judge with previews.**

### F3: the bottleneck is selection, not generation

- Across the candidate pool, oracle selection reaches 0.763 (SQLite), but an execution-result majority vote reaches only 0.663: a **10-point selection gap**.
- Replacing the first candidate with a new LLM output *hurts* in unflagged states (resample: −8.4 points in uncertain states).
- So actions must be **verified before they replace**. A pairwise, execution-clustered selector beats pointwise acceptance: 0.667 vs 0.653 (SQLite) and 0.640 vs 0.634 (PostgreSQL) at matched settings.

### F4: the best next action depends on the DB-observed state

Rescue rate among wrong first candidates:

| state after probes | SQLite: cheap actions | SQLite: any action | PostgreSQL: best action |
|---|---|---|---|
| **DB-flagged** | **50%** (11/22) | 55% | evidence-conditioned repair **0.38** (vs resample 0.30, qwen 0.27) |
| not flagged | 18% (16/89) | 33% | switch to qwen 0.26 |

- In DB-flagged states, DB-only VALUE_FIX has ~45× higher ΔP(correct)/s than the best LLM action (SQLite).
- In unflagged states, only a *different, stronger model* helps.

**The planner learns exactly this split.** At λ=0.02 on SQLite, the first action in flagged states is REPAIR_DB, RESAMPLE or VALUE_FIX, and 98% of unflagged states are accepted or resampled.

### F5: the optimizer prices DB evidence almost perfectly (PostgreSQL)

- `EXPLAIN` takes 0.6 ms (median).
- Spearman correlation between the estimated cost and the actual execution time is **0.962** (2,446 executed candidates).
- It predicts executions slower than 100 ms with **AUROC 0.924**, and slower than 500 ms with 0.999.
- Execution time is heavy-tailed: the top 5% of candidates account for 34% of all execution time.
- **Price-aware acquisition** (execute only if est. cost ≤ p95) keeps decision quality (AUROC 0.759 vs 0.758 for always-execute) with **−21% DB time and −42% max DB time**. It beats random skipping at equal coverage.

### F6: end-to-end, evidence planning dominates LLM-only test-time compute on both engines

**Matched-budget comparison** against the best *external* baseline at each budget. Budgets are multiples of the cost of the cheapest model. External baselines are: fixed models, the one-shot router, the rule cascade, the MoT-style agreement cascade, self-consistency, and 3–5-candidate execution-vote ensembles.

| budget | SQLite: PlanGate | SQLite: best baseline | Δ [95% CI] | PostgreSQL: PlanGate | PostgreSQL: best baseline | Δ [95% CI] |
|---|---|---|---|---|---|---|
| 1.25× | 0.660 | 0.630 | **+3.0** [+1.0, +5.3] | 0.622 | 0.604 | +1.8 [+0.0, +3.6] |
| 1.5× | 0.670 | 0.647 | **+2.3** [+0.7, +4.3] | 0.628 | 0.612 | **+1.6** [+0.2, +3.2] |
| 2× | 0.670 | 0.647 | **+2.3** [+0.3, +4.3] | 0.648 | 0.618 | **+3.0** [+0.4, +5.6] |
| 3× | 0.673 | 0.670 | +0.3 (ns) | 0.670 | 0.632 | **+3.8** [+1.2, +6.4] |
| 4× | 0.680 | 0.673 | +0.7 (ns) | **0.676** | 0.642 | **+3.4** [+1.0, +5.8] |

**Split-half validation**, with operating points chosen on held-out halves:
- SQLite: **+3.3 [+1.0, +5.7]** at 1.25×, **+3.3 [+1.3, +5.7]** at 1.5× and 2×.
- PostgreSQL: **+2.0 [+0.0, +4.0]** at 1.5×, **+4.2 [+1.6, +6.8]** at 3×.

**Cost needed to match each baseline's accuracy:**

| baseline | SQLite cost | SQLite p95 latency | PostgreSQL cost | PostgreSQL p95 latency |
|---|---|---|---|---|
| strong model alone (qwen) | −14% | +48% ✗ | **−47%** | **−27%** |
| self-consistency@3 / @2 | −66% / −49% | −8% / −4% | — / −49% | — / −23% |
| execution-vote ensemble (3–5 candidates; PRISM-like budget) | **−82% to −86%** | −13% to −17% | **−76% to −78%** | **−43% to −44%** |
| MoT-style agreement cascade | −55% | −23% | −41% | −21% |

**DBMS ablation.** The same planner with no DB access (SQL-text agreement only):

| | SQLite (≤2×) | PostgreSQL 2× | PostgreSQL 4× |
|---|---|---|---|
| no DB | 0.637–0.657 | 0.624 | 0.656 |
| with DB | 0.660–0.670 | 0.648 | 0.676 |

The no-DB planner needs about 2–4× the cost to approach the DB-aware planner, and never reaches it.

Figures: `results/planner_d_frontier.png` (SQLite) and `results/planner_d_frontier_pg.png` (PostgreSQL), each showing accuracy vs cost and accuracy vs p95 latency.

---

## System design (what the paper proposes)

1. **Operator catalog.** Every operator has a measured or estimated cost and latency.
   - LLM: GEN(model, profile), RESAMPLE, REPAIR(evidence), SWITCH(model).
   - DBMS: PARSE_BIND, EXPLAIN, VALUE_PROBE, EXEC_BOUNDED, DIFF_PROBE, and VALUE_FIX (DB-only repair).
2. **Optimizer-priced evidence.** EXPLAIN estimates are the a-priori price of execution-based evidence (F5).
3. **DB-verified selection.** Candidates are clustered by execution fingerprint. A cross-fitted pairwise preference model over cluster representatives uses DB features (F2, F3).
4. **Two-phase evidence planner.**
   - Phase 1: cheap DB-guided local actions chosen by learned Q(s,a) − V(s) − λ·cost(a).
   - Phase 2: escalate to a strong model iff V(s) < τ.
   - The answer is always the DB-verified selection over all candidates.
   - All value models are offline-trained gradient-boosted classifiers; no RL.
5. **Safety.** Read-only sessions, statement timeouts, a single-SELECT guard, and bounded result fetches.

## Positioning (to verify against the papers before writing related work)

- **Cost-aware routers** (e.g. EllieSQL) decide *before* generation. F1/F6 show that pre-generation information is weak here: P0 AUROC is about 0.58, and the one-shot router is dominated on both engines.
- **Database-aware configuration / ensembles** (e.g. PRISM) spend compute on many candidates. PlanGate matches 3–5-candidate execution-vote ensembles at 76–86% lower cost. The comparison should be repeated against the real PRISM pipeline.
- **Execution-guided decoding / execution-feedback repair / LLM selectors** (e.g. CHASE-SQL-style pairwise selection) use the DB *as a verifier inside a fixed pipeline*. PlanGate makes DB operations *first-class, priced plan operators* chosen per state. F2 shows that what makes selection work is the DB evidence, not the judge.
- **LLM cascades** (FrugalGPT, AutoMix, MoT) use LLM self-verification or agreement. F1/F2 show that LLM-only verification signals are weak without execution.

## Honest limitations / open items

- **Scale of evidence.**
  - n=300 (SQLite) and n=500 (PostgreSQL).
  - The gains are 2–4 points at matched budget. They are significant at most budgets but modest in absolute terms.
  - The oracle (0.75–0.76) shows that most headroom is still unrealised.
- **High-budget SQLite latency.** Matching qwen's accuracy costs +48% p95 latency, because the plan is sequential. A latency-aware objective exists but does not yet fix this. Possible fixes: speculative parallel escalation, or using the latency weight.
- **The "confidence cascade" baselines reuse PlanGate's own DB-aware acceptor and selector,** so they are ablations, not external baselines. A faithful AutoMix (LLM self-verification) and the real PRISM still need to be run.
- **Differential probes** help a cheap judge but are not yet integrated as a planner action (JUDGE_DIFF). That is the most promising next action to add, given F3.
- **DB costs on BIRD are small** (ms to sub-second). F5's pricing result matters most on larger data. Next: a scaled PostgreSQL copy (10–100×) and concurrent load, where price-aware acquisition should show large p95 savings.
- **Labels.** About 23–25% of questions are unrescuable by any action. Part of this is likely BIRD label noise; only the official tied-answer fix has been applied (7 labels in the 300-question sample).
- **Single run per configuration;** the API proxy adds roughly 2–3k prompt tokens per call, which compresses cost ratios.

## Next experiments, in priority order

1. **JUDGE_DIFF as a planner action.** A cheap judge with differential evidence, invoked only when clusters disagree. It targets the 10-point selection gap (F3) directly.
2. **Scaled-data PostgreSQL** (synthetic 10×/100× scale of the large BIRD tables) plus concurrent DB load, so that F5 becomes a first-class systems result (p95 and DB-CPU savings).
3. **External baselines run faithfully:** PRISM (or its published BIRD configuration budget), AutoMix with LLM self-verification, and a CHASE-SQL-style selector.
4. **Full BIRD dev (1,534 questions)** on both engines, for the final budget-matched frontier with split-half / held-out operating points.

## Reproduce (all from cache)

```
python experiments/build_action_table.py results/matrix.jsonl     # SQLite action table
python experiments/analyze_actions.py                             # oracle, Exp A/B/C (SQLite)
python experiments/judge_exp.py deepseek-v4-pro                   # Exp J (pairwise selection)
python experiments/judge_stats.py
python experiments/pg_build.py 500                                # PostgreSQL action table (needs PG on :5544)
python experiments/pg_analysis.py                                 # PG: Exp A, optimizer pricing, price-aware acquisition
python experiments/planner_d.py sqlite ; python experiments/planner_d.py pg   # Exp D frontiers + split-half
```

PostgreSQL server:

```
data/pg/pgsql/bin/pg_ctl -D data/pg/cluster -o "-p 5544" start
```
