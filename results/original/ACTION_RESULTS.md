# PlanGate — action-level analysis (Experiments A, B, C + oracle action)

Date: 2026-10-06. The same 300 BIRD dev questions as `RESULTS.md`. Raw output is in `results/action_analysis.txt`, and the per-query table is in `results/action_table.jsonl`.

## Setup

- **First candidate c0** is `flash_direct` (deepseek-v4-flash). It is correct on 63.0% of questions.
- **The pool is limited to 3 models**, so that ensemble size cannot explain the results: flash, `pro_direct` and `qwen_max_direct`.

| action | type | what it does |
|---|---|---|
| PARSE_BIND | DB | prepare / bind check |
| EXPLAIN | DB | SQLite plan features |
| VALUE_PROBE | DB | `EXISTS` micro-probe per literal predicate, plus a near-value lookup |
| EXEC_BOUNDED | DB | time-bounded execution, keeping the result shape and a fingerprint |
| **VALUE_FIX** | DB only, no LLM | rewrite a non-existent literal to the DB value it matches ignoring case, or to the single near value |
| REPAIR_DB | LLM (flash) | repair conditioned on c0's DB evidence |
| REPAIR_BLIND | LLM (flash) | the same prompt with generic feedback (control) |
| RESAMPLE_1/2 | LLM (flash) | independent samples at T=0.8 |
| SWITCH_PRO / SWITCH_QWEN | LLM | another model, temperature 0 |

Latency is measured wall-clock. For SQLite in-process, DB time ≈ DB CPU. Cost is in token-weighted units, as in `RESULTS.md`.

---

## Oracle action analysis: is the cheapest successful action state-dependent?

Cheapest action after c0 that yields a correct answer (exclusive; actions are tried in order of mean added cost):

| share | cheapest successful action |
|---|---|
| **63.0%** | ACCEPT immediately |
| 1.0% | VALUE_FIX (DB only, ~10 ms) |
| 6.0% | RESAMPLE_1 (same model) |
| 2.0% | REPAIR (0.7% blind + 1.3% with DB evidence) |
| 4.7% | SWITCH model (2.7% pro + 2.0% qwen) |
| **23.3%** | **no action in the pool succeeds** |

Non-exclusive rescue rate among the 111 wrong first candidates, split by the state *after* the cheap DB probes:

| state after probes | n wrong | VALUE_FIX | REPAIR_DB | REPAIR_BLIND | RESAMPLE | SWITCH_PRO | SWITCH_QWEN | any cheap (≤1 flash call) | any action |
|---|---|---|---|---|---|---|---|---|---|
| **DB-flagged** | 22 | 0.14 | 0.27 | 0.18 | 0.32 | 0.18 | 0.23 | **0.50** | 0.55 |
| not flagged | 89 | 0.00 | 0.04 | 0.06 | 0.15 | 0.17 | 0.21 | **0.18** | 0.33 |

**Reading:**

- **Different states do need different actions.**
  - When the DB flags the candidate, cheap actions recover half of the errors, and expensive models add almost nothing (11 → 12 of 22).
  - When the DB sees nothing wrong, cheap same-model actions rarely help. The recoverable errors there need *a different model* (16 → 29 of 89).
- **But the action space is thin.**
  - The pure DB action (VALUE_FIX) solves only 1% of questions.
  - **70/300 (23%) cannot be rescued by any single action in the pool.** Some of these are BIRD label noise; that hasn't been checked yet.
  - The hoped-for distribution ("18% DB probe sufficient") is **not** what we see. The DB is far more valuable as a **source of decision information** than as a **source of answers**.

---

## Experiment A: marginal decision value of each action

The target is P(c0 is correct). The base state is P1 (question + schema + c0's SQL), with AUROC 0.644. The model is L2 logistic regression with 10×5-fold out-of-fold predictions; CIs are paired bootstrap.

| action | ΔAUROC | 95% CI | applicable | fires | median ms | p95 ms | cost |
|---|---|---|---|---|---|---|---|
| PARSE_BIND | +0.010 | [−0.003, +0.024] | 100% | 1.7% | 1.7 | 14 | ≈0 |
| EXPLAIN (SQLite) | −0.002 | [−0.014, +0.011] | 98% | 2.7% | 2.1 | 13 | ≈0 |
| **VALUE_PROBE** | **+0.036** | [+0.008, +0.067] | 70% | 3.7% | **2.7** | 26 | ≈0 |
| **EXEC_BOUNDED** | **+0.066** | [+0.024, +0.112] | 100% | 6.3% | **9.0** | 378 | 0.09 |
| **all 4 DB probes** | **+0.080** | [+0.034, +0.131] | 100% | 7.7% | **20** | 394 | 0.10 |
| extra LLM sample, SQL-text agreement only | −0.004 | [−0.007, −0.000] | 100% | 76% | 4,399 | 13,843 | 2.96 |
| extra LLM sample + execution agreement | +0.046 | [+0.015, +0.082] | 100% | 28% | 4,433 | 13,856 | 2.96 |
| other-model candidate + execution agreement | +0.042 | [+0.013, +0.073] | 100% | 26% | 3,361 | 28,993 | 5.67 |
| evidence repair leaves SQL unchanged | +0.043 | [+0.014, +0.073] | 100% | 25% | 4,499 | 13,376 | 3.05 |
| all DB probes + extra sample with execution agreement | +0.098 | [+0.047, +0.148] | 100% | — | 4,445 | 13,862 | 2.96 |

**Reading:**

- **The cheap DB probes carry about 1.7× more decision information than one more LLM call** (+0.080 vs +0.046), at about **1/200 of the latency** (20 ms vs 4.4 s median) and about **1/30 of the cost**.
- **An extra LLM sample is worthless as evidence unless the DB executes it.** SQL-text agreement gives −0.004; execution-result agreement gives +0.046. Even the "ask another LLM" signal is really a DB signal.
- VALUE_PROBE is cheap and applicable to 70% of queries, and it has a positive CI on its own. EXEC_BOUNDED is the strongest single probe.
- EXPLAIN and PARSE_BIND add nothing on SQLite. The plan is too coarse, and nearly all candidates bind. EXPLAIN should be re-tested on PostgreSQL, which exposes cardinalities and costs.

This table is the justification for **active evidence acquisition**: the cheapest information sources are DB actions, and they beat an extra LLM call.

---

## Experiment C: DB action vs another LLM call

Each action's output *replaces* c0. dP = acc(action) − acc(keep c0).

| region (n, acc of c0) | best action by dP/s | dP | +ms | dP/s | dP/cost |
|---|---|---|---|---|---|
| DB-flagged (23, 0.04) | **VALUE_FIX** | +0.130 | 53 | **+2.48** | **+2.48** |
| | RESAMPLE_1 | +0.304 | 5,498 | +0.055 | +0.093 |
| | REPAIR_DB | +0.261 | 8,711 | +0.030 | +0.074 |
| | REPAIR_BLIND | +0.174 | 8,621 | +0.020 | +0.056 |
| | SWITCH_QWEN | +0.217 | 13,587 | +0.016 | +0.020 |
| unflagged & uncertain (95, 0.50) | SWITCH_QWEN | +0.021 | 11,739 | +0.002 | +0.002 |
| | RESAMPLE_1 | **−0.084** | 6,596 | — | — |
| | REPAIR_DB | −0.032 | 5,809 | — | — |
| unflagged & confident (182, 0.78) | everything except qwen | ≤ 0 | | | |

**Reading:**

- **"Ask the DB" wins clearly in a well-defined state region: when a probe has flagged the candidate.** There, VALUE_FIX gives about 45× more dP per second than the best LLM action, and an evidence repair or resample recovers 26–30% more.
- **Outside that region, replacing the candidate with another LLM output hurts.** A resample loses 8.4 points in the uncertain region. So actions must be **verified before they replace c0**: generate an alternative, and keep it only if DB evidence such as execution agreement supports it. This is a direct design requirement for the planner in Experiment D.
- Evidence repair beats blind repair on flagged states (0.30 vs 0.22), but **a plain same-model resample is at least as good (0.35)**. With n=23, the "evidence-conditioned repair" claim is **not supported** yet.

---

## Experiment B: do I need this probe?

Policy: c0 → (probes) → ACCEPT if P(correct | collected evidence) ≥ τ, otherwise the fallback.

- The acceptor is trained on all 16 probe subsets, so it can decide from partial evidence.
- The adaptive gate runs probes from cheapest to most expensive, and runs a probe only if it predicts that probe would flip the decision. It is cross-fitted.

Fallback = SWITCH_QWEN, τ = 0.5:

| policy | EX acc | DB ms/q (mean) | DB ms p95 | probes/q | LLM cost/q |
|---|---|---|---|---|---|
| No-Probes | 0.647 | 0 | 0 | 0 | 6.37 |
| All-Probes | **0.657** | 102.6 | 394 | 4.0 | 6.39 |
| Adaptive (gate ≥ 0.5) | 0.650 | 30.8 | 151 | 0.77 | 6.21 |
| **Adaptive (gate ≥ 0.7)** | **0.650** | **19.0** | **52** | 0.46 | 6.12 |
| Adaptive (gate ≥ 0.85) | 0.650 | 11.4 | 14 | 0.28 | 6.24 |

**Reading:**

- Mechanically, adaptive probing works. It keeps about 0.3 of the 1.0-point gain from All-Probes while cutting **mean DB time by 81% and p95 DB time by 87%**.
- **This is not yet a systems result.**
  1. The end-to-end accuracy gain from probes is about 1 point, which is within noise. The reason is that the fallback rarely rescues: the decision information exists (Experiment A), but **the action space can't cash it in** (oracle).
  2. On SQLite with BIRD-sized DBs, all four probes cost about 100 ms, under 2% of a 6–10 s LLM call. Saving DB work only matters when DB work is expensive: larger databases, PostgreSQL, concurrent load, or `EXPLAIN ANALYZE`-style probes.
- With REPAIR_DB as the fallback the picture is the same: All-Probes 0.643 vs No-Probes 0.633.

---

## Verdict: should PlanGate become a cost-based hybrid DB–LLM optimizer?

**Supported:**

1. **DB actions are the most cost-effective source of decision information** (Experiment A): +0.080 AUROC for 20 ms, versus +0.046 for a 4.4 s LLM call. LLM self-agreement only becomes informative *through* DB execution.
2. **The best action depends on the DB-observed state** (oracle and Experiment C).
   - In the flagged state, cheap DB or same-model actions recover 50% of errors, and DB-only VALUE_FIX is about 45× more time-efficient than any LLM action.
   - In the unflagged state, cheap actions do nothing or harm, and only a different model helps.

**Weakened:**

3. **The DB rarely produces the answer itself.** VALUE_FIX solves 1%. The oracle shows **no evidence for a large "DB probe sufficient" mass**.
4. **The actionable headroom is small.** Only 41/300 (13.7%) wrong first candidates can be rescued by any single action, and 23% can't be rescued at all.
5. **The adaptive-probe saving is real in relative terms (−81% DB time) but negligible end to end on SQLite/BIRD.**

**Recommended reformulation:** treat PlanGate as **verification-centric planning**. DB probes are the cheap *sensors* that decide (a) whether to stop, (b) *which kind* of action to take (cheap local fix vs a different model), and (c) whether an alternative candidate may *replace* the current one (accept only with execution support). The claimed contribution becomes "DB evidence is the cheapest decision signal and routes the next action", not "DB actions replace LLM calls".

## Next steps

1. **Experiment D: a budgeted planner with verified replacement.** Actions: ACCEPT, PROBE(type), VALUE_FIX, RESAMPLE, SWITCH, REPAIR. An alternative replaces c0 only if the acceptor prefers it given execution agreement. Optimise Q(s,a) − λ_c·C − λ_l·L with logistic or gradient-boosted value models. All the data needed is already in `action_table.jsonl`, so this needs no new LLM calls.
2. **Audit the 70 unrescuable questions for BIRD label noise.** `dev_tied_append.json` and the known BIRD error lists would help. If many are label errors, the real headroom is larger.
3. **Make DB cost matter: a PostgreSQL arm** on a larger-scale copy of the BIRD DBs. That is where Experiment B can become a systems result, and where EXPLAIN gains real features.
4. Only after D: scale to 500–700 questions, then run the 1534-question budget-matched frontier (Experiment E) against best-single, router, cascade, best-of-N and a PRISM-like ensemble budget.

## Caveats

- n=300, with 23 flagged questions in the DB-flagged region, so the Experiment C numbers in that region are noisy.
- The bootstrap CIs in Experiment A hold the out-of-fold predictions fixed; they don't include training variance.
- The oracle ordering puts RESAMPLE (2.96) before REPAIR (3.03–3.05) by tiny cost differences. The non-exclusive table is the fairer view.
- DB times were measured while other probes ran concurrently in threads, so they are slightly inflated.
