# PlanGate — preliminary results (real LLMs, BIRD dev)

Date: 2026-10-05. Status: **preliminary**. n=300, a single run, not tuned.

## Setup

| | |
|---|---|
| Benchmark | BIRD dev, 300 questions stratified by (db, difficulty), seed 0, 11 SQLite DBs, with BIRD hints |
| Metric | BIRD execution accuracy (set equality of result rows) |
| LLM ops (`configs/real_catalog.json`) | `flash_direct` deepseek-v4-flash · `pro_direct` deepseek-v4-pro · `glm_direct` glm-5.3-flash · `pro_decompose` deepseek-v4-pro + step-by-step prompt · `qwen_max_direct` qwen3.8-max · `kimi_decompose` kimi-k3 + step-by-step prompt. All run through MiraiAPI with temperature 0 and the DDL + 3 sample rows schema prompt |
| DB ops (`plangate/db_evidence.py`) | `PARSE_BIND`, `EXPLAIN` (plan features), `VALUE_PROBE` (literal-predicate micro-probes plus near-value lookup), `EXEC_BOUNDED` (time-bounded execution that keeps only the result *shape* and a fingerprint) |
| Cost | Generator cost = model rate × (prompt + 4·completion tokens)/1000. DB cost = 1 unit per DB-second. Both are also reported separately |
| Protocol | A counterfactual matrix: every config on every question, 1800 real calls, cached in `cache/llm`. All policies replay over the same candidates and DB evidence. Learned parts are cross-fitted with 5-fold GroupKFold by question, so no question is ever scored by a model that saw it |

## 1. Oracle opportunity (E5): large

| config | EX acc | cost/q |
|---|---|---|
| flash_direct | 0.630 | 3.03 |
| pro_direct | 0.603 | 5.67 |
| glm_direct | 0.597 | 5.95 |
| pro_decompose | 0.607 | 6.52 |
| **qwen_max_direct** (best fixed) | **0.670** | 9.74 |
| kimi_decompose | 0.583 | 13.66 |
| **oracle** (cheapest correct config) | **0.757** | **3.47** |

- Only 46% of questions are solved by *all* configs, and 75.7% are solved by *some* config.
- The oracle is +8.7 points over the best fixed config at about ⅓ of its cost.
- **The ordering is not monotone in cost.** The cheapest model is the second-best fixed config, and the most expensive (kimi + decomposition) is the worst. On this model pool, the headroom comes from **picking the candidate that happens to be right**, not from "hard query → bigger model". That is an argument *against* a cost-ordered cascade, and in favour of state-dependent planning.

## 2. RQ1 — does post-generation information help? Yes, monotonically

Cross-fitted AUROC of P(candidate is correct | state):

| state | first candidate | all ladder states |
|---|---|---|
| P0: question + schema | 0.581 | 0.575 |
| P1: + candidate SQL / model signals | 0.643 | 0.646 |
| **P2: + DBMS evidence** | **0.715** | **0.753** |

- A question-only router can barely predict correctness (0.58).
- Each later state adds signal, and the DB evidence adds the most.
- At matched thresholds, adding DB evidence (P1 → P2) flips the accept/escalate decision for **13–23%** of questions.

This is the "novelty kill-test" from the research summary, and at the level of predictive information it **passes**.

## 3. RQ2 — cost–quality frontier: directionally positive, not yet significant

Best accuracy reachable within a budget (multiples of the cheapest config's cost):

| policy | 1.25× | 1.5× | 2× | 3× |
|---|---|---|---|---|
| one-shot router (P0) | 0.637 | 0.640 | 0.640 | 0.653 |
| sequential, P1 | 0.633 | 0.637 | 0.637 | 0.640 |
| **sequential, P2** (acceptor threshold, ladder) | **0.643** | **0.643** | **0.660** | **0.673** |
| rule cascade (exec error/empty → escalate) | – | 0.650 | 0.650 | 0.650 |
| value planner P2 (rescue rule) | 0.633 | 0.640 | 0.640 | 0.640 |

- At **3×** budget, the sequential P2 planner reaches the accuracy of the best fixed config (0.67), which costs **3.2×**.
- Paired bootstrap of sequential P2 vs the one-shot router at matched budget:
  - 2×: **+2.0 points**, 95% CI [−0.3, +4.3], P(>0) = 0.95.
  - 3×: +2.0 points, CI [−1.0, +5.0].
- **Not significant at n=300.**
- Sequential planning *without* DB evidence (P1) does **not** beat the router. The gain appears only once DBMS evidence is in the state, which is exactly the separation the paper needs.
- The hand-written value planner (`plangate/learned_planner.py`) is still **bimodal**: it either accepts, or it keeps generating once candidates disagree. It underperforms the simple threshold cascade. That is a planner-design problem to fix next, not a result.

Figure: `results/frontier.png`.

## 4. RQ3 — DB evidence as an *action trigger* (repair): high precision, low recall

On top of `flash_direct`, the trigger fires if any of these hold: parse/bind error, join without a condition, a literal value that does not exist, execution error or timeout, an empty result, or an all-NULL result.

- **23/300 (7.7%) are flagged. Flash accuracy on flagged questions is 4.3%, versus 67.9% on unflagged**, so the trigger is almost perfectly precise.
- Next action on the 23 flagged questions:

| next action | acc on flagged | extra cost / flagged q |
|---|---|---|
| keep (ACCEPT) | 0.043 | 0 |
| escalate to qwen_max | 0.261 | 10.75 |
| REPAIR, generic feedback (control) | 0.217 | 3.10 |
| **REPAIR with DB evidence** | **0.304** | **3.25** |

- Evidence-driven repair beats escalation at **30% of its cost**, and beats the blind second look.
- The extra fixes are value-grounding errors, for example `'stealth'` → `'Stealth'` and `'Outliers'` → `'outliers'`.
- One of them returned `COUNT = 0` rather than an empty set. Only the **value micro-probe** caught it, which an execution or `LIMIT` check would have missed.
- n=23 is tiny. Treat this as a qualitative result.

**The recall ceiling is the main limitation.** In an error taxonomy of flash's wrong answers, about 50% are "same result shape, different values". These are semantic or annotation-convention mismatches (`'Yes'` vs `'normal'`, `>=` vs `>`), and no hard DB check can see them. For those errors, the useful DB signal is **cross-candidate execution-result agreement**, which is a P2 feature. That is why the P2 acceptor beats P1.

## 5. DB cost is small but not zero

Running all four evidence operators costs a mean of 95 ms per candidate (p95 439 ms, max 5 s on a slow join). With the cost model above, that is about 0.1 units against about 3 units for the cheapest LLM call. Adaptive evidence selection (VOI) is not needed on BIRD-sized DBs. It would matter on larger databases, which supports the PostgreSQL arm.

## Honest caveats

- n=300, one sample per config, and temperature 0. Differences of 1–2 points are within noise.
- BIRD gold labels have known noise, which caps attainable EX.
- Operating points (λ, τ) are picked on the same questions for **every** policy, which is optimistic for all of them equally. A proper train/test split is the next step.
- The MiraiAPI proxy adds roughly 2–3k prompt tokens per call. This inflates costs and compresses the cost ratio between models: the real price spread is larger than measured here.
- The repair acceptor was trained on un-repaired candidates. The repair loop is not yet integrated into the learned planner.

## What this means for the research story

1. **Supported:** post-generation candidate and DB information carries decision value that question/schema routing lacks (RQ1). DBMS evidence is what makes sequential planning beat one-shot routing (P1 ≈ router < P2).
2. **Supported, qualitatively:** DB evidence changes *which* action is best. Repair is better than escalation when a micro-probe fires.
3. **Open:**
   - The frontier gain is small (+2 points at matched cost) and not yet significant.
   - The model pool is too homogeneous for "escalate to a stronger model" to be a strong action.
   - The value-based planner needs a better design. Candidates are multi-step lookahead or a learned stopping rule.

## Next steps (suggested)

1. Scale to the full BIRD dev set (1534 questions) to get statistical power, with a held-out split for picking λ and τ.
2. Add REPAIR(model) to the learned planner's action set. Add multi-sample generation at temperature > 0, since agreement is the strongest P2 signal.
3. Use a more heterogeneous pool: a small local model, plus one of claude-sonnet-5 / gpt-5.6-luna if the policy allows non-Chinese models.
4. Move the DB arm to PostgreSQL, where `EXPLAIN (FORMAT JSON)` gives cardinalities and costs.

## Reproduce

```
python experiments/run_matrix.py --n 300 --configs catalog --out results/matrix.jsonl
python experiments/evaluate_policies.py results/matrix.jsonl        # E5, RQ1, RQ2, bootstrap
python experiments/run_repair.py results/matrix.jsonl flash_direct qwen_max_direct   # RQ3
python experiments/error_analysis.py results/matrix.jsonl flash_direct
python experiments/plot_report.py
```

LLM responses and DB evidence are cached in `cache/`, so every re-run is free and deterministic.
