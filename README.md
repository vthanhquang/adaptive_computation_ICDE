# PlanGate: cost-based inference planning over LLM and DBMS operators for Text-to-SQL

PlanGate treats LLM Text-to-SQL serving as cost-based planning over two kinds of operators: LLM
calls (generate, resample, repair, switch to a stronger model) and DBMS evidence operations
(parse/bind, `EXPLAIN`, literal value probes, a DB-only value fix, bounded execution). A planner
runs a plan → act → observe loop: it executes one operator at a time, updates its state with what
it observed and re-plans. DB observations are priced before they run, using the optimizer's
estimate, and the final answer comes from verified selection over execution-result clusters.

This repository contains the library, every experiment script, the released intermediate
artifacts (replay tables, per-question traces, LLM response cache) and all reported results.
Every table and figure of the paper can be regenerated from the released artifacts without LLM API
calls and, except where noted, without a database server.

## Repository layout

```text
src/plangate/              library (pip install -e .)
  benchmark.py             BIRD dev / train / mini-dev loaders
  engines/                 read-only SQLite and PostgreSQL sandboxes, EX scoring, cross-engine gold
  evidence/                DB evidence operators, differential probe, evidence cache
  llm/                     provider clients, prompts, GEN / REPAIR operators, response cache
  features.py              state features
  planning/adaptive.py     adaptive evidence acquisition (priced probes, value of information)
  planning/crossfit.py     cross-fitted sequential planner (two-phase policy, verified selection)
  evaluation/              held-out harness, external baselines on replay tables, statistics
  baselines/               EllieSQL, SQLens, DIN-SQL, MAC-SQL re-implementations
experiments/               one module per experiment, run as `python -m experiments.<group>.<name>`
  crossfit/                study on the BIRD dev samples (300 SQLite / 500 PostgreSQL entries)
  baselines/               published methods re-implemented in the same setting
  heldout/                 held-out train / test evaluation of adaptive evidence acquisition
  scaling/                 resource replay on 10x / 100x scaled PostgreSQL data
  figures/                 paper figures
configs/                   generation configs, held-out split manifests (configs/splits/)
results/                   all outputs; see results/README.md
artifacts/cache.zip        cached LLM responses and DB evidence
docs/                      data setup and step-by-step reproduction
tests/                     unit tests (no data needed) and data-dependent smoke tests
third_party/               prompt templates vendored from DIN-SQL and MAC-SQL
```

## Installation

Python 3.10 or newer. The experiments were run with Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3,
sqlglot 30.21, psycopg 3.3 and PostgreSQL 16.

```bash
pip install -e .                 # library and experiment dependencies
pip install -e ".[elliesql]"     # optional: EllieSQL router (torch, transformers)
pip install -e ".[dev]"          # optional: pytest, ruff
```

Benchmark data (BIRD dev, BIRD train, BIRD mini-dev, PostgreSQL 16 with the BIRD dump) is not part
of the repository. [`docs/data.md`](docs/data.md) explains how to obtain each item. Paths can be
redirected with `PLANGATE_DATA_DIR`, `PLANGATE_CACHE_DIR` and `PLANGATE_RESULTS_DIR`.

## Quick start: regenerate tables and figures from the released artifacts

These steps need neither API keys nor a database server:

```bash
python -m experiments.heldout.external_baselines pg pg_x10 pg_x100 sqlite
python -m experiments.heldout.leave_db_out_summary
python -m experiments.heldout.make_tables          # results/tables/*.tex
python -m experiments.scaling.cost_summary
python -m experiments.figures.heldout_results      # results/figures/heldout_frontier.pdf, scaling_dbtime.pdf
```

With the BIRD databases in `data/` and the cache unpacked (`make cache`), the cross-fitted study
also runs from the cache alone, for example:

```bash
python -m experiments.crossfit.planner sqlite      # about 2 minutes
python -m experiments.crossfit.summary_table
```

[`docs/reproducibility.md`](docs/reproducibility.md) lists every command in order: generation,
action and replay tables, split manifests, held-out runs, scaled databases, ablations, tables and
figures. The `Makefile` wraps the common steps (`make help`).

## Evaluation protocols

The results come from three protocols, which are reported separately and never pooled.

| Protocol | What is learned on what | Data | Scripts | Results |
|---|---|---|---|---|
| Cross-fitted | estimators cross-fitted (5 folds) on the study questions | 300 BIRD dev questions (SQLite), 500 mini-dev entries (PostgreSQL) | `experiments/crossfit`, `experiments/baselines` | `results/crossfit`, `results/baselines` |
| Held-out | fitted and calibrated on training questions only; the test split is touched once | train: 287 BIRD train (SQLite) / 417 BIRD dev questions outside mini-dev (PostgreSQL); test: 300 / 498 | `experiments/heldout` | `results/heldout` |
| Resource replay | held-out protocol with DB costs measured on 10x and 100x scaled copies of the database | PostgreSQL | `experiments/scaling`, `experiments/heldout` | `results/scaling` |

All split manifests pass the leakage checks: no shared question id, normalized question text or
(database, gold SQL) between training and test (`configs/splits/MANIFEST.json`).

## Results

Costs are normalized units: per-model token-weighted LLM cost (`plangate.llm.generation.cost_units`)
plus DB time at a price of 1 unit per DB-second unless stated otherwise. Budgets are multiples of the
cost of the cheap first candidate.

**Cross-fitted study** (`results/crossfit/summary_table.md`; full comparison with six
re-implemented published methods in [`results/baselines/README.md`](results/baselines/README.md)):

| | SQLite (n=300) | PostgreSQL (n=500) |
|---|---|---|
| Cheap model (deepseek-v4-flash) | 0.630 EX, cost 3.03 | 0.592 EX, cost 2.90 |
| Strongest single model (qwen3.8-max) | 0.673 EX, cost 9.74 | 0.640 EX, cost 9.64 |
| PlanGate at 2x budget | 0.670 EX, cost 4.06 | 0.648 EX, cost 5.68 |
| PlanGate, most accurate point | 0.680 EX, cost 9.52 | 0.676 EX, cost 11.09 |
| PlanGate without DB access, most accurate point | 0.667 EX, cost 13.06 | 0.656 EX, cost 11.51 |

**Held-out evaluation** (`results/tables/heldout_main.tex`, `results/heldout/*/summary.json`).
Accuracy differences between policies at matched budgets are within about two points, and most
paired bootstrap intervals include zero (`results/tables/heldout_ci.tex`). The clearer effect is
the DB work spent: when probes are expensive, adaptive acquisition buys far less evidence than a
fixed bundle or the optimizer gate at similar accuracy. At 100x data costs and a 1.5x budget, the
adaptive policy uses 89 ms of DB time per question at 0.598 EX, against 788 ms at 0.600 EX for the
optimizer-gated policy. Against the best external baseline that fits each budget,
PlanGate is ahead by 1.2 to 2.0 points on PostgreSQL at 1.25x to 3x budgets, and within the
confidence interval elsewhere (`results/tables/external_heldout.tex`). The leave-databases-out
folds give the same picture (`results/tables/leave_db_out.tex`).

## Tests

```bash
pytest                    # unit tests; data-dependent tests are skipped when data/ is absent
ruff check src experiments tests
```

## LLM access

New generations call each provider's official OpenAI-compatible API (DeepSeek, Z.ai, Moonshot AI,
Alibaba Cloud DashScope, Tencent Hunyuan; `src/plangate/llm/providers.py`). Keys are read from
`DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `DASHSCOPE_API_KEY` and `HUNYUAN_API_KEY`;
only the keys of the models actually called are needed. Responses are cached by
(model, messages, temperature, sample), so with the released cache no call is made. Token counts,
and therefore costs, are the ones recorded in the cache; a fresh run can report slightly different
token counts.

## Attribution

BIRD, BIRD train and BIRD mini-dev are from the BIRD benchmark (https://bird-bench.github.io/).
Baseline prompts are vendored from the DIN-SQL and MAC-SQL repositories (see
`third_party/README.md`). The EllieSQL router checkpoint is downloaded from the HuggingFace Hub at
run time.
