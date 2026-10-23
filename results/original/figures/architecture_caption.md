# Paper figures: captions and status

Regenerate everything with:

```
python experiments/plot_teaser.py
python experiments/plot_architecture.py
python experiments/plot_running_example.py
```

The PDFs are vector files; include them directly in LaTeX.

---

## Figure 1: teaser (`teaser.pdf`, `figure*`)

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/teaser.pdf}
  \caption{(a) Existing cost-aware Text-to-SQL systems decide once (a model, a pipeline or a cascade
  order) and use the DBMS only to execute the final answer. PlanGate instead plans over LLM \emph{and}
  DBMS operators, re-planning after every observation; DB evidence is priced by the optimizer before
  it is acquired. (b, c) Execution accuracy versus mean cost per query against re-implemented
  published methods (AutoMix, CHASE-SQL, EllieSQL, SQLens, DIN-SQL, MAC-SQL), a MoT-style
  agreement cascade and single models; each method is shown at its most accurate setting, and
  PlanGate as its cost--accuracy frontier.}
  \label{fig:teaser}
\end{figure*}
```

## Figure 2: architecture (`architecture.pdf`, `figure*`)

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/architecture.pdf}
  \caption{PlanGate architecture. \emph{Online}, the adaptive planner (C2) runs a
  PLAN--ACT--OBSERVE loop: it chooses the next operator $a_t$ by estimated value minus
  $\lambda\cdot$cost, runs exactly one operator from the catalog, adds the result ($y_t$, a candidate
  SQL, or $o_t$, a DB observation) to its state $s_t$, and re-plans until it accepts. DB evidence
  operators (C1) are first-class operators priced by the optimizer ($\hat{C}$ from
  \texttt{EXPLAIN}) before they run, inside a read-only, time-bounded sandbox. The answer comes from
  verified selection (C3) over execution-result clusters. \emph{Offline}, counterfactual replay of
  every action trains the value models, which are calibrated per budget or SLO into a policy
  $\pi_v$ deployed with the operator cost model. Dashed: periodic feedback.}
  \label{fig:architecture}
\end{figure*}
```

Contribution badges (match the contribution list in the introduction):

| Badge | Contribution |
|---|---|
| C1 | DBMS operations as first-class, optimizer-priced inference operators |
| C2 | Adaptive re-planning after every observation (mid-inference re-optimization) |
| C3 | Verified selection: a new candidate replaces the current one only with execution evidence |

## Figure 3: running example (`running_example.pdf`, single column)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/running_example.pdf}
  \caption{Running example (BIRD dev, question 650). The cheap model follows the hint and filters on
  \texttt{'Outliers'}, which returns no rows. A 36\,ms value probe finds that the stored value is
  \texttt{'outliers'}; the planner chooses the DB-only \textsc{value fix} instead of another LLM call
  and accepts after re-execution (1 LLM call, cost 2.50). Every LLM-only alternative, including two
  stronger models, repeats the wrong literal. With a larger budget PlanGate also calls the strongest
  model, which is wrong; verified selection keeps the \textsc{value fix} answer.}
  \label{fig:example}
\end{figure}
```

All numbers in Figure 3 are measured: `results/action_table.jsonl`, question 650. The plan
PlanGate executes is confirmed by `python experiments/trace_example.py 650`.

---

## Implementation status (state this precisely in the paper)

| Component | Status |
|---|---|
| Online loop, LLM and DB operators, verified selection, two-phase planner | implemented and evaluated (Experiment D, both engines) |
| Optimizer-priced acquisition of bounded execution (C1) | **implemented and evaluated in the planner on PostgreSQL** (`experiments/pg_priced_planner.py`) |
| Probe gate | evaluated separately (Experiment B); not part of the planner, so not shown in the v2 figure |
| Offline workload = training split | **planned**: current value models are cross-fitted on the evaluation questions |
| Telemetry, periodic retraining, shadow runs | **planned** (dashed in the figure) |

## Result for C1 integrated in the planner (PostgreSQL, n = 500)

BOUNDED EXEC runs only if the optimizer price $\hat{C}$ is at most $\theta$; all other probes always run.

| $\theta$ | Candidates executed | Best EX | EX at 2× budget | DB time per query | p95 DB time |
|---|---|---|---|---|---|
| ∞ (always execute) | 100% | 0.676 | 0.648 | 47.9 ms | 187 ms |
| p90 of $\hat{C}$ | 88% | 0.682 | 0.652 | 32.5 ms (−32%) | 129 ms (−31%) |
| p75 | 74% | 0.674 | 0.646 | 16.5 ms (−66%) | 36 ms (−81%) |
| p50 | 49% | 0.674 | 0.654 | 13.1 ms (−73%) | 26 ms (−86%) |
