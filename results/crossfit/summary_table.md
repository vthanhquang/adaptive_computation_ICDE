
### SQLite (n=300)

| Group | Method | EX | Cost/query | p95 latency (s) |
|---|---|---|---|---|
| Fixed model | flash | 0.630 | 3.03 | 32.5 |
| Fixed model | pro | 0.603 | 5.67 | 29.0 |
| Fixed model | qwen | 0.673 | 9.74 | 30.0 |
| Generic baseline | Self-consistency@3 | 0.640 | 9.18 | 35.3 |
| Generic baseline | Self-consistency@2 | 0.630 | 6.14 | 34.0 |
| Generic baseline | Ensemble 3 models (exec vote) | 0.657 | 18.70 | 38.8 |
| Generic baseline | Ensemble 5 candidates (exec vote) | 0.657 | 24.76 | 39.6 |
| Generic baseline | MoT-style agreement cascade | 0.670 | 9.08 | 44.5 |
| Generic baseline | Rule cascade (DB trigger) | 0.647 | 3.96 | 32.9 |
| Generic baseline | One-shot router (best point) | 0.640 | 4.85 | 32.7 |
| Published method (re-impl.) | AutoMix (theta=0.90) | 0.667 | 15.11 | 62.6 |
| Published method (re-impl.) | AutoMix few-shot (theta=0.90) | 0.673 | 16.44 | 60.7 |
| Published method (re-impl.) | CHASE-SQL selection (flash x2, pro, qwen) | 0.637 | 25.57 | 54.0 |
| Published method (re-impl.) | CHASE-SQL selection (flash, pro, qwen) | 0.637 | 21.42 | 44.5 |
| Published method (re-impl.) | CHASE-SQL selection few-shot (flash x2, pro, qwen) | 0.643 | 26.03 | 50.0 |
| Published method (re-impl.) | CHASE-SQL selection few-shot (flash, pro, qwen) | 0.633 | 21.74 | 45.4 |
| Published method (re-impl.) | DIN-SQL | 0.537 | 20.96 | 178.4 |
| Published method (re-impl.) | EllieSQL (DPO router) | 0.633 | 7.21 | 56.8 |
| Published method (re-impl.) | EllieSQL (in-domain router, t=0.9) | 0.650 | 7.92 | 56.7 |
| Published method (re-impl.) | EllieSQL (in-domain router) | 0.637 | 6.42 | 53.2 |
| Published method (re-impl.) | EllieSQL G_A (always) | 0.640 | 9.52 | 65.3 |
| Published method (re-impl.) | EllieSQL G_B (always) | 0.593 | 5.68 | 38.0 |
| Published method (re-impl.) | EllieSQL G_M (always) | 0.640 | 6.83 | 44.6 |
| Published method (re-impl.) | MAC-SQL | 0.660 | 6.63 | 56.1 |
| Published method (re-impl.) | SQLens (flash) | 0.643 | 9.43 | 72.8 |
| Published method (re-impl.) | SQLens (pro) | 0.633 | 13.22 | 59.8 |
| Published method (re-impl.) | SQLens separate-prompts (flash) | 0.637 | 23.02 | 72.1 |
| Published method (re-impl.) | SQLens separate-prompts (pro) | 0.627 | 37.83 | 79.0 |
| PlanGate | PlanGate (budget 1.25x) | 0.660 | 3.66 | 35.8 |
| PlanGate | PlanGate (budget 2.0x) | 0.670 | 4.06 | 34.1 |
| PlanGate | PlanGate (best point) | 0.680 | 9.52 | 45.3 |
| Ablation | PlanGate without DB (best point) | 0.667 | 13.06 | 50.1 |
| Upper bound | Oracle (best single action) | 0.763 | 3.70 | 35.1 |

### PostgreSQL (n=500)

| Group | Method | EX | Cost/query | p95 latency (s) |
|---|---|---|---|---|
| Fixed model | flash | 0.592 | 2.90 | 12.3 |
| Fixed model | pro | 0.600 | 5.69 | 14.5 |
| Fixed model | qwen | 0.640 | 9.64 | 22.7 |
| Generic baseline | Self-consistency@2 | 0.602 | 5.81 | 16.2 |
| Generic baseline | Ensemble 3 models (exec vote) | 0.630 | 18.27 | 29.1 |
| Generic baseline | MoT-style agreement cascade | 0.642 | 8.94 | 26.7 |
| Generic baseline | Rule cascade (DB trigger) | 0.612 | 3.69 | 17.1 |
| Generic baseline | One-shot router (best point) | 0.632 | 6.97 | 20.7 |
| Published method (re-impl.) | AutoMix (theta=0.90) | 0.632 | 14.99 | 54.1 |
| Published method (re-impl.) | AutoMix few-shot (theta=0.90) | 0.634 | 16.39 | 55.2 |
| Published method (re-impl.) | CHASE-SQL selection (flash x2, pro, qwen) | 0.624 | 26.02 | 46.5 |
| Published method (re-impl.) | CHASE-SQL selection (flash, pro, qwen) | 0.620 | 21.46 | 43.3 |
| Published method (re-impl.) | CHASE-SQL selection few-shot (flash x2, pro, qwen) | 0.628 | 26.66 | 45.1 |
| Published method (re-impl.) | CHASE-SQL selection few-shot (flash, pro, qwen) | 0.620 | 21.79 | 42.7 |
| PlanGate | PlanGate (budget 1.25x) | 0.622 | 3.62 | 18.9 |
| PlanGate | PlanGate (budget 2.0x) | 0.648 | 5.68 | 29.5 |
| PlanGate | PlanGate (best point) | 0.676 | 11.09 | 32.3 |
| Ablation | PlanGate without DB (best point) | 0.656 | 11.51 | 28.9 |
| Upper bound | Oracle (best single action) | 0.750 | 3.70 | 15.9 |
