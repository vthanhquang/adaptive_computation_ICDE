"""PlanGate: cost-based inference planning over LLM and DBMS operators for Text-to-SQL.

Package layout:
  benchmark   BIRD dev / train / mini-dev loaders
  engines     SQLite and PostgreSQL sandboxes, execution-accuracy scoring
  evidence    DB evidence operators (parse/bind, EXPLAIN, value probe, bounded execution) and caches
  llm         provider clients, prompts, generation and repair with a persistent response cache
  features    state features for the learned planner components
  planning    the adaptive evidence-acquisition planner and the cross-fitted planner
  evaluation  held-out evaluation harness and statistics
  baselines   re-implementations of published Text-to-SQL methods
"""
