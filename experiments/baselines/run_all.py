"""Runs every published-method step that needs the LLM APIs, then the matched-budget comparison.

Checks that the provider keys work (short AND long prompt) before spending anything, then runs:
  1. DIN-SQL / MAC-SQL                            (experiments.baselines.run_din_mac_sql)
  2. AutoMix + CHASE-SQL with few-shot prompts    (experiments.baselines.automix_chase_sql --fewshot, both engines)
  3. SQLens with one prompt per LLM signal        (experiments.baselines.run_sqlens --separate)
  4. matched-budget comparison                    (experiments.crossfit.planner sqlite / pg)
Each step's stdout is written to the matching results/ text file.
Usage: python -m experiments.baselines.run_all
"""
from __future__ import annotations
import subprocess, sys

from plangate.paths import REPO_ROOT, RESULTS_DIR

CHECK_MODELS = ["deepseek-v4-flash", "qwen3.8-max"]


def key_ok():
    from openai import OpenAI
    from plangate.llm import providers as official
    for model in CHECK_MODELS:
        p = official.provider_of(model)
        c = OpenAI(api_key=official.api_key(p), base_url=official.base_url(p), max_retries=0, timeout=120)
        for n in (5, 6000):   # quota problems first show up on long prompts
            try:
                c.chat.completions.create(model=model, max_tokens=5,
                                          messages=[{"role": "user", "content": "word " * n + "Reply OK"}])
            except Exception as e:
                print(f"API check failed for {model} (prompt of {n} words): {str(e)[:200]}")
                return False
    print("API keys OK")
    return True


STEPS = [
    ["experiments.baselines.run_din_mac_sql"],
    ["experiments.baselines.automix_chase_sql", "sqlite", "--fewshot"],
    ["experiments.baselines.automix_chase_sql", "pg", "--fewshot"],
    ["experiments.baselines.run_sqlens", "deepseek-v4-flash", "deepseek-v4-pro", "--separate"],
    ["experiments.crossfit.planner", "sqlite"],
    ["experiments.crossfit.planner", "pg"],
]
LOGS = ["baselines/din_mac_sql_run.txt", "baselines/automix_chase_sql_fewshot_sqlite.txt",
        "baselines/automix_chase_sql_fewshot_pg.txt", "baselines/sqlens_separate_run.txt",
        "crossfit/planner_sqlite.txt", "crossfit/planner_pg.txt"]


def main():
    if not key_ok():
        sys.exit("Stopping: check the provider API keys (see plangate/providers/official.py), then rerun.")
    for args, log in zip(STEPS, LOGS):
        print("running", " ".join(args), "->", log, flush=True)
        with open(RESULTS_DIR / log, "w", encoding="utf8") as f:
            rc = subprocess.call([sys.executable, "-m"] + args, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0:
            sys.exit(f"step failed ({rc}): {' '.join(args)}; see results/{log}")
        if args[0] != "experiments.crossfit.planner" and not key_ok():
            sys.exit("API stopped working mid-run; rerun this script later (finished calls are cached).")
    print("all steps done")


if __name__ == "__main__":
    main()
