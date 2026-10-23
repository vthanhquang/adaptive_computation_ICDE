"""Filesystem layout of the repository.

Large inputs (benchmark databases, LLM and evidence caches) live outside version control. Their
locations default to ``data/`` and ``cache/`` under the repository root and can be redirected with
the environment variables ``PLANGATE_DATA_DIR``, ``PLANGATE_CACHE_DIR`` and ``PLANGATE_RESULTS_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).resolve() if value else default


DATA_DIR = _env_dir("PLANGATE_DATA_DIR", REPO_ROOT / "data")
CACHE_DIR = _env_dir("PLANGATE_CACHE_DIR", REPO_ROOT / "cache")
RESULTS_DIR = _env_dir("PLANGATE_RESULTS_DIR", REPO_ROOT / "results")
CONFIG_DIR = REPO_ROOT / "configs"
SPLITS_DIR = CONFIG_DIR / "splits"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"

# Result groups, one per evaluation protocol of the paper.
CROSSFIT_DIR = RESULTS_DIR / "crossfit"      # cross-fitted study on the BIRD dev samples
BASELINES_DIR = RESULTS_DIR / "baselines"    # re-implemented published methods
HELDOUT_DIR = RESULTS_DIR / "heldout"        # held-out train / test evaluation
SCALING_DIR = RESULTS_DIR / "scaling"        # resource replay on scaled PostgreSQL data
REPLAY_DIR = HELDOUT_DIR / "replay"          # evidence-conditioned replay tables
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
