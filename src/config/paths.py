"""
Centralized path configuration for the quantitative trading platform.

All data, log, and config directories are derived from a single PROJECT_ROOT.
This module has zero dependencies beyond stdlib to avoid circular imports.

Resolution order for PROJECT_ROOT:
  1. QUANT_PROJECT_ROOT environment variable (set in Docker to /app)
  2. Auto-detect from this file's location: Path(__file__).parents[2]
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────

_env_root = os.environ.get("QUANT_PROJECT_ROOT")
PROJECT_ROOT: Path = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parents[2]

# ── Top-level directories ─────────────────────────────────────────────────────

DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CONFIG_DIR: Path = PROJECT_ROOT / "config"

# ── Session database ──────────────────────────────────────────────────────────

SESSIONS_DB: Path = DATA_DIR / "sessions.db"

# ── Alpha research ────────────────────────────────────────────────────────────

ALPHA_DIR: Path = DATA_DIR / "alpha"
ALPHA_RUNS_DIR: Path = ALPHA_DIR / "runs"
ALPHA_ZOO_DIR: Path = ALPHA_DIR / "zoo"

# ── Alpha lab (experiments, training, checkpoints) ────────────────────────────

ALPHA_LAB_DIR: Path = DATA_DIR / "alpha_lab"
ALPHA_LAB_CHECKPOINTS_DIR: Path = ALPHA_LAB_DIR / "checkpoints"
ALPHA_LAB_NEURAL_DIR: Path = ALPHA_LAB_DIR / "neural"
STRATEGY_MEMORY_PATH: Path = ALPHA_LAB_DIR / "strategy_memory.json"
AUTO_SEARCH_STATE_PATH: Path = ALPHA_LAB_DIR / "auto_search_state.json"
SEARCH_JOBS_STATE_PATH: Path = ALPHA_LAB_DIR / "search_jobs_state.json"

# ── Market data sync state ────────────────────────────────────────────────────

CRYPTO_SYNC_STATE_PATH: Path = DATA_DIR / "crypto_sync_state.json"
TDX_SYNC_STATE_PATH: Path = DATA_DIR / "tdx_sync_state.json"
TDX_FIN_DATA_DIR: Path = DATA_DIR / "fin_data"

# ── Backtest checkpoints ─────────────────────────────────────────────────────

CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"

# ── Reports ───────────────────────────────────────────────────────────────────

REPORTS_DIR: Path = DATA_DIR / "reports"

# ── Logs ──────────────────────────────────────────────────────────────────────

LOG_FILE_PATH: Path = LOGS_DIR / "quant.log"
JSON_LOG_PATH: Path = LOGS_DIR / "quant.jsonl"


def ensure_data_dirs() -> None:
    """Create all required data directories. Idempotent — safe to call multiple times."""
    for d in (
        DATA_DIR,
        LOGS_DIR,
        ALPHA_RUNS_DIR,
        ALPHA_ZOO_DIR,
        ALPHA_LAB_DIR,
        ALPHA_LAB_CHECKPOINTS_DIR,
        ALPHA_LAB_NEURAL_DIR,
        TDX_FIN_DATA_DIR,
        CHECKPOINTS_DIR,
        REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
