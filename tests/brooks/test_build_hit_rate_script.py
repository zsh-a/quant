"""Smoke tests for scripts/brooks_build_hit_rate.py dry-run path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "brooks_build_hit_rate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("brooks_build_hit_rate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dry_run_writes_empty_parquet(tmp_path):
    out = tmp_path / "hit_rate.parquet"
    script = _load_script()
    exit_code = script.main(["--dry-run", "--output", str(out)])
    assert exit_code == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert df.empty
    assert list(df.columns) == [
        "pattern",
        "regime",
        "htf_aligned",
        "side",
        "samples",
        "hit_rate_1r",
        "hit_rate_2r",
        "avg_realized_r",
    ]


def test_missing_db_writes_empty_parquet(tmp_path):
    out = tmp_path / "hit_rate.parquet"
    missing_db = tmp_path / "does_not_exist.db"
    script = _load_script()
    exit_code = script.main(["--db-path", str(missing_db), "--output", str(out)])
    assert exit_code == 0
    assert out.exists()
    assert pd.read_parquet(out).empty
