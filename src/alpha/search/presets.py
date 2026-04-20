"""Alpha Lab search presets loaded from ``config/alpha_lab/presets.yaml``.

The front-end used to hard-code these; now the API is authoritative so
ops can tune them without a redeploy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from src.config.paths import CONFIG_DIR

_PRESETS_PATH = CONFIG_DIR / "alpha_lab" / "presets.yaml"


def _load_presets() -> dict[str, Any]:
    if not _PRESETS_PATH.exists():
        return {"presets": [], "auto_archive": {"top_k": 5}}
    try:
        data = yaml.safe_load(_PRESETS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"presets": [], "auto_archive": {"top_k": 5}}
    return data


@lru_cache(maxsize=1)
def get_presets_payload() -> dict[str, Any]:
    """Returned as-is by the API endpoint; structure matches the YAML."""
    payload = _load_presets()
    # Defensive defaults
    payload.setdefault("presets", [])
    payload.setdefault("auto_archive", {"top_k": 5})
    return payload


def invalidate_presets_cache() -> None:
    """Force a fresh read from disk on the next call (useful for admin tooling)."""
    get_presets_payload.cache_clear()


def list_presets() -> list[dict[str, Any]]:
    return list(get_presets_payload().get("presets") or [])


def auto_archive_top_k() -> int:
    try:
        return int(get_presets_payload().get("auto_archive", {}).get("top_k", 5))
    except Exception:
        return 5
