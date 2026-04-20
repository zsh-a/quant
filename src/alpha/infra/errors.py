"""Graded exception hierarchy for Alpha Lab automation.

These classes let the auto-runner and Celery tasks distinguish between
retryable problems (network blips, LLM rate limits), cycle-skippable
problems (data not yet available), and hard stops (bad config).

The orchestrator also uses ``SearchAborted`` to signal cooperative
cancellation from a user-triggered cancel request.
"""

from __future__ import annotations


class AlphaAutomationError(Exception):
    """Base for all automation-layer errors."""


class TransientError(AlphaAutomationError):
    """Retryable — expected to resolve on its own within seconds/minutes.

    Examples: LLM rate limit, temporary network failure, Redis momentary
    disconnect. The Celery retry policy triggers on these.
    """


class DataError(AlphaAutomationError):
    """The current cycle lacks viable data; skip this run, try again later.

    Examples: ClickHouse table missing rows for the target window, all
    symbols filtered out by min_quote_volume, no seeds and enum_max=0.
    """


class FatalError(AlphaAutomationError):
    """Unrecoverable — human intervention required.

    Examples: config schema violation, code exception that indicates a
    bug, missing required API key. These halt the loop and alert.
    """


class SearchAborted(AlphaAutomationError):
    """Raised when an in-flight search detects a user cancel request.

    Treated as a non-error terminal state by callers.
    """


def classify_exception(exc: BaseException) -> type[AlphaAutomationError]:
    """Best-effort bucket for an arbitrary exception into our hierarchy.

    Used by the Celery task to decide whether to retry or fail-fast.
    """
    # Explicit classes take precedence
    if isinstance(exc, AlphaAutomationError):
        return type(exc)

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Network / rate-limit heuristics → transient
    transient_hints = (
        "timeout",
        "rate limit",
        "ratelimit",
        "connection reset",
        "temporarily unavailable",
        "503",
        "502",
        "504",
    )
    if any(hint in msg for hint in transient_hints):
        return TransientError
    if name in {"connectionerror", "timeouterror", "readtimeout", "connecttimeout"}:
        return TransientError

    # Data-layer heuristics → skip this cycle
    if any(hint in msg for hint in ("no data", "empty dataset", "0 symbols", "insufficient")):
        return DataError

    # Default: treat as transient for one retry round, elevate to fatal if it persists.
    return TransientError
