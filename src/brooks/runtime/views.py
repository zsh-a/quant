"""Pure projections from strategy state to BarEvent / WS payload dicts.

Lifted out of the live task so the same helpers serve both live and replay.
Each function is small and side-effect-free; the call sites are what
decides where the produced dict ends up (session_logs / WS / a unit test).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def decision_to_dict(decision: Any) -> Dict[str, Any]:
    """Flatten a :class:`Decision` for legacy ``strategy_step`` payloads."""
    sig = decision.signals[0] if decision.signals else None
    return {
        "side": decision.side,
        "entry_px": float(decision.entry_px),
        "stop_px": float(decision.stop_px),
        "target_px": float(decision.target_px) if decision.target_px is not None else None,
        "quantity": float(decision.quantity),
        "probability": float(decision.probability),
        "expected_r": float(decision.expected_r),
        "regime": decision.regime,
        "htf_aligned": bool(decision.htf_aligned),
        "pattern": sig.pattern if sig is not None else "unknown",
        "source": decision.source,
        "reasoning": decision.reasoning,
    }


def decision_to_full_dict(decision: Any) -> Dict[str, Any]:
    """Pydantic-friendly Decision dump (loader rebuilds via ``Decision(**dict)``)."""
    return decision.model_dump(mode="json")


def features_to_view_dict(feat: Any) -> Dict[str, Any]:
    return {
        "is_bull": bool(getattr(feat, "is_bull", False)),
        "body_pct": int(getattr(feat, "body_pct", 0)),
        "close_position": getattr(feat, "close_position", "mid"),
        "ema_relation": getattr(feat, "ema_relation", "at"),
        "leg_dir": getattr(feat, "leg_dir", "flat"),
        "leg_length": int(getattr(feat, "leg_length", 0)),
        "is_doji": bool(getattr(feat, "is_doji", False)),
        "is_inside_bar": bool(getattr(feat, "is_inside_bar", False)),
    }


def structure_to_view_dict(struct: Any) -> Dict[str, Any]:
    swings = []
    for s in getattr(struct, "confirmed_swing_highs", []) or []:
        swings.append({"idx": s.bar_idx, "kind": "high", "price": s.price})
    for s in getattr(struct, "confirmed_swing_lows", []) or []:
        swings.append({"idx": s.bar_idx, "kind": "low", "price": s.price})
    top = getattr(struct, "micro_channel_top", None)
    bot = getattr(struct, "micro_channel_bot", None)
    return {
        "always_in": getattr(struct, "always_in", "neutral"),
        "confirmed_swings": swings,
        "micro_channel_top": top.to_dict() if top is not None else None,
        "micro_channel_bot": bot.to_dict() if bot is not None else None,
        "last_breakout_lookback_high": getattr(struct, "last_breakout_lookback_high", None),
        "last_breakout_lookback_low": getattr(struct, "last_breakout_lookback_low", None),
    }


def regime_to_view_dict(
    regime_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not regime_payload:
        return None
    return {
        "name": regime_payload.get("regime") or regime_payload.get("name"),
        "confidence": float(regime_payload.get("confidence") or 0.0),
        "reasons": list(regime_payload.get("reasons") or []),
    }


__all__ = [
    "decision_to_dict",
    "decision_to_full_dict",
    "features_to_view_dict",
    "regime_to_view_dict",
    "structure_to_view_dict",
]
