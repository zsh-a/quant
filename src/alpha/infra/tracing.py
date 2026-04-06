"""Lightweight LLM chain observability.

Provides span-based tracing for LLM calls, formula evaluation, and search
orchestration.  Zero external dependencies — uses ``contextvars`` for
propagation and ``loguru`` for structured output.

Design goals:
- Trace every LLM API call with token/cost/latency breakdown
- Link LLM outputs to downstream formula quality (extracted → valid → compiled)
- Correlate parent metrics → suggestion → offspring quality in breeding loops
- Pluggable collectors for future integration (OpenTelemetry, LangFuse, etc.)

Quick start::

    from src.alpha.tracing import tracer

    with tracer.start_span("llm_call", kind="llm") as span:
        response = client.chat.completions.create(...)
        span.set_response(response)           # auto-extracts tokens/cost
        span.set("formulas_extracted", 5)
        span.set("formulas_valid", 3)
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from loguru import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpanEvent:
    """A point-in-time observation within a span."""
    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A timed operation in the LLM chain."""
    trace_id: str
    span_id: str
    parent_id: str | None
    operation: str
    kind: str  # "llm", "eval", "search", "breed", "mcts", "internal"
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    end_time: float | None = None
    status: str = "ok"
    error: str | None = None

    # --- builder API ---

    def set(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span."""
        self.attributes[key] = value
        return self

    def event(self, name: str, **attrs: Any) -> "Span":
        """Record a point-in-time event."""
        self.events.append(SpanEvent(name, time.time(), dict(attrs)))
        return self

    def set_error(self, err: str | Exception) -> "Span":
        self.status = "error"
        self.error = str(err)
        return self

    def set_response(self, response: Any) -> "Span":
        """Extract token/cost info from an OpenAI-style response object."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.attributes["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            self.attributes["completion_tokens"] = getattr(usage, "completion_tokens", None)
            self.attributes["total_tokens"] = getattr(usage, "total_tokens", None)
            total = self.attributes.get("total_tokens") or 0
            model = self.attributes.get("model", "")
            self.attributes["estimated_cost_usd"] = estimate_cost(model, total)
        choices = getattr(response, "choices", None)
        if choices:
            choice = choices[0]
            self.attributes["finish_reason"] = getattr(choice, "finish_reason", None)
            msg = getattr(choice, "message", None)
            if msg:
                content = getattr(msg, "content", "") or ""
                self.attributes["response_length"] = len(content)
        return self

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "kind": self.kind,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "start_time": self.start_time,
        }
        if self.error:
            d["error"] = self.error
        if self.attributes:
            d["attributes"] = self.attributes
        if self.events:
            d["events"] = [
                {"name": e.name, "timestamp": e.timestamp, **e.attributes}
                for e in self.events
            ]
        return d


# ---------------------------------------------------------------------------
# Collector protocol — pluggable backends
# ---------------------------------------------------------------------------

class SpanCollector(Protocol):
    """Interface for span consumers (loguru, OTLP, LangFuse, file, etc.)."""
    def on_span_end(self, span: Span) -> None: ...


class LoguruCollector:
    """Default collector: emits spans as structured loguru messages.

    Shows hierarchy via indentation (depth derived from parent chain)
    and formats each span kind with its most useful attributes.

    All messages carry the ``alpha.trace`` prefix for easy grep filtering.
    """

    _PREFIX = "alpha.trace"

    # Kind → (icon, key attributes to show)
    _KIND_FMT: dict[str, str] = {
        "llm":    "LLM",
        "breed":  "BREED",
        "eval":   "EVAL",
        "search": "SEARCH",
        "mcts":   "MCTS",
    }

    def __init__(self) -> None:
        self._depth: dict[str, int] = {}

    def on_span_end(self, span: Span) -> None:
        depth = 0
        if span.parent_id and span.parent_id in self._depth:
            depth = self._depth[span.parent_id] + 1
        self._depth[span.span_id] = depth
        if len(self._depth) > 500:
            for k in list(self._depth)[:200]:
                del self._depth[k]

        indent = "  " * depth
        a = span.attributes
        ms = f"{span.duration_ms:.0f}ms"
        tag = self._KIND_FMT.get(span.kind, span.kind.upper() if span.kind else "SPAN")

        # Build key=value detail string per kind
        parts: list[str] = [ms]
        if span.kind == "llm":
            tokens = a.get("total_tokens")
            if tokens:
                parts.append(f"tokens={tokens}")
            cost = a.get("estimated_cost_usd")
            if cost:
                parts.append(f"${cost:.4f}")
        elif span.kind == "breed":
            gen = a.get("generated", a.get("finalized"))
            if gen is not None:
                parts.append(f"generated={gen}")
        elif span.kind == "eval":
            count = a.get("count")
            if count is not None:
                parts.append(f"n={count}")
            best = a.get("best_fitness")
            if best is not None:
                parts.append(f"best={best:.4f}")
        elif span.kind == "search":
            for key in ("archive", "archive_size"):
                v = a.get(key)
                if v is not None:
                    parts.append(f"archive={v}")
                    break
            bf = a.get("best_fitness")
            if bf is not None:
                parts.append(f"best={bf}")
            rej = a.get("rejected", a.get("total_rejected"))
            if rej:
                parts.append(f"rejected={rej}")
        elif span.kind == "mcts":
            for key in ("iteration", "child_score", "zoo_size", "budget"):
                v = a.get(key)
                if v is not None:
                    parts.append(f"{key}={v}")

        detail = "  ".join(parts)
        line = f"{self._PREFIX} [{tag}] {indent}{span.operation}  {detail}"

        if span.status == "error":
            logger.warning(f"{line}  ERR: {span.error}")
        else:
            logger.info(line)


def _epoch_to_datetime(epoch: float | None) -> datetime | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class LangfuseCollector:
    """Sends spans to Langfuse with proper trace → span → generation hierarchy.

    Uses explicit ``id`` / ``parent_observation_id`` so that Langfuse
    reconstructs the full tree regardless of emission order (inner spans
    are emitted before their parents due to context-manager semantics).
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._client = None
        self._traces: dict[str, Any] = {}  # trace_id → Langfuse trace object
        if not enabled:
            return
        resolved_pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        resolved_sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        if not resolved_pk or not resolved_sk:
            self.enabled = False
            return
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=resolved_pk, secret_key=resolved_sk,
                host=host or os.getenv("LANGFUSE_HOST"),
            )
        except Exception as exc:
            logger.warning("langfuse init failed: {}", exc)
            self.enabled = False

    def on_span_end(self, span: Span) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._send(span)
        except Exception as exc:
            logger.debug("langfuse send failed: {}", exc)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

    def _get_trace(self, span: Span) -> Any:
        """Get or create a Langfuse trace for the given trace_id."""
        if span.trace_id not in self._traces:
            trace = self._client.trace(
                id=span.trace_id,
                name="alpha_search",
            )
            self._traces[span.trace_id] = trace
            if len(self._traces) > 200:
                del self._traces[next(iter(self._traces))]
        return self._traces[span.trace_id]

    def _send(self, span: Span) -> None:
        _NATIVE = {"model", "prompt_tokens", "completion_tokens", "total_tokens",
                    "estimated_cost_usd", "temperature", "input", "output"}
        metadata = {k: v for k, v in span.attributes.items() if k not in _NATIVE}
        level = "ERROR" if span.status == "error" else "DEFAULT"

        trace = self._get_trace(span)

        # When root span closes, update the trace name with the actual operation
        if span.parent_id is None:
            trace.update(name=span.operation, metadata=metadata)

        if span.kind == "llm":
            trace.generation(
                id=span.span_id,
                name=span.operation,
                parent_observation_id=span.parent_id,
                start_time=_epoch_to_datetime(span.start_time),
                end_time=_epoch_to_datetime(span.end_time),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                model=span.attributes.get("model"),
                model_parameters={"temperature": span.attributes.get("temperature")},
                usage={
                    "input": span.attributes.get("prompt_tokens") or 0,
                    "output": span.attributes.get("completion_tokens") or 0,
                    "total": span.attributes.get("total_tokens") or 0,
                },
                metadata=metadata,
                level=level,
                status_message=span.error,
            )
        else:
            name = span.operation if span.kind == "search" else f"{span.kind}.{span.operation}"
            trace.span(
                id=span.span_id,
                name=name,
                parent_observation_id=span.parent_id,
                start_time=_epoch_to_datetime(span.start_time),
                end_time=_epoch_to_datetime(span.end_time),
                metadata=metadata,
                level=level,
                status_message=span.error,
            )


class InMemoryCollector:
    """Collects spans in memory for testing / analysis.

    Spans are grouped by ``trace_id`` so callers can retrieve a single
    search run without noise from previous runs.
    """

    _MAX_TRACES = 20  # keep last N traces to bound memory

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._traces: dict[str, list[Span]] = {}  # trace_id → spans

    def on_span_end(self, span: Span) -> None:
        self.spans.append(span)
        self._traces.setdefault(span.trace_id, []).append(span)
        # Evict oldest traces if too many
        if len(self._traces) > self._MAX_TRACES:
            oldest = next(iter(self._traces))
            evicted = self._traces.pop(oldest)
            self.spans = [s for s in self.spans if s.trace_id != oldest]

    def clear(self) -> None:
        self.spans.clear()
        self._traces.clear()

    @property
    def trace_ids(self) -> list[str]:
        """All known trace IDs, oldest first."""
        return list(self._traces.keys())

    @property
    def latest_trace_id(self) -> str | None:
        """The most recent trace ID, or None."""
        return list(self._traces.keys())[-1] if self._traces else None

    def find(
        self,
        operation: str | None = None,
        kind: str | None = None,
        trace_id: str | None = None,
    ) -> list[Span]:
        if trace_id:
            result = self._traces.get(trace_id, [])
        else:
            result = self.spans
        if operation:
            result = [s for s in result if s.operation == operation]
        if kind:
            result = [s for s in result if s.kind == kind]
        return result

    def summary(self, trace_id: str | None = None) -> dict[str, Any]:
        """Aggregate statistics, optionally scoped to a single trace."""
        spans = self._traces.get(trace_id, self.spans) if trace_id else self.spans
        llm_spans = [s for s in spans if s.kind == "llm"]
        total_tokens = sum(s.attributes.get("total_tokens", 0) or 0 for s in llm_spans)
        total_cost = sum(s.attributes.get("estimated_cost_usd", 0) or 0 for s in llm_spans)
        total_latency = sum(s.duration_ms for s in llm_spans)
        errors = sum(1 for s in llm_spans if s.status == "error")
        return {
            "trace_id": trace_id or self.latest_trace_id,
            "llm_calls": len(llm_spans),
            "llm_errors": errors,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_ms": round(total_latency, 1),
            "avg_latency_ms": round(total_latency / len(llm_spans), 1) if llm_spans else 0,
            "total_spans": len(spans),
        }


# ---------------------------------------------------------------------------
# Tracer — span lifecycle management
# ---------------------------------------------------------------------------

_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)


class _SpanContext:
    """Context manager returned by ``tracer.start_span()``."""

    def __init__(self, span: Span, tracer: "Tracer") -> None:
        self._span = span
        self._tracer = tracer
        self._token = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._span.end_time = time.time()
        if exc_type is not None:
            self._span.set_error(exc_val or exc_type.__name__)
        self._tracer._emit(self._span)
        if self._token is not None:
            _current_span.reset(self._token)
        return None  # don't suppress exceptions


class Tracer:
    """Lightweight LLM chain tracer.

    Usage::

        tracer = Tracer()
        tracer.add_collector(InMemoryCollector())

        with tracer.start_span("genesis", kind="llm") as span:
            span.set("model", "gpt-4.1-mini")
            ...
    """

    def __init__(self) -> None:
        self._collectors: list[SpanCollector] = [LoguruCollector()]

    def add_collector(self, collector: SpanCollector) -> None:
        self._collectors.append(collector)

    def remove_collector(self, collector: SpanCollector) -> None:
        self._collectors = [c for c in self._collectors if c is not collector]

    def start_span(
        self,
        operation: str,
        kind: str = "internal",
        trace_id: str | None = None,
        **initial_attrs: Any,
    ) -> _SpanContext:
        parent = _current_span.get(None)
        resolved_trace = trace_id or (parent.trace_id if parent else _new_id())
        span = Span(
            trace_id=resolved_trace,
            span_id=_new_id(),
            parent_id=parent.span_id if parent else None,
            operation=operation,
            kind=kind,
            start_time=time.time(),
            attributes=dict(initial_attrs),
        )
        return _SpanContext(span, self)

    @property
    def current_span(self) -> Span | None:
        return _current_span.get(None)

    @property
    def current_trace_id(self) -> str | None:
        span = _current_span.get(None)
        return span.trace_id if span else None

    def flush(self) -> None:
        """Flush all collectors that support it (e.g. Langfuse)."""
        for collector in self._collectors:
            flush_fn = getattr(collector, "flush", None)
            if flush_fn:
                try:
                    flush_fn()
                except Exception:
                    pass

    def _emit(self, span: Span) -> None:
        for collector in self._collectors:
            try:
                collector.on_span_end(span)
            except Exception:
                pass  # never let collector errors break the pipeline


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tracer = Tracer()


def _auto_configure_langfuse() -> None:
    """Auto-register LangfuseCollector if LANGFUSE_SECRET_KEY is set."""
    # Ensure .env is loaded so Langfuse keys are available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if os.getenv("LANGFUSE_SECRET_KEY"):
        collector = LangfuseCollector()
        if collector.enabled:
            tracer.add_collector(collector)
            logger.info("alpha.tracing langfuse collector auto-registered host={}", os.getenv("LANGFUSE_HOST", "default"))


_auto_configure_langfuse()


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Approximate costs per 1M tokens (input+output blended).
# Update as needed; this is for order-of-magnitude tracking.
_COST_PER_1M_TOKENS: dict[str, float] = {
    "gpt-4.1-mini": 0.40,
    "gpt-4.1": 2.00,
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.0,
    "deepseek-v3": 0.27,
    "deepseek-v3.2": 0.27,
    "deepseek-ai/deepseek-v3.2": 0.27,
    "deepseek-chat": 0.27,
}


def estimate_cost(model: str, total_tokens: int) -> float | None:
    """Rough cost estimate in USD.  Returns None if model unknown."""
    key = model.lower().strip()
    rate = _COST_PER_1M_TOKENS.get(key)
    if rate is None:
        # Try partial match
        for k, v in _COST_PER_1M_TOKENS.items():
            if k in key or key in k:
                rate = v
                break
    if rate is None:
        return None
    return total_tokens * rate / 1_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def prompt_hash(text: str) -> str:
    """Short hash of prompt content for dedup detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
