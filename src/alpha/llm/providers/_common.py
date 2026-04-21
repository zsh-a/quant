"""Shared helpers for provider implementations."""

from __future__ import annotations

import base64
import time
from typing import Any

from pydantic import BaseModel


def pydantic_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return the plain JSON schema for a Pydantic v2 model."""
    return schema.model_json_schema()


def strict_openai_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema shape accepted by OpenAI strict mode.

    OpenAI's `response_format={"type": "json_schema", "strict": true}` requires
    every object to declare `additionalProperties: false` and list every property
    in `required`. Pydantic v2 emits `required` only for non-default fields, so
    we tighten the schema recursively (including `$defs`).
    """
    js = schema.model_json_schema()
    _tighten_objects(js)
    for defn in (js.get("$defs") or {}).values():
        _tighten_objects(defn)
    return js


def _tighten_objects(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
    for value in node.values():
        if isinstance(value, dict):
            _tighten_objects(value)
        elif isinstance(value, list):
            for item in value:
                _tighten_objects(item)


def encode_image_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Uniform attribute / mapping accessor: works on dicts and SDK response objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def raw_to_dict(raw: Any) -> dict[str, Any]:
    """Best-effort conversion of a vendor SDK response to a plain dict."""
    if isinstance(raw, dict):
        return dict(raw)
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(raw, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                continue
            if isinstance(dumped, dict):
                return dumped
    return {"repr": repr(raw)}


class Timer:
    """Context manager that captures wall-clock elapsed milliseconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._end: float | None = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0
