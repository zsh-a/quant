"""Shared fixtures for provider integration tests."""

from __future__ import annotations

from pydantic import BaseModel


class HelloOut(BaseModel):
    """Minimal schema used for round-trip structured-output tests."""

    greeting: str
