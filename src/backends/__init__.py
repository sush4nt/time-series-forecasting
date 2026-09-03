"""Backend registry — maps a model name to its backend implementation."""

from __future__ import annotations

from ..config import SUPPORTED_MODELS
from .base import Backend
from .classical import ClassicalBackend
from .gru import GRUBackend


def get_backend(model: str) -> Backend:
    if model in SUPPORTED_MODELS:
        return ClassicalBackend(model)
    if model == "gru":
        return GRUBackend()
    raise ValueError(
        f"Unknown model '{model}'. Choose one of {SUPPORTED_MODELS + ('gru',)}."
    )


__all__ = ["Backend", "get_backend"]
