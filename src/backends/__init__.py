"""Backend registry — maps a model name to its backend implementation."""

from __future__ import annotations

from ..config import SUPPORTED_MODELS
from .base import Backend
from .classical import ClassicalBackend


def get_backend(model: str) -> Backend:
    if model in SUPPORTED_MODELS:
        return ClassicalBackend(model)
    if model == "gru":
        # Lazy import: torch is only loaded when the GRU backend is actually
        # requested. Importing torch at module level causes a threading-library
        # conflict (PyTorch BLAS vs. LightGBM's libomp) that segfaults during
        # LightGBM training even when GRU is never used.
        from .gru import GRUBackend  # noqa: PLC0415
        return GRUBackend()
    raise ValueError(
        f"Unknown model '{model}'. Choose one of {SUPPORTED_MODELS + ('gru',)}."
    )


__all__ = ["Backend", "get_backend"]
