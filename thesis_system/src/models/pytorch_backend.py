"""Single source of truth for the optional CPU PyTorch backend.

The Markov baseline has no neural dependency.  Keeping the guarded import in
this small module lets every neural component report the same availability
state without importing PyTorch independently. PyTorch is imported only when
neural availability or functionality is explicitly requested.
"""

from __future__ import annotations

from threading import Lock
from typing import Any


_LOAD_LOCK = Lock()
_LOAD_ATTEMPTED = False
_TORCH: Any = None
_NN: Any = None
_IMPORT_ERROR: Exception | None = None


def _load_pytorch() -> tuple[Any, Any]:
    """Try the optional import once and cache either outcome."""

    global _IMPORT_ERROR, _LOAD_ATTEMPTED, _NN, _TORCH
    if not _LOAD_ATTEMPTED:
        with _LOAD_LOCK:
            if not _LOAD_ATTEMPTED:
                try:
                    import torch as imported_torch
                    from torch import nn as imported_nn
                except Exception as exc:  # pragma: no cover - platform-specific.
                    _IMPORT_ERROR = exc
                else:
                    _TORCH = imported_torch
                    _NN = imported_nn
                finally:
                    _LOAD_ATTEMPTED = True
    return _TORCH, _NN


def pytorch_available() -> bool:
    """Return whether PyTorch imported successfully."""

    imported_torch, imported_nn = _load_pytorch()
    return imported_torch is not None and imported_nn is not None


def backend_error_message(algorithm: str = "GRU/LSTM") -> str | None:
    """Return one consistent, user-facing dependency message."""

    if pytorch_available():
        return None

    detail = str(_IMPORT_ERROR) if _IMPORT_ERROR is not None else "not installed"
    return (
        f"{algorithm} training requires PyTorch. "
        "Markov Chain/N-gram training can still run. "
        f"Backend detail: {detail}"
    )


def require_pytorch(algorithm: str = "GRU/LSTM") -> tuple[Any, Any]:
    """Return the imported ``torch`` and ``nn`` modules or raise clearly."""

    imported_torch, imported_nn = _load_pytorch()
    if imported_torch is None or imported_nn is None:
        raise RuntimeError(backend_error_message(algorithm))
    return imported_torch, imported_nn


def neural_backend_status() -> dict[str, object]:
    """Describe neural-model availability for the Streamlit interface."""

    imported_torch, _ = _load_pytorch()
    if imported_torch is not None:
        return {
            "available": True,
            "backend": "PyTorch",
            "device": "cpu",
            "version": str(imported_torch.__version__),
            "message": "PyTorch is available. GRU and LSTM training will run on CPU.",
        }

    return {
        "available": False,
        "backend": None,
        "device": "cpu",
        "version": None,
        "message": backend_error_message() or "PyTorch is unavailable.",
    }


__all__ = [
    "backend_error_message",
    "neural_backend_status",
    "pytorch_available",
    "require_pytorch",
]
