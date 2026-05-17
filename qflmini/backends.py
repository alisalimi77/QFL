"""Minimal backend abstraction for quantum circuit execution."""

from __future__ import annotations

from typing import Any, Protocol

from qflmini.circuits import run_single_qubit_expectation


class QuantumBackend(Protocol):
    """Protocol for quantum execution backends."""

    def run_expectation(self, theta: float) -> float:
        """Run the backend's expectation-value circuit for a scalar theta."""
        ...


class PennyLaneBackend:
    """PennyLane-backed quantum execution backend."""

    name = "pennylane"

    def run_expectation(self, theta: float) -> float:
        return run_single_qubit_expectation(theta)


class ConstantBackend:
    """Deterministic backend that always returns a fixed value.

    Intended for tests and demos only. Not a quantum simulator.
    Ignores theta intentionally.
    """

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def run_expectation(self, theta: float) -> float:
        return self.value


def get_backend_metadata(backend: Any) -> dict[str, str]:
    """Return a small metadata dict describing a backend instance.

    Args:
        backend: Any object satisfying the QuantumBackend protocol.

    Returns:
        A dict with keys "name" and "class". "name" is backend.name if it
        exists and is a non-empty string, otherwise the class name. "class"
        is always the class name.
    """
    class_name = backend.__class__.__name__
    raw_name = getattr(backend, "name", None)
    name = raw_name if isinstance(raw_name, str) and raw_name else class_name
    return {"name": name, "class": class_name}
