"""Minimal backend abstraction for quantum circuit execution."""

from __future__ import annotations

from typing import Protocol

from qflmini.circuits import run_single_qubit_expectation


class QuantumBackend(Protocol):
    """Protocol for quantum execution backends."""

    def run_expectation(self, theta: float) -> float:
        """Run the backend's expectation-value circuit for a scalar theta."""
        ...


class PennyLaneBackend:
    """PennyLane-backed quantum execution backend."""

    def run_expectation(self, theta: float) -> float:
        return run_single_qubit_expectation(theta)
