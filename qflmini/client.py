"""Quantum client abstraction for local execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qflmini.circuits import run_single_qubit_expectation


@dataclass(frozen=True)
class QuantumClient:
    """A local quantum execution node in a federated quantum workload.

    For Phase 0, a quantum client is a local Python object. It owns a local
    parameter and executes a deterministic PennyLane circuit.
    """

    client_id: str
    theta: float

    def run(self) -> dict[str, Any]:
        """Execute the client's local quantum circuit.

        Returns:
            A dictionary containing the client identifier, local parameter, and
            circuit result.
        """
        result = run_single_qubit_expectation(self.theta)
        return {
            "client_id": self.client_id,
            "theta": self.theta,
            "result": result,
        }
