"""Quantum client abstraction for local execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qflmini.backends import PennyLaneBackend, QuantumBackend


@dataclass(frozen=True)
class QuantumClient:
    """A local quantum execution node in a federated quantum workload.

    A quantum client owns a local parameter and delegates circuit execution
    to a backend. The default backend is PennyLaneBackend.
    """

    client_id: str
    theta: float
    backend: QuantumBackend = field(default_factory=PennyLaneBackend)

    def run(self) -> dict[str, Any]:
        """Execute the client's local quantum circuit.

        Returns:
            A dictionary containing the client identifier, local parameter, and
            circuit result.
        """
        result = self.backend.run_expectation(self.theta)
        return {
            "client_id": self.client_id,
            "theta": self.theta,
            "result": result,
        }
