"""Minimal parameter update loop for qfl-mini."""

from __future__ import annotations

from typing import Any

from qflmini.client import QuantumClient


class ParameterUpdateCoordinator:
    """Minimal coordinator for repeated federated quantum parameter updates.

    This is a deterministic Phase 1 seed. It demonstrates how a classical
    coordinator can update one shared scalar parameter across repeated local
    execution rounds.
    """

    def __init__(
        self,
        clients: list[QuantumClient],
        initial_theta: float,
        learning_rate: float,
        target: float = 0.0,
    ) -> None:
        if not clients:
            raise ValueError(
                "ParameterUpdateCoordinator requires at least one quantum client."
            )
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.clients = clients
        self.initial_theta = initial_theta
        self.learning_rate = learning_rate
        self.target = target

    def run_updates(self, num_rounds: int) -> dict[str, Any]:
        """Run repeated federated quantum execution with scalar updates.

        Args:
            num_rounds: Number of parameter update rounds to execute.

        Returns:
            A dictionary containing per-round execution results and the final
            updated parameter.

        Raises:
            ValueError: If ``num_rounds`` is less than 1.
        """
        if num_rounds < 1:
            raise ValueError("num_rounds must be at least 1.")

        theta = self.initial_theta
        rounds = []

        for round_index in range(1, num_rounds + 1):
            round_clients = [
                QuantumClient(client_id=client.client_id, theta=theta)
                for client in self.clients
            ]
            client_results = [client.run() for client in round_clients]
            result_values = [client_result["result"] for client_result in client_results]
            aggregated_result = sum(result_values) / len(result_values)
            loss = (aggregated_result - self.target) ** 2
            next_theta = theta - self.learning_rate * aggregated_result

            rounds.append(
                {
                    "round": round_index,
                    "theta": theta,
                    "client_results": client_results,
                    "aggregated_result": aggregated_result,
                    "target": self.target,
                    "loss": loss,
                    "next_theta": next_theta,
                }
            )

            theta = next_theta

        return {
            "num_clients": len(self.clients),
            "num_rounds": num_rounds,
            "initial_theta": self.initial_theta,
            "learning_rate": self.learning_rate,
            "target": self.target,
            "final_theta": theta,
            "rounds": rounds,
        }
