"""Trace-first scalar federated averaging helpers for qfl-mini."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalarFedAvgClient:
    """Client configuration for scalar FedAvg."""

    client_id: str
    target: float


class ScalarFedAvgCoordinator:
    """Trace-first scalar FedAvg coordinator.

    The coordinator maintains one scalar global parameter. Each client computes
    a local finite-difference update against its own target, then the
    coordinator aggregates local updated parameters using a mean.
    """

    def __init__(
        self,
        clients: list[ScalarFedAvgClient],
        backend: Any,
        initial_theta: float,
        learning_rate: float,
        epsilon: float = 1e-3,
    ) -> None:
        if not clients:
            raise ValueError("ScalarFedAvgCoordinator requires at least one client.")
        if not hasattr(backend, "run_expectation"):
            raise ValueError("backend must provide run_expectation(theta).")
        if not _is_number(initial_theta):
            raise ValueError("initial_theta must be numeric.")
        if not _is_number(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not _is_number(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        normalized_clients: list[ScalarFedAvgClient] = []
        for client in clients:
            client_id = getattr(client, "client_id", "")
            target = getattr(client, "target", None)
            if not isinstance(client_id, str) or not client_id.strip():
                raise ValueError("client_id must be a non-empty string.")
            if not _is_number(target):
                raise ValueError("target must be numeric.")
            normalized_clients.append(
                ScalarFedAvgClient(client_id=client_id.strip(), target=float(target))
            )

        self.clients = normalized_clients
        self.backend = backend
        self.initial_theta = float(initial_theta)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)

    def _evaluate_loss(self, theta: float, target: float) -> dict[str, float]:
        """Evaluate backend result and scalar loss for a theta/target pair."""
        result = float(self.backend.run_expectation(theta))
        loss = (result - target) ** 2
        return {
            "theta": float(theta),
            "target": float(target),
            "result": result,
            "loss": loss,
        }

    def _compute_client_update(
        self,
        client: ScalarFedAvgClient,
        global_theta: float,
    ) -> dict[str, Any]:
        """Compute one client's finite-difference local parameter update."""
        current = self._evaluate_loss(global_theta, client.target)
        plus = self._evaluate_loss(global_theta + self.epsilon, client.target)
        minus = self._evaluate_loss(global_theta - self.epsilon, client.target)

        gradient = (plus["loss"] - minus["loss"]) / (2 * self.epsilon)
        local_next_theta = global_theta - self.learning_rate * gradient

        return {
            "client_id": client.client_id,
            "target": client.target,
            "theta": float(global_theta),
            "result": current["result"],
            "loss": current["loss"],
            "loss_plus": plus["loss"],
            "loss_minus": minus["loss"],
            "gradient": gradient,
            "local_next_theta": local_next_theta,
        }

    def run_rounds(self, num_rounds: int) -> dict[str, Any]:
        """Run scalar FedAvg rounds and return a full trace dictionary."""
        if num_rounds < 1:
            raise ValueError("num_rounds must be at least 1.")

        global_theta = self.initial_theta
        rounds: list[dict[str, Any]] = []

        for round_number in range(1, num_rounds + 1):
            client_updates = [
                self._compute_client_update(client, global_theta)
                for client in self.clients
            ]
            aggregation_inputs = [
                {
                    "client_id": update["client_id"],
                    "local_next_theta": update["local_next_theta"],
                }
                for update in client_updates
            ]
            next_global_theta = (
                sum(item["local_next_theta"] for item in aggregation_inputs)
                / len(aggregation_inputs)
            )
            mean_local_loss = (
                sum(update["loss"] for update in client_updates)
                / len(client_updates)
            )

            rounds.append(
                {
                    "round": round_number,
                    "global_theta": global_theta,
                    "client_updates": client_updates,
                    "aggregation": {
                        "method": "mean",
                        "inputs": aggregation_inputs,
                        "next_global_theta": next_global_theta,
                    },
                    "mean_local_loss": mean_local_loss,
                }
            )
            global_theta = next_global_theta

        return {
            "algorithm": "scalar_fedavg",
            "num_clients": len(self.clients),
            "num_rounds": num_rounds,
            "initial_theta": self.initial_theta,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "aggregation": {"method": "mean"},
            "clients": [
                {"client_id": client.client_id, "target": client.target}
                for client in self.clients
            ],
            "rounds": rounds,
            "final_theta": global_theta,
            "final_mean_local_loss": rounds[-1]["mean_local_loss"],
        }


def _is_number(value: Any) -> bool:
    """Return True for int/float values except bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
