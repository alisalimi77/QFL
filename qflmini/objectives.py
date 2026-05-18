"""Client-specific objective evaluation helpers for qfl-mini."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qflmini.client import QuantumClient


@dataclass(frozen=True)
class ClientObjective:
    """Local objective context for a quantum client."""

    client: QuantumClient
    target: float


def evaluate_client_objective(objective: ClientObjective) -> dict[str, Any]:
    """Evaluate one client's local objective.

    Args:
        objective: Client-specific objective context.

    Returns:
        A dictionary containing client output, local target, and local loss.
    """
    result = objective.client.run()
    local_loss = (result["result"] - objective.target) ** 2

    return {
        "client_id": result["client_id"],
        "theta": result["theta"],
        "target": objective.target,
        "result": result["result"],
        "loss": local_loss,
    }


def evaluate_client_objectives(objectives: list[ClientObjective]) -> dict[str, Any]:
    """Evaluate multiple client-specific objectives.

    Args:
        objectives: Client-specific objective contexts.

    Returns:
        A dictionary containing per-client objective results, mean aggregation,
        and mean local loss.

    Raises:
        ValueError: If objectives is empty.
    """
    if not objectives:
        raise ValueError("objectives must not be empty.")

    client_objectives = [
        evaluate_client_objective(objective) for objective in objectives
    ]
    aggregated_result = sum(item["result"] for item in client_objectives) / len(
        client_objectives
    )
    mean_local_loss = sum(item["loss"] for item in client_objectives) / len(
        client_objectives
    )

    return {
        "num_clients": len(client_objectives),
        "client_objectives": client_objectives,
        "aggregated_result": aggregated_result,
        "mean_local_loss": mean_local_loss,
    }
