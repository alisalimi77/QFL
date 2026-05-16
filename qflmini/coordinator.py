"""Classical coordinator for federated quantum workload execution."""

from __future__ import annotations

from typing import Any

from qflmini.client import QuantumClient


class Coordinator:
    """Classical coordination layer for local quantum clients.

    The coordinator collects local execution results from quantum clients and
    applies simple mean aggregation.
    """

    def __init__(self, clients: list[QuantumClient]) -> None:
        if not clients:
            raise ValueError("Coordinator requires at least one quantum client.")
        self.clients = clients

    def run_round(self) -> dict[str, Any]:
        """Run one local execution round and aggregate client results.

        Returns:
            A dictionary containing the number of clients, individual client
            results, and the mean aggregated result.
        """
        client_results = [client.run() for client in self.clients]
        result_values = [client_result["result"] for client_result in client_results]
        aggregated_result = sum(result_values) / len(result_values)

        return {
            "num_clients": len(self.clients),
            "client_results": client_results,
            "aggregated_result": aggregated_result,
        }

    def run_rounds(self, num_rounds: int) -> dict[str, Any]:
        """Run multiple deterministic local execution rounds.

        Args:
            num_rounds: Number of coordination rounds to execute.

        Returns:
            A dictionary containing the number of clients, number of rounds,
            and per-round results.

        Raises:
            ValueError: If ``num_rounds`` is less than 1.
        """
        if num_rounds < 1:
            raise ValueError("num_rounds must be at least 1.")

        rounds = []
        for round_index in range(1, num_rounds + 1):
            round_result = self.run_round()
            rounds.append(
                {
                    "round": round_index,
                    "client_results": round_result["client_results"],
                    "aggregated_result": round_result["aggregated_result"],
                }
            )

        return {
            "num_clients": len(self.clients),
            "num_rounds": num_rounds,
            "rounds": rounds,
        }
