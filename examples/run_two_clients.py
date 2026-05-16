"""Run a minimal federated quantum workload with two local quantum clients."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator
from qflmini.reporting import format_round_report


def main() -> None:
    """Execute one Phase 0 coordination round and print a report."""
    clients = [
        QuantumClient(client_id="client_1", theta=0.2),
        QuantumClient(client_id="client_2", theta=0.8),
    ]

    coordinator = Coordinator(clients)
    round_result = coordinator.run_round()
    print(format_round_report(round_result))


if __name__ == "__main__":
    main()
