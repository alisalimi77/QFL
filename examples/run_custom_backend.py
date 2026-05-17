"""Demonstrate backend injection with a deterministic constant backend."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.backends import ConstantBackend
from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator
from qflmini.reporting import format_custom_backend_report


def main() -> None:
    """Run one coordination round using clients with injected constant backends."""
    clients = [
        QuantumClient(client_id="client_1", theta=0.0, backend=ConstantBackend(0.2)),
        QuantumClient(client_id="client_2", theta=0.0, backend=ConstantBackend(0.6)),
    ]
    round_result = Coordinator(clients).run_round()
    print(format_custom_backend_report(round_result))


if __name__ == "__main__":
    main()
