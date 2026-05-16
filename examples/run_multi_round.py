"""Run repeated coordination rounds and save a JSON artifact."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import save_json_artifact
from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator
from qflmini.reporting import format_multi_round_report


def main() -> None:
    """Execute three coordination rounds and save the run artifact."""
    clients = [
        QuantumClient(client_id="client_1", theta=0.2),
        QuantumClient(client_id="client_2", theta=0.8),
    ]

    coordinator = Coordinator(clients)
    run_result = coordinator.run_rounds(3)
    print(format_multi_round_report(run_result))

    artifact_path = save_json_artifact(run_result, "runs/demo_multi_round.json")
    print(f"Saved artifact: {artifact_path.as_posix()}")


if __name__ == "__main__":
    main()
