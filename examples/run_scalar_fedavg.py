"""Run a transparent scalar FedAvg demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.backends import PennyLaneBackend, get_backend_metadata
from qflmini.federated import ScalarFedAvgClient, ScalarFedAvgCoordinator
from qflmini.metadata import build_run_artifact
from qflmini.reporting import format_scalar_fedavg_report


def main() -> None:
    """Run the scalar FedAvg demo and save a JSON artifact."""
    backend = PennyLaneBackend()
    clients = [
        ScalarFedAvgClient(client_id="client_1", target=0.0),
        ScalarFedAvgClient(client_id="client_2", target=0.5),
    ]
    coordinator = ScalarFedAvgCoordinator(
        clients=clients,
        backend=backend,
        initial_theta=0.5,
        learning_rate=0.1,
        epsilon=1e-3,
    )

    result = coordinator.run_rounds(3)
    print(format_scalar_fedavg_report(result))

    artifact = build_run_artifact(
        example_name="run_scalar_fedavg",
        run_result={
            "backend": get_backend_metadata(backend),
            "result": result,
        },
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path}")


if __name__ == "__main__":
    main()
