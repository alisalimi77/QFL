"""Run a minimal repeated parameter update demo and save a JSON artifact."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.client import QuantumClient
from qflmini.metadata import build_run_artifact
from qflmini.optimization import ParameterUpdateCoordinator
from qflmini.reporting import format_parameter_update_report


def main() -> None:
    """Execute three parameter update rounds and save the run artifact."""
    clients = [
        QuantumClient(client_id="client_1", theta=0.0),
        QuantumClient(client_id="client_2", theta=0.0),
    ]

    coordinator = ParameterUpdateCoordinator(
        clients=clients,
        initial_theta=0.5,
        learning_rate=0.1,
    )
    update_result = coordinator.run_updates(3)
    print(format_parameter_update_report(update_result))

    artifact = build_run_artifact(
        example_name="run_parameter_update",
        run_result=update_result,
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
