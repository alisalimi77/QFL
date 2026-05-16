"""Run a supported experiment from a JSON manifest and save a JSON artifact."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.client import QuantumClient
from qflmini.manifest import load_gradient_update_manifest
from qflmini.metadata import build_run_artifact
from qflmini.optimization import FiniteDifferenceGradientCoordinator
from qflmini.reporting import format_gradient_update_report


def main() -> None:
    """Load a JSON manifest, run the specified experiment, and save an artifact."""
    if len(sys.argv) < 2:
        print("Usage: python examples/run_from_manifest.py <manifest.json>")
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    config = load_gradient_update_manifest(manifest_path)

    clients = [
        QuantumClient(client_id=f"client_{index + 1}", theta=0.0)
        for index in range(config["num_clients"])
    ]

    coordinator = FiniteDifferenceGradientCoordinator(
        clients=clients,
        initial_theta=config["initial_theta"],
        learning_rate=config["learning_rate"],
        target=config["target"],
        epsilon=config["epsilon"],
    )
    update_result = coordinator.run_updates(config["num_rounds"])
    print(format_gradient_update_report(update_result))

    artifact = build_run_artifact(
        example_name="run_from_manifest_gradient_update",
        run_result={
            "manifest": config,
            "result": update_result,
        },
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
