"""Run a supported experiment from a JSON manifest and save a JSON artifact."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.backends import get_backend_metadata
from qflmini.client import QuantumClient
from qflmini.federated import ScalarFedAvgClient, ScalarFedAvgCoordinator
from qflmini.manifest import build_backend_from_config, load_manifest
from qflmini.metadata import build_run_artifact
from qflmini.objectives import ClientObjective, evaluate_client_objectives
from qflmini.optimization import FiniteDifferenceGradientCoordinator
from qflmini.reporting import (
    format_client_objectives_report,
    format_gradient_update_report,
    format_scalar_fedavg_report,
)


def main() -> None:
    """Load a JSON manifest, run the specified experiment, and save an artifact."""
    if len(sys.argv) < 2:
        print("Usage: python examples/run_from_manifest.py <manifest.json>")
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    config = load_manifest(manifest_path)
    backend = build_backend_from_config(config["backend"])

    if config["experiment"] == "client_objectives":
        clients = [
            QuantumClient(
                client_id=client_config["client_id"],
                theta=client_config["theta"],
                backend=backend,
            )
            for client_config in config["clients"]
        ]
        objectives = [
            ClientObjective(client=client, target=client_config["target"])
            for client, client_config in zip(clients, config["clients"])
        ]
        result = evaluate_client_objectives(objectives)
        print(format_client_objectives_report(result))

        artifact = build_run_artifact(
            example_name="run_from_manifest_client_objectives",
            run_result={
                "manifest_path": manifest_path.as_posix(),
                "manifest": config,
                "backend": get_backend_metadata(backend),
                "result": result,
            },
        )
        artifact_path = artifact_path_for_run(artifact["run_id"])
        saved_path = save_json_artifact(artifact, artifact_path)
        print(f"Saved artifact: {saved_path.as_posix()}")
        return

    if config["experiment"] == "scalar_fedavg":
        clients = [
            ScalarFedAvgClient(
                client_id=client_config["client_id"],
                target=client_config["target"],
            )
            for client_config in config["clients"]
        ]
        coordinator = ScalarFedAvgCoordinator(
            clients=clients,
            backend=backend,
            initial_theta=config["initial_theta"],
            learning_rate=config["learning_rate"],
            epsilon=config["epsilon"],
        )
        result = coordinator.run_rounds(config["num_rounds"])
        print(format_scalar_fedavg_report(result))

        artifact = build_run_artifact(
            example_name="run_from_manifest_scalar_fedavg",
            run_result={
                "manifest_path": manifest_path.as_posix(),
                "manifest": config,
                "backend": get_backend_metadata(backend),
                "result": result,
            },
        )
        artifact_path = artifact_path_for_run(artifact["run_id"])
        saved_path = save_json_artifact(artifact, artifact_path)
        print(f"Saved artifact: {saved_path.as_posix()}")
        return

    clients = [
        QuantumClient(client_id=f"client_{index + 1}", theta=0.0, backend=backend)
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
            "manifest_path": manifest_path.as_posix(),
            "manifest": config,
            "backend": get_backend_metadata(backend),
            "result": update_result,
        },
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
