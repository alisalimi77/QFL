"""Compare clean PennyLane execution with deterministic noisy backend execution."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.backends import NoisyBackend, PennyLaneBackend, get_backend_metadata
from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator
from qflmini.metadata import build_run_artifact
from qflmini.reporting import format_clean_vs_noisy_backend_report


def main() -> None:
    """Run one coordination round with a clean and a noisy backend and save an artifact."""
    clean_backend = PennyLaneBackend()
    noisy_backend = NoisyBackend(
        base_backend=PennyLaneBackend(),
        noise=0.05,
        seed=42,
    )

    clean_clients = [
        QuantumClient(client_id="client_1", theta=0.2, backend=clean_backend),
        QuantumClient(client_id="client_2", theta=0.8, backend=clean_backend),
    ]
    noisy_clients = [
        QuantumClient(client_id="client_1", theta=0.2, backend=noisy_backend),
        QuantumClient(client_id="client_2", theta=0.8, backend=noisy_backend),
    ]

    clean_result = Coordinator(clean_clients).run_round()
    noisy_result = Coordinator(noisy_clients).run_round()
    difference = noisy_result["aggregated_result"] - clean_result["aggregated_result"]

    run_result = {
        "clean": {
            "backend": get_backend_metadata(clean_backend),
            "result": clean_result,
        },
        "noisy": {
            "backend": get_backend_metadata(noisy_backend),
            "result": noisy_result,
        },
        "difference": difference,
    }

    print(format_clean_vs_noisy_backend_report(run_result))

    artifact = build_run_artifact(
        example_name="run_clean_vs_noisy_backend",
        run_result=run_result,
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
