"""Run a client-specific objective evaluation demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.client import QuantumClient
from qflmini.metadata import build_run_artifact
from qflmini.objectives import ClientObjective, evaluate_client_objectives
from qflmini.reporting import format_client_objectives_report


def main() -> None:
    """Run the client-specific objective demo and save a JSON artifact."""
    client_1 = QuantumClient(client_id="client_1", theta=0.2)
    client_2 = QuantumClient(client_id="client_2", theta=0.8)

    result = evaluate_client_objectives(
        [
            ClientObjective(client=client_1, target=0.0),
            ClientObjective(client=client_2, target=0.5),
        ]
    )

    print(format_client_objectives_report(result))

    artifact = build_run_artifact(
        example_name="run_client_objectives",
        run_result=result,
    )
    artifact_path = artifact_path_for_run(artifact["run_id"])
    saved_path = save_json_artifact(artifact, artifact_path)
    print(f"Saved artifact: {saved_path}")


if __name__ == "__main__":
    main()
