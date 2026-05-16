from __future__ import annotations

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from qflmini.artifacts import artifact_path_for_run, save_json_artifact
from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator
from qflmini.metadata import (
    build_run_artifact,
    collect_environment_metadata,
    generate_run_id,
)
from qflmini.optimization import ParameterUpdateCoordinator


def test_quantum_client_run_returns_expected_keys() -> None:
    client = QuantumClient(client_id="client_1", theta=0.2)

    result = client.run()

    assert set(result) == {"client_id", "theta", "result"}
    assert result["client_id"] == "client_1"
    assert result["theta"] == 0.2
    assert isinstance(result["result"], float)


def test_coordinator_run_round_returns_aggregated_result() -> None:
    coordinator = Coordinator(
        [
            QuantumClient(client_id="client_1", theta=0.2),
            QuantumClient(client_id="client_2", theta=0.8),
        ]
    )

    result = coordinator.run_round()

    assert result["num_clients"] == 2
    assert len(result["client_results"]) == 2
    assert "aggregated_result" in result
    assert result["aggregated_result"] == pytest.approx(
        sum(client_result["result"] for client_result in result["client_results"]) / 2
    )


def test_coordinator_requires_clients() -> None:
    with pytest.raises(ValueError, match="at least one quantum client"):
        Coordinator([])


def test_coordinator_run_rounds_returns_three_rounds() -> None:
    coordinator = Coordinator(
        [
            QuantumClient(client_id="client_1", theta=0.2),
            QuantumClient(client_id="client_2", theta=0.8),
        ]
    )

    result = coordinator.run_rounds(3)

    assert result["num_clients"] == 2
    assert result["num_rounds"] == 3
    assert len(result["rounds"]) == 3
    assert [round_result["round"] for round_result in result["rounds"]] == [1, 2, 3]


def test_run_rounds_requires_positive_count() -> None:
    coordinator = Coordinator([QuantumClient(client_id="client_1", theta=0.2)])

    with pytest.raises(ValueError, match="at least 1"):
        coordinator.run_rounds(0)


def test_save_json_artifact_writes_file(tmp_path) -> None:
    artifact_path = save_json_artifact(
        {"num_clients": 1, "rounds": [{"round": 1}]},
        tmp_path / "nested" / "artifact.json",
    )

    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {
        "num_clients": 1,
        "rounds": [{"round": 1}],
    }


def test_parameter_update_coordinator_requires_clients() -> None:
    with pytest.raises(ValueError, match="at least one quantum client"):
        ParameterUpdateCoordinator([], initial_theta=0.5, learning_rate=0.1)


def test_parameter_update_coordinator_requires_positive_learning_rate() -> None:
    clients = [QuantumClient(client_id="client_1", theta=0.0)]

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        ParameterUpdateCoordinator(clients, initial_theta=0.5, learning_rate=0.0)


def test_parameter_update_run_updates_returns_three_rounds() -> None:
    coordinator = ParameterUpdateCoordinator(
        [
            QuantumClient(client_id="client_1", theta=0.0),
            QuantumClient(client_id="client_2", theta=0.0),
        ],
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_updates(3)

    assert result["num_clients"] == 2
    assert result["num_rounds"] == 3
    assert result["initial_theta"] == 0.5
    assert result["learning_rate"] == 0.1
    assert result["target"] == 0.0
    assert "final_theta" in result
    assert "rounds" in result
    assert len(result["rounds"]) == 3
    assert [round_result["round"] for round_result in result["rounds"]] == [1, 2, 3]
    assert all("target" in round_result for round_result in result["rounds"])
    assert all("loss" in round_result for round_result in result["rounds"])


def test_parameter_update_run_updates_requires_positive_count() -> None:
    coordinator = ParameterUpdateCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
    )

    with pytest.raises(ValueError, match="at least 1"):
        coordinator.run_updates(0)


def test_save_json_artifact_writes_parameter_update_artifact(tmp_path) -> None:
    coordinator = ParameterUpdateCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
    )
    update_result = coordinator.run_updates(2)

    artifact_path = save_json_artifact(
        update_result,
        tmp_path / "parameter_update.json",
    )

    saved_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved_data["initial_theta"] == 0.5
    assert saved_data["learning_rate"] == 0.1
    assert saved_data["target"] == 0.0
    assert len(saved_data["rounds"]) == 2


def test_collect_environment_metadata_returns_expected_keys() -> None:
    metadata = collect_environment_metadata()

    assert set(metadata) == {
        "python_version",
        "platform",
        "system",
        "machine",
        "pennylane_version",
    }


def test_build_run_artifact_returns_metadata_wrapper() -> None:
    artifact = build_run_artifact("example_name", {"ok": True})

    assert artifact["project"] == "qfl-mini"
    assert artifact["artifact_version"] == "0.1"
    assert artifact["run_id"].startswith("example_name_")
    assert artifact["example"] == "example_name"
    assert "created_at" in artifact
    assert "environment" in artifact
    assert artifact["run"] == {"ok": True}


def test_build_run_artifact_requires_example_name() -> None:
    with pytest.raises(ValueError, match="example_name must not be empty"):
        build_run_artifact("", {"ok": True})


def test_build_run_artifact_requires_run_result_dict() -> None:
    with pytest.raises(TypeError, match="run_result must be a dictionary"):
        build_run_artifact("example_name", "not a dict")


def test_save_json_artifact_writes_metadata_artifact(tmp_path) -> None:
    artifact = build_run_artifact("example_name", {"ok": True})

    artifact_path = save_json_artifact(artifact, tmp_path / "artifact.json")

    saved_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved_data["project"] == "qfl-mini"
    assert saved_data["run"] == {"ok": True}


def test_generate_run_id_uses_example_name_and_timestamp() -> None:
    run_id = generate_run_id("run_parameter_update")

    assert run_id.startswith("run_parameter_update_")
    assert run_id.endswith("Z")
    assert re.match(r"^run_parameter_update_\d{8}T\d{6}Z$", run_id)


def test_generate_run_id_requires_example_name() -> None:
    with pytest.raises(ValueError, match="example_name must not be empty"):
        generate_run_id("")


def test_artifact_path_for_run_uses_default_runs_directory() -> None:
    assert artifact_path_for_run("abc123") == Path("runs") / "abc123.json"


def test_artifact_path_for_run_accepts_output_dir(tmp_path) -> None:
    assert artifact_path_for_run("abc123", output_dir=tmp_path) == (
        tmp_path / "abc123.json"
    )


def test_artifact_path_for_run_requires_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must not be empty"):
        artifact_path_for_run("")


def test_save_json_artifact_with_run_id_matches_filename(tmp_path) -> None:
    artifact = build_run_artifact("example_name", {"ok": True})
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)

    saved_path = save_json_artifact(artifact, artifact_path)

    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["run_id"] == saved_path.stem


def test_parameter_update_loss_matches_target_objective() -> None:
    coordinator = ParameterUpdateCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
        target=0.0,
    )

    result = coordinator.run_updates(1)
    round_result = result["rounds"][0]
    expected_loss = (
        round_result["aggregated_result"] - round_result["target"]
    ) ** 2

    assert round_result["loss"] == pytest.approx(expected_loss)


def test_parameter_update_accepts_nonzero_target() -> None:
    coordinator = ParameterUpdateCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
        target=0.5,
    )

    result = coordinator.run_updates(1)

    assert result["target"] == 0.5
    assert result["rounds"][0]["target"] == 0.5
