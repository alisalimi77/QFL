from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from qflmini.artifacts import save_json_artifact
from qflmini.client import QuantumClient
from qflmini.coordinator import Coordinator


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
