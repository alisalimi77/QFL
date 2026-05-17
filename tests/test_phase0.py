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
from qflmini.comparison import (
    format_artifact_comparison,
    load_artifact,
    summarize_artifact,
    summarize_artifacts,
)
from qflmini.manifest import (
    load_gradient_update_manifest,
    load_json_manifest,
    validate_gradient_update_manifest,
)
from qflmini.optimization import FiniteDifferenceGradientCoordinator, ParameterUpdateCoordinator


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


# --- FiniteDifferenceGradientCoordinator tests ---


def test_gradient_coordinator_requires_clients() -> None:
    with pytest.raises(ValueError, match="at least one quantum client"):
        FiniteDifferenceGradientCoordinator([], initial_theta=0.5, learning_rate=0.1)


def test_gradient_coordinator_requires_positive_learning_rate() -> None:
    clients = [QuantumClient(client_id="client_1", theta=0.0)]

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        FiniteDifferenceGradientCoordinator(clients, initial_theta=0.5, learning_rate=0.0)


def test_gradient_coordinator_requires_positive_epsilon() -> None:
    clients = [QuantumClient(client_id="client_1", theta=0.0)]

    with pytest.raises(ValueError, match="epsilon must be positive"):
        FiniteDifferenceGradientCoordinator(clients, initial_theta=0.5, learning_rate=0.1, epsilon=0.0)


def test_gradient_coordinator_run_updates_returns_three_rounds() -> None:
    coordinator = FiniteDifferenceGradientCoordinator(
        [
            QuantumClient(client_id="client_1", theta=0.0),
            QuantumClient(client_id="client_2", theta=0.0),
        ],
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_updates(3)

    assert result["num_rounds"] == 3
    assert "epsilon" in result
    assert "final_theta" in result
    assert len(result["rounds"]) == 3


def test_gradient_coordinator_rounds_contain_expected_keys() -> None:
    coordinator = FiniteDifferenceGradientCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_updates(1)
    round_result = result["rounds"][0]

    for key in ("theta", "aggregated_result", "target", "loss", "loss_plus", "loss_minus", "gradient", "next_theta"):
        assert key in round_result, f"Missing key: {key}"


def test_gradient_coordinator_gradient_is_consistent() -> None:
    coordinator = FiniteDifferenceGradientCoordinator(
        [QuantumClient(client_id="client_1", theta=0.0)],
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_updates(1)
    round_result = result["rounds"][0]
    expected_gradient = (round_result["loss_plus"] - round_result["loss_minus"]) / (2 * result["epsilon"])

    assert round_result["gradient"] == pytest.approx(expected_gradient)


def test_gradient_coordinator_artifact_save(tmp_path) -> None:
    coordinator = FiniteDifferenceGradientCoordinator(
        [
            QuantumClient(client_id="client_1", theta=0.0),
            QuantumClient(client_id="client_2", theta=0.0),
        ],
        initial_theta=0.5,
        learning_rate=0.1,
    )
    update_result = coordinator.run_updates(2)

    artifact = build_run_artifact(
        example_name="run_gradient_update",
        run_result=update_result,
    )
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)
    saved_path = save_json_artifact(artifact, artifact_path)

    assert saved_path.exists()
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "run_id" in saved_data
    assert "gradient" in saved_data["run"]["rounds"][0]


# --- manifest tests ---

_VALID_MANIFEST = {
    "manifest_version": "0.1",
    "name": "test-manifest",
    "description": "A test manifest.",
    "experiment": "gradient_update",
    "num_clients": 2,
    "num_rounds": 3,
    "initial_theta": 0.5,
    "learning_rate": 0.1,
    "target": 0.0,
    "epsilon": 0.001,
}


def test_load_json_manifest_returns_dict(tmp_path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text('{"experiment": "gradient_update"}', encoding="utf-8")

    result = load_json_manifest(manifest_file)

    assert isinstance(result, dict)
    assert result["experiment"] == "gradient_update"


def test_load_json_manifest_rejects_non_object(tmp_path) -> None:
    manifest_file = tmp_path / "bad.json"
    manifest_file.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_manifest(manifest_file)


def test_validate_gradient_update_manifest_returns_normalized_values() -> None:
    config = validate_gradient_update_manifest(_VALID_MANIFEST)

    assert config["manifest_version"] == "0.1"
    assert config["name"] == "test-manifest"
    assert config["description"] == "A test manifest."
    assert config["experiment"] == "gradient_update"
    assert isinstance(config["num_clients"], int)
    assert isinstance(config["num_rounds"], int)
    assert isinstance(config["initial_theta"], float)
    assert isinstance(config["learning_rate"], float)
    assert isinstance(config["target"], float)
    assert isinstance(config["epsilon"], float)
    assert config["num_clients"] == 2
    assert config["num_rounds"] == 3
    assert config["initial_theta"] == 0.5
    assert config["learning_rate"] == 0.1
    assert config["target"] == 0.0
    assert config["epsilon"] == 0.001


def test_validate_gradient_update_manifest_missing_field_raises() -> None:
    bad = dict(_VALID_MANIFEST)
    del bad["num_rounds"]

    with pytest.raises(ValueError, match="num_rounds"):
        validate_gradient_update_manifest(bad)


def test_validate_gradient_update_manifest_wrong_experiment_raises() -> None:
    bad = dict(_VALID_MANIFEST, experiment="parameter_update")

    with pytest.raises(ValueError, match="Unsupported experiment type"):
        validate_gradient_update_manifest(bad)


def test_validate_gradient_update_manifest_num_clients_below_one_raises() -> None:
    bad = dict(_VALID_MANIFEST, num_clients=0)

    with pytest.raises(ValueError, match="num_clients"):
        validate_gradient_update_manifest(bad)


def test_validate_gradient_update_manifest_num_rounds_below_one_raises() -> None:
    bad = dict(_VALID_MANIFEST, num_rounds=0)

    with pytest.raises(ValueError, match="num_rounds"):
        validate_gradient_update_manifest(bad)


def test_validate_gradient_update_manifest_nonpositive_learning_rate_raises() -> None:
    bad = dict(_VALID_MANIFEST, learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate"):
        validate_gradient_update_manifest(bad)


def test_validate_gradient_update_manifest_nonpositive_epsilon_raises() -> None:
    bad = dict(_VALID_MANIFEST, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon"):
        validate_gradient_update_manifest(bad)


def test_load_gradient_update_manifest_loads_and_validates(tmp_path) -> None:
    import json as _json

    manifest_file = tmp_path / "gradient_update.json"
    manifest_file.write_text(_json.dumps(_VALID_MANIFEST), encoding="utf-8")

    config = load_gradient_update_manifest(manifest_file)

    assert config["experiment"] == "gradient_update"
    assert config["num_clients"] == 2


def test_manifest_run_artifact_shape(tmp_path) -> None:
    config = validate_gradient_update_manifest(_VALID_MANIFEST)
    coordinator = FiniteDifferenceGradientCoordinator(
        [QuantumClient(client_id=f"client_{i + 1}", theta=0.0) for i in range(config["num_clients"])],
        initial_theta=config["initial_theta"],
        learning_rate=config["learning_rate"],
        target=config["target"],
        epsilon=config["epsilon"],
    )
    update_result = coordinator.run_updates(config["num_rounds"])

    artifact = build_run_artifact(
        example_name="run_from_manifest_gradient_update",
        run_result={"manifest": config, "result": update_result},
    )
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)
    saved_path = save_json_artifact(artifact, artifact_path)

    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "run_id" in saved_data
    assert "manifest" in saved_data["run"]
    assert "result" in saved_data["run"]
    assert saved_data["run"]["manifest"]["experiment"] == "gradient_update"


def test_validate_manifest_missing_manifest_version_raises() -> None:
    bad = dict(_VALID_MANIFEST)
    del bad["manifest_version"]

    with pytest.raises(ValueError, match="manifest_version"):
        validate_gradient_update_manifest(bad)


def test_validate_manifest_unsupported_version_raises() -> None:
    bad = dict(_VALID_MANIFEST, manifest_version="0.2")

    with pytest.raises(ValueError, match="manifest_version"):
        validate_gradient_update_manifest(bad)


def test_validate_manifest_missing_name_raises() -> None:
    bad = dict(_VALID_MANIFEST)
    del bad["name"]

    with pytest.raises(ValueError, match="name"):
        validate_gradient_update_manifest(bad)


def test_validate_manifest_blank_name_raises() -> None:
    bad = dict(_VALID_MANIFEST, name="   ")

    with pytest.raises(ValueError, match="name"):
        validate_gradient_update_manifest(bad)


def test_validate_manifest_missing_description_normalizes_to_empty_string() -> None:
    without_desc = {k: v for k, v in _VALID_MANIFEST.items() if k != "description"}

    config = validate_gradient_update_manifest(without_desc)

    assert config["description"] == ""


def test_validate_manifest_non_string_description_raises() -> None:
    bad = dict(_VALID_MANIFEST, description=42)

    with pytest.raises(ValueError, match="description"):
        validate_gradient_update_manifest(bad)


def test_all_example_manifests_are_valid() -> None:
    manifest_dir = Path(__file__).resolve().parents[1] / "examples" / "manifests"
    manifest_files = sorted(manifest_dir.glob("*.json"))

    assert len(manifest_files) > 0, "No manifest files found in examples/manifests/"

    for manifest_file in manifest_files:
        config = load_gradient_update_manifest(manifest_file)
        assert config["experiment"] == "gradient_update", (
            f"{manifest_file.name}: expected experiment='gradient_update'"
        )


# --- comparison tests ---

_DIRECT_ARTIFACT = {
    "run_id": "run_gradient_update_20260101T000000Z",
    "example": "run_gradient_update",
    "run": {
        "num_rounds": 3,
        "final_theta": 0.773778,
        "rounds": [
            {"loss": 0.770151},
            {"loss": 0.695861},
            {"loss": 0.608376},
        ],
    },
}

_MANIFEST_ARTIFACT = {
    "run_id": "run_from_manifest_gradient_update_20260101T000001Z",
    "example": "run_from_manifest_gradient_update",
    "run": {
        "manifest": {
            "manifest_version": "0.1",
            "name": "more-rounds",
            "experiment": "gradient_update",
            "num_rounds": 5,
        },
        "result": {
            "num_rounds": 5,
            "final_theta": 0.972194,
            "rounds": [
                {"loss": 0.770151},
                {"loss": 0.695861},
                {"loss": 0.608376},
                {"loss": 0.511619},
                {"loss": 0.412106},
            ],
        },
    },
}


def test_load_artifact_returns_dict(tmp_path) -> None:
    artifact_file = tmp_path / "artifact.json"
    artifact_file.write_text(json.dumps(_DIRECT_ARTIFACT), encoding="utf-8")

    result = load_artifact(artifact_file)

    assert isinstance(result, dict)
    assert result["run_id"] == _DIRECT_ARTIFACT["run_id"]


def test_load_artifact_rejects_non_object(tmp_path) -> None:
    artifact_file = tmp_path / "bad.json"
    artifact_file.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_artifact(artifact_file)


def test_summarize_artifact_direct_shape() -> None:
    summary = summarize_artifact(_DIRECT_ARTIFACT)

    assert summary["run_id"] == "run_gradient_update_20260101T000000Z"
    assert summary["example"] == "run_gradient_update"
    assert summary["experiment"] == "gradient_update"
    assert summary["num_rounds"] == 3
    assert summary["final_theta"] == pytest.approx(0.773778)
    assert summary["final_loss"] == pytest.approx(0.608376)


def test_summarize_artifact_manifest_shape() -> None:
    summary = summarize_artifact(_MANIFEST_ARTIFACT)

    assert summary["experiment"] == "gradient_update"
    assert summary["manifest_name"] == "more-rounds"
    assert summary["manifest_version"] == "0.1"
    assert summary["num_rounds"] == 5
    assert summary["final_theta"] == pytest.approx(0.972194)
    assert summary["final_loss"] == pytest.approx(0.412106)


def test_summarize_artifact_handles_missing_fields() -> None:
    summary = summarize_artifact({})

    assert summary["run_id"] == "unknown"
    assert summary["example"] == "unknown"
    assert summary["experiment"] == "unknown"
    assert summary["manifest_name"] == "unknown"
    assert summary["manifest_version"] == "unknown"
    assert summary["num_rounds"] is None
    assert summary["final_theta"] is None
    assert summary["final_loss"] is None


def test_summarize_artifacts_preserves_order() -> None:
    summaries = summarize_artifacts_from_dicts = [
        summarize_artifact(_DIRECT_ARTIFACT),
        summarize_artifact(_MANIFEST_ARTIFACT),
    ]

    assert summaries[0]["run_id"] == _DIRECT_ARTIFACT["run_id"]
    assert summaries[1]["run_id"] == _MANIFEST_ARTIFACT["run_id"]


def test_summarize_artifacts_list(tmp_path) -> None:
    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    file_a.write_text(json.dumps(_DIRECT_ARTIFACT), encoding="utf-8")
    file_b.write_text(json.dumps(_MANIFEST_ARTIFACT), encoding="utf-8")

    summaries = summarize_artifacts([file_a, file_b])

    assert len(summaries) == 2
    assert summaries[0]["run_id"] == _DIRECT_ARTIFACT["run_id"]
    assert summaries[1]["run_id"] == _MANIFEST_ARTIFACT["run_id"]


def test_format_artifact_comparison_contains_expected_content() -> None:
    summaries = [
        summarize_artifact(_DIRECT_ARTIFACT),
        summarize_artifact(_MANIFEST_ARTIFACT),
    ]

    output = format_artifact_comparison(summaries)

    assert "qfl-mini: artifact comparison" in output
    assert "more-rounds" in output
    assert "0.773778" in output
    assert "0.608376" in output
    assert "0.972194" in output
    assert "0.412106" in output


def test_format_artifact_comparison_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        format_artifact_comparison([])
