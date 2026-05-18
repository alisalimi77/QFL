from __future__ import annotations

from dataclasses import FrozenInstanceError
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
    extract_experiment_metrics,
    format_artifact_comparison,
    load_artifact,
    summarize_artifact,
    summarize_artifacts,
)
from qflmini.manifest import (
    build_backend_from_config,
    load_client_objectives_manifest,
    load_gradient_update_manifest,
    load_json_manifest,
    load_manifest,
    validate_backend_config,
    validate_client_objectives_manifest,
    validate_gradient_update_manifest,
    validate_manifest,
)
from qflmini.backends import ConstantBackend, NoisyBackend, PennyLaneBackend, get_backend_metadata
from qflmini.federated import ScalarFedAvgClient, ScalarFedAvgCoordinator
from qflmini.objectives import (
    ClientObjective,
    evaluate_client_objective,
    evaluate_client_objectives,
)
from qflmini.optimization import FiniteDifferenceGradientCoordinator, ParameterUpdateCoordinator
from qflmini.reporting import (
    format_clean_vs_noisy_backend_report,
    format_client_objectives_report,
    format_custom_backend_report,
    format_scalar_fedavg_report,
)


def test_quantum_client_run_returns_expected_keys() -> None:
    client = QuantumClient(client_id="client_1", theta=0.2)

    result = client.run()

    assert set(result) == {"client_id", "theta", "result"}
    assert result["client_id"] == "client_1"
    assert result["theta"] == 0.2
    assert isinstance(result["result"], float)


def test_quantum_client_is_immutable() -> None:
    client = QuantumClient(client_id="client_1", theta=0.2)
    with pytest.raises(FrozenInstanceError):
        client.theta = 0.4


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


_VALID_CLIENT_OBJECTIVES_MANIFEST = {
    "manifest_version": "0.1",
    "name": "client-objectives-demo",
    "description": "Client-specific objective evaluation with local targets.",
    "experiment": "client_objectives",
    "clients": [
        {"client_id": "client_1", "theta": 0.0, "target": 0.0},
        {"client_id": "client_2", "theta": 0.0, "target": 0.5},
    ],
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
    assert config["backend"] == {"type": "pennylane"}


def test_validate_backend_config_accepts_pennylane() -> None:
    assert validate_backend_config({"type": "pennylane"}) == {"type": "pennylane"}


def test_validate_backend_config_accepts_constant() -> None:
    assert validate_backend_config({"type": "constant", "value": 0.5}) == {
        "type": "constant",
        "value": 0.5,
    }


def test_validate_backend_config_accepts_noisy() -> None:
    config = validate_backend_config(
        {
            "type": "noisy",
            "base": {"type": "pennylane"},
            "noise": 0.05,
            "seed": 42,
        }
    )

    assert config == {
        "type": "noisy",
        "base": {"type": "pennylane"},
        "noise": 0.05,
        "seed": 42,
    }


def test_validate_backend_config_noisy_defaults_seed() -> None:
    config = validate_backend_config(
        {
            "type": "noisy",
            "base": {"type": "constant", "value": 0.5},
            "noise": 0.05,
        }
    )

    assert config["seed"] == 0
    assert config["base"] == {"type": "constant", "value": 0.5}


def test_validate_backend_config_requires_object() -> None:
    with pytest.raises(ValueError, match="backend"):
        validate_backend_config("pennylane")


def test_validate_backend_config_requires_type() -> None:
    with pytest.raises(ValueError, match="backend.type"):
        validate_backend_config({})


def test_validate_backend_config_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported backend type"):
        validate_backend_config({"type": "qiskit"})


def test_validate_backend_config_constant_requires_value() -> None:
    with pytest.raises(ValueError, match="backend.value"):
        validate_backend_config({"type": "constant"})


def test_validate_backend_config_constant_requires_numeric_value() -> None:
    with pytest.raises(ValueError, match="backend.value"):
        validate_backend_config({"type": "constant", "value": "0.5"})


def test_validate_backend_config_noisy_requires_base() -> None:
    with pytest.raises(ValueError, match="backend.base"):
        validate_backend_config({"type": "noisy", "noise": 0.05})


def test_validate_backend_config_noisy_requires_noise() -> None:
    with pytest.raises(ValueError, match="backend.noise"):
        validate_backend_config({"type": "noisy", "base": {"type": "pennylane"}})


def test_validate_backend_config_noisy_rejects_negative_noise() -> None:
    with pytest.raises(ValueError, match="backend.noise"):
        validate_backend_config(
            {"type": "noisy", "base": {"type": "pennylane"}, "noise": -0.1}
        )


def test_validate_backend_config_noisy_requires_integer_seed() -> None:
    with pytest.raises(ValueError, match="backend.seed"):
        validate_backend_config(
            {
                "type": "noisy",
                "base": {"type": "pennylane"},
                "noise": 0.05,
                "seed": 4.2,
            }
        )


def test_build_backend_from_config_returns_pennylane_backend() -> None:
    backend = build_backend_from_config({"type": "pennylane"})

    assert isinstance(backend, PennyLaneBackend)


def test_build_backend_from_config_returns_constant_backend() -> None:
    backend = build_backend_from_config({"type": "constant", "value": 0.5})

    assert isinstance(backend, ConstantBackend)
    assert backend.run_expectation(99.0) == pytest.approx(0.5)


def test_build_backend_from_config_returns_noisy_backend() -> None:
    backend = build_backend_from_config(
        {
            "type": "noisy",
            "base": {"type": "pennylane"},
            "noise": 0.05,
            "seed": 42,
        }
    )

    metadata = get_backend_metadata(backend)
    assert isinstance(backend, NoisyBackend)
    assert metadata["name"] == "noisy"
    assert metadata["base_backend"] == "pennylane"
    assert metadata["noise"] == "0.05"
    assert metadata["seed"] == "42"


def test_validate_client_objectives_manifest_returns_normalized_values() -> None:
    config = validate_client_objectives_manifest(_VALID_CLIENT_OBJECTIVES_MANIFEST)

    assert config["manifest_version"] == "0.1"
    assert config["name"] == "client-objectives-demo"
    assert config["description"] == "Client-specific objective evaluation with local targets."
    assert config["experiment"] == "client_objectives"
    assert config["backend"] == {"type": "pennylane"}
    assert config["clients"] == [
        {"client_id": "client_1", "theta": 0.0, "target": 0.0},
        {"client_id": "client_2", "theta": 0.0, "target": 0.5},
    ]


def test_validate_client_objectives_manifest_missing_clients_raises() -> None:
    bad = dict(_VALID_CLIENT_OBJECTIVES_MANIFEST)
    del bad["clients"]

    with pytest.raises(ValueError, match="clients"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_empty_clients_raises() -> None:
    bad = dict(_VALID_CLIENT_OBJECTIVES_MANIFEST, clients=[])

    with pytest.raises(ValueError, match="clients"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_client_must_be_object() -> None:
    bad = dict(_VALID_CLIENT_OBJECTIVES_MANIFEST, clients=["client_1"])

    with pytest.raises(ValueError, match="clients\\[0\\]"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_missing_client_id_raises() -> None:
    bad = dict(
        _VALID_CLIENT_OBJECTIVES_MANIFEST,
        clients=[{"theta": 0.2, "target": 0.0}],
    )

    with pytest.raises(ValueError, match="client_id"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_blank_client_id_raises() -> None:
    bad = dict(
        _VALID_CLIENT_OBJECTIVES_MANIFEST,
        clients=[{"client_id": "   ", "theta": 0.2, "target": 0.0}],
    )

    with pytest.raises(ValueError, match="client_id"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_non_numeric_theta_raises() -> None:
    bad = dict(
        _VALID_CLIENT_OBJECTIVES_MANIFEST,
        clients=[{"client_id": "client_1", "theta": "0.2", "target": 0.0}],
    )

    with pytest.raises(ValueError, match="theta"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_non_numeric_target_raises() -> None:
    bad = dict(
        _VALID_CLIENT_OBJECTIVES_MANIFEST,
        clients=[{"client_id": "client_1", "theta": 0.2, "target": "0.0"}],
    )

    with pytest.raises(ValueError, match="target"):
        validate_client_objectives_manifest(bad)


def test_validate_client_objectives_manifest_accepts_backend_config() -> None:
    config = validate_client_objectives_manifest(
        {
            **_VALID_CLIENT_OBJECTIVES_MANIFEST,
            "backend": {"type": "constant", "value": 0.5},
        }
    )

    assert config["backend"] == {"type": "constant", "value": 0.5}


def test_validate_manifest_dispatches_gradient_update() -> None:
    config = validate_manifest(_VALID_MANIFEST)

    assert config["experiment"] == "gradient_update"


def test_validate_manifest_dispatches_client_objectives() -> None:
    config = validate_manifest(_VALID_CLIENT_OBJECTIVES_MANIFEST)

    assert config["experiment"] == "client_objectives"


def test_validate_manifest_rejects_unsupported_experiment() -> None:
    bad = dict(_VALID_MANIFEST, experiment="unknown")

    with pytest.raises(ValueError, match="Unsupported experiment type"):
        validate_manifest(bad)


def test_load_manifest_loads_and_validates(tmp_path) -> None:
    manifest_file = tmp_path / "client_objectives.json"
    manifest_file.write_text(
        json.dumps(_VALID_CLIENT_OBJECTIVES_MANIFEST),
        encoding="utf-8",
    )

    config = load_manifest(manifest_file)

    assert config["experiment"] == "client_objectives"


def test_load_client_objectives_manifest_loads_and_validates(tmp_path) -> None:
    manifest_file = tmp_path / "client_objectives.json"
    manifest_file.write_text(
        json.dumps(_VALID_CLIENT_OBJECTIVES_MANIFEST),
        encoding="utf-8",
    )

    config = load_client_objectives_manifest(manifest_file)

    assert config["experiment"] == "client_objectives"
    assert len(config["clients"]) == 2


def test_build_backend_from_client_objectives_manifest() -> None:
    config = validate_client_objectives_manifest(
        {
            **_VALID_CLIENT_OBJECTIVES_MANIFEST,
            "backend": {"type": "constant", "value": 0.5},
        }
    )

    backend = build_backend_from_config(config["backend"])

    assert isinstance(backend, ConstantBackend)


def test_client_objectives_manifest_run_path_with_constant_backend() -> None:
    config = validate_client_objectives_manifest(
        {
            **_VALID_CLIENT_OBJECTIVES_MANIFEST,
            "backend": {"type": "constant", "value": 0.5},
        }
    )
    backend = build_backend_from_config(config["backend"])
    objectives = [
        ClientObjective(
            QuantumClient(client["client_id"], theta=client["theta"], backend=backend),
            target=client["target"],
        )
        for client in config["clients"]
    ]

    result = evaluate_client_objectives(objectives)

    assert result["aggregated_result"] == pytest.approx(0.5)
    assert result["mean_local_loss"] == pytest.approx(0.125)


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
    backend = build_backend_from_config(config["backend"])
    coordinator = FiniteDifferenceGradientCoordinator(
        [
            QuantumClient(client_id=f"client_{i + 1}", theta=0.0, backend=backend)
            for i in range(config["num_clients"])
        ],
        initial_theta=config["initial_theta"],
        learning_rate=config["learning_rate"],
        target=config["target"],
        epsilon=config["epsilon"],
    )
    update_result = coordinator.run_updates(config["num_rounds"])

    artifact = build_run_artifact(
        example_name="run_from_manifest_gradient_update",
        run_result={
            "manifest_path": "examples/manifests/gradient_update.json",
            "manifest": config,
            "backend": get_backend_metadata(backend),
            "result": update_result,
        },
    )
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)
    saved_path = save_json_artifact(artifact, artifact_path)

    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "run_id" in saved_data
    assert saved_data["run"]["manifest_path"] == "examples/manifests/gradient_update.json"
    assert "manifest" in saved_data["run"]
    assert saved_data["run"]["backend"]["name"] == "pennylane"
    assert "result" in saved_data["run"]
    assert saved_data["run"]["manifest"]["experiment"] == "gradient_update"


def test_client_objectives_manifest_artifact_shape(tmp_path) -> None:
    config = validate_client_objectives_manifest(_VALID_CLIENT_OBJECTIVES_MANIFEST)
    backend = build_backend_from_config(config["backend"])
    objectives = [
        ClientObjective(
            QuantumClient(client["client_id"], theta=client["theta"], backend=backend),
            target=client["target"],
        )
        for client in config["clients"]
    ]
    result = evaluate_client_objectives(objectives)

    artifact = build_run_artifact(
        example_name="run_from_manifest_client_objectives",
        run_result={
            "manifest_path": "examples/manifests/client_objectives.json",
            "manifest": config,
            "backend": get_backend_metadata(backend),
            "result": result,
        },
    )
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)
    saved_path = save_json_artifact(artifact, artifact_path)
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_data["run"]["manifest"]["experiment"] == "client_objectives"
    assert saved_data["run"]["backend"]["name"] == "pennylane"
    assert "client_objectives" in saved_data["run"]["result"]
    assert "mean_local_loss" in saved_data["run"]["result"]


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
        config = load_manifest(manifest_file)
        assert config["experiment"] in {"gradient_update", "client_objectives"}, (
            f"{manifest_file.name}: expected supported experiment"
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
        "manifest_path": "examples/manifests/gradient_update_more_rounds.json",
        "manifest": {
            "manifest_version": "0.1",
            "name": "more-rounds",
            "experiment": "gradient_update",
            "num_rounds": 5,
        },
        "backend": {
            "name": "pennylane",
            "class": "PennyLaneBackend",
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


_CLIENT_OBJECTIVES_MANIFEST_ARTIFACT = {
    "run_id": "run_from_manifest_client_objectives_20260101T000002Z",
    "example": "run_from_manifest_client_objectives",
    "run": {
        "manifest_path": "examples/manifests/client_objectives.json",
        "manifest": {
            "manifest_version": "0.1",
            "name": "client-objectives-demo",
            "experiment": "client_objectives",
            "clients": [
                {"client_id": "client_1", "theta": 0.2, "target": 0.0},
                {"client_id": "client_2", "theta": 0.8, "target": 0.5},
            ],
        },
        "backend": {
            "name": "pennylane",
            "class": "PennyLaneBackend",
        },
        "result": {
            "num_clients": 2,
            "client_objectives": [
                {
                    "client_id": "client_1",
                    "theta": 0.2,
                    "target": 0.0,
                    "result": 0.980066,
                    "loss": 0.96053,
                },
                {
                    "client_id": "client_2",
                    "theta": 0.8,
                    "target": 0.5,
                    "result": 0.696707,
                    "loss": 0.038694,
                },
            ],
            "aggregated_result": 0.838387,
            "mean_local_loss": 0.499612,
        },
    },
}


_SCALAR_FEDAVG_ARTIFACT = {
    "run_id": "run_scalar_fedavg_20260101T000003Z",
    "example": "run_scalar_fedavg",
    "run": {
        "backend": {
            "name": "pennylane",
            "class": "PennyLaneBackend",
        },
        "result": {
            "algorithm": "scalar_fedavg",
            "num_clients": 2,
            "num_rounds": 2,
            "initial_theta": 0.5,
            "learning_rate": 0.1,
            "epsilon": 0.001,
            "aggregation": {"method": "mean"},
            "clients": [
                {"client_id": "client_1", "target": 0.0},
                {"client_id": "client_2", "target": 0.5},
            ],
            "rounds": [
                {
                    "round": 1,
                    "global_theta": 0.5,
                    "client_updates": [
                        {
                            "client_id": "client_1",
                            "target": 0.0,
                            "theta": 0.5,
                            "result": 0.877583,
                            "loss": 0.770151,
                            "loss_plus": 0.76931,
                            "loss_minus": 0.770992,
                            "gradient": -0.841,
                            "local_next_theta": 0.5841,
                        }
                    ],
                    "aggregation": {
                        "method": "mean",
                        "local_next_thetas": [0.5841, 0.5337],
                        "next_global_theta": 0.5589,
                    },
                    "mean_local_loss": 0.456359,
                },
                {
                    "round": 2,
                    "global_theta": 0.5589,
                    "client_updates": [],
                    "aggregation": {
                        "method": "mean",
                        "local_next_thetas": [0.62, 0.58],
                        "next_global_theta": 0.6,
                    },
                    "mean_local_loss": 0.3,
                },
            ],
            "final_theta": 0.6,
            "final_mean_local_loss": 0.3,
        },
    },
}


def _manifest_artifact_with_backend(backend: dict) -> dict:
    artifact = json.loads(json.dumps(_MANIFEST_ARTIFACT))
    artifact["run"]["backend"] = backend
    return artifact


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
    assert summary["manifest_path"] == "unknown"
    assert summary["manifest_file"] == "unknown"
    assert summary["backend_name"] == "unknown"
    assert summary["backend_class"] == "unknown"
    assert summary["backend_detail"] == "-"
    assert summary["num_rounds"] == 3
    assert summary["final_theta"] == pytest.approx(0.773778)
    assert summary["final_loss"] == pytest.approx(0.608376)
    assert summary["primary_metric"] == "final_loss"
    assert summary["primary_value"] == pytest.approx(0.608376)
    assert summary["secondary_metric"] == "final_theta"
    assert summary["secondary_value"] == pytest.approx(0.773778)


def test_summarize_artifact_manifest_shape() -> None:
    summary = summarize_artifact(_MANIFEST_ARTIFACT)

    assert summary["experiment"] == "gradient_update"
    assert summary["manifest_name"] == "more-rounds"
    assert summary["manifest_version"] == "0.1"
    assert summary["manifest_path"] == "examples/manifests/gradient_update_more_rounds.json"
    assert summary["manifest_file"] == "gradient_update_more_rounds.json"
    assert summary["backend_name"] == "pennylane"
    assert summary["backend_class"] == "PennyLaneBackend"
    assert summary["backend_detail"] == "-"
    assert summary["num_rounds"] == 5
    assert summary["final_theta"] == pytest.approx(0.972194)
    assert summary["final_loss"] == pytest.approx(0.412106)
    assert summary["primary_metric"] == "final_loss"
    assert summary["primary_value"] == pytest.approx(0.412106)
    assert summary["secondary_metric"] == "final_theta"
    assert summary["secondary_value"] == pytest.approx(0.972194)


def test_summarize_artifact_client_objectives_shape() -> None:
    summary = summarize_artifact(_CLIENT_OBJECTIVES_MANIFEST_ARTIFACT)

    assert summary["experiment"] == "client_objectives"
    assert summary["manifest_name"] == "client-objectives-demo"
    assert summary["manifest_file"] == "client_objectives.json"
    assert summary["num_rounds"] is None
    assert summary["final_theta"] is None
    assert summary["mean_local_loss"] == pytest.approx(0.499612)
    assert summary["final_loss"] == pytest.approx(0.499612)
    assert summary["primary_metric"] == "mean_local_loss"
    assert summary["primary_value"] == pytest.approx(0.499612)
    assert summary["secondary_metric"] == "aggregated_result"
    assert summary["secondary_value"] == pytest.approx(0.838387)


def test_summarize_artifact_handles_missing_fields() -> None:
    summary = summarize_artifact({})

    assert summary["run_id"] == "unknown"
    assert summary["example"] == "unknown"
    assert summary["experiment"] == "unknown"
    assert summary["manifest_name"] == "unknown"
    assert summary["manifest_version"] == "unknown"
    assert summary["manifest_path"] == "unknown"
    assert summary["manifest_file"] == "unknown"
    assert summary["backend_name"] == "unknown"
    assert summary["backend_class"] == "unknown"
    assert summary["backend_detail"] == "-"
    assert summary["num_rounds"] is None
    assert summary["final_theta"] is None
    assert summary["final_loss"] is None
    assert summary["primary_metric"] == "n/a"
    assert summary["primary_value"] is None
    assert summary["secondary_metric"] == "n/a"
    assert summary["secondary_value"] is None


def test_extract_experiment_metrics_gradient_update_manifest() -> None:
    metrics = extract_experiment_metrics(_MANIFEST_ARTIFACT)

    assert metrics["primary_metric"] == "final_loss"
    assert metrics["primary_value"] == pytest.approx(0.412106)
    assert metrics["secondary_metric"] == "final_theta"
    assert metrics["secondary_value"] == pytest.approx(0.972194)


def test_extract_experiment_metrics_gradient_update_direct() -> None:
    metrics = extract_experiment_metrics(_DIRECT_ARTIFACT)

    assert metrics["primary_metric"] == "final_loss"
    assert metrics["primary_value"] == pytest.approx(0.608376)
    assert metrics["secondary_metric"] == "final_theta"
    assert metrics["secondary_value"] == pytest.approx(0.773778)


def test_extract_experiment_metrics_client_objectives() -> None:
    metrics = extract_experiment_metrics(_CLIENT_OBJECTIVES_MANIFEST_ARTIFACT)

    assert metrics["primary_metric"] == "mean_local_loss"
    assert metrics["primary_value"] == pytest.approx(0.499612)
    assert metrics["secondary_metric"] == "aggregated_result"
    assert metrics["secondary_value"] == pytest.approx(0.838387)


def test_extract_experiment_metrics_scalar_fedavg() -> None:
    metrics = extract_experiment_metrics(_SCALAR_FEDAVG_ARTIFACT)

    assert metrics["primary_metric"] == "final_mean_local_loss"
    assert metrics["primary_value"] == pytest.approx(0.3)
    assert metrics["secondary_metric"] == "final_theta"
    assert metrics["secondary_value"] == pytest.approx(0.6)


def test_extract_experiment_metrics_unknown_without_metrics() -> None:
    metrics = extract_experiment_metrics({"run": {"example": "unknown"}})

    assert metrics["primary_metric"] == "n/a"
    assert metrics["primary_value"] is None
    assert metrics["secondary_metric"] == "n/a"
    assert metrics["secondary_value"] is None


def test_extract_experiment_metrics_unknown_with_fallback_metrics() -> None:
    artifact = {
        "run": {
            "final_theta": 0.3,
            "rounds": [{"loss": 0.4}],
        }
    }

    metrics = extract_experiment_metrics(artifact)

    assert metrics["primary_metric"] == "final_loss"
    assert metrics["primary_value"] == pytest.approx(0.4)
    assert metrics["secondary_metric"] == "final_theta"
    assert metrics["secondary_value"] == pytest.approx(0.3)


def test_summarize_artifact_noisy_backend_detail() -> None:
    artifact = _manifest_artifact_with_backend(
        {
            "name": "noisy",
            "class": "NoisyBackend",
            "base_backend": "pennylane",
            "noise": "0.05",
            "seed": "42",
        }
    )

    summary = summarize_artifact(artifact)

    assert summary["backend_detail"] == "base=pennylane, noise=0.05, seed=42"


def test_summarize_artifact_constant_backend_detail() -> None:
    artifact = _manifest_artifact_with_backend(
        {
            "name": "constant",
            "class": "ConstantBackend",
            "value": "0.5",
        }
    )

    summary = summarize_artifact(artifact)

    assert summary["backend_detail"] == "value=0.5"


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
        summarize_artifact(
            _manifest_artifact_with_backend(
                {
                    "name": "noisy",
                    "class": "NoisyBackend",
                    "base_backend": "pennylane",
                    "noise": "0.05",
                    "seed": "42",
                }
            )
        ),
        summarize_artifact(
            _manifest_artifact_with_backend(
                {
                    "name": "constant",
                    "class": "ConstantBackend",
                    "value": "0.5",
                }
            )
        ),
    ]

    output = format_artifact_comparison(summaries)

    assert "qfl-mini: artifact comparison" in output
    assert "manifest_file" in output
    assert "backend" in output
    assert "backend_detail" in output
    assert "primary_metric" in output
    assert "primary_value" in output
    assert "secondary_metric" in output
    assert "secondary_value" in output
    assert "final_loss" in output
    assert "final_theta" in output
    assert "pennylane" in output
    assert "noise=0.05" in output
    assert "value=0.5" in output
    assert "gradient_update_more_rounds.json" in output
    assert "more-rounds" in output
    assert "0.773778" in output
    assert "0.608376" in output
    assert "0.972194" in output
    assert "0.412106" in output


def test_format_artifact_comparison_handles_mixed_experiments() -> None:
    summaries = [
        summarize_artifact(_MANIFEST_ARTIFACT),
        summarize_artifact(_CLIENT_OBJECTIVES_MANIFEST_ARTIFACT),
        summarize_artifact(_SCALAR_FEDAVG_ARTIFACT),
    ]

    output = format_artifact_comparison(summaries)

    assert "gradient_update" in output
    assert "client_objectives" in output
    assert "scalar_fedavg" in output
    assert "client-objectives-demo" in output
    assert "mean_local_loss" in output
    assert "aggregated_result" in output
    assert "final_mean_local_loss" in output
    assert "0.499612" in output


def test_format_artifact_comparison_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        format_artifact_comparison([])


# --- client-specific objective tests ---


def test_client_objective_is_immutable() -> None:
    objective = ClientObjective(
        client=QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.5)),
        target=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        objective.target = 0.5


def test_evaluate_client_objective_computes_loss() -> None:
    objective = ClientObjective(
        client=QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.5)),
        target=0.0,
    )

    result = evaluate_client_objective(objective)

    assert result["client_id"] == "client_1"
    assert result["result"] == pytest.approx(0.5)
    assert result["target"] == pytest.approx(0.0)
    assert result["loss"] == pytest.approx(0.25)


def test_evaluate_client_objective_zero_loss_for_matching_target() -> None:
    objective = ClientObjective(
        client=QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.5)),
        target=0.5,
    )

    result = evaluate_client_objective(objective)

    assert result["loss"] == pytest.approx(0.0)


def test_evaluate_client_objectives_computes_means() -> None:
    result = evaluate_client_objectives(
        [
            ClientObjective(
                QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.2)),
                target=0.0,
            ),
            ClientObjective(
                QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.6)),
                target=0.5,
            ),
        ]
    )

    assert result["num_clients"] == 2
    assert len(result["client_objectives"]) == 2
    assert result["aggregated_result"] == pytest.approx(0.4)
    assert result["mean_local_loss"] == pytest.approx(0.025)


def test_evaluate_client_objectives_empty_raises() -> None:
    with pytest.raises(ValueError, match="objectives"):
        evaluate_client_objectives([])


def test_format_client_objectives_report_contains_expected_content() -> None:
    result = evaluate_client_objectives(
        [
            ClientObjective(
                QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.2)),
                target=0.0,
            ),
            ClientObjective(
                QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.6)),
                target=0.5,
            ),
        ]
    )

    output = format_client_objectives_report(result)

    assert "client-specific objective demo" in output
    assert "client_1" in output
    assert "client_2" in output
    assert "Mean local loss" in output
    assert "0.400000" in output
    assert "0.025000" in output


def test_client_objectives_artifact_shape(tmp_path) -> None:
    result = evaluate_client_objectives(
        [
            ClientObjective(
                QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.2)),
                target=0.0,
            ),
            ClientObjective(
                QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.6)),
                target=0.5,
            ),
        ]
    )
    artifact = build_run_artifact("run_client_objectives", result)
    path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)

    saved_path = save_json_artifact(artifact, path)
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert "client_objectives" in saved_data["run"]
    assert "mean_local_loss" in saved_data["run"]
    assert "aggregated_result" in saved_data["run"]


# --- scalar FedAvg tests ---


def test_scalar_fedavg_requires_clients() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        ScalarFedAvgCoordinator(
            clients=[],
            backend=ConstantBackend(0.5),
            initial_theta=0.5,
            learning_rate=0.1,
        )


def test_scalar_fedavg_requires_non_empty_client_id() -> None:
    with pytest.raises(ValueError, match="client_id"):
        ScalarFedAvgCoordinator(
            clients=[ScalarFedAvgClient(client_id=" ", target=0.0)],
            backend=ConstantBackend(0.5),
            initial_theta=0.5,
            learning_rate=0.1,
        )


def test_scalar_fedavg_requires_numeric_target() -> None:
    with pytest.raises(ValueError, match="target"):
        ScalarFedAvgCoordinator(
            clients=[ScalarFedAvgClient(client_id="client_1", target="0.0")],
            backend=ConstantBackend(0.5),
            initial_theta=0.5,
            learning_rate=0.1,
        )


def test_scalar_fedavg_requires_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        ScalarFedAvgCoordinator(
            clients=[ScalarFedAvgClient(client_id="client_1", target=0.0)],
            backend=ConstantBackend(0.5),
            initial_theta=0.5,
            learning_rate=0.0,
        )


def test_scalar_fedavg_requires_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        ScalarFedAvgCoordinator(
            clients=[ScalarFedAvgClient(client_id="client_1", target=0.0)],
            backend=ConstantBackend(0.5),
            initial_theta=0.5,
            learning_rate=0.1,
            epsilon=0.0,
        )


def test_scalar_fedavg_run_rounds_requires_positive_count() -> None:
    coordinator = ScalarFedAvgCoordinator(
        clients=[ScalarFedAvgClient(client_id="client_1", target=0.0)],
        backend=ConstantBackend(0.5),
        initial_theta=0.5,
        learning_rate=0.1,
    )

    with pytest.raises(ValueError, match="at least 1"):
        coordinator.run_rounds(0)


def test_scalar_fedavg_constant_backend_has_zero_gradient() -> None:
    coordinator = ScalarFedAvgCoordinator(
        clients=[ScalarFedAvgClient(client_id="client_1", target=0.0)],
        backend=ConstantBackend(0.5),
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_rounds(1)
    update = result["rounds"][0]["client_updates"][0]

    assert update["result"] == pytest.approx(0.5)
    assert update["loss"] == pytest.approx(0.25)
    assert update["loss_plus"] == pytest.approx(update["loss_minus"])
    assert update["gradient"] == pytest.approx(0.0)
    assert update["local_next_theta"] == pytest.approx(0.5)


def test_scalar_fedavg_aggregates_local_updates() -> None:
    coordinator = ScalarFedAvgCoordinator(
        clients=[
            ScalarFedAvgClient(client_id="client_1", target=0.0),
            ScalarFedAvgClient(client_id="client_2", target=0.5),
        ],
        backend=ConstantBackend(0.5),
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_rounds(1)
    round_result = result["rounds"][0]

    assert round_result["aggregation"]["method"] == "mean"
    assert round_result["aggregation"]["next_global_theta"] == pytest.approx(0.5)
    assert result["final_theta"] == pytest.approx(0.5)


def test_scalar_fedavg_trace_structure() -> None:
    coordinator = ScalarFedAvgCoordinator(
        clients=[
            ScalarFedAvgClient(client_id="client_1", target=0.0),
            ScalarFedAvgClient(client_id="client_2", target=0.5),
        ],
        backend=ConstantBackend(0.5),
        initial_theta=0.5,
        learning_rate=0.1,
    )

    result = coordinator.run_rounds(2)

    assert result["algorithm"] == "scalar_fedavg"
    assert len(result["rounds"]) == 2
    for round_result in result["rounds"]:
        assert "global_theta" in round_result
        assert "client_updates" in round_result
        assert "aggregation" in round_result
        assert "mean_local_loss" in round_result
        for update in round_result["client_updates"]:
            for key in (
                "client_id",
                "target",
                "theta",
                "result",
                "loss",
                "loss_plus",
                "loss_minus",
                "gradient",
                "local_next_theta",
            ):
                assert key in update


def test_format_scalar_fedavg_report_contains_expected_content() -> None:
    coordinator = ScalarFedAvgCoordinator(
        clients=[
            ScalarFedAvgClient(client_id="client_1", target=0.0),
            ScalarFedAvgClient(client_id="client_2", target=0.5),
        ],
        backend=ConstantBackend(0.5),
        initial_theta=0.5,
        learning_rate=0.1,
    )

    output = format_scalar_fedavg_report(coordinator.run_rounds(2))

    assert "transparent scalar FedAvg demo" in output
    assert "Rounds:" in output
    assert "Client updates, final round:" in output
    assert "Final theta:" in output


def test_scalar_fedavg_artifact_shape(tmp_path) -> None:
    backend = ConstantBackend(0.5)
    coordinator = ScalarFedAvgCoordinator(
        clients=[ScalarFedAvgClient(client_id="client_1", target=0.0)],
        backend=backend,
        initial_theta=0.5,
        learning_rate=0.1,
    )
    result = coordinator.run_rounds(1)
    artifact = build_run_artifact(
        "run_scalar_fedavg",
        {"backend": get_backend_metadata(backend), "result": result},
    )
    path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)

    saved_path = save_json_artifact(artifact, path)
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert "backend" in saved_data["run"]
    assert saved_data["run"]["result"]["algorithm"] == "scalar_fedavg"
    assert "rounds" in saved_data["run"]["result"]
    assert "final_theta" in saved_data["run"]["result"]
    assert "final_mean_local_loss" in saved_data["run"]["result"]


def test_scalar_fedavg_comparison_support() -> None:
    summary = summarize_artifact(_SCALAR_FEDAVG_ARTIFACT)
    output = format_artifact_comparison([summary])

    assert summary["experiment"] == "scalar_fedavg"
    assert summary["primary_metric"] == "final_mean_local_loss"
    assert summary["secondary_metric"] == "final_theta"
    assert "scalar_fedavg" in output
    assert "final_mean_local_loss" in output


# --- backend tests ---


def test_default_backend_runs_and_returns_result() -> None:
    client = QuantumClient(client_id="client_1", theta=0.2)
    result = client.run()

    assert result["client_id"] == "client_1"
    assert isinstance(result["result"], float)


def test_pennylane_backend_direct_call() -> None:
    backend = PennyLaneBackend()
    value = backend.run_expectation(0.2)

    assert isinstance(value, float)


def test_custom_backend_injection() -> None:
    client = QuantumClient(
        client_id="client_1",
        theta=123.0,
        backend=ConstantBackend(0.42),
    )

    assert client.run()["result"] == pytest.approx(0.42)


def test_coordinator_with_custom_backend() -> None:
    clients = [
        QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.2)),
        QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.6)),
    ]
    result = Coordinator(clients).run_round()

    assert result["aggregated_result"] == pytest.approx(0.4)


def test_parameter_update_coordinator_preserves_backend() -> None:
    clients = [
        QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.5)),
        QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.5)),
    ]
    coordinator = ParameterUpdateCoordinator(
        clients, initial_theta=1.0, learning_rate=0.1
    )
    result = coordinator.run_updates(1)

    assert result["rounds"][0]["aggregated_result"] == pytest.approx(0.5)
    assert result["rounds"][0]["next_theta"] == pytest.approx(1.0 - 0.1 * 0.5)


def test_gradient_coordinator_preserves_backend() -> None:
    clients = [
        QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.5)),
        QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.5)),
    ]
    coordinator = FiniteDifferenceGradientCoordinator(
        clients, initial_theta=1.0, learning_rate=0.1, target=0.0
    )
    result = coordinator.run_updates(1)
    round_result = result["rounds"][0]

    # Constant backend: loss_plus == loss_minus, so gradient == 0
    assert round_result["gradient"] == pytest.approx(0.0)
    assert round_result["next_theta"] == pytest.approx(1.0)


# --- backend metadata tests ---


def test_pennylane_backend_metadata() -> None:
    meta = get_backend_metadata(PennyLaneBackend())

    assert meta["name"] == "pennylane"
    assert meta["class"] == "PennyLaneBackend"


def test_constant_backend_metadata() -> None:
    meta = get_backend_metadata(ConstantBackend(0.5))

    assert meta["name"] == "constant"
    assert meta["class"] == "ConstantBackend"


def test_backend_metadata_falls_back_to_class_name_when_no_name() -> None:
    class NoNameBackend:
        def run_expectation(self, theta: float) -> float:
            return 0.0

    meta = get_backend_metadata(NoNameBackend())

    assert meta["name"] == "NoNameBackend"
    assert meta["class"] == "NoNameBackend"


def test_backend_metadata_falls_back_to_class_name_when_name_is_empty() -> None:
    class EmptyNameBackend:
        name = ""

        def run_expectation(self, theta: float) -> float:
            return 0.0

    meta = get_backend_metadata(EmptyNameBackend())

    assert meta["name"] == "EmptyNameBackend"
    assert meta["class"] == "EmptyNameBackend"


# --- ConstantBackend behavior tests ---


def test_constant_backend_returns_fixed_value() -> None:
    backend = ConstantBackend(0.42)

    assert backend.run_expectation(0.0) == pytest.approx(0.42)
    assert backend.run_expectation(999.0) == pytest.approx(0.42)


def test_constant_backend_via_client() -> None:
    client = QuantumClient(client_id="client_1", theta=0.0, backend=ConstantBackend(0.42))

    assert client.run()["result"] == pytest.approx(0.42)


# --- custom backend report tests ---


def test_format_custom_backend_report_contains_expected_content() -> None:
    clients = [
        QuantumClient("client_1", theta=0.0, backend=ConstantBackend(0.2)),
        QuantumClient("client_2", theta=0.0, backend=ConstantBackend(0.6)),
    ]
    round_result = Coordinator(clients).run_round()
    output = format_custom_backend_report(round_result)

    assert "custom backend demo" in output
    assert "client_1" in output
    assert "client_2" in output
    assert "0.200000" in output
    assert "0.600000" in output
    assert "0.400000" in output


# --- NoisyBackend behavior tests ---


def test_noisy_backend_rejects_negative_noise() -> None:
    with pytest.raises(ValueError, match="noise must be >= 0"):
        NoisyBackend(base_backend=ConstantBackend(0.5), noise=-0.1)


def test_noisy_backend_is_deterministic() -> None:
    backend = NoisyBackend(base_backend=ConstantBackend(0.5), noise=0.1, seed=1)

    result_a = backend.run_expectation(0.3)
    result_b = backend.run_expectation(0.3)

    assert result_a == result_b


def test_noisy_backend_clips_upper_bound() -> None:
    backend = NoisyBackend(base_backend=ConstantBackend(1.0), noise=1.0, seed=0)

    result = backend.run_expectation(1.0)

    assert result == pytest.approx(1.0)


def test_noisy_backend_clips_lower_bound() -> None:
    backend = NoisyBackend(base_backend=ConstantBackend(-1.0), noise=1.0, seed=0)

    result = backend.run_expectation(-1.0)

    assert result == pytest.approx(-1.0)


def test_noisy_backend_output_differs_from_base() -> None:
    base = ConstantBackend(0.5)
    noisy = NoisyBackend(base_backend=ConstantBackend(0.5), noise=0.1, seed=7)

    theta = 1.0
    base_result = base.run_expectation(theta)
    noisy_result = noisy.run_expectation(theta)

    assert base_result != pytest.approx(noisy_result)


# --- Backend metadata for NoisyBackend and ConstantBackend ---


def test_noisy_backend_metadata_has_expected_keys() -> None:
    backend = NoisyBackend(base_backend=PennyLaneBackend(), noise=0.05, seed=42)

    meta = get_backend_metadata(backend)

    assert meta["name"] == "noisy"
    assert meta["class"] == "NoisyBackend"
    assert meta["base_backend"] == "pennylane"
    assert meta["noise"] == "0.05"
    assert meta["seed"] == "42"


def test_constant_backend_metadata_has_value_key() -> None:
    backend = ConstantBackend(0.77)

    meta = get_backend_metadata(backend)

    assert meta["name"] == "constant"
    assert meta["class"] == "ConstantBackend"
    assert meta["value"] == "0.77"


# --- clean-vs-noisy report tests ---


def test_format_clean_vs_noisy_backend_report_contains_expected_content() -> None:
    clean_backend = PennyLaneBackend()
    noisy_backend = NoisyBackend(base_backend=PennyLaneBackend(), noise=0.05, seed=42)
    clean_clients = [
        QuantumClient("client_1", theta=0.2, backend=clean_backend),
        QuantumClient("client_2", theta=0.8, backend=clean_backend),
    ]
    noisy_clients = [
        QuantumClient("client_1", theta=0.2, backend=noisy_backend),
        QuantumClient("client_2", theta=0.8, backend=noisy_backend),
    ]
    clean_result = Coordinator(clean_clients).run_round()
    noisy_result = Coordinator(noisy_clients).run_round()
    difference = noisy_result["aggregated_result"] - clean_result["aggregated_result"]
    run_result = {
        "clean": {"backend": get_backend_metadata(clean_backend), "result": clean_result},
        "noisy": {"backend": get_backend_metadata(noisy_backend), "result": noisy_result},
        "difference": difference,
    }

    output = format_clean_vs_noisy_backend_report(run_result)

    assert "clean vs noisy backend demo" in output
    assert "backend=pennylane" in output
    assert "backend=noisy" in output
    assert "Difference:" in output


# --- clean-vs-noisy artifact shape tests ---


def test_clean_vs_noisy_run_result_has_expected_structure() -> None:
    clean_backend = PennyLaneBackend()
    noisy_backend = NoisyBackend(base_backend=PennyLaneBackend(), noise=0.05, seed=42)
    clean_clients = [QuantumClient("c1", theta=0.3, backend=clean_backend)]
    noisy_clients = [QuantumClient("c1", theta=0.3, backend=noisy_backend)]
    clean_result = Coordinator(clean_clients).run_round()
    noisy_result = Coordinator(noisy_clients).run_round()
    run_result = {
        "clean": {"backend": get_backend_metadata(clean_backend), "result": clean_result},
        "noisy": {"backend": get_backend_metadata(noisy_backend), "result": noisy_result},
        "difference": noisy_result["aggregated_result"] - clean_result["aggregated_result"],
    }

    assert "clean" in run_result
    assert "noisy" in run_result
    assert "difference" in run_result
    assert run_result["clean"]["backend"]["name"] == "pennylane"
    assert run_result["noisy"]["backend"]["name"] == "noisy"
    assert run_result["noisy"]["backend"]["base_backend"] == "pennylane"
    assert isinstance(run_result["difference"], float)


def test_clean_vs_noisy_artifact_contains_backend_metadata(tmp_path) -> None:
    clean_backend = PennyLaneBackend()
    noisy_backend = NoisyBackend(base_backend=PennyLaneBackend(), noise=0.05, seed=42)
    clean_result = Coordinator(
        [QuantumClient("c1", theta=0.3, backend=clean_backend)]
    ).run_round()
    noisy_result = Coordinator(
        [QuantumClient("c1", theta=0.3, backend=noisy_backend)]
    ).run_round()
    run_result = {
        "clean": {"backend": get_backend_metadata(clean_backend), "result": clean_result},
        "noisy": {"backend": get_backend_metadata(noisy_backend), "result": noisy_result},
        "difference": noisy_result["aggregated_result"] - clean_result["aggregated_result"],
    }
    artifact = build_run_artifact("run_clean_vs_noisy_backend", run_result)
    artifact_path = artifact_path_for_run(artifact["run_id"], output_dir=tmp_path)

    saved_path = save_json_artifact(artifact, artifact_path)

    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["run"]["clean"]["backend"]["name"] == "pennylane"
    assert saved_data["run"]["noisy"]["backend"]["name"] == "noisy"
    assert saved_data["run"]["noisy"]["backend"]["base_backend"] == "pennylane"
    assert "difference" in saved_data["run"]
