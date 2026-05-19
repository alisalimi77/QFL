"""Minimal JSON manifest loading and validation for qfl-mini experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qflmini.backends import (
    ConstantBackend,
    NoisyBackend,
    PennyLaneBackend,
    QuantumBackend,
)

SUPPORTED_MANIFEST_VERSION = "0.1"
SUPPORTED_BACKEND_TYPES = {"pennylane", "constant", "noisy"}
SUPPORTED_AGGREGATION_TYPES = {"mean"}
SUPPORTED_EXPERIMENTS = {"gradient_update", "client_objectives", "scalar_fedavg"}


def load_json_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON manifest file and return it as a dictionary.

    Args:
        path: Path to the JSON manifest file.

    Returns:
        The manifest as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level JSON value is not an object.
    """
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Manifest must be a JSON object. Got {type(raw).__name__}."
        )
    return raw


def _validate_common_manifest_fields(
    manifest: dict[str, Any],
    expected_experiment: str,
) -> dict[str, Any]:
    """Validate fields shared by all qfl-mini manifest types."""
    for field in ("manifest_version", "name", "experiment"):
        if field not in manifest:
            raise ValueError(f"Manifest is missing required field: '{field}'.")

    manifest_version = manifest["manifest_version"]
    if (
        not isinstance(manifest_version, str)
        or manifest_version != SUPPORTED_MANIFEST_VERSION
    ):
        raise ValueError(
            f"Unsupported manifest_version: expected '{SUPPORTED_MANIFEST_VERSION}', "
            f"got '{manifest_version}'."
        )

    name = manifest["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string.")

    description_raw = manifest.get("description")
    if description_raw is None:
        description = ""
    elif not isinstance(description_raw, str):
        raise ValueError("'description' must be a string if provided.")
    else:
        description = description_raw.strip()

    experiment = manifest["experiment"]
    if experiment != expected_experiment:
        raise ValueError(
            f"Unsupported experiment type: '{experiment}'. "
            f"Expected '{expected_experiment}'."
        )

    backend = validate_backend_config(manifest.get("backend", {"type": "pennylane"}))

    return {
        "manifest_version": SUPPORTED_MANIFEST_VERSION,
        "name": name.strip(),
        "description": description,
        "experiment": expected_experiment,
        "backend": backend,
    }


def validate_gradient_update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a gradient update manifest and return a normalized config.

    Args:
        manifest: Raw manifest dictionary.

    Returns:
        Normalized configuration dictionary with correct types.

    Raises:
        ValueError: If any required field is missing or invalid.
    """
    required_fields = (
        "num_clients",
        "num_rounds",
        "initial_theta",
        "learning_rate",
        "target",
        "epsilon",
    )
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest is missing required field: '{field}'.")

    common = _validate_common_manifest_fields(manifest, "gradient_update")

    num_clients = manifest["num_clients"]
    if not isinstance(num_clients, int):
        raise ValueError("'num_clients' must be an integer.")
    if num_clients < 1:
        raise ValueError("'num_clients' must be at least 1.")

    num_rounds = manifest["num_rounds"]
    if not isinstance(num_rounds, int):
        raise ValueError("'num_rounds' must be an integer.")
    if num_rounds < 1:
        raise ValueError("'num_rounds' must be at least 1.")

    initial_theta = manifest["initial_theta"]
    if not isinstance(initial_theta, (int, float)):
        raise ValueError("'initial_theta' must be a number.")

    learning_rate = manifest["learning_rate"]
    if not isinstance(learning_rate, (int, float)):
        raise ValueError("'learning_rate' must be a number.")
    if learning_rate <= 0:
        raise ValueError("'learning_rate' must be positive.")

    target = manifest["target"]
    if not isinstance(target, (int, float)):
        raise ValueError("'target' must be a number.")

    epsilon = manifest["epsilon"]
    if not isinstance(epsilon, (int, float)):
        raise ValueError("'epsilon' must be a number.")
    if epsilon <= 0:
        raise ValueError("'epsilon' must be positive.")

    return {
        **common,
        "num_clients": int(num_clients),
        "num_rounds": int(num_rounds),
        "initial_theta": float(initial_theta),
        "learning_rate": float(learning_rate),
        "target": float(target),
        "epsilon": float(epsilon),
    }


def validate_client_objectives_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a client objectives manifest and return a normalized config.

    Args:
        manifest: Raw manifest dictionary.

    Returns:
        Normalized configuration dictionary with client local targets.

    Raises:
        ValueError: If any required field is missing or invalid.
    """
    if "clients" not in manifest:
        raise ValueError("Manifest is missing required field: 'clients'.")

    common = _validate_common_manifest_fields(manifest, "client_objectives")

    clients = manifest["clients"]
    if not isinstance(clients, list):
        raise ValueError("'clients' must be a list.")
    if not clients:
        raise ValueError("'clients' must not be empty.")

    normalized_clients = []
    for index, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError(f"'clients[{index}]' must be an object.")
        for field in ("client_id", "theta", "target"):
            if field not in client:
                raise ValueError(f"'clients[{index}].{field}' is required.")

        client_id = client["client_id"]
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError(
                f"'clients[{index}].client_id' must be a non-empty string."
            )

        theta = client["theta"]
        if not isinstance(theta, (int, float)):
            raise ValueError(f"'clients[{index}].theta' must be a number.")

        target = client["target"]
        if not isinstance(target, (int, float)):
            raise ValueError(f"'clients[{index}].target' must be a number.")

        normalized_clients.append(
            {
                "client_id": client_id.strip(),
                "theta": float(theta),
                "target": float(target),
            }
        )

    return {
        **common,
        "clients": normalized_clients,
    }


def validate_aggregation_config(config: Any) -> dict[str, Any]:
    """Validate and normalize an aggregation configuration.

    Only mean aggregation is supported for now. The explicit aggregation block
    is a small scenario-runtime foundation, not a registry or plugin system.
    """
    if not isinstance(config, dict):
        raise ValueError("'aggregation' must be an object.")

    aggregation_type = config.get("type")
    if not isinstance(aggregation_type, str) or not aggregation_type:
        raise ValueError("'aggregation.type' is required and must be a string.")
    if aggregation_type not in SUPPORTED_AGGREGATION_TYPES:
        raise ValueError(
            f"Unsupported aggregation type: '{aggregation_type}'. "
            "Supported types are: mean."
        )

    return {"type": "mean"}


def validate_scalar_fedavg_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a scalar FedAvg manifest and return a normalized config."""
    required_fields = (
        "num_rounds",
        "initial_theta",
        "learning_rate",
        "epsilon",
        "clients",
    )
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest is missing required field: '{field}'.")

    common = _validate_common_manifest_fields(manifest, "scalar_fedavg")
    aggregation = validate_aggregation_config(
        manifest.get("aggregation", {"type": "mean"})
    )

    num_rounds = manifest["num_rounds"]
    if not isinstance(num_rounds, int):
        raise ValueError("'num_rounds' must be an integer.")
    if num_rounds < 1:
        raise ValueError("'num_rounds' must be at least 1.")

    initial_theta = manifest["initial_theta"]
    if not isinstance(initial_theta, (int, float)):
        raise ValueError("'initial_theta' must be a number.")

    learning_rate = manifest["learning_rate"]
    if not isinstance(learning_rate, (int, float)):
        raise ValueError("'learning_rate' must be a number.")
    if learning_rate <= 0:
        raise ValueError("'learning_rate' must be positive.")

    epsilon = manifest["epsilon"]
    if not isinstance(epsilon, (int, float)):
        raise ValueError("'epsilon' must be a number.")
    if epsilon <= 0:
        raise ValueError("'epsilon' must be positive.")

    clients = manifest["clients"]
    if not isinstance(clients, list):
        raise ValueError("'clients' must be a list.")
    if not clients:
        raise ValueError("'clients' must not be empty.")

    normalized_clients = []
    for index, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError(f"'clients[{index}]' must be an object.")
        for field in ("client_id", "target"):
            if field not in client:
                raise ValueError(f"'clients[{index}].{field}' is required.")

        client_id = client["client_id"]
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError(
                f"'clients[{index}].client_id' must be a non-empty string."
            )

        target = client["target"]
        if not isinstance(target, (int, float)):
            raise ValueError(f"'clients[{index}].target' must be a number.")

        normalized_clients.append(
            {
                "client_id": client_id.strip(),
                "target": float(target),
            }
        )

    return {
        **common,
        "aggregation": aggregation,
        "num_rounds": int(num_rounds),
        "initial_theta": float(initial_theta),
        "learning_rate": float(learning_rate),
        "epsilon": float(epsilon),
        "clients": normalized_clients,
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate any supported qfl-mini manifest and return normalized config."""
    experiment = manifest.get("experiment")
    if experiment == "gradient_update":
        return validate_gradient_update_manifest(manifest)
    if experiment == "client_objectives":
        return validate_client_objectives_manifest(manifest)
    if experiment == "scalar_fedavg":
        return validate_scalar_fedavg_manifest(manifest)
    supported = ", ".join(sorted(SUPPORTED_EXPERIMENTS))
    raise ValueError(
        f"Unsupported experiment type: '{experiment}'. "
        f"Supported experiments are: {supported}."
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate any supported qfl-mini manifest from a JSON file."""
    raw = load_json_manifest(path)
    return validate_manifest(raw)


def load_gradient_update_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a gradient update manifest from a JSON file.

    Args:
        path: Path to the JSON manifest file.

    Returns:
        Normalized configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the manifest is invalid.
    """
    raw = load_json_manifest(path)
    return validate_gradient_update_manifest(raw)


def load_client_objectives_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a client objectives manifest from a JSON file."""
    raw = load_json_manifest(path)
    return validate_client_objectives_manifest(raw)


def load_scalar_fedavg_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a scalar FedAvg manifest from a JSON file."""
    raw = load_json_manifest(path)
    return validate_scalar_fedavg_manifest(raw)


def validate_backend_config(config: Any) -> dict[str, Any]:
    """Validate and normalize a built-in backend configuration.

    Supported backend types are ``pennylane``, ``constant``, and ``noisy``.
    This function does not support arbitrary imports or plugin loading.
    """
    if not isinstance(config, dict):
        raise ValueError("'backend' must be an object.")

    backend_type = config.get("type")
    if not isinstance(backend_type, str) or not backend_type:
        raise ValueError("'backend.type' is required and must be a string.")
    if backend_type not in SUPPORTED_BACKEND_TYPES:
        raise ValueError(
            f"Unsupported backend type: '{backend_type}'. "
            "Supported types are: pennylane, constant, noisy."
        )

    if backend_type == "pennylane":
        return {"type": "pennylane"}

    if backend_type == "constant":
        if "value" not in config:
            raise ValueError("'backend.value' is required for constant backend.")
        value = config["value"]
        if not isinstance(value, (int, float)):
            raise ValueError("'backend.value' must be a number.")
        return {"type": "constant", "value": float(value)}

    if "base" not in config:
        raise ValueError("'backend.base' is required for noisy backend.")
    if "noise" not in config:
        raise ValueError("'backend.noise' is required for noisy backend.")

    noise = config["noise"]
    if not isinstance(noise, (int, float)):
        raise ValueError("'backend.noise' must be a number.")
    if noise < 0:
        raise ValueError("'backend.noise' must be >= 0.")

    seed = config.get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError("'backend.seed' must be an integer if provided.")

    return {
        "type": "noisy",
        "base": validate_backend_config(config["base"]),
        "noise": float(noise),
        "seed": int(seed),
    }


def build_backend_from_config(config: dict[str, Any]) -> QuantumBackend:
    """Build one of qfl-mini's built-in backends from validated config."""
    backend_type = config.get("type")

    if backend_type == "pennylane":
        return PennyLaneBackend()
    if backend_type == "constant":
        return ConstantBackend(config["value"])
    if backend_type == "noisy":
        return NoisyBackend(
            base_backend=build_backend_from_config(config["base"]),
            noise=config["noise"],
            seed=config["seed"],
        )

    raise ValueError(f"Unsupported backend type: '{backend_type}'.")
