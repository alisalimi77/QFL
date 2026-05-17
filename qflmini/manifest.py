"""Minimal JSON manifest loading and validation for qfl-mini experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qflmini.backends import ConstantBackend, NoisyBackend, PennyLaneBackend, QuantumBackend

SUPPORTED_MANIFEST_VERSION = "0.1"
SUPPORTED_BACKEND_TYPES = {"pennylane", "constant", "noisy"}


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
        "manifest_version",
        "name",
        "experiment",
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

    manifest_version = manifest["manifest_version"]
    if not isinstance(manifest_version, str) or manifest_version != SUPPORTED_MANIFEST_VERSION:
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
    if experiment != "gradient_update":
        raise ValueError(
            f"Unsupported experiment type: '{experiment}'. Only 'gradient_update' is supported."
        )

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

    backend = validate_backend_config(manifest.get("backend", {"type": "pennylane"}))

    return {
        "manifest_version": SUPPORTED_MANIFEST_VERSION,
        "name": name.strip(),
        "description": description,
        "experiment": str(experiment),
        "backend": backend,
        "num_clients": int(num_clients),
        "num_rounds": int(num_rounds),
        "initial_theta": float(initial_theta),
        "learning_rate": float(learning_rate),
        "target": float(target),
        "epsilon": float(epsilon),
    }


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
