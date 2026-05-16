"""Minimal JSON manifest loading and validation for qfl-mini experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

    return {
        "experiment": str(experiment),
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
