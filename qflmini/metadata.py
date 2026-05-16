"""Lightweight reproducibility metadata for qfl-mini artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform as platform_module
import sys
from typing import Any


def collect_environment_metadata() -> dict[str, str]:
    """Collect lightweight software and platform metadata for a run artifact.

    Returns:
        A dictionary describing the Python runtime, platform, and installed
        PennyLane version when it is available.
    """
    try:
        pennylane_version = version("pennylane")
    except PackageNotFoundError:
        pennylane_version = "unknown"

    return {
        "python_version": sys.version.split()[0],
        "platform": platform_module.platform(),
        "system": platform_module.system(),
        "machine": platform_module.machine(),
        "pennylane_version": pennylane_version,
    }


def generate_run_id(example_name: str) -> str:
    """Generate a timestamped run identifier for an example.

    Args:
        example_name: Name of the example producing the artifact.

    Returns:
        A run identifier such as ``run_parameter_update_20260516T203956Z``.

    Raises:
        ValueError: If ``example_name`` is empty or only whitespace.
    """
    clean_name = _clean_example_name(example_name)
    return _run_id_from_datetime(clean_name, datetime.now(timezone.utc))


def build_run_artifact(example_name: str, run_result: dict[str, Any]) -> dict[str, Any]:
    """Build a reproducibility artifact around a run result.

    Args:
        example_name: Name of the example that produced the run.
        run_result: Result dictionary produced by a coordinator.

    Returns:
        A JSON-serializable artifact dictionary with metadata and run results.

    Raises:
        ValueError: If ``example_name`` is empty.
        TypeError: If ``run_result`` is not a dictionary.
    """
    clean_name = _clean_example_name(example_name)
    if not isinstance(run_result, dict):
        raise TypeError("run_result must be a dictionary.")

    created_at = datetime.now(timezone.utc)
    run_id = _run_id_from_datetime(clean_name, created_at)

    return {
        "project": "qfl-mini",
        "artifact_version": "0.1",
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "example": clean_name,
        "environment": collect_environment_metadata(),
        "run": run_result,
    }


def _clean_example_name(example_name: str) -> str:
    clean_name = example_name.strip()
    if not clean_name:
        raise ValueError("example_name must not be empty.")
    return clean_name


def _run_id_from_datetime(example_name: str, created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{example_name}_{timestamp}"
