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
    if not example_name:
        raise ValueError("example_name must not be empty.")
    if not isinstance(run_result, dict):
        raise TypeError("run_result must be a dictionary.")

    return {
        "project": "qfl-mini",
        "artifact_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example": example_name,
        "environment": collect_environment_metadata(),
        "run": run_result,
    }
