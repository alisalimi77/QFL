"""Reproducibility artifact helpers for qfl-mini."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def artifact_path_for_run(run_id: str, output_dir: str | Path = "runs") -> Path:
    """Return the JSON artifact path for a run identifier.

    Args:
        run_id: Unique run identifier.
        output_dir: Directory where run artifacts are stored.

    Returns:
        The artifact path for the run.

    Raises:
        ValueError: If ``run_id`` is empty or only whitespace.
    """
    clean_run_id = run_id.strip()
    if not clean_run_id:
        raise ValueError("run_id must not be empty.")

    return Path(output_dir) / f"{clean_run_id}.json"


def save_json_artifact(data: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON reproducibility artifact and return its final path.

    Args:
        data: JSON-serializable artifact data.
        path: Destination file path.

    Returns:
        The final artifact path.

    Raises:
        TypeError: If ``data`` is not a dictionary or ``path`` is not a string
            or ``Path``.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path.")

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path
