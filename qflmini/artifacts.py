"""Reproducibility artifact helpers for qfl-mini."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_artifact(data: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON reproducibility artifact and return its final path.

    Args:
        data: JSON-serializable artifact data.
        path: Destination file path.

    Returns:
        The final artifact path.
    """
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path
