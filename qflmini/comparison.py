"""Dependency-free artifact loading, summarization, and comparison for qfl-mini."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MAX_RUN_ID_WIDTH = 46
_MAX_MANIFEST_WIDTH = 25
_MAX_MANIFEST_FILE_WIDTH = 40
_MAX_BACKEND_WIDTH = 20


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a saved qfl-mini JSON artifact and return it as a dictionary.

    Args:
        path: Path to the artifact JSON file.

    Returns:
        The artifact as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level JSON value is not an object.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Artifact must be a JSON object. Got {type(raw).__name__}."
        )
    return raw


def summarize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalized summary from a saved qfl-mini artifact.

    Supports both direct run result artifacts (from run_gradient_update.py) and
    manifest-wrapped artifacts (from run_from_manifest.py).

    Args:
        artifact: Artifact dictionary loaded from a JSON file.

    Returns:
        A summary dictionary with keys: run_id, example, experiment, manifest_name,
        manifest_version, manifest_path, manifest_file, backend_name, backend_class,
        num_rounds, final_theta, final_loss. Missing fields are represented as None
        or "unknown".
    """
    run_id = artifact.get("run_id", "unknown")
    example = artifact.get("example", "unknown")
    run = artifact.get("run", {})

    # Distinguish manifest-wrapped (Shape B) from direct (Shape A)
    if "manifest" in run and "result" in run:
        result_data = run.get("result", {})
        manifest_data = run.get("manifest", {})
    else:
        result_data = run
        manifest_data = {}

    # experiment
    if manifest_data.get("experiment"):
        experiment = manifest_data["experiment"]
    elif "gradient_update" in str(example):
        experiment = "gradient_update"
    else:
        experiment = "unknown"

    # manifest metadata
    manifest_name = manifest_data.get("name", "unknown")
    manifest_version = manifest_data.get("manifest_version", "unknown")

    # manifest provenance
    manifest_path_str = run.get("manifest_path", "unknown")
    if manifest_path_str != "unknown":
        manifest_file = Path(manifest_path_str).name
    else:
        manifest_file = "unknown"

    # backend metadata
    backend_data = run.get("backend", {})
    backend_name = backend_data.get("name", "unknown") if isinstance(backend_data, dict) else "unknown"
    backend_class = backend_data.get("class", "unknown") if isinstance(backend_data, dict) else "unknown"

    # num_rounds
    num_rounds = result_data.get("num_rounds", manifest_data.get("num_rounds"))

    # final_theta
    final_theta = result_data.get("final_theta")

    # final_loss — last round's loss
    rounds = result_data.get("rounds", [])
    if rounds and isinstance(rounds[-1], dict) and "loss" in rounds[-1]:
        final_loss = rounds[-1]["loss"]
    else:
        final_loss = None

    return {
        "run_id": run_id,
        "example": example,
        "experiment": experiment,
        "manifest_name": manifest_name,
        "manifest_version": manifest_version,
        "manifest_path": manifest_path_str,
        "manifest_file": manifest_file,
        "backend_name": backend_name,
        "backend_class": backend_class,
        "num_rounds": num_rounds,
        "final_theta": final_theta,
        "final_loss": final_loss,
    }


def summarize_artifacts(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load and summarize multiple saved artifacts.

    Args:
        paths: Ordered list of artifact file paths.

    Returns:
        List of summary dictionaries in the same order as paths.
    """
    return [summarize_artifact(load_artifact(p)) for p in paths]


def format_artifact_comparison(summaries: list[dict[str, Any]]) -> str:
    """Format a list of artifact summaries as a plain text comparison table.

    Args:
        summaries: List of summary dictionaries from summarize_artifact().

    Returns:
        A plain text table string.

    Raises:
        ValueError: If summaries is empty.
    """
    if not summaries:
        raise ValueError("summaries must not be empty.")

    def _trunc(value: Any, max_len: int) -> str:
        s = str(value)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def _fmt_float(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{value:.6f}"

    def _fmt_int(value: Any) -> str:
        if value is None:
            return "n/a"
        return str(int(value))

    headers = ["run_id", "manifest", "manifest_file", "backend", "experiment", "rounds", "final_theta", "final_loss"]

    rows = [
        [
            _trunc(s["run_id"], _MAX_RUN_ID_WIDTH),
            _trunc(s["manifest_name"], _MAX_MANIFEST_WIDTH),
            _trunc(s["manifest_file"], _MAX_MANIFEST_FILE_WIDTH),
            _trunc(s["backend_name"], _MAX_BACKEND_WIDTH),
            str(s["experiment"]),
            _fmt_int(s["num_rounds"]),
            _fmt_float(s["final_theta"]),
            _fmt_float(s["final_loss"]),
        ]
        for s in summaries
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "  "

    def _fmt_row(cells: list[str]) -> str:
        parts = [
            cell.ljust(col_widths[i]) if i < len(cells) - 1 else cell
            for i, cell in enumerate(cells)
        ]
        return sep.join(parts)

    lines = [
        "qfl-mini: artifact comparison",
        "",
        _fmt_row(headers),
    ]
    for row in rows:
        lines.append(_fmt_row(row))

    return "\n".join(lines)
