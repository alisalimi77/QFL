"""Compare saved qfl-mini JSON artifacts from multiple runs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflmini.comparison import format_artifact_comparison, summarize_artifacts


def main() -> None:
    """Load the provided artifact files and print a plain text comparison table."""
    if len(sys.argv) < 2:
        print(
            "Usage: python examples/compare_artifacts.py"
            " <artifact1.json> [artifact2.json ...]"
        )
        sys.exit(1)

    paths = [Path(arg) for arg in sys.argv[1:]]
    summaries = summarize_artifacts(paths)
    print(format_artifact_comparison(summaries))


if __name__ == "__main__":
    main()
