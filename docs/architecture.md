# Architecture

qfl-mini is intentionally small. The design keeps each concern in a separate module so the execution, aggregation, reporting, and artifact paths are easy to read independently.

## Module layout

| Module            | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| `circuits.py`     | PennyLane circuit definition and execution                  |
| `client.py`       | Quantum client abstraction (`QuantumClient`)                |
| `coordinator.py`  | Basic multi-round coordination and mean aggregation         |
| `optimization.py` | Parameter and gradient update coordinators                  |
| `reporting.py`    | Human-readable report formatters                            |
| `metadata.py`     | Run ID generation and environment metadata collection       |
| `artifacts.py`    | JSON artifact path resolution and saving                    |

## Execution flow

```text
QuantumClient.run()
  -> circuits.run_single_qubit_expectation(theta)
  -> returns { client_id, theta, result }

Coordinator / ParameterUpdateCoordinator / FiniteDifferenceGradientCoordinator
  -> instantiates temporary QuantumClient objects per round
  -> collects per-client results
  -> computes mean aggregation
  -> computes loss / gradient / next_theta
  -> returns structured round trace
```

## Artifact flow

```text
run_result = coordinator.run_updates(n)

artifact = build_run_artifact(example_name, run_result)
  -> generates run_id from example_name + UTC timestamp
  -> collects environment metadata
  -> wraps run_result under "run" key

artifact_path = artifact_path_for_run(artifact["run_id"])
  -> resolves to runs/<run_id>.json

save_json_artifact(artifact, artifact_path)
  -> creates runs/ directory if needed
  -> writes JSON with 2-space indentation
  -> returns final path
```

## Why the design is intentionally small

- No base classes or plugin registries. There are three coordinator classes; each is self-contained and easy to read from top to bottom.
- No global state. Each coordinator takes clients and hyperparameters at construction time.
- No autograd. The finite-difference gradient is computed with plain arithmetic, making the math visible in the source.
- No external storage. Artifacts are plain JSON files written to `runs/` with standard library `json` and `pathlib`.
- No network layer. Clients and coordinators are all in-process. Networking is a future concern.

The goal is that a new reader can understand the full execution path by reading five to ten files of moderate length.
