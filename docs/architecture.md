# Architecture

qfl-mini is intentionally small. The design keeps each concern in a separate module so the execution, aggregation, reporting, and artifact paths are easy to read independently.

## Module layout

| Module            | Role                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------- |
| `circuits.py`     | PennyLane circuit definition and execution                                                   |
| `backends.py`     | Backend protocol, `PennyLaneBackend`, `ConstantBackend`, `NoisyBackend`, and metadata helper |
| `client.py`       | Quantum client abstraction (`QuantumClient`)                                                 |
| `coordinator.py`  | Basic multi-round coordination and mean aggregation                                          |
| `optimization.py` | Parameter and gradient update coordinators                                                   |
| `reporting.py`    | Human-readable report formatters                                                             |
| `metadata.py`     | Run ID generation and environment metadata collection                                        |
| `artifacts.py`    | JSON artifact path resolution and saving                                                     |
| `manifest.py`     | Loading and validating minimal JSON experiment manifests                                     |
| `comparison.py`   | Loading, summarizing, and formatting saved artifact comparisons                              |

## Execution flow

```text
QuantumClient.run()
  -> backend.run_expectation(theta)  (default: PennyLaneBackend)
  -> circuits.run_single_qubit_expectation(theta)
  -> returns { client_id, theta, result }

Coordinator / ParameterUpdateCoordinator / FiniteDifferenceGradientCoordinator
  -> instantiates temporary QuantumClient objects per round, preserving backend
  -> collects per-client results
  -> computes mean aggregation
  -> computes loss / gradient / next_theta
  -> returns structured round trace
```

The backend interface (`QuantumBackend`) is a small seam between `QuantumClient` and the circuit implementation. It is not a plugin system and adds no new runtime dependencies. `PennyLaneBackend` is the only real quantum backend. `ConstantBackend` and `NoisyBackend` are deterministic support backends for tests and controlled examples.

```text
PennyLaneBackend -> NoisyBackend wrapper -> QuantumClient -> Coordinator -> artifact
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

The clean-vs-noisy backend demo stores both backend metadata blocks under the
artifact run payload:

```text
run.clean.backend -> get_backend_metadata(PennyLaneBackend())
run.noisy.backend -> get_backend_metadata(NoisyBackend(PennyLaneBackend(), ...))
```

## Manifest flow

```text
JSON manifest file
  -> load_json_manifest()      reads and parses JSON
  -> validate_gradient_update_manifest()  checks types and constraints,
       normalizes manifest_version, name, description, and numeric fields
  -> returns normalized config dict

config dict
  -> creates QuantumClient objects
  -> creates FiniteDifferenceGradientCoordinator
  -> run_updates(num_rounds)
  -> format_gradient_update_report()
  -> build_run_artifact({ manifest_path: posix_path, manifest: config,
                          backend: get_backend_metadata(clients[0].backend),
                          result: update_result })
  -> save_json_artifact()
```

## Comparison flow

```text
Saved artifact files
  -> load_artifact()          reads and validates JSON
  -> summarize_artifact()     extracts run_id, experiment, manifest_name,
                              manifest_version, manifest_path, manifest_file,
                              backend_name, backend_class,
                              num_rounds, final_theta, final_loss
  -> format_artifact_comparison()  produces a plain text table
                              (columns: run_id, manifest, manifest_file,
                               backend, experiment, rounds, final_theta, final_loss)
```

## Why the design is intentionally small

- No base classes or plugin registries. There are three coordinator classes and one backend class; each is self-contained and easy to read from top to bottom.
- No global state. Each coordinator takes clients and hyperparameters at construction time.
- No autograd. The finite-difference gradient is computed with plain arithmetic, making the math visible in the source.
- No external storage. Artifacts are plain JSON files written to `runs/` with standard library `json` and `pathlib`.
- No network layer. Clients and coordinators are all in-process. Networking is a future concern.

The goal is that a new reader can understand the full execution path by reading five to ten files of moderate length.

For a command-by-command walkthrough of the full workflow, see [walkthrough.md](walkthrough.md).
