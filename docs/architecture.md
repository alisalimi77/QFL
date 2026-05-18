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
| `objectives.py`   | Client-specific objective evaluation helpers                                                 |
| `reporting.py`    | Human-readable report formatters                                                             |
| `metadata.py`     | Run ID generation and environment metadata collection                                        |
| `artifacts.py`    | JSON artifact path resolution and saving                                                     |
| `manifest.py`     | Loading, validating, normalizing, and building built-in backend configs from JSON manifests  |
| `comparison.py`   | Loading, summarizing, and formatting saved artifact comparisons with backend details         |

## Execution flow

```text
QuantumClient -> QuantumBackend -> backend implementation

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

Client-specific objective artifacts follow the same artifact wrapper and store
local objective results under the run payload:

```text
QuantumClient -> local target -> local loss -> mean local loss artifact
```

## Manifest flow

```text
JSON manifest file
  -> load_json_manifest()      reads and parses JSON
  -> validate_manifest()       dispatches by experiment type
  -> validate_gradient_update_manifest() or validate_client_objectives_manifest()
       checks types and constraints, normalizes manifest_version, name,
       description, experiment-specific fields, and backend config
  -> build_backend_from_config()  explicitly builds one built-in backend
  -> gradient_update path:
       creates QuantumClient objects with the selected backend
       creates FiniteDifferenceGradientCoordinator
       run_updates(num_rounds)
       format_gradient_update_report()
  -> client_objectives path:
       creates QuantumClient objects with the selected backend
       creates ClientObjective objects from manifest client targets
       evaluate_client_objectives()
       format_client_objectives_report()
  -> build_run_artifact({ manifest_path: posix_path, manifest: config,
                          backend: get_backend_metadata(backend),
                          result: result })
  -> save_json_artifact()
```

The backend builder is intentionally explicit and limited to built-in backend
configs: `pennylane`, `constant`, and `noisy`. It does not use dynamic imports,
a registry, or a plugin system.

## Comparison flow

```text
Saved artifact files
  -> load_artifact()          reads and validates JSON
  -> summarize_artifact()     extracts run_id, experiment, manifest_name,
                              manifest_version, manifest_path, manifest_file,
                              backend_name, backend_class, backend_detail,
                              legacy fields, and experiment-aware metrics
  -> format_artifact_comparison()  produces a plain text table
                              (columns: run_id, manifest, manifest_file,
                               backend, backend_detail, experiment,
                               primary_metric, primary_value,
                               secondary_metric, secondary_value)
```

Backend detail extraction is display-only. It reads metadata already stored in
artifacts, such as `base_backend`, `noise`, and `seed` for `NoisyBackend`, or
`value` for `ConstantBackend`.

Comparison also extracts experiment-aware primary and secondary metrics from
artifact JSON. For example, `gradient_update` compares final loss and final
theta, while `client_objectives` compares mean local loss and aggregated result.

## Why the design is intentionally small

- No broad plugin registries. There are three coordinator classes and a few built-in backend classes; each is self-contained and easy to read from top to bottom.
- No global state. Each coordinator takes clients and hyperparameters at construction time.
- No autograd. The finite-difference gradient is computed with plain arithmetic, making the math visible in the source.
- No external storage. Artifacts are plain JSON files written to `runs/` with standard library `json` and `pathlib`.
- No network layer. Clients and coordinators are all in-process. Networking is a future concern.

The goal is that a new reader can understand the full execution path by reading five to ten files of moderate length.

For a command-by-command walkthrough of the full workflow, see [walkthrough.md](walkthrough.md).
