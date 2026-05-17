# Concepts

This document explains the terms used throughout qfl-mini.

## Federated Quantum-Classical Workload

A federated quantum-classical workload is a computation where multiple quantum-capable clients execute local quantum circuits, and a classical coordinator aggregates or uses their outputs — without treating the whole computation as one centralized quantum process.

In qfl-mini, this is demonstrated locally using simulated clients. No networking is involved. The term "federated" here refers to the structural pattern: separate local executions coordinated by a shared classical layer.

## Quantum Client

A local execution node. In this prototype, a quantum client is a Python object that owns a local parameter (`theta`) and executes a single-qubit PennyLane circuit when asked.

Later versions could map a client to a different simulator backend, a cloud API, or real hardware — but the coordination interface stays the same.

## Quantum Backend

A backend is the object responsible for running a scalar-theta expectation circuit and returning a float. `QuantumClient` delegates circuit execution to its backend via `backend.run_expectation(theta)`.

Three backend-related objects exist:

- **`PennyLaneBackend`** — the only real quantum backend. Calls `run_single_qubit_expectation` from `circuits.py`. Default backend for all clients.
- **`ConstantBackend`** — a deterministic backend that always returns a fixed value regardless of theta. Intended for tests and demos only; not a quantum simulator.
- **`get_backend_metadata(backend)`** — returns `{"name": ..., "class": ...}` for any backend. Used to record which backend ran a manifest experiment.

The `QuantumBackend` protocol creates a seam for future adapters. It is not a plugin system and adds no new runtime dependencies.

## Classical Coordinator

The coordination layer that:

- asks each client to execute its local circuit
- collects the per-client results
- applies aggregation (currently mean)
- drives repeated rounds or parameter updates

The coordinator does not run quantum circuits itself. It only coordinates.

## Execution Sandbox

An execution sandbox is a controlled environment for trying coordination patterns before adding networking, backend adapters, hardware, or experiment services.

qfl-mini is not just a circuit simulator. It is a place to try and observe coordination ideas at the smallest useful scale.

## Aggregation

The process of combining per-client results into one value. qfl-mini currently implements mean aggregation:

```text
aggregated_result = mean(client_results)
```

This is intentionally simple. Future versions may support weighted aggregation, secure aggregation, or richer strategies.

## Objective and Loss

Some examples compute a simple squared loss to make optimization progress observable:

```text
loss = (aggregated_result - target)^2
```

This is for observability only. It is not full training. The target is a fixed scalar, not a dataset.

## Finite-Difference Gradient

The gradient update demo estimates a gradient numerically using central finite differences:

```text
gradient ≈ (loss(theta + epsilon) - loss(theta - epsilon)) / (2 * epsilon)
next_theta = theta - learning_rate * gradient
```

This requires no automatic differentiation. It is a minimal numerical seed, not FedAvg, not dataset-based training, and not PennyLane autograd.

## Reproducibility Artifact

A timestamped JSON file written by artifact-producing examples. It contains:

- project name and artifact version
- run ID and creation timestamp
- example name
- environment metadata (Python version, platform, PennyLane version)
- full run trace (per-round results, parameters, loss values)

Manifest-run artifacts additionally include backend metadata (`name` and `class`) so the execution backend is traceable from the artifact alone.

Artifacts are designed to be inspectable by humans and machines without any special tooling beyond a JSON reader.

## Artifact Comparison

Artifact comparison is a lightweight way to inspect saved run artifacts side by side. It reads saved JSON files, extracts summary fields (run ID, manifest name, manifest file, backend name, experiment, rounds, final theta, final loss), and prints a plain text table.

It is intentionally plain text and dependency-free. It is not a dashboard, not a plotting tool, and not an experiment tracking server.

## Experiment Manifest

A manifest is a small JSON file that declares the parameters for a supported experiment — number of clients, rounds, initial theta, learning rate, target, and epsilon. Running `run_from_manifest.py` with a manifest is equivalent to editing the Python example directly, but without touching code.

Each manifest includes:

- `manifest_version` — currently `"0.1"`. Required. Controls which validation rules apply.
- `name` — a short human-readable identifier such as `"default-gradient-update"`. Required. Appears in artifact comparison output so runs can be told apart at a glance.
- `description` — a sentence explaining what the manifest does. Optional; defaults to empty string.

In the current version, manifests are limited to finite-difference gradient update experiments. Multiple example manifests live under `examples/manifests/`.

## Run ID

A unique identifier for each run. It is derived from the example name and a UTC timestamp:

```text
run_gradient_update_20260516T205502Z
```

The artifact filename matches the run ID. Repeated runs produce new files and never overwrite previous artifacts.
