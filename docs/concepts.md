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

Backend-related objects:

- **`QuantumBackend`** — a small protocol requiring `run_expectation(theta: float) -> float`.
- **`PennyLaneBackend`** — the only real quantum backend. Calls `run_single_qubit_expectation` from `circuits.py`. Default backend for all clients.
- **`ConstantBackend`** — a deterministic backend that always returns a fixed value regardless of theta. Intended for tests and demos only; not a quantum simulator.
- **`NoisyBackend`** — a deterministic noisy wrapper around another backend. Adds a controlled perturbation to the base result. See "Deterministic Noise" below.
- **`get_backend_metadata(backend)`** — returns a `dict[str, str]` with `name`, `class`, and optional extra fields for each backend type. Used to record which backend ran an experiment.

The `QuantumBackend` protocol creates a seam for future adapters. It is not a plugin system and adds no new runtime dependencies.

JSON manifests can select one of qfl-mini's built-in backend configs:
`pennylane`, `constant`, or `noisy`. This support is explicit and limited; it
does not load arbitrary Python classes, use plugins, or add external backend
SDKs.

## Backend-Aware Manifest

A backend-aware manifest is still a JSON v0.1 manifest. It adds an optional `backend` object using one of the built-in backend configs:

- `{"type": "pennylane"}`
- `{"type": "constant", "value": 0.5}`
- `{"type": "noisy", "base": {"type": "pennylane"}, "noise": 0.05, "seed": 42}`

If `backend` is missing, qfl-mini normalizes the manifest to use `{"type": "pennylane"}`. Backend-aware manifests do not import Python classes, load plugins, or select external quantum SDKs.

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

## Client-Specific Objective

A client-specific objective gives each quantum client its own local target. The client runs its local circuit, then qfl-mini computes:

```text
client_loss = (client_result - client_target)^2
```

The `run_client_objectives.py` example reports each client result, each local target, each local loss, the aggregated result, and the mean local loss. The same evaluation can be run from `examples/manifests/client_objectives.json`. This demonstrates different local objective contexts across clients. It is not dataset training, not FedAvg, and not full QFL training.

## Scalar FedAvg

Scalar FedAvg is qfl-mini's minimal FedAvg-style loop over one scalar parameter. The server owns a global `theta`, each client computes a local finite-difference update from its own local target, and the server averages the local updated theta values:

```text
global theta -> client local updates -> mean aggregation -> next global theta
```

The trace records every round, every client result, local loss, finite-difference gradient, local updated theta, aggregation input, and next global theta. This is not FedAvg over model weights, not dataset training, and not a full federated learning framework.

## Deterministic Noise

`NoisyBackend` wraps any base backend and applies a deterministic perturbation to its output:

```text
noise_value = noise * sin(theta + seed)
result = clip(base_result + noise_value, -1.0, 1.0)
```

The perturbation is fully determined by `theta`, `noise`, and `seed` — no random module is used. The output is clipped to `[-1.0, 1.0]` to stay within valid expectation-value range.

This is a controlled demo backend, not a hardware noise model. It is useful for comparing clean and noisy execution in a reproducible way: given the same inputs, the noisy result is always the same. The `run_clean_vs_noisy_backend.py` example demonstrates this pattern.

The same deterministic noisy backend can also be selected from a JSON manifest:

```json
"backend": {
  "type": "noisy",
  "base": {
    "type": "pennylane"
  },
  "noise": 0.05,
  "seed": 42
}
```

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
For backend-aware manifest runs, artifacts also include the normalized manifest backend config and backend-specific metadata such as `value` for `ConstantBackend` or `base_backend`, `noise`, and `seed` for `NoisyBackend`.

Artifacts are designed to be inspectable by humans and machines without any special tooling beyond a JSON reader.

## Artifact Comparison

Artifact comparison is a lightweight way to inspect saved run artifacts side by side. It reads saved JSON files, extracts summary fields, and prints a plain text table. Comparison is experiment-aware: `gradient_update` uses `final_loss` and `final_theta`, `client_objectives` uses `mean_local_loss` and `aggregated_result`, and direct `scalar_fedavg` artifacts use `final_mean_local_loss` and `final_theta`. This avoids forcing all experiments into gradient-specific columns.

It is intentionally plain text and dependency-free. It is not a dashboard, not a plotting tool, and not an experiment tracking server.

## Experiment Manifest

A manifest is a small JSON file that declares the parameters for a supported experiment. qfl-mini currently supports `gradient_update` and `client_objectives` manifests. Running `run_from_manifest.py` with a manifest is equivalent to editing the Python example directly, but without touching code.

Each manifest includes:

- `manifest_version` — currently `"0.1"`. Required. Controls which validation rules apply.
- `name` — a short human-readable identifier such as `"default-gradient-update"`. Required. Appears in artifact comparison output so runs can be told apart at a glance.
- `description` — a sentence explaining what the manifest does. Optional; defaults to empty string.
- `experiment` — currently either `"gradient_update"` or `"client_objectives"`.
- `backend` — optional built-in backend config. If omitted, it defaults to `{"type": "pennylane"}`.

For `gradient_update`, manifests include fields such as `num_clients`, `num_rounds`, `initial_theta`, `learning_rate`, `target`, and `epsilon`. For `client_objectives`, manifests include a `clients` list where each client has `client_id`, `theta`, and `target`.

Backend configs are limited to `pennylane`, `constant`, and `noisy`; there are no arbitrary imports or plugin paths in manifest files. Multiple example manifests live under `examples/manifests/`.

## Run ID

A unique identifier for each run. It is derived from the example name and a UTC timestamp:

```text
run_gradient_update_20260516T205502Z
```

The artifact filename matches the run ID. Repeated runs produce new files and never overwrite previous artifacts.
