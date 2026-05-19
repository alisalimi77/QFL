# Walkthrough

This document walks through the current qfl-mini workflow from setup to artifact inspection.
All commands assume you have cloned the repository and are running from the repository root.

---

## 1. Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

`requirements.txt` contains the runtime dependencies (PennyLane and its stack).
`requirements-dev.txt` adds pytest for running the test suite.
A virtual environment is recommended to keep dependencies isolated.

---

## 2. Run the smallest federated quantum workload

```bash
python examples/run_two_clients.py
```

Two local quantum clients each run a single-qubit PennyLane circuit parametrised by a scalar `theta`.
The coordinator collects their results and applies mean aggregation.
No artifact is saved.

Expected output (exact values depend on PennyLane version):

```text
qfl-mini: federated quantum workload demo

Client results:
- client_1 | theta=0.2 | result=0.980067
- client_2 | theta=0.8 | result=0.696707

Aggregated result:
0.838387
```

---

## 3. Run multiple rounds

```bash
python examples/run_multi_round.py
```

Three coordination rounds are executed. The same pair of clients runs in each round.
A JSON artifact is saved under `runs/` with a timestamped filename that includes a unique run ID.

```text
qfl-mini: multi-round federated quantum workload demo

Rounds:
- round 1 | aggregated_result=0.838387
- round 2 | aggregated_result=0.838387
- round 3 | aggregated_result=0.838387

Final aggregated result:
0.838387
Saved artifact: runs/run_multi_round_<timestamp>.json
```

Generated artifacts are ignored by git via `.gitignore`.
Running the same example twice produces two separate files that never overwrite each other.

---

## 4. Run a parameter update trace

```bash
python examples/run_parameter_update.py
```

A shared scalar parameter `theta` is updated each round using the rule:

```text
next_theta = theta - learning_rate * aggregated_result
```

A simple squared loss is tracked for observability:

```text
loss = (aggregated_result - target)^2
```

This is not full training. There is no dataset, no labels, and no convergence guarantee.
The trace shows how theta and loss evolve round by round.

```text
qfl-mini: parameter update demo

Rounds:
- round 1 | theta=0.500000 | aggregated_result=0.877583 | target=0.000000 | loss=0.770152 | next_theta=0.412242
- round 2 | theta=0.412242 | aggregated_result=0.921722 | target=0.000000 | loss=0.849571 | next_theta=0.320070
- round 3 | theta=0.320070 | aggregated_result=0.949477 | target=0.000000 | loss=0.901508 | next_theta=0.225122

Final theta:
0.225122
Saved artifact: runs/run_parameter_update_<timestamp>.json
```

---

## 5. Run finite-difference gradient updates

```bash
python examples/run_gradient_update.py
```

Each round estimates a gradient using central finite differences:

```text
gradient ≈ (loss(theta + epsilon) - loss(theta - epsilon)) / (2 * epsilon)
```

The update rule is then:

```text
next_theta = theta - learning_rate * gradient
```

No automatic differentiation is used. The gradient computation is plain arithmetic visible in `optimization.py`.
This is still a minimal demo, not a full optimizer framework.

```text
qfl-mini: finite-difference gradient update demo

Rounds:
- round 1 | theta=0.500000 | loss=0.770151 | gradient=-0.841470 | next_theta=0.584147
- round 2 | theta=0.584147 | loss=0.695861 | gradient=-0.920083 | next_theta=0.676155
- round 3 | theta=0.676155 | loss=0.608376 | gradient=-0.976226 | next_theta=0.773778

Final theta:
0.773778
Saved artifact: runs/run_gradient_update_<timestamp>.json
```

---

## 6. Run from a JSON manifest

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
```

The manifest file declares the experiment parameters:

```json
{
  "manifest_version": "0.1",
  "name": "default-gradient-update",
  "description": "Default finite-difference gradient update experiment.",
  "experiment": "gradient_update",
  "num_clients": 2,
  "num_rounds": 3,
  "initial_theta": 0.5,
  "learning_rate": 0.1,
  "target": 0.0,
  "epsilon": 0.001
}
```

This produces the same output as running `run_gradient_update.py` directly,
but without editing any Python code. The artifact records the manifest path and
manifest config alongside the run result and backend metadata.

Current constraints:

- supported experiments are `gradient_update`, `client_objectives`, and `scalar_fedavg`
- JSON only, no YAML
- backend configs are limited to built-in `pennylane`, `constant`, and `noisy`
- no arbitrary imports, plugins, or external quantum SDK adapters

---

## 7. Try multiple manifests

Eight example manifests are provided. Some change optimizer parameters, two select deterministic built-in backends, one runs client-specific objectives, and one runs scalar FedAvg:

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
python examples/run_from_manifest.py examples/manifests/gradient_update_low_lr.json
python examples/run_from_manifest.py examples/manifests/gradient_update_target_half.json
python examples/run_from_manifest.py examples/manifests/gradient_update_more_rounds.json
python examples/run_from_manifest.py examples/manifests/gradient_update_noisy.json
python examples/run_from_manifest.py examples/manifests/gradient_update_constant.json
python examples/run_from_manifest.py examples/manifests/client_objectives.json
python examples/run_from_manifest.py examples/manifests/scalar_fedavg.json
```

| Manifest                             | Name                       | Backend             | What changes                   |
| ------------------------------------ | -------------------------- | ------------------- | ------------------------------ |
| `gradient_update.json`               | `default-gradient-update`  | `pennylane` default | Default settings               |
| `gradient_update_low_lr.json`        | `low-learning-rate`        | `pennylane` default | Smaller learning rate          |
| `gradient_update_target_half.json`   | `target-half`              | `pennylane` default | Non-zero target (0.5)          |
| `gradient_update_more_rounds.json`   | `more-rounds`              | `pennylane` default | Five rounds instead of three   |
| `gradient_update_noisy.json`         | `noisy-gradient-update`    | `noisy`             | Deterministic noisy backend    |
| `gradient_update_constant.json`      | `constant-gradient-update` | `constant`          | Deterministic constant backend |
| `client_objectives.json`             | `client-objectives-demo`   | `pennylane`         | Client-specific objectives     |
| `scalar_fedavg.json`                 | `scalar-fedavg-demo`       | `pennylane`         | Scalar FedAvg + mean aggregation |

Each run produces a separate artifact under `runs/`.

---

## 8. Run backend-aware manifests

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update_noisy.json
python examples/run_from_manifest.py examples/manifests/gradient_update_constant.json
```

The noisy manifest uses `NoisyBackend` around `PennyLaneBackend` with
deterministic noise. The constant manifest uses `ConstantBackend`. Both are
still JSON manifest v0.1 gradient update experiments, and both artifacts record
the normalized manifest backend config plus backend metadata.

You can compare a default, noisy, and constant manifest run:

```bash
python examples/compare_artifacts.py runs/<default>.json runs/<noisy>.json runs/<constant>.json
```

Replace the angle-bracket placeholders with the real filenames printed by the
example commands. The comparison table includes `backend` and `backend_detail`
columns, so backend differences are visible without opening each artifact.

---

## 9. Compare artifacts

After generating at least two artifacts, compare them:

```bash
python examples/compare_artifacts.py runs/<artifact-a>.json runs/<artifact-b>.json
```

Replace `<artifact-a>` and `<artifact-b>` with the actual filenames printed after each run.
The comparison table shows key fields side by side:

```text
qfl-mini: artifact comparison

run_id                                          manifest                 manifest_file           backend    backend_detail  experiment         primary_metric   primary_value  secondary_metric   secondary_value
run_from_manifest_gradient_update_...           default-gradient-update  gradient_update.json    pennylane  -               gradient_update    final_loss       0.608376       final_theta        0.773778
run_from_manifest_client_objectives_...         client-objectives-demo   client_objectives.json  pennylane  -               client_objectives  mean_local_loss  0.499612       aggregated_result  0.838387
```

The `primary_metric` and `secondary_metric` columns depend on the experiment type. This is a plain text, dependency-free helper. It is not a dashboard or experiment tracking server.

---

## 10. Inspect an artifact

```bash
python -m json.tool runs/<artifact>.json
```

Artifacts include:

- `run_id` and creation timestamp
- `example` name
- environment metadata (Python version, platform, PennyLane version)
- manifest path and manifest config (for manifest-run artifacts)
- backend metadata (`name`, `class`, and backend-specific details when available)
- full run result (per-round traces, final theta, final loss)

Abbreviated shape of a manifest-run artifact:

```json
{
  "project": "qfl-mini",
  "artifact_version": "0.1",
  "run_id": "run_from_manifest_gradient_update_20260517T075455Z",
  "example": "run_from_manifest_gradient_update",
  "environment": {
    "python_version": "3.12.6",
    "platform": "...",
    "pennylane_version": "0.45.0"
  },
  "run": {
    "manifest_path": "examples/manifests/gradient_update.json",
    "manifest": {
      "manifest_version": "0.1",
      "name": "noisy-gradient-update",
      "backend": {
        "type": "noisy",
        "base": {
          "type": "pennylane"
        },
        "noise": 0.05,
        "seed": 42
      }
    },
    "backend": {
      "name": "noisy",
      "class": "NoisyBackend",
      "base_backend": "pennylane",
      "noise": "0.05",
      "seed": "42"
    },
    "result": {
      "final_theta": 0.75274294792475
    }
  }
}
```

---

## 11. Run the custom backend demo

```bash
python examples/run_custom_backend.py
```

This example injects a `ConstantBackend` directly in Python — no manifest, no artifact.
`ConstantBackend` always returns a fixed value regardless of `theta`.
It is provided for tests and demos only; it is not a real quantum backend.

```text
qfl-mini: custom backend demo

Client results:
- client_1 | theta=0.0 | result=0.200000
- client_2 | theta=0.0 | result=0.600000

Aggregated result:
0.400000
```

Backend injection requires no manifest changes. Pass any object that implements
`run_expectation(theta: float) -> float` to `QuantumClient(backend=...)`.

---

## 12. Run the clean-vs-noisy backend demo

```bash
python examples/run_clean_vs_noisy_backend.py
```

This example runs two coordinated rounds side by side using the same clients and thetas:
one with `PennyLaneBackend` (clean) and one with `NoisyBackend` wrapping `PennyLaneBackend`.

`NoisyBackend` applies a deterministic perturbation to the base result:

```text
noise_value = noise * sin(theta + seed)
result = clip(base_result + noise_value, -1.0, 1.0)
```

No randomness is used. Given the same `theta`, `noise`, and `seed`, the noisy result is always identical.

Expected output (values depend on PennyLane version):

```text
qfl-mini: clean vs noisy backend demo

Clean backend:
- backend=pennylane
- aggregated_result=0.838387

Noisy backend:
- backend=noisy
- aggregated_result=0.790807

Difference:
-0.047580
Saved artifact: runs/run_clean_vs_noisy_backend_<timestamp>.json
```

The artifact records backend metadata for both the clean and noisy backends so runs are fully traceable.

---

## 13. Run client-specific objectives

```bash
python examples/run_client_objectives.py
```

This example gives each client its own local target. Each client runs its local
quantum circuit, computes a local squared loss against its own target, and the
run reports both mean aggregation and mean local loss. This is not
dataset-based training and not FedAvg.

```text
qfl-mini: client-specific objective demo

Client objectives:
- client_1 | theta=0.2 | target=0.0 | result=0.980067 | loss=0.960531
- client_2 | theta=0.8 | target=0.5 | result=0.696707 | loss=0.038693

Aggregated result:
0.838387

Mean local loss:
0.499612
Saved artifact: runs/run_client_objectives_<timestamp>.json
```

The artifact stores `client_objectives`, `aggregated_result`, and
`mean_local_loss` under the top-level `run` payload.

---

## 14. Run client-specific objectives from a manifest

```bash
python examples/run_from_manifest.py examples/manifests/client_objectives.json
```

This runs the same kind of local objective evaluation from JSON. The manifest
declares a list of clients, and each client entry has `client_id`, `theta`, and
`target`. There are still no datasets, no FedAvg fields, and no client-specific
backend configs. All clients use the manifest-level backend.

The saved artifact records:

- `manifest_path`
- normalized manifest config, including `clients`
- backend metadata
- `result.client_objectives`
- `result.mean_local_loss`

Client-objective artifacts can be compared with gradient artifacts. In the
plain text comparison table, `primary_metric` is `mean_local_loss` and
`secondary_metric` is `aggregated_result` for `client_objectives` runs.

---

## 15. Run transparent scalar FedAvg

```bash
python examples/run_scalar_fedavg.py
```

This demo is a minimal FedAvg-style loop over one scalar global parameter. The
server sends `global_theta` to each client context, each client computes a
local finite-difference gradient against its own target, and the server averages
the local updated theta values.

This is trace-first: the artifact records every round, every client update,
local losses, finite-difference losses, gradients, local updated theta values,
the aggregation inputs, and the next global theta. It is not dataset-based
training, not model-weight FedAvg, and not a full federated learning framework.

```text
qfl-mini: transparent scalar FedAvg demo

Rounds:
- round 1 | global_theta=0.500000 | mean_local_loss=0.456360 | next_global_theta=0.560176
- round 2 | global_theta=0.560176 | mean_local_loss=0.419102 | next_global_theta=0.623634
- round 3 | global_theta=0.623634 | mean_local_loss=0.378076 | next_global_theta=0.689247

Client updates, final round:
- client_1 | target=0.0 | result=0.811761 | loss=0.658957 | gradient=-0.948119 | local_next_theta=0.718446
- client_2 | target=0.5 | result=0.811761 | loss=0.097195 | gradient=-0.364130 | local_next_theta=0.660047

Final theta:
0.689247
Saved artifact: runs/run_scalar_fedavg_<timestamp>.json
```

Direct scalar FedAvg artifacts can be compared with other artifacts. In the
comparison table, `primary_metric` is `final_mean_local_loss` and
`secondary_metric` is `final_theta`.

---

## 16. Run scalar FedAvg from a manifest

```bash
python examples/run_from_manifest.py examples/manifests/scalar_fedavg.json
```

This is the scenario-defined form of the same scalar FedAvg path. The manifest
declares clients, local targets, backend, aggregation, number of rounds,
initial theta, learning rate, and epsilon:

```json
{
  "experiment": "scalar_fedavg",
  "backend": {
    "type": "pennylane"
  },
  "aggregation": {
    "type": "mean"
  }
}
```

The execution is still local-only. The important step is structural: the
scenario describes the aggregation policy, and the artifact records the
aggregation method, client-tagged inputs, and output for every round.

```text
qfl-mini: transparent scalar FedAvg demo

Rounds:
- round 1 | global_theta=0.500000 | mean_local_loss=0.456360 | next_global_theta=0.560176
- round 2 | global_theta=0.560176 | mean_local_loss=0.419102 | next_global_theta=0.623634
- round 3 | global_theta=0.623634 | mean_local_loss=0.378076 | next_global_theta=0.689247

Final theta:
0.689247
Saved artifact: runs/run_from_manifest_scalar_fedavg_<timestamp>.json
```

In the saved artifact, each round contains:

```json
"aggregation": {
  "method": "mean",
  "inputs": [
    {
      "client_id": "client_1",
      "local_next_theta": 0.584147
    }
  ],
  "next_global_theta": 0.560176
}
```

This is not a full Scenario Contract v0.2 yet, but it is shaped toward the
same-scenario local-to-real direction.

---

## 17. Run checks

```bash
python -m compileall qflmini examples
pytest
```

`compileall` catches import errors and syntax problems across all modules and examples.
`pytest` runs the full test suite. GitHub Actions runs both on every push and pull request.

---

## 18. Where this leaves the project

```text
Phase 0: minimal execution                            [done]
Phase 1: parameter/loss/gradient traces               [done]
Phase 2: manifest/artifact/comparison workflow        [done/active]
Phase 3: backend abstraction                          [active]
Phase 4: deterministic backend realism                [active]
Phase 5: client-specific objectives + manifest support [active]
Phase 6: scenario-defined transparent scalar FedAvg    [active]
```

**What the project can do now:**

- run local quantum clients with PennyLane circuit execution
- aggregate client results (mean)
- run multi-round coordination
- track loss across rounds
- run finite-difference gradient updates
- run experiments from JSON manifests
- choose a built-in backend from JSON manifests
- run backend-aware client-objective manifests
- save timestamped reproducibility artifacts
- compare artifacts in a plain text table
- compare direct scalar FedAvg artifacts with experiment-aware metrics
- inject a custom backend in Python
- record backend metadata in artifacts
- run a deterministic noisy backend and compare clean vs. noisy results
- evaluate client-specific local objectives
- run transparent scalar FedAvg with full per-round and per-client trace
- run scalar FedAvg from a JSON manifest with explicit mean aggregation

**What is intentionally not supported:**

- Qiskit, Braket, Cirq adapters
- real quantum hardware execution
- arbitrary backend imports or plugin systems
- hardware noise models or density-matrix simulation
- FedAvg over model weights
- vector parameters, local epochs, and client sampling
- PyTorch, Flower, or FedML integration
- dataset-based training
- full QFL training
- dashboard or experiment tracking server
- plugin system
