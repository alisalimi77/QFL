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

- only `gradient_update` is supported
- JSON only, no YAML
- no backend selection in manifests yet

---

## 7. Try multiple manifests

Four example manifests are provided. Each changes one parameter from the default:

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
python examples/run_from_manifest.py examples/manifests/gradient_update_low_lr.json
python examples/run_from_manifest.py examples/manifests/gradient_update_target_half.json
python examples/run_from_manifest.py examples/manifests/gradient_update_more_rounds.json
```

| Manifest                             | Name                      | What changes              |
| ------------------------------------ | ------------------------- | ------------------------- |
| `gradient_update.json`               | `default-gradient-update` | Default settings          |
| `gradient_update_low_lr.json`        | `low-learning-rate`       | Smaller learning rate     |
| `gradient_update_target_half.json`   | `target-half`             | Non-zero target (0.5)     |
| `gradient_update_more_rounds.json`   | `more-rounds`             | Five rounds instead of three |

Each run produces a separate artifact under `runs/`.

---

## 8. Compare artifacts

After generating at least two artifacts, compare them:

```bash
python examples/compare_artifacts.py runs/<artifact-a>.json runs/<artifact-b>.json
```

Replace `<artifact-a>` and `<artifact-b>` with the actual filenames printed after each run.
The comparison table shows key fields side by side:

```text
qfl-mini: artifact comparison

run_id                                          manifest                 manifest_file                     backend    experiment       rounds  final_theta  final_loss
run_from_manifest_gradient_update_...           default-gradient-update  gradient_update.json              pennylane  gradient_update  3       0.773778     0.608376
run_from_manifest_gradient_update_...           more-rounds              gradient_update_more_rounds.json  pennylane  gradient_update  5       0.972194     0.412106
```

This is a plain text, dependency-free helper. It is not a dashboard or experiment tracking server.

---

## 9. Inspect an artifact

```bash
python -m json.tool runs/<artifact>.json
```

Artifacts include:

- `run_id` and creation timestamp
- `example` name
- environment metadata (Python version, platform, PennyLane version)
- manifest path and manifest config (for manifest-run artifacts)
- backend metadata (`name` and `class`)
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
      "name": "default-gradient-update"
    },
    "backend": {
      "name": "pennylane",
      "class": "PennyLaneBackend"
    },
    "result": {
      "final_theta": 0.7737779663622264
    }
  }
}
```

---

## 10. Run the custom backend demo

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

## 11. Run checks

```bash
python -m compileall qflmini examples
pytest
```

`compileall` catches import errors and syntax problems across all modules and examples.
`pytest` runs the full test suite. GitHub Actions runs both on every push and pull request.

---

## 12. Where this leaves the project

```text
Phase 0: minimal execution                            [done]
Phase 1: parameter/loss/gradient traces               [done]
Phase 2: manifest/artifact/comparison workflow        [done]
Phase 3: backend abstraction                          [active]
```

**What the project can do now:**

- run local quantum clients with PennyLane circuit execution
- aggregate client results (mean)
- run multi-round coordination
- track loss across rounds
- run finite-difference gradient updates
- run experiments from JSON manifests
- save timestamped reproducibility artifacts
- compare artifacts in a plain text table
- inject a custom backend in Python
- record backend metadata in manifest-run artifacts

**What is intentionally not supported:**

- Qiskit, Braket, Cirq adapters
- real quantum hardware execution
- backend selection in manifests
- noise models
- FedAvg
- dataset-based training
- full QFL training
- dashboard or experiment tracking server
- plugin system

The next natural direction is adding controlled backend realism, such as a small noisy backend demo, while keeping the project minimal.
