# qfl-mini
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

qfl-mini is a minimal execution sandbox for federated quantum-classical workloads.

## What is this?

qfl-mini is a small Python prototype for executing federated quantum workloads with local quantum clients and a classical coordinator.

Each quantum client owns local parameters and runs a simple PennyLane circuit. The classical coordinator collects the local execution results, applies mean aggregation, and produces readable reports. The demos also write JSON artifacts with lightweight reproducibility metadata so runs can be inspected and reproduced later.

The project is intentionally small. It is meant to make the basic execution, observation, and reproducibility path clear before larger federated quantum infrastructure is added.

## Why this exists

Before Quantum Federated Learning can scale, we need a way to execute, observe, and reproduce federated quantum-classical workloads. qfl-mini is the smallest working prototype of that idea.

## What this is not

qfl-mini is:

* not a Quantum OS
* not a full QFL framework
* not full QFL training
* not a full optimizer framework
* not FedAvg
* not dataset-based training
* not automatic differentiation-based training yet
* not a production system
* not a replacement for PennyLane, Qiskit, Flower, Braket, or Cirq
* not connected to real quantum hardware yet

## Core concepts

**Quantum Client**

A local quantum execution node. In this prototype, a quantum client is a Python object with a local parameter and a PennyLane circuit execution.

**Classical Coordinator**

The classical coordination layer. It asks each quantum client to run, collects results, and performs aggregation.

**Federated Quantum Workload**

A computation where multiple quantum-capable clients execute local quantum circuits and a classical coordinator aggregates their results without requiring all computation to happen in one centralized quantum process.

**Execution Sandbox**

A small, controlled environment for defining and running a federated quantum workload before adding networking, backend adapters, hardware execution, or experiment services.

**Aggregation**

The process of combining client results. This version implements mean aggregation only.

**Reproducibility Artifact**

A saved JSON file containing run results and lightweight metadata that can be inspected, shared, and used as a basis for reproducing an execution.

## Installation

```bash
pip install -r requirements.txt
```

## Run the smallest demo

```bash
python examples/run_two_clients.py
```

## Run the multi-round demo

```bash
python examples/run_multi_round.py
```

This writes:

```text
runs/run_multi_round_<timestamp>.json
```

## Run the parameter update demo

```bash
python examples/run_parameter_update.py
```

This writes:

```text
runs/run_parameter_update_<timestamp>.json
```

The parameter update demo tracks a simple objective value:

```text
loss = (aggregated_result - target)^2
```

The update rule is intentionally simple:

```text
next_theta = theta - learning_rate * aggregated_result
```

This is objective tracking only. It is not gradient-based training.

## Run the gradient update demo

```bash
python examples/run_gradient_update.py
```

This writes:

```text
runs/run_gradient_update_<timestamp>.json
```

The gradient update demo estimates a gradient using central finite differences:

```text
gradient ≈ (loss(theta + epsilon) - loss(theta - epsilon)) / (2 * epsilon)
```

It updates the parameter with:

```text
next_theta = theta - learning_rate * gradient
```

This is still not full QFL training. It is not PennyLane autograd. It is a
minimal gradient-based optimization trace built on the existing execution,
aggregation, and artifact system.

## Reproducibility artifacts

Saved artifacts include:

* project name
* artifact version
* run ID
* timestamp
* example name
* Python version
* platform, system, and machine
* PennyLane version
* run results

Each artifact has a unique `run_id`. The artifact filename is derived from
that `run_id`, so repeated runs do not overwrite previous artifacts. The JSON
artifact includes the same top-level `run_id` for traceability.

Example structure:

```json
{
  "project": "qfl-mini",
  "artifact_version": "0.1",
  "run_id": "run_parameter_update_20260516T203956Z",
  "created_at": "...",
  "example": "run_parameter_update",
  "environment": {
    "python_version": "...",
    "platform": "...",
    "system": "...",
    "machine": "...",
    "pennylane_version": "..."
  },
  "run": {
    "num_rounds": 3,
    "final_theta": 0.225715,
    "rounds": [
      {
        "round": 1,
        "theta": 0.5,
        "aggregated_result": 0.877583,
        "target": 0.0,
        "loss": 0.770151,
        "next_theta": 0.412242
      }
    ]
  }
}
```

This is intentionally lightweight. qfl-mini does not implement a full experiment tracking system.

## Development checks

```bash
pip install -r requirements-dev.txt
pytest
python -m compileall qflmini examples
```

## Current status

Only Phase 0 and a small Phase 1 seed are implemented.

Implemented:

* local quantum clients
* simple PennyLane circuit execution
* classical coordinator
* mean aggregation
* one-round report
* multi-round execution
* JSON artifact export
* minimal parameter update loop
* reproducibility metadata in saved artifacts
* run IDs for saved artifacts
* non-overwriting artifact filenames
* simple objective/loss tracking for parameter update runs
* finite-difference gradient update demo

Not implemented yet:

* full experiment tracking
* full QFL training
* full optimizer framework
* FedAvg
* dataset-based training
* automatic differentiation-based training
* training loops
* noise models
* non-IID data
* backend adapters
* real hardware
* dashboard

## Roadmap

This version still belongs to the Phase 0 / early Phase 1 seed. The parameter update loop and the finite-difference gradient update are Phase 1 seeds, not a full training framework.

* Phase 0: minimal federated quantum execution
* Phase 1: multi-round execution, run history, minimal parameter update loop, and finite-difference gradient update (Phase 1.4 seed)
* Phase 2: parameter updates
* Phase 3: federated variational quantum training
* Phase 4: noise and backend realism
* Phase 5: experiment manifests
* Phase 6: backend adapters
* Phase 7: reproducibility artifacts
* Phase 8: real hardware integration
