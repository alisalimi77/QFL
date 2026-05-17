# Roadmap

This roadmap is aspirational and may change. Each phase is intended to stay small and executable — a working prototype before the next phase begins.

The current project is alpha. Do not rely on API stability between phases.

For the current end-to-end workflow, see [walkthrough.md](walkthrough.md).

---

## Phase 0 — Minimal federated quantum execution

**Status: implemented.**

- local quantum clients with PennyLane circuit execution
- one-round classical coordination
- mean aggregation
- human-readable report

---

## Phase 1 — Observable and reproducible parameter experiments

**Status: implemented.**

- multi-round execution
- JSON artifact export
- reproducibility metadata and run IDs
- non-overwriting artifact filenames
- heuristic parameter update loop
- objective/loss tracking
- finite-difference gradient update demo

---

## Phase 2 — Experiment manifests

**Status: started.**

Started:

- JSON manifest v0 for finite-difference gradient update experiments
- `load_gradient_update_manifest()` loads and validates a JSON manifest file
- `run_from_manifest.py` runs a gradient update experiment from a manifest
- multiple example manifests showing different parameter settings (`learning_rate`, `target`, `num_rounds`)
- manifest versioning (`manifest_version`) and human-readable names (`name`)
- optional `description` field in manifests
- artifact comparison shows manifest names
- simple dependency-free artifact comparison helper (`comparison.py`, `compare_artifacts.py`)
- manifest provenance in artifacts (`manifest_path` recorded in run dict)
- artifact comparison shows manifest filename (`manifest_file` column)

Remaining planned items:

- more experiment types (parameter update, multi-round)
- richer artifact comparison later
- no YAML support yet
- no plugin system yet
- no dashboard or plotting yet

The Python API stays primary. The manifest format is an optional convenience layer on top of it.

---

## Phase 3 — Backend adapters

**Status: started.**

Started:

- minimal `QuantumBackend` protocol (`backends.py`)
- `PennyLaneBackend` as the only real quantum backend
- `ConstantBackend` for tests and demos (deterministic, not a simulator)
- `get_backend_metadata()` helper returns `name` and `class` for any backend
- `QuantumClient` now delegates to its backend; defaults to `PennyLaneBackend`
- coordinators preserve the backend when creating temporary clients
- backend metadata recorded in manifest-run artifacts
- artifact comparison table shows backend name column
- custom backend demo (`run_custom_backend.py`)

Remaining planned items:

- backend selection in manifests
- backend metadata in all artifact types (currently manifest-run only)
- possible adapters: Qiskit, Braket, Cirq (no commitments yet)

PennyLane stays the first and default backend. The interface is intentionally small — it is not a plugin system and adds no new runtime dependencies.

---

## Phase 4 — Noise and backend realism

**Status: started.**

Started:

- `NoisyBackend` — deterministic noisy wrapper around any base backend
- perturbation formula: `noise * sin(theta + seed)`, clipped to `[-1.0, 1.0]`
- `get_backend_metadata()` enriched with `base_backend`, `noise`, and `seed` for `NoisyBackend`
- clean-vs-noisy comparison demo (`run_clean_vs_noisy_backend.py`) with artifact saving
- `format_clean_vs_noisy_backend_report()` report helper

Remaining planned items:

- noise in manifest experiments
- metrics aggregating clean-vs-noisy difference across rounds
- no hardware noise models or density-matrix simulation

---

## Phase 5 — Federated quantum training examples

**Status: planned.**

Richer objective functions and simple local data examples. Possibly FedAvg-style aggregation experiments. Still demos, not a full training framework.

---

## Phase 6 — Real hardware integration

**Status: planned, optional.**

Optional execution on real quantum hardware via a cloud API. Queue and runtime awareness. Artifact comparison between simulator and hardware runs.

This remains optional and exploratory. qfl-mini does not aim to be a production hardware execution system.

---

## Non-goals (permanent)

These are not planned for any phase:

- a Quantum OS
- a full QFL production framework
- a dashboard or experiment tracking server
- guaranteed convergence or optimization correctness claims
- FedAvg as the default aggregation strategy
- dataset-based training as a core feature
