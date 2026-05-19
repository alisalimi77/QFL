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

## Phase 1 — Observable parameter/loss/gradient traces

**Status: implemented.**

- multi-round execution
- JSON artifact export
- reproducibility metadata and run IDs
- non-overwriting artifact filenames
- heuristic parameter update loop
- objective/loss tracking
- finite-difference gradient update demo

---

## Phase 2 — Manifest/artifact/comparison workflow

**Status: implemented / active.**

Implemented:

- JSON manifest v0.1 for finite-difference gradient update experiments
- `load_gradient_update_manifest()` loads and validates a JSON manifest file
- `run_from_manifest.py` runs a gradient update experiment from a manifest
- multiple example manifests showing different parameter settings (`learning_rate`, `target`, `num_rounds`)
- manifest versioning (`manifest_version`) and human-readable names (`name`)
- optional `description` field in manifests
- simple dependency-free artifact comparison helper (`comparison.py`, `compare_artifacts.py`)
- manifest provenance in artifacts (`manifest_path` recorded in run dict)
- artifact comparison shows manifest names, manifest filenames, backend names, backend details, final theta, and final loss

Remaining planned items:

- more experiment types (parameter update, multi-round)
- possible comparison ergonomics later
- no YAML support yet
- no plugin system yet
- no dashboard or plotting yet

The Python API stays primary. The manifest format is an optional convenience layer on top of it.

---

## Phase 3 — Backend abstraction

**Status: active.**

Implemented:

- minimal `QuantumBackend` protocol (`backends.py`)
- `PennyLaneBackend` as the only real quantum backend
- `ConstantBackend` for tests and demos (deterministic, not a simulator)
- `get_backend_metadata()` helper returns `name` and `class` for any backend
- `QuantumClient` now delegates to its backend; defaults to `PennyLaneBackend`
- coordinators preserve the backend when creating temporary clients
- backend metadata recorded in manifest-run artifacts
- artifact comparison table shows backend name and backend detail columns
- custom backend demo (`run_custom_backend.py`)

Remaining planned items:

- richer backend config ergonomics
- backend metadata consistency across future artifact types
- possible adapters much later (no commitments yet)

PennyLane stays the first and default backend. The interface is intentionally small — it is not a plugin system and adds no new runtime dependencies.

---

## Phase 4 — Deterministic backend realism and backend-aware manifests

**Status: active.**

Implemented:

- `NoisyBackend` — deterministic noisy wrapper around any base backend
- perturbation formula: `noise * sin(theta + seed)`, clipped to `[-1.0, 1.0]`
- `get_backend_metadata()` enriched with `base_backend`, `noise`, and `seed` for `NoisyBackend`
- clean-vs-noisy comparison demo (`run_clean_vs_noisy_backend.py`) with artifact saving
- `format_clean_vs_noisy_backend_report()` report helper
- backend-aware manifest experiments for built-in backends (`pennylane`, `constant`, `noisy`)
- noisy and constant backend example manifests
- backend detail comparison for noisy and constant artifacts

Remaining planned items:

- metrics aggregating clean-vs-noisy difference across rounds
- richer deterministic backend realism
- manifest UX refinement
- external backend adapters much later
- no hardware noise models or density-matrix simulation

---

## Phase 5 — Client-specific objectives

**Status: started.**

Implemented:

- client-specific objective evaluation
- per-client local target and local loss
- mean local loss summary
- artifact-producing `run_client_objectives.py` example
- manifest-driven `client_objectives` experiment support
- example `client_objectives.json` manifest

Remaining planned items:

- richer local objectives later
- local data examples later
- more manifest ergonomics later
- scalar FedAvg as a separate trace-first seed

This phase is still not dataset training and not a full QFL training framework.

---

## Phase 6 — Transparent scalar FedAvg

**Status: started.**

Implemented:

- scalar FedAvg coordinator (`ScalarFedAvgCoordinator`)
- one scalar global parameter
- per-client local finite-difference updates
- mean aggregation over local updated theta values
- full per-round and per-client trace
- artifact-producing `run_scalar_fedavg.py` example
- scalar FedAvg support in artifact comparison metrics
- manifest-driven scalar FedAvg (`examples/manifests/scalar_fedavg.json`)
- explicit scalar FedAvg aggregation config (`{"type": "mean"}`)
- aggregation trace inputs and output recorded in artifacts

Remaining planned items:

- Aggregator Extension Contract
- Scenario Contract v0.2
- Trace Event Model
- ClientRuntime Protocol
- vector parameters later
- richer deterministic scenarios later
- toy local data examples later
- full FedAvg over model weights much later, if ever

This phase is still not dataset training, not PyTorch/Flower integration, and
not a general federated learning framework.

---

## Later — External adapter exploration

**Status: planned, optional.**

Possible adapters for external quantum SDKs or services. This is not implemented and not committed to a specific provider.

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
- arbitrary backend imports or plugin systems
- production real hardware support
