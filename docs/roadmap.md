# Roadmap

This roadmap is aspirational and may change. Each phase is intended to stay small and executable — a working prototype before the next phase begins.

The current project is alpha. Do not rely on API stability between phases.

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

**Status: planned.**

A declarative way to describe an experiment: clients, rounds, initial theta, learning rate, target, epsilon. The goal is to run examples from a manifest file without editing Python.

The Python API stays primary. The manifest format is an optional convenience layer on top of it.

---

## Phase 3 — Backend adapters

**Status: planned.**

PennyLane stays the first and default backend. A simple adapter interface would allow other simulators or APIs to be wired in without changing the coordinator logic.

Possible future backends: Qiskit, Braket, Cirq. No commitments yet.

---

## Phase 4 — Noise and backend realism

**Status: planned.**

Simple noise models to make simulation results more realistic. Metrics for comparing noisy vs. ideal runs. No hardware required.

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
