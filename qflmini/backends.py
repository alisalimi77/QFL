"""Minimal backend abstraction for quantum circuit execution."""

from __future__ import annotations

import math
from typing import Any, Protocol

from qflmini.circuits import run_single_qubit_expectation


class QuantumBackend(Protocol):
    """Protocol for quantum execution backends."""

    def run_expectation(self, theta: float) -> float:
        """Run the backend's expectation-value circuit for a scalar theta."""
        ...


class PennyLaneBackend:
    """PennyLane-backed quantum execution backend."""

    name = "pennylane"

    def run_expectation(self, theta: float) -> float:
        return run_single_qubit_expectation(theta)


class ConstantBackend:
    """Deterministic backend that always returns a fixed value.

    Intended for tests and demos only. Not a quantum simulator.
    Ignores theta intentionally.
    """

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def run_expectation(self, theta: float) -> float:
        return self.value


class NoisyBackend:
    """Deterministic noisy wrapper around another quantum backend.

    Applies a deterministic perturbation to the base backend's output using a
    sine-based formula. The perturbation depends on theta and a fixed seed, so
    results are fully reproducible given the same inputs.

    This is not a real hardware noise model. It is a controlled demo backend.
    No stochastic behavior; no random module.
    """

    name = "noisy"

    def __init__(
        self,
        base_backend: QuantumBackend,
        noise: float,
        seed: int = 0,
    ) -> None:
        if noise < 0:
            raise ValueError("noise must be >= 0.")
        self.base_backend = base_backend
        self.noise = float(noise)
        self.seed = int(seed)

    def run_expectation(self, theta: float) -> float:
        base_result = self.base_backend.run_expectation(theta)
        noise_value = self.noise * math.sin(theta + self.seed)
        return max(-1.0, min(1.0, base_result + noise_value))


def get_backend_metadata(backend: Any) -> dict[str, str]:
    """Return a metadata dict describing a backend instance.

    Args:
        backend: Any object satisfying the QuantumBackend protocol.

    Returns:
        A dict with string keys and string values. Always includes "name" and
        "class". Additional keys are added for ConstantBackend ("value") and
        NoisyBackend ("base_backend", "noise", "seed").
    """
    class_name = backend.__class__.__name__
    raw_name = getattr(backend, "name", None)
    name = raw_name if isinstance(raw_name, str) and raw_name else class_name

    meta: dict[str, str] = {"name": name, "class": class_name}

    if isinstance(backend, ConstantBackend):
        meta["value"] = str(backend.value)
    elif isinstance(backend, NoisyBackend):
        base_name = getattr(backend.base_backend, "name", None)
        if not (isinstance(base_name, str) and base_name):
            base_name = backend.base_backend.__class__.__name__
        meta["base_backend"] = base_name
        meta["noise"] = str(backend.noise)
        meta["seed"] = str(backend.seed)

    return meta
