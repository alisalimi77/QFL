"""PennyLane circuits used by the qfl-mini execution sandbox."""

from __future__ import annotations

import pennylane as qml


_DEVICE = qml.device("default.qubit", wires=1)


@qml.qnode(_DEVICE)
def _single_qubit_expectation(theta: float) -> float:
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


def run_single_qubit_expectation(theta: float) -> float:
    """Run a deterministic one-qubit circuit and return a Pauli-Z expectation.

    Args:
        theta: Rotation angle for the local RX gate.

    Returns:
        The expectation value of Pauli-Z after applying RX(theta).
    """
    return float(_single_qubit_expectation(theta))
