"""Human-readable reporting helpers for qfl-mini."""

from __future__ import annotations

from typing import Any


def format_round_report(round_result: dict[str, Any]) -> str:
    """Format one coordinator round result as a readable text report.

    Args:
        round_result: Result dictionary returned by ``Coordinator.run_round``.

    Returns:
        A clean human-readable report for the federated quantum workload demo.
    """
    lines = [
        "qfl-mini: federated quantum workload demo",
        "",
        "Client results:",
    ]

    for client_result in round_result["client_results"]:
        lines.append(
            f"- {client_result['client_id']} | "
            f"theta={client_result['theta']} | "
            f"result={client_result['result']:.6f}"
        )

    lines.extend(
        [
            "",
            "Aggregated result:",
            f"{round_result['aggregated_result']:.6f}",
        ]
    )

    return "\n".join(lines)


def format_multi_round_report(run_result: dict[str, Any]) -> str:
    """Format a multi-round coordinator result as a readable text report.

    Args:
        run_result: Result dictionary returned by ``Coordinator.run_rounds``.

    Returns:
        A clean human-readable report for repeated coordination rounds.
    """
    lines = [
        "qfl-mini: multi-round federated quantum workload demo",
        "",
        "Rounds:",
    ]

    for round_result in run_result["rounds"]:
        lines.append(
            f"- round {round_result['round']} | "
            f"aggregated_result={round_result['aggregated_result']:.6f}"
        )

    final_result = run_result["rounds"][-1]["aggregated_result"]
    lines.extend(
        [
            "",
            "Final aggregated result:",
            f"{final_result:.6f}",
        ]
    )

    return "\n".join(lines)


def format_parameter_update_report(update_result: dict[str, Any]) -> str:
    """Format a parameter update result as a readable text report.

    Args:
        update_result: Result dictionary returned by
            ``ParameterUpdateCoordinator.run_updates``.

    Returns:
        A clean human-readable report for the parameter update demo.
    """
    lines = [
        "qfl-mini: parameter update demo",
        "",
        "Rounds:",
    ]

    for round_result in update_result["rounds"]:
        lines.append(
            f"- round {round_result['round']} | "
            f"theta={round_result['theta']:.6f} | "
            f"aggregated_result={round_result['aggregated_result']:.6f} | "
            f"target={round_result['target']:.6f} | "
            f"loss={round_result['loss']:.6f} | "
            f"next_theta={round_result['next_theta']:.6f}"
        )

    lines.extend(
        [
            "",
            "Final theta:",
            f"{update_result['final_theta']:.6f}",
        ]
    )

    return "\n".join(lines)


def format_custom_backend_report(round_result: dict[str, Any]) -> str:
    """Format one coordinator round result for the custom backend demo.

    Args:
        round_result: Result dictionary returned by ``Coordinator.run_round``.

    Returns:
        A clean human-readable report showing per-client results and aggregation.
    """
    lines = [
        "qfl-mini: custom backend demo",
        "",
        "Client results:",
    ]

    for client_result in round_result["client_results"]:
        lines.append(
            f"- {client_result['client_id']} | "
            f"theta={client_result['theta']} | "
            f"result={client_result['result']:.6f}"
        )

    lines.extend(
        [
            "",
            "Aggregated result:",
            f"{round_result['aggregated_result']:.6f}",
        ]
    )

    return "\n".join(lines)


def format_clean_vs_noisy_backend_report(run_result: dict[str, Any]) -> str:
    """Format a clean-vs-noisy backend comparison result as a readable text report.

    Args:
        run_result: Result dictionary with keys "clean", "noisy", and "difference".
            Each of "clean" and "noisy" should have "backend" and "result" sub-dicts.

    Returns:
        A clean human-readable report for the clean-vs-noisy backend demo.
    """
    clean_name = run_result["clean"]["backend"]["name"]
    clean_agg = run_result["clean"]["result"]["aggregated_result"]
    noisy_name = run_result["noisy"]["backend"]["name"]
    noisy_agg = run_result["noisy"]["result"]["aggregated_result"]
    difference = run_result["difference"]

    lines = [
        "qfl-mini: clean vs noisy backend demo",
        "",
        "Clean backend:",
        f"- backend={clean_name}",
        f"- aggregated_result={clean_agg:.6f}",
        "",
        "Noisy backend:",
        f"- backend={noisy_name}",
        f"- aggregated_result={noisy_agg:.6f}",
        "",
        "Difference:",
        f"{difference:.6f}",
    ]

    return "\n".join(lines)


def format_client_objectives_report(result: dict[str, Any]) -> str:
    """Format client-specific objective results as a readable text report.

    Args:
        result: Result dictionary returned by
            ``evaluate_client_objectives``.

    Returns:
        A clean human-readable report for the client-specific objective demo.
    """
    lines = [
        "qfl-mini: client-specific objective demo",
        "",
        "Client objectives:",
    ]

    for client_objective in result["client_objectives"]:
        lines.append(
            f"- {client_objective['client_id']} | "
            f"theta={client_objective['theta']} | "
            f"target={client_objective['target']} | "
            f"result={client_objective['result']:.6f} | "
            f"loss={client_objective['loss']:.6f}"
        )

    lines.extend(
        [
            "",
            "Aggregated result:",
            f"{result['aggregated_result']:.6f}",
            "",
            "Mean local loss:",
            f"{result['mean_local_loss']:.6f}",
        ]
    )

    return "\n".join(lines)


def format_gradient_update_report(update_result: dict[str, Any]) -> str:
    """Format a finite-difference gradient update result as a readable text report.

    Args:
        update_result: Result dictionary returned by
            ``FiniteDifferenceGradientCoordinator.run_updates``.

    Returns:
        A clean human-readable report for the gradient update demo.
    """
    lines = [
        "qfl-mini: finite-difference gradient update demo",
        "",
        "Rounds:",
    ]

    for round_result in update_result["rounds"]:
        lines.append(
            f"- round {round_result['round']} | "
            f"theta={round_result['theta']:.6f} | "
            f"loss={round_result['loss']:.6f} | "
            f"gradient={round_result['gradient']:.6f} | "
            f"next_theta={round_result['next_theta']:.6f}"
        )

    lines.extend(
        [
            "",
            "Final theta:",
            f"{update_result['final_theta']:.6f}",
        ]
    )

    return "\n".join(lines)
