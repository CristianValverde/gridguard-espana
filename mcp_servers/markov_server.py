"""
mcp_servers/markov_server.py — MCP: markov

Expone la capa de Markov (tools/markov_state_tracker.py) como herramientas MCP.
Pure Python, sin LLM (shift-left). Consumido por orchestrator, anomaly_detector,
report_agent y regulatory_compliance (Plan 4).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ml.incidents import S3_MAJOR, S4_BLACKOUT, STATE_LABELS
from tools import markov_state_tracker as mst

mcp = FastMCP("gridguard-markov")


def _get_grid_state_impl(features: dict) -> dict:
    state = mst.compute_state(features)
    traj = mst.get_trajectory(state, horizon_hours=6)
    A = mst.get_transition_matrix()
    confidence = float(A[state].max())  # masa de la transición dominante
    return {
        "state": state,
        "label": STATE_LABELS.get(state, "?"),
        "confidence": confidence,
        "trajectory_6h": traj,
    }


def _get_transition_probabilities_impl() -> dict:
    A = mst.get_transition_matrix()
    return {"matrix": [[float(x) for x in row] for row in A], "states": [0, 1, 2, 3, 4]}


def _get_markov_trajectory_impl(features: dict, horizon_hours: int) -> dict:
    state = mst.compute_state(features)
    return {
        "trajectory": mst.get_trajectory(state, horizon_hours),
        "prob_reach_s3": mst.prob_reach_severity(state, S3_MAJOR, horizon_hours),
        "prob_reach_s4": mst.prob_reach_severity(state, S4_BLACKOUT, horizon_hours),
    }


@mcp.tool()
def get_grid_state(features: dict) -> dict:
    """Estado actual S0–S4 + trayectoria probable a 6 h desde los proxies de red."""
    return _get_grid_state_impl(features)


@mcp.tool()
def get_transition_probabilities() -> dict:
    """Matriz de transición A[5x5] de la cadena de Markov."""
    return _get_transition_probabilities_impl()


@mcp.tool()
def get_markov_trajectory(features: dict, horizon_hours: int = 6) -> dict:
    """Trayectoria de estado proyectada y P(alcanzar S3/S4) en el horizonte dado."""
    return _get_markov_trajectory_impl(features, horizon_hours)


if __name__ == "__main__":
    mcp.run()
