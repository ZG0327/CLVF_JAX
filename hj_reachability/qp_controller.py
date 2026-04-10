"""Quadratic-programming-based CLVF controller utilities."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def _as_box_bound(umax, m: int) -> np.ndarray:
    """Normalize scalar or vector box bounds to shape ``(m,)``."""
    umax_arr = np.asarray(umax, dtype=float)
    if umax_arr.ndim == 0:
        return np.full((m,), float(umax_arr))
    umax_arr = umax_arr.reshape(-1)
    if umax_arr.shape != (m,):
        raise ValueError(f"umax must be scalar or shape ({m},), got {umax_arr.shape}.")
    return umax_arr


def solve_two_stage_qp(
    x_now,
    grid,
    target_values,
    dynamics,
    gamma,
    umax,
    u_ref=None,
    eps_delta=1e-6,
    time=0.0,
):
    """Solve the two-stage CLVF-QP at a continuous state.

    Stage 1 minimizes the slack variable ``delta`` needed to satisfy
    ``LgV(x) u <= -LfV(x) - gamma V(x) + delta`` subject to box control bounds.
    Stage 2 finds the control closest to ``u_ref`` among controls using the optimal
    slack from stage 1.
    """
    grad = grid.grad_values(target_values)
    grad_value = grid.interpolate(grad, x_now)
    V_value = float(grid.interpolate(target_values, x_now))

    G_jax = grad_value @ dynamics.control_jacobian(x_now, time)
    h_jax = -(grad_value @ dynamics.open_loop_dynamics(x_now, time)) - gamma * V_value

    G = np.asarray(G_jax, dtype=float).reshape(1, -1)
    h = np.asarray(h_jax, dtype=float).reshape(1,)
    m = G.shape[1]
    umax_vec = _as_box_bound(umax, m)

    if u_ref is None:
        u_ref = np.zeros(m)
    else:
        u_ref = np.asarray(u_ref, dtype=float).reshape(m,)

    u1 = cp.Variable(m)
    delta1 = cp.Variable(nonneg=True)
    constraints1 = [
        cp.Constant(G) @ u1 <= cp.Constant(h) + delta1,
        u1 <= umax_vec,
        u1 >= -umax_vec,
    ]
    prob1 = cp.Problem(cp.Minimize(cp.square(delta1)), constraints1)
    prob1.solve(solver=cp.OSQP, warm_start=True)

    if prob1.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, {
            "status_stage1": prob1.status,
            "status_stage2": None,
            "V": V_value,
            "G": G,
            "h": h,
            "umax": umax_vec,
        }

    delta_star = float(delta1.value)

    u2 = cp.Variable(m)
    constraints2 = [
        cp.Constant(G) @ u2 <= cp.Constant(h) + (delta_star + eps_delta),
        u2 <= umax_vec,
        u2 >= -umax_vec,
    ]
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(u2 - u_ref)), constraints2)
    prob2.solve(solver=cp.OSQP, warm_start=True)

    if prob2.status not in ["optimal", "optimal_inaccurate"]:
        return None, delta_star, {
            "status_stage1": prob1.status,
            "status_stage2": prob2.status,
            "V": V_value,
            "G": G,
            "h": h,
            "umax": umax_vec,
        }

    u_val = np.asarray(u2.value, dtype=float).reshape(-1)
    return u_val, delta_star, {
        "status_stage1": prob1.status,
        "status_stage2": prob2.status,
        "V": V_value,
        "G": G,
        "h": h,
        "umax": umax_vec,
    }
