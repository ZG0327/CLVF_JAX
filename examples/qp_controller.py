# qp_controller.py

import numpy as np
import jax.numpy as jnp
import cvxpy as cp

def solve_two_stage_qp(x_now, grid, target_values, dynamics, gamma, umax, u_ref=None, eps_delta=1e-6):
    grad = grid.grad_values(target_values)
    grad_value = grid.interpolate(grad, x_now)
    V_value = float(grid.interpolate(target_values, x_now))

    G_jax = grad_value @ dynamics.control_jacobian(x_now, 0.0)
    h_jax = -(grad_value @ dynamics.open_loop_dynamics(x_now, 0.0)) - gamma * V_value

    G = np.asarray(G_jax, dtype=float).reshape(1, -1)
    h = np.asarray(h_jax, dtype=float).reshape(1,)
    m = G.shape[1]

    if u_ref is None:
        u_ref = np.zeros(m)
    else:
        u_ref = np.asarray(u_ref, dtype=float).reshape(m,)

    # stage 1
    u1 = cp.Variable(m)
    delta1 = cp.Variable(nonneg=True)

    prob1 = cp.Problem(
        cp.Minimize(cp.square(delta1)),
        [
            cp.Constant(G) @ u1 <= cp.Constant(h) + delta1,
            u1 <= umax,
            u1 >= -umax,
        ],
    )
    prob1.solve(solver=cp.OSQP, warm_start=True)

    if prob1.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, {
            "status_stage1": prob1.status,
            "status_stage2": None,
            "V": V_value,
            "G": G,
            "h": h,
        }

    delta_star = float(delta1.value)

    # stage 2
    u2 = cp.Variable(m)
    prob2 = cp.Problem(
        cp.Minimize(cp.sum_squares(u2 - u_ref)),
        [
            cp.Constant(G) @ u2 <= cp.Constant(h) + (delta_star + eps_delta),
            u2 <= umax,
            u2 >= -umax,
        ],
    )
    prob2.solve(solver=cp.OSQP, warm_start=True)

    if prob2.status not in ["optimal", "optimal_inaccurate"]:
        return None, delta_star, {
            "status_stage1": prob1.status,
            "status_stage2": prob2.status,
            "V": V_value,
            "G": G,
            "h": h,
        }

    u_val = np.asarray(u2.value, dtype=float).reshape(-1)

    return u_val, delta_star, {
        "status_stage1": prob1.status,
        "status_stage2": prob2.status,
        "V": V_value,
        "G": G,
        "h": h,
    }