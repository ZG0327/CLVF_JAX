"""Custom solver settings and convergence stepping utilities for CLVF experiments."""

from __future__ import annotations

import contextlib
import functools
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from . import artificial_dissipation
from . import solver as hj_solver
from .finite_differences import upwind_first
from .zg_time_integration import third_order_tvd_rk_div_freeze


identity = lambda *x: x[-1]


@struct.dataclass
class ZGSolverSettings:
    """Solver settings with convergence and divergence controls."""

    upwind_scheme: Callable = struct.field(default=upwind_first.WENO3, pytree_node=False)
    artificial_dissipation_scheme: Callable = struct.field(
        default=artificial_dissipation.global_lax_friedrichs,
        pytree_node=False,
    )
    hamiltonian_postprocessor: Callable = struct.field(default=identity, pytree_node=False)
    time_integrator: Callable = struct.field(
        default=third_order_tvd_rk_div_freeze,
        pytree_node=False,
    )
    value_postprocessor: Callable = struct.field(default=identity, pytree_node=False)
    CFL_number: float = 0.75
    convergence_threshold: float = 1e-4
    divergence_threshold: float = 1e3


@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def step_until_converged(
    solver_settings,
    dynamics,
    grid,
    time,
    values,
    target_time,
    convergence_threshold=None,
    progress_bar=True,
):
    """Advance until either the target time is reached or the update converges.

    Parameters
    ----------
    convergence_threshold : float or None
        If omitted, ``solver_settings.convergence_threshold`` is used.
    """
    if convergence_threshold is None:
        convergence_threshold = solver_settings.convergence_threshold

    with (
        hj_solver._try_get_progress_bar(time, target_time)
        if progress_bar is True
        else contextlib.nullcontext(progress_bar)
    ) as bar:

        def cond_fun(state):
            t, v, prev_v, done = state
            del prev_v
            not_at_target = jnp.abs(target_time - t) > 0
            not_converged = jnp.logical_not(done)
            return jnp.logical_and(not_at_target, not_converged)

        def body_fun(state):
            t, v, prev_v, done = state
            del prev_v, done

            new_t, new_v = solver_settings.time_integrator(
                solver_settings, dynamics, grid, t, v, target_time
            )

            dt = jnp.maximum(jnp.abs(new_t - t), jnp.finfo(new_v.dtype).eps)

            divergence_threshold = getattr(solver_settings, "divergence_threshold", None)

            if divergence_threshold is None:
                delta_v = jnp.max(jnp.abs(new_v - v))
                diff = delta_v / dt
                active_mask = jnp.ones_like(v, dtype=bool)
            else:
        # 和你的 divergence freezing 口径保持一致：
        # 只把 value > divergence_threshold 的点视为 diverged
                active_mask = jnp.logical_and(
                        v <= divergence_threshold,
                        new_v <= divergence_threshold,
                        )

                delta_field = jnp.where(active_mask, jnp.abs(new_v - v), 0.0)
                has_active = jnp.any(active_mask)

        # 只在未 diverge 的 grid 上取 max change
                delta_v = jnp.where(has_active, jnp.max(delta_field), jnp.inf)
                diff = delta_v / dt

            new_done = diff < convergence_threshold

            
            # diff = jnp.max(jnp.abs(new_v - v)) / dt
            # delta_v = jnp.max(jnp.abs(new_v - v))
            # new_done = diff < convergence_threshold
            
            
            jax.debug.print("t={:.4f}, max|ΔV|={:.6e}, max|ΔV|/dt={:.6e}",
                            new_t, delta_v, diff)

            if bar is not False:
                bar.update_to(jnp.abs(new_t - bar.reference_time))

            return new_t, new_v, v, new_done

        init_state = (time, values, values, False)
        final_t, final_v, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        del final_t
        return final_v
