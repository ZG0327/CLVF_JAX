import contextlib
import functools
import jax
import jax.numpy as jnp
from flax import struct
from hj_reachability import solver as hj_solver
from hj_reachability import artificial_dissipation
from hj_reachability import time_integration
from hj_reachability.finite_differences import upwind_first
from ZG_time_integration import third_order_tvd_rk_div_freeze

@struct.dataclass
class ZGSolverSettings:
    upwind_scheme: callable = struct.field(default=upwind_first.WENO3, pytree_node=False)
    artificial_dissipation_scheme: callable = struct.field(
        default=artificial_dissipation.global_lax_friedrichs, pytree_node=False
    )
    hamiltonian_postprocessor: callable = struct.field(default=lambda *x: x[-1], pytree_node=False)
    time_integrator: callable = struct.field(default=third_order_tvd_rk_div_freeze, pytree_node=False)
    value_postprocessor: callable = struct.field(default=lambda *x: x[-1], pytree_node=False)
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
    convergence_threshold,
    progress_bar=True,
):
    with (hj_solver._try_get_progress_bar(time, target_time)
          if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:

        def cond_fun(state):
            t, v, prev_v, done = state
            not_at_target = jnp.abs(target_time - t) > 0
            not_converged = jnp.logical_not(done)
            return jnp.logical_and(not_at_target, not_converged)

        def body_fun(state):
            t, v, prev_v, done = state

            new_t, new_v = solver_settings.time_integrator(
                solver_settings, dynamics, grid, t, v, target_time
            )

            diff = jnp.max(jnp.abs(new_v - v)) / jnp.abs(new_t - t)
            
            jax.debug.print("time = {t:.6f}, diff = {d:.6e}", t=new_t, d=diff)
            
            new_done = diff < convergence_threshold

            if bar is not False:
                bar.update_to(jnp.abs(new_t - bar.reference_time))

            return new_t, new_v, v, new_done

        init_state = (time, values, values, False)
        final_t, final_v, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_v
