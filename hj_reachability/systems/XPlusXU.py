import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class XPlusXU(dynamics.ControlAndDisturbanceAffineDynamics):
    # xdot = -x + x u
    def __init__(self,
                 u_max=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 gamma=0.,
                 control_space=None,
                 disturbance_space=None):
        self.gamma = gamma

        if control_space is None:
            control_space = sets.Box(jnp.array([-u_max]), jnp.array([u_max]))

        # no disturbance: use a 0-dimensional box
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([]), jnp.array([]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, = state
        return jnp.array([-x])

    def control_jacobian(self, state, time):
        x, = state
        return jnp.array([
            [x],
        ])

    def disturbance_jacobian(self, state, time):
        # no disturbance
        return jnp.zeros((1, 0))

    def hamiltonian(self, state, time, value, grad_value):
        u_opt, d_opt = self.optimal_control_and_disturbance(state, time, grad_value)
        xdot = self(state, u_opt, d_opt, time)
        return grad_value @ xdot + self.gamma * value