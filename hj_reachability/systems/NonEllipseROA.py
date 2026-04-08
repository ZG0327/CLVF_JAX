import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class NonEllipseROA(dynamics.ControlAndDisturbanceAffineDynamics):
    # x1_dot = -x1 * (1 - x1^4 - x2^4)
    # x2_dot = -x2 * (1 - x1^4 - x2^4)

    def __init__(self,
                 control_mode="max",
                 disturbance_mode="min",
                 gamma=0.,
                 control_space=None,
                 disturbance_space=None):
        self.gamma = gamma

        if control_space is None:
            control_space = sets.Box(jnp.zeros((0,)), jnp.zeros((0,)))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.zeros((0,)), jnp.zeros((0,)))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x1, x2 = state
        s = 1. - x1**4 - x2**4
        return jnp.array([
            -x1 * s,
            -x2 * s,
        ])

    def control_jacobian(self, state, time):
        return jnp.zeros((2, 0))

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((2, 0))

    def hamiltonian(self, state, time, value, grad_value):
        xdot = self.open_loop_dynamics(state, time)
        return grad_value @ xdot + self.gamma * value