import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class DoubleInt(dynamics.ControlAndDisturbanceAffineDynamics):
# x1 dot = x2 + d1, x2dot = u + d2
    def __init__(self,
                 u_max=1.,
                 d1_max=0.,
                 d2_max=0.,
                 control_mode="max",
                 disturbance_mode="min",
                 gamma = 0.,
                 control_space=None,
                 disturbance_space=None):
        self.gamma = gamma
        if control_space is None:
            control_space = sets.Box(jnp.array([-u_max]), jnp.array([u_max]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([-d1_max, -d2_max]), jnp.array([d1_max, d2_max]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, x2 = state
        return jnp.array([x2,  0.])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1. , 0. ],
            [0. , 1.],
        ])

    def hamiltonian(self, state, time, value, grad_value):
        u_opt, d_opt = self.optimal_control_and_disturbance(state, time, grad_value)
        xdot = self(state, u_opt, d_opt, time)
        return grad_value @ xdot + self.gamma * value


