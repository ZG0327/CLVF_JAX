import jax.numpy as jnp
import numpy as np


class Controller:
    """
    Hamilton-Jacobi feedback controller for hj_reachability.

    Parameters
    ----------
    dynamics : hj_reachability.dynamics.Dynamics
        Your system dynamics object.
    grid : hj_reachability.Grid
        Computational grid.
    value_function : jnp.ndarray or np.ndarray
        Either:
          - static value function with shape grid.shape
          - time-varying value function with shape (T, *grid.shape)
    times : array-like or None
        If value_function is time-varying, provide the corresponding time stamps.
        Example: np.linspace(0.0, -2.0, 41)
    upwind_scheme : callable or None
        Optional scheme passed to grid.grad_values(...).
    time_interp : str
        "nearest" or "linear". For now, both are supported.
    """

    def __init__(
        self,
        dynamics,
        grid,
        value_function,
        times=None,
        upwind_scheme=None,
        time_interp="nearest",
        mode="opt",
    ):
        self.dynamics = dynamics
        self.grid = grid
        self.value_function = jnp.asarray(value_function)
        self.times = None if times is None else np.asarray(times, dtype=float)
        self.upwind_scheme = upwind_scheme
        self.time_interp = time_interp
        self.mode = mode
        
        # Detect static vs time-varying value function
        if self.value_function.ndim == self.grid.ndim:
            self.is_time_varying = False
            self.grad_value_function = self.grid.grad_values(
                self.value_function, upwind_scheme=self.upwind_scheme
            )
        elif self.value_function.ndim == self.grid.ndim + 1:
            self.is_time_varying = True
            if self.times is None:
                raise ValueError(
                    "For time-varying value_function, `times` must be provided."
                )
            if len(self.times) != self.value_function.shape[0]:
                raise ValueError(
                    "`times` length must equal value_function.shape[0]."
                )

            self.grad_value_function = jnp.stack(
                [
                    self.grid.grad_values(v, upwind_scheme=self.upwind_scheme)
                    for v in self.value_function
                ],
                axis=0,
            )
        else:
            raise ValueError(
                "value_function must have shape grid.shape or (T, *grid.shape)."
            )

    def _get_time_index_nearest(self, t):
        return int(np.argmin(np.abs(self.times - float(t))))

    def _get_grad_at_time(self, t):
        if not self.is_time_varying:
            return self.grad_value_function

        if self.time_interp == "nearest":
            k = self._get_time_index_nearest(t)
            return self.grad_value_function[k]

        if self.time_interp == "linear":
            tt = float(t)

            # Clamp outside range
            if tt <= self.times.min():
                return self.grad_value_function[np.argmin(self.times)]
            if tt >= self.times.max():
                return self.grad_value_function[np.argmax(self.times)]

            order = np.argsort(self.times)
            times_sorted = self.times[order]
            grads_sorted = self.grad_value_function[order]

            idx_hi = np.searchsorted(times_sorted, tt)
            idx_lo = idx_hi - 1

            t0, t1 = times_sorted[idx_lo], times_sorted[idx_hi]
            g0, g1 = grads_sorted[idx_lo], grads_sorted[idx_hi]

            alpha = (tt - t0) / (t1 - t0)
            return (1.0 - alpha) * g0 + alpha * g1

        raise ValueError("time_interp must be 'nearest' or 'linear'.")

    def value_at(self, state, time=None):
        """
        Interpolate V(x,t) at a continuous state.
        """
        state = jnp.asarray(state)

        if not self.is_time_varying:
            return self.grid.interpolate(self.value_function, state)

        if time is None:
            raise ValueError("time must be provided for time-varying value_function.")

        if self.time_interp == "nearest":
            k = self._get_time_index_nearest(time)
            return self.grid.interpolate(self.value_function[k], state)

        if self.time_interp == "linear":
            tt = float(time)

            if tt <= self.times.min():
                k = int(np.argmin(self.times))
                return self.grid.interpolate(self.value_function[k], state)

            if tt >= self.times.max():
                k = int(np.argmax(self.times))
                return self.grid.interpolate(self.value_function[k], state)

            order = np.argsort(self.times)
            times_sorted = self.times[order]
            values_sorted = self.value_function[order]

            idx_hi = np.searchsorted(times_sorted, tt)
            idx_lo = idx_hi - 1

            t0, t1 = times_sorted[idx_lo], times_sorted[idx_hi]
            v0 = self.grid.interpolate(values_sorted[idx_lo], state)
            v1 = self.grid.interpolate(values_sorted[idx_hi], state)

            alpha = (tt - t0) / (t1 - t0)
            return (1.0 - alpha) * v0 + alpha * v1

        raise ValueError("time_interp must be 'nearest' or 'linear'.")

    def grad_at(self, state, time=None):
        """
        Interpolate ∇V(x,t) at a continuous state.
        """
        state = jnp.asarray(state)
        grad_grid = self._get_grad_at_time(time)
        grad = self.grid.interpolate(grad_grid, state)

        if jnp.any(jnp.isnan(grad)):
            raise ValueError(
                "State is outside non-periodic grid domain; interpolation returned NaN."
            )

        return grad

    def __call__(self, state, time):
        """
        Return optimal control and disturbance at (state, time).

        Returns
        -------
        u, d
        """
        state = jnp.asarray(state)
        grad = self.grad_at(state, time)

        u, d = self.dynamics.optimal_control_and_disturbance(state, time, grad)
        return u, d

    def control(self, state, time):
        if self.mode == "opt":
            state = jnp.asarray(state)
            grad = self.grad_at(state, time)
            return self.dynamics.optimal_control(state, time, grad)

    def disturbance(self, state, time):
        state = jnp.asarray(state)
        grad = self.grad_at(state, time)
        return self.dynamics.optimal_disturbance(state, time, grad)