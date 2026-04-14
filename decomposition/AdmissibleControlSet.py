import numpy as np
import jax
import jax.numpy as jnp


class AdmissibleControlSet:
    """
    Build CLVF linear inequalities of the form

        A(x, t) u <= b(x, t)

    from a value function V defined on a grid, where

        A = L_g V = ∇V(x, t)^T g(x, t)
        b = -gamma * V(x, t) - L_f V

    for control-affine dynamics

        x_dot = f(x, t) + g(x, t) u + h(x, t) d.

    Notes
    -----
    - `compute_ab_state(...)` uses batch interpolation when x has shape (N, n).
    - This assumes `grid.interpolate(values, X)` supports batched states
      `X.shape == (N, n)`. If not, you can replace the interpolation call with
      `jax.vmap(lambda x: grid.interpolate(values, x))(X)`.
    """

    def __init__(
        self,
        dynamics,
        grid,
        value_function,
        gamma,
        u_range=None,
        d_range=None,
        times=None,
        upwind_scheme=None,
        time_interp="nearest",
    ):
        self.dynamics = dynamics
        self.grid = grid
        self.value_function = jnp.asarray(value_function)
        self.gamma = float(gamma)
        self.u_range = self._normalize_range(u_range)
        self.d_range = self._normalize_range(d_range)
        self.times = None if times is None else np.asarray(times, dtype=float)
        self.upwind_scheme = upwind_scheme
        self.time_interp = time_interp

        if self.value_function.ndim == self.grid.ndim:
            self.is_time_varying = False
            self.grad_value_function = self.grid.grad_values(
                self.value_function,
                upwind_scheme=self.upwind_scheme,
            )
        elif self.value_function.ndim == self.grid.ndim + 1:
            self.is_time_varying = True
            if self.times is None:
                raise ValueError(
                    "For time-varying value_function, `times` must be provided."
                )
            if len(self.times) != self.value_function.shape[0]:
                raise ValueError("`times` length must equal value_function.shape[0].")
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

    @staticmethod
    def _normalize_range(bounds):
        if bounds is None:
            return None

        arr = np.asarray(bounds, dtype=float)

        if arr.ndim == 0:
            val = float(arr)
            return np.array([[-val, val]], dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] == 2:
                return arr.reshape(1, 2)
            return np.stack([-np.abs(arr), np.abs(arr)], axis=1)

        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

        raise ValueError(
            "Range must be None, a scalar, a length-2 pair, a 1D array of symmetric bounds, or a (dim, 2) array."
        )

    def _get_time_index_nearest(self, t):
        return int(np.argmin(np.abs(self.times - float(t))))

    def _get_grid_at_time(self, arr, t):
        if not self.is_time_varying:
            return arr

        if self.time_interp == "nearest":
            k = self._get_time_index_nearest(t)
            return arr[k]

        if self.time_interp == "linear":
            tt = float(t)
            order = np.argsort(self.times)
            times_sorted = self.times[order]
            arr_sorted = arr[order]

            if tt <= times_sorted[0]:
                return arr_sorted[0]
            if tt >= times_sorted[-1]:
                return arr_sorted[-1]

            idx_hi = np.searchsorted(times_sorted, tt)
            idx_lo = idx_hi - 1
            t0, t1 = times_sorted[idx_lo], times_sorted[idx_hi]
            a0, a1 = arr_sorted[idx_lo], arr_sorted[idx_hi]
            alpha = (tt - t0) / (t1 - t0)
            return (1.0 - alpha) * a0 + alpha * a1

        raise ValueError("time_interp must be 'nearest' or 'linear'.")

    def value_and_grad(self, x, t=0.0):
        x = jnp.asarray(x)
        value_grid = self._get_grid_at_time(self.value_function, t)
        grad_grid = self._get_grid_at_time(self.grad_value_function, t)

        V = self.grid.interpolate(value_grid, x)
        gradV = self.grid.interpolate(grad_grid, x)

        if jnp.any(jnp.isnan(gradV)) or jnp.any(jnp.isnan(V)):
            raise ValueError(
                "State is outside non-periodic grid domain; interpolation returned NaN."
            )

        return gradV, V

    def value_and_grad_batch(self, X, t=0.0):
        X = jnp.asarray(X)
        value_grid = self._get_grid_at_time(self.value_function, t)
        grad_grid = self._get_grid_at_time(self.grad_value_function, t)

        if X.ndim == 1:
            V = self.grid.interpolate(value_grid, X)
            gradV = self.grid.interpolate(grad_grid, X)
            return gradV, V

        interp_value = jax.vmap(lambda x: self.grid.interpolate(value_grid, x))
        interp_grad = jax.vmap(lambda x: self.grid.interpolate(grad_grid, x))

        V = interp_value(X)
        gradV = interp_grad(X)

        if jnp.any(jnp.isnan(gradV)) or jnp.any(jnp.isnan(V)):
            raise ValueError(
                "NaNs encountered during interpolation. "
                "Some states may lie outside the grid bounds."
            )

        return gradV, V

    @staticmethod
    def _worst_case_box_support(c, bounds):
        c = np.asarray(c, dtype=float).reshape(-1)
        bounds = np.asarray(bounds, dtype=float)

        if bounds.shape[0] != c.shape[0]:
            raise ValueError(
                f"Dimension mismatch: disturbance term has dim {c.shape[0]}, but d_range has dim {bounds.shape[0]}."
            )

        low = bounds[:, 0]
        high = bounds[:, 1]
        d_star = np.where(c >= 0.0, high, low)
        support_value = float(c @ d_star)
        return support_value, d_star

    def _compute_one(self, x, t=0.0, robust_disturbance=False):
        x = jnp.asarray(x)
        gradV, V = self.value_and_grad(x, t)

        f = self.dynamics.open_loop_dynamics(x, t)
        g = self.dynamics.control_jacobian(x, t)

        LfV = gradV @ f
        LgV = gradV @ g

        A = np.asarray(LgV, dtype=float).reshape(-1)
        b = float(-self.gamma * V - LfV)

        if self.u_range is not None and self.u_range.shape[0] != A.shape[0]:
            raise ValueError(
                f"Dimension mismatch: control term has dim {A.shape[0]}, but u_range has dim {self.u_range.shape[0]}."
            )

        info = {
            "x": np.asarray(x, dtype=float),
            "t": float(t),
            "V": float(V),
            "gradV": np.asarray(gradV, dtype=float),
            "LfV": float(LfV),
            "LgV": A.copy(),
            "gamma": self.gamma,
            "u_range": None if self.u_range is None else self.u_range.copy(),
            "d_range": None if self.d_range is None else self.d_range.copy(),
        }

        if robust_disturbance:
            if not hasattr(self.dynamics, "disturbance_jacobian"):
                raise AttributeError(
                    "robust_disturbance=True requires dynamics.disturbance_jacobian(x, t)."
                )
            if self.d_range is None:
                raise ValueError(
                    "robust_disturbance=True requires d_range to be provided."
                )

            h = self.dynamics.disturbance_jacobian(x, t)
            LhV = gradV @ h
            support, d_star = self._worst_case_box_support(LhV, self.d_range)
            b = b - support

            info.update(
                {
                    "LhV": np.asarray(LhV, dtype=float).reshape(-1),
                    "disturbance_support": float(support),
                    "d_star": np.asarray(d_star, dtype=float),
                }
            )

        return A, b, info

    @staticmethod
    def _batched_inner(a, b):
        return jnp.sum(a * b, axis=-1)

    @staticmethod
    def _batched_lg(gradV, g):
        """
        Compute batched LgV.

        gradV : (N, n)
        g     : (N, n, m) or (N, n)

        Returns
        -------
        LgV : (N, m)
        """
        if g.ndim == 2:
            return jnp.sum(gradV * g, axis=-1, keepdims=True)
        if g.ndim == 3:
            return jnp.einsum("bn,bnm->bm", gradV, g)
        raise ValueError(f"Unexpected batched control Jacobian shape: {g.shape}")

    def compute_ab_state(self, x, t=0.0, robust_disturbance=False):
        """
        Compute A and b for one state or a batch of states.

        Parameters
        ----------
        x : array-like, shape (n,) or (N, n)
            One state or multiple states.
        t : float, optional
            Time for time-varying value functions.
        robust_disturbance : bool, optional
            Whether to subtract the worst-case bounded disturbance contribution.

        Returns
        -------
        A : np.ndarray
            - shape (m,) for a single state
            - shape (N, m) for multiple states
        b : float or np.ndarray
            - scalar for a single state
            - shape (N,) for multiple states
        info : dict
            Diagnostics. For batched input, values are stacked across states.
        """
        x_arr = np.asarray(x, dtype=float)
        state_dim = int(self.grid.states.shape[-1])

        if x_arr.ndim == 1:
            if x_arr.shape[0] != state_dim:
                raise ValueError(
                    f"Single state must have shape ({state_dim},), got {x_arr.shape}."
                )
            return self._compute_one(x_arr, t=t, robust_disturbance=robust_disturbance)

        if x_arr.ndim != 2:
            raise ValueError("x must have shape (n,) or (N, n).")
        if x_arr.shape[1] != state_dim:
            raise ValueError(
                f"Batched states must have shape (N, {state_dim}), got {x_arr.shape}."
            )

        X = jnp.asarray(x_arr)
        gradV, V = self.value_and_grad_batch(X, t=t)

        f_fun = lambda x_i: self.dynamics.open_loop_dynamics(x_i, t)
        g_fun = lambda x_i: self.dynamics.control_jacobian(x_i, t)

        f = jax.vmap(f_fun)(X)
        g = jax.vmap(g_fun)(X)

        LfV = self._batched_inner(gradV, f)
        LgV = self._batched_lg(gradV, g)
        b = -self.gamma * V - LfV

        A = np.asarray(LgV, dtype=float)
        b_np = np.asarray(b, dtype=float)

        control_dim = A.shape[1]
        if self.u_range is not None and self.u_range.shape[0] != control_dim:
            raise ValueError(
                f"Dimension mismatch: control term has dim {control_dim}, but u_range has dim {self.u_range.shape[0]}."
            )

        info = {
            "x": x_arr.copy(),
            "t": float(t),
            "V": np.asarray(V, dtype=float),
            "gradV": np.asarray(gradV, dtype=float),
            "LfV": np.asarray(LfV, dtype=float),
            "LgV": A.copy(),
            "gamma": self.gamma,
            "u_range": None if self.u_range is None else self.u_range.copy(),
            "d_range": None if self.d_range is None else self.d_range.copy(),
        }

        if robust_disturbance:
            if not hasattr(self.dynamics, "disturbance_jacobian"):
                raise AttributeError(
                    "robust_disturbance=True requires dynamics.disturbance_jacobian(x, t)."
                )
            if self.d_range is None:
                raise ValueError(
                    "robust_disturbance=True requires d_range to be provided."
                )

            h_fun = lambda x_i: self.dynamics.disturbance_jacobian(x_i, t)
            h = jax.vmap(h_fun)(X)
            LhV = self._batched_lg(gradV, h)

            supports = []
            d_stars = []
            for i in range(LhV.shape[0]):
                support_i, d_star_i = self._worst_case_box_support(LhV[i], self.d_range)
                supports.append(support_i)
                d_stars.append(d_star_i)

            supports = np.asarray(supports, dtype=float)
            d_stars = np.stack(d_stars, axis=0)
            b_np = b_np - supports

            info.update(
                {
                    "LhV": np.asarray(LhV, dtype=float),
                    "disturbance_support": supports,
                    "d_star": d_stars,
                }
            )

        return A, b_np, info

    def compute_ab_grid(self, t=0.0, robust_disturbance=False):
        """
        Compute A and b on the full grid.

        Returns
        -------
        A_grid : np.ndarray, shape (*grid.shape, m)
        b_grid : np.ndarray, shape grid.shape
        info : dict
        """
        grid_shape = tuple(self.grid.states.shape[:-1])
        state_dim = int(self.grid.states.shape[-1])
        X = np.asarray(self.grid.states, dtype=float).reshape(-1, state_dim)

        A, b, info_batch = self.compute_ab_state(
            X,
            t=t,
            robust_disturbance=robust_disturbance,
        )

        control_dim = A.shape[1]
        A_grid = A.reshape(*grid_shape, control_dim)
        b_grid = b.reshape(*grid_shape)

        info = {
            "t": float(t),
            "grid_shape": grid_shape,
            "control_dim": control_dim,
            "gamma": self.gamma,
            "u_range": None if self.u_range is None else self.u_range.copy(),
            "d_range": None if self.d_range is None else self.d_range.copy(),
            "V_grid": info_batch["V"].reshape(*grid_shape),
            "LfV_grid": info_batch["LfV"].reshape(*grid_shape),
            "gradV_grid": info_batch["gradV"].reshape(*grid_shape, state_dim),
            "LgV_grid": A_grid.copy(),
        }

        if robust_disturbance:
            dist_dim = info_batch["d_star"].shape[-1]
            info.update(
                {
                    "LhV_grid": info_batch["LhV"].reshape(*grid_shape, dist_dim),
                    "disturbance_support_grid": info_batch["disturbance_support"].reshape(*grid_shape),
                    "d_star_grid": info_batch["d_star"].reshape(*grid_shape, dist_dim),
                }
            )

        return A_grid, b_grid, info


def compute_ab_state(
    value_function,
    grid,
    dynamics,
    x,
    gamma,
    u_range=None,
    d_range=None,
    t=0.0,
    times=None,
    upwind_scheme=None,
    time_interp="nearest",
    robust_disturbance=False,
):
    builder = AdmissibleControlSet(
        dynamics=dynamics,
        grid=grid,
        value_function=value_function,
        gamma=gamma,
        u_range=u_range,
        d_range=d_range,
        times=times,
        upwind_scheme=upwind_scheme,
        time_interp=time_interp,
    )
    return builder.compute_ab_state(x=x, t=t, robust_disturbance=robust_disturbance)



def compute_ab_grid(
    value_function,
    grid,
    dynamics,
    gamma,
    u_range=None,
    d_range=None,
    t=0.0,
    times=None,
    upwind_scheme=None,
    time_interp="nearest",
    robust_disturbance=False,
):
    builder = AdmissibleControlSet(
        dynamics=dynamics,
        grid=grid,
        value_function=value_function,
        gamma=gamma,
        u_range=u_range,
        d_range=d_range,
        times=times,
        upwind_scheme=upwind_scheme,
        time_interp=time_interp,
    )
    return builder.compute_ab_grid(t=t, robust_disturbance=robust_disturbance)


if __name__ == "__main__":
    print(
        "This file defines AdmissibleControlSet, compute_ab_state(...), and compute_ab_grid(...)."
    )
