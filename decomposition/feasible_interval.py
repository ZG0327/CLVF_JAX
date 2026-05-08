import numpy as np

def feasible_interval_grid_1d(A_grid, b_grid, u_range, tol=1e-9):
    """
    Compute feasible interval [u_low, u_high] at every grid point for 1D control.

    Constraint at each grid point:
        A(x) * u <= b(x),
        u in u_range

    Parameters
    ----------
    A_grid : array-like, shape grid.shape or (*grid.shape, 1)
        Gridwise A values for 1D control.
    b_grid : array-like, shape grid.shape
        Gridwise b values.
    u_range :
        either
        - scalar umax, meaning u in [-umax, umax]
        - length-2 iterable [u_min, u_max]
    tol : float

    Returns
    -------
    u_low_grid : ndarray, shape grid.shape
    u_high_grid : ndarray, shape grid.shape
    feasible_mask : ndarray of bool, shape grid.shape
        True where feasible interval is nonempty.
    """
    A_grid = np.asarray(A_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)

    # allow A_grid to be shape (*grid.shape, 1)
    if A_grid.ndim == b_grid.ndim + 1 and A_grid.shape[-1] == 1:
        A_grid = A_grid[..., 0]

    if A_grid.shape != b_grid.shape:
        raise ValueError(
            f"A_grid and b_grid must have matching spatial shape, got "
            f"{A_grid.shape} and {b_grid.shape}"
        )

    # parse u_range
    if np.isscalar(u_range):
        umax = float(u_range)
        base_low = -umax
        base_high = umax
    else:
        u_range = np.asarray(u_range, dtype=float).reshape(-1)
        if u_range.shape[0] != 2:
            raise ValueError("For 1D control, u_range must be a scalar or [u_min, u_max].")
        base_low = float(u_range[0])
        base_high = float(u_range[1])

    u_low_grid = np.full_like(b_grid, base_low, dtype=float)
    u_high_grid = np.full_like(b_grid, base_high, dtype=float)

    pos = A_grid > tol
    neg = A_grid < -tol
    zero = ~pos & ~neg

    # A > 0  => u <= b/A
    u_high_grid[pos] = np.minimum(u_high_grid[pos], b_grid[pos] / A_grid[pos])

    # A < 0  => u >= b/A
    u_low_grid[neg] = np.maximum(u_low_grid[neg], b_grid[neg] / A_grid[neg])

    # A == 0
    # if b < 0, infeasible; if b >= 0, box constraint unchanged
    feasible_mask = np.ones_like(b_grid, dtype=bool)
    feasible_mask[zero & (b_grid < -tol)] = False

    # interval empty => infeasible
    feasible_mask &= (u_low_grid <= u_high_grid + tol)

    return u_low_grid, u_high_grid, feasible_mask



def feasible_interval_state_1d(A, b, u_range, tol=1e-9):
    """
    Compute feasible interval [u_low, u_high] for 1D control at a single state.

    Constraint:
        A * u <= b,
        u in u_range

    Parameters
    ----------
    A : scalar, array-like shape (1,), or array-like shape (1,1)
    b : scalar or array-like shape (1,)
    u_range :
        either
        - scalar umax, meaning u in [-umax, umax]
        - length-2 iterable [u_min, u_max]
    tol : float

    Returns
    -------
    interval : tuple or None
        (u_low, u_high) if feasible, else None
    feasible : bool
    """
    A = float(np.asarray(A, dtype=float).reshape(-1)[0])
    b = float(np.asarray(b, dtype=float).reshape(-1)[0])

    # parse u_range
    if np.isscalar(u_range):
        umax = float(u_range)
        u_low = -umax
        u_high = umax
    else:
        u_range = np.asarray(u_range, dtype=float).reshape(-1)
        if u_range.shape[0] != 2:
            raise ValueError("For 1D control, u_range must be a scalar or [u_min, u_max].")
        u_low = float(u_range[0])
        u_high = float(u_range[1])

    if A > tol:
        u_high = min(u_high, b / A)
    elif A < -tol:
        u_low = max(u_low, b / A)
    else:
        if b < -tol:
            return None, False

    feasible = (u_low <= u_high + tol)
    if not feasible:
        return None, False

    return (u_low, u_high), True