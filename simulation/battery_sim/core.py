"""Pack scoring with stable cell IDs and explicit remainder handling."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sorted_assignment(lifetimes: ArrayLike, pack_size: int) -> NDArray[np.int64]:
    """Sort cells into equal packs. Reject leftovers rather than silently drop cells."""
    values = np.asarray(lifetimes, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("RUL must be a finite, nonnegative 1-D array.")
    if pack_size < 1 or values.size == 0 or values.size % pack_size:
        raise ValueError("Use a positive pack size dividing the number of cells exactly.")
    return np.argsort(values, kind="stable").reshape(-1, pack_size)


def pack_lifetimes(lifetimes: ArrayLike, assignment: ArrayLike) -> NDArray[np.float64]:
    """Return each pack minimum for one RUL vector or a batch of scenarios.

    Assignment remains fixed across the scenarios. Integer cell indices are IDs,
    not ranks. Every input cell must occur in exactly one pack.
    """
    values = np.asarray(lifetimes, dtype=float)
    packs = np.asarray(assignment)
    if values.ndim not in (1, 2) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("RUL must be finite and nonnegative, with shape (N,) or (S, N).")
    if packs.ndim != 2 or packs.size == 0 or not np.issubdtype(packs.dtype, np.integer):
        raise ValueError("Assignment must be a nonempty integer matrix.")
    if not np.array_equal(np.sort(packs.ravel()), np.arange(values.shape[-1])):
        raise ValueError("Assignment must include every cell exactly once.")
    return values[..., packs].min(axis=-1)


def utility(lifetimes: ArrayLike, assignment: ArrayLike) -> NDArray[np.float64]:
    """Sum pack minima; units are pack-cycles when RUL is in cycles."""
    return pack_lifetimes(lifetimes, assignment).sum(axis=-1)
