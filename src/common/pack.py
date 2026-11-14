"""Pack performance primitives shared by every stage.

A pack is a set of cells wired in series, so at each measurement point the
weakest cell sets the pack's capacity. The pack is retired the first time that
weakest capacity falls below the threshold.

    m_k = min over cells in pack of C[i, j0_i + k]
    K   = first k with m_k < thr
    V   = 5 * sum of m_k for k < K

The factor 5 is there because one 0.2C measurement stands for the five cycles
that follow it.

Index rule (shared across stages):
    real cycle number = 1 + 5 * measurement_idx
    a cell diagnosed for n_i cycles starts its pack life at column j0 = n_i / 5

An assignment is a label array, one entry per inventory cell: the pack number
it belongs to, or UNUSED (-1) for a cell left out. Unused cells add nothing to
the assignment's value, and changing their capacities must not change it.
"""
from __future__ import annotations

import numpy as np

CYCLES_PER_MEASUREMENT = 5
UNUSED = -1


# --------------------------------------------------------------- index rule

def future_measurement_idx(n_i) -> int:
    """Column where a cell's pack life begins, given its diagnostic length."""
    n = int(n_i)
    if n != n_i or n < 0:
        raise ValueError(f"n_i must be a non-negative integer, got {n_i!r}")
    if n % CYCLES_PER_MEASUREMENT:
        raise ValueError(
            f"n_i must be a multiple of {CYCLES_PER_MEASUREMENT}, got {n}")
    return n // CYCLES_PER_MEASUREMENT


def cycle_of(measurement_idx) -> int:
    """Real cycle number of a column."""
    return 1 + CYCLES_PER_MEASUREMENT * int(measurement_idx)


def _n_of(n_vec, i) -> int:
    """Diagnostic length of cell i; n_vec may be a scalar or per-cell array."""
    return int(n_vec) if np.isscalar(n_vec) else int(n_vec[i])


# ----------------------------------------------------------- pack behaviour

def _pack_trace(C, cells, n_vec, thr):
    """The m_k series up to (not including) the first point below thr."""
    cells = np.asarray(cells, dtype=int).ravel()
    if cells.size == 0:
        raise ValueError("a pack needs at least one cell")

    starts = np.array([future_measurement_idx(_n_of(n_vec, i)) for i in cells])
    room = C.shape[1] - starts
    mins = []
    for k in range(int(room.min())):
        vals = C[cells, starts + k]
        if np.isnan(vals).any():
            bad = cells[np.isnan(vals)]
            raise ValueError(
                f"NaN at measurement {k} for cell(s) {bad.tolist()} "
                f"before the pack reached {thr}; check C_full or the threshold")
        m = vals.min()
        if m < thr:
            return np.asarray(mins, dtype=float)
        mins.append(m)
    raise ValueError(
        f"cells {cells.tolist()} never fall below {thr} within the data; "
        "these cells are censored at this threshold")


def pack_value(C, cells, n_vec, thr) -> float:
    """V of one pack, in Ah*cycle."""
    return CYCLES_PER_MEASUREMENT * float(_pack_trace(C, cells, n_vec, thr).sum())


def pack_eol(C, cells, n_vec, thr) -> int:
    """K, the number of measurement points the pack survives."""
    return int(_pack_trace(C, cells, n_vec, thr).size)


def v_single(C, i, n_i, thr) -> float:
    """A cell's value on its own. The factor 5 is already inside pack_value."""
    return pack_value(C, [i], n_i, thr)


# -------------------------------------------------------------- assignments

def assignment_value(C, packs, n_vec, thr) -> float:
    """Sum of pack values. `packs` is a list of cell-index lists."""
    return float(sum(pack_value(C, p, n_vec, thr) for p in packs))


def packs_from_labels(labels):
    """Label array -> list of packs, ordered by pack number. Unused dropped."""
    labels = np.asarray(labels, dtype=int)
    used = labels[labels != UNUSED]
    if used.size == 0:
        return []
    n_packs = int(used.max()) + 1
    packs = [np.flatnonzero(labels == p) for p in range(n_packs)]
    empty = [p for p, c in enumerate(packs) if c.size == 0]
    if empty:
        raise ValueError(f"pack number(s) {empty} have no cells")
    return packs


def labels_value(C, labels, n_vec, thr, cells=None) -> float:
    """V of an assignment given as a label array.

    `cells` maps label positions to rows of C when the inventory is a subset;
    without it, position j is row j.
    """
    packs = packs_from_labels(labels)
    if cells is not None:
        cells = np.asarray(cells, dtype=int)
        packs = [cells[p] for p in packs]
    return assignment_value(C, packs, n_vec, thr)


def unused_cells(labels, cells=None):
    """Rows of C left out by this assignment."""
    labels = np.asarray(labels, dtype=int)
    idx = np.flatnonzero(labels == UNUSED)
    return idx if cells is None else np.asarray(cells, dtype=int)[idx]


def check_labels(labels, pack_size, n_cells=None) -> None:
    """Every pack holds exactly pack_size cells and pack numbers are 0..P-1."""
    labels = np.asarray(labels, dtype=int)
    if n_cells is not None and labels.size != n_cells:
        raise ValueError(f"expected {n_cells} labels, got {labels.size}")
    expected_packs = labels.size // pack_size
    used = labels[labels != UNUSED]
    if used.size != expected_packs * pack_size:
        raise ValueError(
            f"{used.size} cells are packed, expected "
            f"{expected_packs} x {pack_size} = {expected_packs * pack_size}")
    if expected_packs:
        counts = np.bincount(used, minlength=expected_packs)
        if counts.size != expected_packs or not np.all(counts == pack_size):
            raise ValueError(
                f"pack sizes are {counts.tolist()}, expected all {pack_size}")


# -------------------------------------------------------------- upper bound

def upper_bound(C, cells, n_vec, thr, pack_size) -> float:
    """Upper bound on V over every assignment of this inventory.

    A pack can never outlast its weakest member, so replacing the pack value
    with the smallest v_i in the pack can only raise the total. That relaxed
    problem is solved exactly by keeping the best P * pack_size cells, sorting
    them, cutting into consecutive groups, and summing each group's last value.
    Dropping a cell for a higher-valued one never lowers any group's last
    value, so keeping the top P * pack_size is optimal even though cells may
    be left unused.
    """
    cells = np.asarray(cells, dtype=int).ravel()
    if pack_size < 1:
        raise ValueError("pack_size must be at least 1")
    n_packs = cells.size // pack_size
    if n_packs == 0:
        return 0.0
    v = np.sort([v_single(C, i, _n_of(n_vec, i), thr) for i in cells])[::-1]
    kept = v[:n_packs * pack_size].reshape(n_packs, pack_size)
    return float(kept[:, -1].sum())


# ------------------------------------------------------------ fast scorer

class PackScorer:
    """Precomputed view of C for the millions of calls a search makes.

    Every row is shifted to its own pack start, so a pack's value is one
    column-wise minimum. Exhausted data is stored as -inf: the minimum then
    drops below any threshold, which is how the scorer notices that a curve
    ran out before the pack retired and raises, as the plain path does.
    """

    __slots__ = ("S", "thr", "n_cells", "_calls")

    def __init__(self, C, n_vec, thr):
        C = np.asarray(C, dtype=float)
        starts = np.array([future_measurement_idx(_n_of(n_vec, i))
                           for i in range(C.shape[0])])
        width = C.shape[1] - int(starts.max())
        if width < 1:
            raise ValueError("diagnostic uses up every column of C")
        S = np.empty((C.shape[0], width), dtype=float)
        for i, s0 in enumerate(starts):
            S[i] = C[i, s0:s0 + width]
        S[~np.isfinite(S)] = -np.inf
        self.S = S
        self.thr = float(thr)
        self.n_cells = C.shape[0]
        self._calls = 0

    @property
    def calls(self) -> int:
        """How many pack evaluations have been made."""
        return self._calls

    def pack(self, cells) -> float:
        """V of one pack."""
        self._calls += 1
        m = np.minimum.reduce(self.S[cells])
        below = m < self.thr
        k = int(below.argmax())
        if not below[k]:
            raise ValueError(
                f"cells {list(cells)} never fall below {self.thr}; censored here")
        if m[k] == -np.inf:
            raise ValueError(
                f"cells {list(cells)} ran out of data before reaching "
                f"{self.thr}; check C_full or the threshold")
        return CYCLES_PER_MEASUREMENT * float(m[:k].sum())

    def eol(self, cells) -> int:
        """K of one pack."""
        m = np.minimum.reduce(self.S[cells])
        below = m < self.thr
        k = int(below.argmax())
        if not below[k] or m[k] == -np.inf:
            raise ValueError(f"cells {list(cells)} do not retire inside the data")
        return k

    def single(self, i) -> float:
        return self.pack([i])

    def objective_for(self, cells):
        """Objective for an inventory given as rows of C.

        The search hands the objective *positions* in its label array; this
        turns them back into rows before scoring. Use it whenever the
        inventory is not all of C (e.g. a 64-cell subset): passing `self.pack`
        directly would score the wrong cells without raising.
        """
        cells = np.asarray(cells, dtype=int).ravel()
        return lambda pos: self.pack(cells[np.asarray(pos, dtype=int)])

    def labels(self, labels) -> float:
        """V of an assignment given as a label array."""
        return float(sum(self.pack(p) for p in packs_from_labels(labels)))

    def upper_bound(self, cells, pack_size) -> float:
        """Same bound as the module-level upper_bound, off the cached view."""
        cells = np.asarray(cells, dtype=int).ravel()
        n_packs = cells.size // pack_size
        if n_packs == 0:
            return 0.0
        v = np.sort([self.single(i) for i in cells])[::-1]
        return float(v[:n_packs * pack_size].reshape(n_packs, pack_size)[:, -1].sum())


# ------------------------------------------------- pure-Python reference

def pack_value_reference(C, cells, n_vec, thr) -> float:
    """Same answer as pack_value, written without numpy, for the speed tests."""
    rows = [list(C[i]) for i in cells]
    starts = [future_measurement_idx(_n_of(n_vec, i)) for i in cells]
    total, k = 0.0, 0
    while True:
        vals = []
        for row, s in zip(rows, starts):
            if s + k >= len(row):
                raise ValueError("ran past the end of the data")
            x = row[s + k]
            if x != x:
                raise ValueError("NaN before the pack reached the threshold")
            vals.append(x)
        m = min(vals)
        if m < thr:
            return CYCLES_PER_MEASUREMENT * total
        total += m
        k += 1
