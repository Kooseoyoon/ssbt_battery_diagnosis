"""Team metric contract: integrated pack capacity, diagnostic-cycle offsets.

This evaluator consumes future truth for scoring only. Policies must receive a
separate observed prefix. Missing capacity is never filled with zero before EOL.
"""

import itertools

import numpy as np


def diagnostic_columns(n_cycles, n_cells):
    n = np.asarray(n_cycles)
    if n.ndim == 0:
        n = np.full(n_cells, n.item())
    if n.shape != (n_cells,) or not np.all(np.isfinite(n)) or np.any(n < 0) or np.any(n % 5):
        raise ValueError('Diagnostic cycles must be nonnegative multiples of five, one per cell.')
    return n.astype(np.int64) // 5


def observed_prefix(capacities, n_cycles, enforce_protocol=True):
    """n=10 reveals columns 0,1 (cycles 1,6); column 2 is still hidden."""
    c = np.asarray(capacities, dtype=float)
    if c.ndim != 2:
        raise ValueError('Expected a cell-by-RPT matrix.')
    columns = diagnostic_columns(n_cycles, len(c))
    if enforce_protocol and (np.any(columns < 2) or np.any(columns > 8)):
        raise ValueError('The deployment diagnostic protocol permits 10..40 cycles per cell.')
    if np.any(columns > c.shape[1]):
        raise ValueError('Diagnostic amount exceeds the matrix.')
    prefixes = [c[i, :j].copy() for i, j in enumerate(columns)]
    if any(not np.isfinite(row).all() for row in prefixes):
        raise ValueError('Missing observations inside a diagnostic prefix.')
    return prefixes


def effective_future(capacities, n_cycles, threshold=1.44):
    """Capacity after diagnosis, zero contribution only *after* observed first EOL.

    Columns are aligned by deployment time k, not by original absolute cycle.
    The zero tail encodes termination; it is not an imputation of missing data.
    """
    c = np.asarray(capacities, dtype=float)
    if c.ndim != 2 or len(c) == 0:
        raise ValueError('Expected a nonempty cell-by-RPT capacity matrix.')
    offsets = diagnostic_columns(n_cycles, len(c))
    curves = []
    for i, start in enumerate(offsets):
        tail = c[i, start:]
        below = np.flatnonzero(np.isfinite(tail) & (tail < threshold))
        if not len(below):
            raise ValueError(f'Cell index {i} has no observed post-diagnosis EOL; censor handling is required.')
        stop = int(below[0])
        if not np.isfinite(tail[:stop]).all() or np.any(tail[:stop] < 0):
            raise ValueError(f'Cell index {i} has missing or invalid capacity before EOL.')
        curves.append(tail[:stop].copy())
    length = max(map(len, curves), default=0)
    aligned = np.zeros((len(c), length))
    for i, curve in enumerate(curves):
        aligned[i, :len(curve)] = curve
    return aligned


def validate_partition(packs, n_cells):
    packs = np.asarray(packs)
    if packs.ndim != 2 or packs.shape[1] < 1 or not np.issubdtype(packs.dtype, np.integer):
        raise ValueError('Expected equal-sized integer packs.')
    if packs.shape[0] != n_cells // packs.shape[1] or packs.size == 0:
        raise ValueError('Use floor(N / pack_size) packs; only the remainder may be unused.')
    if np.any(packs < 0) or np.any(packs >= n_cells) or len(np.unique(packs)) != packs.size:
        raise ValueError('Used cell indices must be unique and inside the full inventory.')
    return packs


def unused_cells(packs, n_cells):
    return np.setdiff1d(np.arange(n_cells), validate_partition(packs, n_cells).ravel())


def swap_neighbors(packs, n_cells):
    """All pack-to-pack and pack-to-unused one-cell swaps, no pack-label swaps."""
    packs = validate_partition(packs, n_cells)
    unused = unused_cells(packs, n_cells)
    for p, q in itertools.combinations(range(len(packs)), 2):
        for a, b in itertools.product(range(packs.shape[1]), repeat=2):
            candidate = packs.copy()
            candidate[p, a], candidate[q, b] = packs[q, b], packs[p, a]
            yield candidate
    for p, a in np.ndindex(packs.shape):
        for cell in unused:
            candidate = packs.copy()
            candidate[p, a] = cell
            yield candidate


def value_from_aligned(aligned, packs):
    aligned = np.asarray(aligned, dtype=float)
    packs = validate_partition(packs, len(aligned))
    if aligned.ndim != 2 or not np.isfinite(aligned).all() or np.any(aligned < 0):
        raise ValueError('Expected finite effective capacities.')
    return 5.0 * aligned[packs].min(axis=1).sum(axis=1)


def pack_values(capacities, n_cycles, packs, threshold=1.44):
    """Per-pack V, in Ah*cycles, stopping before first minimum below threshold."""
    return value_from_aligned(effective_future(capacities, n_cycles, threshold), packs)


def total_value(capacities, n_cycles, packs, threshold=1.44):
    return float(pack_values(capacities, n_cycles, packs, threshold).sum())


def fixed_random_pool(cell_ids, pack_size, seed):
    """LEGACY protocol only: random pre-exclusion, superseded by full inventory."""
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or pack_size < 1:
        raise ValueError('Use unique IDs and a positive pack size.')
    count = len(ids) // pack_size * pack_size
    if not count:
        raise ValueError('Not enough cells for one pack.')
    permutation = np.random.default_rng(seed).permutation(len(ids))
    excluded = np.sort(permutation[:len(ids) - count])
    retained = np.sort(permutation[len(ids) - count:])
    return retained, excluded


def current_only_assignment(capacities, pack_size):
    """Team baseline: everyone n=10, sort column 1 descending, pack consecutively."""
    c = np.asarray(capacities, dtype=float)
    if pack_size < 1 or len(c) < pack_size or c.ndim != 2 or c.shape[1] < 2 or not np.isfinite(c[:, 1]).all():
        raise ValueError('Supply the full inventory and a complete cycle-6 capacity column.')
    count = len(c) // pack_size * pack_size
    return np.argsort(-c[:, 1], kind='stable')[:count].reshape(-1, pack_size)


def regret_components(v_oracle_zero, v_oracle_same_diagnosis, v_method):
    """All terms share V_O(0) as denominator; preserve signed values, never clip."""
    if not np.isfinite([v_oracle_zero, v_oracle_same_diagnosis, v_method]).all() or v_oracle_zero <= 0:
        raise ValueError('Oracle and method values must be finite, with positive V_O(0).')
    r_diag = (v_oracle_zero - v_oracle_same_diagnosis) / v_oracle_zero
    r_assign = (v_oracle_same_diagnosis - v_method) / v_oracle_zero
    r_total = (v_oracle_zero - v_method) / v_oracle_zero
    return {'R_total': r_total, 'R_diag': r_diag, 'R_assign': r_assign,
            'decomposition_error': r_total - r_diag - r_assign}


def regrouping_upper_bound(aligned, pack_size):
    """Valid relaxation: allow a fresh optimal partition at every deployment step.

    A fixed partition cannot exceed this bound. This is an upper bound, not an
    implementable oracle assignment, because cells can change packs per step.
    """
    aligned = np.asarray(aligned, dtype=float)
    if pack_size < 1 or len(aligned) < pack_size:
        raise ValueError('Supply enough cells for a positive pack size.')
    sorted_capacities = np.sort(aligned, axis=0)
    return float(5 * sorted_capacities[len(aligned) % pack_size::pack_size].sum())


def exact_small_oracle(aligned, pack_size):
    """Exhaustive oracle for unit checks only; full 80/84-cell cases are not enumerated."""
    aligned = np.asarray(aligned, dtype=float)
    n = len(aligned)
    if n > 10 or pack_size < 1 or n < pack_size:
        raise ValueError('Exact enumeration requires <=10 cells and at least one pack.')
    def partitions(remaining):
        if not remaining:
            yield []
            return
        first, *rest = remaining
        for other in itertools.combinations(rest, pack_size - 1):
            pack = (first, *other)
            tail = [i for i in rest if i not in other]
            for suffix in partitions(tail):
                yield [pack, *suffix]
    best_value, best = -np.inf, None
    for used in itertools.combinations(range(n), n // pack_size * pack_size):
        for packs in partitions(list(used)):
            value = float(value_from_aligned(aligned, packs).sum())
            if value > best_value:
                best_value, best = value, np.array(packs, dtype=int)
    return best_value, best
