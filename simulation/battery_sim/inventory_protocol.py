"""Full-inventory diagnostic accounting and provisional companion-change DFF."""
import numpy as np
from .value_metrics import validate_partition

N_INVENTORY = 86
BUDGET_GRID = tuple(range(860, 3441, 430))
ROUND_SIZE = 9


def diagnostic_batch(n_cycles, ranking, checkpoint):
    """All 86 candidates ranked; cap precedes nominal nine-cell batch size."""
    n, order = np.asarray(n_cycles), np.asarray(ranking)
    if n.shape != (86,) or np.any(n < 10) or np.any(n > 40) or np.any(n % 5):
        raise ValueError('Expected 86 diagnostic amounts in 10..40, by five.')
    if not np.issubdtype(order.dtype, np.integer) or not np.array_equal(np.sort(order), np.arange(86)):
        raise ValueError('Rank every inventory cell exactly once, including unused cells.')
    if checkpoint not in BUDGET_GRID or checkpoint < n.sum():
        raise ValueError('Expected an unreached common checkpoint.')
    count = min(9, int((checkpoint - n.sum()) // 5))
    chosen = order[n[order] < 40][:count]
    updated = n.copy()
    updated[chosen] += 5
    return chosen, updated


def companion_dff(reference, samples, n_cells):
    """Experimental definition: mean fraction of pack companions replaced.

    Both used: 1 - |companions_ref intersect companions_sample| / (m - 1).
    Exactly one unused: 1. Both unused: 0. Invariant to pack labels.
    This is assignment instability, not proven diagnostic value or EVOI.
    """
    def companions(packs):
        packs = validate_partition(packs, n_cells)
        if packs.shape[1] < 2:
            raise ValueError('Companion DFF requires pack size >= 2.')
        groups = {}
        for pack in packs:
            members = frozenset(map(int, pack))
            for cell in pack:
                groups[int(cell)] = members - {int(cell)}
        return groups, packs.shape[1]
    ref, size = companions(reference)
    rows = []
    for sample in samples:
        other, other_size = companions(sample)
        if size != other_size:
            raise ValueError('Reference and sampled assignments need the same pack size.')
        row = np.zeros(n_cells)
        for cell in range(n_cells):
            if (cell in ref) != (cell in other):
                row[cell] = 1.
            elif cell in ref:
                row[cell] = 1 - len(ref[cell] & other[cell]) / (size - 1)
        rows.append(row)
    if not rows:
        raise ValueError('Supply at least one sampled assignment.')
    return np.mean(rows, axis=0)
