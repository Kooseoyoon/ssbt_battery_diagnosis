"""Unit tests for src/common/pack.py (spec section 3-3, plus the unused rule)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import pack as P  # noqa: E402

THR = 1.44


def spec_example():
    """2-series example from the spec: V = 33.5 Ah*cycle."""
    C = np.full((2, 8), np.nan)
    C[0, :6] = [1.90, 1.85, 1.80, 1.60, 1.50, 1.43]
    C[1, :5] = [1.95, 1.70, 1.60, 1.50, 1.40]
    return C


def toy(seed=0, n_cells=12, n_cols=40):
    """Monotone decreasing curves that all cross THR inside the window."""
    rng = np.random.default_rng(seed)
    C = np.full((n_cells, n_cols), np.nan)
    for i in range(n_cells):
        start = rng.uniform(1.85, 2.00)
        life = rng.integers(12, n_cols - 2)
        end = rng.uniform(1.15, 1.35)
        C[i, :life] = np.linspace(start, end, life)
    return C


# --------------------------------------------------------------- index rule

def test_future_measurement_idx():
    assert P.future_measurement_idx(0) == 0
    assert P.future_measurement_idx(10) == 2
    assert P.future_measurement_idx(15) == 3
    assert P.future_measurement_idx(40) == 8


def test_cycle_of():
    assert P.cycle_of(0) == 1
    assert P.cycle_of(1) == 6
    assert P.cycle_of(2) == 11


def test_n_not_multiple_of_five_raises():
    for bad in (7, 11, -5, 2.5):
        with pytest.raises(ValueError):
            P.future_measurement_idx(bad)


def test_start_column_moves_with_diagnostic_length():
    C = toy()
    assert P.future_measurement_idx(15) - P.future_measurement_idx(10) == 1
    # a longer diagnostic can only shorten what is left of the pack's life
    assert P.pack_eol(C, [0, 1], 15, THR) <= P.pack_eol(C, [0, 1], 10, THR)


# ------------------------------------------------------------- pack value

def test_spec_example_reproduces_33_5():
    assert P.pack_value(spec_example(), [0, 1], 0, THR) == pytest.approx(33.5)
    assert P.pack_eol(spec_example(), [0, 1], 0, THR) == 4


def test_single_cell_pack_spans_its_own_eol():
    C = toy()
    for i in range(C.shape[0]):
        row = C[i]
        eol_col = int(np.flatnonzero(np.isfinite(row) & (row < THR))[0])
        assert P.pack_eol(C, [i], 0, THR) == eol_col


def test_v_single_matches_pack_value_without_extra_factor():
    C = toy()
    assert P.v_single(C, 3, 10, THR) == pytest.approx(P.pack_value(C, [3], 10, THR))


def test_nan_before_threshold_raises():
    C = spec_example().copy()
    C[0, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        P.pack_value(C, [0, 1], 0, THR)


def test_censored_cell_raises():
    C = np.full((1, 6), 2.0)  # never drops below THR
    with pytest.raises(ValueError, match="censored"):
        P.pack_value(C, [0], 0, THR)


def test_reference_matches_vectorised():
    C = toy(seed=5, n_cells=16)
    rng = np.random.default_rng(1)
    for _ in range(1000):
        cells = rng.choice(16, size=4, replace=False)
        n_vec = rng.choice([0, 10], size=16)
        assert P.pack_value(C, cells, n_vec, THR) == pytest.approx(
            P.pack_value_reference(C, cells, n_vec, THR))


# ------------------------------------------------- assignments and unused

def test_labels_round_trip_and_value():
    C = toy(n_cells=12)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    packs = P.packs_from_labels(labels)
    assert [p.tolist() for p in packs] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert P.labels_value(C, labels, 0, THR) == pytest.approx(
        P.assignment_value(C, packs, 0, THR))


def test_unused_cells_contribute_nothing():
    C = toy(n_cells=13)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, P.UNUSED])
    assert P.unused_cells(labels).tolist() == [12]
    base = P.labels_value(C, labels, 0, THR)

    # rewriting the unused cell's whole curve must not move V
    bumped = C.copy()
    bumped[12, :] = np.nan
    bumped[12, :20] = np.linspace(2.4, 1.0, 20)
    assert P.labels_value(bumped, labels, 0, THR) == pytest.approx(base)

    crushed = C.copy()
    crushed[12, :] = np.nan
    crushed[12, :3] = [1.50, 1.45, 1.40]
    assert P.labels_value(crushed, labels, 0, THR) == pytest.approx(base)


def test_check_labels_accepts_valid_and_rejects_broken():
    P.check_labels([0, 0, 0, 0, 1, 1, 1, 1, P.UNUSED], pack_size=4)
    with pytest.raises(ValueError):                      # pack of the wrong size
        P.check_labels([0, 0, 0, 1, 1, 1, 1, 1], pack_size=4)
    with pytest.raises(ValueError):                      # too many left unused
        P.check_labels([0, 0, 0, 0, P.UNUSED, P.UNUSED, P.UNUSED, P.UNUSED,
                        P.UNUSED], pack_size=4)


# ------------------------------------------------------------ upper bound

def _random_labels(n_cells, pack_size, rng):
    order = rng.permutation(n_cells)
    n_packs = n_cells // pack_size
    labels = np.full(n_cells, P.UNUSED)
    labels[order[:n_packs * pack_size]] = np.repeat(np.arange(n_packs), pack_size)
    return labels


def test_upper_bound_dominates_random_assignments_no_unused():
    C = toy(seed=7, n_cells=12)
    cells = np.arange(12)
    rng = np.random.default_rng(2)
    for pack_size in (2, 3, 4, 6):
        ub = P.upper_bound(C, cells, 0, THR, pack_size)
        for _ in range(300):
            labels = _random_labels(12, pack_size, rng)
            P.check_labels(labels, pack_size)
            assert P.labels_value(C, labels, 0, THR) <= ub + 1e-9


def test_upper_bound_dominates_random_assignments_with_unused():
    C = toy(seed=11, n_cells=13)
    cells = np.arange(13)
    rng = np.random.default_rng(3)
    for pack_size, n_unused in ((4, 1), (5, 3), (6, 1)):
        ub = P.upper_bound(C, cells, 0, THR, pack_size)
        for _ in range(300):
            labels = _random_labels(13, pack_size, rng)
            assert (labels == P.UNUSED).sum() == n_unused
            P.check_labels(labels, pack_size)
            assert P.labels_value(C, labels, 0, THR) <= ub + 1e-9


def test_upper_bound_uses_only_the_best_cells():
    """Adding a hopeless cell to the inventory must not raise the bound."""
    C = toy(seed=13, n_cells=12)
    base = P.upper_bound(C, np.arange(12), 0, THR, 4)
    padded = np.full((13, C.shape[1]), np.nan)
    padded[:12] = C
    padded[12, :3] = [1.50, 1.46, 1.40]          # dies almost at once
    assert P.upper_bound(padded, np.arange(13), 0, THR, 4) == pytest.approx(base)


def test_upper_bound_tight_for_single_cell_packs():
    C = toy(seed=17, n_cells=6)
    cells = np.arange(6)
    ub = P.upper_bound(C, cells, 0, THR, 1)
    exact = sum(P.v_single(C, i, 0, THR) for i in cells)
    assert ub == pytest.approx(exact)


# ------------------------------------------------------------ fast scorer

def test_scorer_matches_plain_path():
    C = toy(seed=23, n_cells=16)
    rng = np.random.default_rng(4)
    for n_i in (0, 10, 20):
        sc = P.PackScorer(C, n_i, THR)
        for _ in range(400):
            cells = rng.choice(16, size=rng.integers(1, 6), replace=False)
            assert sc.pack(cells) == pytest.approx(P.pack_value(C, cells, n_i, THR))
            assert sc.eol(cells) == P.pack_eol(C, cells, n_i, THR)


def test_scorer_handles_mixed_diagnostic_lengths():
    C = toy(seed=29, n_cells=10)
    n_vec = np.array([0, 10, 10, 15, 0, 20, 10, 5, 0, 10])
    sc = P.PackScorer(C, n_vec, THR)
    rng = np.random.default_rng(5)
    for _ in range(200):
        cells = rng.choice(10, size=4, replace=False)
        assert sc.pack(cells) == pytest.approx(P.pack_value(C, cells, n_vec, THR))


def test_scorer_spec_example():
    assert P.PackScorer(spec_example(), 0, THR).pack([0, 1]) == pytest.approx(33.5)


def test_scorer_raises_when_data_runs_out_before_threshold():
    C = np.full((2, 6), np.nan)
    C[0, :6] = [1.90, 1.85, 1.80, 1.70, 1.60, 1.50]   # never crosses
    C[1, :3] = [1.95, 1.90, 1.85]                      # data ends early
    sc = P.PackScorer(C, 0, THR)
    with pytest.raises(ValueError, match="ran out of data"):
        sc.pack([0, 1])
    with pytest.raises(ValueError, match="censored"):
        sc.pack([0])


def test_scorer_upper_bound_matches_module_level():
    C = toy(seed=31, n_cells=13)
    sc = P.PackScorer(C, 10, THR)
    for ps in (2, 4, 5):
        assert sc.upper_bound(np.arange(13), ps) == pytest.approx(
            P.upper_bound(C, np.arange(13), 10, THR, ps))


def test_scorer_labels_value_and_unused():
    C = toy(seed=37, n_cells=13)
    sc = P.PackScorer(C, 10, THR)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, P.UNUSED])
    assert sc.labels(labels) == pytest.approx(P.labels_value(C, labels, 10, THR))
    before = sc.labels(labels)
    C2 = C.copy()
    C2[12, :] = np.nan
    C2[12, :4] = [2.0, 1.9, 1.5, 1.3]
    assert P.PackScorer(C2, 10, THR).labels(labels) == pytest.approx(before)


def test_scorer_counts_calls():
    C = toy(seed=41, n_cells=8)
    sc = P.PackScorer(C, 0, THR)
    assert sc.calls == 0
    sc.pack([0, 1])
    sc.pack([2, 3])
    assert sc.calls == 2


# ------------------------------------------------ fixes after review (11/15)

def test_objective_for_scores_rows_not_positions():
    """On a subset inventory the search hands over positions; they must map to rows."""
    C = toy(seed=4, n_cells=12)
    sc = P.PackScorer(C, 0, THR)
    cells = np.array([3, 7, 9, 11])
    obj = sc.objective_for(cells)
    assert obj([0, 1]) == sc.pack([3, 7])
    assert obj(np.array([2, 3])) == sc.pack([9, 11])
    assert obj([0, 1]) != sc.pack([0, 1])        # the silent mistake it prevents


def test_objective_for_matches_labels_value_on_a_subset():
    C = toy(seed=6, n_cells=12)
    sc = P.PackScorer(C, 10, THR)
    cells = np.array([0, 2, 4, 5, 8, 9, 10, 11, 1])   # 9 cells -> 2 packs of 4 + 1 unused
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, P.UNUSED])
    obj = sc.objective_for(cells)
    via_obj = sum(obj(p) for p in P.packs_from_labels(labels))
    assert via_obj == pytest.approx(P.labels_value(C, labels, 10, THR, cells))
