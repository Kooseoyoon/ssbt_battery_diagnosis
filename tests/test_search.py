"""Unit tests for src/common/search.py."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import pack as P          # noqa: E402
from common import search as S        # noqa: E402

THR = 1.44


def toy(seed=0, n_cells=12, n_cols=40):
    rng = np.random.default_rng(seed)
    C = np.full((n_cells, n_cols), np.nan)
    for i in range(n_cells):
        life = rng.integers(12, n_cols - 2)
        C[i, :life] = np.linspace(rng.uniform(1.85, 2.00),
                                  rng.uniform(1.15, 1.35), life)
    return C


def setup(seed=0, n_cells=12, n_i=0):
    C = toy(seed, n_cells)
    sc = P.PackScorer(C, n_i, THR)
    return C, sc, sc.pack


def brute_force(sc, n_cells, pack_size):
    """Exact best over every partition, unused cells included."""
    cells = list(range(n_cells))
    n_packs = n_cells // pack_size
    best = -np.inf

    def rec(remaining, packs):
        nonlocal best
        if len(packs) == n_packs:
            best = max(best, sum(sc.pack(list(p)) for p in packs))
            return
        head = remaining[0] if len(remaining) == n_packs * pack_size - \
            sum(len(p) for p in packs) else None
        # choose the next pack; fix the lowest free cell when none may be unused
        pool = remaining
        for combo in itertools.combinations(pool, pack_size):
            if head is not None and combo[0] != head:
                continue
            rec([c for c in pool if c not in combo], packs + [combo])

    rec(cells, [])
    return best


# ---------------------------------------------------------------- starts

def test_sorted_start_keeps_the_best_cells():
    scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    lab = S.sorted_start(scores, pack_size=2)
    assert lab.tolist() == [0, P.UNUSED, 0, 1, 1]      # cell 1 is worst -> unused
    P.check_labels(lab, 2)


def test_random_start_is_valid_and_reproducible():
    a = S.random_start(13, 4, np.random.default_rng(0))
    b = S.random_start(13, 4, np.random.default_rng(0))
    assert np.array_equal(a, b)
    P.check_labels(a, 4)
    assert (a == P.UNUSED).sum() == 1


def test_neighbourhood_size():
    # 12 cells, packs of 4: C(12,2) - 3*C(4,2) = 66 - 18 = 48, no unused
    assert S.neighbourhood_size(12, 4) == 48
    # 13 cells, packs of 4: 48 pack-pack plus 12*1 pack-unused
    assert S.neighbourhood_size(13, 4) == 60
    # 86 cells, packs of 8: 2880 + 80*6
    assert S.neighbourhood_size(86, 8) == 3360


# ----------------------------------------------------------- hill climbing

def test_swap_search_never_loses_ground():
    C, sc, obj = setup(seed=1, n_cells=12)
    rng = np.random.default_rng(0)
    for _ in range(20):
        start = S.random_start(12, 4, rng)
        before = sc.labels(start)
        res = S.swap_search(start, obj, 4, S.SearchConfig(), rng)
        assert res.value >= before - 1e-9
        P.check_labels(res.labels, 4)
        assert res.value == pytest.approx(sc.labels(res.labels))


def test_swap_search_reaches_a_local_optimum():
    C, sc, obj = setup(seed=2, n_cells=12)
    res = S.swap_search(S.random_start(12, 4, np.random.default_rng(1)),
                        obj, 4, S.SearchConfig(), np.random.default_rng(1))
    assert not res.hit_max_sweeps
    st = S._State(res.labels, obj, 4)
    for a in range(12):
        for b in range(12):
            d = st.delta(a, b)
            assert d is None or d[0] <= 1e-9


def test_same_seed_same_result():
    C, sc, obj = setup(seed=3, n_cells=13)
    start = S.random_start(13, 4, np.random.default_rng(7))
    cfg = S.SearchConfig(sa_iters=2000)
    a = S.anneal(start, obj, 4, cfg, np.random.default_rng(99))
    b = S.anneal(start, obj, 4, cfg, np.random.default_rng(99))
    assert a.value == pytest.approx(b.value)
    assert np.array_equal(a.labels, b.labels)


# -------------------------------------------------------- unused mechanics

def test_unused_swaps_are_proposed_and_taken():
    """With a clearly bad cell in the inventory, the search must set it aside."""
    C = toy(seed=4, n_cells=12)
    C = np.vstack([C, np.full((1, C.shape[1]), np.nan)])
    C[12, :3] = [1.50, 1.46, 1.40]                      # dies at once
    sc = P.PackScorer(C, 0, THR)
    # start with the bad cell packed and a good one set aside
    labels = np.array([0, 0, 0, 12 and 0] + [1] * 4 + [2] * 4 + [P.UNUSED])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, P.UNUSED, 2])
    res = S.swap_search(labels, sc.pack, 4, S.SearchConfig(),
                        np.random.default_rng(0), kind="forced")
    assert res.accepted_unused > 0
    assert res.labels[12] == P.UNUSED                   # bad cell ends up out
    P.check_labels(res.labels, 4)


def test_pack_sizes_stay_exact_throughout():
    C, sc, obj = setup(seed=5, n_cells=14)
    rng = np.random.default_rng(2)
    res = S.anneal(S.random_start(14, 4, rng), obj, 4,
                   S.SearchConfig(sa_iters=3000), rng)
    P.check_labels(res.labels, 4, n_cells=14)
    assert (res.labels == P.UNUSED).sum() == 2


# ------------------------------------------------------ against brute force

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force_small_no_unused(seed):
    C, sc, obj = setup(seed=seed, n_cells=8)
    exact = brute_force(sc, 8, 4)
    rng = np.random.default_rng(seed)
    starts = [("random", S.random_start(8, 4, rng)) for _ in range(5)]
    res = S.best_of(starts, obj, 4, S.SearchConfig(sa_iters=3000), rng)
    assert res.value <= exact + 1e-9
    assert res.value >= exact - 1e-9


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_brute_force_small_with_unused(seed):
    C, sc, obj = setup(seed=seed, n_cells=9)
    exact = brute_force(sc, 9, 4)
    rng = np.random.default_rng(seed)
    starts = [("random", S.random_start(9, 4, rng)) for _ in range(6)]
    res = S.best_of(starts, obj, 4, S.SearchConfig(sa_iters=4000), rng)
    assert res.value == pytest.approx(exact)


# ------------------------------------------------------------- best_of

def test_best_of_never_falls_below_a_supplied_start():
    C, sc, obj = setup(seed=6, n_cells=13)
    rng = np.random.default_rng(3)
    baseline = S.sorted_start(C[:, 1], 4)
    v_base = sc.labels(baseline)
    starts = [("current", baseline)] + \
             [("random", S.random_start(13, 4, rng)) for _ in range(3)]
    res = S.best_of(starts, obj, 4, S.SearchConfig(sa_iters=2000), rng)
    assert res.value >= v_base - 1e-9
    assert len(res.log_rows()) == len(starts) * 2        # hill climb + SA each


def test_result_value_matches_a_fresh_rescore():
    C, sc, obj = setup(seed=8, n_cells=13)
    rng = np.random.default_rng(4)
    starts = [("random", S.random_start(13, 4, rng)) for _ in range(3)]
    res = S.best_of(starts, obj, 4, S.SearchConfig(sa_iters=2000), rng)
    assert res.value == pytest.approx(sc.labels(res.labels))


def test_config_round_trips_through_yaml():
    import yaml
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "search.yaml")
    cfg = S.SearchConfig.from_yaml(path)
    assert cfg.n_random_starts == 20
    assert cfg.max_sweeps == 100
    assert cfg.sa_enabled is True
    assert yaml.safe_load(yaml.safe_dump(cfg.to_dict())) == cfg.to_dict()


# ------------------------------------------------------- handoff hardening

def test_pack_larger_than_inventory_is_refused():
    with pytest.raises(ValueError, match="do not fit"):
        S.sorted_start(np.arange(6.0), 8)
    with pytest.raises(ValueError, match="do not fit"):
        S.random_start(6, 8, np.random.default_rng(0))


def test_returned_labels_are_the_caller_s_own_copy():
    C, sc, obj = setup(seed=9, n_cells=13)
    rng = np.random.default_rng(0)
    res = S.swap_search(S.random_start(13, 4, rng), obj, 4, S.SearchConfig(), rng)
    kept = res.value
    res.labels[:] = P.UNUSED                      # vandalise the returned array
    again = S.swap_search(S.random_start(13, 4, np.random.default_rng(0)),
                          obj, 4, S.SearchConfig(), np.random.default_rng(0))
    assert again.value == pytest.approx(kept)     # nothing internal was shared


def test_best_of_is_independent_of_start_order():
    C, sc, obj = setup(seed=10, n_cells=13)
    starts = [("s%d" % i, S.random_start(13, 4, np.random.default_rng(i)))
              for i in range(4)]
    cfg = S.SearchConfig(sa_iters=2000)
    fwd = S.best_of(starts, obj, 4, cfg, seed=7)
    rev = S.best_of(starts[::-1], obj, 4, cfg, seed=7)
    assert fwd.value == pytest.approx(rev.value)
    by_kind = {r.kind: r.value for r in fwd.starts}
    for r in rev.starts:                          # each start keeps its own answer
        assert r.value == pytest.approx(by_kind[r.kind])


def test_annealing_survives_a_very_long_cooling_schedule():
    C, sc, obj = setup(seed=12, n_cells=13)
    cfg = S.SearchConfig(sa_iters=200000, sa_cooling=0.999, sa_polish=False)
    res = S.anneal(S.random_start(13, 4, np.random.default_rng(0)), obj, 4,
                   cfg, np.random.default_rng(0))
    assert np.isfinite(res.value)
