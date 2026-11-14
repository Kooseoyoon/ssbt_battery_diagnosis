"""Assignment search: hill climbing on swaps, with simulated annealing.

The exact problem is out of reach, so a good assignment is found by search.
Stages differ only in what a pack is worth, so the objective is passed in:

    objective(cells) -> float        value of ONE pack

Stage 1 passes a PackScorer's `pack`, which reads the true future curves.
Stage 4 passes the mean over predicted scenarios. A per-pack objective is what
makes incremental scoring possible: a swap touches one or two packs, so only
those are rescored instead of the whole assignment.

An assignment is a label array over the inventory: the pack number a cell sits
in, or UNUSED (-1). With N cells and packs of n, P = floor(N / n) packs are
filled and the remaining N - P*n cells are left unused. Which cells those are
is part of the decision, so the neighbourhood holds two kinds of swap:

    pack <-> pack      two packed cells trade places
    pack <-> unused    a packed cell is set aside for an unused one

Both are plain swaps, so every pack keeps exactly n cells throughout.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field, asdict

import numpy as np

from .pack import UNUSED, check_labels

__all__ = ["SearchConfig", "StartResult", "SearchResult",
           "sorted_start", "random_start", "swap_search", "anneal", "best_of",
           "start_stream"]


# ------------------------------------------------------------------ config

@dataclass
class SearchConfig:
    n_random_starts: int = 20         # read by the caller when it builds starts
    seed: int = 0
    max_sweeps: int = 100
    tol: float = 1e-9                 # a swap must beat this to be accepted

    sa_enabled: bool = True
    sa_iters: int = 20000
    sa_t0: float | None = None        # None -> scale from the data
    sa_t0_quantile: float = 0.5       # which |delta| sets the start temperature
    sa_t0_samples: int = 400
    sa_cooling: float = 0.9995        # used only when sa_end_ratio is None
    sa_end_ratio: float | None = None # final temp / start temp; sets the cooling
                                      # from the iteration count when given
    sa_polish: bool = True            # hill climb once the temperature is gone

    # scale sa_iters with the neighbourhood so one config fits every condition
    sa_iters_per_neighbour: float | None = None

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as fh:
            return cls(**(yaml.safe_load(fh) or {}))

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ results

@dataclass
class StartResult:
    labels: np.ndarray
    value: float
    kind: str                         # which start this came from
    sweeps: int = 0
    hit_max_sweeps: bool = False
    accepted_pack: int = 0
    accepted_unused: int = 0
    sa_accepted_worse: int = 0
    sa_temp_start: float = 0.0        # annealing only: first and last temperature
    sa_temp_end: float = 0.0
    seconds: float = 0.0


@dataclass
class SearchResult:
    labels: np.ndarray
    value: float
    best_kind: str
    seed: int = 0                     # the one number that reproduces this run
    starts: list = field(default_factory=list)
    seconds: float = 0.0

    def log_rows(self):
        """One dict per start, for search_log.csv."""
        return [{"kind": s.kind, "value": s.value, "sweeps": s.sweeps,
                 "hit_max_sweeps": s.hit_max_sweeps,
                 "accepted_pack": s.accepted_pack,
                 "accepted_unused": s.accepted_unused,
                 "sa_accepted_worse": s.sa_accepted_worse,
                 "sa_temp_start": s.sa_temp_start,
                 "sa_temp_end": s.sa_temp_end,
                 "seconds": s.seconds} for s in self.starts]


# ------------------------------------------------------------- start points

def sorted_start(scores, pack_size, n_cells=None):
    """Best-scoring P*n cells, cut into packs in order; the rest unused.

    `scores` is one number per inventory cell, higher meaning better: T_eol
    for the Oracle, cycle-6 capacity for the industry baseline.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size if n_cells is None else int(n_cells)
    if scores.size != n:
        raise ValueError(f"{scores.size} scores for {n} cells")
    return _chunk(np.argsort(-scores, kind="stable"), pack_size, n)


def random_start(n_cells, pack_size, rng):
    return _chunk(rng.permutation(n_cells), pack_size, n_cells)


def _chunk(order, pack_size, n_cells):
    if pack_size < 1:
        raise ValueError("pack_size must be at least 1")
    if pack_size > n_cells:
        raise ValueError(
            f"packs of {pack_size} do not fit in an inventory of {n_cells}")
    n_packs = n_cells // pack_size
    labels = np.full(n_cells, UNUSED, dtype=int)
    labels[order[:n_packs * pack_size]] = np.repeat(np.arange(n_packs), pack_size)
    return labels


# -------------------------------------------------------------------- state

class _State:
    """Labels plus a cached value per pack, so a swap rescores only what moved."""

    def __init__(self, labels, objective, pack_size):
        self.labels = np.asarray(labels, dtype=int).copy()
        self.objective = objective
        self.pack_size = pack_size
        check_labels(self.labels, pack_size)
        self.n_packs = self.labels.size // pack_size
        self.packs = [np.flatnonzero(self.labels == p) for p in range(self.n_packs)]
        self.values = np.array([objective(p) for p in self.packs], dtype=float)
        self.packed = np.flatnonzero(self.labels != UNUSED)
        self.unused = np.flatnonzero(self.labels == UNUSED)

    @property
    def total(self):
        return float(self.values.sum())

    def delta(self, a, b):
        """Gain from swapping cells a and b, with the packs it would make."""
        la, lb = self.labels[a], self.labels[b]
        if la == lb:
            return None
        if la != UNUSED and lb != UNUSED:
            pa = np.where(self.packs[la] == a, b, self.packs[la])
            pb = np.where(self.packs[lb] == b, a, self.packs[lb])
            va, vb = self.objective(pa), self.objective(pb)
            return (va + vb - self.values[la] - self.values[lb],
                    ((la, pa, va), (lb, pb, vb)), "pack")
        inside, outside = (a, b) if la != UNUSED else (b, a)
        p = self.labels[inside]
        new = np.where(self.packs[p] == inside, outside, self.packs[p])
        v = self.objective(new)
        return (v - self.values[p], ((p, new, v),), "unused")

    def apply(self, a, b, moves):
        for p, cells, v in moves:
            self.packs[p] = cells
            self.values[p] = v
        self.labels[a], self.labels[b] = self.labels[b], self.labels[a]
        if UNUSED in (self.labels[a], self.labels[b]):
            self.packed = np.flatnonzero(self.labels != UNUSED)
            self.unused = np.flatnonzero(self.labels == UNUSED)


def neighbourhood_size(n_cells, pack_size):
    """How many distinct swaps the search may try at one state."""
    n_packs = n_cells // pack_size
    packed = n_packs * pack_size
    unused = n_cells - packed
    pack_pack = packed * (packed - 1) // 2 - n_packs * pack_size * (pack_size - 1) // 2
    return pack_pack + packed * unused


# ------------------------------------------------------------- hill climbing

def swap_search(labels, objective, pack_size, cfg=None, rng=None, kind="swap"):
    """Take the first swap that strictly improves V, until none is left."""
    cfg = cfg or SearchConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    st = _State(labels, objective, pack_size)
    res = StartResult(labels=st.labels, value=st.total, kind=kind)

    n = st.labels.size
    for sweep in range(cfg.max_sweeps):
        improved = False
        # one sweep is one full pass over every cell; within a pass each cell
        # takes the first swap that helps, then the pass moves on
        for a in rng.permutation(n):
            for b in rng.permutation(n):
                if st.labels[a] == st.labels[b]:
                    continue
                d = st.delta(a, b)
                if d is None or d[0] <= cfg.tol:
                    continue
                st.apply(a, b, d[1])
                if d[2] == "pack":
                    res.accepted_pack += 1
                else:
                    res.accepted_unused += 1
                improved = True
                break
        res.sweeps = sweep + 1
        if not improved:
            break
    else:
        res.hit_max_sweeps = True

    res.labels, res.value = st.labels.copy(), st.total
    res.seconds = time.perf_counter() - t0
    return res


# --------------------------------------------------------------- annealing

def _start_temperature(st, cfg, rng):
    """A temperature on the scale of the swaps this problem actually offers."""
    if cfg.sa_t0 is not None:
        return float(cfg.sa_t0)
    mags = []
    n = st.labels.size
    for _ in range(cfg.sa_t0_samples):
        a, b = rng.integers(0, n, 2)
        d = st.delta(int(a), int(b))
        if d is not None and d[0] < 0:
            mags.append(-d[0])
    if not mags:
        return 1.0
    return max(float(np.quantile(mags, cfg.sa_t0_quantile)), 1e-12)


def anneal(labels, objective, pack_size, cfg=None, rng=None, kind="sa"):
    """Simulated annealing over the same swaps, then a hill climb to finish."""
    cfg = cfg or SearchConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    t_start = time.perf_counter()
    st = _State(labels, objective, pack_size)
    res = StartResult(labels=st.labels, value=st.total, kind=kind)

    iters = cfg.sa_iters
    if cfg.sa_iters_per_neighbour:
        iters = int(cfg.sa_iters_per_neighbour
                    * neighbourhood_size(st.labels.size, pack_size))

    temp = _start_temperature(st, cfg, rng)
    res.sa_temp_start = temp
    cooling = cfg.sa_cooling
    if cfg.sa_end_ratio is not None and iters > 0:
        cooling = float(cfg.sa_end_ratio) ** (1.0 / iters)
    best_labels, best_value = st.labels.copy(), st.total
    n = st.labels.size

    for _ in range(iters):
        # every draw cools, so the last temperature is set by iters alone and
        # does not depend on how often a draw lands inside one pack
        a, b = (int(x) for x in rng.integers(0, n, 2))
        if st.labels[a] == st.labels[b]:
            temp *= cooling
            continue
        d = st.delta(a, b)
        if d is None:
            temp *= cooling
            continue
        gain = d[0]
        if gain > cfg.tol:
            take = True
        else:
            take = temp > 0 and rng.random() < np.exp(gain / temp)
            if take:
                res.sa_accepted_worse += 1
        if take:
            st.apply(a, b, d[1])
            if d[2] == "pack":
                res.accepted_pack += 1
            else:
                res.accepted_unused += 1
            if st.total > best_value:
                best_labels, best_value = st.labels.copy(), st.total
        temp *= cooling

    res.sa_temp_end = temp
    res.labels, res.value = best_labels.copy(), best_value
    if cfg.sa_polish:
        polished = swap_search(best_labels, objective, pack_size, cfg, rng, kind)
        res.accepted_pack += polished.accepted_pack
        res.accepted_unused += polished.accepted_unused
        res.sweeps = polished.sweeps
        res.hit_max_sweeps = polished.hit_max_sweeps
        res.labels, res.value = polished.labels, polished.value
    res.seconds = time.perf_counter() - t_start
    return res


# ------------------------------------------------------------------ best of

def start_stream(base, kind, which):
    """Random stream for one start, keyed by the start's name (not its place).

    `which` is 0 for the hill climb and 1 for annealing.
    """
    return np.random.default_rng([int(base), zlib.crc32(str(kind).encode()), int(which)])


def best_of(starts, objective, pack_size, cfg=None, rng=None, seed=None):
    """Run every start through hill climbing (and SA), keep the best.

    `starts` is a sequence of (kind, labels). Ties keep the earlier start, so
    passing the industry assignment first makes the result at least as good
    as that baseline and keeps G from turning negative.

    Each start draws its own streams from `seed` (falling back to cfg.seed)
    and its own *name*, not its position, so a start lands on the same answer
    wherever it sits in the list and whether the starts are run one after
    another or in parallel. Start names must therefore be unique.
    """
    cfg = cfg or SearchConfig()
    t0 = time.perf_counter()
    starts = list(starts)
    base = cfg.seed if seed is None else seed
    kinds = [kind for kind, _ in starts]
    if len(set(kinds)) != len(kinds):
        raise ValueError("start names must be unique: they pick the random streams")

    out = []
    for kind, labels in starts:
        out.append(swap_search(labels, objective, pack_size, cfg,
                               start_stream(base, kind, 0), kind))
        if cfg.sa_enabled:
            out.append(anneal(labels, objective, pack_size, cfg,
                              start_stream(base, kind, 1), kind + "+sa"))

    best = max(out, key=lambda r: r.value)
    return SearchResult(labels=best.labels, value=best.value,
                        best_kind=best.kind, seed=int(base), starts=out,
                        seconds=time.perf_counter() - t0)
