"""Tests for src/stage1_g1.py (실제 data/processed 데이터를 씀)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stage1_g1 as G                    # noqa: E402
from common import pack as P             # noqa: E402
from common import search as S           # noqa: E402


def test_t_eol_marks_censored_cells_as_inf():
    C = np.array([[1.9, 1.5, 1.4, np.nan],      # 열 2에서 1.44 아래 -> 사이클 11
                  [1.9, 1.8, 1.7, np.nan]])     # 끝까지 안 내려감 -> +inf
    t = G.t_eol_from_C(C, np.array([1, 6, 11, 16]), 1.44)
    assert t[0] == 11 and np.isinf(t[1])


def test_t_eol_matches_stage0_file_at_60():
    C, cycles, _ = G.load_curves("raw")
    t = G.t_eol_from_C(C, cycles, G.thr_of(60))
    assert np.array_equal(t, np.load(G.PROCESSED_DIR / "T_eol.npy"))


def test_cycle6_column_is_column_1():
    _, cycles, _ = G.load_curves("raw")
    assert G.cycle6_column(cycles) == 1


def test_instances_are_fixed_by_seed_and_have_no_repeats():
    a, b = G.make_instances(86, 3), G.make_instances(86, 3)
    assert [x[:2] for x in a] == [(32, 0), (32, 1), (32, 2), (64, 0), (64, 1), (64, 2), (86, 0)]
    for (s, k, r), (_, _, r2) in zip(a, b):
        assert np.array_equal(r, r2) and len(set(r.tolist())) == s


def test_task_keeps_G_info_nonnegative_and_unused_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    G._init_worker("raw")
    G._G["cfg"] = S.SearchConfig(sa_enabled=False, n_random_starts=3)
    rows = np.arange(0, 86, 6)[:13]                       # 13셀 4직렬 -> 3팩 + 미사용 1
    path, _ = G.run_task(13, 0, rows, 4, 60)
    z = np.load(path)
    row = json.loads(str(z["row"]))
    assert row["V_O10"] >= row["V_Current"] - 1e-9        # Current가 시작점이라 G_info >= 0
    assert row["n_unused"] == 1
    for col in ("unused_Current", "unused_O10", "unused_O0"):
        assert len(json.loads(row[col])) == 1             # 방식마다 미사용 1개씩
    # Current의 미사용 셀 = 사이클 6 용량이 가장 낮은 셀 (9가지 변경 4번)
    C, _, ids = G.load_curves("raw")
    assert json.loads(row["unused_Current"]) == [int(ids[rows[np.argmin(C[rows, 1])]])]
    # 기록한 V와 다시 채점한 V가 같다
    sc = P.PackScorer(C, 10, 1.44).objective_for(rows)
    assert sum(sc(p) for p in P.packs_from_labels(z["current"])) == pytest.approx(row["V_Current"])
    # 다시 실행하면 건너뛴다 (이어서 실행)
    assert G.run_task(13, 0, rows, 4, 60)[1] == 0.0


def test_finalize_takes_the_best_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    G._init_worker("raw")
    G._G["cfg"] = S.SearchConfig(sa_enabled=False, n_random_starts=2)
    rows = np.arange(86)
    G.run_task(86, 0, rows, 16, 60)                       # 탐색 시드 0
    G.run_task(86, 0, rows, 16, 60, search_seed=1)        # 같은 조건, 다른 시드
    inst, gate = G.finalize("raw")
    v0 = [json.loads(str(np.load(f)["row"]))["V_O10"]
          for f in sorted((tmp_path / "assignments" / "raw").glob("*.npz"))]
    assert inst.V_O10.iloc[0] == pytest.approx(max(v0))   # 두 시드 중 더 좋은 쪽으로 갱신
    assert inst.G_info.iloc[0] >= 0
