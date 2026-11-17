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


def test_interrupted_save_leaves_no_result_file(tmp_path, monkeypatch):
    """저장 도중 멈추면(여기서는 오류로 흉내) 결과 파일이 생기지 않아, 다시 실행하면 그 과제를 새로 계산한다."""
    monkeypatch.setattr(G, "OUT", tmp_path)
    G._init_worker("raw")
    G._G["cfg"] = S.SearchConfig(sa_enabled=False, n_random_starts=2)
    real = np.savez_compressed

    def broken(fh, **arrays):
        fh.write(b"PK\x03\x04 half written")             # 반쯤 쓰다가
        raise KeyboardInterrupt                           # 강제 종료

    monkeypatch.setattr(np, "savez_compressed", broken)
    rows = np.arange(0, 86, 6)[:12]
    with pytest.raises(KeyboardInterrupt):
        G.run_task(12, 0, rows, 4, 60)
    assert not G.task_path("raw", 12, 0, 4, 60).exists()   # '끝난 과제'로 착각할 파일이 없다
    monkeypatch.setattr(np, "savez_compressed", real)
    path, sec = G.run_task(12, 0, rows, 4, 60)             # 다시 실행하면 새로 계산
    assert sec > 0 and np.load(path)["row"]


# ------------------------------------------------------------ LFP (UConn) — docs/stage1_lfp_rules.md

def test_uconn_thresholds_use_1p2_ah_nominal():
    assert G.thr_of(60, "uconn") == pytest.approx(0.72)
    assert G.thr_of(55, "uconn") == pytest.approx(0.66)
    assert G.thr_of(60) == pytest.approx(1.44)            # NMC는 그대로


def test_uconn_t_eol_matches_preprocess_file_and_all_cells_reach_55():
    C, cycles, _ = G.load_curves("raw", "uconn")
    assert G.cycle6_column(cycles) == 1                   # 같은 5사이클 격자
    t = G.t_eol_from_C(C, cycles, G.thr_of(60, "uconn"))
    assert np.array_equal(t, np.load(G.PROCESSED_DIR / "uconn" / "T_eol_uconn.npy"))
    assert np.isfinite(G.t_eol_from_C(C, cycles, G.thr_of(55, "uconn"))).all()


def test_uconn_excl_variant_is_rejected():
    with pytest.raises(ValueError):
        G.load_curves("excl", "uconn")


def test_uconn_uses_one_full_inventory_and_separate_folder():
    inst = G.make_instances(24, 1000, G.DATASETS["uconn"]["inv"])
    assert len(inst) == 1 and inst[0][0] == 24 and np.array_equal(inst[0][2], np.arange(24))
    assert G.run_dir("raw", "uconn") != G.run_dir("raw")  # NMC 결과와 섞이지 않음


def test_uconn_task_and_finalize_write_lfp_files_without_touching_nmc(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    nmc_fig = tmp_path / "figure_b.png"
    nmc_fig.write_bytes(b"nmc figure")                    # 기존 Figure B가 있다고 치고
    G._init_worker("raw", "uconn")
    G._G["cfg"] = S.SearchConfig(sa_enabled=False, n_random_starts=2)
    rows = np.arange(24)
    path, _ = G.run_task(24, 0, rows, 8, 60)
    row = json.loads(str(np.load(path)["row"]))
    assert row["chemistry"] == "LFP" and row["n_packs"] == 3 and row["n_unused"] == 0
    assert row["V_O10"] >= row["V_Current"] - 1e-9
    C, _, _ = G.load_curves("raw", "uconn")
    sc = P.PackScorer(C, 10, 0.72).objective_for(rows)    # 0.72Ah로 채점됐는지
    cur = np.load(path)["current"]
    assert sum(sc(p) for p in P.packs_from_labels(cur)) == pytest.approx(row["V_Current"])
    inst, gate = G.finalize("raw", "uconn")
    assert (tmp_path / "g1_gate_lfp.csv").exists() and (tmp_path / "figure_b_lfp.png").exists()
    assert not (tmp_path / "g1_gate.csv").exists()        # NMC 표를 만들지 않음
    assert nmc_fig.read_bytes() == b"nmc figure"          # NMC 그림을 덮어쓰지 않음
    assert gate.passes_g1.isna().all()                    # LFP는 판정 없음
