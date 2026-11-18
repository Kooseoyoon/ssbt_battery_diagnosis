"""Stage 3 표현·후처리 단위 테스트 (명세 8절). 실행: python -m pytest tests/test_stage3.py -q

실제 데이터 테스트는 data/processed/C_full.npz, T_eol.npy가 있을 때만 돈다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stage3_repr as R  # noqa: E402

CYC = 1 + 5 * np.arange(390)
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


def fake_cell(q0=1.9, t_eol=301, knee=0.6):
    """단조 감소 가짜 궤적: knee 이후 빨라지고, t_eol 열에서 처음 1.44 미만."""
    u = CYC / t_eol
    s = np.where(u < knee, 0.5 * u / knee, 0.5 + 0.5 * (u - knee) / (1 - knee)) * 1.0
    s = s + 0.002 * (u >= 1)          # t_eol 열에서 1.44 바로 아래
    q = q0 - s * (q0 - R.EOL_AH)
    q[q < 1.2] = 1.2
    return q


def test_curve_to_u_ends():
    q = fake_cell()
    s = R.curve_to_u(q, CYC, 301)
    assert s.shape == (101,)
    assert s[0] == pytest.approx(0.0, abs=1e-12)
    assert s[-1] >= 1.0
    assert np.all(np.diff(s) >= -1e-12)


def test_curve_to_u_rejects_bad_teol_and_nan():
    q = fake_cell()
    with pytest.raises(ValueError):
        R.curve_to_u(q, CYC, 300)              # 격자에 없는 사이클
    q2 = q.copy()
    q2[10] = np.nan
    with pytest.raises(ValueError):
        R.curve_to_u(q2, CYC, 301)


def test_fpca_roundtrip_all_components():
    rng = np.random.default_rng(0)
    S = np.stack([R.curve_to_u(fake_cell(1.85 + 0.1 * rng.random(), 301, 0.4 + 0.4 * rng.random()), CYC, 301)
                  for _ in range(30)])
    fp = R.FPCA().fit(S, var_target=1.0, k_max=100)
    assert np.abs(fp.inverse(fp.transform(S)) - S).max() < 1e-10


def test_fpca_k_rule():
    rng = np.random.default_rng(1)
    S = np.stack([R.curve_to_u(fake_cell(1.9, 301, 0.3 + 0.5 * rng.random()), CYC, 301) for _ in range(40)])
    fp = R.FPCA().fit(S)
    assert 1 <= fp.k <= R.K_MAX
    assert fp.explained >= R.VAR_TARGET or fp.k == R.K_MAX


def test_trajectory_reconstructs_monotone_curve():
    q = fake_cell(1.9, 301, 0.6)
    s = R.curve_to_u(q, CYC, 301)
    qr = R.trajectory(s, q[0], 301, CYC)
    j = int(np.searchsorted(CYC, 301))
    assert np.abs(qr[: j + 1] - q[: j + 1]).max() < 0.01      # 격자 왕복 보간 오차
    assert qr[j] < R.EOL_AH
    assert np.all(np.diff(qr) <= 1e-12)                      # EOL 이후 연장도 비증가


def test_finalize_rules():
    q_true = fake_cell()
    q_pred = R.trajectory(R.curve_to_u(fake_cell(1.9, 401, 0.5), CYC, 401), 1.95, 401, CYC)
    for n in (10, 25, 40):
        t0 = n // 5 - 1
        out = R.finalize(q_true, n, q_pred, CYC)
        assert np.array_equal(out[: t0 + 1], q_true[: t0 + 1])        # 관측 열 그대로
        assert np.all(np.diff(out[t0 + 1:]) <= 1e-12)                 # 예측 열 비증가
        assert out.min() >= 0 and not np.isnan(out).any()
        assert abs(out[t0 + 1] - q_true[t0]) < abs(q_pred[t0 + 1] - q_true[t0])   # 연속 보정
    with pytest.raises(ValueError):
        R.finalize(q_true, 12, q_pred, CYC)


def test_t_eol_of():
    q = fake_cell(1.9, 301)
    assert R.t_eol_of(q, CYC) == 301
    assert np.isnan(R.t_eol_of(np.full(390, 1.9), CYC))


@pytest.mark.skipif(not (DATA / "C_full.npz").exists(), reason="실제 데이터 없음")
def test_real_data_representation():
    d = np.load(DATA / "C_full.npz")
    C, cyc = d["C_full"], d["cycles"]
    t = np.load(DATA / "T_eol.npy")
    S = np.stack([R.curve_to_u(C[i], cyc, t[i]) for i in range(len(t))])
    assert np.isfinite(S).all()
    fp = R.FPCA().fit(S)
    assert 1 <= fp.k <= R.K_MAX
    # 실제 곡선을 k개 성분으로 복원했을 때 EOL까지 용량 RMSE(셀별). 최대 절대 오차는 회복 점프·잡음 때문에
    # 커지므로(11/17 확인: 성분 전부 써도 0.03Ah 수준의 RMSE가 남음) RMSE로 본다
    Sk = fp.inverse(fp.transform(S))
    rmse = []
    for i in range(len(t)):
        q = R.trajectory(Sk[i], C[i, 0], t[i], cyc)
        j = int(np.searchsorted(cyc, t[i]))
        rmse.append(np.sqrt(np.mean((q[: j + 1] - C[i, : j + 1]) ** 2)))
    assert np.median(rmse) < 0.02 and max(rmse) < 0.05
