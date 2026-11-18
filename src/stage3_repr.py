"""Stage 3 — 궤적 표현 (명세 3절, 5절 3~7).

셀 하나의 용량 궤적 ↔ (log T_eol, FPCA 점수 z₁…z_k) 변환과, 뽑은 출력을 C_full 격자 궤적으로 되돌리는 후처리.
GP는 여기 없다(stage3_gp.py). 이 파일의 함수는 모두 결정적이라(난수 없음) 단위 테스트로 고정한다.

  낙폭 비율   s(c) = (Q₀ − Q(c)) / (Q₀ − EOL)      Q₀ = 사이클 1(열 0) 용량, EOL = 1.44Ah
  수명 정규화 u = c / T_eol, 격자 U_GRID = 0, 0.01, …, 1 (101점)
  FPCA       학습 셀 s(u)에서 평균 μ와 주성분 φ. k = 누적 분산 ≥ 95%인 최소값, 상한 3
  복원       s(u) = μ + Σ zⱼφⱼ → (누적 최댓값) → Q(c) = Q₀ − s(c/T_eol)(Q₀ − EOL), c > T_eol은 직선 연장

사용 예:
    S = np.stack([curve_to_u(C[i], cycles, t_eol[i]) for i in train])
    fp = FPCA().fit(S)                     # fp.k, fp.explained
    z = fp.transform(S)                    # (학습 셀, k)
    s_u = fp.inverse(z_sample)             # (U,)
    q = trajectory(s_u, q0, t_sample, cycles)          # (열 수,)
    q = finalize(c_row, n, q, cycles)                  # 연속 보정 + 관측 열 채우기 + 비증가 + 0 하한
"""
from __future__ import annotations

import numpy as np

EOL_AH = 1.44
U_GRID = np.linspace(0.0, 1.0, 101)
VAR_TARGET, K_MAX = 0.95, 3
TAIL_U = 0.9          # EOL 이후 연장 기울기를 잴 구간 u ∈ [0.9, 1]
DECAY_CYC = 50.0      # 연속 보정 감쇠 척도(사이클)
STEP = 5              # C_full 열 간격(사이클)


# ------------------------------------------------------------------ 셀 곡선 → s(u)
def curve_to_u(c_row, cycles, t_eol, eol=EOL_AH):
    """한 셀의 C_full 행 → U_GRID 위 낙폭 비율 s(u). 열 0 ~ T_eol 열까지만 쓴다."""
    c_row, cycles = np.asarray(c_row, float), np.asarray(cycles, float)
    j_end = int(np.searchsorted(cycles, t_eol))
    if j_end >= len(cycles) or cycles[j_end] != t_eol:
        raise ValueError(f"T_eol {t_eol}이 cycles 격자에 없다")
    q = c_row[: j_end + 1]
    if np.isnan(q).any():
        raise ValueError("EOL 전에 NaN이 있다")
    q0 = q[0]
    s = (q0 - q) / (q0 - eol)
    u = cycles[: j_end + 1] / t_eol
    return np.interp(U_GRID, u, s)          # u < 1/T_eol 구간은 s(사이클 1) = 0으로 고정


# ------------------------------------------------------------------ FPCA
class FPCA:
    """학습 셀 곡선만으로 맞춘다. k는 누적 분산 VAR_TARGET 이상인 최소값, 상한 K_MAX."""

    def fit(self, S, var_target=VAR_TARGET, k_max=K_MAX):
        """var_target·k_max는 테스트용 인자. 본 실행은 기본값(0.95, 3) 고정."""
        S = np.asarray(S, float)
        self.mean = S.mean(0)
        _, sv, vt = np.linalg.svd(S - self.mean, full_matrices=False)
        var = sv ** 2
        self.explained_all = var / var.sum()
        cum = np.cumsum(self.explained_all)
        self.k = int(min(np.searchsorted(cum, var_target - 1e-12) + 1, k_max, len(sv)))
        self.components = vt[: self.k]                     # (k, U)
        self.explained = float(cum[self.k - 1])
        return self

    def transform(self, S):
        return (np.asarray(S, float) - self.mean) @ self.components.T

    def inverse(self, Z):
        return self.mean + np.asarray(Z, float) @ self.components


# ------------------------------------------------------------------ 복원
def trajectory(s_u, q0, t_eol, cycles, eol=EOL_AH):
    """s(u) 하나 → 전체 열의 용량. 예측 구간 비증가는 finalize에서 다시 맞춘다."""
    s_u = np.maximum.accumulate(np.asarray(s_u, float))   # 용량이 다시 오르지 않게
    cycles = np.asarray(cycles, float)
    u = cycles / t_eol
    s = np.interp(u, U_GRID, s_u)
    tail = U_GRID >= TAIL_U
    slope = max(np.polyfit(U_GRID[tail], s_u[tail], 1)[0], 0.0)
    after = u > 1
    s[after] = s_u[-1] + slope * (u[after] - 1)
    return q0 - s * (q0 - eol)


def finalize(c_row, n, q_pred, cycles, decay=DECAY_CYC):
    """연속 보정 → 관측 열(t < n/5)은 C_full 그대로 → 예측 열 비증가(누적 최솟값) → 0 하한.

    n = 진단 사이클 수(5의 배수). 마지막 관측 열 t₀ = n/5 − 1.
    """
    if n % STEP:
        raise ValueError(f"n = {n}은 5의 배수가 아니다")
    t0 = n // STEP - 1
    cycles = np.asarray(cycles, float)
    q = np.asarray(q_pred, float).copy()
    obs = np.asarray(c_row, float)[: t0 + 1]
    if np.isnan(obs).any():
        raise ValueError("관측 구간에 NaN")
    delta = obs[-1] - q[t0]
    fut = slice(t0 + 1, None)
    q[fut] += delta * np.exp(-(cycles[fut] - cycles[t0]) / decay)
    q[fut] = np.minimum.accumulate(q[fut])
    q[: t0 + 1] = obs
    return np.clip(q, 0.0, None)


def t_eol_of(q, cycles, eol=EOL_AH):
    """처음 eol 미만이 되는 열의 사이클(Stage 0 규칙). 없으면 NaN."""
    b = np.where(np.asarray(q) < eol)[0]
    return float(np.asarray(cycles)[b[0]]) if b.size else np.nan
