"""
Stage 1 — G1 게이트 (개선 여지 정량화)

experiment-design.md Stage 1 명세. V_current vs V_oracle의 pack regret(지표 C)을
팩 크기 x thr x 재고 크기 그리드에서 몬테카를로로 계산해 g1_gate.csv를 만든다.
"""

import itertools

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR

PACK_SIZES = [4, 8, 12, 16]
THR_SOH = [0.55, 0.60, 0.65]
INVENTORY_SIZES = [32, 64, 85]
N_TRIALS = 1000
NOMINAL_CAP_AH = 2.4


def pack_value(cell_curves: np.ndarray, thr_ah: float) -> float:
    """지표 C: EOL까지 누적 min-capacity."""
    m = cell_curves.min(axis=0)
    below = np.where(m < thr_ah)[0]
    t_eol = below[0] if below.size else len(m)
    return float(m[:t_eol].sum())


def oracle_sort_key(c_full: np.ndarray, thr_ah: float) -> np.ndarray:
    n = c_full.shape[0]
    keys = np.zeros(n)
    for i in range(n):
        below = np.where(c_full[i] < thr_ah)[0]
        keys[i] = below[0] if below.size else c_full.shape[1]
    return keys


def assemble_packs(order: np.ndarray, pack_size: int) -> list[np.ndarray]:
    n_packs = len(order) // pack_size
    return [order[k * pack_size:(k + 1) * pack_size] for k in range(n_packs)]


def run_grid():
    data = np.load(PROCESSED_DIR / "C_full.npz")
    c_full = data["C_full"]  # (N, T), NaN padded
    c_full = np.nan_to_num(c_full, nan=0.0)

    rng = np.random.default_rng(0)
    rows = []
    n_total = c_full.shape[0]

    for pack_size, thr_frac, inv_size in itertools.product(
        PACK_SIZES, THR_SOH, INVENTORY_SIZES
    ):
        if inv_size > n_total or inv_size < pack_size:
            continue
        thr_ah = NOMINAL_CAP_AH * thr_frac
        gains = []
        for _ in range(N_TRIALS):
            inv = rng.choice(n_total, size=inv_size, replace=False)
            curves = c_full[inv]

            current_key = curves[:, 0]
            current_order = inv[np.argsort(-current_key)]
            oracle_key = oracle_sort_key(curves, thr_ah)
            oracle_order = inv[np.argsort(oracle_key)]

            v_current = sum(
                pack_value(c_full[idx], thr_ah)
                for idx in assemble_packs(current_order, pack_size)
            )
            v_oracle = sum(
                pack_value(c_full[idx], thr_ah)
                for idx in assemble_packs(oracle_order, pack_size)
            )
            if v_current > 0:
                gains.append((v_oracle - v_current) / v_current)

        gains = np.array(gains)
        rows.append({
            "pack_size": pack_size,
            "thr_soh": thr_frac,
            "inventory_size": inv_size,
            "oracle_gain_mean": gains.mean(),
            "oracle_gain_p5": np.percentile(gains, 5),
            "oracle_gain_p50": np.percentile(gains, 50),
            "oracle_gain_p95": np.percentile(gains, 95),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "g1_gate.csv", index=False)
    print(df.sort_values("oracle_gain_p5", ascending=False).head())


if __name__ == "__main__":
    run_grid()
