"""
Stage 0 — 데이터 수집 및 표준화

experiment-design.md의 Stage 0 명세를 그대로 따른다.
입력: Primary_use_phase.zip, Second_life_phase.zip (Zenodo 14859405, Deep Sorting 데이터셋)
출력: cap_raw.parquet, C_full.npz, T_eol.npy, censor_mask.npy, meta.csv

현재 이 파일은 파이프라인 스켈레톤이다. 각 TODO를 실제 압축 해제 ·
파싱 로직으로 채워 넣으면 된다. 데이터셋을 내려받아 config.RAW_DIR
아래 두면 바로 이어서 쓸 수 있는 구조로 짜여 있다.
"""

import numpy as np
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR

MIN_DISCHARGE_CAP_AH = 0.3   # 불완전 사이클 제거 기준
MIN_CYCLES = 20              # 20사이클 미만 셀 제외 (64.csv 등)
NOMINAL_CAP_AH = 2.4          # SOH 정규화 기준


def load_second_life_phase() -> pd.DataFrame:
    """Second_life_phase.zip에서 셀 x 사이클 레벨 방전 용량을 집계한다."""
    # TODO: RAW_DIR / "Second_life_phase.zip" 압축 해제 후 파싱
    raise NotImplementedError("Second_life_phase 파서를 구현하세요")


def load_primary_use_protocol_ids() -> pd.DataFrame:
    """Primary_use_phase.zip에서 셀별 Protocol_ID(16종)를 추출한다."""
    # TODO: RAW_DIR / "Primary_use_phase.zip" 압축 해제 후 파싱
    raise NotImplementedError("Primary_use_phase 파서를 구현하세요")


def build_cap_raw(second_life: pd.DataFrame, protocol: pd.DataFrame) -> pd.DataFrame:
    """cell_id, cycle, cap_Ah, soh, protocol_id 컬럼을 가진 long-format 테이블."""
    df = second_life.merge(protocol, on="cell_id", how="left")
    df = df[df["cap_Ah"] >= MIN_DISCHARGE_CAP_AH]
    cycle_counts = df.groupby("cell_id")["cycle"].count()
    keep_ids = cycle_counts[cycle_counts >= MIN_CYCLES].index
    df = df[df["cell_id"].isin(keep_ids)]
    df["soh"] = df["cap_Ah"] / NOMINAL_CAP_AH
    return df


def build_c_full(cap_raw: pd.DataFrame) -> tuple[np.ndarray, list]:
    """(N_cells, T_max) 우측 절단 허용 행렬. 미관측은 NaN."""
    cell_ids = sorted(cap_raw["cell_id"].unique())
    t_max = int(cap_raw["cycle"].max())
    mat = np.full((len(cell_ids), t_max), np.nan)
    for i, cid in enumerate(cell_ids):
        sub = cap_raw[cap_raw["cell_id"] == cid].sort_values("cycle")
        idx = sub["cycle"].to_numpy(dtype=int) - 1
        mat[i, idx] = sub["cap_Ah"].to_numpy()
    return mat, cell_ids


def compute_t_eol(c_full: np.ndarray, thr_ah: float) -> tuple[np.ndarray, np.ndarray]:
    """셀별 EOL 도달 사이클과 censored 여부."""
    n_cells, t_max = c_full.shape
    t_eol = np.full(n_cells, t_max, dtype=float)
    censored = np.ones(n_cells, dtype=bool)
    for i in range(n_cells):
        below = np.where(c_full[i] < thr_ah)[0]
        if below.size:
            t_eol[i] = below[0] + 1
            censored[i] = False
    return t_eol, censored


def main():
    second_life = load_second_life_phase()
    protocol = load_primary_use_protocol_ids()
    cap_raw = build_cap_raw(second_life, protocol)
    cap_raw.to_parquet(PROCESSED_DIR / "cap_raw.parquet")

    c_full, cell_ids = build_c_full(cap_raw)
    np.savez(PROCESSED_DIR / "C_full.npz", C_full=c_full, cell_ids=cell_ids)

    thr_ah = NOMINAL_CAP_AH * 0.60  # SOH 60%
    t_eol, censored = compute_t_eol(c_full, thr_ah)
    np.save(PROCESSED_DIR / "T_eol.npy", t_eol)
    np.save(PROCESSED_DIR / "censor_mask.npy", censored)

    meta = cap_raw.groupby("cell_id").agg(
        protocol_id=("protocol_id", "first"),
        n_cycles=("cycle", "count"),
    ).reset_index()
    meta["censored"] = censored[: len(meta)]
    meta.to_csv(PROCESSED_DIR / "meta.csv", index=False)

    print(f"{len(cell_ids)}개 셀 처리 완료. censored 비율 = {censored.mean():.1%}")


if __name__ == "__main__":
    main()
