"""
Stage 0 — Preprocess

입력: Primary_use_phase.zip, Second_life_phase.zip (Zenodo 14859405, Deep Sorting 데이터셋)
출력: cap_raw.parquet, C_full.npz, T_eol.npy, censor_mask.npy, meta.csv
"""

import functools
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR

MIN_DISCHARGE_CAP_AH = 0.3   # 불완전 사이클 제거 기준
MIN_CYCLES = 20              # 20사이클 미만 셀 제외
NOMINAL_CAP_AH = 2.4         # SOH 정규화 기준
EOL_SOH = 0.60               # SOH 60% 기준 EOL 사이클 계산
RPT_CURRENT_MAX_A = 1.0  # A, 판별 기준값: 방전 전류 중앙값이 이보다 작으면 RPT(약 0.48A), 크면 aging(2.4A)
CAP_SOURCE = "rpt" # "rpt" 또는 "aging"

# 0. 파일 경로, Second_life_phase.zip 압축 해제
SECOND_LIFE_DIR = RAW_DIR / "Second_life_phase"
SECOND_LIFE_ZIP = RAW_DIR / "Second_life_phase.zip"
USECOLS = ["Battery_ID", "Cycle_Index", "Current(A)", "Discharge_Capacity(Ah)", "Protocol_ID"]

def _ensure_second_life_extracted() -> None:
    if SECOND_LIFE_DIR.exists() and any(SECOND_LIFE_DIR.rglob("*.csv")):
        return
 
    if not SECOND_LIFE_ZIP.exists():
        raise FileNotFoundError(
            f"{SECOND_LIFE_ZIP} 파일이 존재하지 않습니다."
            f" 구글 드라이브나 Zenodo 14859405에서 다운로드 후 {RAW_DIR}에 파일 있는지 확인."
        )
 
    print(f"{SECOND_LIFE_ZIP.name} 압축 해제 중 → {SECOND_LIFE_DIR} ...")
    SECOND_LIFE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SECOND_LIFE_ZIP) as zf:
        zf.extractall(SECOND_LIFE_DIR)
    print("압축 해제 완료.")

# 1. 각 셀의 사이클 별 Discharge_Capacity 최댓값과 방전 전류 중앙값을 계산하고, rpt/aging 판별
def _summarize_cell(df: pd.DataFrame) -> pd.DataFrame:
    """
    - cap_Ah: 사이클 안 방전 용량의 최댓값 (값이 사이클마다 0부터 다시 쌓임을 확인함)
    - cycle_type: 방전 구간 전류 중앙값으로 rpt / aging 판별
    """
    cap = df.groupby("Cycle_Index")["Discharge_Capacity(Ah)"].max()
    discharge = df[df["Current(A)"] < 0]
    i_med = discharge.groupby("Cycle_Index")["Current(A)"].median().abs()
 
    out = pd.DataFrame({"cycle": cap.index.astype(int), "cap_Ah": cap.to_numpy()})
    out["i_dis_A"] = out["cycle"].map(i_med)
    out["cycle_type"] = np.where(out["i_dis_A"] < RPT_CURRENT_MAX_A, "rpt", "aging")
    # 방전이 아예 없는 사이클(전류 정보 없음)은 판별 불가 → 이후 용량 필터에서 대부분 제거됨
    out.loc[out["i_dis_A"].isna(), "cycle_type"] = "unknown"

    # 중단된 사이클 제거: 저속(RPT) 방전인데 용량이 직전 1C 사이클보다 작으면 측정이 끊긴 것
    # (예: cell 74의 487사이클 0.38Ah). C_full에서 빠지도록 "invalid"로 표시한다.
    prev_aging = out["cap_Ah"].where(out["cycle_type"] == "aging").ffill().shift(1)
    out.loc[(out["cycle_type"] == "rpt") & (out["cap_Ah"] < prev_aging), "cycle_type"] = "invalid"
    return out
 
 
@lru_cache(maxsize=1)
def _read_all_second_life_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSV 86개를 한 번만 읽어서 (cap_df, protocol_df)를 만들어 캐시한다."""
    _ensure_second_life_extracted()
 
    csv_paths = sorted(SECOND_LIFE_DIR.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"{SECOND_LIFE_DIR} 안에 CSV가 없습니다.")
 
    cap_rows, protocol_rows = [], []
    for path in csv_paths:
        df = pd.read_csv(path, usecols=USECOLS)
        if df.empty:
            continue
 
        cell_id = int(df["Battery_ID"].iloc[0])
        protocol_ids = df["Protocol_ID"].dropna().unique()
        if len(protocol_ids) != 1:
            print(f"⚠ {path.name}: Protocol_ID가 {len(protocol_ids)}개 — 첫 값을 사용")
 
        cyc = _summarize_cell(df)
        cyc["cell_id"] = cell_id
        cap_rows.append(cyc)
        protocol_rows.append({"cell_id": cell_id, "protocol_id": int(protocol_ids[0])})
 
    cap_df = pd.concat(cap_rows, ignore_index=True)
    protocol_df = pd.DataFrame(protocol_rows)
 
    dup = protocol_df["cell_id"].duplicated()
    if dup.any():
        raise ValueError(f"같은 Battery_ID가 여러 파일에 있음: {protocol_df.loc[dup, 'cell_id'].tolist()}")
    return cap_df, protocol_df
 
 
def load_second_life_phase() -> pd.DataFrame:
    """반환 컬럼: cell_id, cycle, cycle_type, cap_Ah, i_dis_A"""
    cap_df, _ = _read_all_second_life_csvs()
    return cap_df
 
 
def load_primary_use_protocol_ids() -> pd.DataFrame:
    """셀별 Protocol_ID(16종). Second_life CSV에 이미 있는 값을 쓴다.
 
    반환 컬럼: cell_id, protocol_id
    """
    _, protocol_df = _read_all_second_life_csvs()
    return protocol_df
 

# 2. cap_raw, C_full, T_eol, meta.csv 생성-> 기준에 맞지 않는 데이터는 삭제
def build_cap_raw(second_life: pd.DataFrame, protocol: pd.DataFrame) -> pd.DataFrame:
    """모든 사이클(rpt + aging)을 담은 long-format 테이블."""
    df = second_life.merge(protocol, on="cell_id", how="left")
    df = df[df["cap_Ah"] >= MIN_DISCHARGE_CAP_AH]
    cycle_counts = df.groupby("cell_id")["cycle"].count()
    keep_ids = cycle_counts[cycle_counts >= MIN_CYCLES].index
    df = df[df["cell_id"].isin(keep_ids)].copy()
    df["soh"] = df["cap_Ah"] / NOMINAL_CAP_AH
    return df[["cell_id", "cycle", "cycle_type", "cap_Ah", "soh", "protocol_id"]]
 
 
def build_c_full(cap_raw: pd.DataFrame, source: str = CAP_SOURCE) -> tuple[np.ndarray, list, np.ndarray]:
    """선택한 종류의 사이클만으로 (N_cells, N_cols) 행렬을 만든다.
 
    열은 그 종류 사이클 번호들의 합집합(오름차순). 미관측은 NaN.
    반환: (행렬, cell_ids, cycles) — cycles[j] = j번째 열의 실제 사이클 번호
    """
    sub = cap_raw[cap_raw["cycle_type"] == source]
    cell_ids = sorted(sub["cell_id"].unique())
    cycles = np.sort(sub["cycle"].unique())
    col_of = {c: j for j, c in enumerate(cycles)}
 
    mat = np.full((len(cell_ids), len(cycles)), np.nan)
    for i, cid in enumerate(cell_ids):
        s = sub[sub["cell_id"] == cid]
        mat[i, [col_of[c] for c in s["cycle"]]] = s["cap_Ah"].to_numpy()
    return mat, cell_ids, cycles
 
 
def compute_t_eol(c_full: np.ndarray, cycles: np.ndarray, thr_ah: float) -> tuple[np.ndarray, np.ndarray]:
    """셀별 EOL 도달 사이클 번호와 미도달 여부.
 
    EOL = 용량이 처음으로 thr_ah 아래로 내려간 사이클.
    미도달(censored)이면 마지막으로 관측된 사이클 번호를 넣는다.
    """
    n_cells = c_full.shape[0]
    t_eol = np.zeros(n_cells)
    censored = np.ones(n_cells, dtype=bool)
    for i in range(n_cells):
        below = np.where(c_full[i] < thr_ah)[0]  # NaN은 비교 결과 False라 자동 제외
        if below.size:
            t_eol[i] = cycles[below[0]]
            censored[i] = False
        else:
            observed = np.where(~np.isnan(c_full[i]))[0]
            t_eol[i] = cycles[observed[-1]] if observed.size else np.nan
    return t_eol, censored
 


 
def main():
    second_life = load_second_life_phase()
    protocol = load_primary_use_protocol_ids()
    cap_raw = build_cap_raw(second_life, protocol)
    cap_raw.to_parquet(PROCESSED_DIR / "cap_raw.parquet")
 
    c_full, cell_ids, cycles = build_c_full(cap_raw, CAP_SOURCE)
    np.savez(PROCESSED_DIR / "C_full.npz", C_full=c_full, cell_ids=np.array(cell_ids),
             cycles=cycles, source=CAP_SOURCE)
 
    thr_ah = NOMINAL_CAP_AH * EOL_SOH
    t_eol, censored = compute_t_eol(c_full, cycles, thr_ah)
    np.save(PROCESSED_DIR / "T_eol.npy", t_eol)
    np.save(PROCESSED_DIR / "censor_mask.npy", censored)
 
    meta = (
        cap_raw.groupby("cell_id")
        .agg(protocol_id=("protocol_id", "first"),
             n_cycles=("cycle", "count"),
             last_cycle=("cycle", "max"))
        .reindex(cell_ids)
        .reset_index()
    )
    meta["t_eol"] = t_eol
    meta["censored"] = censored
    meta.to_csv(PROCESSED_DIR / "meta.csv", index=False)
 
    type_counts = cap_raw["cycle_type"].value_counts().to_dict()
    print(f"셀 {len(cell_ids)}개 | 사이클 종류별 개수 {type_counts}")
    print(f"C_full 기준: {CAP_SOURCE} ({c_full.shape[1]}개 열, 사이클 {cycles[0]}~{cycles[-1]})")
    print(f"EOL 기준 {thr_ah:.2f}Ah (SOH {EOL_SOH:.0%}) | 미도달(censored) 비율 {censored.mean():.1%}")
    if censored.mean() > 0.2:
        print("⚠ 미도달 비율이 20%를 넘음 — EOL_SOH 완화 검토 필요 (experiment-design.md Stage 0)")
    if meta["protocol_id"].isna().any():
        print("⚠ protocol_id 누락 셀 있음 — Battery_ID 매칭 확인 필요")

 
 
if __name__ == "__main__":
    main()