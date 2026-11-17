"""UConn LFP 2단계 — G1용 용량 행렬 C_uconn.npz와 수명 종료 T_eol_uconn.npy를 만든다.

입력: uconn_scan.py 출력 (data/processed/uconn/uconn_cycles.parquet 또는 .csv, uconn_scan.csv)
      → zip을 다시 읽지 않는다. Batch 2 Part 3이 나중에 들어오면 scan부터 다시 돌리고 이것도 다시 돌린다.

규칙 (G1을 보기 전에 고정, 11/16)
  1. 셀: uconn_scan.csv의 lr_usable = True만
     (0.5C 구간 시작 ≥ 0.90Ah이고 연속 3사이클 0.72Ah 미만까지 내려감)
  2. 사이클: 방전 전류 중앙값 |I|가 0.4~0.8A(0.5C)인 사이클만, 누적 순서대로 2차 수명 사이클 1, 2, 3, …으로 다시 번호를 매긴다.
     Phase 표시는 쓰지 않는다(1차/2차 표시가 율속 전환과 맞지 않는 셀이 있음)
  3. 중단 사이클 제외: 같은 셀 앞뒤 7사이클 중앙값보다 0.03Ah 넘게 낮은 사이클 하나를 invalid로 표시하고 뺀다
     (0.5C 구간 안에서 한 사이클만 갑자기 낮게 찍히는 점. 셀마다 2~5개, 위로 튀는 점은 없음).
     Deep Sorting Stage 0의 '중단된 RPT invalid'와 같은 취지
  4. 휴지 뒤 위로 점프하는 회복은 실제 현상이라 그대로 둔다(Deep Sorting 주 결과와 같음)
  5. 격자: Deep Sorting C_full과 같게 5사이클 간격(2차 수명 사이클 1, 6, 11, …). 격자 사이클이 invalid로 빠졌으면
     같은 셀의 앞뒤 유효 사이클로 선형 보간. 셀 시험이 끝난 뒤는 NaN
  6. T_eol = 격자 용량이 처음 0.72Ah(정격 1.2Ah × 60%) 미만이 된 격자 사이클(Stage 0과 같은 규칙)

출력: data/processed/uconn/
    C_uconn.npz          C_full(셀 × 열), cell_ids(G#C# 문자열), cycles(1, 6, 11, …), source="uconn_0.5C",
                         nominal_Ah=1.2, eol_Ah=0.72
    T_eol_uconn.npy      셀별 T_eol (2차 수명 사이클 번호, 셀 순서 = cell_ids)
    uconn_invalid.csv    뺀 사이클: cell, sl_cycle, cap_Ah, 이웃 중앙값
    uconn_meta.csv       셀별: source, 시작 용량, 2차 수명 사이클 수, invalid 수, T_eol, 1차 수명 방전 전류
    uconn_C_plot.png     격자 궤적

Stage 1(pack.py)에 넘길 때: EOL 기준이 1.44Ah가 아니라 0.72Ah이고, cell_ids가 문자열이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR  # noqa: E402

NOMINAL, EOL = 1.2, 0.72
I_LO, I_HI = 0.4, 0.8
WIN, DIP = 7, 0.03
STEP = 5


def load_cycles(d):
    p = d / "uconn_cycles.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(d / "uconn_cycles.csv")


def main():
    d = PROCESSED_DIR / "uconn"
    scan = pd.read_csv(d / "uconn_scan.csv")
    cyc = load_cycles(d)
    cells = sorted(scan.loc[scan.lr_usable.astype(bool), "cell"],
                   key=lambda c: tuple(int(x) for x in c[1:].replace("C", " ").split()))
    print(f"사용 셀 {len(cells)}개 (uconn_scan.csv lr_usable)")

    series, inval, meta = {}, [], []
    for c in cells:
        g = cyc[cyc.cell == c].sort_values("cum_cycle")
        is_lr = g.i_dchg_med.abs().between(I_LO, I_HI)
        lr = g[is_lr].reset_index(drop=True)
        x = lr.cap_Ah.to_numpy(float)
        med = pd.Series(x).rolling(WIN, center=True, min_periods=3).median().to_numpy()
        bad = x < med - DIP
        sl = np.arange(1, len(x) + 1)
        for i in np.where(bad)[0]:
            inval.append({"cell": c, "sl_cycle": int(sl[i]), "cap_Ah": x[i], "neighbor_median": med[i]})
        ok = ~bad
        grid = np.arange(1, len(x) + 1, STEP)
        v = np.interp(grid, sl[ok], x[ok])
        series[c] = v
        meta.append({"cell": c, "source": g.source.iloc[0], "sl_n_cycles": len(x), "n_invalid": int(bad.sum()),
                     "start_Ah": x[ok][0], "fl_i_dchg_med": g[~is_lr].i_dchg_med.median()})

    T = max(len(v) for v in series.values())
    cycles = 1 + STEP * np.arange(T)
    C = np.full((len(cells), T), np.nan)
    for i, c in enumerate(cells):
        C[i, :len(series[c])] = series[c]
    t_eol = np.full(len(cells), np.nan)
    for i in range(len(cells)):
        b = np.where(C[i] < EOL)[0]
        t_eol[i] = cycles[b[0]] if b.size else np.nan
    for m, t in zip(meta, t_eol):
        m["t_eol"] = t

    np.savez(d / "C_uconn.npz", C_full=C, cell_ids=np.array(cells), cycles=cycles, source="uconn_0.5C",
             nominal_Ah=NOMINAL, eol_Ah=EOL)
    np.save(d / "T_eol_uconn.npy", t_eol)
    pd.DataFrame(inval).to_csv(d / "uconn_invalid.csv", index=False, encoding="utf-8-sig")
    meta = pd.DataFrame(meta)
    meta.to_csv(d / "uconn_meta.csv", index=False, encoding="utf-8-sig")

    print(f"C_uconn: {C.shape} (열 = 2차 수명 사이클 1~{cycles[-1]}, 5사이클 간격)")
    print(f"invalid로 뺀 사이클: {len(inval)}개 (셀당 {meta.n_invalid.min()}~{meta.n_invalid.max()})")
    print(f"시작 용량 {meta.start_Ah.min():.3f}~{meta.start_Ah.max():.3f}Ah")
    print(f"T_eol(0.72Ah): 최소 {np.nanmin(t_eol):.0f}, 중앙값 {np.nanmedian(t_eol):.0f}, 최대 {np.nanmax(t_eol):.0f} "
          f"| 미도달 {int(np.isnan(t_eol).sum())}셀")
    nan_before = sum(np.isnan(C[i, :int(np.argmax(cycles >= t_eol[i]))]).any()
                     for i in range(len(cells)) if not np.isnan(t_eol[i]))
    print(f"EOL 전에 NaN이 있는 셀: {nan_before}개 (0이어야 pack.py가 돈다)")
    print(meta[["cell", "source", "start_Ah", "sl_n_cycles", "n_invalid", "t_eol"]].round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, c in enumerate(cells):
            ax.plot(cycles, C[i], lw=0.8, label=c)
        ax.axhline(EOL, color="k", ls="--", lw=0.8)
        ax.set_xlabel("second-life cycle (0.5C)")
        ax.set_ylabel("capacity (Ah), 5-cycle grid")
        ax.set_title(f"C_uconn: {len(cells)} cells (dashed: EOL 0.72 Ah)")
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(d / "uconn_C_plot.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("그림 저장 실패:", e)
    print(f"저장: {d}/ C_uconn.npz, T_eol_uconn.npy, uconn_invalid.csv, uconn_meta.csv, uconn_C_plot.png")


if __name__ == "__main__":
    main()
