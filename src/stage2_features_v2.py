"""Stage 2 (재시작, 11/4) — 기준 양 35개 × 집계 → 특징 추출.

`stage2_feature_candidates_v2.xlsx`의 '특징 후보' 시트에서 물리·정의 관계로 제거한 15개를 뺀
기준 양 34개를 계산한다(35개 중 dvdq_spacing은 계산 불가로 뺐다: 이 셀들의 0.2C dV/dQ에는
양 끝을 빼면 뚜렷한 피크가 둘이 아니라 넓은 골 하나뿐이고, 그 골은 dQ/dV 주 피크와 같은 점이다). 각 기준 양은 사이클마다 값이 하나씩 나오고(측정 사이클 또는 1C 사이클),
진단량 n의 특징은 사이클 1~n 값만 집계해서 만든다.

집계 (특징 이름 = 기준양__집계):
  측정 사이클(0.2C) 기준 양 13개 : last, delta, slope1(값 3개 이상), slope2(값 4개 이상)
  1C 사이클 기준 양 15개         : last, delta, mean, std, slope1, slope2
  한 값 기준 양 6개              : single (ΔQ(V) 계열)
  1C 기준 양은 측정 사이클 바로 다음 1C 사이클(2, 7, 12, …)을 뺀 값만 집계한다(n = 10에서 6개).
  first는 넣지 않는다. delta = last − first라 셋 중 둘이면 충분하다.
  측정 사이클은 n = 10에서 값이 2개(사이클 1·6)라 slope1이 delta ÷ 5와 같아져서, 값 3개 이상일 때만 계산한다.

실행 (저장소 루트):
    python src/stage2_features_v2.py                  # 86셀 전부
    python src/stage2_features_v2.py --jobs 8         # 병렬(파일 읽기가 대부분이라 효과 큼)
    python src/stage2_features_v2.py --limit 2 --check-leak
입력:  data/raw/Second_life_phase/*.csv, data/processed/C_full.npz(셀 순서), cap_raw.parquet(있으면 invalid 사이클 제외)
출력:  data/processed/features_v2.npz      X (셀, 7, D), cell_ids, budgets, feature_names, base_names
       data/processed/features_v2_long.csv  셀 × 진단량 × D (엑셀 확인용)
       data/processed/cycle_summary_v2.csv  셀 × 사이클별 기준 양 원값(디버깅용)
       (이전 features.npz는 덮어쓰지 않는다)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

BUDGETS = (10, 15, 20, 25, 30, 35, 40)
MAX_CYCLE = max(BUDGETS)

COLS = {
    "t": "Test_Time(s)", "st": "Step_Time(s)", "cyc": "Cycle_Index", "step": "Step_Index",
    "I": "Current(A)", "V": "Voltage(V)", "Qc": "Charge_Capacity(Ah)", "Qd": "Discharge_Capacity(Ah)",
    "Ec": "Charge_Energy(Wh)", "Ed": "Discharge_Energy(Wh)", "IR": "Internal_Resistance(Ohm)",
    "T": "Aux_Temperature_1(C)", "bid": "Battery_ID",
}

# 시험 프로토콜 상수 (13.csv로 확인: 충전 1.2A CC → 4.2V CV → 0.048A, 방전 0.48A(측정) 또는 2.4A(1C))
I_CC = 1.2
CC_TOL = 0.01
RPT_MAX_I = 1.0
MIN_DCHG_AH = 0.3
TEMP_RISE_MIN = 0.5      # 1C 방전 온도 상승 중앙값이 이보다 작으면 센서 이상(셀 67·68) → 온도 특징 전부 결측

V_GRID = np.arange(3.05, 4.10, 0.002)        # 방전 dQ/dV·ΔQ(V) 공통 격자(2mV)
V_GRID_CHG = np.arange(3.40, 4.19, 0.002)    # 충전 정전류 구간 dQ/dV 격자
P1_WIN, P2_WIN = (3.6, 3.8), (3.4, 3.6)      # 주·부 피크 창. p1 면적 = 3.8→3.6V 구간 방전 용량
Q_HIGH_V, Q_LOW_V = 3.8, 3.5                 # q_rpt_high = 4.2→3.8V 용량, q_rpt_low = 3.5→3.0V 용량
SG_WIN = 31                                  # savgol 창(격자 31점 ≈ 60mV)

# (기준 양, 사이클 종류, 단위, 설명)  종류: rpt = 측정 사이클, 1c = 1C 사이클, single = 한 값
BASES = [
    ("q_rpt",           "rpt", "Ah",   "0.2C 방전 용량"),
    ("q_rpt_high",      "rpt", "Ah",   "0.2C 방전 4.2→3.8V 구간 용량"),
    ("q_rpt_low",       "rpt", "Ah",   "0.2C 방전 3.5→3.0V 구간 용량"),
    ("v_mean_dchg_rpt", "rpt", "V",    "0.2C 방전 에너지 ÷ 용량"),
    ("dqdv_p1_h",       "rpt", "Ah/V", "0.2C dQ/dV 주 피크(3.6~3.8V) 높이"),
    ("dqdv_p1_v",       "rpt", "V",    "같은 피크 전압"),
    ("dqdv_p1_area",    "rpt", "Ah",   "3.8→3.6V 구간 방전 용량(= 그 구간 dQ/dV 적분)"),
    ("dqdv_p2_h",       "rpt", "Ah/V", "0.2C dQ/dV 부 피크(3.4~3.6V) 높이"),
    ("dqdv_p2_v",       "rpt", "V",    "같은 피크 전압"),
    ("ir_chg",          "rpt", "Ohm",  "충전 후 휴지 다음 펄스 저항(Step 14)"),
    ("ir_dchg",         "rpt", "Ohm",  "방전 후 휴지 다음 펄스 저항(Step 17)"),
    ("t_rise_rpt",      "rpt", "°C",   "0.2C 방전 중 최고 − 시작 온도"),
    ("rate_ratio",      "rpt", "-",    "바로 다음 1C 사이클 방전 용량 ÷ 이 측정 사이클 0.2C 용량"),
    ("ce",              "1c",  "-",    "1C 사이클 방전 용량 ÷ 같은 사이클 충전 용량"),
    ("v_mean_dchg_1c",  "1c",  "V",    "1C 방전 에너지 ÷ 용량"),
    ("cv_frac",         "1c",  "-",    "정전압 충전 용량 ÷ 전체 충전 용량"),
    ("v_chg_start",     "1c",  "V",    "충전 첫 기록 전압 − 직전 사이클 방전 후 휴지 끝 전압"),
    ("dqdv_chg_p_h",    "1c",  "Ah/V", "정전류 충전 구간 dQ/dV 최댓값(3.5~4.1V)"),
    ("r_dc_1c",         "1c",  "Ohm",  "(충전 후 휴지 끝 전압 − 1C 방전 첫 기록 전압) ÷ 방전 전류"),
    ("v_hyst",          "1c",  "V",    "평균 충전 전압 − 평균 방전 전압(같은 1C 사이클)"),
    ("v_jump_dchg",     "1c",  "V",    "방전 후 휴지 첫 기록 전압 − 방전 마지막 전압"),
    ("relax_dchg",      "1c",  "V",    "방전 후 휴지 동안 전압 상승량"),
    ("relax_dchg_tau",  "1c",  "s",    "방전 후 휴지 전압에 V∞ − A·e^(−t/τ) 맞춘 τ"),
    ("relax_chg",       "1c",  "V",    "충전 후 휴지 동안 전압 하강량"),
    ("t_rise_1c",       "1c",  "°C",   "1C 방전 중 최고 − 시작 온도"),
    ("t_rise_chg",      "1c",  "°C",   "충전 중 최고 − 시작 온도"),
    ("cool_rate",       "1c",  "°C/s", "1C 방전 후 휴지 (시작 − 끝 온도) ÷ 휴지 시간"),
    ("heat_integral",   "1c",  "°C·s", "1C 방전 중 (온도 − 시작 온도)의 시간 적분"),
    ("dq_var",          "single", "log10 Ah²", "첫·마지막 측정 사이클 ΔQ(V) 분산(log10)"),
    ("dq_min",          "single", "Ah",  "ΔQ(V) 최솟값"),
    ("dq_mean",         "single", "Ah",  "ΔQ(V) 평균"),
    ("dq_skew",         "single", "-",   "ΔQ(V) 왜도"),
    ("dq_kurt",         "single", "-",   "ΔQ(V) 첨도"),
    ("dq1c_var",        "single", "log10 Ah²", "첫·마지막 1C 사이클(직전 측정 사이클 다음 것 제외) ΔQ(V) 분산(log10)"),
]
assert len(BASES) == 34
TEMP_BASES = {"t_rise_rpt", "t_rise_1c", "t_rise_chg", "cool_rate", "heat_integral"}
AGGS = {"rpt": ("last", "delta", "slope1", "slope2"),
        "1c": ("last", "delta", "mean", "std", "slope1", "slope2"),
        "single": ("single",)}
MIN_PTS = {"last": 1, "delta": 2, "mean": 1, "std": 2, "slope1": 2, "slope2": 3, "single": 1}
MIN_PTS_RPT = {"slope1": 3, "slope2": 4}     # 측정 사이클은 점이 적어서 기준을 올린다

FEATURE_NAMES = [f"{b}__{a}" for b, kind, _, _ in BASES for a in AGGS[kind]]
FEATURE_BASE = [b for b, kind, _, _ in BASES for a in AGGS[kind]]


# ---------------------------------------------------------------- 읽기와 구간 나누기
def load_cell(path: Path, max_cycle: int = MAX_CYCLE) -> pd.DataFrame:
    chunks = []
    for ch in pd.read_csv(path, usecols=list(COLS.values()), chunksize=200_000):
        chunks.append(ch[ch[COLS["cyc"]] <= max_cycle])
        if ch[COLS["cyc"]].min() > max_cycle:
            break
    return pd.concat(chunks, ignore_index=True).rename(columns={v: k for k, v in COLS.items()})


def split_steps(cyc: pd.DataFrame) -> dict:
    """한 사이클을 역할별 구간으로 나눈다. Step_Index 번호가 아니라 전류로 판별한다."""
    out = {"chg": None, "dchg": None, "rest_after_chg": None, "rest_after_dchg": None, "pulses": []}
    prev = None
    for _, g in cyc.groupby("step", sort=False):
        if len(g) <= 2:
            out["pulses"].append((prev, g))
            continue
        med = g["I"].median()
        role = "chg" if med > 0.01 else "dchg" if med < -0.01 else "rest"
        if role in ("chg", "dchg"):
            if out[role] is None:
                out[role] = g
            prev = role
        elif prev == "chg" and out["rest_after_chg"] is None:
            out["rest_after_chg"] = g
        elif prev == "dchg" and out["rest_after_dchg"] is None:
            out["rest_after_dchg"] = g
    return out


# ---------------------------------------------------------------- 곡선 계산
def _sorted_unique(x, y):
    o = np.argsort(x)
    x, y = x[o], y[o]
    x, i = np.unique(x, return_index=True)
    return x, y[i]


def q_on_grid(v, q, grid):
    v, q = _sorted_unique(v, q)
    return np.interp(grid, v, q, left=np.nan, right=np.nan)


def q_at_v(v, q, v0):
    """방전 곡선에서 전압 v0까지 방전한 용량."""
    v, q = _sorted_unique(v, q)
    return float(np.interp(v0, v, q)) if v[0] <= v0 <= v[-1] else np.nan


def smoothed_dqdv(v, q, grid, sign):
    qg = q_on_grid(v, q, grid)
    ok = ~np.isnan(qg)
    if ok.sum() < 2 * SG_WIN:
        return None, None
    vg, qg = grid[ok], qg[ok]
    d = sign * np.gradient(qg, vg)
    return vg, savgol_filter(d, SG_WIN, 3)


def peak_in(vg, d, win):
    """창 안 최댓값과 그 전압(꼭짓점 주변 2차식 보간). 최댓값이 창 경계면 피크가 아니라서 결측."""
    if vg is None:
        return np.nan, np.nan
    m = (vg >= win[0]) & (vg <= win[1])
    if m.sum() < 5:
        return np.nan, np.nan
    vw, dw = vg[m], d[m]
    i = int(np.argmax(dw))
    if i == 0 or i == len(dw) - 1:
        return np.nan, np.nan
    y0, y1, y2 = dw[i - 1], dw[i], dw[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(vw[i] + off * (vw[1] - vw[0])), float(y1 - 0.25 * (y0 - y2) * off)


def relax_tau(rest: pd.DataFrame):
    t = (rest["st"] - rest["st"].iloc[0]).to_numpy()
    v = rest["V"].to_numpy()
    if len(t) < 10 or v[-1] - v[0] <= 0:
        return np.nan
    f = lambda t, a, b, tau: a - b * np.exp(-t / tau)  # noqa: E731
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(f, t, v, p0=(v[-1], v[-1] - v[0], 100.0),
                             bounds=([v[-1] - 0.5, 0, 1.0], [v[-1] + 0.5, 2.0, 1e4]), maxfev=2000)
        return float(p[2])
    except (RuntimeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- 사이클별 기준 양
def summarize_cycles(df: pd.DataFrame, invalid: set[int]) -> pd.DataFrame:
    """사이클마다 한 줄. 각 값은 그 사이클(v_chg_start만 직전 사이클 휴지 끝 전압 포함) 데이터로만 계산."""
    rows, prev_rest_end = [], None
    for c, cyc in df.groupby("cyc"):
        c = int(c)
        s = split_steps(cyc)
        d, ch = s["dchg"], s["chg"]
        valid = c not in invalid and d is not None and ch is not None and d["Qd"].max() >= MIN_DCHG_AH
        if valid:
            r = {"cyc": c, "type": "rpt" if d["I"].abs().median() < RPT_MAX_I else "1c"}
            qd, qc, ed, ec = d["Qd"].max(), ch["Qc"].max(), d["Ed"].max(), ch["Ec"].max()
            r["Qd"] = qd
            r["v_mean_dchg"] = ed / qd
            r["T_rise_dchg"] = d["T"].max() - d["T"].iloc[0]
            dv, dq = d["V"].to_numpy(), d["Qd"].to_numpy()
            if r["type"] == "rpt":
                r["q_rpt_high"] = q_at_v(dv, dq, Q_HIGH_V)
                r["q_rpt_low"] = qd - q_at_v(dv, dq, Q_LOW_V)
                r["dqdv_p1_area"] = q_at_v(dv, dq, P1_WIN[0]) - q_at_v(dv, dq, P1_WIN[1])
                vg, dd = smoothed_dqdv(dv, dq, V_GRID, -1)
                r["dqdv_p1_v"], r["dqdv_p1_h"] = peak_in(vg, dd, P1_WIN)
                r["dqdv_p2_v"], r["dqdv_p2_h"] = peak_in(vg, dd, P2_WIN)
                for prev, g in s["pulses"]:
                    val = g["IR"].dropna()
                    if len(val) and prev in ("chg", "dchg"):
                        r["ir_chg" if prev == "chg" else "ir_dchg"] = float(val.iloc[-1])
            else:
                r["ce"] = qd / qc if qc > 0 else np.nan
                r["v_hyst"] = ec / qc - ed / qd if qc > 0 else np.nan
                cc = ch[ch["I"] >= I_CC - CC_TOL]
                r["cv_frac"] = (qc - cc["Qc"].max()) / qc if len(cc) and qc > 0 else np.nan
                r["v_chg_start"] = ch["V"].iloc[0] - prev_rest_end if prev_rest_end is not None else np.nan
                if len(cc):
                    vg, dd = smoothed_dqdv(cc["V"].to_numpy(), cc["Qc"].to_numpy(), V_GRID_CHG, +1)
                    r["dqdv_chg_p_h"] = peak_in(vg, dd, (3.5, 4.1))[1]
                r["t_rise_chg"] = ch["T"].max() - ch["T"].iloc[0]
                r["heat_integral"] = float(np.trapezoid(d["T"] - d["T"].iloc[0], d["st"])) \
                    if hasattr(np, "trapezoid") else float(np.trapz(d["T"] - d["T"].iloc[0], d["st"]))
                rc = s["rest_after_chg"]
                if rc is not None and len(rc) >= 2:
                    r["relax_chg"] = rc["V"].iloc[0] - rc["V"].iloc[-1]
                    r["r_dc_1c"] = (rc["V"].iloc[-1] - d["V"].iloc[0]) / d["I"].abs().median()
                rd = s["rest_after_dchg"]
                if rd is not None and len(rd) >= 2:
                    r["v_jump_dchg"] = rd["V"].iloc[0] - d["V"].iloc[-1]
                    r["relax_dchg"] = rd["V"].iloc[-1] - rd["V"].iloc[0]
                    r["relax_dchg_tau"] = relax_tau(rd)
                    dur = rd["st"].iloc[-1] - rd["st"].iloc[0]
                    r["cool_rate"] = (rd["T"].iloc[0] - rd["T"].iloc[-1]) / dur if dur > 0 else np.nan
            r["_curve"] = (dv, dq)
            rows.append(r)
        rd = s["rest_after_dchg"]
        prev_rest_end = float(rd["V"].iloc[-1]) if rd is not None and len(rd) else None
    out = pd.DataFrame(rows).sort_values("cyc").reset_index(drop=True)
    out = out.rename(columns={"Qd": "q", "v_mean_dchg": "v_mean", "T_rise_dchg": "t_rise"})
    return out


# ---------------------------------------------------------------- 진단량 n의 특징
def _agg(cyc, val, how, min_pts):
    m = ~np.isnan(val)
    cyc, val = cyc[m], val[m]
    if len(val) < min_pts:
        return np.nan
    if how == "last":
        return val[-1]
    if how == "delta":
        return val[-1] - val[0]
    if how == "mean":
        return val.mean()
    if how == "std":
        return val.std(ddof=1)
    if how == "slope1":
        return np.polyfit(cyc, val, 1)[0]
    if how == "slope2":
        return np.polyfit(cyc, val, 2)[0]
    raise ValueError(how)


def _dq_stats(c0, c1):
    a, b = q_on_grid(*c0, V_GRID), q_on_grid(*c1, V_GRID)
    dq = (b - a)[~np.isnan(b - a)]
    if len(dq) < 20:
        return dict.fromkeys(("var", "min", "mean", "skew", "kurt"), np.nan)
    return {"var": np.log10(max(np.var(dq), 1e-14)), "min": dq.min(), "mean": dq.mean(),
            "skew": float(skew(dq)), "kurt": float(kurtosis(dq))}


def features_for_budget(summ: pd.DataFrame, n: int) -> np.ndarray:
    s = summ[summ["cyc"] <= n]
    rpt, c1 = s[s["type"] == "rpt"].copy(), s[s["type"] == "1c"]
    # 측정 사이클 기준 양 열 이름 맞추기
    rpt["q_rpt"], rpt["v_mean_dchg_rpt"], rpt["t_rise_rpt"] = rpt["q"], rpt["v_mean"], rpt["t_rise"]
    q1c = dict(zip(c1["cyc"], c1["q"]))
    rpt["rate_ratio"] = [q1c.get(c + 1, q1c.get(c - 1, np.nan)) / q for c, q in zip(rpt["cyc"], rpt["q"])]
    # 측정 사이클 바로 다음 1C 사이클(2, 7, 12, …)은 0.2C로 더 깊게 방전된 상태에서 충전을 시작해
    # 충전 용량·정전압 비율·충전 시작 점프·쿨롱 효율이 톱니처럼 튄다(13.csv에서 ce 0.87~0.92). 집계에서 뺀다
    c1 = c1[~(c1["cyc"] - 1).isin(set(s.loc[s["type"] == "rpt", "cyc"]))]
    c1 = c1.assign(v_mean_dchg_1c=c1["v_mean"], t_rise_1c=c1["t_rise"])

    f = {}
    for base, kind, _, _ in BASES:
        if kind == "single":
            continue
        fr = rpt if kind == "rpt" else c1
        cyc = fr["cyc"].to_numpy(float)
        val = fr[base].to_numpy(float) if base in fr else np.full(len(fr), np.nan)
        for a in AGGS[kind]:
            mp = MIN_PTS_RPT.get(a, MIN_PTS[a]) if kind == "rpt" else MIN_PTS[a]
            f[f"{base}__{a}"] = _agg(cyc, val, a, mp)
    cr = list(rpt["_curve"])
    st = _dq_stats(cr[0], cr[-1]) if len(cr) >= 2 else dict.fromkeys(("var", "min", "mean", "skew", "kurt"), np.nan)
    for k in ("var", "min", "mean", "skew", "kurt"):
        f[f"dq_{k}__single"] = st[k]
    c1c = list(c1["_curve"])
    f["dq1c_var__single"] = _dq_stats(c1c[0], c1c[-1])["var"] if len(c1c) >= 2 else np.nan
    # 온도 센서 이상(셀 67·68): 1C 방전 온도 상승 중앙값이 0.5°C 미만이면 온도 특징 전부 결측
    if not (np.nanmedian(c1["t_rise"]) >= TEMP_RISE_MIN if len(c1) else False):
        for k in f:
            if k.split("__")[0] in TEMP_BASES:
                f[k] = np.nan
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)


def cell_features(df, invalid):
    summ = summarize_cycles(df, invalid)
    return np.stack([features_for_budget(summ, n) for n in BUDGETS]), summ


def leak_check(df, invalid) -> bool:
    """n 이후 사이클을 잡음으로 바꿔도 진단량 n의 특징이 같은지 확인."""
    base, _ = cell_features(df, invalid)
    rng, ok = np.random.default_rng(0), True
    for b, n in enumerate(BUDGETS[:-1]):
        bad = df.copy()
        fut = bad["cyc"] > n
        for col in ("V", "I", "Qc", "Qd", "Ec", "Ed", "T", "IR"):
            bad.loc[fut, col] = rng.normal(size=fut.sum())
        got, _ = cell_features(bad, invalid)
        same = np.isclose(base[b], got[b], equal_nan=True)
        if not same.all():
            ok = False
            print(f"  누수 의심 n={n}:", np.array(FEATURE_NAMES)[~same])
    return ok


def _one(args):
    path, inv_map = args
    df = load_cell(Path(path))
    if df.empty:
        return None
    bid = int(df["bid"].iloc[0])
    X, summ = cell_features(df, inv_map.get(bid, set()))
    summ = summ.drop(columns=["_curve"])
    summ.insert(0, "cell_id", bid)
    return bid, X, summ, Path(path).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DIR / "Second_life_phase"))
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--check-leak", action="store_true")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    files = sorted(raw.rglob("*.csv"), key=lambda p: p.stem)
    if args.limit:
        files = files[: args.limit]
    inv_map: dict[int, set[int]] = {}
    if (out / "cap_raw.parquet").exists():
        cr = pd.read_parquet(out / "cap_raw.parquet")
        for cid, g in cr[cr["cycle_type"] == "invalid"].groupby("cell_id"):
            inv_map[int(cid)] = set(int(x) for x in g["cycle"])

    if args.check_leak:
        df = load_cell(files[0])
        print("누수 점검(첫 파일):", "통과" if leak_check(df, inv_map.get(int(df["bid"].iloc[0]), set())) else "실패")

    feats, summs = {}, []
    tasks = [(str(p), inv_map) for p in files]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            results = ex.map(_one, tasks)
            for k, res in enumerate(results, 1):
                if res:
                    feats[res[0]] = res[1]; summs.append(res[2])
                    print(f"[{k}/{len(files)}] {res[3]} → Battery_ID {res[0]}")
    else:
        for k, t in enumerate(tasks, 1):
            res = _one(t)
            if res:
                feats[res[0]] = res[1]; summs.append(res[2])
                print(f"[{k}/{len(files)}] {res[3]} → Battery_ID {res[0]}")

    cf = out / "C_full.npz"
    order = [int(c) for c in np.load(cf)["cell_ids"] if int(c) in feats] if cf.exists() else sorted(feats)
    order += [c for c in sorted(feats) if c not in order]
    X = np.stack([feats[c] for c in order])
    np.savez(out / "features_v2.npz", X=X, cell_ids=np.array(order), budgets=np.array(BUDGETS),
             feature_names=np.array(FEATURE_NAMES), base_names=np.array(FEATURE_BASE))
    long = pd.DataFrame(X.reshape(-1, X.shape[2]), columns=FEATURE_NAMES)
    long.insert(0, "budget", np.tile(BUDGETS, len(order)))
    long.insert(0, "cell_id", np.repeat(order, len(BUDGETS)))
    long.to_csv(out / "features_v2_long.csv", index=False, encoding="utf-8-sig")
    pd.concat(summs).to_csv(out / "cycle_summary_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: features_v2.npz {X.shape} (셀, 진단량, 특징 {len(FEATURE_NAMES)}개 = 기준 양 34 × 집계)")
    nan = pd.DataFrame(np.isnan(X).mean(axis=0).T, index=FEATURE_NAMES, columns=BUDGETS)
    nan = nan[(nan > 0).any(axis=1)]
    print(f"\n[결측률이 0이 아닌 특징 {len(nan)}개] (slope1·slope2의 짧은 진단량 결측은 설계상 정상)")
    print(nan.round(2).to_string())


if __name__ == "__main__":
    main()
