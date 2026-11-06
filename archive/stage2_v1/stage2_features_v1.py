"""Stage 2 — 진단 특징량 추출.

셀 86개 × 진단량 7단계(n = 10, 15, …, 40) × 특징 D개를 계산해 저장한다.
각 진단량 n의 특징은 실제 사이클 1~n의 원본 데이터만으로 계산한다(미래 누수 방지).

실행 (저장소 루트, conda env ssbt-battery):
    python src/stage2_features.py                 # 86셀 전부
    python src/stage2_features.py --limit 3       # 앞 3개 파일만 (빠른 확인)
    python src/stage2_features.py --check-leak    # 누수 점검 포함

입력:  data/raw/Second_life_phase/*.csv (원본 10초 간격 시계열)
       data/processed/C_full.npz          (셀 순서 cell_ids, 없으면 Battery_ID 오름차순)
       data/processed/cap_raw.parquet     (Stage 0이 invalid로 표시한 사이클, 있으면 제외)
출력:  data/processed/features.npz        X (86, 7, D) float, fail (86, 7, D) bool,
                                          cell_ids (86,), budgets (7,), feature_names (D,)
       data/processed/feature_names.json  특징 이름·그룹·단위·설명
       data/processed/features_long.csv   셀 × 진단량 602행 × 특징 D열 (엑셀 확인용)

주의: 여기서는 원래 단위의 값만 저장한다. 결측 대체와 표준화는 fold마다 학습 셀 기준으로
      모델 학습 단계에서 한다(전체 86셀로 미리 하면 검증 셀 정보가 샌다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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

# 시험 프로토콜 상수 (13.csv, 1.csv로 확인: 충전 1.2A CC → 4.2V CV → 0.048A, 방전 0.48A 또는 2.4A)
I_CC = 1.2              # 정전류 충전 전류(A)
CC_TOL = 0.01           # 이 값 이상이면 정전류 구간으로 본다(1.2A − 0.01A)
RPT_MAX_I = 1.0         # 방전 전류 중앙값이 이보다 작으면 측정 사이클(0.2C)
MIN_DCHG_AH = 0.3       # Stage 0 규칙: 0.3Ah 미만 방전은 불완전 사이클
TEMP_RISE_MIN = 0.5     # 방전 후 온도 상승량 최소값
V_GRID = np.arange(3.05, 4.10, 0.002)   # dQ/dV·ΔQ(V)용 공통 전압 격자(2mV, 방전 곡선 범위 안)
DQDV_WINDOW = (3.4, 4.0)                # 주 피크를 찾는 전압 범위

# (이름, 그룹 번호, 그룹 이름, 단위, 설명) — 그룹 번호는 스프레드시트 '특징 후보' 시트의 행 순서
FEATURES = [
    ("cap_rpt_last",      1,  "0.2C 방전 용량",      "Ah", "마지막 측정 사이클의 0.2C 방전 용량"),
    ("cap_1c_last",       2,  "1C 방전 용량",        "Ah", "마지막 1C 사이클의 방전 용량"),
    ("cap_chg_last",      3,  "충전 용량",           "Ah", "마지막 사이클의 충전 용량"),
    ("rate_ratio",        4,  "율속 용량비",         "-",  "마지막 1C 용량 ÷ 마지막 0.2C 용량"),
    ("cc_time_last",      5,  "정전류 구간",         "s",  "마지막 1C 사이클 충전의 정전류 구간 시간"),
    ("cc_cap_last",       5,  "정전류 구간",         "Ah", "마지막 1C 사이클 충전의 정전류 구간 용량"),
    ("cv_time_last",      6,  "정전압 구간",         "s",  "마지막 1C 사이클 충전의 정전압 구간 시간"),
    ("cv_frac_last",      6,  "정전압 구간",         "-",  "정전압 구간 용량 ÷ 전체 충전 용량"),
    ("cv_tau_last",       7,  "정전압 전류 감쇠",    "s",  "정전압 구간 ln(I) 대 시간 기울기의 역수(감쇠 시상수)"),
    ("e_dchg_1c_last",    8,  "충방전 에너지",       "Wh", "마지막 1C 사이클 방전 에너지"),
    ("e_chg_last",        8,  "충방전 에너지",       "Wh", "마지막 1C 사이클 충전 에너지"),
    ("ce_mean",           9,  "효율",               "-",  "1C 사이클 쿨롱 효율(방전 용량 ÷ 충전 용량) 평균"),
    ("ee_mean",           9,  "효율",               "-",  "1C 사이클 에너지 효율(방전 에너지 ÷ 충전 에너지) 평균"),
    ("v_dchg_mean_1c",    10, "평균 방전 전압",      "V",  "마지막 1C 사이클 방전 에너지 ÷ 방전 용량"),
    ("dqdv_peak_v",       11, "dQ/dV 피크",         "V",  "마지막 측정 사이클 0.2C 방전 dQ/dV 주 피크 전압"),
    ("dqdv_peak_h",       11, "dQ/dV 피크",         "Ah/V", "같은 피크의 높이"),
    ("dq_var",            12, "ΔQ(V)",              "Ah²", "첫·마지막 측정 사이클 방전 곡선 ΔQ(V)의 분산"),
    ("dq_min",            12, "ΔQ(V)",              "Ah", "ΔQ(V) 최솟값"),
    ("dq_mean",           12, "ΔQ(V)",              "Ah", "ΔQ(V) 평균"),
    ("ir_after_chg",      13, "펄스 내부저항",       "Ohm", "마지막 측정 사이클, 충전 후 휴지 다음 펄스 저항"),
    ("ir_after_dchg",     13, "펄스 내부저항",       "Ohm", "마지막 측정 사이클, 방전 후 휴지 다음 펄스 저항"),
    ("relax_dchg",        14, "휴지 전압 회복",      "V",  "1C 방전 후 휴지 동안 전압 상승량(평균)"),
    ("relax_chg",         14, "휴지 전압 회복",      "V",  "충전 후 휴지 동안 전압 하강량(1C 사이클 평균)"),
    ("v_jump_dchg",       15, "휴지 첫 점 전압 점프", "V",  "1C 방전 마지막 점 → 휴지 첫 점 전압 차(평균)"),
    ("v_hyst_1c",         16, "충방전 히스테리시스", "V",  "마지막 1C 사이클 평균 충전 전압 − 평균 방전 전압"),
    ("temp_rise_1c",      17, "1C 방전 온도",        "°C", "1C 방전 중 온도 상승폭(평균)"),
    ("temp_max_1c",       17, "1C 방전 온도",        "°C", "1C 방전 중 최고온도(평균)"),
    ("temp_rise_rpt",     18, "0.2C 방전 온도",      "°C", "마지막 측정 사이클 0.2C 방전 중 온도 상승폭"),
    ("cool_rate_1c",      19, "휴지 냉각 속도",      "°C/s", "1C 방전 후 휴지 동안 (시작 − 끝 온도) ÷ 휴지 시간(평균)"),
    ("slope1_cap_1c",     20, "열화 속도",           "Ah/cycle", "1C 방전 용량 대 사이클 1차 회귀 기울기"),
    ("slope2_cap_1c",     20, "열화 속도",           "Ah/cycle²", "1C 방전 용량 대 사이클 2차 회귀의 2차 계수(점 3개 이상)"),
    ("drop_cap_rpt",      20, "열화 속도",           "Ah", "첫 측정 사이클 − 마지막 측정 사이클 0.2C 용량"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]


# ---------------------------------------------------------------- 읽기와 사이클 분해
def load_cell(path: Path, max_cycle: int = MAX_CYCLE) -> pd.DataFrame:
    """원본 CSV에서 필요한 컬럼, 사이클 1~max_cycle만 읽는다."""
    use = list(COLS.values())
    chunks = []
    for ch in pd.read_csv(path, usecols=use, chunksize=200_000):
        chunks.append(ch[ch[COLS["cyc"]] <= max_cycle])
        if ch[COLS["cyc"]].min() > max_cycle:
            break
    df = pd.concat(chunks, ignore_index=True)
    return df.rename(columns={v: k for k, v in COLS.items()})


def split_steps(cyc: pd.DataFrame) -> dict:
    """한 사이클을 역할별 구간으로 나눈다. Step_Index 번호가 아니라 전류로 판별한다."""
    out = {"chg": None, "dchg": None, "rest_after_chg": None, "rest_after_dchg": None, "pulses": []}
    order = []
    for step, g in cyc.groupby("step", sort=False):
        if len(g) <= 2:
            role = "pulse"
        else:
            med = g["I"].median()
            role = "chg" if med > 0.01 else "dchg" if med < -0.01 else "rest"
        order.append((role, g))
    prev = None
    for role, g in order:
        if role in ("chg", "dchg") and out[role] is None:
            out[role] = g
        elif role == "rest":
            if prev == "chg" and out["rest_after_chg"] is None:
                out["rest_after_chg"] = g
            elif prev == "dchg" and out["rest_after_dchg"] is None:
                out["rest_after_dchg"] = g
        elif role == "pulse":
            out["pulses"].append((prev, g))
        if role in ("chg", "dchg"):
            prev = role
    return out


def summarize_cycles(df: pd.DataFrame, invalid: set[int]) -> pd.DataFrame:
    """사이클마다 한 줄 요약(용량, 에너지, 충전 구간, 저항, 휴지, 온도)."""
    rows = []
    for c, cyc in df.groupby("cyc"):
        c = int(c)
        if c in invalid:
            continue
        s = split_steps(cyc)
        d, ch = s["dchg"], s["chg"]
        if d is None or ch is None:
            continue
        qd = d["Qd"].max()
        if qd < MIN_DCHG_AH:
            continue
        r = {"cyc": c, "type": "rpt" if d["I"].abs().median() < RPT_MAX_I else "1c",
             "Qd": qd, "Qc": ch["Qc"].max(), "Ed": d["Ed"].max(), "Ec": ch["Ec"].max()}
        # 충전: 정전류(1.2A 근처) / 정전압 구간
        cc = ch[ch["I"] >= I_CC - CC_TOL]
        cv = ch[ch["I"] < I_CC - CC_TOL]
        cc_end_t = cc["st"].max() if len(cc) else np.nan
        r["cc_time"] = cc_end_t
        r["cc_cap"] = cc["Qc"].max() if len(cc) else np.nan
        r["cv_time"] = ch["st"].max() - cc_end_t if len(cc) else np.nan
        r["cv_frac"] = (r["Qc"] - r["cc_cap"]) / r["Qc"] if r["Qc"] > 0 and len(cc) else np.nan
        r["cv_tau"] = np.nan
        cvp = cv[(cv["I"] > 0.06) & (cv["st"] > cc_end_t)] if len(cc) else cv.iloc[0:0]
        if len(cvp) >= 5:
            slope = np.polyfit(cvp["st"].to_numpy(), np.log(cvp["I"].to_numpy()), 1)[0]
            r["cv_tau"] = -1.0 / slope if slope < 0 else np.nan
        # 방전 중 온도, 휴지
        r["T_rise"] = d["T"].max() - d["T"].iloc[0]
        r["T_max"] = d["T"].max()
        rd = s["rest_after_dchg"]
        if rd is not None and len(rd) >= 2:
            r["relax_dchg"] = rd["V"].iloc[-1] - rd["V"].iloc[0]
            r["v_jump"] = rd["V"].iloc[0] - d["V"].iloc[-1]
            dur = rd["st"].iloc[-1] - rd["st"].iloc[0]
            r["cool_rate"] = (rd["T"].iloc[0] - rd["T"].iloc[-1]) / dur if dur > 0 else np.nan
        rc = s["rest_after_chg"]
        if rc is not None and len(rc) >= 2:
            r["relax_chg"] = rc["V"].iloc[0] - rc["V"].iloc[-1]
        # 펄스 저항: 충전 후 / 방전 후
        for prev, g in s["pulses"]:
            val = g["IR"].dropna()
            if len(val):
                r["ir_after_chg" if prev == "chg" else "ir_after_dchg"] = float(val.iloc[-1])
        r["_dchg_curve"] = (d["V"].to_numpy(), d["Qd"].to_numpy()) if r["type"] == "rpt" else None
        rows.append(r)
    return pd.DataFrame(rows).sort_values("cyc").reset_index(drop=True)


# ---------------------------------------------------------------- 곡선 특징
def q_on_grid(curve) -> np.ndarray:
    """방전 곡선(V 감소, Q 증가)을 공통 전압 격자 위 용량으로 보간한다."""
    v, q = curve
    order = np.argsort(v)
    v, q = v[order], q[order]
    v, idx = np.unique(v, return_index=True)
    q = q[idx]
    out = np.interp(V_GRID, v, q, left=np.nan, right=np.nan)
    return out


def dqdv_peak(curve) -> tuple[float, float]:
    qg = q_on_grid(curve)
    ok = ~np.isnan(qg)
    if ok.sum() < 61:
        return np.nan, np.nan
    v, q = V_GRID[ok], qg[ok]
    dq = -np.gradient(q, v)                      # 방전이라 V가 내려갈수록 Q 증가 → 부호 반전
    dq = savgol_filter(dq, 31, 3)            # 약 60mV 창으로 스무딩
    win = (v >= DQDV_WINDOW[0]) & (v <= DQDV_WINDOW[1])
    if not win.any():
        return np.nan, np.nan
    vw, dw = v[win], dq[win]
    i = int(np.argmax(dw))
    if 0 < i < len(dw) - 1:                      # 격자 간격보다 정밀하게: 꼭짓점 주변 2차식 보간
        y0, y1, y2 = dw[i - 1], dw[i], dw[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        h = vw[1] - vw[0]
        return float(vw[i] + off * h), float(y1 - 0.25 * (y0 - y2) * off)
    return float(vw[i]), float(dw[i])


def delta_q(curve_first, curve_last) -> tuple[float, float, float]:
    # Severson et al. (2019) 방식: 같은 전압에서 '방전한 용량'의 차이 ΔQ(V) = Q_last(V) − Q_first(V)
    # 두 사이클 모두 4.2V 정전압 충전 뒤 같은 0.2C로 방전하므로 시작점이 맞는다
    a, b = q_on_grid(curve_first), q_on_grid(curve_last)
    dq = b - a
    dq = dq[~np.isnan(dq)]
    if len(dq) < 20:
        return np.nan, np.nan, np.nan
    return float(np.var(dq)), float(np.min(dq)), float(np.mean(dq))


# ---------------------------------------------------------------- 진단량 하나의 특징
def features_for_budget(summ: pd.DataFrame, n: int) -> np.ndarray:
    """사이클 1~n 요약만으로 특징 벡터를 만든다. summ에 n 이후 사이클이 있으면 오류."""
    s = summ[summ["cyc"] <= n]
    rpt, c1 = s[s["type"] == "rpt"], s[s["type"] == "1c"]
    f = dict.fromkeys(FEATURE_NAMES, np.nan)

    def last(frame, col):
        v = frame[col].dropna() if col in frame else pd.Series(dtype=float)
        return float(v.iloc[-1]) if len(v) else np.nan

    def mean(frame, col):
        v = frame[col].dropna() if col in frame else pd.Series(dtype=float)
        return float(v.mean()) if len(v) else np.nan

    f["cap_rpt_last"] = last(rpt, "Qd")
    f["cap_1c_last"] = last(c1, "Qd")
    f["cap_chg_last"] = last(s, "Qc")
    f["rate_ratio"] = f["cap_1c_last"] / f["cap_rpt_last"] if f["cap_rpt_last"] > 0 else np.nan
    for k, col in [("cc_time_last", "cc_time"), ("cc_cap_last", "cc_cap"), ("cv_time_last", "cv_time"),
                   ("cv_frac_last", "cv_frac"), ("cv_tau_last", "cv_tau"),
                   ("e_dchg_1c_last", "Ed"), ("e_chg_last", "Ec")]:
        f[k] = last(c1, col)
    if len(c1):
        f["ce_mean"] = float((c1["Qd"] / c1["Qc"]).mean())
        f["ee_mean"] = float((c1["Ed"] / c1["Ec"]).mean())
        lc = c1.iloc[-1]
        f["v_dchg_mean_1c"] = lc["Ed"] / lc["Qd"]
        f["v_hyst_1c"] = lc["Ec"] / lc["Qc"] - lc["Ed"] / lc["Qd"]
    curves = [x for x in rpt["_dchg_curve"] if x is not None] if len(rpt) else []
    if curves:
        f["dqdv_peak_v"], f["dqdv_peak_h"] = dqdv_peak(curves[-1])
    if len(curves) >= 2:
        f["dq_var"], f["dq_min"], f["dq_mean"] = delta_q(curves[0], curves[-1])
    f["ir_after_chg"] = last(rpt, "ir_after_chg")
    f["ir_after_dchg"] = last(rpt, "ir_after_dchg")
    f["relax_dchg"] = mean(c1, "relax_dchg")
    f["relax_chg"] = mean(c1, "relax_chg")
    f["v_jump_dchg"] = mean(c1, "v_jump")
    f["temp_rise_1c"] = mean(c1, "T_rise")
    f["temp_max_1c"] = mean(c1, "T_max")
    f["temp_rise_rpt"] = last(rpt, "T_rise")
    f["cool_rate_1c"] = mean(c1, "cool_rate")
    if len(c1) >= 2:
        f["slope1_cap_1c"] = float(np.polyfit(c1["cyc"], c1["Qd"], 1)[0])
    if len(c1) >= 3:
        f["slope2_cap_1c"] = float(np.polyfit(c1["cyc"], c1["Qd"], 2)[0])
    if len(rpt) >= 2:
        f["drop_cap_rpt"] = float(rpt["Qd"].iloc[0] - rpt["Qd"].iloc[-1])
    if not (f["temp_rise_1c"] >= TEMP_RISE_MIN):
        for k in ("temp_rise_1c", "temp_max_1c", "temp_rise_rpt", "cool_rate_1c"):
            f[k] = np.nan
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)


def cell_features(df: pd.DataFrame, invalid: set[int]) -> np.ndarray:
    """셀 하나의 (7, D). 진단량마다 1~n으로 자른 원본에서 따로 요약해 미래 누수를 막는다."""
    out = np.full((len(BUDGETS), len(FEATURE_NAMES)), np.nan)
    for b, n in enumerate(BUDGETS):
        part = df[df["cyc"] <= n]
        summ = summarize_cycles(part, invalid)
        out[b] = features_for_budget(summ, n)
    return out


# ---------------------------------------------------------------- 누수 점검
def leak_check(df: pd.DataFrame, invalid: set[int]) -> bool:
    """n 이후 데이터를 망가뜨려도 진단량 n의 특징이 같은지 확인한다."""
    base = cell_features(df, invalid)
    rng = np.random.default_rng(0)
    ok = True
    for b, n in enumerate(BUDGETS):
        bad = df.copy()
        fut = bad["cyc"] > n
        for col in ("V", "I", "Qc", "Qd", "Ec", "Ed", "T", "IR"):
            bad.loc[fut, col] = rng.normal(size=fut.sum())
        got = cell_features(bad, invalid)[b]
        same = np.allclose(base[b], got, equal_nan=True)
        ok &= same
        if not same:
            print(f"  누수 의심: n={n}", np.array(FEATURE_NAMES)[~np.isclose(base[b], got, equal_nan=True)])
    return ok


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DIR / "Second_life_phase"))
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 파일만 처리(확인용)")
    ap.add_argument("--check-leak", action="store_true", help="첫 파일로 누수 점검")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    files = sorted(raw.rglob("*.csv"), key=lambda p: p.stem)
    if args.limit:
        files = files[: args.limit]
    invalid_by_cell: dict[int, set[int]] = {}
    cap_raw = out / "cap_raw.parquet"
    if cap_raw.exists():
        cr = pd.read_parquet(cap_raw)
        for cid, g in cr[cr["cycle_type"] == "invalid"].groupby("cell_id"):
            invalid_by_cell[int(cid)] = set(int(x) for x in g["cycle"])

    feats, ids = {}, []
    for k, p in enumerate(files, 1):
        df = load_cell(p)
        if df.empty:
            print(f"[{k}/{len(files)}] {p.name}: 사이클 1~{MAX_CYCLE} 데이터 없음, 건너뜀")
            continue
        bid = int(df["bid"].iloc[0])
        inv = invalid_by_cell.get(bid, set())
        if args.check_leak and k == 1:
            print("누수 점검:", "통과" if leak_check(df, inv) else "실패")
        feats[bid] = cell_features(df, inv)
        print(f"[{k}/{len(files)}] {p.name} → Battery_ID {bid}, 결측 {np.isnan(feats[bid]).sum()}칸")

    cfull = out / "C_full.npz"
    if cfull.exists():
        order = [int(c) for c in np.load(cfull)["cell_ids"] if int(c) in feats]
    else:
        order = sorted(feats)
    X = np.stack([feats[c] for c in order])                 # (셀, 7, D)
    fail = np.isnan(X)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features.npz", X=X, fail=fail, cell_ids=np.array(order),
             budgets=np.array(BUDGETS), feature_names=np.array(FEATURE_NAMES))
    meta = [{"name": n, "group": g, "group_name": gn, "unit": u, "desc": d} for n, g, gn, u, d in FEATURES]
    (out / "feature_names.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    long = pd.DataFrame(X.reshape(-1, X.shape[2]), columns=FEATURE_NAMES)
    long.insert(0, "budget", np.tile(BUDGETS, len(order)))
    long.insert(0, "cell_id", np.repeat(order, len(BUDGETS)))
    long.to_csv(out / "features_long.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장: features.npz {X.shape}, features_long.csv {long.shape}")
    rate = fail.mean(axis=0)                                 # (7, D) 진단량별 계산 실패율
    bad = [(FEATURE_NAMES[j], BUDGETS[b], rate[b, j]) for b in range(len(BUDGETS))
           for j in range(len(FEATURE_NAMES)) if rate[b, j] > 0]
    print("계산 실패가 있는 (특징, 진단량, 실패율):", bad if bad else "없음")


if __name__ == "__main__":
    main()
