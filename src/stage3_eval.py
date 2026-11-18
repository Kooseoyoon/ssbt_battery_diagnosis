"""Stage 3 평가 — M1·M2·M3 비교 (명세 7절). stage3_gp.py --run 다음에 돌린다.

  M1 리지      Stage 2 리지(log T_eol, RidgeCV GCV, alpha 10^-4~10^3 29개) + 학습 fold 평균 모양 μ(u). 점 예측
  M2 GP-EOL    GP(log T_eol)에서 T_eol만 뽑고 모양은 평균 μ(u). conformal 없음
  M3 GP 전체   posterior_samples_r{r}.npz 그대로(주 모델, conformal 후)
  세 모델 모두 같은 folds.csv, 같은 입력(S7b), 같은 연속 보정·관측 열 규칙(stage3_repr.finalize).
  주 모델은 M3로 고정이고, 이 비교로 바꾸지 않는다.

지표 (진단량마다, 시험 셀 = 86셀 전체 out-of-fold)
  숨은 미래      열 n/5 ~ 각 셀 EOL 열: RMSE·MAE·CRPS(Ah), 시점별 90% PICP, 90% 구간 폭
  공통 horizon   사이클 41(열 8) ~ 각 셀 EOL 열: RMSE·CRPS·PICP (진단량끼리 같은 구간 비교)
  EOL            샘플 중앙값 기준 MAE(사이클), 90% 구간 포괄률, Spearman, Kendall, 쌍별 순서 정확도
  궤적 단위      숨은 미래 전체가 5~95% 띠 안에 든 셀 비율(trajectory-level coverage, 보조)
  셀 평균 → 셀 사이 평균. M1은 점 예측이라 CRPS = MAE, PICP·구간 지표는 비움.
  범위(사이클 1946) 안 미도달 샘플은 1e6으로 두고 분위수를 재며, 비율을 따로 적는다(결정 15).

실행 (저장소 루트):
    python src/stage3_eval.py                 # 반복 0
    python src/stage3_eval.py --reps 5        # M2·M3 지표를 반복 0~4 평균(샘플링 변동 확인)
    python src/stage3_eval.py --tag s2        # 민감도 실행 결과 평가
출력: results/stage3/{tag}/eval/
    metrics.csv            모델 × 진단량 × 지표 (반복 평균, --reps > 1이면 표준편차 열 추가)
    metrics_vs_budget.png  RMSE·CRPS·PICP·EOL Spearman vs 진단량, 세 모델
    calibration.png        M3 시점별 중앙 구간 명목 수준(10~90%) vs 실제 포괄률, 진단량별
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR  # noqa: E402
import stage3_gp as G  # noqa: E402
import stage3_repr as R  # noqa: E402

BIG = 1e6
COMMON_START_COL = 8          # 사이클 41
ALPHAS = np.logspace(-4, 3, 29)
LEVELS = np.arange(0.1, 1.0, 0.1)
COLORS = {"M1_ridge": "C0", "M2_gp_eol": "C1", "M3_gp_full": "C2"}


# ------------------------------------------------------------------ 비교 모델 예측
def predict_m1(D, f):
    """리지 log T_eol + 평균 모양. 반환: 셀별 (1, 열) 궤적."""
    b, n, k, idx = f["b"], f["n"], f["k"], f["idx"]
    tr = np.where(D["fold"] != k)[0]
    Xtr, Xte = G.impute_std(D["X"][tr, b], D["X"][idx, b])
    m = RidgeCV(alphas=ALPHAS).fit(Xtr, np.log(D["t_eol"][tr]))
    t_hat = np.exp(m.predict(Xte))
    out = []
    for ii, i in enumerate(idx):
        t = max(t_hat[ii], n + 5.0)
        q = R.trajectory(f["fp"].mean, D["C"][i, 0], t, D["cycles"])
        out.append(R.finalize(D["C"][i], n, q, D["cycles"])[None, :])
    return out


def predict_m2(D, f, r, S):
    """GP(log T_eol)만 뽑고 모양은 평균. 반환: 셀별 (S, 열)."""
    b, n, idx = f["b"], f["n"], f["idx"]
    out = []
    for ii, i in enumerate(idx):
        rng = G.cell_rng(r, b, D["cell_ids"][i], tag=99)
        Q = np.empty((S, len(D["cycles"])))
        for s in range(S):
            for _ in range(G.RESAMPLE_MAX):
                t = float(np.exp(f["mu"][ii, 0] + f["sd"][ii, 0] * rng.standard_normal()))
                if t > n:
                    break
            else:
                t = n + 5.0
            q = R.trajectory(f["fp"].mean, D["C"][i, 0], t, D["cycles"])
            Q[s] = R.finalize(D["C"][i], n, q, D["cycles"])
        out.append(Q)
    return out


# ------------------------------------------------------------------ 지표
def crps_samples(q, y):
    S = q.shape[0]
    if S == 1:
        return np.abs(q[0] - y)
    e1 = np.abs(q - y).mean(0)
    qs = np.sort(q, 0)
    w = (2 * np.arange(1, S + 1) - S - 1) / (S * S)
    return e1 - (w[:, None] * qs).sum(0)


def window_metrics(q, y):
    m = q.mean(0)
    out = {"rmse": np.sqrt(np.mean((m - y) ** 2)), "mae": np.mean(np.abs(m - y)), "crps": np.mean(crps_samples(q, y))}
    if q.shape[0] > 1:
        lo, hi = np.percentile(q, [5, 95], axis=0)
        inside = (y >= lo) & (y <= hi)
        out.update(picp=np.mean(inside), width=np.mean(hi - lo), traj_cover=float(inside.all()))
    return out


def evaluate(D, n, idx, Qs):
    """한 모델, 한 진단량, 86셀. Qs = 셀별 (S, 열) 배열."""
    cyc = D["cycles"]
    rows_h, rows_c, t_med, t_lo, t_hi, cens = [], [], [], [], [], []
    for i, Q in zip(idx, Qs):
        j_end = int(np.searchsorted(cyc, D["t_eol"][i]))
        a = n // 5
        rows_h.append(window_metrics(Q[:, a: j_end + 1], D["C"][i, a: j_end + 1]))
        c0 = max(COMMON_START_COL, a)
        rows_c.append(window_metrics(Q[:, c0: j_end + 1], D["C"][i, c0: j_end + 1]))
        T = np.array([R.t_eol_of(q, cyc) for q in Q])
        cens.append(np.isnan(T).mean())
        T = np.where(np.isnan(T), BIG, T)
        t_med.append(np.median(T))
        t_lo.append(np.percentile(T, 5))
        t_hi.append(np.percentile(T, 95))
    H, Cm = pd.DataFrame(rows_h), pd.DataFrame(rows_c)
    tt = D["t_eol"][idx]
    tm = np.minimum(np.array(t_med), cyc[-1] + 5)
    # 쌍별 순서 정확도: 실제 수명이 다른 모든 쌍 중 예측 순서가 맞는 비율(예측 동점은 반으로)
    ii, jj = np.triu_indices(len(tt), 1)
    keep = tt[ii] != tt[jj]
    agree = np.sign(tm[ii] - tm[jj])[keep] * np.sign(tt[ii] - tt[jj])[keep]
    out = {
        "hidden_rmse_Ah": H.rmse.mean(), "hidden_mae_Ah": H.mae.mean(), "hidden_crps_Ah": H.crps.mean(),
        "common_rmse_Ah": Cm.rmse.mean(), "common_crps_Ah": Cm.crps.mean(),
        "eol_mae_cyc": np.mean(np.abs(tm - tt)), "eol_spearman": spearmanr(tm, tt)[0],
        "eol_kendall": kendalltau(tm, tt)[0], "pair_order_acc": np.mean(agree > 0) + 0.5 * np.mean(agree == 0),
        "eol_censored_frac": np.mean(cens),
    }
    if "picp" in H:
        out.update(hidden_picp90=H.picp.mean(), hidden_width90_Ah=H.width.mean(), traj_cover90=H.traj_cover.mean(),
                   common_picp90=Cm.picp.mean(),
                   eol_cover90=np.mean((tt >= np.array(t_lo)) & (tt <= np.array(t_hi))))
    return out


def calibration(D, n, idx, Qs):
    """시점별 중앙 구간 명목 수준별 실제 포괄률(숨은 미래 모든 셀·열 풀링)."""
    hits = {lv: [] for lv in LEVELS}
    for i, Q in zip(idx, Qs):
        j_end = int(np.searchsorted(D["cycles"], D["t_eol"][i]))
        a = n // 5
        q, y = Q[:, a: j_end + 1], D["C"][i, a: j_end + 1]
        for lv in LEVELS:
            lo, hi = np.percentile(q, [50 - 50 * lv, 50 + 50 * lv], axis=0)
            hits[lv].append((y >= lo) & (y <= hi))
    return {lv: float(np.mean(np.concatenate(h))) for lv, h in hits.items()}


# ------------------------------------------------------------------ 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    base = RESULTS_DIR / "stage3" / a.tag
    out = base / "eval"
    out.mkdir(parents=True, exist_ok=True)
    D = G.load("sensitivity" if a.tag == "s2" else "main")
    fits = pickle.load(open(base / "fits.pkl", "rb"))
    rows, calib = [], []
    for r in range(a.reps):
        d = np.load(base / f"posterior_samples_r{r}.npz")
        Qall = d["Q"]
        S = Qall.shape[2]
        print(f"반복 {r}: 샘플 {S}개")
        for n in D["budgets"]:
            parts = [f for f in fits if f["n"] == n]
            b = parts[0]["b"]
            idx = np.concatenate([f["idx"] for f in parts])
            m3 = [Qall[i, b].astype(float) for i in idx]
            m2 = [q for f in parts for q in predict_m2(D, f, r, S)]
            models = {"M2_gp_eol": m2, "M3_gp_full": m3}
            if r == 0:
                models = {"M1_ridge": [q for f in parts for q in predict_m1(D, f)], **models}
                cal = calibration(D, n, idx, m3)
                calib += [{"budget": n, "nominal": lv, "empirical": v} for lv, v in cal.items()]
            for name, Qs in models.items():
                rows.append({"rep": r, "model": name, "budget": n, **evaluate(D, n, idx, Qs)})
            print(f"  n = {n} 완료")

    M = pd.DataFrame(rows)
    agg = M.groupby(["model", "budget"]).mean(numeric_only=True).drop(columns="rep")
    if a.reps > 1:
        sd = M.groupby(["model", "budget"]).std(numeric_only=True).drop(columns="rep").add_suffix("_sd")
        agg = agg.join(sd)
    agg.reset_index().to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calib).to_csv(out / "calibration.csv", index=False, encoding="utf-8-sig")
    show = ["hidden_rmse_Ah", "hidden_crps_Ah", "hidden_picp90", "hidden_width90_Ah", "common_rmse_Ah",
            "eol_mae_cyc", "eol_cover90", "eol_spearman", "pair_order_acc", "traj_cover90"]
    with pd.option_context("display.width", 250):
        print(agg[[c for c in show if c in agg]].round(4).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        A = agg.reset_index()
        panels = [("hidden_rmse_Ah", "hidden-future RMSE (Ah)"), ("hidden_crps_Ah", "hidden-future CRPS (Ah)"),
                  ("hidden_picp90", "pointwise 90% PICP"), ("eol_spearman", "EOL Spearman")]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for ax, (col, lab) in zip(axes, panels):
            for name, g in A.groupby("model"):
                if g[col].notna().any():          # 색은 모델에 고정(M1은 PICP가 없어 빠져도 색이 밀리지 않게)
                    ax.plot(g.budget, g[col], marker="o", label=name, color=COLORS[name])
            if col == "hidden_picp90":
                ax.axhline(0.9, color="gray", ls="--", lw=0.8)
            ax.set_xlabel("diagnostic budget n (cycles)")
            ax.set_title(lab, fontsize=10)
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "metrics_vs_budget.png", dpi=140)
        Cb = pd.DataFrame(calib)
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for n, g in Cb.groupby("budget"):
            ax.plot(g.nominal, g.empirical, marker=".", label=f"n = {n}")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
        ax.set_xlabel("nominal central-interval level")
        ax.set_ylabel("empirical pointwise coverage")
        ax.set_title("M3 calibration (hidden future)", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "calibration.png", dpi=140)
    except Exception as e:  # noqa: BLE001
        print("그림 저장 실패:", e)
    print(f"저장: {out}/ metrics.csv, calibration.csv, metrics_vs_budget.png, calibration.png")


if __name__ == "__main__":
    main()
