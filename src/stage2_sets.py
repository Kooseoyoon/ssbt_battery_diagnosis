"""Stage 2 — 미리 정한 특징 묶음 비교 (11/6 설계).

안정성 점검(stage2_stability.py, 분할 시드 10 × 5 fold)에서 곡선 정보가 두 묶음 중 하나로 들어가는 구조를 확인했다.
  묶음 A(미분 곡선): dqdv_p1_h__last + dqdv_p1_v__last
  묶음 B(구간 용량): q_rpt__last + q_rpt_low__last + dqdv_p1_area__last
그래서 104만 조합 탐색 대신, 결과를 보기 전에 고정한 소수의 묶음만 같은 반복 교차검증으로 비교한다.
묶음이 고정이라 fold 안에서 고르는 것은 리지 강도뿐이다(바깥 학습 셀 안 GCV).

비교 묶음 (아래 SETS, 실행 전에 고정. 바꾸면 결정 기록에 남길 것)
  S0 현재 용량(기준선), S1 ce__mean, S2 ce + A, S3 ce + B, S4 S2 + 분극 쌍, S5 S3 + 분극 쌍,
  S6a/S6b = S4/S5 + ce__std, S7a/S7b = S4/S5 + cool_rate__slope1,
  R = fold 안에서 1차 거르기로 고른 상위 20개(기존 절차, 정직한 기준값)
  ※ 설계에서 'S4·S5 중 나은 쪽 + …'이던 S6·S7은, 결과를 보고 고르는 단계를 없애려고 a·b 둘 다 계산한다

분할: 분할 시드 1~10, 시드 s의 바깥 5-fold는 stage2_stability.py와 같다(strat_folds(y, 5, 1000 + s)) → 묶음끼리 짝지어 비교
지표 (분할 시드마다, 진단량마다, 5 fold 예측을 모은 86셀로 계산)
  spearman : 예측 수명과 실제 T_eol의 Spearman (주 지표, 7단계 평균을 묶음 점수로)
  mae      : 평균 절대 오차(사이클)
  bottom6  : 실제 수명 하위 6셀 중 예측 하위 6셀과 겹치는 수(0~6). 8직렬 미사용 6셀을 맞게 고르는가(결정 38)
             실제 하위 6번째 값과 동률인 셀은 모두 '실제 하위'로 센다
선택 규칙 (실행 전에 고정, 코드가 자동 적용)
  1. 7단계 모두에서 시드 평균 Spearman이 S0보다 높은 묶음만 후보
  2. 후보 중 7단계 평균 점수가 최고인 묶음의 점수에서 그 묶음의 시드 간 표준편차를 뺀 값 이상인 묶음 중 특징 수가 가장 적은 것
  3. 동점이면 bottom6 평균이 높은 쪽, 그래도 같으면 점수가 높은 쪽

실행 (저장소 루트, CPU로 충분):
    python src/stage2_sets.py                 # 묶음 비교
    python src/stage2_sets.py --robust        # S7b 강건성 확인(11/6) → results/stage2_robust/
입력:  data/processed/features_v2.npz, T_eol.npy, C_full.npz (stage2_select.load와 같음)
출력:  results/stage2_sets/
       scores.csv     분할 시드 × 묶음 × 진단량별 spearman, mae, bottom6
       summary.csv    묶음별 7단계 평균 점수(시드 평균 ± 표준편차), 진단량별 평균, bottom6, 최고 묶음 대비 짝지은 차이
       decision.json  선택 규칙 적용 결과와 과정
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR  # noqa: E402
import stage2_select as S  # noqa: E402

CE = ["ce__mean"]
A = ["dqdv_p1_h__last", "dqdv_p1_v__last"]
B = ["q_rpt__last", "q_rpt_low__last", "dqdv_p1_area__last"]
POL = ["rate_ratio__last", "relax_dchg__last"]
SETS = {
    "S0_current_cap": ["q_rpt__last"],
    "S1_ce": CE,
    "S2_ce+A": CE + A,
    "S3_ce+B": CE + B,
    "S4_ce+A+pol": CE + A + POL,
    "S5_ce+B+pol": CE + B + POL,
    "S6a_S4+ce_std": CE + A + POL + ["ce__std"],
    "S6b_S5+ce_std": CE + B + POL + ["ce__std"],
    "S7a_S4+heat": CE + A + POL + ["cool_rate__slope1"],
    "S7b_S5+heat": CE + B + POL + ["cool_rate__slope1"],
    "R_screen_top20": None,          # fold 안에서 1차 거르기
}
BASE = "S0_current_cap"

# 강건성 확인(11/6): S7b의 분극·열 대표 지표를 같은 유형의 다른 지표로 바꿔도 점수가 버티는지.
# 고르는 실험이 아니다. 결과와 무관하게 최종 묶음은 S7b이고, 이 결과는 주장 범위(유형 수준인가, 특정 지표인가)만 정한다.
POL_ALT = ["v_chg_start__mean", "r_dc_1c__mean"]
ROBUST_SETS = {
    "S7b_main": CE + B + POL + ["cool_rate__slope1"],
    "V1_pol_alt": CE + B + POL_ALT + ["cool_rate__slope1"],
    "V2_heat_alt": CE + B + POL + ["t_rise_rpt__last"],
    "V3_both_alt": CE + B + POL_ALT + ["t_rise_rpt__last"],
    "V4_no_heat(S5)": CE + B + POL,
    "S2_ce+A(ref)": CE + A,
    "S0_current_cap": ["q_rpt__last"],
}
ROBUST_REF = "S7b_main"
# 통과 기준(실행 전 고정): S7b와의 짝지은 차이가 Spearman ±(S7b 시드 간 표준편차) 안, bottom6 −0.5 이상
ROBUST_BOTTOM_TOL = 0.5
K_SCREEN = 20
N_BOTTOM = 6


def bottom_hits(pred, t, n=N_BOTTOM):
    thr = np.sort(t)[n - 1]
    true_low = set(np.where(t <= thr)[0])
    pred_low = set(np.argsort(pred, kind="stable")[:n])
    return len(true_low & pred_low)


def run(seeds, sets=SETS):
    X, ids, budgets, names, y, t, _ = S.load()
    miss = [f for v in sets.values() if v for f in v if f not in names]
    if miss:
        raise SystemExit(f"후보에 없는 특징: {sorted(set(miss))}")
    cols = {k: ([names.index(f) for f in v] if v else None) for k, v in sets.items()}
    B_ = len(budgets)
    rows = []
    for seed in seeds:
        fold = S.strat_folds(y, 5, 1000 + seed)
        pred = {k: np.empty((len(y), B_)) for k in sets}
        for k in range(5):
            tr, te = fold != k, fold == k
            screened = S.screen(X[tr], y[tr], names, K_SCREEN)[0] if any(c is None for c in cols.values()) else None
            for name, c in cols.items():
                cc = screened if c is None else c
                for b in range(B_):
                    pred[name][te, b] = S.outer_fit_predict(X[tr][:, b, cc], y[tr], X[te][:, b, cc])
        for name, p in pred.items():
            for b, n in enumerate(budgets):
                th = np.exp(p[:, b])
                rows.append({"seed": seed, "set": name, "budget": n,
                             "n_features": K_SCREEN if cols[name] is None else len(cols[name]),
                             "spearman": spearmanr(th, t)[0], "mae": float(np.mean(np.abs(th - t))),
                             "bottom6": bottom_hits(th, t)})
        print(f"분할 시드 {seed} 끝")
    return pd.DataFrame(rows), budgets


def summarize(sc, budgets):
    per_seed = sc.groupby(["set", "seed"]).agg(score=("spearman", "mean"), bottom6=("bottom6", "mean"),
                                               mae=("mae", "mean"), n=("n_features", "first")).reset_index()
    summ = per_seed.groupby("set").agg(score_mean=("score", "mean"), score_sd=("score", "std"),
                                       bottom6_mean=("bottom6", "mean"), mae_mean=("mae", "mean"),
                                       n_features=("n", "first"))
    byb = sc.pivot_table(index="set", columns="budget", values="spearman", aggfunc="mean")
    byb.columns = [f"sp_n{c}" for c in byb.columns]
    summ = summ.join(byb).loc[list(SETS)]

    # 선택 규칙
    base = byb.loc[BASE].to_numpy()
    eligible = [s for s in SETS if s != BASE and np.all(byb.loc[s].to_numpy() > base)]
    log = {"eligible(7단계 모두 S0보다 높음)": eligible}
    if not eligible:
        log["결과"] = "후보 없음: 어느 묶음도 7단계 모두에서 현재 용량을 넘지 못함"
        return summ, log, None
    best = max(eligible, key=lambda s: summ.loc[s, "score_mean"])
    sd = summ.loc[best, "score_sd"]
    thr = summ.loc[best, "score_mean"] - (0.0 if np.isnan(sd) else sd)   # 시드 1개면 표준편차 없음
    within = [s for s in eligible if summ.loc[s, "score_mean"] >= thr]
    pick = sorted(within, key=lambda s: (summ.loc[s, "n_features"], -summ.loc[s, "bottom6_mean"],
                                         -summ.loc[s, "score_mean"]))[0]
    # 최고 묶음 대비 짝지은 차이(같은 분할 시드끼리)
    ps = per_seed.pivot(index="seed", columns="set", values="score")
    d = ps.sub(ps[best], axis=0)
    summ["diff_vs_best_mean"] = d.mean()
    summ["diff_vs_best_sd"] = d.std()
    log.update({"best": best, "best_score": round(float(summ.loc[best, "score_mean"]), 4),
                "best_seed_sd": round(float(summ.loc[best, "score_sd"]), 4), "threshold": round(float(thr), 4),
                "within_threshold": within, "selected": pick,
                "selected_features": SETS[pick] if SETS[pick] else f"86셀 전체 1차 거르기 상위 {K_SCREEN}개"})
    return summ, log, pick


def robust(seeds):
    out = RESULTS_DIR / "stage2_robust"
    out.mkdir(parents=True, exist_ok=True)
    sc, budgets = run(seeds, ROBUST_SETS)
    sc.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    ps = sc.groupby(["set", "seed"]).agg(score=("spearman", "mean"), b6=("bottom6", "mean"),
                                         n=("n_features", "first")).reset_index()
    sp = ps.pivot(index="seed", columns="set", values="score")
    b6 = ps.pivot(index="seed", columns="set", values="b6")
    ref_sd = sp[ROBUST_REF].std()
    ref_sd = 0.0 if np.isnan(ref_sd) else ref_sd
    rows = []
    for s_ in ROBUST_SETS:
        d, e = sp[s_] - sp[ROBUST_REF], b6[s_] - b6[ROBUST_REF]
        rows.append({"set": s_, "features": ";".join(ROBUST_SETS[s_]), "score_mean": sp[s_].mean(),
                     "score_sd": sp[s_].std(), "bottom6_mean": b6[s_].mean(),
                     "d_score_vs_S7b": d.mean(), "d_score_sd": d.std(), "d_bottom6_vs_S7b": e.mean(),
                     "pass": (abs(d.mean()) <= ref_sd) and (e.mean() >= -ROBUST_BOTTOM_TOL)})
    res = pd.DataFrame(rows).set_index("set")
    byb = sc.pivot_table(index="set", columns="budget", values="spearman", aggfunc="mean")
    byb.columns = [f"sp_n{c}" for c in byb.columns]
    res = res.join(byb)
    res.to_csv(out / "summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"\n[강건성 확인, 분할 시드 {len(list(seeds))}개] 기준: S7b와 Spearman 차이 |d| ≤ {ref_sd:.3f}(S7b 시드 간 표준편차), "
          f"bottom6 차이 ≥ −{ROBUST_BOTTOM_TOL}")
    print(res.drop(columns="features").round(3).to_string())
    alts = [s_ for s_ in ["V1_pol_alt", "V2_heat_alt", "V3_both_alt"]]
    ok = res.loc[alts, "pass"].all()
    print("\n판정:", "대표 지표를 바꿔도 버팀 → '유형 수준의 정보가 중요하다'고 주장 가능" if ok else
          "대표 지표에 따라 결과가 달라짐 → 최종 묶음은 S7b 그대로, 주장은 '이 지표 조합'으로 한정")
    print(f"저장: {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--robust", action="store_true", help="S7b 대표 지표 바꿔 끼우기 강건성 확인")
    args = ap.parse_args()
    if args.robust:
        return robust(range(1, args.seeds + 1))
    out = RESULTS_DIR / "stage2_sets"
    out.mkdir(parents=True, exist_ok=True)
    sc, budgets = run(range(1, args.seeds + 1))
    sc.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    summ, log, pick = summarize(sc, budgets)
    summ.to_csv(out / "summary.csv", encoding="utf-8-sig")
    (out / "decision.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"\n[묶음별 결과, 분할 시드 {args.seeds}개] score = 7단계 평균 Spearman, bottom6 = 하위 6셀 적중 수(0~6)")
    show = summ[["n_features", "score_mean", "score_sd", "bottom6_mean", "mae_mean"]
                + [c for c in summ.columns if c.startswith("sp_n")]
                + [c for c in ["diff_vs_best_mean", "diff_vs_best_sd"] if c in summ.columns]]
    print(show.round(3).to_string())
    print("\n[선택 규칙 적용]")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print(f"\n저장: {out}/")


if __name__ == "__main__":
    main()
