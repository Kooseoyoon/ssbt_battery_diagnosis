"""Stage 2 (재시작) — 특징끼리 Spearman 상관을 진단량 7단계마다 전부 계산한다.

T_eol(정답)은 읽지 않는다. 특징 값끼리만 비교하므로 86셀 전체로 한 번 계산해도 선택 편향이 없다.
여기서는 r 값을 구해 저장하고 요약만 보여준다. 무엇을 뺄지는 결과를 보고 정한다.

실행 (저장소 루트):
    python src/stage2_corr.py                 # 기준선 0.95
    python src/stage2_corr.py --thr 0.9       # 요약에 쓰는 기준만 바뀐다(저장되는 r 값은 같다)
입력:  data/processed/features_v2.npz  (stage2_features_v2.py 출력)
출력:  results/stage2_corr/
       corr_n10.csv … corr_n40.csv   진단량별 특징 × 특징 Spearman 행렬 전체
       pairs_all.csv                 모든 특징 쌍 × 7단계 r, 쌍마다 같이 관측된 셀 수 (넓은 표)
       pairs_flagged.csv             한 단계라도 |r| ≥ 기준인 쌍만, 걸린 단계 수 순
       base_max_r.csv                기준 양 × 기준 양: 집계를 가리지 않은 최대 |r| (7단계 중 최댓값)
       feature_status.csv            특징별 결측률·상수 여부(상관 계산 불가 사유)
       heatmap_n10.png, heatmap_n40.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

MIN_OBS = 30   # 두 특징이 같이 관측된 셀이 이보다 적으면 r을 계산하지 않는다(86셀 중)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--inp", default=str(PROCESSED_DIR / "features_v2.npz"))
    args = ap.parse_args()
    d = np.load(args.inp)
    X, budgets = d["X"], [int(b) for b in d["budgets"]]
    names, bases = list(d["feature_names"]), list(d["base_names"])
    n_cell = X.shape[0]
    out = RESULTS_DIR / "stage2_corr"
    out.mkdir(parents=True, exist_ok=True)
    min_obs = min(MIN_OBS, n_cell)
    print(f"features_v2.npz {X.shape}: 셀 {n_cell}, 진단량 {budgets}, 특징 {len(names)}개")
    if n_cell < 20:
        print(f"  ※ 셀이 {n_cell}개뿐이라 r 값이 의미 없다(코드 동작 확인용). 86셀로 다시 돌릴 것")

    # 특징 상태: 결측률, 상수(서로 다른 값 1개) → 그 단계에서는 상관 계산 불가
    st = []
    for b, n in enumerate(budgets):
        for j, f in enumerate(names):
            v = X[:, b, j]
            ok = v[~np.isnan(v)]
            st.append({"feature": f, "base": bases[j], "budget": n, "nan_rate": 1 - len(ok) / n_cell,
                       "n_unique": len(np.unique(np.round(ok, 12))),
                       "usable": len(ok) >= min_obs and len(np.unique(np.round(ok, 12))) > 1})
    st = pd.DataFrame(st)
    st.to_csv(out / "feature_status.csv", index=False, encoding="utf-8-sig")

    iu = np.triu_indices(len(names), 1)
    wide = pd.DataFrame({"feature_a": np.array(names)[iu[0]], "feature_b": np.array(names)[iu[1]],
                         "base_a": np.array(bases)[iu[0]], "base_b": np.array(bases)[iu[1]]})
    for b, n in enumerate(budgets):
        df = pd.DataFrame(X[:, b, :], columns=names)
        usable = st[(st["budget"] == n) & st["usable"]]["feature"].tolist()
        c = df[usable].corr(method="spearman", min_periods=min_obs).reindex(index=names, columns=names)
        c.to_csv(out / f"corr_n{n}.csv", encoding="utf-8-sig")
        obs = (~df.isna()).astype(int)
        nobs = obs.T @ obs
        wide[f"r_n{n}"] = c.to_numpy()[iu]
        wide[f"obs_n{n}"] = nobs.to_numpy()[iu]
        print(f"  n={n}: 계산 가능 특징 {len(usable)}/{len(names)}, "
              f"|r| ≥ {args.thr} 쌍 {(np.abs(wide[f'r_n{n}']) >= args.thr).sum()}개")

    rcols = [f"r_n{n}" for n in budgets]
    A = wide[rcols].abs()
    wide["n_budgets_flag"] = (A >= args.thr).sum(axis=1)
    wide["max_abs_r"] = A.max(axis=1)
    wide["min_abs_r"] = A.min(axis=1)
    wide["same_base"] = wide["base_a"] == wide["base_b"]
    wide.to_csv(out / "pairs_all.csv", index=False, encoding="utf-8-sig")
    flag = wide[wide["n_budgets_flag"] > 0].sort_values(["n_budgets_flag", "max_abs_r"], ascending=False)
    flag.to_csv(out / "pairs_flagged.csv", index=False, encoding="utf-8-sig")

    # 기준 양 수준 요약: 두 기준 양의 어떤 집계끼리든, 7단계 중 가장 큰 |r|
    ub = list(dict.fromkeys(bases))
    bm = (wide.groupby(["base_a", "base_b"])["max_abs_r"].max().reset_index())
    M = pd.DataFrame(np.nan, index=ub, columns=ub)
    for _, r in bm.iterrows():
        M.loc[r.base_a, r.base_b] = M.loc[r.base_b, r.base_a] = r.max_abs_r
    M.to_csv(out / "base_max_r.csv", encoding="utf-8-sig")

    # 화면 요약
    print(f"\n[한 단계라도 |r| ≥ {args.thr}인 쌍: {len(flag)}개 / 전체 {len(wide)}쌍]")
    print(f"  같은 기준 양의 다른 집계끼리: {flag['same_base'].sum()}쌍, 다른 기준 양끼리: {(~flag['same_base']).sum()}쌍")
    show = flag[~flag["same_base"]].head(40)
    if len(show):
        print("\n[다른 기준 양끼리, 걸린 단계 수 순 상위 40] r 부호 그대로")
        print(show[["feature_a", "feature_b", "n_budgets_flag"] + rcols].round(3).to_string(index=False))
    same = flag[flag["same_base"]].head(20)
    if len(same):
        print("\n[같은 기준 양의 다른 집계끼리 상위 20]")
        print(same[["feature_a", "feature_b", "n_budgets_flag"] + rcols].round(3).to_string(index=False))
    bad = st[~st["usable"]].groupby("feature")["budget"].apply(list)
    if len(bad):
        print(f"\n[상관 계산 불가(값이 있는 셀이 {min_obs}개 미만이거나 모든 셀 값이 같음): 특징 {len(bad)}개]")
        for f, bl in bad.items():
            print(f"  {f:28s} 진단량 {bl}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n in (budgets[0], budgets[-1]):
            c = pd.read_csv(out / f"corr_n{n}.csv", index_col=0)
            fig, ax = plt.subplots(figsize=(16, 14))
            im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=5)
            ax.set_yticks(range(len(c)), c.index, fontsize=5)
            # 기준 양 경계선
            edges = [i for i in range(1, len(bases)) if bases[i] != bases[i - 1]]
            for e in edges:
                ax.axhline(e - 0.5, color="k", lw=0.3)
                ax.axvline(e - 0.5, color="k", lw=0.3)
            fig.colorbar(im, shrink=0.6, label="Spearman r")
            ax.set_title(f"feature Spearman correlation, n = {n} (gray = not computable)")
            fig.tight_layout()
            fig.savefig(out / f"heatmap_n{n}.png", dpi=150)
            plt.close(fig)
    except Exception as e:  # 그림은 부가물
        print("그림 저장 실패:", e)
    print(f"\n저장: {out}/")


if __name__ == "__main__":
    main()
