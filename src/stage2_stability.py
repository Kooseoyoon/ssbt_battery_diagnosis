"""Stage 2 — 특징 선택 안정성 점검: 분할 시드를 바꿔 stage2_select.py의 nested 절차를 반복한다.

11/4 결과(시드 0)는 바깥 5 fold마다 고른 조합이 크게 달랐다. 그래서 '86셀 전체 상위 50 조합에 모두 든 5개'
(잠정 최종)가 분할 운인지 확인한다.

시드 s마다:
  - 바깥 5-fold를 시드 s로 새로 나눈다(T_eol 5분위 층화). 안쪽 5-fold도 시드에 따라 바뀐다
  - 바깥 fold마다 학습 셀만으로 stage2_select.py와 똑같이 ① 1차 거르기(상위 K) ② 2^K − 1 조합 전부 채점
  - 그 fold의 선택을 두 규칙으로 기록한다
      top1 : 안쪽 점수 1등 조합
      core : 안쪽 점수 상위 50 조합 모두에 든 특징 (잠정 5개를 고른 규칙과 같음)
             상위 50개에 공통 특징이 하나도 없으면 core는 빈 칸으로 기록하고, 바깥 채점은 top1으로 대신한다
             (K가 작으면 조합 수가 적어 자주 비므로 core는 K = 20일 때만 의미가 있다)
  - 두 규칙의 선택을 바깥 시험 셀로 채점(정직한 성능)
결과 = 시드 × 5 fold번 고른 것 중 특징마다 몇 번 뽑혔나 + 시드별 바깥 Spearman 분포.

folds.csv(Stage 3과 공유하는 분할)는 건드리지 않는다. 이 스크립트 안에서만 분할을 새로 만든다.

실행 (저장소 루트):
    python src/stage2_stability.py --seeds 10            # 시드 1~10 (0은 11/4에 이미 돌림)
    python src/stage2_stability.py --seeds 10 --k 16     # 빠르게(조합 6.5만 개)
    이미 끝난 시드는 결과 파일에서 읽고 건너뛴다(중간에 끊겨도 이어서 돌릴 수 있다)
출력:  results/stage2_stability/
       selections.csv   시드 × fold × 규칙별 고른 특징, 1차 후보, 안쪽 점수
       scores.csv       시드 × 진단량별 바깥 Spearman: top1 / core / 1차 후보 전부 / 현재 용량
       frequency.csv    특징별 뽑힌 비율(top1·core·1차 후보), 잠정 5개 표시
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR  # noqa: E402
import stage2_select as S  # noqa: E402

PROVISIONAL = ["ce__mean", "v_chg_start__mean", "r_dc_1c__mean", "dqdv_p1_h__last", "cool_rate__slope1"]
TOP_N_CORE = 50


def core_of(tops, n=TOP_N_CORE):
    """상위 n 조합 모두에 든 특징."""
    sets = [set(fs) for fs, _ in tops[:n]]
    return sorted(set.intersection(*sets)) if sets else []


def run_seed(seed, X, y, t, names, args, quiet):
    fold = S.strat_folds(y, 5, 1000 + seed)          # 바깥 분할. 시드 0의 folds.csv와 겹치지 않게 1000 더함
    B = X.shape[1]
    jb = names.index(S.BASELINE)
    pred = {k: np.empty((len(y), B)) for k in ("top1", "core", "top_k_all", "current_cap")}
    sel = []
    for k in range(5):
        tr, te = fold != k, fold == k
        chosen, best, tops, _ = S.run_selection(X[tr], y[tr], names, args.k, args.device, args.batch,
                                                f"시드 {seed} fold {k}", inner_seed=2000 + 10 * seed + k, log=quiet)
        core = core_of(tops)
        cols = {"top1": best, "core": [names.index(f) for f in core] or best,
                "top_k_all": chosen, "current_cap": [jb]}
        for b in range(B):
            for key, c in cols.items():
                pred[key][te, b] = S.outer_fit_predict(X[tr][:, b, c], y[tr], X[te][:, b, c])
        sel.append({"seed": seed, "fold": k, "rule": "top1", "features": ";".join(names[j] for j in best),
                    "n": len(best), "inner_score": tops[0][1], "screened": ";".join(names[j] for j in chosen)})
        sel.append({"seed": seed, "fold": k, "rule": "core", "features": ";".join(core),
                    "n": len(core), "inner_score": np.nan, "screened": ";".join(names[j] for j in chosen)})
        print(f"  시드 {seed} fold {k}: core {core}")
    sc = [{"seed": seed, "method": key, "budget": n, "spearman": spearmanr(np.exp(p[:, b]), t)[0],
           "mae_cycles": float(np.mean(np.abs(np.exp(p[:, b]) - t)))}
          for key, p in pred.items() for b, n in enumerate(S_BUDGETS)]
    return sel, sc


def summarize(sel, sc, names, out):
    total = sel[sel.rule == "top1"].shape[0]
    rows = []
    for f in names:
        def frac(rule):
            ss = sel[sel.rule == rule]["features"].fillna("")
            return np.mean([f in s.split(";") for s in ss])
        scr = np.mean([f in s.split(";") for s in sel[sel.rule == "top1"]["screened"]])
        rows.append({"feature": f, "freq_top1": frac("top1"), "freq_core": frac("core"), "freq_screened": scr,
                     "provisional": f in PROVISIONAL})
    fr = pd.DataFrame(rows).sort_values(["freq_top1", "freq_core"], ascending=False)
    fr = fr[(fr.freq_top1 > 0) | (fr.freq_core > 0) | fr.provisional]
    fr.to_csv(out / "frequency.csv", index=False, encoding="utf-8-sig")
    n_seed = sel.seed.nunique()
    print(f"\n[뽑힌 비율: 시드 {n_seed}개 × 5 fold = {total}번 선택 중]")
    print("  top1 = 1등 조합에 든 비율, core = 상위 50 조합 모두에 든 비율, screened = 1차 후보 20개에 든 비율")
    print(fr.head(25).round(2).to_string(index=False))
    print("\n[잠정 5개]")
    print(fr[fr.provisional].round(2).to_string(index=False))

    piv = sc.groupby(["method", "budget"])["spearman"].agg(["mean", "std", "min", "max"])
    print(f"\n[바깥 Spearman, 시드 {n_seed}개 평균 ± 표준편차]")
    m = sc.pivot_table(index="method", columns="budget", values="spearman", aggfunc="mean")
    s = sc.pivot_table(index="method", columns="budget", values="spearman", aggfunc="std")
    print((m.round(3).astype(str) + " ± " + s.round(3).astype(str)).to_string())
    avg = sc.groupby(["seed", "method"])["spearman"].mean().unstack()
    print("\n[시드별 7단계 평균 Spearman]")
    print(avg.round(3).to_string())
    n_core = sel[sel.rule == "core"]["n"]
    print(f"\ncore 크기: 평균 {n_core.mean():.1f}, 범위 {n_core.min()}~{n_core.max()} | "
          f"top1 크기: 평균 {sel[sel.rule == 'top1']['n'].mean():.1f}")
    return piv


def main():
    global S_BUDGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="시드 1..N")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default=("cuda" if S.torch.cuda.is_available() else "cpu") if S.torch else "numpy")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--verbose", action="store_true", help="조합 진행 로그 표시")
    args = ap.parse_args()
    X, ids, budgets, names, y, t, _ = S.load()
    S_BUDGETS = budgets
    out = RESULTS_DIR / "stage2_stability"
    out.mkdir(parents=True, exist_ok=True)
    fs, fc = out / "selections.csv", out / "scores.csv"
    sel = pd.read_csv(fs) if fs.exists() else pd.DataFrame()
    sc = pd.read_csv(fc) if fc.exists() else pd.DataFrame()
    if len(sel) and (sel.get("k", pd.Series([args.k])) != args.k).any():
        raise SystemExit(f"{fs}에 다른 K로 돌린 결과가 있다. 폴더를 비우거나 같은 --k를 쓸 것")
    done = set(sel["seed"]) if len(sel) else set()
    quiet = print if args.verbose else (lambda *a, **k: None)
    print(f"장치 {args.device}, K = {args.k}, 시드 1~{args.seeds} (이미 끝난 시드: {sorted(done) or '없음'})")
    for seed in range(1, args.seeds + 1):
        if seed in done:
            continue
        print(f"\n=== 시드 {seed} ===")
        s1, s2 = run_seed(seed, X, y, t, names, args, quiet)
        s1 = pd.DataFrame(s1).assign(k=args.k)
        sel = pd.concat([sel, s1], ignore_index=True)
        sc = pd.concat([sc, pd.DataFrame(s2)], ignore_index=True)
        sel.to_csv(fs, index=False, encoding="utf-8-sig")      # 시드마다 저장 → 끊겨도 이어서
        sc.to_csv(fc, index=False, encoding="utf-8-sig")
    summarize(sel, sc, names, out)
    print(f"\n저장: {out}/")


S_BUDGETS = []
if __name__ == "__main__":
    main()
