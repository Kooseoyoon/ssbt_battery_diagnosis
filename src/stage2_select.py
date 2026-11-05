"""Stage 2 (재시작) — 특징 선택 C안: 1차 거르기 + 상위 K개 전체 조합 비교, 전부 nested CV 안에서.

바깥 5-fold(folds.csv)마다 학습 셀(약 69개)만으로 아래 1~3을 하고, 고른 조합을 시험 셀(약 17개)로 채점한다.
  1. 1차 거르기: 특징마다 |Spearman(특징, T_eol)|을 진단량 7단계 평균으로 매겨 높은 순으로 줄 세운다.
     위에서부터 하나씩 받되, 이미 받은 특징과 |r| ≥ 0.95(7단계 중 하나라도)이면 건너뛴다 → 상위 K개
     (특징끼리 상관은 정답을 안 쓰므로, 겹치는 묶음에서 누구를 대표로 남길지만 점수로 정하는 셈)
  2. 전체 조합: K개의 공집합 아닌 부분집합 2^K − 1개 전부를 리지 회귀 + 안쪽 5-fold CV로 채점
     점수 = 안쪽 out-of-fold 예측의 Spearman, 진단량 7단계 평균. 리지 강도는 조합·진단량마다
     안쪽 CV 오차(log T_eol MSE)가 가장 작은 값을 고른다
  3. 점수 1등 조합을 그 바깥 fold의 선택으로 한다
바깥 시험 셀은 1~3 어디에도 쓰지 않으므로, 바깥 5 fold를 이어 붙인 Spearman이 '이 선택 절차'의 정직한 성능이다.
마지막으로 같은 절차를 86셀 전체에 한 번 돌려 GP에 넘길 최종 특징 묶음을 정한다(관례: 성능은 nested로,
최종 모델은 전체로).

모든 진단량에 같은 특징 묶음을 쓴다(GP 35개가 같은 입력 정의를 쓰게). 그래서 7단계 모두 계산되는 특징만 후보다.
측정 사이클 기준 양의 slope1(n = 10 불가), slope2(n = 10·15 불가)는 후보에서 빠진다.

실행 (저장소 루트, GPU 권장):
    python src/stage2_select.py                     # K = 20 (조합 1,048,575개)
    python src/stage2_select.py --k 12 --device cpu # 빠른 확인
입력:  data/processed/features_v2.npz, T_eol.npy, C_full.npz, folds.csv
출력:  results/stage2_select/
       outer_summary.csv     바깥 fold별 1차 후보 K개, 고른 조합, 안쪽 점수, 바깥 시험 Spearman
       outer_scores.csv      진단량별 바깥 Spearman: 고른 조합 / 1차 후보 K개 전부 / 현재 용량만
       final_top_subsets.csv 86셀 전체로 돌린 결과 상위 50개 조합과 점수
       final_selection.json  최종 특징 묶음
       screen_ranks.csv      1차 거르기 순위(바깥 fold별 + 전체)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # GPU 없는 환경에서는 numpy로 (느림, --k 작게)
    torch = None
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

SEED = 0
N_INNER = 5
ALPHAS = np.logspace(-2, 3, 11)       # 안쪽 조합 채점용 리지 강도 후보
ALPHAS_OUTER = np.logspace(-4, 3, 29) # 바깥 채점은 Stage 2 점검 코드와 같은 GCV 규칙
CORR_DUP = 0.95
MAX_NAN = 0.10                        # 어느 진단량에서든 결측이 10% 넘는 특징은 후보에서 뺀다
EXCLUDE_BASES = {"dqdv_p2_h", "dqdv_p2_v"}   # 86셀 중 77~88% 결측(피크가 아니라 어깨)
BASELINE = "q_rpt__last"


# ---------------------------------------------------------------- 데이터
def load():
    d = np.load(PROCESSED_DIR / "features_v2.npz")
    X, ids, budgets = d["X"].astype(float), d["cell_ids"].astype(int), [int(b) for b in d["budgets"]]
    names, bases = [str(x) for x in d["feature_names"]], [str(x) for x in d["base_names"]]
    cf = np.load(PROCESSED_DIR / "C_full.npz")
    t_map = dict(zip((int(c) for c in cf["cell_ids"]), np.load(PROCESSED_DIR / "T_eol.npy")))
    t = np.array([t_map[c] for c in ids], float)
    f = pd.read_csv(PROCESSED_DIR / "folds.csv").set_index("cell_id").loc[ids, "fold"].to_numpy()
    keep = [j for j, (n, b) in enumerate(zip(names, bases))
            if b not in EXCLUDE_BASES and np.isnan(X[:, :, j]).mean(axis=0).max() <= MAX_NAN
            and all(np.nanstd(X[:, k, j]) > 0 for k in range(len(budgets)))]
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    print(f"후보 특징 {len(keep)}개 (제외 {len(dropped)}개: dqdv_p2 계열, 진단량 일부에서 계산 불가, 결측 > {MAX_NAN:.0%})")
    return X[:, :, keep], ids, budgets, [names[j] for j in keep], np.log(t), t, f


def impute_std(Xtr, Xte):
    """학습 셀 중앙값 대체 + 학습 셀 기준 표준화. X: (셀, 진단량, 특징)."""
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def strat_folds(y, k, seed):
    bins = pd.qcut(y, 5, labels=False, duplicates="drop")
    out = np.empty(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(y, bins)):
        out[te] = i
    return out


def spearman_cols(X, y):
    return np.array([abs(pd.Series(X[:, j]).corr(pd.Series(y), method="spearman")) for j in range(X.shape[1])])


# ---------------------------------------------------------------- 1차 거르기
def screen(X, y, names, k):
    """X: (셀, 진단량, 특징) 학습 셀만. 반환: 고른 K개 인덱스, 전체 순위 표."""
    B = X.shape[1]
    score = np.nanmean([spearman_cols(X[:, b, :], y) for b in range(B)], axis=0)
    corr = np.stack([pd.DataFrame(X[:, b, :]).corr(method="spearman").abs().to_numpy() for b in range(B)])
    cmax = np.nanmax(corr, axis=0)
    order = np.argsort(-score)
    chosen, dup_of = [], {}
    for j in order:
        hit = [c for c in chosen if cmax[j, c] >= CORR_DUP]
        if hit:
            dup_of[j] = hit[0]
            continue
        if len(chosen) < k:
            chosen.append(j)
    tab = pd.DataFrame({"feature": [names[j] for j in order], "score": score[order],
                        "dup_of": [names[dup_of[j]] if j in dup_of else "" for j in order],
                        "in_top_k": [j in chosen for j in order]})
    return chosen, tab


# ---------------------------------------------------------------- 전체 조합 (GPU)
class Backend:
    """torch(GPU)가 있으면 torch, 없으면 numpy로 같은 계산을 한다. numpy는 느리니 K를 줄여서 쓴다."""

    def __init__(self, device):
        self.torch = device != "numpy"
        if self.torch:
            self.dt = torch.float64 if device == "cpu" else torch.float32
            self.dev = device

    def arr(self, x):
        return torch.as_tensor(np.asarray(x), dtype=self.dt, device=self.dev) if self.torch else np.asarray(x, float)

    def empty(self, *shape):
        return torch.empty(*shape, dtype=self.dt, device=self.dev) if self.torch else np.empty(shape)

    def solve(self, A, b):
        return torch.linalg.solve(A, b) if self.torch else np.linalg.solve(A, b)

    def rank(self, x):
        """마지막 축 순위. 예측값은 연속값이라 동률이 사실상 없다."""
        if self.torch:
            return x.argsort(-1).argsort(-1).to(x.dtype)
        return x.argsort(-1).argsort(-1).astype(float)

    def zstd(self, x):
        if self.torch:
            return (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
        return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)

    def take_best(self, pred, best):
        """pred (m, a, B, n)에서 (m, B)마다 best 리지 강도를 골라 (m, B, n)."""
        m, _, B, n = pred.shape
        if self.torch:
            return torch.gather(pred, 1, best[:, None, :, None].expand(m, 1, B, n))[:, 0]
        return np.take_along_axis(pred, best[:, None, :, None], axis=1)[:, 0]

    def np(self, x):
        return x.cpu().numpy() if self.torch else x


def all_masks(k):
    idx = np.arange(1, 2 ** k, dtype=np.int64)
    return ((idx[:, None] >> np.arange(k)) & 1).astype(bool)


def exhaustive(X, y, device, batch, log=print, inner_seed=SEED + 1):
    """X: (셀, 진단량, K) 학습 셀만, y: log T_eol. 반환: 조합별 진단량 평균 Spearman (2^K − 1,), 조합 마스크."""
    be = Backend(device)
    n, B, K = X.shape
    na = len(ALPHAS)
    inner = strat_folds(y, N_INNER, inner_seed)
    # 안쪽 fold × 진단량별로 표준화한 학습 Gram 행렬과 시험 행렬을 미리 만든다
    # (결측 대체·표준화는 특징마다 따로 하는 연산이라 조합과 무관하게 한 번만 하면 된다)
    pre = []
    for b in range(B):
        for f in range(N_INNER):
            tr, te = inner != f, inner == f
            Xtr, Xte = impute_std(X[tr][:, b, :], X[te][:, b, :])
            ym = y[tr].mean()
            te_idx = np.where(te)[0]
            te_idx = torch.as_tensor(te_idx, device=be.dev) if be.torch else te_idx
            pre.append((b, te_idx, be.arr(Xtr.T @ Xtr), be.arr(Xtr.T @ (y[tr] - ym)), be.arr(Xte), ym))
    al = be.arr(ALPHAS)[None, :, None, None]
    eye = be.arr(np.eye(K))
    yt = be.arr(y)
    yr = be.zstd(be.arr(pd.Series(y).rank().to_numpy()))
    masks = all_masks(K)
    out = np.empty(len(masks))
    t0 = time.time()
    for s in range(0, len(masks), batch):
        M = be.arr(masks[s:s + batch].astype(float))            # (m, K)
        m = M.shape[0]
        pred = be.empty(m, na, B, n)
        for b, te, G, g, Xte, ym in pre:
            # 고르지 않은 특징은 행·열을 0으로, 대각을 α로 두면 그 계수는 정확히 0이 된다
            # → 크기가 다른 조합을 같은 K×K 연립방정식으로 한꺼번에 푼다
            A = (M[:, :, None] * G[None] * M[:, None, :])[:, None] + al * eye     # (m, a, K, K)
            rhs = (M * g[None])[:, None, :, None] + 0 * al[..., :1]              # (m, a, K, 1)
            w = be.solve(A, rhs)[..., 0]                                         # (m, a, K)
            pred[:, :, b, te] = w @ Xte.T + ym
        mse = ((pred - yt) ** 2).mean(-1)                         # (m, a, B)
        pb = be.take_best(pred, mse.argmin(1))                    # (m, B, n)
        rho = (be.zstd(be.rank(pb)) * yr).mean(-1)                # (m, B) 안쪽 out-of-fold Spearman
        out[s:s + m] = be.np(rho.mean(-1))
        if s == 0 or (s // batch) % 20 == 0:
            done = s + m
            log(f"    조합 {done:,}/{len(masks):,}  {time.time() - t0:.0f}s"
                f" (예상 총 {(time.time() - t0) / done * len(masks):.0f}s)")
    return np.nan_to_num(out, nan=-1.0), masks


# ---------------------------------------------------------------- 바깥 채점
def outer_fit_predict(Xtr, ytr, Xte):
    Xtr, Xte = impute_std(Xtr, Xte)
    m = RidgeCV(alphas=ALPHAS_OUTER).fit(Xtr, ytr)
    return m.predict(Xte)


def run_selection(X, y, names, k, device, batch, tag, inner_seed=SEED + 1, log=print):
    chosen, tab = screen(X, y, names, k)
    print(f"  [{tag}] 1차 후보 {len(chosen)}개: {[names[j] for j in chosen]}")
    sc, masks = exhaustive(X[:, :, chosen], y, device, batch, log=log, inner_seed=inner_seed)
    top = np.argsort(-sc)[:50]
    tops = [([names[chosen[i]] for i in np.where(masks[t])[0]], float(sc[t])) for t in top]
    best = [chosen[i] for i in np.where(masks[top[0]])[0]]
    print(f"  [{tag}] 1등 조합({len(best)}개, 안쪽 점수 {sc[top[0]]:.3f}): {[names[j] for j in best]}")
    return chosen, best, tops, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu") if torch else "numpy",
                    help="cuda / cpu(torch) / numpy")
    ap.add_argument("--batch", type=int, default=4096, help="한 번에 푸는 조합 수(GPU 메모리에 맞게)")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()
    print(f"장치: {args.device}, K = {args.k} → 조합 {2 ** args.k - 1:,}개")
    X, ids, budgets, names, y, t, fold = load()
    out = RESULTS_DIR / "stage2_select"
    out.mkdir(parents=True, exist_ok=True)
    B = len(budgets)
    jb = names.index(BASELINE)

    pred = {key: np.empty((len(y), B)) for key in ("selected", "top_k_all", "current_cap")}
    rows, ranks = [], []
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        print(f"\n바깥 fold {k}: 학습 {tr.sum()}셀, 시험 {te.sum()}셀")
        chosen, best, tops, tab = run_selection(X[tr], y[tr], names, args.k, args.device, args.batch, f"fold {k}")
        tab.insert(0, "outer_fold", k)
        ranks.append(tab)
        for b in range(B):
            for key, cols in (("selected", best), ("top_k_all", chosen), ("current_cap", [jb])):
                pred[key][te, b] = outer_fit_predict(X[tr][:, b, cols], y[tr], X[te][:, b, cols])
        rho = [spearmanr(pred["selected"][te, b], t[te])[0] for b in range(B)]
        rows.append({"outer_fold": k, "screened": ";".join(names[j] for j in chosen),
                     "selected": ";".join(names[j] for j in best), "n_selected": len(best),
                     "inner_score": tops[0][1], "outer_spearman_mean": float(np.mean(rho))})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "outer_summary.csv", index=False, encoding="utf-8-sig")
    sc = pd.DataFrame([{"method": key, "budget": n, "spearman": spearmanr(np.exp(p[:, b]), t)[0],
                        "mae_cycles": np.mean(np.abs(np.exp(p[:, b]) - t))}
                       for key, p in pred.items() for b, n in enumerate(budgets)])
    sc.to_csv(out / "outer_scores.csv", index=False, encoding="utf-8-sig")
    print("\n[바깥 시험 셀 Spearman (86셀 out-of-fold) — 선택 절차의 정직한 성능]")
    print(sc.pivot(index="method", columns="budget", values="spearman").round(3).to_string())
    print("  안쪽 점수 평균", round(summ["inner_score"].mean(), 3), "vs 바깥 selected 평균",
          round(sc[sc.method == "selected"]["spearman"].mean(), 3), "← 차이가 선택 편향 크기")
    freq = pd.Series([f for s in summ["selected"] for f in s.split(";")]).value_counts()
    print("\n[바깥 5 fold에서 뽑힌 횟수]")
    print(freq.to_string())

    if not args.skip_final:
        print("\n최종: 86셀 전체로 같은 절차")
        chosen, best, tops, tab = run_selection(X, y, names, args.k, args.device, args.batch, "전체")
        tab.insert(0, "outer_fold", "all")
        ranks.append(tab)
        pd.DataFrame([{"rank": i + 1, "n": len(fs), "score": s, "features": ";".join(fs)}
                      for i, (fs, s) in enumerate(tops)]).to_csv(out / "final_top_subsets.csv", index=False,
                                                                 encoding="utf-8-sig")
        (out / "final_selection.json").write_text(json.dumps(
            {"features": [names[j] for j in best], "k": args.k, "screened": [names[j] for j in chosen],
             "inner_score": tops[0][1], "outer_fold_frequency": freq.to_dict()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  최종 특징 {len(best)}개: {[names[j] for j in best]}")
    pd.concat(ranks).to_csv(out / "screen_ranks.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}/")


if __name__ == "__main__":
    main()
