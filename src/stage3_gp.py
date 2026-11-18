"""Stage 3 — GP 학습, conformal 보정, 궤적 샘플링 (명세 4~6절).

실행 (저장소 루트):
    python src/stage3_gp.py --pilot                       # 시험 실행: 진단량 10, fold 0
    python src/stage3_gp.py --pilot --budget 30 --fold 2
    python src/stage3_gp.py --run --jobs 8                # 전체: 진단량 7 × fold 5 학습 → 반복 20 샘플
    python src/stage3_gp.py --run --jobs 8 --features sensitivity --tag s2   # 민감도(S2 입력)
전체 실행 출력: results/stage3/{tag}/   (tag 기본 main)
    fits.pkl                    (진단량, fold)별 FPCA·GP·예측 분포·c_n (중간 저장, 있으면 학습을 건너뜀)
    gp_log.csv, conformal.csv   커널 모수·주변우도·경계 도달·FPCA k·잔차 상관 / c_n
    posterior_samples_r{r}.npz  Q (86, 7, 200, 390) float32, t_eol (86, 7, 200)(미도달 NaN), cycles, cell_ids,
                                budgets, fold. 이미 있는 r은 건너뜀(끊겨도 이어서)
    summary_r0.csv              반복 0의 진단량별 지표(86셀 전체, 보정 전·후)
출력: results/stage3/pilot/
    pilot_n{n}_f{k}.png        시험 셀 6개: 관측(검정 점), 실제 미래(검정 선), 샘플 중앙값, 5~95% 띠, 샘플 20개
    pilot_n{n}_f{k}.txt        FPCA k·분산, GP 커널, 시험 셀 지표(보정 전)

입력: data/processed/features_v2.npz, results/stage2_final_features.json(S7b),
      data/processed/C_full.npz, T_eol.npy, folds.csv
"""
from __future__ import annotations

import argparse
import pickle
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import stage3_repr as R  # noqa: E402

S_DEFAULT = 200
RESAMPLE_MAX = 100
N_RESTARTS = 5


# ------------------------------------------------------------------ 데이터
def load(feature_set="main"):
    d = np.load(PROCESSED_DIR / "features_v2.npz")
    spec = json.load(open(RESULTS_DIR / "stage2_final_features.json", encoding="utf-8"))
    feats = spec["main" if feature_set == "main" else "sensitivity"]["features"]
    names = list(d["feature_names"])
    X = d["X"][:, :, [names.index(f) for f in feats]]                 # (셀, 진단량, 특징)
    fid = [int(c) for c in d["cell_ids"]]
    c = np.load(PROCESSED_DIR / "C_full.npz")
    cid = [int(x) for x in c["cell_ids"]]
    t_eol = np.load(PROCESSED_DIR / "T_eol.npy")
    folds = pd.read_csv(PROCESSED_DIR / "folds.csv").set_index("cell_id")["fold"]
    # 셀 순서를 C_full 기준으로 맞춘다(ID로 연결, 순서를 가정하지 않음)
    order = [fid.index(i) for i in cid]
    return {"X": X[order], "feats": feats, "budgets": [int(b) for b in d["budgets"]],
            "C": c["C_full"].astype(float), "cycles": c["cycles"].astype(float), "cell_ids": np.array(cid),
            "t_eol": t_eol.astype(float), "fold": folds.loc[cid].to_numpy()}


def impute_std(Xtr, Xte):
    med = np.nanmedian(Xtr, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ------------------------------------------------------------------ GP
def make_gp(d):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=2.5) \
        + WhiteKernel(0.1, (1e-6, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=N_RESTARTS, random_state=0)


class OutputGPs:
    """출력마다 독립 GP. 출력은 학습 셀 평균·표준편차로 표준화."""

    def fit(self, X, Y):
        self.ym, self.ys = Y.mean(0), Y.std(0)
        self.ys[self.ys == 0] = 1.0
        Z = (Y - self.ym) / self.ys
        self.gps, self.warn = [], []
        for j in range(Y.shape[1]):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.gps.append(make_gp(X.shape[1]).fit(X, Z[:, j]))
            # 경계에 닿은 모수 경고(길이 척도 상한 = 그 특징을 사실상 안 씀, 잡음 하한 등)는 화면에 뿌리지 않고 기록만
            self.warn.append(sum("bound" in str(x.message) for x in w))
        return self

    def predict(self, X):
        """예측 평균과 표준편차(잡음 포함), 원래 단위."""
        out = [g.predict(X, return_std=True) for g in self.gps]
        mu = np.stack([m for m, _ in out], 1) * self.ys + self.ym
        sd = np.stack([s for _, s in out], 1) * self.ys
        return mu, sd

    def log(self):
        return [{"output": j, "kernel": str(g.kernel_), "lml": g.log_marginal_likelihood_value_,
                 "bound_warnings": self.warn[j]} for j, g in enumerate(self.gps)]


def fit_on(D, b, tr, te):
    """진단량 인덱스 b에서 학습 셀 tr로 FPCA + 출력 GP, 시험 셀 te의 예측 분포. tr·te는 셀 인덱스 배열."""
    S = np.stack([R.curve_to_u(D["C"][i], D["cycles"], D["t_eol"][i]) for i in tr])
    fp = R.FPCA().fit(S)
    Y = np.column_stack([np.log(D["t_eol"][tr]), fp.transform(S)])
    Xtr, Xte = impute_std(D["X"][tr, b], D["X"][te, b])
    gp = OutputGPs().fit(Xtr, Y)
    resid = Y - gp.predict(Xtr)[0]
    rcorr = np.corrcoef(resid.T) if Y.shape[1] > 1 else np.ones((1, 1))
    mu, sd = gp.predict(Xte)
    return fp, gp, mu, sd, rcorr


def fit_fold(D, b, k):
    """진단량 인덱스 b, 바깥 fold k: 학습 셀만 쓴다."""
    tr, te = np.where(D["fold"] != k)[0], np.where(D["fold"] == k)[0]
    return (te, *fit_on(D, b, tr, te))


# ------------------------------------------------------------------ 샘플링
def sample_cell(rng, mu, sd, fp, c_row, n, cycles, S=S_DEFAULT):
    """한 셀의 궤적 S개. 반환 Q (S, 열), t_eol (S,), 다시 뽑기 실패 수."""
    q0 = c_row[0]
    Q = np.empty((S, len(cycles)))
    T = np.empty(S)
    n_fail = 0
    for s in range(S):
        for _ in range(RESAMPLE_MAX):
            y = mu + sd * rng.standard_normal(len(mu))
            t = float(np.exp(y[0]))
            if t > n:
                break
        else:
            t, n_fail = n + 5.0, n_fail + 1
        s_u = fp.inverse(y[1:])
        q = R.trajectory(s_u, q0, t, cycles)
        q = R.finalize(c_row, n, q, cycles)
        Q[s], T[s] = q, R.t_eol_of(q, cycles)
    return Q, T, n_fail


def cell_rng(r, b, cell_id, tag=0):
    """반복 r, 진단량 b, 셀마다 독립 난수. tag = 0 본 샘플, 1 + 바깥 fold = conformal 보정용 안쪽 예측."""
    return np.random.default_rng(np.random.SeedSequence(r, spawn_key=(b, int(cell_id), tag)))


def sample_cells(D, b, n, idx, fp, mu, sd, r=0, tag=0, S=S_DEFAULT):
    Qs, Ts, fails = [], [], 0
    for ii, i in enumerate(idx):
        Q, T, f = sample_cell(cell_rng(r, b, D["cell_ids"][i], tag), mu[ii], sd[ii], fp, D["C"][i], n, D["cycles"], S)
        Qs.append(Q)
        Ts.append(T)
        fails += f
    return Qs, Ts, fails


# ------------------------------------------------------------------ conformal (명세 6절, 결정 26)
SD_FLOOR = 1e-4
INNER_SEED = 3000


def strat_folds(t_eol, k=5, seed=0):
    """T_eol 5분위 층화 k-fold (Stage 2와 같은 방식)."""
    rng = np.random.default_rng(seed)
    q = pd.qcut(t_eol, 5, labels=False, duplicates="drop")
    f = np.zeros(len(t_eol), int)
    for g in np.unique(q):
        m = np.where(q == g)[0]
        rng.shuffle(m)
        f[m] = np.arange(len(m)) % k
    return f


def conformal_factor(D, b, n, k, S=S_DEFAULT):
    """바깥 fold k의 학습 셀만으로 c_n: 안쪽 5-fold 예측의 숨은 미래 점수 90% 분위수 / 1.645."""
    tr = np.where(D["fold"] != k)[0]
    inner = strat_folds(D["t_eol"][tr], 5, INNER_SEED + 10 * k + b)
    scores = []
    for j in range(5):
        itr, ite = tr[inner != j], tr[inner == j]
        fp, _, mu, sd, _ = fit_on(D, b, itr, ite)
        Qs, _, _ = sample_cells(D, b, n, ite, fp, mu, sd, r=0, tag=1 + k, S=S)
        for i, Q in zip(ite, Qs):
            j_end = int(np.searchsorted(D["cycles"], D["t_eol"][i]))
            y = D["C"][i, n // 5: j_end + 1]
            q = Q[:, n // 5: j_end + 1]
            scores.append(np.abs(y - q.mean(0)) / np.maximum(q.std(0), SD_FLOOR))
    sc = np.concatenate(scores)
    return float(np.quantile(sc, 0.9) / 1.645), len(sc)


def apply_conformal(Q, n, c):
    """예측 열에서 샘플 평균 둘레 차이를 c배 → 비증가·0 하한 다시 적용. 관측 열은 그대로."""
    Q = Q.copy()
    f = slice(n // 5, None)
    m = Q[:, f].mean(0)
    Q[:, f] = m + c * (Q[:, f] - m)
    Q[:, f] = np.minimum(np.minimum.accumulate(Q[:, f], axis=1), Q[:, n // 5 - 1: n // 5])
    Q = np.clip(Q, 0.0, None)
    return Q


# ------------------------------------------------------------------ 지표 (보정 전, 시험 셀)
def metrics(D, idx, n, Qs, Ts):
    cyc = D["cycles"]
    t_start = n // 5
    rmse, mae, picp, crps = [], [], [], []
    for i, Q, in zip(idx, Qs):
        j_end = int(np.searchsorted(cyc, D["t_eol"][i]))
        y = D["C"][i, t_start: j_end + 1]
        q = Q[:, t_start: j_end + 1]
        m = q.mean(0)
        lo, hi = np.percentile(q, [5, 95], axis=0)
        rmse.append(np.sqrt(np.mean((m - y) ** 2)))
        mae.append(np.mean(np.abs(m - y)))
        picp.append(np.mean((y >= lo) & (y <= hi)))
        # 샘플 기반 CRPS = E|X − y| − ½E|X − X'|
        e1 = np.abs(q - y).mean(0)
        qs = np.sort(q, 0)
        S = q.shape[0]
        w = (2 * np.arange(1, S + 1) - S - 1) / (S * S)
        e2 = (w[:, None] * qs).sum(0)
        crps.append(np.mean(e1 - e2))
    t_true = D["t_eol"][idx]
    # 예측 범위(사이클 1946) 안에서 1.44 아래로 안 내려간 샘플 = 미도달(NaN). 무한대로 두고 분위수를 잰다(결정 15)
    big = 1e6                                    # 무한대 대신 큰 유한값(분위수 보간에서 inf − inf = NaN 방지)
    Tin = [np.where(np.isnan(t), big, t) for t in Ts]
    cens = np.mean(np.concatenate([np.isnan(t) for t in Ts]))
    t_med = np.array([np.median(t) for t in Tin])
    t_lo = np.array([np.percentile(t, 5) for t in Tin])
    t_hi = np.array([np.percentile(t, 95) for t in Tin])
    horizon = cyc[-1] + 5
    return {"n_test": len(idx), "rmse_Ah": np.mean(rmse), "mae_Ah": np.mean(mae), "crps_Ah": np.mean(crps),
            "picp90_pointwise": np.mean(picp), "eol_censored_frac": cens,
            "eol_mae_cyc": np.mean(np.abs(np.minimum(t_med, horizon) - t_true)),
            "eol_cover90": np.mean((t_true >= t_lo) & (t_true <= t_hi)),
            "eol_spearman": spearmanr(np.minimum(t_med, horizon), t_true)[0]}


# ------------------------------------------------------------------ 시험 실행
def pilot(D, n, k, S, out):
    b = D["budgets"].index(n)
    t0 = time.time()
    idx, fp, gp, mu, sd, rcorr = fit_fold(D, b, k)
    Qs, Ts, fails = sample_cells(D, b, n, idx, fp, mu, sd, r=0, tag=0, S=S)
    m0 = metrics(D, idx, n, Qs, Ts)
    c_n, n_sc = conformal_factor(D, b, n, k, S)
    Qs = [apply_conformal(Q, n, c_n) for Q in Qs]
    Ts = [np.array([R.t_eol_of(q, D["cycles"]) for q in Q]) for Q in Qs]
    m = metrics(D, idx, n, Qs, Ts)
    lines = [f"진단량 n = {n}, fold {k}: 학습 {int((D['fold'] != k).sum())}셀, 시험 {len(idx)}셀, "
             f"특징 {D['feats']}",
             f"FPCA k = {fp.k}, 누적 분산 {fp.explained:.4f} (성분별 {np.round(fp.explained_all[:4], 4).tolist()})",
             *[f"  GP 출력 {r['output']} ({'log T_eol' if r['output'] == 0 else 'z' + str(r['output'])}): "
               f"{r['kernel']} | lml {r['lml']:.2f} | 경계 도달 {r['bound_warnings']}" for r in gp.log()],
             f"학습 잔차 상관(출력 간):\n{np.round(rcorr, 2)}",
             f"T_eol 다시 뽑기 실패(n + 5로 둠): {fails}회 / {len(idx) * S}",
             "시험 셀 지표 보정 전(반복 0): " + ", ".join(f"{k_}={v:.4f}" for k_, v in m0.items()),
             f"conformal 배율 c_n = {c_n:.3f} (안쪽 5-fold 점수 {n_sc}개)",
             "시험 셀 지표 보정 후(반복 0): " + ", ".join(f"{k_}={v:.4f}" for k_, v in m.items()),
             f"시간 {time.time() - t0:.1f}초"]
    txt = "\n".join(lines)
    print(txt)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pilot_n{n}_f{k}.txt").write_text(txt, encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(D["t_eol"][idx])
        pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]        # 수명 짧은 셀 ~ 긴 셀
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        cyc = D["cycles"]
        for ax, p in zip(axes.flat, pick):
            i = idx[p]
            Q = Qs[p]
            jt = int(np.searchsorted(cyc, D["t_eol"][i]))
            xmax = max(D["t_eol"][i], np.percentile(Ts[p], 95)) * 1.15
            lo, med, hi = np.percentile(Q, [5, 50, 95], axis=0)
            ax.fill_between(cyc, lo, hi, color="C0", alpha=0.2, label="5~95%")
            for s in range(0, len(Q), max(1, len(Q) // 20)):
                ax.plot(cyc, Q[s], color="C0", lw=0.3, alpha=0.5)
            ax.plot(cyc, med, color="C0", lw=1.5, label="sample median")
            ax.plot(cyc[: jt + 5], D["C"][i, : jt + 5], color="k", lw=1, label="actual")
            ax.plot(cyc[: n // 5], D["C"][i, : n // 5], "ko", ms=3, label="observed")
            ax.axhline(R.EOL_AH, color="gray", ls="--", lw=0.8)
            ax.axvline(D["t_eol"][i], color="k", ls=":", lw=0.8)
            ax.set_xlim(0, xmax)
            ax.set_title(f"cell {D['cell_ids'][i]}: T_eol {D['t_eol'][i]:.0f}, "
                         f"pred {np.median(Ts[p]):.0f} [{np.percentile(Ts[p], 5):.0f}, {np.percentile(Ts[p], 95):.0f}]",
                         fontsize=9)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Stage 3 pilot: n = {n}, fold {k} (after conformal, c_n = {c_n:.2f})")
        fig.supxlabel("cycle")
        fig.supylabel("capacity (Ah)")
        fig.tight_layout()
        fig.savefig(out / f"pilot_n{n}_f{k}.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("그림 저장 실패:", e)
    print(f"저장: {out}/pilot_n{n}_f{k}.png, .txt")


# ------------------------------------------------------------------ 전체 실행
def fit_one(D, b, n, k, S):
    idx, fp, gp, mu, sd, rcorr = fit_fold(D, b, k)
    c_n, n_sc = conformal_factor(D, b, n, k, S)
    logs = [{"budget": n, "fold": k, **r, "fpca_k": fp.k, "fpca_explained": fp.explained} for r in gp.log()]
    for i in range(len(rcorr)):
        for j in range(i + 1, len(rcorr)):
            logs.append({"budget": n, "fold": k, "output": f"resid_corr_{i}_{j}", "kernel": "", "lml": rcorr[i, j],
                         "bound_warnings": 0, "fpca_k": fp.k, "fpca_explained": fp.explained})
    return {"b": b, "n": n, "k": k, "idx": idx, "fp": fp, "mu": mu, "sd": sd, "c_n": c_n, "n_scores": n_sc, "log": logs}


def run_all(D, reps, S, jobs, out):
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / "fits.pkl"
    combos = [(b, n, k) for b, n in enumerate(D["budgets"]) for k in sorted(set(D["fold"]))]
    if fpath.exists():
        fits = pickle.load(open(fpath, "rb"))
        print(f"학습 결과 불러옴: {fpath} ({len(fits)}개)")
    else:
        t0 = time.time()
        print(f"학습: 진단량 {len(D['budgets'])} × fold {len(set(D['fold']))} = {len(combos)}개 (conformal 안쪽 5-fold 포함), "
              f"병렬 {jobs}")
        if jobs > 1:
            from joblib import Parallel, delayed
            fits = Parallel(n_jobs=jobs, verbose=5)(delayed(fit_one)(D, b, n, k, S) for b, n, k in combos)
        else:
            fits = []
            for b, n, k in combos:
                fits.append(fit_one(D, b, n, k, S))
                print(f"  n = {n}, fold {k}: c_n = {fits[-1]['c_n']:.3f} ({time.time() - t0:.0f}초)")
        pickle.dump(fits, open(fpath, "wb"))
        print(f"학습 끝 {time.time() - t0:.0f}초")
    pd.DataFrame([r for f in fits for r in f["log"]]).to_csv(out / "gp_log.csv", index=False, encoding="utf-8-sig")
    cf = pd.DataFrame([{"budget": f["n"], "fold": f["k"], "c_n": f["c_n"], "n_scores": f["n_scores"]} for f in fits])
    cf.to_csv(out / "conformal.csv", index=False, encoding="utf-8-sig")
    print("c_n (진단량 × fold):\n" + cf.pivot(index="budget", columns="fold", values="c_n").round(3).to_string())

    N, B, T = len(D["cell_ids"]), len(D["budgets"]), len(D["cycles"])
    for r in range(reps):
        p = out / f"posterior_samples_r{r}.npz"
        if p.exists():
            continue
        t0 = time.time()
        Q = np.empty((N, B, S, T), np.float32)
        TE = np.empty((N, B, S))
        rows = []
        for f in fits:
            b, n = f["b"], f["n"]
            Qs, _, fails = sample_cells(D, b, n, f["idx"], f["fp"], f["mu"], f["sd"], r=r, tag=0, S=S)
            Qraw = Qs
            Qs = [apply_conformal(q, n, f["c_n"]) for q in Qs]
            for ii, i in enumerate(f["idx"]):
                Q[i, b] = Qs[ii]
                TE[i, b] = [R.t_eol_of(q, D["cycles"]) for q in Qs[ii]]
            if r == 0:
                rows.append((n, f["idx"], Qraw, Qs, fails))
        np.savez(p, Q=Q, t_eol=TE, cycles=D["cycles"], cell_ids=D["cell_ids"], budgets=np.array(D["budgets"]),
                 fold=D["fold"])
        print(f"반복 {r}: 저장 {p.name} ({time.time() - t0:.0f}초, 미도달 샘플 {np.isnan(TE).mean():.4f})")
        if r == 0:
            summ = []
            for n in D["budgets"]:
                parts = [x for x in rows if x[0] == n]
                idx = np.concatenate([x[1] for x in parts])
                for lab, j in (("before", 2), ("after", 3)):
                    Qs = [q for x in parts for q in x[j]]
                    Ts = [np.array([R.t_eol_of(q, D["cycles"]) for q in Qc]) for Qc in Qs]
                    summ.append({"budget": n, "conformal": lab, **metrics(D, idx, n, Qs, Ts)})
            sm = pd.DataFrame(summ)
            sm.to_csv(out / "summary_r0.csv", index=False, encoding="utf-8-sig")
            print("반복 0, 86셀 전체 지표:\n" + sm.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--S", type=int, default=S_DEFAULT)
    ap.add_argument("--features", default="main", choices=["main", "sensitivity"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default=None, help="출력 폴더 이름(기본: main 또는 s2)")
    a = ap.parse_args()
    D = load(a.features)
    if a.pilot:
        pilot(D, a.budget, a.fold, a.S, RESULTS_DIR / "stage3" / "pilot")
    elif a.run:
        tag = a.tag or ("main" if a.features == "main" else "s2")
        run_all(D, a.reps, a.S, a.jobs, RESULTS_DIR / "stage3" / tag)
    else:
        raise SystemExit("--pilot 또는 --run")


if __name__ == "__main__":
    main()
