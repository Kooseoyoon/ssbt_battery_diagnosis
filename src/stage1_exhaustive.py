"""
Stage 1 - 전수 검사로 SA 설정 고르기 (명세서 4절 3단계, 9가지 변경 6번 반영)

작은 문제는 모든 배정을 채점해 '진짜 최적'을 구할 수 있다. 그 정답과 비교해
SA 설정 후보마다 전체 탐색(T_eol 정렬 + 무작위 20, 각 시작점에서 교환 탐색과 SA)이
진짜 최적을 얼마나 놓치는지 잰다.

선택 규칙: 모든 문제에서 놓친 폭 <= 1% 인 설정 중 평균 시간이 가장 짧은 것
           -> configs/search.yaml 에 고정하고 86셀 본 계산에도 그대로 쓴다.

※ 작은 문제에서의 정황 증거일 뿐 86셀 문제의 최적성을 증명하지 못한다(명세서 3-f).

문제 종류 (결정 30 + 9가지 변경 6번)
    12c_4s   12셀 4직렬 3팩            5,775 분할
    16c_8s   16셀 8직렬 2팩            6,435 분할
    13c_4s   13셀 4직렬 3팩 + 미사용 1  75,075 분할
    18c_8s   18셀 8직렬 2팩 + 미사용 2  984,555 분할

실행 (저장소 루트에서)
    python src/stage1_exhaustive.py                     # 명세서대로 종류별 30문제
    python src/stage1_exhaustive.py --n-problems 3      # 빠른 시험 실행
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR
from common import pack as P
from common import search as S

THR = 1.44                        # EOL 60% (주 결과)
N_VALUES = (0, 10)                # n_i = 0 과 10 둘 다

PROBLEM_TYPES = [
    dict(name="12c_4s", n_cells=12, pack_size=4),
    dict(name="16c_8s", n_cells=16, pack_size=8),
    dict(name="13c_4s", n_cells=13, pack_size=4),   # 미사용 1 (변경 6번)
    dict(name="18c_8s", n_cells=18, pack_size=8),   # 미사용 2 (변경 6번)
]


# SA 설정 후보. 맨 앞은 'SA 없이 교환 탐색만' 기준선.
# 냉각률은 sa_end_ratio(마지막 온도 / 처음 온도)로 시도 횟수에 맞춰 자동 계산된다.
CANDIDATES = {
    "swap_only": dict(sa_enabled=False),
    "sa_2k":     dict(sa_enabled=True, sa_iters=2000,  sa_end_ratio=1e-4),
    "sa_5k":     dict(sa_enabled=True, sa_iters=5000,  sa_end_ratio=1e-4),
    "sa_20k":    dict(sa_enabled=True, sa_iters=20000, sa_end_ratio=1e-4),
}


# --------------------------------------------------------------------------
# 진짜 최적: 모든 배정을 채점
# --------------------------------------------------------------------------
def exact_optimum(obj, n_cells, pack_size):
    """모든 배정(미사용 포함)을 채점해 (최적값, 최적 라벨, 배정 수)를 돌려준다.

    가능한 '팩'은 C(n_cells, pack_size)개뿐이라 먼저 전부 채점해 두고(비트마스크로 저장),
    배정은 미리 구한 팩 점수의 합만 계산한다.
    """
    n_packs = n_cells // pack_size
    n_unused = n_cells - n_packs * pack_size

    pv = {}
    for combo in itertools.combinations(range(n_cells), pack_size):
        mask = 0
        for c in combo:
            mask |= 1 << c
        pv[mask] = obj(np.array(combo))

    best = [-np.inf, None]
    count = 0

    def partitions(rest, acc_value, acc_masks):
        """rest를 pack_size개 묶음으로 나누는 모든 방법. 맨 앞 셀을 고정해 중복을 없앤다."""
        nonlocal count
        if not rest:
            count += 1
            if acc_value > best[0]:
                best[0], best[1] = acc_value, list(acc_masks)
            return
        head, tail = rest[0], rest[1:]
        for others in itertools.combinations(tail, pack_size - 1):
            mask = 1 << head
            for c in others:
                mask |= 1 << c
            remaining = [c for c in tail if not (mask >> c) & 1]
            acc_masks.append(mask)
            partitions(remaining, acc_value + pv[mask], acc_masks)
            acc_masks.pop()

    for unused in itertools.combinations(range(n_cells), n_unused):
        used = [c for c in range(n_cells) if c not in unused]
        partitions(used, 0.0, [])

    labels = np.full(n_cells, P.UNUSED, dtype=int)
    for p, mask in enumerate(best[1]):
        for c in range(n_cells):
            if (mask >> c) & 1:
                labels[c] = p
    return best[0], labels, count


# --------------------------------------------------------------------------
# 도우미
# --------------------------------------------------------------------------
def t_eol_from_C(C, cycles, thr):
    """셀별로 처음 thr 아래로 내려간 사이클. 끝까지 안 내려가면 +inf (검열 셀을 '가장 좋은 셀'로)."""
    out = np.full(C.shape[0], np.inf)
    for i in range(C.shape[0]):
        below = np.flatnonzero(C[i] < thr)
        if below.size:
            out[i] = cycles[below[0]]
    return out


def make_starts(scores, n_cells, pack_size, n_random, rng):
    starts = [("sorted_teol", S.sorted_start(scores, pack_size))]
    starts += [(f"random_{k:02d}", S.random_start(n_cells, pack_size, rng))
               for k in range(n_random)]
    return starts


def same_unused(lab_a, lab_b):
    return set(np.flatnonzero(lab_a == P.UNUSED)) == set(np.flatnonzero(lab_b == P.UNUSED))


def repro_info(cfg):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    return dict(data_sha=sha(PROCESSED_DIR / "C_full.npz"), code_commit=commit,
                thr=THR, base_config=cfg.to_dict(), candidates=CANDIDATES,
                run_at=datetime.now().isoformat(timespec="seconds"))


# --------------------------------------------------------------------------
# 본체
# --------------------------------------------------------------------------
def run(n_problems=30, seed=0, types=None, candidates=None, config_path="configs/search.yaml"):
    d = np.load(PROCESSED_DIR / "C_full.npz")
    C, cycles = d["C_full"], d["cycles"]
    teol = t_eol_from_C(C, cycles, THR)
    base_cfg = S.SearchConfig.from_yaml(config_path)
    types = [t for t in PROBLEM_TYPES if types is None or t["name"] in types]
    candidates = {k: v for k, v in CANDIDATES.items() if candidates is None or k in candidates}
    scorers = {n: P.PackScorer(C, n, THR) for n in N_VALUES}

    rng_prob = np.random.default_rng(seed)
    rows = []
    for pt in types:
        for pid in range(n_problems):
            cells = np.sort(rng_prob.choice(C.shape[0], size=pt["n_cells"], replace=False))
            for n in N_VALUES:
                sc = scorers[n]
                obj = sc.objective_for(cells)          # 위치 -> 행 번호

                t = time.perf_counter()
                opt, opt_labels, n_parts = exact_optimum(obj, len(cells), pt["pack_size"])
                t_exact = time.perf_counter() - t
                ub = sc.upper_bound(cells, pt["pack_size"])

                rng_start = np.random.default_rng([seed, pid, n, pt["n_cells"], pt["pack_size"]])
                starts = make_starts(teol[cells], len(cells), pt["pack_size"],
                                     base_cfg.n_random_starts, rng_start)

                for name, override in candidates.items():
                    cfg = S.SearchConfig(**{**base_cfg.to_dict(), **override})
                    t = time.perf_counter()
                    res = S.best_of(starts, obj, pt["pack_size"], cfg, seed=seed)
                    sec = time.perf_counter() - t
                    rows.append(dict(
                        problem_type=pt["name"], problem_id=pid, n_i=n,
                        cells=json.dumps(cells.tolist()), n_partitions=n_parts,
                        true_opt=opt, exact_sec=t_exact,
                        ub=ub, ub_vs_opt_pct=100 * (ub - opt) / opt,
                        config=name, found=res.value,
                        miss_pct=100 * (opt - res.value) / opt,
                        found_same_unused=same_unused(res.labels, opt_labels),
                        best_kind=res.best_kind, search_sec=sec,
                    ))
            print(f"  {pt['name']} #{pid} done", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exhaustive_check.csv", index=False)
    (out / "exhaustive_check_meta.json").write_text(
        json.dumps(repro_info(base_cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def summarize(df, tol_pct=1.0):
    s = (df.groupby("config")
           .agg(max_miss_pct=("miss_pct", "max"), mean_miss_pct=("miss_pct", "mean"),
                n_missed=("miss_pct", lambda x: int((x > 1e-9).sum())),
                n_problems=("miss_pct", "size"), mean_sec=("search_sec", "mean"))
           .reset_index().sort_values("mean_sec"))
    by_type = df.pivot_table(index="config", columns="problem_type",
                             values="miss_pct", aggfunc="max")
    ok = s[s.max_miss_pct <= tol_pct]
    chosen = ok.iloc[0]["config"] if len(ok) else None
    return s, by_type, chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", nargs="*", help="예: 12c_4s 18c_8s")
    ap.add_argument("--candidates", nargs="*", help="예: swap_only sa_5k")
    args = ap.parse_args()

    df = run(args.n_problems, args.seed, args.types, args.candidates)
    summary, by_type, chosen = summarize(df)
    pd.set_option("display.width", 200)
    print("\n[설정별 요약]\n", summary.round(4).to_string(index=False))
    print("\n[문제 종류별 최대 놓친 폭 %]\n", by_type.round(4).to_string())
    base = df.drop_duplicates(["problem_type", "problem_id", "n_i"])
    print("\nUB - 진짜 최적: 평균 %.2f%%, 최대 %.2f%%"
          % (base.ub_vs_opt_pct.mean(), base.ub_vs_opt_pct.max()))
    print("선택된 설정:", chosen if chosen else "없음 -> 후보 범위를 넓힐 것")
