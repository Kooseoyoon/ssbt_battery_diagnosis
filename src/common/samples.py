"""Stage 3 → Stage 4~7 연결: 셀별 현재 진단량에 맞는 예측 샘플을 꺼낸다.

posterior_samples_r{r}.npz 하나 = 반복 r. Q (셀 86, 진단량 7, 샘플 S, 열 390), t_eol (86, 7, S).
셀 i가 지금 n_i사이클 진단된 상태라면 그 셀의 미래 S개는 Q[i, budgets.index(n_i)].
셀 사이 샘플은 독립으로 뽑았으므로(결정 27) 셀마다 다른 진단량 칸을 꺼내 이어 붙여도 된다.
같은 샘플 번호 s를 셀끼리 묶은 것이 '재고 전체의 시나리오 s'다.

    from common.samples import load_samples
    S = load_samples(r=0, n_vec=np.full(86, 10))
    S.Q        # (86, S, 390) float32 — 시나리오 s의 셀 i 용량 궤적. 열 j = 사이클 1 + 5j (C_full과 같음)
    S.t_eol    # (86, S) — 각 궤적이 처음 1.44Ah 미만이 되는 사이클(범위 안 미도달 NaN)
    S.cell_ids, S.cycles, S.budgets

규칙
  - n_vec는 셀 순서 = C_full.npz의 cell_ids. 값은 {10, 15, …, 40}만 (n_i = 0은 샘플이 없음 → Oracle은 C_full로)
  - 열 < n_i/5는 C_full 관측값 그대로, 이후는 예측. pack.py의 j0_i = n_i/5부터 팩에 쓴다
  - 파일 위치: 환경변수 SSBT_SAMPLES_DIR가 있으면 그 폴더(Google Drive 등), 없으면 results/stage3/{tag}/
  - 같은 r 파일은 한 번만 읽어 둔다(약 0.19GB)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CACHE: dict = {}


@dataclass
class Samples:
    Q: np.ndarray          # (N, S, T) float32
    t_eol: np.ndarray      # (N, S)
    cell_ids: np.ndarray
    cycles: np.ndarray
    budgets: list
    n_vec: np.ndarray


def samples_dir(tag="main"):
    env = os.environ.get("SSBT_SAMPLES_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "results" / "stage3" / tag


def _load(r, tag):
    key = (r, tag, str(samples_dir(tag)))
    if key not in _CACHE:
        _CACHE.clear()                                   # 한 번에 반복 하나만 메모리에
        p = samples_dir(tag) / f"posterior_samples_r{r}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음. SSBT_SAMPLES_DIR 또는 results/stage3/{tag}/ 확인")
        d = np.load(p)
        _CACHE[key] = {k: d[k] for k in ("Q", "t_eol", "cell_ids", "cycles", "budgets")}
    return _CACHE[key]


def load_samples(r, n_vec, tag="main"):
    d = _load(r, tag)
    budgets = [int(b) for b in d["budgets"]]
    n_vec = np.asarray(n_vec).astype(int)
    N = d["Q"].shape[0]
    if n_vec.shape != (N,):
        raise ValueError(f"n_vec 길이 {n_vec.shape}가 셀 수 {N}과 다르다")
    bad = sorted(set(n_vec.tolist()) - set(budgets))
    if bad:
        raise ValueError(f"샘플이 없는 진단량 {bad}. 가능한 값 {budgets} (n_i = 0은 C_full로)")
    b = np.searchsorted(budgets, n_vec)
    i = np.arange(N)
    return Samples(Q=d["Q"][i, b], t_eol=d["t_eol"][i, b], cell_ids=d["cell_ids"], cycles=d["cycles"],
                   budgets=budgets, n_vec=n_vec)
