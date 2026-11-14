"""
Stage 1 - 미래 정보 가치 G1 본 계산 (명세서 4절 4~7단계, 5절 출력, 9가지 변경 4·7·9번)

무엇을 재나
    같은 재고에서 세 가지로 묶어 팩 성능 합 V를 비교한다.
      V_Current : 모든 셀 n=10, 사이클 6 용량 큰 순으로 묶음 (현재 산업 방식, 탐색 없음)
      V_O(10)   : 모든 셀 n=10, 진짜 미래를 알고 탐색한 최고 (시작점에 Current 포함 -> G_info >= 0)
      V_O(0)    : 모든 셀 n=0,  진짜 미래를 알고 탐색한 최고
    G_info  = (V_O(10) - V_Current) / V_Current   <- 이 연구가 주장하는 값
    G_total = (V_O(0)  - V_Current) / V_Current

미사용 셀 (9가지 변경 7번)
    재고가 팩 크기로 나눠떨어지지 않으면 남는 셀은 '미사용'. 어느 셀을 안 쓸지는 방식마다 스스로 정한다
    (Current: 사이클 6 용량이 낮은 셀 / Oracle: 탐색이 고름). 세 방식 모두 같은 재고에서 출발한다.

실행 단계 (저장소 맨 위 폴더에서)
    python src/stage1_g1.py run --inv 64 --eol 60 --workers 4      # 본 계산 (중단해도 다시 실행하면 이어서 함)
    python src/stage1_g1.py seedvar --workers 4                    # 86셀 탐색 시드 변동 (7단계)
    python src/stage1_g1.py finalize                               # 기준선 갱신 -> CSV, Figure B (6단계, 5절)

    시험 실행: python src/stage1_g1.py run --inv 64 --eol 60 --n-instances 3 --workers 2
"""
from __future__ import annotations

import argparse
import multiprocessing
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR
from common import pack as P
from common import search as S

NOMINAL_AH = 2.4
EOL_PCTS = (55, 60, 65)
PACK_SIZES = (4, 8, 12, 16)
INVENTORY_SIZES = (32, 64, 86)
N_INSTANCES = 1000                 # 32셀, 64셀 각각 (시드 0~999). 86셀은 1개
N_DIAG = 10                        # Current와 V_O(10)의 진단량
SEEDVAR_SEEDS = (0, 1, 2, 3, 4)    # 7단계: 86셀만 탐색 시드 5개
VARIANTS = ("raw", "monotone", "excl")   # 주 결과 raw, 민감도 ① 단조 곡선 ② C_full_excl

OUT = RESULTS_DIR / "stage1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "search.yaml"


# ============================================================ 데이터

def load_curves(variant="raw"):
    """(C, cycles, cell_ids). variant: raw / monotone(누적 최솟값) / excl(C_full_excl)."""
    name = "C_full_excl.npz" if variant == "excl" else "C_full.npz"
    d = np.load(PROCESSED_DIR / name)
    C, cycles, ids = d["C_full"].astype(float), d["cycles"], d["cell_ids"]
    if variant == "monotone":
        C = C.copy()
        for r in range(C.shape[0]):
            ok = np.isfinite(C[r])
            C[r, ok] = np.minimum.accumulate(C[r, ok])
    elif variant not in ("raw", "excl"):
        raise ValueError(f"unknown variant {variant}")
    return C, cycles, ids


def thr_of(eol_pct):
    return NOMINAL_AH * eol_pct / 100.0


def t_eol_from_C(C, cycles, thr):
    """셀별로 처음 thr 아래로 내려간 사이클. 끝까지 안 내려가면 +inf.
    검열 셀을 -1로 두면 sorted_start가 '가장 나쁜 셀'로 착각해 미사용으로 밀어낸다(형 README 4번)."""
    out = np.full(C.shape[0], np.inf)
    for i in range(C.shape[0]):
        below = np.flatnonzero(C[i] < thr)
        if below.size:
            out[i] = cycles[below[0]]
    return out


def cycle6_column(cycles):
    """사이클 6의 열 번호. 변환은 pack.py 함수로만 한다 (사이클 6 = 진단 5사이클이 지난 뒤의 측정)."""
    j = P.future_measurement_idx(5)
    if int(cycles[j]) != 6:
        raise ValueError(f"column {j} is cycle {cycles[j]}, expected 6")
    return j


# ============================================================ 재고 인스턴스 (4단계)

def make_instances(n_cells_total, n_instances=N_INSTANCES):
    """[(inventory_size, subset_seed, rows)] - 시드마다 고정된 부분집합. 86셀은 전부 1개."""
    inst = []
    for size in INVENTORY_SIZES:
        if size >= n_cells_total:
            inst.append((n_cells_total, 0, np.arange(n_cells_total)))
            continue
        for seed in range(n_instances):
            rng = np.random.default_rng([size, seed])
            inst.append((size, seed, np.sort(rng.choice(n_cells_total, size=size, replace=False))))
    return inst


def save_instance_list(instances, ids):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [dict(inventory_size=s, subset_seed=k, cell_ids=json.dumps(ids[r].tolist()))
            for s, k, r in instances]
    pd.DataFrame(rows).to_csv(OUT / "instances.csv", index=False)


# ============================================================ 한 과제 = (재고, 팩 크기, EOL)

_G = {}   # 작업 프로세스마다 한 번 불러 두는 데이터


def _init_worker(variant):
    C, cycles, ids = load_curves(variant)
    _G.update(C=C, cycles=cycles, ids=ids, variant=variant,
              cfg=S.SearchConfig.from_yaml(CONFIG_PATH), scorers={}, teol={})


def _scorer(n, thr):
    key = (n, thr)
    if key not in _G["scorers"]:
        _G["scorers"][key] = P.PackScorer(_G["C"], n, thr)
    return _G["scorers"][key]


def _teol(thr):
    if thr not in _G["teol"]:
        _G["teol"][thr] = t_eol_from_C(_G["C"], _G["cycles"], thr)
    return _G["teol"][thr]


def _pack_values(obj, labels):
    return [obj(p) for p in P.packs_from_labels(labels)]


def _starts(rows, pack_size, teol, rng, n_random, current=None):
    starts = []
    if current is not None:
        starts.append(("current", current))
    starts.append(("sorted_teol", S.sorted_start(teol[rows], pack_size)))
    starts += [(f"random_{k:02d}", S.random_start(len(rows), pack_size, rng))
               for k in range(n_random)]
    return starts


def task_path(variant, size, seed, pack_size, eol, search_seed=None):
    tag = "" if search_seed is None else f"_ss{search_seed}"
    return OUT / "assignments" / variant / f"inv{size}_s{seed:04d}_p{pack_size}_e{eol}{tag}.npz"


def run_task(size, seed, rows, pack_size, eol, search_seed=None):
    """한 조건을 계산하고 파일 하나로 저장. 이미 있으면 건너뜀(이어서 실행)."""
    path = task_path(_G["variant"], size, seed, pack_size, eol, search_seed)
    if path.exists():
        return str(path), 0.0
    t0 = time.perf_counter()
    C, cycles, ids, cfg = _G["C"], _G["cycles"], _G["ids"], _G["cfg"]
    if search_seed is not None:
        cfg = S.SearchConfig(**{**cfg.to_dict(), "seed": int(search_seed)})
    thr = thr_of(eol)
    teol = _teol(thr)
    rows = np.asarray(rows, dtype=int)

    out = {}
    # --- Current (9가지 변경 4번): 사이클 6 용량 상위 P*n개를 차례로, 나머지 미사용
    obj10 = _scorer(N_DIAG, thr).objective_for(rows)
    current = S.sorted_start(C[rows, cycle6_column(cycles)], pack_size)
    pv_cur = _pack_values(obj10, current)

    # --- Oracle: n=10 (Current 포함), n=0
    for n in (N_DIAG, 0):
        obj = _scorer(n, thr).objective_for(rows)
        rng = np.random.default_rng([size, seed, pack_size, eol, n, cfg.seed])
        starts = _starts(rows, pack_size, teol, rng, cfg.n_random_starts,
                         current=current if n == N_DIAG else None)
        res = S.best_of(starts, obj, pack_size, cfg)
        out[n] = dict(res=res, pv=_pack_values(obj, res.labels),
                      ub=_scorer(n, thr).upper_bound(rows, pack_size))

    unused = lambda lab: json.dumps(ids[rows[lab == P.UNUSED]].tolist())
    r10, r0 = out[N_DIAG], out[0]
    row = dict(
        chemistry="NMC", variant=_G["variant"], inventory_size=size, subset_seed=seed,
        pack_size=pack_size, eol_pct=eol, search_seed=cfg.seed,
        n_packs=len(rows) // pack_size, n_unused=len(rows) % pack_size,
        V_Current=float(sum(pv_cur)), V_O10=r10["res"].value, V_O0=r0["res"].value,
        UB10=r10["ub"], UB0=r0["ub"],
        worst_Current=float(min(pv_cur)), worst_O10=float(min(r10["pv"])),
        worst_O0=float(min(r0["pv"])),
        best_start_O10=r10["res"].best_kind, best_start_O0=r0["res"].best_kind,
        unused_Current=unused(current), unused_O10=unused(r10["res"].labels),
        unused_O0=unused(r0["res"].labels),
        seconds=time.perf_counter() - t0,
    )
    log = []
    for n in (N_DIAG, 0):
        for lr in out[n]["res"].log_rows():
            log.append({**{k: row[k] for k in ("variant", "inventory_size", "subset_seed",
                                              "pack_size", "eol_pct", "search_seed")},
                        "n_i": n, **lr})

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, rows=rows, current=current,
        cand10=np.stack([s.labels for s in r10["res"].starts]),
        cand0=np.stack([s.labels for s in r0["res"].starts]),
        kinds10=np.array([s.kind for s in r10["res"].starts]),
        kinds0=np.array([s.kind for s in r0["res"].starts]),
        row=json.dumps(row), log=json.dumps(log))
    return str(path), row["seconds"]


def _run_task_star(args):
    return run_task(*args)


def run_tasks(tasks, variant, workers):
    todo = [t for t in tasks if not task_path(variant, t[0], t[1], t[3], t[4],
                                              t[5] if len(t) > 5 else None).exists()]
    print(f"[{variant}] 전체 {len(tasks)}개 중 남은 과제 {len(todo)}개, 작업자 {workers}개", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    done = 0
    # Windows와 같은 방식(spawn)으로 작업 프로세스를 만든다: 어느 컴퓨터에서나 같은 동작
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(variant,),
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [ex.submit(_run_task_star, t) for t in todo]
        for f in as_completed(futs):
            f.result()                        # 오류가 나면 여기서 멈춘다
            done += 1
            if done % max(1, len(todo) // 50) == 0 or done == len(todo):
                el = time.perf_counter() - t0
                eta = el / done * (len(todo) - done)
                print(f"  {done}/{len(todo)}  경과 {el/60:.1f}분  남은 예상 {eta/60:.1f}분", flush=True)


# ============================================================ 6단계: 기준선 갱신

def finalize(variant="raw"):
    """같은 조건 키의 후보 배정을 모두 모아 다시 채점 -> 최고를 최종 V_O로, G를 일괄 계산."""
    files = sorted((OUT / "assignments" / variant).glob("*.npz"))
    if not files:
        raise SystemExit(f"[{variant}] 결과 파일이 없습니다. 먼저 run 을 실행하세요.")
    C, cycles, ids = load_curves(variant)
    scorers = {}

    def sc(n, thr):
        if (n, thr) not in scorers:
            scorers[(n, thr)] = P.PackScorer(C, n, thr)
        return scorers[(n, thr)]

    # 1) 모든 파일을 읽어 조건 키별로 후보 배정을 모은다 (C 행 번호로 바꿔 저장)
    recs, pool, logs = [], {}, []
    for f in files:
        z = np.load(f)
        row = json.loads(str(z["row"]))
        rows = z["rows"]
        key_base = (tuple(rows.tolist()), row["pack_size"], row["eol_pct"])
        for n, cand in ((N_DIAG, z["cand10"]), (0, z["cand0"])):
            bucket = pool.setdefault(key_base + (n,), {})
            for lab in cand:
                packs = tuple(sorted(tuple(sorted(rows[p].tolist())) for p in P.packs_from_labels(lab)))
                bucket[packs] = None                       # 같은 배정은 한 번만
        recs.append((row, key_base))
        logs += json.loads(str(z["log"]))

    # 2) 키마다 후보 전부를 다시 채점해 최고를 고른다 (실행 순서와 무관)
    best = {}
    for key, bucket in pool.items():
        rows_t, ps, eol, n = key
        s = sc(n, thr_of(eol))
        scored = [(sum(s.pack(list(p)) for p in packs), packs) for packs in bucket]
        v, packs = max(scored, key=lambda x: x[0])
        best[key] = (v, min(s.pack(list(p)) for p in packs),
                     json.dumps(sorted(set(ids[list(rows_t)].tolist())
                                       - set(ids[[c for p in packs for c in p]].tolist()))))

    # 3) 인스턴스 표: 최종 V_O로 G를 다시 계산
    out_rows = []
    for row, kb in recs:
        for n, tag in ((N_DIAG, "O10"), (0, "O0")):
            v, worst, unused = best[kb + (n,)]
            row[f"V_{tag}_search"] = row[f"V_{tag}"]
            row[f"refreshed_{tag}"] = v > row[f"V_{tag}"] + 1e-9
            row[f"V_{tag}"], row[f"worst_{tag}"], row[f"unused_{tag}"] = v, worst, unused
        row["G_info"] = (row["V_O10"] - row["V_Current"]) / row["V_Current"]
        row["G_total"] = (row["V_O0"] - row["V_Current"]) / row["V_Current"]
        row["UB_gap_O10"] = (row["UB10"] - row["V_O10"]) / row["UB10"]
        row["UB_gap_O0"] = (row["UB0"] - row["V_O0"]) / row["UB0"]
        row["worst_gain_O10"] = row["worst_O10"] - row["worst_Current"]
        out_rows.append(row)
    inst = pd.DataFrame(out_rows)
    main = inst[inst.search_seed == S.SearchConfig.from_yaml(CONFIG_PATH).seed]
    seedvar = inst[inst.inventory_size == C.shape[0]]

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "" if variant == "raw" else f"_{variant}"
    main.to_csv(OUT / f"g1_instances{sfx}.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / f"search_log{sfx}.csv", index=False)
    gate = make_gate(main, seedvar)
    gate.to_csv(OUT / f"g1_gate{sfx}.csv", index=False)
    if variant == "raw":
        figure_b(main)
    write_repro(variant, sfx)
    print(f"[{variant}] 인스턴스 {len(main)}줄, 기준선이 갱신된 경우 "
          f"O10 {int(main.refreshed_O10.sum())} / O0 {int(main.refreshed_O0.sum())}")
    return main, gate


def make_gate(inst, seedvar):
    q = lambda s, p: float(np.quantile(s, p))
    rows = []
    for (chem, var, size, ps, eol), g in inst.groupby(
            ["chemistry", "variant", "inventory_size", "pack_size", "eol_pct"]):
        r = dict(chemistry=chem, variant=var, inventory_size=size, pack_size=ps, eol_pct=eol,
                 n_instances=len(g))
        for col in ("G_info", "G_total"):
            r[f"{col}_mean"], r[f"{col}_p05"] = g[col].mean(), q(g[col], 0.05)
            r[f"{col}_median"], r[f"{col}_p95"] = q(g[col], 0.5), q(g[col], 0.95)
        r["worst_gain_mean"], r["worst_gain_p05"] = g.worst_gain_O10.mean(), q(g.worst_gain_O10, 0.05)
        r["UB_gap_O10_mean"], r["UB_gap_O10_max"] = g.UB_gap_O10.mean(), g.UB_gap_O10.max()
        r["UB_gap_O0_mean"], r["UB_gap_O0_max"] = g.UB_gap_O0.mean(), g.UB_gap_O0.max()
        sv = seedvar[(seedvar.pack_size == ps) & (seedvar.eol_pct == eol) & (seedvar.variant == var)]
        if size == seedvar.inventory_size.max() and len(sv) > 1:
            r["seedvar_n"] = len(sv)
            r["seedvar_V_O10_min"], r["seedvar_V_O10_max"] = sv.V_O10_search.min(), sv.V_O10_search.max()
            r["seedvar_G_info_min"] = ((sv.V_O10_search - sv.V_Current) / sv.V_Current).min()
            r["seedvar_G_info_max"] = ((sv.V_O10_search - sv.V_Current) / sv.V_Current).max()
        r["passes_g1"] = (bool(r["G_info_p05"] > 0.10)
                          if (eol == 60 and size == 64 and var == "raw") else None)
        rows.append(r)
    return pd.DataFrame(rows)


def figure_b(inst, size=64, eol=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = inst[(inst.inventory_size == size) & (inst.eol_pct == eol)]
    if g.empty:
        print("Figure B: 64셀 EOL 60% 결과가 아직 없어 건너뜀")
        return
    agg = g.groupby("pack_size").agg(
        gi_med=("G_info", "median"), gi_lo=("G_info", lambda s: s.quantile(0.05)),
        gi_hi=("G_info", lambda s: s.quantile(0.95)), gt_med=("G_total", "median")).reset_index()
    x = agg.pack_size.values
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.fill_between(x, 100 * agg.gi_lo, 100 * agg.gi_hi, color="#2a78d6", alpha=0.18,
                    label="G_info p05-p95")
    ax.plot(x, 100 * agg.gi_med, "o-", color="#2a78d6", lw=2, label="G_info median (NMC)")
    ax.plot(x, 100 * agg.gt_med, "o--", color="#2a78d6", lw=1.2, alpha=0.7, label="G_total median")
    ax.axhline(10, color="#52514e", lw=1, ls=":", label="10% (G1 criterion on p05)")
    ax.set_xticks(x)
    ax.set_xlabel("pack size (cells in series)")
    ax.set_ylabel("gain over current practice (%)")
    ax.set_title(f"Figure B - value of future information ({size}-cell inventories, EOL {eol}%)",
                 fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_b.png")
    plt.close(fig)


def write_repro(variant, sfx):
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs"],
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    info = dict(variant=variant, code_commit=commit, code_dirty=dirty,
                data_sha={n: sha(PROCESSED_DIR / n) for n in ("C_full.npz", "C_full_excl.npz")},
                search_config=S.SearchConfig.from_yaml(CONFIG_PATH).to_dict(),
                eol_thresholds_ah={e: thr_of(e) for e in EOL_PCTS},
                curve_handling={"raw": "측정값 그대로", "monotone": "누적 최솟값 단조 곡선",
                                "excl": "C_full_excl (저자 제외점 보간)"}[variant],
                instances=f"32/64셀 시드 0~{N_INSTANCES - 1}, 86셀 1개 (instances.csv)",
                unused_rule="방식마다 미사용 셀을 스스로 정함 (9가지 변경 7번)",
                written_at=datetime.now().isoformat(timespec="seconds"))
    (OUT / f"g1_repro{sfx}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                             encoding="utf-8")


# ============================================================ 명령줄

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "seedvar", "finalize"])
    ap.add_argument("--variant", choices=VARIANTS, default="raw")
    ap.add_argument("--inv", type=int, nargs="*", default=list(INVENTORY_SIZES))
    ap.add_argument("--pack", type=int, nargs="*", default=list(PACK_SIZES))
    ap.add_argument("--eol", type=int, nargs="*", default=list(EOL_PCTS))
    ap.add_argument("--n-instances", type=int, default=N_INSTANCES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    if a.step == "finalize":
        finalize(a.variant)
        return

    C, cycles, ids = load_curves(a.variant)
    for e in a.eol:   # 2절: 미도달 셀 수를 먼저 확인
        n_cens = int(np.isinf(t_eol_from_C(C, cycles, thr_of(e))).sum())
        print(f"EOL {e}% ({thr_of(e):.2f} Ah): 미도달 셀 {n_cens}개")

    if a.step == "run":
        inst = make_instances(C.shape[0], a.n_instances)
        save_instance_list(inst, ids)
        inst = [x for x in inst if x[0] in a.inv]
        tasks = [(s, k, r, p, e) for s, k, r in inst for e in a.eol for p in a.pack]
    else:   # seedvar: 86셀, 탐색 시드 5개 (시드 0은 run 결과와 같은 파일)
        rows = np.arange(C.shape[0])
        tasks = [(C.shape[0], 0, rows, p, e, (None if ss == 0 else ss))
                 for ss in SEEDVAR_SEEDS for e in a.eol for p in a.pack]
    run_tasks(tasks, a.variant, a.workers)


if __name__ == "__main__":
    main()
