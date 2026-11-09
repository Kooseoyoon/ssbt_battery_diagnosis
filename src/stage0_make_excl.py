"""Stage 0 보조 — 민감도용 용량 행렬 C_full_excl.npz를 만든다.

주 결과는 C_full.npz(측정값 그대로)를 쓴다. 이 파일은 "이상치 처리에 따라 결과가 바뀌는가"를 보는 민감도용이다.

C_full에서 바꾸는 점 (나머지는 C_full과 똑같다)
  1. Deep Sorting 저자가 수동 제외한 측정점 131개(52셀)
     출처: 저자 코드 3_capa_trajectory_extraction.py의 제외 목록 (Zenodo 15350607)
     저자 목록의 키는 CSV 파일 번호다. Battery_ID가 아니다 → ds_file_to_battery_id.csv로 변환
  2. Battery_ID 82의 191사이클 (2.00Ah, 앞뒤 1.84·1.88Ah). 11/9 실행 결과 저자 목록의 파일 88번 191사이클과
     같은 점이었다(파일 88 = Battery_ID 82). 그래서 실제로 바뀌는 점은 131개. 확인용으로 목록에 남겨 둔다
  바꾸는 방법: 같은 셀에서 바꾸지 않는 가장 가까운 앞·뒤 유효값을 사이클 번호 기준으로 선형 보간.
             뒤쪽 유효값이 없으면(셀의 마지막 측정점) 보간하지 않고 NaN으로 둔다(외삽 안 함).

T_eol: C_full_excl로 stage0_preprocess.py와 같은 규칙(1.44Ah 첫 미만)으로 다시 계산해 T_eol_excl.npy로 저장한다.
T_eol.npy(주 결과)는 건드리지 않는다. 민감도 분석에서는 C_full_excl과 T_eol_excl을 한 쌍으로 쓴다.
11/9 실행: Battery_ID 3(파일 31)만 286 → 281. 저자가 뺀 281사이클(1.477Ah)을 276·286 사이로 보간하면
1.414Ah가 되어 1.44 아래로 내려가기 때문이다.

실행 (저장소 루트):
    python src/stage0_make_excl.py
입력:  data/processed/C_full.npz, T_eol.npy, ds_file_to_battery_id.csv
출력:  data/processed/C_full_excl.npz   C_full과 같은 키(C_full, cell_ids, cycles, source) + changed_mask
       data/processed/T_eol_excl.npy      C_full_excl 기준 T_eol (셀 순서 = cell_ids)
       data/processed/C_full_excl_log.csv  바꾼 점마다: battery_id, file_num, cycle, 원래 값, 새 값, 출처, 처리 결과
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR  # noqa: E402

EOL_AH = 1.44   # 2.4Ah × 60%, stage0_preprocess.py와 같은 기준

# 저자 제외 목록 (키 = 저자 bat_num = CSV 파일 번호). check_ds_exclusions.py와 같은 목록
AUTHOR_EXCLUDED = {
    3: [221, 226], 5: [161, 166, 171], 10: [116], 13: [261, 266], 14: [266, 271, 276, 281],
    15: [256, 261, 266, 271], 16: [266], 17: [81, 86], 18: [101, 106, 111, 116, 216],
    19: [76, 81, 86, 236, 241], 20: [86, 91, 176, 196, 241, 246], 21: [821, 826, 831, 836, 1221, 1511],
    22: [736, 741, 746, 751, 756, 761, 766, 811, 816, 1086], 23: [116, 721, 726, 731, 736, 741, 1106, 1111],
    24: [946, 951, 956], 27: [316], 29: [156, 161, 166, 171, 176], 30: [216, 221, 226], 31: [281],
    32: [231, 236], 37: [36, 41, 46], 40: [166, 171], 41: [206], 42: [196, 201], 43: [16, 266], 44: [16],
    47: [56], 48: [56, 61], 51: [216, 221], 52: [216, 221], 53: [201], 54: [201], 55: [136], 56: [201],
    59: [151, 201], 60: [241], 63: [106, 111], 64: [96, 101], 68: [276], 69: [346, 601], 70: [316, 561],
    73: [106], 74: [226, 336, 491, 496], 77: [11, 511, 516], 79: [56, 61, 66, 446], 80: [56, 61, 66, 446],
    83: [396], 84: [41], 85: [261], 86: [276], 87: [146, 151], 88: [191],
}
# 저자 목록 밖에서 추가로 바꾸는 점 (키 = Battery_ID)
EXTRA_BY_BATTERY_ID = {82: [191]}


def compute_t_eol(c, cycles, thr):
    """stage0_preprocess.compute_t_eol과 같은 규칙."""
    t = np.zeros(c.shape[0])
    for i in range(c.shape[0]):
        below = np.where(c[i] < thr)[0]
        if below.size:
            t[i] = cycles[below[0]]
        else:
            obs = np.where(~np.isnan(c[i]))[0]
            t[i] = cycles[obs[-1]] if obs.size else np.nan
    return t


def main():
    n_author = sum(map(len, AUTHOR_EXCLUDED.values()))
    assert n_author == 131 and len(AUTHOR_EXCLUDED) == 52, "저자 목록 개수가 131개/52셀이 아니다"

    d = np.load(PROCESSED_DIR / "C_full.npz")
    C, cycles, ids = d["C_full"].astype(float), d["cycles"], [int(x) for x in d["cell_ids"]]
    t_eol = np.load(PROCESSED_DIR / "T_eol.npy")
    mp = pd.read_csv(PROCESSED_DIR / "ds_file_to_battery_id.csv")
    f2b = dict(zip(mp["file_num"].astype(int), mp["battery_id"].astype(int)))
    b2f = {b: f for f, b in f2b.items()}

    # 바꿀 점 목록: (battery_id, cycle) → 출처
    targets: dict[tuple[int, int], str] = {}
    for fnum, cs in AUTHOR_EXCLUDED.items():
        if fnum not in f2b:
            raise SystemExit(f"파일 번호 {fnum}이 ds_file_to_battery_id.csv에 없다")
        for c in cs:
            targets[(f2b[fnum], c)] = "author"
    for bid, cs in EXTRA_BY_BATTERY_ID.items():
        for c in cs:
            key = (bid, c)
            targets[key] = "author+extra" if key in targets else "extra"

    col = {int(c): j for j, c in enumerate(cycles)}
    mask = np.zeros_like(C, dtype=bool)
    rows = []
    for (bid, c), src in sorted(targets.items()):
        r = {"battery_id": bid, "file_num": b2f.get(bid), "cycle": c, "source": src,
             "orig_Ah": np.nan, "new_Ah": np.nan}
        if bid not in ids:
            rows.append({**r, "status": "셀 없음(C_full에 없는 Battery_ID)"}); continue
        if c not in col:
            rows.append({**r, "status": "사이클 없음(RPT 격자 밖)"}); continue
        i, j = ids.index(bid), col[c]
        r["orig_Ah"] = C[i, j]
        if np.isnan(C[i, j]):
            rows.append({**r, "status": "원래 NaN(바꿀 것 없음)"}); continue
        mask[i, j] = True
        rows.append({**r, "status": "대상"})

    out = C.copy()
    out[mask] = np.nan            # 먼저 다 지운 뒤, 바꾸지 않는 점만으로 보간
    keep = ~np.isnan(C) & ~mask
    for r in rows:
        if r["status"] != "대상":
            continue
        i, j = ids.index(r["battery_id"]), col[r["cycle"]]
        left = np.where(keep[i, :j])[0]
        right = np.where(keep[i, j + 1:])[0] + j + 1
        if left.size and right.size:
            a, b = left[-1], right[0]
            x0, x1, x = cycles[a], cycles[b], cycles[j]
            out[i, j] = C[i, a] + (C[i, b] - C[i, a]) * (x - x0) / (x1 - x0)
            r["status"] = f"보간({int(x0)}~{int(x1)})"
        else:
            r["status"] = "NaN(앞 또는 뒤 유효값 없음, 외삽 안 함)"
        r["new_Ah"] = out[i, j]

    log = pd.DataFrame(rows)
    np.savez(PROCESSED_DIR / "C_full_excl.npz", C_full=out, cell_ids=d["cell_ids"], cycles=cycles,
             source=str(d["source"]) + "+excl", changed_mask=mask)
    log.to_csv(PROCESSED_DIR / "C_full_excl_log.csv", index=False, encoding="utf-8-sig")
    t2 = compute_t_eol(out, cycles, EOL_AH)
    np.save(PROCESSED_DIR / "T_eol_excl.npy", t2)

    # 요약·점검
    diff = np.where(t2 != t_eol)[0]
    same_elsewhere = np.array_equal(np.nan_to_num(out[~mask], nan=-1), np.nan_to_num(C[~mask], nan=-1))
    print(f"바꿀 점 {len(targets)}개(저자 {n_author} + 추가 {sum(map(len, EXTRA_BY_BATTERY_ID.values()))}, "
          f"겹침 {sum(v == 'author+extra' for v in targets.values())})")
    print(log["status"].str.replace(r"\(.*\)", "", regex=True).value_counts().to_string())
    done = log[log.status.str.startswith("보간")]
    if len(done):
        delta = done.new_Ah - done.orig_Ah
        print(f"보간 전후 차이(새 − 원래, Ah): 중앙값 {delta.median():+.4f}, 최소 {delta.min():+.4f}, "
              f"최대 {delta.max():+.4f}, 아래로 바뀐 점 {(delta < 0).mean():.0%}")
    print(f"바꾸지 않은 점이 C_full과 완전히 같은가: {same_elsewhere}")
    print(f"T_eol_excl이 T_eol.npy와 다른 셀: {len(diff)}개"
          + ("" if not len(diff) else " → " + ", ".join(f"{ids[i]}({int(t_eol[i])}→{int(t2[i])})" for i in diff)))
    print(f"저장: {PROCESSED_DIR}/ C_full_excl.npz, T_eol_excl.npy, C_full_excl_log.csv")


if __name__ == "__main__":
    main()
