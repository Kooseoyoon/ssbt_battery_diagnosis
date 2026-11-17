"""UConn LFP 1단계 — 셀마다 사이클별 방전 용량 궤적을 뽑고, G1에 쓸 수 있는 셀인지 판정한다.

결정 20·21: UConn은 G1에만 쓴다. cycling 파일의 사이클별 방전 용량으로 정격(1.2Ah)의 80% → 60% 구간.
이 스크립트는 G1을 계산하지 않는다. "몇 셀이 쓸 수 있는가"만 본다(G1 값은 보지 않음 → 이후 규칙을 정해도 편향 없음).

읽는 곳: data/raw/external/uconn/ 아래
  - 내려받은 zip을 그대로 둬도 된다(압축을 풀지 않고 zip 안의 CSV를 바로 읽는다 → 디스크 절약)
  - 이미 풀어 둔 폴더도 읽는다
  셀 = 'cycling N.csv'가 들어 있는 폴더(G#C#). 같은 셀이 zip과 폴더에 둘 다 있으면 한 번만 쓴다.

셀마다 하는 일
  1. cycling 파일을 첫 기록 시각(Date) 순으로 정렬해 이어 붙이고, 누적 사이클 번호를 매긴다
  2. 사이클마다: 방전 용량 = 전류 < −0.05A 기록의 Capacity(Ah) 절댓값 최댓값, 방전 전류 중앙값, Phase
  3. 판정(정격 1.2Ah 기준): 처음 0.96Ah(80%) 미만이 된 누적 사이클, 처음 0.72Ah(60%) 미만이 된 누적 사이클,
     그 사이 길이. 튀는 값 하나로 판정이 바뀌지 않게 '연속 3사이클 미만'을 기준으로 한다
  4. 2차 수명 구간(Phase에 'second' 포함)의 시작·끝 용량과 방전 전류(0.5C = 0.6A 근처인지)

실행 (저장소 루트):
    python src/uconn_scan.py
    python src/uconn_scan.py --root /다른/경로
출력: data/processed/uconn/
    uconn_cycles.parquet   셀 × 누적 사이클: cell, source, file_k, cycle_in_file, cum_cycle, phase, cap_Ah, i_dchg_med
    uconn_scan.csv         셀별 요약과 판정(usable_g1)
    uconn_traj.png         셀별 궤적(0.96·0.72Ah 선)
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

NOMINAL = 1.2
TH80, TH60 = 0.8 * NOMINAL, 0.6 * NOMINAL
RUN = 3                     # 연속 RUN 사이클이 기준 미만일 때 '도달'
LR_START_MIN = 0.90          # 0.5C 구간 시작 용량이 이 이상(정격 75%)이어야 80%대에서 시작한 것으로 본다
CELL_RE = re.compile(r"G\d+C\d+", re.I)
CYC_RE = re.compile(r"cycling\D*(\d+)\.csv$", re.I)


def find_col(cols, *keys):
    for c in cols:
        if all(k.lower() in str(c).lower() for k in keys):
            return c
    return None


# ------------------------------------------------------------------ 파일 찾기
def list_cells(root: Path):
    """{셀 이름: [(파일 번호, 열기 함수, 출처 문자열), ...]}"""
    cells: dict[str, dict[int, tuple]] = {}
    src_of: dict[str, str] = {}

    def add(cell, k, opener, src):
        if cell in src_of and src_of[cell] != src:
            return                         # 다른 출처(zip/폴더)에서 이미 읽은 셀 → 건너뜀
        src_of[cell] = src
        cells.setdefault(cell, {})[k] = (opener, src)

    for z in sorted(root.rglob("*.zip")):
        try:
            zf = zipfile.ZipFile(z)
            names = zf.namelist()
            opener_of = lambda n, zf=zf: (lambda: zf.open(n))  # noqa: E731
        except zipfile.BadZipFile as e:
            # 파이썬 zipfile이 못 여는 zip(ZIP64 헤더 등)이라도 unzip 명령은 여는 경우가 있다 → unzip으로 읽는다
            try:
                names = subprocess.run(["unzip", "-Z1", str(z)], capture_output=True, text=True,
                                       check=True).stdout.splitlines()
            except Exception as e2:  # noqa: BLE001
                print(f"  ! zip을 열 수 없음: {z.name} (zipfile: {e} / unzip: {e2})")
                continue
            print(f"  (파이썬 zipfile 실패 → unzip 명령으로 읽음: {z.name})")
            opener_of = lambda n, z=z: (lambda: io.BytesIO(subprocess.run(  # noqa: E731
                ["unzip", "-p", str(z), n], capture_output=True, check=True).stdout))
        for name in names:
            p = PurePosixPath(name)
            m = CYC_RE.search(p.name)
            cm = [part for part in p.parts if CELL_RE.fullmatch(part)]
            if m and cm:
                add(cm[-1].upper(), int(m.group(1)), opener_of(name), z.name)
    for f in sorted(root.rglob("*.csv")):
        m = CYC_RE.search(f.name)
        cm = [part for part in f.parts if CELL_RE.fullmatch(part)]
        if m and cm:
            add(cm[-1].upper(), int(m.group(1)), (lambda f=f: open(f, "rb")), "folder:" + f.parent.parent.name)
    return {c: sorted(((k,) + v for k, v in d.items()), key=lambda x: x[0]) for c, d in sorted(cells.items())}


# ------------------------------------------------------------------ 파일 하나 → 사이클 요약
def read_cycling(opener):
    with opener() as fh:
        raw = fh.read()
    head = pd.read_csv(io.BytesIO(raw), nrows=2, encoding_errors="replace")  # 일부 파일 헤더에 cp949 문자(° 등)
    c_cyc, c_cur, c_cap = find_col(head.columns, "cycle"), find_col(head.columns, "current"), \
        find_col(head.columns, "capacity")
    c_date, c_ph = find_col(head.columns, "date"), find_col(head.columns, "phase")
    use = [c for c in (c_cyc, c_cur, c_cap, c_date, c_ph) if c]
    df = pd.read_csv(io.BytesIO(raw), usecols=use, low_memory=False, encoding_errors="replace")
    t0 = pd.to_datetime(df[c_date].iloc[0], errors="coerce") if c_date else pd.NaT
    cyc = pd.to_numeric(df[c_cyc], errors="coerce")
    cur = pd.to_numeric(df[c_cur], errors="coerce")
    cap = pd.to_numeric(df[c_cap], errors="coerce").abs()
    d = pd.DataFrame({"cyc": cyc, "cur": cur, "cap": cap,
                      "ph": df[c_ph].astype(str) if c_ph else "nan"}).dropna(subset=["cyc"])
    dis = d[d.cur < -0.05]
    g = dis.groupby("cyc").agg(cap_Ah=("cap", "max"), i_dchg_med=("cur", "median"))
    g["phase"] = d.groupby("cyc")["ph"].agg(lambda s: s.mode().iat[0] if len(s) else "nan")
    return t0, g.reset_index().rename(columns={"cyc": "cycle_in_file"})


def first_run_below(cap, th, run=RUN):
    """cap < th가 run번 연속 처음 성립하는 위치(시작 인덱스). 없으면 None."""
    b = (np.asarray(cap) < th).astype(int)
    if len(b) < run:
        return None
    s = np.convolve(b, np.ones(run, int), "valid")
    i = np.where(s == run)[0]
    return int(i[0]) if i.size else None


def lowrate_stats(df):
    lr = df[df.i_dchg_med.abs().between(0.4, 0.8)]
    cap = lr.cap_Ah.to_numpy()
    j60 = first_run_below(cap, TH60) if len(cap) else None
    ok = bool(len(cap) and cap[0] >= LR_START_MIN and j60 is not None)
    return {"lr_n": len(lr), "lr_start_cap": cap[0] if len(cap) else np.nan,
            "lr_min_cap": cap.min() if len(cap) else np.nan,
            "lr_cyc_to60": (j60 + 1) if j60 is not None else np.nan, "lr_usable": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RAW_DIR / "external" / "uconn"))
    ap.add_argument("--min-cap", type=float, default=0.3, help="이보다 작은 방전 용량 사이클은 버림(중단 사이클)")
    a = ap.parse_args()
    root = Path(a.root)
    out = PROCESSED_DIR / "uconn"
    out.mkdir(parents=True, exist_ok=True)
    cells = list_cells(root)
    print(f"셀 {len(cells)}개 발견 ({root})")
    if not cells:
        return

    all_rows, summ = [], []
    for cell, files in cells.items():
        parts = []
        for k, opener, src in files:
            try:
                t0, g = read_cycling(opener)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {cell} cycling {k} 읽기 실패: {e}")
                continue
            g["file_k"], g["t0"], g["source"] = k, t0, src
            parts.append(g)
        if not parts:
            continue
        df = pd.concat(parts)
        df = df.sort_values(["t0", "file_k", "cycle_in_file"], na_position="last")
        n_drop = int((df.cap_Ah < a.min_cap).sum())
        df = df[df.cap_Ah >= a.min_cap].reset_index(drop=True)
        df["cum_cycle"] = np.arange(1, len(df) + 1)
        df["cell"] = cell
        all_rows.append(df)

        cap = df.cap_Ah.to_numpy()
        i80, i60 = first_run_below(cap, TH80), first_run_below(cap, TH60)
        sl = df[df.phase.str.contains("second", case=False, na=False)]
        c80 = int(df.cum_cycle[i80]) if i80 is not None else np.nan
        c60 = int(df.cum_cycle[i60]) if i60 is not None else np.nan
        sl_i = sl.i_dchg_med.median() if len(sl) else np.nan
        fl = df[~df.phase.str.contains("second", case=False, na=False)]
        # 방전 전류 구간: 전류(0.1A 반올림)가 같은 연속 구간별 사이클 수 → 율속이 바뀌는 지점 확인용
        seg = (df.i_dchg_med.round(1) != df.i_dchg_med.round(1).shift()).cumsum()
        segs = df.groupby(seg).agg(i=("i_dchg_med", "median"), c0=("cum_cycle", "first"),
                                   c1=("cum_cycle", "last"), n=("cum_cycle", "size"))
        segs = segs[segs.n >= 20]
        rate_txt = " / ".join(f"{r.c0:.0f}-{r.c1:.0f}: {r.i:.2f}A" for r in segs.itertuples())
        summ.append({
            "cell": cell, "source": df.source.iloc[0], "n_files": len(files), "n_cycles": len(df),
            "dropped_short": n_drop, "cap_first": cap[0], "cap_min": cap.min(), "cap_last": cap[-1],
            "cyc_below80": c80, "cyc_below60": c60, "len_80_60": c60 - c80,
            "sl_n_cycles": len(sl), "sl_start_cap": sl.cap_Ah.iloc[0] if len(sl) else np.nan,
            "sl_last_cap": sl.cap_Ah.iloc[-1] if len(sl) else np.nan,
            "sl_first_cum_cycle": int(sl.cum_cycle.iloc[0]) if len(sl) else np.nan,
            "sl_i_dchg_med": sl_i, "fl_i_dchg_med": fl.i_dchg_med.median() if len(fl) else np.nan,
            "rate_segments": rate_txt,
            "below80_in_sl": bool(len(sl) and i80 is not None and c80 >= sl.cum_cycle.iloc[0]),
            "usable_g1": bool(i80 is not None and i60 is not None),
            # 0.5C(0.4~0.8A) 방전 사이클만: 2차 수명 표준 조건. Phase 표시와 무관하게 전류로 고른다
            **lowrate_stats(df),
        })
        s = summ[-1]
        print(f"  {cell:8s} 사이클 {s['n_cycles']:5d} | 용량 {s['cap_first']:.3f}→{s['cap_last']:.3f}Ah "
              f"(최저 {s['cap_min']:.3f}) | 80%↓ {c80} 60%↓ {c60} | 2차 수명 {s['sl_n_cycles']}사이클, "
              f"전류 {sl_i:.2f}A")
        print(f"           방전 전류 구간(20사이클 이상): {rate_txt}")
        print(f"           0.5C 구간: {s['lr_n']}사이클, 시작 {s['lr_start_cap']:.3f}Ah, 최저 {s['lr_min_cap']:.3f}Ah, "
              f"0.72Ah 도달 {s['lr_cyc_to60']}사이클째 → {'사용 가능' if s['lr_usable'] else '불가'}")

    cyc = pd.concat(all_rows)
    cols = ["cell", "source", "file_k", "cycle_in_file", "cum_cycle", "phase", "cap_Ah", "i_dchg_med"]
    try:
        cyc[cols].to_parquet(out / "uconn_cycles.parquet", index=False)
    except ImportError:     # pyarrow 없는 환경 → csv로
        cyc[cols].to_csv(out / "uconn_cycles.csv", index=False, encoding="utf-8-sig")
        print("  (pyarrow 없음 → uconn_cycles.csv로 저장)")
    s = pd.DataFrame(summ)
    s.to_csv(out / "uconn_scan.csv", index=False, encoding="utf-8-sig")
    u = s[s.usable_g1]
    print(f"\n[판정] 정격 80%(0.96Ah)와 60%(0.72Ah)를 모두 연속 {RUN}사이클 넘어 내려간 셀: {len(u)}/{len(s)}")
    if len(u):
        print(f"  80→60 길이(사이클): 최소 {u.len_80_60.min():.0f}, 중앙값 {u.len_80_60.median():.0f}, "
              f"최대 {u.len_80_60.max():.0f} | 80% 아래가 2차 수명 안에서 시작: {int(u.below80_in_sl.sum())}셀")
    print(f"  60%에 못 미친 셀: {', '.join(s[~s.usable_g1].cell) or '없음'}")
    lu = s[s.lr_usable]
    print(f"\n[UConn 사용 판정 기준 = 0.5C 구간] 시작 ≥ {LR_START_MIN}Ah이고 0.72Ah까지 내려간 셀: "
          f"{len(lu)}/{len(s)} ({', '.join(lu.cell) or '없음'})")
    if len(lu):
        print(f"  0.72Ah 도달까지 0.5C 사이클 수: 최소 {lu.lr_cyc_to60.min():.0f}, 중앙값 {lu.lr_cyc_to60.median():.0f}, "
              f"최대 {lu.lr_cyc_to60.max():.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for c, g in cyc.groupby("cell"):
            ax.plot(g.cum_cycle, g.cap_Ah, lw=0.6, label=c)
        for th in (TH80, TH60):
            ax.axhline(th, color="k", ls="--", lw=0.8)
        ax.set_xlabel("cumulative cycle")
        ax.set_ylabel("discharge capacity (Ah)")
        ax.set_title("UConn LFP: per-cycle discharge capacity (dashed: 0.96 / 0.72 Ah)")
        if cyc.cell.nunique() <= 20:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "uconn_traj.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("그림 저장 실패:", e)
    print(f"저장: {out}/")


if __name__ == "__main__":
    main()
