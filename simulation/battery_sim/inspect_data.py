"""Read file metadata without guessing array-to-cell mappings or loading pickle."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ("cap_raw.parquet", "C_full.npz", "T_eol.npy", "censor_mask.npy", "meta.csv")


def inspect(path: Path) -> dict:
    if path.suffix == ".parquet":
        file = pq.ParquetFile(path)
        return {"rows": file.metadata.num_rows, "row_groups": file.num_row_groups,
                "schema": str(file.schema_arrow)}
    if path.suffix == ".csv":
        sample = pd.read_csv(path, nrows=5)
        return {"columns": list(sample.columns),
                "sample_dtypes": {str(k): str(v) for k, v in sample.dtypes.items()}}
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return {"shape": list(array.shape), "dtype": str(array.dtype)}
    with np.load(path, allow_pickle=False) as archive:
        entries = {}
        for name in archive.files:
            try:
                array = archive[name]
                entries[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            except ValueError as exc:
                entries[name] = {"error": str(exc), "note": "Object arrays require an explicit data-format review."}
        return {"arrays": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    real_config = ROOT / 'configs/real86.json'
    configured_dir = Path(json.loads(real_config.read_text())['data_dir']) if real_config.exists() else Path('data/raw')
    default_dir = (configured_dir if configured_dir.is_absolute() else ROOT / configured_dir).resolve()
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    try:
        portable_data_dir = data_dir.relative_to(ROOT.parent).as_posix()
    except ValueError:
        portable_data_dir = data_dir.name
    report = {"data_directory": portable_data_dir, "files": {}}
    for name in EXPECTED:
        path = args.data_dir / name
        if not path.is_file():
            report["files"][name] = {"status": "missing"}
            print(f"미배치: {name}")
            continue
        try:
            report["files"][name] = {"status": "found", **inspect(path)}
            print(f"확인 완료: {name}")
        except Exception as exc:
            report["files"][name] = {"status": "error", "error": str(exc)}
            print(f"읽기 확인 필요: {name}: {exc}")
    output = ROOT / "results/data_inventory.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"구조 확인 결과: {output.relative_to(ROOT).as_posix()}")
    print("셀 ID 대응, 배열 축과 RUL 기준은 이 결과를 검토한 뒤 연결합니다.")


if __name__ == "__main__":
    main()
