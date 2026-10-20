"""
환경(로컬/서버 vs Colab)에 따라 데이터 경로를 자동으로 잡아주는 설정 모듈.

사용법:
    from config import DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR

- Colab에서 실행 중이면 Google Drive가 마운트돼 있다고 가정하고
  GDRIVE_DATA_DIR 환경변수(또는 기본 경로)를 사용한다.
- 로컬/서버에서 실행 중이면 저장소 루트 기준 data/ 폴더를 사용한다.
- 두 경우 모두 환경변수 SSBT_DATA_DIR로 강제 지정할 수 있다.
"""

import os
from pathlib import Path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# 저장소 루트 (이 파일 기준 한 단계 위)
REPO_ROOT = Path(__file__).resolve().parent.parent

# 팀 공유 Google Drive 폴더 경로 (Colab 전용 기본값).
# 실제 공유 드라이브 폴더를 만든 뒤 이 경로를 맞춰서 바꿔두면 된다.
DEFAULT_COLAB_DATA_DIR = "/content/drive/MyDrive/SSBT_battery_diagnosis/data"

if os.environ.get("SSBT_DATA_DIR"):
    DATA_DIR = Path(os.environ["SSBT_DATA_DIR"])
elif _in_colab():
    DATA_DIR = Path(os.environ.get("GDRIVE_DATA_DIR", DEFAULT_COLAB_DATA_DIR))
else:
    DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"IN_COLAB      = {_in_colab()}")
    print(f"REPO_ROOT     = {REPO_ROOT}")
    print(f"DATA_DIR      = {DATA_DIR}")
    print(f"RAW_DIR       = {RAW_DIR}")
    print(f"PROCESSED_DIR = {PROCESSED_DIR}")
    print(f"RESULTS_DIR   = {RESULTS_DIR}")
