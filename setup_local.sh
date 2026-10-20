#!/usr/bin/env bash
# 로컬/서버용 환경 세팅 스크립트. Colab 노트북(00_colab_bootstrap.ipynb)과
# 동일한 requirements.txt를 써서 두 환경의 패키지 버전을 맞춘다.
#
# 사용법:
#   bash setup_local.sh
set -euo pipefail

ENV_NAME="ssbt-battery"
PYTHON_VERSION="3.11"

if command -v conda >/dev/null 2>&1; then
    echo "[conda] 가상환경 생성: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
else
    echo "[venv] 가상환경 생성: .venv"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "설치 완료. 데이터 경로를 로컬 data/ 폴더가 아닌 다른 곳에 두려면"
echo "export SSBT_DATA_DIR=/path/to/data 를 셸 프로필에 추가하세요."
echo
python src/config.py
