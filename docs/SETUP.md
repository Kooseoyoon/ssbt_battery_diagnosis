# 환경 세팅 가이드

역할 분담: **Colab = 팀원 공유/빠른 실험**, **로컬 = 본인 개인 환경에서 개발하고자 하는 경우**.
두 환경 모두 이 GitHub 저장소를 공유하고,
실험에 필요한 원본 데이터는 용량이 크므로 Google Drive에 둔다.

---

## 0. 구조 설명

-  Colab 노트북 코드가 꼬이지 않도록 코드는 Git으로 버전관리 → 회의 후 main에 push하기.

- Colab은 세션이 꺼지면 로컬 디스크가 날아가므로, 데이터는 **Google Drive**에 둬야 매번 다시 받지 않는다.
- 로컬/서버는 디스크가 유지되므로 Drive에 의존할 필요 없이 원본 URL에서 직접 받아도 된다.


## 1. Google Drive 공유 폴더 (데이터용)

- 원본 데이터(Deep Sorting 85셀 데이터셋만 해도 Primary_use_phase 2.69GB + Second_life_phase 1.79GB ≈ 4.5GB)는 GitHub에 올리면 안 된다(용량 제한, 저장소 오염).
- 
1. 팀 공유 드라이브나, 본인 드라이브에 폴더를 만들고 **팀원 전체를 "편집자"로 공유**:
   ```
   SSBT_battery_diagnosis/
     data/
       raw/         # 원본 zip/csv (Primary_use_phase.zip, Second_life_phase.zip 등)
       processed/   # Stage 0 산출물 (cap_raw.parquet, C_full.npz 등) — 선택적으로 Drive에도 캐시
   ```
2. `notebooks/00_colab_bootstrap.ipynb`의 `GDRIVE_DATA_DIR`이 이 경로를 가리키도록 맞춘다(기본값이 이미 `/content/drive/MyDrive/SSBT_battery_diagnosis/data`로 돼 있으니, 본인 드라이브가 아니라 팀 공유 드라이브를 쓴다면 경로만 바꾸면 된다).
3. 원본 데이터셋(Zenodo 14859405, Deep Sorting) 다운로드 후 `raw/`에 업로드. 용량이 크므로 Drive 데스크톱 앱이나 `rclone`으로 올리는 게 브라우저 업로드보다 빠르다.


---

## 3. 로컬/서버 환경 세팅

```bash
cd ssbt-battery-diagnosis
bash setup_local.sh
```

데이터를 저장소 내 `data/`가 아니라 다른 경로(예: Shaperon 서버의 별도 디스크)에 두고 싶다면:

```bash
export SSBT_DATA_DIR=/path/to/data   # ~/.bashrc 등에 추가해두면 편함
```

`src/config.py`가 Colab인지 로컬인지 자동 감지해서 경로를 잡아준다 — 코드에서 경로를 하드코딩하지 않아도 된다.

---

## 4. Colab에서 시작하기

1. `notebooks/00_colab_bootstrap.ipynb`를 Colab에서 열기 (Drive에 올려두고 더블클릭, 또는 GitHub 저장소 URL로 직접 열기: colab.research.google.com → File → Open notebook → GitHub 탭).
2. 셀을 순서대로 실행 (Drive 마운트 → 저장소 클론/풀 → `requirements.txt` 설치 → 경로 확인).
3. 이후 `from stage1_g1_gate import run_grid` 식으로 `src/` 모듈을 그대로 import해서 쓴다.

**주의**: Colab에서 코드를 수정했다면 노트북 안에서 수정하지 말고, 가능하면 `src/` 파일을 고쳐서 커밋 → 로컬에서도 바로 반영되게 한다. 노트북 셀에 임시로 패치 코드를 쓰는 건 디버깅 용도로만.

---

## 5. 다음 단계

- `src/stage0_preprocess.py`의 TODO(zip 파싱 로직)부터 채운다 — 이게 끝나야 이후 단계가 전부 돌아간다.
- 최소 경로(공모전 일정 기준, `claude/experiment-design.md` 참고): Stage 0 → 1 → 3(GP) → 4 → 5(DFF) → 7(Figure A).
