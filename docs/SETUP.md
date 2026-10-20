# 환경 세팅 가이드

역할 분담: **Colab = 팀원 공유/빠른 실험**, **로컬 = 본인 메인 개발**.
두 환경 모두 이 GitHub 저장소를 "진실의 소스(single source of truth)"로 공유하고,
무거운 원본 데이터는 저장소 밖(Google Drive)에 둔다.

---

## 0. 왜 이 구조인가

- 코드는 Git으로 버전관리 → 팀원 간 diff·merge가 가능해야 Colab 노트북 코드가 꼬이지 않는다.
- 원본 데이터(Deep Sorting 85셀 데이터셋만 해도 Primary_use_phase 2.69GB + Second_life_phase 1.79GB ≈ 4.5GB)는 GitHub에 올리면 안 된다(용량 제한, 저장소 오염).
- Colab은 세션이 꺼지면 로컬 디스크가 날아가므로, 데이터는 **Google Drive**에 둬야 매번 다시 받지 않는다.
- 로컬/서버는 디스크가 유지되므로 Drive에 의존할 필요 없이 원본 URL에서 직접 받아도 된다.

---

## 1. GitHub 저장소 만들기

지금 이 환경에는 GitHub 계정이 연결돼 있지 않아서, 저장소 생성은 직접 해야 한다.

1. GitHub에서 새 저장소 생성 (예: `ssbt-battery-diagnosis`), private 추천(공모전 진행 중이므로 비공개 아이디어 보호).
2. 이 폴더(`ssbt-battery-diagnosis/`) 전체를 그 저장소에 푸시:
   ```bash
   cd ssbt-battery-diagnosis
   git init
   git add .
   git commit -m "init: 실험 파이프라인 스캐폴드"
   git branch -M main
   git remote add origin https://github.com/<ORG>/<REPO>.git
   git push -u origin main
   ```
3. 팀원 5명 + 지도교수(선택)를 Collaborator로 추가.
4. `notebooks/00_colab_bootstrap.ipynb`의 `REPO_URL`을 실제 주소로 바꿔서 다시 커밋.

---

## 2. Google Drive 공유 폴더 만들기 (데이터용)

1. 팀 공유 드라이브나, 본인 드라이브에 폴더를 만들고 **팀원 전체를 "편집자"로 공유**:
   ```
   SSBT_battery_diagnosis/
     data/
       raw/         # 원본 zip/csv (Primary_use_phase.zip, Second_life_phase.zip 등)
       processed/   # Stage 0 산출물 (cap_raw.parquet, C_full.npz 등) — 선택적으로 Drive에도 캐시
   ```
2. `notebooks/00_colab_bootstrap.ipynb`의 `GDRIVE_DATA_DIR`이 이 경로를 가리키도록 맞춘다(기본값이 이미 `/content/drive/MyDrive/SSBT_battery_diagnosis/data`로 돼 있으니, 본인 드라이브가 아니라 팀 공유 드라이브를 쓴다면 경로만 바꾸면 된다).
3. 원본 데이터셋(Zenodo 14859405, Deep Sorting) 다운로드 후 `raw/`에 업로드. 용량이 크므로 Drive 데스크톱 앱이나 `rclone`으로 올리는 게 브라우저 업로드보다 빠르다.

### 왜 데이터는 Drive에, 코드는 GitHub에 나누는가
- Drive 15GB 무료 용량 기준으로 원본 4.5GB + 향후 추가할 보조 데이터셋(`claude/dataset-urls.md`에 정리된 MATR 180셀, ISU-ILCC 238셀 등, 각 수 GB~10GB)까지 감당하려면 일부만 선별해서 올려야 할 수 있다. 처음엔 Deep Sorting 85셀 데이터만 올리고, Stage 1 이후 외부 데이터셋이 실제로 필요해지면 그때 추가하는 식으로 가면 된다.
- 로컬/서버에서는 Drive를 거치지 않고 원본 URL(`claude/dataset-urls.md` 참고)에서 바로 받는 걸 추천한다(동기화 지연·용량 중복 방지).

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
