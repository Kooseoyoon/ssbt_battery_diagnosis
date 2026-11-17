# SSBT 공모전 — 팩 조립 결정 중심(Decision-Aware) 사용 후 배터리 적응형 진단

2026 SSBT 배터리 아이디어 공모전 제출 주제의 실험 코드 저장소.

제안 내용·배경은 `docs/proposal.md`(제출 신청서), 실험 단계별 설계는
`docs/experiment-design.md`를 참고할 것.

## 환경

- **Colab**: 팀원 공유, 빠른 프로토타이핑용. `notebooks/00_colab_bootstrap.ipynb`로 시작.
- **로컬/서버**: 본인 메인 개발용. `bash setup_local.sh`로 시작.
- 자세한 세팅 순서는 [`docs/SETUP.md`](docs/SETUP.md) 참고.

## 구조

```
src/
  config.py                  # 환경(Colab/로컬) 자동 감지, 데이터 경로 설정
  stage0_preprocess.py       # 데이터 전처리 (Primary/Second_life phase)
  common/
    pack.py                  # 팩 성능 V, 상한 UB, 빠른 채점기 PackScorer (Stage 1·4~7 공용)
    search.py                # 배정 탐색: 정렬/무작위 시작, 교환 탐색, (선택) SA, best_of
  stage1_exhaustive.py       # 작은 문제 전수 검사로 탐색 설정 확인
  stage1_g1.py               # 미래 정보 가치 G1 본 계산 (Current vs Oracle), Figure B. --dataset uconn 으로 LFP
  stage2_features.py         # 진단 특징량 추출
  stage3_gp_predict.py       # Gaussian Process 궤적 예측
  stage4_pack_assembly.py    # 팩 조립 최적화
  stage5_adaptive_diagnosis.py  # 적응형 진단 선택 (DFF/EVOI)
  stage6_stopping_rule.py    # 조기 중단 규칙
  stage7_evaluate.py         # 최종 평가 및 그림
notebooks/
  00_colab_bootstrap.ipynb   # Colab 공통 환경 세팅 노트북
  stage1_colab_run.ipynb     # Stage 1 본 계산을 Colab에서 (결과는 Drive에 저장)
configs/
  search.yaml                # 탐색 설정 (결정 근거: docs/stage1_sa_decision.md)
tests/                       # python -m pytest tests -q
archive/                     # 더 이상 쓰지 않는 옛 코드 (예: 예전 설계의 stage1_g1_gate.py)
data/                        # 원본/가공 데이터 (용량 이슈로 google drive에 보관, git에는 올리지 않음)
results/                     # 실험 결과 CSV/그림
docs/
  SETUP.md                   # GitHub + Google Drive 세팅 가이드
  stage1_common_README.txt   # pack.py / search.py 설명과 수정 내역
  stage1_sa_decision.md      # 탐색 설정 결정 (SA 끔) 근거
  stage1_lfp_rules.md        # LFP(UConn) G1 실행 전 규칙
```

## 빠른 시작 (로컬)

```bash
bash setup_local.sh
source .venv/bin/activate   # 또는 conda activate ssbt-battery
python -m pytest tests -q                              # 테스트
python src/stage1_g1.py run --inv 64 --eol 60 --workers 4   # Stage 1 본 계산 (중단해도 이어서 함)
python src/stage1_g1.py finalize                       # 결과 표와 Figure B (results/stage1/)
python src/stage1_g1.py run --dataset uconn && python src/stage1_g1.py seedvar --dataset uconn \
  && python src/stage1_g1.py finalize --dataset uconn   # LFP G1 -> g1_gate_lfp.csv, figure_b_lfp.png (1분 이내)
```
