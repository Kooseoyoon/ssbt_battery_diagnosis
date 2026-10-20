# SSBT 공모전 — 팩 조립 결정 중심(Decision-Aware) 사용 후 배터리 적응형 진단

2026 SSBT 배터리 아이디어 공모전 제출 주제의 실험 코드 저장소.

제안 내용·배경은 `docs/proposal.md`(제출 신청서), 실험 단계별 설계는
`docs/experiment-design.md`를 참고. (둘 다 claude.ai 프로젝트 문서에서 복사해 넣을 것.)

## 환경

- **Colab**: 팀원 공유, 빠른 프로토타이핑용. `notebooks/00_colab_bootstrap.ipynb`로 시작.
- **로컬/서버**: 본인 메인 개발용. `bash setup_local.sh`로 시작.
- 자세한 세팅 순서는 [`docs/SETUP.md`](docs/SETUP.md) 참고.

## 구조

```
src/
  config.py                  # 환경(Colab/로컬) 자동 감지, 데이터 경로 설정
  stage0_preprocess.py       # 데이터 전처리 (Primary/Second_life phase)
  stage1_g1_gate.py          # 개선 여지 정량화 (Oracle gain)
  stage2_features.py         # 진단 특징량 추출
  stage3_gp_predict.py       # Gaussian Process 궤적 예측
  stage4_pack_assembly.py    # 팩 조립 최적화
  stage5_adaptive_diagnosis.py  # 적응형 진단 선택 (DFF/EVOI)
  stage6_stopping_rule.py    # 조기 중단 규칙
  stage7_evaluate.py         # 최종 평가 및 그림
notebooks/
  00_colab_bootstrap.ipynb   # Colab 공통 환경 세팅 노트북
data/                        # 원본/가공 데이터 (git에는 올리지 않음, .gitignore 처리)
results/                     # 실험 결과 CSV/그림
docs/
  SETUP.md                   # GitHub + Google Drive 세팅 가이드
```

## 빠른 시작 (로컬)

```bash
bash setup_local.sh
source .venv/bin/activate   # 또는 conda activate ssbt-battery
python src/stage1_g1_gate.py   # Stage 0 데이터가 준비된 뒤에 실행
```
