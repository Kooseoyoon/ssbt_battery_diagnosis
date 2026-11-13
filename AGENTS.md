# 배터리 연구 프로젝트 — Codex 작업 기준

이 지침은 이 저장소 전체에 적용한다. 기본 응답은 한국어다.

## 시작할 때
- `simulation/START_HERE.txt`, `simulation/configs/experiment_plan_v3.json`부터 읽는다.
- 현재 V3 구현은 `simulation/battery_sim/`이다. `src/`의 Stage 뼈대와 혼동하지 않는다.
- `docs/simulation_handoff/research_record.tex`와 `experiment_v3_design.tex`에 연구 판단·실행 범위가 있다.
- ChatGPT 프로젝트 안의 이전 simulation 사본은 이관 당시 스냅샷이며, 이 저장소가 이후 코드의 기준이다.

## 실행
- V3 환경은 `simulation/.venv`에 둔다. 기존 conda 환경을 변경하지 않는다.
- 저장소 루트에서 `uv run --directory simulation python -m battery_sim.experiment_v3 --stage preflight`.
- 실제 용량 대조군은 같은 명령의 `--stage controls`이다.
- 검증은 `uv run --directory simulation python -m unittest discover -s tests -q`.
- 새 컴퓨터에서는 `uv sync --project simulation --frozen`으로 환경을 재현한다.

## 데이터와 평가
- 원본 `data/`와 제공받은 특징 파일은 읽기 전용으로 취급하고 덮어쓰지 않는다.
- 전체 재고 N=86, 미사용 셀 진단 비용 포함, 셀당 10~40사이클·5사이클 추가.
- 공통 비용: 860,1290,1720,2150,2580,3010,3440 셀-사이클.
- n_i/5 미만 열만 관측 공개; 실제 미래·EOL은 평가기에만 전달한다.
- V는 첫 팩 최소 용량 <1.44Ah 이전의 최소 용량 합 ×5, 단위 Ah·사이클.
- 팩 크기 4,8,12,16; 미사용은 배정 변수이며 팩↔미사용 교환을 허용한다.
- Current-only는 전 셀 n=10, 사이클6 용량 내림차순, 하위 나머지 셀 미사용.
- `capacity_updated`는 각 예산의 최신 공개 용량 정렬로, Current-only와 별도다.
- 공유 fold는 `data/processed/folds.csv`; 현재 임시본, 공식본 우선. 셀의 모든 시점은 같은 fold다.
- 결측 처리·표준화·특징 선정·튜닝은 학습 셀에서만 수행한다.
- 86셀 OOF 팩 결과를 독립 86셀 재고의 일반화 평가로 주장하지 않는다.
- 정확한 oracle 절차 미정. 근사 탐색값·완화 상한을 정확한 oracle로 표시하지 않는다.

## 현재 미완료와 다음 작업
- `features_v2.npz`와 `results/stage2_select/final_selection.json`은 아직 미수신이다.
- 특징 배열 키는 제안 형식이다. 실제 파일 확인 없이 축·셀 ID·선정 목록을 추정하지 않는다.
- C_full만 쓰는 n=10·8직렬 OOF 비교에 단조성·매끄러움 제약 NN과 제곱근+선형/포화 지수+선형 ODE 잔차 PINN을 구현·실행했다. PINN 결과는 독립 검증이 아니며 Current-only 대비 V 향상도 확인되지 않았다.
- Stage2 학습 비교 실행기, GP+DFF 적응 진단, 정확한 regret은 미구현/미실행이다.
- 용량 특징 모델과 Stage2 모델의 같은 fold·모델 후보·튜닝·정렬 배정 비교를 먼저 완성한다.
- DFF는 실험용 동료 교체 비율이지 보상이나 진단 가치의 보장이 아니다.

## 결과와 기록
- 설계·구현·실행·검증 완료를 구분하고 없는 결과를 생성하지 않는다.
- 주요 변경마다 설정, 입력 해시, 실제 실행 결과, 해석과 다음 작업을 남긴다.
- 인계자료는 시도와 판단을 기록한다. 대화 원문이나 긴 코드를 그대로 붙이지 않는다.
- 결과는 `simulation/results/`, 보존할 기준 결과는 `simulation/reference_results/`에 둔다.
- 데이터·비밀키·캐시·Python 환경을 Git에 추가하지 않는다. 원격 게시·메시지 전송은 별도 사용자 지시를 따른다.
