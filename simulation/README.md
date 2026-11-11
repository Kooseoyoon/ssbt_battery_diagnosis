활성 코드 기준은 저장소의 `simulation/` 폴더다. 저장소 루트에서 `battery-research.code-workspace`를 연다.
저장소 루트의 `battery-research.code-workspace`를 열고 `simulation/START_HERE.txt`부터 읽는다.
활성 V3 설정의 입력 경로는 simulation 기준 상대 경로이며, 공유 fold는 저장소 루트 data/processed에 있다.
과거 ChatGPT 폴더의 simulation 사본은 이관 당시 스냅샷이다.

# 배터리 적응형 진단 시뮬레이션

## V3 개선 실험: Stage 2 특징 → 실제 팩 가치

VS Code에서 `V3. 새 실험 입력·준비 상태 확인` 또는 `V3. 실제 용량 기준선 7개 예산 실행`을 선택한다.
기본 빌드 작업은 입력 점검이다. 실험 설계는 `configs/experiment_plan_v3.json`에 있다.

```bash
uv run --directory simulation python -m battery_sim.experiment_v3 --stage preflight
uv run --directory simulation python -m battery_sim.experiment_v3 --stage controls
```

핵심 비교는 초기 10사이클에서 Current-only, 용량 특징 모델, Stage 2 screened20 모델의 실제 V이다.
용량 특징은 기존 E2-B의 10개 특징을 두 RPT에서도 계산하도록 수정한 V3 정의이며 과거 실행의 재현은 아니다.
RUL 목표는 미래 평가 시작 사이클 n+1부터의 잔여 수명으로 통일한다.
각 모델 계열별 및 동일 내부 검증으로 선택한 결과를 함께 보고한다. 먼저 정렬 배정을 고정해 특징 효과를 비교한다.
RUL 예측으로 배정한 뒤 실제 미래 용량으로 V를 평가하므로 1차 비교에 용량 궤적 예측기는 필수가 아니다.

**실행된 범위:** 입력 점검, 28개 균등 진단·최신 용량 정렬 및 초기 Current-only 4개 조건.
**아직 미실행:** Stage 2 학습 비교, nested 학습 실행기, GP와 연결한 DFF 적응 진단, 정확한 oracle regret.
최신 `origin/main`에는 `features_v2.npz`와 `final_selection.json`이 추가되어 있다. 배열은 `X`(86×7×148), `cell_ids`, `budgets`, `feature_names`, `base_names`이며 입력 셀 ID는 `C_full.npz`와 일치한다. 선정 JSON의 `screened`는 20개, `features`는 6개다. V3 설정은 `budgets`를 진단량 키로 사용한다.
새 입력이 V3의 실제 관측 공개 범위와 선정 근거를 만족하는지 preflight 및 분석으로 확인해야 하며, 파일 존재만으로 학습 비교가 완료된 것은 아니다.
결과는 `results/experiment_design_v3/`의 `readiness.json`, `control_metrics.csv`, `control_assignments.json`에 저장한다.

Current-only는 n=10, 비용 860의 고정 비교점이다. `capacity_updated`는 각 예산에서 최신 용량으로 다시 정렬하는 별도 대조군이다.
전체 86셀 OOF 배정은 탐색용이다. 서로 다른 fold의 학습에 다른 셀 정답이 들어가므로 독립 86셀 팩 일반화 평가로 주장하지 않는다.
공유 5-fold의 셀별 예측 오차와 17~18셀 fold 내 배정 결과는 별도로 보고한다. screened20이 평가 셀 정답으로 선정됐다면 fold 안에서 재선정한다.

## 현재 규약: 팀 피드백에 따른 V 지표

현재 기준은 `configs/evaluation_contract_v3.json`이다. 팩 가치는 RUL 최솟값의 합이 아니라
EOL 전까지의 최소 용량 누적값 `V`(Ah·cycles)다. `battery_sim/value_metrics.py`에 이 평가 코어를 구현했다.
진단량 n=10이면 사이클 1·6만 공개하고, 미래 평가는 0-based 열 2(사이클 11)부터 시작한다.
5사이클씩 추가하고 셀당 상한은 40이다. N은 미사용 후보를 포함한 재고 전체 86으로 고정한다. 모든 방법은 86셀 전체를 진단 대상으로 둔다.
배정은 floor(86/m)개 팩과 미사용 셀로 구성하며, 미사용 셀의 진단 비용도 포함한다.
Current-only는 사이클 6 용량 하위 셀을 미사용으로 남긴다. 팩↔팩과 팩↔미사용 교환을 지원한다.

주 지표는 `(V_O - V_method)/V_O`이고, 진단 손실과 배정 손실도 같은 `V_O(n=0)`로 나누어 분해한다.
공통 진단량은 860, 1290, 1720, 2150, 2580, 3010, 3440 셀-사이클이다.
라운드는 9셀, 한 구간은 기본 9셀×9라운드+5셀×1라운드다. 상한 도달로 후보가 부족하면 라운드를 축소한다.
대규모 oracle 계산 방법은 아직 `null`이다. 미사용 셀의 재고 가치·폐기 비용은 V에 포함하지 않는다.

DFF 첫 실험안은 두 배정에서 팩 동료가 교체된 비율이다. 모두 사용이면
`1 - |동료집합 교집합|/(m-1)`, 한쪽만 미사용이면 1, 모두 미사용이면 0을 시나리오 평균한다.
팩 번호 재명명에 영향을 받지 않는다. DFF가 높다고 추가 진단의 V 개선이 보장되지는 않는다.
`battery_sim/inventory_protocol.py`는 예산 및 DFF 코어이며 GP 적응 진단 연결은 후속 작업이다.
수명 정렬을 V의 정확한 oracle이라고 부르거나, 근사해를 전역 상한처럼 사용하지 않는다.

```bash
uv run python -m battery_sim.value_contract_smoke
uv run python -m unittest discover -s tests -v
```

이는 Current-only의 새 V 계산 확인이며, 새 기준의 GP·적응 검사 전체 실행 결과가 아니다.
결과는 `results/value_contract_v3/`에 저장한다. 균등 선택의 7개 예산·60개 라운드 장부도 함께 검증한다.
이전 무작위 사전 제외 결과는 `results/value_contract_v3_random_exclusion_superseded/`에 보관했다. 예전 V1/V2 숫자는 아래의 구지표 실험으로 보존하며 새 regret으로 환산하지 않는다.

`data/processed/folds.csv`의 최신 공유 버전은 저장소 `origin/main`에 있다. 이전 로컬 작업의 시드 42 임시 fold와 `simulation/data/processed/` 복사본은 이를 대체하지 않으며 이번 공유 변경에는 포함하지 않는다. 각 셀의 모든 관측은 같은 fold에 속한다.
기존 V2의 2-fold 실행을 5-fold 실행으로 소급 표기하지 않는다. 5-fold 평가군은 17~18셀이므로
12S/16S는 한 팩만 남는다. 예측 평가 fold와 여러 팩의 배정을 비교할 평가 후보군의 설계를 구분해야 한다.

## 연구를 이어받는 사람이 먼저 볼 문서

프로젝트의 `outputs/대화_문답기록/research_record.tex`와 `연구진행_인계자료.pdf`가 연구 진행·인계 기준 문서다.
연구 질문, 데이터·평가 규칙, 시도 E1/E2의 목적·변경·결과·판단, 재현 절차와 다음 작업을 연결한다.
이후 기록도 대화 원문을 누적하지 않고 실험 단위로 이 문서에 갱신한다. 긴 코드는 소스로 연결한다.
새로운 시도는 이유·고정 조건·변경점·실행 설정·결과·판단·다음 작업을 남긴다.

## V2: 실제 셀 분리 모델 비교

```bash
uv run --directory simulation python -m battery_sim.compare_models
uv run --directory simulation python -m battery_sim.audit_comparison
```

`model_comparison_v2.ipynb` 또는 F5의 **V2. 셀 분리 학습·보정·팩 구성 비교**로 실행한다.
변경 내역은 `CHANGES_V2.txt`, 설정은 `configs/model_comparison_v2.json`에 있다.
결과는 `results/model_comparison_v2/`에 저장한다. 원본 데이터와 V1 결과는 유지한다.

개발 43셀 중 32셀을 학습에, 11셀을 예측구간 보정에 사용하고, 별도 43셀을 평가한다.
내부 교차검증에서 모델 설정과 Ridge/GP 중 사용할 종류를 정한다. 평가 정답으로 선택하지 않는다.
각 평가에서는 8셀 팩 5개와 미사용 후보 3개를 사용하므로, 비교는 같은 평가 집합의 기준 모델과 한다.
외부 2-fold와 분할 시드 3개를 반복하여 모든 셀이 각 시드에서 한 번씩 평가에 들어간다.
같은 86셀의 반복 분할이므로 독립 데이터셋의 일반화 신뢰구간이라고 해석하지 않는다.

실험은 예측모델과 불확실성 보정, 팩 구성을 분리해서 비교한다. 새 적응형 검사·중단 정책을
평가한 결과는 아니다. 모든 평가 셀은 같은 양의 관측을 받는다. 기존 국소 선형모델, 전체 관측
선형모델, Ridge와 GP, raw/calibrated 구간, 정렬/시나리오 셀 교환을 비교한다.
보정은 80% 구간에 대한 것이며 시나리오 꼬리나 셀 간 상관의 정확성까지 검증한 것은 아니다.

`metrics.csv`와 `predictions.csv`는 원시 결과, `summary.csv`는 평균과 분할 간 산포,
`comparison_deltas.csv`는 같은 조건의 변경 전후 차이, `fitted_model_log.json`은 개발 데이터에서
선택한 설정과 보정 배율, `splits.json`은 셀별 역할이다. `manifest.json`은 입력과 코드의 해시,
`audit.json`은 결과 재계산 및 데이터 분리 확인 결과를 남긴다.

## 실제 86셀 실험

`real86_start.ipynb`를 열거나 F5의 **0. 실제 86셀 진단 시뮬레이션**을 선택한다.

```bash
uv run --directory simulation python -m battery_sim.real86
```

설정은 `configs/real86.json`, 결과는 `results/real86_pilot_v1/`에 저장한다. 데이터 경로는 저장소의 `data/processed/`를 기준으로 해석한다. 외부 폴더의 데이터를 사용할 때만 `--data-dir` 또는 설정을 바꾼다. 원본 파일은 수정하지 않는다.

이 첫 실험은 86개 후보 중 예측 RUL이 긴 80개로 8셀 팩 10개를 구성하고 나머지 6개는 미사용 후보로 남긴다. 같은 정책이라도 새 관측 후 선택한 80개와 배정이 바뀔 수 있다. 미사용 후보의 잔여 가치나 폐기 비용은 목적함수에 넣지 않았다.

모든 셀의 첫 RPT 5개(사이클 1, 6, 11, 16, 21)를 공개하고 시작한다. 한 번의 검사는 선택한 셀의 다음 RPT를 공개하며 5 cell-cycles를 사용한다. 초기 공통 비용은 1,806 cell-cycles이고, 그래프의 예산은 여기에 더한 추가 비용이다. 검사에 소비한 사이클은 배치 시점의 실제 RUL에서도 차감한다. 공통 완전 관측 구간 중 최대 101사이클까지만 사용한다.

수명모델은 관측된 용량과 사이클만 사용하는 Bayesian linear regression이다. 최근 최대 8개 RPT로 직선을 적합하고 선형 외삽한 EOL에서 현재 사이클을 빼 RUL을 얻는다. 회귀계수의 정규 사후분포를 RUL로 변환하므로 RUL 자체의 분포는 정규분포로 가정하지 않는다. 정밀 측정 오차 0.01 Ah, 사전분포, 기울기 제한과 2,000사이클의 예측 상한은 첫 실행용 가정이며 정답 수명으로 학습하거나 보정하지 않았다. 해당 수치들은 일반화 성능이 검증된 최종 설정이 아니다.

검사 선택 방법은 균등, 무작위, RUL 불확실성 우선, 의사결정 가치 근사다. 마지막 방법은 구성 변화 가능성이 큰 최대 16개 후보를 추린 뒤 다음 용량 관측을 8가지로 가상 생성하고 관측 후 예상 팩 성능을 평가한다. 기대 개선량에서 검사로 소비한 수명의 영향을 반영한다. 후보 제한, 샘플 수, 선형모델, 평균 정렬 구성 규칙에 의존하는 Monte Carlo 근사이며 정확한 EVOI나 학습된 최종 제안 알고리즘이라고 주장하지 않는다. 예산을 모두 사용하는 비교이고 중단 규칙은 아직 없다.

`metrics.csv`는 정책·시드·검사 예산별 성능, `inspection_trace.csv`는 공개한 셀·사이클·용량의 전체 이력, `summary.csv`는 반복 평균과 표준편차다. `manifest.json`에 실행 설정과 원본 5개 파일의 SHA-256을 저장한다. 서로 다른 시드는 같은 86셀의 알고리즘 난수 반복이며 독립 데이터셋 반복이나 일반화 오차 추정이 아니다.

성능점수는 검사 종료 상태의 실제 RUL로 최적 배정했을 때를 100점으로 한다. 후보 86개와 팩 10개라는 제약은 oracle에도 동일하다. 정답 EOL은 평가기에서만 읽고 예측·검사 선택 함수에는 전달하지 않는다. 최종 팩 수명 합계와 초기 oracle 대비 성능도 함께 저장해서 검사로 소비한 수명이 점수 정규화에 가려지지 않도록 한다. 현재 관측 용량만 정렬한 단순 구성의 점수도 별도 비교값으로 저장한다.

실제 팩 내부 셀 간 상호작용, 달력 열화, 검사 장비의 병렬 경과 시간은 모델링하지 않았다. EOL은 제공된 RPT에서 최초 1.44 Ah 미만인 관측 시점이다.

## VS Code에서 시작

1. `battery-simulation.code-workspace`를 VS Code로 연다.
2. `F5`를 누르고 **1. 합성 데이터로 시뮬레이션 시작**을 선택한다.
3. `results/demo_summary.json`과 `figures/demo_pack_lifetimes.png`에서 결과를 확인한다.

노트북을 선호하면 `start_here.ipynb`를 열어 위에서부터 실행해도 된다.

이 예제는 **합성 데이터로 실행 흐름을 확인하는 시작점**이다. 실제 배터리에서 검증한 결과가 아니다. 셀별 수명 예측모델, 추가 검사 선택 및 중단 정책은 이후 구현한다.

## 예제에서 하는 일

- 80셀을 8셀씩 묶고, 셀별 예측 평균을 정렬하여 팩을 구성한다.
- 전체 셀 RUL 튜플을 512개 생성한다. 생성된 튜플마다 같은 팩 구성을 평가한다.
- 숨겨진 합성 정답으로 팩 수명 합계와 regret을 계산한다.
- 각 시나리오를 완전히 알고 재구성하는 비교용 상한은 별도로 계산한다.
- 팩 수명은 구성 셀 RUL의 최솟값이다. 합계의 단위는 pack-cycles다.

예측분포는 실행 예제용으로 임의 생성한 것이며, 실제로 학습하거나 검증한 분포가 아니다. 예측 평균 정렬은 시작용 기준 방법이다. 불확실성이 있는 상황에서 기대 성능을 최적화한다고 주장하지 않는다.

`configs/demo.json`에서 셀 수, 팩 크기, 시나리오 수, 난수 시드를 바꿀 수 있다. 현재는 나머지 셀을 임의로 버리지 않도록 셀 수가 팩 크기의 배수일 때만 실행된다.

## 실제 전처리 파일 연결

다음 파일을 `data/raw/`에 복사한다. 원본을 보존하고, `sources/` 아래 자료는 수정하지 않는다.

```text
cap_raw.parquet
C_full.npz
T_eol.npy
censor_mask.npy
meta.csv
```

VS Code의 실행 메뉴에서 **2. 실제 데이터 파일 구조 확인**을 실행하면 `results/data_inventory.json`에 열 이름, 배열 키·크기, 자료형을 기록한다. 외부 경로를 그대로 읽으려면 아래처럼 실행할 수 있다.

```bash
uv run --directory simulation python -m battery_sim.inspect_data --data-dir "/실제/데이터/폴더"
```

데이터 구조를 확인한 뒤 셀 ID 대응, RUL 기준, 정밀 관측 간격, 검사 비용을 정한다. 수명 정답은 평가용으로만 사용하며 검사 정책에 공개하지 않는다. 구조 확인 도구는 pickle 역직렬화를 사용하지 않는다. object 배열이면 형식을 검토한 후 연결한다.

## 폴더 구성

```text
battery_sim/     팩 성능 계산, 합성 예제, 데이터 구조 확인
configs/         실행 조건
data/raw/        팀원 전처리 파일의 로컬 사본
data/processed/  구조 확인 후 변환한 데이터
results/         수치 결과
figures/         그림
.vscode/         실행·디버깅·Python 환경 설정
.venv/           이 프로젝트의 Python 가상환경
uv.lock          의존성 버전 기록
```

## 터미널 실행과 팀원 환경 재현

Python 3.12와 `uv`가 있는 환경에서는 이 폴더에서 다음을 실행한다.

```bash
uv sync --locked
uv run python -m battery_sim.demo
uv run python -m battery_sim.inspect_data
```

VS Code에서 노트북 커널은 `simulation/.venv` 가상환경을 선택한다. Python 확장은 운영체제에 맞는 실행 파일을 찾는다. 실행 환경에는 NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyArrow, ipykernel을 준비한다. 실제 프로젝트 데이터와 가상환경은 팀원에게 공유하는 소스 코드에 포함하지 않는다.
