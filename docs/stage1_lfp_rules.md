# Stage 1 LFP G1 — 실행 전 규칙 (결과 보기 전에 고정, 11/17)

명세서 4절 "LFP(UConn)" 단락과 결정 21에 따라, 측정 간격·진단량 단위를 실행 전에 정한다.
데이터는 서윤 `src/uconn_preprocess.py`가 만든 `data/processed/uconn/C_uconn.npz` (24셀, 2차 수명 0.5C).

| 항목 | 규칙 | 이유 |
|---|---|---|
| 측정 간격 | 5사이클 격자 (2차 수명 사이클 1, 6, 11, …) | 전처리에서 NMC C_full과 같은 격자로 맞춤 → pack.py 그대로 사용 |
| 진단량 n | 10사이클 (Current, V_O(10)) / 0 (V_O(0)) | NMC와 같은 단위 |
| Current 기준 | 사이클 6 용량(열 1) 큰 순, 남는 셀은 미사용 | NMC와 같음 (9가지 변경 4번) |
| EOL | 60% = 0.72Ah (주), 55% = 0.66Ah, 65% = 0.78Ah (민감도) | 정격 1.2Ah 기준. 24셀 모두 55%까지 도달 확인 (미도달 0) |
| 재고 | 24셀 전체 1개 | 셀이 적어 부분집합 1,000개는 거의 같은 재고가 반복됨 → NMC 86셀과 같은 방식 |
| 팩 크기 | 4, 8, 12직렬 | 16직렬은 1팩 + 미사용 8셀이라 제외 |
| 탐색 | configs/search.yaml 그대로 (swap + 시작점 22개, SA 끔) | NMC와 같음 |
| 시드 변동 | 탐색 시드 0~4 | NMC 86셀과 같음 |
| 곡선 처리 | raw(주), monotone(민감도) | excl은 NMC 전용 파일이라 해당 없음 |

보고 규칙
- p05 > 10% 판정은 하지 않는다 (재고 1개라 분포 없음, G1 판정은 NMC 64셀 기준). LFP는 보조 증거.
- NMC와 숫자를 직접 비교하지 않고 따로 보고한다 (결정 21).
- 같은 G 그룹(1차 수명 조건이 같은 셀)끼리 닮아 24셀이 완전히 독립은 아님을 해석에 적는다.

출력 (NMC 결과를 덮어쓰지 않음)
- `results/stage1/assignments/uconn_raw/` 등 — 과제 파일
- `results/stage1/g1_instances_lfp.csv`, `g1_gate_lfp.csv`, `search_log_lfp.csv`, `g1_repro_lfp.json`
- `results/stage1/figure_b_lfp.png` — Figure B(NMC 3줄)에 LFP 줄을 더한 새 그림. 기존 `figure_b.png`는 그대로
