# condition_pipeline

투수 컨디션 저하 신호를 정량화해, 이닝/아웃팅 단위로 `KEEP / WARM_UP / MOUND_VISIT / PULL` 의사결정을 만드는 파이프라인입니다.

## 프로젝트 핵심

- **목표:** 투구 단위 Statcast 데이터를 받아 투수 컨디션과 타구 결과를 결합해 교체 신호를 생성
- **핵심 지표 축:**
  - Condition 축: `delta_release_speed` (baseline 대비 구속 변화)
  - Outcome 축: `rolling_xwoba_10`, `rolling_whiff_rate_10` (없으면 `xwoba`, `is_whiff` 대체)
- **핵심 구조:** pitch 레벨 특성 생성 → inning/outing 집계 → 상태(State_A~D) 부여 → 최종 decision 매핑
- **안정성 포인트:** 집계 단계(`aggregate.py`)에서 의사결정 핵심 피처가 누락되지 않도록 보정/전달

## 데이터 수집 (Statcast)

- **우선순위 1:** `config.yaml`의 `data.file_name`(기본값 `statcast_2025.parquet`)이 **`data/raw/`** 아래에 있으면 그 파일을 그대로 읽는다.
- **우선순위 2:** 해당 raw 파일이 없으면, [SmartPitch](../SmartPitch)와 동일하게 **pybaseball `statcast(start_dt, end_dt)`**로 기간별 Statcast를 가져와 `data/raw/<file_name>`에 저장한 뒤 진행한다.
  - CSV 파싱 오류가 나는 경우를 줄이기 위해 SmartPitch와 같은 **statcast CSV patch**를 적용한다.
  - 기간·분할은 `config.yaml`의 **`data_fetch`**에서 조정한다 (`start_date`, `end_date`, `chunk_days`, `use_cache`).
- **폴백:** `pybaseball`이 설치되어 있지 않으면, 로컬 개발용 **최소 합성 데이터**로 파이프라인만 통과시킨다. 실제 분석에는 `pip install -e .` 등으로 의존성을 맞춘 뒤 다시 실행하는 것을 권장한다.

## 파이프라인 흐름

1. 데이터 수집/정렬 — `collect.py`
2. 선발/불펜 분리 — `split_pitchers.py`
3. 투수 필터링 — `filter_pitchers.py`
4. baseline 생성 — `baseline.py`
5. delta 생성 — `condition_features.py`
6. outcome 생성 — `outcome_features.py`
7. rolling 지표 생성 — `rolling_features.py`
8. 라벨링/진단 — `labeling.py`, `quadrant_analysis.py`, `plots.py`, `threshold_analysis.py`
9. inning/outing 집계 — `aggregate.py`
10. 상태/점수 계산 — `scoring.py`
11. 추세 반영 — `trend.py`
12. 최종 의사결정 — `decision.py`

전체 오케스트레이션 — `pipeline.py`

## 최종 의사결정 로직

- `scoring.py`에서 4분면 상태를 부여
  - `State_A` → `KEEP`
  - `State_B` → `WARM_UP`
  - `State_C` → `MOUND_VISIT`
  - `State_D` → `PULL`
- `decision.py`에서 상태를 최종 decision 문자열로 매핑하고 분포를 `debug_decision_counts.parquet`에 저장

## 디렉터리

- `config/config.yaml` — 경로·하이퍼파라미터·Statcast 수집 기간(`data_fetch`)
- `data/raw`, `interim`, `processed` — 입력·중간·최종 데이터
- `outputs/figures`, `tables`, `logs` — 그림·표·로그
- `src/` — 단계별 로직
- `scripts/run_pipeline.py` — 실행 진입점

## 실행

프로젝트 루트(`condition_pipeline/`)에서:

```bash
pip install -e .
python scripts/run_pipeline.py
```

- **그래프:** `plots` 단계는 `matplotlib`이 있을 때만 생성된다. 설치되어 있지 않으면 해당 단계는 건너뛴다.
- **Statcast 자동 수집:** raw parquet이 없을 때만 네트워크로 가져온다. 이미 `data/raw/statcast_2025.parquet`를 두었다면 재수집하지 않는다.
- **결정 분포 확인:** `outputs/tables/debug_decision_counts.parquet`

## 설정 참고

- `project.season_year`: 기본 수집 기간의 연도 힌트로 쓰일 수 있다 (`data_fetch`에 날짜를 직접 쓰는 것이 우선).
- `data.file_name`: raw parquet 파일명.
- `data_fetch`: Statcast 자동 수집 시 사용할 시작일·종료일·chunk 크기·pybaseball 캐시 여부.
