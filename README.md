# condition_pipeline

투수 컨디션 분석 파이프라인. 단계는 아래 순서로 연결한다.

## 데이터 수집 (Statcast)

- **우선순위 1:** `config.yaml`의 `data.file_name`(기본값 `statcast_2025.parquet`)이 **`data/raw/`** 아래에 있으면 그 파일을 그대로 읽는다.
- **우선순위 2:** 해당 raw 파일이 없으면, [SmartPitch](../SmartPitch)와 동일하게 **pybaseball `statcast(start_dt, end_dt)`**로 기간별 Statcast를 가져와 `data/raw/<file_name>`에 저장한 뒤 진행한다.
  - CSV 파싱 오류가 나는 경우를 줄이기 위해 SmartPitch와 같은 **statcast CSV patch**를 적용한다.
  - 기간·분할은 `config.yaml`의 **`data_fetch`**에서 조정한다 (`start_date`, `end_date`, `chunk_days`, `use_cache`).
- **폴백:** `pybaseball`이 설치되어 있지 않으면, 로컬 개발용 **최소 합성 데이터**로 파이프라인만 통과시킨다. 실제 분석에는 `pip install -e .` 등으로 의존성을 맞춘 뒤 다시 실행하는 것을 권장한다.

## 파이프라인 흐름

1. 데이터 수집 — `collect.py`
2. 선발/불펜 분리 — `split_pitchers.py`
3. 기준 필터링 — `filter_pitchers.py`
4. baseline 생성 — `baseline.py`
5. delta 계산(컨디션) — `condition_features.py`
6. 결과 지표 생성 — `outcome_features.py`
7. rolling 집계 — `rolling_features.py`
8. good/bad 기준 생성 — `labeling.py`
9. 4분면 분석 — `quadrant_analysis.py`
10. 그래프·threshold 분석 — `plots.py`, `threshold_analysis.py`
11. inning/outing 집계 — `aggregate.py`
12. score 계산 — `scoring.py`
13. trend 분석 — `trend.py`
14. 교체 판단 — `decision.py`

전체 오케스트레이션 — `pipeline.py`

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

## 설정 참고

- `project.season_year`: 기본 수집 기간의 연도 힌트로 쓰일 수 있다 (`data_fetch`에 날짜를 직접 쓰는 것이 우선).
- `data.file_name`: raw parquet 파일명.
- `data_fetch`: Statcast 자동 수집 시 사용할 시작일·종료일·chunk 크기·pybaseball 캐시 여부.
