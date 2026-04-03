"""단계별 산출물 파일명 (starter / reliever / 단위 구분)."""

from __future__ import annotations

# 공통 전처리
PITCH_SORTED = "pitch_sorted.parquet"

# starter — pitch grain
STARTER_PITCH_SPLIT = "starter_pitch_split.parquet"
STARTER_PITCH_FILTERED = "starter_pitch_filtered.parquet"
STARTER_PITCH_BASELINE = "starter_pitch_baseline.parquet"
STARTER_PITCH_DELTA = "starter_pitch_delta.parquet"
STARTER_PITCH_OUTCOMES = "starter_pitch_outcomes.parquet"
STARTER_PITCH_ROLLING = "starter_pitch_rolling.parquet"
STARTER_PITCH_LABELED = "starter_pitch_labeled.parquet"
STARTER_PITCH_QUADRANT = "starter_pitch_quadrant.parquet"
STARTER_PITCH_THRESHOLD = "starter_pitch_threshold.parquet"

# starter — inning grain
STARTER_INNING_AGG = "starter_inning_aggregated.parquet"
STARTER_INNING_SCORE = "starter_inning_scored.parquet"
STARTER_INNING_TREND = "starter_inning_trend.parquet"
STARTER_INNING_DECISION = "starter_inning_decision.parquet"

# reliever — pitch grain
RELIEVER_PITCH_SPLIT = "reliever_pitch_split.parquet"
RELIEVER_PITCH_FILTERED = "reliever_pitch_filtered.parquet"
RELIEVER_PITCH_BASELINE = "reliever_pitch_baseline.parquet"
RELIEVER_PITCH_DELTA = "reliever_pitch_delta.parquet"
RELIEVER_PITCH_OUTCOMES = "reliever_pitch_outcomes.parquet"
RELIEVER_PITCH_ROLLING = "reliever_pitch_rolling.parquet"
RELIEVER_PITCH_LABELED = "reliever_pitch_labeled.parquet"
RELIEVER_PITCH_QUADRANT = "reliever_pitch_quadrant.parquet"
RELIEVER_PITCH_THRESHOLD = "reliever_pitch_threshold.parquet"

# reliever — outing grain
RELIEVER_OUTING_AGG = "reliever_outing_aggregated.parquet"
RELIEVER_OUTING_SCORE = "reliever_outing_scored.parquet"
RELIEVER_OUTING_TREND = "reliever_outing_trend.parquet"
RELIEVER_OUTING_DECISION = "reliever_outing_decision.parquet"

# 표·로그
TABLE_QUADRANT_SUMMARY = "quadrant_summary.parquet"
LOG_PIPELINE_RUN = "pipeline_run.log"
