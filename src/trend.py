"""trend: 최근 이닝/최근 구 기반 단순 추세 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def add_starter_inning_trend(df: pd.DataFrame, recent_n: int) -> pd.DataFrame:
    """투수별 경기·이닝 순으로 최근 n이닝 평균 스코어 대비."""
    # TODO: 경기 간 일정 간격 가중
    out = df.copy()
    if out.empty or "condition_score" not in out.columns:
        out["trend_delta"] = np.nan
        return out
    out = out.sort_values(["pitcher", "game_pk", "inning"])
    g = out.groupby("pitcher", group_keys=False)
    roll = g["condition_score"].transform(
        lambda s: s.rolling(recent_n, min_periods=1).mean()
    )
    overall = g["condition_score"].transform("mean")
    out["trend_delta"] = roll - overall
    return out


def add_reliever_pitch_trend_placeholder(df: pd.DataFrame, recent_pitches: int) -> pd.DataFrame:
    """outing 단위에서는 최근 구 수 기반 추세는 pitch 레벨과 결합할 때 확장."""
    # TODO: RELIEVER recent_pitches와 pitch_count 연동
    out = df.copy()
    if out.empty:
        return out
    out["trend_delta"] = 0.0
    _ = recent_pitches
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    tr = config["trend"]
    st_in = pd.read_parquet(paths.processed_dir / A.STARTER_INNING_SCORE)
    rp_in = pd.read_parquet(paths.processed_dir / A.RELIEVER_OUTING_SCORE)
    st_t = add_starter_inning_trend(st_in, int(tr["starter_recent_innings"]))
    rp_t = add_reliever_pitch_trend_placeholder(rp_in, int(tr["reliever_recent_pitches"]))

    log_step_io(
        "trend",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_t, rp_t], ignore_index=True),
    )
    save_debug_sample(paths, "trend", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input_scored")
    save_debug_sample(paths, "trend", df=pd.concat([st_t, rp_t], ignore_index=True), suffix="output_trend")

    # trend_direction 분포 저장
    def direction_counts(df: pd.DataFrame, *, role: str) -> pd.DataFrame:
        if df.empty or "trend_delta" not in df.columns:
            return pd.DataFrame()
        td = pd.to_numeric(df["trend_delta"], errors="coerce")
        direction = np.select([td > 0, td < 0], ["up", "down"], default="flat")
        return (
            pd.DataFrame({"role": role, "trend_direction": direction})
            .value_counts(subset=["role", "trend_direction"])
            .rename("n")
            .reset_index()
        )

    dc = pd.concat(
        [
            direction_counts(st_t, role="starter"),
            direction_counts(rp_t, role="reliever"),
        ],
        ignore_index=True,
    )
    if not dc.empty:
        dc.to_parquet(paths.output_tables_dir / "debug_trend_direction_distribution.parquet", index=False)

    st_t.to_parquet(paths.processed_dir / A.STARTER_INNING_TREND, index=False)
    rp_t.to_parquet(paths.processed_dir / A.RELIEVER_OUTING_TREND, index=False)
    logger.info("trend 저장: starter=%s reliever=%s", len(st_t), len(rp_t))
