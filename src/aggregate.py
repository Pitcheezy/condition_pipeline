"""inning(선발) / outing(불펜) 단위 집계."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def mean_numeric_features(df: pd.DataFrame) -> list[str]:
    """집계 평균 대상 숫자형·파생 컬럼."""
    # TODO: 설정으로 명시적 컬럼 리스트 관리
    out = []
    for c in df.columns:
        if c in ("starter_inning_unit_id", "reliever_outing_unit_id"):
            continue
        if c.startswith("delta_") or c.startswith("rolling_") or c == "z_aggregate":
            out.append(c)
        elif c in ("is_whiff", "is_strike", "is_hard_hit", "xwoba"):
            out.append(c)
    return [c for c in out if c in df.columns]


def aggregate_starter_inning(df: pd.DataFrame) -> pd.DataFrame:
    gid = "starter_inning_unit_id"
    if gid not in df.columns or df.empty:
        return pd.DataFrame()
    num_cols = mean_numeric_features(df)
    gcols = [gid, "pitcher", "game_pk", "inning"]
    g = df.groupby(gcols, dropna=False)
    counts = g.size().reset_index(name="pitch_count")
    if num_cols:
        means = g[num_cols].mean().reset_index()
        out = counts.merge(means, on=gcols, how="left")
    else:
        out = counts
    return out


def aggregate_reliever_outing(df: pd.DataFrame) -> pd.DataFrame:
    gid = "reliever_outing_unit_id"
    if gid not in df.columns or df.empty:
        return pd.DataFrame()
    num_cols = mean_numeric_features(df)
    gcols = [gid, "pitcher", "game_pk"]
    g = df.groupby(gcols, dropna=False)
    counts = g.size().reset_index(name="pitch_count")
    if num_cols:
        means = g[num_cols].mean().reset_index()
        out = counts.merge(means, on=gcols, how="left")
    else:
        out = counts
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    _ = config
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_THRESHOLD)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_THRESHOLD)
    st_i = aggregate_starter_inning(st_in)
    rp_o = aggregate_reliever_outing(rp_in)

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "aggregate",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_i, rp_o], ignore_index=True),
    )
    save_debug_sample(paths, "aggregate", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input_threshold")
    save_debug_sample(paths, "aggregate", df=pd.concat([st_i, rp_o], ignore_index=True), suffix="output_aggregated")
    st_i.to_parquet(paths.processed_dir / A.STARTER_INNING_AGG, index=False)
    rp_o.to_parquet(paths.processed_dir / A.RELIEVER_OUTING_AGG, index=False)
    logger.info("집계 저장: starter_inning=%s outing=%s", len(st_i), len(rp_o))

    # 단위 row count 확인
    unit_rows = []
    if not st_i.empty and "starter_inning_unit_id" in st_i.columns:
        unit_rows.append(
            {
                "role": "starter",
                "rows": int(len(st_i)),
                "unique_unit_ids": int(st_i["starter_inning_unit_id"].nunique(dropna=True)),
                "unique_pitchers": int(st_i["pitcher"].nunique(dropna=True)) if "pitcher" in st_i.columns else 0,
                "unique_games": int(st_i["game_pk"].nunique(dropna=True)) if "game_pk" in st_i.columns else 0,
            }
        )
    if not rp_o.empty and "reliever_outing_unit_id" in rp_o.columns:
        unit_rows.append(
            {
                "role": "reliever",
                "rows": int(len(rp_o)),
                "unique_unit_ids": int(rp_o["reliever_outing_unit_id"].nunique(dropna=True)),
                "unique_pitchers": int(rp_o["pitcher"].nunique(dropna=True)) if "pitcher" in rp_o.columns else 0,
                "unique_games": int(rp_o["game_pk"].nunique(dropna=True)) if "game_pk" in rp_o.columns else 0,
            }
        )
    if unit_rows:
        pd.DataFrame(unit_rows).to_parquet(paths.output_tables_dir / "debug_aggregate_unit_counts.parquet", index=False)
