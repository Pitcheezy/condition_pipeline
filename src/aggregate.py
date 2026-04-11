"""inning(선발) / outing(불펜) 단위 집계."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def ensure_required_features_for_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """집계 전 필수 피처를 보정한다.

    필수 피처:
    - delta_release_speed
    - rolling_xwoba_10 (또는 xwoba 대체)
    - rolling_whiff_rate_10 (또는 is_whiff 대체)
    """
    out = df.copy()
    if out.empty:
        return out

    # 1) delta_release_speed: 없으면 raw-baseline으로 복원
    if "delta_release_speed" not in out.columns:
        if "release_speed" in out.columns and "release_speed_baseline" in out.columns:
            out["delta_release_speed"] = (
                pd.to_numeric(out["release_speed"], errors="coerce")
                - pd.to_numeric(out["release_speed_baseline"], errors="coerce")
            )

    # 2) rolling_xwoba_10: 없으면 xwoba를 대체 입력으로 사용
    if "rolling_xwoba_10" not in out.columns and "xwoba" in out.columns:
        out["rolling_xwoba_10"] = pd.to_numeric(out["xwoba"], errors="coerce")

    # 3) rolling_whiff_rate_10: 없으면 is_whiff를 대체 입력으로 사용
    if "rolling_whiff_rate_10" not in out.columns and "is_whiff" in out.columns:
        out["rolling_whiff_rate_10"] = pd.to_numeric(out["is_whiff"], errors="coerce")

    return out


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
    st_in_raw = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_THRESHOLD)
    rp_in_raw = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_THRESHOLD)
    st_in = ensure_required_features_for_aggregation(st_in_raw)
    rp_in = ensure_required_features_for_aggregation(rp_in_raw)
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

    # 핵심 피처 전달 여부 점검용 디버그 저장
    required_rows = []
    for role, df in (("starter", st_i), ("reliever", rp_o)):
        required_rows.append(
            {
                "role": role,
                "has_delta_release_speed": "delta_release_speed" in df.columns,
                "has_rolling_xwoba_10": "rolling_xwoba_10" in df.columns,
                "has_rolling_whiff_rate_10": "rolling_whiff_rate_10" in df.columns,
                "has_xwoba": "xwoba" in df.columns,
                "has_is_whiff": "is_whiff" in df.columns,
            }
        )
    pd.DataFrame(required_rows).to_parquet(
        paths.output_tables_dir / "debug_aggregate_required_feature_pass_through.parquet",
        index=False,
    )

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
