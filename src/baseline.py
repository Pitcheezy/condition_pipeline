"""baseline: 시즌 평균(mean) 기준 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, pick_sample_pitchers, save_debug_sample

logger = logging.getLogger(__name__)


def load_filtered(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_FILTERED)
    rp = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_FILTERED)
    return st, rp


def compute_season_baseline(
    df: pd.DataFrame,
    use_cols: list[str],
    by_pitch_type: bool,
) -> pd.DataFrame:
    """투수(및 옵션으로 구종)별 시즌 평균."""
    # TODO: median/가중평균 등 method 분기
    gcols = ["pitcher"]
    if by_pitch_type and "pitch_type" in df.columns:
        gcols.append("pitch_type")
    for c in use_cols:
        if c not in df.columns:
            df[c] = pd.NA
    agg = df.groupby(gcols, dropna=False)[use_cols].mean().reset_index()
    return agg


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    st_in, rp_in = load_filtered(paths)
    bc = config["baseline"]
    use_cols = list(config["condition_features"]["use_columns"])
    by_pt = bool(bc.get("by_pitch_type", False))
    st_b = compute_season_baseline(st_in, use_cols, by_pt)
    rp_b = compute_season_baseline(rp_in, use_cols, by_pt)

    # 공통 IO 로그/샘플 저장 (baseline는 game_pk가 없을 수 있어 unique_games는 0)
    log_step_io("baseline", pd.concat([st_in, rp_in], ignore_index=True), pd.concat([st_b, rp_b], ignore_index=True))
    save_debug_sample(paths, "baseline", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input")
    save_debug_sample(paths, "baseline", df=pd.concat([st_b, rp_b], ignore_index=True), suffix="output")

    st_b.to_parquet(paths.interim_dir / A.STARTER_PITCH_BASELINE, index=False)
    rp_b.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_BASELINE, index=False)
    logger.info("baseline 저장: starter=%s행 reliever=%s행", len(st_b), len(rp_b))

    # 특정 pitcher 샘플에 대해 raw vs baseline 저장 (5행)
    for role, raw_df, base_df, base_out_name in [
        ("starter", st_in, st_b, "debug_baseline_starter_sample_pid_"),
        ("reliever", rp_in, rp_b, "debug_baseline_reliever_sample_pid_"),
    ]:
        sample_pids = pick_sample_pitchers(raw_df, n=1)
        if not sample_pids:
            continue
        pid = sample_pids[0]
        sample_raw = raw_df[raw_df["pitcher"] == pid].head(5).copy()

        key = ["pitcher"]
        if by_pt and "pitch_type" in sample_raw.columns and "pitch_type" in base_df.columns:
            key.append("pitch_type")

        base_ren = base_df.copy()
        for c in use_cols:
            if c in base_ren.columns:
                base_ren = base_ren.rename(columns={c: f"{c}_baseline"})

        baseline_cols = [f"{c}_baseline" for c in use_cols if f"{c}_baseline" in base_ren.columns]
        keep_cols = [*key, *baseline_cols]
        merged = sample_raw.merge(base_ren[keep_cols], on=key, how="left")

        # delta = raw - baseline (샘플에 한해 재계산)
        for c in use_cols:
            c_base = f"{c}_baseline"
            dcol = f"delta_{c}"
            if c in merged.columns and c_base in merged.columns:
                merged[dcol] = pd.to_numeric(merged[c], errors="coerce") - pd.to_numeric(merged[c_base], errors="coerce")

        merged.to_parquet(
            paths.output_tables_dir / f"{base_out_name}{pid}.parquet",
            index=False,
        )
