"""컨디션: baseline 대비 delta 특성 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, pick_sample_pitchers, save_debug_sample

logger = logging.getLogger(__name__)


def attach_deltas(
    pitches: pd.DataFrame,
    baseline: pd.DataFrame,
    use_cols: list[str],
    prefix: str = "delta_",
) -> pd.DataFrame:
    key = ["pitcher"]
    if "pitch_type" in baseline.columns and "pitch_type" in pitches.columns:
        key = ["pitcher", "pitch_type"]
    rename = {c: f"{c}_baseline" for c in use_cols if c in baseline.columns}
    b2 = baseline.rename(columns=rename)
    m = pitches.merge(b2, on=key, how="left")
    for c in use_cols:
        bl = f"{c}_baseline"
        if bl in m.columns and c in m.columns:
            m[f"{prefix}{c}"] = pd.to_numeric(m[c], errors="coerce") - pd.to_numeric(
                m[bl], errors="coerce"
            )
    return m


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    use_cols = list(config["condition_features"]["use_columns"])
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_FILTERED)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_FILTERED)
    st_b = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_BASELINE)
    rp_b = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_BASELINE)
    st_d = attach_deltas(st_in, st_b, use_cols)
    rp_d = attach_deltas(rp_in, rp_b, use_cols)

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "condition_features",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_d, rp_d], ignore_index=True),
    )
    save_debug_sample(paths, "condition_features", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input")
    save_debug_sample(paths, "condition_features", df=pd.concat([st_d, rp_d], ignore_index=True), suffix="output")
    st_d.to_parquet(paths.interim_dir / A.STARTER_PITCH_DELTA, index=False)
    rp_d.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_DELTA, index=False)
    logger.info("delta 저장: starter=%s행 reliever=%s행", len(st_d), len(rp_d))

    # delta = raw - baseline 검증 (샘플 5행)
    def validate_delta(raw_df: pd.DataFrame, delta_df: pd.DataFrame, *, role: str) -> None:
        if raw_df.empty or delta_df.empty or "pitcher" not in raw_df.columns:
            return
        pids = pick_sample_pitchers(raw_df, n=1)
        if not pids:
            return
        pid = pids[0]
        sample = delta_df[delta_df["pitcher"] == pid].head(5).copy()
        if sample.empty:
            return

        max_abs_err = 0.0
        for c in use_cols:
            c_base = f"{c}_baseline"
            dcol = f"delta_{c}"
            if c in sample.columns and c_base in sample.columns and dcol in sample.columns:
                r = pd.to_numeric(sample[c], errors="coerce")
                b = pd.to_numeric(sample[c_base], errors="coerce")
                d = pd.to_numeric(sample[dcol], errors="coerce")
                err = (d - (r - b)).abs()
                emax = float(err.max(skipna=True)) if len(err) else 0.0
                max_abs_err = max(max_abs_err, emax)
                sample[f"delta_check_{c}"] = err

        logger.info("condition_features delta_check[%s]: max_abs_error=%s", role, max_abs_err)
        sample.to_parquet(
            paths.output_tables_dir / f"debug_condition_features_delta_sample_{role}_pid_{pid}.parquet",
            index=False,
        )

    validate_delta(st_in, st_d, role="starter")
    validate_delta(rp_in, rp_d, role="reliever")
