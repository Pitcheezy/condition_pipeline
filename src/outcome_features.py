"""결과 지표: 타석/투구 결과 파생 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def add_outcome_columns(df: pd.DataFrame, hard_hit_threshold: float) -> pd.DataFrame:
    out = df.copy()
    if "description" in out.columns:
        desc = out["description"].astype(str).str.lower()
        # Statcast: swinging_strike, swinging_strike_blocked 등
        out["is_whiff"] = desc.str.contains("swinging_strike", na=False)
        out["is_strike"] = desc.str.contains("strike", na=False) | desc.str.contains("foul", na=False)
    else:
        out["is_whiff"] = False
        out["is_strike"] = False
    if "release_speed" in out.columns:
        out["is_hard_hit"] = pd.to_numeric(out["release_speed"], errors="coerce") >= hard_hit_threshold
    else:
        out["is_hard_hit"] = False
    if "estimated_woba_using_speedangle" in out.columns:
        out["xwoba"] = pd.to_numeric(out["estimated_woba_using_speedangle"], errors="coerce")
    else:
        out["xwoba"] = np.nan
    return out


def assign_outcome_quadrant_label(df: pd.DataFrame) -> pd.DataFrame:
    """4분면용 outcome 축: whiff가 변별이 있으면 whiff=good, 아니면 xwoba median 기준(낮을수록 투수 좋음)."""
    out = df.copy()
    if "is_whiff" in out.columns:
        w = pd.to_numeric(out["is_whiff"], errors="coerce").fillna(0).astype(int)
        if w.nunique() > 1:
            out["outcome_label_quadrant"] = np.where(w.astype(bool), "outcome_good", "outcome_bad")
            return out
    xw = pd.to_numeric(out.get("xwoba"), errors="coerce")
    if xw.notna().any():
        med = float(xw.median())
        out["outcome_label_quadrant"] = np.where(xw <= med, "outcome_good", "outcome_bad")
    else:
        out["outcome_label_quadrant"] = "outcome_bad"
    return out


def drop_delta_columns_for_outcome_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """delta_*는 delta parquet에만 두고, outcome parquet에 중복 저장하면 merge 시 _x/_y로 갈라진다."""
    drop = [c for c in df.columns if c.startswith("delta_")]
    return df.drop(columns=drop, errors="ignore")


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    thr = float(config["outcome_features"]["hard_hit_threshold"])
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_DELTA)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_DELTA)
    st_o = add_outcome_columns(st_in, thr)
    rp_o = add_outcome_columns(rp_in, thr)
    st_o = assign_outcome_quadrant_label(st_o)
    rp_o = assign_outcome_quadrant_label(rp_o)
    st_o = drop_delta_columns_for_outcome_parquet(st_o)
    rp_o = drop_delta_columns_for_outcome_parquet(rp_o)

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "outcome_features",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_o, rp_o], ignore_index=True),
    )
    save_debug_sample(paths, "outcome_features", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input")
    save_debug_sample(paths, "outcome_features", df=pd.concat([st_o, rp_o], ignore_index=True), suffix="output")

    st_o.to_parquet(paths.interim_dir / A.STARTER_PITCH_OUTCOMES, index=False)
    rp_o.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES, index=False)
    logger.info("outcome 저장: starter=%s행 reliever=%s행", len(st_o), len(rp_o))

    def outcome_counts(df: pd.DataFrame, *, role: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        is_whiff = df.get("is_whiff", pd.Series(False, index=df.index))
        is_strike = df.get("is_strike", pd.Series(False, index=df.index))
        is_hard_hit = df.get("is_hard_hit", pd.Series(False, index=df.index))
        in_play_proxy = (~is_whiff) & (~is_strike)
        return pd.DataFrame(
            {
                "role": [role],
                "n": [int(len(df))],
                "n_whiff": [int(is_whiff.sum())],
                "n_strike": [int(is_strike.sum())],
                "n_in_play_proxy": [int(in_play_proxy.sum())],
                "n_hard_hit": [int(is_hard_hit.sum())],
            }
        )

    def mapping_summary(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["description", "events", "bb_type"] if c in df.columns]
        if not cols:
            return pd.DataFrame()
        g = df.groupby(cols, dropna=False)
        summary = g.agg(
            n=("pitcher", "size"),
            whiff_n=("is_whiff", "sum"),
            strike_n=("is_strike", "sum"),
            hard_hit_n=("is_hard_hit", "sum"),
        ).reset_index()
        summary["whiff_rate"] = summary["whiff_n"] / summary["n"].replace(0, np.nan)
        summary["strike_rate"] = summary["strike_n"] / summary["n"].replace(0, np.nan)
        summary["hard_hit_rate"] = summary["hard_hit_n"] / summary["n"].replace(0, np.nan)
        return summary.sort_values("n", ascending=False).head(200)

    cnt = pd.concat([outcome_counts(st_o, role="starter"), outcome_counts(rp_o, role="reliever")], ignore_index=True)
    cnt.to_parquet(paths.output_tables_dir / "debug_outcome_counts_by_role.parquet", index=False)
    logger.info("outcome counts debug:\n%s", cnt.to_string(index=False))

    ms = mapping_summary(st_o)
    if not ms.empty:
        ms.to_parquet(paths.output_tables_dir / "debug_outcome_mapping_starter_top.parquet", index=False)
    ms = mapping_summary(rp_o)
    if not ms.empty:
        ms.to_parquet(paths.output_tables_dir / "debug_outcome_mapping_reliever_top.parquet", index=False)
