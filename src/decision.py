"""교체 판단: KEEP / WATCH / PULL (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def apply_decisions(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["decision"] = pd.Series(dtype=object)
        return out
    s = pd.to_numeric(out["condition_score"], errors="coerce").fillna(0.0)
    if cfg.get("use_trend", True) and "trend_delta" in out.columns:
        td = pd.to_numeric(out["trend_delta"], errors="coerce").fillna(0.0)
        adj = s + 0.1 * td
    else:
        adj = s
    if cfg.get("use_pitch_count", True) and "pitch_count" in out.columns:
        pass  # TODO: 장이닝·고구수 페널티
    keep_th = float(cfg["keep_score_threshold"])
    watch_th = float(cfg["watch_score_threshold"])
    pull_th = float(cfg["pull_score_threshold"])
    out["decision"] = np.select(
        [adj >= keep_th, adj >= watch_th, adj >= pull_th],
        ["KEEP", "WATCH", "WATCH"],
        default="PULL",
    )
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    cfg = config["decision"]
    st_in = pd.read_parquet(paths.processed_dir / A.STARTER_INNING_TREND)
    rp_in = pd.read_parquet(paths.processed_dir / A.RELIEVER_OUTING_TREND)
    st_d = apply_decisions(st_in, cfg)
    rp_d = apply_decisions(rp_in, cfg)

    log_step_io(
        "decision",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_d, rp_d], ignore_index=True),
    )
    save_debug_sample(paths, "decision", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input_trend")
    save_debug_sample(paths, "decision", df=pd.concat([st_d, rp_d], ignore_index=True), suffix="output_decision")

    def add_trend_direction(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "trend_delta" not in df.columns:
            df = df.copy()
            df["trend_direction"] = "flat"
            return df
        td = pd.to_numeric(df["trend_delta"], errors="coerce")
        df = df.copy()
        df["trend_direction"] = np.select([td > 0, td < 0], ["up", "down"], default="flat")
        return df

    st_dir = add_trend_direction(st_d)
    rp_dir = add_trend_direction(rp_d)
    st_dir["role"] = "starter"
    rp_dir["role"] = "reliever"
    all_d = pd.concat([st_dir, rp_dir], ignore_index=True)

    # decision KEEP/WATCH/PULL count
    if not all_d.empty and "decision" in all_d.columns:
        counts = all_d["decision"].value_counts(dropna=False).rename_axis("decision").reset_index(name="n")
        counts.to_parquet(paths.output_tables_dir / "debug_decision_counts.parquet", index=False)
        logger.info("decision counts:\n%s", counts.to_string(index=False))

    # trend_direction x decision 교차표
    if not all_d.empty and "decision" in all_d.columns and "trend_direction" in all_d.columns:
        cross = (
            all_d.groupby(["role", "trend_direction", "decision"])
            .size()
            .reset_index(name="n")
        )
        cross.to_parquet(paths.output_tables_dir / "debug_decision_trend_cross.parquet", index=False)
    st_d.to_parquet(paths.processed_dir / A.STARTER_INNING_DECISION, index=False)
    rp_d.to_parquet(paths.processed_dir / A.RELIEVER_OUTING_DECISION, index=False)
    logger.info("판단 저장: starter=%s reliever=%s", len(st_d), len(rp_d))
