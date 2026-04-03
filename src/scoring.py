"""inning / outing 단위 score (starter·reliever 가중치 동일 구조)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def weighted_score_row(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """가중치에 있는 컬럼만 합산 (없으면 0)."""
    # TODO: 정규화·누락값 대체 전략
    total = np.zeros(len(df))
    for col, w in weights.items():
        if col in df.columns:
            total += w * pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return pd.Series(total, index=df.index)


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    weights = dict(config["scoring"]["weights"])
    st_in = pd.read_parquet(paths.processed_dir / A.STARTER_INNING_AGG)
    rp_in = pd.read_parquet(paths.processed_dir / A.RELIEVER_OUTING_AGG)

    st = st_in.copy()
    rp = rp_in.copy()
    if not st.empty:
        st["condition_score"] = weighted_score_row(st, weights)
    else:
        st["condition_score"] = pd.Series(dtype=float)
    if not rp.empty:
        rp["condition_score"] = weighted_score_row(rp, weights)
    else:
        rp["condition_score"] = pd.Series(dtype=float)

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "scoring",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st, rp], ignore_index=True),
    )
    save_debug_sample(paths, "scoring", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input_agg")
    save_debug_sample(paths, "scoring", df=pd.concat([st, rp], ignore_index=True), suffix="output_scored")

    st.to_parquet(paths.processed_dir / A.STARTER_INNING_SCORE, index=False)
    rp.to_parquet(paths.processed_dir / A.RELIEVER_OUTING_SCORE, index=False)
    logger.info("스코어 저장: starter=%s reliever=%s", len(st), len(rp))

    # score 분포 저장
    def score_dist(df: pd.DataFrame, *, role: str) -> pd.DataFrame:
        if df.empty or "condition_score" not in df.columns:
            return pd.DataFrame()
        s = pd.to_numeric(df["condition_score"], errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "role": [role],
                "count": [int(s.shape[0])],
                "mean": [float(s.mean())],
                "std": [float(s.std())],
                "min": [float(s.min())],
                "p05": [float(s.quantile(0.05))],
                "p50": [float(s.quantile(0.5))],
                "p95": [float(s.quantile(0.95))],
                "max": [float(s.max())],
            }
        )

    dist = pd.concat([score_dist(st, role="starter"), score_dist(rp, role="reliever")], ignore_index=True)
    if not dist.empty:
        dist.to_parquet(paths.output_tables_dir / "debug_scoring_score_distribution.parquet", index=False)
