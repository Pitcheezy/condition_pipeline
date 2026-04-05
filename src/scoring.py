"""inning / outing 단위 score 및 4분면 State 부여 (starter·reliever 분리)."""

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
    total = np.zeros(len(df))
    for col, w in weights.items():
        if col in df.columns:
            total += w * pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return pd.Series(total, index=df.index)


def assign_pitcher_state(df: pd.DataFrame) -> pd.DataFrame:
    """투수의 현재 구위(Condition)와 타격 결과(Outcome)를 바탕으로 4분면 상태를 부여합니다."""
    out = df.copy()
    
    # 평가할 기준 컬럼 설정 (이닝/아웃팅 집계 데이터에 존재하는 컬럼 기준)
    cond_col = "delta_release_speed"
    # rolling 데이터가 집계 테이블에 없다면 일반 xwoba 사용
    out_col = "rolling_xwoba_10" if "rolling_xwoba_10" in out.columns else "xwoba"

    if cond_col not in out.columns or out_col not in out.columns:
        out["pitcher_state"] = "Unknown"
        return out

    x = pd.to_numeric(out[cond_col], errors="coerce")
    y = pd.to_numeric(out[out_col], errors="coerce")

    # 절대 임계값 (분석가가 찾는 핵심 변곡점. 필요시 config로 뺄 수 있음)
    cond_threshold = -1.0   # 예: 평소보다 구속이 1마일 이상 하락하면 Bad
    out_threshold = 0.350   # 예: xwOBA가 0.350 이상이면 Bad

    # 조건(True/False) 정의
    cond_good = x >= cond_threshold
    cond_bad = x < cond_threshold
    out_good = y < out_threshold
    out_bad = y >= out_threshold

    conditions = [
        cond_good & out_good,  # State A: 지표 Good, 결과 Good
        cond_bad & out_good,   # State B: 지표 Bad,  결과 Good
        cond_good & out_bad,   # State C: 지표 Good, 결과 Bad
        cond_bad & out_bad     # State D: 지표 Bad,  결과 Bad
    ]
    
    choices = ['State_A', 'State_B', 'State_C', 'State_D']
    out["pitcher_state"] = np.select(conditions, choices, default="Unknown")
    
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    weights = dict(config["scoring"]["weights"])
    st_in = pd.read_parquet(paths.processed_dir / A.STARTER_INNING_AGG)
    rp_in = pd.read_parquet(paths.processed_dir / A.RELIEVER_OUTING_AGG)

    st = st_in.copy()
    rp = rp_in.copy()
    
    # 1. 기존 가중치 합산 스코어 (보조 지표)
    if not st.empty:
        st["condition_score"] = weighted_score_row(st, weights)
    else:
        st["condition_score"] = pd.Series(dtype=float)
        
    if not rp.empty:
        rp["condition_score"] = weighted_score_row(rp, weights)
    else:
        rp["condition_score"] = pd.Series(dtype=float)

    # 2. 4분면 상태(State) 부여
    st = assign_pitcher_state(st)
    rp = assign_pitcher_state(rp)

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
    logger.info("스코어 및 State 저장: starter=%s reliever=%s", len(st), len(rp))

    # score 분포 저장
    def score_dist(df: pd.DataFrame, *, role: str) -> pd.DataFrame:
        if df.empty or "condition_score" not in df.columns:
            return pd.DataFrame()
        s = pd.to_numeric(df["condition_score"], errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "role": [role], "count": [int(s.shape[0])], "mean": [float(s.mean())],
            "std": [float(s.std())], "min": [float(s.min())], "p05": [float(s.quantile(0.05))],
            "p50": [float(s.quantile(0.5))], "p95": [float(s.quantile(0.95))], "max": [float(s.max())],
        })

    dist = pd.concat([score_dist(st, role="starter"), score_dist(rp, role="reliever")], ignore_index=True)
    if not dist.empty:
        dist.to_parquet(paths.output_tables_dir / "debug_scoring_score_distribution.parquet", index=False)

    # 상태(State) 분포 요약본 추가 저장
    state_counts = pd.concat([st, rp], ignore_index=True)["pitcher_state"].value_counts().reset_index()
    if not state_counts.empty:
        state_counts.to_parquet(paths.output_tables_dir / "debug_scoring_state_distribution.parquet", index=False)