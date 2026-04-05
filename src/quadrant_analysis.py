"""4분면 분석 요약 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)

def quadrant_summary(df: pd.DataFrame, x_col: str, y_col: str, role: str) -> pd.DataFrame:
    """두 축의 절대적 기준점(Threshold) 기반 4분면 카운트."""
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame({"role": [role], "quadrant": ["undefined"], "n": [0]})
    
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    
    # 1. x축 (지표 - Condition) 절대 기준선 설정
    if "delta_release_speed" in x_col:
        xm = -1.0  # 평소(Baseline)보다 1마일 이상 떨어졌을 때를 저하 기준으로 설정
    elif "delta_release_spin_rate" in x_col:
        xm = -100.0  # 회전수가 100rpm 이상 감소했을 때를 저하 기준으로 설정
    else:
        xm = x.median()  # 그 외의 알 수 없는 지표는 임시로 중앙값 사용
        
    # 2. y축 (결과 - Outcome) 절대 기준선 설정
    if "xwoba" in y_col:
        ym = 0.350  # xwOBA가 0.350 이상이면 타격 결과가 나쁜 것(리그 평균 대비 높음)으로 판단
    elif "whiff_rate" in y_col:
        ym = 0.200  # 헛스윙률이 20% 이하면 구위가 나쁜 것으로 판단
    else:
        ym = y.median()

    # 3. 4분면 분류 (기준점 적용)
    # x >= xm : 구속/회전수 등 지표가 기준치 이상 (Good Condition)
    # x < xm  : 구속/회전수 등 지표가 기준치 미만 (Bad Condition)
    # y >= ym : 결과 지표가 기준치 이상 (xwOBA라면 Bad Outcome, whiff_rate라면 Good Outcome)
    quad = np.where(
        (x >= xm) & (y >= ym),
        "Q1_high_high",
        np.where(
            (x < xm) & (y >= ym),
            "Q2_low_high",
            np.where((x >= xm) & (y < ym), "Q3_high_low", "Q4_low_low"),
        ),
    )
    
    tmp = pd.DataFrame({"quad": quad})
    c = tmp["quad"].value_counts().rename_axis("quadrant").reset_index(name="n")
    c["role"] = role
    
    # 결과 데이터에 사용된 기준점(Threshold)을 함께 저장하여 디버깅 및 분석에 활용
    c["x_threshold"] = xm
    c["y_threshold"] = ym
    
    return c


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    _ = config
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_LABELED)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_LABELED)
    x_col = "delta_release_speed" if "delta_release_speed" in st_in.columns else st_in.columns[0]
    if "rolling_xwoba_10" in st_in.columns:
        y_col = "rolling_xwoba_10"
    elif "delta_release_spin_rate" in st_in.columns:
        y_col = "delta_release_spin_rate"
    elif "z_aggregate" in st_in.columns:
        y_col = "z_aggregate"
    else:
        y_col = x_col
    s1 = quadrant_summary(st_in, x_col, y_col, "starter")
    s2 = quadrant_summary(rp_in, x_col, y_col, "reliever")
    out = pd.concat([s1, s2], ignore_index=True)

    # 공통 IO 로그/샘플 저장
    log_step_io("quadrant_analysis", pd.concat([st_in, rp_in], ignore_index=True), out)
    save_debug_sample(paths, "quadrant_analysis", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input_labeled")
    if not out.empty:
        save_debug_sample(paths, "quadrant_analysis", df=out, suffix="output_quadrant_summary")

    out.to_parquet(paths.output_tables_dir / A.TABLE_QUADRANT_SUMMARY, index=False)
    st_in.to_parquet(paths.interim_dir / A.STARTER_PITCH_QUADRANT, index=False)
    rp_in.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_QUADRANT, index=False)
    logger.info("4분면 요약 저장: %s", paths.output_tables_dir / A.TABLE_QUADRANT_SUMMARY)

    # quadrant 조합 count + 사용 축 저장
    if not out.empty:
        comb = out.pivot_table(index="quadrant", columns="role", values="n", aggfunc="sum", fill_value=0).reset_index()
        comb.to_parquet(paths.output_tables_dir / "debug_quadrant_analysis_combinations.parquet", index=False)
        pd.DataFrame([{"x_col": x_col, "y_col": y_col, "n_roles_rows": len(out)}]).to_parquet(
            paths.output_tables_dir / "debug_quadrant_analysis_axis_choice.parquet",
            index=False,
        )
