"""디버그/검증 공통 유틸.

요구사항:
- 각 단계마다 input/output rows, unique pitchers, unique games를 공통 로그로 남긴다.
- 상위 20행을 `outputs/tables/debug_*` 형태로 저장한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config_paths import ProjectPaths

logger = logging.getLogger(__name__)


def _safe_unique_count(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def common_step_stats(df: pd.DataFrame, *, pitcher_col: str = "pitcher", games_col: str = "game_pk") -> dict[str, Any]:
    rows = int(len(df)) if df is not None else 0
    return {
        "rows": rows,
        "unique_pitchers": _safe_unique_count(df, pitcher_col),
        "unique_games": _safe_unique_count(df, games_col),
    }


def save_debug_sample(
    paths: ProjectPaths,
    step_name: str,
    *,
    df: pd.DataFrame,
    suffix: str,
    limit: int = 20,
) -> None:
    if df is None:
        return
    out = df.head(limit)
    paths.output_tables_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"debug_{step_name}_{suffix}.parquet"
    out.to_parquet(paths.output_tables_dir / file_name, index=False)


def log_step_io(
    step_name: str,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    *,
    logger_: logging.Logger | None = None,
) -> None:
    log = logger_ or logger
    ist = common_step_stats(input_df)
    ost = common_step_stats(output_df)
    log.info(
        "%s IO: in_rows=%s in_unique_pitchers=%s in_unique_games=%s | out_rows=%s out_unique_pitchers=%s out_unique_games=%s",
        step_name,
        ist["rows"],
        ist["unique_pitchers"],
        ist["unique_games"],
        ost["rows"],
        ost["unique_pitchers"],
        ost["unique_games"],
    )


def pick_sample_pitchers(df: pd.DataFrame, *, n: int = 1) -> list[int]:
    """결정론적으로 pitcher 샘플을 뽑는다."""
    if df is None or df.empty or "pitcher" not in df.columns:
        return []
    uniq = sorted(df["pitcher"].dropna().astype(int).unique().tolist())
    return uniq[:n]

