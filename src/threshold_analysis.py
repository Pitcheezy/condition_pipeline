"""threshold: binning 방식 뼈대 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths

logger = logging.getLogger(__name__)


def binning_thresholds(series: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """등간 구간(bin)별 표본 수·평균 점수 요약."""
    # TODO: 최적 임계값 선택·최소 표본 재할당
    n_bins = int(cfg["n_bins"])
    min_samples = int(cfg["min_samples_per_bin"])
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_value", "min_v", "max_v", "dropped_low_n"])
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        edges = np.linspace(lo - 1e-9, hi + 1e-9, n_bins + 1)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    cats = pd.cut(s, bins=edges, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for interval in cats.cat.categories:
        m = s[cats == interval]
        if len(m) == 0:
            continue
        rows.append(
            {
                "bin": str(interval),
                "n": int(len(m)),
                "mean_value": float(m.mean()),
                "min_v": float(m.min()),
                "max_v": float(m.max()),
                "dropped_low_n": int(len(m)) < min_samples,
            }
        )
    return pd.DataFrame(rows)


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    cfg = config["threshold"]
    value_col = "z_aggregate"
    st = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_LABELED)
    rp = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_LABELED)
    if value_col not in st.columns:
        st[value_col] = 0.0
    if value_col not in rp.columns:
        rp[value_col] = 0.0
    st_t = binning_thresholds(st[value_col], cfg)
    rp_t = binning_thresholds(rp[value_col], cfg)
    st_t["role"] = "starter"
    rp_t["role"] = "reliever"
    both = pd.concat([st_t, rp_t], ignore_index=True)
    paths.output_tables_dir.mkdir(parents=True, exist_ok=True)
    both.to_parquet(paths.output_tables_dir / "threshold_bin_summary.parquet", index=False)
    st.to_parquet(paths.interim_dir / A.STARTER_PITCH_THRESHOLD, index=False)
    rp.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_THRESHOLD, index=False)
    logger.info("threshold bin 요약 저장")
