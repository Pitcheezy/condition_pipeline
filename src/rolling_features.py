"""rolling 집계 (starter / reliever 분리, 투수별 시계열)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, pick_sample_pitchers, save_debug_sample

logger = logging.getLogger(__name__)


def sort_keys_from_config(config: dict[str, Any]) -> list[str]:
    return list(config["data"]["sort_keys"])


def add_rolling_rates(df: pd.DataFrame, windows: list[int], sort_keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    use_keys = [k for k in sort_keys if k in out.columns]
    out = out.sort_values(use_keys, kind="mergesort")
    g = out.groupby("pitcher", group_keys=False)

    for w in windows:
        minp = max(1, w // 3)
        if "is_whiff" in out.columns:
            out[f"rolling_whiff_rate_{w}"] = g["is_whiff"].transform(
                lambda s: s.astype(float).rolling(w, min_periods=minp).mean()
            )
        if "is_strike" in out.columns:
            out[f"rolling_strike_rate_{w}"] = g["is_strike"].transform(
                lambda s: s.astype(float).rolling(w, min_periods=minp).mean()
            )
        if "xwoba" in out.columns:
            out[f"rolling_xwoba_{w}"] = g["xwoba"].transform(
                lambda s: pd.to_numeric(s, errors="coerce").rolling(w, min_periods=minp).mean()
            )
        if "is_hard_hit" in out.columns:
            out[f"rolling_hard_hit_rate_{w}"] = g["is_hard_hit"].transform(
                lambda s: s.astype(float).rolling(w, min_periods=minp).mean()
            )
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    windows = [int(x) for x in config["rolling"]["windows"]]
    sk = sort_keys_from_config(config)
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_OUTCOMES)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES)
    st_r = add_rolling_rates(st_in, windows, sk)
    rp_r = add_rolling_rates(rp_in, windows, sk)

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "rolling_features",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_r, rp_r], ignore_index=True),
    )
    save_debug_sample(paths, "rolling_features", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input")
    save_debug_sample(paths, "rolling_features", df=pd.concat([st_r, rp_r], ignore_index=True), suffix="output")

    # 특정 pitcher-game 샘플 rolling 값 검증 테이블 저장 + 누수 검증(간이 manual 비교)
    def rolling_leakage_check(raw_df: pd.DataFrame, rolled_df: pd.DataFrame, *, role: str) -> None:
        if raw_df.empty or rolled_df.empty or "pitcher" not in raw_df.columns or "game_pk" not in raw_df.columns:
            return
        pids = pick_sample_pitchers(raw_df, n=1)
        if not pids:
            return
        pid = pids[0]
        games = sorted(raw_df.loc[raw_df["pitcher"] == pid, "game_pk"].dropna().unique().tolist())
        if not games:
            return
        gpk = games[0]

        # rolling 누수 검증용: pitcher's full 시퀀스에서 rolling을 만든 뒤
        # 특정 game_pk 구간만 잘라 비교한다(이전 게임 데이터 포함).
        use_keys = [k for k in sk if k in raw_df.columns]
        pitcher_all = raw_df[raw_df["pitcher"] == pid].copy()
        if use_keys:
            pitcher_all = pitcher_all.sort_values(use_keys, kind="mergesort")
        if pitcher_all.empty:
            return
        w = 10 if 10 in windows else windows[0]
        minp = max(1, w // 3)

        if "is_whiff" in pitcher_all.columns and f"rolling_whiff_rate_{w}" in rolled_df.columns:
            s_all = pd.to_numeric(pitcher_all["is_whiff"], errors="coerce").astype(float)
            manual_all = s_all.rolling(w, min_periods=minp).mean()
            pitcher_all["manual_rolling_whiff_rate_w"] = manual_all.values

            sample_manual = pitcher_all[pitcher_all["game_pk"] == gpk].copy()
            sample_rolled = rolled_df[(rolled_df["pitcher"] == pid) & (rolled_df["game_pk"] == gpk)].copy()
            if use_keys and all(k in sample_rolled.columns for k in use_keys):
                sample_manual = sample_manual.sort_values(use_keys, kind="mergesort")
                sample_rolled = sample_rolled.sort_values(use_keys, kind="mergesort")

            if sample_manual.empty or sample_rolled.empty:
                return

            diff = (
                sample_manual["manual_rolling_whiff_rate_w"].values
                - pd.to_numeric(sample_rolled[f"rolling_whiff_rate_{w}"], errors="coerce").values
            ).astype(float)
            max_abs = float(np.nanmax(np.abs(diff))) if len(diff) else 0.0

            check_df = sample_rolled.copy()
            check_df[f"manual_rolling_whiff_rate_{w}"] = sample_manual["manual_rolling_whiff_rate_w"].values
            check_df[f"diff_manual_vs_col_{w}"] = diff
            check_df.to_parquet(
                paths.output_tables_dir / f"debug_rolling_leakage_check_{role}_pid_{pid}_gamepk_{gpk}.parquet",
                index=False,
            )
            logger.info("rolling_features leakage_check[%s]: pid=%s game_pk=%s w=%s max_abs_diff=%s", role, pid, gpk, w, max_abs)

        # rolling 값 존재 여부 테이블 저장
        cols = [c for c in rolled_df.columns if c.startswith("rolling_") and any(str(w) in c for w in windows)] if not rolled_df.empty else []
        if cols:
            sample_out = rolled_df[(rolled_df["pitcher"] == pid) & (rolled_df["game_pk"] == gpk)].head(200)
            sample_out.to_parquet(paths.output_tables_dir / f"debug_rolling_sample_{role}_pid_{pid}_gamepk_{gpk}.parquet", index=False)

    rolling_leakage_check(st_in, st_r, role="starter")
    rolling_leakage_check(rp_in, rp_r, role="reliever")
    st_r.to_parquet(paths.interim_dir / A.STARTER_PITCH_ROLLING, index=False)
    rp_r.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_ROLLING, index=False)
    logger.info("rolling 저장: starter=%s행 reliever=%s행", len(st_r), len(rp_r))
