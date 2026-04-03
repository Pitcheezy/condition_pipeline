"""기준 필터링: starter / reliever 각각 독립 기준 적용."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def load_split_tables(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_SPLIT)
    rp = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_SPLIT)
    return st, rp


def estimate_innings_per_game_game_pitcher(df: pd.DataFrame) -> pd.DataFrame:
    """경기·투수별 이닝(=아웃/3) 근사치를 계산한다.

    - `inning_topbot`(Top/Bottom)과 `outs_when_up`을 이용해
      등장한 구간의 부분 아웃을 추정한다.
    - TODO: 향후 `events`/`outcome` 기준으로 실제 아웃 누적(가능 시)으로 교체.
    """
    required = {"inning", "inning_topbot", "outs_when_up", "game_pk", "pitcher"}
    missing = required - set(df.columns)
    if missing:
        # TODO: 정식 데이터로 교체 시 이 폴백 제거
        g = df.groupby(["game_pk", "pitcher"], as_index=False).agg(
            inn_min=("inning", "min"),
            inn_max=("inning", "max"),
        )
        g["innings_est"] = (g["inn_max"] - g["inn_min"] + 1).clip(lower=1)
        return g.loc[:, ["game_pk", "pitcher", "innings_est"]]

    x = df.copy()
    x["inning"] = pd.to_numeric(x["inning"], errors="coerce")
    x["outs_when_up"] = pd.to_numeric(x["outs_when_up"], errors="coerce")
    bot_flag = (
        x["inning_topbot"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.startswith("B")
        .astype(int)
    )
    # half-inning phase: inning(1..N) * 2 + top/bot
    x["phase"] = x["inning"].astype("Int64").fillna(0).astype(int) * 2 + bot_flag
    # clamp to [0,2] (outs_when_up is typically 0/1/2)
    x["outs_when_up"] = x["outs_when_up"].fillna(0).clip(lower=0, upper=2)

    rows: list[dict[str, Any]] = []
    for (game_pk, pitcher), sub in x.groupby(["game_pk", "pitcher"], sort=False):
        sub = sub.dropna(subset=["phase"])
        if sub.empty:
            rows.append({"game_pk": game_pk, "pitcher": pitcher, "innings_est": 0.0})
            continue

        min_phase = int(sub["phase"].min())
        max_phase = int(sub["phase"].max())

        first = sub[sub["phase"] == min_phase]
        last = sub[sub["phase"] == max_phase]

        outs_first_before = int(first["outs_when_up"].min())
        outs_last_before = int(last["outs_when_up"].max())

        if min_phase == max_phase:
            # 동일 half-inning 안에서의 등장이라면,
            # 시작 시점(outs_first_before)부터 마지막 pitch 직전(outs_last_before)까지의
            # 대략 아웃 수를 (마지막 앞에서 +1)로 근사한다.
            outs_est = (outs_last_before - outs_first_before) + 1
            outs_est = int(np.clip(outs_est, 0, 3))
        else:
            # 첫 half-inning: 이미 만들어진 outs(outs_first_before)를 제외한 잔여 아웃
            outs_first_partial = int(np.clip(3 - outs_first_before, 0, 3))
            # 마지막 half-inning: 마지막 pitch 직전 outs_last_before + 1(다음 아웃 발생 가정)
            outs_last_partial = int(np.clip(outs_last_before + 1, 0, 3))
            # 중간 half-inning들은 full 3 outs로 가정
            mid_halves_full = max(0, max_phase - min_phase - 1)
            outs_est = outs_first_partial + outs_last_partial + mid_halves_full * 3

        innings_est = float(outs_est) / 3.0
        innings_est = max(0.0, innings_est)
        rows.append({"game_pk": game_pk, "pitcher": pitcher, "innings_est": innings_est})

    return pd.DataFrame(rows)


def starter_season_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ipg = estimate_innings_per_game_game_pitcher(df)
    games = ipg.groupby("pitcher", as_index=False).agg(
        games_started=("game_pk", "nunique"),
        total_innings=("innings_est", "sum"),
    )
    games["avg_innings_per_start"] = games["total_innings"] / games["games_started"].replace(0, np.nan)
    return games


def reliever_season_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ipg = estimate_innings_per_game_game_pitcher(df)
    pitches = df.groupby(["game_pk", "pitcher"], as_index=False).size().rename(columns={"size": "pitches"})
    m = ipg.merge(pitches, on=["game_pk", "pitcher"], how="left")
    per = m.groupby("pitcher", as_index=False).agg(
        games=("game_pk", "nunique"),
        total_innings=("innings_est", "sum"),
        total_pitches=("pitches", "sum"),
    )
    per["avg_pitches_per_inning"] = per["total_pitches"] / per["total_innings"].replace(0, np.nan)
    return per


def filter_starter_pitchers(metrics: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    f = cfg["pitcher_filters"]["starter"]
    m = metrics
    mask = (
        (m["games_started"] >= f["min_games_started"])
        & (m["avg_innings_per_start"] >= f["min_avg_innings_per_start"])
        & (m["total_innings"] >= f["min_total_innings"])
    )
    return m.loc[mask, "pitcher"].dropna().astype(int).unique()


def filter_reliever_pitchers(metrics: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    f = cfg["pitcher_filters"]["reliever"]
    m = metrics
    mask = (
        (m["games"] >= f["min_games"])
        & (m["avg_pitches_per_inning"] >= f["min_avg_pitches_per_inning"])
        & (m["total_innings"] >= f["min_total_innings"])
    )
    return m.loc[mask, "pitcher"].dropna().astype(int).unique()


def apply_pitcher_filter(df: pd.DataFrame, pitchers: np.ndarray) -> pd.DataFrame:
    if len(pitchers) == 0:
        logger.warning("필터 결과 투수 0명 — 빈 테이블 유지")
        return df.iloc[0:0].copy()
    return df[df["pitcher"].isin(pitchers)].copy()


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    st_in, rp_in = load_split_tables(paths)
    st_m = starter_season_metrics(st_in)
    rp_m = reliever_season_metrics(rp_in)

    st_keep = filter_starter_pitchers(st_m, config)
    rp_keep = filter_reliever_pitchers(rp_m, config)

    st_f = apply_pitcher_filter(st_in, st_keep)
    rp_f = apply_pitcher_filter(rp_in, rp_keep)

    # 공통 IO 로그/샘플 저장
    df_in = pd.concat([st_in, rp_in], ignore_index=True)
    df_out = pd.concat([st_f, rp_f], ignore_index=True)
    log_step_io("filter_pitchers", df_in, df_out)
    save_debug_sample(paths, "filter_pitchers", df=df_in, suffix="input")
    save_debug_sample(paths, "filter_pitchers", df=df_out, suffix="output")
    save_debug_sample(paths, "filter_pitchers", df=st_in, suffix="starter_input")
    save_debug_sample(paths, "filter_pitchers", df=rp_in, suffix="reliever_input")
    save_debug_sample(paths, "filter_pitchers", df=st_f, suffix="starter_output")
    save_debug_sample(paths, "filter_pitchers", df=rp_f, suffix="reliever_output")

    # 시즌 통계 테이블 저장
    paths.output_tables_dir.mkdir(parents=True, exist_ok=True)
    st_m.to_parquet(paths.output_tables_dir / "debug_filter_pitchers_starter_season_metrics.parquet", index=False)
    rp_m.to_parquet(paths.output_tables_dir / "debug_filter_pitchers_reliever_season_metrics.parquet", index=False)

    # 필터 통과/탈락 수 로그
    all_st = int(st_m["pitcher"].nunique()) if not st_m.empty and "pitcher" in st_m.columns else 0
    all_rp = int(rp_m["pitcher"].nunique()) if not rp_m.empty and "pitcher" in rp_m.columns else 0
    logger.info(
        "filter_pitchers pass/fail: starter pass=%s/%s, reliever pass=%s/%s",
        len(st_keep),
        all_st,
        len(rp_keep),
        all_rp,
    )

    # 경계값 근처 샘플 출력
    def boundary_sample(metrics: pd.DataFrame, f: dict[str, Any], *, role: str) -> pd.DataFrame:
        if metrics.empty:
            return metrics
        mask = pd.Series(False, index=metrics.index)
        for k, v in f.items():
            if k not in metrics.columns:
                continue
            if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
                tol = 1.0
            else:
                tol = max(0.1, abs(float(v)) * 0.05)
            mask = mask | ((metrics[k] - float(v)).abs() <= tol)
        out = metrics.loc[mask].copy()
        if not out.empty:
            sort_col = "total_innings" if "total_innings" in out.columns else out.columns[0]
            out = out.sort_values(sort_col, ascending=False).head(20)
        return out

    st_cfg = config["pitcher_filters"]["starter"]
    rp_cfg = config["pitcher_filters"]["reliever"]
    st_boundary = boundary_sample(st_m, st_cfg, role="starter")
    rp_boundary = boundary_sample(rp_m, rp_cfg, role="reliever")
    if not st_boundary.empty:
        st_boundary.to_parquet(paths.output_tables_dir / "debug_filter_pitchers_starter_boundary_sample.parquet", index=False)
    if not rp_boundary.empty:
        rp_boundary.to_parquet(paths.output_tables_dir / "debug_filter_pitchers_reliever_boundary_sample.parquet", index=False)

    st_f.to_parquet(paths.interim_dir / A.STARTER_PITCH_FILTERED, index=False)
    rp_f.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_FILTERED, index=False)
    logger.info(
        "필터 완료: starter=%s명(%s행) reliever=%s명(%s행)",
        len(st_keep),
        len(st_f),
        len(rp_keep),
        len(rp_f),
    )
