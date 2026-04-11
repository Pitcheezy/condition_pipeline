"""선발/불펜 분리: 경기별 선발 투수 식별 후 pitch 테이블 분리."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import common_step_stats, log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def load_sorted_pitches(paths: ProjectPaths) -> pd.DataFrame:
    p = paths.interim_dir / A.PITCH_SORTED
    return pd.read_parquet(p)


def compute_game_starters(df: pd.DataFrame) -> pd.DataFrame:
    """이닝 1 Top/Bottom 첫 투구 투수를 각각 홈·원정 선발로 간주한다."""
    # TODO: 더 정교한 선발 식별(선발 교체·더블헤더 등)은 후속 확장
    need = {"game_pk", "inning", "inning_topbot", "pitcher", "at_bat_number", "pitch_number"}
    if not need.issubset(df.columns):
        raise ValueError(f"필요 컬럼 누락: {sorted(need - set(df.columns))}")
    g1 = df[df["inning"] == 1].copy()
    g1 = g1.sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    g1["inning_topbot_u"] = g1["inning_topbot"].astype(str).str.upper().str.strip()
    first_rows = g1.groupby(["game_pk", "inning_topbot_u"], as_index=False).first()
    home_sp = (
        first_rows[first_rows["inning_topbot_u"].str.startswith("T")]
        .loc[:, ["game_pk", "pitcher"]]
        .rename(columns={"pitcher": "home_starter_pitcher"})
    )
    away_sp = (
        first_rows[first_rows["inning_topbot_u"].str.startswith("B")]
        .loc[:, ["game_pk", "pitcher"]]
        .rename(columns={"pitcher": "away_starter_pitcher"})
    )
    out = home_sp.merge(away_sp, on="game_pk", how="outer")
    return out


def attach_starter_flags(df: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    x = df.merge(starters, on="game_pk", how="left")
    pid = x["pitcher"]
    hs = x["home_starter_pitcher"]
    aw = x["away_starter_pitcher"]
    is_starter = (pid == hs) | (pid == aw)
    x["pitcher_role"] = np.where(is_starter, "starter", "reliever")
    return x


def validate_starter_identification(starters: pd.DataFrame, pitched: pd.DataFrame) -> None:
    """starter 식별 결과가 팀당 1명 조건을 만족하는지 검증한다."""
    if starters.empty:
        logging.getLogger(__name__).warning("starter 식별: starters가 비어있다")
        return

    # starters 테이블 자체에서 팀별 중복(동일 game_pk에 동일 역할 투수 다중)이 없어야 한다.
    home_nunique = (
        starters.groupby("game_pk")["home_starter_pitcher"].nunique(dropna=True).max()
    )
    away_nunique = (
        starters.groupby("game_pk")["away_starter_pitcher"].nunique(dropna=True).max()
    )
    if home_nunique > 1:
        logging.getLogger(__name__).error("home starter 중복 감지: game 당 %s명", home_nunique)
    if away_nunique > 1:
        logging.getLogger(__name__).error("away starter 중복 감지: game 당 %s명", away_nunique)
    assert home_nunique <= 1, f"home starter 중복 감지: max nunique={home_nunique}"
    assert away_nunique <= 1, f"away starter 중복 감지: max nunique={away_nunique}"

    # pitched 전체에서 starter로 분류된 투수 수는 game 당 최대 2명이어야 한다.
    x = attach_starter_flags(pitched, starters)
    n_starters = (
        x[x["pitcher_role"] == "starter"]
        .groupby("game_pk")["pitcher"]
        .nunique(dropna=True)
        .max()
    )
    if pd.isna(n_starters):
        n_starters = 0
    if n_starters > 2:
        logging.getLogger(__name__).error("starter 분류: game 당 starter 투수 수가 %s명", n_starters)
    assert n_starters <= 2, f"game 당 starter 투수 수 초과: max nunique={n_starters}"


def build_starter_inning_unit_id(df: pd.DataFrame) -> pd.Series:
    """starter 분석 단위: inning — (game_pk, pitcher, inning)."""
    return (
        df["game_pk"].astype(str)
        + "_"
        + df["pitcher"].astype(str)
        + "_"
        + df["inning"].astype(str)
    )


def build_reliever_outing_unit_id(df: pd.DataFrame, *, mode: str = "by_inning_topbot") -> pd.Series:
    """reliever 분석 단위: outing(appearance 단위 확장 가능 구조).

    현재 모드:
    - `by_inning_topbot`: inning_topbot(Top/Bottom) 변화가 있으면 appearance_id를 분리 (초기 근사)
    - `game_pitcher`: (기존 임시 규칙) 경기 내 동일 투수는 1 outing으로 간주
    """
    if df.empty:
        return pd.Series(dtype=object)

    if mode == "game_pitcher":
        return df["game_pk"].astype(str) + "_" + df["pitcher"].astype(str)

    required = {"inning", "inning_topbot", "game_pk", "pitcher"}
    if not required.issubset(df.columns):
        # TODO: true appearance segmentation을 위한 데이터(교체/재등판 식별자) 필요
        return df["game_pk"].astype(str) + "_" + df["pitcher"].astype(str)

    x = df.copy()
    x["inning"] = pd.to_numeric(x["inning"], errors="coerce")
    x["inning_topbot_u"] = x["inning_topbot"].astype(str).str.upper().str.strip()
    x["topbot_flag"] = x["inning_topbot_u"].str.startswith("B").astype(int)
    x["phase"] = x["inning"].fillna(0).astype(int) * 2 + x["topbot_flag"]

    order_keys = ["game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number"]
    order_keys = [k for k in order_keys if k in x.columns]
    if order_keys:
        x = x.sort_values(order_keys, kind="mergesort")

    # 동일 게임/투수 안에서 phase가 바뀌면 새로운 appearance(근사)로 본다.
    x["appearance_id"] = (
        (x["phase"] != x.groupby(["game_pk", "pitcher"])["phase"].shift(1))
        .groupby([x["game_pk"], x["pitcher"]])
        .cumsum()
    ).astype(int)

    return (
        x["game_pk"].astype(str)
        + "_"
        + x["pitcher"].astype(str)
        + "_"
        + x["appearance_id"].astype(str)
    )


def split_frames(
    df: pd.DataFrame,
    *,
    reliever_outing_mode: str = "by_inning_topbot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = df[df["pitcher_role"] == "starter"].copy()
    rp = df[df["pitcher_role"] == "reliever"].copy()
    st["starter_inning_unit_id"] = build_starter_inning_unit_id(st)
    # TODO: 실제 appearance 단위로 분리되도록 `mode`를 확장할 것
    rp["reliever_outing_unit_id"] = build_reliever_outing_unit_id(rp, mode=reliever_outing_mode)
    return st, rp


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    df_in = load_sorted_pitches(paths)
    starters = compute_game_starters(df_in)
    validate_starter_identification(starters, df_in)
    df = attach_starter_flags(df_in, starters)
    reliever_outing_mode = config.get("aggregation", {}).get("reliever_outing_mode", "by_inning_topbot")
    starter_df, reliever_df = split_frames(df, reliever_outing_mode=reliever_outing_mode)

    # 공통 IO 로그/샘플 저장
    df_out = pd.concat([starter_df, reliever_df], ignore_index=True)
    log_step_io("split_pitchers", df_in, df_out)
    save_debug_sample(paths, "split_pitchers", df=df_in, suffix="input")
    save_debug_sample(paths, "split_pitchers", df=starter_df, suffix="starter_output")
    save_debug_sample(paths, "split_pitchers", df=reliever_df, suffix="reliever_output")

    # starter: game 당 최대 2명 검증용 테이블/로그
    if not starter_df.empty and "game_pk" in starter_df.columns:
        st_counts = (
            starter_df.groupby("game_pk")["pitcher"]
            .nunique(dropna=True)
            .reset_index(name="starter_pitcher_n")
        )
        max_n = int(st_counts["starter_pitcher_n"].max()) if len(st_counts) else 0
        logger.info("split_pitchers: starter max per game=%s", max_n)
        st_stat = common_step_stats(starter_df)
        rp_stat = common_step_stats(reliever_df)
        logger.info("split_pitchers: starter stats rows=%s pitchers=%s games=%s", st_stat["rows"], st_stat["unique_pitchers"], st_stat["unique_games"])
        logger.info("split_pitchers: reliever stats rows=%s pitchers=%s games=%s", rp_stat["rows"], rp_stat["unique_pitchers"], rp_stat["unique_games"])
        st_counts.to_parquet(paths.output_tables_dir / "debug_split_pitchers_starter_counts.parquet", index=False)

    # reliever: appearance_id가 game 내에서 어떻게 나뉘는지 샘플 출력
    if not reliever_df.empty and reliever_outing_mode == "by_inning_topbot":
        rp = reliever_df.copy()
        rp["inning"] = pd.to_numeric(rp.get("inning"), errors="coerce")
        rp["inning_topbot_u"] = rp["inning_topbot"].astype(str).str.upper().str.strip()
        rp["topbot_flag"] = rp["inning_topbot_u"].str.startswith("B").astype(int)
        rp["phase"] = rp["inning"].fillna(0).astype(int) * 2 + rp["topbot_flag"]

        order_keys = ["game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number"]
        order_keys = [k for k in order_keys if k in rp.columns]
        if order_keys:
            rp = rp.sort_values(order_keys, kind="mergesort")

        rp["appearance_id"] = (
            (rp["phase"] != rp.groupby(["game_pk", "pitcher"])["phase"].shift(1))
            .groupby([rp["game_pk"], rp["pitcher"]])
            .cumsum()
            .astype(int)
        )

        # game 내 appearance 분리 샘플
        sample_game_pitcher = rp.groupby(["game_pk", "pitcher"], as_index=False).agg(
            appearance_n=("appearance_id", "nunique")
        )
        sample_game_pitcher = sample_game_pitcher.sort_values("appearance_n", ascending=False).head(10)
        sample_game_pitcher.to_parquet(
            paths.output_tables_dir / "debug_split_pitchers_reliever_appearance_sample_game_pitcher.parquet",
            index=False,
        )
        rp_out = rp[["game_pk", "pitcher", "appearance_id", "inning", "inning_topbot", "at_bat_number", "pitch_number"]].head(200)
        rp_out.to_parquet(
            paths.output_tables_dir / "debug_split_pitchers_reliever_appearance_detail.parquet",
            index=False,
        )

    starter_df.to_parquet(paths.interim_dir / A.STARTER_PITCH_SPLIT, index=False)
    reliever_df.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_SPLIT, index=False)
    logger.info(
        "분리 저장: starter=%s행 reliever=%s행",
        len(starter_df),
        len(reliever_df),
    )
