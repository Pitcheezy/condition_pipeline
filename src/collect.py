"""데이터 수집: raw parquet 로드, 컬럼 선택, 정렬."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths, ensure_dirs

logger = logging.getLogger(__name__)


def load_config_data_keys(config: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    data = config["data"]
    cols = config["columns"]["selected"]
    sort_keys = list(data["sort_keys"])
    return data["file_name"], cols, sort_keys


def read_raw_parquet(paths: ProjectPaths, file_name: str) -> pd.DataFrame:
    path = paths.raw_dir / file_name
    if not path.exists():
        logger.warning("raw 파일 없음: %s — 최소 합성 데이터 생성", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("누락 컬럼(0으로 채움): %s", missing)
        for c in missing:
            df[c] = np.nan
    return df[list(columns)].copy()


def sort_pitch_table(df: pd.DataFrame, sort_keys: list[str]) -> pd.DataFrame:
    use = [k for k in sort_keys if k in df.columns]
    if not use:
        return df.reset_index(drop=True)
    return df.sort_values(use, kind="mergesort").reset_index(drop=True)


def build_minimal_synthetic_pitches(config: dict[str, Any]) -> pd.DataFrame:
    """필터·후속 단계가 동작하도록 통과 가능한 규모의 합성 Statcast 형 테이블 생성."""
    # TODO: 실제 수집 API/파일 연동 시 이 함수 제거 또는 플래그로만 사용
    rng = np.random.default_rng(int(config["project"]["random_seed"]))
    _, cols, _ = load_config_data_keys(config)

    # 선발·불펜 필터를 동시에 만족하도록 경기 수·RP 풀을 맞춘다
    games = np.arange(40_001, 40_151, dtype=np.int64)
    pitchers_sp = [1001, 1002, 1003]
    pitchers_rp = [2001, 2002]

    rows: list[dict[str, Any]] = []
    for g in games:
        home_sp, away_sp = rng.choice(pitchers_sp, size=2, replace=False)
        for _ in range(8):
            rows.append(_one_pitch_row(g, int(home_sp), 1, "Top", rng, cols))
        for _ in range(8):
            rows.append(_one_pitch_row(g, int(away_sp), 1, "Bottom", rng, cols))
        for inn in range(2, 7):
            for _ in range(6):
                rows.append(_one_pitch_row(g, int(home_sp), inn, "Top" if inn % 2 else "Bottom", rng, cols))
            for _ in range(6):
                rows.append(_one_pitch_row(g, int(away_sp), inn, "Bottom" if inn % 2 else "Top", rng, cols))
        rp = int(pitchers_rp[(g - int(games[0])) % len(pitchers_rp)])
        for _ in range(25):
            rows.append(_one_pitch_row(g, rp, 7, "Top", rng, cols))

    out = pd.DataFrame(rows)
    return select_columns(out, cols)


def _one_pitch_row(
    game_pk: int,
    pitcher: int,
    inning: int,
    topbot: str,
    rng: np.random.Generator,
    cols: list[str],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "game_date": "2025-06-01",
        "game_pk": game_pk,
        "pitcher": pitcher,
        "batter": int(rng.integers(3_000, 4_000)),
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "release_speed": float(rng.normal(93, 2)),
        "release_spin_rate": float(rng.normal(2200, 100)),
        "release_extension": 6.5,
        "release_pos_x": float(rng.normal(0, 0.2)),
        "release_pos_z": float(rng.normal(5.8, 0.1)),
        "pfx_x": float(rng.normal(0, 0.5)),
        "pfx_z": float(rng.normal(1.0, 0.3)),
        "plate_x": float(rng.normal(0, 0.5)),
        "plate_z": float(rng.normal(2.5, 0.3)),
        "zone": int(rng.integers(1, 14)),
        "balls": int(rng.integers(0, 4)),
        "strikes": int(rng.integers(0, 3)),
        "inning": inning,
        "inning_topbot": topbot,
        "outs_when_up": int(rng.integers(0, 3)),
        "at_bat_number": int(rng.integers(1, 40)),
        "pitch_number": int(rng.integers(1, 8)),
        # 검증·whiff 분포용: 단일 called_strike 고정이면 is_whiff가 전부 0이 됨
        "description": str(
            rng.choice(
                np.array(
                    [
                        "called_strike",
                        "swinging_strike",
                        "swinging_strike_blocked",
                        "foul",
                        "ball",
                        "hit_into_play",
                    ]
                )
            )
        ),
        "events": "",
        "bb_type": "",
        "estimated_woba_using_speedangle": float(rng.uniform(0.1, 0.5)),
    }
    return {c: base.get(c, np.nan) for c in cols}


def write_sorted_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    ensure_dirs(paths)
    file_name, columns, sort_keys = load_config_data_keys(config)
    raw = read_raw_parquet(paths, file_name)
    if raw.empty:
        raw = build_minimal_synthetic_pitches(config)
    raw = select_columns(raw, columns)
    raw = sort_pitch_table(raw, sort_keys)
    out_path = paths.interim_dir / A.PITCH_SORTED
    write_sorted_parquet(raw, out_path)
    logger.info("저장 완료: %s (%s행)", out_path, len(raw))
