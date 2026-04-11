"""데이터 수집: Statcast raw parquet 로드/생성, 컬럼 선택, 정렬."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths, ensure_dirs

logger = logging.getLogger(__name__)

try:
    # 런타임에서 import가 안 되어도(의존성 미설치) 합성 폴백이 가능해야 하므로
    # top-level import는 피하고, 여기서는 존재 여부만 확인한다.
    import pybaseball  # noqa: F401

    _HAS_PYBASEBALL = True
except Exception:
    _HAS_PYBASEBALL = False


def load_config_data_keys(config: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    data = config["data"]
    cols = config["columns"]["selected"]
    sort_keys = list(data["sort_keys"])
    return data["file_name"], cols, sort_keys


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _date_chunks(start_ymd: str, end_ymd: str, *, chunk_days: int) -> list[tuple[str, str]]:
    start = _parse_ymd(start_ymd)
    end = _parse_ymd(end_ymd)
    if end < start:
        raise ValueError(f"end_date < start_date: {start_ymd} ~ {end_ymd}")

    chunk_days = int(chunk_days)
    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be > 0, got {chunk_days}")

    out: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + timedelta(days=1)
    return out


@dataclass(frozen=True)
class FetchConfig:
    use_cache: bool = True


def _apply_statcast_csv_patch() -> None:
    """
    pybaseball statcast CSV 파싱 시 ParserError 방지.
    SmartPitch의 patch를 그대로 재사용.
    """
    import pybaseball.datasources.statcast as statcast_ds

    _original = statcast_ds.get_statcast_data_from_csv
    _ = _original  # unused: patch를 위해 원본을 잡아두는 목적

    def _patched(csv_content: str, null_replacement=None, known_percentages=None):
        if null_replacement is None:
            null_replacement = __import__("numpy").nan
        if known_percentages is None:
            known_percentages = []
        data = pd.read_csv(io.StringIO(csv_content), on_bad_lines="skip")
        return statcast_ds.postprocessing.try_parse_dataframe(
            data,
            parse_numerics=False,
            null_replacement=null_replacement,
            known_percentages=known_percentages,
        )

    statcast_ds.get_statcast_data_from_csv = _patched


def fetch_statcast_by_date(start_date: str, end_date: str, cfg: FetchConfig) -> pd.DataFrame:
    """
    기간별 Statcast pitch-by-pitch 데이터 수집.
    """
    from pybaseball import statcast, cache

    _apply_statcast_csv_patch()

    if cfg.use_cache:
        cache.enable()

    df = statcast(start_dt=start_date, end_dt=end_date)
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    return df


def save_parquet_arrow_safe(df: pd.DataFrame, path: Path) -> None:
    """
    SmartPitch의 Parquet 저장 안정화 로직 적용.
    (PyArrow string dtype 조합에서 발생하는 Windows Arrow 호환 이슈 완화)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df2 = df.copy()

    for c in df2.columns:
        dt_str = str(df2[c].dtype).lower()
        if ("string" in dt_str) or (dt_str == "object") or ("category" in dt_str) or ("arrow" in dt_str):
            try:
                arr = df2[c].to_numpy(dtype=object, na_value=None, copy=True)
                converted = [None if v is pd.NA or pd.isna(v) else str(v) for v in arr]
                df2[c] = np.array(converted, dtype=object)
            except Exception:
                s = df2[c].astype("object")
                s = s.fillna(None)
                s = s.apply(lambda v: None if v is None else str(v))
                df2[c] = np.array(s.to_list(), dtype=object)

    try:
        df2.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        df2.to_parquet(path, index=False, engine="fastparquet")


def load_or_fetch_raw_parquet(paths: ProjectPaths, config: dict[str, Any], file_name: str) -> pd.DataFrame:
    path = paths.raw_dir / file_name
    if path.exists() and path.stat().st_size > 0:
        return pd.read_parquet(path)

    season_year = int(config.get("project", {}).get("season_year", 2025))
    fetch_cfg = config.get("data_fetch") or config.get("fetch") or {}
    start_date = str(fetch_cfg.get("start_date", f"{season_year}-03-01"))
    end_date = str(fetch_cfg.get("end_date", f"{season_year}-11-01"))
    chunk_days = int(fetch_cfg.get("chunk_days", 30))
    use_cache = bool(fetch_cfg.get("use_cache", True))

    if not _HAS_PYBASEBALL:
        logger.warning(
            "raw 파일 없음(%s) & pybaseball 미설치 — 최소 합성 데이터 생성합니다. "
            "pyproject.toml에 pybaseball 의존성 추가됨",
            path,
        )
        return pd.DataFrame()

    logger.info(
        "raw 파일 없음(%s) — Statcast fetch로 raw 생성: %s ~ %s (chunk_days=%s)",
        path,
        start_date,
        end_date,
        chunk_days,
    )

    chunks = _date_chunks(start_date, end_date, chunk_days=chunk_days)
    dfs: list[pd.DataFrame] = []
    fcfg = FetchConfig(use_cache=use_cache)
    for i, (cs, ce) in enumerate(chunks, start=1):
        logger.info("fetch statcast chunk %s/%s: %s ~ %s", i, len(chunks), cs, ce)
        df = fetch_statcast_by_date(cs, ce, fcfg)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        logger.warning("Statcast fetch 결과가 비어있습니다: %s ~ %s", start_date, end_date)
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    save_parquet_arrow_safe(raw, path)
    logger.info("raw 저장 완료: %s (%s행)", path, len(raw))
    return raw


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
    raw = load_or_fetch_raw_parquet(paths, config, file_name)
    if raw.empty:
        raw = build_minimal_synthetic_pitches(config)
    raw = select_columns(raw, columns)
    # 이후 단계의 groupby/필터를 위해 숫자형 캐스팅(가능한 컬럼만)
    for c in ("pitcher", "batter", "game_pk", "balls", "strikes", "inning", "outs_when_up", "at_bat_number", "pitch_number"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = sort_pitch_table(raw, sort_keys)
    out_path = paths.interim_dir / A.PITCH_SORTED
    write_sorted_parquet(raw, out_path)
    logger.info("저장 완료: %s (%s행)", out_path, len(raw))
