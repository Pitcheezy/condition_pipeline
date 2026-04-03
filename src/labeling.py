"""good/bad 기준: z-score 기반 라벨 (starter / reliever 분리)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import artifacts as A
from config_paths import ProjectPaths
from debug_utils import log_step_io, save_debug_sample

logger = logging.getLogger(__name__)


def zscore_labels(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """delta_* 컬럼에 대해 투수 내 z-score 후 종합(z_aggregate)로 good/bad 라벨링.

    - 전역 표준화 대신, 투수 개인(해당 pitcher 내) 분포 대비 delta 중심으로 판단한다.
    """
    # TODO: quantile 모드 및 구종별 라벨
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if not delta_cols:
        out = df.copy()
        out["z_aggregate"] = np.nan
        out["condition_label"] = "neutral"
        return out
    out = df.copy()
    zcols = []
    for c in delta_cols:
        zname = f"_z_{c}"
        out[zname] = out.groupby("pitcher")[c].transform(
            lambda x: (pd.to_numeric(x, errors="coerce") - pd.to_numeric(x, errors="coerce").mean())
            / (pd.to_numeric(x, errors="coerce").std() + 1e-9)
        )
        zcols.append(zname)
    out["z_aggregate"] = out[zcols].mean(axis=1)
    zg = float(cfg["zscore_good"])
    zb = float(cfg["zscore_bad"])
    out["condition_label"] = np.where(
        out["z_aggregate"] >= zg,
        "good",
        np.where(out["z_aggregate"] <= zb, "bad", "neutral"),
    )
    out = out.drop(columns=zcols, errors="ignore")
    return out


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    cfg = config["labeling"]
    method = str(cfg.get("method", "zscore")).lower()
    st_in = pd.read_parquet(paths.interim_dir / A.STARTER_PITCH_ROLLING)
    rp_in = pd.read_parquet(paths.interim_dir / A.RELIEVER_PITCH_ROLLING)
    if method == "zscore":
        st_l = zscore_labels(st_in, cfg)
        rp_l = zscore_labels(rp_in, cfg)
    else:
        st_l = st_in.copy()
        rp_l = rp_in.copy()
        st_l["condition_label"] = "neutral"
        rp_l["condition_label"] = "neutral"

    # 공통 IO 로그/샘플 저장
    log_step_io(
        "labeling",
        pd.concat([st_in, rp_in], ignore_index=True),
        pd.concat([st_l, rp_l], ignore_index=True),
    )
    save_debug_sample(paths, "labeling", df=pd.concat([st_in, rp_in], ignore_index=True), suffix="input")
    save_debug_sample(paths, "labeling", df=pd.concat([st_l, rp_l], ignore_index=True), suffix="output")

    # good/bad/neutral count + z_aggregate 분포 요약 저장
    def label_stats(df: pd.DataFrame, *, role: str) -> dict[str, Any]:
        counts = df["condition_label"].value_counts(dropna=False).to_dict() if "condition_label" in df.columns else {}
        z_desc = df["z_aggregate"].describe().to_dict() if "z_aggregate" in df.columns and df["z_aggregate"].notna().any() else {}
        # 빈 dict는 Parquet struct(자식 필드 없음)로 쓰일 때 Arrow 오류가 나므로 None으로 둔다.
        return {
            "role": role,
            "label_counts": counts if counts else None,
            "z_aggregate_desc": z_desc if z_desc else None,
        }

    st_stats = label_stats(st_l, role="starter")
    rp_stats = label_stats(rp_l, role="reliever")
    stats_df = pd.DataFrame([st_stats, rp_stats])
    stats_df.to_parquet(paths.output_tables_dir / "debug_labeling_label_and_z_stats.parquet", index=False)

    # role별 라벨 카운트 테이블(행 형태) 추가 저장
    def to_count_table(df: pd.DataFrame, *, role: str) -> pd.DataFrame:
        if df.empty or "condition_label" not in df.columns:
            return pd.DataFrame(columns=["role", "condition_label", "n"])
        return df["condition_label"].value_counts(dropna=False).rename_axis("condition_label").reset_index(name="n").assign(role=role)

    to_count_table(st_l, role="starter").to_parquet(
        paths.output_tables_dir / "debug_labeling_condition_label_counts_starter.parquet",
        index=False,
    )
    to_count_table(rp_l, role="reliever").to_parquet(
        paths.output_tables_dir / "debug_labeling_condition_label_counts_reliever.parquet",
        index=False,
    )

    st_l.to_parquet(paths.interim_dir / A.STARTER_PITCH_LABELED, index=False)
    rp_l.to_parquet(paths.interim_dir / A.RELIEVER_PITCH_LABELED, index=False)
    logger.info("라벨 저장: starter=%s reliever=%s", len(st_l), len(rp_l))
