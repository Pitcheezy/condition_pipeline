"""시각화: 기준 검증용 진단 플롯 (scatter / threshold binning / quadrant / trend)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import artifacts as A
from config_paths import ProjectPaths, ensure_dirs

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], paths: ProjectPaths) -> None:
    ensure_dirs(paths)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib 미설치 — plots 스킵")
        return

    # config/threshold 후보 확인을 위한 파라미터
    thr_cfg = config.get("threshold", {})
    n_bins = int(thr_cfg.get("n_bins", 20))
    min_samples_per_bin = int(thr_cfg.get("min_samples_per_bin", 30))

    def _read(path: Any) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    # -----------------------------
    # 1) delta vs outcome scatter plot
    # -----------------------------
    # x: delta_release_speed, delta_release_spin_rate (요청의 delta_spin_rate 대응)
    # y: xwoba 또는 rolling whiff rate(가능하면)
    def plot_delta_outcome_scatter(role: str) -> None:
        if role == "starter":
            delta_p = paths.interim_dir / A.STARTER_PITCH_DELTA
            out_p = paths.interim_dir / A.STARTER_PITCH_OUTCOMES
            roll_p = paths.interim_dir / A.STARTER_PITCH_ROLLING
        else:
            delta_p = paths.interim_dir / A.RELIEVER_PITCH_DELTA
            out_p = paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES
            roll_p = paths.interim_dir / A.RELIEVER_PITCH_ROLLING

        d = _read(delta_p)
        o = _read(out_p)
        r = _read(roll_p)
        if d.empty or o.empty:
            logger.warning("scatter 입력 누락(role=%s): delta/or outcome parquet 없음", role)
            return

        # 가능한 조인 키만 자동 선택
        preferred_keys = ["game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number", "reliever_outing_unit_id"]
        keys = [k for k in preferred_keys if k in d.columns and k in o.columns]
        if not keys:
            logger.warning("scatter join keys 없음(role=%s) — 인덱스 기반 merge 스킵", role)
            return

        base = d.merge(o, on=keys, how="inner", suffixes=("", "_out"))
        if not r.empty:
            r_keys = [k for k in ["pitcher", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number", "reliever_outing_unit_id"] if k in base.columns and k in r.columns]
            if r_keys:
                base = base.merge(r, on=r_keys, how="left", suffixes=("", "_roll"))

        # y 우선순위: rolling_whiff_rate_10 -> xwoba
        y_whiff_col = "rolling_whiff_rate_10" if "rolling_whiff_rate_10" in base.columns else None
        y_xwoba_col = "xwoba" if "xwoba" in base.columns else None
        if y_xwoba_col is None and y_whiff_col is None:
            logger.warning("scatter y 컬럼 없음(role=%s): xwoba/rolling_whiff_rate_10 둘 다 없음", role)
            return

        x1 = "delta_release_speed"
        x2 = "delta_release_spin_rate"
        if x1 not in base.columns or x2 not in base.columns:
            logger.warning("scatter x 컬럼 없음(role=%s): %s 또는 %s 없음", role, x1, x2)
            return

        # y 컬럼 선택
        y_col = y_xwoba_col if y_xwoba_col is not None else y_whiff_col
        y_label = "estimated_woba_using_speedangle (xwoba)" if y_col == y_xwoba_col else "whiff_rate (rolling_whiff_rate_10)"

        # scatter 2개를 한 figure로
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        sample = base[[x1, x2, y_col]].copy()
        sample = sample.dropna(subset=[x1, x2, y_col])
        if len(sample) > 30_000:
            sample = sample.sample(30_000, random_state=42)

        axes[0].scatter(pd.to_numeric(sample[x1], errors="coerce"), pd.to_numeric(sample[y_col], errors="coerce"), s=6, alpha=0.25)
        axes[0].set_xlabel("delta_release_speed")
        axes[0].set_ylabel(y_label)
        axes[0].set_title(f"{role}: delta_release_speed vs {y_label}")

        axes[1].scatter(pd.to_numeric(sample[x2], errors="coerce"), pd.to_numeric(sample[y_col], errors="coerce"), s=6, alpha=0.25, color="orange")
        axes[1].set_xlabel("delta_release_spin_rate")
        axes[1].set_title(f"{role}: delta_release_spin_rate vs {y_label}")

        out = paths.output_figures_dir / f"debug_plot1_delta_outcome_scatter_{role}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots 저장: %s", out)

    plot_delta_outcome_scatter("starter")
    plot_delta_outcome_scatter("reliever")

    # -----------------------------
    # 2) binning 기반 threshold plot
    # -----------------------------
    def plot_threshold_bins(role: str) -> None:
        if role == "starter":
            d = _read(paths.interim_dir / A.STARTER_PITCH_DELTA)
            o = _read(paths.interim_dir / A.STARTER_PITCH_OUTCOMES)
            labeled = _read(paths.interim_dir / A.STARTER_PITCH_LABELED)
        else:
            d = _read(paths.interim_dir / A.RELIEVER_PITCH_DELTA)
            o = _read(paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES)
            labeled = _read(paths.interim_dir / A.RELIEVER_PITCH_LABELED)

        if d.empty or o.empty or labeled.empty:
            logger.warning("threshold plot 입력 누락(role=%s): delta/outcome/labeled", role)
            return

        preferred_keys = ["game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number", "reliever_outing_unit_id"]
        keys = [k for k in preferred_keys if k in d.columns and k in o.columns and k in labeled.columns]
        if not keys:
            logger.warning("threshold plot join keys 없음(role=%s)", role)
            return

        # outcomes에는 delta_*를 넣지 않는 것이 원칙이나, 방어적으로 suffix 지정
        base = d.merge(o, on=keys, how="inner", suffixes=("", "_out")).merge(
            labeled[["z_aggregate", "condition_label"] + keys],
            on=keys,
            how="inner",
        )

        if "delta_release_speed" not in base.columns:
            logger.warning("threshold plot delta_release_speed 없음(role=%s)", role)
            return

        # outcome: xwoba 또는 whiff_rate proxy(rolling이 없으면 is_whiff 0/1)
        if "xwoba" in base.columns:
            outcome_col = "xwoba"
            outcome_label = "xwoba"
        elif "is_whiff" in base.columns:
            outcome_col = "is_whiff"
            outcome_label = "is_whiff (0/1)"
        else:
            logger.warning("threshold plot outcome 컬럼 없음(role=%s)", role)
            return

        good_label_col = "condition_label" if "condition_label" in base.columns else None

        cols = ["delta_release_speed", outcome_col, "pitcher"]
        if good_label_col is not None:
            cols.append(good_label_col)

        tmp = base[cols].copy()
        tmp = tmp.dropna(subset=["delta_release_speed", outcome_col])
        if len(tmp) < 100:
            logger.warning("threshold plot 데이터 부족(role=%s): %s행", role, len(tmp))
            return

        # 구간 분할(등간이 아니라 quantile 기반에 가까운 bins)
        s = pd.to_numeric(tmp["delta_release_speed"], errors="coerce").dropna()
        if s.empty:
            return

        # pd.qcut이 rank 기반이라 outlier에 강함. 다만 중복 bin edge 이슈가 있으면 linspace로 대체.
        try:
            cats = pd.qcut(tmp["delta_release_speed"], q=n_bins, duplicates="drop")
        except Exception:
            lo, hi = float(tmp["delta_release_speed"].min()), float(tmp["delta_release_speed"].max())
            edges = np.linspace(lo, hi, n_bins + 1)
            cats = pd.cut(tmp["delta_release_speed"], bins=edges, include_lowest=True)

        tmp["delta_bin"] = cats.astype(str)
        grp = tmp.groupby("delta_bin", observed=False).agg(
            n=("pitcher", "size"),
            mean_delta=("delta_release_speed", "mean"),
            mean_outcome=(outcome_col, "mean"),
        )
        grp = grp.reset_index()
        grp = grp.sort_values("mean_delta")
        grp = grp[grp["n"] >= min_samples_per_bin].copy()
        if grp.empty:
            logger.warning("threshold plot bin이 모두 min_samples 미만(role=%s)", role)
            return

        # 구간별 good rate(라벨 기반)도 같이 그리기
        if good_label_col is not None:
            good_rate = (
                tmp.assign(is_good=(tmp[good_label_col] == "good").astype(int))
                .groupby("delta_bin")["is_good"]
                .mean()
                .reset_index(name="good_rate")
            )
            grp = grp.merge(good_rate, on="delta_bin", how="left")
        else:
            grp["good_rate"] = np.nan

        fig, ax1 = plt.subplots(1, 1, figsize=(8.5, 4.8), constrained_layout=True)
        ax1.plot(grp["mean_delta"], grp["mean_outcome"], marker="o", linewidth=1.5)
        ax1.set_xlabel("delta_release_speed (bin mean)")
        ax1.set_ylabel(f"mean outcome: {outcome_label}")
        ax1.set_title(f"{role}: outcome by delta_release_speed bins")

        # threshold 후보 확인을 위해 good_rate를 보조축에 표시
        if not grp["good_rate"].isna().all():
            ax2 = ax1.twinx()
            ax2.plot(grp["mean_delta"], grp["good_rate"], marker="x", color="green", linewidth=1.2, alpha=0.9)
            ax2.set_ylabel("condition_label == good rate")

        out = paths.output_figures_dir / f"debug_plot2_threshold_binning_{role}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots 저장: %s", out)

    plot_threshold_bins("starter")
    plot_threshold_bins("reliever")

    # -----------------------------
    # 3) quadrant visualization: metric good/bad vs outcome good/bad
    # -----------------------------
    def plot_quadrants(role: str) -> None:
        if role == "starter":
            labeled = _read(paths.interim_dir / A.STARTER_PITCH_LABELED)
            outcomes = _read(paths.interim_dir / A.STARTER_PITCH_OUTCOMES)
        else:
            labeled = _read(paths.interim_dir / A.RELIEVER_PITCH_LABELED)
            outcomes = _read(paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES)

        if labeled.empty or outcomes.empty:
            logger.warning("quadrant plot 입력 누락(role=%s)", role)
            return

        preferred_keys = ["game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number", "reliever_outing_unit_id"]
        keys = [k for k in preferred_keys if k in labeled.columns and k in outcomes.columns]
        if not keys:
            logger.warning("quadrant plot join keys 없음(role=%s)", role)
            return

        base = labeled.merge(outcomes, on=keys, how="inner", suffixes=("", "_out"))
        if "delta_release_speed" not in base.columns or "condition_label" not in base.columns:
            logger.warning("quadrant plot 필수 컬럼 없음(role=%s)", role)
            return

        # metric: good/bad만 사용(중립 제거)
        base = base[base["condition_label"].isin(["good", "bad"])].copy()
        if base.empty:
            return

        # outcome 축: outcome_features에서 만든 outcome_label_quadrant 우선 (whiff 변별 없을 때 xwoba median)
        if "outcome_label_quadrant" in base.columns:
            base["outcome_label"] = base["outcome_label_quadrant"]
        elif "is_whiff" in base.columns:
            whiff = base["is_whiff"].astype(bool)
            if whiff.nunique() > 1:
                base["outcome_label"] = np.where(whiff, "outcome_good", "outcome_bad")
            elif "xwoba" in base.columns:
                xw_med = pd.to_numeric(base["xwoba"], errors="coerce").median()
                base["outcome_label"] = np.where(
                    pd.to_numeric(base["xwoba"], errors="coerce") <= xw_med,
                    "outcome_good",
                    "outcome_bad",
                )
            else:
                logger.warning("quadrant plot outcome 정의 불가(role=%s)", role)
                return
        elif "xwoba" in base.columns:
            xw_med = pd.to_numeric(base["xwoba"], errors="coerce").median()
            base["outcome_label"] = np.where(
                pd.to_numeric(base["xwoba"], errors="coerce") <= xw_med,
                "outcome_good",
                "outcome_bad",
            )
        else:
            logger.warning("quadrant plot outcome 컬럼 없음(role=%s)", role)
            return

        base["quad_key"] = base["condition_label"] + "_" + base["outcome_label"].map(
            {"outcome_good": "good", "outcome_bad": "bad"}
        )

        # scatter
        color_map = {
            "good_good": "green",
            "good_bad": "orange",
            "bad_good": "blue",
            "bad_bad": "red",
        }
        base["quad_key2"] = base["condition_label"] + "_" + np.where(base["outcome_label"] == "outcome_good", "good", "bad")
        base["color"] = base["quad_key2"].map(color_map).fillna("gray")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        sample = base.dropna(subset=["delta_release_speed"]).copy()
        if len(sample) > 25_000:
            sample = sample.sample(25_000, random_state=42)

        y_col = "xwoba" if "xwoba" in sample.columns else ("is_whiff" if "is_whiff" in sample.columns else None)
        if y_col is None:
            return

        axes[0].scatter(sample["delta_release_speed"], pd.to_numeric(sample[y_col], errors="coerce"), c=sample["color"], s=6, alpha=0.25)
        axes[0].set_xlabel("delta_release_speed")
        axes[0].set_ylabel(y_col)
        axes[0].set_title(f"{role}: metric good/bad vs outcome good/bad (colors)")

        # count heatmap
        comb = (
            base.groupby(["condition_label", "outcome_label"], observed=False)
            .size()
            .reset_index(name="n")
        )
        # pivot to heatmap
        pivot = comb.pivot_table(index="condition_label", columns="outcome_label", values="n", aggfunc="sum", fill_value=0)
        # normalize? keep raw
        im = axes[1].imshow(pivot.values, aspect="auto")
        axes[1].set_xticks(range(pivot.shape[1]))
        axes[1].set_yticks(range(pivot.shape[0]))
        axes[1].set_xticklabels([str(x) for x in pivot.columns], rotation=20)
        axes[1].set_yticklabels([str(x) for x in pivot.index])
        axes[1].set_title(f"{role}: count by (metric,outcome)")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        out = paths.output_figures_dir / f"debug_plot3_quadrant_metric_vs_outcome_{role}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots 저장: %s", out)

    plot_quadrants("starter")
    plot_quadrants("reliever")

    # -----------------------------
    # 4) starter inning trend plot: x inning, y score
    # -----------------------------
    def plot_starter_inning_trend() -> None:
        df = _read(paths.processed_dir / A.STARTER_INNING_SCORE)
        if df.empty or "inning" not in df.columns or "condition_score" not in df.columns:
            logger.warning("starter trend 입력 누락")
            return
        df = df.copy()
        df["inning"] = pd.to_numeric(df["inning"], errors="coerce")
        df = df.dropna(subset=["inning", "condition_score"])
        if df.empty:
            return

        grp = df.groupby("inning")["condition_score"]
        summary = grp.agg(
            mean="mean",
            p10=lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.10)),
            p90=lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.90)),
            n="size",
        ).reset_index()
        summary = summary.sort_values("inning")

        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8), constrained_layout=True)
        ax.plot(summary["inning"], summary["mean"], marker="o", linewidth=1.6, label="mean score")
        ax.fill_between(summary["inning"], summary["p10"], summary["p90"], alpha=0.18, label="p10-p90")
        ax.set_xlabel("inning")
        ax.set_ylabel("condition_score")
        ax.set_title("starter: condition_score trend by inning")
        ax.legend(loc="best")
        out = paths.output_figures_dir / "debug_plot4_starter_inning_trend_score.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots 저장: %s", out)

    plot_starter_inning_trend()

    # -----------------------------
    # 5) reliever outing/pitch trend plot: x pitch_number, y velocity or score
    # -----------------------------
    def plot_reliever_pitch_trend() -> None:
        # pitch-level velocity/whiff
        pitch_out = _read(paths.interim_dir / A.RELIEVER_PITCH_OUTCOMES)
        if pitch_out.empty:
            logger.warning("reliever pitch trend 입력 누락: reliever_pitch_outcomes")
            return

        out_score = _read(paths.processed_dir / A.RELIEVER_OUTING_SCORE)

        df = pitch_out.copy()
        # x axis
        if "pitch_number" in df.columns:
            df["pitch_number"] = pd.to_numeric(df["pitch_number"], errors="coerce")
        else:
            logger.warning("reliever pitch trend: pitch_number 컬럼 없음")
            return
        df = df.dropna(subset=["pitch_number"])

        # y1: velocity(=release_speed)
        if "release_speed" in df.columns:
            df["release_speed_num"] = pd.to_numeric(df["release_speed"], errors="coerce")
        # y2: score at outing level (optional join)
        if not out_score.empty and "reliever_outing_unit_id" in df.columns and "reliever_outing_unit_id" in out_score.columns and "condition_score" in out_score.columns:
            df = df.merge(out_score[["reliever_outing_unit_id", "condition_score"]], on="reliever_outing_unit_id", how="left")
        if "condition_score" in df.columns:
            df["condition_score_num"] = pd.to_numeric(df["condition_score"], errors="coerce")

        # whiff rate
        if "is_whiff" in df.columns:
            df["is_whiff_num"] = df["is_whiff"].astype(int)
        else:
            df["is_whiff_num"] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        # velocity trend
        if "release_speed_num" in df.columns:
            g1 = df.groupby("pitch_number")["release_speed_num"].agg(["mean", "count"]).reset_index()
            axes[0].plot(g1["pitch_number"], g1["mean"], marker="o")
            axes[0].set_title("reliever: avg velocity vs pitch_number")
            axes[0].set_xlabel("pitch_number")
            axes[0].set_ylabel("release_speed")
        else:
            axes[0].set_title("reliever: velocity unavailable")

        # score trend
        if "condition_score_num" in df.columns:
            g2 = df.dropna(subset=["condition_score_num"]).groupby("pitch_number")["condition_score_num"].mean().reset_index()
            axes[1].plot(g2["pitch_number"], g2["condition_score_num"], marker="o", color="purple")
            axes[1].set_title("reliever: avg outing score vs pitch_number")
            axes[1].set_xlabel("pitch_number")
            axes[1].set_ylabel("condition_score")
        else:
            axes[1].set_title("reliever: score unavailable")

        # whiff rate
        if "is_whiff_num" in df.columns and not pd.isna(df["is_whiff_num"]).all():
            g3 = df.dropna(subset=["is_whiff_num"]).groupby("pitch_number")["is_whiff_num"].mean().reset_index(name="whiff_rate")
            axes[2].plot(g3["pitch_number"], g3["whiff_rate"], marker="o", color="green")
            axes[2].set_title("reliever: whiff_rate vs pitch_number")
            axes[2].set_xlabel("pitch_number")
            axes[2].set_ylabel("whiff_rate")
        else:
            axes[2].set_title("reliever: whiff_rate unavailable")

        out = paths.output_figures_dir / "debug_plot5_reliever_pitch_trend_velocity_score_whiff.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots 저장: %s", out)

    plot_reliever_pitch_trend()
