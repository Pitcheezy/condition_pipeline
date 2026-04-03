"""전체 파이프라인 실행 순서만 관리한다."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

import aggregate
import baseline
import collect
import condition_features
import config_paths
import decision
import filter_pitchers
import labeling
import outcome_features
import plots
import quadrant_analysis
import rolling_features
import scoring
import split_pitchers
import threshold_analysis
import trend

logger = logging.getLogger(__name__)


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or Path("config/config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def run_all(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    root = project_root or Path.cwd()
    paths = config_paths.project_paths_from_config(config, root)
    config_paths.ensure_dirs(paths)
    setup_logging(paths.output_logs_dir)

    collect.run(config, paths)
    split_pitchers.run(config, paths)
    filter_pitchers.run(config, paths)
    baseline.run(config, paths)
    condition_features.run(config, paths)
    outcome_features.run(config, paths)
    rolling_features.run(config, paths)
    labeling.run(config, paths)
    quadrant_analysis.run(config, paths)
    plots.run(config, paths)
    threshold_analysis.run(config, paths)
    aggregate.run(config, paths)
    scoring.run(config, paths)
    trend.run(config, paths)
    decision.run(config, paths)
    logger.info("파이프라인 완료")
