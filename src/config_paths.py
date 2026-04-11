"""config.yaml의 paths를 절대 경로로 해석한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    output_tables_dir: Path
    output_figures_dir: Path
    output_logs_dir: Path


def project_paths_from_config(config: dict[str, Any], root: Path) -> ProjectPaths:
    p = config["paths"]
    return ProjectPaths(
        root=root,
        raw_dir=root / p["raw_dir"],
        interim_dir=root / p["interim_dir"],
        processed_dir=root / p["processed_dir"],
        output_tables_dir=root / p["output_tables_dir"],
        output_figures_dir=root / p["output_figures_dir"],
        output_logs_dir=root / p["output_logs_dir"],
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    for d in (
        paths.raw_dir,
        paths.interim_dir,
        paths.processed_dir,
        paths.output_tables_dir,
        paths.output_figures_dir,
        paths.output_logs_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
