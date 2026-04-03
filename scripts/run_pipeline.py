#!/usr/bin/env python3
"""파이프라인 실행 진입점. 프로젝트 루트에서 실행하는 것을 권장한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 넣어 src 모듈을 찾는다.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="condition_pipeline 실행")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config" / "config.yaml",
        help="config.yaml 경로",
    )
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    pipeline.run_all(config, project_root=_ROOT)


if __name__ == "__main__":
    main()
