"""Command-line interface for one reference-image/video pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Phase1Config, Phase2Config, PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find a reference person in a video and analyze their visible activity."
    )
    parser.add_argument("reference_image", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--search-fps", type=float, default=4.0)
    parser.add_argument("--identity-threshold", type=float, default=0.4)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Remove intermediate evidence frames after writing results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        phase1=Phase1Config(
            search_fps=args.search_fps,
            identity_threshold=args.identity_threshold,
            yolo_model=args.yolo_model,
        ),
        phase2=Phase2Config(model_id=args.vlm_model),
        runs_dir=args.runs_dir,
        keep_artifacts=not args.delete_artifacts,
    )
    result = run_pipeline(
        args.reference_image,
        args.video,
        output_path=args.output,
        debug_output_path=args.debug_output,
        config=config,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0
