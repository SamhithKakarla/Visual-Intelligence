"""Command-line interface for one pair or a dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import run_batch
from .config import Phase1Config, Phase2Config, PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find a reference person in a video and analyze their visible activity."
    )
    parser.add_argument("reference_image", type=Path, nargs="?")
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON dataset manifest; cannot be combined with positional media paths.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--search-fps", type=float, default=4.0)
    parser.add_argument("--identity-threshold", type=float, default=0.4)
    parser.add_argument("--max-evidence-frames", type=int, default=48)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Remove intermediate evidence frames after writing results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.manifest and (args.reference_image or args.video):
        parser.error("--manifest cannot be combined with positional media paths")
    if not args.manifest and not (args.reference_image and args.video):
        parser.error("provide reference_image and video, or use --manifest")
    config = PipelineConfig(
        phase1=Phase1Config(
            search_fps=args.search_fps,
            identity_threshold=args.identity_threshold,
            yolo_model=args.yolo_model,
            max_evidence_frames=args.max_evidence_frames,
        ),
        phase2=Phase2Config(model_id=args.vlm_model),
        runs_dir=args.runs_dir,
        keep_artifacts=not args.delete_artifacts,
    )
    if args.manifest:
        result = run_batch(
            args.manifest,
            output_path=args.output or Path("batch_results.json"),
            debug_output_path=args.debug_output,
            config=config,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = run_pipeline(
            args.reference_image,
            args.video,
            output_path=args.output or Path("result.json"),
            debug_output_path=args.debug_output,
            config=config,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0
