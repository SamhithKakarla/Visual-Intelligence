"""Conditional orchestration of identity matching and activity analysis."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Callable

from .config import PipelineConfig
from .phase1.pipeline import ArcFaceEmbedder, run_phase1
from .phase2.pipeline import AppearanceAnalyzer, run_phase2
from .schemas import FinalResult, Phase1Result


Phase1Runner = Callable[..., Phase1Result]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finalize_from_phase1(
    phase1_result: Phase1Result,
    analyzer: AppearanceAnalyzer | None = None,
    config: PipelineConfig | None = None,
) -> tuple[FinalResult, dict]:
    """Apply the Phase 1 gate and return public plus diagnostic results."""
    config = config or PipelineConfig()
    if not phase1_result.person_exists:
        final = FinalResult(
            reference_image=phase1_result.reference_image,
            reference_video=phase1_result.reference_video,
            person_exists=False,
            activity_description=None,
            activity_classification=None,
        )
        return final, {"phase1": phase1_result.to_dict(), "phase2": None}

    phase2_result = run_phase2(
        phase1_result,
        analyzer=analyzer,
        config=config.phase2,
    )
    final = FinalResult(
        reference_image=phase1_result.reference_image,
        reference_video=phase1_result.reference_video,
        person_exists=True,
        activity_description=phase2_result.activity_description,
        activity_classification=phase2_result.activity_classification,
    )
    return final, {
        "phase1": phase1_result.to_dict(),
        "phase2": phase2_result.to_dict(),
    }


def run_pipeline(
    reference_image: str | Path,
    video_path: str | Path,
    output_path: str | Path = "result.json",
    debug_output_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    analyzer: AppearanceAnalyzer | None = None,
    embedder: ArcFaceEmbedder | None = None,
    phase1_runner: Phase1Runner = run_phase1,
) -> FinalResult:
    """Run the complete pipeline and persist stable public/debug JSON files."""
    config = config or PipelineConfig()
    reference_image = Path(reference_image).resolve()
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    debug_output_path = (
        Path(debug_output_path).resolve()
        if debug_output_path
        else output_path.with_name(output_path.stem + ".debug.json")
    )

    run_id = f"{video_path.stem}-{uuid.uuid4().hex[:10]}"
    run_directory = config.runs_dir.resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    try:
        phase1_result = phase1_runner(
            reference_image,
            video_path,
            run_directory,
            config.phase1,
            embedder=embedder,
        )
        final, diagnostics = finalize_from_phase1(
            phase1_result,
            analyzer=analyzer,
            config=config,
        )
        diagnostics["run_id"] = run_id
        _write_json(output_path, final.to_dict())
        _write_json(debug_output_path, diagnostics)
        return final
    finally:
        if not config.keep_artifacts:
            shutil.rmtree(run_directory, ignore_errors=True)
