"""Manifest-based execution for datasets of reference-image/video pairs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import PipelineConfig
from .phase2.pipeline import AppearanceAnalyzer, QwenVideoAnalyzer
from .pipeline import run_pipeline
from .schemas import FinalResult


@dataclass(frozen=True, slots=True)
class BatchItem:
    case_id: str
    reference_image: Path
    reference_video: Path


@dataclass(frozen=True, slots=True)
class BatchManifest:
    dataset_id: str
    items: list[BatchItem]


PipelineRunner = Callable[..., FinalResult]
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(manifest_path: str | Path) -> BatchManifest:
    """Load and validate a dataset manifest, resolving relative media paths."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Batch manifest must be a JSON object")

    dataset_id = str(payload.get("dataset_id", manifest_path.stem)).strip()
    raw_items = payload.get("items")
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Batch manifest must contain a non-empty items list")

    items: list[BatchItem] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Manifest item {index} must be a JSON object")
        case_id = str(raw_item.get("case_id", "")).strip()
        if not SAFE_CASE_ID.fullmatch(case_id):
            raise ValueError(
                f"Invalid case_id {case_id!r}; use letters, numbers, '.', '_', or '-'"
            )
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)

        try:
            image_value = raw_item["reference_image"]
            video_value = raw_item["reference_video"]
        except KeyError as error:
            raise ValueError(
                f"Manifest item {case_id!r} is missing {error.args[0]}"
            ) from error

        reference_image = Path(image_value)
        reference_video = Path(video_value)
        if not reference_image.is_absolute():
            reference_image = manifest_path.parent / reference_image
        if not reference_video.is_absolute():
            reference_video = manifest_path.parent / reference_video
        reference_image = reference_image.resolve()
        reference_video = reference_video.resolve()
        if not reference_image.is_file():
            raise FileNotFoundError(
                f"Reference image for {case_id!r} not found: {reference_image}"
            )
        if not reference_video.is_file():
            raise FileNotFoundError(
                f"Reference video for {case_id!r} not found: {reference_video}"
            )
        items.append(BatchItem(case_id, reference_image, reference_video))

    return BatchManifest(dataset_id=dataset_id, items=items)


def run_batch(
    manifest_path: str | Path,
    output_path: str | Path = "batch_results.json",
    debug_output_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    analyzer: AppearanceAnalyzer | None = None,
    pipeline_runner: PipelineRunner = run_pipeline,
) -> dict:
    """Run every manifest item and write combined public/debug JSON outputs."""
    manifest = load_manifest(manifest_path)
    config = config or PipelineConfig()
    output_path = Path(output_path).resolve()
    debug_output_path = (
        Path(debug_output_path).resolve()
        if debug_output_path
        else output_path.with_name(output_path.stem + ".debug.json")
    )
    item_output_root = output_path.parent / f"{output_path.stem}.items"

    # Reuse one lazily loaded Qwen instance for every present-person item.
    # It remains unloaded when every item exits after Phase 1.
    shared_analyzer = analyzer or QwenVideoAnalyzer(config.phase2)
    public_items: list[dict] = []
    debug_items: list[dict] = []

    for item in manifest.items:
        item_dir = item_output_root / item.case_id
        item_output = item_dir / "result.json"
        item_debug_output = item_dir / "result.debug.json"
        result = pipeline_runner(
            item.reference_image,
            item.reference_video,
            output_path=item_output,
            debug_output_path=item_debug_output,
            config=config,
            analyzer=shared_analyzer,
        )
        public_items.append({"case_id": item.case_id, **result.to_dict()})
        diagnostics = json.loads(item_debug_output.read_text(encoding="utf-8"))
        debug_items.append({"case_id": item.case_id, **diagnostics})

    public_payload = {
        "dataset_id": manifest.dataset_id,
        "item_count": len(public_items),
        "results": public_items,
    }
    debug_payload = {
        "dataset_id": manifest.dataset_id,
        "item_count": len(debug_items),
        "items": debug_items,
    }
    _write_json(output_path, public_payload)
    _write_json(debug_output_path, debug_payload)
    return public_payload
