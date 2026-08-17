import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
import re
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


# ---------------------------------------------------------------------------
# Dataset / manifest helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def safe_extract(zip_path: Path, extract_dir: Path) -> None:
    """Safely extract a ZIP file and prevent path traversal (zip-slip)."""
    extract_dir = Path(extract_dir).resolve()

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target = (extract_dir / member.filename).resolve()
            if not str(target).startswith(str(extract_dir)):
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        z.extractall(extract_dir)


def find_dataset_root(extract_dir: Path) -> Optional[Path]:
    """
    Finds the folder containing query/, video/, manifest.csv.

    Supports both a flat layout and one wrapped in an outer folder.
    """
    extract_dir = Path(extract_dir)

    if (
        (extract_dir / "query").is_dir()
        and (extract_dir / "video").is_dir()
        and (extract_dir / "manifest.csv").is_file()
    ):
        return extract_dir

    for manifest_path in extract_dir.rglob("manifest.csv"):
        parent = manifest_path.parent
        if (parent / "query").is_dir() and (parent / "video").is_dir():
            return parent

    return None


def find_matching_file(directory: Path, value, allowed_extensions: set) -> Optional[Path]:
    """
    Resolve CSV values to actual files.

    Examples:
        1            -> person1.png
        person1      -> person1.png
        person1.png  -> person1.png
        100          -> 100.mp4
        100.mp4      -> 100.mp4
    """
    directory = Path(directory)
    value = str(value).strip()

    if not value:
        return None

    exact = directory / value
    if exact.is_file():
        return exact

    for ext in allowed_extensions:
        candidate = directory / f"{value}{ext}"
        if candidate.is_file():
            return candidate

    value_stem = Path(value).stem.lower()

    for file in directory.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in allowed_extensions:
            continue
        if file.stem.lower() == value_stem:
            return file

    normalized_value = re.sub(r"[^a-zA-Z0-9]", "", value_stem).lower()

    for file in directory.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in allowed_extensions:
            continue

        normalized_stem = re.sub(r"[^a-zA-Z0-9]", "", file.stem).lower()

        if normalized_stem == normalized_value:
            return file

        # person1 -> 1
        if normalized_stem.endswith(normalized_value):
            prefix = normalized_stem[: -len(normalized_value)]
            if prefix in {"person", "target", "query"}:
                return file

    return None


def process_manifest(dataset_root: Path):
    """
    Converts manifest.csv (query_image,video) into the JSON manifest expected
    by run_batch(), and writes it to <dataset_root>/streamlit_manifest.json.

    Returns (dataframe, manifest_dict, errors_list).
    """
    dataset_root = Path(dataset_root)
    query_dir = dataset_root / "query"
    video_dir = dataset_root / "video"
    csv_path = dataset_root / "manifest.csv"

    errors = []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, None, [f"Could not read manifest.csv: {e}"]

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "query_image" not in df.columns:
        errors.append("manifest.csv must contain `query_image`.")
    if "video" not in df.columns:
        errors.append("manifest.csv must contain `video`.")

    if errors:
        return df, None, errors

    if df.empty:
        return df, None, ["manifest.csv contains no rows."]

    items = []

    for index, row in df.iterrows():
        csv_row = index + 2

        query_value = row["query_image"]
        video_value = row["video"]

        if pd.isna(query_value):
            errors.append(f"CSV row {csv_row}: query_image is empty.")
            continue

        if pd.isna(video_value):
            errors.append(f"CSV row {csv_row}: video is empty.")
            continue

        image_path = find_matching_file(query_dir, query_value, IMAGE_EXTENSIONS)
        if image_path is None:
            errors.append(f"CSV row {csv_row}: Could not find query image for '{query_value}'.")
            continue

        video_path = find_matching_file(video_dir, video_value, VIDEO_EXTENSIONS)
        if video_path is None:
            errors.append(f"CSV row {csv_row}: Could not find video for '{video_value}'.")
            continue

        # Row number in the CSV (1-based) is used as-is for the case ID.
        case_id = str(csv_row - 1)

        relative_image = image_path.relative_to(dataset_root)
        relative_video = video_path.relative_to(dataset_root)

        items.append(
            {
                "case_id": case_id,
                "reference_image": str(relative_image),
                "reference_video": str(relative_video),
            }
        )

    if errors:
        return df, None, errors

    manifest = {"dataset_id": dataset_root.name, "items": items}

    manifest_path = dataset_root / "streamlit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return df, manifest, []


def create_results_zip(dataset_root: Path) -> bytes:
    """Bundle manifest + results + debug + per-item artifacts + run artifacts into a ZIP."""
    dataset_root = Path(dataset_root)
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        files = [
            "streamlit_manifest.json",
            "batch_results.json",
            "batch_results.debug.json",
        ]
        for filename in files:
            path = dataset_root / filename
            if path.is_file():
                z.write(path, arcname=filename)

        items_dir = dataset_root / "batch_results.items"
        if items_dir.is_dir():
            for file in items_dir.rglob("*"):
                if not file.is_file():
                    continue
                arcname = Path("batch_results.items") / file.relative_to(items_dir)
                z.write(file, arcname=str(arcname))

        # Phase 1 / Phase 2 intermediate artifacts (frames, crops, per-case
        # working dirs, etc.), written by run_batch() when keep_artifacts=True.
        runs_dir = dataset_root / "visual_intelligence_runs"
        if runs_dir.is_dir():
            for file in runs_dir.rglob("*"):
                if not file.is_file():
                    continue
                arcname = Path("visual_intelligence_runs") / file.relative_to(runs_dir)
                z.write(file, arcname=str(arcname))

    buffer.seek(0)
    return buffer.getvalue()


try:
    from visual_intelligence.batch import run_batch
    from visual_intelligence.config import Phase1Config, Phase2Config, PipelineConfig
except Exception as e:
    run_batch = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None

app = FastAPI(
    title="Visual Intelligence API",
    description="Person Recognition → Tracking → Activity Analysis. "
    "Use POST /upload first to get a session_id, then POST /run/{session_id}, "
    "then GET /download/{session_id}.",
)

# session_id -> dataset_root (Path)
SESSIONS: dict[str, Path] = {}

# session_id -> most recent run_batch() result
RESULTS: dict[str, dict] = {}


@app.post("/upload", summary="Upload a dataset ZIP (query/, video/, manifest.csv)")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    temp_dir = Path(tempfile.mkdtemp(prefix="vi_"))
    zip_path = temp_dir / "upload.zip"
    zip_path.write_bytes(await file.read())

    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir()
    try:
        safe_extract(zip_path, extract_dir)
    except Exception as e:
        raise HTTPException(400, f"Could not extract ZIP: {e}")

    dataset_root = find_dataset_root(extract_dir)
    if dataset_root is None:
        raise HTTPException(400, "ZIP must contain query/, video/, and manifest.csv.")

    df, manifest, errors = process_manifest(dataset_root)
    if errors:
        raise HTTPException(422, "; ".join(errors))

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = dataset_root

    query_dir, video_dir = dataset_root / "query", dataset_root / "video"

    return {
        "session_id": session_id,
        "dataset_name": dataset_root.name,
        "image_count": len([p for p in query_dir.iterdir() if p.is_file()]),
        "video_count": len([p for p in video_dir.iterdir() if p.is_file()]),
        "pairs": manifest["items"],
        "pipeline_available": IMPORT_ERROR is None,
        "pipeline_import_error": IMPORT_ERROR,
    }


@app.post("/run/{session_id}", summary="Run Phase 1 + Phase 2 for a previously uploaded session")
def run(
    session_id: str,
    search_fps: float = Form(2.0),
    identity_threshold: float = Form(0.4),
    max_evidence_frames: int = Form(48),
    keep_artifacts: bool = Form(True),
):
    if IMPORT_ERROR:
        raise HTTPException(500, f"Pipeline unavailable: {IMPORT_ERROR}")

    dataset_root = SESSIONS.get(session_id)
    if dataset_root is None:
        raise HTTPException(404, "Unknown session_id. Upload a dataset first.")

    config = PipelineConfig(
        phase1=Phase1Config(
            search_fps=search_fps,
            identity_threshold=identity_threshold,
            max_evidence_frames=max_evidence_frames,
        ),
        phase2=Phase2Config(model_id="Qwen/Qwen3-VL-2B-Instruct"),
        runs_dir=dataset_root / "visual_intelligence_runs",
        keep_artifacts=keep_artifacts,
    )

    try:
        result = run_batch(
            manifest_path=dataset_root / "streamlit_manifest.json",
            output_path=dataset_root / "batch_results.json",
            debug_output_path=dataset_root / "batch_results.debug.json",
            config=config,
        )
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")

    RESULTS[session_id] = result
    return result


@app.get("/download/{session_id}", summary="Download a ZIP of manifest + results + artifacts")
def download(session_id: str):
    dataset_root = SESSIONS.get(session_id)
    if dataset_root is None:
        raise HTTPException(404, "Unknown session_id.")

    zip_bytes = create_results_zip(dataset_root)
    out_path = dataset_root / "results.zip"
    out_path.write_bytes(zip_bytes)
    return FileResponse(out_path, filename="visual_intelligence_results.zip")