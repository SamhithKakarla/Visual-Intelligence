"""Track-level face matching and disjoint target-appearance construction."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import Phase1Config
from ..schemas import (
    Appearance,
    Detection,
    EvidenceFrame,
    Phase1Result,
    TrackIdentityDecision,
)


def extract_search_frames(
    video_path: Path,
    frames_dir: Path,
    fps: float,
) -> list[Path]:
    """Decode a chronological, run-isolated frame sequence with FFmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame_%06d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-start_number",
            "0",
            str(output_pattern),
            "-hide_banner",
            "-loglevel",
            "error",
        ],
        check=True,
    )
    return sorted(frames_dir.glob("frame_*.jpg"))


def detect_and_track_people(
    frame_paths: list[Path],
    fps: float,
    config: Phase1Config,
) -> list[Detection]:
    """Run YOLO and persistent BoT-SORT over frames in chronological order."""
    from ultralytics import YOLO

    model = YOLO(config.yolo_model)
    detections: list[Detection] = []
    for frame_index, frame_path in enumerate(frame_paths):
        result = model.track(
            source=str(frame_path),
            persist=True,
            conf=config.person_confidence,
            iou=0.5,
            classes=[0],
            tracker=config.tracker,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for box_index, box in enumerate(result.boxes):
            if box.id is None:
                # Preserve the detection as a one-frame track without allowing
                # unrelated untracked detections to be merged.
                track_id = -((frame_index + 1) * 10_000 + box_index + 1)
            else:
                track_id = int(box.id.item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp=round(frame_index / fps, 4),
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=round(float(box.conf.item()), 4),
                    frame_path=str(frame_path),
                )
            )
    return detections


@dataclass(slots=True)
class FaceObservation:
    """Face metadata and ArcFace embedding returned by InsightFace."""

    embedding: Any
    bbox: tuple[float, float, float, float]
    detector_confidence: float
    width: float
    height: float


class ArcFaceEmbedder:
    """Lazy InsightFace wrapper so absent/error paths remain lightweight."""

    def __init__(self) -> None:
        self._app = None

    def _get_app(self):
        if self._app is None:
            import onnxruntime
            from insightface.app import FaceAnalysis

            available = onnxruntime.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            self._app = FaceAnalysis(name="buffalo_l", providers=providers)
            self._app.prepare(
                ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1,
                det_size=(640, 640),
            )
        return self._app

    def analyze_image(self, image) -> FaceObservation | None:
        """Return the largest detected face with its quality metadata."""
        if image is None or image.size == 0:
            return None
        faces = self._get_app().get(image)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1]),
        )
        x1, y1, x2, y2 = (float(value) for value in largest.bbox)
        embedding = getattr(largest, "normed_embedding", None)
        if embedding is None:
            return None
        return FaceObservation(
            embedding=embedding,
            bbox=(x1, y1, x2, y2),
            detector_confidence=float(getattr(largest, "det_score", 0.0)),
            width=max(0.0, x2 - x1),
            height=max(0.0, y2 - y1),
        )

    def embed_image(self, image):
        """Return a normalized embedding for compatibility with reference callers."""
        observation = self.analyze_image(image)
        return observation.embedding if observation is not None else None


def _crop(image, bbox: tuple[int, int, int, int], margin_ratio: float = 0.0):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    return image[y1:y2, x1:x2]


def _cosine_similarity(left, right) -> float:
    import math

    dot = sum(l * r for l, r in zip(left, right))
    denominator = math.sqrt(sum(l * l for l in left)) * math.sqrt(sum(r * r for r in right))
    if denominator == 0:
        return -1.0
    return float(dot / denominator)


def aggregate_similarities(scores: Iterable[float], top_k: int) -> float:
    """Use a top-k mean to reduce one-frame false matches."""
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return -1.0
    selected = ordered[: max(1, top_k)]
    return sum(selected) / len(selected)


def embed_reference_image(
    reference_image: Path,
    embedder: ArcFaceEmbedder,
):
    """Embed the single reference face used for every track comparison."""
    import cv2

    reference = cv2.imread(str(reference_image))
    embedding = embedder.embed_image(reference)
    if embedding is None:
        raise ValueError(f"No face detected in reference image: {reference_image}")
    return embedding


def score_tracks(
    reference_image: Path,
    detections: list[Detection],
    config: Phase1Config,
    embedder: ArcFaceEmbedder | None = None,
) -> tuple[dict[int, TrackIdentityDecision], list[str]]:
    """Make a quality-gated, multi-frame identity decision for every track."""
    import cv2

    embedder = embedder or ArcFaceEmbedder()
    reference_embedding = embed_reference_image(reference_image, embedder)

    by_track: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_track[detection.track_id].append(detection)

    warnings: list[str] = []
    track_decisions: dict[int, TrackIdentityDecision] = {}
    for track_id, track_detections in by_track.items():
        ordered = sorted(track_detections, key=lambda item: item.timestamp)
        candidates = select_evenly_spaced(
            ordered,
            config.max_identity_observations_per_track,
        )
        eligible: list[tuple[float, float]] = []
        faces_detected = 0
        rejected_no_face = 0
        rejected_too_small = 0
        rejected_low_confidence = 0
        for detection in candidates:
            frame = cv2.imread(detection.frame_path)
            if frame is None:
                warnings.append(
                    f"Track {track_id} frame could not be read: {detection.frame_path}"
                )
                continue
            person = _crop(frame, detection.bbox, margin_ratio=0.05)
            face = embedder.analyze_image(person)
            if face is None:
                rejected_no_face += 1
                continue
            faces_detected += 1
            if min(face.width, face.height) < config.minimum_face_dimension:
                rejected_too_small += 1
                continue
            if face.detector_confidence < config.minimum_face_detector_confidence:
                rejected_low_confidence += 1
                continue
            similarity = _cosine_similarity(reference_embedding, face.embedding)
            detection.identity_similarity = round(similarity, 4)
            eligible.append((similarity, detection.timestamp))

        selected = sorted(eligible, key=lambda item: item[0], reverse=True)[
            : config.similarity_top_k
        ]
        selected_scores = [score for score, _ in selected]
        selected_timestamps = [timestamp for _, timestamp in selected]
        aggregate_score = aggregate_similarities(
            (score for score, _ in eligible),
            config.similarity_top_k,
        )
        scores_above_threshold = sum(
            score >= config.identity_threshold for score in selected_scores
        )

        if len(eligible) < config.minimum_consensus_faces:
            matched = False
            decision_reason = "insufficient_faces"
        elif aggregate_score < config.identity_threshold:
            matched = False
            decision_reason = "below_average_threshold"
        elif scores_above_threshold < config.minimum_scores_above_threshold:
            matched = False
            decision_reason = "insufficient_scores_above_threshold"
        else:
            matched = True
            decision_reason = "matched_consensus"

        if not eligible:
            warnings.append(f"Track {track_id} had no quality-eligible face observations")
        track_decisions[track_id] = TrackIdentityDecision(
            track_id=track_id,
            matched=matched,
            decision_reason=decision_reason,
            aggregate_score=round(aggregate_score, 4),
            total_detections=len(track_detections),
            observations_examined=len(candidates),
            faces_detected=faces_detected,
            eligible_faces=len(eligible),
            rejected_no_face=rejected_no_face,
            rejected_too_small=rejected_too_small,
            rejected_low_confidence=rejected_low_confidence,
            selected_scores=[round(score, 4) for score in selected_scores],
            selected_timestamps=selected_timestamps,
        )
    return track_decisions, warnings


def build_appearance_groups(
    detections: list[Detection],
    gap_seconds: float,
) -> list[list[Detection]]:
    """Split matches into disjoint temporal appearances.

    Track IDs are intentionally not used as the only boundary: the same
    identity can receive a new tracker ID after an occlusion, while duplicated
    matched tracks at the same moment should still describe one appearance.
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: (item.timestamp, item.track_id))
    groups: list[list[Detection]] = [[ordered[0]]]
    latest_timestamp = ordered[0].timestamp
    for detection in ordered[1:]:
        if detection.timestamp - latest_timestamp > gap_seconds:
            groups.append([detection])
        else:
            groups[-1].append(detection)
        latest_timestamp = max(latest_timestamp, detection.timestamp)
    return groups


def select_evenly_spaced(items: list[Detection], maximum: int) -> list[Detection]:
    if maximum <= 0:
        return []
    if len(items) <= maximum:
        return items
    if maximum == 1:
        return [items[len(items) // 2]]
    indices = {
        round(index * (len(items) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [items[index] for index in sorted(indices)]


def create_evidence_frames(
    group: list[Detection],
    appearance_id: int,
    output_dir: Path,
    config: Phase1Config,
) -> list[EvidenceFrame]:
    """Write context-preserving target frames and expanded person crops."""
    import cv2

    context_dir = output_dir / "context"
    crop_dir = output_dir / "targets"
    context_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # At most one box per tracker and timestamp is retained. Duplicate target
    # tracks can otherwise waste the VLM frame budget.
    unique: dict[tuple[float, int], Detection] = {}
    for detection in group:
        unique[(detection.timestamp, detection.track_id)] = detection
    selected = select_evenly_spaced(
        sorted(unique.values(), key=lambda item: item.timestamp),
        config.max_evidence_frames,
    )

    evidence: list[EvidenceFrame] = []
    for sequence_index, detection in enumerate(selected):
        frame = cv2.imread(detection.frame_path)
        if frame is None:
            continue
        annotated = frame.copy()
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            annotated,
            "TARGET",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        stem = f"appearance_{appearance_id:03d}_{sequence_index:04d}"
        context_path = context_dir / f"{stem}.jpg"
        crop_path = crop_dir / f"{stem}.jpg"
        cv2.imwrite(str(context_path), annotated)
        cv2.imwrite(
            str(crop_path),
            _crop(frame, detection.bbox, config.crop_margin_ratio),
        )
        evidence.append(
            EvidenceFrame(
                frame_index=detection.frame_index,
                timestamp=detection.timestamp,
                bbox=detection.bbox,
                detection_confidence=detection.confidence,
                identity_similarity=detection.identity_similarity,
                context_frame_path=str(context_path),
                target_crop_path=str(crop_path),
            )
        )
    return evidence


def run_phase1(
    reference_image: str | Path,
    video_path: str | Path,
    run_directory: str | Path,
    config: Phase1Config | None = None,
    embedder: ArcFaceEmbedder | None = None,
) -> Phase1Result:
    """Run person discovery, identity matching, and evidence construction."""
    config = config or Phase1Config()
    reference_image = Path(reference_image).resolve()
    video_path = Path(video_path).resolve()
    run_directory = Path(run_directory).resolve()
    if not reference_image.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = extract_search_frames(
        video_path,
        run_directory / "search_frames",
        config.search_fps,
    )
    detections = detect_and_track_people(frames, config.search_fps, config)
    if not detections:
        return Phase1Result(
            reference_image=str(reference_image),
            reference_video=str(video_path),
            person_exists=False,
            identity_score=-1.0,
            threshold=config.identity_threshold,
            run_directory=str(run_directory),
            warnings=["No people were detected in the sampled video frames"],
        )

    track_decisions, warnings = score_tracks(
        reference_image, detections, config, embedder=embedder
    )
    matched_track_ids = {
        track_id
        for track_id, decision in track_decisions.items()
        if decision.matched
    }
    best_track_candidates = matched_track_ids or set(track_decisions)
    best_track_id = (
        max(
            best_track_candidates,
            key=lambda track_id: track_decisions[track_id].aggregate_score,
        )
        if best_track_candidates
        else None
    )
    best_score = (
        track_decisions[best_track_id].aggregate_score
        if best_track_id is not None
        else -1.0
    )
    if not matched_track_ids:
        return Phase1Result(
            reference_image=str(reference_image),
            reference_video=str(video_path),
            person_exists=False,
            identity_score=round(best_score, 4),
            threshold=config.identity_threshold,
            best_track_id=best_track_id,
            run_directory=str(run_directory),
            warnings=warnings,
            track_identity_decisions=list(track_decisions.values()),
        )

    matched_detections = [
        detection
        for detection in detections
        if detection.track_id in matched_track_ids
    ]
    groups = build_appearance_groups(
        matched_detections, config.appearance_gap_seconds
    )
    appearances: list[Appearance] = []
    for appearance_id, group in enumerate(groups, start=1):
        frames_for_vlm = create_evidence_frames(
            group, appearance_id, run_directory / "phase2_inputs", config
        )
        start = max(
            0.0,
            min(item.timestamp for item in group) - config.context_padding_seconds,
        )
        end = max(item.timestamp for item in group) + config.context_padding_seconds
        appearances.append(
            Appearance(
                appearance_id=appearance_id,
                start_time=round(start, 4),
                end_time=round(end, 4),
                track_ids=sorted({item.track_id for item in group}),
                frames=frames_for_vlm,
            )
        )

    return Phase1Result(
        reference_image=str(reference_image),
        reference_video=str(video_path),
        person_exists=True,
        identity_score=round(best_score, 4),
        threshold=config.identity_threshold,
        appearances=appearances,
        best_track_id=best_track_id,
        run_directory=str(run_directory),
        warnings=warnings,
        track_identity_decisions=list(track_decisions.values()),
    )
