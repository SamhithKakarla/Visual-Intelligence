"""Track-level face matching and disjoint target-appearance construction."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..config import Phase1Config
from ..schemas import Appearance, Detection, EvidenceFrame, Phase1Result


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


def estimate_camera_motion(frame_paths: list[Path], sample_pairs: int = 8) -> float:
    """Cheaply estimate whether the camera itself is moving.

    Measures global frame-to-frame displacement via phase correlation
    (a single FFT-based shift estimate, not a full optical-flow pass) on
    a handful of evenly spaced consecutive-frame pairs, at a small
    downsampled scale. Returns the median displacement in pixels at that
    downsampled scale -- near zero for a genuinely fixed/mounted camera,
    and meaningfully positive for any handheld, panning, or otherwise
    moving camera. This decides only whether tracking bothers to run
    global motion compensation; it never touches detection quality.
    """
    import cv2
    import numpy as np

    if len(frame_paths) < 2:
        return 0.0

    step = max(1, (len(frame_paths) - 1) // sample_pairs)
    indices = range(0, len(frame_paths) - 1, step)

    shifts = []
    for i in indices:
        a = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(frame_paths[i + 1]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        target_width = 320
        target_height = max(1, int(target_width * a.shape[0] / a.shape[1]))
        small_a = cv2.resize(a, (target_width, target_height)).astype("float32")
        small_b = cv2.resize(b, (target_width, target_height)).astype("float32")
        (dx, dy), _response = cv2.phaseCorrelate(small_a, small_b)
        shifts.append((dx**2 + dy**2) ** 0.5)

    return float(np.median(shifts)) if shifts else 0.0


def detect_and_track_people(
    frame_paths: list[Path],
    fps: float,
    config: Phase1Config,
) -> list[Detection]:
    """Run YOLO and persistent BoT-SORT over frames in chronological order."""
    from ultralytics import YOLO

    tracker = config.tracker
    if config.auto_detect_static_camera:
        motion_px = estimate_camera_motion(frame_paths, config.static_camera_motion_samples)
        if motion_px < config.static_camera_threshold_px:
            tracker = config.static_camera_tracker

    model = YOLO(config.yolo_model)
    detections: list[Detection] = []
    for frame_index, frame_path in enumerate(frame_paths):
        result = model.track(
            source=str(frame_path),
            persist=True,
            conf=config.person_confidence,
            iou=0.5,
            classes=[0],
            tracker=tracker,
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


class ArcFaceEmbedder:
    """Lazy InsightFace wrapper so absent/error paths remain lightweight."""

    def __init__(self, det_thresh: float = 0.5) -> None:
        self._app = None
        self._det_thresh = det_thresh

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
                det_thresh=self._det_thresh,
            )
        return self._app

    def embed_image(self, image):
        """Return the largest detected face's normalized embedding."""
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
        return largest.normed_embedding


def _bbox_area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


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
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return -1.0
    return float(np.dot(left, right) / denominator)


def _appearance_histogram(image):
    """Cheap, lighting-tolerant appearance descriptor for track stitching.

    Hue/saturation only (no value channel) so shadows and exposure shifts
    across a brief occlusion don't break the match.
    """
    import cv2

    if image is None or image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _histogram_similarity(left, right) -> float:
    import cv2

    if left is None or right is None:
        return -1.0
    return float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


class _UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def merge_fragmented_tracks(
    detections: list[Detection],
    config: Phase1Config,
) -> list[Detection]:
    """Stitch tracks likely split by an ID switch (occlusion, motion blur)
    into one continuous track, so a single person's evidence isn't diluted
    across many short-lived fragments before face scoring runs.

    A candidate merge requires, between one track's last sighting and
    another's first sighting shortly after: temporal adjacency, spatial
    proximity scaled to how large the person appears, and a matching
    color-histogram appearance at the handoff. Matching is greedy by
    descending appearance similarity, capping each track to at most one
    predecessor and one successor, then unioned into merged groups.
    """
    import cv2

    if not detections:
        return detections

    by_track: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_track[detection.track_id].append(detection)
    for track_detections in by_track.values():
        track_detections.sort(key=lambda item: item.timestamp)

    track_ids = list(by_track.keys())
    starts = {tid: dets[0] for tid, dets in by_track.items()}
    ends = {tid: dets[-1] for tid, dets in by_track.items()}

    frame_cache: dict[str, object] = {}

    def _load_crop(detection: Detection):
        frame = frame_cache.get(detection.frame_path)
        if frame is None:
            frame = cv2.imread(detection.frame_path)
            frame_cache[detection.frame_path] = frame
        if frame is None:
            return None
        return _crop(frame, detection.bbox, margin_ratio=0.05)

    end_histograms = {tid: _appearance_histogram(_load_crop(det)) for tid, det in ends.items()}
    start_histograms = {tid: _appearance_histogram(_load_crop(det)) for tid, det in starts.items()}

    candidate_edges = []
    for a in track_ids:
        a_end = ends[a]
        for b in track_ids:
            if a == b:
                continue
            b_start = starts[b]
            gap = b_start.timestamp - a_end.timestamp
            if gap <= 0 or gap > config.merge_max_gap_seconds:
                continue

            distance = (
                (_bbox_center(a_end.bbox)[0] - _bbox_center(b_start.bbox)[0]) ** 2
                + (_bbox_center(a_end.bbox)[1] - _bbox_center(b_start.bbox)[1]) ** 2
            ) ** 0.5
            scale = max(_bbox_diagonal(a_end.bbox), _bbox_diagonal(b_start.bbox), 1.0)
            if distance > config.merge_max_distance_ratio * scale:
                continue

            similarity = _histogram_similarity(end_histograms[a], start_histograms[b])
            if similarity < config.merge_min_appearance_similarity:
                continue

            candidate_edges.append((gap, similarity, a, b))

    # Nearest-in-time first, so a track always prefers its immediate next
    # sighting over a temporally-farther one with marginally higher
    # appearance similarity -- otherwise a "skip connection" can strand a
    # track that sat chronologically in between (similarity ties broken by
    # highest similarity).
    candidate_edges.sort(key=lambda edge: (edge[0], -edge[1]))
    has_successor: set[int] = set()
    has_predecessor: set[int] = set()
    accepted_edges = []
    for gap, similarity, a, b in candidate_edges:
        if a in has_successor or b in has_predecessor:
            continue
        has_successor.add(a)
        has_predecessor.add(b)
        accepted_edges.append((a, b))

    union_find = _UnionFind(track_ids)
    for a, b in accepted_edges:
        union_find.union(a, b)

    canonical_id = {tid: union_find.find(tid) for tid in track_ids}
    for detection in detections:
        detection.track_id = canonical_id[detection.track_id]
    return detections


def aggregate_similarities(scores: Iterable[float], top_k: int) -> float:
    """Use a top-k mean to reduce one-frame false matches."""
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return -1.0
    selected = ordered[: max(1, top_k)]
    return sum(selected) / len(selected)


def score_tracks(
    reference_image: Path,
    detections: list[Detection],
    config: Phase1Config,
    embedder: ArcFaceEmbedder | None = None,
) -> tuple[dict[int, float], list[str]]:
    """Compute identity similarity from several high-quality faces per track."""
    import cv2

    embedder = embedder or ArcFaceEmbedder(det_thresh=config.face_detection_threshold)
    reference = cv2.imread(str(reference_image))
    reference_embedding = embedder.embed_image(reference)
    if reference_embedding is None:
        raise ValueError(f"No face detected in reference image: {reference_image}")

    by_track: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_track[detection.track_id].append(detection)

    # Multiple tracks (different people) can share a frame_path when more
    # than one person is visible in the same sampled frame -- cache the
    # decoded frame across tracks instead of re-reading it from disk once
    # per track that happens to have a candidate there.
    frame_cache: dict[str, object] = {}

    def _load_frame(path: str):
        frame = frame_cache.get(path)
        if frame is None and path not in frame_cache:
            frame = cv2.imread(path)
            frame_cache[path] = frame
        return frame

    warnings: list[str] = []
    track_scores: dict[int, float] = {}
    for track_id, track_detections in by_track.items():
        # Larger, higher-confidence person crops are more likely to contain a
        # usable face, so they're tried first, but bbox size says nothing
        # about pose -- a big box can still be facing away or looking down.
        # No early exit on candidate *count* (stopping after N successes
        # let a worse-but-bigger-boxed frame crowd out a genuinely better
        # one still waiting later in the pool). The only early exit is
        # confidence-based: see early_exit_on_high_confidence below, which
        # requires several samples to agree rather than trusting one.
        candidates = sorted(
            track_detections,
            key=lambda item: (_bbox_area(item), item.confidence),
            reverse=True,
        )[: max(config.face_samples_per_track * 3, 10)]
        similarities: list[float] = []
        for detection in candidates:
            frame = _load_frame(detection.frame_path)
            if frame is None:
                continue
            person = _crop(frame, detection.bbox, margin_ratio=0.05)
            embedding = embedder.embed_image(person)
            if embedding is None:
                continue
            similarity = _cosine_similarity(reference_embedding, embedding)
            detection.identity_similarity = round(similarity, 4)
            similarities.append(similarity)

            if (
                config.early_exit_on_high_confidence
                and len(similarities) >= config.similarity_top_k
            ):
                top_values = sorted(similarities, reverse=True)[: config.similarity_top_k]
                if top_values[-1] >= config.early_exit_confidence_threshold:
                    # The `similarity_top_k` samples that will decide this
                    # track's score already all independently clear a bar
                    # well outside the observed noise/false-positive band
                    # -- a wrong person is unlikely to produce that many
                    # agreeing high scores by chance, so further candidates
                    # are very unlikely to change the outcome.
                    break
        if similarities:
            track_scores[track_id] = aggregate_similarities(
                similarities, config.similarity_top_k
            )
        else:
            warnings.append(f"Track {track_id} had no usable face observations")
    return track_scores, warnings


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
    if detections and config.merge_fragmented_tracks:
        detections = merge_fragmented_tracks(detections, config)
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

    track_scores, warnings = score_tracks(
        reference_image, detections, config, embedder=embedder
    )
    best_track_id = max(track_scores, key=track_scores.get) if track_scores else None
    best_score = track_scores.get(best_track_id, -1.0)
    matched_track_ids = {
        track_id
        for track_id, score in track_scores.items()
        if score >= config.identity_threshold
    }
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
    )
