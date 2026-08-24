"""Instrumented, fully local Phase 1 rerun for phase1_v2's actual algorithm:
evenly-spaced observation sampling -> quality gates -> top-K-by-similarity
aggregation. Produces the real Phase1Result (for Phase 2 handoff) plus a rich
per-detection distribution (for the frontend), all from one detection pass."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from visual_intelligence.config import Phase1Config
from visual_intelligence.phase1.pipeline import (
    ArcFaceEmbedder,
    build_appearance_groups,
    create_evidence_frames,
    detect_and_track_people,
    embed_reference_image,
    extract_search_frames,
    select_evenly_spaced,
    _cosine_similarity,
    _crop,
    aggregate_similarities,
)

CASE_NAME = sys.argv[1]
REFERENCE_IMAGE = Path(sys.argv[2])
VIDEO_PATH = Path(sys.argv[3])
OUT_DIR = Path(sys.argv[4])

IDENTITY_THRESHOLD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
config = Phase1Config(identity_threshold=IDENTITY_THRESHOLD)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== {CASE_NAME}: extract + detect ===")
frames = extract_search_frames(VIDEO_PATH, OUT_DIR / "search_frames", config.search_fps)
detections = detect_and_track_people(frames, config.search_fps, config)
by_track = defaultdict(list)
for d in detections:
    by_track[d.track_id].append(d)
print(f"{len(frames)} frames, {len(detections)} detections, tracks: {sorted(by_track.keys())}")

embedder = ArcFaceEmbedder()
reference_embedding = embed_reference_image(REFERENCE_IMAGE, embedder)

scoring_dir = OUT_DIR / "track_scoring"
scoring_dir.mkdir(parents=True, exist_ok=True)

track_report = {}
track_decisions = {}
for track_id, track_detections in by_track.items():
    track_dir = scoring_dir / f"track_{track_id}"
    track_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(track_detections, key=lambda item: item.timestamp)
    examined = select_evenly_spaced(ordered, config.max_identity_observations_per_track)
    examined_ts = {d.timestamp for d in examined}

    all_entries = []
    eligible = []
    faces_detected = 0
    rejected_no_face = 0
    rejected_too_small = 0
    rejected_low_confidence = 0
    for detection in ordered:
        entry = {
            "timestamp": detection.timestamp,
            "bbox": list(detection.bbox),
            "confidence": detection.confidence,
            "examined": detection.timestamp in examined_ts,
            "similarity": None,
            "face_w": None,
            "face_h": None,
            "det_conf": None,
            "reason": "not_examined" if detection.timestamp not in examined_ts else None,
            "face_crop_path": None,
            "person_crop_path": None,
        }
        if entry["reason"] == "not_examined":
            all_entries.append(entry)
            continue

        frame = cv2.imread(detection.frame_path)
        if frame is None:
            entry["reason"] = "frame_unreadable"
            all_entries.append(entry)
            continue
        person = _crop(frame, detection.bbox, margin_ratio=0.05)
        face = embedder.analyze_image(person)
        if face is None:
            entry["reason"] = "no_face_detected"
            rejected_no_face += 1
            all_entries.append(entry)
            continue

        faces_detected += 1
        similarity = _cosine_similarity(reference_embedding, face.embedding)
        entry["similarity"] = round(float(similarity), 4)
        entry["face_w"] = round(face.width, 1)
        entry["face_h"] = round(face.height, 1)
        entry["det_conf"] = round(face.detector_confidence, 3)

        # Save crops for every examined, face-bearing detection.
        ph, pw = person.shape[:2]
        fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(pw, fx2), min(ph, fy2)
        annotated = person.copy()
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        stem = f"t{detection.timestamp:.2f}"
        person_path = track_dir / f"{stem}_person.jpg"
        cv2.imwrite(str(person_path), annotated)
        entry["person_crop_path"] = str(person_path)
        face_only = person[fy1:fy2, fx1:fx2]
        if face_only.size > 0:
            big = cv2.resize(
                face_only, (face_only.shape[1] * 3, face_only.shape[0] * 3),
                interpolation=cv2.INTER_CUBIC,
            )
            face_path = track_dir / f"{stem}_face.jpg"
            cv2.imwrite(str(face_path), big)
            entry["face_crop_path"] = str(face_path)

        too_small = min(face.width, face.height) < config.minimum_face_dimension
        low_conf = face.detector_confidence < config.minimum_face_detector_confidence
        if too_small:
            entry["reason"] = "rejected_too_small"
            rejected_too_small += 1
        elif low_conf:
            entry["reason"] = "rejected_low_confidence"
            rejected_low_confidence += 1
        else:
            entry["reason"] = "eligible"
            eligible.append(entry)

        all_entries.append(entry)

    selected = sorted(eligible, key=lambda e: e["similarity"], reverse=True)[: config.similarity_top_k]
    selected_ts = {e["timestamp"] for e in selected}
    for e in all_entries:
        e["selected"] = e["timestamp"] in selected_ts

    agg = (
        aggregate_similarities((e["similarity"] for e in eligible), config.similarity_top_k)
        if eligible
        else None
    )
    scores_above_threshold = sum(e["similarity"] >= config.identity_threshold for e in selected)
    if len(eligible) < config.minimum_consensus_faces:
        matched, decision_reason = False, "insufficient_faces"
    elif agg is None or agg < config.identity_threshold:
        matched, decision_reason = False, "below_average_threshold"
    elif scores_above_threshold < config.minimum_scores_above_threshold:
        matched, decision_reason = False, "insufficient_scores_above_threshold"
    else:
        matched, decision_reason = True, "matched_consensus"

    track_report[str(track_id)] = {
        "track_id": track_id,
        "total_detections": len(track_detections),
        "observations_examined": len(examined),
        "faces_detected": faces_detected,
        "eligible_faces": len(eligible),
        "rejected_no_face": rejected_no_face,
        "rejected_too_small": rejected_too_small,
        "rejected_low_confidence": rejected_low_confidence,
        "selected_scores": [e["similarity"] for e in selected],
        "selected_timestamps": [e["timestamp"] for e in selected],
        "aggregate_score": round(float(agg), 4) if agg is not None else None,
        "matched": matched,
        "decision_reason": decision_reason,
        "entries": all_entries,
    }
    track_decisions[track_id] = track_report[str(track_id)]
    print(f"  track {track_id}: agg={agg} matched={matched} ({decision_reason})")

matched_track_ids = {tid for tid, d in track_decisions.items() if d["matched"]}
best_track_candidates = matched_track_ids or set(track_decisions)
best_track_id = (
    max(best_track_candidates, key=lambda tid: track_decisions[tid]["aggregate_score"] or -1.0)
    if best_track_candidates
    else None
)
best_score = track_decisions[best_track_id]["aggregate_score"] if best_track_id is not None else -1.0
print(f"best_track_id={best_track_id} best_score={best_score} matched={sorted(matched_track_ids)}")

appearances_out = []
if matched_track_ids:
    matched_detections = [d for d in detections if d.track_id in matched_track_ids]
    groups = build_appearance_groups(matched_detections, config.appearance_gap_seconds)
    for appearance_id, group in enumerate(groups, start=1):
        bbox_to_track = {(d.timestamp, tuple(d.bbox)): d.track_id for d in group}
        evidence = create_evidence_frames(group, appearance_id, OUT_DIR / "phase2_inputs", config)
        start = max(0.0, min(item.timestamp for item in group) - config.context_padding_seconds)
        end = max(item.timestamp for item in group) + config.context_padding_seconds
        frames_out = []
        for e in evidence:
            tid = bbox_to_track.get((e.timestamp, tuple(e.bbox)))
            frames_out.append({
                "timestamp": e.timestamp,
                "bbox": e.bbox,
                "identity_similarity": e.identity_similarity,
                "track_id": tid,
                "context_frame_path": e.context_frame_path,
                "target_crop_path": e.target_crop_path,
            })
        appearances_out.append({
            "appearance_id": appearance_id,
            "start_time": round(start, 4),
            "end_time": round(end, 4),
            "track_ids": sorted({item.track_id for item in group}),
            "num_evidence_frames": len(evidence),
            "frames": frames_out,
        })

report = {
    "case_name": CASE_NAME,
    "reference_image": str(REFERENCE_IMAGE),
    "reference_video": str(VIDEO_PATH),
    "config": {
        "search_fps": config.search_fps,
        "identity_threshold": config.identity_threshold,
        "max_identity_observations_per_track": config.max_identity_observations_per_track,
        "minimum_consensus_faces": config.minimum_consensus_faces,
        "minimum_scores_above_threshold": config.minimum_scores_above_threshold,
        "minimum_face_dimension": config.minimum_face_dimension,
        "minimum_face_detector_confidence": config.minimum_face_detector_confidence,
        "similarity_top_k": config.similarity_top_k,
        "appearance_gap_seconds": config.appearance_gap_seconds,
        "max_evidence_frames": config.max_evidence_frames,
    },
    "num_search_frames": len(frames),
    "num_detections": len(detections),
    "tracks": track_report,
    "best_track_id": best_track_id,
    "best_score": round(float(best_score), 4),
    "matched_track_ids": sorted(matched_track_ids),
    "person_exists": bool(matched_track_ids),
    "appearances": appearances_out,
}
(OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
print(f"=== {CASE_NAME} DONE (person_exists={report['person_exists']}) ===")
