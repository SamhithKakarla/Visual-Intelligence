"""
Phase 1 — Facial Recognition & Tracking
Steps 7, 8, 9: Similarity Analysis, Decision Thresholding, Data Persistence

Compares the query embedding against every person-crop embedding using
cosine similarity, flags matches above threshold, and saves the
structured output for Phase 2 (if pursued separately).

NOTE ON THRESHOLD: this was 0.85 when using dlib's embeddings. ArcFace
embeddings have a different similarity distribution -- genuine same-person
matches typically score in the 0.4-0.6 cosine-similarity range, not 0.85+.
0.4 below is a reasonable starting point for ArcFace, but like the old
value, it's still a guess, not a measured number. Run evaluate_threshold.py
against labeled data once you have it to pick a real value.

MATCHING MODES:
  - "frame" (default): score every detection independently and flag frames
    above threshold. This is the original behavior.
  - "track": group a person's detections into a track (greedy IoU association
    across frames), average each track's embeddings into a single template,
    and score once per track. Averaging cancels per-frame noise (blur, pose,
    partial occlusion), which the FaceSurv benchmark showed lifts rank-1 from
    ~92% (frame) to ~98% (track). See facesurv_benchmark/.
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.4


def match_query_to_detections(query_vec: np.ndarray, detections: list, threshold: float = THRESHOLD):
    """
    Compare the query embedding against every detection's embedding.

    Args:
        query_vec: embedding of the reference query photo (128-dim)
        detections: list of dicts, each with an "embedding" key
        threshold: cosine similarity cutoff for a positive match

    Returns:
        (matches, all_scored_detections)
        - matches: detections that meet/exceed threshold, sorted by similarity desc
        - all_scored_detections: every detection with a "similarity" key added
    """
    embeddings = np.array([d["embedding"] for d in detections])
    scores = cosine_similarity([query_vec], embeddings)[0]

    for det, score in zip(detections, scores):
        det["similarity"] = round(float(score), 4)

    matches = [d for d in detections if d["similarity"] >= threshold]
    matches.sort(key=lambda d: d["similarity"], reverse=True)

    return matches, detections


# ---------------------------------------------------------------------------
# Track-level matching (averaging)
# ---------------------------------------------------------------------------

def aggregate_track_embeddings(embeddings) -> np.ndarray:
    """
    Combine several face embeddings from the same identity into one template
    by averaging, then L2-normalizing (so cosine similarity stays well-defined).

    This is the core "averaging" primitive shared by the pipeline's track mode
    and the FaceSurv benchmark, so both measure the exact same operation.

    Args:
        embeddings: iterable of 1-D vectors (list or ndarray), same dimension.

    Returns:
        a 1-D L2-normalized numpy vector (the track template).
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / (norm + 1e-9)


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter)


def group_detections_into_tracks(detections: list, iou_thresh: float = 0.3) -> list:
    """
    Assign a "track_id" to each detection with a lightweight greedy tracker:
    walking frames in time order, a detection extends an existing track if its
    box overlaps that track's most recent box (IoU >= iou_thresh); otherwise it
    starts a new track. Adequate for the pipeline's ~1 fps sampling where a
    person's box moves only modestly between frames; it is deliberately simple
    (no motion model / re-identification) and is not meant for crowded scenes.

    Mutates each detection in place (adds "track_id") and returns detections.
    """
    ordered = sorted(detections, key=lambda d: (d.get("timestamp", 0), d.get("frame_path", "")))
    active = []  # list of {"id": int, "bbox": last box}
    next_id = 0
    for det in ordered:
        best_id, best_iou = None, iou_thresh
        for tr in active:
            iou = _iou(det["bbox"], tr["bbox"])
            if iou >= best_iou:
                best_id, best_iou = tr["id"], iou
        if best_id is None:
            best_id = next_id
            next_id += 1
            active.append({"id": best_id, "bbox": det["bbox"]})
        else:
            for tr in active:
                if tr["id"] == best_id:
                    tr["bbox"] = det["bbox"]  # advance the track's latest box
                    break
        det["track_id"] = best_id
    return detections


def match_query_to_tracks(query_vec: np.ndarray, detections: list,
                          threshold: float = THRESHOLD, iou_thresh: float = 0.3):
    """
    Track-level matching: group detections into tracks, average each track's
    embeddings into one template, and score once per track.

    Returns:
        (matches, tracks)
        - matches: track dicts with similarity >= threshold, sorted desc
        - tracks: every track dict, each with keys track_id, similarity,
          num_frames, best_frame (the member detection closest to the query),
          timestamps, and member detection list.
    """
    group_detections_into_tracks(detections, iou_thresh=iou_thresh)

    by_track = {}
    for det in detections:
        by_track.setdefault(det["track_id"], []).append(det)

    tracks = []
    for track_id, members in by_track.items():
        template = aggregate_track_embeddings([m["embedding"] for m in members])
        sim = float(cosine_similarity([query_vec], [template])[0][0])
        # best_frame = the single detection whose own embedding is closest to
        # the query, useful as a human-verifiable exemplar of the track.
        best = max(members, key=lambda m: float(
            cosine_similarity([query_vec], [m["embedding"]])[0][0]))
        tracks.append({
            "track_id": int(track_id),
            "similarity": round(sim, 4),
            "num_frames": len(members),
            "timestamps": sorted(m.get("timestamp") for m in members),
            "best_frame": {k: v for k, v in best.items() if k != "embedding"},
        })

    matches = [t for t in tracks if t["similarity"] >= threshold]
    matches.sort(key=lambda t: t["similarity"], reverse=True)
    return matches, tracks


def run_phase1(query_path: str, detections_path: str = "detections_with_embeddings.json",
               output_path: str = "phase1_output.json", threshold: float = THRESHOLD,
               mode: str = "frame"):
    """
    End-to-end Step 6-9: embed query, match, threshold, save.

    mode: "frame" (score each detection) or "track" (average per-track first).
    """
    from step5_6_embed import embed_query_image

    if mode not in ("frame", "track"):
        raise ValueError(f"mode must be 'frame' or 'track', got {mode!r}")

    query_vec = embed_query_image(query_path)

    with open(detections_path) as f:
        detections = json.load(f)

    if mode == "track":
        matches, all_scored = match_query_to_tracks(query_vec, detections, threshold)
    else:
        matches, all_scored = match_query_to_detections(query_vec, detections, threshold)

    unit = "track" if mode == "track" else "frame"
    if matches:
        print(f"MATCH — {len(matches)} {unit}(s) above threshold:")
        for m in matches:
            if mode == "track":
                print(f"  track {m['track_id']}  similarity={m['similarity']}  "
                      f"{m['num_frames']} frames  best_frame={m['best_frame'].get('crop_path')}")
            else:
                print(f"  similarity={m['similarity']}  at {m['timestamp']}s  "
                      f"bbox={m['bbox']}  crop={m['crop_path']}")
    else:
        best_score = max((d["similarity"] for d in all_scored), default=0.0)
        print(f"NO MATCH — best {unit} similarity was {best_score:.4f} (threshold={threshold})")

    output = {
        "query_image": query_path,
        "mode": mode,
        "threshold": threshold,
        "match_found": len(matches) > 0,
        "matches": [_drop_embedding(m) for m in matches],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {output_path}")

    return output


def _drop_embedding(detection: dict) -> dict:
    """Return a copy of a detection dict without the raw embedding vector --
    it's only needed for the similarity computation, not for a human-readable
    result file, and at 512 floats per detection it bloats the JSON badly."""
    return {k: v for k, v in detection.items() if k != "embedding"}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step7_8_9_match.py <query_image_path> [frame|track]")
        sys.exit(1)
    mode = sys.argv[2] if len(sys.argv) > 2 else "frame"
    run_phase1(sys.argv[1], mode=mode)