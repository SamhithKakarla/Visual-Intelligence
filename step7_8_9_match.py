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


def run_phase1(query_path: str, detections_path: str = "detections_with_embeddings.json",
               output_path: str = "phase1_output.json", threshold: float = THRESHOLD):
    """
    End-to-end Step 6-9: embed query, match, threshold, save.
    """
    from step5_6_embed import embed_query_image

    query_vec = embed_query_image(query_path)

    with open(detections_path) as f:
        detections = json.load(f)

    matches, all_scored = match_query_to_detections(query_vec, detections, threshold)
    if matches:
        print(f"MATCH — {len(matches)} frame(s) above threshold:")
        for m in matches:
            print(f"  similarity={m['similarity']}  at {m['timestamp']}s  "
                  f"bbox={m['bbox']}  crop={m['crop_path']}")
    else:
        best_score = max(d["similarity"] for d in all_scored) if all_scored else 0.0
        print(f"NO MATCH — best similarity was {best_score:.4f} (threshold={threshold})")
    trajectory=[]
    for det in matches:
        trajectory.extend(det['trajectory'])

    output = {
        "query_image": query_path,
        "threshold": threshold,
        "match_found": len(matches) > 0,
        "matches": process_segments(trajectory),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {output_path}")

    return output


def process_segments(trajectory: list) -> list:
    if not trajectory:
        return []

    trajectory = sorted(trajectory, key=lambda x: (x["track_id"], x["timestamp"]))

    results = []
    segment_id = 1

    current_segment = [trajectory[0]]
    start_segment = trajectory[0]["timestamp"]
    prev_time = trajectory[0]["timestamp"]
    prev_track_id = trajectory[0]["track_id"]

    for traj in trajectory[1:]:
        track_changed = traj["track_id"] != prev_track_id
        time_gap = traj["timestamp"] - prev_time > 2.0

        if track_changed or time_gap:
            # close current segment
            results.append({
                "segment_id": segment_id,
                "track_id": current_segment[0]["track_id"],
                "start_time": start_segment,
                "end_time": prev_time,
                "trajectory": current_segment
            })
            segment_id += 1

            # start new segment
            start_segment = traj["timestamp"]
            current_segment = [traj]
        else:
            current_segment.append(traj)

        prev_time = traj["timestamp"]
        prev_track_id = traj["track_id"]

    # add last segment
    results.append({
        "segment_id": segment_id,
        "track_id": current_segment[0]["track_id"],
        "start_time": start_segment,
        "end_time": prev_time,
        "trajectory": current_segment
    })

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python step7_8_9_match.py <query_image_path>")
        sys.exit(1)
    run_phase1(sys.argv[1])
