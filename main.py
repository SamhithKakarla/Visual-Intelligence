"""
Phase 1 — Facial Recognition & Tracking
Main entry point: runs the full pipeline end to end.

    python main.py <reference_photo.jpg> <video.mp4>

Output:
    phase1_output.json — match result with timestamp, bbox, crop path
"""

import sys
import os

from step1_extract_frames import extract_frames
from step3_detect_people import detect_people
from step4_crop_people import crop_people
from step5_6_embed import embed_person_crops
from step7_8_9_match import run_phase1, THRESHOLD


def main(query_path: str, video_path: str, fps: int = 1, threshold: float = THRESHOLD):
    print("=" * 60)
    print("PHASE 1 — Facial Recognition & Tracking")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[1/4] Extracting frames...")
    extract_frames(video_path, output_dir="frames", fps=fps)

    # Step 3-4: Detect and crop people
    print("\n[2/4] Detecting and cropping people...")
    detections = detect_people("frames", fps=fps)
    detections = crop_people(detections)

    # Step 5: Embed all crops
    print("\n[3/4] Embedding face crops...")
    detections = embed_person_crops(detections)

    if not detections:
        print("\nNo faces detected in any frame. Cannot proceed with matching.")
        return None

    # Save intermediate state in case you want to re-run matching without
    # re-running detection/embedding (which are the slow steps)
    import json
    with open("detections_with_embeddings.json", "w") as f:
        json.dump(detections, f, indent=2)

    # Steps 6-9: Embed query, match, threshold, save
    print("\n[4/4] Matching against reference photo...")
    result = run_phase1(query_path, "detections_with_embeddings.json",
                         output_path="phase1_output.json", threshold=threshold)

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <reference_photo.jpg> <video.mp4>")
        sys.exit(1)

    query_photo, video_file = sys.argv[1], sys.argv[2]
    main(query_photo, video_file)
