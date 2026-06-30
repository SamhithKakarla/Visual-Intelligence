"""
Phase 1 — Facial Recognition & Tracking
Steps 5 & 6: Feature Vector Embedding (ArcFace via InsightFace)

Converts every cropped person image into a face embedding vector, and does
the same for the single reference query photo.

CHANGE: switched from `face_recognition` (dlib-based) to `insightface`
(ArcFace). Reasons:
  - ArcFace is a stronger face-verification model than dlib's encoder
  - insightface ships prebuilt wheels (onnxruntime backend) — no C++
    compilation, no cmake, avoids the dlib/PyTorch OpenMP conflict
  - This is the model your original proposal actually named

Embeddings are 512-dimensional (InsightFace's buffalo_l default), vs.
dlib's 128-dim. This doesn't affect anything elsewhere in the pipeline —
every other step just treats embeddings as opaque vectors.

Function signatures (embed_face, embed_person_crops, embed_query_image)
are unchanged from the dlib version, so step7_8_9_match.py and main.py
require no changes.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from collections import defaultdict

_face_app = None  # lazy-loaded singleton — loading the model is slow, do it once


def _get_face_app():
    global _face_app
    if _face_app is None:
        print("[step5_6_embed] Loading InsightFace (buffalo_l / ArcFace)...")
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def embed_face(image_path: str):
    """
    Compute a face embedding for a single image.

    Returns:
        a 512-dim numpy vector, or None if no face was detected, or if the
        image couldn't be read / was an invalid crop (e.g. zero-size)
    """
    bgr = cv2.imread(image_path)
    if bgr is None or bgr.size == 0:
        return None  # unreadable file or empty crop (e.g. a degenerate bounding box)

    app = _get_face_app()
    faces = app.get(bgr)  # insightface expects BGR (OpenCV's native format) — no conversion needed

    if len(faces) == 0:
        return None  # no face found in this crop — likely a partial/occluded detection

    # If multiple faces are found in one crop, take the largest (by bounding box area)
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.normed_embedding  # already L2-normalized, ideal for cosine similarity


def embed_person_crops(detections: list):
    """
    Compute face embeddings for every person crop in detections.

    Args:
        detections: list of dicts with a "crop_path" key (from step4)

    Returns:
        filtered list of detections that had a detectable face, each with
        an added "embedding" key (list of floats, JSON-serializable)
    """
    results = []
    skipped = 0

    combined = defaultdict(list)
    max_track_id = max(det["track_id"] for det in detections if det["track_id"] != -1)
    for det in detections:
        if det['track_id']==-1:
            max_track_id+=1
            combined[max_track_id].append(det)
        else:
            combined[det["track_id"]].append(det)
    
    for track_id, det_list in combined.items():
        det = det_list[0]
        vec = embed_face(det["crop_path"])
        if vec is None:
            skipped += 1
            continue
        results.append({
            "track_id": track_id,
            "embedding": vec.tolist(),
            "bbox":det['bbox'],
            "representative_frame": det["frame_path"],
            "timestamp": det["timestamp"],
            "trajectory": det_list,
            "crop_path":det['crop_path']
        })
    
    print(f"[embed_person_crops] number of segments={len(results)}")
    
    return results


    # for det in detections:
    #     vec = embed_face(det["crop_path"])
    #     if vec is None:
    #         skipped += 1
    #         continue
    #     det["embedding"] = vec.tolist()
    #     results.append(det)

    # print(f"[embed_person_crops] Embedded {len(results)} crops, skipped {skipped} (no face detected)")
    # return results


def embed_query_image(query_path: str):
    """
    Compute the embedding for the single reference query photo (e.g. passport/ID image).

    Returns:
        a 512-dim numpy vector

    Raises:
        ValueError if no face is found in the reference photo
    """
    vec = embed_face(query_path)
    if vec is None:
        raise ValueError(f"No face detected in reference photo: {query_path}")
    return vec


if __name__ == "__main__":
    import json
    with open("detections_with_crops.json") as f:
        detections = json.load(f)
    detections = embed_person_crops(detections)
    with open("detections_with_embeddings.json", "w") as f:
        json.dump(detections, f, indent=2)
    print("Saved detections_with_embeddings.json")
