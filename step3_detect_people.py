"""
Phase 1 — Facial Recognition & Tracking
Step 3: Person Detection (YOLOv8 / OpenCV)

For every extracted frame, detect all visible human entities and catalog
their spatial and temporal metadata: frame path, timestamp, bounding box
coordinates [x1, y1, x2, y2], and detection confidence.
"""

import os
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person" in YOLOv8


def detect_people(frames_dir: str, fps: int = 1, conf_threshold: float = 0.5, model_name: str = "yolov8n.pt"):
    """
    Run YOLOv8 person detection on every frame in frames_dir.

    Returns:
        list of dicts, one per detected person, with keys:
        frame_path, timestamp, bbox, confidence
    """
    model = YOLO(model_name)
    results=model.track(source=frames_dir,persist=True,conf=conf_threshold,classes=[0],stream=True,tracker="ocsort.yaml")
    detections = []
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    for idx,result in enumerate(results):
        frame_path = os.path.join(frames_dir, frame_files[idx])
        timestamp = (idx + 1) / fps

        if result.boxes is None:
            continue


        for box in result.boxes:
            # cls_id = int(box.cls[0])
            track_id = int(box.id.item()) if box.id else -1
            conf = float(box.conf[0])
            # if cls_id != PERSON_CLASS_ID or conf < conf_threshold:
            #     continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append({
                "frame_path": frame_path,
                "timestamp": timestamp,
                "track_id":track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
            })

    print(f"[detect_people] Found {len(detections)} person detections across {len(frame_files)} frames")
    return detections


if __name__ == "__main__":
    import sys, json
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else "frames"
    dets = detect_people(frames_dir)
    with open("detections.json", "w") as f:
        json.dump(dets, f, indent=2)
    print("Saved detections.json")
