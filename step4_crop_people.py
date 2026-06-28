"""
Phase 1 — Facial Recognition & Tracking
Step 4: Spatial Target Cropping (OpenCV / Pillow)

Crops each detected person from their source frame and saves the crop
independently. Optionally saves an annotated copy of the full frame with
the bounding box drawn.
"""

import os
import cv2


def crop_people(detections: list, crops_dir: str = "person_crops", boxed_dir: str = "boxed_frames",
                 save_boxed: bool = True):
    """
    Crop every detected person from their source frame.

    Args:
        detections: list of dicts from step3 (frame_path, timestamp, bbox, confidence)
        crops_dir: output directory for individual person crops
        boxed_dir: output directory for full frames with boxes drawn
        save_boxed: whether to also save annotated full frames

    Returns:
        the same detections list, with a "crop_path" key added to each entry
    """
    os.makedirs(crops_dir, exist_ok=True)
    if save_boxed:
        os.makedirs(boxed_dir, exist_ok=True)

    # Group detections by frame so we only draw boxes once per frame
    by_frame = {}
    for i, det in enumerate(detections):
        by_frame.setdefault(det["frame_path"], []).append(i)

    for frame_path, indices in by_frame.items():
        image = cv2.imread(frame_path)
        if image is None:
            print(f"[crop_people] WARNING: could not read {frame_path}")
            continue

        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        boxed_image = image.copy() if save_boxed else None

        for person_num, idx in enumerate(indices, start=1):
            x1, y1, x2, y2 = detections[idx]["bbox"]
            crop = image[y1:y2, x1:x2]

            crop_filename = f"{frame_name}_person_{person_num:02d}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            detections[idx]["crop_path"] = crop_path

            if save_boxed:
                cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if save_boxed:
            cv2.imwrite(os.path.join(boxed_dir, f"{frame_name}.jpg"), boxed_image)

    print(f"[crop_people] Saved {len(detections)} crops to '{crops_dir}/'")
    return detections


if __name__ == "__main__":
    import json
    with open("detections.json") as f:
        detections = json.load(f)
    detections = crop_people(detections)
    with open("detections_with_crops.json", "w") as f:
        json.dump(detections, f, indent=2)
    print("Saved detections_with_crops.json")
