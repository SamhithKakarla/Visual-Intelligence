import sys
import os
import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from phase2_classification import run_phase2

CLASSES = [
    "Person loading an object into a vehicle",
    "Person unloading an object from a vehicle",
    "Person opening a vehicle trunk",
    "Person closing a vehicle trunk",
    "Person getting into a vehicle",
    "Person getting out of a vehicle",
    "Person gesturing",
    "Person digging",
    "Person carrying an object",
    "Person running",
    "Person entering a facility",
    "Person exiting a facility",
]

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from step1_extract_frames import extract_frames, frame_index_to_timestamp
from step3_detect_people import detect_people
from step4_crop_people import crop_people
from step7_8_9_match import process_segments


dataset_dir = r"VIRAT Ground Dataset"
annotations_dir = os.path.join(dataset_dir, "annotations (1)")
video_dir = os.path.join(dataset_dir, "videos_original")
frames_root = "frames"
csv_path = "virat-eval-results.csv"
dataset_summary_csv_path = "virat-eval-results-dataset-summary.csv"


def clear_pipeline_dirs():
    """Removes stale intermediate output dirs so each video starts clean."""
    for d in ("frames", "boxed_frames", "person_crops"):
        if os.path.exists(d):
            shutil.rmtree(d)


def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0.0

    return inter / union


def evaluate_matches(matches, events_df,
                      fps=1,
                      temporal_threshold=1,
                      iou_weight=0.8,
                      iou_threshold=0.5):
    """
    Evaluate predicted bounding boxes against VIRAT ground-truth boxes.

    For every predicted detection, find the best-matching GT box within
    `temporal_threshold` frames (ranked by a blend of IoU + temporal
    closeness). A predicted box counts as a True Positive if the IoU of
    its best match is >= `iou_threshold`, otherwise it's a False Positive
    (including predictions with no nearby GT candidate at all). Any GT
    box in the dataframe that was never claimed as a >=threshold match by
    some prediction counts as a False Negative.

    Parameters
    ----------
    matches : list
        tracking_json["matches"]

    events_df : DataFrame
        VIRAT event dataframe (must contain frame, x, y, x2, y2, event_id,
        event_type columns)

    fps : int
        FPS used while extracting frames

    temporal_threshold : int
        Max frame difference allowed when searching for a GT candidate

    iou_weight : float
        Weight given to IoU (vs temporal closeness) when ranking candidates

    iou_threshold : float
        Minimum IoU for a matched box to count as a correct detection (TP)

    Returns
    -------
    results : list
        Per-prediction match records (includes a "is_tp" flag)
    metrics : dict
        Aggregate bbox metrics: mean_iou, mean_temporal_difference,
        bbox_precision, bbox_recall, bbox_f1, true_positives,
        false_positives, false_negatives
    """

    # Create x2,y2 if they don't exist
    if "x2" not in events_df.columns:
        events_df["x2"] = events_df["x"] + events_df["w"]
        events_df["y2"] = events_df["y"] + events_df["h"]

    results = []

    ious = []
    temporal_errors = []

    matched_gt_indices = set()  # GT rows (by dataframe index) claimed by a TP prediction
    tp = 0
    fp = 0

    for segment in matches:

        for det in segment["trajectory"]:

            pred_frame = round(det["timestamp"] * fps)
            pred_box = det["bbox"]

            candidates = events_df[
                abs(events_df["frame"] - pred_frame) <= temporal_threshold
            ]

            best = None
            best_score = -1
            best_idx = None

            for idx, row in candidates.iterrows():

                gt_box = [
                    row["x"],
                    row["y"],
                    row["x2"],
                    row["y2"]
                ]

                iou = compute_iou(pred_box, gt_box)

                dt = abs(int(row["frame"]) - pred_frame)

                temporal_similarity = 1 - dt / temporal_threshold if temporal_threshold > 0 else 1

                score = (
                    iou_weight * iou +
                    (1 - iou_weight) * temporal_similarity
                )

                if score > best_score:

                    best_score = score
                    best_idx = idx

                    # event_type is 1-indexed (1-12) in VIRAT; CLASSES is 0-indexed
                    event_type = int(row["event_type"])

                    best = {
                        "segment_id": segment["segment_id"],
                        "track_id": segment["track_id"],
                        "predicted_frame": pred_frame,
                        "ground_truth_frame": int(row["frame"]),
                        "temporal_difference": dt,
                        "iou": round(iou, 4),
                        "score": round(score, 4),
                        "event_id": int(row["event_id"]),
                        "event_type": event_type,
                        "event_name": CLASSES[event_type - 1] if 1 <= event_type <= len(CLASSES) else "unknown",
                    }

            if best is not None:
                is_tp = best["iou"] >= iou_threshold
                best["is_tp"] = is_tp

                results.append(best)
                ious.append(best["iou"])
                temporal_errors.append(best["temporal_difference"])

                if is_tp:
                    tp += 1
                    matched_gt_indices.add(best_idx)
                else:
                    fp += 1
            else:
                # No GT candidate at all within the temporal window -> FP
                fp += 1

    total_gt_boxes = len(events_df)
    fn = total_gt_boxes - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # NOTE: classic accuracy = (TP+TN)/(TP+TN+FP+FN) requires a countable
    # set of negatives. For bounding-box detection there's no fixed universe
    # of "boxes that weren't drawn" -> TN is undefined, so real accuracy
    # can't be computed here (this is why COCO/PASCAL VOC-style detection
    # evaluation reports precision/recall/F1, never accuracy). What we
    # report instead is the standard detection "accuracy" proxy, TP/(TP+FP+FN),
    # sometimes called the Jaccard/IoU-style detection score.
    detection_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    metrics = {
        "matched_boxes": len(results),
        "mean_iou": round(np.mean(ious), 4) if ious else 0,
        "mean_temporal_difference": round(np.mean(temporal_errors), 4) if temporal_errors else 0,
        "bbox_true_positives": tp,
        "bbox_false_positives": fp,
        "bbox_false_negatives": fn,
        "bbox_precision": round(precision, 4),
        "bbox_recall": round(recall, 4),
        "bbox_f1": round(f1, 4),
        "bbox_detection_accuracy": round(detection_accuracy, 4),
    }

    return results, metrics


def run_pipeline_for_video(video_file_name: str):
    """
    Runs your existing detect -> crop -> segment -> phase2 pipeline for one
    video and returns (matches, phase2_results) for downstream evaluation.
    """
    video_basename = os.path.splitext(video_file_name)[0]
    video_path = os.path.join(video_dir, video_file_name)

    frames_dir = frames_root
    clear_pipeline_dirs()

    frames = extract_frames(video_path, frames_dir, fps=30)

    detections = detect_people(frames_dir)
    detections = crop_people(detections)
    matches = process_segments(detections)
    results = run_phase2(matches)

    if not os.path.exists("inference"):
        os.makedirs("inference")

    out_path = f"inference/phase2_output_{video_basename}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    return matches, results


def parse_events_file(event_file: str) -> pd.DataFrame:
    """Parse a VIRAT '<clip>.viratdata.events.txt' file into a DataFrame.

    Columns per the VIRAT spec:
    event_id, event_type, duration, start_frame, end_frame, frame, x, y, w, h
    """
    columns = [
        "event_id", "event_type", "duration",
        "start_frame", "end_frame", "frame",
        "x", "y", "w", "h",
    ]
    rows = []
    with open(event_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != len(columns):
                continue  # skip malformed/blank lines
            rows.append([int(p) for p in parts])
    return pd.DataFrame(rows, columns=columns)


def eval_data(video_file_name: str, matches, fps: int = 30,
              temporal_threshold: int = 1, iou_weight: float = 0.8,
              iou_threshold: float = 0.5):
    """
    Compare phase2's predicted per-segment activities against VIRAT
    ground-truth events overlapping that segment's time window, AND
    evaluate the predicted bounding boxes (via evaluate_matches) against
    the same ground truth.

    Writes a single combined summary row (activity metrics + bbox metrics)
    to `csv_path` and returns (per_segment_results, bbox_results, metrics).
    """
    video_name = video_file_name[:-4]
    event_file = os.path.join(annotations_dir, f"{video_name}.viratdata.events.txt")
    inference_path = f"inference/phase2_output_{video_name}.json"

    events_df = parse_events_file(event_file)
    events_df["x2"] = events_df["x"] + events_df["w"]
    events_df["y2"] = events_df["y"] + events_df["h"]

    with open(inference_path, "r") as f:
        inference_data = json.load(f)

    # ---- Activity (segment-level) evaluation ----
    per_segment_results = []
    tp = fp = fn = tn = 0
    num_classes = len(CLASSES)

    for segment in inference_data:
        start_frame = round(segment["start_time"] * fps)
        end_frame = round(segment["end_time"] * fps)

        gt_rows = events_df[
            (events_df["frame"] >= start_frame) &
            (events_df["frame"] <= end_frame)
        ]

        # event_type is 1-indexed in VIRAT (1-12); CLASSES is 0-indexed
        gt_activities = {
            CLASSES[int(et) - 1]
            for et in gt_rows["event_type"].unique()
            if 1 <= int(et) <= len(CLASSES)
        }
        pred_activities = set(segment.get("activities", []))

        seg_tp = len(gt_activities & pred_activities)
        seg_fp = len(pred_activities - gt_activities)
        seg_fn = len(gt_activities - pred_activities)
        # Classes neither predicted nor present in GT for this segment
        seg_tn = num_classes - seg_tp - seg_fp - seg_fn
        tp += seg_tp
        fp += seg_fp
        fn += seg_fn
        tn += seg_tn

        per_segment_results.append({
            "segment_id": segment["segment_id"],
            "track_id": segment["track_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "ground_truth_activities": sorted(gt_activities),
            "predicted_activities": sorted(pred_activities),
            "true_positives": seg_tp,
            "false_positives": seg_fp,
            "false_negatives": seg_fn,
            "true_negatives": seg_tn,
        })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Standard accuracy: (TP+TN) / (TP+TN+FP+FN). TN is well-defined here
    # because each segment has a fixed universe of 12 possible classes, so
    # "neither predicted nor in ground truth" is a countable outcome.
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # ---- Bounding-box evaluation ----
    bbox_results, bbox_metrics = evaluate_matches(
        matches,
        events_df,
        fps=fps,
        temporal_threshold=temporal_threshold,
        iou_weight=iou_weight,
        iou_threshold=iou_threshold,
    )

    metrics = {
        "video": video_name,
        "num_segments": len(inference_data),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        **bbox_metrics,
    }

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    return per_segment_results, bbox_results, metrics


def compute_prf1_accuracy(tp, fp, fn, tn=None):
    """Shared precision/recall/F1(/accuracy) computation.

    If `tn` is provided, accuracy is the standard (TP+TN)/(TP+TN+FP+FN).
    If `tn` is None (no well-defined negative set, e.g. bbox detection),
    a detection-style TP/(TP+FP+FN) proxy is returned instead.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if tn is not None:
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    else:
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4), round(accuracy, 4)


def main():
    """Run the full pipeline + evaluation over every video in video_dir."""
    video_files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    )

    # Dataset-wide accumulators
    ds_act_tp = ds_act_fp = ds_act_fn = ds_act_tn = 0
    ds_bbox_tp = ds_bbox_fp = ds_bbox_fn = 0
    ds_ious = []
    ds_temporal_errors = []
    ds_num_segments = 0
    ds_videos_evaluated = 0

    for video_file_name in tqdm(video_files, desc="Processing videos"):
        try:
            matches, results = run_pipeline_for_video(video_file_name)
            per_segment_results, bbox_results, metrics = eval_data(video_file_name, matches)
            print(f"{video_file_name}: {metrics}")

            # Roll this video's counts into the dataset totals
            ds_act_tp += metrics["true_positives"]
            ds_act_fp += metrics["false_positives"]
            ds_act_fn += metrics["false_negatives"]
            ds_act_tn += metrics["true_negatives"]
            ds_bbox_tp += metrics["bbox_true_positives"]
            ds_bbox_fp += metrics["bbox_false_positives"]
            ds_bbox_fn += metrics["bbox_false_negatives"]
            ds_num_segments += metrics["num_segments"]
            ds_videos_evaluated += 1

            # mean_iou / mean_temporal_difference are per-video averages, so
            # weight them by matched_boxes to get a correct dataset-wide mean
            if metrics["matched_boxes"]:
                ds_ious.extend([metrics["mean_iou"]] * metrics["matched_boxes"])
                ds_temporal_errors.extend(
                    [metrics["mean_temporal_difference"]] * metrics["matched_boxes"]
                )
        except Exception as e:
            print(f"Failed on {video_file_name}: {e}")

    # ---- Aggregate dataset-wide metrics (computed from totals, not an
    # average of per-video metrics, so ratio metrics stay mathematically
    # correct) ----
    act_precision, act_recall, act_f1, act_accuracy = compute_prf1_accuracy(
        ds_act_tp, ds_act_fp, ds_act_fn, tn=ds_act_tn
    )
    bbox_precision, bbox_recall, bbox_f1, bbox_detection_accuracy = compute_prf1_accuracy(
        ds_bbox_tp, ds_bbox_fp, ds_bbox_fn, tn=None
    )

    dataset_metrics = {
        "videos_evaluated": ds_videos_evaluated,
        "num_segments": ds_num_segments,
        "true_positives": ds_act_tp,
        "false_positives": ds_act_fp,
        "false_negatives": ds_act_fn,
        "true_negatives": ds_act_tn,
        "precision": act_precision,
        "recall": act_recall,
        "f1": act_f1,
        "accuracy": act_accuracy,
        "matched_boxes": ds_bbox_tp + ds_bbox_fp,
        "mean_iou": round(np.mean(ds_ious), 4) if ds_ious else 0,
        "mean_temporal_difference": round(np.mean(ds_temporal_errors), 4) if ds_temporal_errors else 0,
        "bbox_true_positives": ds_bbox_tp,
        "bbox_false_positives": ds_bbox_fp,
        "bbox_false_negatives": ds_bbox_fn,
        "bbox_precision": bbox_precision,
        "bbox_recall": bbox_recall,
        "bbox_f1": bbox_f1,
        "bbox_detection_accuracy": bbox_detection_accuracy,
    }

    with open(dataset_summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataset_metrics.keys()))
        writer.writeheader()
        writer.writerow(dataset_metrics)

    print(f"\nDataset-wide summary ({ds_videos_evaluated} videos): {dataset_metrics}")
    print(f"Saved: {dataset_summary_csv_path}")


if __name__ == "__main__":
    main()