"""
Chunked Ollama (qwen2.5vl + mistral) evaluation pipeline for UCF-Crime.

Pipeline per video:
  1. extract_frames() (from step1_extract_frames, assumed to already exist
     in your codebase) writes ordered frames into a local "frames" folder.
  2. Frames are grouped into chunks of CHUNK_SIZE and each chunk is described
     by qwen2.5vl via process_chunk().
  3. All chunk descriptions are concatenated and handed to mistral, which
     returns a structured {class, summary} JSON via summarize_description().
  4. Per-video and aggregate results (classification report) are saved.

You will need:
    pip install --break-system-packages ollama pandas scikit-learn tqdm
    ollama pull qwen2.5vl:7b
    ollama pull mistral:latest
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from ollama import chat
from sklearn.metrics import classification_report
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from step1_extract_frames import extract_frames  

CHUNK_SIZE = 18
DATA_DIR = "Data"
FRAMES_DIR = "frames"
RESULTS_CSV = "ucf-eval.csv"
REPORT_JSON = "data.json"

NORMAL_DIRS = {
    "Testing_Normal_Videos_Anomaly",
    "Training_Normal_Videos_Anomaly",
}

CLASSES = ["Suspicious", "Not suspicious"]

SUMMARY_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "enum": CLASSES},
        "summary": {"type": "string"},
    },
    "required": ["class", "summary"],
}

CHUNK_PROMPT = (
    "You are a video surveillance analyst. You are given a sequence of "
    "frames sampled over time from a single CCTV/surveillance video clip, "
    "in chronological order.\n\n"
    "Describe the activity of any person(s) visible in the frames in "
    "detail, including their appearance, and note whether they appear to "
    "be security personnel or an ordinary member of the public.\n\n"
    "Return enough details to decide whether the clip as a whole shows "
    "SUSPICIOUS behavior -- for example: getting into fights, being "
    "aggressive with others, breaking rules, being arrested, intentionally "
    "hurting others, cornering/bullying someone, abusing mentally or "
    "physically through actions, stealing, shoplifting, vandalism, arson, shooting or "
    "pushing/pulling someone around aggressively -- OR whether it shows "
    "NORMAL, everyday activity with nothing concerning happening."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the text: {combined_results}. "
    "Then decide whether the clip as a whole shows SUSPICIOUS behavior -- "
    "for example: getting into fights, being aggressive with others, "
    "breaking rules, being arrested, intentionally hurting others, or "
    "cornering/bullying someone, abusing mentally or physically through "
    "actions, stealing, shoplifting, vandalism, pushing or pulling someone "
    "around aggressively, arson, shooting -- OR whether it shows NORMAL, everyday activity "
    "with nothing concerning happening. If there seems to be nothing suspicious with the behavior of the people(even if one is suspicious give suspicious) then give not suspicious."
)


def process_chunk(chunk: list[str], model: str = "qwen2.5vl:7b", num_ctx: int = 10000) -> str:
    """Send one chunk of frame image paths to the VLM and return its
    free-text description of the activity in that chunk."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CHUNK_PROMPT,
                "images": chunk,
            }
        ],
        options={"num_ctx": num_ctx},
    )
    return response["message"]["content"]


def summarize_description(combined_results: str) -> dict:
    """Summarize all chunk descriptions for a video and return a
    {"class": "Suspicious"|"Not suspicious", "summary": str} dict."""
    response = chat(
        model="mistral:latest",
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT_TEMPLATE.format(combined_results=combined_results),
            }
        ],
        format=SUMMARY_FORMAT_SCHEMA,
    )
    return json.loads(response["message"]["content"])


def _frame_sort_key(filename: str):
    """Sort frame filenames by the numeric index they contain, so chunks are
    built in true chronological order regardless of OS listing order."""
    digits = re.findall(r"\d+", filename)
    return int(digits[-1]) if digits else 0


def process_video(video_path: str) -> dict:
    """Extract frames, describe them chunk by chunk, then summarize into a
    final {class, summary} dict for the whole video."""
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    extract_frames(video_path,fps=5)

    frame_files = sorted(os.listdir(FRAMES_DIR), key=_frame_sort_key)

    chunk_descriptions = []
    segment = []
    for file in tqdm(frame_files, total=len(frame_files), desc=f"Frames: {os.path.basename(video_path)}", leave=False):
        segment.append(os.path.join(FRAMES_DIR, file))
        if len(segment) == CHUNK_SIZE:
            chunk_descriptions.append(process_chunk(segment))
            segment = []
    if segment:
        chunk_descriptions.append(process_chunk(segment))

    combined_results = "\n\n---\n\n".join(chunk_descriptions)
    return summarize_description(combined_results)


def evaluate_all_videos():

    if os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        res_data = existing_df.to_dict("records")

        already_done = set(existing_df.loc[existing_df["pred_label"].isin(CLASSES), "video_name"])
        num_errors_to_retry = (existing_df["pred_label"] == "Error").sum()
        print(f"Resuming: found {len(already_done)} previously completed videos in {RESULTS_CSV}"
              + (f", will retry {num_errors_to_retry} that errored last time" if num_errors_to_retry else ""))

        res_data = [r for r in res_data if r["pred_label"] in CLASSES]
    else:
        res_data = []
        already_done = set()

    true_label, pred_label = [], []
    for row in res_data:
        if row["pred_label"] in CLASSES:
            true_label.append(1 if row["true_label"] == "Suspicious" else 0)
            pred_label.append(1 if row["pred_label"] == "Suspicious" else 0)

    class_dirs = [
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    for cd in class_dirs:
        class_name = os.path.basename(cd)
        is_normal = class_name.strip().lower().replace(" ", "_") in NORMAL_DIRS
        gt_label = "Not suspicious" if is_normal else "Suspicious"
        tl = 0 if is_normal else 1

        files = [os.path.join(cd, f) for f in os.listdir(cd)]
        remaining = [f for f in files if f not in already_done]
        skipped = len(files) - len(remaining)
        if skipped:
            print(f"{class_name}: skipping {skipped} already-processed video(s)")

        for file_path in tqdm(remaining, total=len(remaining), desc=class_name):
            try:
                res = process_video(file_path)
                pred_class = res.get("class", "Error")
                summary = res.get("summary", "")
            except Exception as e:
                pred_class = "Error"
                summary = str(e)

            res_data.append(
                {
                    "video_name": file_path,
                    "true_label": gt_label,
                    "pred_label": pred_class,
                    "vlm_description": summary,
                }
            )

            # Only feed valid (non-error) predictions into the metrics arrays.
            if pred_class in CLASSES:
                true_label.append(tl)
                pred_label.append(1 if pred_class == "Suspicious" else 0)

            pd.DataFrame(res_data).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(res_data)
    df.to_csv(RESULTS_CSV, index=False)

    num_errors = sum(1 for r in res_data if r["pred_label"] not in CLASSES)
    if num_errors:
        print(f"Note: {num_errors} videos failed processing and were excluded from the report.")

    if true_label:
        report = classification_report(
            true_label,
            pred_label,
            target_names=["Not suspicious", "Suspicious"],
            output_dict=True,
            zero_division=0,
        )
        print(classification_report(
            true_label, pred_label, target_names=["Not suspicious", "Suspicious"], zero_division=0
        ))
    else:
        report = {"error": "No valid predictions to score."}
        print(report["error"])

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    evaluate_all_videos()