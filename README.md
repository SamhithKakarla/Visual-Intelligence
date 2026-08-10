# Visual Intelligence

Given one reference face image and one video, this pipeline determines whether
the person appears in the video. If the person is found, it analyzes only that
person's disjoint appearances and returns an open-ended activity description
plus a positive, negative, or neutral classification.

## Output contract

Present subject:

```json
{
  "reference_image": "/data/person_041.jpg",
  "reference_video": "/data/camera12_clip004.mp4",
  "person_exists": true,
  "activity_description": "The subject walks toward the camera.",
  "activity_classification": "neutral"
}
```

Absent subject:

```json
{
  "reference_image": "/data/person_041.jpg",
  "reference_video": "/data/camera12_clip019.mp4",
  "person_exists": false,
  "activity_description": null,
  "activity_classification": null
}
```

## Pipeline

### Phase 1: reference-person discovery

1. FFmpeg samples the video in chronological order.
2. YOLO detects people and BoT-SORT maintains local person tracks.
3. InsightFace/ArcFace embeds several high-quality face observations per track.
4. Track-level cosine scores are aggregated and compared with a configurable
   threshold.
5. Matching observations are split into disjoint appearances. A subject seen
   at 2–5 seconds and 40–43 seconds produces two appearances, never one 2–43
   second interval.
6. Each appearance receives a bounded sequence of full-scene frames with the
   target highlighted, plus context-preserving target crops.

### Phase 2: arbitrary activity analysis

Phase 2 runs only when Phase 1 reports a match. Qwen3-VL analyzes each target
appearance independently and returns a description, valence, confidence, and
evidence timestamps. There is no fixed activity vocabulary.

The visible-behavior rubric is:

- `positive`: clearly helpful, cooperative, protective, affectionate, or
  prosocial behavior
- `negative`: clearly harmful, dangerous, aggressive, destructive, illegal,
  or antisocial behavior
- `neutral`: ordinary behavior with no clear positive/negative effect, or
  behavior whose intent is ambiguous

For multiple appearances, a confidently negative appearance makes the overall
video negative; otherwise positive takes precedence over neutral. Diagnostic
segment results are retained separately from the five-field public output.

## Installation

Python 3.11 is recommended. Install FFmpeg separately, then install Python
dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Colab:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

The first real run downloads YOLO, InsightFace, and Qwen model weights. Model
weights and run artifacts are intentionally excluded from Git.

## Run

Either entry point is supported:

```bash
python main.py reference_photo.jpg video.mp4 --output result.json
```

```bash
python -m visual_intelligence reference_photo.jpg video.mp4 \
  --output result.json
```

Useful configuration options:

```text
--search-fps 4
--identity-threshold 0.4
--yolo-model yolov8n.pt
--vlm-model Qwen/Qwen3-VL-4B-Instruct
--runs-dir runs
--delete-artifacts
```

## Batch datasets

For multiple reference-image/video pairs, create a JSON manifest. Relative
media paths are resolved from the manifest's directory:

```json
{
  "dataset_id": "two-video-demo",
  "items": [
    {
      "case_id": "known-present",
      "reference_image": "test_image.jpg",
      "reference_video": "test_video.mp4"
    },
    {
      "case_id": "known-absent",
      "reference_image": "test_image.jpg",
      "reference_video": "test_video_absent.mp4"
    }
  ]
}
```

Run the full dataset with one command:

```bash
python -m visual_intelligence --manifest dataset.json \
  --output batch_results.json \
  --max-evidence-frames 48
```

The public output contains one five-field prediction per `case_id`.
`batch_results.debug.json` contains the corresponding Phase 1 and Phase 2
diagnostics. Individual outputs are retained under `batch_results.items/`.
The Qwen model is lazily loaded once and reused across the batch.

The default threshold remains a starting point. It must be calibrated on
labeled present/absent pairs before accuracy claims are made.

## Outputs

For `--output result.json`, the pipeline writes:

- `result.json`: the stable five-field public result
- `result.debug.json`: identity score, threshold, appearances, evidence paths,
  per-appearance VLM outputs, confidence, and warnings
- `runs/<run-id>/`: isolated intermediate frames and Phase 2 evidence, unless
  `--delete-artifacts` is supplied

Run isolation prevents frames from one video contaminating another run.

## Tests

The core control flow and segmentation tests do not download models:

```bash
python -m unittest discover -s tests -v
```

Tests cover the absent-person gate, strict public output schema, structured VLM
validation, valence aggregation, and disjoint appearances separated by a large
time gap.

## FaceSurv evaluation status

FaceSurv integration is intentionally the next stage. The dataset will be kept
outside Git and used to calibrate/evaluate Phase 1 with balanced present/absent
pairs. Because FaceSurv contains people walking toward cameras but does not
provide activity-valence annotations, it can exercise Phase 2 as a neutral
pipeline smoke test but cannot establish three-class Phase 2 accuracy.
