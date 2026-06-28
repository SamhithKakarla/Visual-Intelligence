# Phase 1 — Facial Recognition & Tracking

Detects whether a person from a reference photo appears in a video, and
if so, returns the timestamp, bounding box, and cropped frame of the match.

Uses **ArcFace** (via the `insightface` package) for face embeddings.

## Setup (do this once)

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(If you already have the `capstone` conda environment from before, that's
fine too — just `pip install -r requirements.txt` inside it. You can
remove `dlib`/`face_recognition` from it if you want to reclaim space:
`pip uninstall face_recognition dlib`.)

**ffmpeg** (system tool, installed separately from Python packages):
```bash
brew install ffmpeg        # macOS — requires Homebrew (brew.sh)
sudo apt install ffmpeg    # Linux
```

If `brew install ffmpeg` fails with an Xcode license error:
```bash
sudo xcodebuild -license accept
```

If `brew` itself fails due to an untrusted tap (e.g. an old `mongodb` tap
from a different project), it's unrelated to this project — just allow it
for your terminal session and retry:
```bash
export HOMEBREW_NO_REQUIRE_TAP_TRUST=1
brew install ffmpeg
```

**Verify everything is installed correctly** before running the pipeline:
```bash
python -c "from insightface.app import FaceAnalysis; print('insightface: ok')"
ffmpeg -version
```

**macOS only — if you see an OpenMP error** (`OMP: Error #15: Initializing
libomp.dylib...`), this can still happen if PyTorch (used by `ultralytics`
for YOLO) collides with another package's bundled OpenMP runtime. Standard
workaround:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
Add it to your shell profile (`~/.zshrc`) if you don't want to repeat it
every session.

## Run

```bash
python main.py reference_photo.jpg video.mp4
```

This runs all 9 steps end to end and writes `phase1_output.json`:

```json
{
  "query_image": "reference_photo.jpg",
  "threshold": 0.4,
  "match_found": true,
  "matches": [
    {
      "frame_path": "frames/frame_0039.jpg",
      "timestamp": 38.0,
      "bbox": [320, 110, 455, 520],
      "confidence": 0.91,
      "crop_path": "person_crops/frame_0039_person_01.jpg",
      "similarity": 0.52
    }
  ]
}
```

## File-by-file (matches the proposal's 9 pipeline steps)

| File | Steps | What it does |
|---|---|---|
| `step1_extract_frames.py` | 1 | ffmpeg frame extraction at 1 fps |
| `step3_detect_people.py` | 3 | YOLOv8 person detection per frame |
| `step4_crop_people.py` | 4 | Crop each detected person, optional boxed-frame overlay |
| `step5_6_embed.py` | 5, 6 | ArcFace embeddings for crops and the reference photo |
| `step7_8_9_match.py` | 7, 8, 9 | Cosine similarity, threshold decision, JSON output |
| `main.py` | — | Wires all steps together |

## Notes on model choice

This pipeline uses ArcFace via `insightface`, not dlib's older face
encoder and not DINOv2 (a general-purpose image encoder that isn't
specialized for faces). ArcFace was chosen because:

- It's a stronger face-verification model than dlib's encoder
- `insightface` ships prebuilt wheels (onnxruntime backend) — no C++
  compiler, no cmake, and it avoids the OpenMP conflict that dlib caused
  when combined with PyTorch
- It's the model the original proposal named as the intended choice

**Important — the similarity threshold changed.** ArcFace's cosine
similarities for genuine matches typically fall in the 0.4–0.6 range,
much lower than dlib's ~0.85+. The default `THRESHOLD` in
`step7_8_9_match.py` has been updated to `0.4` to reflect this, but like
before, it's a starting point — tune it on labeled data (see below).

A backup of the old dlib-based embedding file is kept as
`step5_6_embed_dlib_OLD.py.bak` for reference, in case you ever want to
compare the two models directly.

## Tuning

- **`fps`** in `main.py` — increase if the video is fast-moving and you
  worry about missing the person between sampled frames.
- **`threshold`** — 0.4 is a reasonable ArcFace starting point but should
  be tuned on a labeled validation set (e.g. FaceSurv) to balance false
  positives vs. missed matches before trusting results on new video.
- **`yolov8n.pt`** in `step3_detect_people.py` — the nano model is fast
  but less accurate; swap to `yolov8s.pt` or `yolov8m.pt` for better
  detection at the cost of speed.

## Troubleshooting

- **`zsh: command not found: ffmpeg`** — ffmpeg isn't a Python package,
  so `pip install` never installs it. Install it with Homebrew (macOS) or
  apt (Linux) as shown above.
- **Homebrew refuses to run any command, mentions an untrusted tap** —
  unrelated to this project; some other tool you installed earlier (e.g.
  MongoDB) added an untrusted Homebrew tap. Use
  `export HOMEBREW_NO_REQUIRE_TAP_TRUST=1` for your terminal session
  rather than trying to fix the unrelated tap.
- **`OMP: Error #15` on macOS** — see the OpenMP note in Setup above.
- **`insightface` model download is slow on first run** — it downloads
  the `buffalo_l` model weights from its model zoo the first time
  `FaceAnalysis` is initialized. This is a one-time download, cached
  afterward.
