# FaceSurv — Phase-1 Identification Benchmark

Evaluates the Phase-1 face-recognition core (InsightFace `buffalo_l` / ArcFace,
the same embedder as [`step5_6_embed.py`](../step5_6_embed.py)) on the **FaceSurv**
surveillance dataset, and reports identification + verification accuracy.

FaceSurv ships pre-cropped, identity-labeled face frames, so this benchmark skips
the pipeline's frame-extraction / YOLO / crop stages and goes straight to
embed → cosine-match against per-subject gallery templates.

## Setup

1. Follow the [root README](../README.md) first — venv, `pip install -r
   requirements.txt`, `ffmpeg`, and the macOS OpenMP workaround
   (`KMP_DUPLICATE_LIB_OK=TRUE`, used below). The benchmark doesn't call
   `ffmpeg` itself (FaceSurv ships pre-cropped frames), but it still needs
   `insightface`/`onnxruntime` from that same venv.
2. Get the FaceSurv dataset from whoever shared it for the capstone and save
   it anywhere on disk (see layout below).
3. Point the benchmark at it — by default it looks in
   `~/Desktop/UCLA/Capstone/FaceSurv`, but that's just a fallback, not a
   requirement. Override it with the `FACESURV_DIR` env var if you saved the
   dataset somewhere else:

   ```bash
   export FACESURV_DIR=/path/to/your/FaceSurv
   ```

## Dataset (not committed)

Layout expected under `$FACESURV_DIR`:

- `Gallery/<id>_<n>.jpg` — 3 enrollment portraits per subject (230 subjects)
- `FaceSurv_DayData [Annotations]/<track>/…` and `FaceSurv_NightData [Annotations]/…`
  — per-subject-per-video dirs of cropped face frames. Track name is
  `D_S<sess>_V<vid>_<subjectID>`; the **trailing number is the ground-truth id**.

## Run

```bash
# from the repo root, with the venv active
KMP_DUPLICATE_LIB_OK=TRUE python facesurv_benchmark/facesurv_benchmark.py \
    --sessions day,night --max-frames 10
```

Useful flags: `--limit-tracks N` (quick smoke test), `--sessions day`,
`--det-size`, `--pad`. Gallery templates are cached to
`facesurv_benchmark/gallery_templates_det<sz>_pad<pad>.npz` (enrolling the
hi-res portraits takes ~4 min). The cache is gitignored and regenerates on
first run; delete it if you change `--det-size`/`--pad`.

## Results (day + night, 720 tracks, 10 frames/track)

| Metric | Frame-level | Track-level |
|---|---|---|
| Rank-1 accuracy | 91.9% | **98.4%** |
| Rank-5 accuracy | 96.1% | 99.1% |

Verification: ROC-AUC **0.986**, EER 4.7%, TAR@FAR=1% 90.9%. Face-detection rate
on the tight crops: 85.8% (black-border padding is required or SCRFD returns 0
faces). See [`facesurv_RESULTS.md`](facesurv_RESULTS.md) for the full report,
`facesurv_results.json` for machine-readable metrics, and
`facesurv_per_track.csv` for per-track predictions.

**Note:** the pipeline's hardcoded match threshold of 0.4 is too high for
frame-level surveillance matching (drops frame rank-1 to 67%); track-level
aggregation avoids this. Consider ~0.25–0.3 (EER-optimal) or track-level scoring.

## Frame-level vs. track-level (this branch)

Two matching strategies are compared:

- **Frame-level** — score every detection independently (the original pipeline
  behavior). Baseline branch: `eagle/facesurv-benchmark`.
- **Track-level** — group a person's detections into a track and *average* their
  embeddings into one template before scoring. This branch
  (`eagle/facesurv-track-level`) implements it in the pipeline itself
  (`step7_8_9_match.py` + `main.py`, `mode="track"`), and the benchmark calls the
  pipeline's shared `aggregate_track_embeddings()` so both measure the same op.

| Approach | Rank-1 | Rank-5 |
|---|---|---|
| Frame-level | 91.9% | 96.1% |
| **Track-level (averaging)** | **98.4%** | **99.1%** |

Averaging cancels per-frame noise (blur, pose, partial occlusion), recovering
~6.5 points of rank-1 accuracy. Run the production pipeline in either mode:

```bash
python main.py <reference_photo.jpg> <video.mp4> frame   # per-frame (default)
python main.py <reference_photo.jpg> <video.mp4> track    # averaged per track
```

In `track` mode the pipeline groups detections with a lightweight greedy-IoU
tracker (suitable for the ~1 fps sampling), averages each track, and reports one
score per track instead of per frame.
