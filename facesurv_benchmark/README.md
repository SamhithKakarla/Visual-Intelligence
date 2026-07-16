# FaceSurv — Phase-1 Identification Benchmark

Evaluates the Phase-1 face-recognition core (InsightFace `buffalo_l` / ArcFace,
the same embedder as [`step5_6_embed.py`](../step5_6_embed.py)) on the **FaceSurv**
surveillance dataset, and reports identification + verification accuracy.

FaceSurv ships pre-cropped, identity-labeled face frames, so this benchmark skips
the pipeline's frame-extraction / YOLO / crop stages and goes straight to
embed → cosine-match against per-subject gallery templates.

## Dataset (not committed)

Expected at `~/Desktop/UCLA/capstone/FaceSurv`:

- `Gallery/<id>_<n>.jpg` — 3 enrollment portraits per subject (230 subjects)
- `FaceSurv_DayData [Annotations]/<track>/…` and `FaceSurv_NightData [Annotations]/…`
  — per-subject-per-video dirs of cropped face frames. Track name is
  `D_S<sess>_V<vid>_<subjectID>`; the **trailing number is the ground-truth id**.

## Run

```bash
# from the repo root, with the pipeline venv active
KMP_DUPLICATE_LIB_OK=TRUE python facesurv_benchmark/facesurv_benchmark.py \
    --sessions day,night --max-frames 10
```

Useful flags: `--limit-tracks N` (quick smoke test), `--sessions day`,
`--det-size`, `--pad`. Gallery templates are cached to
`gallery_templates_det<sz>_pad<pad>.npz` (enrolling the hi-res portraits takes
~4 min); delete the cache if you change `--det-size`/`--pad`.

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
