#!/usr/bin/env python3
"""
Build a single self-contained HTML report walking the FaceSurv benchmark's
track-mode evaluation end to end for one probe track: gallery enrollment ->
probe crops (FaceSurv ships these pre-cropped, no 1fps/YOLO step here) ->
per-frame embedding -> track averaging -> match decision -- shown alongside
the full 720-track benchmark numbers for context.

Run the benchmark at least once first so the gallery cache and
facesurv_per_track.csv exist:

    cd facesurv_benchmark && python facesurv_benchmark.py

Then build the report:

    python demo/build_facesurv_report.py                # auto-picks a track
    python demo/build_facesurv_report.py --track D_S1_V3_12

Opens as demo/facesurv_report.html -- no server needed.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "facesurv_benchmark")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facesurv_benchmark as fb  # noqa: E402
from step7_8_9_match import aggregate_track_embeddings  # noqa: E402
from report_common import (  # noqa: E402
    thumb_data_uri, img_strip, build_similarity_chart, page_shell, verdict_block,
)


def pick_track(per_track_csv, requested):
    with open(per_track_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if requested:
        for r in rows:
            if r["track"] == requested:
                return r
        raise ValueError(f"track '{requested}' not found in {per_track_csv}")

    correct = [r for r in rows if r["track_rank1_correct"] == "True"]
    if not correct:
        raise ValueError("no correctly-matched tracks in facesurv_per_track.csv to auto-pick from")
    return max(correct, key=lambda r: int(r["frames_used"]))


def render(**c):
    m = c["metrics"]
    fl, tl = m["frame_level"], m["track_level"]

    def stat(value, label):
        return f'<div class="stat"><div class="value">{value}</div><div class="label">{label}</div></div>'

    body = f"""
<section>
  <div class="step-badge">GALLERY ENROLLMENT</div>
  <h2>Reference portraits &mdash; subject {c['gt_id']}</h2>
  <div class="desc">Enrollment template = mean of these portraits' embeddings (from <code>Gallery/</code>).</div>
  <div class="inputs-row">
    <div class="strip">{img_strip(c['gallery_uris'], 'thumb wide')}</div>
    <div class="meta">
      <div>Track: <strong>{c['track_name']}</strong> ({c['session']})</div>
      <div><strong>{c['num_frames']}</strong> probe frames used</div>
    </div>
  </div>
</section>

<section>
  <div class="step-badge">PROBE TRACK</div>
  <h2>Pre-cropped face frames</h2>
  <div class="desc">
    FaceSurv ships identity-labeled face crops directly (this is why the benchmark
    skips the pipeline's frame-extraction / YOLO / crop steps and goes straight to embedding).
  </div>
  <div class="strip">{img_strip(c['probe_uris'], 'thumb wide')}</div>
</section>

<section>
  <div class="step-badge">EMBED + TRACK AVERAGING</div>
  <h2>Per-frame similarity &amp; track averaging</h2>
  <div class="desc">
    Every probe crop gets a 512-dim ArcFace embedding, scored against subject {c['gt_id']}'s gallery
    template individually (bars). Track mode averages this track's embeddings into one template
    before the final score (dashed "track avg" line).
  </div>
  {c['chart_svg']}
</section>

<section>
  <div class="step-badge">DECISION</div>
  <h2>This track</h2>
  {verdict_block(c['is_match'], c['track_similarity'], c['threshold'],
                 f"{c['num_frames']} frames averaged &middot; ground truth subject {c['gt_id']}")}
</section>

<section>
  <div class="step-badge">FULL BENCHMARK</div>
  <h2>In context: all {m['probe_tracks_total']} probe tracks ({m['config']['sessions']})</h2>
  <div class="desc">This one track is a single data point inside the full evaluation:</div>
  <div class="stat-row">
    {stat(f"{tl['rank1_accuracy']*100:.1f}%", "Track-level rank-1")}
    {stat(f"{tl['rank5_accuracy']*100:.1f}%", "Track-level rank-5")}
    {stat(f"{fl['rank1_accuracy']*100:.1f}%", "Frame-level rank-1")}
    {stat(m.get('verification', {}).get('roc_auc', '—'), "Verification ROC-AUC")}
    {stat(f"{m.get('verification', {}).get('eer', 0)*100:.1f}%" if m.get('verification') else '—', "EER")}
    {stat(m['gallery_subjects'], "Gallery subjects")}
  </div>
</section>
"""
    return page_shell(
        "FaceSurv Benchmark — Track-mode Walkthrough",
        "One probe track in detail, in context of the full evaluation &middot; ArcFace / InsightFace buffalo_l",
        body,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=None, help="specific track name (default: auto-pick a strong correct example)")
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--pad", type=float, default=0.4)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "facesurv_report.html"))
    args = ap.parse_args()

    per_track_csv = os.path.join(BENCH_DIR, "facesurv_per_track.csv")
    results_json = os.path.join(BENCH_DIR, "facesurv_results.json")
    if not os.path.exists(per_track_csv) or not os.path.exists(results_json):
        print(f"run facesurv_benchmark.py first (missing {per_track_csv} or {results_json})", file=sys.stderr)
        sys.exit(1)

    row = pick_track(per_track_csv, args.track)
    session, track_name, gt_id = row["session"], row["track"], int(row["gt_id"])

    gal_ids, gal_templates = fb.build_gallery(args.det_size, args.pad, BENCH_DIR)
    gal_pos = {sid: i for i, sid in enumerate(gal_ids.tolist())}
    if gt_id not in gal_pos:
        print(f"subject {gt_id} not in cached gallery -- rebuild the gallery cache first", file=sys.stderr)
        sys.exit(1)
    gt_template = gal_templates[gal_pos[gt_id]]

    track_dir = os.path.join(fb.SESSION_DIRS[session], track_name)
    frame_paths = sorted(glob.glob(os.path.join(glob.escape(track_dir), "*.jpg")), key=fb.frame_index)

    members = []
    for fp in frame_paths:
        v = fb.embed(fp, args.det_size, args.pad)
        if v is None:
            continue
        sim = float(gt_template @ v)
        members.append({
            "path": fp, "embedding": v, "own_similarity": sim,
            "chart_label": os.path.basename(fp),
        })

    if not members:
        print(f"no detectable faces in track '{track_name}'", file=sys.stderr)
        sys.exit(1)

    track_template = aggregate_track_embeddings([m["embedding"] for m in members])
    track_similarity = float(gt_template @ track_template)
    is_match = track_similarity >= fb.PIPELINE_THRESHOLD

    gallery_paths = sorted(glob.glob(os.path.join(fb.GALLERY_DIR, f"{gt_id}_*.jpg")))

    with open(results_json) as f:
        metrics = json.load(f)

    chart_svg = build_similarity_chart(members, track_similarity, fb.PIPELINE_THRESHOLD, avg_label="track avg")

    html = render(
        gt_id=gt_id,
        track_name=track_name,
        session=session,
        num_frames=len(members),
        gallery_uris=[thumb_data_uri(p, max_w=150) for p in gallery_paths],
        probe_uris=[thumb_data_uri(m["path"], max_w=150) for m in members],
        chart_svg=chart_svg,
        track_similarity=track_similarity,
        threshold=fb.PIPELINE_THRESHOLD,
        is_match=is_match,
        metrics=metrics,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  (track={track_name}, subject={gt_id}, similarity={track_similarity:.3f})")


if __name__ == "__main__":
    main()
