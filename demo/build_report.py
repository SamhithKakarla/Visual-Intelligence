#!/usr/bin/env python3
"""
Build a single self-contained HTML report walking Phase-1's track-mode
pipeline end to end for one video + one reference photo: 1fps frames ->
detection/bbox -> crop -> embed -> track averaging -> match decision.

Run the production pipeline first so its intermediate artifacts exist in
the repo root:

    python main.py <headshot.jpg> <video.mp4> track

Then build the report from those same artifacts:

    python demo/build_report.py <headshot.jpg>

Opens as demo/report.html -- no server needed, everything (images, chart)
is embedded inline.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step5_6_embed import embed_query_image  # noqa: E402
from step7_8_9_match import match_query_to_tracks, THRESHOLD  # noqa: E402
from report_common import (  # noqa: E402
    thumb_data_uri, evenly_spaced, img_strip, build_similarity_chart,
    page_shell, verdict_block,
)


def render(**c):
    body = f"""
<section>
  <div class="step-badge">INPUTS</div>
  <h2>Reference photo &amp; query video</h2>
  <div class="desc">The two inputs to <code>main.py &lt;headshot&gt; &lt;video&gt; track</code>.</div>
  <div class="inputs-row">
    <img class="headshot" src="{c['query_uri']}">
    <div class="meta">
      <div><strong>{c['num_frames']}</strong> frames sampled at 1&nbsp;fps in this track</div>
      <div><strong>{c['n_all_frames']}</strong> total frames extracted from the video</div>
    </div>
  </div>
</section>

<section>
  <div class="step-badge">STEP 1</div>
  <h2>1 fps frame extraction</h2>
  <div class="desc">ffmpeg samples the video at 1 frame/sec ({c['n_all_frames']} frames total; showing an even sample below).</div>
  <div class="strip">{img_strip(c['frame_uris'])}</div>
</section>

<section>
  <div class="step-badge">STEP 3</div>
  <h2>Person detection &amp; bounding box</h2>
  <div class="desc">YOLOv8 detects each person per frame; boxes shown for this track's member frames.</div>
  <div class="strip">{img_strip(c['boxed_uris'], 'thumb wide')}</div>
</section>

<section>
  <div class="step-badge">STEP 4</div>
  <h2>Crop</h2>
  <div class="desc">Each detected person is cropped independently before embedding.</div>
  <div class="strip">{img_strip(c['crop_uris'], 'thumb crop')}</div>
</section>

<section>
  <div class="step-badge">STEPS 5&ndash;6, TRACK MODE</div>
  <h2>Per-frame embedding &amp; track averaging</h2>
  <div class="desc">
    Every crop gets a 512-dim ArcFace embedding, scored against the reference photo individually (bars).
    In track mode, this track's embeddings are averaged into one template before the final score
    (dashed "track avg" line) &mdash; this is what cancels per-frame noise like blur and pose.
  </div>
  {c['chart_svg']}
</section>

<section>
  <div class="step-badge">STEPS 7&ndash;9</div>
  <h2>Decision</h2>
  {verdict_block(c['is_match'], c['track_similarity'], c['threshold'],
                 f"{c['num_frames']} frames averaged into this track's template")}
</section>
"""
    return page_shell(
        "Phase 1 — Facial Recognition & Tracking",
        "Track-mode walkthrough for one data point &middot; ArcFace / InsightFace buffalo_l",
        body,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="path to reference headshot photo")
    ap.add_argument("--detections", default=os.path.join(ROOT, "detections_with_embeddings.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "report.html"))
    ap.add_argument("--frame-samples", type=int, default=18)
    ap.add_argument("--track-samples", type=int, default=10)
    args = ap.parse_args()

    with open(args.detections) as f:
        detections = json.load(f)

    query_vec = embed_query_image(args.query)
    matches, tracks = match_query_to_tracks(query_vec, detections, THRESHOLD)
    # match_query_to_tracks mutates `detections` in place, adding "track_id" to each

    if not tracks:
        print("No tracks found -- did detection/embedding produce any faces?", file=sys.stderr)
        sys.exit(1)

    chosen = matches[0] if matches else max(tracks, key=lambda t: t["similarity"])
    track_id = chosen["track_id"]
    members = sorted(
        (d for d in detections if d["track_id"] == track_id),
        key=lambda d: d.get("timestamp", 0),
    )
    for m in members:
        m["own_similarity"] = float(cosine_similarity([query_vec], [np.array(m["embedding"])])[0][0])
        m["chart_label"] = f"t={m.get('timestamp', 0):.1f}s"

    is_match = chosen["similarity"] >= THRESHOLD

    frames_dir = os.path.join(ROOT, "frames")
    all_frames = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")
    ) if os.path.isdir(frames_dir) else []
    frame_sample = evenly_spaced(all_frames, args.frame_samples)
    track_sample = evenly_spaced(members, args.track_samples)

    boxed_uris, crop_uris = [], []
    for m in track_sample:
        frame_name = os.path.splitext(os.path.basename(m["frame_path"]))[0]
        boxed_path = os.path.join(ROOT, "boxed_frames", f"{frame_name}.jpg")
        boxed_uris.append(thumb_data_uri(boxed_path, max_w=220))
        crop_uris.append(thumb_data_uri(m.get("crop_path"), max_w=140))

    chart_svg = build_similarity_chart(members, chosen["similarity"], THRESHOLD, avg_label="track avg")

    html = render(
        query_uri=thumb_data_uri(args.query, max_w=260),
        frame_uris=[thumb_data_uri(p, max_w=160) for p in frame_sample],
        n_all_frames=len(all_frames),
        boxed_uris=boxed_uris,
        crop_uris=crop_uris,
        chart_svg=chart_svg,
        track_similarity=chosen["similarity"],
        threshold=THRESHOLD,
        is_match=is_match,
        num_frames=len(members),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
