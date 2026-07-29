#!/usr/bin/env python3
"""
Build a landing page tying the two demos together into one story:
  Part 1 - the pipeline walkthrough on one self-filmed clip (report.html)
  Part 2 - the same track-mode matching validated at scale on FaceSurv (facesurv_report.html)

Run both build_report.py and build_facesurv_report.py first so their
outputs (phase1_output.json, facesurv_benchmark/facesurv_results.json) and
HTML pages exist, then:

    python demo/build_index.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "facesurv_benchmark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_common import thumb_data_uri, page_shell  # noqa: E402


def card(href, badge, title, desc, stat_html, thumb_html=""):
    return f"""
<a class="card" href="{href}">
  <div class="step-badge">{badge}</div>
  <h2>{title}</h2>
  <div class="desc">{desc}</div>
  <div class="card-bottom">
    {thumb_html}
    <div class="stat-row">{stat_html}</div>
  </div>
  <div class="card-cta">Open report &rarr;</div>
</a>
"""


def stat(value, label):
    return f'<div class="stat"><div class="value">{value}</div><div class="label">{label}</div></div>'


def main():
    phase1_path = os.path.join(ROOT, "phase1_output.json")
    results_path = os.path.join(BENCH_DIR, "facesurv_results.json")
    missing = [p for p in (phase1_path, results_path) if not os.path.exists(p)]
    if missing:
        print("missing: " + ", ".join(missing) + " -- run build_report.py and build_facesurv_report.py first",
              file=sys.stderr)
        sys.exit(1)

    with open(phase1_path) as f:
        phase1 = json.load(f)
    with open(results_path) as f:
        bench = json.load(f)

    top_match = phase1["matches"][0] if phase1.get("matches") else None
    query_thumb = thumb_data_uri(phase1["query_image"], max_w=200) or ""

    part1_stats = (
        stat(f"{top_match['similarity']:.3f}" if top_match else "no match", "Top similarity")
        + stat(top_match["num_frames"] if top_match else "-", "Frames averaged")
        + stat(f"&ge; {phase1['threshold']}", "Threshold")
    )
    part1_thumb = f'<img class="headshot" src="{query_thumb}">' if query_thumb else ""

    tl = bench["track_level"]
    v = bench.get("verification", {})
    part2_stats = (
        stat(f"{tl['rank1_accuracy']*100:.1f}%", "Track-level rank-1")
        + stat(f"{tl['rank5_accuracy']*100:.1f}%", "Track-level rank-5")
        + stat(v.get("roc_auc", "-"), "ROC-AUC")
        + stat(bench["probe_tracks_total"], "Probe tracks")
    )

    body = f"""
<div class="cards">
{card("report.html", "PART 1", "Pipeline walkthrough",
      "One data point, start to finish: 1fps frames &rarr; detection &amp; bounding box &rarr; "
      "crop &rarr; embed &rarr; track averaging &rarr; match decision, on a self-filmed clip.",
      part1_stats, part1_thumb)}
{card("facesurv_report.html", "PART 2", "Validated at scale &mdash; FaceSurv benchmark",
      "The same track-mode matching, this time measured against the FaceSurv surveillance "
      f"dataset across all {bench['probe_tracks_total']} probe tracks, with one example shown in detail.",
      part2_stats)}
</div>
"""
    extra_css = """
.cards { display:flex; flex-direction:column; gap: 20px; }
.card { display:block; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 22px 24px; text-decoration:none; color: inherit; }
.card:hover { border-color: var(--series-1); }
.card-bottom { display:flex; gap: 24px; align-items:center; margin-top: 10px; }
.card-cta { margin-top: 14px; font-size: 13px; font-weight:600; color: var(--series-1); }
"""
    html = page_shell(
        "Phase 1 — Facial Recognition & Tracking: Demo",
        "Two-part story: one live example, then validated across the full benchmark.",
        body,
    ).replace("</style>", extra_css + "</style>")

    out_path = os.path.join(ROOT, "demo", "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
