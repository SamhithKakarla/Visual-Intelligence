#!/usr/bin/env python3
"""
FaceSurv Phase-1 identification benchmark.

Runs the Phase-1 face-recognition core (ArcFace / InsightFace buffalo_l, the
same model step5_6_embed.py uses) against the FaceSurv surveillance dataset and
reports identification + verification accuracy metrics.

FaceSurv already ships pre-cropped face frames with ground-truth identity, so
the frame-extraction / YOLO person-detection / cropping steps of the pipeline
are not needed here -- we go straight to embedding + matching.

Dataset layout (under $FACESURV_DIR, default ~/Desktop/UCLA/Capstone/FaceSurv):
  Gallery/<id>_<n>.jpg                         enrollment portraits (3 per subject)
  FaceSurv_DayData [Annotations]/<track>/...   probe face crops, one dir per subject-in-video
  FaceSurv_NightData [Annotations]/<track>/... probe face crops (night)
  where <track> = D_S<sess>_V<vid>_<subjectID>, trailing number = ground-truth id.

Protocol:
  * Enroll one L2-normalized template per gallery subject (mean of their portraits).
  * For each probe track, subsample frames, embed, and match by cosine similarity
    against every gallery template.
  * Report frame-level and track-level rank-1 / rank-5 identification accuracy,
    plus verification ROC-AUC / EER / TAR@FAR from genuine vs impostor scores.
  * Tracks whose subject is not enrolled are treated as open-set distractors.

Usage:
  python facesurv_benchmark.py                    # full run (day + night)
  python facesurv_benchmark.py --limit-tracks 5   # quick smoke test
  python facesurv_benchmark.py --sessions day --max-frames 8
"""

import os
import re
import sys
import json
import glob
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Use the pipeline's own averaging primitive so the benchmark's track-level
# numbers reflect exactly what the shipped pipeline (step7_8_9_match.py) does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step7_8_9_match import aggregate_track_embeddings  # noqa: E402

FACESURV = os.path.expanduser(
    os.environ.get("FACESURV_DIR", "~/Desktop/UCLA/Capstone/FaceSurv")
)
GALLERY_DIR = os.path.join(FACESURV, "Gallery")
SESSION_DIRS = {
    "day": os.path.join(FACESURV, "FaceSurv_DayData [Annotations]"),
    "night": os.path.join(FACESURV, "FaceSurv_NightData [Annotations]"),
}

# Same operating threshold the Phase-1 pipeline uses for a positive match.
PIPELINE_THRESHOLD = 0.4

_app = None


def get_app(det_size):
    """Lazy-load the InsightFace ArcFace model once (slow to construct)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        print(f"[facesurv] Loading InsightFace (buffalo_l / ArcFace), det_size={det_size}...")
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return _app


def embed(path, det_size, pad):
    """
    Return a 512-d L2-normalized ArcFace embedding for one image, or None if no
    face is detected. FaceSurv probe crops are tight face boxes; InsightFace's
    SCRFD detector often misses edge-to-edge faces, so we pad with a black
    border first (same trick the pipeline uses for tight portraits).
    """
    bgr = cv2.imread(path)
    if bgr is None or bgr.size == 0:
        return None
    if pad > 0:
        h, w = bgr.shape[:2]
        py, px = int(round(h * pad)), int(round(w * pad))
        bgr = cv2.copyMakeBorder(bgr, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    faces = get_app(det_size).get(bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.normed_embedding


def frame_index(path):
    """Trailing integer in a probe frame filename (for stable ordering)."""
    m = re.search(r"_(\d+)\.jpg$", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def track_subject_id(track_name):
    """Ground-truth subject id = trailing number of the track dir name."""
    m = re.search(r"_(\d+)$", track_name)
    return int(m.group(1)) if m else None


def subsample(items, k):
    """Evenly-spaced subsample of at most k items, preserving order."""
    if k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def build_gallery(det_size, pad, cache_dir):
    """Enroll one mean, L2-normalized template per gallery subject (cached to disk)."""
    cache = os.path.join(cache_dir, f"gallery_templates_det{det_size}_pad{pad}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"[facesurv] Loaded cached gallery templates from {os.path.basename(cache)}")
        return z["ids"], z["templates"]

    paths = sorted(glob.glob(os.path.join(GALLERY_DIR, "*.jpg")))
    by_id = defaultdict(list)
    for p in paths:
        sid = int(os.path.basename(p).split("_")[0])
        by_id[sid].append(p)

    ids, templates = [], []
    failed = 0
    print(f"[facesurv] Enrolling {len(by_id)} gallery subjects from {len(paths)} portraits...")
    for sid in sorted(by_id):
        vecs = [v for v in (embed(p, det_size, pad) for p in by_id[sid]) if v is not None]
        if not vecs:
            failed += 1
            continue
        t = np.mean(vecs, axis=0)
        t = t / (np.linalg.norm(t) + 1e-9)
        ids.append(sid)
        templates.append(t)
    if failed:
        print(f"[facesurv] WARNING: {failed} gallery subjects had no detectable face")
    ids_arr, templ_arr = np.array(ids), np.vstack(templates)
    np.savez(cache, ids=ids_arr, templates=templ_arr)
    return ids_arr, templ_arr


def collect_tracks(sessions, limit):
    """Return list of (session, track_name, [frame_paths]) for requested sessions."""
    tracks = []
    for sess in sessions:
        base = SESSION_DIRS[sess]
        # Dir names contain "[Annotations]"; "[...]" is a glob char-class, so
        # escape the literal path before globbing under it.
        for d in sorted(glob.glob(os.path.join(glob.escape(base), "*/"))):
            name = os.path.basename(d.rstrip("/"))
            frames = sorted(glob.glob(os.path.join(glob.escape(d), "*.jpg")), key=frame_index)
            if frames:
                tracks.append((sess, name, frames))
    if limit:
        tracks = tracks[:limit]
    return tracks


def eer_and_tar(y_true, scores):
    """Equal-error-rate and TAR at fixed FARs from genuine/impostor scores."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)

    def tar_at(far):
        ok = np.where(fpr <= far)[0]
        return float(tpr[ok[-1]]) if len(ok) else 0.0

    return eer, tar_at(0.01), tar_at(0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="day,night",
                    help="comma list of day,night (default both)")
    ap.add_argument("--max-frames", type=int, default=10,
                    help="max frames sampled per probe track (default 10)")
    ap.add_argument("--det-size", type=int, default=320,
                    help="InsightFace detector input size (default 320)")
    ap.add_argument("--pad", type=float, default=0.4,
                    help="black-border padding fraction for tight crops (default 0.4)")
    ap.add_argument("--impostors-per-frame", type=int, default=5,
                    help="random impostor templates sampled per frame for the ROC")
    ap.add_argument("--limit-tracks", type=int, default=0,
                    help="cap number of probe tracks (0 = all; for quick tests)")
    ap.add_argument("--out-prefix", default="facesurv",
                    help="output filename prefix (written into this dir)")
    args = ap.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    rng = np.random.default_rng(0)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()

    gal_ids, gal_templates = build_gallery(args.det_size, args.pad, out_dir)
    gal_pos = {sid: i for i, sid in enumerate(gal_ids.tolist())}
    print(f"[facesurv] Gallery: {len(gal_ids)} enrolled subjects\n")

    tracks = collect_tracks(sessions, args.limit_tracks)
    print(f"[facesurv] Probe tracks: {len(tracks)} across sessions {sessions}\n")

    # Accumulators
    frame_rank1 = frame_rank5 = frame_known = 0
    track_rank1 = track_rank5 = track_known = 0
    probe_faces = probe_missing = 0
    at_threshold_correct = 0        # frame-level rank-1 correct AND sim >= pipeline threshold
    track_at_threshold_correct = 0  # track-level rank-1 correct AND sim >= pipeline threshold

    ver_scores, ver_labels = [], []      # verification genuine/impostor
    known_top1_sims, distractor_top1_sims = [], []  # for open-set separability
    per_track_rows = []

    for ti, (sess, name, frames) in enumerate(tracks, 1):
        gt = track_subject_id(name)
        known = gt in gal_pos
        sampled = subsample(frames, args.max_frames)

        track_vecs = []
        track_frame_top1 = []
        for fp in sampled:
            v = embed(fp, args.det_size, args.pad)
            if v is None:
                probe_missing += 1
                continue
            probe_faces += 1
            track_vecs.append(v)

            sims = gal_templates @ v  # cosine (all L2-normalized)
            order = np.argsort(-sims)
            top1_id = int(gal_ids[order[0]])
            top1_sim = float(sims[order[0]])
            track_frame_top1.append(top1_id)

            if known:
                frame_known += 1
                top5 = gal_ids[order[:5]].tolist()
                if top1_id == gt:
                    frame_rank1 += 1
                    if top1_sim >= PIPELINE_THRESHOLD:
                        at_threshold_correct += 1
                if gt in top5:
                    frame_rank5 += 1
                # verification: genuine vs sampled impostors
                gpos = gal_pos[gt]
                ver_scores.append(float(sims[gpos]))
                ver_labels.append(1)
                others = [j for j in range(len(gal_ids)) if j != gpos]
                for j in rng.choice(others, size=min(args.impostors_per_frame, len(others)),
                                    replace=False):
                    ver_scores.append(float(sims[j]))
                    ver_labels.append(0)
                known_top1_sims.append(top1_sim)
            else:
                distractor_top1_sims.append(top1_sim)

        # Track-level: average the track's embeddings into one template (via the
        # pipeline's shared aggregator) -> single template -> rank.
        track_pred = None
        track_correct = track_top5 = False
        if track_vecs:
            m = aggregate_track_embeddings(track_vecs)
            sims = gal_templates @ m
            order = np.argsort(-sims)
            track_pred = int(gal_ids[order[0]])
            track_top1_sim = float(sims[order[0]])
            if known:
                track_known += 1
                top5 = gal_ids[order[:5]].tolist()
                track_correct = (track_pred == gt)
                track_top5 = (gt in top5)
                track_rank1 += int(track_correct)
                track_rank5 += int(track_top5)
                if track_correct and track_top1_sim >= PIPELINE_THRESHOLD:
                    track_at_threshold_correct += 1

        per_track_rows.append({
            "session": sess, "track": name, "gt_id": gt, "in_gallery": known,
            "frames_used": len(track_vecs), "track_pred": track_pred,
            "track_rank1_correct": bool(track_correct) if known else None,
        })

        if ti % 25 == 0 or ti == len(tracks):
            el = time.time() - t0
            r1 = frame_rank1 / frame_known if frame_known else 0
            print(f"  [{ti}/{len(tracks)}] {el:5.0f}s  frames={probe_faces} "
                  f"frame-rank1={r1:.3f} track-rank1="
                  f"{(track_rank1/track_known if track_known else 0):.3f}")

    # ---- Metrics ----
    metrics = {
        "config": vars(args),
        "gallery_subjects": int(len(gal_ids)),
        "probe_tracks_total": len(tracks),
        "probe_tracks_known": int(track_known),
        "probe_tracks_distractor": int(len(tracks) - track_known),
        "probe_faces_embedded": probe_faces,
        "probe_faces_no_detection": probe_missing,
        "face_detection_rate": round(probe_faces / (probe_faces + probe_missing), 4)
                               if (probe_faces + probe_missing) else 0.0,
        "frame_level": {
            "n": frame_known,
            "rank1_accuracy": round(frame_rank1 / frame_known, 4) if frame_known else None,
            "rank5_accuracy": round(frame_rank5 / frame_known, 4) if frame_known else None,
            "rank1_accuracy_at_threshold": round(at_threshold_correct / frame_known, 4)
                                           if frame_known else None,
            "pipeline_threshold": PIPELINE_THRESHOLD,
        },
        "track_level": {
            "n": track_known,
            "rank1_accuracy": round(track_rank1 / track_known, 4) if track_known else None,
            "rank5_accuracy": round(track_rank5 / track_known, 4) if track_known else None,
            "rank1_accuracy_at_threshold": round(track_at_threshold_correct / track_known, 4)
                                           if track_known else None,
            "pipeline_threshold": PIPELINE_THRESHOLD,
        },
    }

    if len(set(ver_labels)) == 2:
        y = np.array(ver_labels)
        s = np.array(ver_scores)
        eer, tar1, tar01 = eer_and_tar(y, s)
        metrics["verification"] = {
            "genuine_pairs": int(y.sum()),
            "impostor_pairs": int((y == 0).sum()),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "eer": round(eer, 4),
            "tar_at_far_1pct": round(tar1, 4),
            "tar_at_far_0.1pct": round(tar01, 4),
            "genuine_mean_cosine": round(float(s[y == 1].mean()), 4),
            "impostor_mean_cosine": round(float(s[y == 0].mean()), 4),
        }

    if known_top1_sims and distractor_top1_sims:
        metrics["open_set_top1_similarity"] = {
            "known_mean": round(float(np.mean(known_top1_sims)), 4),
            "distractor_mean": round(float(np.mean(distractor_top1_sims)), 4),
            "note": "known top-1 sims should sit above distractor top-1 sims; "
                    "the gap indicates open-set separability.",
        }

    metrics["runtime_seconds"] = round(time.time() - t0, 1)

    # ---- Write outputs ----
    json_path = os.path.join(out_dir, f"{args.out_prefix}_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(out_dir, f"{args.out_prefix}_per_track.csv")
    with open(csv_path, "w") as f:
        f.write("session,track,gt_id,in_gallery,frames_used,track_pred,track_rank1_correct\n")
        for r in per_track_rows:
            f.write(f"{r['session']},{r['track']},{r['gt_id']},{r['in_gallery']},"
                    f"{r['frames_used']},{r['track_pred']},{r['track_rank1_correct']}\n")

    md_path = os.path.join(out_dir, f"{args.out_prefix}_RESULTS.md")
    write_report(md_path, metrics)

    print("\n" + "=" * 60)
    print("FaceSurv Phase-1 benchmark complete")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote:\n  {json_path}\n  {csv_path}\n  {md_path}")


def write_report(path, m):
    fl, tl = m["frame_level"], m["track_level"]
    v = m.get("verification", {})
    lines = [
        "# FaceSurv — Phase-1 Face Identification Benchmark",
        "",
        "Model: InsightFace **buffalo_l / ArcFace** (512-d), the same embedder used by "
        "the Phase-1 pipeline (`step5_6_embed.py`). Match = cosine similarity vs. "
        "per-subject gallery templates.",
        "",
        "## Setup",
        f"- Gallery (enrolled subjects): **{m['gallery_subjects']}** "
        "(mean of each subject's portraits)",
        f"- Probe tracks: **{m['probe_tracks_total']}** total "
        f"({m['probe_tracks_known']} enrolled / {m['probe_tracks_distractor']} open-set distractors)",
        f"- Probe faces embedded: **{m['probe_faces_embedded']}** "
        f"(face-detection rate {m['face_detection_rate']:.1%}; "
        f"{m['probe_faces_no_detection']} crops had no detectable face)",
        f"- Frames sampled per track: up to {m['config']['max_frames']}; "
        f"sessions: {m['config']['sessions']}",
        "",
        "## Identification accuracy (closed-set, enrolled subjects only)",
        "",
        "| Metric | Frame-level | Track-level |",
        "|---|---|---|",
        f"| Rank-1 accuracy | {_pct(fl['rank1_accuracy'])} | {_pct(tl['rank1_accuracy'])} |",
        f"| Rank-5 accuracy | {_pct(fl['rank5_accuracy'])} | {_pct(tl['rank5_accuracy'])} |",
        f"| Rank-1 @ threshold (≥{fl['pipeline_threshold']}) | "
        f"{_pct(fl['rank1_accuracy_at_threshold'])} | {_pct(tl['rank1_accuracy_at_threshold'])} |",
        f"| N (probe items) | {fl['n']} | {tl['n']} |",
        "",
        "Rank-1 counts a probe correct when the true subject is its nearest gallery "
        "template (no threshold). Rank-1 @ threshold additionally requires that top "
        f"similarity to clear the pipeline's operating point (cosine ≥ {fl['pipeline_threshold']}) "
        "— i.e. what the live pipeline would actually accept.",
        "",
    ]
    if v:
        lines += [
            "## Verification metrics (genuine vs. impostor cosine scores)",
            "",
            f"- ROC-AUC: **{v['roc_auc']}**",
            f"- EER: **{_pct(v['eer'])}**",
            f"- TAR @ FAR=1%: **{_pct(v['tar_at_far_1pct'])}**",
            f"- TAR @ FAR=0.1%: **{_pct(v['tar_at_far_0.1pct'])}**",
            f"- Mean cosine — genuine {v['genuine_mean_cosine']} vs. "
            f"impostor {v['impostor_mean_cosine']}",
            f"- Pairs: {v['genuine_pairs']} genuine / {v['impostor_pairs']} impostor",
            "",
        ]
    if "open_set_top1_similarity" in m:
        o = m["open_set_top1_similarity"]
        lines += [
            "## Open-set separability",
            f"- Mean top-1 cosine for **enrolled** probes: {o['known_mean']}",
            f"- Mean top-1 cosine for **distractor** probes: {o['distractor_mean']}",
            "",
        ]
    lines += [f"_Runtime: {m['runtime_seconds']}s. Generated by `facesurv_benchmark.py`._"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _pct(x):
    return "—" if x is None else f"{x*100:.2f}%"


if __name__ == "__main__":
    main()
