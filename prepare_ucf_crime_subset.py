"""
Phase 2 (UCF-Crime) -- Dataset prep: pull a manageable evaluation subset out
of the official UCF-Crime download.

Extracts only the specific .mp4 files named in
Temporal_Anomaly_Annotation_for_Testing_Videos.txt -- these are the only
videos with frame-level ground truth (training anomaly videos have no
localization annotation at all, by the dataset's weakly-supervised design;
see readme). Reads zip central directories to find members (fast, no
decompression) and extracts only the selected files, since the four
Anomaly-Videos-Part-*.zip archives are tens of GB each and nothing here
needs the rest of their contents.

Also attaches UCA (UCF-Crime Annotation, github.com/Xuange923/Surveillance-Video-Understanding)
captions to each selected video, when available: free-text sentences with
second-level temporal grounding, keyed by video basename. UCA's own
train/val/test split is INDEPENDENT of the anomaly-detection split used
above (cross-checked: of the 290 officially-testing videos, only 50 fall in
UCA's own Test.json -- the rest are scattered across UCA's Train/Val), so
this downloads and searches all three UCA files together rather than just
the one named "Test".

Usage:
    python prepare_ucf_crime_subset.py \\
        --zips ~/Downloads/Anomaly-Videos-Part-1.zip \\
               ~/Downloads/Anomaly-Videos-Part-2.zip \\
               ~/Downloads/Anomaly-Videos-Part-3.zip \\
               ~/Downloads/Anomaly-Videos-Part-4.zip \\
               ~/Downloads/Testing_Normal_Videos.zip \\
        --annotations ~/Downloads/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \\
        --out ./ucf_crime_subset \\
        --per-category 3

Pass --no-captions to skip the UCA download/attach step (e.g. if offline).
"""

import argparse
import json
import os
import random
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict

# Raw-file URLs for UCA's three splits. Small (low single-digit MB total)
# compared to the video zips, so unlike those this just downloads outright
# rather than needing selective extraction.
UCA_JSON_URLS = {
    "train": "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/main/UCF%20Annotation/json/UCFCrime_Train.json",
    "val": "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/main/UCF%20Annotation/json/UCFCrime_Val.json",
    "test": "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/main/UCF%20Annotation/json/UCFCrime_Test.json",
}


def parse_temporal_annotations(path: str) -> list[dict]:
    """Parse Temporal_Anomaly_Annotation_for_Testing_Videos.txt.

    Each line: video_name  category  start1  end1  start2  end2
    A value of -1 means "no event instance in that slot" (per the readme);
    Normal rows are always -1 -1 -1 -1. Returns one dict per video with an
    "events" list of (start_frame, end_frame) tuples, empty for Normal.
    """
    records = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            video, category = parts[0], parts[1]
            nums = [int(x) for x in parts[2:6]]
            events = [
                (nums[i], nums[i + 1])
                for i in range(0, 4, 2)
                if nums[i] >= 0 and nums[i + 1] >= 0
            ]
            records.append({"video": video, "category": category, "events": events})
    return records


def index_zips(zip_paths: list[str]) -> dict:
    """Map video basename -> (zip_path, member_name) across all given zips.

    Only reads each zip's central directory (namelist()), not file
    contents -- cheap even on multi-GB archives.
    """
    index = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".mp4"):
                    basename = os.path.basename(name)
                    if basename in index and index[basename][0] != zp:
                        print(f"[index_zips] WARNING: {basename} found in both "
                              f"{index[basename][0]} and {zp}; keeping the first")
                        continue
                    index[basename] = (zp, name)
    return index


def select_subset(records: list[dict], per_category: int, categories: list[str] | None,
                  seed: int) -> list[dict]:
    """Deterministically pick up to `per_category` videos per category."""
    by_category = defaultdict(list)
    for r in records:
        if categories is None or r["category"] in categories:
            by_category[r["category"]].append(r)

    rng = random.Random(seed)
    selected = []
    for category, items in sorted(by_category.items()):
        items = items[:]
        rng.shuffle(items)
        chosen = items[:per_category]
        if len(chosen) < per_category:
            print(f"[select_subset] {category}: only {len(chosen)} available "
                  f"(requested {per_category})")
        selected.extend(chosen)
    return selected


def extract_subset(selected: list[dict], index: dict, out_dir: str) -> list[dict]:
    """Extract each selected video's .mp4 into out_dir (flat, no subfolders).

    Returns the subset of `selected` that was actually found and extracted
    -- any name missing from `index` is skipped with a warning rather than
    failing the whole run, since a handful of not-yet-downloaded parts
    shouldn't block extracting the rest.
    """
    os.makedirs(out_dir, exist_ok=True)
    extracted = []

    # Group by source zip so each archive is opened once, not once per file.
    by_zip = defaultdict(list)
    for record in selected:
        video = record["video"]
        if video not in index:
            print(f"[extract_subset] WARNING: {video} not found in any provided "
                  f"zip -- skipping (is the zip containing its category downloaded?)")
            continue
        zip_path, member_name = index[video]
        by_zip[zip_path].append((member_name, record))

    for zip_path, members in by_zip.items():
        with zipfile.ZipFile(zip_path) as zf:
            for member_name, record in members:
                dst_path = os.path.join(out_dir, record["video"])
                with zf.open(member_name) as src, open(dst_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(record)
                print(f"[extract_subset] {record['video']} ({record['category']}) "
                      f"<- {os.path.basename(zip_path)}")

    return extracted


def write_subset_annotations(extracted: list[dict], out_path: str) -> None:
    """Write a Temporal_Anomaly_Annotation-format file covering only the
    extracted subset, so downstream eval code can read a small file instead
    of re-filtering the full 290-line original every time."""
    with open(out_path, "w") as f:
        for r in sorted(extracted, key=lambda r: (r["category"], r["video"])):
            events = r["events"] or [(-1, -1)]
            events = (events + [(-1, -1)])[:2]  # pad to exactly 2 slots
            flat = [x for pair in events for x in pair]
            f.write(f"{r['video']}  {r['category']}  " + "  ".join(map(str, flat)) + "\n")
    print(f"[write_subset_annotations] wrote {len(extracted)} rows to {out_path}")


def load_uca_captions(uca_dir: str) -> dict:
    """Download (if needed) and merge UCA's three split files into one dict
    keyed by video basename (no extension), e.g. "Abuse037_x264".

    Merged across train/val/test deliberately -- UCA's split is independent
    of the anomaly-detection split this script otherwise works from, so
    restricting to just Test.json would silently drop most of the overlap
    (measured: only 50/290 of the officially-testing videos land in UCA's
    own Test split; 286/290 have coverage once all three are combined).
    Each value also gets a "uca_split" key recording which UCA file it came
    from, in case that distinction matters downstream.
    """
    os.makedirs(uca_dir, exist_ok=True)
    merged = {}

    for split, url in UCA_JSON_URLS.items():
        dest = os.path.join(uca_dir, f"UCFCrime_{split.capitalize()}.json")
        if not os.path.exists(dest):
            print(f"[load_uca_captions] downloading {split} split -> {dest}")
            try:
                urllib.request.urlretrieve(url, dest)
            except urllib.error.URLError as e:
                print(f"[load_uca_captions] WARNING: could not download {url}: {e}")
                continue

        with open(dest) as f:
            split_data = json.load(f)
        for video_key, entry in split_data.items():
            entry = dict(entry)
            entry["uca_split"] = split
            merged[video_key] = entry

    split_counts = defaultdict(int)
    for entry in merged.values():
        split_counts[entry["uca_split"]] += 1
    counts_str = ", ".join(f"{s}={split_counts[s]}" for s in UCA_JSON_URLS)
    print(f"[load_uca_captions] {len(merged)} videos with UCA captions ({counts_str})")
    return merged


def attach_captions(extracted: list[dict], uca_captions: dict) -> None:
    """Attach a "captions" key (UCA entry, or None) to each extracted
    record in place, matching by video basename with the extension and any
    "_x264" suffix quirks left untouched -- UCA's keys already include
    "_x264" (e.g. "Abuse037_x264"), matching the video filename minus
    ".mp4" directly."""
    n_matched = 0
    for r in extracted:
        key = os.path.splitext(r["video"])[0]
        captions = uca_captions.get(key)
        r["captions"] = captions
        if captions is not None:
            n_matched += 1
        else:
            print(f"[attach_captions] no UCA captions found for {r['video']}")
    print(f"[attach_captions] {n_matched}/{len(extracted)} selected videos have UCA captions")


def write_manifest(extracted: list[dict], out_path: str) -> None:
    """Write one combined JSON manifest per video: category, VIRAT-style
    anomaly frame windows, and UCA captions (sentences + timestamps) side by
    side, so downstream code has a single file to read instead of joining
    the annotation txt and UCA json separately."""
    manifest = {}
    for r in extracted:
        manifest[r["video"]] = {
            "category": r["category"],
            "anomaly_frame_windows": r["events"],
            "captions": r.get("captions"),
        }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write_manifest] wrote {len(manifest)} entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", nargs="+", required=True,
                        help="Anomaly-Videos-Part-*.zip and/or Testing_Normal_Videos.zip")
    parser.add_argument("--annotations", required=True,
                        help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    parser.add_argument("--out", required=True, help="Output directory for extracted videos")
    parser.add_argument("--per-category", type=int, default=3,
                        help="Max videos to pull per category, including Normal (default 3)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Restrict to these categories (default: all present)")
    parser.add_argument("--seed", type=int, default=0, help="Selection RNG seed")
    parser.add_argument("--no-captions", action="store_true",
                        help="Skip downloading/attaching UCA captions")
    parser.add_argument("--uca-dir", default="./uca_annotations",
                        help="Directory to cache/download UCA's json files into")
    args = parser.parse_args()

    zip_paths = [os.path.expanduser(p) for p in args.zips]
    records = parse_temporal_annotations(os.path.expanduser(args.annotations))
    print(f"[main] {len(records)} testing-video annotations parsed "
          f"({len({r['category'] for r in records})} categories)")

    selected = select_subset(records, args.per_category, args.categories, args.seed)
    print(f"[main] selected {len(selected)} videos across "
          f"{len({r['category'] for r in selected})} categories")

    index = index_zips(zip_paths)
    print(f"[main] indexed {len(index)} .mp4 files across {len(zip_paths)} zip(s)")

    out_dir = os.path.expanduser(args.out)
    extracted = extract_subset(selected, index, out_dir)

    annotations_out = os.path.join(out_dir, "temporal_anomaly_annotation_subset.txt")
    write_subset_annotations(extracted, annotations_out)

    if not args.no_captions:
        uca_captions = load_uca_captions(os.path.expanduser(args.uca_dir))
        attach_captions(extracted, uca_captions)
    else:
        for r in extracted:
            r["captions"] = None

    manifest_out = os.path.join(out_dir, "subset_manifest.json")
    write_manifest(extracted, manifest_out)

    missing = len(selected) - len(extracted)
    if missing:
        print(f"[main] {missing} selected video(s) were not found in the provided zips")
    print(f"[main] done: {len(extracted)} videos in {out_dir}")


if __name__ == "__main__":
    main()
