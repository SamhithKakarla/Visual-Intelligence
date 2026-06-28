"""
Phase 1 — Facial Recognition & Tracking
Step 1: Frame Extraction (FFmpeg)

Extracts frames from the source video at a fixed interval (default 1 FPS)
into an intermediate storage directory.
"""

import subprocess
import os


def extract_frames(video_path: str, output_dir: str = "frames", fps: int = 1):
    """
    Extract frames from a video at the given fps rate.

    Args:
        video_path: path to the source video file
        output_dir: directory to write extracted frames into
        fps: frames per second to extract (1 = one frame every second)

    Returns:
        sorted list of frame filenames written to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            f"{output_dir}/frame_%04d.jpg",
            "-hide_banner", "-loglevel", "error",
        ],
        check=True,
    )

    frames = sorted(f for f in os.listdir(output_dir) if f.endswith(".jpg"))
    print(f"[extract_frames] Extracted {len(frames)} frames to '{output_dir}/' at {fps} fps")
    return frames


def frame_index_to_timestamp(frame_filename: str, fps: int = 1) -> float:
    """
    Convert a frame filename (e.g. frame_0039.jpg) to a timestamp in seconds,
    given the extraction fps used.
    """
    idx = int(frame_filename.replace("frame_", "").replace(".jpg", ""))
    return (idx - 1) / fps


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python step1_extract_frames.py <video_path>")
        sys.exit(1)
    extract_frames(sys.argv[1])
