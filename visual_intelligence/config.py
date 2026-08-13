"""Configuration objects for the two pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Phase1Config:
    search_fps: float = 4.0
    yolo_model: str = "yolov8n.pt"
    tracker: str = "botsort.yaml"
    # Auto-selected in place of `tracker` for a video whose measured
    # frame-to-frame global motion falls below `static_camera_threshold_px`
    # -- skips global motion compensation, which is wasted compute (and
    # only wasted compute) on a genuinely fixed/mounted camera. Any video
    # with real measured motion keeps using `tracker` unchanged.
    static_camera_tracker: str = str(
        Path(__file__).parent / "phase1" / "trackers" / "botsort_static_camera.yaml"
    )
    auto_detect_static_camera: bool = True
    static_camera_threshold_px: float = 1.5
    static_camera_motion_samples: int = 8
    person_confidence: float = 0.5
    identity_threshold: float = 0.15
    face_detection_threshold: float = 0.15
    face_samples_per_track: int = 20
    similarity_top_k: int = 3
    merge_fragmented_tracks: bool = True
    merge_max_gap_seconds: float = 3.0
    merge_max_distance_ratio: float = 4.0
    merge_min_appearance_similarity: float = 0.35
    appearance_gap_seconds: float = 1.5
    context_padding_seconds: float = 0.75
    max_evidence_frames: int = 48
    crop_margin_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.search_fps <= 0:
            raise ValueError("search_fps must be greater than zero")
        if not 0 <= self.person_confidence <= 1:
            raise ValueError("person_confidence must be between zero and one")
        if self.static_camera_threshold_px < 0:
            raise ValueError("static_camera_threshold_px cannot be negative")
        if self.static_camera_motion_samples <= 0:
            raise ValueError("static_camera_motion_samples must be positive")
        if not 0 <= self.face_detection_threshold <= 1:
            raise ValueError("face_detection_threshold must be between zero and one")
        if self.face_samples_per_track <= 0 or self.similarity_top_k <= 0:
            raise ValueError("face sampling counts must be positive")
        if self.merge_max_gap_seconds < 0:
            raise ValueError("merge_max_gap_seconds cannot be negative")
        if self.merge_max_distance_ratio <= 0:
            raise ValueError("merge_max_distance_ratio must be positive")
        if not 0 <= self.merge_min_appearance_similarity <= 1:
            raise ValueError("merge_min_appearance_similarity must be between zero and one")
        if self.appearance_gap_seconds < 0 or self.context_padding_seconds < 0:
            raise ValueError("appearance timing values cannot be negative")
        if self.max_evidence_frames <= 0:
            raise ValueError("max_evidence_frames must be positive")
        if self.crop_margin_ratio < 0:
            raise ValueError("crop_margin_ratio cannot be negative")


@dataclass(slots=True)
class Phase2Config:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_new_tokens: int = 256
    attn_implementation: str | None = "sdpa"
    device_map: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(slots=True)
class PipelineConfig:
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    runs_dir: Path = Path("runs")
    keep_artifacts: bool = True
