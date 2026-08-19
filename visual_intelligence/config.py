"""Configuration objects for the two pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Phase1Config:
    search_fps: float = 4.0
    yolo_model: str = "yolov8n.pt"
    tracker: str = "botsort.yaml"
    person_confidence: float = 0.5
    identity_threshold: float = 0.4
    face_samples_per_track: int = 5
    similarity_top_k: int = 3
    appearance_gap_seconds: float = 1.5
    context_padding_seconds: float = 0.75
    max_evidence_frames: int = 48
    crop_margin_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.search_fps <= 0:
            raise ValueError("search_fps must be greater than zero")
        if not 0 <= self.person_confidence <= 1:
            raise ValueError("person_confidence must be between zero and one")
        if self.face_samples_per_track <= 0 or self.similarity_top_k <= 0:
            raise ValueError("face sampling counts must be positive")
        if self.appearance_gap_seconds < 0 or self.context_padding_seconds < 0:
            raise ValueError("appearance timing values cannot be negative")
        if self.max_evidence_frames <= 0:
            raise ValueError("max_evidence_frames must be positive")
        if self.crop_margin_ratio < 0:
            raise ValueError("crop_margin_ratio cannot be negative")


@dataclass(slots=True)
class Phase2Config:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_new_tokens: int = 512
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
