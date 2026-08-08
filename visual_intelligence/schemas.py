"""Typed data contracts shared by Phase 1, Phase 2, and evaluation code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Valence = Literal["positive", "negative", "neutral"]
BBox = tuple[int, int, int, int]


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Detection:
    frame_index: int
    timestamp: float
    track_id: int
    bbox: BBox
    confidence: float
    frame_path: str
    identity_similarity: float | None = None


@dataclass(slots=True)
class EvidenceFrame(Serializable):
    frame_index: int
    timestamp: float
    bbox: BBox
    detection_confidence: float
    identity_similarity: float | None
    context_frame_path: str
    target_crop_path: str


@dataclass(slots=True)
class Appearance(Serializable):
    appearance_id: int
    start_time: float
    end_time: float
    track_ids: list[int]
    frames: list[EvidenceFrame] = field(default_factory=list)


@dataclass(slots=True)
class Phase1Result(Serializable):
    reference_image: str
    reference_video: str
    person_exists: bool
    identity_score: float
    threshold: float
    appearances: list[Appearance] = field(default_factory=list)
    best_track_id: int | None = None
    run_directory: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AppearanceAnalysis(Serializable):
    appearance_id: int
    start_time: float
    end_time: float
    description: str
    classification: Valence
    confidence: float
    evidence_timestamps: list[float] = field(default_factory=list)


@dataclass(slots=True)
class Phase2Result(Serializable):
    activity_description: str
    activity_classification: Valence
    appearances: list[AppearanceAnalysis] = field(default_factory=list)


@dataclass(slots=True)
class FinalResult(Serializable):
    reference_image: str
    reference_video: str
    person_exists: bool
    activity_description: str | None
    activity_classification: Valence | None
