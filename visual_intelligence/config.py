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
    identity_threshold: float = 0.2
    face_detection_threshold: float = 0.15
    face_samples_per_track: int = 20
    similarity_top_k: int = 3
    # Stop trying more candidates for a track once `similarity_top_k` of
    # them have ALL independently cleared this bar -- unlike the naive
    # "stop after N successes" exit removed earlier (which let one lucky
    # frame end the search on thin, unconfirmed evidence), this requires
    # several samples to agree, which a genuinely wrong person is unlikely
    # to produce by chance. 0.6 is well outside the noisy overlap zone
    # observed between real weak matches and coincidental false positives.
    early_exit_on_high_confidence: bool = True
    early_exit_confidence_threshold: float = 0.6
    # When True, a second track that also independently clears
    # identity_threshold is only admitted as evidence if its own face
    # (compared directly to the best track's face, NOT the reference
    # photo) matches the single highest-scoring track. merge_fragmented_tracks
    # already consolidates genuine tracker-fragmentation for the SAME
    # person upstream of this, so an unmatching second track by this point
    # is more likely to be a different, similarly-scored person (observed
    # directly: a track can score consistently above threshold, averaged
    # over hundreds of detections, and still be the wrong person if they
    # happen to look similar to the reference photo) than a missed
    # fragment of the real target. A track that DOES match the best
    # track's face is kept in full, preserving multi-fragment evidence for
    # genuine same-person cases instead of discarding it outright.
    #
    # A color-histogram version of this check was tried first and failed:
    # shared background/lighting/skin tone swamped the actual clothing
    # signal, so two genuinely different people scored 0.56-0.78 "similar"
    # to each other on a 0-1 scale. Comparing the two tracks' faces
    # directly (cosine similarity between ArcFace embeddings, same scale
    # as identity_score, not the histogram's 0-1 correlation) is the
    # signal we've already shown is somewhat discriminative here.
    restrict_evidence_to_best_track: bool = True
    evidence_face_min_similarity: float = 0.35
    merge_fragmented_tracks: bool = True
    merge_max_gap_seconds: float = 3.0
    merge_max_distance_ratio: float = 4.0
    merge_min_appearance_similarity: float = 0.35
    appearance_gap_seconds: float = 1.5
    # A group with fewer raw detections than this is discarded rather than
    # reported as its own appearance. Without this, a handful of low-
    # confidence tracks that repeatedly interleave in time (rather than one
    # continuous handoff) can each trigger the spatial/appearance continuity
    # check in build_appearance_groups, producing dozens of single-frame
    # "appearances" -- each gets its own Phase 2 call, which is both
    # incoherent output and a real cost blowup for no signal.
    min_appearance_detections: int = 3
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
        if not -1 <= self.early_exit_confidence_threshold <= 1:
            raise ValueError("early_exit_confidence_threshold must be between -1 and 1")
        if self.merge_max_gap_seconds < 0:
            raise ValueError("merge_max_gap_seconds cannot be negative")
        if self.merge_max_distance_ratio <= 0:
            raise ValueError("merge_max_distance_ratio must be positive")
        if not 0 <= self.merge_min_appearance_similarity <= 1:
            raise ValueError("merge_min_appearance_similarity must be between zero and one")
        if not -1 <= self.evidence_face_min_similarity <= 1:
            raise ValueError("evidence_face_min_similarity must be between -1 and 1")
        if self.appearance_gap_seconds < 0 or self.context_padding_seconds < 0:
            raise ValueError("appearance timing values cannot be negative")
        if self.min_appearance_detections <= 0:
            raise ValueError("min_appearance_detections must be positive")
        if self.max_evidence_frames <= 0:
            raise ValueError("max_evidence_frames must be positive")
        if self.crop_margin_ratio < 0:
            raise ValueError("crop_margin_ratio cannot be negative")


@dataclass(slots=True)
class Phase2Config:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    # Enough headroom to finish a well-formed JSON object even for an
    # appearance with a long description and a full evidence_timestamps
    # array (up to max_evidence_frames entries) -- 256 was tight enough
    # that generation could hit the cap mid-object, producing truncated,
    # unparseable JSON for exactly the appearances with the most evidence.
    max_new_tokens: int = 768
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
