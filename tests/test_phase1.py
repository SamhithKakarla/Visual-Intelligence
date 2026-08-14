from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_intelligence.config import Phase1Config
from visual_intelligence.phase1.pipeline import (
    FaceObservation,
    build_appearance_groups,
    embed_reference_image,
    run_phase1,
    score_tracks,
    select_evenly_spaced,
)
from visual_intelligence.schemas import (
    Detection,
    EvidenceFrame,
    TrackIdentityDecision,
)


def detection(timestamp: float, track_id: int = 1) -> Detection:
    return Detection(
        frame_index=round(timestamp * 4),
        timestamp=timestamp,
        track_id=track_id,
        bbox=(10, 20, 110, 220),
        confidence=0.9,
        frame_path=f"frame_{timestamp}.jpg",
    )


def embedding_with_similarity(similarity: float) -> list[float]:
    return [similarity, math.sqrt(max(0.0, 1.0 - similarity * similarity))]


def face(
    similarity: float,
    *,
    width: float = 80.0,
    height: float = 90.0,
    confidence: float = 0.9,
) -> FaceObservation:
    return FaceObservation(
        embedding=embedding_with_similarity(similarity),
        bbox=(0.0, 0.0, width, height),
        detector_confidence=confidence,
        width=width,
        height=height,
    )


class FakeImage:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.shape = (300, 200, 3)
        self.size = 180_000

    def __getitem__(self, _key):
        return self


def _fake_cv2():
    fake_module = types.ModuleType("cv2")
    fake_module.imread = lambda path: FakeImage(tag=str(path))
    return patch.dict(sys.modules, {"cv2": fake_module})


class FakeEmbedder:
    def __init__(self, observations_by_tag: dict[str, FaceObservation]) -> None:
        self._observations_by_tag = observations_by_tag

    def analyze_image(self, image: FakeImage):
        return self._observations_by_tag.get(image.tag)

    def embed_image(self, image: FakeImage):
        observation = self.analyze_image(image)
        return observation.embedding if observation is not None else None


def score_one_track(
    scores: list[float],
    *,
    config: Phase1Config | None = None,
    observations: dict[str, FaceObservation] | None = None,
):
    track = [detection(index / 4) for index in range(len(scores))]
    mapping = {"reference.jpg": face(1.0)}
    mapping.update(
        {
            item.frame_path: face(score)
            for item, score in zip(track, scores)
        }
    )
    if observations:
        mapping.update(observations)
    with _fake_cv2():
        decisions, warnings = score_tracks(
            "reference.jpg",
            track,
            config or Phase1Config(),
            embedder=FakeEmbedder(mapping),
        )
    return decisions[1], warnings


class AppearanceGroupingTests(unittest.TestCase):
    def test_far_apart_observations_remain_disjoint(self):
        observations = [
            detection(2.0),
            detection(3.0),
            detection(4.0),
            detection(40.0, track_id=8),
            detection(41.0, track_id=8),
            detection(42.0, track_id=8),
        ]
        groups = build_appearance_groups(observations, gap_seconds=1.5)
        self.assertEqual(len(groups), 2)
        self.assertEqual([item.timestamp for item in groups[0]], [2.0, 3.0, 4.0])
        self.assertEqual([item.timestamp for item in groups[1]], [40.0, 41.0, 42.0])

    def test_adjacent_tracker_fragments_are_one_appearance(self):
        groups = build_appearance_groups(
            [detection(2.0, 1), detection(2.5, 9), detection(3.0, 9)],
            gap_seconds=1.5,
        )
        self.assertEqual(len(groups), 1)

    def test_even_sampling_keeps_endpoints(self):
        observations = [detection(float(index)) for index in range(10)]
        selected = select_evenly_spaced(observations, maximum=4)
        self.assertEqual(selected[0].timestamp, 0.0)
        self.assertEqual(selected[-1].timestamp, 9.0)
        self.assertEqual(len(selected), 4)


class IdentityConsensusTests(unittest.TestCase):
    def test_reference_image_must_contain_a_face(self):
        with _fake_cv2():
            with self.assertRaisesRegex(ValueError, "No face detected"):
                embed_reference_image("reference.jpg", FakeEmbedder({}))

    def test_three_clear_matches_accept_track(self):
        decision, _ = score_one_track([0.47, 0.44, 0.42])
        self.assertTrue(decision.matched)
        self.assertEqual(decision.decision_reason, "matched_consensus")
        self.assertAlmostEqual(decision.aggregate_score, 0.4433, places=4)

    def test_one_strong_frame_is_not_enough(self):
        decision, _ = score_one_track([0.8])
        self.assertFalse(decision.matched)
        self.assertEqual(decision.decision_reason, "insufficient_faces")

    def test_two_strong_frames_are_not_enough(self):
        decision, _ = score_one_track([0.8, 0.7])
        self.assertFalse(decision.matched)
        self.assertEqual(decision.decision_reason, "insufficient_faces")

    def test_consensus_requires_two_scores_above_threshold(self):
        decision, _ = score_one_track([0.8, 0.3, 0.2])
        self.assertFalse(decision.matched)
        self.assertEqual(
            decision.decision_reason,
            "insufficient_scores_above_threshold",
        )

    def test_all_observations_can_supply_the_best_three(self):
        decision, _ = score_one_track(
            [0.1, 0.12, 0.14, 0.16, 0.18, 0.43, 0.45, 0.47]
        )
        self.assertTrue(decision.matched)
        self.assertEqual(decision.observations_examined, 8)
        self.assertEqual(decision.selected_scores, [0.47, 0.45, 0.43])

    def test_small_and_low_confidence_faces_are_excluded(self):
        track = [detection(index / 4) for index in range(5)]
        mapping = {
            "reference.jpg": face(1.0),
            track[0].frame_path: face(0.9, width=30),
            track[1].frame_path: face(0.85, confidence=0.4),
            track[2].frame_path: face(0.47),
            track[3].frame_path: face(0.45),
            track[4].frame_path: face(0.43),
        }
        with _fake_cv2():
            decisions, _ = score_tracks(
                "reference.jpg",
                track,
                Phase1Config(),
                embedder=FakeEmbedder(mapping),
            )
        decision = decisions[1]
        self.assertTrue(decision.matched)
        self.assertEqual(decision.eligible_faces, 3)
        self.assertEqual(decision.rejected_too_small, 1)
        self.assertEqual(decision.rejected_low_confidence, 1)
        self.assertNotIn(0.9, decision.selected_scores)

    def test_missing_face_is_ignored_without_lowering_consensus(self):
        track = [detection(index / 4) for index in range(4)]
        mapping = {
            "reference.jpg": face(1.0),
            track[1].frame_path: face(0.47),
            track[2].frame_path: face(0.45),
            track[3].frame_path: face(0.43),
        }
        with _fake_cv2():
            decisions, _ = score_tracks(
                "reference.jpg",
                track,
                Phase1Config(),
                embedder=FakeEmbedder(mapping),
            )
        decision = decisions[1]
        self.assertTrue(decision.matched)
        self.assertEqual(decision.rejected_no_face, 1)
        self.assertEqual(decision.aggregate_score, 0.45)

    def test_long_tracks_use_bounded_deterministic_observations(self):
        track = [detection(index / 4) for index in range(100)]
        mapping = {"reference.jpg": face(1.0)}
        mapping.update({item.frame_path: face(0.5) for item in track})
        with _fake_cv2():
            decisions, _ = score_tracks(
                "reference.jpg",
                track,
                Phase1Config(max_identity_observations_per_track=64),
                embedder=FakeEmbedder(mapping),
            )
        decision = decisions[1]
        self.assertTrue(decision.matched)
        self.assertEqual(decision.observations_examined, 64)
        self.assertEqual(decision.eligible_faces, 64)


class Phase1HandoffTests(unittest.TestCase):
    def test_accepted_identity_forwards_complete_track_to_appearance_builder(self):
        track = [detection(index / 4) for index in range(5)]
        decision = TrackIdentityDecision(
            track_id=1,
            matched=True,
            decision_reason="matched_consensus",
            aggregate_score=0.45,
            total_detections=5,
            observations_examined=5,
            faces_detected=3,
            eligible_faces=3,
            rejected_no_face=2,
            rejected_too_small=0,
            rejected_low_confidence=0,
            selected_scores=[0.47, 0.45, 0.43],
            selected_timestamps=[0.0, 0.25, 0.5],
        )
        captured_timestamps: list[float] = []

        def fake_evidence(group, appearance_id, output_dir, config):
            captured_timestamps.extend(item.timestamp for item in group)
            first = group[0]
            return [
                EvidenceFrame(
                    frame_index=first.frame_index,
                    timestamp=first.timestamp,
                    bbox=first.bbox,
                    detection_confidence=first.confidence,
                    identity_similarity=first.identity_similarity,
                    context_frame_path="context.jpg",
                    target_crop_path="target.jpg",
                )
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.jpg"
            video = root / "video.mp4"
            reference.touch()
            video.touch()
            with (
                patch(
                    "visual_intelligence.phase1.pipeline.extract_search_frames",
                    return_value=[root / "frame.jpg"],
                ),
                patch(
                    "visual_intelligence.phase1.pipeline.detect_and_track_people",
                    return_value=track,
                ),
                patch(
                    "visual_intelligence.phase1.pipeline.score_tracks",
                    return_value=({1: decision}, []),
                ),
                patch(
                    "visual_intelligence.phase1.pipeline.create_evidence_frames",
                    side_effect=fake_evidence,
                ),
            ):
                result = run_phase1(reference, video, root / "run")

        self.assertTrue(result.person_exists)
        self.assertEqual(captured_timestamps, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(len(result.appearances), 1)


if __name__ == "__main__":
    unittest.main()
