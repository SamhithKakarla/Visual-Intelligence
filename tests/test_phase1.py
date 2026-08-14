from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from visual_intelligence.config import Phase1Config
from visual_intelligence.phase1.pipeline import (
    aggregate_similarities,
    build_appearance_groups,
    embed_reference_images,
    score_tracks,
    select_evenly_spaced,
)
from visual_intelligence.schemas import Detection


def detection(timestamp: float, track_id: int = 1) -> Detection:
    return Detection(
        frame_index=round(timestamp * 4),
        timestamp=timestamp,
        track_id=track_id,
        bbox=(10, 20, 110, 220),
        confidence=0.9,
        frame_path=f"frame_{timestamp}.jpg",
    )


class FakeImage:
    """Stands in for a decoded cv2 frame; tagged so a fake embedder can
    recognize which source path it came from without real image data."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.shape = (10, 10, 3)
        self.size = 300

    def __getitem__(self, _key):
        return self


def _fake_cv2():
    """cv2 is imported lazily inside the pipeline functions under test, so
    patch("cv2.imread", ...) would require the real package to be installed
    just to resolve the target. Install a stand-in module instead, keeping
    these tests runnable without the (heavy, unavailable-in-CI) cv2 dependency."""
    fake_module = types.ModuleType("cv2")
    fake_module.imread = lambda path: FakeImage(tag=path)
    return patch.dict(sys.modules, {"cv2": fake_module})


class FakeEmbedder:
    """Returns a deterministic embedding per source path instead of running
    a real face model, so multi-reference aggregation can be tested exactly."""

    def __init__(self, embeddings_by_tag: dict[str, "object"]) -> None:
        self._embeddings_by_tag = embeddings_by_tag

    def embed_image(self, image: FakeImage):
        return self._embeddings_by_tag.get(image.tag)


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

    def test_top_k_similarity_aggregation(self):
        self.assertAlmostEqual(
            aggregate_similarities([0.1, 0.9, 0.7, 0.5], top_k=2),
            0.8,
        )


class MultiReferenceScoringTests(unittest.TestCase):
    """A candidate frame matching only one of several reference angles should
    still score on that match, not get diluted by the angles it doesn't."""

    def setUp(self):
        # Two reference "angles" of the same subject: deliberately orthogonal
        # embeddings so a candidate can match one perfectly and the other not
        # at all, making the aggregation behavior unambiguous to assert on.
        self.frontal_embedding = [1.0, 0.0]
        self.profile_embedding = [0.0, 1.0]
        # A candidate face that only resembles the profile reference.
        self.candidate_embedding = [0.0, 1.0]

    def test_embed_reference_images_collects_one_embedding_per_image(self):
        embedder = FakeEmbedder({
            "ref_frontal.jpg": self.frontal_embedding,
            "ref_profile.jpg": self.profile_embedding,
        })
        with _fake_cv2():
            embeddings = embed_reference_images(
                ["ref_frontal.jpg", "ref_profile.jpg"], embedder
            )
        self.assertEqual(len(embeddings), 2)

    def test_embed_reference_images_raises_when_every_image_has_no_face(self):
        embedder = FakeEmbedder({})  # no path recognized -> embed_image returns None
        with _fake_cv2():
            with self.assertRaisesRegex(ValueError, "No face detected"):
                embed_reference_images(["ref_frontal.jpg"], embedder)

    def test_candidate_scores_on_best_matching_reference_angle(self):
        embedder = FakeEmbedder({
            "ref_frontal.jpg": self.frontal_embedding,
            "ref_profile.jpg": self.profile_embedding,
            "candidate.jpg": self.candidate_embedding,
        })
        track_detection = detection(0.0)
        track_detection.frame_path = "candidate.jpg"

        with _fake_cv2():
            single_ref_scores, _ = score_tracks(
                ["ref_frontal.jpg"], [track_detection], Phase1Config(), embedder=embedder
            )
            multi_ref_scores, _ = score_tracks(
                ["ref_frontal.jpg", "ref_profile.jpg"],
                [track_detection],
                Phase1Config(),
                embedder=embedder,
            )

        # Against only the frontal reference, the profile-matching candidate
        # scores as a total mismatch (orthogonal embeddings -> cosine 0).
        self.assertAlmostEqual(single_ref_scores[1], 0.0)
        # Adding the profile reference lets the same candidate match on its
        # best angle instead of being averaged down or missed entirely.
        self.assertAlmostEqual(multi_ref_scores[1], 1.0)


if __name__ == "__main__":
    unittest.main()
