from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_intelligence.phase1.pipeline import (
    aggregate_similarities,
    build_appearance_groups,
    select_evenly_spaced,
)
from visual_intelligence.schemas import Detection

MAX_DISTANCE_RATIO = 4.0
MIN_APPEARANCE_SIMILARITY = 0.35


def detection(
    timestamp: float,
    track_id: int = 1,
    bbox: tuple[int, int, int, int] = (10, 20, 110, 220),
    frame_path: str = "frame.jpg",
) -> Detection:
    return Detection(
        frame_index=round(timestamp * 4),
        timestamp=timestamp,
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        frame_path=frame_path,
    )


class AppearanceGroupingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Same solid-color frame for a plausible, visually-consistent handoff.
        self.blue_frame = self.tmpdir / "blue.jpg"
        cv2.imwrite(str(self.blue_frame), np.full((240, 320, 3), (200, 120, 20), dtype="uint8"))
        # A distinctly different-colored frame for an implausible handoff --
        # a different person's clothing, not the same one under a new track ID.
        self.red_frame = self.tmpdir / "red.jpg"
        cv2.imwrite(str(self.red_frame), np.full((240, 320, 3), (20, 20, 220), dtype="uint8"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _groups(self, observations, gap_seconds=1.5):
        return build_appearance_groups(
            observations,
            gap_seconds,
            max_distance_ratio=MAX_DISTANCE_RATIO,
            min_appearance_similarity=MIN_APPEARANCE_SIMILARITY,
        )

    def test_far_apart_observations_remain_disjoint(self):
        observations = [
            detection(2.0, frame_path=str(self.blue_frame)),
            detection(3.0, frame_path=str(self.blue_frame)),
            detection(4.0, frame_path=str(self.blue_frame)),
            detection(40.0, track_id=8, frame_path=str(self.blue_frame)),
            detection(41.0, track_id=8, frame_path=str(self.blue_frame)),
            detection(42.0, track_id=8, frame_path=str(self.blue_frame)),
        ]
        groups = self._groups(observations)
        self.assertEqual(len(groups), 2)
        self.assertEqual([item.timestamp for item in groups[0]], [2.0, 3.0, 4.0])
        self.assertEqual([item.timestamp for item in groups[1]], [40.0, 41.0, 42.0])

    def test_same_track_id_always_merges_regardless_of_frame_content(self):
        # No continuity check needed (or run) when the track_id is unchanged
        # -- same tracker identity, trivially the same appearance.
        groups = self._groups([
            detection(2.0, track_id=1, frame_path=str(self.blue_frame)),
            detection(2.5, track_id=1, frame_path="does-not-exist.jpg"),
            detection(3.0, track_id=1, frame_path="also-does-not-exist.jpg"),
        ])
        self.assertEqual(len(groups), 1)

    def test_adjacent_tracker_fragment_of_the_same_person_still_merges(self):
        # Different track_id (tracker ID switch after a brief occlusion), but
        # spatially adjacent and visually consistent -- a plausible handoff.
        groups = self._groups([
            detection(2.0, track_id=1, bbox=(10, 20, 110, 220), frame_path=str(self.blue_frame)),
            detection(2.5, track_id=9, bbox=(15, 22, 115, 222), frame_path=str(self.blue_frame)),
            detection(3.0, track_id=9, bbox=(18, 24, 118, 224), frame_path=str(self.blue_frame)),
        ])
        self.assertEqual(len(groups), 1)

    def test_different_person_temporally_adjacent_does_not_merge(self):
        # This is the bug this test suite is here to catch: two different
        # people who each independently cleared the identity threshold,
        # temporally adjacent but NOT the same person -- must not be folded
        # into one appearance just because they're close in time.
        groups = self._groups([
            detection(2.0, track_id=1, bbox=(10, 20, 110, 220), frame_path=str(self.blue_frame)),
            detection(2.5, track_id=9, bbox=(10, 20, 110, 220), frame_path=str(self.red_frame)),
        ])
        self.assertEqual(len(groups), 2)

    def test_spatially_distant_handoff_does_not_merge_even_if_visually_similar(self):
        groups = self._groups([
            detection(2.0, track_id=1, bbox=(10, 20, 110, 220), frame_path=str(self.blue_frame)),
            detection(2.5, track_id=9, bbox=(2000, 2000, 2100, 2200), frame_path=str(self.blue_frame)),
        ])
        self.assertEqual(len(groups), 2)

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


if __name__ == "__main__":
    unittest.main()
