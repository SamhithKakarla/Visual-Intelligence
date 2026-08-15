from __future__ import annotations

import unittest

from visual_intelligence.phase1.pipeline import (
    aggregate_similarities,
    build_appearance_groups,
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


if __name__ == "__main__":
    unittest.main()
