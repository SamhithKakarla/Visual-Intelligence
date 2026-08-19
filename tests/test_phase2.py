from __future__ import annotations

import unittest

from visual_intelligence.phase2.pipeline import (
    aggregate_analyses,
    parse_json_object,
    split_video_metadata,
    validate_analysis_payload,
)
from visual_intelligence.schemas import (
    Appearance,
    AppearanceAnalysis,
    EvidenceFrame,
)


def appearance() -> Appearance:
    return Appearance(
        appearance_id=1,
        start_time=2.0,
        end_time=5.0,
        track_ids=[4],
        frames=[
            EvidenceFrame(
                frame_index=8,
                timestamp=2.0,
                bbox=(1, 2, 3, 4),
                detection_confidence=0.9,
                identity_similarity=0.62,
                context_frame_path="context.jpg",
                target_crop_path="target.jpg",
            )
        ],
    )


class Phase2Tests(unittest.TestCase):
    def test_qwen3_video_metadata_is_split_from_video_tensors(self):
        videos, metadata = split_video_metadata(
            [("frames-a", {"fps": 2.0}), ("frames-b", {"fps": 1.0})]
        )
        self.assertEqual(videos, ["frames-a", "frames-b"])
        self.assertEqual(metadata, [{"fps": 2.0}, {"fps": 1.0}])

    def test_missing_qwen3_video_inputs_remain_none(self):
        self.assertEqual(split_video_metadata(None), (None, None))

    def test_fenced_json_is_parsed(self):
        payload = parse_json_object('```json\n{"classification":"neutral"}\n```')
        self.assertEqual(payload["classification"], "neutral")

    def test_payload_validation_filters_unknown_timestamps(self):
        result = validate_analysis_payload(
            {
                "description": "The target walks toward the camera.",
                "classification": "Neutral",
                "confidence": 1.2,
                "evidence_timestamps": [2.0, 99],
            },
            appearance(),
        )
        self.assertEqual(result.classification, "neutral")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.evidence_timestamps, [2.0])

    def test_negative_appearance_dominates_video_valence(self):
        analyses = [
            AppearanceAnalysis(1, 2.0, 5.0, "The target walks.", "neutral", 0.9),
            AppearanceAnalysis(2, 40.0, 43.0, "The target strikes another person.", "negative", 0.8),
        ]
        result = aggregate_analyses(analyses)
        self.assertEqual(result.activity_classification, "negative")
        self.assertIn("2.00s", result.activity_description)
        self.assertIn("40.00s", result.activity_description)

    def test_invalid_valence_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_analysis_payload(
                {
                    "description": "The target walks.",
                    "classification": "suspicious",
                    "confidence": 0.5,
                },
                appearance(),
            )


if __name__ == "__main__":
    unittest.main()
