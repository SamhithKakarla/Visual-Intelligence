from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from visual_intelligence.config import PipelineConfig
from visual_intelligence.pipeline import finalize_from_phase1, run_pipeline
from visual_intelligence.schemas import (
    Appearance,
    AppearanceAnalysis,
    EvidenceFrame,
    Phase1Result,
)


class FailingAnalyzer:
    def analyze(self, appearance):
        raise AssertionError("Phase 2 must not run for an absent subject")


class NeutralAnalyzer:
    def analyze(self, appearance):
        return AppearanceAnalysis(
            appearance_id=appearance.appearance_id,
            start_time=appearance.start_time,
            end_time=appearance.end_time,
            description="The target walks toward the camera.",
            classification="neutral",
            confidence=0.95,
            evidence_timestamps=[appearance.frames[0].timestamp],
        )


def absent_result() -> Phase1Result:
    return Phase1Result(
        reference_image="person.jpg",
        reference_video="video.mp4",
        person_exists=False,
        identity_score=0.2,
        threshold=0.4,
    )


def present_result() -> Phase1Result:
    frame = EvidenceFrame(
        frame_index=8,
        timestamp=2.0,
        bbox=(1, 2, 3, 4),
        detection_confidence=0.9,
        identity_similarity=0.62,
        context_frame_path="context.jpg",
        target_crop_path="target.jpg",
    )
    return Phase1Result(
        reference_image="person.jpg",
        reference_video="video.mp4",
        person_exists=True,
        identity_score=0.62,
        threshold=0.4,
        appearances=[Appearance(1, 2.0, 5.0, [4], [frame])],
    )


class PipelineTests(unittest.TestCase):
    def test_absent_gate_never_invokes_phase2(self):
        final, diagnostics = finalize_from_phase1(
            absent_result(), analyzer=FailingAnalyzer()
        )
        self.assertFalse(final.person_exists)
        self.assertIsNone(final.activity_description)
        self.assertIsNone(final.activity_classification)
        self.assertIsNone(diagnostics["phase2"])

    def test_present_result_is_analyzed(self):
        final, diagnostics = finalize_from_phase1(
            present_result(), analyzer=NeutralAnalyzer()
        )
        self.assertTrue(final.person_exists)
        self.assertEqual(final.activity_classification, "neutral")
        self.assertIsNotNone(diagnostics["phase2"])

    def test_runner_writes_public_and_debug_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "person.jpg"
            video = root / "video.mp4"
            reference.touch()
            video.touch()

            def fake_phase1(reference_image, video_path, run_directory, config, embedder=None):
                result = absent_result()
                result.reference_image = str(reference_image)
                result.reference_video = str(video_path)
                result.run_directory = str(run_directory)
                return result

            output = root / "result.json"
            config = PipelineConfig(runs_dir=root / "runs", keep_artifacts=False)
            run_pipeline(
                reference,
                video,
                output_path=output,
                config=config,
                analyzer=FailingAnalyzer(),
                phase1_runner=fake_phase1,
            )
            public = json.loads(output.read_text())
            debug = json.loads((root / "result.debug.json").read_text())
            self.assertEqual(
                set(public),
                {
                    "reference_image",
                    "reference_video",
                    "person_exists",
                    "activity_description",
                    "activity_classification",
                },
            )
            self.assertIsNone(debug["phase2"])
            self.assertEqual(list((root / "runs").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
