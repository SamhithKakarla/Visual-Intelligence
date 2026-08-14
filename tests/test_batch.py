from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from visual_intelligence.batch import load_manifest, run_batch
from visual_intelligence.config import PipelineConfig
from visual_intelligence.schemas import FinalResult


class BatchTests(unittest.TestCase):
    def test_batch_writes_combined_and_per_item_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "person.jpg").touch()
            (root / "present.mp4").touch()
            (root / "absent.mp4").touch()
            manifest_path = root / "dataset.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "two-case-test",
                        "items": [
                            {
                                "case_id": "present",
                                "reference_image": "person.jpg",
                                "reference_video": "present.mp4",
                            },
                            {
                                "case_id": "absent",
                                "reference_image": "person.jpg",
                                "reference_video": "absent.mp4",
                            },
                        ],
                    }
                )
            )

            analyzers = []
            embedders = []

            def fake_pipeline(
                reference_image,
                video_path,
                output_path,
                debug_output_path,
                config,
                analyzer,
                embedder,
            ):
                analyzers.append(analyzer)
                embedders.append(embedder)
                person_exists = Path(video_path).stem == "present"
                result = FinalResult(
                    reference_image=str(reference_image),
                    reference_video=str(video_path),
                    person_exists=person_exists,
                    activity_description=("The target walks." if person_exists else None),
                    activity_classification=("neutral" if person_exists else None),
                )
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text(json.dumps(result.to_dict()))
                Path(debug_output_path).write_text(
                    json.dumps(
                        {
                            "phase1": {"person_exists": person_exists},
                            "phase2": ({"activity_classification": "neutral"} if person_exists else None),
                        }
                    )
                )
                return result

            output = root / "batch_results.json"
            sentinel_analyzer = object()
            payload = run_batch(
                manifest_path,
                output_path=output,
                config=PipelineConfig(runs_dir=root / "runs"),
                analyzer=sentinel_analyzer,
                pipeline_runner=fake_pipeline,
            )

            self.assertEqual(payload["dataset_id"], "two-case-test")
            self.assertEqual(payload["item_count"], 2)
            self.assertTrue(payload["results"][0]["person_exists"])
            self.assertFalse(payload["results"][1]["person_exists"])
            self.assertEqual(analyzers, [sentinel_analyzer, sentinel_analyzer])
            self.assertIs(embedders[0], embedders[1])
            self.assertTrue((root / "batch_results.debug.json").is_file())
            self.assertTrue(
                (root / "batch_results.items" / "present" / "result.json").is_file()
            )
            debug = json.loads((root / "batch_results.debug.json").read_text())
            self.assertIsNone(debug["items"][1]["phase2"])

    def test_manifest_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "person.jpg").touch()
            (root / "video.mp4").touch()
            manifest = {
                "items": [
                    {
                        "case_id": "duplicate",
                        "reference_image": "person.jpg",
                        "reference_video": "video.mp4",
                    },
                    {
                        "case_id": "duplicate",
                        "reference_image": "person.jpg",
                        "reference_video": "video.mp4",
                    },
                ]
            }
            path = root / "dataset.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
