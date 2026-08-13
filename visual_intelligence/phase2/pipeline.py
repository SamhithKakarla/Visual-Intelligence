"""Video-language inference over target-highlighted appearance frames."""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

from ..config import Phase2Config
from ..schemas import Appearance, AppearanceAnalysis, Phase1Result, Phase2Result
from .prompts import build_appearance_prompt


VALID_VALENCES = {"positive", "negative", "neutral"}


class AppearanceAnalyzer(Protocol):
    def analyze(self, appearance: Appearance) -> AppearanceAnalysis: ...


def split_video_metadata(video_inputs):
    """Separate Qwen3-VL's ``(video, metadata)`` input pairs."""
    if video_inputs is None:
        return None, None
    videos, metadata = zip(*video_inputs)
    return list(videos), list(metadata)


def parse_json_object(text: str) -> dict:
    """Parse a model response, tolerating a single fenced JSON object."""
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("VLM response did not contain a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ValueError("VLM returned malformed JSON") from error
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


def validate_analysis_payload(payload: dict, appearance: Appearance) -> AppearanceAnalysis:
    description = str(payload.get("description", "")).strip()
    if not description:
        raise ValueError("VLM response is missing a description")
    classification = str(payload.get("classification", "")).strip().lower()
    if classification not in VALID_VALENCES:
        raise ValueError(f"Invalid activity classification: {classification!r}")
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError) as error:
        raise ValueError("VLM confidence must be numeric") from error

    allowed = {round(frame.timestamp, 4) for frame in appearance.frames}
    evidence = []
    for value in payload.get("evidence_timestamps", []):
        try:
            timestamp = round(float(value), 4)
        except (TypeError, ValueError):
            continue
        if timestamp in allowed:
            evidence.append(timestamp)

    return AppearanceAnalysis(
        appearance_id=appearance.appearance_id,
        start_time=appearance.start_time,
        end_time=appearance.end_time,
        description=description,
        classification=classification,  # type: ignore[arg-type]
        confidence=round(confidence, 4),
        evidence_timestamps=sorted(set(evidence)),
    )


class QwenVideoAnalyzer:
    """Lazy Qwen3-VL analyzer; model weights load only after a Phase 1 match."""

    def __init__(self, config: Phase2Config | None = None) -> None:
        self.config = config or Phase2Config()
        self._model = None
        self._processor = None
        self.device: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL requires a current transformers release. "
                "Install the phase2 dependencies from requirements.txt."
            ) from error

        kwargs = {
            "torch_dtype": "auto",
            "device_map": self.config.device_map,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.device = str(next(self._model.parameters()).device)
        return self._model, self._processor

    def analyze(self, appearance: Appearance) -> AppearanceAnalysis:
        if not appearance.frames:
            raise ValueError(
                f"Appearance {appearance.appearance_id} has no visual evidence frames"
            )
        from qwen_vl_utils import process_vision_info

        model, processor = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [frame.context_frame_path for frame in appearance.frames],
                        "min_pixels": 12544,
                        "max_pixels": 200704,
                    },
                    {"type": "text", "text": build_appearance_prompt(appearance)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_inputs, video_metadata = split_video_metadata(video_inputs)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)
        generated = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        response = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return validate_analysis_payload(parse_json_object(response), appearance)


def aggregate_analyses(analyses: list[AppearanceAnalysis]) -> Phase2Result:
    """Create a chronological description and safety-oriented video valence."""
    if not analyses:
        raise ValueError("Cannot aggregate an empty Phase 2 result")
    ordered = sorted(analyses, key=lambda item: item.start_time)
    if any(item.classification == "negative" for item in ordered):
        overall = "negative"
    elif any(item.classification == "positive" for item in ordered):
        overall = "positive"
    else:
        overall = "neutral"

    if len(ordered) == 1:
        description = ordered[0].description
    else:
        descriptions = []
        for index, item in enumerate(ordered):
            prefix = "Initially" if index == 0 else "Later"
            descriptions.append(
                f"{prefix}, from {item.start_time:.2f}s to {item.end_time:.2f}s, "
                f"{item.description.rstrip('.')}"
            )
        description = ". ".join(descriptions) + "."

    return Phase2Result(
        activity_description=description,
        activity_classification=overall,  # type: ignore[arg-type]
        appearances=ordered,
    )


def run_phase2(
    phase1_result: Phase1Result,
    analyzer: AppearanceAnalyzer | None = None,
    config: Phase2Config | None = None,
) -> Phase2Result:
    if not phase1_result.person_exists:
        raise ValueError("Phase 2 must not run when Phase 1 reports no match")
    analyzer = analyzer or QwenVideoAnalyzer(config)
    analyses = []
    for item in phase1_result.appearances:
        start = time.perf_counter()
        analysis = analyzer.analyze(item)
        analysis.elapsed_seconds = round(time.perf_counter() - start, 4)
        analyses.append(analysis)
    result = aggregate_analyses(analyses)
    result.device = getattr(analyzer, "device", None)
    return result
