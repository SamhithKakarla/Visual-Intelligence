"""Prompts and policy for arbitrary visible activity classification."""

from __future__ import annotations

from ..schemas import Appearance


def build_appearance_prompt(appearance: Appearance) -> str:
    timestamps = ", ".join(f"{frame.timestamp:.2f}s" for frame in appearance.frames)
    return f"""You are analyzing chronological surveillance frames from one continuous appearance.

The subject to analyze is marked with a green bounding box labeled TARGET. Ignore the behavior of other people except where they visibly interact with the target. Describe only actions that are visibly supported by the frames. Do not guess identity, ownership, intent, motive, or events outside the supplied evidence.

Classify the target's visible behavior using this rubric:
- positive: clearly helpful, cooperative, protective, affectionate, or prosocial behavior
- negative: clearly harmful, dangerous, aggressive, destructive, illegal, or antisocial behavior
- neutral: ordinary behavior without a clear positive or negative effect, or behavior whose intent is ambiguous

The supplied frame timestamps are: {timestamps}.

Return only one JSON object with exactly these fields:
{{
  "description": "concise chronological description of what the target visibly does",
  "classification": "positive | negative | neutral",
  "confidence": 0.0,
  "evidence_timestamps": [0.0]
}}

Confidence must be between 0 and 1. Evidence timestamps must come from the supplied list.
List at most 5 evidence_timestamps — pick only the frames most clearly supporting your
classification, not every frame you were given.
"""
