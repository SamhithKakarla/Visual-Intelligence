from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM
from qwen_vl_utils import process_vision_info
import torch
import json
from ollama import chat
from tqdm import tqdm


def chunk_segment(segment, chunk_size=8):
    """Splits a trajectory's frame list into consecutive chunks of chunk_size,
    so every frame still gets sent to the model -- just across multiple calls.
    The last chunk may be smaller than chunk_size; nothing is dropped."""
    return [segment[i:i + chunk_size] for i in range(0, len(segment), chunk_size)]


CLASSES = ["Suspicious", "Not suspicious"]


CHUNK_DESCRIPTION_PROMPT = (
    "Describe ONLY what is literally visible about the boxed person in this "
    "clip: their physical actions, posture, position relative to objects or "
    "other people, and appearance (including whether they appear to be "
    "wearing a uniform consistent with security/staff, if visible). "
    "Do NOT speculate about intent, emotional state, or what might happen "
    "next -- report only observable behavior."
)

DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    },
    "required": ["summary"]
}

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {
            "type": "string",
            "enum": CLASSES
        },
        "summary": {
            "type": "string"
        }
    },
    "required": ["class", "summary"]
}


CLASSIFY_PROMPT_TEMPLATE = (
    "Below is a full description of one person's behavior across an entire "
    "video, built from a series of short clips. Read the whole thing and "
    "classify the person as 'Suspicious' or 'Not suspicious'. "
    "Ordinary actions on their own -- walking, standing, sitting, "
    "approaching someone, observing -- are NOT suspicious. Only classify as "
    "'Suspicious' if the description, taken as a whole, reports concrete "
    "behavior such as physical violence or fighting, aggressive/threatening "
    "physical contact, cornering or bullying someone, breaking rules, "
    "getting arrested, or intentionally hurting others. If the description "
    "doesn't clearly support that, classify as 'Not suspicious'.\n\n"
    "Description:\n{text}"
)


def describe_chunk(chunk, model="qwen2.5vl:7b", num_ctx=8096):
    """Sends one chunk of images to the VLM and returns the parsed
    {summary} dict. Raises on failure -- caller decides whether to skip or
    abort."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CHUNK_DESCRIPTION_PROMPT,
                "images": chunk,
            }
        ],
        format=DESCRIPTION_SCHEMA,
        options={"num_ctx": num_ctx}
    )
    return json.loads(response["message"]["content"])


def classify_text(text, model="gemma3:4b"):
    """Classifies a block of text (a track's full combined description, or
    the whole video's combined description) into Suspicious / Not suspicious
    in a single call, based on the entire text at once."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CLASSIFY_PROMPT_TEMPLATE.format(text=text)
            }
        ],
        format=CLASSIFICATION_SCHEMA
    )
    return json.loads(response["message"]["content"])


def run_phase2(matches, mode="train", chunk_size=18, num_ctx=8096):

    output_str = ""
    timeline = []

    # Video Frame processing
    for match in tqdm(matches, total=len(matches)):
        if match['track_id'] == -1:
            continue

        trajectory = match['trajectory']
        segment = []
        for frame in trajectory:
            crop_path = frame['crop_path']
            boxed_path = crop_path.replace("person_crops", "boxed_frames")
            segment.append(boxed_path)

        if not segment:
            continue

        chunks = chunk_segment(segment, chunk_size=chunk_size)

        chunk_summaries = []
        for chunk in chunks:
            try:
                chunk_summaries.append(describe_chunk(chunk, num_ctx=num_ctx)["summary"])
            except Exception as e:
                print(f"  Skipping a chunk for track {match['track_id']}: {e}")
                continue

        if not chunk_summaries:
            print(f"  All chunks failed for track {match['track_id']}, skipping track")
            continue

        combined_summary = " ".join(chunk_summaries)
        
        try:
            classification = classify_text(combined_summary)
            final_class = classification["class"]
        except Exception as e:
            print(f"  Classification failed for track {match['track_id']}: {e}")
            continue

        result = {
            "start_time": match['start_time'],
            "end_time": match['end_time'],
            "class": final_class,
            "activity": combined_summary,
            "track_id": match['track_id'],
            "bbox": [traj['bbox'] for traj in match["trajectory"]]
        }
        output_str += combined_summary
        timeline.append(result)

    result = classify_text(output_str)

    result_phase2 = {
        "overall_summary": result['summary'],
        "class": result['class'],
        "timeline": timeline
    }
    return result_phase2


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python phase2.py <phase1_output_path>")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        phase1_result = json.load(f)
    result_phase2 = run_phase2(phase1_result['matches'])
    with open("phase2_output.json", "w") as f:
        json.dump(result_phase2, f, indent=2)
