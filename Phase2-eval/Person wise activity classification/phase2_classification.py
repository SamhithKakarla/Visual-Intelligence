from ollama import chat
import json
from tqdm import tqdm

CHUNK_PROMPT = (
    "You are a video surveillance analyst. You are given a sequence of "
    "frames sampled over time from a single CCTV/surveillance video clip, "
    "in chronological order.\n\n"
    "Describe in detail what each visible person is doing throughout the "
    "sequence. Include their appearance, interactions with objects, vehicles, "
    "or buildings, and any changes in their actions over time.\n\n"
    "Identify whether any of the following activities are being performed:\n"
    "1. Person loading an object into a vehicle\n"
    "2. Person unloading an object from a vehicle\n"
    "3. Person opening a vehicle trunk\n"
    "4. Person closing a vehicle trunk\n"
    "5. Person getting into a vehicle\n"
    "6. Person getting out of a vehicle\n"
    "7. Person gesturing\n"
    "8. Person digging\n"
    "9. Person carrying an object\n"
    "10. Person running\n"
    "11. Person entering a facility\n"
    "12. Person exiting a facility\n\n"
    "If multiple activities occur, describe all of them in chronological "
    "order. If none of these activities are visible, describe the observed "
    "behavior without forcing it into one of the categories."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Given the following observations from different portions of the video:\n\n"
    "{combined_results}\n\n"
    "Summarize the events in chronological order and determine which of the "
    "following activities are present in the video:\n"
    "1. Person loading an object into a vehicle\n"
    "2. Person unloading an object from a vehicle\n"
    "3. Person opening a vehicle trunk\n"
    "4. Person closing a vehicle trunk\n"
    "5. Person getting into a vehicle\n"
    "6. Person getting out of a vehicle\n"
    "7. Person gesturing\n"
    "8. Person digging\n"
    "9. Person carrying an object\n"
    "10. Person running\n"
    "11. Person entering a facility\n"
    "12. Person exiting a facility\n\n"
    "Return your answer in the following format:\n\n"
    "Summary: <brief chronological summary>\n"
    "Detected Activities:\n"
    "- <activity 1>\n"
    "- <activity 2>\n\n"
    "If none of the listed activities are present, write:\n"
    "Detected Activities: None."
)

CLASSES = [
    "Person loading an object into a vehicle",
    "Person unloading an object from a vehicle",
    "Person opening a vehicle trunk",
    "Person closing a vehicle trunk",
    "Person getting into a vehicle",
    "Person getting out of a vehicle",
    "Person gesturing",
    "Person digging",
    "Person carrying an object",
    "Person running",
    "Person entering a facility",
    "Person exiting a facility",
]

# NOTE: key is "activity" (singular) to match the field name used below and
# in summarize_description's return value. Keep these in sync.
SUMMARY_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "activity": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": CLASSES
            }
        },
        "summary": {
            "type": "string"
        }
    },
    "required": ["activity", "summary"],
}


def process_chunk(chunk: list[str], model: str = "qwen2.5vl:7b", num_ctx: int = 10000) -> str:
    """Send one chunk of frame image paths to the VLM and return its
    free-text description of the activity in that chunk."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CHUNK_PROMPT,
                "images": chunk,
            }
        ],
        options={"num_ctx": num_ctx},
    )
    return response["message"]["content"]


def summarize_description(combined_results: str) -> dict:
    """Summarize all chunk descriptions for a segment and return a
    {"activity": [...], "summary": str} dict per SUMMARY_FORMAT_SCHEMA."""
    response = chat(
        model="mistral:latest",
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT_TEMPLATE.format(combined_results=combined_results),
            }
        ],
        format=SUMMARY_FORMAT_SCHEMA,
    )
    return json.loads(response["message"]["content"])


def run_phase2(matches, chunk_size: int = 8):
    """Run the two-stage VLM pipeline over each segment independently.

    For each match/segment: gather its frame_paths in chronological order,
    split into chunks, describe each chunk with process_chunk(), then feed
    that segment's chunk descriptions into summarize_description() to get
    a per-segment activity classification.

    Returns a list of per-segment results, each with an "activities" key
    (renamed from the schema's "activity" for a clearer external API).
    """
    segment_results = []

    for match in tqdm(matches,total=len(matches)):
        segment_id = match["segment_id"]
        track_id = match["track_id"]
        start_time = match["start_time"]
        end_time = match["end_time"]

        frame_paths = [step["frame_path"] for step in match["trajectory"]]

        # Split this segment's frames into chunks for the VLM
        chunk_descriptions = []
        for i in range(0, len(frame_paths), chunk_size):
            chunk = frame_paths[i:i + chunk_size]
            description = process_chunk(chunk)
            chunk_descriptions.append(description)

        combined_results = "\n".join(chunk_descriptions)
        summary = summarize_description(combined_results)

        segment_results.append({
            "segment_id": segment_id,
            "track_id": track_id,
            "start_time": start_time,
            "end_time": end_time,
            "descriptions": chunk_descriptions,
            "activities": summary["activity"],
            "summary": summary["summary"],
        })

    return segment_results