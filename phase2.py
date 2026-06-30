from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM
from qwen_vl_utils import process_vision_info
import torch
import json
import ollama

def run_phase2(phase1_output_path:str="phase1_output.json"):
    with open(phase1_output_path, "r") as f:
        phase1_result = json.load(f)

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",local_files_only=True)

    output_str=""
    timeline=[]
    # Video Frame processing
    for match in phase1_result['matches']:
        trajectory=match['trajectory']
        segment=[]
        for frame in trajectory:
            frame_path=frame['frame_path']
            boxed_path=frame_path.replace("frames","boxed_frames")
            segment.append(boxed_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video":segment,
                    },
                    {"type": "text", "text": "Describe only what the boxed person is visibly doing."},
                ],
            }
        ]
        text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_kwargs.pop("fps", None)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result={
            "start_time":match['start_time'],
            "end_time":match['end_time'],
            "activity":output_text[0]
        }
        output_str+=output_text[0]
        timeline.append(result)

    #LLM Summarization
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(output_str, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs, max_length=130, min_length=30, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    result_phase2={"overall_summary":summary,"timeline":timeline}
    with open("phase2_output.json","w") as f:
        json.dump(result_phase2, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python phase2.py <phase1_output_path>")
        sys.exit(1)
    run_phase2(sys.argv[1])