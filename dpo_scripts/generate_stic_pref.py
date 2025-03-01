import math
import os
import argparse
import json
import random

import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

prompt_list = ["Illustrate the sequence of events unfolding in the video.",
                "Summarize the dynamic visual content presented over time.",
                "Explain the progression of actions depicted in the video.",
                "Outline the key events and their interactions captured in the footage.",
                "Detail the evolving composition and subjects throughout the video.",
                "Convey the atmosphere and mood as it changes, if it changes, during the video.",
                "Interpret the sequence of scenes shown in the video.",
                "Identify and describe the main focal points and their development in the video."]

full_prompt = """Please provide a detailed description of the video, focusing on the following aspects. 
Identify the main subjects (people, animals, objects) in the video and describe what they do over time. 
Note how their actions evolve or interact as the video progresses. Describe the setting of the video—is 
it indoors or outdoors, and what kind of environment or location is depicted? Discuss the mood conveyed 
by the video, noting any temporal changes in lighting, weather, or expressions that contribute to this 
atmosphere. Describe the dominant colors and overall composition as they shift throughout the video, and 
explain how these elements interact over time to affect its impact. Point out any details or symbols that 
might be relevant to understanding the sequence of events or the video's underlying meaning, and if 
applicable, provide interpretations of what the video might represent or communicate through its unfolding narrative."""

hallu_prompt_list = [
    "Describe the video with imaginative sequences of events that may unfold over time.",
    "Enrich the video narrative by adding hypothetical events or interactions that could occur between characters or objects.",
    "Suggest and detail practical sequences or interactions that could logically happen within the video's timeline.",
    "Incorporate elements that, though absent, would seamlessly fit into the temporal flow of the video.",
    "Imagine and describe additional everyday activities or interactions taking place just out of frame in the video.",
    "Augment the video with details of potential events or interactions that are plausible over time.",
    "Conceive of and detail natural phenomena, such as weather changes or animal movements, that could realistically occur during the video's duration. Make the description affirmative.",
    "Invent and incorporate details of practical tools, vehicles, or gadgets that could be expected to appear or be used in a similar video scenario."
]

def eval_model(args, model_dict):
    processor = model_dict['processor']
    model = model_dict['model'].to(torch.bfloat16)
    modal_type = args.modal_type 
    query = args.query

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": os.path.join(args.video_dir, args.video_file), "fps": 1, "max_frames": 180}},
                {"type": "text", "text": query},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        return_tensors='pt',
    )[1].to("cuda")

    inputs["eval"] = True

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    # output_ids = model(
    #     input_ids=inputs.
    # )
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='DAMO-NLP-SG/VideoLLaMA3-7B')
    parser.add_argument("--video_dir", type=str, default="/home/cr8dl-user/sameep/datasets/timewarp/llava_hound_frames")
    parser.add_argument("--save_dir", type=str, default="/home/cr8dl-user/sameep/datasets/timewarp/stic_vl3_pref.json")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--query", type=str, default="Describe the Video.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument('--modal_type', type=str, default="VIDEO")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model_dict = {
        "processor": processor,
        "model": model
    }

    response_dicts = []

    with open('/home/cr8dl-user/sameep/datasets/llava-hound/sft_dpo_17k.jsonl', 'r') as f:
        video_data = [json.loads(line) for line in f]
    
    with open('/home/cr8dl-user/sameep/datasets/timewarp/stic_lh_pref.json', 'r') as f:
        frame_corruption_data = json.load(f)

    for idx, video_info in tqdm(enumerate(video_data)):
        video_id = video_info['id'].rsplit("_", 1)[0]
        args.query = full_prompt
        args.video_file = video_id

        # For preferred output, use the original frames
        args.video_dir = "/home/cr8dl-user/sameep/datasets/timewarp/llava_hound_frames"

        # Compute preferred response
        preferred_output = eval_model(args, model_dict)

        # For dispreferred output, use the corrupted frames
        args.video_dir = "/home/cr8dl-user/sameep/datasets/timewarp/stic_lh_frames"
        
        hallu_prompt = ""
        prompt = random.choice(prompt_list)

        frame_corruption = frame_corruption_data[idx].get(video_id, False)

        if frame_corruption:
            args.query = prompt
            corrupted_output = eval_model(args, model_dict)
        else:
            hallu_prompt = random.choice(hallu_prompt_list)
            args.query = hallu_prompt
            corrupted_output = eval_model(args, model_dict)
        
        d = {
                "id": video_id + "_" + str(idx),
                "video": video_id,
                "prompt": prompt,
                "frame_corruption": frame_corruption,
                "answer": preferred_output,
                "chosen": preferred_output,
                "chosen_score": random.randint(3, 5),
                "rejected": corrupted_output,
                "rejected_score": random.randint(0, 3),
            }
        
        response_dicts.append(d)
        
    output_file_path = os.path.join(args.save_dir)
    with open(output_file_path, 'w') as outfile:
        json.dump(response_dicts, outfile, indent=4)
    print(f"Saved response data to {output_file_path}")