import math
import os
import argparse
import json
import random

import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

FPV=10
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    tokenizer = model_dict['tokenizer']
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

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Custom Function
def load_video(video_path, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    frame_count = 0

    for frame in os.listdir(video_path):
        if frame_count == FPV:
            break
        frame_count += 1
        frame_path = os.path.join(video_path, frame)
        img = Image.open(frame_path).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# eval setting
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='OpenGVLab/InternVideo2_5_Chat_8B')
    parser.add_argument("--video_dir", type=str, default="/scratch/svani/timewarp/sharegpt4frames")
    parser.add_argument("--save_dir", type=str, default="/scratch/svani/timewarp/stic_ivl_pref.json")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--query", type=str, default="Describe the Video.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument('--modal_type', type=str, default="VIDEO")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model_dict = {
        "tokenizer": tokenizer,
        "model": model
    }

    response_dicts = []

    with open('/scratch/svani/timewarp/pref_sharegpt4video.jsonl', 'r') as f:
        video_data = [json.loads(line) for line in f]
    
    with open('/scratch/svani/timewarp_sameep_backup/json_files/stic_lh_pref.json', 'r') as f:
        frame_corruption_data = json.load(f)

    for idx, video_info in tqdm(enumerate(video_data)):
        video_id = video_info['id'].rsplit("_", 1)[0]
        args.query = full_prompt
        args.video_file = video_id

        # For preferred output, use the original frames
        args.video_dir = "/scratch/svani/timewarp/sharegpt4frames"

        # Compute preferred response
        preferred_output = eval_model(args, model_dict)

        # For dispreferred output, use the corrupted frames
        args.video_dir = "/scratch/svani/timewarp/stic_lh_frames/"
        
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