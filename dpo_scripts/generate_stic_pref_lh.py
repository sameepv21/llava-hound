import math
import os
import argparse
import json
import random

import numpy as np
from PIL import Image

import torchvision.transforms as T
import torch
import transformers
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

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

def corrupt_frames(directory, video):
    path = os.path.join(directory, video)
    if random.random() > 0.5:
        for image_file in os.listdir(path):
            image = Image.open(os.path.join(path, image_file))
            image = T.Resize(size=20)(image)
            image.save(os.path.join(path, image_file))
    else:
        for image_file in os.listdir(path):
            jitter = T.ColorJitter(brightness=.5, hue=.3)
            image = Image.open(os.path.join(path, image_file))
            image = jitter(image)
            image.save(os.path.join(path, image_file))

def eval_model(args, model_dict, frame_corruption = False):
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    processor = model_dict['processor']
    video_processor = processor.get('video', None)
    context_len = model_dict['context_len']
    modal_type = args.modal_type 
    query = DEFAULT_X_TOKEN[modal_type] + "\n" + args.query

    conv_mode = 'v1'
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if frame_corruption:
        corrupt_frames(args.video_dir, args.video_file)

    if modal_type=="VIDEO":
        video_decode_backend = "frames" # Change if frames are not send.
        video_path = os.path.join(args.video_dir, args.video_file)
        modal_tensor = video_processor(video_path, return_tensors='pt', video_decode_backend=video_decode_backend)['pixel_values'][0].half().to('cuda')
    else:
        raise ValueError(f"model_type {modal_type} not supported")
    
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[modal_type], return_tensors='pt')
    input_ids = input_ids.reshape(1, -1)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    temperature = args.temperature
    if temperature < 0.01:
        temperature = -1 # Greedy
    
    top_p = args.top_p
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048
    )
    max_new_tokens = args.max_new_tokens
    max_new_tokens = min(max_context_length - input_ids.shape[1], max_new_tokens)

    with torch.inference_mode():
        input_ids = input_ids.to('cuda')
        model = model.to('cuda')
        modal_tensor = modal_tensor.to("cuda")
        model.half()
        output_ids = model.generate(
            input_ids,
            images=[[modal_tensor], [modal_type.lower()]],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = 'ShareGPTVideo/LLaVA-Hound-SFT')
    parser.add_argument("--model_base", type=str, default = None)
    parser.add_argument("--video_dir", type=str, default="/home/cr8dl-user/sameep/datasets/timewarp/stic_lh_frames")
    parser.add_argument("--save_dir", type=str, default="/home/cr8dl-user/sameep/datasets/timewarp/stic_lh_pref.json")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--query", type=str, default="Describe the Video.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument('--modal_type', type=str, default="VIDEO")
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path = args.model_path,
        model_base = args.model_base,
        model_name = model_name,
    )

    if args.peft_path:
        model.load_adapter(args.peft_path)
        print(f"Loaded adapters from {args.peft_path}")
    
    model_dict = {
        "tokenizer": tokenizer,
        "model": model,
        "processor": processor,
        "context_len": context_len
    }

    directory = args.video_dir

    response_dicts = []

    with open('/home/cr8dl-user/sameep/datasets/llava-hound/sft_dpo_17k.jsonl', 'r') as f:
        video_data = [json.loads(line) for line in f]

    for idx, video_info in tqdm(enumerate(video_data)):
        video_id = video_info['id'].rsplit("_", 1)[0]
        args.query = full_prompt
        args.video_file = video_id

        preferred_output = eval_model(args, model_dict)
        
        hallu_prompt = ""
        prompt = random.choice(prompt_list)
        

        if random.random() > 0.5:
            hallu_prompt = random.choice(hallu_prompt_list)
            args.query = hallu_prompt
            corrupted_output = eval_model(args, model_dict)
        else:
            args.query = prompt
            corrupted_output = eval_model(args, model_dict, frame_corruption=True)
            frame_corruption = True
        
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