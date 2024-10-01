from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

template_ = """
                You are given the description for a video. You should provide a mostly similar description, 
                changing the original one slightly, but introducing enough significant differences such that the two descriptions could not possibly be for the same video. 
                Keep the description length the same. Only modify a small number of things (such as counting, objects, attributes, and relationships) that significantly 
                changes the video structure.\nProvide just the updated description. Overall, you want to perturb the spatio-temporal aspect of the video.
                \n\nExamples:\nInput: A dog to the left of the cat.\nOutput: A dog to the right of the cat. \n\nInput: A person wearing a red helmet is driving a 
                motorbike on a dirt road.\nOutput:  A person in a blue helmet is riding a motorbike on a gravel path.\n\nNow, do the same for the following captions:
                \n\nInput: {}\nOutput: 
            """


import numpy as np
import pandas as pd
from pathlib import Path 
import torch
import time
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--start_idx', required=True, type=int)
# parser.add_argument('--end_idx', required=True, type=int)
# parser.add_argument('--batch_idx', required=True, type=int)
# parser.add_argument('--total_batch', default=10, type=int)
# parser.add_argument('--save_interval', default=100, type=int, help="Interval of iterations to save the output JSON")
parser.add_argument('--batch_size', default=33, type=int)
parser.add_argument('--data_path', type=str, default="/scratch/svani/data/finevideo", help='path to the save directory')
parser.add_argument('--save_path', type=str, default="/scratch/svani/data/finevideo", help='path to the save directory')
args = parser.parse_args()

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

org_path = os.path.join(args.data_path, "finevideo-15k-descriptions.json")
gpt_path = os.path.join(args.data_path, "finevideo-15k-dpo-descriptions.json")

org_df = pd.read_csv(org_path, header=None, names=["image_id", "caption", "url"])
data = []
with open(gpt_path) as h:
    tmp_ = h.readlines()
    for t in tmp_:
        data.append(t.strip())

assert len(org_df["caption"].tolist()) == len(data)

# each_batch_length = len(data) // args.total_batch
# start_idx = each_batch_length * args.batch_idx
# end_idx = start_idx + each_batch_length
start_idx = args.start_idx 
end_idx = args.end_idx

org_df = org_df.iloc[start_idx:end_idx].reset_index(drop=True)
data = data[start_idx:end_idx]

print(len(org_df), len(data))

import json
def update_json_file(filepath, new_data):
    try:
        with open(filepath, 'r+') as file:
            existing_data = json.load(file)
            existing_data.extend(new_data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    except FileNotFoundError:
        with open(filepath, 'w') as file:
            json.dump(new_data, file, indent=4)


new_captions = []
batch_size = args.batch_size
tokenizer.pad_token = tokenizer.eos_token

ci = 0
save_ck = 0

pbar = tqdm(total=len(data))
while ci <= len(data):
    batch = data[ci:min(ci+batch_size, args.end_idx+1)]
    prompts = [template_.format(d) for d in batch]

    encodeds = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=64, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for i in range(len(batch)):
        new_captions.append({
            "image_id": org_df.loc[ci+i, "image_id"], 
            "url": org_df.loc[ci+i, "url"], 
            "caption": org_df.loc[ci+i, "caption"], 
            "language-rewrite": batch[i], 
            "Mixtral_instruct_v02": decoded[i][len(prompts[i]):].strip(), 
            "negative_caption": decoded[i][len(prompts[i]):].strip().split("Input")[0].split("Output")[0].split(".")[0].strip() + "."
        })

    if len(new_captions) // args.save_interval != 0:
        update_json_file(os.path.join(save_path, f"cc12m_batch-start_idx-{args.start_idx}-end_idx-{args.end_idx}.json"), new_captions)
        new_captions = []
        save_ck += 1


    pbar.update(batch_size)
    ci += batch_size
pbar.close()

if len(new_captions)>0:
    update_json_file(os.path.join(save_path, f"cc12m_batch-start_idx-{args.start_idx}-end_idx-{args.end_idx}.json"), new_captions)