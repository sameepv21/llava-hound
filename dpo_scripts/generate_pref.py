from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from pathlib import Path 
import torch
import time
import os
from tqdm import tqdm
import json
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=15101)
parser.add_argument('--save_interval', default=100, type=int, help="Interval of iterations to save the output JSON")
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--data_path', type=str, default="/scratch/svani/data/finevideo", help='path to the save directory')
parser.add_argument('--save_path', type=str, default="/scratch/svani/data/finevideo", help='path to the save directory')
args = parser.parse_args()

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

org_path = os.path.join(args.data_path, "finevideo-15k-description.json")
gpt_path = os.path.join(args.data_path, "finevideo-15k-gpt.txt")

org_df = pd.read_json(org_path, header=None, names=["name", "caption"])
data = org_df['caption'].tolist()

start_idx = args.start_idx 
end_idx = args.end_idx

org_df = org_df.iloc[start_idx:end_idx].reset_index(drop=True)
data = data[start_idx:end_idx]

print(len(org_df), len(data))

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
            "name": org_df.loc[ci+i, "name"], 
            "caption": org_df.loc[ci+i, "caption"], 
            "negative_caption": decoded[i][len(prompts[i]):].strip().split("Input")[0].split("Output")[0].split(".")[0].strip() + "."
        })

    if len(new_captions) // args.save_interval != 0:
        update_json_file(os.path.join(save_path, f"finevideo-dpo-15k-descriptions.json"), new_captions)
        new_captions = []
        save_ck += 1


    pbar.update(batch_size)
    ci += batch_size
pbar.close()

if len(new_captions)>0:
    update_json_file(os.path.join(save_path, f"finevideo-dpo-15k-descriptions.json"), new_captions)