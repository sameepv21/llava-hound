import argparse
import os
import json
from pprint import pprint
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='JSON File Reader')
    parser.add_argument('--root_dir', type=str, help='Root directory of the JSON file')
    parser.add_argument('--json_filename', type=str, help='Name of the JSON file')
    parser.add_argument("--save_dir", type=str, help='Path to save the preference data')
    return parser.parse_args()

args = parse_args()

with open(os.path.join(args.root_dir, args.json_filename), 'r') as f:
    data = json.load(f)

# pprint(data[0])
new_dicts = []

for index, entry in enumerate(tqdm(data)):
    new_dicts.append({
        "id": entry['id'].replace(".mp4", "") + "_" + str(index),
        "video": entry['id'].replace(".mp4", ""),
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n" + entry['question'],
            },
            {
                "from": "gpt",
                "value": entry['answer']
            }
        ]
    })

with open(os.path.join(args.save_dir, 'temporal_benchmark.json'), 'w') as f:
    json.dump(new_dicts, f)

