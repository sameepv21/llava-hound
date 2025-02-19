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

"""
CHANGE THE CODE ACCORDING TO EVERY ANNOTATION THAT YOU WANT TO EVAL
"""

counter = 0

for index, entry in enumerate(tqdm(data)):


    # THIS IS FOR TEMPCOMPASS BENCHMARK
    for sub_cat in data[entry]:
        sub_data = data[entry][sub_cat]
        for qa in sub_data:
            new_dicts.append({
                "id": entry + "_" + str(counter),
                "video": "frames/" + entry,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\n" + qa['question'],
                    },
                    {
                        "from": "gpt",
                        "value": qa['answer']
                    }
                ]
            })
            counter += 1
    # new_dicts.append({
    #     "id": entry['id'],
    #     "video": "videos/frames/" + entry['video_id'].split('/')[-1], # Take the last entry
    #     "conversations": [
    #         {
    #             "from": "human",
    #             "value": "<video>\n" + entry['question'],
    #         },
    #         {
    #             "from": "gpt",
    #             "value": entry['answer']
    #         }
    #     ]
    # })

with open(os.path.join(args.save_dir, 'mcq_tempcompass_lh.json'), 'w') as f:
    json.dump(new_dicts, f)