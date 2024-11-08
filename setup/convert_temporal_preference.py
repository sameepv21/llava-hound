import json
import os
import argparse
from pprint import pprint
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser(description='JSON File Reader')
    parser.add_argument('--root_dir', type=str, help='Root directory of the JSON file')
    parser.add_argument('--json_filename', type=str, help='Name of the JSON file')
    parser.add_argument("--save_dir", type=str, help='Path to save the preference data')
    return parser.parse_args()

def main():
    args = parse_args()
    json_path = os.path.join(args.root_dir, args.json_filename)
    with open(json_path, 'r') as f:
        preference_data = json.load(f)

    all_new_dict = []

    for index, record in tqdm(enumerate(preference_data)):
        new_dict = {
            "id": record['id'].replace(".mp4", "") + "_" + str(index),
            "video": record['id'].replace(".mp4", ""),
            "prompt": record['question'],
            "answer": record['answer'],
            "chosen": record['answer'],
            "chosen_score": float(random.randint(3, 5)),
            "rejected": record['rejected'],
            "rejected_score": float(random.randint(1, 3))
        }

        all_new_dict.append(new_dict)

    save_path = os.path.join(args.save_dir, "temporal_dpo_15.json")
    with open(save_path, 'w') as f:
        json.dump(all_new_dict, f)
        
if __name__ == '__main__':
    # Set seed value
    random.seed(42)
    main()