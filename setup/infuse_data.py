import json
import os
import argparse

parser = argparse.ArgumentParser(description='Infuse data')
parser.add_argument('--llava_hound_dpo_data', type=str, required=True, help='Path to llava-hound-dpo-data')
parser.add_argument('--temporal_data', type=str, required=True, help='Path to temporal data')
parser.add_argument('--temporal_frame_dir_name', type=str, required=True, help='Name of the temporal frame directory')
parser.add_argument("--save_dir", type=str, required=True, help="Path to save the infused data")

args = parser.parse_args()

llava_hound_dpo_data = args.llava_hound_dpo_data
temporal_data = args.temporal_data
temporal_frame_dir_name = args.temporal_frame_dir_name

with open(temporal_data, "r") as f:
    data_temporal = json.load(f)

data_lh = []
with open(llava_hound_dpo_data, "r") as f:
    for line in f:
        data_lh.append(json.loads(line))

assert set(data_temporal[0].keys()) == set(data_lh[0].keys()), "Keys do not match for the first input"

for item in data_temporal:
    item["video"] = os.path.join(temporal_frame_dir_name, item["video"].removesuffix(".mp4"))

for lh_item in data_lh:
    lh_item['video'] = os.path.join("train_300k/train_zip", lh_item['video'].removesuffix('.mp4'))

combined_data = data_temporal + data_lh

# Assert statement to check the length of the combined data
assert len(combined_data) == len(data_temporal) + len(data_lh), "Length of combined data does not match the sum of individual lengths"

with open(os.path.join(args.save_dir, "temporal_infused_good.json"), "w") as f:
    json.dump(combined_data, f, indent=4)