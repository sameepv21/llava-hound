import json
import argparse
from tqdm import tqdm

model = "sys.argv[1]"
TYPE = "group"

parser = argparse.ArgumentParser()
parser.add_argument("--text_eval_results", type=str, required=True, help="Path to input text eval results.json")
parser.add_argument("--video_eval_results", type=str, required=True, help="Path to input video eval results.json")
parser.add_argument("--group_output_path", type=str,required=True, help='Group output path')

args = parser.parse_args()

with open(args.text_eval_results, 'r') as f:
    text_results = json.load(f)

with open(args.video_eval_results, 'r') as f:
    video_results = json.load(f)

results = {}

score = 0

for vid in text_results:
    video_res = video_results[vid]['overall']
    text_res = text_results[vid]['overall']
    results[vid] = {
        'text': text_res,
        'video': video_res,
        'overall': text_res and video_res    
    }
    
    score += 1 if results[vid]['overall'] else 0
    
print(f"Group Score: ", score/len(results)*100.0)

with open(args.group_output_path, 'w') as f:
    json.dump(results, f)