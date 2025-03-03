import json
import argparse
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input JSONL file')
parser.add_argument('--output_file_path', type=str, required=True, help='Path to the output score file')

# Parse the arguments
args = parser.parse_args()

# Open files using the provided command line arguments
f = open(args.input_file_path, 'r')

results = {}

for line in tqdm(f):
    data = json.loads(line)
    vid = data['video_id'].split("/")[-1].split('_')[0] if '_' in data['video_id'].split("/")[-1] else data['video_id'].split("/")[-1]
    
    if vid not in results:
        results[vid] = {}
        
    answer = data['answer']
    pred = data['pred']
    ans = True if answer[0] == pred[0] else False
    vid_type = "first" if len(results[vid]) == 0 else "second"
    
    results[vid][vid_type] = ans
    
score = 0

for vid, result in results.items():
    first_result: bool = result['first']
    second_result: bool = result['second']
    result['overall'] = first_result and second_result
    score += 1 if result['overall'] else 0

print(f"Video Score: ", score/len(results)*100.0)

with open(args.output_file_path, 'w') as f:
    json.dump(results, f)