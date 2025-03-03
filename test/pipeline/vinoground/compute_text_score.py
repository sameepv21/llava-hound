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
    vid, vid_type = data["video_id"].split("/")[-1].split("_")
    
    if vid not in results:
        results[vid] = {}
        
    answer = data['answer']
    pred = data['pred']
    ans = True if answer[0] == pred[0] else False
    
    results[vid][vid_type] = ans
    
score = 0

for vid, result in results.items():
    pos_result: bool = result['pos']
    neg_result: bool = result['neg']
    result['overall'] = pos_result and neg_result
    score += 1 if result['overall'] else 0

print(f"Text Score: ", score/len(results)*100.0)
    
with open(args.output_file_path, 'w') as f:
    json.dump(results, f)