import os
import sys
import json

model = "sys.argv[1]"
TYPE = "group"

# os.makedirs(f"{model}/{TYPE}", exist_ok=True)

with open(f"{model}/text/eval_results.json", 'r') as f:
    text_results = json.load(f)

with open(f"{model}/video/eval_results.json", 'r') as f:
    video_results = json.load(f)
    
fres = open(f"{model}/{TYPE}/{TYPE}score.txt", 'w')

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
    
print(f"{TYPE} Score: ", score/len(results)*100.0)
fres.write(f"{TYPE} Score: {score/len(results)*100.0}")
fres.close()

with open(f"{model}/{TYPE}/eval_results.json", 'w') as f:
    json.dump(results, f)