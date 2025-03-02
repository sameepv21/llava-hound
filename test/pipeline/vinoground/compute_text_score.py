import os
import sys
import json

model = "videollama3_base_text"
TYPE = "text"
dataset = f"vinoground-{TYPE}"

f = open(f"/home/cr8dl-user/sameep/Video-LLMs/finetune_all/video-llama3/videollama3_temporal_text.jsonl", 'r')
fres = open(f"{model}_score.txt", 'w')

results = {}

for line in f:
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

print(f"{TYPE} Score: ", score/len(results)*100.0)
# fres.write(f"{TYPE} Score: {score/len(results)*100.0}")
# fres.close()
    
# with open(f"{model}/{TYPE}/eval_results.json", 'w') as f:
#     json.dump(results, f)