import os
import sys
import json

model = "sys.argv[1]"
TYPE = "video"
dataset = f"vinoground-{TYPE}"

f = open(f"/home/cr8dl-user/sameep/Video-LLMs/finetune_all/video-llama3/videollama3_base_video.jsonl", 'r')
# fres = open(f"{model}/{TYPE}/{TYPE}score.txt", 'w')

results = {}

for line in f:
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

print(f"{TYPE} Score: ", score/len(results)*100.0)
# fres.write(f"{TYPE} Score: {score/len(results)*100.0}")
# fres.close()
    
# with open(f"{model}/{TYPE}/eval_results.json", 'w') as f:
#     json.dump(results, f)