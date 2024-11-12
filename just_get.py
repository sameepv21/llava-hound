import json
import os

data_dict = []
correct = 0
CONSTANT = "perceptiontest/eval_test_official"
BASE_DIR = '/home/cr8dl-user/sameep/evaluation/perceptiontest/'

# Change this
EVAL_DIR = "llava_hound_dpo_temporal_v2_scaled"
JSON = 'llava-hound-dpo-temporal_scaled.jsonl'

path = os.path.join(BASE_DIR, EVAL_DIR, CONSTANT, JSON)

with open(path, "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        data_dict.append(line)
        # print(data['response']['pred'])
        if "yes" in data['response'].lower():
            correct += 1
    print(correct / len(data_dict))