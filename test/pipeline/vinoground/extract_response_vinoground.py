import json
from tqdm import tqdm
from pprint import pprint

# File paths
eval_file_path = '/home/cr8dl-user/sameep/evaluation/vinoground/videollama3_base_video/vinoground/eval_test_official/videollama3_base_video.jsonl'
inference_file_path = '/home/cr8dl-user/sameep/evaluation/vinoground/videollama3_base_video/vinoground/inference_test_official/videollama3_base_video.jsonl'
output_file_path = './videollama3_base_video.jsonl'

# Read eval data as a list
with open(eval_file_path, 'r') as eval_file:
    eval_data = [json.loads(line) for line in eval_file]

# Read inference data as a list
with open(inference_file_path, 'r') as inference_file:
    inference_data = [json.loads(line) for line in inference_file]

# Prepare output data
output_data = []

# Create a dictionary for quick lookup of inference data by id
inference_dict = {entry['id']: entry for entry in inference_data}

# Use tqdm to show progress
for eval_entry in tqdm(eval_data):
    id_key = eval_entry['id']

    if id_key in inference_dict:
        inference_entry = inference_dict[id_key]
        response_dict = eval(eval_entry['response'])
        pred_value = response_dict.get('pred')
        answer = inference_entry['answer']
        
        # Determine pred based on the condition
        pred = answer if pred_value == 'yes' else 'X'  # 'X' is a placeholder for any incorrect answer

        # Construct the output entry
        output_entry = {
            "id": id_key,
            "video_id": str("vinoground/frames_concated/" + inference_entry.get('modal_path', '').split('/')[-1]),
            "question": inference_entry.get('query', ''),
            "answer": answer,
            "pred": pred
        }
        output_data.append(output_entry)

# Write to output jsonl file
with open(output_file_path, 'w') as output_file:
    for entry in output_data:
        output_file.write(json.dumps(entry) + '\n')