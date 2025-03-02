# import json
# from tqdm import tqdm
# from pprint import pprint

# # File paths
# eval_file_path = '/home/cr8dl-user/sameep/evaluation/vinoground/videollama3_temporal/vinoground/eval_test_official/videollama3_temporal.jsonl'
# inference_file_path = '/home/cr8dl-user/sameep/evaluation/vinoground/videollama3_temporal/vinoground/inference_test_official/videollama3_temporal.jsonl'
# output_file_path = './output.jsonl'

# # Read eval data as a list
# with open(eval_file_path, 'r') as eval_file:
#     eval_data = [json.loads(line) for line in eval_file]

# # Read inference data as a list
# with open(inference_file_path, 'r') as inference_file:
#     inference_data = [json.loads(line) for line in inference_file]

# # Prepare output data
# output_data = []

# # Create a dictionary for quick lookup of inference data by id
# inference_dict = {entry['id']: entry for entry in inference_data}

# # Use tqdm to show progress
# for eval_entry in tqdm(eval_data):
#     id_key = eval_entry['id']

#     if id_key in inference_dict:
#         inference_entry = inference_dict[id_key]
#         response_dict = eval(eval_entry['response'])
#         pred_value = response_dict.get('pred')
#         answer = inference_entry['answer']
        
#         # Determine pred based on the condition
#         pred = answer if pred_value == 'yes' else 'X'  # 'X' is a placeholder for any incorrect answer

#         print("vinoground/frames_concated/" + inference_entry.get('modal_path', '').split('/')[-1])

#         # Construct the output entry
#         output_entry = {
#             "id": id_key,
#             "video_id": str("vinoground/frames_concated/" + inference_entry.get('modal_path', '').split('/')[-1]),
#             "question": inference_entry.get('query', ''),
#             "answer": answer,
#             "pred": pred
#         }
#         output_data.append(output_entry)

# # Write to output jsonl file
# with open(output_file_path, 'w') as output_file:
#     for entry in output_data:
#         output_file.write(json.dumps(entry) + '\n')



import os

# Define the paths
video_dir = '/home/cr8dl-user/sameep/datasets/vinoground/vinoground_videos'
frames_dir = '/home/cr8dl-user/sameep/datasets/vinoground/frames'

# Get list of video files and strip the .mp4 extension
video_files = [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Get list of directories in frames
frame_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]

# Check for each video if there is a corresponding directory
for video in video_files:
    if video not in frame_dirs:
        print(f"No match for: {video}")
