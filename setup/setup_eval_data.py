import json
import argparse
from tqdm import tqdm
import os
from pprint import pprint
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="Input json file", default=None)
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--dataset_name", type=str, help="Choose one from ['percetiontest', 'cinepile', 'mvbench', 'sharegpt4video']")
    parser.add_argument("--mvbench_dir", type=str, help="Directory containing all the json files of mvbench", default=None)
    return parser.parse_args()

def read_json_data(filepath):
    if "json" in filepath:
        with open(filepath, "r") as f:
            data = json.load(f)
    elif "jsonl" in filepath:
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def read_mvbench(dir):
    combined_data = []
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            file_path = os.path.join(dir, filename)
            data = read_json_data(file_path)
            # print(filename)
            for index, row in enumerate(data):
                if "fine_grained_action" in filename:
                    row['video'] = os.path.join("Moments_in_Time_Raw", row['video']) 
                elif "action_antonym" in filename:
                    row['video'] = os.path.join("ssv2_video", row['video'])
                elif "action_count" in filename:
                    row['video'] = os.path.join("perception", row['video'])
                elif "character_order" in filename:
                    row['video'] = os.path.join("perception", row['video'])



                elif "fine_grained_pose" in filename: # Skip this as dataset is not there
                    del data[index]
                    # row['video'] = os.path.join("fine_grained_pose", row['video'])



                elif "object_interaction" in filename:
                    row['video'] = os.path.join("star", row['video'])
                elif "counterfactual_inference" in filename:
                    row['video'] = os.path.join("clevrer", row['video'])
                elif "moving_attribute" in filename:
                    row['video'] = os.path.join("clevrer", row['video'])
                elif "object_shuffle" in filename:
                    row['video'] = os.path.join("perception", row['video'])
                elif "action_localization" in filename:
                    row['video'] = os.path.join("sta", row['video'])
                elif "egocentric_navigation" in filename:
                    row['video'] = os.path.join("vlnqa", row['video'])
                elif "moving_count" in filename:
                    row['video'] = os.path.join("clevrer", row['video'])



                elif "scene_transition" in filename: # Skip this as dataset is not there
                    del data[index]

                    
                elif "action_prediction" in filename:
                    row['video'] = os.path.join("star", row['video'])
                elif "episodic_reasoning" in filename:
                    row['video'] = os.path.join("tvqa", row['video'])
                elif "moving_direction" in filename:
                    row['video'] = os.path.join("clevrer", row['video'])
                elif "state_change" in filename:
                    row['video'] = os.path.join("perception", row['video'])
                elif "action_sequence" in filename:
                    row['video'] = os.path.join("star", row['video'])
                elif "object_existence" in filename:
                    row['video'] = os.path.join("clevrer", row['video'])
                elif "unexpected_action" in filename:
                    row['video'] = os.path.join("FunQA_test", row['video'])
            combined_data.extend(data)
    return combined_data

def read_parquet(filepath):
    return pd.read_parquet(filepath)

def generate_data(data, dataset_name):
    new_data = []

    if dataset_name == 'perceptiontest':
        idx = 0
        for key, value in tqdm(data.items()):
            id = f"v_{key}_{idx}"
            video = key
            new_data_dict = {
                "id": id,
                "video": video,
            }
            for questions in value['mc_question']:
                conversations = []
                question = "<video>\n" + questions['question'] + "\nOptions:"
                for index, option in enumerate(questions['options']):
                    question += "\n" + str(index) + "." + option
                question += "\nJust provide the option number and the option's text."
                conversations.append({"from": "human", "value": question})
                answer = f"{questions['answer_id']}. {questions['options'][questions['answer_id']]}"
                conversations.append({"from": "gpt", "value": answer})
            new_data_dict['conversations'] = conversations
            new_data.append(new_data_dict)
            idx += 1
        assert len(new_data) == len(data)
        return new_data
    elif dataset_name == "mvbench":
        for idx, row in enumerate(data):
            print(row)
            break
    elif dataset_name == "cinepile":
        for idx, row in tqdm(data.iterrows()):
            video = row['yt_clip_link'].split("=")[-1] + f"_{idx:04d}"
            id = f"v_{video}_{idx}"
            new_data_dict = {
                "id": id,
                "video": video,
            }
            conversations = []
            question = "<video>\n" + row['question'] + "\nOptions:"
            for index, option in enumerate(row['choices'].tolist()):
                question += "\n" + str(index) + "." + option
            question += "\nJust provide the option number and the option's text."
            conversations.append({"from": "human", "value": question})
            answer = f"{int(row['answer_key_position'])}. {row['answer_key']}"
            conversations.append({"from": "gpt", "value": answer})
            new_data_dict['conversations'] = conversations
            new_data.append(new_data_dict)
        assert len(new_data) == data.shape[0]
        return new_data
    
if __name__ == "__main__":
    args = parse_args()
    filename = args.filename
    data_dir = args.data_dir

    if filename:
        if "json" in filename or "jsonl" in filename:
            data = read_json_data(os.path.join(data_dir, filename))
        
        if "parquet" in filename:
            data = read_parquet(os.path.join(data_dir, filename))

    if args.mvbench_dir:
        data = read_mvbench(args.mvbench_dir)
    
    new_data = generate_data(data, args.dataset_name)

    output_filename = f"processed_{args.dataset_name}.json"
    with open(os.path.join(data_dir, output_filename), 'w') as output_file:
        json.dump(new_data, output_file, indent=4)
    print(f"Processed data saved to {output_filename}")