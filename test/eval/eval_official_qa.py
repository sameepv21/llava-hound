import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
import fire
import re
from logzero import logger
from data_processing.utils import format_docstring, load_json_data, save_jsonl, save_json, get_id_from_frame_path

RESULTING_PATH=os.environ.get("RESULTING_PATH_OFFICIAL", "./eval_official_results.jsonl")

def maybe_truncate(text, max_len=256):
    words = text.split()
    if len(words) > max_len:
        return " ".join(words[:max_len])
    return text

def make_data(dp):
    ret = {}
    ret['id'] = dp['id']
    ret['variables'] = {
        "question": dp['question'],
        "answer": dp['answer'],
        "prediction": dp["prediction"],
    }
    return ret

def main(pred_path, output_dir, output_path, num_tasks=1, model_name='chatgpt-3.5-turbo', temperature=0, top_p=1.0, max_new_tokens=256):
    # Display checks
    print("pred_path:", pred_path)
    print("output_dir:", output_dir)
    print("output_path:", output_path)
    
    pred_data = load_json_data(pred_path) # Load the json with responses.
    
    data_to_send_list = [] 
    for sample in pred_data:
        if sample['model_prediction']['status'] != 'success':
            logger.info(f"no valid prediction {sample['id']}")
            continue
        pred = sample['model_prediction']['message']
        pred = maybe_truncate(pred)
        
        question = sample['query']
        answer = sample['answer']
        data_to_send = {
            'id': sample['id'],
            'question': question,
            'answer': answer,
            'prediction': pred,
        }
        data_to_send = make_data(data_to_send)
        data_to_send_list.append(data_to_send)
    print(data_to_send_list[0])

    combined_contents = load_json_data(output_path)

    # Calculate accuracy
    yes_count = 0
    no_count = 0
    for item in combined_contents:
        try:
            response = item['response']
            
            # Computing accuracy
            if "yes" in response.lower():
                yes_count += 1
            elif "no" in response.lower():
                no_count += 1
        except:
            print(f"invalid response: {item}")

    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)

    resulting_dict = {
        'result': f"{accuracy*100:.2f}",
        'name_or_path': output_path, 
    }
    save_jsonl(RESULTING_PATH, resulting_dict, append=True)

if __name__ == "__main__":
    fire.Fire(main)

