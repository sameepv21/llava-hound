import socket
import json
from PIL import Image
import fire
import os
from tqdm import tqdm
import torch 
from logzero import logger
import random
import math
import numpy as np

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria, tokenizer_X_token_llama3
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

from data_processing.utils import load_jsonl, save_jsonl, load_json, save_json

from transformers import CLIPImageProcessor
from PIL import Image

def get_id_from_path(path):
    return path.split('/')[-1].split('.')[0]

MODAL_TOKEN_LIST=["<video>", "<image>"
                  ]
def remove_special_tokens(text):
    for token in MODAL_TOKEN_LIST:
        if token in text:
            text = text.replace(token, "").strip()
    return text

def model_function(model_dict, input_data):
    """
        input_data:
        {
            "modal_type": "VIDEO" or "IMAGE"
            "query": "query",
            "modal_path": image or video path,
            "video_decode_backend": "frames" or "decord" # for video
        }
    """
    # unpack model dict
    model = model_dict["model"]
    processor = model_dict["processor"]
    qs = input_data['query']

    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'video', "video": {"video_path": input_data['modal_path'], "fps": 1, "max_frames": 180}},
                {'type': 'text', "text": qs},
            ]
        }
    ]

    (_, inputs) = processor(conversation = conversation, return_tensors = 'pt')

    inputs['eval'] = True

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    inputs = inputs.to('cuda')
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_ranged_data(data, range_start, range_end):
    start_idx = int(len(data) * range_start)
    end_idx = int(len(data) * range_end)
    return data[start_idx:end_idx]

def inference_data_list(model_dict, data_list, output_path, proc_func, **kwargs):
    if os.path.exists(output_path):
        res = load_jsonl(output_path)
        res_idx = set([x['id'] for x in res])
        ll = len(data_list)
        logger.info(f"load {len(res)}, full chunck length: {ll}, need process length: {ll - len(res)}")
    else:
        res_idx = set()
        json_data = []
        for i, item in tqdm(enumerate(data_list), total=len(data_list)):
            data_to_send = proc_func(item, **kwargs)
            if data_to_send['id'] in res_idx:
                continue
            resulting_output = model_function(model_dict, data_to_send)
            data_to_send['model_prediction'] = {
                'status': 'success',
                'message': resulting_output,
            }
            json_data.append(data_to_send)
        with open(output_path, 'w') as f:
            for item in json_data:
                json.dump(item, f)
                f.write('\n')