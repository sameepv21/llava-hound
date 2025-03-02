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

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from data_processing.utils import load_json_data, save_jsonl, save_json
from inference.inference import model_function, split_list, get_chunk, get_ranged_data, inference_data_list, remove_special_tokens

from transformers import CLIPImageProcessor, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def main(model_path, base_model_path, data_path, output_dir, output_name, chunk_idx, chunks, video_dir=None, range_start=None, range_end=None, **kwargs):    
    temperature = kwargs.get('temperature', 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 1024)
    peft_path = kwargs.get("load_peft")

    data = load_json_data(data_path)
    chunks = get_chunk(data, chunks, chunk_idx)

    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    def make_data_to_send(dp, temperature=0.0, top_p=0.9, max_new_tokens=1024):
        """
            {
            'id': '39137_1',
            'video': '/mnt/bn/liangkeg/data/frames/activitynet/v_-1CEVKeAyA8-Scene-012',
            'caption': 'The video opens with consecutive stills showing a female athlete preparing to jump. She starts with a focused stance, then progresses through her a...',
            'question': 'How does the athlete transition into the takeoff phase?',
            'answer': 'The athlete transitions into the takeoff phase by arching her body and showing powerful leg push-off.'
            }
        """
        # query = remove_special_tokens(dp['conversations'][0]['value'])
        query = dp['conversations'][0]['value'].replace("<video>", "").strip()
        answer = dp['conversations'][1]['value']
        idx = dp['id'] if 'id' in dp else dp['idx']
        modal_path = dp['video'] if video_dir is None else f"{video_dir}/{dp['video']}"
        data_to_send = {
            'id': idx,
            'modal_path': modal_path,
            'query': query,
            'answer': answer,
            'modal_type': 'VIDEO',
            'video_decode_backend': 'frames',
            'temperature': temperature,
            'top_p': top_p,
            'max_new_tokens': max_new_tokens,
        }
        return data_to_send

    print(make_data_to_send(chunks[0], **kwargs))

    model_name = get_model_name_from_path(model_path)
    logger.info(f"model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_token_compression = False
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     cache_dir='/home/cr8dl-user/.cache', # Change based on your location FIXME
    #     use_fast=False,
    # )
    # tokenizer, model, processor, context_len = load_pretrained_model(model_path, base_model_path, model_name, device_map={"":0})
    
    if peft_path:
        model.load_adapter(peft_path)
        print(f"Loaded adapters from {peft_path}")

    model_dict = {
        "model": model,
        "processor": processor,
        # "context_len": context_len,
    }

    output_path = f"{output_dir}/{output_name}"
    inference_data_list(model_dict, chunks, output_path, make_data_to_send, **kwargs)
    # save_jsonl(f"{output_dir}/{output_name}", item, append=True)

if __name__ == "__main__":
    fire.Fire(main)