import sys
import functools
import itertools
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np

import torch
import torchvision

from decord import VideoReader, cpu
import transformers
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


# from tasks.eval.eval_utils import conv_templates
from eval_utils import (
    MVBenchDataset,
    check_ans,
    save_results,
    load_results,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672 # 
NUM_FRAMES = 16
dtype = torch.float16

dataset_name = "mvbench"
model_name = "llava-onevision"


# debug
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default='"./test_results/test_llava_mvbench"'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conv_mode", 
        type=str,
        required=False,
        default='eval_mvbench',
    )
    parser.add_argument(
        "--pooling_shape", 
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False, transform_frames=False, num_frames=4, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    images_group = list()
    for frame_index in frame_indices:
        if transform_frames:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        else:
            img = vr[frame_index].asnumpy()
            images_group.append(img)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return images_group, msg
    elif not transform_frames:
        return np.stack(images_group)
    else:    
        return images_group
        

def load_model_and_dataset(rank, world_size):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                                                                torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    logger.info('done loading model')

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = MVBenchDataset(num_segments=NUM_FRAMES)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset

def infer_mvbench(
        model,
        processor,
        data_sample
    ):
    video_path = data_sample["video_path"]
    clip = load_video(video_path)
    
    prompt = data_sample["question"]
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
                ],
        },
    ]
    
    processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device, dtype)
    output = model.generate(**inputs_video, max_new_tokens=768, do_sample=False)
    output_conv = processor.decode(output[0], skip_special_tokens=True)
    llm_message = output_conv.split("assistant")[1].strip()

    # conv = conv_templates[conv_mode].copy()
    # conv.user_query(data_sample['question'], pre_query_prompt, post_query_prompt, is_mm=True)
    # if answer_prompt is not None:
    #     conv.assistant_response(answer_prompt)
        
    # llm_message, conv = pllava_answer(
    #     conv=conv,
    #     model=model,
    #     processor=processor,
    #     img_list=video_list,
    #     max_new_tokens=100,
    #     do_sample=False,
    #     print_res=print_res
    # )
    
    # if answer_prompt is not None:
    #     llm_message =  ''.join(llm_message.split(answer_prompt)[1:])

    # if return_prompt is not None:
    #     llm_message = return_prompt + llm_message

    return llm_message
    
# def single_test(model, processor, vid_path, num_frames=4, conv_mode="plain"):

#     if num_frames != 0:
#         vid, msg = load_video(vid_path, num_segments=num_frames, return_msg=True, resolution=RESOLUTION)
#     else:
#         vid, msg = None, 'num_frames is 0, not inputing image'
#     img_list = vid
#     conv = conv_templates[conv_mode].copy()
#     conv.user_query("Describe the video in details.", is_mm=True)
#     llm_response, conv = pllava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=img_list, max_new_tokens=256, print_res=True)

def run(rank, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = False
    # conv_mode= args.conv_mode
    pre_query_prompt = None
    post_query_prompt = "\nOnly give the best option."
    # if args.pooling_shape is not None:
    #     pooling_shape=tuple([int(x) for x in args.pooling_shape.split("-")])

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size)
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')
    # logger.info('single test...')

    # vid_path = "./example/yoga.mp4"
    # # vid_path = "./example/jesse_dance.mp4"
    # if rank == 0:
    #     single_test(model,
    #                 processor,
    #                 vid_path,
    #                 num_frames=args.num_frames,
    #                 conv_mode=args.conv_mode)
    #     logger.info('single test done...')
    
    output_dir = f"/home/cr8dl-user/sameep/evaluation/{dataset_name}/{model_name}/results"
    os.makedirs(output_dir, exist_ok=True)
    
    tbar = tqdm(total=len(dataset))

    correct = 0
    total = 0
    total_score = 0.0
    result_list = []
    acc_dict = {}
    done_count = 0

    for example in dataset:
        # try:
            video_name = example["video_path"].split("/")[-1].split('.')[0]
            task_type = example['task_type']
            key = f"{task_type}___{video_name}"
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0] # correct, total
            acc_dict[task_type][1] += 1
            total += 1
            
            # print(example)
            
            pred = infer_mvbench(
                model,
                processor,
                example
            )

            # print(pred)
            
            question = example['question']
            gt = example['answer']
            
            # evaluate the response using GPT
            response_dict = check_ans(key=key, question=question, pred=pred, gt=gt, output_dir=output_dir)
            check = response_dict['pred']
            score = response_dict['score']

            result_list.append({
                'pred': pred,
                'gt': gt,
                'task_type': task_type,
                'video_path': example['video_path'],
                'question': question,
                'check': check,
                'score': score
            })
            
            if check:
                acc_dict[task_type][0] += 1
                correct += 1
            if rank == 0:
                tbar.update(len(result_list) - done_count, )
                tbar.set_description_str(
                    f"One Chunk--Task Type: {task_type}, Chunk Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%;" 
                    f" Chunk Total Acc: {correct / total * 100 :.2f}%"
                )
                done_count = len(result_list)
        # except Exception as e:
        #     print(f"Could not process {example['video_path']} due to {e}")
    return result_list

def main():
    multiprocess=True
    mp.set_start_method('spawn')
    # args = parse_args()
    save_path = f"/home/cr8dl-user/sameep/evaluation/{dataset_name}/{model_name}"
    json_data = load_results(save_path)
    if json_data is None:
        if multiprocess:
            logger.info(f'started benchmarking, saving to: {save_path}')
            n_gpus = torch.cuda.device_count()
            # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, world_size=world_size)
                result_lists = pool.map(func, range(world_size))
            
            logger.info('finished running')
            result_list = [ res for res in itertools.chain(*result_lists)]
        else:
            result_list = run(0, world_size=1) # debug

    else:
        logger.info(f'loaded results from {save_path}')
        result_list = json_data
    save_results(result_list, save_path)
    
    
if __name__ == "__main__":
    main()