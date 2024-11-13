import copy
import itertools
import re
from enum import auto, Enum
import dataclasses
from typing import Any, List

from PIL import Image
import cv2
import imageio
import os
import json
import ast
import torch
import numpy as np
from torch.utils.data import Dataset

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from openai import OpenAI

from decord import VideoReader, cpu


gpt_model_name = os.environ.get('GPT_MODEL_NAME', 'gpt-3.5-turbo-0125')
api_key = os.environ['OPENAI_API_KEY']


client = OpenAI(api_key=api_key)

# Old version of check_ans -> Author provided the newer version below for more accurate evaluation

# def check_ans(pred, gt):
#     flag = False
    
#     pred_list = pred.lower().split(' ')
#     pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
#     gt_list = gt.lower().split(' ')
#     gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
#     if gt_content[-1] == '.':
#         gt_content = gt_content[:-1]
    
#     if not any([c in pred_option for c in 'abcdefgABCDEFG']):
#         print(f"model doesn't follow instructions: {pred}")
#     elif pred_option.replace('.', '') in gt_option:
#         flag = True
#     elif gt_option in pred_option:
#         flag = True
        
#     return flag

def dump_json(obj_serializable ,save_dir_path, json_file_name):
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, json_file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj_serializable, f, indent=4, ensure_ascii=False, )

def load_json(load_dir_path, json_file_name):
    
    load_path = os.path.join(load_dir_path, json_file_name)
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        obj_serializable = json.load(f)
    return obj_serializable

def check_ans_hardcoded(pred, gt):
    flag = False
    
    # Split predictions and ground truth into options and content
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    
    # Remove trailing period from ground truth content if present
    if gt_content.endswith('.'):
        gt_content = gt_content[:-1]

    # Clean options by removing certain characters
    pred_option = pred_option.replace('.', '').replace('(', '').replace(')', '')
    gt_option = gt_option.replace('.', '').replace('(', '').replace(')', '')
    
    # Additional check: if pred_option does not contain any answer a-e, return False
    if not any(char in pred_option for char in 'abcde'):
        return False
    # Check for equality or inclusion
    if pred_option == gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def check_ans(key, question, pred, gt, output_dir):
    try:
        # Compute the correctness score
        completion = client.chat.completions.create(
            model=gpt_model_name,
            messages=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {gt}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content.strip()
        response_dict = ast.literal_eval(response_message)
        result_qa_pair = [response_dict]

        # Save the question-answer pairs to a json file.
        with open(f"{output_dir}/{key}.json", "w") as f:
            json.dump(result_qa_pair, f)
        
        return response_dict

    except Exception as e:
        print(f"Error processing file '{key}': {e}")


def save_results(result_list, save_path):

    final_res, acc_dict = {}, {}
    correct, total, total_score = 0, 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        check = res['check']
        score = res['score']
        total_score += score
        
        if check == 'yes':
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100
    final_res['Avg_Score'] = total_score / total

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')

def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        result_list = all_results['result_list']
    else:
        result_list = None
    # json_data = load_json(save_path, 'all_results.json')['result_list']
    return result_list


class EvalDataset(Dataset):

    def __init__(self, num_segments, test_ratio=None):
        super().__init__()
        self.num_segments = num_segments
        self.test_ratio = test_ratio
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_clip_gif,
            'frame': self.read_frame,
        }
        
    def __getitem__(self, index) -> Any:
        raise NotImplementedError('')
        
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)     # Changed this to 2 since > 2 gave decode errors (https://github.com/dmlc/decord/issues/145#issuecomment-2185433737)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
                if len(images_group) == len(frame_indices):
                    break

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_clip_gif(self, video_path, bound=None, fps=25):
        gif = VideoFileClip(video_path)
        frames = gif.iter_frames()
        max_frame = gif.reader.nframes - 1
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(frames):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        return images_group

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # self.data_list = self.data_list[::200] # debug
        if self.test_ratio is None:
            self.data_list = self.data_list[rank::world_size]
        else:
            np.random.RandomState(42).shuffle(self.data_list)
            if isinstance(self.test_ratio, float):
                num_samples = int(len(self.data_list) * self.test_ratio)
            else:
                num_samples = int(self.test_ratio)
            self.data_list = self.data_list[rank:num_samples:world_size]
            
            

class MVBenchDataset(EvalDataset):
    data_list_info = {
        # "task_type (sub task name)": ("json file name", "image/video prefix", "data_type", "bound")
        "Action Sequence": ("action_sequence.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/nturgbd/", "video", False),
        "Character Order": ("character_order.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "/home/cr8dl-user/sameep/datasets/mvbench/video/clevrer/video_validation/", "video", False),
    }
    data_dir = "/home/cr8dl-user/sameep/datasets/mvbench/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        # self.data_list = self.data_list[:100] # for debug
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
                
        # # transform
        # crop_size = resolution
        # scale_size = resolution
        # input_mean = [0.48145466, 0.4578275, 0.40821073]
        # input_std = [0.26862954, 0.26130258, 0.27577711]
        # self.transform = T.Compose([
        #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        #     GroupCenterCrop(crop_size),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     GroupNormalize(input_mean, input_std) 
        # ])
    
    def __getitem__(self, idx):
        question, answer = self.qa_template(self.data_list[idx]['data'])
        task_type = self.data_list[idx]['task_type']
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])


        # images_group = decord_method(video_path, bound)
        try: # might be problem with decord
            images_group = decord_method(video_path, bound)
        except Exception as e:
            print(f'error decoding {video_path}')
            task_type = 'error_reading_video'
            images_group = None

        return {
            'video_path': video_path, 
            'video_pils': images_group, # some might use the original pils and do their own transforms
            'question': question, 
            'answer': answer,
            'task_type': task_type,
        }
        

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

