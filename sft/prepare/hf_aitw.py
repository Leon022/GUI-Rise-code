import os
import cv2
import re
import pdb
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from IPython.display import display
from PIL import Image, ImageDraw
from data_utils import is_english_simple, bbox_2_point
import jsonlines
import random
from collections import defaultdict


parent_dir = "../DATASET"
imgs_dir =  f"{parent_dir}/AITW/images"
anno_dir = f"{parent_dir}/AITW"

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, "r") as f:
        for line in f:
            data.append(line)  # 每行是一个JSON对象，添加到列表
    return data

def load_thinking_data(file_path: str) -> defaultdict:
    """Loads and structures the thinking/pseudo-labels data from a JSONL file."""
    loaded_answers = load_jsonl(file_path)
    thinking = defaultdict(dict)
    for text in loaded_answers:
        annotation_id = text['annotation_id']
        step_id = text['step_id']
        thinking[annotation_id][step_id] = text['pseudo_labels']
    return thinking

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def data_transform(version='train', mini=False):
    if version == 'cold_start':
        load_version = 'train'
        action_dict = defaultdict(list)
    else:
        load_version = version
    aitw_data = json.load(open(f"{anno_dir}/aitw_data_{load_version}.json", 'r'))
    thinking_data = load_thinking_data(f"{anno_dir}/metadata/{load_version}_thinking_results.jsonl")
    
    total_step = []
    step_i = 0
    for scenario in aitw_data:
        aitw_subset = aitw_data[scenario]
        for sample in tqdm(aitw_subset):
            confirmed_task = sample[0]['goal']
    
            step_history = []
            for i, step in enumerate(sample):
                annotation_id = step['ep_id']
                filename = step['img_filename']
                img_url = os.path.join(imgs_dir, filename) + '.png'
                if not os.path.exists(img_url):
                    print(img_url)
                    continue
                image = Image.open(img_url)
                action_id = step["action_type_id"]
                action_type = step["action_type_text"]
                if i == 0:
                    memory = ''
                else:
                    memory = thinking_data[annotation_id][i-1]['history_summary']
                gt_think = thinking_data[annotation_id][i]
                # if action_id == 4:
                #     if action_type == "click":
                #         touch_point = step['touch']
                #         step_point = step['lift']
                #         click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
                # elif action_type == 3:
                #     typed_text = step["type_text"]
                data = {
                        "split": version,
                        "id": "aitw_{}".format(step_i), 
                        # "annot_id": annot_id,
                        # "action_uid": step["action_uid"],
                        "domain": scenario,
                        "ep_id": step['ep_id'],
                        "step_id": i,

                        "task": confirmed_task,
                        "img_url": filename,
                        "img_size": image.size,

                        "action_type_id": action_id,
                        "action_type_text": action_type,
                        "annot_position": step['annot_position'],
                        "touch": step['touch'],
                        "lift": step['lift'],
                        "type_text": step['type_text'],
                        
                        "step": step,
                        "step_history": step_history.copy(),

                        "with_think": True,
                        "gt_think": gt_think,
                        "memory": memory,  
                        }
                total_step.append(data)
                if version == 'cold_start':
                    action_dict[action_type].append(data)
                step_history.append(step)
                step_i += 1

                if mini and step_i > 50:
                    break
            if mini and step_i > 50:
                break
    
    if version == 'cold_start':
        select = {
            'click': 600,
            'type': 100,
            'status task complete': 50,
            'press enter': 50,
            'press home': 50,
            'scroll down': 50,
            'press back': 50,
            'scroll up': 50,
            'scroll left': 16,
            'scroll right': 16,
        }
        new_total_step = []
        for k,v in select.items():
            new_total_step.extend(random.sample(action_dict[k], v))
        print(len(new_total_step))
        return new_total_step
    else:
        return total_step

if __name__ == "__main__":
    # for version in ['train', 'test', 'val']: # 
    version = 'cold_start'
    data = data_transform(version=version)
    save_url = f"{anno_dir}/metadata/hf_{version}.json"
    with open(save_url, "w") as file:
        json.dump(data, file, indent=4)
    
    version = 'test'
    data = data_transform(version=version)
    save_url = f"{anno_dir}/metadata/hf_{version}.json"
    with open(save_url, "w") as file:
        json.dump(data, file, indent=4)
    
    version = 'test'
    data = data_transform(version=version, mini=True)
    save_url = f"{anno_dir}/metadata/hf_{version}_mini.json"
    with open(save_url, "w") as file:
        json.dump(data, file, indent=4)
