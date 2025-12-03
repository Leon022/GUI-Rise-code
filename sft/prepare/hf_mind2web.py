from PIL import Image, ImageDraw
import json
import os
import re
import pdb
from tqdm import tqdm
import random
import argparse
import jsonlines
from collections import defaultdict

parent_dir = "../DATASET"
imgs_dir =  f"{parent_dir}/Mind2Web/images"
anno_dir = f"{parent_dir}/Mind2Web"

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

def data_transform(version='train', mini=False):
    if version == 'cold_start':
        load_version = 'train'
    else:
        load_version = version
    mind2web_train = json.load(open(f"{anno_dir}/mind2web_data_{version}.json", 'r'))
    thinking_data = load_thinking_data(f"{anno_dir}/metadata/{load_version}_thinking_results.jsonl")

    total_step = []
    step_i = 0

    for episode in tqdm(mind2web_train):
        annot_id = episode["annotation_id"]
        confirmed_task = episode["confirmed_task"]

        step_history = []
        repr_history = []
        for i, (step, step_repr) in enumerate(zip(episode["actions"], episode["action_reprs"])):
            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(imgs_dir, filename)

            if i == 0:
                memory = ''
            else:
                memory = thinking_data[annotation_id][i-1]['history_summary']
            gt_think = thinking_data[annotation_id][i]
            
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path)

            total_step.append({
                            "split": version,
                            "id": "mind2web_{}".format(step_i), 
                            "annot_id": annot_id,
                            "action_uid": step["action_uid"],
                            
                            "website": episode["website"],
                            "domain": episode["domain"],
                            "subdomain": episode["subdomain"],

                            "task": confirmed_task,
                            "img_url": filename,
                            "img_size": image.size,

                            "step_id": i,
                            "step": step,
                            "step_repr": step_repr,
                            "step_history": step_history.copy(),
                            "repr_history": repr_history.copy(),

                            "with_think": True,
                            "gt_think": gt_think,
                            "memory": memory, 
                            })

            step_history.append(step)
            repr_history.append(step_repr)

            step_i += 1
            
        if mini and step_i > 1:
            break

    if mini:
        return total_step

    if version == 'cold_start':
        new_total_step = random.sample(total_step, 1000)
    else:
        return total_step

if __name__ == "__main__":
    version = 'cold_start'
    data = data_transform(version=version)
    save_url = f"{anno_dir}/metadata/hf_{version}.json"
    with open(save_url, "w") as file:
        json.dump(data, file, indent=4)


    test_full = []
    for version in ['test_task', 'test_domain', 'test_website']:
        test_full.extend(data_transform(version=version))
    save_url = f"{anno_dir}/metadata/hf_test_full.json"
    with open(save_url, "w") as file:
        json.dump(test_full, file, indent=4)

    # miniset
    test_full = []
    for version in ['test_task', 'test_domain', 'test_website']:
        test_full.extend(data_transform(version=version, mini=True))
    
    save_url = f"{anno_dir}/metadata/hf_test_mini.json"
    with open(save_url, "w") as file:
        json.dump(test_full, file, indent=4)
