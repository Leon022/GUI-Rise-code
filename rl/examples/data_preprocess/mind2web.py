# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import re
import random
from collections import defaultdict, Counter
from PIL import Image, ImageDraw
import numpy as np
import jsonlines
import pdb
from tqdm import tqdm
from IPython.display import display
import datasets


_MIND2WEB_SYSTEM_THINKING = """You are an assistant trained to navigate the web. 
Given a task instruction, a screenshot, and a last history action summary, output the think and ext action and wait for the next observation. 
The think must strictly follow these reasoning steps:
(1) Progress Estimation: Task Decomposition, Planning and Progress Estimation
(2) Decision Making: Strategy Formulation
(3) History Summary: Update the history summary according the action you executed

## Action Space
1. `CLICK`: Click on an element, value is the element to click and the position [x,y] is required.
2. `TYPE`: Type a string into an element, value is the string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is the value to select and the position [x,y] is required.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.

## Output Format
<think>
<Progress Estimation>
...
</Progress Estimation>
<Decision Reasoning>
...
</Decision Reasoning>
<action>
{{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}}
</action>
<History Summary>
...
</History Summary>
<think>

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1, where (x=0, y=0) denotes the top-left corner of the screen, 
with x increasing rightward and y increasing downward.
"""

def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with jsonlines.open(file_path, "r") as f:
        for line in f:
            data.append(line)
    return data

def get_bbox(action, image_size):
    """Extract bounding box coordinates and normalize to [0,1] range."""
    bbox = [action["bbox"]["x"], action["bbox"]["y"], 
            action["bbox"]["x"] + action["bbox"]["width"],
            action["bbox"]["y"] + action["bbox"]["height"]]
    bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], 
            bbox[2] / image_size[0], bbox[3] / image_size[1]]
    bbox = [round(item, 3) for item in bbox]
    return bbox


def get_value(step_repr):
    """Extract value from step representation using regex."""
    pattern = r'\]\s+(.*?)\s+->'
    match = re.search(pattern, step_repr)
    if match:
        return match.group(1)
    else:
        return None


def get_answer(sample, step, step_repr):
    """Extract answer from step data."""
    image = sample['img_url']
    image_size = sample['img_size']
    task = sample['task']

    action_type = step['operation']['op']
    if action_type != 'TYPE':
        element = get_value(step_repr)
    else:
        element = step['operation']['value']
    
    bbox = step['bbox']
    point_x = bbox["x"] + (bbox["width"] / 2)
    point_y = bbox["y"] + (bbox["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 2) for item in click_point]
    
    answer = {'action': action_type, 'value': element, 'position': click_point}
    return answer


def convert_data(mind2web_data, loaded_answers_dict, using_memory_input=True):
    """Convert Mind2Web data to training format.
    
    Args:
        mind2web_data: Original Mind2Web dataset
        summaries: Dictionary containing memory summaries
        args: Command line arguments
        version: Dataset version ('train', 'test', etc.)
    """
    total_step = []
    step_i = 0
    
    for episode in tqdm(mind2web_data):
        annot_id = episode["annotation_id"]
        confirmed_task = episode["confirmed_task"]

        previous_actions = []
        for idx, (step, step_repr) in enumerate(zip(episode["actions"], episode["action_reprs"])):
            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(args.imgs_dir, filename)

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
                
            with Image.open(img_path) as image:
                item = {
                    'img_url': filename,
                    'img_size': image.size,
                    'task': confirmed_task
                }
                answer_dict = get_answer(item, step, step_repr)
                cur_answer = str(answer_dict)
            
            # Build previous step summary
            previous_step = ""
            for j, action in enumerate(previous_actions):
                previous_step += 'Step' + str(j) + ', previous action: ' + action[:-1] + "}. "

            # Determine memory based on using_memory_input flag
            if using_memory_input:
                # Use thinking results for memory
                memory = loaded_answers_dict.get(step.get('annotation_id', ''), {}).get(i-1, {}).get('history_summary', '')
            else:
                # Use previous step actions for memory
                memory = previous_step

            prompt = _MIND2WEB_SYSTEM_THINKING
            prompt += f'\nTask: {confirmed_task}'
            prompt += f'\nHistory summary: {memory}'

            bbox_ref = get_bbox(step, image.size)
            
            data = {
                "data_source": step_i,
                "prompt": [
                    {
                        "role": "user",
                        "content": "<image>" + prompt,
                    }
                ],
                "images": [{"image": img_path, 'min_pixels': 640*12*28, 'max_pixels': 1280*36*28}],
                "ability": "gui",
                "reward_model": {"style": "rule", "ground_truth": cur_answer},
                "extra_info": {
                    "id": f"mind2web_{step_i}",
                    "step_id": step_i,
                    "answer": cur_answer,
                    "question": prompt,
                    "history": previous_step,
                    "bbox_ref": bbox_ref, 
                    "step": step,
                    "input_summary": memory,
                    "is_last": idx == len(episode["actions"])-1,
                    "is_first": idx == 0,
                    "task": confirmed_task
                },
            }
            total_step.append(data)
            previous_actions.append(cur_answer)
            step_i += 1

    return total_step


def main():
    """Main function to process Mind2Web data."""
    # Load Mind2Web data
    print(f"Loading Mind2Web {args.version} data from {args.mind2web_data_path}")
    mind2web_data = json.load(open(args.mind2web_data_path, 'r'))
    
    # Load memory summaries if using_memory_input is True
    thinking = defaultdict(dict)
    if args.using_memory_input:
        print(f"Loading thinking results from {args.thinking_results_path}")
        loaded_answers = load_jsonl(args.thinking_results_path)
        for text in loaded_answers:
            annotation_id = text['annotation_id']
            step_id = text['step_id']
            thinking[annotation_id][step_id] = text['pseudo_labels']
    else:
        print("Using previous step actions for memory (not loading thinking results)")
    
    # Convert data
    print("Converting Mind2Web data...")
    train_data_list = convert_data(mind2web_data, thinking, args.using_memory_input)
    
    # Save full dataset
    print(f"Saving full dataset to {args.output_path}")
    train_data = datasets.Dataset.from_list(train_data_list)
    df = train_data.to_pandas()
    df.to_parquet(args.output_path)
    
    # Save subset if specified
    if args.output_subset_path:
        subset_size = min(len(train_data_list), args.subset_size)
        print(f"Saving subset ({subset_size} samples) to {args.output_subset_path}")
        train_data_2 = datasets.Dataset.from_list(train_data_list[:subset_size])
        df_2 = train_data_2.to_pandas()
        df_2.to_parquet(args.output_subset_path)
    
    print("Processing completed!")
    print(f"Total samples processed: {len(train_data_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Mind2Web dataset")
    
    # Data paths
    parser.add_argument('--mind2web_data_path', type=str, required=True,
                       help="Path to the Mind2Web annotation JSON file")
    parser.add_argument('--imgs_dir', type=str, required=True,
                       help="Path to the directory containing Mind2Web images")
    parser.add_argument('--thinking_results_path', type=str, 
                       help="Path to the thinking results JSONL file (required if using_memory_input is True)")
    
    # Output paths
    parser.add_argument('--output_path', type=str, required=True,
                       help="Path to save the output parquet file")
    parser.add_argument('--output_subset_path', type=str, default=None,
                       help="Path to save a subset of the data (optional)")
    parser.add_argument('--subset_size', type=int, default=300,
                       help="Size of the subset to save (if output_subset_path is provided)")
    
    # Processing options
    parser.add_argument('--using_memory_input', action='store_true',
                       help="If set, use memory summaries for input; otherwise use previous step actions")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.using_memory_input and not args.thinking_results_path:
        parser.error("--thinking_results_path is required when --using_memory_input is set")
    
    if not os.path.exists(args.mind2web_data_path):
        parser.error(f"Mind2Web data path does not exist: {args.mind2web_data_path}")
    
    if not os.path.exists(args.imgs_dir):
        parser.error(f"Images directory does not exist: {args.imgs_dir}")
    
    if args.using_memory_input and not os.path.exists(args.thinking_results_path):
        parser.error(f"Thinking results path does not exist: {args.thinking_results_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.output_subset_path:
        os.makedirs(os.path.dirname(args.output_subset_path), exist_ok=True)
    
    main()

# python mind2web.py \
#   --aitw_data_path /path/to/mind2web_data_train.json \
#   --imgs_dir /path/to/images \
#   --thinking_results_path /path/to/thinking_results.jsonl \
#   --output_path /path/to/output/full_data.parquet \
#   --output_subset_path /path/to/output/subset_data.parquet \
#   --subset_size 300 \
#   --using_memory_input