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
"""
Preprocess the AITW dataset to parquet format
"""

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


_AITW_SYSTEM_THINKING = """You are an assistant trained to navigate the mobile.
Given a task instruction, a screenshot, and a last history action summary, output the think and action and wait for the next observation.
The think must strictly follow these reasoning steps:
(1) Progress Estimation: Task Decomposition, Planning and Progress Estimation
(2) Decision Reasoning: Strategy Formulation
(3) Action Prediction: Giving the answer action in <action>...</action>
(4) History Summary: Update the history summary according the action you executed

## Action Space
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required.
2. `INPUT`: Input a string into an element, value is a string to type and the position is not applicable.
3. `REMEMBER`: Remember a string, value is a string to type and the position is not applicable.
4. `SCROLL UP`: Scroll up for the screen.
5. `SCROLL DOWN`: Scroll down for the screen.
6. `SCROLL LEFT`: Scroll left for the screen.
7. `SCROLL RIGHT`: Scroll right for the screen.
8. `PRESS BACK`: Press for returning to the previous step, value and position are not applicable.
9. `PRESS HOME`: Press for returning to the home screen, value and position are not applicable.
10. `PRESS ENTER`: Press for submitting the input content, value and position are not applicable.
11. `STATUS TASK COMPLETE`: Indicate the task is completed, value and position are not applicable.
12. `STATUS TASK IMPOSSIBLE `: Indicate the task is impossible to complete, value and position are not applicable.

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


def get_answer(step):
    """Extract answer from step data."""
    action_type_id = step['action_type_id']
    action_type_text = step['action_type_text']

    click_point = None
    type_text = None
    if action_type_id == 4:
        if action_type_text == 'click':
            touch = step['touch']
            lift = step['lift']
            click_point = [(touch[0] + lift[0]) / 2, (touch[1] + lift[1]) / 2]
            click_point = [round(item, 2) for item in click_point]
    elif action_type_id == 3:
        type_text = step['type_text']

    answer = {'action': action_type_text.upper(), 'value': type_text, 'position': click_point}
    
    if 'think' in step:
        final_answer = """{}\n<answer>{}</answer>""".format(step["think"], answer)
    else:
        final_answer = answer
        
    return final_answer


def convert_data(aitw_data, loaded_answers_dict, using_memory_input=True):
    """Convert AITW data to training format.
    
    Args:
        aitw_data: Original AITW dataset
        loaded_answers_dict: Dictionary containing thinking results
        using_memory_input: If True, use thinking results for memory; 
                           if False, use previous step actions
    """
    total_step = []
    step_i = 0
    
    for scenario in aitw_data:
        aitw_subset = aitw_data[scenario]
        for sample in tqdm(aitw_subset):
            confirmed_task = sample[0]['goal']
            previous_actions = []
            step_history = []
            
            for i, step in enumerate(sample):
                filename = step['img_filename']
                img_url = os.path.join(args.imgs_dir, filename) + '.png'
                
                if not os.path.exists(img_url):
                    print(f"Image not found: {img_url}")
                    continue
                    
                # Get answer for current step
                answer_dict = get_answer({
                    "action_type_id": step["action_type_id"],
                    "action_type_text": step["action_type_text"],
                    "annot_position": step['annot_position'],
                    "touch": step['touch'],
                    "lift": step['lift'],
                    "type_text": step['type_text'],
                })
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
                
                # Build prompt
                prompt = _AITW_SYSTEM_THINKING
                prompt += f'\nTask: {confirmed_task}'
                prompt += f'\nHistory summary: {memory}'

                # Create data entry
                data = {
                    "data_source": step_i,
                    "prompt": [
                        {
                            "role": "user",
                            "content": "<image>" + prompt,
                        }
                    ],
                    "images": [{"image": img_url, 'min_pixels': 640*12*28, 'max_pixels': 1280*36*28}],
                    "ability": "gui",
                    "reward_model": {"style": "rule", "ground_truth": cur_answer},
                    "extra_info": {
                        "id": f"aitw_{step_i}",
                        "step_id": step_i,
                        "answer": cur_answer,
                        "question": prompt,
                        "history": previous_step,
                        "bbox_ref": step['annot_position'],
                        "step": step,
                        "input_summary": memory,
                        "is_last": i == len(sample)-1,
                        "is_first": len(previous_actions) == 0,
                        "task": confirmed_task
                    },
                }
                total_step.append(data)

                previous_actions.append(cur_answer)
                step_history.append(step)
                step_i += 1

    return total_step


def main():
    """Main function to process AITW data."""
    # Load data
    print(f"Loading AITW data from {args.aitw_data_path}")
    aitw_data = json.load(open(args.aitw_data_path, 'r'))
    
    # Load thinking results if using_memory_input is True
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
    print("Converting AITW data...")
    train_data_list = convert_data(aitw_data, thinking, args.using_memory_input)
    
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
    parser = argparse.ArgumentParser(description="Preprocess AITW dataset")
    
    # Data paths
    parser.add_argument('--aitw_data_path', type=str, required=True,
                       help="Path to the AITW annotation JSON file")
    parser.add_argument('--imgs_dir', type=str, required=True,
                       help="Path to the directory containing AITW images")
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
                       help="If set, use thinking results for memory; otherwise use previous step actions")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.using_memory_input and not args.thinking_results_path:
        parser.error("--thinking_results_path is required when --using_memory_input is set")
    
    if not os.path.exists(args.aitw_data_path):
        parser.error(f"AITW data path does not exist: {args.aitw_data_path}")
    
    if not os.path.exists(args.imgs_dir):
        parser.error(f"Images directory does not exist: {args.imgs_dir}")
    
    if args.using_memory_input and not os.path.exists(args.thinking_results_path):
        parser.error(f"Thinking results path does not exist: {args.thinking_results_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.output_subset_path:
        os.makedirs(os.path.dirname(args.output_subset_path), exist_ok=True)
    
    main()


# python aitw.py \
#   --aitw_data_path /mnt/bn/pistis/liutao.0220/DATASET/AITW/aitw_data_train.json \
#   --imgs_dir /mnt/bn/pistis/liutao.0220/DATASET/AITW/images \
#   --thinking_results_path /mnt/bn/pistis/liutao.0220/DATASET/AITW/metadata/aitw_train_thinking_v2_4_1000_from_gpt4o-mini.jsonl \
#   --output_path /mnt/bn/pistis/liutao.0220/DATASET/AITW/metadata/full_data.parquet \
#   --output_subset_path /mnt/bn/pistis/liutao.0220/DATASET/AITW/metadata/su_data.parquet \
#   --subset_size 300 \
#   --using_memory_input
