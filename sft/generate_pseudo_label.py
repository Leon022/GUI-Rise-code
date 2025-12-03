from PIL import Image, ImageDraw
import json
import jsonlines
import os
from tqdm import tqdm
import random
from datetime import datetime
import time
from openai import OpenAI
import re
import base64
import openai
from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
import uuid
import mimetypes
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# --- Constants and Prompts ---
SYSTEM_PROMPT = """You are an AI assistant designed to simulate the model's reasoning process before executing a given action in a gui navigation task.  Given the task instruction, current screenshot, the previous history summary, the current action to be executed and thought, generate a rigorous chain of thought. You must strictly follow these reasoning steps:
(1) Progress Estimation: Interface Comprehension and Progress Estimation
(2) Decision Reasoning: Strategy Formulation
(3) History Summary: Update the history summary according the action you executed

### Output format:
<Progress Estimation>
... (one or two sentence)
</Progress Estimation>
<Decision Reasoning>
... (one or two sentence)
</Decision Reasoning>
<History Summary>
... (one or two sentence)
</History Summary>

###Example Input & Output
Input:
Task Instruction: Find all events taking place in New York City during the month of September.
Current Action: {{'action': CLICK, 'value': 'Apply', 'position':[0.3, 0.66]}}
Previous History Summary: The user first changed the location to New York, then set the start date to September 1, and set the end data to September 30.
Output:
<Progress Estimation>
The user has configured the search parameters by setting the location to New York and selecting the date range of September 1–30. However, the event list still shows entries for March, indicating that the updated filters have been selected but not yet applied.
</Progress Estimation>
<Decision Reasoning>
Clicking “Apply” will submit the selected date range and trigger the interface to refresh the event list, ensuring that only events occurring in New York City during September are displayed.
</Decision Reasoning>
<History Summary> 
So far, the user has completed the main filter configuration for finding September events in New York. The current action applies these settings so the interface can update accordingly.
</History Summary>
"""

USER_PROMPT = """
###Input
Task Instruction: {_TASK}
Current Action: {_ACTION}
Previous History Summary: {_MEMO}
"""

# --- Utility Functions ---

def encode_image_to_base64_optimized(image_path, max_dimension=6000, quality=90):
    """
    优化版本：限制最大边长并控制质量
    
    参数:
        image_path: 图片文件路径
        max_dimension: 最大边长限制（默认6000）
        quality: JPEG/WEBP质量（1-100，默认90）
    
    返回:
        Data URL字符串
    """
    # 猜测MIME类型
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # 默认使用JPEG
    
    try:
        with Image.open(image_path) as img:
            # 转换模式
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            width, height = img.size
            
            # 检查是否需要调整大小
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"优化压缩: {width}x{height} → {new_width}x{new_height}")
            
            # 保存到缓冲区
            buffer = BytesIO()
            
            # 根据格式选择保存参数
            if mime_type == "image/jpeg":
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
            elif mime_type == "image/png":
                # PNG压缩级别（0-9，9是最大压缩）
                img.save(buffer, format="PNG", optimize=True, compress_level=6)
            else:
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                mime_type = "image/jpeg"
            
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
            
            # 输出大小信息
            size_kb = len(buffer.getvalue()) / 1024
            print(f"编码完成: {size_kb:.1f} KB, 尺寸: {img.size[0]}x{img.size[1]}")
    
    except Exception as e:
        raise ValueError(f"无法处理图像文件: {e}")
    
    return f"data:{mime_type};base64,{encoded_string}"

def get_value_mind2web(step_repr):
    """Extracts the value from a Mind2Web step representation string."""
    pattern = r'\]\s+(.*?)\s+->'
    match = re.search(pattern, step_repr)
    return match.group(1) if match else None

def get_answer_mind2web(sample, step, step_repr):
    """Formats the action dictionary for a Mind2Web step."""
    image_size = sample['img_size']
    action_type = step['operation']['op']
    
    if action_type != 'TYPE':
        element = get_value_mind2web(step_repr)
    else:
        element = step['operation']['value']
        
    bbox = step['bbox']
    point_x = bbox["x"] + (bbox["width"] / 2)
    point_y = bbox["y"] + (bbox["height"] / 2)
    click_point = [round(point_x / image_size[0], 2), round(point_y / image_size[1], 2)]
    
    return {'action': action_type, 'value': element, 'position': click_point}

def get_answer_aitw(step):
    """Formats the action dictionary for an AITW step."""
    action_type_id = step['action_type_id']
    action_type_text = step['action_type_text']
    click_point = None
    type_text = None

    if action_type_id == 4 and action_type_text == 'click':
        touch = step['touch']
        lift = step['lift']
        click_point = [round((touch[0] + lift[0]) / 2, 2), round((touch[1] + lift[1]) / 2, 2)]
    elif action_type_id == 3:
        type_text = step['type_text']

    return {'action': action_type_text.upper(), 'value': type_text, 'position': click_point}

def get_image_message(user_content: str, url: str | None = None) -> HumanMessage:

    #  UserPrompt
    user_message = ChatCompletionContentPartTextParam(text=user_content, type="text")
    screenshot_message = HumanMessage(role="user", content=[user_message], id=str(uuid.uuid4()))

    # 添加截图消息
    if url:
        snap_content = ChatCompletionContentPartImageParam(image_url=ImageURL(url=url), type="image_url")
        screenshot_message.content.append(snap_content)

    return screenshot_message

def gpt_response(messages, client, args):
    """
    Sends a request to the GPT model, extracts sections into a dictionary, and retries on failure.
    """
    try_num = 0
    while True:
        try:
            if try_num > 50:
                print("Error: Exceeded maximum retry attempts.")
                return None

            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                top_p=1.0,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content

            progress_match = re.search(r'<Progress Estimation>(.*?)</Progress Estimation>', content, re.DOTALL)
            decision_match = re.search(r'<Decision Reasoning>(.*?)</Decision Reasoning>', content, re.DOTALL)
            history_match = re.search(r'<History Summary>(.*?)</History Summary>', content, re.DOTALL)

            if progress_match and decision_match and history_match:
                extracted_data = {
                    "progress_estimation": progress_match.group(1).strip(),
                    "decision_reasoning": decision_match.group(1).strip(),
                    "history_summary": history_match.group(1).strip(),
                }
                return extracted_data
            else:
                print(f'Error: Invalid response format. Could not extract all sections. Retry {try_num}')
                try_num += 1
        except Exception as e:
            print(f"{datetime.now()} - Error calling API: {e}. Retrying...")
            time.sleep(3)
            try_num += 1

# --- Main Processing Functions ---

def get_summary_mind2web(episode, imgs_dir, client, args, output_path):
    """Processes a single episode from the Mind2Web dataset."""
    pseudo_labels = []
    annot_id = episode["annotation_id"]
    confirmed_task = episode["confirmed_task"]

    for step_id, (step, step_repr) in enumerate(tqdm(zip(episode["actions"], episode["action_reprs"]), desc=f"Processing {annot_id}")):
        filename = f"{annot_id}-{step['action_uid']}.jpg"
        img_path = os.path.join(imgs_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue

        with Image.open(img_path) as image:
            item = {'img_url': filename, 'img_size': image.size}
            answer_dict = get_answer_mind2web(item, step, step_repr)
            gt_action = str(answer_dict)

        input_history = pseudo_labels[-1]['history_summary'] if step_id > 0 else ''
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(_TASK=confirmed_task, _ACTION=gt_action, _MEMO=input_history)

        system_message = SystemMessage(
            content=[
                {
                    "text": system_prompt,
                    "type": "text"
                }
            ],  # 传入已有的content
            role="system"            # 角色固定为"system"
        )
        user_message = HumanMessage(
            content=[
                {
                    "text": user_prompt,
                    "type": "text"
                }
            ],  # 传入已有的content
            role="user"            # 角色固定为"user"
        )
        image_url_base64 = encode_image_to_base64_optimized(img_path)
        snap_content = ChatCompletionContentPartImageParam(image_url=ImageURL(url=image_url_base64), type="image_url")
        user_message.content.append(snap_content)
        total_messages = [system_message, user_message]
        import ipdb; ipdb.set_trace()
        response = gpt_response(total_messages, client, args)
        if response:
            pseudo_labels.append(response)

        line_data = {
                "annotation_id": annot_id,
                "step_id": step_id,
                "pseudo_labels": response
            }
        with jsonlines.open(f'{output_path}_thinking_results.jsonl', mode='a') as writer:
            writer.write(line_data)

def get_summary_aitw(episode, imgs_dir, client, args, output_path):
    """Processes a single episode from the AITW dataset."""
    pseudo_labels = []

    for step in episode:
        step_id = step['step']
        annot_id = step['ep_id']

        confirmed_task = step['goal']
        filename = step['img_filename'] + '.png'
        img_path = os.path.join(imgs_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue

        answer_dict = get_answer_aitw(step)
        gt_action = str(answer_dict)

        input_history = pseudo_labels[-1]['history_summary'] if step_id > 0 else ''
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(_TASK=confirmed_task, _ACTION=gt_action, _MEMO=input_history)

        system_message = SystemMessage(
            content=[
                {
                    "text": system_prompt,
                    "type": "text"
                }
            ],  # 传入已有的content
            role="system"            # 角色固定为"system"
        )
        user_message = HumanMessage(
            content=[
                {
                    "text": user_prompt,
                    "type": "text"
                }
            ],  # 传入已有的content
            role="user"            # 角色固定为"user"
        )
        image_url_base64 = encode_image_to_base64_optimized(img_path)
        snap_content = ChatCompletionContentPartImageParam(image_url=ImageURL(url=image_url_base64), type="image_url")
        user_message.content.append(snap_content)
        total_messages = [system_message, user_message]

        response = gpt_response(total_messages, client, args)
        if response:
            pseudo_labels.append(response)

        line_data = {
                "annotation_id": annot_id,
                "step_id": step_id,
                "pseudo_labels": response
            }
            
        with jsonlines.open(f'{output_path}_thinking_results.jsonl', mode='a') as writer:
            writer.write(line_data)



def main(args, max_workers=1):
    """Main function to orchestrate the data processing."""
    # Initialize the OpenAI client using the provided API key
    client = openai.OpenAI(
        api_key=args.openai_api_key,
    )

    # Ensure output directory exists
    dataset_output_dir = os.path.join(args.output_dir, args.dataset, 'meatadata')
    # os.makedirs(dataset_output_dir, exist_ok=True)
    print(dataset_output_dir)

    executor = ThreadPoolExecutor(max_workers=max_workers)

    if args.dataset == 'AITW':
        imgs_dir = os.path.join(args.parent_dir, 'AITW', 'images')
        data_path = os.path.join(args.parent_dir, 'AITW', f'aitw_data_{args.version}.json')
        
        print(f"Loading AITW data from: {data_path}")
        aitw_data = json.load(open(data_path, 'r'))
        output_path = os.path.join(dataset_output_dir, f"{args.version}")
        tasks = []
        total_episodes = sum(len(aitw_data[scenario]) for scenario in aitw_data)
        
        for scenario in aitw_data:
            aitw_subset = aitw_data[scenario]
            for episode in aitw_subset:
                tasks.append(
                    executor.submit(
                        get_summary_aitw,  
                        episode,
                        imgs_dir,
                        client,
                        args,
                        output_path
                    )
                )
        with tqdm(total=total_episodes, desc="Processing episodes") as pbar:
            for future in as_completed(tasks):
                future.result()  
                pbar.update(1)   

    elif args.dataset == 'Mind2Web':
        imgs_dir = os.path.join(args.parent_dir, 'Mind2Web', 'Mind2Web', 'images')
        anno_dir = os.path.join(args.parent_dir, 'Mind2Web', 'Mind2Web')
        if args.version == 'train':
            sub_data_name = ['train']
        else:
            sub_data_name = ['test_domain', 'test_task', 'test_website']
        mind2web_data = []
        for sub_name in sub_data_name:
            data_path = os.path.join(anno_dir, f'mind2web_data_{sub_name}.json')
            print(f"Loading Mind2Web data from: {data_path}")
            mind2web_data += json.load(open(data_path, 'r'))
        tasks = []
        total_episodes = len(mind2web_data)
        output_path = os.path.join(dataset_output_dir, f"{args.version}")
        
        for episode in tqdm(mind2web_data):
            tasks.append(
                    executor.submit(
                        get_summary_mind2web,  
                        episode,
                        imgs_dir,
                        client,
                        args,
                        output_path
                    )
                )
            
        with tqdm(total=total_episodes, desc="Processing episodes") as pbar:
            for future in as_completed(tasks):
                future.result()  
                pbar.update(1)   
    
    executor.shutdown()
    print("All episodes processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for GUI navigation datasets.")
    
    # --- Dataset Arguments ---
    parser.add_argument("--dataset", type=str, required=True, choices=['AITW', 'Mind2Web'],
                        help="The dataset to process.")
    parser.add_argument("--parent_dir", type=str, default="../DATASET",
                        help="The parent directory containing the dataset folders.")
    parser.add_argument("--output_dir", type=str, default="../DATASET",
                        help="The directory where results will be saved.")
    parser.add_argument("--version", type=str, default='train',
                        help="The data split to process (e.g., 'train', 'test').")

    # --- OpenAI Arguments ---
    parser.add_argument("--openai_api_key", type=str, required=True, 
                        help="Your OpenAI API key.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", 
                        help="The model name to use (e.g., 'gpt-4o').")
    parser.add_argument("--max_tokens", type=int, default=4096, 
                        help="The maximum number of tokens for the model response.")
    
    args = parser.parse_args()
    main(args)
