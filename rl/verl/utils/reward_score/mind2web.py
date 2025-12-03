import torch
import re
import numpy as np
import sys
from datetime import datetime
import os
import sys
sys.path.append('/mnt/bn/pistis/liutao.0220/verl/verl/utils/reward_score/')
from utils_aitw import *
import math

def extract_answer(text):
    """
    从文本中提取 <answer> 标签内容。
    如果 <answer> 标签不存在，返回原始文本。
    """
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def pred2json(prediction):
    prediction = prediction.replace('\"', '\'')
    pattern = r"'action':\s*'(.*?)',\s*'value':\s*(None|'(.*?)'),\s*'position':\s*(None|\[([0-9.]+),\s*([0-9.]+)\])"
    match = re.search(pattern, prediction)

    if match:
        action = match.group(1)
        value = match.group(2)
        if value == 'None':
            value = None
        else:
            value = match.group(3)

        position_group = match.group(4)
        if position_group == 'None':
            position = None
        else:
            position_x = float(match.group(5))
            position_y = float(match.group(6))
            position = [position_x, position_y]

        return {
            "action": action,
            "value": value,
            "position": position
        }
    else:
        raise ValueError(f"Input string '{prediction}' doesn't match the expected format")

def is_location_close(loc1, loc2, threshold=0.1):
    """
    判断两个坐标是否接近。
    :param loc1: 第一个坐标，格式为 (x1, y1)
    :param loc2: 第二个坐标，格式为 (x2, y2)
    :param threshold: 允许的最大距离（默认值为 10 像素）
    :return: 如果距离小于阈值，返回 True；否则返回 False。
    """
    x1, y1 = loc1
    x2, y2 = loc2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # 计算欧几里得距离
    return distance < threshold

def format_reward(completions, **kwargs) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = re.search(pattern, completions, re.DOTALL) is not None
    return 1.0 if matches else 0.0

def format_reward_think(completions):
    """Check if the Qwen model output matches a specific format."""
    patterns = [r"<Progress Estimation>.*?</Progress Estimation>\s", r"<Decision Making>.*?</Decision Making>\s", r"<action>.*?</action>", r"<History Summary>.*?</History Summary>\s"]
    rewards = 0
    for pattern in patterns:
        matches = re.search(pattern, completions, re.DOTALL) is not None 
        rewards += 1.0 if matches else 0.0
    return 1.0 if rewards == 4.0 else 0.0

def format_reward_action(completions):
    """Check if the Qwen model output matches a specific format."""
    try:
        student_answer = extract_answer(completions)
        pred_i = pred2json(student_answer)
        rewards = 1.0
    except Exception as e:
        rewards = 0.0
    return rewards


def mind2web_verify_action(completions, solution, extra_info):
    NUM_HISTORY = 4
    ground_truth = extract_answer(solution)
    student_answer = extract_answer(completions)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
    reward = 0.0
    try:
        sol = pred2json(ground_truth)
        pred_i = pred2json(student_answer)
        
        click_point = pred_i["position"]
        answer_point = sol["position"]
        bbox_ref = extra_info['bbox_ref']

        if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]) and sol["action"].lower() == pred_i["action"].lower():
            reward = 1.5
        elif sol["action"].lower() == pred_i["action"].lower():
            reward = 1.0
        else:
            reward = 0.0
    except Exception as e:
        print(f"Accuracy Position Reward Function Error: {e}")

    try:
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_path = os.getenv("REWARD_LOG_PATH")
        os.makedirs(log_path, exist_ok=True)
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        with open(os.path.join(log_path, 'report_rewards.log'), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
            f.write(f"Content: {completions}\n")
            f.write(f"Solution: {ground_truth}\n")
    except Exception as e:
        print(f"日志写入失败（但不中断流程）: {e}")

    return reward

def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None, format_score: float = 0.1) -> float:
    action_reward = mind2web_verify_action(solution_str, ground_truth, extra_info)
    action_format_reward = format_reward_action(solution_str)
    think_format_reward = format_reward_think(solution_str)
    final_rewatd = 1.0 * action_reward + format_score * (action_format_reward + think_format_reward)
    reward_dict = {
        "score": final_rewatd,
        "action_reward": action_reward,
        "action_format_reward": action_format_reward,
        "think_format_reward": think_format_reward
    }

    return reward_dict