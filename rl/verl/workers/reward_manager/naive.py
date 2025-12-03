# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from collections import defaultdict

import os
import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from typing import Dict, List, Union
from tqdm import tqdm

from threading import Lock

# lock = Lock()
import json

import asyncio
import jsonlines
import ray



def _token_decode(data_item, index, tokenizer):
    """处理单个数据项的内部方法，返回包含所有需要信息的结果字典"""
    # 提取输入数据
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    
    # 计算有效prompt长度和token
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    
    # 计算有效response长度和token
    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    # 解码文本
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    data_item.non_tensor_batch['prompt_str'] = prompt_str
    data_item.non_tensor_batch['response_str'] = response_str
    data_item.non_tensor_batch['valid_response_length'] = valid_response_length

    return {
        "index": index,
        "valid_response_length": valid_response_length,
        "prompt_str": prompt_str,
        "response_str": response_str,
    }



def _ids_preproc(data_item, index):
        
    # 提取输入数据
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    uid = data_item.non_tensor_batch['uid']
    
    # 计算有效prompt长度和token
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    
    # 计算有效response长度和token
    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    return {
        "index": index,
        "valid_response_length": valid_response_length,
        "valid_prompt_ids": valid_prompt_ids,
        "valid_response_ids": valid_response_ids,
    }


def do_reward_cal(compute_score, data_source, response_str, ground_truth, extra_info, index, valid_response_length, prompt_str, data_uid):
    time_start = time.perf_counter()
    extra_info.update({
        "data_uid": data_uid
    })
    score = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    
    return {
        "index": index,
        "valid_response_length": valid_response_length,
        "score": score,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "data_source": data_source,
        'time_consume': time.perf_counter() - time_start
    }

def _process_single_item(data_item, index: int, tokenizer, compute_score, reward_fn_key, decode_results) -> Dict:
        """处理单个数据项的内部方法，返回包含所有需要信息的结果字典"""
        # 提取输入数据
        # prompt_ids = data_item.batch["prompts"]
        # prompt_length = prompt_ids.shape[-1]
        
        # # 计算有效prompt长度和token
        # valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        # valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        # # 计算有效response长度和token
        # response_ids = data_item.batch["responses"]
        # valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        # valid_response_ids = response_ids[:valid_response_length]
        
        # # 解码文本
        prompt_str = decode_results[index]["prompt_str"]
        response_str = decode_results[index]["response_str"]
        valid_response_length = decode_results[index]['valid_response_length']

        
        # # 获取标签和额外信息
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        # prompt_ids = data_item.batch["prompts"]

        
        # 计算得分
        time_start = time.perf_counter()

        score = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        return {
            "index": index,
            "valid_response_length": valid_response_length,
            "score": score,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
            'time_consume': time.perf_counter() - time_start
        }

@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_workers=512, dataset=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.max_workers = os.cpu_count() 
        self.reward_log_path = os.getenv("REWARD_LOG_PATH")
        self.dataset = dataset
        os.makedirs(os.path.dirname(self.reward_log_path), exist_ok=True)

        print( f"NaiveRewardManager max_workers: {self.max_workers}")

    def __call__(self, data: DataProto, return_dict=False) -> Union[torch.Tensor, Dict]:
        """并发计算reward的主方法"""
        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        # 初始化结果存储
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        decode_results = {}
        time_start_all = time.perf_counter()

        all_res = [_ids_preproc( data[i], i) for i in range(len(data))]

        valid_prompt_ids = [each['valid_prompt_ids'] for each in all_res]
        valid_response_ids = [each['valid_response_ids'] for each in all_res]

        # 解码文本
        prompt_str = self.tokenizer.batch_decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.batch_decode(valid_response_ids, skip_special_tokens=True)

        for i in range(len(data)):
            all_res[i].update({
                'prompt_str': prompt_str[i],
                'response_str': response_str[i]
            })
            decode_results[i] = all_res[i]

        print(f"token_decode cost time: { time.perf_counter() - time_start_all:.2f}s")
                
        # 使用线程池并发处理
        time_start = time.perf_counter()

        all_time_consume = []
        with ThreadPoolExecutor(max_workers=self.max_workers*3) as executor:
            # 提交所有任务
            futures = []
            for index in range(len(data)):
                
                data_item = data[index]
                data_uid = data[index].non_tensor_batch['uid']
                prompt_str = decode_results[index]["prompt_str"]
                response_str = decode_results[index]["response_str"]
                valid_response_length = decode_results[index]['valid_response_length']
                # # 获取标签和额外信息
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", None)

                futures.append(executor.submit(do_reward_cal, self.compute_score, data_source, response_str, 
                                                ground_truth, extra_info, index, valid_response_length, prompt_str, data_uid))
            
            # 处理完成的结果
            for future in futures:
                result = future.result()
                i = result["index"]
                score = result["score"]
                all_time_consume.append(result["time_consume"])
                valid_response_length = result["valid_response_length"]
                
                # 处理得分结果
                if isinstance(score, dict):
                    reward = score["score"]
                    # 存储额外信息
                    for key, value in score.items():
                        reward_extra_info[key].append((i, value))
                else:
                    reward = score
                
                # 填充reward tensor
                reward_tensor[i, valid_response_length - 1] = reward
                
                # 打印示例（限制数量）
                data_source = result["data_source"]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("\n[Example Debug Info]")
                    print(f"[data_source] {data_source}")
                    print(f"[prompt] {result['prompt_str']}")
                    print(f"[response] {result['response_str']}")
                    print(f"[ground_truth] {result['ground_truth']}")
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}] {value}")
                    else:
                        print(f"[score] {score}")
        print(f"reward success, timing_s: max {np.max(all_time_consume):.2f} min {np.min(all_time_consume):.2f} avg {np.mean(all_time_consume):.2f}, overall { time.perf_counter() - time_start:.2f}")
    
        print('reward time consume', f'{ time.perf_counter() - time_start_all:.2f}')
        
        if 'save_res' in reward_extra_info:
            # save reward log
            payload = "\n".join(json.dumps(r) for r in reward_extra_info['save_res']) + "\n"
            with open(self.reward_log_path, "a", encoding="utf-8") as f:
                f.write(payload)

        # 返回结果
        if return_dict:

            for key, value in reward_extra_info.items():
                reward_extra_info[key] = [each[1] for each in sorted(reward_extra_info[key], key=lambda x: x[0])]
        
            output_dict = {"reward_tensor": reward_tensor}
            output_dict['reward_extra_info'] = reward_extra_info
            # output_dict.update(reward_extra_info)
            return output_dict
        return reward_tensor