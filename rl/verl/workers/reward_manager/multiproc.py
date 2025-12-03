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
from datetime import datetime
import os
import torch
import pickle
import numpy as np
import jsonlines
import json
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Union

from tqdm import tqdm
import threading

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.qa_agent import qa_agent_reward 
from verl.workers.reward_manager import register


def _ids_preproc(data_item, index):
        
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
    
    return {
        "index": index,
        "valid_response_length": valid_response_length,
        "valid_prompt_ids": valid_prompt_ids,
        "valid_response_ids": valid_response_ids,
    }


def do_reward_cal(data_source, response_str, ground_truth, extra_info, index, valid_response_length, prompt_str, acc_hist_dict):
    time_start = time.perf_counter()

    score, reward_extra_info = qa_agent_reward.compute_score(
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        acc_hist_dict=acc_hist_dict,
    )

    # print(f"model call time consume {time.perf_counter() - time_start:.3f}")

    res =  {
        "index": index,
        "valid_response_length": valid_response_length,
        "score": score,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "data_source": data_source,
        'extra_info': extra_info,
        'time_consume': time.perf_counter() - time_start
    }
    res.update(reward_extra_info)

    return res


def get_proc_num():
    # 获取当前主进程ID
    main_pid = os.getpid()
    main_process = psutil.Process(main_pid)
    
    # 获取所有子进程（即进程池中的工作进程）
    children = main_process.children(recursive=False)
    
    print(f"当前运行的进程数量: {len(children)}")
    print(f"子进程ID: {[child.pid for child in children]}")



@register("multiproc")
class MultiProcRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_workers=512) -> None:
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
        self.compute_score = qa_agent_reward
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.max_workers = os.cpu_count() 

        self.acc_hist_dict = None
        self.acc_dict_path = None
        self.init_acc_hist()

        print( f"NaiveRewardManager max_workers: {self.max_workers}")

    def init_acc_hist(self):
        # major_thres = 0.5
        reference_res_path = os.getenv("REFERENCE_ACC_RES_PATH")
        log_path = os.getenv("REWARD_LOG_PATH")
        os.makedirs(log_path, exist_ok=True)
        self.acc_dict_path = os.path.join(log_path, 'acc_hist_dict.json')

        if not os.path.exists(self.acc_dict_path):
            print(f'create new acc hist dict: {self.acc_dict_path}')
            
            with open(reference_res_path, 'rb') as f:
                error_res = pickle.load(f)

            acc_hist_dict = defaultdict(lambda: defaultdict(dict))
            code_acc_avg = dict()
            for code in error_res.keys():
                code_acc_avg[code] = [[], []]
                for err_typ in error_res[code].keys():
                    for pred_id, pred_cluster in enumerate(error_res[code][err_typ]):
                        pos = []
                        neg = []
                        
                        for each_res in pred_cluster:
                            if each_res.get('material') is not None:
                                material = each_res.get('material')
                                
                        gt = 0
                        for each_res in pred_cluster:
                            if each_res.get('gt') is not None:
                                gt = each_res.get('gt')
                                break
                                
                        for each_pred in pred_cluster:
                            if each_pred['pred'] == each_pred['gt']:
                                pos.append(each_pred)
                            else:
                                neg.append(each_pred)
                        
                        acc = len(pos) / len(pred_cluster)
                        # # 过滤都是0 或者 1的回答
                        # if len(neg) == 0 or len(pos) == 0:
                        #     continue
                        acc_hist_dict[str(code)][err_typ][str(pred_id)] = {'acc': acc, 'gt': gt, 'pred_id': pred_id,
                                                                        'pred_len': len(pred_cluster),
                                                                        'content_identification_result_dict': each_pred['content_identification']['input']['content_identification_result_dict']}

                        with open(self.acc_dict_path, 'w') as f:
                            json.dump(self.acc_hist_dict, f)

        else:
            print(f'load from existed acc hist dict: {self.acc_dict_path}')
            with open(self.acc_dict_path, 'r') as f:
                acc_hist_dict = json.load(f)
        
        self.acc_hist_dict = acc_hist_dict

        self.show_curr_acc()

    def show_curr_acc(self):
        print('curr baseline acc:')
        acc_summary = {}
        for code in self.acc_hist_dict:
            acc_summary[code] = {'pos': [], 'neg': []}
            for all_acc in self.acc_hist_dict[str(code)].values():
                for each_his in all_acc.values():
                    if each_his['gt']:
                        acc_summary[code]['pos'].append(each_his['acc'])
                    else:
                        acc_summary[code]['neg'].append(each_his['acc'])
            print(f"{code}: pos: avg: {np.mean(acc_summary[code]['pos'])*100:.2f} std: {np.std(acc_summary[code]['pos'])*100:.2f} median: {np.median(acc_summary[code]['pos'])*100:.2f} | neg: avg {np.mean(acc_summary[code]['neg'])*100:.2f} std: {np.std(acc_summary[code]['neg'])*100:.2f} median: {np.median(acc_summary[code]['neg'])*100:.2f} ")
        return acc_summary

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

        def do_decode_call():
            time_start = time.perf_counter()

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

            print(f"token_decode cost time: { time.perf_counter() - time_start:.2f}s")

        do_decode_call()

        def do_rew_call(thread_num):
            time_start = time.perf_counter()
            all_time_consume = []
            with ThreadPoolExecutor(max_workers=thread_num) as executor:
                # 提交所有任务
                futures = []
                for index in range(len(data)):
                    data_item = data[index]
                    prompt_str = decode_results[index]["prompt_str"]
                    response_str = decode_results[index]["response_str"]
                    valid_response_length = decode_results[index]['valid_response_length']
                    
                    # # 获取标签和额外信息
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch.get("extra_info", None)

                    code = extra_info["code"]
                    err_typ = extra_info["err_typ"]
                    pred_id = extra_info["pred_id"]
                    extra_info['data_uid'] = data_item.non_tensor_batch['uid']

                    ident_result = self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['content_identification_result_dict']
                    prev_acc = self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['acc']
                    pred_len = self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['pred_len']

                    prev_ref = {'ident_result': ident_result, 'prev_acc': prev_acc, 'pred_len': pred_len}

                    futures.append(executor.submit(do_reward_cal, data_source, response_str, 
                                                    ground_truth, extra_info, index, valid_response_length, 
                                                    prompt_str, prev_ref))
                
                # 处理完成的结果
                all_res = []
                for future in as_completed(futures):
                    result = future.result()
                    i = result["index"]
                    score = result["score"]
                    curr_acc = result["curr_acc"]

                    all_time_consume.append(result["time_consume"])
                    valid_response_length = result["valid_response_length"]

                    extra_info = result["extra_info"]
                    code = extra_info["code"]
                    err_typ = extra_info["err_typ"]
                    pred_id = extra_info["pred_id"]
                    item_name = extra_info["item_name"]
                    
                    prev_acc = self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['acc']
                    if curr_acc > min(prev_acc * 1.1, 0.9):
                        self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['acc'] = curr_acc * 0.88 + prev_acc * 0.12
                        self.acc_hist_dict[str(code)][err_typ][str(pred_id)]['content_identification_result_dict'][item_name]['content_iden'] = result["prompt_str"]

                    # 处理得分结果
                    if isinstance(score, dict):
                        reward = score["score"]
                        # 存储额外信息
                        for key, value in score.items():
                            reward_extra_info[key].append((i, value))
                    else:
                        reward = score

                    res = {
                        'current_time': datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
                        'gt': ground_truth,
                        'reward': reward,
                        'acc_list': result['acc_list'],
                        'all_answer': result['all_answer'],
                        'policy_content_str': result['policy_content_str'],
                        'item_name': extra_info['item_name'],
                        'code': code,
                        'err_typ': err_typ,
                        'pred_id': pred_id,
                    }
                    all_res.append(res)

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

            def dump_log():
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("REWARD_LOG_PATH")
                    os.makedirs(log_path, exist_ok=True)
                    with jsonlines.open(os.path.join(log_path, 'report_rewards.log'), mode="a") as writer:
                        writer.write_all(all_res)
            thr = threading.Thread(target=dump_log)
            thr.start()
            
            print(f"reward success, timing_s: max {np.max(all_time_consume):.2f} min {np.min(all_time_consume):.2f} avg {np.mean(all_time_consume):.2f}, overall { time.perf_counter() - time_start:.2f}")
            
        thread_num = int(self.max_workers * 5)
        print('thread num', thread_num, 'data_num', len(data))
        do_rew_call(thread_num)
        
        self.show_curr_acc()
        def dump_hist():
            with open(self.acc_dict_path, 'w') as f:
                json.dump(self.acc_hist_dict, f)

        thr = threading.Thread(target=dump_hist)
        thr.start()
        print('reward time consume', f'{ time.perf_counter() - time_start_all:.2f}')
        # 返回结果
        if return_dict:
            for key, value in reward_extra_info.items():
                reward_extra_info[key] = [each[1] for each in sorted(reward_extra_info[key], key=lambda x: x[0])]
        
            output_dict = {"reward_tensor": reward_tensor}
            output_dict['reward_extra_info'] = reward_extra_info
            return output_dict
        return reward_tensor