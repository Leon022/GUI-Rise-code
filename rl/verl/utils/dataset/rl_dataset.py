# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union, Dict
import math

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.utils.data import Dataset, DataLoader, Sampler

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.prompt import _AITW_SYSTEM_THINKING, _MIND2WEB_SYSTEM_THINKING, SYSTEM_THINKING



logger = logging.getLogger(__name__)

from PIL import Image, ImageDraw, ImageFont
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import time
from datetime import datetime
import pickle


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def add_data_idx_with_index(batch, indices):
    """为每个样本添加 data_idx"""
    result = {'extra_info': []}
    for i, (extra_info, idx) in enumerate(zip(batch['extra_info'], indices)):
        if isinstance(extra_info, dict):
            # 创建新的字典副本
            new_extra_info = extra_info.copy()
            new_extra_info['data_idx'] = idx
            result['extra_info'].append(new_extra_info)
        else:
            result['extra_info'].append(extra_info)
    return result

class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

        self.history_summary_enabled = config.get("history_summary_enabled", True)
        self.history_summary = []
        self._init_history_summary()
        self.history_summary_score = [0.3] * len(self.history_summary)

        self.dataframe = self.dataframe.map(
            lambda example, idx: {"data_idx": idx},
            with_indices=True
        )
        
        self.data_list = []
        for i in range(len(self.dataframe['reward_model'])):
            self.data_list.append({
                'reward_gt': self.dataframe['reward_model'][i],
                'ground_truth': self.dataframe['reward_model'][i]['ground_truth'],
                'tgt_code': self.dataframe["extra_info"][i].get("code"),
                'audit_log_id': self.dataframe["extra_info"][i].get("audit_log_id"),
            })


    def _init_history_summary(self):
        """初始化 history_summary，从 data['extra_info'] 中的 'input_summary' 字段获取初始值"""
        if hasattr(self, 'dataframe'):
            # 从 extra_info 中的 input_summary 字段初始化
            self.history_summary = []
            for i in range(len(self.dataframe)):
                extra_info = self.dataframe[i].get("extra_info", {})
                # 从 extra_info 中获取 input_summary，如果没有则使用空字符串
                input_summary = extra_info.get("input_summary", "")
                self.history_summary.append(input_summary)
            
            print(f"Initialized history_summary from data, length: {len(self.history_summary)}")

    def update_history_summary(self, index: int, summary: str, summary_score: float):
        """
        更新指定索引的 history_summary
        
        Args:
            index: 要更新的数据索引
            summary: 新的历史摘要
        """
        if 0 <= index < len(self.history_summary):
            if summary_score > self.history_summary_score[index]:
                self.history_summary[index] = summary
                self.history_summary_score[index] = summary_score
        else:
            raise IndexError(f"Index {index} out of range [0, {len(self.history_summary)-1}]")
    
    def get_history_summary(self, index: int) -> str:
        """获取指定索引的 history_summary"""
        if 0 <= index < len(self.history_summary):
            return self.history_summary[index]
        else:
            raise IndexError(f"Index {index} out of range [0, {len(self.history_summary)-1}]")
    
    def get_history_summary_score(self, index: int) -> float:
        """获取指定索引的 history_summary_score"""
        if 0 <= index < len(self.history_summary_score):
            return self.history_summary_score[index]
        else:
            raise IndexError(f"Index {index} out of range [0, {len(self.history_summary_score)-1}]")

    def get_next_item(self, current_index: int, is_replaced=False, replaced_history_summary=None):
        """
        查询下一个数据项
        
        Args:
            current_index: 当前索引
            
        Returns:
            如果下一个索引存在且 next_verify 为 True，返回处理后的数据
            否则返回 None
        """
        next_index = current_index + 1
        
        # 检查索引是否越界
        if next_index >= len(self):
            return None
            
        # 获取下一个数据项的原始数据
        next_row = self.dataframe[next_index]
        
        # 检查 next_verify 字段
        # if not next_row.get("next_verify", False):
        #     return None
            
        # 如果 next_verify 为 True，返回处理后的数据
        return self.__getitem__(next_index, is_replaced, replaced_history_summary)



    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        # 创建一个列表来存储被过滤掉的索引
        self.filtered_indices = []

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in messages.pop(image_key)] if image_key in messages else None
                    )
                    videos = (
                        [process_video(video) for video in messages.pop(video_key)] if video_key in messages else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:
                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            # self.dataframe = self.dataframe.filter(
            #     lambda doc: doc2len(doc) <= self.max_prompt_length,
            #     num_proc=self.num_workers,
            #     desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            # )

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item, is_replaced=False, replaced_history_summary=None):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        if is_replaced:
            current_history_summary = self.history_summary[item] if self.history_summary_enabled else ""
            if isinstance(self.data_files, list):
                if 'aitw' in self.data_files[0].lower():
                    current_prompt = _AITW_SYSTEM_THINKING
                elif 'mind2web' in self.data_files[0].lower():
                    current_prompt = _MIND2WEB_SYSTEM_THINKING
                else:
                    current_prompt = SYSTEM_THINKING
            elif isinstance(self.data_files, str):
                if 'aitw' in self.data_files.lower():
                    current_prompt = _AITW_SYSTEM_THINKING
                elif 'mind2web' in self.data_files.lower():
                    current_prompt = _MIND2WEB_SYSTEM_THINKING
                else:
                    current_prompt = SYSTEM_THINKING
            current_prompt += f'\nTask: {row_dict["extra_info"]["task"]}'
            current_prompt += f'\nHistory summary: {current_history_summary}'
            row_dict['prompt'] = [
                        {
                            "role": "user",
                            "content": "<image>" + current_prompt,
                        }
                    ]


        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        # index = row_dict.get("extra_info", {}).get("index", 0)
        index = row_dict['data_idx']
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["item_id"] = item
        del row_dict['data_idx']
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

