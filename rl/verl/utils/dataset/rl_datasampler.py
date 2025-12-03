

import logging
import matplotlib.pyplot as plt
from typing import List, Iterator, Dict
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np

import math
import os
import pickle
from datetime import datetime

class BalancedDifficultySampler(Sampler):
    def __init__(self, 
                 dataset, 
                 difficulty_labels: List[float],
                 batch_size : int = 32,
                 warmup_steps: int = 100,
                 alpha: float = 0.75,
                 min_difficulty: float = 0.25,
                 max_difficulty: float = 0.75,
                 difficulty_weight_factor: float = 3.0,
                 reassign_freq: int = 5,
                 min_sample_freq: int = 50,
                 save_interval: int = 5,
                 pos_neg_amplify: float = 3.0,
                 under_sampling_amplifier: float = 20.0,
                 log_sampling_extremes_interval=128,
                 min_samlpe_weight = 0.25, 
                 weight_steepness=6,
                 tgt_difficulties=0.6,
                 verbose=True,
                 force_sample=True,
                 log_dir = "/mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/reward_logs/relatedness_qwen3_train_data_v4_aa+qi+cot_pos1_neg20_by_indentification_v7_20250902_difficulty_sampler_v2_enlarge_pos_neg_amplify_20_true/qa_agent"):
        """
        平衡版难度采样器 - 避免重复样本并确保所有样本被采样
        
        Args:
            dataset: 原始数据集
            difficulty_labels: 每个样本的初始难度值 (0.0~1.0)
            batch_size: 每个batch的大小
            warmup_steps: 预热步数，开始时使用均匀采样
            alpha: 难度更新的平滑系数
            min_difficulty: 中等难度范围下限
            max_difficulty: 中等难度范围上限
            difficulty_weight_factor: 中等难度样本的权重倍增因子
            reassign_freq: 重新计算权重的频率（每N个batch）
            min_sample_freq: 确保所有样本至少每N步被采样一次
        """
        super().__init__(None)
        self.dataset = dataset
        self.num_samples = len(difficulty_labels)
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.alpha = alpha
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_weight_factor = difficulty_weight_factor
        self.reassign_freq = reassign_freq
        self.min_sample_freq = min_sample_freq if min_sample_freq > 50 else 50
        self.save_interval = save_interval  # 新增：保存间隔
        self.save_dir = log_dir            # 新增：保存目录
        self.verbose = verbose
        self.log_sampling_extremes_interval = log_sampling_extremes_interval
        self.force_sample = force_sample
        self.min_weight = min_samlpe_weight
        self.weight_steepness = weight_steepness
        self.tgt_difficulties = tgt_difficulties
        
        # 存储样本难度和权重
        self.difficulties = np.array(difficulty_labels, dtype=np.float32)
        self.weights = np.ones(self.num_samples, dtype=np.float32)
        
        # 采样统计
        self.pos_neg_label = np.array([int(each['ground_truth']) for each in self.dataset.data_list])
        self.pos_neg_amplify = pos_neg_amplify
        self.under_sampling_amplifier = under_sampling_amplifier
        self.sampling_counts = np.zeros(self.num_samples, dtype=np.int32)
        self.last_sampled = np.zeros(self.num_samples, dtype=np.int32)  # 记录上次采样步数
        self.sampling_history = []
        self.difficulty_history = []
        self.sample_details: Dict[int, List[int]] = {}  # 记录每个样本被采样的步数
        
        # 初始权重计算（必须在last_sampled初始化之后）
        self.calculate_weights()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"difficulty_sampler_{timestamp}.log")  # 组合完整文件路径

        os.makedirs(log_dir, exist_ok=True) 
        # 记录采样
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # 保存到文件
                logging.StreamHandler()  # 输出到控制台
            ],
            force=True
        )
        self.logger = logging.getLogger("DifficultySampler")
        
        self.logger.info(f"Sampler initialized with {self.num_samples} samples")
        self.logger.info(f"Min difficulty: {self.min_difficulty:.2f}, Max difficulty: {self.max_difficulty:.2f}")
        self.logger.info(f"Weight factor: {self.difficulty_weight_factor}, Reassign freq: {self.reassign_freq}")
    
    def update_difficulties(self, indices: List[int], new_difficulties: List[float]):
        """更新样本难度值"""
        if self.verbose:
            self.logger.info(f"Update difficulties of : {indices}")

        for idx, new_diff in zip(indices, new_difficulties):
            # 使用指数移动平均平滑难度更新
            self.difficulties[idx] = (
                self.alpha * new_diff + 
                (1 - self.alpha) * self.difficulties[idx]
            )
            # self.difficulties[idx] = new_diff
    
    def calculate_weights(self):
        """根据难度值计算样本权重"""
        # 核心逻辑：中等难度样本获得更高权重

        def smooth_peak_function(x, tgt=0.65,):
            distance = np.abs(x - tgt)
            return np.maximum(self.min_weight, 1 - (2 * distance) ** self.weight_steepness)

        # in_zone_mask = (self.difficulties >= self.min_difficulty) & \
        #                (self.difficulties <= self.max_difficulty)
        # self.weights = np.where(in_zone_mask, self.difficulty_weight_factor, 1.0)
        # print(self.current_step, 'pos curr median difficulty', np.median(self.difficulties[self.pos_neg_label > 0]), 'curr mean difficulty', np.mean(self.difficulties[self.pos_neg_label > 0]))
        # print(self.current_step, 'neg curr median difficulty', np.median(self.difficulties[self.pos_neg_label == 0]), 'curr mean difficulty', np.mean(self.difficulties[self.pos_neg_label == 0]))

        self.weights = smooth_peak_function(self.difficulties, self.tgt_difficulties) * self.difficulty_weight_factor
        
        # 确保所有样本都有被采样的机会
        # 对于长时间未被采样的样本，增加权重
        steps_since_sampled = self.current_step - self.last_sampled
        under_sampled_mask = (steps_since_sampled > self.min_sample_freq) 
        init_sampled_mask = (steps_since_sampled <= self.min_sample_freq) 
        
        if np.any(under_sampled_mask):
            under_sample_weight = steps_since_sampled[under_sampled_mask] /self.min_sample_freq + 1
            self.weights[under_sampled_mask] = self.weights[under_sampled_mask] + (self.under_sampling_amplifier * under_sample_weight)
        # 正负样本增强
        self.weights[self.pos_neg_label > 0] = self.weights[self.pos_neg_label > 0] + self.pos_neg_amplify

    
    def __iter__(self) -> Iterator[int]:
        """生成动态采样迭代器"""
        return self.dynamic_batch_generator()
    
    def dynamic_batch_generator(self) -> Iterator[int]:
        """动态batch生成器"""
        while True:
            self.current_step += 1
            if self.verbose:
            # print(f"Current step: {self.current_step}")
                self.logger.info(f"Current step: {self.current_step}")

            # 定期重新计算权重
            if self.current_step % self.reassign_freq == 0:
                self.calculate_weights()

            # 记录难度统计
            self.record_difficulty_stats()

            # 基于权重的概率采样（无放回）
            total_weight = self.weights.sum()
            if total_weight <= 0:
                probs = None  # 退化为均匀采样
            else:
                probs = self.weights / total_weight

            # 无放回采样确保batch内无重复
            indices = np.random.choice(
                self.num_samples,
                size=self.batch_size,
                p=probs,
                replace=False  # 关键修改：无放回采样
            )
            
            if self.force_sample:
                # 确保所有样本至少被采样一次的逻辑
                if self.current_step % self.min_sample_freq == 0:
                    if self.verbose:
                        self.logger.info("强制采样中")
                    average_sampling_baseline = math.ceil(np.mean(self.sampling_counts))
                    never_sampled = np.where(self.sampling_counts == 0)[0]
                    if len(never_sampled) > 0:
                        # 强制采样从未被采样的样本
                        n_to_sample = min(len(never_sampled), self.batch_size)
                        indices[:n_to_sample] = np.random.choice(
                            never_sampled, 
                            size=n_to_sample,
                            replace=False
                        )
                        # 更新这些样本的统计信息
                        # self.sampling_counts[indices[:n_to_sample]] += 1
                        self.last_sampled[indices[:n_to_sample]] = self.current_step
                        # 记录这些新采样的样本
                        for idx in indices[:n_to_sample]:
                            if idx not in self.sample_details:
                                self.sample_details[idx] = []
                            self.sample_details[idx].append(self.current_step)
            
            # 记录当前批次采样的详细数据
            self.log_current_batch(indices)

            # 更新采样计数和最后采样时间
            # self.sampling_counts[indices] += 1
            self.last_sampled[indices] = self.current_step
            
            # 更新每个样本的采样历史
            for idx in indices:
                if idx not in self.sample_details:
                    self.sample_details[idx] = []
                self.sample_details[idx].append(self.current_step)
            
            # 记录采样统计
            self.record_sampling_stats()

            self.sampling_counts[indices] += 1
            # 返回当前batch的索引
            yield from indices.tolist()
    
    def log_current_batch(self, indices: np.ndarray):
        """记录当前批次采样的详细数据"""
        # 记录当前批次的所有样本ID
        if self.verbose:
            self.logger.info(f"Batch {self.current_step} sampled indices: {indices.tolist()}")
        
        # 记录每个样本的采样次数
        sampled_items = []
        for idx in indices:
            count = self.sampling_counts[idx] + 1  # +1 因为即将更新计数
            sampled_items.append(f"{idx}: {count}")
        
        # 分组显示，每行10个
        chunks = [sampled_items[i:i+10] for i in range(0, len(sampled_items), 10)]
        for chunk in chunks:
            if self.verbose:
                self.logger.info("  " + ", ".join(chunk))
    
    def __len__(self):
        # 返回一个极大值表示"无限"采样
        return 10**9
    
    def record_difficulty_stats(self):
        """记录难度统计信息"""
        in_zone = ((self.difficulties >= self.min_difficulty) & 
                  (self.difficulties <= self.max_difficulty)).sum()
        avg_diff = self.difficulties.mean()
        never_sampled = np.sum(self.sampling_counts == 0)
        
        stats = {
            "step": self.current_step,
            "avg_difficulty": avg_diff,
            "in_zone_percent": in_zone / self.num_samples * 100,
            "min_difficulty": self.difficulties.min(),
            "max_difficulty": self.difficulties.max(),
            "never_sampled": never_sampled
        }
        self.difficulty_history.append(stats)
        if self.verbose:
            self.logger.info(
                f"Step {self.current_step}: "
                f"AvgDiff={stats['avg_difficulty']:.3f}, "
                f"InZone={stats['in_zone_percent']:.1f}%, "
                f"MinDiff={stats['min_difficulty']:.3f}, "
                f"MaxDiff={stats['max_difficulty']:.3f}, "
                f"Unsampled={stats['never_sampled']}"
            )
    
    def record_sampling_stats(self):
        """记录采样统计信息"""
        # 按难度分组
        easy_mask = self.difficulties < self.min_difficulty
        medium_mask = (self.difficulties >= self.min_difficulty) & (self.difficulties <= self.max_difficulty)
        hard_mask = self.difficulties > self.max_difficulty
        
        # 计算平均采样次数
        easy_avg = self.sampling_counts[easy_mask].mean() if np.any(easy_mask) else 0
        medium_avg = self.sampling_counts[medium_mask].mean() if np.any(medium_mask) else 0
        hard_avg = self.sampling_counts[hard_mask].mean() if np.any(hard_mask) else 0
        
        # 计算采样比例
        total_samples = self.sampling_counts.sum()
        easy_ratio = self.sampling_counts[easy_mask].sum() / total_samples * 100 if total_samples > 0 else 0
        medium_ratio = self.sampling_counts[medium_mask].sum() / total_samples * 100 if total_samples > 0 else 0
        hard_ratio = self.sampling_counts[hard_mask].sum() / total_samples * 100 if total_samples > 0 else 0
        
        # 计算样本比例
        easy_percent = np.sum(easy_mask) / self.num_samples * 100
        medium_percent = np.sum(medium_mask) / self.num_samples * 100
        hard_percent = np.sum(hard_mask) / self.num_samples * 100
        
        stats = {
            "step": self.current_step,
            "easy_avg": easy_avg,
            "medium_avg": medium_avg,
            "hard_avg": hard_avg,
            "easy_ratio": easy_ratio,
            "medium_ratio": medium_ratio,
            "hard_ratio": hard_ratio,
            "easy_percent": easy_percent,
            "medium_percent": medium_percent,
            "hard_percent": hard_percent,
        }
        self.sampling_history.append(stats)
        if self.verbose:
            self.logger.info(
                f"Sampling Stats [Step {self.current_step}]: "
                f"Easy: avg={easy_avg:.2f} (samples: {easy_percent:.1f}%, sampling: {easy_ratio:.1f}%) | "
                f"Medium: avg={medium_avg:.2f} (samples: {medium_percent:.1f}%, sampling: {medium_ratio:.1f}%) | "
                f"Hard: avg={hard_avg:.2f} (samples: {hard_percent:.1f}%, sampling: {hard_ratio:.1f}%)"
            )
        
        # 记录采样次数最多和最少的样本
        self.log_sampling_extremes()

        if self.current_step % self.save_interval == 0:
            self.save_sampler_state()
    
    def log_sampling_extremes(self):
        """记录采样次数最多和最少的样本"""
        # 找出采样次数最多的样本
        max_count = np.max(self.sampling_counts)
        max_indices = np.where(self.sampling_counts == max_count)[0]
        
        # 找出采样次数最少的样本（包括0次）
        min_count = np.min(self.sampling_counts)
        min_indices = np.where(self.sampling_counts == min_count)[0]
        
        if self.current_step % self.log_sampling_extremes_interval == 0 :
            # 记录最多采样的样本
            if max_count > 0:
                self.logger.info(f"Most sampled samples ({max_count} times): {max_indices.tolist()}")
                # 详细记录每个样本的采样历史
                for idx in max_indices[:10]:  # 最多显示前10个
                    if idx in self.sample_details:
                        sample_times = ", ".join(map(str, self.sample_details[idx]))
                        self.logger.info(f"  Sample {idx} sampled at steps: {sample_times}")
            
            # 记录最少采样的样本
            if min_count == 0:
                if self.verbose:
                    self.logger.info(f"Never sampled samples: {min_indices.tolist()}")
            else:
                if self.verbose:
                    self.logger.info(f"Least sampled samples ({min_count} times): {min_indices.tolist()}")
                    # 详细记录每个样本的采样历史
                    for idx in min_indices[:10]:  # 最多显示前10个
                        if idx in self.sample_details:
                            sample_times = ", ".join(map(str, self.sample_details[idx]))
                            self.logger.info(f"  Sample {idx} sampled at steps: {sample_times}")
        
            # 记录采样分布概况
            unique_counts = np.unique(self.sampling_counts)
            count_distribution = {count: np.sum(self.sampling_counts == count) for count in unique_counts}
            self.logger.info("Sampling count distribution:")
            for count in sorted(count_distribution.keys()):
                self.logger.info(f"  {count} times: {count_distribution[count]} samples")
        
    def save_sampler_state(self):
        """保存采样器的当前状态到单个pkl文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 准备要保存的数据
            sampler_state = {
                'step': self.current_step,
                'timestamp': timestamp,
                'difficulties': self.difficulties.copy(),
                'weights': self.weights.copy(),
                'sampling_counts': self.sampling_counts.copy(),
                'last_sampled': self.last_sampled.copy(),
                'num_samples': self.num_samples,
                'batch_size': self.batch_size,
                'min_difficulty': self.min_difficulty,
                'max_difficulty': self.max_difficulty,
                'weight_factor': self.difficulty_weight_factor
            }
            
            # 保存到pkl文件
            filename = f"sampler_state_step_{self.current_step}_{timestamp}.pkl"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(sampler_state, f)
            if self.verbose:
                self.logger.info(f"Sampler state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save sampler state: {e}")
    
    # 新增方法：加载之前保存的状态
    def load_sampler_state(self, filepath):
        """从pkl文件加载采样器状态"""
        try:
            with open(filepath, 'rb') as f:
                sampler_state = pickle.load(f)
            
            # 恢复状态
            self.current_step = sampler_state['step']
            self.difficulties = sampler_state['difficulties']
            self.weights = sampler_state['weights']
            self.sampling_counts = sampler_state['sampling_counts']
            self.last_sampled = sampler_state['last_sampled']
            
            # 确保数组长度一致
            if len(self.difficulties) != self.num_samples:
                self.logger.warning(f"Loaded state has {len(self.difficulties)} samples, "
                                  f"but current sampler expects {self.num_samples}")
            
            self.logger.info(f"Sampler state loaded from {filepath} (step {self.current_step})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load sampler state: {e}")
            return False

    def plot_stats(self, output_dir="plots"):
        """绘制统计图表"""

        os.path.join( f'step_{self.current_step}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 难度变化图
        if self.difficulty_history:
            steps = [s['step'] for s in self.difficulty_history]
            avg_diffs = [s['avg_difficulty'] for s in self.difficulty_history]
            in_zone = [s['in_zone_percent'] for s in self.difficulty_history]
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(steps, avg_diffs, 'b-', label='Average Difficulty')
            plt.axhline(y=self.min_difficulty, color='r', linestyle='--', label='Min Difficulty')
            plt.axhline(y=self.max_difficulty, color='g', linestyle='--', label='Max Difficulty')
            plt.xlabel('Training Steps')
            plt.ylabel('Difficulty')
            plt.title('Difficulty Distribution Over Time')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(steps, in_zone, 'm-', label='% in Optimal Zone')
            plt.xlabel('Training Steps')
            plt.ylabel('Percentage')
            plt.title('Percentage of Samples in Optimal Difficulty Zone')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'difficulty_stats.png'))
            plt.close()
        
        # 2. 采样统计图
        if self.sampling_history:
            steps = [s['step'] for s in self.sampling_history]
            easy_avg = [s['easy_avg'] for s in self.sampling_history]
            medium_avg = [s['medium_avg'] for s in self.sampling_history]
            hard_avg = [s['hard_avg'] for s in self.sampling_history]
            medium_ratio = [s['medium_ratio'] for s in self.sampling_history]
            medium_percent = [s['medium_percent'] for s in self.sampling_history]
            
            # 平均采样次数
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(steps, easy_avg, 'r-', label='Easy Samples')
            plt.plot(steps, medium_avg, 'g-', label='Medium Samples')
            plt.plot(steps, hard_avg, 'b-', label='Hard Samples')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Sampling Count')
            plt.title('Average Sampling Count by Difficulty Level')
            plt.legend()
            plt.grid(True)
            
            # 采样比例
            plt.subplot(2, 1, 2)
            plt.plot(steps, medium_ratio, 'm-', label='Medium Sampling Ratio')
            plt.plot(steps, medium_percent, 'c--', label='Medium Sample Percentage')
            plt.axhline(y=33.3, color='k', linestyle='--', label='Uniform Ratio')
            plt.xlabel('Training Steps')
            plt.ylabel('Percentage')
            plt.title('Sampling Ratio vs Sample Percentage for Medium Difficulty')
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sampling_stats.png'))
            plt.close()
        
        # 3. 采样分布散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(self.difficulties, self.sampling_counts, alpha=0.5, s=10)
        plt.axvline(x=self.min_difficulty, color='r', linestyle='--', label='Min Difficulty')
        plt.axvline(x=self.max_difficulty, color='g', linestyle='--', label='Max Difficulty')
        plt.xlabel('Difficulty')
        plt.ylabel('Sampling Count')
        plt.title('Sampling Count vs Difficulty')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'sampling_vs_difficulty.png'))
        plt.close()
        
        # 4. 采样分布箱线图
        plt.figure(figsize=(12, 6))
        data = []
        labels = []
        
        # 按难度区间分组
        bins = np.linspace(0, 1, 11)  # 10个区间
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            mask = (self.difficulties >= low) & (self.difficulties < high)
            if np.any(mask):
                data.append(self.sampling_counts[mask])
                labels.append(f"{low:.1f}-{high:.1f}")
        
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=45)
        plt.xlabel('Difficulty Range')
        plt.ylabel('Sampling Count')
        plt.title('Sampling Distribution by Difficulty Range')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sampling_boxplot.png'))
        plt.close()
        
        print("Statistics plots saved to '%s' directory", output_dir)
    
    def analyze_sampling(self):
        """分析采样分布并打印摘要"""
        # 获取所有样本的难度和采样次数
        difficulties = self.difficulties
        sampling_counts = self.sampling_counts
        
        # 按难度分组
        easy_mask = difficulties < self.min_difficulty
        medium_mask = (difficulties >= self.min_difficulty) & (difficulties <= self.max_difficulty)
        hard_mask = difficulties > self.max_difficulty
        
        # 计算基本统计
        total_samples = len(difficulties)
        total_sampling = sampling_counts.sum()
        
        # 中等难度区间统计
        medium_count = np.sum(medium_mask)
        medium_sampling = sampling_counts[medium_mask].sum()
        
        # 打印统计摘要
        self.logger.info("\n" + "="*60)
        self.logger.info("Sampling Statistics Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Total sampling events: {total_sampling}")
        self.logger.info(f"Average sampling per sample: {total_sampling / total_samples:.2f}")
        
        self.logger.info("\nDifficulty Groups:")
        self.logger.info(f"  Easy (0.0-{self.min_difficulty:.2f}):")
        self.logger.info(f"    Samples: {np.sum(easy_mask)} ({np.sum(easy_mask)/total_samples*100:.1f}%)")
        self.logger.info(f"    Sampling: {sampling_counts[easy_mask].sum()} ({sampling_counts[easy_mask].sum()/total_sampling*100:.1f}%)")
        self.logger.info(f"    Avg sampling: {sampling_counts[easy_mask].mean():.2f}")
        
        self.logger.info(f"  Medium ({self.min_difficulty:.2f}-{self.max_difficulty:.2f}):")
        self.logger.info(f"    Samples: {medium_count} ({medium_count/total_samples*100:.1f}%)")
        self.logger.info(f"    Sampling: {medium_sampling} ({medium_sampling/total_sampling*100:.1f}%)")
        self.logger.info(f"    Avg sampling: {sampling_counts[medium_mask].mean():.2f}")
        
        self.logger.info(f"  Hard ({self.max_difficulty:.2f}-1.0):")
        self.logger.info(f"    Samples: {np.sum(hard_mask)} ({np.sum(hard_mask)/total_samples*100:.1f}%)")
        self.logger.info(f"    Sampling: {sampling_counts[hard_mask].sum()} ({sampling_counts[hard_mask].sum()/total_sampling*100:.1f}%)")
        self.logger.info(f"    Avg sampling: {sampling_counts[hard_mask].mean():.2f}")
        
        # 计算采样偏差
        medium_sampling_ratio = medium_sampling / total_sampling
        medium_sample_ratio = medium_count / total_samples
        sampling_bias = medium_sampling_ratio / medium_sample_ratio if medium_sample_ratio > 0 else 0
        
        self.logger.info("\nSampling Bias for Medium Difficulty:")
        self.logger.info(f"  Sample ratio: {medium_sample_ratio:.3f}")
        self.logger.info(f"  Sampling ratio: {medium_sampling_ratio:.3f}")
        self.logger.info(f"  Bias factor: {sampling_bias:.2f}x")
        self.logger.info("="*60 + "\n")
        
        return sampling_bias
