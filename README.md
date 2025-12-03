<div align="center">

# GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation

**[📄 Read the Paper on arXiv](https://arxiv.org/abs/2510.27210)**

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2510.27210"><img src="https://img.shields.io/badge/arXiv-2510.27210-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python" alt="Python 3.11"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

GUI-Rise is an agent designed for GUI navigation with enhanced reasoning capabilities. It employs a three-stage sub-task framework that mimics the human **"think-act-summarize"** decision-making process, ensuring that the agent makes optimal decisions at each step based on sufficient historical information.

<!-- Optional: Add a GIF of the agent in action here -->
<!-- <p align="center">
  <img src="path/to/your/demo.gif" width="80%">
</p> -->

---

## 📚 Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [Environment Setup](#environment-setup)
- [💾 Data Preparation](#-data-preparation)
  - [Navigation Datasets](#navigation-datasets)
  - [Directory Structure](#directory-structure)
- [🏋️ Training](#️-training)
  - [1. Supervised Fine-Tuning (SFT)](#1-supervised-fine-tuning-sft)
  - [2. Reinforcement Learning (RL)](#2-reinforcement-learning-rl)
- [🧪 Evaluation](#-evaluation)
- [🙏 Acknowledgement](#-acknowledgement)
- [✒️ Citation](#️-citation)

---

## 🚀 Quick Start

### Environment Setup

This project has two separate environments for SFT/Evaluation and RL.

<details>
<summary><strong>1. SFT (Supervised Fine-Tuning) & Eval Environment</strong></summary>

The environment for SFT and Evaluation is self-contained. Please navigate to the `sft` directory to set it up:

```bash
# Enter the sft directory
cd sft

# Create a conda environment
conda create -n gui-rise-sft python=3.11
conda activate gui-rise-sft

# Install dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>2. Reinforcement Learning (RL) Environment</strong></summary>

For Reinforcement Learning, we utilize the `verl` framework. Please navigate to the `rl` directory and follow the instructions in its dedicated `README.md` to create the environment.

```bash
# Enter the rl directory
cd rl

# Follow the setup instructions in rl/README.md
```
</details>

---

## 💾 Data Preparation

### Navigation Datasets

1.  **GUIAct**: Download from [Hugging Face](https://huggingface.co/datasets/yiye2023/GUIAct) and use our `prepare/hf_guiact.ipynb` to create metadata for each split (i.e., web, mobile).
2.  **Other Datasets**: Set up Mind2Web, AITW, and MiniWoB by following [SeeClick's Instructions](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md). Then, use our provided scripts (`prepare/hf_mind2web.py`, `prepare/hf_aitw.py`, `prepare/hf_miniwob.py`) to process them and generate the metadata.

### Directory Structure

After completing these steps, your dataset directory should be organized as follows:

```
$_DATA_DIR/
    ├── GUI_Course/
    │   └── GUIAct/
    │       ├── images/
    │       └── metadata/
    ├── Mind2Web/
    │   ├── images/
    │   └── metadata/
    ├── AITW/
    │   ├── images/
    │   └── metadata/
    └── MiniWob/
        ├── images/
        └── metadata/
```

---

## 🏋️ Training

### 1. Supervised Fine-Tuning (SFT)

The SFT stage teaches the model basic reasoning and history summarization skills. For detailed instructions on data generation and the training process, please refer to the dedicated guide:

**➡️ [SFT Training README](./gui_rise_readme_sft.md)**

### 2. Reinforcement Learning (RL)

In the RL stage, we use Group Relative Policy Optimization (GRPO) to further optimize the model. For detailed instructions on data generation and the training process, please refer to the dedicated guide:

**➡️ [RL Training README](./gui_rise_readme_rl.md)**


---

## 🧪 Evaluation

The evaluation stage tests the model's performance on various benchmark test sets.

- **Start Evaluation**:
  ```bash
  # Activate the SFT/Eval environment
  conda activate gui-rise-sft

  # Navigate to the sft directory and run the script
  cd sft
  bash scripts/eval.sh
  ```

---

## 🙏 Acknowledgement

We would like to express our sincere gratitude to the contributors of the open-source projects and datasets used in this project, especially:
- **[ShowUI](https://github.com/showlab/ShowUI/tree/main)** for their foundational work.
- **[verl](https://github.com/volcengine/verl/tree/main)** for the reinforcement learning framework.
- **[SeeClick](https://github.com/njucckevin/SeeClick)** for their clear dataset instructions.

---

## ✒️ Citation

If you use GUI-Rise in your research, please cite our paper:

```bibtex
@article{liu2025gui,
  title={GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation},
  author={Liu, Tao and Wang, Chongyu and Li, Rongjie and Yu, Yingchen and He, Xuming and Song, Bai},
  journal={arXiv preprint arXiv:2510.27210},
  year={2025}
}
```
