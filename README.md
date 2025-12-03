
# GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation

This repository contains the official implementation for the paper **GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation**.

GUI-Rise is an agent designed for GUI navigation with enhanced reasoning capabilities. It employs a three-stage sub-task framework that mimics the human "think-act-summarize" decision-making process, ensuring that the agent makes optimal decisions at each step based on sufficient historical information.

## Quick Start

### Environment Setup

**1. SFT (Supervised Fine-Tuning) & Eval (Evaluation)**

The environment for SFT and Evaluation is self-contained. Please navigate to the `sft` directory to set it up:
```bash
# Enter the sft directory
cd sft

# Create a conda environment
conda create -n gui-rise-sft python=3.10
conda activate gui-rise-sft

# Install dependencies
pip install -r requirements.txt
```

**2. RL (Reinforcement Learning)**

For Reinforcement Learning, we utilize the `verl` framework. Please navigate to the `rl` directory and follow the instructions in its dedicated `README.md` to create the environment.
```bash
# Enter the rl directory
cd rl

# Follow the setup instructions in rl/README.md
```

## Data Preparation

### Navigation Datasets

1.  Download [GUIAct](https://huggingface.co/datasets/yiye2023/GUIAct) and then use our `prepare/hf_guiact.ipynb` to create metadata for each split (i.e., web, mobile).
2.  Set up Mind2Web, AITW, and MiniWoB by following [SeeClick's Instruction](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md). Then, use our provided scripts (`prepare/hf_mind2web.py`, `prepare/hf_aitw.py`, `prepare/hf_miniwob.py`) to process them and generate the metadata.

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

## Training

### 1. SFT (Supervised Fine-Tuning)

The SFT stage aims to teach the model basic reasoning and history summarization skills through supervised fine-tuning.

**Data Preparation**:
Please ensure the SFT data has been prepared according to the "Data Preparation" section above.

**Start Training**:
```bash
# Activate the SFT environment
conda activate gui-rise-sft

# Run the SFT training script from the root directory
bash sft/scripts/run_sft.sh
```

### 2. RL (Reinforcement Learning)

In the RL stage, we use the Group Relative Policy Optimization (GRPO) algorithm to further optimize the model in a simulated GUI environment.

**Data Preparation**:
Please ensure the RL data has been prepared according to the "Data Preparation" section.

**Start Training**:
```bash
# Activate the RL environment
conda activate gui-rise-rl

# Run the RL training script from the root directory
bash rl/scripts/run_rl.sh
```

## Evaluation

The evaluation stage is used to test the model's performance on various benchmark test sets.

**Data Preparation**:
Ensure the evaluation datasets (e.g., AITW, Mind2Web) are set up as described in the "Data Preparation" section.

**Start Evaluation**:
```bash
# Activate the SFT/Eval environment
conda activate gui-rise-sft

# Run the evaluation script from the root directory
bash sft/scripts/run_eval.sh
```

## Acknowledgement

We would like to express our sincere gratitude to the contributors of the open-source projects and datasets used in this project, especially the team behind [SeeClick](https://github.com/njucckevin/SeeClick) for their dataset instructions.

## BibTeX

If you use GUI-Rise in your research, please cite our paper:

```bibtex
@article{liu2025guirise,
  title={GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation},
  author={Liu, Tao and Wang, Chongyu and Li, Rongjie and Yu, Yingchen and He, Xuming and Song, Bai},
  journal={arXiv preprint arXiv:2510.27210},
  year={2025},
  eprint={2510.27210v1},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```