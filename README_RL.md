# RL Training Guide

This document provides instructions for the Reinforcement Learning (RL) training phase. The process includes data generation, executing the training script, and merging the final model checkpoints.

## Step 1: Generate RL Data

First, you need to prepare the training data required for the RL environment. The preprocessing scripts are located in the `rl/examples/data_preprocess/` directory.

Navigate to this directory and run the appropriate script (`aitw.py` or `mind2web.py`) to generate the data.

For example, to generate data for the AITW dataset, use the following command:
```bash
# Navigate to the data preprocessing directory
cd rl/examples/data_preprocess

# Run the script for AITW
python aitw.py \
  --aitw_data_path /path/to/your/AITW/aitw_data_train.json \
  --imgs_dir /path/to/your/AITW/images \
  --thinking_results_path /path/to/your/AITW/metadata/your_thinking_results.jsonl \
  --output_path /path/to/your/AITW/metadata/full_data.parquet \
  --output_subset_path /path/to/your/AITW/metadata/subset_data.parquet \
  --subset_size 300 \
  --using_memory_input
```
**Note:** Remember to replace the placeholder paths with the actual paths to your dataset files.

## Step 2: RL Training

Once the data is prepared, you can start the RL training process. The training is launched via a shell script.

Before running, you may need to review and configure the `run_qwen2_5_vl-3b_gui.sh` script to ensure all paths and parameters are set correctly for your environment.

Execute the following command from the project's root directory to start training:
```bash
bash rl/examples/run_qwen2_5_vl-3b_gui.sh
```

## Step 3: Merge Model Checkpoints

After the training is complete, the final step is to merge the trained model weights into a single checkpoint for evaluation or deployment.

Use the `verl.model_merger` module to perform this action.
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/your/experiment/output/global_step_{step_number}/actor \
    --target_dir /path/to/your/merged_model_output
```
**Note:**
*   Replace `/path/to/your/experiment/output/global_step_{step_number}/actor` with the path to your specific training checkpoint directory.
*   Replace `/path/to/your/merged_model_output` with the directory where you want to save the final merged model.