# SFT Training Guide

This document outlines the step-by-step process for Supervised Fine-Tuning (SFT) using a cold-start strategy. The process involves generating pseudo-labels, preparing the data, and then running the training script.

## Step 1: Generate Pseudo Labels for Cold-Start

First, we need to generate pseudo-labels to create a dataset for cold-start training. This is done by running the `generate_pseudo_label.py` script.

Execute the following command, replacing the dataset name and API key as needed.

```bash
# Example for AITW dataset
python generate_pseudo_label.py \
    --dataset AITW \
    --version test \
    --openai_api_key "your_openai_api_key"
```

After the script finishes, you will find the generated pseudo-label file, `${version}_thinking_results.jsonl`, in the metadata directory of your dataset: `$_DATA_DIR/$_DATA_NAME/metadata/`.

## Step 2: Prepare Cold-Start Training Data

With the pseudo-labels generated, you now need to process them to create the final cold-start training data.

Run the corresponding `prepare/hf_{_DATA_NAME}.py` script. For example, to process the AITW dataset, use the following command:

```bash
# Example for AITW dataset
python prepare/hf_aitw.py \
    --dataset AITW \
    --version cold_start
```

This will create the data in a format suitable for the training script.

## Step 3: Run Cold-Start Training

Before starting the training, please make sure to configure the global settings and paths in the `run.sh` script according to your environment.

Once configured, execute the script to begin the cold-start training:

```bash
bash run.sh
```

## Step 4: Save Model Checkpoints

Once you have finished the training, you can use the following command to save the final model checkpoint. This script will merge the model weights for easy deployment and evaluation.

```bash
bash auto_merge_model.sh
```