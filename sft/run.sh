WANDB_KEY=
_DATA_DIR=
_SAVE_DIR=
_MODEL_DIR=


deepspeed --num_gpus=8 --master_port 5677 train.py \
  --wandb_key=$WANDB_KEY \
  --model_id='Qwen/Qwen2.5-VL-3B-Instruct' \
  --version='Qwen/Qwen2.5-VL-3B-Instruct' \
  --local_weight \
  --local_weight_dir=$_MODEL_DIR \
  --dataset_dir=$_DATA_DIR \
  --log_base_dir=$_SAVE_DIR \
  --epochs=3 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="Qwen2.5-VL-3B-SFT-train-aitw-cold-start" \
  --train_ratio="1"  \
  --train_dataset="aitw"  \
  --train_json="hf_train_cold_start"   \
  --val_dataset="aitw"  \
  --val_json="hf_test_mini"    \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=32 \
  --lora_alpha=64  \
  --min_visual_tokens=1344  \
  --max_visual_tokens=1680  \
  --num_turn=100 \
  --crop_min=0.5 \
  --crop_max=1.5 \
  --random_sample \
  --record_sample \
  --lr=0.00001 \
  --uniform_prompt  \
  --ds_zero="zero2" \
  --gradient_checkpointing  \
  --lm_skip_ratio=0.5   \
  --lm_skip_layer='[1,28,0]'    \
  --num_history=4    \
  --interleaved_history='tttt'