exp_dir="${_SAVE_DIR}/Qwen2.5-VL-3B-SFT-train-aitw-cold-start"
showui_dir=$(pwd)
ckpt_dir="${exp_dir}/ckpt_model/"
merge_dir="${ckpt_dir}/merged_model"

cd "$ckpt_dir" || { echo "Failed to cd to $ckpt_dir"; exit 1; }
python zero_to_fp32.py . pytorch_model.bin
mkdir -p merged_model

cd "$showui_dir"
python3 merge_weight.py --exp_dir="$exp_dir"

echo "$merge_dir"