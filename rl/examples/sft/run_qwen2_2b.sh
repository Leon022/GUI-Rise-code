set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/bn/pistis/liutao.0220/verl/data/aitw/infigui_train_data_cold_start_1005-v2.parquet \
    data.val_files=/mnt/bn/pistis/liutao.0220/verl/data/aitw/infigui_train_data_cold_start_1005-v2.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=12000 \
    optim.lr=1e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=/mnt/bn/pistis/liutao.0220/MODEL/Qwen2-VL-2B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=infigui-sft \
    trainer.experiment_name=infigui-sft-qwen-2-vl-2b-instruct \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
