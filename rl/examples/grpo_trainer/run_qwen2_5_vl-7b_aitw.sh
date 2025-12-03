set -x
ENGINE=${1:-vllm}

proj=qwen2.5-vl-3b_gui_rise_aitw
exps=aitw_true
export DEBUG_MODE="true"
export REWARD_LOG_PATH=./reward_logs/${proj}/${exps}

echo 'clear previous REWARD_LOG_PATH {$REWARD_LOG_PATH}'
rm -rf ${REWARD_LOG_PATH}

train_data=../../../DATASET/AITW/metadata/full_data.parquet
test_data=../../../DATASET/AITW/metadata/subset_data.parquet
base_model_path=
save_model_path=./checkpoints/${proj}/${exps}
reward_function_path=../../verl/utils/reward_score/aitw.py

export WANDB_API_KEY=
export WANDB_PROJECT="gui_rise"
export WANDB_MODE=online

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$test_data \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$base_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=24000 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$proj \
    trainer.experiment_name=$exp \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$save_model_path \
    custom_reward_function.path=&reward_function_path \
    custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=False \
    trainer.val_before_train=False $@