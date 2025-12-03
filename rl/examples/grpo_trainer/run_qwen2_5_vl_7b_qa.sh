set -x
ENGINE=${1:-vllm}

proj=qwen2.5-vl-7b_7_13_all_data
exps=ingifui
export DEBUG_MODE="true"
export LOG_PATH=/mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/reward_logs/${proj}/${exps}/report_rewards.log
# export WANDB_API_KEY=de97ec5dd1e8a81192f8d2ded930307b819cc717
# export WANDB_PROJECT="verl_r1"
train_data=/mnt/bn/pistis/liutao.0220/verl/data/aitw/infigui_train_data_rl.parquet
test_data=/mnt/bn/pistis/liutao.0220/verl/data/aitw/infigui_train_data_rl_sub.parquet

# ray job submit --address="http://10.213.19.24:8265" \
#     --runtime-env=/mnt/bn/pistis/liutao.0220/verl/verl/trainer/runtime_env.yaml \
#     --no-wait \
#     -- 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$test_data \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/mnt/bn/pistis/liutao.0220/LLaMA-Factory/saves/qwen2.5_vl-7b/full/Qwen2.5-VL-7B-infigui-cold-start \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$proj \
    trainer.experiment_name=$exp \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.total_epochs=5 \
    custom_reward_function.path=/mnt/bn/pistis/liutao.0220/verl/verl/utils/reward_score/aitw.py \
    custom_reward_function.name=compute_score $@
    # actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt="image=25,video=5" $@


vllm serve /mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/checkpoints/qwen2.5-vl-7b_7_11/global_step_400/merged_hf_model --port 18901    --gpu-memory-utilization 0.8  --max-model-len 32768   --tensor-parallel-size 8   --served-model-name "judge"   --trust-remote-code   --disable-log-requests