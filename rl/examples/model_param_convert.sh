# source /mnt/bn/pistis/liutao.0220/envs/verl_train/bin/activate

source /mnt/bn/chuhui-v6/lirongjie/venvs/verl_cuda124/bin/activate


src_path=/mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/checkpoints/qwen2.5-vl-32b-cont_iden-data_v9_20250901_qi_pp_cot_more_code_sync_reward_diffsampler_diff_weight_20/global_step_200/actor
tgt_path=/mnt/bn/pistis/liutao.0220/verl/export_model/qa_model/data_v9_20250901_qi_pp_cot_more_code_sync_reward_diffsampler_diff_weight_20/global_step_200/actor

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $src_path \
    --target_dir $tgt_path

cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/config.json  $tgt_path
cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/preprocessor_config.init.json $tgt_path/preprocessor_config.json


src_path=/mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/checkpoints/qwen2.5-vl-32b-cont_iden-data_v9_20250901_qi_pp_cot_more_code_sync_reward_diffsampler_diff_weight_20/global_step_150/actor
tgt_path=/mnt/bn/pistis/liutao.0220/verl/export_model/qa_model/data_v9_20250901_qi_pp_cot_more_code_sync_reward_diffsampler_diff_weight_20/global_step_150/actor

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $src_path \
    --target_dir $tgt_path

cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/config.json  $tgt_path
cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/preprocessor_config.init.json $tgt_path/preprocessor_config.json


# src_path=/mnt/bn/pistis/liutao.0220/verl/examples/grpo_trainer/checkpoints/qa_agent_qwen2.5-vl-32b-identification_train_data_v7_material_pool_20250815_qi_pp_sampledtp_sample_iden/global_step_75/actor
# tgt_path=/mnt/bn/pistis/liutao.0220/verl/MODEL/qa_model/qa_agent_qwen2.5-vl-32b-identification_train_data_v7_material_pool_20250815_qi_pp_sampledtp_sample_iden/global_step_75/actor

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir $src_path \
#     --target_dir $tgt_path

# cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/config.json  $tgt_path
# cp  /mnt/bn/pistis/liutao.0220/verl/MODEL/Qwen/Qwen2.5-VL-32B-Instruct/preprocessor_config.init.json $tgt_path/preprocessor_config.json
