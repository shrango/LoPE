#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # replace it with your local file path
# DATAHOME=/ib-scratch/jiaxinh01/project/langlin/Dataset
SAVE_DIR=/home/nvidia/langlin/Checkpoints/ds_distill_qwen1.5b_scaf_grpo_filtered_data_icl_openmathreasoning_gt_suffix0.2_trsft

DATAHOME=/home/nvidia/langlin/Data

AIME24=$DATAHOME/AIME2024/train-00000-of-00001.parquet
AIME25=$DATAHOME/AIME2025/train-00000-of-00001.parquet
DAPO=$DATAHOME/DAPO/dapo-math-17k.parquet
MATHTEST=$DATAHOME/MATH/test.parquet
MATHTRAIN=$DATAHOME/MATH/train.parquet
MATHHARD=$DATAHOME/MATH/train.hard.parquet
DAPOGT=$DATAHOME/dapo-gt/dapo-filtered.parquet
SCAFDATA=/engrfs/project/jiaxinh/langlin/Dataset/scaf-grpo/train_parsed.parquet
FILTEREDDATA=/engrfs/project/jiaxinh/langlin/Dataset/scaf-grpo/DeepSeek-R1-Distill-Qwen-1.5B_parsed.parquet

# fallback到hint，但是不替换输入。意思就是用不带hint的输入+带hint的输出

mkdir -p $SAVE_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=6144 \
    data.train_files=$FILTEREDDATA \
    data.val_files=$MATHTEST \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=ds_distill_qwen1.5b_scaf_grpo_filtered_data_hint_replace_think_distill_luffy-rollout8_test_MATH \
    trainer.save_checkpoint_path=$SAVE_DIR \
    data.apply_hint=false \
    trainer.save_freq=10 \
    data.rollout_batch_size=256 \
    data.val_batch_size=1024 \
    worker.actor.global_batch_size=256 \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.rollout.replace_input=false \
    worker.rollout.correct_only_replace=true \
    data.think_distill=true \
    algorithm.policy_shaping=true \
    algorithm.policy_shaping_gamma=0.1 \
    worker.rollout.n=8
