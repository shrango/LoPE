#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MAX_JOBS=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export VERBOSE=1


MODEL_PATH=Qwen/Qwen3-1.7B-Base
SAVE_DIR=/data/langlin/Checkpoints/qwen3-1.7b-base_openr1_data_bs128_lorem_test_MATH_wokl

DATAHOME=/data/langlin/Data
# AIME24=$DATAHOME/AIME2024/train-00000-of-00001.parquet
# AIME25=$DATAHOME/AIME2025/train-00000-of-00001.parquet
# DAPO=$DATAHOME/DAPO/dapo-math-17k.parquet
MATHTEST=$DATAHOME/MATH/test.parquet
MATHTRAIN=$DATAHOME/MATH/train.parquet
MATHHARD=$DATAHOME/MATH/train.hard.parquet
# DAPOGT=$DATAHOME/dapo-gt/dapo-filtered.parquet
# SCAFDATA=/engrfs/project/jiaxinh/langlin/Dataset/scaf-grpo/train_parsed.parquet
# FILTEREDDATA=/engrfs/project/jiaxinh/langlin/Dataset/scaf-grpo/DeepSeek-R1-Distill-Qwen-1.5B_parsed.parquet
OPENR1DATA=/data/langlin/Data/openr1_parsed.parquet

mkdir -p $SAVE_DIR
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    data.max_prompt_length=2048 \
    worker.rollout.max_num_batched_tokens=10240 \
    data.train_files=$OPENR1DATA \
    data.format_prompt=examples/format_prompt/math.jinja \
    data.val_files=$MATHTEST \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name="easy_r1_qwen3_1.7b-base" \
    trainer.experiment_name=qwen3-1.7b-base_openr1_data_bs128_lorem_test_MATH_wokl \
    trainer.save_checkpoint_path=$SAVE_DIR \
    worker.rollout.n=8 \
    data.apply_hint=false \
    algorithm.use_kl_loss=false \
    algorithm.disable_kl=true \
    algorithm.kl_coef=0.0 \
    data.apply_icl=true \
    data.icl_examples_path="examples/instruction_with_case_examples.jsonl" \
    data.use_lorem=true \
    data.num_icl_examples=6 \
    data.icl_rollout_n=4 \
    trainer.save_freq=5 \
    trainer.save_limit=-1 \
    data.rollout_batch_size=128 \
    data.val_batch_size=1024 \
    worker.actor.global_batch_size=128 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    worker.rollout.gpu_memory_utilization=0.8
