#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MAX_JOBS=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export VERBOSE=1


# MODEL_PATH=Qwen/Qwen2.5-Math-7B  # replace it with your local file path
MODEL_PATH=/data/langlin/Model/Qwen2.5-Math-rope16384
SAVE_DIR=/data/langlin/Checkpoints/qwen2.5_math_7b_simple_prompt_dapomath_data_baseline

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
# OPENR1DATA=/data/langlin/Data/openr1_parsed.parquet
DAPOMATHDATA=/data/langlin/Data/DAPO/dapo-math-17k.parquet

mkdir -p $SAVE_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    data.train_files=$DAPOMATHDATA \
    data.format_prompt=examples/format_prompt/qwen2.5.jinja \
    data.val_files=$MATHTEST \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2.5_math_7b_simple_prompt_dapomath_data_baseline \
    trainer.save_checkpoint_path=$SAVE_DIR \
    trainer.save_freq=5 \
    data.rollout_batch_size=128 \
    data.val_batch_size=1024 \
    worker.actor.global_batch_size=128 \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    worker.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=8
