#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MAX_JOBS=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export VERBOSE=1


MODEL_PATH=Qwen/Qwen3-4B
SAVE_DIR=/data/langlin/Checkpoints/qwen3-4b_openr1_data_gt_suffix0.3_fixed_mask

DATAHOME=/data/langlin/Data
MATHTEST=$DATAHOME/MATH/test.parquet
OPENR1DATA=/data/langlin/Data/openr1_parsed.parquet

mkdir -p $SAVE_DIR
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    data.train_files=$OPENR1DATA \
    data.val_files=$MATHTEST \
    data.format_prompt=examples/format_prompt/math.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3-4b_openr1_data_gt_suffix0.3_fixed_mask \
    trainer.save_checkpoint_path=$SAVE_DIR \
    trainer.save_freq=10 \
    data.rollout_batch_size=256 \
    data.val_batch_size=1024 \
    worker.actor.global_batch_size=256 \
    worker.rollout.gpu_memory_utilization=0.9 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.n=8 \
    data.apply_icl=false \
    trainer.total_epochs=3 \
    worker.actor.optim.lr=1e-6 \
    data.apply_ground_truth=true \
    data.ground_truth_key="solution" \
    trainer.n_gpus_per_node=4 \
    algorithm.safe_policy_region_method="suffix" \
    algorithm.use_safe_policy_loss=true \
    algorithm.entropy_rate_threshold=0.3 \
    algorithm.prefix_loss_type="mask"
