#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
# 1. 解除内存锁定限制 (修复 NCCL Hang 和 Ray 崩溃的核心)
ulimit -l unlimited

# 2. 增加最大打开文件数 (防止 socket 耗尽)
ulimit -n 65535

# 3. 修复 gRPC 在 SLURM 下的通信不兼容
export GRPC_POLL_STRATEGY=poll

# 4. 显式指定 Ray 使用物理 IP (防止绑定到 127.0.0.1)
export MY_POD_IP=$(hostname -I | awk '{print $1}')
export RAY_ADDRESS='local'
# export NCCL_SOCKET_IFNAME=ens34  # 指定网络接口
# export NCCL_IB_DISABLE=1         # 禁用InfiniBand
# export NCCL_P2P_DISABLE=1       # 禁用点对点通信

# export PYTHONUNBUFFERED=1
# export MAX_JOBS=1
# export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
# export VERBOSE=1


MODEL_PATH=Qwen/Qwen3-4B-Base
SAVE_DIR=/data/langlin/Checkpoints/qwen3-4b-base_openr1_data_gt_suffix0.3_mask

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
    trainer.experiment_name=qwen3-4b-base_openr1_data_gt_suffix0.3_mask \
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
    algorithm.use_safe_policy_loss=true \
    algorithm.suffix_is_ratio_lower_bound=0.3 \
    algorithm.prefix_loss_type="mask"
