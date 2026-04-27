# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import pdb

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")

def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    advantage_shaping: bool = False,
):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    # advantage_shaping=True 且 meta 中有 all_rollout_scores_per_uid 时：对触发 ICL fallback 的 uid
    # 用 base+ICL 完整 scores 算 GRPO 的 mean/std；否则一律用当前 batch 内各 uid 的 scores（标准 GRPO）。
    if advantage_shaping:
        all_rollout_scores_per_uid = data.meta_info.get("all_rollout_scores_per_uid", None)
        if all_rollout_scores_per_uid:
            adv_inputs["all_rollout_scores_per_uid"] = all_rollout_scores_per_uid

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _non_pad_mask_for_response_position_slice(
    response_mask_row: torch.Tensor, position_response_slice: torch.Tensor
) -> torch.Tensor:
    """与模型无关：用 response 的 token 级 mask 标记「真实 token」，对齐到 position 切片形状。

    旧逻辑用固定数值（如 Qwen mRoPE 里 pad 位的 position 填 151643）判断 padding，换模型即失效。
    ``response_mask`` 为 1 的位置才做 ``delta`` 平移，padding 位保持原 position 填充值不变。
    """
    device = position_response_slice.device
    L_pos = position_response_slice.shape[-1]
    row = (response_mask_row > 0).flatten().bool().to(device)
    if row.numel() != L_pos:
        if row.numel() >= L_pos:
            row = row[-L_pos:]
        else:
            pad = L_pos - row.numel()
            row = torch.cat([torch.zeros(pad, dtype=torch.bool, device=device), row])
    if position_response_slice.dim() == 2:
        return row.unsqueeze(0)
    # mRoPE 等: (1, n_dims, seq) 与最后一维 seq 对齐，各维共享同一 valid 模式
    return row.view(*([1] * (position_response_slice.dim() - 1)), L_pos).expand_as(position_response_slice)


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.on_policy_interval_by_pos = None

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            n = self.config.worker.rollout.n
            num_icl_examples = self.config.data.num_icl_examples
            icl_rollout_n = self.config.data.icl_rollout_n
            n_icl_slots = num_icl_examples * icl_rollout_n
            general_exploration = getattr(self.config.data, "general_exploration", False)
            n_main = n - n_icl_slots if general_exploration else None
            if general_exploration:
                if gen_batch.meta_info is None:
                    gen_batch.meta_info = {}
                gen_batch.meta_info["n"] = n_main

            # generate a batch（general_exploration 时只生成 n_main 条 response / prompt）
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            base_input_ids = gen_batch_output.batch["prompts"][:1]
            base_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in base_input_ids]
            base_output_ids = gen_batch_output.batch["responses"][:1]
            base_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in base_output_ids]
            print("=" * 30 + "Base input texts" + "=" * 30)
            print(f"Base input texts: {base_input_texts}")
            print("=" * 30 + "Base output texts" + "=" * 30)
            print(f"Base output texts: {base_output_texts}")
            print("=" * 30 + "=" * 30)

            if self.config.algorithm.adv_estimator == "remax":
                raise ValueError("Logic not implemented")
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            if general_exploration:
                merged = new_batch.repeat(repeat_times=n_main, interleave=True).union(gen_batch_output)
                num_prompts_ge = len(merged) // n_main
                chunk_list = []
                for k in range(num_prompts_ge):
                    main_part = merged.slice_select(start=k * n_main, end=(k + 1) * n_main)
                    one_row = merged.index_select([k * n_main])
                    ph = one_row.repeat(repeat_times=n_icl_slots, interleave=True)
                    chunk_list.append(DataProto.concat([main_part, ph]))
                new_batch = DataProto.concat(chunk_list)
            else:
                new_batch = new_batch.repeat(repeat_times=n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

            # 初始化 real_rollout 标记：True 表示模型自己生成的，False 表示被替换的（ICL/ground_truth等）
            # 用于 compute_safe_policy_loss 中区分是否应用 entropy_rate_threshold mask
            # 注意：这个初始化需要在 fallback 流程之前，确保所有样本都有 real_rollout 标记

            if "real_rollout" not in new_batch.non_tensor_batch:
                # general_exploration 的占位槽会在 ICL fallback 替换时由 _icl_fallback_stage 置 False
                new_batch.non_tensor_batch["real_rollout"] = np.ones(len(new_batch), dtype=bool)
            
            # reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            # ================== 统一的 Fallback 框架 ==================
            self.apply_icl = getattr(self.config.data, "apply_icl", False)
            self.use_lorem = getattr(self.config.data, "use_lorem", False)
            self.use_fake_sentence = getattr(self.config.data, "use_fake_sentence", False)
            self.use_random_token = getattr(self.config.data, "use_random_token", False)
            self.use_random_ascii = getattr(self.config.data, "use_random_ascii", False)
            self.use_markovify = getattr(self.config.data, "use_markovify", False)
            self.use_random_natural_language = getattr(self.config.data, "use_random_natural_language", False)
            self.use_random_la_word = getattr(self.config.data, "use_random_la_word", False)
            self.naive_resample = getattr(self.config.data, "naive_resample", False)
            self.multi_style_templates = getattr(self.config.data, "multi_style_templates", False)
            self.lorem_in_middle = getattr(self.config.data, "lorem_in_middle", False)
            self.has_icl_payload = (
                self.apply_icl
                or self.use_lorem
                or self.use_fake_sentence
                or self.use_random_token
                or self.use_random_ascii
                or self.use_markovify
                or self.use_random_natural_language
                or self.use_random_la_word
                or self.naive_resample
                or self.multi_style_templates
                or self.lorem_in_middle
            )

            # 检查是否有 ground_truth_text（用于 ground truth 替换）
            has_ground_truth_text = "ground_truth_text" in new_batch.non_tensor_batch

            # 如果有任何 fallback stage，或启用了 ICL / 纯随机 system 前缀 / naive_resample（dataset 会产出 icl_*），或有 ground_truth_text，执行统一的 fallback 流程
            if self.has_icl_payload or has_ground_truth_text:
                
                def rename_keys_inplace(dp, mapping, *, drop_old=True):
                    """
                    就地重命名 DataProto 中的键，保持 dp.batch 的原始类型（如 TensorDict），
                    避免构造新的普通 dict 破坏 .batch_size 等属性。

                    mapping: {old_key: new_key}
                    drop_old: True 表示改名后删除旧键（默认建议 True）
                    """
                    # 1) batch（通常是 TensorDict，支持 in / pop / __setitem__）
                    if hasattr(dp, "batch") and dp.batch is not None:
                        for old_k, new_k in list(mapping.items()):
                            if old_k in dp.batch:
                                if (new_k in dp.batch) and dp.batch[new_k] is not dp.batch[old_k]:
                                    raise ValueError(f"rename '{old_k}' -> '{new_k}' 会覆盖已存在且不同的键")
                                val = dp.batch.pop(old_k)
                                dp.batch[new_k] = val
                            # 如果 old_k 不在，就忽略 —— 安全 rename

                    # 2) non_tensor_batch（普通 dict / ndarray 列）
                    if hasattr(dp, "non_tensor_batch") and dp.non_tensor_batch is not None:
                        for old_k, new_k in list(mapping.items()):
                            if old_k in dp.non_tensor_batch:
                                if (new_k in dp.non_tensor_batch) and dp.non_tensor_batch[new_k] is not dp.non_tensor_batch[old_k]:
                                    raise ValueError(f"rename '{old_k}' -> '{new_k}' 会覆盖 non_tensor_batch 里已存在且不同的键")
                                val = dp.non_tensor_batch.pop(old_k)
                                dp.non_tensor_batch[new_k] = val

                    # 3) meta_info（样本无关 dict）
                    if hasattr(dp, "meta_info") and dp.meta_info is not None:
                        for old_k, new_k in list(mapping.items()):
                            if old_k in dp.meta_info:
                                # meta_info 通常不该发生重名冲突；如发生，可按需处理一致性
                                val = dp.meta_info.pop(old_k)
                                dp.meta_info[new_k] = val

                    return dp  # 注意：同一个对象，已就地修改

                def _overwrite_with_original_prompt(target_batch, original_indices, repeat_factor=1):
                    """用 new_batch 中的原始 prompt 覆盖 target_batch 的 prompt 数据。
                    
                    Args:
                        target_batch: 要覆盖的 DataProto（stage_prompt_batch 或 combined_icl_batch）
                        original_indices: 在 new_batch 中的样本索引（list of int）
                        repeat_factor: 每个 prompt 重复的次数（ICL 时为 num_icl_examples）
                    """
                    original_prompts = new_batch.batch["prompts"][original_indices].clone()
                    prompt_len = original_prompts.shape[-1]
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

                    original_attn = new_batch.batch["attention_mask"][original_indices, :prompt_len].clone()
                    if new_batch.batch["position_ids"].dim() == 3:
                        original_pos = new_batch.batch["position_ids"][original_indices, :, :prompt_len].clone()
                    else:
                        original_pos = new_batch.batch["position_ids"][original_indices, :prompt_len].clone()

                    if repeat_factor > 1:
                        original_prompts = original_prompts.repeat_interleave(repeat_factor, dim=0)
                        original_attn = original_attn.repeat_interleave(repeat_factor, dim=0)
                        original_pos = original_pos.repeat_interleave(repeat_factor, dim=0)

                    target_batch.batch["input_ids"] = original_prompts
                    target_batch.batch["attention_mask"] = original_attn
                    target_batch.batch["position_ids"] = original_pos

                    raw_prompt_ids_list = []
                    for i in range(len(original_prompts)):
                        ids = original_prompts[i].cpu().tolist()
                        ids = [x for x in ids if x != pad_token_id]
                        raw_prompt_ids_list.append(ids)
                    target_batch.non_tensor_batch["raw_prompt_ids"] = np.array(raw_prompt_ids_list, dtype=object)

                n = self.config.worker.rollout.n
                # 将n保存到meta_info中，供后续使用
                new_batch.meta_info["n"] = n

                # ================== Step 1: 基础 prompt 的对错判断 ==================
                # 此时 new_batch 已经是：repeat(repeat_times=n) + union(gen_batch_output) 之后的版本
                reward_tensor, reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(new_batch)
                )
                overall = np.asarray(reward_metrics["overall"], dtype=np.float32)

                num_samples = len(new_batch)
                assert overall.size == num_samples, (
                    f"overall length {overall.size} != len(new_batch) {num_samples}"
                )
                assert num_samples % n == 0, "overall length must be divisible by n"

                num_prompts = num_samples // n
                overall = overall.reshape(num_prompts, n)      # [num_prompts, n]

                # 仅 advantage_shaping 时需要保留 base 各条 score，与 ICL scores 拼成完整组供 GRPO 归一化
                base_overall_per_prompt = (
                    overall.copy() if self.config.algorithm.advantage_shaping else None
                )  # [num_prompts, n] or None

                # 默认 reward > 0 视为"答对"
                prompt_any_correct = (overall > 0).any(axis=1)   # [num_prompts]
                prompt_all_wrong   = ~prompt_any_correct         # True = 这一组 n 条全错

                num_all_wrong_base = int(prompt_all_wrong.sum())

                # 统计信息
                fallback_stats = new_batch.meta_info.get("fallback_stats", {})
                fallback_stats.update({
                    "num_prompts": int(num_prompts),
                    "num_all_wrong_base": num_all_wrong_base,
                })
                new_batch.meta_info["fallback_stats"] = fallback_stats

                # 只有在启用filter_on_policy_token时才初始化rollout_type和prompt_origin_indices
                filter_on_policy_token = getattr(self.config.algorithm, "filter_on_policy_token", False)
                print(f"filter_on_policy_token: {filter_on_policy_token}")
                filter_by_suffix_is_ratio = getattr(self.config.algorithm, "filter_by_suffix_is_ratio", False)
                print(f"filter_by_suffix_is_ratio: {filter_by_suffix_is_ratio}")
                if filter_on_policy_token or filter_by_suffix_is_ratio:
                    # 初始化rollout_type：记录哪些sample是origin（没有被替换），哪些是reference（被替换了）
                    if "rollout_type" not in new_batch.non_tensor_batch:
                        new_batch.non_tensor_batch["rollout_type"] = np.array(
                            ["origin"] * len(new_batch), dtype=object
                        )
                    # 初始化prompt_origin_indices：记录每个prompt对应的origin samples索引
                    if "prompt_origin_indices" not in new_batch.meta_info:
                        new_batch.meta_info["prompt_origin_indices"] = {}

                # ================== ICL Fallback Stage ==================
                def _icl_fallback_stage(
                    unresolved_mask: np.ndarray,   # [num_prompts]，True 表示当前仍然"全错"的 prompt
                ):
                    """
                    ICL Fallback Stage:
                    - 使用 num_icl_examples 个不同的 ICL prompt，每个生成 1 个 response
                    - 如果正确数量 >= n (rollout.n)，随机选择 n-1 个替换
                    - 其余逻辑与普通 fallback stage 类似
                    
                    注意区分两个变量：
                    - n: rollout.n，每个 prompt 在原始 batch 中的 response 数量（可能是 5, 8 等）
                    - num_icl_examples: ICL examples 的数量（固定为 5，来自 icl_examples_path 文件）
                    """
                    print("----------------------------ICL Fallback Stage----------------------------")
                    stage_name = "icl"
                    # num_icl_examples: ICL examples 的数量（与 dataset.py 中的配置一致）
                    # 注意：这与 rollout.n（每个 prompt 的 response 数量）是不同的概念
                    num_icl_examples = self.config.data.num_icl_examples
                    
                    # 1) 当前未解决的 prompt 索引（general_exploration：对所有 prompt 跑 ICL 以替换占位槽）
                    if general_exploration:
                        unsolved_prompt_indices = np.arange(num_prompts)
                        M = num_prompts
                        if M == 0:
                            return unresolved_mask
                    else:
                        unsolved_prompt_indices = np.nonzero(unresolved_mask)[0]  # [M]
                        M = len(unsolved_prompt_indices)
                        if M == 0:
                            return unresolved_mask  # 没有需要处理的 prompt 了

                    print(f"----------------------------ICL Stage ({M} prompts, general_exploration={general_exploration})----------------------------")

                    # 2) world_size & padding，让 (prompt 数 * num_icl_examples) 对 world_size 可整除
                    # 因为 combined_icl_batch 的大小是 M_eff * num_icl_examples
                    # 注意：必须使用 actor_rollout_ref_wg.world_size，因为 generate_sequences 内部用它来分割数据
                    world_size = self.actor_rollout_ref_wg.world_size

                    prompt_indices = unsolved_prompt_indices.copy()
                    if world_size > 1:
                        # 需要确保 M_eff * num_icl_examples % world_size == 0
                        # 即 M_eff % (world_size / gcd(world_size, num_icl_examples)) == 0
                        from math import gcd
                        divisor = world_size // gcd(world_size, num_icl_examples)
                        r = M % divisor
                        if r != 0:
                            pad = divisor - r
                            # 使用 np.tile 确保有足够的元素来 padding，即使 M < pad
                            repeat_times = (pad // M) + 1
                            extra = np.tile(prompt_indices, repeat_times)[:pad]
                            prompt_indices = np.concatenate([prompt_indices, extra], axis=0)

                    M_eff = len(prompt_indices)

                    # 3) 收集所有 ICL prompts 并合并成一个 batch，一次性生成
                    # 先收集所有 ICL prompt batches
                    icl_prompt_batches = []
                    # first_row_indices: 使用 n (rollout.n) 计算每个 prompt 在原始 batch 中的第一条样本索引
                    first_row_indices = (prompt_indices * n).tolist()
                    
                    # 共享的 key 只需要获取一次，不能每次 pop（会导致后续 pop 失败）
                    shared_non_tensor_keys = ["multi_modal_data"]
                    shared_meta_info_keys = ["min_pixels", "max_pixels", "video_fps"]
                    
                    for icl_idx in range(num_icl_examples):
                        # 从 new_batch 中 pop 出第 icl_idx 个 ICL prompt 的数据
                        icl_pop_batch_keys = [
                            f"icl_{icl_idx}_input_ids",
                            f"icl_{icl_idx}_attention_mask",
                            f"icl_{icl_idx}_position_ids",
                        ]
                        # 非共享的 key 使用 pop，共享的 key 只在第一次 pop
                        icl_pop_non_tensor_batch_keys = [f"raw_icl_{icl_idx}_prompt_ids"]
                        if icl_idx == 0:
                            icl_pop_non_tensor_batch_keys.extend(shared_non_tensor_keys)
                            icl_pop_meta_info_keys = shared_meta_info_keys
                        else:
                            icl_pop_meta_info_keys = []
                        
                        icl_rename_mapping = {
                            f"icl_{icl_idx}_input_ids": "input_ids",
                            f"icl_{icl_idx}_attention_mask": "attention_mask",
                            f"icl_{icl_idx}_position_ids": "position_ids",
                            f"raw_icl_{icl_idx}_prompt_ids": "raw_prompt_ids",
                        }
                        
                        icl_source_all = new_batch.pop(
                            batch_keys=icl_pop_batch_keys,
                            non_tensor_batch_keys=icl_pop_non_tensor_batch_keys,
                            meta_info_keys=icl_pop_meta_info_keys,
                        )
                        
                        # 构造本 ICL prompt 的 batch：只取每个 prompt 的第一条样本
                        icl_prompt_batch = icl_source_all.index_select(first_row_indices)
                        icl_prompt_batch.non_tensor_batch["ground_truth"] = new_batch.non_tensor_batch["ground_truth"][first_row_indices]
                        
                        # 添加 icl_idx 标记，用于后续拆分
                        icl_prompt_batch.non_tensor_batch["icl_idx"] = np.full(M_eff, icl_idx, dtype=np.int32)
                        
                        # 重命名
                        rename_keys_inplace(icl_prompt_batch, icl_rename_mapping, drop_old=True)
                        
                        icl_prompt_batches.append(icl_prompt_batch)
                    
                    # 合并所有 ICL prompt batches
                    # 原始顺序是按 ICL 归类：q1_icl0, q2_icl0, ..., qM_icl0, q1_icl1, ...
                    combined_icl_batch_by_icl = DataProto.concat(icl_prompt_batches)
                    
                    # 重排为按 question 归类：q1_icl0, q1_icl1, ..., q1_icl4, q2_icl0, ...
                    # 生成重排索引
                    reorder_indices = []
                    for q_idx in range(M_eff):
                        for icl_idx in range(num_icl_examples):
                            # 原索引：icl_idx * M_eff + q_idx
                            reorder_indices.append(icl_idx * M_eff + q_idx)
                    combined_icl_batch = combined_icl_batch_by_icl.index_select(reorder_indices)
                    
                    # 如果 fallback_with_original_prompt，用原始 prompt 覆盖所有 ICL prompt
                    if self.config.algorithm.fallback_with_original_prompt:
                        _overwrite_with_original_prompt(combined_icl_batch, first_row_indices, repeat_factor=num_icl_examples)

                    # 设置每个 ICL prompt 生成的 response 数量
                    icl_rollout_n = self.config.data.icl_rollout_n
                    combined_icl_batch.meta_info["n"] = icl_rollout_n

                    # vLLM：generate_sequences 内会对 SamplingParams 应用 prompts.meta_info（见 vllm_rollout_spmd.update_sampling_params）
                    resample_temperature = getattr(self.config.data, "resample_temperature", None)
                    if resample_temperature is not None:
                        combined_icl_batch.meta_info["temperature"] = float(resample_temperature)

                    # 一次性生成所有 ICL responses
                    combined_icl_gen_output = self.actor_rollout_ref_wg.generate_sequences(combined_icl_batch)
                    
                    # 拆分结果：每个 ICL method 有 M_eff * icl_rollout_n 个样本
                    # generate_sequences 的输出按 question 归类，每个 prompt 有 icl_rollout_n 个 response：
                    #   q0_icl0_r0..r(rn-1), q0_icl1_r0..r(rn-1), ..., q1_icl0_r0..r(rn-1), ...
                    all_icl_gen_outputs = []
                    for icl_idx in range(num_icl_examples):
                        icl_indices = []
                        for q_idx in range(M_eff):
                            base = (q_idx * num_icl_examples + icl_idx) * icl_rollout_n
                            icl_indices.extend(range(base, base + icl_rollout_n))
                        icl_gen_output = combined_icl_gen_output.index_select(icl_indices)
                        all_icl_gen_outputs.append(icl_gen_output)
                        
                        # 打印每种 ICL prompt 的样例（取第一个 question 的第一个 response）
                        icl_input_ids = icl_gen_output.batch["prompts"][:1]
                        if len(icl_input_ids) > 0:
                            icl_input_texts = self.tokenizer.decode(icl_input_ids[0], skip_special_tokens=True)
                            print("=" * 30 + f" icl_case:{icl_idx + 1} input " + "=" * 30)
                            print(f"icl_case:{icl_idx + 1} input: {icl_input_texts}")
                            print("=" * 30 + "=" * 30)
                        icl_output_ids = icl_gen_output.batch["responses"][:1]
                        icl_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in icl_output_ids]
                        print("=" * 30 + f" icl_case:{icl_idx + 1} output " + "=" * 30)
                        print(f"icl_case:{icl_idx + 1} output: {icl_output_texts}")
                        print("=" * 30 + "=" * 30)
                    
                    # 4) 计算每个 ICL response 的正确性
                    # all_icl_gen_outputs: list of DataProto, each with M_eff * icl_rollout_n samples
                    # overall_icl shape: [M_eff, num_icl_examples, icl_rollout_n]
                    overall_icl = np.zeros((M_eff, num_icl_examples, icl_rollout_n), dtype=np.float32)
                    
                    for icl_idx, icl_gen_output in enumerate(all_icl_gen_outputs):
                        # 需要添加 ground_truth 来计算 reward
                        # icl_gen_output 有 M_eff * icl_rollout_n 个样本，每 icl_rollout_n 个属于同一个 prompt
                        first_row_indices = (prompt_indices * n).tolist()
                        gt_per_prompt = new_batch.non_tensor_batch["ground_truth"][first_row_indices]
                        icl_gen_output.non_tensor_batch["ground_truth"] = np.repeat(gt_per_prompt, icl_rollout_n)
                        
                        reward_tensor_icl, reward_metrics_icl = ray.get(
                            self.reward_fn.compute_reward.remote(icl_gen_output)
                        )
                        rewards = np.asarray(reward_metrics_icl["overall"], dtype=np.float32).reshape(M_eff, icl_rollout_n)
                        overall_icl[:, icl_idx, :] = rewards

                    # 4.5) advantage_shaping：记录 base + ICL 完整 scores，供 GRPO 对该 uid 用更大一组算 mean/std
                    if self.config.algorithm.advantage_shaping and base_overall_per_prompt is not None:
                        uids = new_batch.non_tensor_batch["uid"]
                        per_uid_scores = new_batch.meta_info.setdefault(
                            "all_rollout_scores_per_uid", {}
                        )
                        base_cols = n_main if general_exploration else n
                        for i in range(M):
                            p_idx = int(unsolved_prompt_indices[i])
                            uid_p = uids[p_idx * n]
                            base_scores = base_overall_per_prompt[p_idx, :base_cols].astype(np.float32)
                            icl_scores = overall_icl[i, :, :].reshape(-1).astype(np.float32)
                            full_scores = np.concatenate([base_scores, icl_scores], axis=0)
                            per_uid_scores[uid_p] = full_scores

                    # 5) 对于每个 prompt，判断有多少个正确的 ICL response，并进行替换
                    device = None
                    if new_batch.batch is not None and len(new_batch.batch) > 0:
                        device = next(iter(new_batch.batch.values())).device
                    
                    processed_prompts = set()
                    replace_indices_list = []
                    replace_icl_info_list = []  # (icl_idx, local_idx_in_icl_output)
                    original_input_dicts = []
                    n_icl_slots_local = num_icl_examples * icl_rollout_n

                    if general_exploration:
                        # 占位槽位与 ICL 输出一一对应：global p*n + n_main + t 对应 all_icl_gen_outputs[icl_idx][i*rn+r]
                        for i, p_idx in enumerate(prompt_indices):
                            if p_idx in processed_prompts:
                                continue
                            processed_prompts.add(p_idx)
                            for t in range(n_icl_slots_local):
                                icl_idx = t // icl_rollout_n
                                r_idx = t % icl_rollout_n
                                global_idx = int(p_idx * n + n_main + t)
                                local_idx_in_icl = i * icl_rollout_n + r_idx
                                replace_indices_list.append(global_idx)
                                replace_icl_info_list.append((int(icl_idx), int(local_idx_in_icl)))
                                original_input_dict = {}
                                keys_to_save = {"prompts", "input_ids", "attention_mask", "position_ids"}
                                if new_batch.batch is not None:
                                    for key in keys_to_save:
                                        if key in new_batch.batch:
                                            original_input_dict[key] = new_batch.batch[key][global_idx : global_idx + 1].clone()
                                if new_batch.non_tensor_batch is not None and "raw_prompt_ids" in new_batch.non_tensor_batch:
                                    arr = new_batch.non_tensor_batch["raw_prompt_ids"]
                                    original_input_dict["raw_prompt_ids"] = np.asarray(arr)[global_idx : global_idx + 1].copy()
                                original_input_dicts.append(original_input_dict)
                    else:
                        for i, p_idx in enumerate(prompt_indices):
                            if p_idx in processed_prompts:
                                continue
                            processed_prompts.add(p_idx)

                            # overall_icl[i] shape: [num_icl_examples, icl_rollout_n]
                            # 收集所有正确的 (icl_method, rollout_r) 对
                            correct_mask = (overall_icl[i, :, :] > 0)  # [num_icl_examples, icl_rollout_n]
                            correct_pairs = list(zip(*np.nonzero(correct_mask)))  # list of (icl_idx, r_idx)
                            num_correct = len(correct_pairs)

                            if num_correct == 0:
                                continue

                            # 确定要替换的数量
                            # n 是 rollout.n（每个 prompt 的原始 response 数量）
                            # 如果正确数量 >= n，最多替换 n-1 个（保留至少 1 个原始 response）
                            if num_correct >= n:
                                num_to_replace = n - 1
                            else:
                                num_to_replace = num_correct

                            # 如果正确数量 >= n，随机选择；否则选择所有正确的
                            if num_correct >= n:
                                selected_idx = np.random.choice(len(correct_pairs), size=num_to_replace, replace=False)
                                selected_pairs = [correct_pairs[j] for j in selected_idx]
                            else:
                                selected_pairs = correct_pairs[:num_to_replace]

                            # 找出要替换的样本索引（全局）
                            base_global = int(p_idx * n)

                            for k, (icl_idx, r_idx) in enumerate(selected_pairs):
                                global_idx = base_global + k  # 替换第 k 个 response
                                local_idx_in_icl = i * icl_rollout_n + r_idx

                                replace_indices_list.append(global_idx)
                                replace_icl_info_list.append((int(icl_idx), int(local_idx_in_icl)))

                                # 保存 original 的 rollout 信息
                                original_input_dict = {}
                                keys_to_save = {"prompts", "input_ids", "attention_mask", "position_ids"}

                                if new_batch.batch is not None:
                                    for key in keys_to_save:
                                        if key in new_batch.batch:
                                            original_input_dict[key] = new_batch.batch[key][global_idx:global_idx+1].clone()

                                if new_batch.non_tensor_batch is not None and "raw_prompt_ids" in new_batch.non_tensor_batch:
                                    arr = new_batch.non_tensor_batch["raw_prompt_ids"]
                                    original_input_dict["raw_prompt_ids"] = np.asarray(arr)[global_idx:global_idx+1].copy()

                                original_input_dicts.append(original_input_dict)
                    
                    # 6) 执行替换
                    if len(replace_indices_list) > 0:
                        for replace_idx, (global_idx, (icl_idx, local_idx_in_icl)) in enumerate(zip(replace_indices_list, replace_icl_info_list)):
                            global_idx_t = torch.as_tensor([global_idx], device=device)
                            local_idx_t = torch.as_tensor([local_idx_in_icl], device=device)
                            
                            icl_gen_output = all_icl_gen_outputs[icl_idx]
                            
                            # 替换 batch 中的字段
                            if new_batch.batch is not None and icl_gen_output.batch is not None:
                                common_keys = set(new_batch.batch.keys()).intersection(
                                    set(icl_gen_output.batch.keys())
                                )
                                for key in common_keys:
                                    new_batch.batch[key][global_idx_t] = icl_gen_output.batch[key][local_idx_t]
                            
                            # 替换 non_tensor_batch 中的字段
                            if new_batch.non_tensor_batch is not None and icl_gen_output.non_tensor_batch is not None:
                                common_nt_keys = set(new_batch.non_tensor_batch.keys()).intersection(
                                    set(icl_gen_output.non_tensor_batch.keys())
                                )
                                for key in common_nt_keys:
                                    arr_tgt = new_batch.non_tensor_batch[key]
                                    arr_src = icl_gen_output.non_tensor_batch[key]
                                    arr_tgt[global_idx] = np.asarray(arr_src)[local_idx_in_icl]
                        
                        # 7) 保存被替换样本的 original_input（fallback_with_original_prompt 时跳过）
                        replace_indices_np = np.asarray(replace_indices_list, dtype=np.int64)
                        if not self.config.algorithm.fallback_with_original_prompt:
                            if "original_inputs" not in new_batch.meta_info:
                                new_batch.meta_info["original_inputs"] = []
                            
                            for idx, original_input_dict in zip(replace_indices_list, original_input_dicts):
                                if original_input_dict:
                                    new_batch.meta_info["original_inputs"].append({
                                        "original_input": original_input_dict,
                                        "indices": np.asarray([idx], dtype=np.int64),
                                        "stage_name": stage_name,
                                    })
                        
                        # 8) 更新 rollout_type（如果启用）
                        if (filter_on_policy_token or filter_by_suffix_is_ratio):
                            if "rollout_type" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["rollout_type"] = np.array(
                                    ["origin"] * len(new_batch), dtype=object
                                )
                            new_batch.non_tensor_batch["rollout_type"][replace_indices_np] = "reference"
                        
                        # 8.5) 标记 real_rollout 为 False（被替换的样本不是模型自己生成的）
                        if "real_rollout" in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch["real_rollout"][replace_indices_np] = False
                    
                    # 9) 在整体 new_batch 上重新计算 reward，得到新的 unresolved_mask
                    reward_tensor_icl_final, reward_metrics_icl_final = ray.get(
                        self.reward_fn.compute_reward.remote(new_batch)
                    )
                    overall_icl_final = np.asarray(reward_metrics_icl_final["overall"], dtype=np.float32)
                    
                    # reshape 使用 n (rollout.n)，因为 new_batch 中每个 prompt 有 n 个 response
                    overall_icl_final = overall_icl_final.reshape(num_prompts, n)
                    prompt_any_correct_icl = (overall_icl_final > 0).any(axis=1)
                    new_unresolved_mask = ~prompt_any_correct_icl
                    
                    # 10) 更新每个prompt对应的origin samples索引
                    if filter_on_policy_token:
                        for p_idx in range(num_prompts):
                            base = int(p_idx * n)
                            origin_indices_for_prompt = []
                            for k in range(n):
                                sample_idx = base + k
                                if new_batch.non_tensor_batch["rollout_type"][sample_idx] == "origin":
                                    origin_indices_for_prompt.append(int(sample_idx))
                            if len(origin_indices_for_prompt) > 0:
                                new_batch.meta_info["prompt_origin_indices"][p_idx] = origin_indices_for_prompt
                            elif p_idx in new_batch.meta_info["prompt_origin_indices"]:
                                del new_batch.meta_info["prompt_origin_indices"][p_idx]
                    
                    # 11) 统计信息
                    fallback_stats = new_batch.meta_info.get("fallback_stats", {})
                    fallback_stats[f"num_all_wrong_after_{stage_name}"] = int(new_unresolved_mask.sum())
                    
                    # 计算 ICL fallback 的正确率
                    icl_correct_mask = (overall_icl[:M, :, :] > 0)  # [M, num_icl_examples, icl_rollout_n]
                    num_icl_fallback_samples = M * num_icl_examples * icl_rollout_n
                    num_icl_correct_samples = int(icl_correct_mask.sum())
                    icl_sample_correct_ratio = num_icl_correct_samples / num_icl_fallback_samples if num_icl_fallback_samples > 0 else 0.0
                    
                    prompt_any_correct_icl_fallback = icl_correct_mask.any(axis=(1, 2))  # [M]
                    num_icl_correct_prompts = int(prompt_any_correct_icl_fallback.sum())
                    icl_prompt_correct_ratio = num_icl_correct_prompts / M if M > 0 else 0.0
                    
                    fallback_stats[f"{stage_name}_num_fallback_prompts"] = M
                    fallback_stats[f"{stage_name}_num_fallback_samples"] = num_icl_fallback_samples
                    fallback_stats[f"{stage_name}_num_correct_samples"] = num_icl_correct_samples
                    fallback_stats[f"{stage_name}_num_correct_prompts"] = num_icl_correct_prompts
                    fallback_stats[f"{stage_name}_sample_correct_ratio"] = icl_sample_correct_ratio
                    fallback_stats[f"{stage_name}_prompt_correct_ratio"] = icl_prompt_correct_ratio
                    
                    new_batch.meta_info["fallback_stats"] = fallback_stats
                    
                    return new_unresolved_mask

                # ================== Ground Truth Replace Stage ==================
                def _ground_truth_replace(
                    unresolved_mask: np.ndarray,   # [num_prompts]，True 表示当前仍然"全错"的 prompt
                    replace_all: bool = False,     # 如果 True，替换所有 prompt 的 1 个 response；否则只替换 all_wrong 的
                ):
                    """
                    Ground Truth Replace Stage:
                    - 使用 ground_truth_text 替换每个 prompt 的 1 个 response
                    - 如果 replace_all=True：替换所有 prompt 的第 0 个 response
                    - 如果 replace_all=False：只替换 all_wrong 的 prompt 的第 0 个 response
                    """
                    stage_name = "ground_truth"
                    
                    # 检查是否存在 ground_truth_text
                    if "ground_truth_text" not in new_batch.non_tensor_batch:
                        print(f"----------------------------Skip Ground Truth Replace (no ground_truth_text)----------------------------")
                        return unresolved_mask
                    
                    # 1) 确定要处理的 prompt 索引
                    if replace_all:
                        # 替换所有 prompt
                        prompt_indices_to_process = np.arange(num_prompts)
                    else:
                        # 只替换 all_wrong 的 prompt
                        prompt_indices_to_process = np.nonzero(unresolved_mask)[0]
                    
                    M = len(prompt_indices_to_process)
                    if M == 0:
                        print(f"----------------------------Skip Ground Truth Replace (no prompts to process)----------------------------")
                        return unresolved_mask
                    
                    print(f"----------------------------Ground Truth Replace Stage ({M} prompts, replace_all={replace_all})----------------------------")
                    
                    device = None
                    if new_batch.batch is not None and len(new_batch.batch) > 0:
                        device = next(iter(new_batch.batch.values())).device
                    
                    # 2) 获取原始 response 的长度（用于 padding/truncation）
                    # 从 new_batch.batch["responses"] 获取 response 的长度
                    original_response_len = new_batch.batch["responses"].shape[-1]
                    
                    replace_indices_list = []
                    original_input_dicts = []
                    
                    for p_idx in prompt_indices_to_process:
                        # 每个 prompt 替换第 0 个 response
                        global_idx = int(p_idx * n)
                        
                        # 获取 ground_truth_text，直接作为 response 内容使用
                        response_content = new_batch.non_tensor_batch["ground_truth_text"][global_idx]
                        
                        # Tokenize response content
                        response_ids = self.tokenizer.encode(response_content, add_special_tokens=False)
                        
                        # 创建 response tensor 并处理 padding/truncation
                        response_tensor = torch.tensor(response_ids, device=device, dtype=new_batch.batch["responses"].dtype)
                        
                        if len(response_tensor) < original_response_len:
                            # Padding
                            pad_len = original_response_len - len(response_tensor)
                            pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                            response_tensor = torch.cat([response_tensor, torch.full((pad_len,), pad_value, device=device, dtype=response_tensor.dtype)])
                        elif len(response_tensor) > original_response_len:
                            # Truncation
                            response_tensor = response_tensor[:original_response_len]
                        
                        # 保存 original 的 rollout 信息
                        original_input_dict = {}
                        keys_to_save = {"prompts", "input_ids", "attention_mask", "position_ids", "responses", "response_mask"}
                        
                        if new_batch.batch is not None:
                            for key in keys_to_save:
                                if key in new_batch.batch:
                                    original_input_dict[key] = new_batch.batch[key][global_idx:global_idx+1].clone()
                        
                        if new_batch.non_tensor_batch is not None and "raw_prompt_ids" in new_batch.non_tensor_batch:
                            arr = new_batch.non_tensor_batch["raw_prompt_ids"]
                            original_input_dict["raw_prompt_ids"] = np.asarray(arr)[global_idx:global_idx+1].copy()
                        
                        original_input_dicts.append(original_input_dict)
                        replace_indices_list.append(global_idx)
                        
                        # 替换 response
                        new_batch.batch["responses"][global_idx] = response_tensor
                        
                        # 更新 response_mask
                        if "response_mask" in new_batch.batch:
                            actual_len = min(len(response_ids), original_response_len)
                            new_mask = torch.cat([
                                torch.ones(actual_len, device=device, dtype=new_batch.batch["response_mask"].dtype),
                                torch.zeros(original_response_len - actual_len, device=device, dtype=new_batch.batch["response_mask"].dtype)
                            ])
                            new_batch.batch["response_mask"][global_idx] = new_mask
                        
                        # 更新 input_ids（prompts + responses）
                        if "input_ids" in new_batch.batch and "prompts" in new_batch.batch:
                            prompt_tensor = new_batch.batch["prompts"][global_idx]
                            new_input_ids = torch.cat([prompt_tensor, response_tensor])
                            # 确保长度一致
                            original_input_len = new_batch.batch["input_ids"].shape[-1]
                            if len(new_input_ids) < original_input_len:
                                pad_len = original_input_len - len(new_input_ids)
                                pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                new_input_ids = torch.cat([new_input_ids, torch.full((pad_len,), pad_value, device=device, dtype=new_input_ids.dtype)])
                            elif len(new_input_ids) > original_input_len:
                                new_input_ids = new_input_ids[:original_input_len]
                            new_batch.batch["input_ids"][global_idx] = new_input_ids
                        
                        # 打印替换信息（第一个样本）
                        if len(replace_indices_list) == 1:
                            print(f"Ground Truth Replace Sample:")
                            print(f"  Original response: {self.tokenizer.decode(original_input_dict.get('responses', torch.tensor([[0]]))[0].cpu().tolist(), skip_special_tokens=True)[:200]}...")
                            print(f"  Ground truth response: {response_content[:200]}...")
                    
                    # 3) 保存被替换样本的 original_input
                    if len(replace_indices_list) > 0:
                        if "original_inputs" not in new_batch.meta_info:
                            new_batch.meta_info["original_inputs"] = []
                        
                        replace_indices_np = np.asarray(replace_indices_list, dtype=np.int64)
                        for idx, original_input_dict in zip(replace_indices_list, original_input_dicts):
                            if original_input_dict:
                                new_batch.meta_info["original_inputs"].append({
                                    "original_input": original_input_dict,
                                    "indices": np.asarray([idx], dtype=np.int64),
                                    "stage_name": stage_name,
                                })
                        
                        # 4) 更新 rollout_type（如果启用）
                        if (filter_on_policy_token or filter_by_suffix_is_ratio):
                            if "rollout_type" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["rollout_type"] = np.array(
                                    ["origin"] * len(new_batch), dtype=object
                                )
                            new_batch.non_tensor_batch["rollout_type"][replace_indices_np] = "ground_truth"
                        
                        # 5) 标记 is_ground_truth（用于 policy loss 中对 ground_truth 的 advantage 进行处理）
                        if "is_ground_truth" not in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch["is_ground_truth"] = np.zeros(len(new_batch), dtype=bool)
                        new_batch.non_tensor_batch["is_ground_truth"][replace_indices_np] = True
                        
                        # 5.5) 标记 real_rollout 为 False（被替换的样本不是模型自己生成的）
                        if "real_rollout" in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch["real_rollout"][replace_indices_np] = False
                    
                    # 5) 在整体 new_batch 上重新计算 reward，得到新的 unresolved_mask
                    reward_tensor_gt, reward_metrics_gt = ray.get(
                        self.reward_fn.compute_reward.remote(new_batch)
                    )
                    overall_gt = np.asarray(reward_metrics_gt["overall"], dtype=np.float32)
                    
                    overall_gt = overall_gt.reshape(num_prompts, n)
                    prompt_any_correct_gt = (overall_gt > 0).any(axis=1)
                    new_unresolved_mask = ~prompt_any_correct_gt
                    
                    # 6) 统计信息
                    fallback_stats = new_batch.meta_info.get("fallback_stats", {})
                    fallback_stats[f"num_all_wrong_after_{stage_name}"] = int(new_unresolved_mask.sum())
                    fallback_stats[f"{stage_name}_num_replaced"] = len(replace_indices_list)
                    fallback_stats[f"{stage_name}_replace_all"] = replace_all
                    new_batch.meta_info["fallback_stats"] = fallback_stats
                    
                    print(f"Ground Truth Replace: replaced {len(replace_indices_list)} samples, {int(new_unresolved_mask.sum())} prompts still all_wrong")
                    
                    return new_unresolved_mask

                # ================== 执行所有 fallback stages ==================
                unresolved_mask = prompt_all_wrong.copy()  # 当前仍然"全错"的 prompt

                # ================== 执行单独的 ICL Fallback Stage ==================
                if self.has_icl_payload and (general_exploration or unresolved_mask.any()):
                    unresolved_mask = _icl_fallback_stage(
                        unresolved_mask=unresolved_mask,
                    )
                elif self.has_icl_payload:
                    print(f"----------------------------Skip ICL Stage (no unresolved prompts)----------------------------")
                
                # ================== Ground Truth Replace Stage ==================
                # 检查是否有 ground_truth_text 可用
                has_ground_truth = "ground_truth_text" in new_batch.non_tensor_batch
                
                if has_ground_truth:
                    if self.has_icl_payload:
                        # ICL 或纯 lorem（icl_* 来自 dataset）：与 ICL 相同，只对仍然 all_wrong 的 prompt 使用 ground_truth 替换
                        if unresolved_mask.any():
                            unresolved_mask = _ground_truth_replace(
                                unresolved_mask=unresolved_mask,
                                replace_all=False,  # 只替换 all_wrong 的 prompt
                            )
                        else:
                            print(f"----------------------------Skip Ground Truth Replace (no unresolved prompts after ICL)----------------------------")
                    else:
                        # 无 ICL / 无 lorem：直接用 ground_truth 替换每个 prompt 的 1 个 response
                        unresolved_mask = _ground_truth_replace(
                            unresolved_mask=unresolved_mask,
                            replace_all=True,  # 替换所有 prompt
                        )

                # fallback_stats 已经更新在 new_batch.meta_info["fallback_stats"] 里

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]
                
                # 如果进行了过滤，需要相应地更新 original_inputs 的索引
                if self.has_icl_payload and "original_inputs" in new_batch.meta_info:
                    kept_sample_idxs_set = set(kept_sample_idxs)
                    filtered_original_inputs = []
                    for overwrite_result in new_batch.meta_info["original_inputs"]:
                        if "indices" not in overwrite_result:
                            continue
                        indices_np = overwrite_result["indices"]
                        # 找出哪些索引在保留列表中
                        mask = np.array([idx in kept_sample_idxs_set for idx in indices_np], dtype=bool)
                        if mask.any():
                            # 更新索引：映射到新的位置
                            old_indices = indices_np[mask]
                            new_indices = np.array([kept_sample_idxs.index(int(idx)) for idx in old_indices], dtype=np.int64)
                            
                            # 更新 original_input，只保留过滤后的部分
                            filtered_original_input = {}
                            original_input = overwrite_result.get("original_input", {})
                            for key, value in original_input.items():
                                if isinstance(value, torch.Tensor):
                                    filtered_original_input[key] = value[mask]
                                elif isinstance(value, np.ndarray):
                                    filtered_original_input[key] = value[mask]
                            
                            filtered_result = {
                                "original_input": filtered_original_input,
                                "indices": new_indices,
                            }
                            if "stage_name" in overwrite_result:
                                filtered_result["stage_name"] = overwrite_result["stage_name"]
                            filtered_original_inputs.append(filtered_result)
                    new_batch.meta_info["original_inputs"] = filtered_original_inputs

            if batch is not None:
                # 合并 original_inputs
                # 保存 new_batch 的 original_inputs（如果存在），因为 concat 会覆盖 meta_info
                new_batch_original_inputs = new_batch.meta_info.get("original_inputs", [])
                batch_original_inputs = batch.meta_info.get("original_inputs", [])

                # 计算 new_batch 在 concat 后的起始位置
                batch_size_before_concat = len(batch)

                if self.config.algorithm.advantage_shaping:
                    # concat 只保留 data[0].meta_info，需显式合并两侧的 all_rollout_scores_per_uid
                    new_batch_all_rollout_scores = new_batch.meta_info.get(
                        "all_rollout_scores_per_uid", {}
                    )
                    batch_all_rollout_scores = batch.meta_info.get("all_rollout_scores_per_uid", {})

                # 先 concat batch
                batch = DataProto.concat([batch, new_batch])

                if self.config.algorithm.advantage_shaping:
                    merged_all_rollout_scores = {
                        **batch_all_rollout_scores,
                        **new_batch_all_rollout_scores,
                    }
                    if merged_all_rollout_scores:
                        batch.meta_info["all_rollout_scores_per_uid"] = merged_all_rollout_scores
                
                # 合并 original_inputs：调整 new_batch 的索引偏移
                if self.has_icl_payload:
                    # 调整 new_batch 的 original_inputs 索引
                    adjusted_new_batch_original_inputs = []
                    for overwrite_result in new_batch_original_inputs:
                        if "indices" not in overwrite_result:
                            continue
                        adjusted_result = overwrite_result.copy()
                        adjusted_result["indices"] = overwrite_result["indices"] + batch_size_before_concat
                        adjusted_new_batch_original_inputs.append(adjusted_result)
                    
                    # 合并 original_inputs
                    batch.meta_info["original_inputs"] = batch_original_inputs + adjusted_new_batch_original_inputs
            else:
                batch = new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                final_batch_size = self.config.data.rollout_batch_size * self.config.worker.rollout.n
                final_batch = batch[:final_batch_size]
                
                # 如果存在 original_inputs，需要过滤出在 final_batch 范围内的
                if self.has_icl_payload and "original_inputs" in final_batch.meta_info:
                    filtered_original_inputs = []
                    for overwrite_result in final_batch.meta_info["original_inputs"]:
                        if "indices" not in overwrite_result:
                            continue
                        indices_np = overwrite_result["indices"]
                        # 只保留索引在 final_batch 范围内的
                        mask = (indices_np < final_batch_size)
                        if mask.any():
                            filtered_indices = indices_np[mask]
                            filtered_original_input = {}
                            original_input = overwrite_result.get("original_input", {})
                            for key, value in original_input.items():
                                if isinstance(value, torch.Tensor):
                                    filtered_original_input[key] = value[mask]
                                elif isinstance(value, np.ndarray):
                                    filtered_original_input[key] = value[mask]
                            
                            filtered_result = {
                                "original_input": filtered_original_input,
                                "indices": filtered_indices,
                            }
                            if "stage_name" in overwrite_result:
                                filtered_result["stage_name"] = overwrite_result["stage_name"]
                            filtered_original_inputs.append(filtered_result)
                    final_batch.meta_info["original_inputs"] = filtered_original_inputs
                
                return final_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                if "rollout_log_probs" in batch.batch:
                    # 这里rollout_log_probs是vllm用icl生成的
                    # old_log_probs是fsdp算的无icl的prompt的log_probs
                    batch.batch["rollout_is_weights"] = batch.batch["old_log_probs"] / batch.batch["rollout_log_probs"]
                
                # 恢复 original_input（在计算完 old_log_probs 后）：保留 reference 输出，恢复原始 prompt 侧字段
                # 当 train_with_icl_prompt=True 时，跳过恢复，直接使用 ICL/reference prompt 进行训练
                if self.config.algorithm.train_with_icl_prompt and "original_inputs" in batch.meta_info:
                    batch.meta_info.pop("original_inputs", None)
                elif "original_inputs" in batch.meta_info:
                    original_inputs_list = batch.meta_info["original_inputs"]
                    device = None
                    if batch.batch is not None and len(batch.batch) > 0:
                        device = next(iter(batch.batch.values())).device
                    if batch.batch is not None:
                        for overwrite_result in original_inputs_list:
                            if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                continue
                            
                            original_input = overwrite_result["original_input"]
                            indices_np = overwrite_result["indices"]
                            indices_t = torch.as_tensor(indices_np, device=device)
                            # ========== fallback：用 original prompt + reference response 恢复 batch ==========
                            # 1. 恢复 prompts：直接用 original 替换 reference
                            if "prompts" in original_input and "prompts" in batch.batch:
                                batch.batch["prompts"][indices_t] = original_input["prompts"].to(device)
                            
                            # 2. 恢复 input_ids：用 original prompts 和 reference responses 拼接
                            if "prompts" in original_input and "responses" in batch.batch and "input_ids" in batch.batch:
                                original_prompts = original_input["prompts"].to(device)
                                reference_responses = batch.batch["responses"][indices_t]
                                batch.batch["input_ids"][indices_t] = torch.cat([original_prompts, reference_responses], dim=-1)
                            
                            # 3. 恢复 attention_mask：前半部分来自 original，后半部分来自 reference
                            if "attention_mask" in original_input and "attention_mask" in batch.batch and "response_mask" in batch.batch and "prompts" in original_input:
                                original_attention_mask = original_input["attention_mask"].to(device)
                                reference_response_mask = batch.batch["response_mask"][indices_t]
                                original_prompt_len = original_input["prompts"].shape[-1]
                                original_prompt_attention = original_attention_mask[..., :original_prompt_len]
                                batch.batch["attention_mask"][indices_t] = torch.cat([original_prompt_attention, reference_response_mask], dim=-1)
                            
                            # 4. 处理 position_ids
                            if "position_ids" in original_input and "position_ids" in batch.batch and "responses" in batch.batch and "prompts" in original_input:
                                original_position_ids = original_input["position_ids"]
                                reference_position_ids = batch.batch["position_ids"][indices_np]
                                reference_responses = batch.batch["responses"][indices_np]
                                
                                original_prompt_len = original_input["prompts"].shape[-1]
                                reference_response_len = reference_responses.shape[-1]
                                
                                num_samples = len(indices_np)
                                for i in range(num_samples):
                                    idx = indices_np[i]
                                    indices_t_single = torch.as_tensor([idx], device=device)
                                    
                                    if original_position_ids.dim() == 2:
                                        original_prompt_positions_single = original_position_ids[i:i+1, :original_prompt_len]
                                        reference_position_ids_single = reference_position_ids[i:i+1]
                                        reference_response_positions_single = reference_position_ids_single[..., -reference_response_len:]
                                        
                                        original_last_pos = original_prompt_positions_single[..., -1:]
                                        reference_first_pos = reference_response_positions_single[..., :1]
                                        delta = reference_first_pos - original_last_pos - 1
                                        
                                        non_pad_mask = _non_pad_mask_for_response_position_slice(
                                            batch.batch["response_mask"][idx], reference_response_positions_single
                                        )
                                        adjusted_response_positions = reference_response_positions_single.clone()
                                        adjusted_response_positions[non_pad_mask] = adjusted_response_positions[non_pad_mask] - delta.expand_as(adjusted_response_positions)[non_pad_mask]
                                        
                                        batch.batch["position_ids"][indices_t_single] = torch.cat([original_prompt_positions_single, adjusted_response_positions], dim=-1)
                                        
                                    elif original_position_ids.dim() == 3:
                                        original_prompt_positions_single = original_position_ids[i:i+1, :, :original_prompt_len]
                                        reference_position_ids_single = reference_position_ids[i:i+1]
                                        reference_response_positions_single = reference_position_ids_single[..., -reference_response_len:]
                                        
                                        original_last_pos = original_prompt_positions_single[..., -1:]
                                        reference_first_pos = reference_response_positions_single[..., :1]
                                        delta = reference_first_pos - original_last_pos - 1
                                        
                                        non_pad_mask = _non_pad_mask_for_response_position_slice(
                                            batch.batch["response_mask"][idx], reference_response_positions_single
                                        )
                                        adjusted_response_positions = reference_response_positions_single.clone()
                                        delta_expanded = delta.expand_as(adjusted_response_positions)
                                        adjusted_response_positions[non_pad_mask] = adjusted_response_positions[non_pad_mask] - delta_expanded[non_pad_mask]
                                        
                                        batch.batch["position_ids"][indices_t_single] = torch.cat([original_prompt_positions_single, adjusted_response_positions], dim=-1)
                        
                            # 恢复 raw_prompt_ids
                            if batch.non_tensor_batch is not None and "raw_prompt_ids" in original_input:
                                arr = batch.non_tensor_batch["raw_prompt_ids"]
                                arr[indices_np] = original_input["raw_prompt_ids"]
                    
                    # 清理 original_inputs（已经恢复，不再需要）
                    batch.meta_info.pop("original_inputs", None)
                # pdb.set_trace()
                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    # 下面这段计算了loss
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        advantage_shaping=self.config.algorithm.advantage_shaping,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor                pdb.set_trace()
                if self.config.trainer.critic_warmup <= self.global_step:
                    # 将filter_on_policy_token配置传递到batch.meta_info中
                    batch.meta_info["filter_on_policy_token"] = self.config.algorithm.filter_on_policy_token
                    batch.meta_info["token_filter_lower_bound"] = self.config.algorithm.token_filter_lower_bound
                    batch.meta_info["token_filter_upper_bound"] = self.config.algorithm.token_filter_upper_bound
                    if self.config.algorithm.filter_on_policy_token and self.config.algorithm.interval_by_pos_file is not None:
                        if self.on_policy_interval_by_pos is None:
                            # 从interval_by_pos_file中读取interval_by_pos
                            with open(self.config.algorithm.interval_by_pos_file, "r") as f:
                                interval_by_pos_dict = json.load(f)
                            # 数据格式是{"0":[lo0,hi0],"1":[lo1,hi1]}
                            # 纵向拼起来，变成[[lo0,hi0],[lo1,hi1]]
                            positions = sorted(interval_by_pos_dict.keys(), key=int)
                            self.on_policy_interval_by_pos = np.stack(
                                [interval_by_pos_dict[pos] for pos in positions],
                                axis=0
                            )
                        batch.meta_info["on_policy_interval_by_pos"] = self.on_policy_interval_by_pos
                    batch.meta_info["filter_by_suffix_is_ratio"] = self.config.algorithm.filter_by_suffix_is_ratio
                    batch.meta_info["suffix_is_ratio_lower_bound"] = self.config.algorithm.suffix_is_ratio_lower_bound
                    batch.meta_info["suffix_is_ratio_upper_bound"] = self.config.algorithm.suffix_is_ratio_upper_bound
                    batch.meta_info["policy_shaping"] = self.config.algorithm.policy_shaping
                    batch.meta_info["policy_shaping_gamma"] = self.config.algorithm.policy_shaping_gamma
                    # --- Safe Policy Loss 相关参数 ---
                    batch.meta_info["use_safe_policy_loss"] = self.config.algorithm.use_safe_policy_loss
                    batch.meta_info["safe_policy_region_method"] = self.config.algorithm.safe_policy_region_method
                    batch.meta_info["entropy_window_size"] = self.config.algorithm.entropy_window_size
                    batch.meta_info["entropy_rate_threshold"] = self.config.algorithm.entropy_rate_threshold
                    batch.meta_info["prefix_loss_type"] = self.config.algorithm.prefix_loss_type
                    batch.meta_info["trsft_alpha"] = self.config.algorithm.trsft_alpha
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            
            # 添加 fallback 统计信息到 metrics
            fallback_stats = batch.meta_info.get("fallback_stats", {})
            if fallback_stats:
                for key, value in fallback_stats.items():
                    metrics[f"fallback/{key}"] = value

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
