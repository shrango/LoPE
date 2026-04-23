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


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
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

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


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

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            base_input_ids = gen_batch_output.batch["prompts"][:2]
            base_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in base_input_ids]
            base_output_ids = gen_batch_output.batch["responses"][:2]
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
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            # reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            self.think_distill = self.config.data.think_distill
            if self.think_distill:
                print("----------------------------Think distill----------------------------")
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

                # ================== Think distill fallback stage ==================
                unresolved_mask = prompt_all_wrong.copy()  # 当前仍然"全错"的 prompt
                
                if unresolved_mask.any():
                    # 1) 当前未解决的 prompt 索引
                    unsolved_prompt_indices = np.nonzero(unresolved_mask)[0]  # [M]
                    M = len(unsolved_prompt_indices)
                    
                    if M > 0:
                        # 2) world_size & padding，让 prompt 数对 world_size 可整除
                        world_size = self.config.trainer.n_gpus_per_node

                        prompt_indices = unsolved_prompt_indices.copy()
                        if world_size > 1:
                            r = M % world_size
                            if r != 0:
                                pad = world_size - r
                                extra = prompt_indices[:pad]
                                prompt_indices = np.concatenate([prompt_indices, extra], axis=0)

                        M_eff = len(prompt_indices)

                        # 3) 从 new_batch 中 pop 出 think 需要的所有样本
                        think_stage_source_all = new_batch.pop(
                            batch_keys=[
                                "think_input_ids",
                                "think_attention_mask",
                                "think_position_ids",
                            ],
                            non_tensor_batch_keys=[
                                "raw_think_prompt_ids",
                                "multi_modal_data",
                            ],
                            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                        )

                        # 4) 构造 think stage 的 prompt-level batch：只取每个 prompt 的第一条样本
                        first_row_indices = (prompt_indices * n).tolist()
                        think_stage_prompt_batch = think_stage_source_all.index_select(first_row_indices)
                        think_stage_prompt_batch.non_tensor_batch["ground_truth"] = new_batch.non_tensor_batch["ground_truth"][first_row_indices]

                        assert len(think_stage_prompt_batch) == M_eff, (
                            f"think_distill: think_stage_prompt_batch size {len(think_stage_prompt_batch)} "
                            f"!= M_eff {M_eff}"
                        )
                        if world_size > 1:
                            assert len(think_stage_prompt_batch) % world_size == 0, (
                                f"think_distill: think_stage_prompt_batch size {len(think_stage_prompt_batch)} "
                                f"not divisible by world_size {world_size}"
                            )

                        # 5) 重命名为 input_ids / attention_mask / position_ids 等
                        rename_keys_inplace(think_stage_prompt_batch, {
                            "think_input_ids": "input_ids",
                            "think_attention_mask": "attention_mask",
                            "think_position_ids": "position_ids",
                            "raw_think_prompt_ids": "raw_prompt_ids",
                        }, drop_old=True)

                        # 6) 生成这一 stage 的回答（M_eff * n 条）
                        think_stage_gen_output = self.actor_rollout_ref_wg.generate_sequences(think_stage_prompt_batch)
                        assert len(think_stage_gen_output) == M_eff * n, (
                            f"think_distill: expect {M_eff*n} samples from generate_sequences, "
                            f"got {len(think_stage_gen_output)}"
                        )

                        think_stage_input_ids = think_stage_gen_output.batch["prompts"][:2]
                        # think_stage_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in think_stage_input_ids]
                        think_stage_input_texts = self.tokenizer.decode(think_stage_input_ids[0], skip_special_tokens=True)
                        print("=" * 30 + "Think distill Stage input texts" + "=" * 30)
                        print(f"Think distill Stage input texts: {think_stage_input_texts}")
                        print("=" * 30 + "=" * 30)
                        think_stage_output_ids = think_stage_gen_output.batch["responses"][:2]
                        think_stage_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in think_stage_output_ids]
                        # think_stage_output_texts = self.tokenizer.decode(think_stage_output_ids, skip_special_tokens=True)
                        print("=" * 30 + "Think distill Stage output texts" + "=" * 30)
                        print(f"Think distill Stage output texts: {think_stage_output_texts}")
                        print("=" * 30 + "=" * 30)

                        # 7) 计算 reference responses 的正确性（只对 think_stage_gen_output 计算）
                        think_stage_prompt_batch = think_stage_prompt_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                        think_stage_gen_output.non_tensor_batch["ground_truth"] = think_stage_prompt_batch.non_tensor_batch["ground_truth"]
                        reward_tensor_ref, reward_metrics_ref = ray.get(
                            self.reward_fn.compute_reward.remote(think_stage_gen_output)
                        )
                        overall_ref = np.asarray(reward_metrics_ref["overall"], dtype=np.float32)
                        # think_stage_gen_output 的长度是 M_eff * n，需要 reshape 为 [M_eff, n]
                        overall_ref = overall_ref.reshape(M_eff, n)  # [M_eff, n]

                        # 8) 检查每个 response 是否包含 </think>
                        # 解码所有 responses 并检查是否包含 </think>
                        device = None
                        if new_batch.batch is not None and len(new_batch.batch) > 0:
                            device = next(iter(new_batch.batch.values())).device
                        
                        responses_ref = think_stage_gen_output.batch["responses"]  # [M_eff * n, response_len]
                        has_redacted_reasoning = np.zeros(M_eff * n, dtype=bool)
                        
                        for i in range(M_eff * n):
                            response_ids = responses_ref[i].cpu().tolist()
                            # 检查 response 中是否包含 </think> token sequence
                            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                            has_redacted_reasoning[i] = "</think>" in response_text
                        
                        has_redacted_reasoning = has_redacted_reasoning.reshape(M_eff, n)  # [M_eff, n]

                        # 9) 对于每个 question，判断是否有正确的 response（需要同时满足：reward > 0 且包含 </think>），如果有则替换
                        # 使用 set 来跟踪已经处理过的 prompt，避免重复处理（padding 导致的重复）
                        processed_prompts = set()
                        
                        # 收集所有需要替换的样本索引和对应的 original 数据
                        replace_indices_list = []
                        replace_local_indices_list = []  # 在 think_stage_gen_output 中的索引
                        original_input_dicts = []  # 保存每个被替换样本的 original_input
                        
                        for i, p_idx in enumerate(prompt_indices):
                            # 跳过已经处理过的 prompt（padding 导致的重复）
                            if p_idx in processed_prompts:
                                continue
                            processed_prompts.add(p_idx)
                            
                            # 检查这个 question 的 reference responses 哪些是正确的
                            # i 是在 prompt_indices 中的位置，对应 think_stage_gen_output 中的第 i 组
                            # 需要同时满足：reward > 0 且包含 </think>
                            correct_mask = (overall_ref[i, :] > 0) & has_redacted_reasoning[i, :]  # [n]
                            num_correct = int(correct_mask.sum())
                            
                            if num_correct == 0:
                                # 没有正确的，什么都不做
                                continue
                            
                            # 确定要替换的数量：如果有 n 个正确，只替换前 n-1 个
                            num_to_replace = num_correct if num_correct < n else n - 1
                            
                            # 找出要替换的样本索引（全局）
                            base_global = int(p_idx * n)
                            base_local = i * n  # 在 think_stage_gen_output 中的起始位置
                            
                            # 找出所有正确的 response 的索引（在 0 到 n-1 范围内）
                            local_correct_indices = np.nonzero(correct_mask)[0]  # 所有正确的索引
                            # 选择前 num_to_replace 个正确的 response 替换
                            local_correct_indices_to_replace = local_correct_indices[:num_to_replace]
                            
                            for local_k in local_correct_indices_to_replace:
                                global_idx = base_global + local_k
                                local_idx_in_stage = base_local + local_k
                                
                                replace_indices_list.append(global_idx)
                                replace_local_indices_list.append(local_idx_in_stage)
                                
                                # 保存 original 的 rollout 信息
                                original_input_dict = {}
                                keys_to_save = {"prompts", "input_ids", "attention_mask", "position_ids"}
                                
                                if new_batch.batch is not None:
                                    for key in keys_to_save:
                                        if key in new_batch.batch:
                                            # 只保存单个样本
                                            original_input_dict[key] = new_batch.batch[key][global_idx:global_idx+1].clone()
                                
                                if new_batch.non_tensor_batch is not None and "raw_prompt_ids" in new_batch.non_tensor_batch:
                                    arr = new_batch.non_tensor_batch["raw_prompt_ids"]
                                    original_input_dict["raw_prompt_ids"] = np.asarray(arr)[global_idx:global_idx+1].copy()
                                
                                original_input_dicts.append(original_input_dict)
                        
                        # 10) 执行替换：将 reference responses 替换到 new_batch 中，但需要去掉 </think> 及其之前的内容
                        if len(replace_indices_list) > 0:
                            replace_indices_np = np.asarray(replace_indices_list, dtype=np.int64)
                            replace_indices_t = torch.as_tensor(replace_indices_np, device=device)
                            replace_local_indices_np = np.asarray(replace_local_indices_list, dtype=np.int64)
                            replace_local_indices_t = torch.as_tensor(replace_local_indices_np, device=device)
                            
                            # 对于每个要替换的 response，去掉 </think> 及其之前的内容
                            for local_idx, global_idx in zip(replace_local_indices_list, replace_indices_list):
                                response_ids = think_stage_gen_output.batch["responses"][local_idx].cpu().tolist()
                                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                                
                                # 找到 </think> 的位置
                                redacted_pos = response_text.find("</think>")
                                if redacted_pos != -1:
                                    # 去掉 </think> 及其之前的所有内容
                                    remaining_text = response_text[redacted_pos + len("</think>"):].strip()
                                    # 重新 tokenize
                                    remaining_ids = self.tokenizer.encode(remaining_text, add_special_tokens=False)
                                    
                                    # 获取原始 response 的长度
                                    original_len = think_stage_gen_output.batch["responses"][local_idx].shape[0]
                                    # 创建新的 response tensor
                                    remaining_tensor = torch.tensor(remaining_ids, device=device, dtype=think_stage_gen_output.batch["responses"].dtype)
                                    # 如果新 response 比原来的短，需要 padding
                                    if len(remaining_tensor) < original_len:
                                        pad_len = original_len - len(remaining_tensor)
                                        pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                        remaining_tensor = torch.cat([remaining_tensor, torch.full((pad_len,), pad_value, device=device, dtype=remaining_tensor.dtype)])
                                    elif len(remaining_tensor) > original_len:
                                        # 如果新 response 比原来的长，截断
                                        remaining_tensor = remaining_tensor[:original_len]
                                    think_stage_gen_output.batch["responses"][local_idx] = remaining_tensor
                                    
                                    # 同样处理 response_mask
                                    if "response_mask" in think_stage_gen_output.batch:
                                        response_mask = think_stage_gen_output.batch["response_mask"][local_idx]
                                        # 计算新的 response 长度
                                        new_response_len = len(remaining_ids)
                                        original_mask_len = response_mask.shape[0]
                                        if new_response_len < original_mask_len:
                                            # 创建新的 mask，前面是 1，后面是 0
                                            new_mask = torch.ones(new_response_len, device=device, dtype=response_mask.dtype)
                                            pad_mask = torch.zeros(original_mask_len - new_response_len, device=device, dtype=response_mask.dtype)
                                            think_stage_gen_output.batch["response_mask"][local_idx] = torch.cat([new_mask, pad_mask])
                                        elif new_response_len > original_mask_len:
                                            # 截断
                                            think_stage_gen_output.batch["response_mask"][local_idx] = torch.ones(original_mask_len, device=device, dtype=response_mask.dtype)
                            
                            # 替换 batch 中的字段
                            if new_batch.batch is not None and think_stage_gen_output.batch is not None:
                                common_keys = set(new_batch.batch.keys()).intersection(
                                    set(think_stage_gen_output.batch.keys())
                                )
                                for key in common_keys:
                                    new_batch.batch[key][replace_indices_t] = think_stage_gen_output.batch[key][replace_local_indices_t]
                            
                            # 替换 non_tensor_batch 中的字段
                            if new_batch.non_tensor_batch is not None and think_stage_gen_output.non_tensor_batch is not None:
                                common_nt_keys = set(new_batch.non_tensor_batch.keys()).intersection(
                                    set(think_stage_gen_output.non_tensor_batch.keys())
                                )
                                for key in common_nt_keys:
                                    arr_tgt = new_batch.non_tensor_batch[key]
                                    arr_src = think_stage_gen_output.non_tensor_batch[key]
                                    arr_tgt[replace_indices_np] = np.asarray(arr_src)[replace_local_indices_np]
                            
                            # 11) 保存被替换样本的 original_input
                            if "original_inputs" not in new_batch.meta_info:
                                new_batch.meta_info["original_inputs"] = []
                            
                            for idx, original_input_dict in zip(replace_indices_list, original_input_dicts):
                                if original_input_dict:  # 确保不为空
                                    new_batch.meta_info["original_inputs"].append({
                                        "original_input": original_input_dict,
                                        "indices": np.asarray([idx], dtype=np.int64),
                                    })
                            
                            # 12) 更新 rollout_type（如果启用）
                            if (filter_on_policy_token or filter_by_suffix_is_ratio):
                                if "rollout_type" not in new_batch.non_tensor_batch:
                                    new_batch.non_tensor_batch["rollout_type"] = np.array(
                                        ["origin"] * len(new_batch), dtype=object
                                    )
                                # 标记被替换的样本为 reference
                                new_batch.non_tensor_batch["rollout_type"][replace_indices_np] = "reference"

                        # 13) 在整体 new_batch 上重新计算 reward，得到新的 unresolved_mask
                        reward_tensor_stage, reward_metrics_stage = ray.get(
                            self.reward_fn.compute_reward.remote(new_batch)
                        )
                        overall_stage = np.asarray(reward_metrics_stage["overall"], dtype=np.float32)

                        num_samples_stage = len(new_batch)
                        assert overall_stage.size == num_samples_stage, (
                            f"think_distill: reward overall length {overall_stage.size} "
                            f"!= len(new_batch) {num_samples_stage}"
                        )

                        overall_stage = overall_stage.reshape(num_prompts, n)
                        prompt_any_correct_stage = (overall_stage > 0).any(axis=1)
                        new_unresolved_mask = ~prompt_any_correct_stage

                        # 14) 更新每个prompt对应的origin samples索引（用于后续计算on-policy区间）
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

                        # 15) 统计信息
                        fallback_stats = new_batch.meta_info.get("fallback_stats", {})
                        fallback_stats["num_all_wrong_after_think_distill"] = int(new_unresolved_mask.sum())
                        
                        # 统计 fallback 中 reference responses 的正确率
                        # 注意：只统计真实的 M 个 prompt，不统计 padding 导致的重复
                        # overall_ref 的 shape 是 [M_eff, n]，但前 M 个是真实的
                        # has_redacted_reasoning 的 shape 也是 [M_eff, n]
                        
                        # 计算有效的 correct_mask（需要同时满足 reward > 0 且包含 </think>）
                        fallback_correct_mask = (overall_ref[:M, :] > 0) & has_redacted_reasoning[:M, :]  # [M, n]
                        
                        # sample level 正确率：正确 sample 数 / 总 fallback sample 数
                        num_fallback_samples = M * n
                        num_correct_samples = int(fallback_correct_mask.sum())
                        fallback_sample_correct_ratio = num_correct_samples / num_fallback_samples if num_fallback_samples > 0 else 0.0
                        
                        # prompt level 正确率：有至少一个正确 response 的 prompt 数 / 总 fallback prompt 数
                        prompt_any_correct_fallback = fallback_correct_mask.any(axis=1)  # [M]
                        num_correct_prompts = int(prompt_any_correct_fallback.sum())
                        fallback_prompt_correct_ratio = num_correct_prompts / M if M > 0 else 0.0
                        
                        fallback_stats["num_fallback_prompts"] = M
                        fallback_stats["num_fallback_samples"] = num_fallback_samples
                        fallback_stats["num_correct_samples_in_fallback"] = num_correct_samples
                        fallback_stats["num_correct_prompts_in_fallback"] = num_correct_prompts
                        fallback_stats["fallback_sample_correct_ratio"] = fallback_sample_correct_ratio
                        fallback_stats["fallback_prompt_correct_ratio"] = fallback_prompt_correct_ratio
                        
                        new_batch.meta_info["fallback_stats"] = fallback_stats

            self.apply_hint = self.config.data.apply_hint
            if self.apply_hint:
                print("----------------------------Apply hint----------------------------")
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

                # ================== 通用 fallback_stage ==================
                def _fallback_stage(
                    stage_name: str,
                    unresolved_mask: np.ndarray,   # [num_prompts]，True 表示当前仍然"全错"的 prompt
                    pop_batch_keys,
                    pop_non_tensor_batch_keys,
                    pop_meta_info_keys,
                    rename_mapping,
                ):
                    """
                    通用 stage（knowledge / planning / solution）流程：
                    - 只对 unresolved_mask==True 的 prompt 做 rollout
                    - 使用 world_size 对 prompt 数做 padding，使其能被均分到多卡
                    - 用 generate_sequences 得到这一 stage 的输出（长度 M_eff*n）
                    - 检查 reference response 的正确性
                    - 对于每个 prompt，如果有 k 个正确的 response，用这 k 个替换原来的 k 个
                    - 如果 k==n，只替换前 n-1 个（保留 1 个原始的）
                    - 保存被替换的 original_input 以便后续恢复
                    """

                    # 1) 当前未解决的 prompt 索引
                    unsolved_prompt_indices = np.nonzero(unresolved_mask)[0]  # [M]
                    M = len(unsolved_prompt_indices)
                    if M == 0:
                        return unresolved_mask  # 没有需要处理的 prompt 了

                    # 2) world_size & padding，让 prompt 数对 world_size 可整除
                    world_size = self.config.trainer.n_gpus_per_node

                    prompt_indices = unsolved_prompt_indices.copy()
                    if world_size > 1:
                        r = M % world_size
                        if r != 0:
                            pad = world_size - r
                            extra = prompt_indices[:pad]
                            prompt_indices = np.concatenate([prompt_indices, extra], axis=0)

                    M_eff = len(prompt_indices)

                    # 3) 从 new_batch 中 pop 出本 stage 需要的所有样本
                    stage_source_all = new_batch.pop(
                        batch_keys=pop_batch_keys,
                        non_tensor_batch_keys=pop_non_tensor_batch_keys,
                        meta_info_keys=pop_meta_info_keys,
                    )

                    # 4) 构造本 stage 的 prompt-level batch：只取每个 prompt 的第一条样本
                    first_row_indices = (prompt_indices * n).tolist()
                    stage_prompt_batch = stage_source_all.index_select(first_row_indices)
                    stage_prompt_batch.non_tensor_batch["ground_truth"] = new_batch.non_tensor_batch["ground_truth"][first_row_indices]

                    assert len(stage_prompt_batch) == M_eff, (
                        f"{stage_name}: stage_prompt_batch size {len(stage_prompt_batch)} "
                        f"!= M_eff {M_eff}"
                    )
                    if world_size > 1:
                        assert len(stage_prompt_batch) % world_size == 0, (
                            f"{stage_name}: stage_prompt_batch size {len(stage_prompt_batch)} "
                            f"not divisible by world_size {world_size}"
                        )

                    # 5) 重命名为 input_ids / attention_mask / position_ids 等
                    rename_keys_inplace(stage_prompt_batch, rename_mapping, drop_old=True)

                    # 6) 生成这一 stage 的回答（M_eff * n 条）
                    stage_gen_output = self.actor_rollout_ref_wg.generate_sequences(stage_prompt_batch)
                    assert len(stage_gen_output) == M_eff * n, (
                        f"{stage_name}: expect {M_eff*n} samples from generate_sequences, "
                        f"got {len(stage_gen_output)}"
                    )

                    stage_input_ids = stage_gen_output.batch["prompts"][:2]
                    stage_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in stage_input_ids]
                    print("=" * 30 + f"{stage_name} Stage input texts" + "=" * 30)
                    print(f"{stage_name} Stage input texts: {stage_input_texts}")
                    print("=" * 30 + "=" * 30)
                    stage_output_ids = stage_gen_output.batch["responses"][:2]
                    stage_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in stage_output_ids]
                    print("=" * 30 + f"{stage_name} Stage output texts" + "=" * 30)
                    print(f"{stage_name} Stage output texts: {stage_output_texts}")
                    print("=" * 30 + "=" * 30)

                    # 7) 计算 reference responses 的正确性（只对 stage_gen_output 计算）
                    stage_prompt_batch = stage_prompt_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    stage_gen_output.non_tensor_batch["ground_truth"] = stage_prompt_batch.non_tensor_batch["ground_truth"]
                    reward_tensor_ref, reward_metrics_ref = ray.get(
                        self.reward_fn.compute_reward.remote(stage_gen_output)
                    )
                    overall_ref = np.asarray(reward_metrics_ref["overall"], dtype=np.float32)
                    # stage_gen_output 的长度是 M_eff * n，需要 reshape 为 [M_eff, n]
                    overall_ref = overall_ref.reshape(M_eff, n)  # [M_eff, n]

                    # 8) 对于每个 question，判断是否有正确的 response，如果有则替换
                    device = None
                    if new_batch.batch is not None and len(new_batch.batch) > 0:
                        device = next(iter(new_batch.batch.values())).device
                    
                    # 使用 set 来跟踪已经处理过的 prompt，避免重复处理（padding 导致的重复）
                    processed_prompts = set()
                    
                    # 收集所有需要替换的样本索引和对应的 original 数据
                    replace_indices_list = []
                    replace_local_indices_list = []  # 在 stage_gen_output 中的索引
                    original_input_dicts = []  # 保存每个被替换样本的 original_input
                    
                    for i, p_idx in enumerate(prompt_indices):
                        # 跳过已经处理过的 prompt（padding 导致的重复）
                        if p_idx in processed_prompts:
                            continue
                        processed_prompts.add(p_idx)
                        
                        # 检查这个 question 的 reference responses 哪些是正确的
                        # i 是在 prompt_indices 中的位置，对应 stage_gen_output 中的第 i 组
                        correct_mask = overall_ref[i, :] > 0  # [n]
                        num_correct = int(correct_mask.sum())
                        
                        if num_correct == 0:
                            # 没有正确的，什么都不做
                            continue
                        
                        # 确定要替换的数量：如果有 n 个正确，只替换前 n-1 个
                        num_to_replace = num_correct if num_correct < n else n - 1
                        
                        # 找出要替换的样本索引（全局）
                        base_global = int(p_idx * n)
                        base_local = i * n  # 在 stage_gen_output 中的起始位置
                        
                        # 找出所有正确的 response 的索引（在 0 到 n-1 范围内）
                        local_correct_indices = np.nonzero(correct_mask)[0]  # 所有正确的索引
                        # 选择前 num_to_replace 个正确的 response 替换
                        local_correct_indices_to_replace = local_correct_indices[:num_to_replace]
                        
                        for local_k in local_correct_indices_to_replace:
                            global_idx = base_global + local_k
                            local_idx_in_stage = base_local + local_k
                            
                            replace_indices_list.append(global_idx)
                            replace_local_indices_list.append(local_idx_in_stage)
                            
                            # 保存 original 的 rollout 信息
                            original_input_dict = {}
                            keys_to_save = {"prompts", "input_ids", "attention_mask", "position_ids"}
                            
                            if new_batch.batch is not None:
                                for key in keys_to_save:
                                    if key in new_batch.batch:
                                        # 只保存单个样本
                                        original_input_dict[key] = new_batch.batch[key][global_idx:global_idx+1].clone()
                            
                            if new_batch.non_tensor_batch is not None and "raw_prompt_ids" in new_batch.non_tensor_batch:
                                arr = new_batch.non_tensor_batch["raw_prompt_ids"]
                                original_input_dict["raw_prompt_ids"] = np.asarray(arr)[global_idx:global_idx+1].copy()
                            
                            original_input_dicts.append(original_input_dict)
                    
                    # 9) 执行替换：将 reference responses 替换到 new_batch 中
                    if len(replace_indices_list) > 0:
                        replace_indices_np = np.asarray(replace_indices_list, dtype=np.int64)
                        replace_indices_t = torch.as_tensor(replace_indices_np, device=device)
                        replace_local_indices_np = np.asarray(replace_local_indices_list, dtype=np.int64)
                        replace_local_indices_t = torch.as_tensor(replace_local_indices_np, device=device)
                        
                        # 替换 batch 中的字段
                        if new_batch.batch is not None and stage_gen_output.batch is not None:
                            common_keys = set(new_batch.batch.keys()).intersection(
                                set(stage_gen_output.batch.keys())
                            )
                            for key in common_keys:
                                new_batch.batch[key][replace_indices_t] = stage_gen_output.batch[key][replace_local_indices_t]
                        
                        # 替换 non_tensor_batch 中的字段
                        if new_batch.non_tensor_batch is not None and stage_gen_output.non_tensor_batch is not None:
                            common_nt_keys = set(new_batch.non_tensor_batch.keys()).intersection(
                                set(stage_gen_output.non_tensor_batch.keys())
                            )
                            for key in common_nt_keys:
                                arr_tgt = new_batch.non_tensor_batch[key]
                                arr_src = stage_gen_output.non_tensor_batch[key]
                                arr_tgt[replace_indices_np] = np.asarray(arr_src)[replace_local_indices_np]
                        
                        # 10) 保存被替换样本的 original_input
                        if "original_inputs" not in new_batch.meta_info:
                            new_batch.meta_info["original_inputs"] = []
                        
                        for idx, original_input_dict in zip(replace_indices_list, original_input_dicts):
                            if original_input_dict:  # 确保不为空
                                new_batch.meta_info["original_inputs"].append({
                                    "original_input": original_input_dict,
                                    "indices": np.asarray([idx], dtype=np.int64),
                                })
                        
                        # 11) 更新 rollout_type（如果启用）
                        if (filter_on_policy_token or filter_by_suffix_is_ratio):
                            if "rollout_type" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["rollout_type"] = np.array(
                                    ["origin"] * len(new_batch), dtype=object
                                )
                            # 标记被替换的样本为 reference
                            new_batch.non_tensor_batch["rollout_type"][replace_indices_np] = "reference"

                    # 12) 在整体 new_batch 上重新计算 reward，得到新的 unresolved_mask
                    reward_tensor_stage, reward_metrics_stage = ray.get(
                        self.reward_fn.compute_reward.remote(new_batch)
                    )
                    overall_stage = np.asarray(reward_metrics_stage["overall"], dtype=np.float32)

                    num_samples_stage = len(new_batch)
                    assert overall_stage.size == num_samples_stage, (
                        f"{stage_name}: reward overall length {overall_stage.size} "
                        f"!= len(new_batch) {num_samples_stage}"
                    )

                    overall_stage = overall_stage.reshape(num_prompts, n)
                    prompt_any_correct_stage = (overall_stage > 0).any(axis=1)
                    new_unresolved_mask = ~prompt_any_correct_stage

                    # 13) 更新每个prompt对应的origin samples索引（用于后续计算on-policy区间）
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

                    # 15) 统计信息
                    fallback_stats = new_batch.meta_info.get("fallback_stats", {})
                    fallback_stats[f"num_all_wrong_after_{stage_name}"] = int(new_unresolved_mask.sum())
                    new_batch.meta_info["fallback_stats"] = fallback_stats

                    return new_unresolved_mask

                # ================== Step 2: Knowledge fallback ==================
                unresolved_mask = prompt_all_wrong.copy()  # 当前仍然"全错"的 prompt

                if unresolved_mask.any():
                    unresolved_mask = _fallback_stage(
                        stage_name="knowledge",
                        unresolved_mask=unresolved_mask,
                        pop_batch_keys=[
                            "knowledge_components_input_ids",
                            "knowledge_components_attention_mask",
                            "knowledge_components_position_ids",
                        ],
                        pop_non_tensor_batch_keys=[
                            "raw_prompt_with_knowledge_components_ids",
                            "multi_modal_data",
                        ],
                        pop_meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                        rename_mapping={
                            "knowledge_components_input_ids": "input_ids",
                            "knowledge_components_attention_mask": "attention_mask",
                            "knowledge_components_position_ids": "position_ids",
                            "raw_prompt_with_knowledge_components_ids": "raw_prompt_ids",
                        },
                    )

                # ================== Step 3: Planning fallback ==================
                if unresolved_mask.any():
                    unresolved_mask = _fallback_stage(
                        stage_name="planning",
                        unresolved_mask=unresolved_mask,
                        pop_batch_keys=[
                            "planning_skeleton_input_ids",
                            "planning_skeleton_attention_mask",
                            "planning_skeleton_position_ids",
                        ],
                        pop_non_tensor_batch_keys=[
                            "raw_prompt_with_planning_skeleton_ids",
                            "multi_modal_data",
                        ],
                        pop_meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                        rename_mapping={
                            "planning_skeleton_input_ids": "input_ids",
                            "planning_skeleton_attention_mask": "attention_mask",
                            "planning_skeleton_position_ids": "position_ids",
                            "raw_prompt_with_planning_skeleton_ids": "raw_prompt_ids",
                        },
                    )

                # ================== Step 4: Solution fallback ==================
                if unresolved_mask.any():
                    unresolved_mask = _fallback_stage(
                        stage_name="solution",
                        unresolved_mask=unresolved_mask,
                        pop_batch_keys=[
                            "solution_breakdown_input_ids",
                            "solution_breakdown_attention_mask",
                            "solution_breakdown_position_ids",
                        ],
                        pop_non_tensor_batch_keys=[
                            "raw_prompt_with_solution_breakdown_ids",
                            "multi_modal_data",
                        ],
                        pop_meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                        rename_mapping={
                            "solution_breakdown_input_ids": "input_ids",
                            "solution_breakdown_attention_mask": "attention_mask",
                            "solution_breakdown_position_ids": "position_ids",
                            "raw_prompt_with_solution_breakdown_ids": "raw_prompt_ids",
                        },
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
                if self.apply_hint and "original_inputs" in new_batch.meta_info:
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
                            
                            filtered_original_inputs.append({
                                "original_input": filtered_original_input,
                                "indices": new_indices,
                            })
                    new_batch.meta_info["original_inputs"] = filtered_original_inputs

            if batch is not None:
                # 合并 original_inputs
                # 保存 new_batch 的 original_inputs（如果存在），因为 concat 会覆盖 meta_info
                new_batch_original_inputs = new_batch.meta_info.get("original_inputs", [])
                batch_original_inputs = batch.meta_info.get("original_inputs", [])
                
                # 计算 new_batch 在 concat 后的起始位置
                batch_size_before_concat = len(batch)
                
                # 先 concat batch
                batch = DataProto.concat([batch, new_batch])
                
                # 合并 original_inputs：调整 new_batch 的索引偏移
                if self.apply_hint:
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
                if self.apply_hint and "original_inputs" in final_batch.meta_info:
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
                            
                            filtered_original_inputs.append({
                                "original_input": filtered_original_input,
                                "indices": filtered_indices,
                            })
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
                    batch.batch["rollout_is_weights"] = batch.batch["old_log_probs"] / batch.batch["rollout_log_probs"]
                
                # 恢复 original_input（在计算完 old_log_probs 后）
                # 恢复 original 的输入，但保留 reference 的输出
                if self.think_distill and "original_inputs" in batch.meta_info:
                    original_inputs_list = batch.meta_info["original_inputs"]
                    device = None
                    if batch.batch is not None and len(batch.batch) > 0:
                        device = next(iter(batch.batch.values())).device
                    if batch.batch is not None:
                        # ========== 处理 think_distill 的恢复逻辑 ==========
                        # 关键点：
                        # 1. original_prompts_text 里面是以 <think>\n\n</think> 结尾
                        # 2. original input_ids 里是不包括 <think> 或 </think> 的
                        # 3. batch 里被替换的 prompt 以 <think>\n 结尾，它的 input_ids 包括了 </think>
                        # 4. 需要去掉 batch input_ids 中的 </think> 及其左边的所有 token
                        # 5. 需要调整 log_probs 以保持对齐
                        
                        for overwrite_result in original_inputs_list:
                            if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                continue
                            
                            original_input = overwrite_result["original_input"]
                            indices_np = overwrite_result["indices"]
                            indices_t = torch.as_tensor(indices_np, device=device)
                            
                            # 1. 恢复 prompts：直接使用 original_input["prompts"]（包含 <think>\n\n</think>）
                            if "prompts" in original_input and "prompts" in batch.batch:
                                # 直接使用 original_input["prompts"]，不需要 decode/encode，数据已经是 pad 到最大长度的
                                # original_input["prompts"] 是带<think>\n\n</think> 的
                                # 原来的batch.batch["prompts"] 以<think>\n结尾
                                batch.batch["prompts"][indices_t] = original_input["prompts"].to(device)
                            
                            # 2. 处理 input_ids：去掉 </think> 以及它左边的所有 token
                            if "input_ids" in batch.batch:
                                # 对于每个样本，找到 </think> 的位置，去掉它及其左边的所有 token
                                for i, idx in enumerate(indices_np):
                                    idx_int = int(idx)
                                    current_input_ids = batch.batch["input_ids"][idx_int]  # [seq_len]
                                    current_input_ids_list = current_input_ids.cpu().tolist()
                                    
                                    # 解码找到 </think> 的位置
                                    current_input_text = self.tokenizer.decode(current_input_ids_list, skip_special_tokens=False)
                                    redacted_pos = current_input_text.find("</think>")
                                    
                                    if redacted_pos != -1:
                                        # 找到 </think> 在 token ids 中的位置
                                        # 我们需要找到 </think> 对应的 token ids 位置
                                        redacted_reasoning_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                                        
                                        # 在 token ids 中查找 </think> 的位置，只在 response 部分搜索
                                        prompt_len = original_input["prompts"].shape[-1]
                                        token_pos = -1
                                        for j in range(prompt_len, len(current_input_ids_list) - len(redacted_reasoning_ids) + 1):
                                            if current_input_ids_list[j:j+len(redacted_reasoning_ids)] == redacted_reasoning_ids:
                                                token_pos = j
                                                break

                                        if token_pos != -1:
                                            # 去掉 </think> 及其左边的所有 token
                                            # 保留右边的 token（即 response 部分）
                                            remaining_input_ids = current_input_ids_list[token_pos + len(redacted_reasoning_ids):]

                                            # 使用 original_prompts 对应的 token ids（包含 pad）
                                            original_prompts_ids = original_input["prompts"][0].cpu().tolist()
                                            pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                            
                                            # 计算 response 部分应有的长度
                                            original_seq_len = current_input_ids.shape[-1]
                                            prompt_len = len(original_prompts_ids)  # prompt 固定长度（包含 pad）
                                            response_len = original_seq_len - prompt_len  # response 部分的长度
                                            
                                            # 在 remaining_input_ids 后面添加 pad，使其长度等于 response_len
                                            if len(remaining_input_ids) < response_len:
                                                pad_len = response_len - len(remaining_input_ids)
                                                padded_remaining_input_ids = remaining_input_ids + [pad_value] * pad_len
                                            elif len(remaining_input_ids) > response_len:
                                                padded_remaining_input_ids = remaining_input_ids[:response_len]
                                            else:
                                                padded_remaining_input_ids = remaining_input_ids
                                            
                                            # 拼接：original_prompts（包含pad）+ padded_remaining_response
                                            new_input_ids = original_prompts_ids + padded_remaining_input_ids
                                            
                                            # 转换为 tensor
                                            new_input_ids_tensor = torch.tensor(new_input_ids, device=device, dtype=current_input_ids.dtype)
                                            
                                            batch.batch["input_ids"][idx_int] = new_input_ids_tensor
                                            
                                            # 2.1 同步更新 responses：等于 input_ids 的后半部分
                                            if "responses" in batch.batch:
                                                batch.batch["responses"][idx_int] = new_input_ids_tensor[prompt_len:]
                                            
                                            # 3. 调整 log_probs：计算去掉了多少 token，左移相应的距离
                                            # 去掉的 token 数量 = token_pos + len(redacted_reasoning_ids)（即 </think> 及其左边的所有 token）
                                            # 新增的 token 数量 = prompt_len（新的 prompt tokens，包含 pad）
                                            # 所以实际的变化 = 去掉的数量 - 新增的数量
                                            tokens_removed_from_start = token_pos + len(redacted_reasoning_ids)  # 从开头去掉的 token 数量
                                            tokens_added_at_start = prompt_len  # 在开头新增的 token 数量
                                            tokens_removed = tokens_removed_from_start - tokens_added_at_start  # 净变化

                                            # tokens_removed 应该总是 >= 0，因为被替换的 prompt 以 <think>\n 结尾并包含 </think>
                                            # 而 original_prompts 以 <think>\n\n</think> 结尾
                                            assert tokens_removed >= 0, f"tokens_removed should be >= 0, got {tokens_removed}"
                                            
                                            if tokens_removed > 0:
                                                # 调整 old_log_probs：左移，去掉前面的 token，后面补 0
                                                if "old_log_probs" in batch.batch:
                                                    old_log_probs = batch.batch["old_log_probs"][idx_int]  # [seq_len]
                                                    old_log_probs_new = torch.cat([
                                                        old_log_probs[tokens_removed:],
                                                        torch.zeros(tokens_removed, device=device, dtype=old_log_probs.dtype)
                                                    ], dim=-1)
                                                    batch.batch["old_log_probs"][idx_int] = old_log_probs_new
                                                
                                                # 调整 rollout_log_probs（pad 值为 -1）
                                                if "rollout_log_probs" in batch.batch:
                                                    rollout_log_probs = batch.batch["rollout_log_probs"][idx_int]  # [seq_len]
                                                    rollout_log_probs_new = torch.cat([
                                                        rollout_log_probs[tokens_removed:],
                                                        torch.full((tokens_removed,), -1.0, device=device, dtype=rollout_log_probs.dtype)
                                                    ], dim=-1)
                                                    batch.batch["rollout_log_probs"][idx_int] = rollout_log_probs_new
                                                
                                                # 调整 rollout_is_weights（pad 值为 -0.0）
                                                if "rollout_is_weights" in batch.batch:
                                                    rollout_is_weights = batch.batch["rollout_is_weights"][idx_int]  # [seq_len]
                                                    rollout_is_weights_new = torch.cat([
                                                        rollout_is_weights[tokens_removed:],
                                                        torch.full((tokens_removed,), -0.0, device=device, dtype=rollout_is_weights.dtype)
                                                    ], dim=-1)
                                                    batch.batch["rollout_is_weights"][idx_int] = rollout_is_weights_new
                            
                            # 3. 恢复 attention_mask：根据新的 input_ids 重新计算
                            if "attention_mask" in batch.batch and "input_ids" in batch.batch:
                                for i, idx in enumerate(indices_np):
                                    idx_int = int(idx)
                                    new_input_ids = batch.batch["input_ids"][idx_int]  # [seq_len]
                                    pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                    new_attention_mask = (new_input_ids != pad_value).long()  # [seq_len]
                                    batch.batch["attention_mask"][idx_int] = new_attention_mask
                                    
                                    # 3.1 同步更新 response_mask：等于 attention_mask 的后半部分
                                    if "response_mask" in batch.batch and "prompts" in original_input:
                                        prompt_len = original_input["prompts"].shape[-1]
                                        batch.batch["response_mask"][idx_int] = new_attention_mask[prompt_len:]
                        
                        # ========== 第4步：逐个样本处理 position_ids ==========
                        # 根据新的 input_ids 重新计算 position_ids
                        if "position_ids" in batch.batch and "input_ids" in batch.batch:
                            for overwrite_result in original_inputs_list:
                                if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                    continue
                                
                                original_input = overwrite_result["original_input"]
                                indices_np = overwrite_result["indices"]
                                
                                if "position_ids" not in original_input:
                                    continue
                                
                                original_position_ids = original_input["position_ids"]  # [num_samples, seq_len] or [num_samples, 4, seq_len]
                                
                                # 逐个样本处理
                                for i, idx in enumerate(indices_np):
                                    idx_int = int(idx)
                                    new_attention_mask = batch.batch["attention_mask"][idx_int]  # [seq_len]
                                    prompt_len = original_input["prompts"].shape[-1]
                                    seq_len = new_attention_mask.shape[-1]
                                    device = new_attention_mask.device
                                    
                                    # prompt 部分：从第一个非 pad 开始递增
                                    prompt_attention_mask = new_attention_mask[:prompt_len]
                                    prompt_positions = torch.clamp(prompt_attention_mask.cumsum(dim=-1) - 1, min=0)
                                    
                                    # response 部分：不管 pad，从 prompt 最后位置 +1 开始一直递增
                                    response_len = seq_len - prompt_len
                                    
                                    if response_len > 0:
                                        last_prompt_pos = prompt_positions[-1]
                                        response_positions = torch.arange(1, response_len + 1, device=device, dtype=prompt_positions.dtype) + last_prompt_pos
                                    else:
                                        response_positions = torch.tensor([], device=device, dtype=prompt_positions.dtype)
                                    
                                    # 拼接
                                    new_position_ids_1d = torch.cat([prompt_positions, response_positions], dim=-1)
                                    
                                    # 确定 original position_ids 的维度
                                    if original_position_ids.dim() == 2:
                                        # 2D: [num_samples, seq_len]
                                        batch.batch["position_ids"][idx_int] = new_position_ids_1d
                                        
                                    elif original_position_ids.dim() == 3:
                                        # 3D: [num_samples, 4, seq_len] (qwen2vl mrope)
                                        # 扩展为 4 维（qwen2vl mrope）
                                        new_position_ids = new_position_ids_1d.unsqueeze(0).expand(4, -1)  # [4, seq_len]
                                        batch.batch["position_ids"][idx_int] = new_position_ids
                        
                        # 恢复 raw_prompt_ids（如果有）
                        if batch.non_tensor_batch is not None:
                            for overwrite_result in original_inputs_list:
                                if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                    continue
                                original_input = overwrite_result["original_input"]
                                indices_np = overwrite_result["indices"]
                                
                                # 直接使用 original_input["raw_prompt_ids"]
                                if "raw_prompt_ids" in original_input:
                                    arr = batch.non_tensor_batch["raw_prompt_ids"]
                                    arr[indices_np] = original_input["raw_prompt_ids"]
                    
                    # 验证数据一致性
                    assert (batch.batch["input_ids"][:, prompt_len:] == batch.batch["responses"]).all(), \
                        "input_ids 后半部分应该等于 responses"
                    assert (batch.batch["input_ids"][:, :prompt_len] == batch.batch["prompts"]).all(), \
                        "input_ids 前半部分应该等于 prompts"
                    assert (batch.batch["attention_mask"][:, prompt_len:] == batch.batch["response_mask"]).all(), \
                        "attention_mask 后半部分应该等于 response_mask"
                    
                    # 清理 original_inputs（已经恢复，不再需要）
                    batch.meta_info.pop("original_inputs", None)
                if self.apply_hint and "original_inputs" in batch.meta_info:
                    original_inputs_list = batch.meta_info["original_inputs"]
                    device = None
                    if batch.batch is not None and len(batch.batch) > 0:
                        device = next(iter(batch.batch.values())).device
                    if batch.batch is not None:
                        # ========== 前3步：并行处理（tensor操作） ==========
                        for overwrite_result in original_inputs_list:
                            # pdb.set_trace()
                            if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                continue
                            
                            original_input = overwrite_result["original_input"]
                            indices_np = overwrite_result["indices"]
                            indices_t = torch.as_tensor(indices_np, device=device)
                            # pdb.set_trace()
                            # 1. 恢复 prompts：直接用 original 替换 reference
                            if "prompts" in original_input and "prompts" in batch.batch:
                                batch.batch["prompts"][indices_t] = original_input["prompts"]
                            
                            # 2. 恢复 input_ids：用 original prompts 和 reference responses 拼接
                            if "prompts" in original_input and "responses" in batch.batch and "input_ids" in batch.batch:
                                original_prompts = original_input["prompts"]  # [num_samples, prompt_len]
                                reference_responses = batch.batch["responses"][indices_t]  # [num_samples, response_len]
                                # 拼接 prompts 和 responses
                                batch.batch["input_ids"][indices_t] = torch.cat([original_prompts, reference_responses], dim=-1)
                            
                            # 3. 恢复 attention_mask：前半部分来自 original，后半部分来自 reference
                            if "attention_mask" in original_input and "attention_mask" in batch.batch and "response_mask" in batch.batch and "prompts" in original_input:
                                original_attention_mask = original_input["attention_mask"]  # [num_samples, original_total_len]
                                reference_response_mask = batch.batch["response_mask"][indices_t]  # [num_samples, response_len]
                                original_prompt_len = original_input["prompts"].shape[-1]
                                # 拼接：original prompt 的 attention_mask + reference response_mask
                                original_prompt_attention = original_attention_mask[..., :original_prompt_len]
                                batch.batch["attention_mask"][indices_t] = torch.cat([original_prompt_attention, reference_response_mask], dim=-1)
                        
                        # ========== 第4步：逐个样本处理 position_ids ==========
                        if "position_ids" in batch.batch and "responses" in batch.batch:
                            for overwrite_result in original_inputs_list:
                                if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                    continue
                                
                                original_input = overwrite_result["original_input"]
                                indices_np = overwrite_result["indices"]
                                
                                if "position_ids" not in original_input or "prompts" not in original_input:
                                    continue
                                
                                original_position_ids = original_input["position_ids"]  # [num_samples, original_total_len] or [num_samples, 4, original_total_len]
                                reference_position_ids = batch.batch["position_ids"][indices_np]  # [num_samples, reference_total_len] or [num_samples, 4, reference_total_len]
                                reference_responses = batch.batch["responses"][indices_np]  # [num_samples, response_len]
                                
                                original_prompt_len = original_input["prompts"].shape[-1]
                                reference_response_len = reference_responses.shape[-1]
                                
                                # 逐个样本处理
                                num_samples = len(indices_np)
                                for i in range(num_samples):
                                    idx = indices_np[i]
                                    indices_t_single = torch.as_tensor([idx], device=device)
                                    
                                    # 确定 original position_ids 的维度
                                    if original_position_ids.dim() == 2:
                                        # 2D: [num_samples, seq_len]
                                        original_prompt_positions_single = original_position_ids[i:i+1, :original_prompt_len]  # [1, prompt_len]
                                        reference_position_ids_single = reference_position_ids[i:i+1]  # [1, reference_total_len]
                                        reference_response_positions_single = reference_position_ids_single[..., -reference_response_len:]  # [1, response_len]
                                        
                                        # 计算差值：original prompt 最后一个位置和 reference response 第一个位置的差
                                        original_last_pos = original_prompt_positions_single[..., -1:]  # [1, 1]
                                        reference_first_pos = reference_response_positions_single[..., :1]  # [1, 1]
                                        delta = reference_first_pos - original_last_pos - 1  # [1, 1]
                                        
                                        # 调整 reference response 的 position_ids：除了末尾 pad（值是 151643）之外，统统减去这个差值
                                        pad_value = 151643
                                        non_pad_mask = reference_response_positions_single != pad_value
                                        adjusted_response_positions = reference_response_positions_single.clone()
                                        adjusted_response_positions[non_pad_mask] = adjusted_response_positions[non_pad_mask] - delta.expand_as(adjusted_response_positions)[non_pad_mask]
                                        
                                        # 拼接
                                        batch.batch["position_ids"][indices_t_single] = torch.cat([original_prompt_positions_single, adjusted_response_positions], dim=-1)
                                        
                                    elif original_position_ids.dim() == 3:
                                        # 3D: [num_samples, 4, seq_len] (qwen2vl mrope)
                                        original_prompt_positions_single = original_position_ids[i:i+1, :, :original_prompt_len]  # [1, 4, prompt_len]
                                        reference_position_ids_single = reference_position_ids[i:i+1]  # [1, 4, reference_total_len]
                                        reference_response_positions_single = reference_position_ids_single[..., -reference_response_len:]  # [1, 4, response_len]
                                        
                                        # 计算差值：original prompt 最后一个位置和 reference response 第一个位置的差
                                        original_last_pos = original_prompt_positions_single[..., -1:]  # [1, 4, 1]
                                        reference_first_pos = reference_response_positions_single[..., :1]  # [1, 4, 1]
                                        delta = reference_first_pos - original_last_pos - 1  # [1, 4, 1]
                                        
                                        # 调整 reference response 的 position_ids：除了末尾 pad（值是 151643）之外，统统减去这个差值
                                        pad_value = 151643
                                        non_pad_mask = reference_response_positions_single != pad_value
                                        adjusted_response_positions = reference_response_positions_single.clone()
                                        delta_expanded = delta.expand_as(adjusted_response_positions)
                                        adjusted_response_positions[non_pad_mask] = adjusted_response_positions[non_pad_mask] - delta_expanded[non_pad_mask]
                                        
                                        # 拼接
                                        batch.batch["position_ids"][indices_t_single] = torch.cat([original_prompt_positions_single, adjusted_response_positions], dim=-1)
                        
                        # 恢复 raw_prompt_ids（如果有）
                        if batch.non_tensor_batch is not None:
                            for overwrite_result in original_inputs_list:
                                if "original_input" not in overwrite_result or "indices" not in overwrite_result:
                                    continue
                                original_input = overwrite_result["original_input"]
                                indices_np = overwrite_result["indices"]
                                if "raw_prompt_ids" in original_input:
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
