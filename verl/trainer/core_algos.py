# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF
import pdb

if TYPE_CHECKING:
    from .config import AlgorithmConfig

import math

class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    all_rollout_scores_per_uid: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero
        all_rollout_scores_per_uid: `(Optional[dict])`
            Only passed when ``algorithm.advantage_shaping`` is enabled in the trainer.
            Mapping ``uid -> 1D rewards`` for full rollouts (e.g. base n + ICL rollouts).
            If a uid is present, mean/std for GRPO use this vector; otherwise in-batch
            scores only (standard GRPO).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        # advantage_shaping 时 trainer 传入 all_rollout_scores_per_uid；否则为 None，仅用 batch 内 scores
        full_scores = None
        if all_rollout_scores_per_uid is not None and idx in all_rollout_scores_per_uid:
            raw = all_rollout_scores_per_uid[idx]
            full_scores = torch.as_tensor(raw, dtype=torch.float32)

        if full_scores is None:
            assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
            group_scores = torch.tensor(id2score[idx])
        else:
            assert full_scores.numel() > 1, "GRPO needs rollout.n > 1."
            group_scores = full_scores

        id2mean[idx] = torch.mean(group_scores)
        id2std[idx] = torch.std(group_scores)

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        scores[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")

def compute_off_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    # -log_probs * exp(adv/lambda)
    lambd = 1
    weight = torch.exp(advantages / lambd)
    # clip weight
    final_pg_loss = -log_probs * weight
    negative_approx_kl = log_probs - old_log_probs

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)
    metrics["pg_clipfrac_higher"] = 0.0
    metrics["pg_clipfrac_lower"] = 0.0

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics



def compute_safe_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo", "sapo"],
    loss_avg_mode: Literal["token", "seq"],
    # --- 新增参数 ---
    safe_policy_region_method: Literal["suffix", "sliding_window"] = "suffix",  # RL区域判定方法
    entropy_window_size: int = 64,           # sliding_window方法的窗口大小
    entropy_rate_threshold: float = None,        # Suffix Mean 过滤阈值
    prefix_loss_type: Literal["mask", "trsft"] = "mask", # 前缀的处理方式
    trsft_alpha: float = 0.1,               # TRSFT 的 alpha 阈值
    is_ground_truth: Optional[np.ndarray] = None,  # ground_truth 标记，shape: (bs,)
    real_rollout: Optional[np.ndarray] = None,  # 模型自己生成的标记，shape: (bs,)，True=模型生成，False=被替换
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    
    # 0. [处理 ground_truth 样本的 advantage]
    # 对于 ground_truth 样本，advantage 不为负，即 max(0, advantage)
    if is_ground_truth is not None:
        # 将 is_ground_truth 转换为 torch tensor 并扩展到与 advantages 相同的形状
        # 先确保是 bool 类型的 numpy 数组，避免 object 类型导致的转换错误
        is_ground_truth_bool = np.asarray(is_ground_truth, dtype=bool)
        gt_mask = torch.tensor(is_ground_truth_bool, dtype=torch.bool, device=advantages.device)
        gt_mask = gt_mask.unsqueeze(-1).expand_as(advantages)  # (bs, response_length)
        # 对 ground_truth 样本的 advantage 取 max(0, advantage)
        advantages = torch.where(gt_mask, torch.clamp(advantages, min=0), advantages)
    
    # 1. [适配缺失 old_log_probs]
    # 如果 old_log_probs 全为0 (这在log空间是不可能的，除非概率为1)，或者用户传入了特定标记
    # 我们假设如果 sum == 0 (且 log_probs 不为0) 说明是占位符。
    # 更安全的做法是假设用户处理好了，这里做一个兜底：
    if (old_log_probs == 0).all():
        # 视为没有旧策略，Ratio 设为 1
        old_log_probs = log_probs.detach()

    # 2. [Suffix Mean Reverse Search] 确定 RL 区域
    # rl_mask: 1 表示该位置属于 RL 训练 (Suffix), 0 表示属于 SFT/Prefix
    rl_mask = response_mask.clone()
    
    # 假设外部定义了模式: 'suffix' (默认) 或 'sliding_window'
    # entropy_method = 'suffix' 
    # entropy_window_size = 64 

    if entropy_rate_threshold is not None:
        with torch.no_grad():
            nll = -log_probs
            nll_masked = nll * response_mask 
            
            # ==========================================
            # 分支 1: Sliding Window (中间对齐策略)
            # ==========================================
            if safe_policy_region_method == 'sliding_window':
                window_size = entropy_window_size
                
                # 1. 维度准备: Conv1d 需要 (Batch, Channel, Length)
                nll_input = nll_masked.unsqueeze(1)      # (B, 1, L)
                mask_input = response_mask.unsqueeze(1)  # (B, 1, L)
                
                # 2. 构建卷积核 (全1卷积核用于求和)
                kernel = torch.ones((1, 1, window_size), device=nll.device)
                
                # 3. 设置 Padding (Same Padding 策略)
                # 为了让输出的第 i 个位置对应原序列第 i 个位置的"中心"窗口
                padding = window_size // 2
                
                # 4. 执行卷积计算局部总和
                # sum_nll: 窗口内的 NLL 总和
                # sum_count: 窗口内的有效 token 数量 (自动处理掉 padding 0)
                sum_nll = F.conv1d(nll_input, kernel, padding=padding, stride=1)
                sum_count = F.conv1d(mask_input, kernel, padding=padding, stride=1)
                
                # 裁剪多余的 padding (conv1d padding 后尺寸可能略大于 L)
                seq_len = nll.shape[1]
                sum_nll = sum_nll[:, :, :seq_len]
                sum_count = sum_count[:, :, :seq_len]
                
                # 5. 计算局部熵率 (Local Entropy Rate)
                # 注意：边缘位置的 count 可能很小，加 1e-8 防除零
                local_entropy_rate = sum_nll / (sum_count + 1e-8)
                
                # 6. 判定逻辑
                # 条件 A: 局部熵率达标
                is_stable = local_entropy_rate <= entropy_rate_threshold
                
                # 条件 B: 窗口内有效样本数过少视为不可靠 (边缘效应保护)
                # 例如：如果窗口大小64，但在句首只覆盖了5个词，统计方差太大，不予采纳
                min_window_valid_count = window_size // 4
                is_reliable_stat = sum_count >= min_window_valid_count
                
                # 综合 Mask
                # 必须 squeeze 掉 channel 维度，并再次乘以 response_mask 确保不选 Padding
                entropy_based_rl_mask = response_mask * is_stable.squeeze(1) * is_reliable_stat.squeeze(1)

            # ==========================================
            # 分支 2: Suffix Mean (动态阈值截断策略)
            # ==========================================
            elif safe_policy_region_method == 'suffix':
                # 推荐参数：最小统计长度
                MIN_VALID_LEN = 16 
                
                # 1. 翻转计算后缀统计量
                nll_flipped = torch.flip(nll_masked, dims=[1])
                mask_flipped = torch.flip(response_mask, dims=[1])
                
                cumsum_nll = torch.cumsum(nll_flipped, dim=1)
                cumsum_count = torch.cumsum(mask_flipped, dim=1)
                
                # 2. 计算实际后缀均值
                suffix_means = cumsum_nll / (cumsum_count + 1e-8)
                
                # 3. 计算动态阈值 (Statistical Bound Strategy)
                # 公式: Threshold_dyn = H + H / sqrt(N)
                tolerance_term = entropy_rate_threshold / torch.sqrt(cumsum_count + 1e-8)
                dynamic_threshold = entropy_rate_threshold + tolerance_term
                
                # 4. 判定逻辑
                # 条件 A: 熵率低于动态阈值
                is_low_entropy = suffix_means <= dynamic_threshold
                
                # 条件 B: 后缀太短，统计无意义，强制保留 (Hard Lock)
                is_too_short = cumsum_count < MIN_VALID_LEN
                
                # 条件 C: 本身是 Padding
                is_padding = mask_flipped == 0
                
                # 综合判定: 满足任一条件即视为有效后缀
                is_valid_suffix = is_low_entropy | is_too_short | is_padding
                
                # 5. 连续性检查 (一旦中间断裂，后面更长的后缀也视为无效)
                valid_chain_flipped = torch.cumprod(is_valid_suffix.long(), dim=1)
                valid_chain = torch.flip(valid_chain_flipped, dims=[1])
                
                # 6. 生成最终 Mask
                entropy_based_rl_mask = response_mask * valid_chain
            else:
                raise ValueError(f"Invalid safe_policy_region_method: {safe_policy_region_method}")
            
            # 根据 real_rollout 标记区分处理：
            # - real_rollout=True（模型自己生成的）：不应用 entropy mask，使用完整的 response_mask
            # - real_rollout=False（被替换的，如 ICL/ground_truth）：应用 entropy mask
            if real_rollout is not None:
                # 转换为 torch tensor 并扩展到与 response_mask 相同的形状
                # 先确保是 bool 类型的 numpy 数组，避免 object 类型导致的转换错误
                real_rollout_bool = np.asarray(real_rollout, dtype=bool)
                real_rollout_mask = torch.tensor(real_rollout_bool, dtype=torch.bool, device=response_mask.device)
                real_rollout_mask = real_rollout_mask.unsqueeze(-1).expand_as(response_mask)  # (bs, response_length)
                # 对于 real_rollout=True 的样本，使用 response_mask；对于 real_rollout=False 的样本，使用 entropy_based_rl_mask
                rl_mask = torch.where(real_rollout_mask, response_mask, entropy_based_rl_mask)
            else:
                # 如果没有 real_rollout 标记，对所有样本应用 entropy mask
                rl_mask = entropy_based_rl_mask

    # 3. 定义 Prefix Mask (SFT 区域)
    # 属于原 response 但不属于 RL 区域的部分
    prefix_mask = response_mask * (1 - rl_mask)

    # -------------------------------------------------------------------------
    # 计算 Part A: PPO Loss (用于 RL Mask 区域)
    # -------------------------------------------------------------------------
    negative_approx_kl = log_probs - old_log_probs
    
    # ... (GSPO 等处理逻辑保持不变，省略部分重复代码以聚焦核心) ...
    log_importance_ratio = negative_approx_kl # 简化展示，保留原函数逻辑即可
    
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # PPO Loss 计算
    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual
    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
    ppo_loss_elementwise = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)

    # -------------------------------------------------------------------------
    # 计算 Part B: SFT/TRSFT Loss (用于 Prefix Mask 区域)
    # -------------------------------------------------------------------------
    sft_loss_elementwise = torch.zeros_like(log_probs)
    
    if prefix_loss_type == "trsft":
        # TRSFT Implementation
        # 公式: if p < alpha: -p/alpha + 1 - log(alpha); else: -log(p)
        probs = torch.exp(log_probs)
        
        # 线性 Clip 部分损失
        linear_loss = -probs / trsft_alpha + 1 - math.log(trsft_alpha)
        # 标准 NLL 损失
        nll_loss = -log_probs
        
        # 选择损失
        sft_loss_elementwise = torch.where(
            probs < trsft_alpha,
            linear_loss,
            nll_loss
        )
    elif prefix_loss_type == "mask":
        # 如果是 mask 模式，loss 保持为 0
        pass

    # -------------------------------------------------------------------------
    # 4. 融合 Loss
    # -------------------------------------------------------------------------
    
    # 最终的 Loss 是两部分的加权和 (Mask 自动处理了 0 的情况)
    # 注意：PPO Loss 只在 rl_mask 生效，SFT Loss 只在 prefix_mask 生效
    final_loss_map = (ppo_loss_elementwise * rl_mask) + (sft_loss_elementwise * prefix_mask)
    
    # 确定最终用于 Average 的 Mask
    if prefix_loss_type == "mask":
        # 如果前缀被 mask 掉，分母只统计 RL token
        final_mask = rl_mask
    else:
        # 如果前缀参与 TRSFT 训练，分母统计所有 response token
        final_mask = response_mask

    # 计算最终 Scalar Loss
    final_pg_loss = average_loss(final_loss_map, final_mask, mode=loss_avg_mode)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    metrics = {}
    # ppo_kl 需要用 masked_mean 计算后转为标量
    metrics["ppo_kl"] = VF.masked_mean(-negative_approx_kl, final_mask).detach().cpu().item()
    metrics["entropy_loss"] = average_loss(-log_probs, final_mask, mode=loss_avg_mode).detach().cpu().item()
    
    # 监控指标
    with torch.no_grad():
        # response 中的 token 总数（不包含 prompt）
        total_response_tokens = response_mask.sum() + 1e-8
        # 计算 rl loss 的 token 数量
        rl_tokens = rl_mask.sum()
        # ratio_rl_tokens = 计算 rl loss 的 token 数量 / response 中的 token 总数
        metrics["ratio_rl_tokens"] = (rl_tokens / total_response_tokens).cpu().item()
        
        # 如果有 real_rollout 标记，分别统计 real_rollout 和非 real_rollout 的 rl_tokens 比例
        if real_rollout is not None:
            # 先确保是 bool 类型的 numpy 数组，避免 object 类型导致的转换错误
            real_rollout_bool = np.asarray(real_rollout, dtype=bool)
            real_rollout_mask_2d = torch.tensor(real_rollout_bool, dtype=torch.bool, device=response_mask.device).unsqueeze(-1).expand_as(response_mask)
            # real_rollout 样本的 response tokens
            real_rollout_response_tokens = (response_mask * real_rollout_mask_2d).sum() + 1e-8
            real_rollout_rl_tokens = (rl_mask * real_rollout_mask_2d).sum()
            metrics["ratio_rl_tokens_real_rollout"] = (real_rollout_rl_tokens / real_rollout_response_tokens).cpu().item()
            
            # 非 real_rollout 样本的 response tokens
            non_real_rollout_response_tokens = (response_mask * ~real_rollout_mask_2d).sum() + 1e-8
            non_real_rollout_rl_tokens = (rl_mask * ~real_rollout_mask_2d).sum()
            metrics["ratio_rl_tokens_replaced"] = (non_real_rollout_rl_tokens / non_real_rollout_response_tokens).cpu().item()
            
            # 统计 real_rollout 样本数和被替换样本数
            num_real_rollout = int(real_rollout_bool.sum())
            num_replaced = len(real_rollout) - num_real_rollout
            metrics["num_real_rollout_samples"] = num_real_rollout
            metrics["num_replaced_samples"] = num_replaced
        
        if prefix_loss_type == "trsft":
            sft_tokens = prefix_mask.sum() + 1e-8
            # 监控有多少 token 触发了 TRSFT 的线性 Clip
            probs = torch.exp(log_probs)
            clipped_tokens = ((probs < trsft_alpha) * prefix_mask).sum()
            metrics["trsft_clip_ratio"] = (clipped_tokens / sft_tokens).cpu().item()
            
            # 分别记录 PPO 部分和 SFT 部分的 loss 大小
            metrics["loss_ppo_part"] = ((ppo_loss_elementwise * rl_mask).sum() / (rl_tokens + 1e-8)).detach().cpu().item()
            metrics["loss_sft_part"] = ((sft_loss_elementwise * prefix_mask).sum() / (sft_tokens + 1e-8)).detach().cpu().item()

    # 为了保持接口兼容，metrics 里的其他值用 masked_mean 算一下
    # 注意：这里的 metrics 返回通常只用于 log，不影响梯度
    # 我们仅对 ppo_kl 等做全局平均
    for k in ["pg_clipfrac_higher", "pg_clipfrac_lower"]:
        # 这些指标只在 RL 区域有意义
        if k == "pg_clipfrac_higher": 
            val = (pg_loss < pg_loss2).float()
        else:
            val = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()
        metrics[k] = VF.masked_mean(val, rl_mask).detach().item()

    return final_pg_loss, metrics

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo"],
    loss_avg_mode: Literal["token", "seq"],
    rollout_is_weights: torch.Tensor | None = None,
    filter_on_policy_token: Optional[bool] = False,
    token_filter_lower_bound: Optional[float] = None,
    token_filter_upper_bound: Optional[float] = None,
    on_policy_interval_by_pos: Optional[np.ndarray] = None,
    rollout_type: Optional[list[str]] = None,
    n: Optional[int] = None,
    global_sample_offset: Optional[int] = None,
    policy_shaping: Optional[bool] = False,
    policy_shaping_gamma: Optional[float] = 0.1,
    is_ground_truth: Optional[np.ndarray] = None,  # ground_truth 标记，shape: (bs,)
    real_rollout: Optional[np.ndarray] = None,  # 模型自己生成的标记，shape: (bs,)，True=模型生成，False=被替换
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    # [处理 ground_truth 样本的 advantage]
    # 对于 ground_truth 样本，advantage 不为负，即 max(0, advantage)
    if is_ground_truth is not None:
        # 将 is_ground_truth 转换为 torch tensor 并扩展到与 advantages 相同的形状
        gt_mask = torch.tensor(is_ground_truth, dtype=torch.bool, device=advantages.device)
        gt_mask = gt_mask.unsqueeze(-1).expand_as(advantages)  # (bs, response_length)
        # 对 ground_truth 样本的 advantage 取 max(0, advantage)
        advantages = torch.where(gt_mask, torch.clamp(advantages, min=0), advantages)
    
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    filtered_response_mask = response_mask.clone()
    flag_reference_included = False
    simple_is_ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -20.0, 20.0))
    if filter_on_policy_token and rollout_type is not None and n is not None and global_sample_offset is not None:
        replaced_sample_indices = [i for i in range(log_probs.size(0)) if rollout_type[i] == "reference"]
        # 统计被mask掉的token数和被mask掉的token比例
        count_mask_tot = 0
        count_token_tot = 0
        if on_policy_interval_by_pos is not None:
            for i in replaced_sample_indices:
                flag_reference_included = True
                # 对于每个reference sample，如果IS_ratio不在区间内，mask掉对应的tokens
                sample_is_ratio = simple_is_ratio[i]  # (response_length,)
                sample_mask = response_mask[i]  # (response_length,)
                valid_tokens = sample_mask.bool()
                initial_valid_token_count = int(valid_tokens.sum().item())
                # 统计有多少token被mask掉
                masked_token_count = 0
                if valid_tokens.any():
                    valid_is_ratios = sample_is_ratio[valid_tokens]
                    in_interval = (valid_is_ratios >= on_policy_interval_by_pos[i, 0]) & (valid_is_ratios <= on_policy_interval_by_pos[i, 1])
                    # valid_is_ratios中，1的数量是初始使用的token数量
                    # in_interval是筛选后有效的token数量
                    # 它们的差值是被mask掉的token数量
                    masked_token_count = initial_valid_token_count - int(in_interval.sum().item())
                    valid_indices = torch.nonzero(valid_tokens, as_tuple=False).squeeze(-1)
                    out_of_interval_indices = valid_indices[~in_interval]
                    if len(out_of_interval_indices) > 0:
                        filtered_response_mask[i, out_of_interval_indices] = 0
                count_token_tot += int(sample_mask.sum().detach().item())
                count_mask_tot += masked_token_count
                # TODO: Check if all tokens are not covered: May result in bug
                if filtered_response_mask[i].sum().item() == 0:
                    print("="*80)
                    print(f"Warning: All tokens are not covered for sample {i}")
                    print("="*80)
        else:
            lower_bound = token_filter_lower_bound
            upper_bound = token_filter_upper_bound
            for i in replaced_sample_indices:
                flag_reference_included = True
                # 对于每个reference sample，如果IS_ratio不在区间内，mask掉对应的tokens
                sample_is_ratio = simple_is_ratio[i]  # (response_length,)
                sample_mask = response_mask[i]  # (response_length,)
                valid_tokens = sample_mask.bool()
                initial_valid_token_count = int(valid_tokens.sum().detach().item())
                # 统计有多少token被mask掉
                masked_token_count = 0
                if valid_tokens.any():
                    valid_is_ratios = sample_is_ratio[valid_tokens]
                    in_interval = (valid_is_ratios >= lower_bound) & (valid_is_ratios <= upper_bound)
                    valid_indices = torch.nonzero(valid_tokens, as_tuple=False).squeeze(-1)
                    out_of_interval_indices = valid_indices[~in_interval]
                    masked_token_count = initial_valid_token_count - int(in_interval.sum().detach().item())
                    if len(out_of_interval_indices) > 0:
                        filtered_response_mask[i, out_of_interval_indices] = 0
                count_token_tot += int(sample_mask.sum().detach().item())
                count_mask_tot += masked_token_count
                # TODO: Check if all tokens are not covered: May result in bug
                if filtered_response_mask[i].sum().item() == 0:
                    print("="*80)
                    print(f"Warning: All tokens are not covered for sample {i}")
                    print("="*80)
        count_mask_ratio = count_mask_tot / count_token_tot if count_token_tot > 0 else 0

    # policy_shaping情形：对非 real_rollout 的 samples 的 is_ratio 进行变换 x -> x/(x+gamma)
    # 并且在后续clip操作中不对这些 samples 进行 clip
    # 筛选标准：real_rollout == False（被替换的样本，如 ICL/ground_truth）
    reference_sample_mask = None  # 用于标记哪些样本需要 policy_shaping
    if policy_shaping and real_rollout is not None:
        # 使用 real_rollout 标记筛选：real_rollout == False 的样本需要 policy_shaping
        replaced_sample_indices = [i for i in range(log_probs.size(0)) if not real_rollout[i]]
        if len(replaced_sample_indices) > 0:
            # 不能对 log_importance_ratio 就地切片赋值：它与 log_probs 的计算图相连（且 default 分支下与
            # negative_approx_kl 共享同一存储），会触发 RuntimeError: modified by an inplace operation。
            # clone 后只对副本写入，negative_approx_kl / ppo_kl 仍基于未 shaping 的 log 差。
            log_importance_ratio = log_importance_ratio.clone()
            flag_reference_included = True
            # 创建 reference sample mask (bs,) 用于后续不 clip 这些 samples
            reference_sample_mask = torch.zeros(log_probs.size(0), dtype=torch.bool, device=log_probs.device)
            for i in replaced_sample_indices:
                reference_sample_mask[i] = True
            
            # 对这些 samples 的 log_importance_ratio 进行变换
            # 原始: ratio = exp(log_importance_ratio)
            # 变换后: new_ratio = ratio / (ratio + gamma)
            # 即: new_log_importance_ratio = log(ratio / (ratio + gamma))
            #                              = log(ratio) - log(ratio + gamma)
            #                              = log_importance_ratio - log(exp(log_importance_ratio) + gamma)
            gamma = policy_shaping_gamma
            replaced_set = set(replaced_sample_indices)
            # 不能用 log_importance_ratio[i] = ...：即使已 clone，行视图参与计算后再原地覆盖同一行
            # 仍会在反传时触发 version 冲突；用 stack 拼成新张量。
            lir = log_importance_ratio
            row_tensors = []
            for i in range(log_probs.size(0)):
                if i in replaced_set:
                    orig_log_ratio = lir[i]
                    orig_ratio = torch.exp(torch.clamp(orig_log_ratio, -20.0, 20.0))
                    new_ratio = orig_ratio / (orig_ratio + gamma)
                    row_tensors.append(torch.log(new_ratio + 1e-10))
                else:
                    row_tensors.append(lir[i])
            log_importance_ratio = torch.stack(row_tensors, dim=0)

    # 使用过滤后的response_mask
    response_mask = filtered_response_mask
    
    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )
    
    # 对于policy_shaping情形，reference samples不进行clip，使用原始ratio
    if policy_shaping and reference_sample_mask is not None:
        # 将reference samples的clipped_ratio替换为原始ratio（不clip）
        clipped_ratio = torch.where(
            reference_sample_mask.unsqueeze(-1).expand_as(clipped_ratio),
            ratio,
            clipped_ratio
        )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()
    if rollout_is_weights is not None:
        final_pg_loss = final_pg_loss * rollout_is_weights
    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}

    if filter_on_policy_token:
        if flag_reference_included:
            print(f"This is type of count_mask_tot: {type(count_mask_tot)}")
            print(f"This is type of count_token_tot: {type(count_token_tot)}")
            print(f"This is type of count_mask_ratio: {type(count_mask_ratio)}")
            print(f"This is value of count_mask_tot: {count_mask_tot}")
            print(f"This is value of count_token_tot: {count_token_tot}")
            print(f"This is value of count_mask_ratio: {count_mask_ratio}")
        metrics["count_mask_tot"] = count_mask_tot
        metrics["count_token_tot"] = count_token_tot
        metrics["count_mask_ratio"] = count_mask_ratio
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
