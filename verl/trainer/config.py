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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    apply_icl: bool = False  # 是否启用 In-Context Learning fallback
    icl_examples_path: Optional[str] = None  # ICL examples 文件路径
    num_icl_examples: int = 5  # ICL examples 的数量
    icl_rollout_n: int = 1  # 每个 ICL prompt 生成的 response 数量
    multi_style_templates: bool = False  # 是否用 verl/utils/multi_style_templates.py 里的 chat template 作为 fallback 多路 prompt（与 apply_icl 互斥）
    multi_style_template_names: Optional[Tuple[str, ...]] = None  # 可选，指定要使用的模板名称（顺序即 icl_idx 顺序）；为 None 时从 templates 字典按插入顺序取前 num_icl_examples 个
    use_lorem: bool = False  # 是否用 lorem ipsum 替换 ICL prompt 内容（对比实验用）
    lorem_in_middle: bool = False  # 与 use_lorem 类似，但 lorem 放在 user prompt 的 question 之后，system 只保留 _tail
    use_fake_sentence: bool = False  # 是否用 Faker 生成英文随机句子（四种替换方式最多开一个；词数规则同 lorem_word_*）
    use_random_token: bool = False  # 是否用 tokenizer 词表随机 token（非 special）decode 文本作 system 前缀
    use_random_ascii: bool = False  # 是否用随机可打印 ASCII 作 system 前缀（词数与同 lorem_word_*；字符数=词数*fertility）
    faker_locale: Optional[str] = None  # Faker 语言区域，如 en_US；None 为默认
    lorem_word_min: int = 100  # 纯随机 system 前缀模式（apply_icl=False 时：lorem/fake 为词数界；random_token 为采样 token 数界；random_ascii 为词数界）
    lorem_word_max: int = 300  # 纯随机 system 前缀模式词数/token 上界（含）；apply_icl=True 时仍用于与各替换方式对齐原 ICL 前缀的规则
    naive_resample: bool = False  # True 时 num_icl_examples 路 icl_* 与主 prompt 相同（需 apply_icl=False 且未启用任一 system 替换标志）
    general_exploration: bool = False  # True 时主 rollout 只生成 n-num_icl_examples*icl_rollout_n 条，其余槽位由 ICL 占位并在 fallback 中全部替换（需 apply_icl 或任一 use_lorem/use_fake_sentence/use_random_token/use_random_ascii）
    resample_temperature: Optional[float] = None  # ICL fallback 中 generate_sequences 的采样温度；None 表示不覆盖，沿用 worker.rollout 默认
    apply_ground_truth: bool = False  # 是否启用 ground truth 替换功能
    ground_truth_key: str = "solution"  # ground truth 在数据中的 key

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    filter_on_policy_token: bool = False
    """lower bound of the average IS_ratio of the suffix"""
    token_filter_lower_bound: float = 0.57
    """upper bound of the average IS_ratio of the suffix"""
    token_filter_upper_bound: float = 1.42  
    """file to store the interval of the on-policy IS_ratio"""
    interval_by_pos_file: Optional[str] = None
    """filter tokens in reference rollouts based on on-policy IS_ratio interval"""
    filter_by_suffix_is_ratio: bool = False
    """train on suffix of a rollout based on the average IS_ratio of the suffix"""
    suffix_is_ratio_lower_bound: float = 0.97
    """lower bound of the average IS_ratio of the suffix"""
    suffix_is_ratio_upper_bound: float = 1.3
    """upper bound of the average IS_ratio of the suffix"""
    policy_shaping: bool = False
    """apply policy shaping transformation x/(x+gamma) to reference samples' IS_ratio"""
    policy_shaping_gamma: float = 0.1
    """gamma parameter for policy shaping transformation"""
    # --- Safe Policy Loss 相关配置 ---
    use_safe_policy_loss: bool = False
    """use compute_safe_policy_loss instead of compute_policy_loss"""
    safe_policy_region_method: str = "suffix"
    """RL区域判定方法: 'suffix' (Suffix Mean动态阈值) 或 'sliding_window' (滑动窗口局部熵率)"""
    entropy_window_size: int = 64
    """sliding_window方法的窗口大小"""
    entropy_rate_threshold: Optional[float] = 0.2
    """Suffix Mean 过滤阈值，用于 Suffix Mean Reverse Search"""
    prefix_loss_type: str = "trsft"
    """前缀的处理方式: 'mask' 或 'trsft'"""
    trsft_alpha: float = 0.1
    """TRSFT 的 alpha 阈值"""
    train_with_icl_prompt: bool = False
    """when True, skip restoring original prompts after fallback/ICL replacement, so training uses the ICL/reference prompt as-is"""
    fallback_with_original_prompt: bool = False
    """when True, use original prompt (instead of ICL/hint prompt) during fallback generation"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

        if self.data.multi_style_templates and self.data.apply_icl:
            raise ValueError("multi_style_templates 与 apply_icl 互斥，不能同时为 True")

        # 自动将 num_icl_examples 对齐到实际启用的 multi_style_templates 数量
        # 不传 multi_style_template_names 时默认使用全部模板
        if self.data.multi_style_templates:
            from ..utils.multi_style_templates import templates as _multi_style_templates
            if self.data.multi_style_template_names:
                n_templates = len(self.data.multi_style_template_names)
                unknown = [n for n in self.data.multi_style_template_names if n not in _multi_style_templates]
                if unknown:
                    raise ValueError(
                        f"multi_style_template_names 中存在未知模板: {unknown}. "
                        f"可用模板: {list(_multi_style_templates.keys())}"
                    )
            else:
                n_templates = len(_multi_style_templates)
            if self.data.num_icl_examples != n_templates:
                print(
                    f"[multi_style_templates] overriding data.num_icl_examples "
                    f"{self.data.num_icl_examples} -> {n_templates} to match selected templates"
                )
                self.data.num_icl_examples = n_templates

        if self.data.general_exploration:
            if not (
                self.data.apply_icl
                or self.data.use_lorem
                or self.data.use_fake_sentence
                or self.data.use_random_token
                or self.data.use_random_ascii
                or self.data.multi_style_templates
                or self.data.lorem_in_middle
            ):
                raise ValueError(
                    "general_exploration 需要 apply_icl=True 或启用任一 "
                    "use_lorem / use_fake_sentence / use_random_token / use_random_ascii / "
                    "multi_style_templates / lorem_in_middle"
                )
            n_icl_slots = self.data.num_icl_examples * self.data.icl_rollout_n
            if n_icl_slots >= self.worker.rollout.n:
                raise ValueError(
                    "general_exploration 需要 num_icl_examples*icl_rollout_n < worker.rollout.n，"
                    f"当前为 {n_icl_slots} >= {self.worker.rollout.n}"
                )
            if self.data.naive_resample:
                raise ValueError("general_exploration 与 naive_resample 不能同时开启")

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
