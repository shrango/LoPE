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

import json
import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

try:
    import lorem as lorem_module
except ImportError:
    lorem_module = None

try:
    from faker import Faker as FakerFactory
except ImportError:
    FakerFactory = None

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        apply_icl: bool = False,
        icl_examples_path: Optional[str] = None,
        num_icl_examples: int = 5,
        use_lorem: bool = False,
        use_fake_sentence: bool = False,
        use_random_token: bool = False,
        use_random_ascii: bool = False,
        faker_locale: Optional[str] = None,
        lorem_word_min: int = 100,
        lorem_word_max: int = 300,
        naive_resample: bool = False,
        apply_ground_truth: bool = False,
        ground_truth_key: str = "solution",
        multi_style_templates: bool = False,
        multi_style_template_names: Optional[list[str]] = None,
        lorem_in_middle: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.apply_icl = apply_icl
        self.icl_examples = []
        self.num_icl_examples = num_icl_examples
        self.use_lorem = use_lorem
        self.use_fake_sentence = use_fake_sentence
        self.use_random_token = use_random_token
        self.use_random_ascii = use_random_ascii
        self.lorem_in_middle = lorem_in_middle
        self.lorem_word_min = lorem_word_min
        self.lorem_word_max = lorem_word_max
        self.multi_style_templates = multi_style_templates
        self.multi_style_template_chat_templates: list[str] = []
        self.multi_style_template_used_names: list[str] = []
        if multi_style_templates and (
            apply_icl or use_lorem or use_fake_sentence or use_random_token or use_random_ascii
            or naive_resample or lorem_in_middle
        ):
            raise ValueError(
                "multi_style_templates=True 与 apply_icl / use_lorem / use_fake_sentence / "
                "use_random_token / use_random_ascii / naive_resample / lorem_in_middle 互斥，不能同时启用"
            )
        _prompt_noise_flags = [use_lorem, use_fake_sentence, use_random_token, use_random_ascii, lorem_in_middle]
        if sum(bool(x) for x in _prompt_noise_flags) > 1:
            raise ValueError(
                "at most one of use_lorem, use_fake_sentence, use_random_token, use_random_ascii, "
                "lorem_in_middle can be True"
            )
        if use_lorem and lorem_module is None:
            raise ImportError("use_lorem=True requires the `python-lorem` package. Install via: pip install python-lorem")
        if lorem_in_middle and lorem_module is None:
            raise ImportError("lorem_in_middle=True requires the `python-lorem` package. Install via: pip install python-lorem")
        if use_fake_sentence and FakerFactory is None:
            raise ImportError("use_fake_sentence=True requires the `faker` package. Install via: pip install Faker")
        self._faker = FakerFactory(faker_locale) if use_fake_sentence else None
        if (
            use_lorem or use_fake_sentence or use_random_token or use_random_ascii or lorem_in_middle
        ) and lorem_word_min > lorem_word_max:
            raise ValueError(f"lorem_word_min ({lorem_word_min}) must be <= lorem_word_max ({lorem_word_max})")
        self._non_special_token_ids_cache: Optional[list[int]] = None
        self.naive_resample = naive_resample
        if naive_resample and (
            apply_icl or use_lorem or use_fake_sentence or use_random_token or use_random_ascii or lorem_in_middle
        ):
            raise ValueError(
                "naive_resample=True requires apply_icl=False and all prompt-replacement flags "
                "(use_lorem, use_fake_sentence, use_random_token, use_random_ascii, lorem_in_middle) to be False"
            )
        self.apply_ground_truth = apply_ground_truth
        self.ground_truth_key = ground_truth_key

        # 加载 multi_style_templates（来自 verl/utils/multi_style_templates.py）
        # multi_style_template_names 传入的是 templates 字典的 key，例如 ["abel", "simplerl", ...]
        # 不传入时默认使用全部 templates，并将 num_icl_examples 对齐到模板数量
        if multi_style_templates:
            from .multi_style_templates import templates as _multi_style_templates
            if multi_style_template_names is not None and len(multi_style_template_names) > 0:
                selected_names = list(multi_style_template_names)
            else:
                selected_names = list(_multi_style_templates.keys())
            missing = [n for n in selected_names if n not in _multi_style_templates]
            if missing:
                raise ValueError(
                    f"multi_style_template_names 中存在未知模板: {missing}. "
                    f"可用模板: {list(_multi_style_templates.keys())}"
                )
            # 自动对齐 num_icl_examples 到实际选中的模板数
            if self.num_icl_examples != len(selected_names):
                print(
                    f"[multi_style_templates] overriding num_icl_examples "
                    f"{self.num_icl_examples} -> {len(selected_names)} to match selected templates"
                )
                self.num_icl_examples = len(selected_names)
            self.multi_style_template_used_names = selected_names
            self.multi_style_template_chat_templates = [_multi_style_templates[n] for n in selected_names]

        # 读取 ICL examples
        if apply_icl:
            if icl_examples_path is None:
                raise ValueError("icl_examples_path must be provided when apply_icl is True")
            with open(icl_examples_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)
                    # 保留文件中的所有字段，支持多种格式：
                    # 格式1: method, problem, solution
                    # 格式2: Explanation, Details, problem, solution
                    # 格式3: Explanation, Details (无示例)
                    self.icl_examples.append(example)
            # 确保 examples 数量与配置一致
            assert len(self.icl_examples) == self.num_icl_examples, f"Expected {self.num_icl_examples} ICL examples, got {len(self.icl_examples)}"

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _format_prompt_wrap_with_middle(self, question_text: str, middle_text: str) -> str:
        """仅供 lorem_in_middle 使用。

        用 format_prompt 渲染时 `{{ content | trim }}` 会吃掉我们放在 content 末尾的换行符，
        导致 middle_text 与 format_prompt 的后缀（如 math.jinja 里的 "You FIRST ..."）之间
        没有换行。这里用唯一占位符反解析出 format_prompt 在 content 前后包了什么，
        再手工控制换行：
            <prefix><question>\n<middle>\n<suffix(lstrip)>
        若未配置 format_prompt，则直接返回 "<question>\n<middle>"。
        """
        core = f"{question_text}\n{middle_text}"
        if not self.format_prompt:
            return core
        unique_placeholder = "\x00__LOREM_IN_MIDDLE_CONTENT_PLACEHOLDER__\x00"
        fp_tpl = Template(self.format_prompt.strip())
        full = fp_tpl.render(content=unique_placeholder)
        idx = full.find(unique_placeholder)
        if idx < 0:
            # format_prompt 不包含 content（极少见），直接回退
            return core
        prefix = full[:idx]
        suffix = full[idx + len(unique_placeholder):]
        # suffix 形如 " You FIRST ..."（紧跟占位符）；左侧空白去掉，改用换行连接
        return prefix + core + "\n" + suffix.lstrip()

    def _non_special_token_ids(self) -> list[int]:
        if self._non_special_token_ids_cache is None:
            special = set(self.tokenizer.all_special_ids)
            self._non_special_token_ids_cache = sorted(
                tid for tid in self.tokenizer.get_vocab().values() if tid not in special
            )
            if not self._non_special_token_ids_cache:
                raise ValueError("Tokenizer has no non-special tokens; cannot use use_random_token")
        return self._non_special_token_ids_cache

    def _random_token_prefix(self, num_tokens: int) -> str:
        """从词表中随机采样 num_tokens 个非 special token，decode 为文本（不含 special tokens）。"""
        n = max(1, int(num_tokens))
        pool = self._non_special_token_ids()
        sampled = random.choices(pool, k=n)
        return self.tokenizer.decode(
            sampled, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _random_ascii_prefix(self, word_count: int) -> str:
        """词数为 word_count，fertility ~ U[3,5]，生成长度为 round(word_count * fertility) 的随机可打印 ASCII。"""
        wc = max(1, int(word_count))
        fertility = random.uniform(3.0, 5.0)
        n_chars = max(1, int(round(wc * fertility)))
        return "".join(chr(random.randint(32, 126)) for _ in range(n_chars))

    def _fake_sentence_prefix(self, word_count: Union[int, tuple[int, int]]) -> str:
        """使用 Faker 生成随机英文句子（sentence），词数规则与 lorem 分支一致。"""
        assert self._faker is not None
        if isinstance(word_count, tuple):
            lo, hi = int(word_count[0]), int(word_count[1])
            n = random.randint(lo, hi)
        else:
            n = int(word_count)
        n = max(1, n)
        return self._faker.sentence(nb_words=n, variable_nb_words=False).strip()

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
    
    def _build_raw_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        # print(f"example keys:{example.keys()}")
        messages = self._build_messages(example)
        raw_messages = self._build_raw_messages(example)
        example.pop(self.prompt_key, None)
        
        # 初始化特殊功能标志，图像和视频模式下不支持这些功能
        has_icl_payload = False

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            # 处理 ICL (In-Context Learning) prompts / 纯 lorem 对照（字段名仍为 icl_* 以便与下游处理一致）
            # 每个槽位对应一个 prompt，共 num_icl_examples 个
            # 命名格式: icl_{idx}_input_ids, icl_{idx}_attention_mask, icl_{idx}_position_ids, raw_icl_{idx}_prompt_ids
            has_icl_payload = (
                self.apply_icl
                or self.use_lorem
                or self.use_fake_sentence
                or self.use_random_token
                or self.use_random_ascii
                or self.naive_resample
                or self.multi_style_templates
                or self.lorem_in_middle
            )
            icl_data = {}  # 存储所有 ICL / lorem / naive_resample prompt 的数据
            if self.apply_icl:
                question_text = raw_messages[-1]['content']
                
                # 使用 format_prompt 处理 user prompt（与 _build_messages 保持一致）
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                
                for icl_idx, icl_example in enumerate(self.icl_examples):
                    # 构建 ICL prompt 格式
                    # instruction 和 example 部分放到 system prompt，实际问题放到 user prompt
                    _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                    # _tail = ""
                    if "Explanation" in icl_example:
                        if "problem" in icl_example:
                            system_prompt = (
                                f"Solve the following problem using the **{icl_example['method']}** strategy.\n"
                                f"### Strategy Definition\n"
                                f"{icl_example['Explanation']}\n"
                                f"### Execution Guidelines\n"
                                f"{icl_example['Details']}\n"
                                f"### Demonstration (Example of this strategy)\n"
                                f"**Question:**\n{icl_example['problem']}\n"
                                f"**Answer:**\n{icl_example['solution']}\n"
                                f"{_tail}"
                            )
                        else:
                            system_prompt = (
                                f"Solve the following problem using the **{icl_example['method']}** strategy.\n"
                                f"### Strategy Definition\n"
                                f"{icl_example['Explanation']}\n"
                                f"### Execution Guidelines\n"
                                f"{icl_example['Details']}\n"
                                f"{_tail}"
                            )
                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    else:
                        system_prompt = (
                            f"Use **{icl_example['method']}** method to solve questions.\n"
                            f"Example:\n"
                            f"Question: {icl_example['problem']}\n"
                            f"Answer: {icl_example['solution']}\n"
                            f"{_tail}"
                        )
                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]

                    if self.use_lorem:
                        system_prefix = system_prompt[: system_prompt.rfind(_tail)]
                        prefix_word_count = len(system_prefix.split())
                        lorem_system_prefix = lorem_module.get_word(count=prefix_word_count) + "\n"
                        system_prompt = lorem_system_prefix + _tail

                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    elif self.use_fake_sentence:
                        system_prefix = system_prompt[: system_prompt.rfind(_tail)]
                        prefix_word_count = len(system_prefix.split())
                        fake_system_prefix = self._fake_sentence_prefix(prefix_word_count) + "\n"
                        system_prompt = fake_system_prefix + _tail

                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    elif self.use_random_token:
                        system_prefix = system_prompt[: system_prompt.rfind(_tail)]
                        prefix_token_count = len(
                            self.tokenizer.encode(system_prefix, add_special_tokens=False)
                        )
                        token_system_prefix = self._random_token_prefix(prefix_token_count) + "\n"
                        system_prompt = token_system_prefix + _tail
                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    elif self.use_random_ascii:
                        system_prefix = system_prompt[: system_prompt.rfind(_tail)]
                        prefix_word_count = len(system_prefix.split())
                        ascii_system_prefix = self._random_ascii_prefix(prefix_word_count) + "\n"
                        system_prompt = ascii_system_prefix + _tail
                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    elif self.lorem_in_middle:
                        # 与 use_lorem 的词数对齐策略一致：lorem 词数 = ICL system prefix 的词数
                        # 但 lorem 放在 user prompt 中 question 之后，system 只保留 _tail
                        # 使用 _format_prompt_wrap_with_middle 保证 lorem 之后有换行，
                        # 而不会被 format_prompt 的 `| trim` 吃掉
                        system_prefix = system_prompt[: system_prompt.rfind(_tail)]
                        prefix_word_count = len(system_prefix.split())
                        lorem_middle = lorem_module.get_word(count=prefix_word_count)
                        new_user_prompt = self._format_prompt_wrap_with_middle(
                            question_text=question_text,
                            middle_text=lorem_middle,
                        )
                        system_prompt = _tail
                        messages_with_icl = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": new_user_prompt},
                        ]
                    # 这么写就兼容了下面两种形式
                    # icl_prompt_content = (
                    #     f"Use **{icl_example['method']}** method to solve questions.\n"
                    #     f"Example:\n"
                    #     f"Question: {icl_example['problem']}\n"
                    #     f"Answer: {icl_example['solution']}\n\n"
                    #     f"Question: {question_text}\n"
                    #     f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    # )
                    # icl_prompt_content = (
                    #     f"Use **{icl_example['method']}** method to solve questions.\n"
                    #     f"Example:\n"
                    #     f"Question: {icl_example['problem']}\n"
                    #     f"Answer: {icl_example['solution']}\n\n"
                    #     f"Question: {question_text}\n"
                    #     f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}.\n"
                    # )
                    # messages_with_icl = [{"role": "user", "content": icl_prompt_content}]

                    icl_prompt = self.tokenizer.apply_chat_template(messages_with_icl, add_generation_prompt=True, tokenize=False)
                    
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.use_lorem:
                # apply_icl=False：无真实 ICL 示例，仅用 lorem 填充 system 前缀（词数在 [lorem_word_min, lorem_word_max] 内随机）
                question_text = raw_messages[-1]["content"]
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                # _tail = ""
                for icl_idx in range(self.num_icl_examples):
                    lorem_system_prefix = (
                        lorem_module.get_word(count=(self.lorem_word_min, self.lorem_word_max)) + "\n"
                    )
                    system_prompt = lorem_system_prefix + _tail
                    messages_with_icl = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_with_icl, add_generation_prompt=True, tokenize=False
                    )
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.lorem_in_middle:
                # 与 use_lorem 类似，但 lorem 放在 user prompt 中 question 之后；system 只保留 _tail
                # 词数与 use_lorem 一致：[lorem_word_min, lorem_word_max] 内随机
                # 使用 _format_prompt_wrap_with_middle 保证 lorem 之后有换行，
                # 而不会被 format_prompt 的 `| trim` 吃掉
                question_text = raw_messages[-1]["content"]
                _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                system_prompt = _tail
                for icl_idx in range(self.num_icl_examples):
                    lorem_middle = lorem_module.get_word(count=(self.lorem_word_min, self.lorem_word_max))
                    user_prompt = self._format_prompt_wrap_with_middle(
                        question_text=question_text,
                        middle_text=lorem_middle,
                    )
                    messages_with_icl = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_with_icl, add_generation_prompt=True, tokenize=False
                    )
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.use_fake_sentence:
                # apply_icl=False：与纯 lorem 相同，仅将 system 前缀换为 Faker 随机句子
                question_text = raw_messages[-1]["content"]
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                for icl_idx in range(self.num_icl_examples):
                    fake_system_prefix = (
                        self._fake_sentence_prefix((self.lorem_word_min, self.lorem_word_max)) + "\n"
                    )
                    system_prompt = fake_system_prefix + _tail
                    messages_with_icl = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_with_icl, add_generation_prompt=True, tokenize=False
                    )
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.use_random_token:
                question_text = raw_messages[-1]["content"]
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                for icl_idx in range(self.num_icl_examples):
                    n_tok = random.randint(self.lorem_word_min, self.lorem_word_max)
                    token_system_prefix = self._random_token_prefix(n_tok) + "\n"
                    system_prompt = token_system_prefix + _tail
                    messages_with_icl = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_with_icl, add_generation_prompt=True, tokenize=False
                    )
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.use_random_ascii:
                question_text = raw_messages[-1]["content"]
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                _tail = "Please reason step by step, and put your final answer within \\boxed{}."
                for icl_idx in range(self.num_icl_examples):
                    wc = random.randint(self.lorem_word_min, self.lorem_word_max)
                    ascii_system_prefix = self._random_ascii_prefix(wc) + "\n"
                    system_prompt = ascii_system_prefix + _tail
                    messages_with_icl = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_with_icl, add_generation_prompt=True, tokenize=False
                    )
                    icl_model_inputs = self.tokenizer([icl_prompt], add_special_tokens=False, return_tensors="pt")
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

            elif self.naive_resample:
                # 与主样本相同的 prompt，复制 num_icl_examples 份（tensor 用 clone，避免后续 postprocess 原地写共享存储）
                for icl_idx in range(self.num_icl_examples):
                    icl_data[icl_idx] = {
                        "input_ids": input_ids.clone(),
                        "attention_mask": attention_mask.clone(),
                        "prompt": prompt,
                    }

            elif self.multi_style_templates:
                # 使用 verl/utils/multi_style_templates.py 中的多套 chat template
                # 每个模板生成一路 icl_* prompt，与原始 ICL 路径在下游逻辑上完全一致
                question_text = raw_messages[-1]["content"]
                user_prompt = question_text
                if self.format_prompt:
                    format_prompt = Template(self.format_prompt.strip())
                    user_prompt = format_prompt.render(content=question_text)
                messages_for_template = [{"role": "user", "content": user_prompt}]
                for icl_idx in range(self.num_icl_examples):
                    chat_template_str = self.multi_style_template_chat_templates[icl_idx]
                    icl_prompt = self.tokenizer.apply_chat_template(
                        messages_for_template,
                        chat_template=chat_template_str,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    icl_model_inputs = self.tokenizer(
                        [icl_prompt], add_special_tokens=False, return_tensors="pt"
                    )
                    icl_input_ids = icl_model_inputs.pop("input_ids")[0]
                    icl_attention_mask = icl_model_inputs.pop("attention_mask")[0]
                    icl_data[icl_idx] = {
                        "input_ids": icl_input_ids,
                        "attention_mask": icl_attention_mask,
                        "prompt": icl_prompt,
                    }

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
            if has_icl_payload:
                for icl_idx in icl_data:
                    icl_data[icl_idx]["position_ids"] = torch.clip(icl_data[icl_idx]["attention_mask"].cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # 后处理 ICL / lorem 数据
        if has_icl_payload:
            for icl_idx in range(self.num_icl_examples):
                icl_input_ids, icl_attention_mask, icl_position_ids = VF.postprocess_data(
                    input_ids=icl_data[icl_idx]["input_ids"],
                    attention_mask=icl_data[icl_idx]["attention_mask"],
                    position_ids=icl_data[icl_idx]["position_ids"],
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                raw_icl_prompt_ids = self.tokenizer.encode(icl_data[icl_idx]["prompt"], add_special_tokens=False)
                if len(raw_icl_prompt_ids) > self.max_prompt_length:
                    if self.truncation == "left":
                        raw_icl_prompt_ids = raw_icl_prompt_ids[-self.max_prompt_length :]
                    elif self.truncation == "right":
                        raw_icl_prompt_ids = raw_icl_prompt_ids[: self.max_prompt_length]
                    elif self.truncation == "error":
                        raise RuntimeError(f"ICL Prompt length {len(raw_icl_prompt_ids)} is longer than {self.max_prompt_length}.")
                
                # 更新 icl_data 中的处理后数据
                icl_data[icl_idx]["input_ids"] = icl_input_ids
                icl_data[icl_idx]["attention_mask"] = icl_attention_mask
                icl_data[icl_idx]["position_ids"] = icl_position_ids
                icl_data[icl_idx]["raw_prompt_ids"] = raw_icl_prompt_ids

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        
        # 处理 ground_truth_text：从 ground_truth_key 获取数据并转换为 text
        # 格式为 [{"content": gt_text, "role": "assistant"}]
        # 处理 ground_truth_text：仅当 apply_ground_truth=True 时才从 ground_truth_key 获取文本
        if self.apply_ground_truth and self.ground_truth_key in example:
            example["ground_truth_text"] = example.pop(self.ground_truth_key)
        # 添加 ICL 数据到 example
        # 命名格式: icl_{idx}_input_ids, icl_{idx}_attention_mask, icl_{idx}_position_ids, raw_icl_{idx}_prompt_ids
        # idx 从 0 到 num_icl_examples-1
        if has_icl_payload:
            for icl_idx in range(self.num_icl_examples):
                example[f"icl_{icl_idx}_input_ids"] = icl_data[icl_idx]["input_ids"]
                example[f"icl_{icl_idx}_attention_mask"] = icl_data[icl_idx]["attention_mask"]
                example[f"icl_{icl_idx}_position_ids"] = icl_data[icl_idx]["position_ids"]
                example[f"raw_icl_{icl_idx}_prompt_ids"] = icl_data[icl_idx]["raw_prompt_ids"]
        
        return example
