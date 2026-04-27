# LoPE: Lorem Perturbation for Exploration

> **Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration**
>
> [Langlin Huang](https://shrango.github.io/), [Chengsong Huang](https://chengsong-huang.github.io/), [Jinyuan Li](https://sites.google.com/view/jinyuanli), [Donghong Cai](https://ilikevegetable.github.io/), [Yuyi Yang](https://yyuyi.github.io/), [Jiaxin Huang](https://teapot123.github.io/)
>
> [HINT Lab](https://teapot123.github.io/application/), Washington University in St. Louis
>
> 📄 [Paper (arXiv)]() · 💻 [Code](https://github.com/shrango/LoPE)

---

## 🚀 Overview

**LoPE** is a simple yet effective resampling strategy for GRPO-style reinforcement learning that breaks through the **zero-advantage problem** — the situation where all sampled rollouts for a hard question fail, the relative advantage collapses to zero, and the training signal is wasted.

Instead of just throwing more compute at logit-space exploration (e.g., higher temperature), **LoPE prepends a randomly generated *Lorem Ipsum* sequence to the prompt** before resampling. This semantically neutral, prompt-space perturbation shifts the model's output distribution just enough to unlock orthogonal reasoning trajectories — without distorting its understanding of the question.

<p align="center">
  <img src="assets/figure1_overview.png" width="85%" alt="LoPE Overview"/>
</p>

> **Figure 1.** During the standard rollout phase, if all *G* responses fail, LoPE prepends a Lorem Ipsum sequence to the prompt and resamples *G′* responses. Successful responses are regrouped with the original failed ones to form a mixed batch of size *G* for policy update.

---

## ✨ Key Findings

- **🎯 Zero-Advantage Recovery.** When all initial rollouts fail, LoPE-perturbed resampling recovers correct trajectories that neither naive resampling nor high-temperature sampling can find.
- **🧭 Orthogonal Exploration.** On a hard 352-question subset, Lorem-perturbed prompts independently solve **50 unique questions** that other methods miss (see Figure 2).
- **🧬 Controlled Perplexity is Key.** Among all tested perturbations (random English, ASCII, tokens, multi-style), Lorem Ipsum's perplexity is closest to natural language — strong enough to induce exploration, gentle enough not to corrupt question semantics.
- **📈 Consistent Gains.** Average improvement of **+2.79** on Qwen3-1.7B-Base, **+4.62** on Qwen3-4B-Base, and **+6.20** on Qwen2.5-Math-7B across five math benchmarks.

<p align="center">
  <img src="assets/figure2_venn.png" width="85%" alt="Venn Diagram of Solved Questions"/>
</p>

> **Figure 2.** Venn diagrams of questions successfully resolved (Pass@8) by naive prompting, high-temperature sampling, and Lorem perturbation. LoPE unlocks reasoning paths that pure logit-space methods cannot reach.

---

## 🧠 Why Lorem Ipsum?

We need a perturbation that is **structurally similar to natural language** but **semantically empty** — so it doesn't leak hints or distort the question. Lorem Ipsum fits perfectly:

| Perturbation Type | Mean Perplexity | Effect |
|---|---|---|
| Question Text (reference) | 4.82 | — |
| **Lorem Ipsum** ✅ | **25.12** | Closest to natural language; preserves comprehension |
| Random ASCII | 492.9 | Higher noise, weaker gains |
| Fake English | 2,430 | Breaks N-gram transitions |
| Random Tokens ❌ | 456,914.9 | Linguistic structure collapses |

The key insight: **moderate perturbation (Lorem Ipsum) increases response entropy and promotes exploration without harming the input representation.** Excessively high-perplexity perturbations corrupt the model's understanding of the question itself.

---

## 📊 Main Results

Results on five math reasoning benchmarks (MATH-500, GSM8K, AMC, AIME24, AIME25):

| Model & Method | MATH-500 | GSM8K | AMC | AIME24 | AIME25 | **Avg.** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Qwen3-1.7B-Base** | 63.40 | 76.92 | 26.87 | 5.33 | 2.00 | 34.90 |
| &nbsp;&nbsp;+ GRPO | 64.20 | 82.71 | 27.61 | 6.15 | 4.47 | 37.03 |
| &nbsp;&nbsp;+ Resample w/ Naive Prompt | 67.00 | 82.18 | 28.36 | 8.70 | 4.58 | 38.16 |
| &nbsp;&nbsp;+ Resample w/ **LoPE** | 68.00 | 83.55 | 33.58 | 7.97 | 5.83 | **39.79** |
| &nbsp;&nbsp;+ **LoPE** + Training Signal Shaping | 68.80 | 82.94 | 32.84 | 8.80 | 5.73 | **39.82** |
| **Qwen3-4B-Base** | 65.80 | 82.71 | 32.84 | 9.38 | 7.24 | 39.59 |
| &nbsp;&nbsp;+ GRPO | 77.80 | 91.74 | 47.76 | 16.41 | 13.12 | 49.37 |
| &nbsp;&nbsp;+ Resample w/ Naive Prompt | 79.80 | 92.87 | 45.52 | 14.90 | 11.67 | 48.95 |
| &nbsp;&nbsp;+ Resample w/ **LoPE** | 85.40 | 92.95 | 52.99 | 19.01 | 13.85 | **52.84** |
| &nbsp;&nbsp;+ **LoPE** + Training Signal Shaping | 82.60 | 92.95 | 58.21 | 19.90 | 16.27 | **53.99** |
| **Qwen2.5-Math-7B** | 52.80 | 65.50 | 35.40 | 12.90 | 7.90 | 34.90 |
| &nbsp;&nbsp;+ GRPO | 78.00 | 85.06 | 47.76 | 17.66 | 9.90 | 47.68 |
| &nbsp;&nbsp;+ Resample w/ Naive Prompt | 78.20 | 83.02 | 50.00 | 17.19 | 9.17 | 47.52 |
| &nbsp;&nbsp;+ Resample w/ **LoPE** | 77.40 | 86.35 | 47.01 | 15.31 | 10.52 | 47.32 |
| &nbsp;&nbsp;+ **LoPE** + Training Signal Shaping | 81.80 | 90.30 | 61.19 | 19.58 | 16.51 | **53.88** |

### Comparison of Prompt Perturbation Strategies (Qwen3-1.7B-Base)

| Method | MATH-500 | GSM8K | AMC | AIME24 | AIME25 | **Avg.** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GRPO | 64.20 | 82.71 | 27.61 | 6.15 | 4.47 | 37.03 |
| w/ Naive Prompt | 67.00 | 82.18 | 28.36 | 8.70 | 4.58 | 38.16 |
| w/ Naive Prompt (Temp=1.2) | 64.40 | 82.87 | 31.34 | 8.65 | 4.48 | 38.35 |
| w/ Multi-style Prompt | 66.40 | 82.26 | 32.09 | 8.23 | 5.10 | 38.82 |
| w/ Fake English | 67.00 | 82.94 | 29.10 | 7.29 | 4.11 | 38.09 |
| w/ Random ASCII | 65.00 | 82.87 | 30.60 | 7.60 | 5.00 | 38.15 |
| w/ Random Token | 63.40 | 81.20 | 30.60 | 7.92 | 3.44 | 37.31 |
| **w/ LoPE** | 68.00 | 83.55 | 33.58 | 7.97 | 5.83 | **39.79** |
| w/ LoPE after question | 68.20 | 83.55 | 31.34 | 8.91 | 6.41 | 39.68 |

---

## 🛠️ Method at a Glance

LoPE follows the standard GRPO training loop with three modifications when all initial rollouts fail:

1. **Rollout with Perturbation.** Prepend a random Lorem Ipsum sequence *δ* (100–300 tokens) to the original prompt *p*, then sample *G′* additional responses from $π_{θ_{old}}(o' | δ ⊕ p, q)$.
2. **Regroup Responses.** Replace failed rollouts with successful resampled ones, keeping the group size at *G* and at least one incorrect response so advantages remain non-zero.
3. **Advantage Estimation with Importance Correction.** Convert resampled responses into pseudo rollouts paired with the naive prompt, and correct the distribution shift via:

$$\rho_{i,t} = \frac{\pi_\theta(o'_{i,t} \mid p, q, o'_{i,<t})}{\pi_{\theta_\text{old}}(o'_{i,t} \mid \delta \oplus p, q, o'_{i,<t})}$$

4. **(Optional) Training Signal Shaping.** Reshape the importance ratio to `ρ' = ρ / (ρ + 0.1)` to amplify low-probability tokens corresponding to critical reasoning steps, and compute group advantage over the full *G + G′* set while restricting gradients to the *G* selected responses.
5. **No KL Regularization.** KL constraints counteract the broader exploration LoPE aims to promote.

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/shrango/LoPE.git
cd LoPE

# Install dependencies
pip install -r requirements.txt

# Install python-lorem for perturbation generation
pip install python-lorem
```

Our implementation is built on top of [EasyR1](https://github.com/hiyouga/EasyR1).

---

## 🏃 Quick Start

### Training

```bash
bash scripts/train_lope.sh \
    --model Qwen/Qwen3-1.7B-Base \
    --dataset openr1-math-46k-8192 \
    --group_size 8 \
    --resample_size 24
```

### Evaluation

```bash
bash scripts/eval.sh --model_path checkpoints/lope-qwen3-1.7b
```

We use [EvalScope](https://github.com/modelscope/evalscope) with sampling temperature 0.6 and top-p 0.95. We report Acc@1 for MATH-500, GSM8K, and AMC, and Mean@32 for AIME24 and AIME25.

---

## 🔍 Hyperparameters

| Parameter | Value |
|---|---|
| Group size *G* | 8 |
| Resample size *G′* | 24 |
| Rollout temperature | 1.0 |
| Eval temperature | 0.6 |
| Eval top-p | 0.95 |
| Lorem sequence length | 100–300 tokens (uniform) |
| Max response length | 8,192 tokens |
| Max input length | 2,048 tokens |
| KL coefficient | 0 (removed) |

A short boundary instruction `\nPlease reason step by step, and put your final answer within \boxed{}.` is appended after the Lorem Ipsum sequence to prevent the model from generating corrupted outputs.

---

## 📌 Citation

If you find LoPE useful in your research, please cite:

```bibtex
@article{huang2026lope,
  title   = {Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration},
  author  = {Huang, Langlin and Huang, Chengsong and Li, Jinyuan and Cai, Donghong and Yang, Yuyi and Huang, Jiaxin},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements

Our code is built upon [EasyR1](https://github.com/hiyouga/EasyR1). We thank the authors of [Qwen](https://github.com/QwenLM/Qwen3), [OpenR1-Math](https://huggingface.co/datasets/open-r1), and [python-lorem](https://github.com/JarryShaw/lorem) for their open-source contributions.
