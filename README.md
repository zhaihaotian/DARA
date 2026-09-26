# Make Sparse Rewards Count: Density-Aware Reward Aggregation for Multi-Reward RL

<p align="center">
🛠️ <a href="./tool_calling">Tool calling</a> &nbsp;|&nbsp; 🧮 <a href="./math">Math reasoning</a>
</p>

This repository contains the implementation of **DARA** (**D**ensity-**A**ware **R**eward **A**ggregation), a multi-reward reinforcement learning method that calibrates reward contributions using their active-group densities. It includes training and evaluation for tool calling and mathematical reasoning, with **DARA-Asym** as our main method and **DARA-Sym** as the symmetric variant.

## Method

Reward-wise normalization preserves comparisons within each rollout group, but rewards that vary in fewer groups still supply less learning signal across the batch. Our analysis quantifies this: under ideal group normalization, a reward's advantage energy is $E_k=B\pi_k(G-1)$, where $\pi_k$ is its active-group density. This relation yields an inverse-square-root density correction for balancing reward contributions. DARA recomputes these weights on each batch and caps amplification at 5. The paper develops the derivation, its connection to policy-gradient signals, and energy bounds for asymmetric calibration.

**DARA-Asym** applies the extra amplification to positive advantages, strengthening favorable comparisons from sparsely active rewards while retaining negative coefficients at their original scale before aggregation. **DARA-Sym** applies the same density weights to both signs.

## ⚙️ Implementation

We implement DARA in the GDPO training pipeline: compute reward-wise group advantages, calibrate them by density, then aggregate and normalize them for the clipped policy objective.

```python
# A[k]: group-normalized advantages of reward k, shape (B, G)
pi = [(A[k].abs().sum(dim=1) > tau).float().mean() for k in range(K)]
pi_ref = max(pi)
w = [min(w_max, (pi_ref / p) ** 0.5) if p > 0 else 1.0 for p in pi]

if variant == "asym":   # DARA-Asym
    S = sum(A[k] + (w[k] - 1) * A[k].clamp(min=0) for k in range(K))
else:                   # DARA-Sym
    S = sum(w[k] * A[k] for k in range(K))

adv = batch_normalize(S)  # use the GDPO normalization of the task's training stack
```

The implementation records reward densities, active-group counts, and calibration weights throughout training. The supplied CPU checks verify the density–energy relation, the positive-only transformation, and recovery of GDPO when all weights equal one. See the [tool-calling estimators](./tool_calling/docs/ALGORITHMS.md) and the [math estimator checks](./math/tools/check_advantages.py) for the implementation details.

## 📁 What's in this repo

We ran the two tasks on two separate verl-based training stacks, so the repo has two self-contained folders:

- [`tool_calling/`](./tool_calling): tool calling following ToolRL and GDPO, with Qwen2.5-1.5B/3B-Instruct, evaluated on BFCL-v4. Built on the verl training code released with GDPO.
- [`math/`](./math): math reasoning with a correctness reward and a 4,000-token length reward, with DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3-4B-Instruct/Thinking-2507, evaluated on MATH-500, AIME 2024, AMC, Minerva and OlympiadBench.

Both folders implement the same set of methods, so every comparison differs only in advantage aggregation: GRPO, GDPO, DVAO, GD²PO-Hard, DARA-Asym and DARA-Sym.

## 🚀 Tool calling

We follow the GDPO tool-calling setup:
- **Data and rewards.** ToolRL's 4k training set, with a binary format reward and a correctness reward in [−3, 3].
- **Training.** 100 steps, with 512 prompts × 4 rollouts per step. Each run takes one node with 4× A100 40GB, about 2.5 hours for 1.5B.

```bash
cd tool_calling
bash environments/install_training.sh dara-train && conda activate dara-train
python scripts/fetch_data.py

python training/launch.py --method dara_asym --seed 0 --model-size 1.5b \
  --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-dara_asym-g4-s0
```

The rollout-group study (G = 4/8/16/32 at a fixed 2,048 responses per step) and the three-reward setting (adding a reward for a `<think>` block of at most 16 words) are listed in [`configs/`](./tool_calling/configs) and can be launched run by run. Evaluation runs BFCL-v4 on one GPU per checkpoint. See [`tool_calling/README.md`](./tool_calling/README.md) for all of it.

## 🚀 Math reasoning

We follow GDPO §4.2:
- **Data and rewards.** DeepScaleR-Preview, with a binary correctness reward and a binary length reward (response ≤ 4,000 tokens).
- **Training.** 512 prompt groups × 16 rollouts, DAPO-style dynamic sampling, and an 8,000-token response limit. Each run uses one 8-GPU node.

```bash
cd math
bash scripts/bootstrap_runtime.sh --prefix /path/to/dara-math-v07
cp configs/machine.example.json my_machine.json   # point it to your venv, runtime, models and output dir

python3 scripts/train.py --model r1-1.5b --method dara_asym --seed 0 --profile my_machine.json --execute
```

Use `--model qwen3-instruct`, `qwen3-thinking` or `r1-7b` for the other backbones. Use `--method static_asym` / `static_sym` for the fixed-weight ablation (correctness weight 1, length weight 5). The benchmark evaluation script and the length-compliance threshold tool are described in [`math/README.md`](./math/README.md).

## 🙏 Acknowledgements

This work builds directly on [GDPO](https://github.com/NVlabs/GDPO). Our tool-calling code starts from their verl implementation and training setup, and we thank the authors for releasing it. We also thank the teams behind [verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), [ToolRL](https://github.com/qiancheng0/ToolRL) (training data and prompt format), the [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla) and [DeepScaleR](https://github.com/agentica-project/rllm). The baseline implementations follow the papers of DVAO and GD²PO.

## License

Apache-2.0. Third-party code keeps its original license; see [`tool_calling/THIRD_PARTY.md`](./tool_calling/THIRD_PARTY.md) and [`math/third_party_dependency.LICENSE`](./math/third_party_dependency.LICENSE).
