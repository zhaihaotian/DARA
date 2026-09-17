# 论文实验设置与评测协议

## Experimental setup

We conduct tool-calling experiments with Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct using a verl-based training implementation following GDPO. All methods use the same ToolRL rlla_4k split, consisting of 3,920 training examples and 80 validation examples. Each run is trained for 100 steps on four NVIDIA A100 40GB GPUs with FSDP and vLLM rollout generation. At each training step, we sample 512 prompts and generate four responses per prompt, yielding 2,048 responses. The learning rate is $10^{-6}$, and the maximum prompt and response lengths are 2,048 and 1,024 tokens, respectively. We use correctness and format rewards, with ranges $[-3,3]$ and $\{0,1\}$. DARA uses a maximum density-calibration weight of 5. We evaluate the final model at step 100. Our main comparisons of GRPO, GDPO, and DARA use five training seeds, $\{0,1,2,4,5\}$, for both model sizes. Additional baseline experiments on the 1.5B model use three seeds, $\{0,1,2\}$. Results are reported as the mean and sample standard deviation across training seeds.

## Evaluation protocol

We evaluate tool-calling performance on BFCL-V3 and BFCL-V4 using the BFCL checkers and a shared ToolRL prompting and response-parsing interface. For BFCL-V3, we report Non-Live AST and Live AST accuracy over 1,150 and 1,351 examples, respectively. For BFCL-V4, we evaluate these two subsets together with 800 Multi-Turn examples. Non-Live AST accuracy is the mean of Simple, Multiple, Parallel, and Parallel Multiple accuracy, where Simple is averaged equally over Python, Java, and JavaScript. Live AST accuracy is computed over all examples in its four categories. Multi-Turn accuracy is the mean over Base, Missing Function, Missing Parameter, and Long Context, with 200 examples per category. We report the arithmetic mean of the three BFCL-V4 subset accuracies as Average.

We also measure Format, which evaluates whether an assistant output follows the prescribed response structure. An output receives a score of one when its complete text matches an accepted sequence of `<think>`, `<tool_call>`, and `<response>` tags, including the required ordering, closing tags, and newlines; otherwise, it receives zero. For cases with multiple assistant outputs, we first average the output-level scores within each case and then average across cases within each category. Subset scores use the same aggregation weights as accuracy. Average Format is the arithmetic mean of the Non-Live, Live, and Multi-Turn format scores. Accuracy and Format are reported as percentages.

We generate one trajectory per evaluation case, using temperature 0 and top-p 1 for BFCL-V3, and temperature 0.6 and top-p 0.95 for BFCL-V4. The maximum generation budgets are 4,096 and 8,192 tokens, respectively, with a context limit of 32,768 tokens. The inference seed is fixed to 0 for all methods. Evaluation uses vLLM 0.11.0 with tensor parallelism 1 and bfloat16 precision.

## Table caption

Tool-calling performance on BFCL-V3 and BFCL-V4. Average is the arithmetic mean of BFCL-V4 Non-Live AST, Live AST, and Multi-Turn accuracy. Average Format applies the same subset averaging to format scores. Values are percentages, reported as mean $\pm$ sample standard deviation across five training seeds for GRPO, GDPO, and DARA, and three seeds for the additional 1.5B baselines.

## Appendix: training configuration

| Parameter | Value |
|---|---|
| Backbone | Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct |
| Hardware per run | 4 × NVIDIA A100 40GB |
| Framework | Haotian verl/GDPO implementation |
| Python / PyTorch | 3.10.21 / 2.4.0+cu121 |
| Training vLLM / Ray / Transformers | 0.6.3 / 2.10.0 / 4.47.1 |
| FlashAttention / xFormers | 2.6.3 / 0.0.27.post2 |
| Training steps | 100 |
| Validation | Before training and every 10 steps |
| Final evaluation checkpoint | Step 100 |
| Prompts per training step | 512 |
| Responses per prompt | 4 |
| Responses per training step | 2,048 |
| Responses per PPO minibatch | 512 |
| Optimizer updates per training step | 4 |
| PPO epochs / clip ratio | 1 / 0.2 |
| Learning rate / gradient clipping | $10^{-6}$ / 1.0 |
| Entropy coefficient | 0.001 |
| Reward-KL coefficient | 0.001 |
| Training generation temperature / top-p / top-k | 1 / 1 / -1 |
| Training prompt / response limit | 2,048 / 1,024 tokens |
| Dynamic batching token budget per GPU | 16,384 |
| Rollout tensor parallelism / GPU memory utilization | 1 / 0.6 |
| Actor parameter / gradient / optimizer offload | False / False / False |
| Reference parameter offload | True |
| Gradient checkpointing / padding removal | True / True |
| DARA weight cap | 5 |
| Training/data seeds | 0, 1, 2, 4, 5 |
| Training rollout engine seed | 0 |

The policy loss uses the Haotian implementation's dynamic microbatch accumulation: each microbatch token-mean loss is divided by the nominal accumulation factor of two. GRPO computes advantages from the summed reward after the reward-KL penalty. GDPO and DARA compute advantages from the individual reward channels and apply token-level batch whitening. The entropy coefficient is 0.001 for all methods. For 3B runs, the PyTorch allocator uses `expandable_segments:True`.

## Appendix: evaluation details

The evaluator is fixed to `bfcl-eval==2025.8.6.2` for BFCL-V3 and gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` for BFCL-V4. The category sizes and exact aggregation formulas are provided in [EVALUATION.md](EVALUATION.md).

The accepted output structures are:

| Output type | Structure |
|---|---|
| Text response | `<think>...</think>\n<response>...</response>` |
| Tool call | `<think>...</think>\n<tool_call>\n...\n</tool_call>` |
| Tool call and text response | `<think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>` |

Matching is anchored to the complete assistant output after Qwen assistant-marker extraction and whitespace trimming. The tool-call and response tags must each occur exactly once in the corresponding structure. A case with no assistant output receives a format score of zero.

For a case $i$ containing $m_i$ assistant outputs with binary format scores $f_{ij}$, we compute $F_i=m_i^{-1}\sum_{j=1}^{m_i}f_{ij}$. Category format scores are the mean of $F_i$ across cases. Non-Live format uses the same nested macro averaging as Non-Live accuracy; Live format is weighted by category sample counts; Multi-Turn format is averaged equally over its four categories. Average Format is the mean of these three subset scores.

## 后续消融的协议文本

以下为已准备的两组消融方案，实验清单与执行状态见 [HANDOFF.md](HANDOFF.md)。

**Group size.** We will compare GRPO, GDPO, and DARA on Qwen2.5-1.5B-Instruct with group sizes $G\in\{4,8,16,32\}$ and five training seeds. We fix the response budget at 2,048 per training step and the PPO minibatch size at 512 responses, giving four optimizer updates per step. The prompt batch size is $2,048/G$. All runs use 100 training steps and the same learning rate, reward functions, and data split.

**Three rewards.** We will train GRPO, GDPO, and DARA with correctness, format, and length rewards on both model sizes, using group size 4 and five training seeds. The length reward is $r_{\mathrm{length}}=\min(1,\operatorname{round}(n/512,2))$, where $n$ is the whitespace-delimited word count inside the extracted think segment; outputs missing the think markers receive zero. We will evaluate the final models with the same BFCL protocols and report Length Reward and the proportion of outputs satisfying $n\geq512$, alongside Accuracy and Format. Length metrics follow the same output-to-case and subset aggregation as Format.
