# BFCL evaluation protocol

## Inference

Each training run is evaluated at its step-100 model, and the base model of each size is evaluated once with the same protocol. Every checkpoint produces one trajectory per case. All methods use the same ToolRL prompt, output parser, test subsets and decoding parameters.

| Item | Setting |
|---|---|
| Evaluator | BFCL V4, gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| Subsets | Non-Live AST + Live AST + Multi-Turn, 3,301 cases in 14 categories |
| Prompt / parser | ToolRL RLLA handler adapted to the V4 API (`evaluation/v4/toolrl_adapter.py`) |
| Interface | completion, `is_fc_model=False` |
| Temperature / top-p | 0.6 / 0.95 |
| Max new tokens / context | 8,192 / 32,768 |
| Seed / top-k / repetition penalty | 0 / −1 / 1 |
| Server | vLLM 0.11.0, TP 1, bf16, memory utilization 0.85, eager mode |

For Multi-Turn cases, the BFCL driver executes the tools and advances the conversation; the tool state is initialized per case. A case that exceeds the context limit counts as a model failure.

## Accuracy

Per-category accuracy comes from the official BFCL checkers. AST categories check the parsed function names, parameters and acceptable values. Multi-Turn categories check the execution state and the conversation outcome.

| Subset | Category | Cases |
|---|---|---:|
| Simple Python / Java / JavaScript | `simple_python`, `simple_java`, `simple_javascript` | 400 / 100 / 50 |
| Multiple | `multiple` | 200 |
| Parallel | `parallel` | 200 |
| Parallel Multiple | `parallel_multiple` | 200 |
| Live Simple / Multiple / Parallel / Parallel Multiple | `live_*` | 258 / 1,053 / 16 / 24 |
| Multi-Turn Base / Missing Function / Missing Parameter / Long Context | `multi_turn_*` | 200 each |

Group accuracies:

$$
A_{\text{simple}}=\tfrac{1}{3}(A_{\text{Python}}+A_{\text{Java}}+A_{\text{JavaScript}}),\qquad
A_{\text{non-live}}=\tfrac{1}{4}(A_{\text{simple}}+A_{\text{multiple}}+A_{\text{parallel}}+A_{\text{parallel multiple}}),
$$

$$
A_{\text{live}}=\frac{\sum_{c\in \text{Live}} N_c A_c}{1351},\qquad
A_{\text{multi-turn}}=\text{mean of the four Multi-Turn categories},\qquad
\text{Average}=\tfrac{1}{3}(A_{\text{non-live}}+A_{\text{live}}+A_{\text{multi-turn}}).
$$

## Format

After extracting the assistant text (Qwen assistant marker, trimmed whitespace), an output scores 1 if it matches one of these structures exactly, with tag order, closing tags, newlines, and a single tool-call/response block each. Otherwise it scores 0.

| Output type | Structure |
|---|---|
| Text response | `<think>...</think>\n<response>...</response>` |
| Tool call | `<think>...</think>\n<tool_call>\n...\n</tool_call>` |
| Tool call and response | `<think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>` |

The check calls the training scorer (`rlla.py::customize_format_reward_func`). For case $i$ with $m_i$ assistant outputs, $F_i=\frac{1}{m_i}\sum_j f_{ij}$, and a case with no output scores 0. Category Format is the mean of $F_i$. Group Format uses the same weights as accuracy, and **Average Format** is the mean of the Non-Live, Live and Multi-Turn format scores.

## Length (three-reward experiment)

With `--length-max-words 16`, the evaluator also scores the training length reward. For each output, $n$ is the whitespace word count of the text between the last `<think>` and the following `</think>`, and the output is compliant (Length = 1) if the think span is present and closed and $n \le 16$. Length is aggregated output → case → category → group exactly like Format, and the mean think-word count is recorded as well.

## Seeds and aggregation

Each checkpoint is scored independently, and the reported values are mean ± sample standard deviation (ddof = 1) over training seeds, computed from the per-run group scores. The inference seed is fixed at 0. Accuracy and Format are reported as percentages.

`evaluation/diagnostics.py` computes the per-case metrics, and `evaluation/protocol.py` computes the group scores.
