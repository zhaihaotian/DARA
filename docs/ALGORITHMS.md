# 方法与实现对齐

DARA 保留历史 Haotian RD-GDPO symmetric 的数值定义，仅统一方法标识和日志前缀为 `dara`。核心组合函数是 `vendor/verl/verl/trainer/ppo/core_algos.py::compute_dara_combined_advantage`，分通道接线位于 `ray_trainer.py::compute_advantage`。四个附加基线共用 `multi_reward_algos.py`；Hard 的 query 权重还接入真实 actor PPO loss。

记 Z_j 为 prompt group 内通道 j 的 reward z-score。GRPO/GDPO/DARA 的 z-score 使用 sample std，分母为 `std + 1e-6`。

| 方法 | 核心定义与固定设置 |
|---|---|
| GRPO | reward 求和，使用原 GRPO 组内 z-score |
| GDPO | 每通道独立 Z_j 后相加，最终 token whitening |
| DARA | π_j 为非零通道优势的 prompt group 比例；`w_j=min(5,sqrt(max(π)/π_j))`；组合 `Σ_j w_j Z_j` 后 token whitening；正负优势同权缩放 |
| RVPO | 两通道 `-(logsumexp(-k*Z)-log(2))/k`，固定 k=1；常数通道仍参与聚合；最终 token whitening |
| DVAO | 通道权重 0.5/0.5；加权中心化 reward 除以加权 population std 之和；epsilon1e-8；不做最终 batch whitening |
| GDPO-SAW | 用理论下界[-3,0]平移 raw reward；batch sample std / mean 为 CV；权重 `2*CV/sum(CV)`；零 CV 时等权；最终 token whitening |
| GD²PO-Hard | 同一 response 通道优势有正有负则删除；零中性；保留项参与 token whitening；每 prompt 保留比例乘在 clipped PPO surrogate 外 |

这些定义延续已经移植到 Haotian 的算法。Jay 上游对照点为 `917da15`；RVPO 是后续按论文 k=1 配置同时加入两侧的实现。给相同 reward、mask 和 UID 时，已有对照核验了方法公式。GD²PO-Hard 这里保留 Jay 移植的“最终 loss 乘 query 权重”位置；该权重位置在 whitening 和 clipped PPO 计算之后。

## 保留的 Haotian 行为

| 部位 | 本仓库 / Haotian | Jay runtime |
|---|---|---|
| 最终归一化 | 有效 token 等权的 sample variance，sqrt(var+1e-8) | response 等权 sample std，std+1e-6 |
| 动态 loss 累积 | microbatch token mean / 名义累积次数 | 按全局 token/DP 进行缩放 |
| KL | 原 reward KL；分通道 estimator 读取未惩罚的分通道 reward | 单独 actor KL loss |
| Entropy | 0.001 | 0 |
| PPO | 原 clipped PPO | 另有 negative-advantage dual clip 与 log-ratio clamp |
| 数据加载 | 3920/80 | 同一原 parquet 过滤后3901/79 |
| vLLM seed | 默认0 | 随实验 seed |
| checkpoint | 最终 HF 模型 | 周期性完整可恢复状态 |

两套实现使用相同的原始数据文件。Reward parser 的具体行为是：内容正确但缺少 `</tool_call>` 时，Haotian correctness 为−3，Jay 可给+3；两者 format 都为0。本仓库使用 Haotian scorer，附加方法的优势公式沿用已有的 Jay→Haotian 移植。

DARA π 的有效组阈值为广播后的 token 优势绝对值和 >1e-8。死通道 π=0 时日志权重为5，实际通道优势为0。新三奖励实验将相同定义扩展至 length 通道；两奖励默认行为通过回归测试保持一致。

GRPO、GDPO与DARA共用 `core_algos.py::compute_channel_densities` 统计各通道的π和活跃group数。三个方法均以原始通道reward的组内z-score计算这些统计量。GRPO记录原reward求和系数1，GDPO记录标准化通道优势的组合系数1，DARA记录实际密度校准权重。每step的字段为 `<method>/pi_<channel>`、`<method>/w_<channel>` 和 `<method>/active_groups_<channel>`；三奖励时包含length通道。GRPO的通道统计用于日志，策略优势继续由含原KL惩罚的总reward计算。
