# 验证记录

验证使用实际Haotian训练环境。独立仓库路径为 `/users/2/lian0190/DARA`，复跑入口为 `bash scripts/test.sh`。

| 项目 | 结果 |
|---|---|
| CPU测试 | 38项通过：独立NumPy算法参考、PPO query权重与梯度、动态累积、100step/checkpoint调度、三奖励接线、通道密度日志、两卡配对启动、四路逻辑分组与rollout随机流、DVAO seed3过程训练入口、评测权重与格式/长度统计 |
| 过程模型保存 | `--save-freq 10` 仅修改保存间隔；训练循环在steps10/20/…/100保存，验证保持steps0/10/…/100 |
| 本轮排队与评测接线 | 调度33项测试、V4过程评测5项测试通过，覆盖任务优先级、到期后新attempt、A100/H100分配、两H100先验证后训练、同节点端口、156个目标模型、逐step统计与最终表复用 |
| 两卡配对训练 | 单节点四卡与两个Ray节点各两卡的训练配置仅节点布局不同；agb04+agb02各两A100的四rank NCCL实跑通过，全部求和为10，退出码0；双节点完整训练与导出待验证 |
| H100两卡训练 | 四路逻辑分组已接通；CPU与两张H100上的GPU/FSDP回放均通过，microbatch顺序一致，GPU四次更新的梯度最大差2.98e-8、参数最大差9.31e-10。详见TWO_GPU_LOGICAL_FOUR.md。此前3B DVAO seed0的100step训练与结果保留 |
| GRPO/GDPO/DARA密度日志 | 两奖励与三奖励共用π口径；UID重排、常数通道、通道抵消均通过；优势、returns及输入张量逐元素一致，实际训练循环每步写出JSONL指标 |
| 整理前后的优势对照 | G4/8/16/32下140次比较逐元素一致 |
| Hydra配置对照 | 28份展开配置对应一致 |
| 仓库初始整理时Actor与FSDP worker | 保留方法的AST一致；后续两卡四路分组修改由固定batch更新对照验证 |
| V3重新评分 | 3B GDPO的2501cases，10类accuracy与原结果完全一致 |
| V4重新评分 | GDPO的3301cases，14类accuracy及各组Format与原结果完全一致 |
| 数据下载 | 固定ToolRL源的train/test parquet与原训练文件逐字节一致 |
| 训练依赖 | 当前环境pip check通过，指定FlashAttention wheel可下载 |
| 全新机器安装 | 待执行 |
| 新group消融与三奖励100step训练 | 待执行 |

算法与配置对照记录见 [verification.json](verification.json)，重新评分记录见 [evaluation_validation.json](evaluation_validation.json)。
