# 验证记录

验证使用实际Haotian训练环境。独立仓库路径为 `/users/2/lian0190/DARA`，复跑入口为 `bash scripts/test.sh`。

| 项目 | 结果 |
|---|---|
| CPU测试 | 25项通过：独立NumPy算法参考、PPO query权重与梯度、动态累积、100step/checkpoint调度、三奖励接线、通道密度日志、评测权重与格式/长度统计 |
| GRPO/GDPO/DARA密度日志 | 两奖励与三奖励共用π口径；UID重排、常数通道、通道抵消均通过；优势、returns及输入张量逐元素一致，实际训练循环每步写出JSONL指标 |
| 整理前后的优势对照 | G4/8/16/32下140次比较逐元素一致 |
| Hydra配置对照 | 28份展开配置对应一致 |
| Actor与FSDP worker | 保留方法的AST一致 |
| V3重新评分 | 3B GDPO的2501cases，10类accuracy与原结果完全一致 |
| V4重新评分 | GDPO的3301cases，14类accuracy及各组Format与原结果完全一致 |
| 数据下载 | 固定ToolRL源的train/test parquet与原训练文件逐字节一致 |
| 训练依赖 | 当前环境pip check通过，指定FlashAttention wheel可下载 |
| 全新机器安装 | 待执行 |
| 新group消融与三奖励100step训练 | 待执行 |

算法与配置对照记录见 [verification.json](verification.json)，重新评分记录见 [evaluation_validation.json](evaluation_validation.json)。
