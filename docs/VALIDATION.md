# 本次整理的验证范围

当前验证使用实际Haotian训练环境，独立新仓库路径为`/users/2/lian0190/DARA`。原训练仓库、已完成模型与集群实验调度未修改。

CPU回归通过21项测试，覆盖附加算法的独立NumPy参考、真实PPO actor的query权重/梯度路径、旧动态累积、100step与最终checkpoint调度，以及新增三奖励的真实reward scorer与GRPO/GDPO/DARA优势接线。也检查了format多轮平均、Non-Live/Live权重和长度四舍五入与512words达标判定的区别。复跑入口为`scripts/test.sh`。

另将新代码与整理前的Haotian代码在G4/8/16/32、稀疏/常数/冲突/连续部分reward等输入上进行140次优势比较，结果逐元素完全相同；28份完整Hydra展开配置在路径、方法命名等显式对应后相同。原reward scorer、rollout、数据加载和附加算法文件保持一致，保留的actor/FSDP方法AST相同。具体证据见[verification.json](verification.json)。这是数值和接线回归，不是重新训练140个模型。

新的可携带BFCL评分脚本对保存的GDPO输出做实际官方重新评分。V4的3301cases得到与原结果完全相同的14类accuracy和所有分组Format*，没有重新生成答案。V3另对3B GDPO的2501cases重新评分，10类accuracy同样完全一致。记录见[evaluation_validation.json](evaluation_validation.json)。

固定ToolRL数据源下载的两份parquet与原训练文件逐字节一致。公开FlashAttention wheel下载地址已确认可访问，原训练环境`pip check`通过。新的安装脚本和锁定文件来自实际已安装依赖；尚未在独立全新机器完成安装。GPU上完整100step的新group与三奖励实验尚未运行；这些仍是交接矩阵中的待办，不能声称已验证它们的收敛效果。
