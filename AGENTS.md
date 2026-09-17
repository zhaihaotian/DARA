# DARA research handoff

先阅读 docs/HANDOFF.md、docs/TRAINING.md、docs/EVALUATION.md。本仓库采用 Haotian infra；DARA 使用双侧密度校准。用户只要求局部修改时，其他现有行为冻结。

训练每个 run 固定4×A10040GB、100steps、相同数据和固定参数。实验只按 configs/group_ablation.json 与 configs/three_rewards.json 执行；seeds为0/1/2/4/5。Group实验固定2048回答/step，三奖励实验固定G4。不能静默改数据、batch预算、学习率、rollout seed、KL、entropy、whitening或checkpoint选择规则。

评测只使用final step100，固定V3/V4协议。保留全部seed、原始输出和基础设施失败记录。筛选seed的诊断不得冒充完整结果。新增length实验必须训练第三通道，不能以旧输出后处理代替。

代码修改需运行与故障直接相关的检查；算法/训练接线修改运行scripts/test.sh。不要添加无用途的哈希、迁移框架、兼容层或泛化开关。不要覆盖已有实验目录，不更改其他项目的调度。与用户交流用连贯段落，不分点。
