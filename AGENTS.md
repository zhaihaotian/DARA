# DARA research handoff

先阅读 docs/HANDOFF.md、docs/TRAINING.md、docs/EVALUATION.md。本仓库采用Haotian infra，DARA使用双侧密度校准。局部修改时，其余现有行为保持冻结；范围扩展以用户明确授权为准。

训练按 configs/group_ablation.json 与 configs/three_rewards.json 执行，每run固定4×A10040GB、100steps和相同数据，seeds为0/1/2/4/5。Group实验固定2048回答/step，三奖励固定G4。batch预算、学习率、rollout seed、KL、entropy、whitening和checkpoint规则使用仓库规定值。

评测step100模型，执行固定V3/V4协议。完整保留各seed、原始输出和运行记录；筛选分析注明seed列表与样本数。三奖励run把length接入训练reward与优势计算。

检查与当前修改直接相关的行为；算法和训练接线修改运行scripts/test.sh。每个新run使用独立目录，其他项目调度保持原样。实现保持简洁，按任务实际需要添加代码。与用户交流使用连贯段落。文档直接写操作步骤、参数和计算定义，指标统一称为Format和Average Format。
