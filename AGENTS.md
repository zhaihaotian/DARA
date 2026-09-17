# DARA research handoff

先阅读 docs/HANDOFF.md、docs/TRAINING.md、docs/EVALUATION.md。本仓库采用Haotian infra，DARA使用双侧密度校准。局部修改时，其余现有行为保持冻结；范围扩展以用户明确授权为准。

训练按 configs/group_ablation.json 与 configs/three_rewards.json 执行，每run固定4×A10040GB、100steps和相同数据，seeds为0/1/2/4/5。Group实验固定2048回答/step，三奖励固定G4。batch预算、学习率、rollout seed、KL、entropy、whitening和checkpoint规则使用仓库规定值。

最终比较评测step100模型，执行固定V3/V4协议。本轮补齐与过程比较依照configs/checkpoint_study.json和docs/CHECKPOINT_STUDY.md；1.5B过程比较评测steps10/20/…/100，新训练每10steps保存。完整保留各seed、原始输出和运行记录；筛选分析注明seed列表与样本数。三奖励run把length接入训练reward与优势计算。

检查与当前修改直接相关的行为；算法和训练接线修改运行scripts/test.sh。每个新run使用独立目录，其他项目调度保持原样。实现保持简洁，按任务实际需要添加代码。与用户交流使用连贯段落。文档直接写操作步骤、参数和计算定义，指标统一称为Format和Average Format。

两卡训练必须在原四路逻辑分组下执行，保持每个optimizer更新的样本、动态microbatch边界、loss权重和rollout分配；上线前对同一固定batch的loss、梯度与参数更新做对照。当前正式自动训练使用四个物理rank，单独两卡allocation可用于评测或与另一份同型两卡allocation拼接。2026-09-17的3B DVAO seed0两H100实验按用户要求保留训练和结果。

8×A100 80GB SXM4租机任务按docs/RENTAL_A100_80G_SXM4_AGENT.md执行，目标为保留四路逻辑分组的两卡训练、最多四个实验并行。先完成相应实现和GPU对照，再更新实际启动布局。第一调度规则是任何时候最大化利用所有已租GPU：任务结束立即补位，训练资源不足或剩余租期较短时转入checkpoint评测，使用tmux持续维护任务。其他项目现有调度按各自配置执行。
