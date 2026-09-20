# DARA A100 租机执行记录

服务器为 `ssh ubuntu@137.131.62.225`，8×A100-SXM4-40GB。仓库 `/home/ubuntu/DARA`，基线提交 `726d47ab7f80e3920bc2333e70414d2faf77bdb8`。早期运行使用的2026-09-18 18:26:54 UTC估算截止已被用户明确取消；只要服务器可访问就继续，当前控制器兜底deadline为2026-09-25 19:19:08 UTC，并会在全部任务完成时提前自行退出。

执行目标固定为 `configs/rental_1p5b_20h.json` 中15次100step训练、150份BFCL V4过程评测、15份step100最终评测。第15项DVAO seed3包含全部过程评测和最终评测。两个GPU组分别为0–3与4–7，首批为DVAO seed4、GD²PO-Hard seed4，结束后各自立即补位。完整训练时间不足时空闲组拆为四路单卡评测，每个完成run先评step100。推理与CPU官方评分独立派发。

持久产物根目录为 `/lambda/nfs/haotian/dara-rental-20260917`。仓库的 `outputs` 是指向此根目录 `outputs` 的符号链接。训练产物在 `outputs/rental-1p5b`，评测产物在 `outputs/eval-v4-checkpoints`，失败attempt保留在独立目录。操作脚本、任务登记、进展和进程日志在 `operations`。代码快照在 `code`。

Conda在 `/home/ubuntu/miniforge3`。环境名称为 `rdgdpo`、`dara-inference`、`dara-bfcl-v4`，按仓库锁定安装。安装日志在 `/home/ubuntu/bootstrap-logs`，三个 `.ready` 文件表示对应阶段成功完成。训练安装执行36项仓库测试并全部通过，pip check通过。BFCL V4源码提交为 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，已实际导入adapter并验证14类3301题。

模型位于 `/home/ubuntu/dara-models/Qwen2.5-1.5B-Instruct`。固定ToolRL数据在 `/home/ubuntu/DARA/data/rlla_4k`。

控制器在tmux会话 `dara-controller` 运行，命令为：

```bash
python3 -u /home/ubuntu/controller.py --model-root /home/ubuntu/dara-models
```

2026-09-17 22:31:29 UTC派发首批两项，22:34:35 UTC已从各自真实 `metrics.jsonl` 确认完成step1并继续运行。DVAO seed4第一步101.67秒，GD²PO-Hard seed4第一步106.17秒，均记录实际policy loss和有限梯度范数。两路训练前validation均完成，runtime记录的world_size均为4、SEED均为4、save_freq均为10。此时完成的完整训练为0/15、BFCL评分为0/150、final为0/15；运行中的训练为2项。

2026-09-17 22:51:28 UTC两路均完成step10，首份checkpoint已保存在持久目录，每份权重7,108,390,440 bytes。22:52:44 UTC使用safe_open读取分片元数据并核对index中所有339个tensor的归属，两个分片均可读取，AutoTokenizer及AutoConfig均成功加载；记录在服务器 `operations/first_checkpoints_verified.json`。GPU利用率97–100%，训练继续。后续巡检先比较本地 `LAST_CHECK.json`，首份checkpoint成功已向用户说明。

控制器每5秒检查退出并独立补位，每分钟记录硬件和任务状态，每30分钟记录进展与ETA。`operations/queue_state.json` 保存完整任务归属与所有150份评测状态，`operations/status.json` 为一分钟快照，`operations/controller.log` 与 `operations/events.jsonl` 记录派发事件，各任务日志和退出记录在 `operations/jobs/`。训练worker为 `/home/ubuntu/train_worker.py`，每项先读取官方launch的环境展开结果，随后建立独立四GPU Ray runtime，保持seed、reward环境和四rank设置。

控制器可独立重启，现有job wrapper保持运行，重启后按PID及退出记录接回任务。不要重复手动启动manifest项目。若需要修复训练失败，保留原attempt并用新目录重试。算法、数据、seed、batch、reward、save10和评测协议保持不变。

Codex本任务已建立每10分钟跟进巡检，automation ID为 `dara`。用户明确要求每次读完巡检结果都反馈，包括正常运行或状态未变。巡检必须持续到15次训练、150份过程评分和15份final全部完成，只要服务器可访问就不能依据旧时间估算停机。每次结束时用简短、连贯的中文段落说明当前任务与step、完整训练数、已保存checkpoint数、过程评分及final数、异常或修复，并明确后台是否继续运行。不能空白结束，不分点，不将巡检回合结束表述为训练停止或全部完成。服务器自身的派发不依赖此巡检。

原集群已于2026-09-17 22:30:21 UTC转移这15项。原队列为 `/scratch.global/lian0190/RD-GDPO-3B/20260913/dara_training_tasks.json`，仅移出对应15项，原6项3B保留。登记与完整原队列备份的路径在本目录 `rental_transfer_20260917.json`。Slurm allocation和原tmux会话保持原样。

仓库执行侧增加 `evaluation/run.py --inference-only`，用于GPU推理退出后立即补位，随后在CPU运行同一 `evaluation/score.py`。默认一体化评测入口的行为保持原样。其他操作脚本独立存放，不改训练算法与协议。

2026-09-18 00:53:48 UTC按用户提速请求部署执行优化。首批两个seed4已加载的代码继续运行，之后新启动的run采用三处变化：compute_log_prob省去丢弃的entropy；vLLM rollout省去丢弃的logprobs输出与搬运（训练old_log_probs仍由actor原样重算）；HF checkpoint先同步写本地NVMe，再单线程复制到原持久目录旁的.copying目录、原子发布，最后一次保存等待全部复制完成。四rank、样本数、微批边界、梯度累积、loss、奖励、KL、训练entropy、采样参数及save10保持原协议。

优化源码快照在持久根目录 `code_optimized_20260918`，早期 `code` 快照保留。之后每个run的runtime.json记录source_snapshot、execution_optimizations和checkpoint_staging路径。暂存路径是 `/home/ubuntu/checkpoint-staging/<run-id>-<worker-pid>`，复制成功只清理本地重复文件；复制失败保留完整暂存并向训练报告错误。step100必须等全部十份持久化后才完成。

隔离worktree `/home/ubuntu/DARA-perf-validation` 的38项测试通过。实际CUDA核对使用真实parquet前两行的16个token、完整151936词表和固定logits，原/新log-prob、训练entropy/loss/gradient逐值完全相同，vLLM实际sampler的token与RNG状态完全相同，None输出适配器通过，验证进程显存760MiB。该检查验证修改涉及的计算，不是完整模型生成的端到端加速测量。记录在本地 `kernel-equivalence.json` 和 `optimization-unit-tests.log`。

真实6.62GiB checkpoint CPU实验核对339个tensor及11个文件内容完全相同。直接写NFS23.30秒，本地序列化返回7.61秒，后台完成持久化36.39秒；计入最后一次等待，单次100step训练预计净省约128秒。结果在本地 `checkpoint_io_validation/result.json`，重复benchmark权重已清理。原首批最近20步均值为DVAO94.59秒、GD²PO-Hard93.08秒，反向更新约41.5–42.1秒与历史相近；额外耗时主要在rollout和NFS。后续巡检必须在seed5实际启动后检查新代码版本、采样输出正常、梯度有限、每步timing_s/gen/ref/update_actor，以及首份暂存checkpoint最终发布和step100全部落盘。只用真实新run计时汇报端到端收益。

首批GD²PO-Hard seed4于01:15:05 UTC完成，DVAO seed4于01:16:17 UTC完成，退出码均0，实耗分别9815.93秒和9887.62秒。各自十份完整checkpoint及101行metrics/training_dynamics.csv已核对。通道在不到1秒内各自补位，DVAO seed5在GPU4–7、GD²PO-Hard seed5在GPU0–3运行；01:20:18 UTC分别已完成2步和1步，均启用三个优化且实际梯度有限。BFCL仍待空卡后派发，任务数保留150份过程、15份final。首批完成和自动补位已向用户汇报。

01:36:08 UTC首次正式训练的异步保存已验证：两个seed5的step10均完整发布至NFS，各7,108,390,440 bytes、339 tensors、tokenizer齐全，本地暂存及.copying目录均已清理。实际训练侧save_checkpoint分别13.55秒与14.57秒，后台复制期间训练继续，DVAO已到step11。现有22份checkpoint、2/15完整训练，BFCL评分0/150、final0/15。这个阶段只确认保存阻塞降低，整体训练加速仍以完整新run计时为准。

04:03:20 UTC第二批完整验收：DVAO seed5于03:54:14完成，耗时9548.30秒（159.14分钟）；GD²PO-Hard seed5于03:57:10完成，耗时9653.05秒（160.88分钟）。各自十份checkpoint、step100、101行metrics与CSV全部保留，退出码0，暂存和.copying均无剩余。最终保存耗时42.14/44.16秒包含等待持久复制，证明最终成功状态在落盘后返回。本批比相应seed4分别短5.66/2.71分钟；seed不同，差值不是严格的优化因果测量。GRPO seed0和GDPO seed0分别在前项退出约4秒后补位，04:03分别到step4/2，实际日志已包含两路pi、权重和active_groups。累计完整训练4/15、40份checkpoint，BFCL评分仍0/150、final0/15。

06:22 UTC巡检记录GDPO seed0在step89出现有限值尖峰：actor/grad_norm=1.6221783153e12、actor/pg_loss=16226.78。源码记录的是clip_grad_norm_返回的裁剪前范数，实际展开配置grad_clip=1.0；该run截至step91全部日志数值没有NaN/Inf，step90/91梯度范数恢复为227.03/65.59，policy loss恢复为0.1082/0.0326。step90验证Format=0.975、Correctness=1.79364，与step80同一量级。原始参考GDPO seed0、seed4日志也分别有3.406e11和1.303e13级别范数记录。保留本次原始尖峰和后续恢复记录，按固定协议继续，不调整算法或超参数。该观察已向用户报告；后续仅在再次出现或指标持续异常时进一步核对。

06:49 UTC第三批完整验收：GDPO seed0于06:36:13完成，耗时9539.01秒；GRPO seed0于06:37:29完成，耗时9790.90秒。均exit0，十份checkpoint和101行metrics/CSV齐全，全部持久化、暂存无剩余。GDPO step91之后没有新增大幅梯度尖峰，至step100日志无NaN/Inf。GPU0–3于06:36:18补入DARA seed0，GPU4–7于06:37:33补入GRPO seed2；06:49分别已完成7/6步，两路pi、权重及active_groups实际日志完整。累计训练6/15、60份checkpoint，BFCL评分0/150、final0/15。

09:17 UTC第四批首项验收：DARA seed0于09:12:23完成，耗时9364.96秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU0–3约3.5秒后补入GDPO seed1并已完成step2；GRPO seed2已完成step96且step90已持久化。累计训练7/15、79份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

09:30 UTC第四批完整验收：GRPO seed2于09:25:00完成，耗时10046.70秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU4–7约3.4秒后补入DARA seed1并已完成step2；GDPO seed1已完成step9。累计训练8/15、80份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

11:54 UTC第五批首项验收：GDPO seed1于11:50:41完成，耗时9494.59秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU0–3约3.8秒后补入GRPO seed5并已完成step1；DARA seed1已完成step94且step90已持久化。累计训练9/15、99份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

12:09 UTC第五批完整验收：DARA seed1于12:02:50完成，耗时9467.19秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU4–7约3.3秒后补入GDPO seed5并已完成step2；GRPO seed5已完成step9。累计训练10/15、100份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

14:37 UTC第六批首项验收：GRPO seed5于14:36:42完成，耗时9957.22秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU0–3约3.4秒后补入DARA seed2，正在初始化Ray；GDPO seed5已完成step98且step90已持久化。累计训练11/15、119份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

14:49 UTC第六批完整验收：GDPO seed5于14:39:28完成，耗时9393.98秒，exit0；十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。GPU4–7约3.4秒后补入GD²PO-Hard seed0并已完成step4；DARA seed2已完成step6。累计训练12/15、120份checkpoint，BFCL评分0/150、final0/15，无失败任务或复制积压。

17:25 UTC第七批完整验收：DARA seed2于17:15:59完成，耗时9553.87秒；GD²PO-Hard seed0于17:16:20完成，耗时9408.63秒。二者均exit0，十份checkpoint（含step100）、101行metrics与101行CSV全部持久化，暂存和.copying均为空。累计训练14/15、140份checkpoint。剩余租期不足完成一项新训练，DVAO seed3保持pending，控制器按约定将8张GPU全部转为BFCL推理，已并行派发DVAO seed4的step100、10、20、30、40、50、60、70；BFCL评分0/150、final0/15，无失败任务。

18:10 UTC首份BFCL完整评分验收：DVAO seed4 step10推理完成并经CPU评分exit0，生成summary.json、3301行案例诊断CSV及各分类评分文件；四份汇总CSV均已更新，checkpoint_per_model与checkpoint_method_results各含一条结果，两个final汇总CSV当前仅表头。GPU1在推理退出后立即补入step80，其余step100、20、30、40、50、60、70继续运行。累计过程评分1/150、final0/15，无失败任务。

18:33 UTC旧估算截止触发了一次阶段中断：控制器于18:27:23以reason=lease_ended退出，当时完整训练14/15、checkpoint 140份、过程评分3/150、final0/15。DVAO seed4的step10、20、30完成评分，step100、40、50、60、70、80的raw进度持久化；中断的evaluation均恢复为pending_inference，队列无failed项。该截止随后被用户明确取消，不再作为停止条件。

19:21 UTC恢复执行：服务器仍可访问，控制器deadline改为2026-09-25 19:19:08 UTC兜底并重启，automation重新设为ACTIVE。DVAO seed3已在GPU0–3启动并进入step1；GPU4–7分别从持久raw断点续跑DVAO seed4的step100、40、50、60，行数继续增长，没有重跑已保存案例。当前完整训练14/15、checkpoint 140份、过程评分3/150、final0/15；控制器将持续到全部任务完成并以all_complete自行退出。

19:37 UTC恢复后首份final完成：DVAO seed4 step100完成3301题推理与CPU评分，过程评分4/150、final1/15。四份汇总CSV均已更新：两个checkpoint CSV各为表头加四条结果，两个final CSV各为表头加一条结果。GPU4立即补入step70；GPU4–7继续step70、40、50、60评测。DVAO seed3已完成step9，GPU0–3训练利用率99–100%，评测GPU利用率89–100%，8张卡全部有有效任务，无failed项。

21:57 UTC全部训练完成并转入八路评测：DVAO seed3于21:55:08完成step100并exit0，实耗9314.54秒；十份checkpoint、101行metrics及最终验证均已落盘。累计15/15次训练、150份checkpoint全部完成。GD²PO-Hard seed4 step20随后完成3301题推理和CPU评分，累计过程评分13/150、final2/15；四份汇总CSV行数与该计数一致。GPU0–7已全部运行单卡BFCL V4推理，利用率88–95%，无活动失败项，控制器继续自动补位。

22:55 UTC八路评测连续补位：GD²PO-Hard seed4的steps90、80、40、50、30依次完成推理和CPU评分，累计过程评分18/150、final2/15。释放的GPU均在数秒内补入DVAO seed5的steps10、20、30、40、50；当前另有GD²PO-Hard seed4 steps60/70和DVAO seed5 step100运行。四份CSV分别为19、19、3、3行，与18份checkpoint评分和2份final一致；无活动失败项。

23:29 UTC DVAO seed5首批完成：其step100和step10均完成3301题推理及CPU评分，累计过程评分22/150、final3/15；其中step100计入第三份final。释放的GPU7、3立即补入该run的steps80、90，八路继续运行。逐模型checkpoint/final CSV分别为23/4行；方法聚合CSV按方法和step合并重复seed，当前为21/3行，均与已完成结果一致。无活动失败项。

00:54 UTC过程评分达到30份：DVAO seed5的steps40、50、60、70、90及GD²PO-Hard seed5 step10完成推理和CPU评分，累计30/150、final3/15。释放的GPU立即补入GD²PO-Hard seed5的steps20–70；另有DVAO seed5 step80和GD²PO-Hard seed5 step100继续。checkpoint_per_model.csv为31行，匹配30份评分；无基础设施错误或活动失败项。

01:13 UTC第四份final完成：GD²PO-Hard seed5 step100完成3301题推理和CPU评分，累计32/150、final4/15；GPU2立即补入同一run的step90，八路继续运行。逐模型checkpoint/final CSV分别为33/5行，方法聚合CSV按方法与step合并seed后为21/3行，均与已完成结果一致；无活动失败项。

02:21 UTC过程评分达到40份：GD²PO-Hard seed5的steps20–80均已陆续完成评分，累计40/150、final4/15；仅该run step90仍在推理。释放的GPU已依次补入GRPO seed0的step100及steps10–70，八路继续运行。checkpoint_per_model.csv为41行，匹配40份评分；无活动失败项。

02:42 UTC GD²PO-Hard seed5全部十个checkpoint完成：step90完成推理和CPU评分后，该run的steps10–100已齐全；累计42/150、final4/15。GPU2立即补入GRPO seed0 step90，此时八路均为GRPO seed0 checkpoints。checkpoint_per_model.csv为43行，匹配42份评分；无活动失败项。

03:13 UTC第五份final完成：GRPO seed0 step100完成3301题推理和CPU评分，随后该run step40也完成，累计45/150、final5/15。GPU4、3分别立即补入GDPO seed0 steps20、10；八路继续运行。逐模型checkpoint/final CSV分别为46/6行，汇总CSV为26/4行，均与已完成结果一致；无活动失败项。

03:35 UTC第六份final完成：GDPO seed0 step100完成3301题推理和CPU评分；同阶段GRPO seed0 steps50/60也完成，累计48/150、final6/15。释放的GPU立即补入GDPO seed0 steps30/40/50，八路继续运行。四份CSV分别为49、29、7、5行，与已完成结果一致；无活动失败项。

03:52 UTC过程评分达到50份：GRPO seed0 step70与GDPO seed0 step10完成推理和CPU评分，累计50/150、final6/15。GPU7、3立即补入GDPO seed0 steps60、70，八路继续运行。checkpoint_per_model.csv为51行，匹配50份评分；无活动失败项。

04:56 UTC第七份final完成：DARA seed0 step100完成3301题推理和CPU评分，累计54/150、final7/15；GPU4立即补入DARA seed0 step10。四份CSV分别为55、35、8、6行，与已完成结果一致；无活动失败项。

05:42 UTC过程评分超过60份：GDPO seed0 steps40/60完成推理和CPU评分，累计61/150、final7/15。GPU5、7立即补入DARA seed0 steps80、70，此时八路均为DARA seed0 checkpoints。checkpoint_per_model.csv为62行，匹配61份评分；无活动失败项。

06:48 UTC过程评分达到70份：DARA seed0 step40完成推理和CPU评分，累计70/150、final7/15；GPU1立即补入GRPO seed2 step70。checkpoint_per_model.csv为71行，匹配70份评分；无活动失败项。

07:26 UTC第八份final完成：GRPO seed2 step100完成3301题推理和CPU评分，随后该run step30也完成，累计74/150、final8/15。GPU5、4分别立即补入GDPO seed1 steps100、10。四份CSV分别为75、51、9、6行，与已完成结果一致；无活动失败项。

08:17 UTC第九份final及第80份过程评分完成：GDPO seed1 step100与step10、GRPO seed2 steps60/70完成评分，累计80/150、final9/15。释放的GPU立即补入GDPO seed1 steps40–70，八路继续运行。checkpoint_per_model.csv为81行，匹配80份评分；无活动失败项。

09:08 UTC第十份final完成：DARA seed1 step100完成3301题推理和CPU评分；同阶段GDPO seed1 steps20/80也完成，累计86/150、final10/15。释放的GPU立即补入DARA seed1 steps10/20/30，八路继续运行。逐模型checkpoint/final CSV分别为87/11行；无活动失败项。

09:47 UTC过程评分达到90份：GDPO seed1 step40完成推理和CPU评分，累计90/150、final10/15；GPU6立即补入DARA seed1 step70，八路继续运行。checkpoint_per_model.csv为91行，匹配90份评分；无活动失败项。

10:36 UTC过程评分达到100份：GRPO seed5 step10和DARA seed1 step30完成推理和CPU评分，累计100/150、final10/15。GPU4、2立即补入GRPO seed5 steps60、70，八路继续运行。checkpoint_per_model.csv为101行，匹配100份评分；无活动失败项。

11:41 UTC第十一份final完成：GRPO seed5 step100与step50先后完成3301题推理和CPU评分，累计106/150、final11/15；GRPO的三个final seed现已齐全。释放的GPU1、0立即补入GDPO seed5 steps20、30，短暂加载后八卡恢复59–92%利用率。四份CSV分别为107、51、12、6行，与已完成结果一致；无活动失败项。

12:20 UTC第十二份final完成：GDPO seed5 step100、step10及GRPO seed5 step80完成推理和CPU评分，累计111/150、final12/15；GDPO的三个final seed现已齐全。释放的GPU7、3、5立即补入GDPO seed5 steps60、70、80，加载后八卡利用率73–93%。四份CSV分别为112、51、13、6行，与已完成结果一致；无活动失败项。

12:43 UTC GRPO seed5全部十个checkpoint完成：step90完成推理和CPU评分后，该run的steps10–100已齐全；累计113/150、final12/15。GPU6立即补入DARA seed2 step100，其余七卡继续GDPO seed5 steps30–90，八卡利用率81–93%。四份CSV分别为114、51、13、6行，与已完成结果一致；无活动失败项。

13:19 UTC第十三份final完成：DARA seed2 step100完成推理和CPU评分，累计115/150、final13/15；DARA的三个final seed现已齐全。GPU6立即补入DARA seed2 step20，其余GPU继续GDPO seed5 steps40–90及DARA seed2 step10，八卡利用率75–97%。四份CSV分别为116、51、14、6行，与已完成结果一致；无活动失败项。

13:43 UTC GDPO seed5连续收尾：steps90、70、80完成推理和CPU评分，累计119/150、final13/15。释放的GPU1、3、5立即补入DARA seed2 steps40、50、60；GDPO seed5仅余steps50、60，其中step60为3288/3301。八卡利用率69–93%，四份CSV分别为120、51、14、6行，无活动失败项。

13:56 UTC过程评分达到120份：GDPO seed5 step60完成推理和CPU评分，累计120/150、final13/15。GPU7立即补入DARA seed2 step70；GDPO seed5仅余step50（3297/3301），DARA seed2 step10为3300/3301。八卡均有活动任务，四份CSV分别为121、51、14、6行，无活动失败项。

14:08 UTC GDPO seed5全部十个checkpoint完成：其step50与DARA seed2 step10先后完成推理和CPU评分，累计122/150、final13/15。GPU0、2立即补入DARA seed2 steps80、90，八卡现全部运行DARA seed2 steps20–90，利用率76–93%。四份CSV分别为123、51、14、6行，无活动失败项。

15:12 UTC第十四份final完成：GD²PO-Hard seed0 step100完成推理和CPU评分，累计129/150、final14/15；仅剩DVAO seed3 final尚未评分。GPU7立即补入GD²PO-Hard seed0 step60，加载后八卡利用率54–95%。四份CSV分别为130、51、15、6行，与已完成结果一致；无活动失败项。

15:36 UTC DARA seed2全部十个checkpoint完成：其step30与GD²PO-Hard seed0 step20完成推理和CPU评分，累计133/150、final14/15。GPU4、1立即补入GD²PO-Hard seed0 step90和必须保留的DVAO seed3 step100 final；DVAO final已到2637/3301。八卡利用率65–97%，四份CSV分别为134、51、15、6行，无活动失败项。

16:26 UTC全部十五份final完成：必须保留的DVAO seed3 step100 final、GD²PO-Hard seed0 steps40、50完成推理和CPU评分，累计137/150、final15/15。GPU0、1、3立即补入DVAO seed3 steps20、30、40；八路继续GD²PO-Hard seed0 steps60–90与DVAO seed3 steps10–40，利用率68–92%。四份CSV分别为138、51、16、6行，无活动失败项；巡检继续至150/150过程评分完成。

16:39 UTC过程评分达到140份：GD²PO-Hard seed0 steps80、60、90完成推理和CPU评分，累计140/150、final15/15。GPU6、7、4立即补入DVAO seed3 steps50、60、70；八路现运行GD²PO-Hard seed0 step70及DVAO seed3 steps10–70，仅steps80、90待派发。四份CSV分别为141、51、16、6行，无活动失败项。

16:51 UTC全部剩余过程项均已派发：DVAO seed3 step10与GD²PO-Hard seed0最后的step70完成推理和CPU评分，累计142/150、final15/15。GPU2、5立即补入DVAO seed3 steps80、90；队列不再有pending，八卡全部运行DVAO seed3 steps20–90。四份CSV分别为143、51、16、6行，无活动失败项。

17:40 UTC最后五份过程评测运行中：DVAO seed3 steps30、80完成推理和CPU评分，累计145/150、final15/15。剩余steps40、50、60、70、90全部在GPU3–7运行，其中step90为3300/3301、step70为3292/3301；GPU0–2因队列只剩五项而空闲。四份CSV分别为146、51、16、6行，无活动失败项。

17:53 UTC最后三份过程评测运行中：DVAO seed3 steps90、70完成推理和CPU评分，累计147/150、final15/15。剩余steps40、50、60分别在GPU3、6、7运行，raw进度为3275/3301、3235/3301、3176/3301，均在处理long-context尾部；GPU0–2与4–5因队列只剩三项而空闲，活动卡利用率92–93%。四份CSV分别为148、51、16、6行，无活动队列或基础设施失败，控制器继续运行。

18:07 UTC仅余最后一份过程评测：DVAO seed3 steps40、50已完成推理和CPU评分，累计149/150、final15/15。step60在GPU7运行，raw进度3278/3301、利用率91%，其余GPU因已无其他队列项而空闲。四份CSV分别为150、51、16、6行；日志中的超长上下文BadRequest是协议定义的单题结果，不是基础设施失败，控制器继续运行至最后评分完成。

18:11 UTC全部目标完成：DVAO seed3 step60完成3301题推理并经官方CPU评分exit0，累计15/15次100-step训练、150/150份checkpoint保存、150/150份BFCL V4过程评分和15/15份step100 final评分。queue_state中150项evaluation均为complete，无活动或待派发任务；四份CSV分别为151、51、16、6行。控制器记录reason=all_complete后自行退出，tmux会话消失，8张GPU均空闲。本地过程表、final表与CSV快照完成最终刷新，定时巡检到此停止。
