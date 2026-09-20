# DARA 3B DVAO / GD²PO-Hard execution

Source handoff: DARA main commit `4465304`, `docs/HANDOFF_3B_DVAO_HARD.md`.

The 8×A100 40GB server is split into GPU groups 0–3 and 4–7. The first wave is DVAO seed1 and GD²PO-Hard seed1. A vacated lane immediately starts the next pending seed2 run, keeping two 3B trainings active until all four finish. Every run uses four physical ranks, 100 steps, G4, correctness+format, and save/validation frequency 10. All four runs evaluate steps10/20/…/100 with BFCL V4, for 40 process evaluations; the four step100 results also form the final subset.

Persistent root: `/lambda/nfs/haotian/dara-3b-20260919`. Controller session: `dara-3b-controller`. Training outputs: `outputs/training`. BFCL process and final outputs: `outputs/eval-v4-checkpoints`. Operations, queue state, logs, and hardware snapshots: `operations`.

2026-09-19 19:04 UTC启动第一批。Qwen2.5-3B-Instruct已完整下载，仓库38项测试通过，两个dry-run确认train batch512、rollout G4、optimizer mini-batch128、micro-batch64、learning rate1e-6、100steps、save/test frequency10和四rank。DVAO seed1使用GPU0–3，GD²PO-Hard seed1使用GPU4–7；两项均完成step0 validation并进入step1，初始validation score1.23376、Format0.1125、Correctness1.12126。8张GPU显存约28.8–29.0GiB并开始计算，无训练失败。控制器会自动补入两个seed2，并在GPU空出后对四个run的steps10/20/…/100执行40份BFCL V4过程评测，其中step100同时计入final。checkpoint权重上传保持停止。

2026-09-19用户确认DVAO与GD²PO-Hard的四个新增run均需完整过程checkpoint评测。队列目标固定为4次训练、40份checkpoint保存、40份BFCL V4过程评分和4份step100 final评分；训练中已保存的过程checkpoint全部保留并进入同一评测协议。

评测队列已在不中断训练worker的情况下扩展：控制器先正常退出并保留两条活动训练，queue_state从4项step100扩为40个run-step，旧的空final-only目录已清除，输出统一改为`outputs/eval-v4-checkpoints`；随后控制器重启并接回原训练PID。重启事件确认expected_checkpoints=40、expected_finals=4，当前DVAO seed1/Hard seed1继续推进至step55/56，8张GPU仍在工作。

2026-09-19 19:18 UTC两条通道均完成step4并进入step5。DVAO seed1的有限grad norm78.54、policy loss0.0240，GD²PO-Hard seed1的有限grad norm52.53、policy loss-0.0488；未出现NaN/Inf。当前完整训练0/4、已保存checkpoint0/40、final0/4。控制器初始汇总因定制manifest缺少evaluation字段报错两次，已补入固定V4/final-step100字段并验证空表导出成功，queue的report_dirty已恢复false；训练未受影响，后台继续。

2026-09-19 19:29 UTC DVAO seed1完成step7，GD²PO-Hard seed1完成step8并进入step9；两路全部记录值有限，最新grad norm38.92/43.09、policy loss0.0170/-0.0268。当前完整训练0/4、已保存checkpoint0/40、final0/4。GPU0–3利用率99–100%、显存38.7–40.1GiB，GPU4–7利用率62–68%、显存35.3–35.5GiB，阶段差异来自各自当前计算段；控制器与两路训练均存活，后台继续。

2026-09-19 19:40 UTC两路seed1均完成step11，step10 checkpoint各一份已完整发布到持久盘，每份13,604,379,681 bytes、12个文件，本地staging已清空。DVAO/Hard最新grad norm11.63/20.78、policy loss-0.0119/0.0780，全部指标有限。当前完整训练0/4、checkpoint保存2/40、final0/4；8张GPU利用率96–100%、显存37.6–40.2GiB，无活动失败，后台继续。

2026-09-19 19:51 UTC两路seed1均完成step15，step10两份checkpoint继续完整保留，无copying积压。DVAO/Hard最新grad norm2.45/3.77、policy loss0.0600/-0.0184，累计日志无NaN/Inf。当前完整训练0/4、checkpoint保存2/40、final0/4；8张GPU利用率90–100%、显存36.8–39.1GiB，控制器与训练均正常，后台继续。

2026-09-19 20:02 UTC两路seed1均完成step19，step10两份checkpoint完整且无copying积压；下一步将执行step20验证与保存。DVAO/Hard最新grad norm2.86/3.45、policy loss0.0351/-0.1196，累计日志无NaN/Inf。当前完整训练0/4、checkpoint保存2/40、final0/4；GPU0–3利用率96–98%，GPU4–7处于较低显存计算阶段但仍有39–62%利用率，两路日志均持续推进，后台继续。

2026-09-19 20:13 UTC两路seed1均完成step23并进入step24，step20 checkpoint均已完整发布且无copying积压；累计checkpoint4/40。step20 validation Format两路均为0.95，Correctness为1.70975/1.69428；最新grad norm22.99/3.82，累计日志无NaN/Inf。当前完整训练0/4、final0/4；GPU0–7利用率98–100%，后台继续。

2026-09-19 20:24 UTC DVAO seed1完成step27，GD²PO-Hard seed1完成step28并进入step29；累计checkpoint4/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward为0.9214/0.9292，grad norm1.90/58.05，累计日志无NaN/Inf。GPU0–3利用率99–100%，GPU4–7为64–70%，两路持续推进，后台继续。

2026-09-19 20:35 UTC两路seed1均完成step31并进入step32，step30 checkpoint均已完整发布，累计checkpoint6/40且无copying积压。step30 validation Format均为0.975，Correctness为1.81755/1.70474；最新grad norm1.59/2.73，累计日志无NaN/Inf。当前完整训练0/4、final0/4；8张GPU利用率96–100%，后台继续。

2026-09-19 20:46 UTC DVAO seed1完成step35，GD²PO-Hard seed1完成step36并进入step37；累计checkpoint6/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward0.9521/0.9473，grad norm1.68/3.20，累计日志无NaN/Inf。各GPU处在不同计算/权重阶段，利用率48–100%，两路持续推进，后台继续。

2026-09-19 20:57 UTC DVAO seed1完成step39并正在执行step40，GD²PO-Hard seed1已完成step40并进入step41。Hard step40 validation Format0.975、Correctness1.67489，checkpoint已从staging完整发布，copying目录清空；累计checkpoint7/40、完整训练0/4、final0/4。两路累计日志无NaN/Inf，8张GPU继续工作，后台继续。

2026-09-19 21:09 UTC DVAO seed1完成step43，GD²PO-Hard seed1完成step44；两份step40 checkpoint均已完整发布，累计checkpoint8/40、完整训练0/4、final0/4，无copying积压。step40 validation Format均为0.975，Correctness为1.72410/1.67489；最新grad norm1.64/4.60，累计日志无NaN/Inf。GPU0–3利用率65–68%，GPU4–7利用率99–100%，两路持续推进，后台继续。

2026-09-19 21:20 UTC DVAO seed1完成step48并进入step49，GD²PO-Hard seed1完成step49并进入step50；累计checkpoint8/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward0.9468/0.9624，policy loss-0.0171/-0.0242；Hard的记录grad norm为有限的467.25，协议grad clip1.0继续生效，当前loss与其他指标正常且无NaN/Inf。8张GPU利用率50–71%，两路日志持续推进，后台继续。

2026-09-19 21:31 UTC DVAO seed1完成step52，GD²PO-Hard seed1完成step53；两份step50 checkpoint均已完整发布，累计checkpoint10/40、完整训练0/4、final0/4，无copying积压。step50 validation Format均为0.975，Correctness为1.70785/1.72197；最新grad norm1.69/3.57，累计日志无NaN/Inf。8张GPU处于不同计算阶段但两路持续推进，后台继续。

2026-09-19 21:50 UTC DVAO seed1完成step59并进入step60，GD²PO-Hard seed1已完成step60；Hard step60 checkpoint已完整发布，累计checkpoint11/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。Hard step60 validation Format0.975、Correctness1.72823；两路累计日志无NaN/Inf。8张GPU利用率97–100%，后台继续。

2026-09-19 22:02 UTC DVAO seed1完成step63，GD²PO-Hard seed1完成step65并进入step66；两份step60 checkpoint均已完整发布，累计checkpoint12/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。step60 validation Format均为0.975，Correctness为1.68339/1.72823；最新grad norm1.49/3.84，累计日志无NaN/Inf。GPU0–3利用率98–100%，GPU4–7为49–68%，两路持续推进，后台继续。

2026-09-19 22:14 UTC DVAO seed1完成step68并进入step69，GD²PO-Hard seed1完成step69；累计checkpoint12/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward0.9644/0.9604，DVAO有限grad norm91.47、Hard3.62，policy loss均正常且累计无NaN/Inf。GPU0–7利用率95–100%，后台继续。

2026-09-19 22:25 UTC DVAO seed1完成step72，GD²PO-Hard seed1完成step73；两份step70 checkpoint均已完整发布，累计checkpoint14/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。step70 validation Format均为0.975，Correctness为1.70214/1.79815；DVAO有限grad norm191.44、Hard3.77，policy loss正常且累计无NaN/Inf。8张GPU继续工作，后台继续。

2026-09-19 22:37 UTC DVAO seed1完成step77并进入step78，GD²PO-Hard seed1完成step78并进入step79；累计checkpoint14/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward0.9565/0.9756，grad norm1.34/3.61，累计日志无NaN/Inf。GPU0–3利用率62–70%，GPU4–7为88–100%，两路持续推进，后台继续。

2026-09-19 22:48 UTC DVAO seed1完成step81并进入step82，GD²PO-Hard seed1完成step82；两份step80 checkpoint均已完整发布，累计checkpoint16/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。step80 validation Format均为0.975，Correctness为1.73035/1.74731；最新grad norm1.23/4.90，累计日志无NaN/Inf。GPU0–3利用率51–63%，GPU4–7为98–100%，后台继续。

2026-09-19 22:59 UTC DVAO seed1完成step85，GD²PO-Hard seed1完成step87并进入step88；累计checkpoint16/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。两路最新Format reward0.9580/0.9614，grad norm1.10/4.96，累计日志无NaN/Inf。8张GPU利用率97–100%，后台继续。

2026-09-19 23:10 UTC DVAO seed1完成step89并进入step90，GD²PO-Hard seed1完成step91；Hard step90 checkpoint已完整发布，累计checkpoint17/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。Hard step90 validation Format0.975、Correctness1.82632；两路累计日志无NaN/Inf。8张GPU利用率97–100%，后台继续。

2026-09-19 23:21 UTC DVAO seed1完成step93，GD²PO-Hard seed1完成step96并进入step97；两份step90 checkpoint均已完整发布，累计checkpoint18/40、过程评分0/40、完整训练0/4、final0/4，无copying积压。step90 validation Format均为0.975，Correctness为1.76578/1.82632；最新grad norm52.78/5.42，累计日志无NaN/Inf。GPU0–3利用率97–99%，GPU4–7为60–68%，两路持续推进，后台继续。

2026-09-19 23:33 UTC GD²PO-Hard seed1完成100steps、101行metrics和十份checkpoint，exit0，实耗16,110.09秒；完整训练达到1/4、checkpoint保存19/40。释放的GPU4–7在约1.5秒后补入DVAO seed2，已完成step0 validation并进入step1；DVAO seed1正在执行step100。过程评分0/40、final0/4，无copying积压或活动失败，8张GPU继续两路训练。

2026-09-19 23:46 UTC第一批全部完成：DVAO seed1于23:39:06 UTC完成100steps、101行metrics和十份checkpoint，exit0，实耗16,455.55秒；其step100 validation Format0.975、Correctness1.75857。GD²PO-Hard seed1对应结果为Format0.975、Correctness1.82632。完整训练达到2/4、checkpoint保存20/40，两个完成run的20份评测均已转pending_inference。GPU0–3已补入GD²PO-Hard seed2并完成step1，GPU4–7的DVAO seed2完成step3；过程评分0/40、final0/4，无copying积压或活动失败，8张GPU利用率99–100%，后台继续。

2026-09-19 23:57 UTC第二批持续训练：DVAO seed2完成step7，GD²PO-Hard seed2完成step5；最新Format reward0.1563/0.1782、grad norm14.84/46.42，全部指标有限。当前完整训练2/4、checkpoint保存20/40、过程评分0/40、final0/4；已完成seed1的20份评测保持pending_inference，等待训练卡释放。8张GPU利用率88–100%，无活动失败，后台继续。

2026-09-20 00:08 UTC DVAO seed2完成step11并进入step12，step10 checkpoint已完整发布；GD²PO-Hard seed2完成step9并正在执行step10。当前完整训练2/4、checkpoint保存21/40、过程评分0/40、final0/4，无copying积压。DVAO seed2 step10 validation Format0.8125、Correctness1.71216；两路累计日志无NaN/Inf。8张GPU继续两路训练，后台继续。

2026-09-20 00:19 UTC DVAO seed2完成step15，GD²PO-Hard seed2完成step13；两份step10 checkpoint均已完整发布，累计checkpoint22/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step10 validation Format为0.8125/0.925，Correctness为1.71216/1.74647；最新grad norm2.14/4.32，累计日志无NaN/Inf。8张GPU利用率96–98%，后台继续。

2026-09-20 00:31 UTC DVAO seed2完成step19并正在执行step20，GD²PO-Hard seed2完成step17；累计checkpoint22/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9072/0.9160，grad norm2.14/4.46，累计日志无NaN/Inf。8张GPU利用率99–100%，后台继续。

2026-09-20 00:42 UTC DVAO seed2完成step23并进入step24，GD²PO-Hard seed2完成step21并进入step22；两份step20 checkpoint均已完整发布，累计checkpoint24/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step20 validation Format均为0.95，Correctness为1.79145/1.73549；最新grad norm2.17/3.18，累计日志无NaN/Inf。8张GPU利用率97–98%，后台继续。

2026-09-20 00:53 UTC DVAO seed2完成step27并进入step28，GD²PO-Hard seed2完成step25并进入step26；累计checkpoint24/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9419/0.9458，grad norm1.80/3.32，累计日志无NaN/Inf。8张GPU利用率98–99%，后台继续。

2026-09-20 01:04 UTC DVAO seed2完成step31并进入step32，step30 checkpoint已完整发布；GD²PO-Hard seed2完成step29并正在执行step30。当前完整训练2/4、checkpoint保存25/40、过程评分0/40、final0/4，无copying积压。DVAO seed2 step30 validation Format0.975、Correctness1.77431；两路累计日志无NaN/Inf。8张GPU继续工作，后台继续。

2026-09-20 01:15 UTC DVAO seed2完成step35并进入step36，GD²PO-Hard seed2完成step33并进入step34；两份step30 checkpoint均已完整发布，累计checkpoint26/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step30 validation Format均为0.975，Correctness为1.77431/1.88859；最新grad norm1.49/13.76，累计日志无NaN/Inf。8张GPU利用率97–98%，后台继续。

2026-09-20 01:26 UTC DVAO seed2完成step39并正在执行step40，GD²PO-Hard seed2完成step37；累计checkpoint26/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9365/0.9580，grad norm1.46/3.25，累计日志无NaN/Inf。8张GPU利用率97–100%，后台继续。

2026-09-20 01:37 UTC DVAO seed2完成step43并进入step44，GD²PO-Hard seed2完成step41；两份step40 checkpoint均已完整发布，累计checkpoint28/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step40 validation Format均为0.975，Correctness为1.73292/1.78288；Hard有限grad norm313.18，固定grad clip1.0生效，policy loss正常且累计无NaN/Inf。8张GPU继续两路训练，后台继续。

2026-09-20 01:48 UTC DVAO seed2完成step47并进入step48，GD²PO-Hard seed2完成step45；累计checkpoint28/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9556/0.9639，grad norm1.27/3.47，累计日志无NaN/Inf。巡检时DVAO正进行CPU reward计算，GPU4–7瞬时空闲但日志持续写入；Hard训练卡利用率96–100%，后台继续。

2026-09-20 02:00 UTC DVAO seed2完成step51，GD²PO-Hard seed2完成step50；两份step50 checkpoint均已完整发布，累计checkpoint30/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step50 validation Format均为0.975，Correctness为1.72164/1.76472；Hard有限grad norm136.82，固定grad clip1.0生效，policy loss正常且累计无NaN/Inf。8张GPU利用率98–100%，后台继续。

2026-09-20 02:11 UTC DVAO seed2完成step56并进入step57，GD²PO-Hard seed2完成step54；累计checkpoint30/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9629/0.9683，grad norm1.45/4.51，累计日志无NaN/Inf。GPU0–3利用率99–100%，GPU4–7为60–65%，两路持续推进，后台继续。

2026-09-20 02:22 UTC DVAO seed2完成step60，step60 checkpoint已完整发布；GD²PO-Hard seed2完成step59并正在执行step60。当前完整训练2/4、checkpoint保存31/40、过程评分0/40、final0/4，无copying积压。DVAO seed2 step60 validation Format0.975、Correctness1.85132；两路累计日志无NaN/Inf。8张GPU继续两路训练，后台继续。

2026-09-20 02:34 UTC DVAO seed2完成step64，GD²PO-Hard seed2完成step63；两份step60 checkpoint均已完整发布，累计checkpoint32/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step60 validation Format均为0.975，Correctness为1.85132/1.77179；最新grad norm1.44/3.81，累计日志无NaN/Inf。8张GPU利用率99–100%，后台继续。

2026-09-20 02:45 UTC DVAO seed2完成step69并进入step70，GD²PO-Hard seed2完成step68并进入step69；累计checkpoint32/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。Hard step68记录有限的裁剪前grad norm485,824.12，固定grad clip1.0生效，policy loss0.0362、Format reward0.9526及其他指标正常；两路累计日志无NaN/Inf，后台继续。

2026-09-20 02:56 UTC DVAO seed2完成step73并进入step74，GD²PO-Hard seed2完成step72并进入step73；两份step70 checkpoint均已完整发布，累计checkpoint34/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step70 validation Format均为0.975，Correctness为1.81685/1.79276；最新grad norm1.38/4.47，累计日志无NaN/Inf。两路持续推进，后台继续。

2026-09-20 03:08 UTC DVAO seed2完成step77，GD²PO-Hard seed2完成step76；累计checkpoint34/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9536/0.9790，grad norm1.22/4.64，累计日志无NaN/Inf。8张GPU利用率98–100%，后台继续。

2026-09-20 03:19 UTC DVAO seed2完成step81，GD²PO-Hard seed2完成step80；两份step80 checkpoint均已完整发布，累计checkpoint36/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step80 validation Format均为0.975，Correctness为1.81036/1.79926；最新grad norm1.71/3.99，累计日志无NaN/Inf。8张GPU利用率98–100%，后台继续。

2026-09-20 03:30 UTC DVAO seed2完成step86并进入step87，GD²PO-Hard seed2完成step85；累计checkpoint36/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。两路最新Format reward0.9492/0.9761，grad norm1.35/4.02，累计日志无NaN/Inf。8张GPU利用率97–98%，后台继续。

2026-09-20 03:41 UTC DVAO seed2完成step90并进入step91，step90 checkpoint已完整发布；GD²PO-Hard seed2完成step89并正在执行step90。当前完整训练2/4、checkpoint保存37/40、过程评分0/40、final0/4，无copying积压。DVAO step90 validation Format0.9625、Correctness1.78919；两路累计日志无NaN/Inf。8张GPU继续两路训练，后台继续。
2026-09-20 03:52 UTC DVAO seed2完成step95，GD²PO-Hard seed2完成step94；两份step90 checkpoint均已完整发布，累计checkpoint38/40、过程评分0/40、完整训练2/4、final0/4，无copying积压。step90 validation Format为0.9625/0.9875，Correctness为1.78919/1.79120；最新grad norm1.12/93.98，累计日志无NaN/Inf。8张GPU利用率96–100%，两路接近完成，后台继续。
2026-09-20 04:04 UTC两条seed2通道均已完成step99并进入step100的最终验证与保存阶段；当前完整训练2/4、checkpoint保存38/40、过程评分0/40、final0/4，无copying积压。DVAO/Hard最新grad norm3.17/5.61、policy loss-0.0097/-0.0533，四个run累计指标均无NaN/Inf。GPU0–7处于最终验证的不同计算阶段，利用率38–100%；控制器和两路训练仍存活，后台继续，完成后会自动转为8卡并行BFCL V4评测。
2026-09-20 04:15 UTC两条训练通道均已完成：DVAO seed2和GD²PO-Hard seed2分别于04:06/04:11 UTC以exit0结束，四个run全部为101行metrics且steps10–100十份checkpoint齐全，累计完整训练4/4、checkpoint保存40/40。seed2 step100 validation的Format为0.975/0.9875，Correctness为1.79857/1.78435；四个run累计无NaN/Inf。控制器已立即切换到评测，8张GPU分别运行DVAO seed1的step100及step10–70，raw BFCL V4 JSONL持续写入，GPU利用率47–92%、显存约35.4GiB；当前过程评分0/40、final0/4，无失败，后台继续并会逐项补位。
2026-09-20 04:26 UTC训练通道均已完成step100，完整训练4/4、checkpoint40/40；四个run均为101行metrics且累计无NaN/Inf。首批8份DVAO seed1 BFCL V4推理在8张GPU上并行，14类输出均已创建，每份已生成约2694–3077/3301个case，GPU利用率59–92%、显存约35.4GiB。当前过程评分0/40、final0/4，日志持续推进且无基础设施失败；个别empty response由固定评测器按case记录并继续，后台继续自动补位。
2026-09-20 04:47 UTC训练与checkpoint保持4/4、40/40。首份过程评分和final已经完成：DVAO seed1 step100覆盖3301个case，Live Acc/Format 70.32/100.00，Non-Live 82.46/100.00，Multi-Turn 9.75/89.57，Average 54.18/96.52；四份汇总CSV均已生成并包含该行。step100释放GPU后已立即补入step80，当前8份推理并行、队列为complete1、running8、pending31，GPU利用率56–94%、显存约35.4GiB。固定协议中的超长请求400和空响应均按case记录，任务exit0且无基础设施失败；过程评分1/40、final1/4，后台继续自动补位。
2026-09-20 04:58 UTC训练与checkpoint保持4/4、40/40。DVAO seed1 step70完成3301-case推理并exit0评分：Live Acc/Format 70.17/100.00，Non-Live 82.69/100.00，Multi-Turn 7.88/85.49，Average 53.58/95.16；四份CSV已更新为两条checkpoint记录和一条final记录。GPU3释放后立即补入step90，当前过程评分2/40、final1/4，队列running8、pending30；GPU利用率35–94%、显存约35.4GiB，8份推理均持续推进，无基础设施失败，后台继续自动补位。
2026-09-20 05:09 UTC训练与checkpoint保持4/4、40/40，四个run的101行metrics均无NaN/Inf。DVAO seed1的step30/40/50/60已陆续完成3301-case推理并exit0评分，加上step70/100，实际过程评分达到6/40、final1/4；四份CSV对应checkpoint数据现有6行。控制器已补入GD²PO-Hard seed1的step100/10/20/30，当前仍保持8份GPU推理，剩余pending26；GPU2–7利用率59–97%，GPU0/1正处新任务模型加载阶段，其中GPU0显存已升至34.9GiB、GPU1约6.6GiB，无任务失败，后台继续自动补位。
2026-09-20 05:20 UTC训练与checkpoint保持4/4、40/40，四个run指标仍全部有限。DVAO seed1新增完成step20和step80评分，累计完成step20/30/40/50/60/70/80/100八份，过程评分8/40、final1/4，四份CSV已更新。当前DVAO seed1 step10/90与GD²PO-Hard seed1 step100/10/20/30/40/50占满8张GPU，remaining pending24；GPU利用率74–92%、显存约35.4GiB，无基础设施失败，后台继续自动补位。
2026-09-20 05:31 UTC训练与checkpoint保持4/4、40/40。DVAO seed1的十份checkpoint已全部完成评分；GD²PO-Hard seed1 step100也已完成，Live Acc/Format 70.10/99.78，Non-Live 84.42/100.00，Multi-Turn 8.25/92.36，Average 54.25/97.38。当前过程评分11/40、final2/4，四份CSV已同步更新；Hard seed1的step10–80占满8张GPU，pending21，GPU0/1/3–7利用率58–95%，GPU2正加载step80新任务，无失败，后台继续自动补位。
2026-09-20 05:42 UTC训练与checkpoint保持4/4、40/40，过程评分11/40、final2/4。GD²PO-Hard seed1的step10/20/30/40/50/60/70/80在8张GPU并行，每份已生成约2501–2956/3301个case；GPU利用率88–95%、显存约35.4GiB。控制器、推理日志和队列持续推进，无失败或空槽，后台继续自动补位。
2026-09-20 05:53 UTC训练与checkpoint保持4/4、40/40，过程评分11/40、final2/4。GD²PO-Hard seed1的step10–80八份推理均已推进至约2765–3170/3301个case，距离首批过程项收尾很近；GPU利用率79–97%、显存约35.4GiB，控制器和全部任务日志持续更新，无失败或空槽，后台继续自动补位。
2026-09-20 06:04 UTC训练与checkpoint保持4/4、40/40。GD²PO-Hard seed1的step70和step80已完成3301-case推理并exit0评分，过程评分升至13/40、final2/4；四份CSV已更新。两张释放GPU已立即补入Hard seed1 step90和DVAO seed2 step100，当前仍有8份推理并行、pending19；GPU0/1/3/4/6/7利用率57–95%，GPU2/5正处新任务加载阶段，无失败，后台继续自动补位。
2026-09-20 06:15 UTC训练与checkpoint保持4/4、40/40。GD²PO-Hard seed1的step50和step60已完成评分，过程评分升至15/40、final2/4，四份CSV已更新。当前Hard seed1 step10/20/30/40/90与DVAO seed2 step100/10/20共8份并行，pending17；GPU利用率61–92%、显存约35.4GiB，所有任务日志持续推进且无失败，后台继续自动补位。
2026-09-20 06:26 UTC训练与checkpoint保持4/4、40/40。GD²PO-Hard seed1的step10/30/40已完成评分，过程评分升至18/40、final2/4，四份CSV已更新。当前Hard seed1 step20/90与DVAO seed2 step100/10/20/30/40/50共8份并行，pending14；GPU0–6利用率42–93%，GPU7刚补入DVAO seed2 step50并处加载阶段，无失败，后台继续自动补位。
2026-09-20 06:47 UTC训练与checkpoint保持4/4、40/40。GD²PO-Hard seed1剩余step20/90已完成，使该run十份过程评分全部结束；DVAO seed2 step100 final也已完成，Live Acc/Format 70.32/99.85，Non-Live 83.23/100.00，Multi-Turn 9.88/91.59，Average 54.47/97.15。当前过程评分21/40、final3/4，四份CSV已更新；DVAO seed2 step10–80占满8张GPU且各自约2800–2940/3301 case，pending11，GPU利用率90–94%，无失败，后台继续自动补位。
2026-09-20 06:58 UTC训练与checkpoint保持4/4、40/40，过程评分21/40、final3/4。DVAO seed2的step10–80八份并行推理已分别推进到约2880–3186/3301个case，进入当前批次收尾阶段；GPU利用率89–94%、显存约35.4GiB，控制器和任务日志持续更新，无失败或空槽，后台继续自动评分与补位。
2026-09-20 07:09 UTC训练与checkpoint保持4/4、40/40。DVAO seed2 step80已完成评分，过程评分22/40、final3/4，四份CSV已更新；释放的GPU5立即补入step90。当前DVAO seed2 step10–70与step90共8份并行，其余七份已推进至约2976–3283/3301个case，pending10；GPU0–4/6/7利用率86–93%，GPU5处新任务加载阶段，无失败，后台继续自动补位。
2026-09-20 07:20 UTC训练与checkpoint保持4/4、40/40。DVAO seed2的step70、step80和step20已完成评分，实际过程评分24/40、final3/4，四份CSV持续更新。当前DVAO seed2 step10/30/40/50/60/90与GD²PO-Hard seed2 step100/10组成8份并行，pending8；GPU3刚补入Hard seed2 step10并处加载阶段，其余GPU持续计算，无失败，后台继续自动补位。
2026-09-20 07:31 UTC训练与checkpoint保持4/4、40/40。DVAO seed2的step10/40/50/60等本批过程项陆续完成，过程评分升至28/40、final3/4，四份CSV已更新。当前DVAO seed2仅剩step30/90仍在跑，GD²PO-Hard seed2的step100/10/20/30/40/50并行，pending仅剩Hard step60–90四份；8张GPU利用率59–96%、显存约35.4GiB，无失败，后台继续自动补位。
2026-09-20 07:43 UTC训练与checkpoint保持4/4、40/40。DVAO seed2 step90与GD²PO-Hard seed2 step100已完成评分，四份final现已全部完成；Hard seed2 final的Live Acc/Format 70.61/99.70，Non-Live 83.23/100.00，Multi-Turn 10.50/88.47，Average 54.78/96.06。实际过程评分30/40、final4/4，四份CSV已更新；当前DVAO seed2仅剩step30，Hard seed2 step10–70共7份占满8张GPU，pending只剩step80/90两份。GPU2刚补入step70处加载阶段，其余GPU持续计算，无失败，后台继续。
2026-09-20 07:54 UTC训练与checkpoint保持4/4、40/40，final4/4。DVAO seed2最后的step30已完成，因此该run十份过程评分全部结束；当前过程评分31/40。GD²PO-Hard seed2的step10–80共8份正在并行，step90是队列中唯一pending项；GPU利用率81–97%、显存约35.4GiB，所有任务持续推进且无失败，后台继续自动补位。
2026-09-20 08:05 UTC训练与checkpoint保持4/4、40/40，final4/4。GD²PO-Hard seed2 step50已完成评分，过程评分32/40；step90已补入，队列现在没有pending项。最后8份step10/20/30/40/60/70/80/90全部正在GPU上运行，GPU利用率74–98%、显存约35.4GiB，无失败，后台继续直至全部评分和CSV汇总完成。
2026-09-20 08:17 UTC训练与checkpoint保持4/4、40/40，final4/4。GD²PO-Hard seed2的step40/50/60/70/80已完成评分，实际过程评分36/40；队列没有pending项，仅余step10/20/30/90四份推理，均已生成约2982–3064/3301个case。GPU0/3/6/7继续计算，其他GPU因无剩余可并行项已释放；无失败，后台继续完成最后四份并核对CSV。
2026-09-20 08:28 UTC训练与checkpoint保持4/4、40/40，final4/4。GD²PO-Hard seed2的step10/30/40/60/70/80等已完成评分，实际过程评分38/40。仅剩step20和step90两份推理，分别已生成3277/3301和3292/3301个case；GPU0/6继续最后计算，其他GPU因无待运行项已释放。无失败，后台继续完成最后两份并核对最终四份CSV。
2026-09-20 08:39 UTC全部目标完成。四次100-step训练均exit0，完整训练4/4；四个run各保存steps10/20/…/100十份HF checkpoint，合计40/40且无copying积压；四个metrics.jsonl各101行且无NaN/Inf。BFCL V4过程评分40/40、step100 final4/4，所有evaluation的inference和score均只执行一次且状态complete；控制器以reason=all_complete正常结束，8张GPU已释放。最终CSV核对通过：checkpoint_per_model.csv 41行、checkpoint_method_results.csv 21行、process_final_per_model.csv 5行、process_final_method_results.csv 3行，均无空行。checkpoint权重未上传。
