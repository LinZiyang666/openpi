# 协调员命令：真实并发服务路径（2026-10-18 更新）

旧 B1 命令保存在 `COMMANDS_B1_ARCHIVE.md`，不能作为部署等价证据。本文件所有 4090/H100 GPU 命令只由协调员执行；工具只创建自己的 residents、stock proxy 和私有 MPS，关闭时只回收这些进程。不会减副本、恢复副本、停止 lane 或改 chain。端口占用/内存不足即拒绝启动。新`--run`会独立核对封存cache文件（前后SHA256），再保存内容寻址的只读harness源码快照，随后resident和客户端从快照运行，工具更新不会改变它；结束后另写EXECUTION_HARNESS.json。H100运行前须先用下面的stage脚本同步当前工具。

## 最新：π0.5 H100已修复CPU身份失败；FW跨卡已配对

π0.5 H100失败是四个下载metadata时间戳造成checkpoint身份不匹配。已完成专用checkpoint目录、同SHA权重硬链接、输入/cache/依赖/runtime的CPU预检；新增内容8,457 bytes，无GPU启动。见 [修复与收据](PI05_H100_STAGING_FIX_1018.md:1)。**23177留给正式campaign，下面历史章节的23177/stage_h100.sh命令不再用于本次补传。** 新临时脚本自动检查ss与活跃campaign全部ports，传完自动关闭；这两项CPU工作已实际完成，以下可复现：

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/temporary_h100_stage.py pi05 --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/temporary_h100_stage.py fw --run
```

均GPU **0GiB**；pi最多164MB增量，约3–5分钟（含限速本地权重hash、不传权重）；FW约306MB，约2–6分钟。16MiB/s单流，精确allowlist、只读、原IP白名单；成功和失败均关闭自有daemon。本次用23198，主动避开未监听但活跃campaign保留的23199。FW跨卡2,283/2,304超暂定预算、145/2,304超5×floor（全cache）、top1一致928/960；见 [Fast-WAM分析](FASTWAM_4090_H100_1018.md:1)。

**H100下一lane边界：π0.5下式执行两次，每次44GiB空闲、15–25分钟（watchdog25）。** 使用更新后的同版本harness；其余R4/B8/112、数值、cache和正式bootstrap不变：

```bash
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model pi05 --checkpoint-view /data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944 --batch-telemetry --run
```

## 本次新增：五模型闭环 pilot（CPU 接线就绪）

**20个campaign，每GPU每模型pure/cache各500集；正常prepare/launch、autoscale OFF，固定原生worker数112/40/48/12/44；仅timan1 GPU2和timan107。** 每条prepare+launch及每campaign GPUh、renderer预算已列于 [CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md)，脚本在 `closedloop_launch_1018/`。本轮未启动GPU、worker或chain。两条模型链共享本研究renderer锁；真实资源不足时启动前退出，不修改其他lane。

DP残余漂移CPU定位见 [DP_CACHE_LOCALISATION_1018.md](DP_CACHE_LOCALISATION_1018.md)：51个超预算请求的距离→权重可精确回放，corrector均为0；混合+反归一化解释动作到2.38e-7。**仍未恢复等价。** 剩余大显存开环顺序π→GR→Cosmos，各两冷启动、各自lane边界；FW已定窗口不变。

## 0. DP与Fast-WAM复现命令（GPU轮次已完成）

**协调员完成记录（本地CDT）：** FW 4090两次09:05–09:18、H100两次07:36–07:44；日志为`evidence/run_concurrent_{4090,h100}_fw_{1,2}.log`。下面GPU命令保留复现，不是新的待运行项；其余模型/闭环排期保持原样。

**DP两卡TF32-off各两个冷启动已完成并分析（截至07:34 CDT）：** 4090/H100同卡floor分别0.0005117655/0.0001524687，跨卡0.0006102324；51/2,304超硬预算（全cache），131/2,304超逐请求floor，尚未恢复含cache的DP等价。576个cache head的可观测决策一致。详见[双卡分析](DP_BOTH_TF32OFF_1018.md)。本轮无需重复GPU运行；下面4090/H100 TF32-off命令仅保留复现，原8GiB空闲、预计5–10分钟/watchdog15分钟预算不变。FW排期和其余大显存命令保持原样。

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model dp --deployment longseg_recal1018 --batch-telemetry --numerics tf32_off --run
```

DP两卡各两个冷启动已完成并分析：跨卡动作max-abs=0.0014218092，779/2,304请求超硬预算，7个超逐请求5×floor；可观测cache决策一致，但stall/knob活跃分支和闭环尚未覆盖。见 [DP分析](DP_4090_H100.md)。下一模型选择实际4090 CPU-T5 cache profile `fw4090_reduced`；两卡保持同一PIN、R2/B16、32客户端，不与H100 full profile混合。每条GPU命令独立执行两次，建立冷启动floor。

**4090 Fast-WAM：42GiB空闲；每次预计8–15分钟，watchdog35分钟。** 这是接近整张48GB卡的窗口；由协调员安排资源。CPU预检已通过，见[evidence](evidence/next_fw4090_CPU_plan.jsonl)。

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model fastwam --deployment fw4090_reduced --batch-telemetry --run
```

**H100 DP TF32-off复现命令（两次已完成，无待运行项）：8GiB空闲；每次预计5–10分钟，watchdog15分钟。** 保持R4/B8/48客户端，是原消融顺序第一项；native已经highest、matmul TF32=False，所以这一步有效变化主要是cuDNN TF32关闭。native/TF32-off启动驻留均5,916MiB，峰值尚未实测；8GiB是准入要求而非硬显存隔离。若无8GiB空闲，入口拒绝启动。此步用于定位DP问题，不计作下一个模型完成。

```bash
tether exec --timeout 20m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model dp --deployment longseg_recal1018 --batch-telemetry --numerics tf32_off --run
```

**H100下一模型 Fast-WAM：42GiB空闲；每次预计8–15分钟，watchdog35分钟。** 只有正式任务同时留下至少42GiB空闲才可旁跑；否则留到lane边界。保持原R2/B16/54GiB allocator配置；不把单副本低显存筛查当完整部署证据。工具会检查空闲显存和29600–29602端口，不停止任何已有服务。CPU预检已通过，见[evidence](evidence/next_fwh100_CPU_plan.jsonl)。

```bash
tether exec --timeout 40m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model fastwam --deployment fw4090_reduced --batch-telemetry --run
```

协调员已完成H100 stage；本轮只更新CPU分析工具，模型执行harness身份仍为8498998d…a1096，以上命令无需重复传大文件。CPU比较/摘要占GPU **0GiB**；若运行前另有工具更新，再执行第2节stage。原其余模型与闭环计划保持不变。

## 0.1 π0.5、GR00T、Cosmos后续并发窗口

六条命令已完成两主机CPU预检，见[收据](evidence/pi_gr_cosmos_six_CPU_plans_20261018.json)。每条GPU命令执行**两个独立冷启动**；表内分钟为每次预计值，括号为watchdog。显存是空闲准入请求，不是已测峰值或硬隔离。只在协调员分配的窗口启动，缺显存/端口即拒绝，不调整已有lane。

| 模型 | 4090空闲GiB / 分钟 | H100空闲GiB / 分钟 | R/B/客户端与证据边界 |
|---|---|---|---|
| π0.5 MT50 K10 | 44 / 15–25（25） | 44 / 15–25（25） | 两卡R4/B8/112；源C5，没有新longseg overlay |
| GR00T MT50 K4 | 42 / 8–15（25） | 76 / 8–15（25） | 4090 R6/B1/40是显式容量arm；H100原R11/B1/40；新longseg recipe |
| Cosmos spatial reduced | 47 / 15–25（30） | 76 / 15–25（30） | 两卡保留源R8/B1/44及76GiB allocator；4090物理容量尚待验证 |

4090三条命令（逐模型、逐次执行）：

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model pi05 --batch-telemetry --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model groot --deployment longseg_recal1018 --capacity-arm --batch-telemetry --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model cosmos --batch-telemetry --run
```

H100三条命令（逐模型、逐次执行；76GiB需要接近整卡窗口）：

```bash
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model pi05 --checkpoint-view /data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944 --batch-telemetry --run
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model groot --deployment longseg_recal1018 --batch-telemetry --run
tether exec --timeout 35m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model cosmos --batch-telemetry --run
```

GR4090的R6不是已证实的当前正式R11部署，不据此作“仅换GPU”的因果结论；若要相同R，另在H100明确R6容量arm配对。Cosmos4090若R8准入/峰值失败，保留失败收据并单列容量问题，不自动降R或降低每replica上限。以上不会改变原native配置；也不补齐原profile缺失的长段overlay。

## 1. 4090：DP原始复现命令，8GiB，8–15分钟

04:54两次启动的MPS路径问题已修复，下面CLI不变，重新执行即可使用新harness快照。私有campaign目录改为原子保留的两字符名称；调用真实冻结`mps.paths()`校验，4090 pipe恰为85字节、H100/A100为39字节。27项CPU路径矩阵及碰撞/字节边界测试通过。两次失败未启动daemon或resident；只读检查29600–29699空闲，原有其他MPS未操作。其余R/B/cache/graph及两次冷启动计划保持不变。

执行前检查`cat /proc/pressure/io`及`ss -tn state established '( sport = :23177 )'`；若A100正在拉模型，等单流传输结束后再加载本地大权重，遵守本任务不并行大读的限制。H100 stage脚本也会拒绝与现有23177传输重叠。

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model dp --deployment longseg_recal1018 --batch-telemetry --run
```

DP/GR/FW长段候选已在07:35 UTC完成Q3/knob重标定；本文件明确选择`longseg_recal1018`新PIN。原canonical旧PIN及A100旧结果保留，不能与新PIN混比；DP新PIN的4090/H100各两个冷启动现已完成，GR/FW新PIN仍待对应跨卡结果。4090 FW实际CPU-T5部署的独立profile不变。

这保留 formal R4/B8、5ms batch wait、MPS、DP CUDA graph、原始封存 bootstrap 与 long-segment clock；48 客户端，每轮 pure/cache 各24次 delivery，共2轮、4,608请求。A100实测启动驻留4,544MiB；8GiB是给协调员的保守空闲请求，不是降低原 formal allocator cap。输出自动创建 `results/4090_dp_concurrent_<UTC>/`。此版`--batch-telemetry`同时在native recipe query返回后附加已算出的donor distances/gap/corrector ratio，保留native动作/决策；旧A100 DP未开该observer，须重跑匹配版本后才作跨卡比较。当前 cache 是封存的 n50/C10/knob-off；**并未覆盖 active periodic knob**。

其余模型命令，逐个排期：

| 模型 | 要求空闲显存 | watchdog | 拓扑 / 证据范围 |
|---|---:|---:|---|
| π0.5 MT50 K10 | 44GiB | 25min | R4/B8，112客户端；封存源C5，没有长段overlay |
| GR00T MT50 K4 | 42GiB | 25min | R6/B1，40客户端；源H100为R11，这是显式容量arm |
| Fast-WAM 4090正式CPU-T5 cache路径 | 42GiB | 35min | R2/B16；32压力客户端（源execution_workers=12、desired=70），封存fa_fw_r_u_l；K10 pure与K1/C10 cache单列 |
| Cosmos LIBERO reduced | 47GiB | 30min | R8/B1，44客户端；原76GiB allocator上限，每replica 9.5GiB；待验证48GB卡物理容量 |

```bash
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model pi05 --batch-telemetry --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model groot --deployment longseg_recal1018 --capacity-arm --batch-telemetry --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model fastwam --deployment fw4090_reduced --batch-telemetry --run
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/run_concurrent.py --host 4090 --model cosmos --batch-telemetry --run
```

Cosmos保留原76GiB总allocator上限；将它降到44GiB会把每replica限制为5.5GiB，低于已记录约8GiB加载峰值（RUN_CONFIGS.md:1393），可能制造人工OOM。47GiB是物理空闲请求，尚无4090 R8成功证书。

π的44GiB物理空闲请求保留原40GiB allocator cap，额外空间给多个CUDA context/driver，尚非峰值证书。容量arm用于验证相同正式 server 实现是否能迁移；**不能称为原拓扑已通过**。这些显存数除DP启动值外均是根据已有resident收据加余量的排期预算，尚未实测本 harness 峰值。Fast-WAM上面的命令现使用实际4090 resident收据fa_fw_r_u_l（runtime a7a387ed…，text cache=True）；需留出整卡窗口，42GiB空闲覆盖第二replica原27GiB准入及第一replica约13.3GiB驻留；40GiB可能先被准入拒绝。H100 full runtime 205509…的R2启动收据48,904MiB，已几乎占满4090，A100 R1本轮峰值已超过31GiB；因此不再把该迁移路径冒充4090部署首选命令。显存budget是上限，不能据54/70/76GiB的配置数字直接判定实际用量。

## 2. H100：CPU传输一次，再 DP（8GiB，8–15分钟）

H100已只读确认拥有五份 frozen runtime。新缓存依赖只写入本研究目录；正式 campaign 目录不写入、不覆盖。先运行：

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/xgpu_1017/stage_h100.sh
```

上述命令只有一个rsync流，16MiB/s，约450MB cache metadata/fit加小输入和工具；既有模型/env复用，不复制大权重。然后：

```bash
tether exec --timeout 20m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model dp --deployment longseg_recal1018 --batch-telemetry --run
```

| 模型 | 空闲显存 | watchdog | 拓扑 |
|---|---:|---:|---|
| π0.5 | 44GiB | 25min | R4/B8，112客户端，源4090拓扑迁移 |
| GR00T | 76GiB | 25min | 原H100 R11/B1，40客户端 |
| Fast-WAM | 70GiB | 30min | 原H100 R2/B16，85客户端 |
| Cosmos | 76GiB | 30min | 原H100 R8/B1，44客户端 |

```bash
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model pi05 --checkpoint-view /data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944 --batch-telemetry --run
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model groot --deployment longseg_recal1018 --batch-telemetry --run
tether exec --timeout 35m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model fastwam --deployment longseg_recal1018 --batch-telemetry --run
tether exec --timeout 35m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model cosmos --batch-telemetry --run
```

为比较4090实际CPU-T5 cache部署，H100另跑同一profile，**42GiB空闲、预计8–15min（watchdog35min）**；它与上面H100 full profile分开分析：

```bash
tether exec --timeout 40m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model fastwam --deployment fw4090_reduced --batch-telemetry --run
```

该profile保留原R2/B16、54GiB allocator预算、原bootstrap；32客户端用于填满两个B16（源当前执行12、desired70）。它的旧C10 recipe没有新longseg clock overlay，故补测旧部署不能替代full longseg候选的时钟资格。两个profile的ModelSpec/checkpoint/code/feature identity不同，比较器拒绝混配。

所有命令去掉`--run`即CPU预检；GR已在A100显式R5容量arm完成；FW full及4090 profile已在A100端到端通过（容量arm）；Cosmos原R8容量失败、R4两个冷启动已完成，出现缺依赖/源码identity/metadata不同必须记FAILED，不能修改qualified runtime掩盖错误。路径重定位只作用本次私有assets副本，仍要求原 ModelSpec 和 cache bundle identity完全相同。

## 3. 比较与冷启动重复

每命令包含两轮相同逻辑请求、不同10ms内调度抖动；这是同resident并发floor。还需第二次执行相同命令，得到独立冷启动目录。用两个同GPU目录评估冷启动增量，再比较三GPU；不要把不同客户端数、输入、PIN、K/C、recipe混在一个比较中。

```bash
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/concurrent_compare.py LEFT_COMPLETE_DIRECTORY RIGHT_COMPLETE_DIRECTORY --absolute-budget 0.000345707 --relative-budget 0.001 --output exp/offline_search/rounds/r17/xgpu_1017/evidence/dp_cross.json
```

独立冷启动完成后，在上面的比较命令追加`--left-repeat LEFT_SECOND_COMPLETE --right-repeat RIGHT_SECOND_COMPLETE`；比较器验证同侧GPU UUID、拓扑、harness与request identity，合并两次启动内的全部轮次计算floor，不接受重复目录充数。

这里两个目录需要本地可读；跨主机结果仍只通过已授权的窄范围公共rsync模块传入某一比较主机，不使用tether push/pull、SSH或expose。参数为DP wire action预算；FW的0.015625原值是normalized action，**不能直接用于wire比较**。其他模型wire暂定1e-3需要标为筛查预算。π命令启用batch成员observer；各模型该选项也记录load后flags与native cache距离（DP/FW已有原生成员ledger，GR/Cosmos为B1），三卡必须一致开启；该包装有Python开销，原无observer的失败/旧数据不能直接混合做floor。当前比较器始终输出`MEASURED_NOT_CERTIFIED`，缺分支、阈值margin、独立样本与SR时不能给等价证书。

完成4090并发结果后，发布**仅本研究已COMPLETE的结果文件**（read-only，保留原IP白名单）：

```bash
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/publish_results.py
cat /proc/pressure/io
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xr_tools/ /home/exouser/xgpu_1017/tools/
tether exec --timeout 10m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/compare_latest.py --host h100 --model dp --pull-reference
```

这是CPU比较，不占GPU显存；按模型替换最后的dp即可。它选择相同PIN和observer设置的最新native COMPLETE结果，并自动纳入同GPU同harness/拓扑的另一独立冷启动（如有）；不会误把numerics消融当native。缺任一卡结果即报缺测，不能用B1补位。A100也可拉同一4090参考后比较，无需反向上传或SSH。元数据/小action文件传输仍单流16MiB/s。

## 4. 闭环设计（待实施，不是已有结果）

**最新可排期小pilot：** 五模型均已用冻结formal planner生成每GPU每arm500个任务（MT50每task10官方init，LIBERO spatial每task50官方init），放在`closedloop_pilot_1018/<model>/{pure,cache}/PLAN.json`。固定相同plan/seed/init，真实worker数量分别112/40/48/12/44；FW开环32 clients不能代替闭环源12 workers。每GPU pure+cache合计1,000 episodes的服务器GPUh初排：π1–2；GR4090 1–2.6/H100 0.6–1.6；DP0.5–1.5；FW实际4090配置1.5–3；Cosmos4090 1.5–3（R8容量通过后）/H100 1–2。renderer另排，不能算在开环显存内。完整设计、依据、CI能力和启动接线缺口见[CLOSEDLOOP_PILOT_1018.md](CLOSEDLOOP_PILOT_1018.md)。原MT50 2,500/arm资格计划保留，以下为该原全量预算及方法。

用每模型原formal_eval及真正workers、同init/seed/plan，pure和一个预注册cache recipe分开。LIBERO首轮固定libero_spatial 500对（10任务×50官方init）；MT50每arm 2,500对（50任务×50init）。五模型pure/cache的固定任务表已生成于closedloop/（π为pi05_v2；DP/GR/FW长段使用dp_longseg_recal1018、groot_longseg_recal1018、fastwam_longseg_recal1018；旧计划留存），须在各GPU复用同一PLAN，不按GPU重生成。第二个LIBERO suite需另pin对应recipe后扩展。在每GPU重复同一任务清单，不能把失败/超时episode从配对表删除。成本pilot先LIBERO 20对、MT50 100对（每task 2 init）估计discordance和吞吐，但不能看到SR后改margin。

| 模型 / 初期suite | 每GPU、每arm episodes | 原worker数 | 每GPU、每arm预算 |
|---|---:|---:|---:|
| π0.5 MT50 | 2500 | 112 | 2–4 GPUh |
| GR00T MT50 | 2500 | 40 | 1–3 GPUh |
| DP spatial | 500 | 48 | 1–2.5 GPUh |
| FW spatial | 500 | 85 | 1.5–3 GPUh |
| Cosmos spatial | 500 | 44 | 2–4 GPUh |

表内预算以对应源拓扑作保守排期，非本轮闭环测量；pure+cache先乘2，不能再盲目对三GPU乘3。A100减R时，若吞吐近似随R缩放，πR3约2.7–5.3、GR R5约2.2–6.6、FW full R1约3–6、Cosmos R4约4–8 GPUh/arm；这些是等待cost pilot校准的估计，未计不同架构速度。4090 GR R6约1.8–5.5 GPUh/arm。FW实际4090 CPU-T5 profile是独立12-worker闭环计划（开环32客户端），另预留2–4 GPUh/arm并先跑20对；不能与H100 full profile直接合并。真实workers还需要独立renderer预算，不能把开环客户端显存当闭环预算。正式50init池之外扩容必须预注册额外seed，并按task/init聚类分析，不能编造新init。

±2pp双侧95%CI全落入margin才算SR等价；McNemar p>.05不是等价。若discordance=.10，500对通常CI半宽约2.8pp（正态近似），仍可能不确定；100对只够暴露历史6.4pp量级风险，不能认证。`paired_sr.py`提供配对discordant cells的保守精确95%区间和exact McNemar，0 discordant也有非零区间；paired_analysis.py另提供按model/suite/arm拆分、task-cluster bootstrap敏感性及first-divergence/right-censoring分析。500对是有意义的风险pilot，不保证±2pp认证；失败/缺测不能删除。

实际预注册区间的CPU功效模拟（20,000次、真差0、独立配对；不计task聚类）见`evidence/paired_power_design.json`：discordance=.10时，500对通过±2pp的概率约0%，2,500对约3.6%，5,000对约81.7%；discordance=.02时，500对约6.7%、2,500对约99.98%。因此本轮500/2,500对可给有意义CI，但不能承诺认证。5,000对只是样本量敏感性，不授权超出官方init池或复用相关episode冒充独立样本。

**闭环启动命令现已发布：**使用 [CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md) 的20个正常prepare/launch脚本。CPU已完成native planner、transport/trace接线和固定rendererplacement测试；实际remote prepare/launch待协调员。原冻结launcher传输审计保留为历史证据；本研究coordinator复用当前正常direct_transfer公共rsync，不调用旧tether_transfer。

闭环完成后的CPU分析入口如下，`PLAN_DIRECTORY`须使用本次预生成的同一plan，`DRIVER_DIRECTORY`须是该GPU正式run的driver目录。此处是待填的结果路径参数，不是闭环启动命令；不得拿本轮开环目录填入。`--trace`可重复指定新worker的query JSONL；对应的`*.executed.jsonl`单独用`--execution-trace`传入，不能混到query流。工具核对PIN、benchmark文件SHA、正式manifest完整任务集、PASS/all_done及accepted generation；少一个episode、重复accepted或runner error即拒绝，普通success=False正常保留。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/normalize_formal.py --bundle exp/offline_search/rounds/r17/xgpu_1017/concurrent/bundles/dp_longseg_recal1018 --plan PLAN_DIRECTORY/PAIRED_PLAN.json --attempts DRIVER_DIRECTORY/attempts.jsonl --driver-receipt DRIVER_DIRECTORY/DRIVER_RECEIPT.json --trace WORKER_TRACE.jsonl --execution-trace WORKER_EXECUTED_TRACE.jsonl --output NEW_NORMALIZED_DIRECTORY
CUDA_VISIBLE_DEVICES='' .venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/paired_analysis.py sr LEFT_NORMALIZED_DIRECTORY/outcomes.jsonl RIGHT_NORMALIZED_DIRECTORY/outcomes.jsonl --output NEW_SR_REPORT.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/paired_analysis.py divergence LEFT_NORMALIZED_DIRECTORY/traces.jsonl RIGHT_NORMALIZED_DIRECTORY/traces.jsonl --trace-clock env_step --action-atol 0.000345707 --output NEW_DIVERGENCE_REPORT.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/paired_analysis.py controls LEFT_NORMALIZED_DIRECTORY/executions.jsonl RIGHT_NORMALIZED_DIRECTORY/executions.jsonl --outcomes-left LEFT_NORMALIZED_DIRECTORY/outcomes.jsonl --outcomes-right RIGHT_NORMALIZED_DIRECTORY/outcomes.jsonl --action-atol 0.000345707 --output NEW_EXECUTED_DIVERGENCE_REPORT.json
```

上面action tolerance为DP wire预算，其他模型暂定0.001；不能跨模型混用。env_seed/policy_seed按冻结worker契约区分：Cosmos为环境0/策略base195，其余所选协议使用各自benchmark/task seed；实际每次denoise seed另保留。缺trace不伪造first-divergence；普通未记录字段不能证明一致。新worker hook在实际stock client和_send_ctrl层观测（冻结end_episode helper会绕过episode_end方法）；已通过128请求/16末次ACK的CPU协议集成。controls模式仅比较ACK确认实际选去执行的wire动作前缀（控制器/夹爪后处理之前），包含terminal partial tail；输出首次动作/夹爪符号分歧、右删失及描述性KM分布。须传两侧完整outcomes，使双侧都无trace的episode仍计unknown。query模式仍只定位已观测state/RGB；不是全模拟器隐状态。统计工具拒绝混用PLAN/PIN/runtime/protocol；缺身份时即使CI够窄也不签SR等价。

## 5. A100隔离顺序

同模型完成native与跨GPU比较后，逐项累积：TF32 matmul/cuDNN关 → highest → 共同attention backend → compile/CUDA graph关或固定 → deterministic（进程启动前CUBLAS_WORKSPACE_CONFIG=:4096:8）→ reduced-precision reduction关。π/Cosmos必须在loader主动设flag之后再次设置；GR直接FA2、Cosmos TE不能被SDPA math开关冒充统一。每个开关是单独新resident/新结果目录；先和前一项比较，再回退验证。`numerics_bootstrap.py`现已提供独立new-resident消融：`run_concurrent.py --numerics tf32_off`、再追加`--numerics highest`、`--numerics sdpa_math`（仅π/DP/FW）、`--numerics compile_off --numerics graphs_off`、`--numerics deterministic`。flags在load后重设并写NUMERICS.json，关闭graph时显式标为不同执行路径；FW实际4090 profile的TF32-off已在A100验证，仍有5/960个top1不稳定及0.029296875动作最大差；其余未测开关不能说成已恢复等价。GR/Cosmos的sdpa_math直接拒绝。Cosmos使用`--numerics cosmos_math`：禁用vendor FA3分支、将其局部sdpa_kernel selector固定为MATH，同时关TE Flash/Fused；首次实际进入写ATTENTION_OVERRIDE_VISITED.json。`te_unfused`仅切TE原生unfused族，仍含fused masked softmax，而且不影响minimal_a2a的SM90 cuDNN/SM80 Flash分派；两者都尚未GPU验证。GR仍需同build FA2的实际dispatch审计或另行资格化替代实现。旧B1 profiles不能冒充完成了E。

A100最新容量证据：πR4/B8/112在38GiB cap和原40GiB cap都OOM。下一次需显式R3容量arm并重做floor，不能把缩R叫原R4等价。GR/FW也会明确标记A100较小R；Cosmos原R8/76GiB allocator预算已实测：7个resident通过parity，第8个free-VRAM准入拒绝，采样峰值37,455MiB；不是精度失败。不可把预算降40GiB（每replica仅5GiB）引发的单进程加载OOM误称为物理R8容量证据。显式R4/38GiB两次冷启动已完成，四轮动作误差0、top1一致100%，2秒采样峰值23,956MiB；保留原R8失败收据。Cosmos另记录NUMERICS_AFTER_STAGE1.json，因为run_stage1会重设cuDNN deterministic/benchmark；加载后flag不能冒充最终推理flag。


CPU复核命令（不初始化CUDA）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s exp/offline_search/rounds/r17/xgpu_1017 -p 'test_*.py'
```

A100容量调整必须显式标新arm。π资源准备恢复后，候选R3命令如下；这是待验证的容量修复，不是已恢复等价：

```bash
tether exec --timeout 50m a100 -- /home/exouser/xgpu_1017/env/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host a100 --model pi05 --replicas 3 --memory-gib 38 --free-gib 36 --capacity-arm --batch-telemetry --minutes 45 --run
```

当前原πR4已留存OOM证据，不将失败结果用于floor。原DP/FW/GR longseg源是已封存待启动候选；其bootstrap的SHA被固定，底层模型runtime另有实际resident收据。4090 GR R6是新容量arm。旧4090 FW CPU-T5缓存拓扑曾为3faa6dcf…；新发现实际fa_fw_r_u_l resident收据为a7a387ed…，已追加独立fw4090_reduced profile。205509… H100 full迁移仍不能冒充它。

真实socket的CPU集成验证（输出目录必须新建，不启动GPU/worker）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/cpu_protocol_smoke.py --output exp/offline_search/rounds/r17/xgpu_1017/results/cpu_wire_repeat_02
```

已完成一次128请求的CPU协议检查；假策略仅验证transport，不能计入模型/缓存等价样本。

A100实际4090 Fast-WAM profile的原54GiB allocator budget被第二副本准入拒绝；原R2/B16保持不变、改51GiB budget已完成两次独立冷启动检查。复现该明确容量arm：

```bash
tether exec --timeout 40m a100 -- /home/exouser/xgpu_1017/env/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host a100 --model fastwam --deployment fw4090_reduced --memory-gib 51 --capacity-arm --batch-telemetry --run
```

TF32消融在同命令追加`--numerics tf32_off`，每次必须新结果目录；此profile native已是highest/matmul TF32=False，所以highest单独是无效变更，只剩cuDNN TF32关闭构成这个第一步的实际差异。
