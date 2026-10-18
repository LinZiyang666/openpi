2026-10-18 紧急修订：唯一全局准备开关 `ENABLED` 当前为 `off`，缺失或无法识别也按 OFF。所有新 prepare 恢复原生 worker 数量、原生 payload、原生回收策略；R15/R16 不部署 autoscale overlay。已有冻结运行目录不变。R17、legacy 各完成一条由 owner 执行的真实小 lane，且逐例前后回归无新增失败后，owner 才能启用；本轮不自动打开。

10:57 slot4 工具修订：闭合核验改读节点已有 process receipt 文件，只传路径和原 PID/start_ticks/pgid，引用按 8 KiB 分批，避免内嵌 argv 触发 E2BIG；不可读的非归属 `/proc` 字段跳过并计数，仍拒绝可证明属于 run 的 argv/cwd、存活 receipt identity 和进程组。timan107 支持普通 dmesg，但不支持原 `--time-format raw`；现优先普通 dmesg 并记录保留范围，窗口外 OOM 不归入旧 lane。本次日志已滚动到约 10:37，不能证明 09:06–09:29 验收窗口零 OOM。Coordinator 报告 10:54 原生 R15 39 + R16 64 workers 重叠引发 OOM、计数 37→45、MemAvailable=0；其 30 GiB guard 保留不动，全局仍 OFF。命令与证据见 `followup_1018/slot4/`。

09:40 OFF 故障的遗漏点已经定位到打包之外的 coordinator/按值注入：`filter_failed/timan1_gpu2.py` 的 authorize 无条件把 MT50 N0 改成 N32，bootstrap 又把当前文件直接嵌入远端命令。实际 fa_ls_gr_r_p_m 的 campaign 是 N0、timan1.json 却是 N32，worker payload 仍为原生身份逻辑，因此第 33 次 pull 的 renderer PID 变化触发原生 validator。现在 OFF 的 GPU2 route 恢复 N0，不增加时间回收；bootstrap、driver/worker hooks、capacity RPC 都采用同一 native source 选择规则；OFF placement 也恢复原生算法。真实 driver + worker_command + lifetime 回路在修复前重现相同 ValueError，修复后单代完成 40 个唯一 UID，且测试仍拒绝伪造的 PID 变化。证据、逐例回归及保留 939/补跑 1561 的命令见 `followup_1018/off_r17_fix/`。

验收补充：节点在第一代启动前和清理后分别记录 `/proc/vmstat` 的 `oom_kill`，以及可读且覆盖本进程的 user slice `memory.events`，记录 host、boot ID、时间及 cgroup inode。kernel journal 不可读时按覆盖整个 lane 的 counter delta 审计；缺启动基线、重启、计数倒退或时间覆盖不足保留 UNKNOWN，不补造历史。旧 R15 256 局试验事后读到 oom_kill=15、没有启动基线，故仍只有 zero_oom 未获证明。旧 FW validation 来源缺 runtime_identity 时，从原 archive 的成员哈希计算并核对新内容 pin，绝不换服务端源码；zero-progress reset 增加显式 H100 整目录改名备份，保留 mps_control，拒绝存活 owner，既不删文件也不杀进程。

动态候选的数量语义：旧手调 worker 数量不再隐式限制 runtime，选中节点的 desired maximum 至少取物理默认；原配置更大则保留。LIBERO timan1 GPU2 / timan107 / local 为 **83 / 104 / 83**，MetaWorld 为 **332 / 440 / 332**；来源为已有 renderer SIZES 与 10% 显存余量，local 准入另扣同卡 server 增长。只有带正整数 maximum 和明确技术原因的 `worker_hard_limits` 可以降低这些上限。正式 7 项中没有发现需要保留的小硬上限；原生固定 continuation `fa_co_f_f_oc1` 保留源 driver/control/worker pins，因此不切换模式。本次单独的 `fa_asv_*` 故障验收以 4/8 为显式边界，保证 256 集覆盖多代回收，不作为正式默认。timan108 始终排除；OFF 下其旧份额转给 timan107、总人数不变。

动态运行初始吞吐目标就是 desired maximum。节点每 2 秒执行内存准入，driver 每 180 秒按 accepted 吞吐与排队信号调整；floor=20 GiB、critical=10 GiB、恢复滞回 4 GiB、hold=60 秒。多个 campaign 共用同一节点锁账本，按可回收 RAM/VRAM 做等份渐进分配，未用份额可以借出；新需求出现后借用者在 episode 边界归还，吞吐目标仍分别计算。整数 worker 不足以整除时轮换余数，不承诺物理上无法同时驻留的份额。

两次实跑故障已各用真实 CPU 路径先复现再修复。R17 的新进程复用 slot ID，但 renderer PID 已改变：现在每代通信身份追加 UUID，同代重连保持身份，slot 账本身份不变；没有放宽 endpoint/model/runtime/GPU/probe 校验（src/openpi/conductor/worker.py:111）。Legacy 的完成计数原本从 0 开始；实际根因是 Observer 的 client factory 闭包被循环变量覆盖，调用到 episode_setup，连续四个零步错误后触发 N4。使用默认参数捕获原 factory（autoscale_1017/worker.py:71），真实 LiberoEpisodeRunner 测试完成两代、八个不同 UID。

接线统一在 defaults.apply、legacy_release 与打包 source 选择层，覆盖 R17 formal_eval、ff stage/resume/launch、formal_a_queue dispatch/build-levels、ffk launch、R16 conductor/runner、R15 sync/run_arm。OFF 打包逐项采用 SHA 验证的补丁前源码字节，避免只删 JSON 却留下 overlay。已 prepared 的动态运行必须确认未启动后重新 prepare；有 launch/receipt 的失败运行使用保留证据的 closed-zero reset。运行链脚本、已有运行 payload 与科学字段不改。`control.py` 中 owner 的 SO_REUSEADDR 修复同时保留于当前源码和 OFF 打包字节。

原 INTEGRATION.patch/PATCH_BASE.json 是 00:0x 已应用的历史记录，不再用于重复应用。当前结果以 VERIFICATION.json 和 followup_1018/off_final/BEFORE_AFTER.md 为准；实际传输事件见 followup_1018/incident/INCIDENT.md，禁止将其抹成无远端操作。紧急 OFF 与修复阶段没有远端调用、launch、reset 或进程终止。CPU 插件阻止真实远端命令、GPU 调用和 23100–23199 绑定；真实 rsync daemon 测试在两侧排除。

下面是动态候选设计，只有明确的 owner 验收入口可在全局 OFF 时为新 `fa_asv_*` payload 封装 ON；普通调用者不获得这一例外。

**前线接入点（补丁前文件:行）**

| 位置 | 设计依据与接入 |
|---|---|
| `exp/r17_integration/ops/formal_eval.py:353` | prepare_workers组织payload与每节点配置。node_workers保留该节点自己的最大值，不把250复制成每节点上限。autoscale_initial=count；CPU准备阶段不要求最大数量同时能驻留。 |
| `exp/r17_integration/ops/formal_eval.py:687`、`:714` | 新runtime准备边界，显式加入源码闭包。普通全新runtime使用bundle；冻结正式campaign走本目录formal_trial的新orchestration pin，保留服务端runtime。 |
| `exp/r17_integration/ops/formal_eval.py:1002`、`:1047`、`:1153` | 既有30秒合并status/lease与运行检查，附带控制和node快照，不增加tether轮询。零worker节点也能收配额。 |
| `exp/r17_integration/ops/formal_process.py:64`、`:120`、`:143` | 外层supervisor拥有fleet子进程；真正的worker采样在其直接所有者formal_workers，外层只转发。 |
| `exp/r17_integration/ops/formal_workers.py:130`、`:187`、`:209`、`:245` | 复用launch、所有权清理、退出分类；用持续的run_fleet替换一次性固定spawn/refill决策。清理后收集该次内核证据。 |
| `exp/r17_integration/ops/worker_memory.py:70`、`:129` | 原固定admitted=True；legacy兼容路径恢复真实内存判断。动态路径不使用固定RAM/worker。 |
| `exp/r17_integration/ops/worker_gpu.py:93`、`:172` | 动态空assignments不能回退为整个配置上限的GPU预留；只预留当前代际。 |
| `exp/r17_integration/ops/worker_lifetime.py:11`、`:30` | exit75和PID/start_ticks回收凭据；新增时间条件，仍在episode边界退出。 |
| `exp/r17_integration/conductor/worker_entry.py:113`、`:135`、`:165` | 首次attestation也会创建环境，必须与后续factory使用同一物理GPU创建锁。 |
| `exp/r17_integration/ops/local_workers.py:136`、`:165` | local路由和独立payload overlay；删除timan108和timan1人数40的静态限制。 |
| `exp/offline_search/rounds/r17/filter_failed/timan1_gpu2.py:30`、`:46`、`:180` | 复用GPU2 UUID/PCI/minor/namespace，允许较大desired maximum；实际VRAM准入仍执行。MT50从N0改为N32/1800秒。 |
| `exp/offline_search/rounds/r17/filter_failed/worker_resilience.py:14`、`:108`、`:143` | 删除timan108；动态prepare/launch只探配置中的节点，保留节点最大值，不把一次capacity探测固化成desired count。运行中按本地信号分配。 |
| `exp/offline_search/rounds/r17/filter_failed/worker_resilience.py:333`、`:375` | 保留整节点故障fencing，增加retired_workers边界shutdown。内存/吞吐退役名单通过pull元数据走同一路径。 |
| `src/openpi/conductor/worker.py:116`、`:138`、`:141`、`:188` | pull、结果、执行、循环边界增加observer，科学任务和action不变。 |
| `src/openpi/conductor/driver.py:323`、`:422` | 只在accepted-attempt fence及终局证据持久化后计吞吐；单worker下一次pull返回MSG_SHUTDOWN。 |
| `exp/r17_integration/qualification/driver.py:73` | 动态结束条件不再等待静态expected_workers，all_done后排空连接并发布done。 |
| `exp/r17_integration/ops/driver_release.py:21`、`:36` | 复用已有独立orchestration overlay机制，将新controller和core driver纳入新pin；不替换模型/服务端runtime。 |
| `src/openpi/conductor/agent.py:190`、`:219` | legacy start及supervise/refill接同一本地内存门；拒绝准入不消耗restart预算。 |
| `exp/offline_search/rounds/r16/conductor.py:529`、`:538`、`:587` | 取消数量≤64的硬限制，省略数量取硬件上限，显式配置仍有效。启动数由内存与吞吐决定；无需64个worker同驻一台主机。 |
| `exp/offline_search/closed_loop/ops/h100/run_arm.sh:10`、`:25`、`:79`；`fleet.py:5`；`control.py:70` | R15注入同一governor，默认timan107，timan108调用直接拒绝。新模块由既有源码打包器进入下一次部署，本次不部署。 |

正式队列入口的只读依据：`filter_failed_knob/formal_a_queue.py:197`定义OFF→levels；`filter_failed/stage.py:18`、`:50`、`:105`提供现有staging流程；`filter_failed_knob/launch.py:43`提供原启动入口。新 `formal_trial.py`调用这些入口的组成部分，默认策略已进入共享 stage。 旧Cosmos队列仍含timan108，入口在调用native materialize之前仅替换worker资源参数，避免先进入已禁用节点的topology解析；科学选择不变。各节点复用现有ff/ffk测得的renderer VRAM档位（LIBERO 530MiB、MetaWorld 133MiB），避免timan107沿用旧1GiB档位过早限制活动数；这不是RAM预留，RAM仍由观测p90决定。它允许尚未LAUNCH的正式定义；已PREPARED者必须显式 `--reprepare`，保留原模型/GPU/R/B/MPS、filtered选择、cache_plan_name、UID/init/seed；准备前后核对完整计划。新worker tar只替换release.worker_files明确列出的控制层文件，其他冻结字节逐项保留。

**信号与传输**

- 节点读取MemAvailable、memory PSI some/full；同时尊重更紧的cgroup v1/v2剩余额度。不可读则停止准入。fleet只采样登记的children及wrapper后代，校验PID/start_ticks，记录RSS、VmHWM与代际最大值。worker结果边界的PEAK.json补足最后一段峰值。
- legacy agent 对已拒绝的空槽复用2秒节点采样，GPU拒绝也等待下一次采样再试；500个待填槽不会触发500次全节点扫描。实际spawn前仍在节点锁内重新读取、准入并登记，不使用旧的允许结果直接启动。
- 每个语义campaign/benchmark/runtime键保留最近128个代际峰值。history节点本地存储；不读取大日志、不跨模型混用。worker_max_episodes、worker_max_seconds也属于键。
- Observer包装client调用，记录sim_render_s、server_wait_s、server_timing.infer_ms累积service_s。sim_render包括创建、渲染、预处理、本地开销与环境锁等待，不是假称纯仿真时间。action原样返回。
- driver读取真实accepted rate及episode_inflight；后者是已派出未完成episode的代理量，不称为H100内部请求队列。只有等待占比和相对service膨胀也满足条件才判断排队。缺少service时不编造unqueued cycle。
- telemetry附在pull/progress/result及既有30秒supervisor会话中。零worker节点靠status通道恢复；没有高频tether polling。控制带campaign、epoch、sequence、时间，90秒过期；UNKNOWN不释放旧配额。

**秒级内存保护**

P = 同campaign最近代际lifetime peak的nearest-rank p90 × 1.15。对每个存活worker，remaining growth为 `max(0, max(P_i, peak_i) - RSS_i)`。新增唯一RAM条件：

```text
MemAvailable - floor - sum(all_live_remaining_growth) >= P_new
```

本用户所有升级fleet共用节点本地flock账本。准入、Popen、登记pending growth持同一锁，避免同时消费同一空间。其他用户当前RSS已经包含在MemAvailable，不能重复扣；其未来增长未知，所以仍需floor及持续采样。死owner尚有存活children，或活owner超过3个采样周期未更新，均fail closed。

无历史时仅同节点同campaign一个校准worker，以当时headroom一半暂记未知增长，直到完整计划回收代际结束。失败代际仍保留峰值但不解除校准。校准不是吞吐目标从1爬坡；全局/节点目标从开始就是最大值，校准完成立即按内存公式起量。首次校准耗时单独计入启动区间，不能隐藏在收敛估计中。历史p90不是数学最坏上界；外部突发增长及必须等episode边界的退役，意味着零OOM仍需真实验收证据。

低于floor或memory full avg10≥10停止spawn，包括recycle补位；低于critical选择最新worker排空，预计恢复到floor+hysteresis。已在draining的RSS先计入预计释放，避免重复缩容。恢复要求持续≥24GiB达60秒，回落重新计时。每次加回仍逐一重算准入。

driver的可容纳人数使用 `active + floor(node_headroom/P)`；node_headroom已经扣过所有存活worker的剩余增长，不能再用RSS/P折算存活数，否则会在内存爬升时过度缩小容量。按余量权重分配，受各节点配置上限和全局上限共同约束；合并小于4的多节点残片，不制造local×2。所有新增通过原spawn路径，每节点每2秒尝试；NVML独立线程每10秒刷新，样本过期禁止创建。实际环境创建用物理GPU UUID锁，首次attestation与后续环境创建均覆盖，每GPU同时最多一个。

目标下调不等于容量已经释放：driver必须看到节点ACK及实际退出，才释放全局预留；失联节点继续占用旧额度。每个selected worker先完成并报告当前episode，下一次pull才shutdown；supervisor不补该槽。MT50每32局或30分钟在边界回收，LIBERO保留N4并加30分钟兜底；exit75不消耗失败预算。多线程fleet通过bootstrap子进程设置PDEATHSIG再exec，避免多线程父进程使用Python preexec_fn。

**分钟级吞吐与收敛**

默认180秒窗口、最少8个accepted样本、marginal gain阈值5%、等待比例65%、decrease=0.7、probe周期600秒。初始target=maximum，稀疏/未知窗口维持这个目标并交给内存门，不再按每600秒+1起量。固定模式也仅关闭吞吐调节，内存保护仍在。

排队判据为wait/(sim+wait)≥0.65、wait>1.5×service、有episode_inflight。计算 `useful=ceil(rate×(sim+service)×1.1)`，每窗向其收缩，目标为 `max(useful, floor(min(active,target)×0.7))`。排队缩容不额外等待人数稳定一窗，上探才需要稳定；窗口低样本、相邻yaml_id分布变化>0.2时不比较吞吐。窗口边界采用左开右闭，不重复计刚好落在边界的accepted样本。

当无继续缩容的排队证据后，600秒周期性加 `max(1,N//10)`，若增益不足5%回到前一人数；有效提速则继续增加。缩容期间刷新probe计时，避免过载尚未消退就反向上探。最大值始终是配置值。

默认开启前补充实机信号边界：episode in-flight 只是代理，固定网络延迟也可能使等待占比很高，因此每次排队缩容后比较 accepted rate；下降超过5%就恢复前一人数，并记住该次更好的吞吐，避免反复缩到1人。上探优先用测得的增益决定是否继续，不能被同一个排队代理立即抵消；同一人数的稳态吞吐下降超过20%或任务分布改变时，重新估计这个经验下界。没有可靠排队证据时，从初始maximum做仅依据accepted rate的下降试探；吞吐损失超过5%则回退，不能因没有server timing就永远停在最大人数。稳定窗口要求目标配额稳定一窗、实测活动数处于5%（至少1人）的稳定波动带，容忍N4/N32回收和一次2秒采样误差；显著爬升/退役仍重置计时。VRAM准入使实际人数低于目标时也能测得稳定容量，不强求先达到maximum；上探无法增加实际人数时结束该次探测。日志同时保留实际active、测量容量、回退前后速率和预先确定的probing状态；验收排除探测窗口，不按得到的速率挑窗口。

以平台44估算：原0.8每180秒回退，150需6窗18分钟、250需8窗24分钟；现0.7分别4窗12分钟、5窗15分钟。队列信号成立时不再附加一个稳定窗口，因此该估计在模拟的真实DriverController中也成立。目标到达后实际退出还要等episode边界；若≤180秒，则约15/18分钟完成活动数回退。未知长尾、冷校准、cohort切换另记，不能承诺所有实机条件下固定时限。44不出现在生产策略的默认目标中。

**科学不变量与CPU证明**

`exp/r17_integration/conductor/manifest.py:31`的episode_tasks不接受worker_id/count；`formal_driver.py:90`的语义campaign决定UID namespace。`examples/libero/worker_entry.py:47`使用orig_init_state_idx；`exp/metaworld/episode_runner.py:163`使用显式seed和原init。decision seed来自`benchmarks/libero_bridge.py:113`与`benchmarks/metaworld.py:21`的语义hash。运行控制不重写这些字段。

测试调用真实R17 planner/CompatibleEpisodeQueue、原init helper与两种decision_seed，对LIBERO、MT50按固定1、固定250、动态350→250→150→44→2→6→1分发并逆序完成，验证原manifest、UID、init、显式seed和前五个decision seeds完全一致；真实accepted fence拒绝重复结果。另直接导入已应用的共享代码，验证先报告后回收、单worker边界退役、legacy初始/补位准入、GPU环境创建互斥及CLI大maxima。

模拟含220GiB节点向floor起量、低于floor停止补位、hold后恢复；250初始服务端过载到平台，以及服务器随后提速。SIMULATED.json明确标为synthetic，不是实机收益。实际driver的五个180秒窗口由250回退至44，测试不把settled强行置真。正式prepare测试mock transport/install/staging，验证新worker pin、原server bytes与EPISODE_PLAN保持不变；未调用任何远端。

**审计与正式验收**

每次spawn、退出、退役请求、目标变更、内存原因切换都append到AUTOSCALE_DECISIONS.jsonl，字段为time/node/delta/reason/signals/counts。退役请求delta与实际退出delta不能累加当活动数；counts及最终CSV给出活动曲线。节点每10秒写样本，同时保存每2秒及每次准入采样的累计minimum_available，不遗漏两次落盘之间的低谷。

kernel_evidence仅在fleet清理结束时读取本地kernel journal，核对日志覆盖起点，必要时回退dmesg；OOM时间限定本次区间。空日志窗口也必须有覆盖启动时刻的保留证据；没有权限/覆盖不足记UNKNOWN，不伪装零OOM。证据随原worker快照回收。acceptance只读取已经回收的本地文件：所有稳定180秒窗口≥最佳固定速率95%，至少5窗；zero OOM、最低可用内存≥critical、accepted errors=0、每UID恰好一次、实际缩容/内存拒绝事件与曲线齐全。Cosmos基线86/min对应81.7/min门槛；DP基线必须来自既有同模型4090正式证据，未提供数字就不能宣告通过。

TRIAL.md现指定真实fa_co_r_f_l（150/100）与fa_ls_dp_r_u_m（200/150）；不再单开小lane或固定人数重跑。Maxima高并不自动证明触发了内存门；MW若先受GPU或吞吐限制，验收必须报告内存场景未exercise，不能降floor造压力。

**文件与验证**

policy/node/worker/driver/fleet/legacy/bootstrap负责控制；bundle/release负责明确源码闭包及新pin；formal_trial提供新正式prepare/launch入口；kernel_evidence/acceptance收集与核验本次证据。trial_commands.sh只打印命令。build_patch.py只读共享文件，写本目录的INTEGRATION.patch与PATCH_BASE.json。

历史补丁涉及19个共享文件，精确路径与SHA见PATCH_BASE.json。默认开启后的验证见VERIFICATION.json、TEST_RESULTS_DEFAULT_ON.xml和BEFORE_AFTER.json：换链后83项CPU测试全通过；8组既有套件逐例比较，无原来通过而现在失败的用例，也无超时。原报告卡住的cache_lane用例，备份侧7.965秒、当前侧8.090秒完成。两边保留相同的既有失败及受限测试记录，不能把此结论写成全仓测试通过。sol检查60项，33项fresh本机解释器、27项H100使用既有CPU回执并重新验证本地计划，明确标记PASS_LOCAL，不冒充远端复检。GPU2实机、EGL/NVML、真实吞吐和零OOM均尚未验证；本轮没有启动正式任务。
