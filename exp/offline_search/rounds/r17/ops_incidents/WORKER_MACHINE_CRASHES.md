# Worker 机器故障调查（2026-10-17）

调查范围：openpi worker 运行方式、历史事故、timan107 / timan1 的只读系统证据，以及本机 4090 的容量建议。现场采样时间为 **2026-10-17 21:47–21:54 CDT（UTC−05:00）**；历史 JSON 中的 UTC / epoch 均换算为 CDT。本文的“当前”只指该采样窗口。

**结论：当前跑法存在已经发生、可以归到我方 worker 的资源风险；但不能把每次整机失联都归因于 worker。** 最强的新证据是 timan107 本次启动的内核环形日志中有 **15 次全局 OOM kill**：10-17 00:04:59 一次，以及 19:00:08–21:29:34 十四次。全部被杀 PID 均在我方车队回执中找到，后十四次属于 `fa_co_r_u_l`。这些事故没有导致本次启动结束，不能算成十五次整机宕机。[C2][C5]

Owner 已取消 timan108 修复检查：**不再探测或查询它，不提供重启操作，不把它计入后续容量。本文只用已有记录分析其历史事故。** 更正之前已完成一次从 timan107 / timan1 发起的连通性检查；更正之后未继续，也不以该检查作本文历史归因依据。没有停止、重启或修改任何实验，没有读取凭据、token 或 auth 文件；唯一写入的交付物是本报告。

## 1. 证据口径与必须纠正的摘要

以下缩写均是文件路径前缀，引用的 `:行号` 指向本次读取时的位置；共享 handoff 仍可能由其他会话追加。

| 缩写 | 完整路径 |
|---|---|
| `E/` | `/data/mw_finetune/eval/` |
| `R/` | `/home/weiland/projects/openpi/` |
| `I/` | `R/exp/r17_integration/` |
| `F/` | `R/exp/offline_search/rounds/r17/filter_failed/` |
| `K/` | `R/exp/offline_search/rounds/r17/filter_failed_knob/` |
| `H` | `R/logs/session_handoff.md` |
| `D` | `/home/weiland/projects/dist_experiment_control/docs/devices.md` |
| `B` | `/home/weiland/idea-sprint/dynamical_system/BINDING.md` |

证据优先级：内核记录 + PID/start_ticks 匹配的车队回执；其次逐节点 accepted/attempt 记录、部署清单与启动配置；最后 handoff / 设备笔记。`campaign.json` 的请求拓扑可能是继承值，实际配置要看 `LAUNCH_WORKER_DEPLOYMENT.json`、节点 `*_launch.json` 和现场回执。`WORKER_NODE_STATE.json` 会被覆盖，不是完整的状态历史。

需要更正四点：

1. **10-14 timan107 最后交局不是 00:20:57。** 那是 `cspi_k2` 全局 journal 最后一次接受，来自 timan1。timan107 在该批次最后交局为 **00:20:40.830551**；同期 `csgr_h5` 的 timan107 最后交局为 **00:20:39.765764**。约 00:21 失联的判断仍成立。[E1]
2. **10-16 09:11 不是 timan108 最后正常工作时间。** DP 原段中 timan108 最后五条 accepted attempt 都是 **09:06:24 的 EGL framebuffer 错误**；09:07:27 已被 tether 报 STALE。09:11:35 的最后正常 journal 来自仍在线的 timan107。handoff 里的“1,059 局、0 报错”是终局口径，不能说明底层没有渲染错误。[E3]
3. **10-17 05:43 的“39 LIBERO + 2 MT50”不能作为确证并发量。** R15 的 39-worker 启动有记录；“2 MT50”只是此前的部署清单。该 MT50 批次的 timan108 / timan107 最后交局都停在 **10-16 22:33:13**，最终状态也把两者列为 retired。没有证据证明这两个 MT50 worker 到次日 05:43 仍存活、仍渲染。[E4]
4. **timan1 的七小时事故包含一次系统重启。** 本次 `last -x reboot shutdown runlevel` 直接显示 **10-16 14:35 boot**，不是仅凭 tether OFFLINE 推断。其原因仍未知；可读用户 journal 的首条是 17:11，不能把它当开机时间。[C1]

`TETHER_CALLS.jsonl` 记录发起调用，通常只有 operation/node/timestamp，**最后调用不等于最后成功响应或最后心跳**。例如 `cspi_k2` 对 timan107 的调用一直记到 10-14 00:30:31，已晚于其失联；不能据此说机器活到 00:30。[E1]

## 2. 已知事故逐项重建

表中 `RAM/VRAM` 为每 worker 的配置或预留（MiB），不是实测峰值，也不是强制内存上限。EGL 数表示分配的渲染 worker 槽位；除有现场计数外，不能保证每个时刻都有等量活动上下文。`N4` 表示每完成四局回收；`N0` 表示没有该周期回收。

| 事故与核实程度 | 当时运行、EGL 分布、预留及回收 | 磁盘证据 | 最后生命迹象 / 恢复证据 |
|---|---|---|---|
| **timan108，09-10 历史冻结**：设备笔记明确记载，未获得当时内核原始栈 | RoboCasa 约 **45 workers**，笔记称占可用渲染显存 **98.7%**；当时以三张 A5000 计，不能据此证明均匀 15/卡。每上下文约 1.5–1.6 GiB；RAM 预留、回收周期未知。笔记把换任务时并发创建/销毁 EGL 作为机制解释 | 同日设备盘点说根盘约剩 51 GB、`/srv/local` 约剩 9.9 T；不是故障瞬时采样 | 记录为 NVIDIA 内核模块卡死、D 状态、整机失联；精确最后时间未知。属于历史强警示，**不是已经取得驱动根因证明**。[E0] |
| **timan108，10-05 02:06–08:44**：历史记录称整机重启，无法重读本机日志 | 此次未找到能与 02:06 对齐的 openpi 启动/逐节点日志，worker 数、EGL/卡、RAM/VRAM、回收均 **未知**。不要把后来 R15 的 39-worker 配置倒填到这次。同期 I018 也使用该机作功能检查，资源共享有记录，但具体在跑进程未知 | 同期故障采样缺失。`B:20` 的本地盘 9.8 T / 根盘 54 G 是 **09-26 盘点**，不可充当 10-05 磁盘状态 | `H` 记录 02:06 掉线、08:44 恢复、09:3x verify-worker 通过；I018 handoff 也记 02:0x 掉线、06:2x 仍无法使用。无法建立其因果与我方车队的关系。[E0a] |
| **timan108，10-10 约 11:31 起停滞，持续离线到 10-14 恢复**：worker 内核阻塞有直接历史采样 | Cosmos 官方基线 **24** + DP **20**，共 **44 个都用物理 EGL GPU0**。Cosmos 24 个 PID 全部 `D / rwsem_down_write_slowpath`；DP 留下七个同样阻塞的子进程。旧 Cosmos campaign 没有 RAM/VRAM/N 字段；DP 当班 handoff 记 4 GiB RAM/worker。不能假定已用后来的 N4 或 6 GiB 默认。显存预留未可靠保存 | 故障时磁盘采样未找到；当时 **MemAvailable 约 139 GiB、memory PSI avg10=0**，不支持“当时全机 RAM 已耗尽” | Cosmos 最后 accepted attempt **11:30:53.501412**；服务端最后 serving 约 11:31:38 / 11:32:08。诊断期间 agent 转 OFFLINE。10-14 12:37 恢复重启由 handoff 记载。没有证据把四天划成每天各一次新崩溃。[E2] |
| **timan107，10-14 约 00:21 整机不可达，12:37 boot**：失联来自当时内网检查；boot 本次复核 | `cspi_k2` **24** + `csgr_h5` **40** = **64 MT50 workers，8 张卡各 8**（3+5）。两批均 **N0**。实际节点部署 **3584/1024**；上层 campaign 的 RAM 是 2048，不能混用。π0.5 从 16:30 出局至 00:20，约八小时 | 16:23 剩 **6.47 GiB**，23:07 剩 **6.44 GiB / 211.49 GiB**；`/scratch` 与 `/tmp` 同盘，约 97%。文件虽写最低 8 GiB，仍标 `admitted:true`。00:05 可用 RAM 191 GB 是当时 handoff 样本，不是宕机瞬间 | 两批本节点最后交局 **00:20:39–40**；00:21:13 tether STALE；当时 ping/ssh 不通。handoff 记录无 shutdown、脏 system journal；本次 `last -x` 确认 12:37 boot。**重启人“root”来自历史记录，本次 wtmp 不提供操作人归因。**[E1][C1] |
| **timan1，10-16 07:36 起失联，14:35 boot**：系统重启已核实 | `ff1015_cosmos_li_03` 在 timan1 **40 个 Cosmos LIBERO workers，全部 GPU3**（这是旧授权，不是现在的 GPU2）；**6144/1024，N4**。同批另有 t107×45、本机×60，不能把全批终局时间当 timan1 活着 | 07:20 根盘余 **886.22 GiB / 1865.88 GiB**，没有接近盘满证据 | 本节点最后 accepted attempt **07:36:26.258665**，该行是 `0x8cdd` 错误；全段本节点有四条 EGL 错误。07:36:39 tether STALE。其他机器继续至 08:30；timan1 14:35 reboot，间隔约 6 h 59 min。内核日志权限不足，**OOM / Xid / 维护操作均未定案**。[E3][C1–C3] |
| **timan108，10-16 09:11 重启**：仅历史记录核实，按 owner 要求不查询节点 | `ffk1015_dp_mt50_reduced_descent` 实际在 t107×40、t108×40；timan1 当时没有部署。t108 **40 个 MT50 workers 全指向 EGL 3**；重启前文档映射为物理 index0，仍应以该次硬件身份为准。**3584/1024，N0**，不是八卡均摊 | 09:02 `/srv/local` 余 **8614.11 GiB / 10728.57 GiB**；`/tmp` 所在根盘余 **52.60 GiB / 870.62 GiB**（可用约6%；其余包含已用及文件系统保留空间） | 09:06:24 五条 framebuffer 错误，09:07:27 STALE；handoff 09:11 采到 uptime 0，无正常关机记录，`/tmp` lease 消失。t107 的后续正常终局直到 09:11:35，证明“整批最后时间”不能归给 t108。[E3] |
| **timan108，10-17 05:43 起离线**：按 owner 定为失效节点，不做修复检查 | R15 `ffk1015_pi05_libero_full_formal_A_pure_l10`：**driver + 39 workers 同机**，H100 三副本，WPS=13。未找到该次实际节点渲染分配回执；仓库旧 `run_arm.sh` 默认四卡，名义均摊 9–10/卡，**不能当实际分卡证明**。没有 R17 `worker-max-episodes` 配置证据，RAM/VRAM 预留未保存。另两名 MT50 worker 仅为旧部署，见上面的纠正 | 最近可用样本是前一晚 23:45：`/srv/local` 余 **8150.03 GiB**，根盘余 **43.07 GiB / 870.62 GiB**；距事故约六小时，不能排除后续变化 | 05:23 handoff 记录 R15 **445/500**；其完整 journal 留在失效机器，未取得精确末条时间。05:43 OFFLINE、05:49 本地链 rc=1 见 handoff。MT50 最后来自 t108 的 accepted 早在 **10-16 22:33:13.908415**，不是本次生命迹象。[E4] |

**同在 10-14 12:37 被恢复，不证明 107 与 108 同时出故障或共用一个根因。** 108 的起始停滞早在 10-10；107 的新失联发生在 10-14。handoff 的“更像公共原因”只能保留为待核实假设。[E1][E2]

## 3. 另外找到的事故与容易混淆的中断

| 时间 / 节点 | 事实、当班负载与资源 | 生命迹象、影响与证据 |
|---|---|---|
| **10-10 10:20:15，timan107 全局 OOM** | DP 与 Fast-WAM 两组各 **24 workers** 重叠，旧版本没有跨车队 RAM 检查；PID **419359**（DP worker7）被内核杀。早期 campaign 未保存单 worker 预留/N，不能用后来配置补齐；此条缺逐卡分配与磁盘采样 | DP 已接受 167 局；旧 supervisor 随后结束另外 23 个 worker。这是进程/车队故障，不是已证整机重启。原始 OOM 行与回执保存在 qualification。[E5] |
| **10-10 11:18:43，timan107 全局 OOM** | Fast-WAM 官方基线 **24 workers**；worker18 PID **423719** 被杀，swap 耗尽。该车队 RSS 合计 **53.01 GiB**，剩余内存的分配者当时未确证。早期预留/N缺失；随后 C10 明确是 24 个都渲染 GPU0 | Fast-WAM 原段 391 个有效终局。随后 C10 的内存压力很大，但 **11:51 C10 退出是 deadline/cleanup，不是已经证明的第二次 OOM**；退出后 MemAvailable 55.89→216.27 GiB，其中约 105 GiB 释放不体现在 AnonPages 中。[E5][E6] |
| **10-10 20:4x，timan107 全局 OOM** | `fastwamc1456717485` **48** + `dpc3464789869` **32**，共 **80 LIBERO workers，N4**；RAM 各仅 **2048**，FW/DP VRAM 分别 **768/576**。均匀分散时约 **10/卡**；cache worker 实测 RSS 3.1–3.5 GiB，80×3.1 已超过整机 220 GiB。当天另有 `/scratch`、`/tmp` 零可用块的取证记录，未将其精确绑定到 OOM 秒 | FW 最后 accepted 20:46:17，20:47 handoff 报内核 OOM；DP 后续继续至 22:02。说明 **有 N4 仍可因并发总量过大 OOM**。[E5] |
| **10-17 00:04:59，timan107 全局 OOM（本次新查实）** | `ff1015_cosmos_li_04` 实际 **97 LIBERO workers**，8 卡按 index 轮转，**13/卡一张、12/卡七张**；**2048/530，N4**。节点 receipt 的 RAM 标签虽为4096，`worker_ram_reservation_mib` 为2048。内核杀 PID **1579408**，回执对应 worker54 / generation1 / exit−9。OOM 表中该 UID RSS 合计约 **216.54 GiB** | 本节点最后 accepted **00:04:52.383031**，随后其余节点继续收齐；约 43.72 GiB 根盘空间可用。远端 receipt 显示 PASS 且含一次 worker failure；全批最终本地 receipt 是 FAIL、all_owned_closed=true，handoff 曾写 rc=0。**这些“成功/失败”是不同层的状态，不能抹去 OOM。** 不把旧 MT50 部署的74人额外计入，此前它已停止从t107交局。[E7][C2][C5] |
| **10-17 19:00:08–21:29:34，timan107 十四次全局 OOM（本次新查实）** | `fa_co_r_u_l` Cosmos **39**（EGL 0–7：七卡5、一卡4；**2048/530，N4**）与 R16 `fa_pi_r_u_l` **64**（每卡8，无 R17 周期回收）重叠，合计名义 **103 workers：七卡13、一卡12**。所有14个被杀 PID 都属于 Cosmos 的代际 worker，**不代表只有 Cosmos 造成总压力**。节点 UID RSS 合计约 **209–215 GiB**，最后一次 swap=0 | 18:47 根盘余 **34.84 GiB**。R16 journal 18:36:51–21:32:29 正常收局；Cosmos 也持续交局，21:54 receipt 仍 RUNNING、14 failures、926 planned recycles。**终局零 error 不等于没有 OOM**；目前没有证据指向此次整机 reboot。[E8][C2][C5] |
| **10-17 18:19，timan107 / timan1 控制探测异常** | 当时 `fa_fw_r_u_l` 远端48个（t10736+t1 12），`fa_co_f_f_o` t10734+t1 44+local7：t107 **70**、t1 **56** 个 LIBERO 槽位。两批部署按 2048/530、N4 计；t107 分散8卡、t1 GPU2。handoff 记录 t107 stalled mount、t1 10秒探测超时 | 查询异常被当作 fleet FAIL，协调器退役了两台仍 ONLINE 的机器；导致 FW 批次失败、Cosmos只剩local。后续修复使用 UNKNOWN/持续失败窗口。**不是新的已证整机崩溃**；本次 dmesg 能证实10-16的NFS不响应，未找到10-17 18:19对应的内核NFS行。[E9][C2] |
| **其他短暂控制失联** | 10-10 建库 handoff 记录 t108两次、H100一次心跳受高频调用影响（约322→8次/分钟的修复）。10-13 22:12:34，`cspi_k2` 曾收到t107 STALE，之后仍持续交局两个多小时。10-16 22:33 H100约127秒OFFLINE后恢复 | 这些记录证明 OFFLINE/STALE 不能一概等同整机崩溃。H100短断期间/附近，MT50两台远端最后交局22:33，最终被记retired；此次不重新归因其具体退出机制。[E10][E4] |

本次覆盖了 `E/` 下 525 个条目，并按存在的 campaign/launch/journal 筛查时间交叠；补读不采用同一目录结构的 R15/R16 记录和历史 qualification。**不是全实验历史的完整故障计数**：早期缺配置、部分 journal 只留远端，轮转过的内核日志也不可读。

## 4. 在线机器的只读现场证据

### 4.1 启动、日志权限、磁盘与内存

| 项目 | timan107 | timan1 | 本机 weilandserver / 4090 |
|---|---|---|---|
| 采样 | 21:47–21:54 | 21:47–21:54 | 21:52:30 |
| uptime / boot | up 3天9小时；`last -x` **10-14 12:37**；kernel 5.4.0-216 | up 1天7小时；`last -x` **10-16 14:35**；kernel 5.15.0-198，前次5.15.0-191 | up 8天11小时44分 |
| `journalctl --list-boots` | 可见前次/本次 boot ID；用户可见前次最后10-13 16:48，本次首条10-14 13:57 | 用户可见前次最后10-15 06:33，本次首条10-16 17:11 | 本次未为本机采集历史journal |
| `journalctl -b -1 -k` / `-k` | “No entries”，同时提示无系统日志权限；**不是没有内核错误** | 同左 | 不作历史推断 |
| `journalctl -b -1` | 抽取可见用户日志，未见目标签名；范围受权限限制 | 同左 | — |
| `dmesg --ctime` | **可读**；15次global OOM；10-16 08:10:36–52 NFS不响应后恢复；未见本次boot的NVRM Xid / 指定filesystem error签名 | **Operation not permitted** | — |
| `/var/log/kern.log` | 存在，不可读 | 存在，不可读 | — |
| `free -g`，GiB | total220、used89、available129、swap已用0（21:47） | total503、used118、available380、swap已用0 | total251、used15、available236、swap已用0 |
| 根盘 `df -h -l` | **212G，84%，可用34G**；root inode11% | **1.9T，45%，可用992G**；root inode2% | **913G，24%，可用663G** |
| 其他本地盘 | `/srv/local` **7.3T，99%，可用92G**；inode6%。不能把它与根盘的 `/scratch` 混为一谈 | 主`/srv/local`19T、56%、余8.2T；另一独立ClueWeb盘100%、余120G，不能据此认定worker根盘满 | `/data`3.6T、65%、余1.3T |

命令与权限见[C1–C4]。`df` 使用 `-l` 避免扫描全部网络挂载；没有为复现故障主动访问疑似挂死的NFS目录。只读日志受限时没有尝试sudo。旧脏journal只能请管理员读，不能把用户journal的末条当整机最后事件。

### 4.2 NVIDIA 与我方进程

| 机器 | GPU身份 / NVIDIA健康信息 | 我方worker RSS与VRAM现场观察 |
|---|---|---|
| timan107 | 8×GTX1080 8192 MiB，driver **535.183.01**。`nvidia-smi -q -x` 的 retired pages/ECC/remapped rows为N/A；不能解释为零错误 | 21:54 按receipt的PID+start_ticks核对，`fa_co_r_u_l` **39**个活worker，RSS合计 **97,150.2 MiB≈94.87 GiB**，单个1940.4–3014.6 MiB，无D状态。21:47 `pmon`无活跃条目、卡占用4–5 MiB；21:49查询有极少worker条目，但未形成稳定全车队VRAM测量 |
| timan1 | GPU2=**PCI 0000:81:00.0，kernel minor1，UUID GPU-9bdcf7cc-8a05-a7f9-b1c3-4d53d0a229df**；4×A6000，driver535.183.01。GPU2 remapped rows corrected/uncorrected **0/0**、pending/failure No；retired pages/ECC N/A。GPU0有1条corrected remap，**它不在我方许可GPU2上** | 21:54 `fa_co_r_u_l` **44**个活worker，RSS合计 **115,394.7 MiB≈112.69 GiB**，单个1949.0–3149.2 MiB，无D状态；未记录worker failure。21:49 GPU2用了40 MiB、仅显示Xorg4 MiB，其余卡存在别人的CUDA负载 |
| 本机4090 | RTX4090 **49140 MiB**，PCI81:00.0；21:52查询used4 MiB / free48507 MiB | 2个`local_worker_entry`进程，RSS合计 **5467 MiB≈5.34 GiB**；该次NVML快照未列我方显存进程 |

**现场GPU显存低快照与存活worker RSS同时存在；本次没有连续NVML跟踪，不能据此把这些worker的真实渲染峰值写成0，也不能据此增加容量。** 预算使用历史工作负载峰值，并保留这个观测限制。RSS求和会重复计算共享页，也不涵盖驱动持有的宿主内存，不能替代整机MemAvailable/PSI。[C3–C5]

`nvidia-smi -q` 不是完整的历史Xid日志；timan107本次boot未匹配到Xid，**不能排除重启前的驱动故障**；timan1连内核日志都不可见，更不能报告“没有Xid/OOM”。

## 5. 原因排序：证实了什么，仍不能证明什么

| 排名 / 因素 | 证据及适用事故 | 结论强度 / 限制 |
|---|---|---|
| **1. 跨车队总RAM超配、预留低估，且失败补位掩盖资源事故** | 10-10多次内核OOM；10-17的97-worker车队及39+64混跑；今天15个被杀PID全部匹配我方回执。当前LIBERO按2GiB排程，现场单worker RSS已常超过2GiB，历史还到3.1–3.5GiB | **对列出的OOM事件已证实。** 可确认我方进程是主要RSS负载与被杀对象；不能由此倒推10-14整机失联必为OOM。OOM触发线程名可来自zabbix、nv_queue等，触发分配者并不等于主要耗内存者 |
| **2. EGL / NVIDIA驱动资源压力，尤其单卡集中与并发换环境** | t108 10-10：44个GPU0上下文、24个D状态、`rwsem_down_write_slowpath`、NVML查询异常、139GiB RAM可用；t1 10-16离线前四条0x8cdd；t108 10-16离线前五条0x8cdd；09-10设备笔记 | **导致渲染失败的证据强，导致整机冻结的机制尚未证明。** 没有对应Xid/内核锁持有者/电源日志。不能把“45”推广为所有卡的统一崩溃阈值 |
| **3. 生命周期过长 / 任务相关驱动宿主内存保留** | 10-10 C10：24个GPU0 worker，AnonPages平而MemAvailable快速下降，退出释放约160GiB；后续回收代际观察到内存恢复。MT50长期N0；R16原生WorkerSpec没有周期回收字段 | **LIBERO风险有配对测量支撑；精确驱动分配机制为推断。** 历史每局0.09–0.54GiB是节点消耗估计，不能当每个任务恒定泄漏率。MT50未证明有同样泄漏，仍应避免无限寿命 |
| **4. 存储余量不足与NFS依赖** | 107 10-14根盘97%、低于文件标注阈值仍admitted；10-10曾零可用块；107当前另一本地盘99%；10-16 NFS实际不响应；10-17控制查询遇stalled mount后误退役 | **存储/控制路径风险已证实，造成整机失联未证实。** 磁盘满本身不能解释所有ping失败，但也不能无日志就绝对排除其间接影响。108 `B`明确禁止NFS，旧R15脚本仍导出NFS HOME，存在运行路径设计冲突 |
| **5. 管理员维护、供电、硬件、网络或其他用户负载** | 107/108同日被恢复；t1重启时kernel版本改变；机器共用GPU；某些整机事故缺内核取证 | **未排除，未证实。** 新kernel只证明boot版本改变，不证明该次是计划维护；同一时刻恢复不证明同因；没有证据指控他人进程 |

风险的核心并非“worker都不能多开”，而是当前把显存能容纳的槽位当作可靠容量：`F/timan1_gpu2.py:25`用LIBERO **530 MiB/2048 MiB**、MT50 **133 MiB/512 MiB**；`F/worker_resilience.py:100`用瞬时free和10%余量得出 **83 / 332** 的timan1硬上限。**83、332是算术容量，不是完成持续压测的安全上限。**[E11]

此外：

- `I/ops/worker_disk.py:33`明确 `admitted = True`；原8GiB阈值目前只是记录值。`cspi_k2`的6.47GiB仍admitted与此吻合，不能把文件名“ADMISSION”当作保护生效。
- 预留不是cgroup限制。`I/qualification/HANDBACK_Q.md:1240`早已说明6144MiB也只是协作预留。当前节点receipt还会因cache逻辑写成4096，而独立`worker_ram_reservation_mib=2048`可被patch优先采用；需要看**实际算式和最终配置**，不能只看某个标签。[E11]
- R15/R16有自己的fleet锁和目录；`K/vla_shared.py:33`还支持按批次改锁。避免互相清理是有用的，但不会自动替整台机器留RAM。旧R16的64上限也不是与R17叠加后的总上限。[E8][E11]

## 6. 建议的临时运行上限与资源余量

**以下是基于现有测量提出的保守操作预算，不是已经证明永不崩溃的阈值，也不是本次已执行的配置变更。** 对整机崩溃原因未知的部分不声称因果。先用这个预算恢复可观察的运行方式，再通过受控的逐档测量决定是否增加。所有数量均是**全机/整卡全部批次之和**，不是每个server、每条lane各拿一份。

| 机器 | 建议初始上限：LIBERO | 建议初始上限：MT50 | RAM / VRAM / 磁盘余量 |
|---|---|---|---|
| **timan107，8×8GiB** | **总32，最多4/卡**；每worker按 **6GiB RAM + 1GiB VRAM**预算；N4。历史诊断的干净节点RAM预算支持32，GPU不是首要限制 | 未完成独立长跑包络验证前也先 **总32、4/卡**；每worker先按 **2GiB RAM + 256MiB VRAM**预算。不要恢复80/120等累加配置 | 分配后 **MemAvailable预算至少22GiB（约10%）**，同时计入其他用户、driver和未落地的worker峰值；每卡至少 **2GiB显存余量**。root `/scratch`/`/tmp`：准备后至少 **max(20GiB，容量15%，下一批峰值新增写入+10GiB)**；本机即约32GiB起。99%的`/srv/local`不应再增加我方暂存 |
| **timan1，仅GPU2，48GiB** | **总32/卡**，6GiB RAM + 1GiB VRAM/worker，N4；先不采用83/卡。32不是从旧GPU3事故推导的硬故障阈值 | **总32/卡**，2GiB RAM +256MiB VRAM/worker；避免332/卡这种仅按显存推出来的配置 | 整机保留 **至少50GiB MemAvailable预算（约10%）**；GPU2保留 **至少10GiB**，且额外扣除他人峰值占用。只识别GPU2 UUID/PCI，不用旧GPU3。root/worker目标盘准备后至少 **max(20GiB，15%，下一批峰值写入+10GiB)** |
| **本机4090，48GiB，常与server共卡** | 与server共卡先 **最多16**，6GiB RAM +1GiB VRAM/worker，N4；实际允许数必须再取显存预算较小者，可以是 **0** | 初始 **最多32**，2GiB RAM +256MiB VRAM/worker；同样扣除server全峰值 | 显存保留 **至少8GiB**，再预留server加载/推理实测峰值和其他任务。整机RAM预留 **至少25GiB**，另计server/分析任务。root与`/data`分别满足上述磁盘公式，不能用一个盘的空闲抵另一个盘 |

资源公式应在准备、启动、worker补位三个时点统一计算：

```text
N_new ≤ min(每卡上下文上限剩余,
            floor((RAM可用 − RAM底线 − 既有车队尚未实现的峰值增量) / 每worker RAM预算),
            floor((该GPU可用 − GPU底线 − server尚未实现的峰值增量
                   − 既有worker尚未实现的峰值增量) / 每worker VRAM预算))
```

混跑时按模型/benchmark分别加总预算；不能把“32 LIBERO + 32 MT50”理解成自动许可。上述RAM是额外峰值预算，不应把已在MemAvailable/NVML used中的驻留量再重复扣一次；也不能漏掉回收间隙暂时不驻留的槽位。

**回收建议：** LIBERO统一N4，包括迁入统一管理后的R15/R16；每卡错峰回收、一次只创建/销毁一个新环境，避免所有代际同步重建。MT50建议新批次采用 **每32局或30分钟（先到者），在完成当前episode后回收**；这是待验证的工程参数，**不是已测最佳周期**。`formal_workers.py:66`能路由到`worker_lifetime`，但MT50冻结包、local/timan1包装和R15/R16路径都必须确认实际生效，并验证种子/attempt/结果保持一致。不要只改一个上层flag就宣称完成。[E6][E11]

**减少单卡并发初始化：** 使用跨批次、按GPU UUID加锁的环境创建/销毁队列；不要假定RoboCasa的`RC365_ENV_BUILD_LOCK`已经覆盖LIBERO/MT50。启动先每卡1个，观察首批任务及一次回收，再逐步填至预算；这属于未来运行计划，本次没有执行任何GPU压测。[E0]

## 7. 协调者应改变的事项

1. **首要：统一R15/R16/R17的节点预算。** 把所有worker、driver、残留子进程和其他用户实际占用计入同一本账，按GPU UUID分配；legacy单机“岛”也必须占用真实节点额度。无法在一台机器满足64个时，减少并发或改为可分散的worker协议；不要绕开账本叠加64+39。
2. **不再把“零终局error / 仍在出局”作为机器健康判据。** 单独展示 `global_oom`、worker被signal9杀、非计划重启、planned recycles、MemAvailable最低值/下降速率、PSI和磁盘/inode。今天14次OOM全部被补位吸收，是最明确的例子。报告OOM后停止向该节点增加新worker，安排自然边界降低后续并发；本次不动现有实验。
3. **将预算与事实分开记录，保留状态历史。** 同时保存请求数量、实际数量、实际物理GPU映射、有效RAM预留、receipt标签、PID/start_ticks、boot_id、逐节点last accepted和最后成功控制响应。状态变化追加JSONL，不覆盖掉stage探测原因；不要将缺省/null EGL列表理解成已验全部卡。
4. **把回收纳入全部worker路径；补位仍要验余量。** N4可以缓解累积增长，不能修复97人起步超配；重启次数上限应在节点压力异常时停止继续补载。进程关闭要验证所有后代，旧receipt RUNNING不能覆盖PID已消失的事实。
5. **区分UNKNOWN、网络失联和机器故障。** 已有18:19修复应保持：短控制失败不立即退役机器。将数据面交局、控制面响应、boot_id变化分别观测；tether调用合并/退避，避免心跳被查询洪峰挤掉。不要把一次4秒NVML超时直接当容量0并把负载全部堆到107。
6. **磁盘检查必须能影响排期。** 目前阈值被显式取消，建议由owner采纳本报告预算后统一恢复资源约束；本次不替owner修改策略。预估暂存包、解包、回传副本、journal增长，计同一个底层filesystem；整理仅针对已关闭且已有可校验副本的本线产物，由单独授权执行。跨机移走才释放源盘，同盘改名/移动不释放容量。
7. **运行依赖迁到本地可靠盘，保留取证能力。** worker Python、HOME/cache、TMPDIR、lease、日志应避免易挂NFS；根盘小的节点不能把本地化等同全部放`/tmp`。管理员负责提供只读内核日志权限或自动导出必要诊断，持续保存跨boot日志；不要为诊断重载驱动或重启节点。

### 每次启动前的只读核对清单

| 检查 | 可执行的只读检查 / 判定 |
|---|---|
| 节点与boot | `tether exec --safe --timeout 30s NODE -- /bin/sh -c 'date -Is; uptime; cat /proc/sys/kernel/random/boot_id'`；与上次记录比较，变化后重新核验PID及设备映射 |
| RAM与压力 | `free -g`；读取`/proc/meminfo`、`/proc/pressure/memory`；结合PID/start_ticks的本线RSS/PSS及全机趋势。至少观察首批重任务、任务切换和一次完整回收周期；预算不够就降低下一批数量 |
| GPU及他人占用 | `nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.total,memory.used,memory.free --format=csv`、`nvidia-smi -q`、`nvidia-smi pmon -c 1 -s m`；timan1仅分配GPU2 UUID。检查driver/module一致性，不触碰他人进程 |
| 本地磁盘 | `df -h -l`、`df -i -l`；对已确认健康的实际run/stage路径再做有界`statvfs`，计算准备后的余量；不全盘du、不遍历NFS |
| 内核异常 | 权限允许时 `dmesg --ctime`、`journalctl -k`、`journalctl -b -1 -k`；查OOM/Xid/NFS/hung-task/EXT4或XFS/I/O错误。权限不足记UNKNOWN，不记PASS |
| 跨批次所有权 | 检查全部本线fleet receipt和legacy进程；记录未关闭后代、回收槽位及server未来峰值，不能只查当前run目录 |
| 回收与数据面 | 核对最终节点配置的N、RAM/VRAM值，确认回收receipt中的completed_episodes/maximum与新PID代际；每节点单独跟踪accepted与error attempt |

上表中的`NODE`只适用于 **timan107、timan1、weilandserver**；不包含已停用的timan108。建议的检查流程与限额尚未部署。

## 8. 证据索引与现场命令摘录

### 文件证据

| ID | 可复核文件:行号 / 内容 |
|---|---|
| **E0** | `D:53`–73：每上下文显存、09-10 45-worker事故、环境构建锁；`D:338`旧盘点；`B:20`、`:38`、`:56`：108本地盘/NFS/HOME限制，以及**4张**A5000对旧3张笔记的纠正 |
| **E0a** | `H:1973`附近“02:06–08:44”；`/home/weiland/idea-sprint/dynamical_system/ideas/I018-iterative-step-profile/handoff.md:43`、`:61`：共享负载背景和02:0x失联。没有同一时间点完整worker清单 |
| **E1** | `E/cspi_k2/LAUNCH_WORKER_DEPLOYMENT.json:220`、`:230`、`:233`；`campaign.json:173`；`driver/attempts.jsonl:54927`（107末条）、`:54976`（全局末条来自1）；`TETHER_CALLS.jsonl:1895`。`E/csgr_h5/LAUNCH_WORKER_DEPLOYMENT.json:216`、`:226`、`:229`；`campaign.json:180`；`driver/attempts.jsonl:22740`；`TETHER_BLIPS.jsonl:1`。两批`*timan107_DISK_ADMISSION.json:9`。`H`中“12:37 被root重启”“系统journal脏关闭”“00:21”段，原问题指定664/694–716，调查时约666–720，因追加会漂移 |
| **E2** | `I/handbacks/HANDBACK_COSMOS.md:470`–510；`E/formal_cosmos_libero_official2000_20261010_100533_5f558380/recovery/WORKER_BLOCK.json:1`、`DP_ISOLATION_AUDIT.json:46`；同批`driver/attempts.jsonl:1726`（EGL error）、`:1727`（最后accepted）；原始campaign没有新增资源字段，避免回填默认值 |
| **E3** | `E/ff1015_cosmos_li_03/LAUNCH_WORKER_DEPLOYMENT.json:280`–291；`driver/attempts.jsonl:245`；`TETHER_BLIPS.jsonl:1`；`timan1_DISK_ADMISSION.json:9`。`E/ffk1015_dp_mt50_reduced_descent/LAUNCH_WORKER_DEPLOYMENT.json:229`–235；`campaign.json:5191`；`driver/attempts.jsonl:1`–5；`TETHER_BLIPS.jsonl:1`；`timan108_DISK_ADMISSION.json:9`、`:22`；`H`“09:1x timan108重启”段（调查时约402） |
| **E4** | `K/formal_a_1016/queue/ffk1015_pi05_libero_full_formal_A_pure_l10/UNIT.json:1`（39、WPS13、单机拓扑）；`H`“05:2x旧R15流程首跑”“05:4x–05:5x掉线”（调查时约145–147）。`E/ffk1015_pi05_mt50_full_descent_04/LAUNCH_WORKER_DEPLOYMENT.json:345`–358；`WORKER_NODE_STATE.json:200`–204；`driver/attempts.jsonl:28116`、`:28118`；`timan108_DISK_ADMISSION.json:9`、`:22`。最近盘样本另见`E/ff1015_cosmos_li_04/timan108_DISK_ADMISSION.json:9`、`:22` |
| **E5** | `I/qualification/HANDBACK_Q.md:1109`–1118（10:20 OOM）；`:1221`–1247（11:18 OOM与C10区别）；`:1832`（零可用块）、`:1841`–1848（80-worker RAM超配）；对应`preflight/concurrent_fleets_20261010/DP_REMOTE_EVIDENCE.json`及`preflight/fastwam_baseline_recovery_20261010/{OOM_MEMORY,OOM_BACKTRACE}.json`。`E/fastwamc1456717485/campaign.json:1`、`E/dpc3464789869/campaign.json:1`记录数量/预留/N4 |
| **E6** | `I/REQUESTS/M_to_Q.md:18`–44、`:69`–94；`I/REQUESTS/D_to_M.md:38`–47；诊断明确“driver host allocations为leading inference”，提供N4/6GiB预算，不提供永不失败保证 |
| **E7** | `E/ff1015_cosmos_li_04/LAUNCH_WORKER_DEPLOYMENT.json:418`–422；`timan107_launch.json:291`–306；`WORKER_NODE_STATE.json:156`；`driver/attempts.jsonl:742`；`timan107_DISK_ADMISSION.json:9`；远端PID匹配见[C5] |
| **E8** | `E/fa_co_r_u_l/LAUNCH_WORKER_DEPLOYMENT.json:279`–285、`:415`–419；`timan107_launch.json:410`、`:423`；`WORKER_NODE_STATE.json:203`；`timan107_DISK_ADMISSION.json:9`。`K/formal_a_1016/queue/fa_pi_r_u_l/UNIT.json:10`；`E/fa_pi_r_u_l/launch.json:1`、`journal.jsonl:1`及末条；`R/exp/offline_search/rounds/r16/conductor.py:538`–550（64上限、i%8）；`R/src/openpi/conductor/agent.py:36`–86（原生WorkerSpec） |
| **E9** | `K/formal_a_1016/probe_1017/HANDBACK.md:3`–17（误退役及修复）；`H`“18:2x严重错误”段（调查时约261），包含t107 stalled mount / t1 timeout；`E/fa_co_f_f_o/LAUNCH_WORKER_DEPLOYMENT.json:1`、`E/fa_fw_r_u_l/LAUNCH_WORKER_DEPLOYMENT.json:1`、各节点`*_launch.json` |
| **E10** | `H`10-10建库“约每分钟322次，现在约8次”段（原约1532，最终核对约1542）；`E/cspi_k2/TETHER_BLIPS.jsonl:15`的22:12:34 STALE后仍有数小时accepted；`H:113`附近H100 22:33短断。仅把handoff的高频心跳解释列作历史诊断，不宣称本次重做了网络实验 |
| **E11** | `F/timan1_gpu2.py:19`–42、`:126`–136；`F/worker_resilience.py:66`–103、`:205`、`:238`、`:428`；`I/ops/worker_memory.py:8`–19；`I/ops/formal_workers.py:66`–68、`:110`–116；`I/ops/worker_disk.py:12`–39；`I/ops/formal_eval.py:1761`–1762；`I/ops/worker_lifetime.py:11`–26。`K/vla_shared.py:33`–51；旧R15 `R/exp/offline_search/closed_loop/ops/h100/run_arm.sh:15`–26、`:79`–89 |
| **E12** | 本机显存历史：`I/ops/RUN_CONFIGS.md:1234`–1238，π0.5 R5+24本机worker在48.4/49GB后回收建FBO失败；`:1359`–1362附近DP R4+60本机worker导致CUDA OOM。用于8GiB显存余量建议，不计作整机崩溃 |

### 现场命令证据（本次全部只读）

**C1，21:47。** 分别在timan107/timan1通过以下命令采样；stdout已按第4节摘录：

```sh
tether exec --safe --timeout 40s NODE -- /bin/sh -c \
  'date -Is; hostname; uptime; last -x -n 25; journalctl --list-boots --no-pager; free -g; df -h -l; df -i -l'
tether exec --safe --timeout 40s NODE -- last -x reboot shutdown runlevel
```

关键输出：

```text
timan107: reboot system boot 5.4.0-216-generi Wed Oct 14 12:37 still running
timan1:   reboot system boot 5.15.0-198-gener Fri Oct 16 14:35 still running
两机journalctl均提示：not seeing messages from other users and the system
```

**C2，21:47及21:52。** 通过tether exec执行`dmesg --ctime`，筛选OOM/NVRM Xid/NFS/hung-task/文件系统错误；在内存中解析OOM进程表，没有写远端文件。timan107可读；timan1返回`Operation not permitted`。timan107关键行：

```text
Oct 16 08:10:36 nfs: server cs-czhai-file-01.engrit.illinois.edu not responding, still trying
Oct 16 08:10:52 nfs: server cs-czhai-file-01.engrit.illinois.edu OK
Oct 17 00:04:59 Out of memory: Killed process 1579408 ... UID:1310580
Oct 17 19:00:08 Out of memory: Killed process 980372  ... UID:1310580
Oct 17 19:02:42 Out of memory: Killed process 980362  ... UID:1310580
Oct 17 19:03:08 Out of memory: Killed process 980363  ... UID:1310580
Oct 17 19:03:17 Out of memory: Killed process 980382  ... UID:1310580
Oct 17 19:23:44 Out of memory: Killed process 1096770 ... UID:1310580
Oct 17 20:02:10 Out of memory: Killed process 1257369 ... UID:1310580
Oct 17 21:05:54 Out of memory: Killed process 1522019 ... UID:1310580
Oct 17 21:06:23 Out of memory: Killed process 1532581 ... UID:1310580
Oct 17 21:28:34 Out of memory: Killed process 1660433 ... UID:1310580
Oct 17 21:28:41 Out of memory: Killed process 1656303 ... UID:1310580
Oct 17 21:28:46 Out of memory: Killed process 1656007 ... UID:1310580
Oct 17 21:29:01 Out of memory: Killed process 1653451 ... UID:1310580
Oct 17 21:29:17 Out of memory: Killed process 1662415 ... UID:1310580
Oct 17 21:29:34 Out of memory: Killed process 1659791 ... UID:1310580
最后一轮：global_oom；Free swap = 0kB；Total swap = 3997692kB
```

**C3，21:47–21:49。** 两节点执行`journalctl -b -1 -k --no-pager -n 100`、`journalctl -k --no-pager -n 100`、`journalctl -b -1 --no-pager -n 200`；先用`os.access('/var/log/kern.log', os.R_OK)`判断可读性，两机均false，未尝试提权。执行`nvidia-smi -q -x`、GPU属性query、`nvidia-smi pmon -c 1 -s m`；仅从XML提取设备、错误计数、显存及PID，不读取进程环境。

**C4，21:52:30。** 通过`tether exec --safe --timeout 35s weilandserver -- /usr/bin/python3 ...`执行`date -Is`、`uptime`、`free -g`、`df -h -l / /data /tmp`、`nvidia-smi -q -x`，并仅汇总本UID实验worker的`/proc/PID/status` RSS，结果见第4节。

**C5，21:52–21:54。** 通过tether exec读取如下我方运行回执，匹配[C2]被杀PID与记录的index/generation/exit_code；对当前进程额外匹配`/proc/PID/stat`的start_ticks，读取`/proc/PID/status`：

```text
timan107:/scratch/zixuans8/mw_r17/formal/runs/ff1015_cosmos_li_04_timan107/receipt.json
  workers=97, status=PASS, worker_failures=1, worker_recycles=139
  PID1579408 = worker54, generation1, exit_code=-9
timan107:/scratch/zixuans8/mw_r17/formal/runs/fa_co_r_u_l_timan107/receipt.json
  workers=39, status=RUNNING, worker_failures=14, worker_recycles=926
  14个worker_failure PID全部匹配本次内核OOM行
  当前39个PID出生时间匹配，RSS总和97150.2MiB
timan1:/scratch/zixuans8/mw_r17/formal/runs/fa_co_r_u_l_timan1/receipt.json
  workers=44, status=RUNNING, worker_failures=0, worker_recycles=1329
  当前44个PID出生时间匹配，RSS总和115394.7MiB
```

上述远端receipt也会继续变化，故保留本次关键字段于报告中。当前worker的N4并没有阻止另一个并发车队把整台机器推到OOM，这正是应统一节点预算、而不能只调单个车队参数的直接证据。

### 管理员仍需补齐的证据

对 **timan107的10-14异常结束**、**timan1的10-16异常结束**，请管理员读取对应boot的持久化`journalctl -k`、轮转`kern.log`、OOM/Xid/hung-task/NFS/EXT4-XFS/I/O记录，以及重启/维护/电源管理记录。107脏journal历史路径为：

```text
/var/log/journal/951a3aa1efa9411b9d4c8f176edf599e/system@00065dd062330cc6-3a77f009df7b4cb8.journal~
```

应先保存原日志，再决定维护措施。本报告没有请求或执行任何节点修复；按owner更正，timan108不再调查现场或安排恢复。
