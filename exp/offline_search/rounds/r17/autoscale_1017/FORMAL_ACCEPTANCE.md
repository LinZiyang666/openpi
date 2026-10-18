# 正式 campaign 私有动态验收（2026-10-18 12:2x）

目标改为已授权 fallback **fa_ls_gr_f_p_m**。13:09最终只读检查发现队列已提前以OFF启动fa_ls_fw_f_p_m；没有触碰该运行目录。GR00T仍未prepare/launch，已提供对应_dyn入口。FW先前新建的_dyn文件已在repo内另名.NOT_FOR_RUNNING_1018，不能用于当前运行；代码也在任何写入/prepare lock之前拒绝已启动目录。global ENABLED 始终 off。现有 prepare_/launch_ 文件和所有链不改；下面只新增 _dyn 文件，coordinator 审阅后自行替换入口。本轮未写 /data/mw_finetune/eval 下的任何目录，未 prepare、stage、传输或启动真实实验。

| Campaign | 正式计划 / 本次待执行 | desired maxima（不是人数上限） | Serving |
|---|---:|---|---|
| fa_ls_gr_f_p_m | 2,500 / 2,500 | timan1 GPU2=332；timan107=440；总772 | 原 H100 R11、同批处理/模型/steps |
| fa_pi_f_u_l | 10,000 / 10,000，5 cells ×4档×500 | timan107=104 | 原 H100 R3 B1、R15 |
| fa_pi_f_f_l | 12,000 / 10,298，6 cells ×4档×500 | timan107=104 | 原 H100 R3 B1、R15；原生 journal continuation |

FW的60/60、GR的20/20 execution_worker_nodes 都是下界，经 defaults.desired_maxima 升至332/440；R15 经 physical_maximum 升至104。没有 worker_hard_limits 被注入。真正的技术约束是 VRAM/RAM 逐次准入和已知保护设备；不会把4、8、60或其它固定小人数当活动人数上限。R17 N32、legacy N4 的代际回收属于 worker orchestration，不改变episode UID/init、rollout seed 或 decision seed。

## timan107 内存与 ramp

只读快照 MEMORY_READONLY.json：MemTotal约220.3GiB、MemAvailable约107.37GiB、64个非 ledger legacy worker；RSS合计104.37GiB，VmHWM合计180.05GiB，单worker最大HWM3.3651GiB；oom_kill=45。它们不能只按当前RSS计费。每个非 ledger LIBERO worker 的未来峰值至少按 **4GiB** 计（3.3651×1.15后上取整）；发现更高峰值则上调，保存node-local peaks-external-libero.json，PID回收不释放这个已观测包络。排除所有 ledger-aware worker及其子进程，避免双算conda wrapper或动态peer。

- timan107 reserve **96GiB**；包含60GiB coordinator阈值之外的36GiB缓冲，覆盖已知一分钟>27GiB下降量的量级。另扣所有已知 worker 的 pending growth，以及非 ledger legacy 的 projected_peak−RSS；不是把96GiB当作全部预算。
- 64个 legacy 在场时，4GiB×64=256GiB 包络已经超过该节点可用RAM；新动态fleet在该节点不准入，仍保留440的期望值，MetaWorld可在timan1继续。即便legacy刚换PID、RSS暂小，也不会误认为内存已释放。该技术准入同时重放了31→3GiB轨迹，未产生任何额外spawn。
- 整个节点跨动态fleet共享 **每10秒最多1个新spawn**；替换也必须重新准入，每个pending child按峰值预留。此速率不依赖服务器是否已加载，也不依赖其它60秒脚本的采样时刻。其它节点保持已有节拍。
- MemAvailable低于60GiB时，governor请求自己拥有的worker在episode边界退让。原 coordinator guard **兼容，任何动态campaign都不需跳过**：原DriverController本来就读取WORKER_NODE_STATE.json.retired；测试证明guard删timan107后，targets和demand_targets均为0且不复活。guard和R17原生驱动断连/fence路径保留。

R15仍必须等4090链的四批R16 GR00T LIBERO离开timan107；_dyn launch先执行只读legacy进程检查，忙/未知不启动。R17允许与legacy共存，但上述预算可能将timan107实际准入降为0。这是资源准入结果，不是desired maximum被悄悄改小。

## 过滤臂 continuation 的证据

fa_pi_f_f_l_l50 有三档已native DONE，各500集（ir45/ir40/ir32）。ir25在timan107的journal有202个唯一accepted、0 error，最近记录10:55；只读检查timan107该ir25与H100该cell均无相关进程幸存者。对其10,819,101字节per_step用单读者、8MiB/s上限检查，202条client_timing、0 exception。保留这1,702个结果；ir25先原生resume剩余298个UID，之后其余5个cell的20档共10,000集，总待跑10,298。

**不reset、不丢弃重跑。** _dyn prepare在l50只stage新的worker orchestration，不重新生成已有h100_sync/synced.json，不改变三条closed arm记录的mirror地址。停止的ir25 h100_launch.json会在coordinator真正prepare时另存按SHA命名的backup。native control.chain跳过DONE，并用原journal的UID过滤；测试证明202+298=500，交集为空。后续五个从未启动cell正常sync。零OOM只认证新的动态执行窗口，不能追认旧OFF历史。

## 精确命令（仅 coordinator 执行）

```bash
cd /home/weiland/projects/openpi
# 替换 H100 chain 授权GR fallback项入口；不用改 campaign definition / 重新build。
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_ls_gr_f_p_m_dyn.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_ls_gr_f_p_m_dyn.sh

# R16 GR00T 四批离开 timan107 后，H100空闲时执行；两个R15 unit串行。
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_pi_f_u_l_dyn.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_pi_f_u_l_dyn.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_pi_f_f_l_dyn.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_pi_f_f_l_dyn.sh

# 各unit结束后，audit只读结果，写入repo的formal_acceptance/<NAME>/。
for name in fa_ls_gr_f_p_m fa_pi_f_u_l fa_pi_f_f_l; do
  env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
    .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.formal_dynamic audit --name "$name"
done
```

所有验收要求保留：原UID各accepted一次、无缺失/重复/计划外UID、0 accepted error、长度允许时至少2代完整回收、无probe冲突、minimum MemAvailable记录且≥10GiB、新动态执行的kernel起点覆盖且零OOM、每次人数改变有reason。新AUTOSCALE_ACTIVE.csv列出每node的time、active/target/draining、delta/reason、minimum_available_gib、过去60秒accepted episodes/min及累计accepted。速率只在scheduler正式accepted之后记录，重试/被fence结果不计；legacy原已完成OFF段不伪造动态曲线。

## CPU证据与运行限制

PLAN_fa_ls_gr_f_p_m.json和PLAN_fa_pi_f_{u,f}_l.json逐字段比较OFF和动态TaskGraph，包括所有task UID、init、seed/extra、server endpoint与bundle：2,500 / 10,000 / 12,000全相等。真实GR00T fallback定义还在CPU临时目录走formal_trial.prepare、真实driver/runtime打包安装和archived planner；transport被禁止/替代，目标fixed runtime的SHA/identity不回退到pre-longseg父runtime，2500字段仍相等。R15 task builder本来就不接收worker counts；同server shard列表、同manifest/row/seed，39→104只改变agent spec数量。

前后逐例表、源码SHA、readonly快照见 followup_1018/formal_dynamic_122x/。新包装器只在其进程和新冻结payload内读取on，物理ENABLED仍off；不改现有wrapper，不改queued/running campaign，不改旧fit、gap、Q3、pure引用或serving。CPU-only测试插件禁止未mock的GPU、远端传输及production端口绑定。

最终回归：207个原用例全部保留通过，新增14个全部通过，共221 PASS、0回归。新六份wrapper全部bash -n通过，旧八份wrapper SHA保持；逐例表见 [BEFORE_AFTER.md](followup_1018/formal_dynamic_122x/BEFORE_AFTER.md)。
