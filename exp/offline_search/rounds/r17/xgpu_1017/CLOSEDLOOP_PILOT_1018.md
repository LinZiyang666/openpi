# 五模型小型配对formal试验排期（2026-10-18）

**20个逐campaign prepare/launch脚本及trace/公共rsync/固定renderer槽位接线已完成CPU核验；GPU/worker未启动。** 入口见 [CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md)。 使用冻结formal planner、真实正式worker及client/ACK协议；同一模型、suite、arm的同一PLAN在4090/H100复用。主pilot每GPU每arm **500 episodes**，pure/cache分开；两卡共每模型2,000次物理episode，形成每arm500对。五模型合计10,000次episode。A100另加每模型1,000次，须先明确容量arm；不默认把双卡预算乘三。

## 固定总体、worker和配对

| 模型 / 初期suite | 每GPU每arm任务总体 | 正式有效worker数 | cache arm |
|---|---|---:|---|
| π0.5 / MT50 | 50任务×init 0–9 =500；seed7 | 112 | 封存crg_k50 |
| GR00T / MT50 | 50任务×init 0–9 =500；seed7 | 40 | longseg_recal1018 chrg_k50 |
| DP / libero_spatial | 10任务×官方A池init 0–49 =500 | 48 | longseg_recal1018 n50/C10/chrg_koff |
| Fast-WAM / libero_spatial | 10任务×官方A池init 0–49 =500 | **12** | 实际4090 fw4090_reduced chrg_k50 |
| Cosmos / libero_spatial | 10任务×官方A池init 0–49 =500 | 44 | 封存crg_k50 |

FW开环压力用32 clients填满B16；正式源实际effective workers=12，闭环恢复12，不把源desired70或开环32当实际worker数。其pure为K10、cache为K1；Cosmos pure K5、cache K2，**仅作同arm跨GPU配对**，不据此计算cache相对pure的同K收益。其余细节均绑定各PLAN的PIN和benchmark。Cosmos环境seed0、策略base seed195分开保留；其余LIBERO seed来自封存benchmark，逐decision noise seed按原worker规则生成。

任务表位于 [closedloop_pilot_1018](closedloop_pilot_1018/)，每模型各有`pure/PLAN.json`、`cache/PLAN.json`、campaign和PREREGISTRATION。完整生成收据见 [prepare](evidence/closedloop_pilot_prepare_1018.json:1)。MT50 planner会因所选init集合改变task_uid/yaml_id及num_trials元数据；两卡必须复用同一新PLAN，不能拿旧2500计划的UID混配。初始化本身仍调用冻结`exp/metaworld/env.py:56`的`MT1(task, seed).train_tasks[idx]`（`:67`），不因计划集数重新抽样。CPU测试逐任务核对模型/协议/任务名/原始init/seed等科学字段等于原50-init计划的子集，并核对pure/cache配对集合一致。

机器可读排期与每个PLAN的SHA见 [SCHEDULE.json](closedloop_pilot_1018/SCHEDULE.json:1)。[85项CPU回归](evidence/tests_tf32off_pilot_all.txt:1)通过，包括这10份任务表的种子/init/完整性检查；未执行模型或simulator GPU任务。

保留原全量资格计划：MT50每arm2,500（50×50），LIBERO每arm500。新500-pair MT50是预注册小pilot，不静默替代原全量资格总体。若先做成本smoke，可用每task2 init：MT50 100、LIBERO20 episodes/arm；只测吞吐、worker接线和trace，不把这点样本当等价证书，也不把与主pilot重合的episode重复计数。

## GPU时间与资源预留

下表是**每台模型GPU、pure+cache合计1,000 episodes**的初排窗口，包含两次模型启动及余量；不是已经测到的耗时。服务器节点的renderer如果另占GPU，须另排资源，不能把它们计成免费。主pilot正式开始前，用成本smoke的实际episodes/min更新区间，阈值和episode总体不随SR结果改变。

| 模型 | 4090服务器GPUh | H100服务器GPUh | 服务器拓扑与前提 |
|---|---:|---:|---|
| π0.5 | 1–2 | 1–2 | R4/B8；开环已预留44GiB |
| GR00T | 1–2.6 | 0.6–1.6 | 4090 R6/42GiB为显式新容量arm；H100原R11/76GiB |
| DP | 0.5–1.5 | 0.5–1.5 | R4/B8；8GiB仅服务器开环请求，不含48个renderer |
| Fast-WAM实际4090配置 | 1.5–3 | 1.5–3 | R2/B16、12真实worker；42GiB服务器窗口，4090不放本地renderer |
| Cosmos | 1.5–3（条件） | 1–2 | 原R8/B1；4090须先通过47GiB开环准入，H100预留76GiB |

估计依据和边界：π/GR原2,500-per-arm预算按500/2500缩放再加启动余量；GR的4090按R11→R6容量保守放大。DP历史cache为100–115 episodes/min，K100 R2 replay约26 calls/s；500个pure任务按44 decision/episode估算约14分钟再加启动、worker与传输余量。FW历史85-worker的500 spatial cache组2.9–4.4分钟，对12-worker源按85/12作约21–31分钟量级参考，再留pure K10和启动余量；不同runtime不保证此缩放。Cosmos历史driver吞吐约50 episodes/min，但pure、较少worker和4090容量有不确定性。依据：`exp/r17_integration/ops/RUN_CONFIGS.md:1336`、`:1387`、`:1388`、`:1399`、`:1404`。这些预算按整个GPU窗口计时，未按SM利用率折算。worker无法按源数量到位时须标新拓扑并重新估计，不能沿用表值。

**renderer固定placement及预算已接线：** π使用timan1 GPU2×64+timan107×48；GR为20+20；DP/FW/Cosmos分别仅用timan107×48/12/44。timan1 GPU2沿用原生UUID/PCI隔离与MT50每worker 133MiB VRAM/512MiB RAM；timan107保留各模型VRAM请求，cache按原生fleet至少4GiB RAM/worker预检。具体总量、headroom及逐campaignGPU小时见 [闭环命令表](CLOSEDLOOP_COMMANDS_1018.md)。两条链共享研究renderer锁并在启动前检查实时资源；不足则不创建LAUNCH、不降worker、不借timan108。真实槽位尚未预留，不把固定placement写成当前容量已满足。

## 统计与首分歧

每model/suite/arm单独比较SR(4090)−SR(H100)，报告paired discordant cells、预注册保守精确95% CI、exact McNemar及task-cluster bootstrap敏感性；margin固定±2pp。McNemar不显著不等于等价，不能删掉失败episode；基础设施错误必须按同一episode/seed补齐accepted结果，未补齐则标缺测。

500对的区间有实际分辨率：若零discordance，保守95% CI为±0.873pp；若差值0、discordance=2%，约±2.262pp；若10%，约±4.584pp。因此500对是有意义的风险pilot，不能保证±2pp通过。100对即使零discordance也只有±4.287pp，20对为±19.676pp。见[精确区间例子](evidence/closedloop_pilot_CI_1018.json:1)。若继续MT50全量资格，固定最终2,500对总体后裁决，不反复查看CI直到碰巧通过；LIBERO官方500池不足以认证时应报不确定，不编造新init或把相关重复当独立样本。

在新worker安装已CPU验证的client_trace/terminal ACK hook；记录每个查询的观测state/RGB hash、action、cache决策、decision-step和实际ACK执行前缀，另外记录末次partial tail。分别定位first action/cache/state/RGB分歧，按实际env-step对齐，报告经验分布/右删失；缺trace计unknown。该hook记录的是控制器变换前的wire动作，不能冒充全模拟器隐状态或控制器变换后的真实动作已一致。

## 启动接线完成，等待协调员 prepare/launch

逐campaign命令、服务器GPU小时、renderer槽位和CPU证据见 [CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md)。复用正常队列stage主流程与launch_prepared；模型runtime/workers/封存bootstrap保持，autoscale只在本研究进程设OFF，worker数量固定。正式端口为服务器23180、backend23181起；driver在weilandserver为4090 23196、H100 23197，地址按正常open_tunnels生成，非开环127.0.0.1:29600。

当前coordinator工作树的公共direct_transfer承担准备和回收（16MiB/s、单payload、串行），不运行旧冻结launcher里的tether_transfer；五个冻结serving runtime照常运行。H100只新增研究assets、复用已staged的同SHA cache输入。native worker child和recycle均按值挂接client/ACK trace，随正常worker snapshot收回。没有修改运行中的worker或chain。

两卡每arm的实际 `EPISODE_PLAN.json` 完全配对（仅endpoint不同）。GR/DP cache因当前OFF planner保留longseg task_namespace，与早期预注册的UID/yaml_id有显式映射；init、seed、其余科学字段完全不变。保留原PLAN，分析使用新生成且两卡identity相同的 `PAIRED_PLAN.json`，不要将旧UID清单直接用于新driver结果。

本次为CPU就绪，不是已通过远程prepare或闭环结果。仍保留π/Cosmos源profile的长段overlay范围限制、GR4090容量arm、Cosmos4090 R8容量待验证及500对CI可能不够窄的限制。DP native闭环配置不被TF32-off诊断静默替换。
