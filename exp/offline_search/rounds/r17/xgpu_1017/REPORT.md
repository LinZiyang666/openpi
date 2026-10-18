# 跨 GPU 部署等价性调查（2026-10-17）

**目前不能认定五模型的跨GPU正式部署等价，也没有已验证恢复等价的开关组合。** 历史A100下降可靠（见§1）。本轮范围已升级到真实并发路径；B1仅作筛查。A100专用于本研究、允许任意时长，公共rsync已打通。最新状态和证据见下方§0。最初B1阶段命令保存在COMMANDS_B1_ARCHIVE.md，仅供审计。

工作目录为本文件所在目录，下文相对路径以仓库根目录为基准；`evidence/` 等链接相对本文件。`file:line` 指本次读取时的行号；运行中的仓库文件可能继续变化，冻结 runtime identity 和本目录证据优先。新的 A100 授权覆盖设备文档中“不能使用”的旧状态，但不把它加入正式实验设备。

**π0.5 H100 10:57启动失败已定位并完成CPU修复。** 四个HF下载metadata仅第三行时间戳不同，触发frozen adapter的完整checkpoint身份检查；输入、bundle不是本次根因。已在H100建立同SHA私有checkpoint目录，硬链接本地权重、复制封存小文件，身份恢复4008289b…；新增内容8,457 bytes，原checkpoint/runtime未改。CPU输入/cache/依赖/冻结代码检查和修正命令预检通过，两次失败MPS均关闭。下一H100窗口使用 `--checkpoint-view /data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944`，每次44GiB空闲、15–25分钟，独立两次冷启动。详见 [修复与一键临时传输](PI05_H100_STAGING_FIX_1018.md:1)。本研究未启动GPU。

**Fast-WAM四次正式并发及跨卡CPU配对已完成，未达到筛查标准。** 同PIN实际4090 CPU-T5 profile，R2/B16/32客户端；4090/H100同卡pure max-abs均0.0146484375、cache最大均0.029296875，超暂定wire预算1,860/2,304与1,976/2,304。跨卡最大0.029296875，**2,283/2,304超预算，145/2,304超逐请求5×floor（全cache）**；top1稳定一致928/960（96.6667%），32个不稳定均在近邻band内，其中1个每卡分别稳定而跨卡不同。hit/miss、corrector、knob跨卡一致，stall全inactive；away仍有0.029296875动作差。临时只读rsync本次自动避开活跃23199、使用23198，传完已关闭；23177未使用。见 [Fast-WAM分析](FASTWAM_4090_H100_1018.md:1) 和[跨卡收据](evidence/fw_4090_h100_summary_1018.json:1)。旧profile没有decision-step overlay、noise/stage tensor与闭环SR仍缺，不能认证完整部署；π→GR→Cosmos排期和20个闭环pilot不变。

**2026-10-18最新CPU交付：**五模型共20个闭环campaign已完成正常prepare/launch接线，每卡每arm500集、autoscale OFF、native workers固定112/40/48/12/44，仅timan1 GPU2与timan107。20/20原生planner检查通过；纯CPU测试覆盖冻结MPS路径、原生stage主体、worker/recycle trace和同SHA fit别名。实际remote prepare/launch与槽位预留尚未执行。逐campaign命令与GPUh见 [CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md)；计划/种子/namespace对应见各P0…PJ的EPISODE_PLAN、TASK_NAMESPACE_MAP、PAIRED_PLAN。

| 本次阻塞项 | 已完成 | 仍待运行 |
|---|---|---|
| formal transfer | 当前正常public direct_transfer，16MiB/s串行；H100私有同SHA读取别名 | 协调员运行各prepare，验证远程实际环境 |
| worker trace | native child及recycle按值注入真实socket/terminal ACK hook | 正式500集worker实测及snapshot回收 |
| renderer slots | π64+48，GR20+20，DP/FW/Cosmos 0+48/12/44；GPU2隔离、固定数、共享研究锁、实时准入 | lane边界实际容量，不借timan108/本地模型GPU |
| DP cache漂移 | 全51个失败请求×8记录CPU回放；corrector均0；距离权重重算误差0；动作重构残差≤2.38e-7 | stage1 feature/projected x/y/d2未记录，无法锁定GPU kernel或验证修复 |

DP当前最早观测差异为retrieval distance→连续权重，weighted blend/反归一化传播该差；离散donor一致不能证明动作一致。详见 [DP_CACHE_LOCALISATION_1018.md](DP_CACHE_LOCALISATION_1018.md) 与[408条证据](evidence/dp_cache_localisation_1018.json)。TF32-off仍未恢复等价。余下并发轮次按π→GR→Cosmos逐卡lane边界，各两冷启动，FW既定排期不变。本轮无GPU/model/worker/driver/chain启动、无sudo、无他人文件编辑。

**DP两卡TF32-off、各两次冷启动（截至12:34 UTC）：未恢复含cache的部署等价。** R4/B8/48客户端、MPS、原生graphs及longseg PIN保持；16份resident收据确认唯一torch flag变化为cuDNN TF32关。Pure跨卡max-abs降至0.0000036955，硬预算全部通过；cache跨卡仍为0.0006102324，51/1,152超硬预算，4090同卡cache亦有19个超预算。两组总计131/2,304超逐请求5×floor（pure94、cache37），不能因绝对误差改善而忽略。各卡/跨卡576个cache head的可观测离散决策全部一致，但active stall/knob、cache MISS和SR缺口不变。详见[完整分析](DP_BOTH_TF32OFF_1018.md:1)、[四次汇总](evidence/dp_both_tf32off_summary_20261018.json:1)。四次已完成，无需再补同配置GPU轮次；FW既定lane边界及其余大显存计划不变。

以下保留首轮及native历史分析；其中“待补TF32-off轮次”的状态已由上述四次结果取代。

**H100 DP TF32-off首轮（11:54–11:57 UTC）：稳定化候选，尚未恢复等价。** 四resident确认cuDNN TF32关，matmul原已关/highest；R4/B8、MPS、graph及cache文件身份保持。等两轮同卡floor：native冷启动1为0.0013344288、冷启动2为0.0013872385，TF32-off为**0.0001524687**（pure仅0.0000034571），TF32-off本次0/2,304超硬预算。相对4090 native，max-abs由0.0014218092降至**0.0008635521**，超预算由779降至253；但超逐请求5×floor由7增至82，仍未通过。缓存576个head可观测决策全一致，stall/knob覆盖缺口不变。只有一次ablation冷启动、两卡设置不同，本次另有3个背景context，不能据此认证跨卡恢复或计TF32纯性能成本。详见[完整分析](DP_TF32OFF_1018.md:1)及[摘要](evidence/dp_tf32off_summary_20261018.json:1)。

**后续交付：** π0.5/GR00T/Cosmos六条双卡并发命令均通过CPU预检，显存及分钟见[COMMANDS §0.1](COMMANDS.md)。五模型每GPU每arm500-episode配对pilot已生成封存任务表；MT50取每task10 init、LIBERO spatial取官方50 init，原MT50 2,500计划保留。worker数量112/40/48/12/44，FW不把开环32 clients当正式worker数；GPU小时、renderer预算、CI能力及尚未完成的闭环启动接线见[设计](CLOSEDLOOP_PILOT_1018.md:1)。FW已由协调员排在4090约10:00、H100约08:00的lane边界，本研究未改排期或启动GPU工作。

**DP 4090↔H100 新结果（10:59–11:16 UTC）：** 协调员完成每卡两次独立冷启动，均COMPLETE；R4/B8、48客户端、MPS、原生CUDA graphs、longseg新PIN及cache文件SHA全部匹配。每卡4轮、2,304个逻辑请求，跨卡逐请求比较全部16对。**本轮未达到预注册动作等价标准**，详见 [DP完整分析](DP_4090_H100.md:1)及[原始分析收据](evidence/dp_4090_h100_summary_20261018.json:1)。

| DP组 | 4090同卡floor max-abs | H100同卡floor max-abs | 跨卡max-abs | 跨卡超硬预算 / 超5×floor |
|---|---:|---:|---:|---:|
| pure | 0.0007492304 | 0.0004336834 | 0.0009070635 | 433/1,152；1/1,152 |
| cache n50/C10/knob-off | 0.0010988712 | 0.0013872385 | 0.0014218092 | 346/1,152；6/1,152 |

独立硬预算仍为max-abs≤0.000345707、relative-L2≤0.001，不因同卡漂移增大而放宽。跨卡共779/2,304超预算；7个超逐请求5×floor（这7个均在硬预算内）。同卡自身也有4090的163个、H100的110个请求超预算。cache 576个逻辑head的top1、donor顺序、corrector及可观测stall/knob状态全部一致；近邻边界band内48个head无翻转。stall全inactive、knob全off、cache没有MISS，不能称完整层(b)通过；层(c)仍未测。不是独立2,995-query认证，也没有由数值差推断SR下降。

GPU包版本及报告flag匹配（Torch2.7.1/CUDA12.6/cuDNN90501，highest、matmul TF32关、cuDNN TF32开）；driver/CPU/Python小版本与并行负载不同，不能单独归因架构。4090启动驻留4,374MiB，H100为5,916MiB（均非峰值）。下一模型的同配置Fast-WAM R2/B16检查每卡需42GiB空闲；H100不足时先安排8GiB DP TF32-off消融，完整FW留到满足资源条件的窗口，不缩R冒充部署证明。命令见 [COMMANDS](COMMANDS.md:5)。分析工具新增逐卡floor，执行harness未改；83项CPU回归通过。

**04:54（09:54 UTC）4090 启动失败修复：** 两次 DP 尝试都在 frozen `mps.paths()` 的长度检查处退出，尚未创建 MPS root、启动 daemon 或 resident。原研究目录为69字节，四字符私有名加 `/mps/<6位hash>/p` 后为87字节，超过85字节上限。仅修改本研究 [concurrent_server.py](concurrent_server.py:26)：原子保留两字符 ASCII 私有目录，逐次调用所选 frozen runtime 的真实 `paths()`，按 `os.fsencode()` 复核长度，并写 `MPS_PATHS.json`；目录存在或竞态碰撞即换名，绝不复用会话。frozen runtime、R/B、cache/clock/graph、实验排期均保持原样。

| MPS路径 CPU回归 | 4090 | H100 | A100 |
|---|---:|---:|---:|
| 五模型、全部9个封存配置，各调用实际冻结 `mps.py` | 85 bytes | 39 bytes | 39 bytes |

27项路径矩阵及frozen文件SHA见 [mps_path_CPU_matrix.json](evidence/mps_path_CPU_matrix.json:1)；[test_mps_paths.py](test_mps_paths.py:1)另覆盖旧87字节失败、85字节边界、Unicode字节、原子碰撞、已有目录不复用及过长路径在创建目录前拒绝。只读检查见 [audit](evidence/mps_failed_attempt_audit_20261018.json:1)：两次RUN的process_cleanup均为空、无MPS_READY；29600–29699无TCP监听、本研究无Unix监听或运行中的harness。系统原有MPS control/server PID2077576/2096610分别于02:55/02:58 UTC启动，早于两次失败，未操作。修复后的协调员命令及8GiB、8–15分钟预算仍见 [COMMANDS.md](COMMANDS.md:5)。完整CPU回归82项通过，收据见 [tests_mps_fix_all.txt](evidence/tests_mps_fix_all.txt:1)；GPU复跑待协调员。

最新增量（2026-10-18）：Fast-WAM 实际4090 profile在A100原R2/B16、allocator 51GiB下完成两次独立冷启动；合并四轮，960个有top1的逻辑head中9个不一致，**稳定一致率951/960=99.0625%**，动作最大差0.029296875。这里是“在所有重复中均一致”的保守逻辑请求比例，不是9,216条delivery的独立抽样概率。9次均在按距离自身漂移定义的边界band内；离开该band仍有0.029296875动作差。两次native ledger只有B1…B15，尽管配置上限B16且32客户端足以填两个batch；TF32-off检查才实际观察到每replica各2个B16，不能把native配置值写成已访问B16。新版neighbor_boundary已按相同donor ID重算：960个有近邻head中89个在band内，9次top1不稳定全在其中；871个在已观测候选band外，最大动作差仍0.029296875。另768个无候选支持的head记unknown。此结论只界定已观测top2候选，不约束未观测第三候选；旧sorted-gap分桶由[ID对齐结果](evidence/a100_fastwam_id_boundary_v2_summary.json:1)取代。见[冷启动floor](evidence/a100_fastwam_deployed_cold_floor.json:1)。关闭TF32后，960个head仍有5个top1翻转（99.4792%），最大动作差仍0.029296875，**该操作未恢复本轮同卡一致性**；native已是highest/matmul TF32=False，因此有效变化主要是cuDNN TF32。累积TF32-off + highest + SDPA math设置已完成，同R2/B16下仍4/960个top1不稳定（99.5833%）、动作最大差0.029296875；该组合也未恢复本轮同卡一致性。这里只确认配置及实际输出，尚无全kernel profiler证书，见[math汇总](evidence/a100_fastwam_math_summary.json:1)。见[第二冷启动及TF32-off](evidence/a100_fastwam_cold2_tf32off_summary.json:1)。R2两次native的2秒采样峰值38,122MiB（37.23GiB，下界），见[memory](evidence/a100_fastwam_deployed_memory_summary.json:1)；FW阶段研究目录峰值36,376,724,790 bytes。完成math后仅删除本研究FW权重12,041,638,315 bytes、vendor/T5 14,202,213,669 bytes、env 7,328,128,265 bytes；结果保留，随后串行准备Cosmos。

**新recipe边界（07:35 UTC）：**长段recalibration最终验证已PASS；DP/FW的Q3 stall路径及recipe identity改变，GR另改变periodic_gap。三者bootstrap仍为4e07fb3e…5e4f5e、runtime/R/B未变。已在本目录新增`dp/groot/fastwam_longseg_recal1018`三个bundle，并保留旧canonical文件；后续命令显式选择`--deployment longseg_recal1018`。旧A100并发floor属于旧recipe，不能用于新PIN的跨卡认证；当前正式部署资格仍未完成。参见[差异](evidence/longseg_recalibration_drift.json:1)、[新PIN](evidence/longseg_recal1018_pins.json:1)，外部原证据`exp/offline_search/rounds/r17/longseg_1017/RECALIBRATION_FINAL_VERIFICATION.json:1`。

## 0. 2026-10-18 正式并发路径状态

### 覆盖与当前结果

| 模型 / 被封存路径 | R / B / 客户端 | 当前证据 | 尚缺 |
|---|---|---|---|
| π0.5 MT50 K10，pimk1_fuld | 4 / 8 / 112 | 正式campaign/assets/runtime已pin；A100 R4/B8/112客户端两次OOM（38GiB cap和原40GiB cap） | 需先解决A100并发峰值容量；4090/H100待协调员；源C5无长段overlay |
| GR00T MT50 K4，fa_ls_gr_f_u_m | 11 / 1 / 40 | 旧recipe：**A100 R5/B1、40客户端完成5,760请求，三组同卡action max-abs=0**；记录实际live flags | 4090/H100、冷启动floor、闭环；A100 R5与源H100 R11不等同 |
| DP LIBERO K100，fa_ls_dp_f_u_o_l | 4 / 8 / 48 | 旧recipe：**A100并发native完成：2轮×2组×48客户端×24 delivery=4,608请求** | 4090/H100并发结果、冷启动floor、active knob、闭环 |
| Fast-WAM LIBERO K10，fa_ls_fw_f_u_o_l | 2 / 16 / 85 | 旧longseg recipe：**A100 R1/B16、85客户端两轮完成8,160请求；pure max-abs=.0146484，cache=.00354445** | 4090/H100、冷启动、闭环；实际4090 text-cache部署另有独立profile |
| Cosmos LIBERO reduced，fa_co_r_u_l | 8 / 1 / 44 | 正式bootstrap/5档recipe已pin；A100原R8第8副本容量拒绝；R4两个冷启动共12,672请求完成，合并四轮动作零漂移 | 4090/H100 floor及闭环；R4是显式容量调整（A100实际空闲仅39.49GiB）；不能把76GiB上限当实际需求 |

精确R/B/client/identity来源：`concurrent/bundles/dp/PIN.json:2`、`:28`；`pi05/PIN.json:2`、`:32`；`groot/PIN.json:2`、`:33`；`fastwam/PIN.json:2`、`:29`；`cosmos/PIN.json:2`、`:32`（以上四个简称均在`concurrent/bundles/`下）。客户端数优先采用封存campaign的`execution_workers`，而不是只看旧RUN_CONFIGS或`workers`；例如π的execution_workers=112，源workers=40。这些是选定checkpoint/domain/K的覆盖，不是五模型所有suite/K/recipe的认证。

**A100 DP实测同GPU并发floor：wire action max-abs = 0.001217007637，超过旧DP batching参照0.000345707。** 两轮共有2,304个可配对逻辑请求；按组重算：pure max-abs=0.0005838871（62/1152超abs预算）；cache max-abs=0.0012170076（50/1152超abs预算），见`evidence/a100_dp_concurrent_by_group.json:1`。缓存离散分支一致仍会有连续corrected action漂移。四个replica均形成真实B8，亦有B1…B7；这是实际动态batch组成变化，不是指定B8后偷偷跑B1。已暴露的source/decision seed/decision-step/执行步/NFE/C完全一致；576个有donor_rows的cache head donor行一致；corrector/stall字段在1,152条有值请求上一致。knob-off日志有字段不等于覆盖active periodic call，原比较器未展开的stall_status现已解析；连续threshold margin缺失仍记unknown；实测stall_call全部False，首条stall_status.state=inactive、reason=fewer_than_four_reference_episodes（只有3个参考episode）；active knob未开启。cache组每轮576个head+576个tail，corrector_applied=True有528条；没有cache→policy MISS覆盖，见`evidence/a100_dp_branch_coverage.json:1`。参见 [A100 DP汇总](evidence/a100_dp_concurrent_summary.json:1)。

这说明**同卡部署自身就存在并发数值漂移**；它不证明A100比其他卡差，也不允许把预算事后抬到0.001217自动宣布通过。层(a)同时报同卡并发floor、跨卡误差、独立硬预算；层(b)不能把missing字段或未触发的分支当一致。层(c)尚无本轮SR结果。

**A100 GR00T同卡并发结果：** R5/B1（原H100 R11/B1）、40客户端、两轮pure/k50/koff各960请求，共5,760请求；2,880逻辑配对中动作max-abs/relative-L2均为0，可观测hit/source/top1/corrector/stall/knob/时钟一致率100%。两个cache arm确实访问了MISS、HIT、tail、corrector、slow_confirmed stall及active knob；原始delivery计数包含tail重复的head诊断，不能按delivery夸大独立决策数。新比较器另报head-only计数。启动28,602MiB；B1是该模型正式每副本batch，而非单请求staged replay。见 [GR收据和live flags](evidence/a100_groot_concurrent_summary.json:1)。load后matmul highest/TF32=False、cuDNN TF32=True、deterministic=False；Torch2.5.1/CUDA12.4/FA2.7.4.post1。已有GR日志可计算stall signed margin：confirmed最小绝对值0.164706，ambiguous为0.2，均远大于同卡零漂移的1e−6 band；含tail重复，不是独立边界样本，见[margin审计](evidence/a100_groot_stall_margins.json:1)。整个研究目录GR峰值15,698,102,705 bytes；测试完成后仅清理其新建权重/env/vendor/model_cache。

**A100 Fast-WAM full同卡并发：** R1/B16、85客户端、8,160 delivery /4,080逻辑配对，498.5s完成。pure动作max-abs=0.0146484375、relative-L2=.00764349，1,233/2,040超过暂定wire预算；cache max-abs=.0035444498、relative-L2=.00215829，2/2,040超预算。top1/hit/corrector/stall/knob与时钟均100%一致；donor列表1/1,020个head不一致（99.90196%）。具体client10/seq0同一输入/seed2039741470，在native stage1 B16和B3之间，donor第7/8名21与0交换顺序，weights改变；此时corrector_applied=False，动作差=.00354445。见[原始两条请求](evidence/a100_fastwam_flip_native_records.json:1)及[汇总](evidence/a100_fastwam_concurrent_summary.json:1)。这已经展示并发数值差向缓存动作传播，尚无跨卡或SR因果结论。正式ledger形成378个B16（另有B1…B15）。load后highest/matmulTF32=False、cuDNN TF32=True，Torch2.7.1+cu128/cuDNN90701。启动24,436MiB，2秒采样物理峰值**32,273MiB**（31.52GiB，是采样下界），见[memory](evidence/a100_fastwam_memory_summary.json:1)。整个研究目录本阶段35,486,297,295 bytes，FW数据26,307,116,116、site7,325,410,848，没有重复weights。

**A100 π0.5容量失败（与精度结论分开）：** attempt04在R4/B8/112客户端、38GiB预算下第一轮三组8,064请求完成，第二轮单resident9.5GiB上限OOM；attempt06恢复原正式40GiB cap并记录batch成员，148个初期pure请求后发生物理OOM：申请212MiB、仅173.69MiB空闲。两次均关闭本研究resident/MPS；不能把部分请求当COMPLETE或同卡floor。40GB A100不能直接承接这次完整并发负载，尚未测试R3等容量调整的数值后果。见 [失败收据](evidence/a100_pi05_concurrent_summary.json:1)。

**π实际部署身份差异：** sealed ModelSpec的code/preprocess digest为`38a4e01…a69`，A100实际为`c166be3…b63`；后者与已有4090 `pimk1_fuld/SERVER_ATTESTATIONS.json`四个resident完全相同。使用[单独封存的实际部署reference](concurrent/references/pi05.json:1)严格绑定，不改原ModelSpec、不忽略checkpoint/norm/layout/sampler差异。5个installed Transformers replacement文件与源树SHA一致；仅依赖wheel RECORD会漏掉133-byte `siglip/check.py`，见 [patch审计](evidence/pi05_transformers_patches.json:1)。这使“运行时代码封存”和“完整Python环境封存”必须分别核对。

**实际部署/待启动配置必须区分：**本次读取时，DP/FW/GR的选定`fa_ls_*` longseg源目录没有LAUNCH/RECEIPT；它们是带时钟修复的已封存候选。底层runtime另有正式resident收据，π源已PASS，Cosmos源已有8副本PARTIAL收据。A100回放验证的是明确PIN的这组候选，不是这些源目录已有正式闭环结果；不能把候选overlay自动说成当前在线lane已使用。见 [源状态](evidence/source_deployment_status.json:1)。4090的GR R6是新容量arm；FW的旧4090 R2 CPU-T5缓存部署使用另一个runtime `3faa6dcf…`，本次迁移的H100 `205509…`并不含同一配置，必须单列而非混称已有4090部署。

**Fast-WAM实际4090部署补充：** 发现`/data/mw_finetune/eval/fa_fw_r_u_l/SERVER_ATTESTATIONS.json:1`已有R2 resident收据，campaign使用runtime a7a387ed…及fastwam_text_cache=True；不是只有旧3faa6dcf拓扑记录。已另pin `concurrent/bundles/fastwam_4090_reduced/PIN.json:1`，同名reference绑定实际ModelSpec；原R2/B16/54GiB预算，源execution_workers12、desired70，本检查固定32客户端填两个batch。full.yaml是K10/C5，两个screen cache为K1/C10、knob50与10，逐arm比较，不能混成同一K。该旧bootstrap尚无longseg clock修复，full longseg候选另测。它与205509… full profile的checkpoint/code/feature identity确实不同，因此当前两条部署不能只按“GPU不同”归因；详见[evidence/fw4090_deployed_profile.json](evidence/fw4090_deployed_profile.json:1)。新命令已替换4090的FW首选，A100已复用同一份weights/env补测此实际部署，没有重复大模型。原R2/54GiB预算在第二replica的free-VRAM准入处失败（不是OOM）；保持R2/B16、改51GiB总allocator预算后两次独立冷启动检查完成；数值结果见本文件最新增量。

**A100 Fast-WAM实际4090路径首个完整结果：** R2/B16/32客户端、51GiB allocator budget（原54GiB准入失败）、两轮共4,608请求。K10 pure max-abs=.00732421875；K1/C10 knob50 cache=.0259252191、knob10 cache=.0258337259。可观测head top1一致率 **956/960=99.5833%**（不满足99.9%目标），donor顺序870/960=90.625%；hit/corrector/stall/knob离散决策保持一致。新observer实际得到距离/ramp字段：36个head落在近邻边界band，4次top1翻转全在该区；924个head在已观测top2 gap band之外，但最大动作差仍可达.025925，说明只看top1/hit会漏掉连续action差。旧部署不提供segment decision/env-step字段，比较器记unknown，不当作时钟一致。见[首轮汇总](evidence/a100_fastwam_deployed_first_summary.json:1)。R2启动27,136MiB；4.47分钟完成。此为首轮结果；独立冷启动及TF32-off后续结果见文件开头最新增量，尚无恢复等价结论。

### 五行 harness 契约

1. CPU封存campaign、ModelSpec、load descriptor、YAML、bootstrap SHA；逐文件验证frozen runtime，缺identity/输入协议不匹配即拒绝。
2. 使用formal_eval相同server_env、每replica正式CLI、原sealed bootstrap、私有MPS和stock replica_proxy；保留R/B、DP graph self-check，不调用会改lane的launcher。
3. 真WebsocketClientPolicy并发驱动，episode reset/select_bundle/request/ACK/end均走真实协议；输入与logical request/seed跨GPU固定，调度抖动在两轮间变化。
4. 逐请求保存action chunk、原始hit metadata、donor rows/weights、可用score/distance/corrector/stall/knob、server_timing；保留batch ledger成员和resident attestation。
5. CPU comparator按逻辑请求配对，以同GPU并发floor比较跨GPU，并单列budget/缺失分支/邻近阈值；短序列与B1均不签发SR或部署等价证书。

实际复用依据：`exp/r17_integration/ops/formal_eval.py:604`（env）、`:1314`（replica CLI）、`:1398`（proxy/bundle），`models/fastwam/ops/mps.py:180`（private Session），`models/dp/formal_batch_server.py:36`（real batch server），`common/websocket_policy_server.py:178`（ACK client）。longseg封存overlay来自选定campaign的`assets/ffk_bootstrap.py`，SHA **4e07fb3e…5e4f5e**；不能只检runtime而漏掉bootstrap。`longseg_1017/runtime.py`按值封存执行步时钟修复；本研究未编辑该文件或sol的serving.py。

另有 [client_trace.py](client_trace.py:1) + worker_hook/sitecustomize.py，可在**新建**正式worker中记录原socket返回动作、观察到的state、相机hash、实际segment时钟与noise seed；默认不存图像，显式限制时才录小量真实输入。它不改变native调用或返回对象，CPU测试覆盖ACK时钟和可重读录制manifest；尚未部署到闭环worker，不是已有首次分歧结果。

后续`--run`先用[harness_snapshot.py](harness_snapshot.py:1)把13个执行依赖保存为内容寻址的只读源码快照，resident observer与socket client均从该快照启动；`EXECUTION_HARNESS.json`记录其identity。模型/runtime/cache PIN另行校验。CPU验证覆盖源码后续更新不影响旧快照、封存文件篡改被拒绝，并从真实快照完成Cosmos CPU计划，见[evidence](evidence/frozen_harness_CPU_plan.json:1)。本次A100 R4执行期间保持既有文件不变，两个冷启动结束且进程关闭后才更新远端工具。

工具：[concurrent_prepare.py](concurrent_prepare.py:1)、[concurrent_server.py](concurrent_server.py:1)、[concurrent_clients.py](concurrent_clients.py:1)、[concurrent_compare.py](concurrent_compare.py:1)、[run_concurrent.py](run_concurrent.py:1)、[paired_sr.py](paired_sr.py:1)、[numerics_bootstrap.py](numerics_bootstrap.py:1)。可选 [batch_telemetry.py](batch_telemetry.py:1) 包装原dispatch，仅收集成员，不改forward/队列/RNG；它增加Python开销，启用/关闭必须分成不同观测arm。π原始stage runtime只有size、无members；06目录已实测包装可记录成员。比较器另以原生BUNDLE_ACKS检查完整bundle×client×delivery总体，拒绝两轮共同漏掉请求/整组；longseg时钟按累计ACK与head计数核对，缺字段仍为unknown。测试 [test_concurrent.py](test_concurrent.py:1) 验证ACK、正式graph/boot命令、非有限数/identity失败、并发floor、zero-discordant非零CI；与旧测试一起最新结果见 [tests_concurrent_all.txt](evidence/tests_concurrent_all.txt:1)。五模型最新CPU命令预检均通过，见[evidence/five_model_CPU_plans_current.json](evidence/five_model_CPU_plans_current.json:1)。额外使用冻结真实WebSocket server/client在CPU假策略上完成8客户端×8请求×2轮=128请求，验证episode/select/infer/ACK/terminal及server关闭，CUDA未初始化；见[CPU协议收据](results/cpu_concurrent_wire_01/RECEIPT.json:1)，此结果不含模型或cache质量证据。GPU完整两轮已完成DP、GR、Fast-WAM和Cosmos R4（容量差异逐项说明）；π正式R4实际启动后OOM；Cosmos原R8第8个resident未通过容量准入，R4两次冷启动已完成。旧结果不能用于新版recipe的资格结论。

**新增cache独立文件收据：**[cache_files.py](cache_files.py:1)已对9个bundle实际使用的67个唯一fit/stall/library元数据文件（435,071,777 bytes）串行封存SHA256，并绑定原serving PIN；见[清单汇总](evidence/cache_file_seals.json:1)。新`--run`在GPU启动前及进程关闭后核对每个文件，不改YAML/runtime/推理；跨run比较拒绝文件identity不同或校验失败。旧A100结果没有此独立前后收据，仍按缺测处理，不能用当前文件哈希追认旧运行内容。这补充了native的配置/recipe identity检查，二者不是同一证据。

### 输入和观测边界

原MT50 qualification三帧注明synthetic=true（`/data/mw_finetune/qualification/observations/pi05_mw_v2/OBSERVATIONS.json:15`）；既有MT50 teacher campaign的capture_images=false，因此不伪称从它重建图像。本轮用 [capture_mt50_cpu.py](capture_mt50_cpu.py:1) 在CPU OSMesa/llvmpipe、真实MT50 assembly-v3/init0/envseed7、正式camera/settle约定下采3帧；identity `234a5635a9d0712b68f26964362ebd4d99a47f07eb035fbdf78fde2e240e77c1`，见 [INPUTS](inputs_mt50_real3/INPUTS.json:1)。OSMesa系统包只下载解包到本目录cpu_osmesa，没有sudo安装或GPU渲染。

当前并发客户端循环真实三帧，以正式replan步长发送ACK。这是**真实输入的协议压力序列**，不是动作真正执行后的物理轨迹；未覆盖真实任务分布、独立2,995查询、首次状态分歧分布。native路径不注入per-stage observer；per-stage张量仍由B1工具辅助定位。旧结果缺失最近邻margin；新cache_telemetry.py只提取native已选donor距离、top2 gap及corrector ramp ratio，不重复查询、不改动作/分支，CPU测试及A100 Fast-WAM实际路径验证通过；未触发的stall分支、未覆盖所有模型的active knob、模型原生noise张量的并发观测是明确缺口；source/seed相同不等于已证明每个noise张量相同。

### 判据与闭环排期修订

层(a)以同GPU**并发**两轮（并补独立冷启动）floor替换旧§3的B1 floor：跨GPU max-abs ≤ max(1e-6,5×floor)，relative L2同理；还须独立硬预算通过。旧FW 0.015625是normalized drift，不能不换单位用于wire action。先4090↔H100，收到结果后再判A100对照。

层(b)目标hit/top1≥99.9%，离阈值corrector/stall/knob一致；unknown/未访问分支不能算通过，缓存效果须包含action后果。两轮数据强相关，不能把4,608请求当4,608独立样本。

层(c)每suite/arm ±2pp、95%配对CI全落入margin，exact McNemar另报；初期LIBERO每suite500对、MT50每arm2500对，保留官方init池；LIBERO 20对/MT50 100对先估成本。500对在discordance .10时常仍不足认证，不能靠p>.05结束。`paired_sr.py`采用保守精确配对多项式cell区间（Bonferroni CP），区间可能比正态估算更宽；[paired_analysis.py](paired_analysis.py:1)已提供按model/suite/arm拆分、task-cluster bootstrap敏感性分析和first-divergence/right-censoring计算；需要真实动作/状态trace，不能从SR推断分歧步。[closedloop_prepare.py](closedloop_prepare.py:1)已调用五份frozen正式planner生成固定PLAN，见closedloop下各PREREGISTRATION；π/GR成功版本为pi05_v2/groot_v2。具体episodes/GPUh及**尚未安全实现的闭环启动项**见 [COMMANDS](COMMANDS.md:1)。当前未运行任何formal_eval闭环campaign。

**Cosmos容量计划修正：**4090命令现在保留源76GiB allocator上限、R8/B1，要求47GiB物理空闲；不能用44GiB总cap（每replica 5.5GiB）与已记录约8GiB的单副本加载峰值混淆（`RUN_CONFIGS.md:1393`）。预算上限可以大于物理容量，但总物理峰值仍必须实测，尚无4090 R8通过收据。

**Cosmos环境准备：**初始数据5,113,698,231 bytes、vendor 10,953,669 bytes、site 11,010,091,085 bytes及Python3.10均已单流传完；后补7.83GB原生registry启动依赖。06实际模型已加载，但正式parity缺少453,381-byte原始qualification输入而拒绝启动，未绕过自检；补齐后07重试。当前研究目录实测27,047,222,410 bytes、root剩余21,900,242,944 bytes，见[evidence](evidence/a100_cosmos_startup_06.json:1)。01/02在实际resident的CPU导入阶段因wandb/multistorageclient缺依赖退出，未进入推理，不能称作容量失败。补传原环境的窄依赖清单（加matplotlib后并集242,836,081 bytes，已有NumPy等重叠文件rsync跳过），未升级版本、无sudo。以正式CUDA_HOME/CUDNN_HOME搜索路径运行的完整loader CPU导入已PASS，CUDA未初始化，见[evidence](evidence/a100_cosmos_CPU_import_formal_env_02.txt:1)。03另遇到PolicyEvalConfig导入matplotlib缺失；补齐后正式配置CPU导入PASS（CUDA未初始化），见[evidence](evidence/a100_cosmos_official_config_CPU_import.txt:1)。04缺ftfy，05缺webdataset，均未进入有效推理。专用A100配置探针随后补齐decord，确认动态注册还会先解析3,913,017,214-byte基础checkpoint，再切换微调权重；它是原生启动依赖，不能用空文件或修改vendor绕过。该原文件已单流补传；同一registry另引用3,913,017,345-byte ALOHA checkpoint，虽本轮选LIBERO，也必须先解析。只复制这些原生启动依赖的唯一原文件，合计约7.83GB，测试后随本研究Cosmos目录清理，不建立通用HF cache。见[base依赖](evidence/cosmos_base_config_dependency.json:1)、[ALOHA依赖](evidence/cosmos_aloha_config_dependency.json:1)。探针可初始化A100 CUDA来满足原生FA2可用性guard，但不实例化model、不load权重、不建simulator；不是数值证据。当前extra依赖allowlist并集399,751,240 bytes，重叠文件rsync跳过。第四次动态配置探针已A100_CONFIG_PASS（未实例化模型），现在进入原R8/B1、76GiB allocator cap的并发测试；physical A100只有39.49GiB。Cosmos pure为K5/C16，cache为K2（1 NFE）/C16，必须分组分析。正式worker将环境seed固定0、策略noise base seed195，见`exp/r17_integration/benchmarks/libero_bridge.py:190`；闭环normalizer据此区分，不能把195误记成环境seed。

**Cosmos原R8容量结论：**07共有7个resident通过原生逐字节parity及ModelSpec核验，第8个在冻结`formal_server.py:123–126`的free-VRAM准入处拒绝；是容量准入失败，不是精度失败或实际allocator OOM。2秒采样峰值37,455MiB（36.58GiB，下界），本研究MPS/进程已关闭。见[容量收据](evidence/a100_cosmos_capacity_07.json:1)、[原始尾日志](evidence/a100_cosmos_capacity_07_tail.json:1)。随后按原每resident 9.5GiB上限完成R4/38GiB的两个冷启动；它不能替代原R8的部署结论。

**Cosmos R4首个完整冷启动：** R4/B1/44客户端、38GiB allocator cap，pure/K5、cache/K2 knob50与knob10共6,336请求、3,168逻辑配对；三组max-abs和relative-L2均0，2,112个有近邻的cache head top1及donor rows完全一致。两个cache arm均访问HIT/MISS、corrector及active knob；首轮knob50为528 HIT/528 MISS，knob10为924 HIT/132 MISS（各1,056请求）。stall均inactive，不能声称覆盖触发分支；原部署缺decision/env-step字段，记unknown；第二独立冷启动也已完成。合并四轮3,168逻辑请求，动作max-abs/relative-L2均0、2,112个有近邻head全部一致；2秒采样峰值23,956MiB（23.39GiB，下界）。见[四轮floor](evidence/a100_cosmos_cold_floor_summary.json:1)及[首轮汇总](evidence/a100_cosmos_r4_first_summary.json:1)。它证明这组输入下的同卡重复稳定，不能替代原R8容量资格、跨GPU或闭环SR。

### A100空间与操作审计（最新）

Cosmos两个冷启动完成后的研究目录峰值27,183,983,424 bytes。清理仅删除本研究的Cosmos权重3,954,990,242 bytes、vendor/原生registry依赖8,976,836,498 bytes、env11,187,280,859 bytes、model_cache1,699,688 bytes；结果保留，root恢复约43GB空闲。随后开始单流π0.5准备，见[串行handoff审计](evidence/a100_pi05_r3_handoff.log:1)。

首查root可用47GB。DP依赖沿公共read-only rsync、单流16MiB/s传输：rootfs913,530,556 bytes；Python82,820,828；site-packages6,191,950,155；长段cache39,200,762；原formal graph probe448,606；五模型小型封存描述22,669,314；DP输入1,188,138。DP并发完成后整个本研究目录`du -sb`为 **7,405,126,762 bytes**（约6.90GiB，含结果/失败启动审计）；没有复制teacher H5/RGB dataset。

唯一已执行sudo：`tether exec --timeout 30s a100 -- sudo install -d -o exouser -g exouser /data`。理由：canonical loader所需/data不存在；随后普通用户创建`/data/mw_finetune`到本研究唯一副本的symlink。详见 [操作审计](evidence/a100_operations.jsonl:1)。DP测试完成后只删除本研究创建的DP checkpoint与CLIP目录，见 [cleanup](evidence/a100_dp_cleanup.json:1)；共享openpi env/Python接着用于π，无重复副本。π数据7,620,274,956 bytes已传完，复用openpi env补入1,005,412,282 bytes依赖；首次loader缺chex失败，已补齐586,837,268 bytes实际import闭包及5个installed Transformers replacement文件（缺失check.py为133 bytes）。π阶段总目录峰值实测 **16,231,062,684 bytes**，06失败后只清理本研究创建的π checkpoint（7,473,099,921 bytes）、env（7,801,725,062 bytes）和tokenizer（4,264,023 bytes），root恢复46GB可用。GR下一模型env为独立新建，单模型、无重复权重。

DP第一次启动因正式graph probe未传输失败，已补那一个448KB文件；第二次发现客户端误把episode ACK声明混入RequestContext，已修复并加测试；另一次端口TIME_WAIT预检拒绝已修正为SO_REUSEADDR（仍拒绝现有listener）。失败目录保留。最终05目录COMPLETE，MPS daemon_closed=true、background_contexts_preserved=true，未触碰他人进程。4090/H100本轮GPU工作均未由本agent执行；4090 B1既有结果仅作screen。


## 1. 历史 A100：下降存在，原因未隔离

| 项目 | 可核实记录 |
|---|---|
| 日期 / 对照 | 2026-05-26；旧 A100（Ampere）对 jupyter 的 H200（Hopper），**不是 H100 或 4090** |
| 模型 / suite | π0.5，`pi05_libero`，LIBERO-Spatial 的 10 个任务 |
| 方法 | CP1 pure cache replay，`judge=always_hit`，weighted sum、top_k=1、step_filter=all；不是无缓存原生策略 |
| 特征 / 归一化 | 256-token SigLIP CP1；top10 中 8 个 spatial_pool_16、2 个 max_pool；vision_0 / vision_1 / robot_state 不同权重；逐特征 z-score 后 tanh |
| 样本 | 每配置每 GPU 100 episodes（每任务 10 个 held-out init），各 GPU 3 次；同一组 held-out init。表内没有逐 episode 配对结果或完整噪声 seed 记录 |
| 平均 SR（从逐次整数重算） | A100 **66.6333%**；H200 **73.0333%**；有符号差 **−6.4 pp**，平均绝对差 **7.0667 pp**；8/10 下降，2/10 上升 |
| 最差配置 | max_pool，v0@31/v1@25/rs@44：A100 55/55/54，H200 73/73/73；差 **−18.3333 pp** |
| 同 GPU 重复噪声 | A100 配置内极差均值 0.6 pp、最大 2 pp；H200 0.1 pp、最大 1 pp。这是三次重复的经验极差，不是 SR 差的置信区间 |

来源：`exp/weighted_sum/RESULTS.md:3`、`:74`、`:89`、`:99`、`:196`；原始逐次表 `exp/weighted_sum/data/libero_spatial/phase2/top10_variance.csv:1`，重算见 [history_recomputed.json](evidence/history_recomputed.json:2)。handoff 旁证为 `logs/archive/weighted_sum_libero10_replication.log.md:445`。旧“混用两个 server 的 87%”已被原报告撤回，不能作为参考 GPU baseline（`exp/weighted_sum/RESULTS.md:5`、`:190`）。

**证据边界：**原报告 `RESULTS.md:218` 将原因归为 bf16 累加、CP1 key 漂移和紧邻检索边界翻转。这是合理机制，但没有找到该次对照的逐阶段张量、attention/TF32/compile 单变量试验或逐请求最近邻记录，不能升级为因果证明。top10 在 H200 上选择，不能据此断言任意配置都会下降 6.4 pp；同 GPU 重复稳定也不能排除跨 GPU 选择偏差、不同 renderer 输入或环境版本差异。

`/home/weiland/projects/dist_experiment_control/docs/devices.md:175` 记载 **5 月 26 日当天**旧 `149.165.151.106` 迁移到旧 A100 `149.165.152.105`；`:180`、`:183` 记载 A100 40GB、`/root/openpi`。现存 SR 表没有 run→IP 绑定，不能可靠认定表中 A100 是当天哪台机器。`logs/archive/throughput_util_exploration.log.md:27`、`:61` 另有 `/root/openpi/.venv`、`pi05_libero_pytorch` 的旧节点吞吐运行，但不能当作这次 SR 的环境证明。

未找到GR00T、DP、Fast-WAM、Cosmos相应的旧A100配对SR下降记录。**没有找到可靠的 5 月 26 日逐 run Python/PyTorch/JAX/CUDA/cuDNN/driver、TF32、autocast、compile、attention 环境快照。** 设备页历史 H200 driver 570.211.01（`:126`）只是节点记录，当前 lock/当前包版本均不能补作历史 run 的事实。已检查 logs 的 handoff、相关 `*plan.log.md` / archive 日志、weighted_sum 原表、rounds 报告和 `RUN_CONFIGS`。当前 r17 也没有三种 GPU 的完整配对等价性证书。

## 2. 本次核查的运行时与数值设置

### 2.1 初期 LIBERO / B1 核查对象（当前并发 PIN 见§0）

| 对象 | 证据 / 使用范围 |
|---|---|
| 当前 DP cache `fa_dp_f_u_o` | runtime `5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e`，示例cache为K100/C10；实际 server Python 是 openpi `.venv`，**不是** ModelSpec 写的 DP nominal env；[campaign 摘要](evidence/fa_dp_f_u_o.json:2) |
| 当前 FW cache `fa_fw_f_u_o` | runtime `20550932745afaeb6bf31e858d1bfb91740727cf98bd0c6a5cd9171a8d9baa21`，示例cache K10/C10，Fast-WAM env；[摘要](evidence/fa_fw_f_u_o.json:2) |
| 当前 Cosmos cache `fa_co_r_u_l` | runtime `a7a387ed6d45ce9cf5ece2628f32577ee5a55ab55bbcb2a771e25ebc227b4f07`，示例cache K2（NFE1）/C16，Cosmos env；[摘要](evidence/fa_co_r_u_l.json:2) |
| hs5 DP / hs6 FW 旧控制 | 共享 runtime `1cf76a84…eab7f7`；源码 manifest 已验证。这个版本不认识新 cache 的 `miss_mode`，不能拿它配当前 cache YAML。保留的 CPU 失败记录不是当前版本故障 |
| π / GR | 已核查冻结共享源代码和正式 selector 定义，但本次没有定位到可直接复用的当前 LIBERO `fa_pi` / `fa_gr` sealed campaign.json。当前并发检查已另绑定π MT50的pimk1_fuld与GR MT50的fa_ls_gr_f_u_m，见§0；这里保留的是LIBERO查找边界，不代表遗漏这两个模型 |

冻结完整性证据：[current_cache_runtime_verification.json](evidence/current_cache_runtime_verification.json:1)、[runtime_verification.json](evidence/runtime_verification.json:1)；冻结静态摘录 [frozen_numerics.txt](evidence/frozen_numerics.txt:1)。本工具再次逐文件验证 runtime manifest、加载后的 ModelSpec、输入、sampler、fit/store/stall identity。旧B1工具调用冻结 adapter 的 staged API 和真正的 fitted recipe；**不包含在线动态批次、MPS、socket、并发顺序、DP 正式 CUDA graph 或缓存 warm-resume 服务调度**。这些仍须在所选部署拓扑上再资格验证；B1 通过不允许直接跨 GPU 混排生产请求。

### 2.2 按模型的静态设置

表中“继承”指该加载路径未明确设置，不能推断任一已有进程的最终值。回放会在加载后和回放后保存实际 flags。

| 模型 | dtype / autocast | matmul / TF32 | compile / 图 | attention | deterministic / benchmark |
|---|---|---|---|---|---|
| π0.5 | 默认 bf16；部分 norm / projector 等保持 fp32 | 构造器主动 `set_float32_matmul_precision('high')`；可覆盖进程启动时的 highest/TF32 策略，必须加载后重设 | `pytorch_compile_mode='max-autotune'` 包装 `sample_actions`；当前正式stage adapter直接调用run_stage1/2/3，不经过该sample_actions wrapper。compile_off在本路径首先是负对照；不能把旧legacy路径的compile原因套到当前部署 | language/expert 选 `sdpa`；flash / efficient / math 由设备、shape、dtype 派发 | torch 默认继承；未发现全局 deterministic 契约 |
| GR00T | checkpoint + inference bf16 autocast；LayerNorm 在 autocast 下 fp32，移除上下文会变为 bf16 | 默认继承；没有从源码确认每个 resident 最终 flag | `compile_vision=False`；可选 reduce-overhead；已有 compiled vision 漂移记录 | Eagle/Qwen 路径可用 FA2；checkpoint `use_flash_attention=true` 本身不是实际后端证据；具体类需运行时确认 | 默认继承 |
| DP | 模型 fp32，无推理 autocast；CLIP text CPU 路径；100-step DDPM、私有 RNG | 默认继承；fp32 GEMM/conv 对 TF32 尤其敏感 | staged eager；正式 server denoiser 可要求 CUDA graph，CUDA graph 不等于 torch.compile | Transformer/MHA，可能进入 SDPA / fused fast path，不能仅靠“没装 flash-attn”判定 math | 默认继承 |
| Fast-WAM | 主模型 bf16，混合 fp32 norm / 部分 autocast disabled；native action dtype 受配置约束 | 默认继承 | vendor `compile_action_infer=False` 默认；staged 直接调用 action denoiser | video/action 与 VAE 有 SDPA；派发依架构变化 | 默认继承 |
| Cosmos | 条件张量 bf16、VAE latent/动作输出 fp32；EDM/2ab 内部可用 float64，不能笼统写“全 bf16” | vendor loader **主动打开 matmul 和 cuDNN TF32** | 未找到这条 serving loader 的 torch.compile；不能替代实测 | Transformer Engine / minimal_a2a；vendor函数对SM90优先cuDNN（如FA3已安装则优先FA3），SM80/89优先Flash，并局部重设SDPA后端；全局math/TE开关不能统一它 | loader 从 config 设 cuDNN；`set_seed()` 又设 deterministic=True、benchmark=False；formal env `DETERMINISTIC=True` 还影响 VAE 随机性，但不等于全算子 deterministic |

vendor路径中的 `openpi_ext` 均以 `/home/weiland/projects/` 为前缀。源码锚点：π冻结af22613e…的`models/pi05/adapter.py:139`、`:183`、`:196`和`src/openpi/models_pytorch/pi0_pytorch.py:239`直接确认上述调用关系；其他源码锚点：π `src/openpi/models/pi0_config.py:20`、`:35`，`src/openpi/models_pytorch/pi0_pytorch.py:224`、`:238`，`src/openpi/models_pytorch/gemma_pytorch.py:62`，`exp/r17_integration/models/pi05/adapter.py:139`；GR `src/openpi/cache/groot/staged.py:49`、`:290`、`:334`、`:353`、`:444`，vendor `openpi_ext/third_party/gr00t_n15/gr00t/model/policy.py:190`、`:240`；DP `exp/mw_finetune/dp/policy.py:22`、`:51`，`exp/r17_integration/models/dp/staged.py:211`，`exp/r17_integration/ops/formal_server.py:49`、`:133`；FW `exp/r17_integration/models/fastwam/loader.py:207`、`exp/r17_integration/models/fastwam/staged.py:292`；Cosmos `exp/r17_integration/models/cosmos/staged.py:89`、`:117`、`:139`，vendor `openpi_ext/lines/cosmos-policy/cosmos_policy/_src/predict2/utils/model_loader.py:104`、`cosmos_policy/utils/utils.py:36`、`cosmos_policy/_src/predict2/networks/wan2pt1.py:300`。上表冻结摘录用于防止 mutable source 变化，未把它当所有 live flags 的证书。

**通用开关：**本地全新 CPU interpreter（未初始化 CUDA）观察到 matmul highest / allow_tf32=False，cuDNN allow_tf32=True，deterministic=False、benchmark=False、deterministic_algorithms=False，SDPA flash/efficient/math 都允许，bf16/fp16 reduced precision reduction=True。`CUBLAS_WORKSPACE_CONFIG`、`JAX_DEFAULT_MATMUL_PRECISION` 均未设；没有找到当前五个 PyTorch serving 路径主动设置 JAX matmul precision 的记录。JAX 只在 openpi env 安装，unset/None 不等于跨 GPU 的共同精度承诺。见 [local_inventory.json](evidence/local_inventory.json:25)、[local_cosmos_inventory.json](evidence/local_cosmos_inventory.json:25)。H100 核查仅包 metadata / driver，**没有读取运行中进程 flags**。

PyTorch 不承诺跨平台、版本或不同 batching 的逐 bit 一致；TF32、bf16/fp16 reduction、不同 SDPA kernel 都可改变数值。deterministic 控制相同配置下的算法选择，并非跨架构逐 bit 保证。参考 [PyTorch 2.7 numerical accuracy](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)、[reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html)、[SDPA](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)。JAX 若实际进入计算路径，应明确使用 `jax.default_matmul_precision('highest')` 并单独验证，而非从环境缺省值推断：[JAX API](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)。

### 2.3 环境版本（当前核查；非旧 A100 环境）

| serving island | torch / CUDA build | torchvision / Triton | transformers / diffusers | NumPy | cuDNN package / cuBLAS package | FA / TE |
|---|---|---|---|---|---|---|
| openpi（π；当前 DP cache） | 2.7.1 / 12.6 | 0.22.1 / 3.3.1 | 4.53.2 / 0.37.1 | 1.26.4 | 9.5.1.17 / 12.6.4.1 | 无 / 无 |
| GR | 2.5.1 / 12.4 | 0.20.1 / 3.1.0 | 4.51.3 / 0.30.2 | 1.26.4 | 9.1.0.70 / 12.4.5.8 | 2.7.4.post1 / 无 |
| DP nominal、FW | 2.7.1+cu128 / 12.8 | 0.22.1+cu128 / 3.3.1 | 5.5.4 / 0.39.0 | 2.2.6 | 9.7.1.26 / 12.8.3.14 | 无 / 无 |
| Cosmos | 2.7.0+cu128 / 12.8 | 0.22.0+cu128 / 3.3.0 | 4.57.1 / 0.36.0 | 2.2.6 | 9.7.1.26 / 12.8.3.14 | 2.7.3+cu128.torch27 / 2.2+cu128.torch27 |

以上对应 island 的包版本在 local/H100 一致，**不表示二进制 wheel、GPU kernel 和所有依赖逐字节一致**。Python patch 版本 local/H100 分别：openpi/GR 3.11.15/3.11.16，DP/FW 3.12.13/3.12.3，Cosmos 3.10.20/3.10.21。openpi JAX/JAXLIB 均 0.5.3。来源：[local_inventory.json](evidence/local_inventory.json:1)、[h100_inventory.json](evidence/h100_inventory.json:1)、[local_cosmos_inventory.json](evidence/local_cosmos_inventory.json:1)。DP nominal venv 只有 `.pth` 指向 FW site-packages，不能复制那个空壳来重现当前 DP cache 的 openpi env。

| 设备 | 架构 / 容量 | 当前 driver | 状态 |
|---|---|---|---|
| local RTX 4090 | Ada；**48GB 改装卡**，49,140 MiB，不能按标准 24GB 估算 | 595.91.07 | 仅 CPU 核查；GPU 由协调员排期 |
| h100 | Hopper，80GB | 595.99.02 | tether 只读核查；GPU 由协调员排期 |
| 新 a100 | Ampere，A100-SXM4-40GB，40,960 MiB | 595.99.02 | 仅本项等价性诊断获准；系统 Python 没有 torch / numpy / jax |

容量来源 `exp/r17_integration/ops/RUN_CONFIGS.md:26`；A100 证据 [a100_preflight.json](evidence/a100_preflight.json:1)。设备页原 driver 记录较旧，未用其替代当前采样。架构会影响 Tensor Core / cuBLAS / cuDNN / SDPA / FA2 / TE kernel 和 autotune；Python flag 本身不因架构自动不同，但**相同 flag 并不强制同一种 kernel**。driver、编译缓存、batch composition、renderer 输入也可能不同，均属待控制变量。缓存的PCA/距离/加权动作主要还包含NumPy CPU BLAS；当前H100主机为Intel Xeon Platinum8468，A100为AMD EPYC-Milan，见[CPU主机审计](evidence/current_cpu_hosts.json:1)。相同GPU flags不能控制CPU ISA/BLAS分派，应记录threadpool/NumPy build并做共同CPU scorer对照；本轮没有把缓存差异预先归因GPU kernel。

## 3. 三层预注册判据

在看到跨 GPU 结果前固定以下标准。先比较 4090↔H100，再比较 A100↔4090、A100↔H100；每个模型、suite、checkpoint、K、commit C、pure/cache recipe、batch 配置单独出结论，不把一个通过推广为全部通过。

### (a) 开环张量

输入 RGB/state/prompt、规范化/权重/tokenizer、request seed、**实际初始及每次 posterior noise**、mask、sampler grid/NFE 必须相同。相同 seed 不足以证明相同噪声；噪声张量不相同则本次算术对照无效，先改用预记录的共同噪声，另测 RNG 差异。B1工具观察 native RNG 并拒绝不相同，不会悄悄替换 native noise；并发工具目前只保存wire/decision seed，实际noise张量仍缺测。

主证据使用正式R/B/并发客户端的至少两轮重复，并在独立冷启动后再跑同一完整检查；当前A100 DP/GR只有同resident两轮，独立冷启动floor未完成。B1辅助工具每输入3次，记录 prefix/vision/state embeddings、stage2、每次 denoise 状态、最终整块 normalized 和 wire action、实际执行前 C 步、gripper 符号。并发工具已记录wire action与batch成员，但没有native每阶段tensor/noise；不能用B1替代这一缺口。报告 max-abs、p99-abs、相对 max（分母 max(abs(reference),1e−6)）和 relative L2；**相对 max 接近零会很大，展示但不单独裁决**。

同 GPU noise floor 为该张量所有 repeat pairs 的最大误差。跨 GPU 用所有 repeat cross-pairs 最坏值。`max_abs <= max(1e−6,5×floor_abs)` 且 `relative_L2 <= max(1e−6,5×floor_L2)` 才标“处于重复噪声内”；另设独立硬预算，不能靠一个很不稳定的 reference 放宽预算。

| 张量 | max-abs 硬预算 | relative L2 | 性质 / 已接受漂移参照 |
|---|---:|---:|---|
| noise、mask、整数结构、gripper 执行符号 | exact | exact | 必须一致 |
| 非 DP prefix / stage2 / capture 浮点特征 | 0.03125 | 0.002 | 暂定 bf16 筛查上限；不是历史 SR 认可值 |
| DP wire / committed wire | 0.000345707 | 0.001 | RUN_CONFIGS 352 请求 batching 最大 committed action drift 的参照；原记录不是 SR 等价证明 |
| Fast-WAM normalized final / denoise | 0.015625 | 0.01 | 仅 K10/C5、3 帧 H100 batching 的已有范围；其他 K 必须重新校准 |
| Fast-WAM 前 C 步 normalized | 0.0078125 | 0.01 | 同一已有 K10/C5 测量 |
| 其他浮点张量 / 动作 | 0.001 | 0.001 | 保守暂定筛查值，须以 layer(c) 验证，不是普遍物理动作尺度 |

参照 `exp/r17_integration/ops/RUN_CONFIGS.md:288`、`:375`、`:436`。GR 历史 B2 max≈0.0029 超过 1e−3 gate（`:588`），π compiled/eager 已有资格失败记录（`:543`），不能把“编译更快”默认为相同数值。

输出严格区分“bit exact”“处于重复噪声内”“超过 noise floor 但低于硬预算”“超预算”。工具 `SCREEN_PASS` 只允许进入配对 pilot，**不宣称部署等价**。首个分歧阶段/denoise index 可以从逐 tensor 表定位；完整 trajectory 的首个分歧需下一节的连续记录。

### (b) 缓存决策

必须同 query 序列、同 library view / row ordering / fit / stall model / recipe、同 episode reset、history、上一动作和 execution ACK 数；仅复制 RGB 和 seed 无法再现 stall 状态。当前冻结 cache 的 `on_execution(count)` 影响 controls，不能略去；参见冻结 `exp/r17_integration/search/cache_recipe.py:303`。

| 判据 | 预注册要求 / noise floor |
|---|---|
| hit/miss、top1 neighbour | 同 GPU重复比较作为决策 noise floor；跨 GPU目标 agreement≥99.9%，无解释的离阈值翻转为失败 |
| corrector / stall | 离阈值区域要求 100%；同时检查 corrected action 的数值差、执行动作与首次分支分歧。短筛查任何决策翻转都不通过 |
| 样本量 | 至少 2,995 个**独立**查询零分歧时，单侧95%分歧率上界才≤0.1%（1−0.05^(1/n)）；3帧零翻转的上界约63.2%，远不足以认证 |
| 连续查询 | 不把高度相关的每步当独立样本；按 episode/task 分层聚类 bootstrap 给 CI；额外覆盖初期/末期、success/failure、close neighbour、corrector ramp、stall cooldown / ambiguous / confirmed 分支 |
| 临界区域 | 对已暴露 scalar 取 ε=max(1e−6,5×同GPU scalar drift,2×跨GPU scalar误差)；top1 使用 gap，阈值用 signed margin；接近 ε 的样本单列，报告距离分桶和动作后果。不得靠删掉边界样本提高 agreement；新版另按相同donor ID对齐距离以识别排序gap不变的名次交换，缺共同候选覆盖记unknown |

工具比较 hit/top1/corrector/stall，保留 neighbours/distances/weights/diagnostics/校正动作；提供实测 boundary band。当前 recipe 不统一暴露 hit/stall 的连续 margin，缺失时写 **unknown**，不能写“远离阈值”。corrector ratio 的 .75/2 是 ramp 边界，不等同于布尔 `corrector_applied`（此处还受 history、step 和 MISS 影响）。工具永不自动签发统计证书：query 独立性与覆盖度不能从一份张量包推断。当前三帧循环序列并不提供独立查询样本量；正式ACK/history已运行，GR已触发slow_confirmed stall及active knob，但并非物理闭环轨迹。

### (c) 闭环 paired SR

每个 suite/arm 预注册等价 margin **±2 个百分点**。同一 task/init/env seed/policy seed、一模一样的 renderer/物理版本和动作执行协议，逐 episode 配对；pure 和当前 cache 分开测。对 `d_i=success_A−success_B` 给 paired 差及双侧95%CI，**整个 CI 落在 [−.02,+.02] 内才通过**。用配对多项式 score/exact 区间；可同时报 task-stratified paired bootstrap（10,000 次），但零 discordant 样本不能使用零宽 bootstrap CI。

报 b=仅A成功、c=仅B成功及 exact two-sided McNemar（二项 b|b+c,p=.5；b+c=0 时 p=1），三 GPU pair 的差异检测采用 Holm 校正。p>.05 不是等价证据。每 suite 另外同 GPU 重跑至少20个配对 episode，显示环境/调度 noise floor；若其 CI 本身超出 margin，先解决重复性。二项 exact 检验可用 [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)。

同步记录首个动作超(a)预算、gripper 符号翻转、cache 分支翻转、模拟 state/RGB 分歧的 control step；分布报中位数、p10/p90、按任务与分支分层，直到终止未分歧者右删失，不能丢掉。不同 renderer 的像素差先于策略时，归为环境差异（历史设备说明 `devices.md:197` 已提醒 GPU/software renderer 不能逐像素互换）。

Pilot预注册：LIBERO首suite libero_spatial每arm **500对**（10任务×50官方init）；π/GR MT50每arm **2,500对**（50任务×50init），pure和拟合cache各一组，三GPU复用同一PLAN。固定计划和预算见 [COMMANDS.md](COMMANDS.md:1) 及closedloop各PREREGISTRATION；π/GR用pi05_v2/groot_v2。20对只用于成本估计。不能临时扩大官方init池、使用保留seed9/10，或看过SR后调整margin。

按实际预注册保守exact方法的20,000次CPU模拟，discordance=.10且真差0时，500/2,500/5,000对通过±2pp门槛概率约0%/3.6%/81.7%；见[功效模拟](evidence/paired_power_design.json:1)，独立假设未计task聚类，5,000不是已授权扩容。500对正态95%半宽约2.8pp，本工具区间更宽；因此是有意义的风险pilot，不保证±2pp认证。100对半宽约6.2pp，仅能筛查大幅下降。完成全部任务仍可能结论为“不确定”；不得放宽margin来宣布通过。所有4090/H100工作由协调员执行；A100可直接执行任意时长本研究测试，但闭环仍需要排期独立正式renderer fleet。当前20个闭环campaign、正常prepare/launch及worker trace/renderer接线已CPU核验，实际远程prepare与GPU启动待协调员；见[CLOSEDLOOP_COMMANDS_1018.md](CLOSEDLOOP_COMMANDS_1018.md)。coordinator复用当前公共direct_transfer，模型运行仍是各自冻结runtime/封存bootstrap，不调用旧冻结launcher的tether_transfer。旧传输审计保留为历史证据。

闭环trace分析的first_action指首次在已观测query返回的完整action chunk上超过预算，未必是首次实际执行的control；state是wire proprio，不是完整模拟器隐状态。另报首次camera-byte hash不同，帮助发现renderer输入分歧。不能把这些观测点之间的未记录物理步宣称已排除分歧。`normalize_formal.py`核对封存PLAN与正式driver manifest，保留accepted失败结果，缺episode或runner错误拒绝计算SR。 新增独立controls模式：`client_trace.py`在下一次infer或terminal episode_end收到真实execution ACK时，只记录该返回chunk中确认选去执行的前缀；包含末段未全执行的情形。`executed_controls.py`逐control对齐、报首次动作/夹爪符号差及描述性KM/右删失，双侧缺trace的episode仍计unknown。记录仍是控制器/夹爪后处理之前的wire动作，不是关节力矩或完整模拟器状态。normalizer核对ACK与原query返回动作、时钟及accepted attempt；`paired_identity.py`拒绝PLAN/PIN/runtime/protocol混配，缺身份时即使CI足够窄也不签SR等价。所有接线目前只经CPU测试，尚未部署闭环worker。实际冻结worker factory创建stock client，且`benchmarks/execution.py:43`直接调用`_send_ctrl`提交terminal ACK；hook已在stock client及真实control层观测，避免绕过或重复末次前缀。冻结协议CPU集成已记录128 query、128 execution和16 terminal ACK，CUDA未初始化，见[收据](results/cpu_trace_wire_01/RECEIPT.json:1)。

## 4. 工具、真实输入和验证范围

| 文件 | 实现 |
|---|---|
| [capture.py](capture.py:1) | 有上限的 Recorder API；读取含 observation_file+完整 request 的 per_step；缺 RGB/state/显式噪声 seed 时拒绝。也可复用 Q real archive 或已有 tiny capture；记录来源、SHA、新诊断 seed 与 history/ACK |
| [artifacts.py](artifacts.py:1) | JSON+NPZ，不用 pickle；内容 seal；bf16 可无损存 fp32；字典键排序不改变输入 identity |
| [replay.py](replay.py:1) | 验证 frozen release / loaded spec；五种 staged adapter；真实 noise、prefix、denoise、action；拟合缓存；首repeat每帧比较 capture/plain exact；重复3次；内存上限和独立300s watchdog |
| [compare.py](compare.py:1) | CPU 输出(a)/(b) JSON；最坏跨repeat误差、noise floor、模型预算、gripper、决策CI/边界；缺失/非有限/不同seal拒绝；永不声称完成(c) |
| [inventory.py](inventory.py:1) | 包/driver/flag receipt；CPU import确认未初始化CUDA |
| [normalize_formal.py](normalize_formal.py:1) | 把正式driver accepted outcomes及可选新worker socket trace绑定到封存PLAN；拒绝缺episode、重复accepted、错误结果、改变任务集或seed，保留普通失败与未接受attempt审计；尚无本轮闭环实测数据 |
| [cpu_cache_check.py](cpu_cache_check.py:1) | 用真实冻结 DP recipe/fit/stall、合成 CPU query，测试 reset、history、execution ACK；不作质量结论 |
| [prepare_transfer.py](prepare_transfer.py:1) | DP最小公共rsync清单；真实用到的openpi包依赖闭包；不复制dataset/teacher RGB/H5，不下载大cache |
| [run_dp_local.sh](run_dp_local.sh:1)、[run_dp_remote.sh](run_dp_remote.sh:1)、[run_campaign.py](run_campaign.py:1) | 协调员命令入口；独立新目录、无端口/进程/chain修改 |
| [test_xgpu.py](test_xgpu.py:1) | CPU单元测试，覆盖错误观测、seal篡改、NaN、noise/决策翻转、缺阶段/重复、ACK、torch RNG observer恢复 |

复用已有 `exp/r17_integration/models/pi05/gpu_diagnostic.py:18` 的 action_difference，以及各 adapter 自带的 capture/snapshot / parity contract；`pi05/parity.py:40` 的 eager精确比较和 compiled替代容差只适用于它的资格范围，没有偷换为本项目 SR 门槛。

真实输入来自既有 `/data/mw_finetune/qualification/observations/cosmos_libero_raw_v1`：LIBERO-Spatial task0/init0/envseed0，settle10后3帧，原始双256RGB、state、含quaternion的Cosmos proprio，非合成；沿既有 Cosmos qualify 的旋转约定转换，另明示 request noise seed195。正式使用 **[inputs_dp3/INPUTS.json](inputs_dp3/INPUTS.json:1)**，identity `5361e28384e29ffc28b9e6417e2530538f6a38836c1584eaf43c3b3eed491997`。旧 `inputs_libero3`、`inputs_tiny3*`、`prepared_a100*`、`prepared_dp`、`prepared_dp_v2` 是开发审计遗留，**不传输/不执行**；按用户要求保留，没有删除。

抽样的现有 per_step 仅有 metrics、seed7、task/init等，不包含恢复 RGB/state 所需数据；见 [DP样本](evidence/hs5dp_ctl_142812_step_sample.json:1)、[FW样本](evidence/hs6fw_ctl_143332_step_sample.json:1)。所以本轮没有声称“从日志重建原 episode”。新增连续 capture 只能在独立运行外层接入 Recorder，保存真实 request.explicit_seed、history和execution ACK，不能改在线 lane。

**旧B1阶段实测：30个CPU单测通过（当前总计85个通过）；完整DP CPU回放9 rows，耗时108.17s，3帧各自capture/plain exact且3次重复无张量或缓存决策漂移。** CPU单测与集成结果见 [tests.txt](evidence/tests.txt:1)、[cache_cpu_integration_current.json](evidence/cache_cpu_integration_current.json:1)、[cpu_replay_final.txt](evidence/cpu_replay_final.txt:1)。CPU自比较 [cpu_self_comparison_final.json](evidence/cpu_self_comparison_final.json:1) 为SCREEN_PASS；完整 CPU DP 回放状态以 [REPLAY.json](results/cpu_dp_native_final/REPLAY.json:1) 为准；这不是GPU证明。旧B1端到端GPU recorder只验证过DP；并发结果另见§0。π用只保留原forward输出的 stage2_capture，并强制plain parity；Cosmos记录latent conditioning和x0_fn每次输入/预测，不声称暴露了closure内所有hidden states。未实现通用实时attention kernel profiler或共同noise注入；native noise若不同即停下，不继续给算术等价结论。

## 5. 原因优先级及恢复操作顺序

| 次序 | 对三GPU做同样处理 | 验证 / 解释 |
|---|---|---|
| 0 | 固定权重、规范化、库/fit/stall、完整输入/实际noise、K/C、runtime、island；正式并发/MPS/graph是主对照，B1仅辅助定位 | 先排除环境或输入差异；4090/H100版本相同不等于kernel相同；A100复制使用中的DP依赖，不选 nominal env |
| 1 native | 每GPU两个独立冷启动，每次两轮并发，取(a)/(b)首次分歧 | 不先“一次开所有开关”掩盖来源 |
| 2 tf32_off | **加载后** torch matmul/cudnn allow_tf32=False | DP fp32路径最有直接解释力；π/Cosmos构造器会改全局状态；bf16主计算不会因关TF32自动成为fp32 |
| 3 highest | 加 `set_float32_matmul_precision('highest')`；实际JAX路径再设highest | 与TF32 flag耦合，可能是同一个有效状态而非独立实验；比较receipt，若相同记为冗余操作 |
| 4 共同attention | π/DP/FW试SDPA math；确认实际MHA/fastpath；GR固定三卡都支持的FA2或已资格化math路径；Cosmos固定TE backend并检查真正kernel | flash-attn/TE直接调用不会被SDPA flags替换。未经vendor backend资格检查，不能声称math profile统一全部模型 |
| 5 eager / fixed compile | 关Dynamo、禁用配置中的compile；固定batch、取消autotune变体；之后才单独复测固定compiled产物 | 本工具staged可能原本就是eager；“无变化”不证明live compiled等价；不跨架构复制编译cache |
| 6 deterministic | 新进程启动前 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，cudnn deterministic=True/benchmark=False、`torch.use_deterministic_algorithms(True)` | 不支持算子报错就记录失败，不静默warn-only。仍不保证跨架构相同 |
| 7 完整reduction / fp32局部定位 | 关bf16/fp16 reduced reduction；若最早差异仍在prefix/vision，单独把相应模块/归一化提升fp32，保留其余不变 | 后者需要新runtime身份和模型资格，不是对现有lane的热补丁；测内存/延迟代价 |
| 8 cache稳定化候选 | 若model drift小但最近邻/ramp/stall翻转，固定CPU float64评分/归一化并保留统一的稳定row顺序/tie-break；按共同数值路径重建库/fit并重做(a)(b)(c) | 不能只改query不改library；不能随意扩大阈值或合并近邻来“制造通过”。先定位score误差和边界，再验证修复 |

逐项累积，但每个profile启动新resident、生成独立结果。先比较native、TF32 off、highest；然后按模型固定attention族；再关compile/graph，最后deterministic和完整reduction。恢复候选必须满足跨GPU(a)(b)，并回退单项确认因果；最终还需(c)。未测通过前维持baseline/cache组合的GPU亲和性。

| 模型 | attention / compile专用项 | 当前可执行性与限制 |
|---|---|---|
| π0.5 | `--numerics sdpa_math`；`compile_off`，实际正式阶段路径是否触及compiled sample_actions须记录 | 原R4在A100物理OOM；先标记独立R3容量arm，不能混入R4等价结论 |
| GR00T | 保持同一FA2 2.7.4.post1二进制/输入shape；compile_vision=False | 源Radio对SM≥8主动FA2，Eagle/SigLIP/Qwen亦选FA2；SDPA开关不统一它，尚无合格math替换。不得假称fixed FA2意味着相同架构kernel |
| DP | `sdpa_math`；`graphs_off`单独关闭正式CUDA graph | TF32 off首先作用fp32 GEMM/conv；graph off显式标不同执行路径 |
| Fast-WAM | `sdpa_math`；`compile_off` | 4090 CPU-T5 cache路径和H100路径先分开核对；不能把不同运行时的差归因GPU精度 |
| Cosmos | `--numerics cosmos_math`：禁用vendor FA3分支、只替换其局部sdpa_kernel selector为MATH，同时NVTE_FLASH_ATTN=0、NVTE_FUSED_ATTN=0 | 原attention的cast/scale/transpose/SDPA调用保留；TE部分仍是原生Unfused且含fused masked softmax。首次实际进入math context写ATTENTION_OVERRIDE_VISITED.json；CPU测试通过，GPU未验证。`te_unfused`单独只影响TE，不能统一minimal_a2a |

**Cosmos A100加载后实测：**07原R8启动过程中前五个resident均已写live收据；首个为matmul precision=high、matmul TF32=True、cuDNN TF32=True、benchmark=True、deterministic=False、deterministic_algorithms=False，CUBLAS_WORKSPACE_CONFIG未设。环境DETERMINISTIC=True未覆盖这些最终flag。Torch2.7.0+cu128、cuDNN90701、FA2.7.3+cu128.torch27、TE2.2+cu128.torch27，见[live numerics](evidence/a100_cosmos_live_numerics_07.json:1)。这只是加载后状态：Cosmos `models/cosmos/staged.py:90`、`:94`每次stage1调用set_seed_everywhere，vendor `utils/utils.py:36`、`:37`随后设cuDNN deterministic=True/benchmark=False；正式parity也调用它。故不能把装载时benchmark=True当作正式推理时的最终状态。新增NUMERICS_AFTER_STAGE1.json观测，CPU验证不改返回对象/数值，仅采一次；R4已实测stage1后deterministic=True、benchmark=False，TF32双开/high保持，见[stage1后收据](evidence/a100_cosmos_after_stage1_numerics.json:1)。

**确定存在的架构分派差异：** Cosmos vendor `cosmos_policy/_src/predict2/networks/attention.py:121`、`:136`、`:145`、`:178` 明确按SM选择FA3/cuDNN/Flash优先级；`configs/video2world/defaults/net.py:39`和`networks/a2a_cp.py:208`接入minimal_a2a。它是跨卡数值差异的具体候选机制，尚未证明造成SR下降。

对应命令入口为`run_concurrent.py --numerics <项>`，重复参数表示累积。`numerics_bootstrap.py`在load之后再次设置TF32/precision，避免π/Cosmos loader覆盖；进程环境在import之前设置。TE原生backend依据vendor `.venv/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py:5689`、`:5701`；GR实际FA2路径见`evidence/groot_imported_distributions_complete.json`与A100 live版本收据。参数存在不等于操作已恢复等价；**截至本报告没有经过跨卡验证的恢复项**。

## 6. 操作审计与下一步

资源和唯一sudo命令见§0“空间与操作审计”；机器的最初preflight、随后实际执行、失败启动、成功RUN及清理分别留存。远端严格单模型、公共rsync 16MiB/s单流，不复制大数据集/teacher H5；删除仅限本研究创建且已完成测试的权重/env目录。

首个协调员GPU工作：**DP / 4090，R4/B8/48客户端，空闲8GiB、8–15分钟**；随后H100同一检查。精确命令及五模型排期在 [COMMANDS.md](COMMANDS.md:1)。协调员自行决定如何腾出资源及恢复lane，本工具不会改在线副本、chain或worker。Cosmos R4两次冷启动已完成；当前串行准备π R3；已启动后继队列，只有π两个冷启动COMPLETE、私有MPS关闭且前驱进程结束后，才复用openpi环境重跑新版DP recipe。若π失败，队列停止且保留现场；若DP原生四轮超预算，则另做两次TF32-off冷启动；跨GPU与闭环缺测继续明确保留，不能以同卡repeat通过替代。

A100当前后继队列的可审计入口：[a100_after_pi.py](a100_after_pi.py:1)，提交收据[evidence/a100_continuation_submission.json](evidence/a100_continuation_submission.json:1)，运行日志evidence/a100_pi_then_dp_queue.log。它只等待本研究前驱PID及start-ticks，不发送信号；环境/权重清理仍由原owned-path保护执行。CPU测试确认缺任一冷启动、未关闭MPS或FAILED均不会开始handoff。队列状态不代表这些测试已完成。
