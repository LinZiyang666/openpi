# A2：运行与建批冗余防御普查

日期：2026-10-23。依据 prompts/COMMON.md、DESIGN.md「拔的防御」及 owner 要求。A = 不影响数字、不防混批，删除；B = 数值/数据身份/完整性，保留；C = 现有实现把冗余门与真实约束绑在一起，需要协调者裁定。下表按同一检查链合并列出，逗号后的数字也是该文件的行号；已删除检查定位到当前工作树中相邻行。

本轮修改 72 个 Python 文件和 1 个 shell 文件；新建本文。W1–W4 占用的实现文件未改；本文是任务明确要求的 seeds_1023 交付例外。没有提交，没有运行实验、pytest、拟合、数据汇总、远程命令或 tether。Python 修改只做 py_compile；shell 语法及运行验证交协调者。

普查对象为任务给定的 ops/search/common/debug、R17/R15/R16 运行与建批源码；tests、followup_*、frozen_runtime、before*/after*、native_sources、worker_payload、execution_tools_* 等冻结副本不作为修改对象。哈希生成并不等于哈希防御：兼容回执、内容目录命名、任务 UID、fit/store/Q3 关联字段继续保留。清单不承诺将所有 digest 字段清零。

## 已拔 A，以及同处恢复的必要 B

SHA 参数、manifest、返回值中的 digest/identity/sha256 字段保持兼容；旧 status=verified 或 *_unchanged 字段不能再单独解释为“本轮重新逐字节核验”。inspect_stage、formal_a_short、cache_files 已显式标记 hashes_checked=False。所有数值检查仍需下文回归，编译通过不等于实验验证。

| 当前 file:line | 修改及理由 |
|---|---|
| `exp/offline_search/rounds/r15/descent/analyze.py:7,84,88,93,677,679,695` | 拔源码 AST 锁、PROTOCOL 文件自封条、输出软链门和固定 CPU affinity；pre-outcome 注册顺序、served-fit 和输入 cohort 校验保留。 |
| `exp/offline_search/rounds/r15/descent/audit_fits.py:1,24` | 拔 base/serving 文件重复 SHA；LOEO heldout/donor 不相交、library-only 和 fit 的 episode 绑定保留。 |
| `exp/offline_search/rounds/r15/descent/freeze.py:9` | 拔 PROTOCOL_SHA256.json 一次性存在门；继续记录元数据。 |
| `exp/offline_search/rounds/r15/descent/prepare.py:55,287,316` | 拔 fit 和定义文件 SHA、普通输入软链门；新目标目录和 A500 数据语义保留。 |
| `exp/offline_search/rounds/r15/pure_corr/analyze.py:41,43` | 拔 manifest/summary 字节重验；manifest 实际 A 集合、完整局数和终止回执仍检查。 |
| `exp/offline_search/rounds/r15/pure_corr/build.py:239,247` | 拔旧源码/产物清单逐文件重验；cell/size、实际数组/权重/calibration 一致保留。 |
| `exp/offline_search/rounds/r15/qk/impl/deployment.py:49` | 拔 artifact 文件 SHA；实际 light_key_spec、编码输出形状与数值校验保留。 |
| `exp/offline_search/rounds/r15/qk/impl/descent_safety.py:183,185,204,211` | 拔旧源码 AST 相等锁、fit 重验及 serving source 不准变；继续记录实际 AST 是否相等与训练/评估隔离证据。 |
| `exp/offline_search/rounds/r15/qk/impl/encoder.py:60,77,87,93` | 拔 student/projection SHA、threads==1 和固定 CPU 编号；默认 affinity 继承当前允许集合，形状/有限数保留。 |
| `exp/offline_search/rounds/r16/conductor_deploy.py:52,53` | 拔 payload 源文件软链门，tar 解引用软链。 |
| `exp/offline_search/rounds/r16/conductor_seal.py:22,24,25,36` | 拔原源码 SHA/AST 和精确 loader AST 锁；仍输出实际未变函数/源码元数据。 |
| `exp/offline_search/rounds/r16/k1ref/runner.py:51` | 拔必须先有 R16_SNAPSHOT.sha256 的准入；端口/隔离启动流程保留。 |
| `exp/offline_search/rounds/r16/k1ref/snapshot.py:13,25,37,39,47,52,59` | 拔旧 commit 的 git-show 源码回填、固定目标目录、源码重验和只读封条；保留 fresh 目录及 manifest 成员存在性。 |
| `exp/offline_search/rounds/r16/k1ref/workflow.py:59,60,83,91` | 拔自身 CODE_LOCK、DEV 冻结重验及 read_sealed 的逐文件 SHA；上游数值/结果语义检查仍走原流程。 |
| `exp/offline_search/rounds/r16/prepare.py:61,62,226,229` | 拔 reference source/fit、输入/design/amendment/source 锁；实际冻结 cell 集与 adaptive 口径保留。 |
| `exp/offline_search/rounds/r16/resident.py:94` | 拔 fit/YAML 逐文件重验；fit 必须存在、entry 配置及显式已拟合 artifact 保留。 |
| `exp/offline_search/rounds/r16/retest.py:31` | 拔 baseline/source 文件锁；实际实验 identity、回放/聚合口径保留。 |
| `exp/offline_search/rounds/r16/retest_combined.py:26` | 拔 source 和 lock-on-lock SHA；combined schema/选择逻辑保留。 |
| `exp/offline_search/rounds/r16/retest_superseding.py:27` | 拔 source/combined lock SHA；owner_selection 选定的重做 cohort 仍绑定。 |
| `exp/offline_search/rounds/r17/autoscale_1017/formal_trial.py:122,153` | 拔 runtime archive/fix SHA 门及 staging 后源码重验；longseg patch 和 task/UID/init/seed 一致仍检查。 |
| `exp/offline_search/rounds/r17/autoscale_1017/legacy_release.py:34,56` | 拔 overlay 一次性源码封条、远端 sha256sum 和逐源 SHA；已生成 release 标识仍保留。 |
| `exp/offline_search/rounds/r17/filter_failed/commands.py:23` | 拔复制 fit 后重复算 SHA 的 assert；删除前限定自己生成的路径检查保留。 |
| `exp/offline_search/rounds/r17/filter_failed/cpu_startup.py:29,39,55,69,76` | 把依赖版本、源码/tokenizer/checkpoint 指纹相等改作观察字段；batch/model 配置、输入存在性仍检查。 |
| `exp/offline_search/rounds/r17/filter_failed/inspect_stage.py:1,18,20,24,26,33,103,143` | 阶段诊断改为文件存在清单；旧 sha/identity 字段兼容，增加 hashes_checked=False，不再以源码 SHA 阻塞。 |
| `exp/offline_search/rounds/r17/filter_failed/resume.py:122` | 拔 runtime/workers 复制后 SHA assert；原任务 UID 和 episode cohort 对应保留。 |
| `exp/offline_search/rounds/r17/filter_failed_knob/formal_a_short.py:92,109,146,148` | 拔命名迁移前后文件 SHA 封条；计划/model/entry 语义仍对齐，旧计数字段保留并加 hashes_checked=False。 |
| `exp/offline_search/rounds/r17/filter_failed_knob/startup.py:24,34,42,44,46` | 拔 loader 源码 AST/SHA 门及启动资产重复重验；提取 descriptor 的结构边界、cache 输入绑定及 replica/batch/MPS 配置保留。 |
| `exp/offline_search/rounds/r17/longseg_1017/profiles.py:31,34,45` | 拔读取首行的 16MiB/256-block 上限，读到所需完整 metadata 或 EOF；缺字段仍报错。 |
| `exp/offline_search/rounds/r17/longseg_1017/vfix.py:94,136` | 拔 bootstrap 重复 SHA 门；longseg 补丁锚点和 UID 检查保留。 |
| `exp/offline_search/rounds/r17/scheduler/defaults.py:4,8,10,85` | 拔 driver/workers.tar 中 scheduler VERSION 源码锁；调度服务健康检查保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/a100_closedloop_run_1018.py:159,195,216` | 把 legacy runtime、worker payload 和缓存依赖逐文件重验改为存在性检查；计划、实际模型和专用实验端口/资源边界保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/cache_files.py:1,10,28,36,45,47` | 拔 stat 封条、一次性 CACHE_FILES 门和每次使用前后字节重验；serving bundle 关联、路径、所需文件存在性保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/concurrent_prepare.py:46` | 拔 ffk_bootstrap SHA 准入；仍从所选 campaign 构建依赖和 groups。 |
| `exp/offline_search/rounds/r17/xgpu_1017/concurrent_server.py:69,168,178` | 拔 PIN 自封条、源码/资产重复重验和 model-spec 指纹门；DP CUDA graph、Cosmos native parity 和实际 reference 模型/配置绑定保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/harness_snapshot.py:29` | 拔 harness 源码 snapshot 的自封条和二次读文件门；目录锁和显式成员保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/pi05_checkpoint_view.py:39,41,49,52,60,64` | 拔 checkpoint view 的软链/stat/文件重验和小文件 1MiB 门；所需成员、checkpoint receipt 关联及硬链接同文件系统要求保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/pilot_campaigns.py:82,83` | 拔继承 runtime 的 SHA 相等门；仍按 campaign 选择来源且禁止改写已派发 run。 |
| `exp/offline_search/rounds/r17/xgpu_1017/pilot_formal.py:76,137,156` | 拔 PREPARED/运行时源码重验；私有输入改为存在性检查，canonical plan 关联与端口限制保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/replay.py:24,157` | 拔运行时源文件重验和 fit 自哈希/NPZ 重验；READY→YAML fit identity 及真实模型配置绑定保留。 |
| `exp/offline_search/rounds/r17/xgpu_1017/temporary_h100_stage.py:136,137,199,200,203` | 拔小 checkpoint 文件的 1MiB/SHA 门、tokenizer 和远端源码锁；依赖存在性与完整 source RUN 保留。 |
| `exp/r17_integration/common/batch_runtime.py:23,26` | 拔 max_batch≤32、batch_wait_ms≤50 的硬上限；正整数、有限非负数仍校验。 |
| `exp/r17_integration/common/recipe.py:3,32,37,54,74,130,137,138,141,154,161,166,172` | 拔缓存 store 的 stat/inode/mtime 封条；保留按路径和 digest 缓存。原有 Q3、store、sampler、fit 设置绑定的空 pass 恢复报错（B）。 |
| `exp/r17_integration/debug/reader_profiles.py:9,259,261,313,350` | 拔 capture/物理 block/H5 的逐文件 SHA 重验和软链拒绝；完整 writer、路径归属、ACK/decision/episode 关联仍检查。 |
| `exp/r17_integration/ops/build_client_bundle.py:51,115,117` | 拔源文件软链拒绝、PAYLOAD 自哈希、成员逐文件重验；归档路径、成员完整性、重复成员检查保留。 |
| `exp/r17_integration/ops/cache_capacity.py:87,89,94,98` | 拔 T1 recycle 证据的本地/远端源码 pin；实际 exit75、失败数和回收行为证据保留。 |
| `exp/r17_integration/ops/cache_capacity_lanes.py:38,44` | 拔 prepare_workers 前后对 native 文件重复重验；容量参数及任务计划关联保留。 |
| `exp/r17_integration/ops/cache_control.py:16,29,35,47,83,106,109,114,137,194,207,262` | 拔 control/worker 源码重验；2000 局写死改按计划任务数校验，原有完成/终止错误、对照模型和运行节点空 pass 恢复报错（B）。 |
| `exp/r17_integration/ops/cache_cpu_prepare.py:5,10,15,20,23,26,28,31,34,37,39,43,46,49,63,73,75,83` | 拔 CPU 恢复的目录/机器限制、自封条和代码重验；重复派发、study、完整转移、任务语义等空 pass 恢复报错（B）。 |
| `exp/r17_integration/ops/cache_model_order.py:115` | 拔 PREPARED 文件重验的空 pass；队列锁、前批结束和派发流程保留。 |
| `exp/r17_integration/ops/deploy.py:8,14,43,55,56` | 拔 attempts≤8 空门和归档/文件重验；正整数、解包路径及已有 payload 冲突保留。 |
| `exp/r17_integration/ops/driver_release.py:90,92,124` | 拔 driver 源码/manifest 封条与续跑复制后的重验；实际运行时存在性保留。 |
| `exp/r17_integration/ops/formal_resume.py:30,45,81,93,104,107,118,125,169,180,189,196` | 拔源码/资产/运行时重验与原机器字段锁；续跑 task/protocol、终止 attempt/error、连续 trace、模型/回执闭合的空 pass 恢复报错（B）。 |
| `exp/r17_integration/ops/formal_runtime.py:8,29,34,45,48,57,70,83,135,142` | 拔 native/runtime/worker 源码重验；paired lineage 的完整回执、任务集合/分区、完成局数空 pass 恢复报错（B）。 |
| `exp/r17_integration/ops/immutable_release.py:63,64,80,81` | 拔安装前后源文件/归档/manifest 的重复 SHA 门和软链门；安装锁、常规文件及路径检查保留。 |
| `exp/r17_integration/ops/library_alignment.py:102,109,116,122,124,152,160` | 拔固定旧种子、旧随机排列重算和 receipt 自哈希；subset_seed 记录实际种子，保留 suite/size/跨模型成员对应，输入软链放行。 |
| `exp/r17_integration/ops/library_batch.py:27,49,50,53,94,135,137,138` | 拔只准 weilandserver 回执、无害软链门及整片/归档的重复重验；campaign/batch/shard 关联、partial 和目标冲突保留。 |
| `exp/r17_integration/ops/library_capture_views.py:24,30,44` | 拔 view 目录/成员的单纯软链门；已存在 view block 冲突与 episode 划分仍检查。 |
| `exp/r17_integration/ops/library_init.py:1,26,31,54,67,72,81` | 拔历史 .init 重复重验及归档/成员多轮 SHA；tar 解引用软链，保留 B 池结构和目标冲突检查。 |
| `exp/r17_integration/ops/library_repair.py:178` | 拔 repair provenance 自哈希；原批采集/episode/alignment 关联和修复合并的完整局检查保留。 |
| `exp/r17_integration/ops/library_salvage.py:222,247,250,253,258` | 拔 salvage 自哈希、只准 weilandserver 的 shard 回执门和落地后重复重验；保留 campaign/shard/目标/partial/成员集合与闭合证明。 |
| `exp/r17_integration/ops/live_fleet_worker.py:24,30,46` | 拔用源码字符串判断 worker 能否加入的门及 pin 的十六进制长度门；payload 信息字段保留。 |
| `exp/r17_integration/ops/live_fleets.py:30,102,148,153,322,329` | 拔 worker≤128、名字≤64 字符及若干空 pass 前的源码/归档重验；正数、合法名字、fleet 归属/关闭检查保留。 |
| `exp/r17_integration/ops/pod/setup.sh:40,42` | 拔 proot 下载后硬钉的 sha256sum；现有下载和安装流程不变。 |
| `exp/r17_integration/ops/queue_retry.py:22,25,34,162` | 拔启动快照软链门、输入/复制后源码重验；零 accepted、原批闭合和新续跑目录约束保留。 |
| `exp/r17_integration/ops/remote_json.py:1,5,16,21` | 拔为了校验小 JSON 额外发起的远端 Python 命令及本地重复重验；reference/sha256 返回字段兼容。 |
| `exp/r17_integration/ops/smoke.py:1,41,44` | 拔 qualification 的 900 秒上限；仍用显式 deadline 停止本次 child，回执记录实际配置时限。 |
| `exp/r17_integration/ops/sync_back.py:17,44,65,73,83,85,86,95,111,113,114,118` | 拔软链准入门、快照后二次扫盘、归档/成员/manifest 重验；忙 journal 锁、成员完整性、越界和已有目标冲突保留。 |
| `exp/r17_integration/ops/tether_retry.py:9,133` | 拔 seconds≤300 限制；默认 300 秒、有限正数、单次 deadline 和禁止重放含糊 spawn 的行为保留。 |
| `exp/r17_integration/ops/worker_installs.py:37,40,52,62,72,85,125,138,140,146,149,154` | 拔依赖/资产安装重复 SHA 和软链准入；仍检查所需文件存在，目录身份字段兼容。 |
| `exp/r17_integration/search/audit.py:2,6` | 拔 freeze 自封条和 code 源码依赖锁；其余数值/数据 cohort 依赖一致仍检查。 |
| `exp/r17_integration/search/build_library.py:231` | 拔 stock artifact 文件重验；保留 artifact 的实际 binding、完整局和 B 池资格。 |
| `exp/r17_integration/search/fit.py:137,142,144` | 拔 READY 自哈希、fit.npz 重验；保留 schema、expected_binding 和拟合完整局/无测试泄漏检查。 |
| `exp/r17_integration/search/freeze.py:11` | 不再强制提供 code identity；数值与 cohort 的依赖项仍必须齐全，已有 code 字段可继续保存。 |

## 待拔：占用文件，只列清单

行号为本轮静态阅读位置；其他工作流同时修改这些文件，合并时以函数名和条件内容为准。已被 owner 工作流删除的条目直接划掉，不重复落补丁。

| 归属 / file:line | 分类与具体动作 |
|---|---|
| W1 · `exp/offline_search/rounds/r17/filter_failed_knob/build.py:93,109,121` | A：去掉 official/root 输入逐文件 sha 重验及 run 名必须以 ffk1015_/fa_ 开头的门；保留选中 library/fit/cohort 之间的关联和已派发拒绝。 |
| W1 · `exp/offline_search/rounds/r17/filter_failed/stage.py:154,190` | A：去掉 runtime source SHA/fix receipt SHA、staging 后再次比 runtime SHA；保留 longseg runtime_fix 锚点和 planned_tasks==EPISODE_PLAN。 |
| W1 · `exp/offline_search/rounds/r17/longseg_1017/staging.py:21` | A：去掉普通输入软链拒绝；打包按实际内容处理。任务书中的同目录 sync_back.py 未找到，实际通用 sync_back.py 已在已拔表。 |
| W1 · `exp/offline_search/rounds/r17/longseg_1017/recalibrate.py:300` | C：samefile 要求是否仅用于确认共享 C5 Q3？若已有 fit/metric 绑定可证明同一 gate，去掉 inode/物理文件别名要求；`:107` 的成功标签一致必须保留。 |
| W1 · `exp/r17_integration/ops/cache_study.py:225,273,390,528` | A：validate_manifest/fit_inputs/stage_inputs/seal 中剩余自封条、逐文件 rehash 可删；B：fit 的 store/commit/binding、输入命名碰撞、Q3 缺失和 group 完整性不能删。 |
| W1 · `exp/r17_integration/ops/cache_knob.py:109,124,143,166` | B：实际 K、每任务档位、合法 gap、缺 gap 报错，均决定数字；不因含 raise 删除。 |
| W1 · `exp/offline_search/rounds/r17/filter_failed_knob/runtime.py:22,82`、`build.py:163` | B 修复交接：selection 缺 fit_digest 必须报错、按 seed 查成功局；已看到 build 按 row.seed 索引，运行回归仍要覆盖“缺新 seed 不退旧库”。 |
| W1 · `exp/offline_search/rounds/r17/longseg_off_1022/workflow.py:79`、`build.py:80,163`、`fit500.py:1` | B 修复交接：缺 C5 fit 不静默丢格，复用路径/YAML key 带 seed 和开关；`workflow.py:113,121` 的同卡参照和完整 cohort 留。 |
| W1 · `exp/r17_integration/ops/formal_a_analysis.py:86`、`exp/offline_search/rounds/r17/knob/common.py:99`、`filter_failed_knob/evidence.py:16` | B：分析/evidence 键必须包含 seed、segment；evidence 的 patch 锚点与实际采样字段不要作为“源码锁”拔掉。 |
| W1 · `exp/offline_search/rounds/r17/seeds_1023/results_seeds.py:305`、`results_lib/native.py:39,59,96,116` | B：跨 seed 冲突、完整局、A pool 和参照匹配；该目录只交本文，其余归 owner 工作流。 |
| W2 · `exp/offline_search/rounds/r15/descent/fits.py:43,66,77` | A：固定旧 seed 覆盖集、来源/旧 artifact SHA 门；B：`:46,51,69` query≠donor、步号连续和实际 seed 的 episode 子集完整性。 |
| W2 · `exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020/build.py:228` | A：删除 assert len(offs)==8，改按本次定义的格子处理；已有 DONE/完整计划检查留。 |
| W2 · `exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020/fit.py:66` | B：拟合行与 Q3 episode 对应。DESIGN 指出的旧行 `:173,195` 旧门周期对照若在合并版本仍在则 A 删除；不得解除服务端旧门禁用。 |
| W2 · `exp/offline_search/rounds/r16/analyze.py:41,49,57,63,86,320,352` | A：design/source/catalog/definition/fit 文件重验、自封条及纯输出一次性门；B：实际 served fit 关联、输入 cohort、pre-outcome 选择顺序留。 |
| W2 · `exp/offline_search/rounds/r16/conductor_analysis.py:24,39,172,180,219` | A：源码锁、resident fit 逐文件重验和一次性输出门；B：`:51,79,175` contract.fit/pool、实际 runtime args 和 DONE→group 对应留。 |
| W2 · `exp/offline_search/rounds/r17/filter_failed_knob/formal_a_queue.py:1`、`vla.py:1`、`q3_port_1020/analysis.py:58,64`、`seeds_1023/vla/__init__.py:1` | B：参照同卡同步数、variant、seed/segment 合并键；非数值的旧 source locks 由本 owner 清理，不在 A2 中改。 |
| W3 · `exp/r17_integration/search/store.py:73` | A：manifest 自 digest 子句，若还有数组逐文件 SHA 一并删；B：schema/READY、数组几何和实际 store 绑定。 |
| W3 · `exp/r17_integration/search/teacher_library.py:172,178` | A：VIEW 自哈希、row_indices.npy 重验；B：RowIndexStore 的完整局/完整 quotas，已有 view/fit 与所选库设置冲突不能删。 |
| W3 · `exp/r17_integration/search/cache_recipe.py:132,136,140,146,149,240` | A：READY 自哈希、逐文件 hash；B：receipt.digest 与所选 fit、store_digest、commit、binding 关联及 stall 的 fit/store/metric。原任务 `:160–170` 在当前文件已前移，按 verified_fit 条件识别。 |
| W3 · `exp/r17_integration/debug/library_capture.py:55`、`teacher_library.py:23`、`cache_recipe.py:67` | B：必须是明确 seed 的 library-only B 数据；旧固定 seed 门已见 owner 改动，不再复制修改。 |
| W3 · `exp/r17_integration/ops/library_campaign.py:371` | A 候选：仅要求 Cosmos handback 标记或指定构建机器的门可删；B：实际 bundle 可用性和完整原批配对。`:241,352` cohort/closure 留。 |
| W3 · `exp/r17_integration/ops/library_staging.py:73,117,225,239,244` | A：只拒软链、整片 rehash、archive SHA 重验和旧只准本机 finish；保留 campaign/shard 归属、receipt 关联、无 partial 和关闭 ACK。quota 上限见 C。 |
| W3 · `exp/r17_integration/ops/RUN_CONFIGS.md:1` | 待同步说明：源码 hash 是元数据，不再是通用准入；不要把 SHA 字段删除误当作不再区分数据批次。 |
| W4 · `exp/r17_integration/ops/formal_prestage.py:46,175,193,197,207,231,280,300` | A：verify_code 的软链拒绝和 source/archive/manifest rehash，含仅剩 pass 却仍计算 SHA 的分支；B：`:173,257,296` 防重复计划、实际计划语义、LAUNCH.json 独占认领。 |
| W4 · `exp/r17_integration/ops/formal_eval.py:1`、`formal_finalize.py:363,472` | A：运行时/worker/finalizer 源码封条可清；B：driver_completion、完成局数、终止错误、owned closure。`formal_finalize.py:498` GPU 交还前后 evidence 输入关联暂保留，见 C。 |
| W4 · `exp/r17_integration/ops/formal_ports.py:1`、`exp/offline_search/rounds/r17/filter_failed_knob/launch.py:12,44` | B：端口冲突与 launcher 的实际 bootstrap/evidence 注入；当前 launch 未见额外 source SHA 门，不另造改动。 |

## 保留 B：检查链及理由

这些不是“异常分支一律删”。同文件可能同时有 A；这里只说明保留的条件。简写如 search/、ops/、common/ 均相对 exp/r17_integration/；r15/、r16/ 相对 exp/offline_search/rounds/，同一行未写目录的文件沿用前一完整目录。

| file:line / 检查链 | 为什么留 |
|---|---|
| `exp/r17_integration/common/recipe.py:25,30,35,53,73,132,160` | 服务端旧门禁用；缺 Q3 不能以 stall_only 启动；fit/store/sampler/settings 绑定。原来部分是 pass，本轮恢复失败时显式报错。 |
| `exp/r17_integration/search/cache_recipe.py:140,146,149,180,241` | fit 与 YAML 的 store/commit/task/binding；Q3 的 B 成功 LOEO、fit 与冻结度量映射；浮点误差边界也影响门的数值。归 W3，未改。 |
| `exp/r17_integration/search/teacher_library.py:187,212,215` | RowIndexStore 不能拆局，嵌套库必须完整 quotas 且来自 parent；防行索引拼错局。 |
| `exp/r17_integration/search/fit.py:14,26,31,40,62,65,70,73`、`search/build_library.py:35,47,49,60,81,94,117,126,135,143` | B/library-only、完整 episode、显式特征/状态/动作几何，拒绝测试局混入拟合。 |
| `exp/offline_search/rounds/r10/recipe/fitting.py:24,37` | 用户指定的拟合防测试集泄漏范围外不改；也没有为了删 guard 在服务端启用旧门。 |
| `exp/r17_integration/common/layers/guard.py:14`、`layers/recipe.py:53,298,326` | NoProgressGuard / TriggerCommitJudge / guard_only 的服务端退休门保留。拟合进程例外由相应 owner 限定作用域。 |
| `exp/r17_integration/common/layers/math.py:62,92,111`、`common/contracts.py:96,300` | 矩阵/权重/形状/有限数/真实特征来源检查会直接改变拟合与推理结果。 |
| `exp/r17_integration/common/serving.py:178,194,215,227,253,263,328`、`common/request_rng.py:1` | bundle/request/seed/cursor/执行 ACK 不可错接；RNG identity 是采样输入，不是防御封条。 |
| `exp/r17_integration/debug/attempts.py:17,63,76,95`、`debug/reader_profiles.py:100,234,258,301`、`debug/library_shards.py:58,81` | 不把 torn、未 ACK 或不同 attempt 的 capture 算成完整局；保留完成/终止错误判断。 |
| `exp/r17_integration/debug/capture_bridge.py:57,78,186,252,434,526` | 重复/串局 decision、benchmark、bundle 及 terminal capture 对应，防不同批次证据混在一起。 |
| `exp/r17_integration/search/audit.py:14`、`search/replay.py:1`、`search/analyze.py:23` | accepted 决策连续性、tail 实际调用数和 IR 口径/实际 MISS sampler。 |
| `exp/r17_integration/ops/formal_resume.py:53,75,89,99,113,128,189`、`formal_runtime.py:24,35,52,58` | 续跑不能重用不同协议/模型/任务的 outcome；终止失败、缺决策或未合并完整批次不能通过。 |
| `exp/r17_integration/ops/cache_control.py:24`、`cache_cpu_prepare.py:14` | 对照必须完整、终止错误为零；CPU 恢复必须是未派发的同 study/episode cohort，不把已跑批次重新派发。 |
| `exp/r17_integration/ops/formal_prestage.py:296`、`formal_ports.py:1`、`exp/offline_search/rounds/r17/longseg_1017/ports.py:1` | LAUNCH.json 独占认领、现存监听/占用/端口冲突检查。全部留给 W4 原逻辑，不改。 |
| `exp/offline_search/rounds/r17/longseg_1017/runtime_fix.py:66,77,92`、`vfix.py:54` | 替换源段的锚点确保长段补丁确实装上；不能等同源码 SHA 锁拔掉。 |
| `exp/offline_search/rounds/r17/xgpu_1017/concurrent_server.py:158` | DP CUDA graph 自检及原生模型 parity；去掉文件指纹并不放行错误图。 |
| `exp/offline_search/rounds/r17/longseg_off_1022/workflow.py:113`、`filter_failed_knob/formal_a_queue.py:1`、`knob/descent.py:1` | 同卡同步数纯推理参照、旋钮关先判格、只对未达标完整格下降；seed/segment 不可改变参照口径。 |
| `exp/r17_integration/ops/sync_back.py:31,75,90,98`、`library_init.py:54`、`library_batch.py:127` | 忙 journal、归档成员缺失/重复/越界、现有目标数据冲突；这些防错目录和混批。冲突场景的 hash 比较仍留。 |
| `exp/r17_integration/ops/library_alignment.py:54,59,74,76,84,102,116` | 原始 B500 init 与缺失 GR 索引的身份恢复证据；不能凭行顺序捏造 reset 身份。跨模型的实际 seed/成员一致保留。 |
| `exp/r17_integration/ops/library_repair.py:139,148,205`、`library_salvage.py:225,270,303,329,354,368` | 只能补缺局；已返回的失败局不能挑掉；node failure/进度/隔离/完整 capture 证据防选择偏差。 |
| `exp/r17_integration/ops/worker_storage.py:30,53,80,94,104,137,160`、`library_cleanup.py:1` | 删除前确认确属本批、进程已关、证据已持久化；清理时拒绝越界软链不是普通读取软链门。重复字节验算与归属混合的细节列 C。 |
| `exp/r17_integration/ops/incremental_sync.py:65,74,176,206,223,240`、`evidence_transfer.py:170` | 增量回收会删除远端来源；ACK、目标冲突、原 manifest、pruning 归属和已有文件冲突不能误放。 |
| `exp/r17_integration/ops/formal_process.py:1`、`worker_gpu.py:1`、`worker_lifetime.py:35` | PID+start_ticks、owner/lease、只关闭自己的进程，以及区分正常局边界 recycle 和终止错误，防伤到其他批和计数错乱。 |
| `exp/r17_integration/ops/tether_retry.py:25,131` | 不把远端应用错误或响应含糊的 spawn 当作可自动重放；否则可能同计划重复执行。 |
| `exp/r17_integration/ops/plan.py:31`、`debug/capture_bridge.py:342` | Unix socket/共享控制块的真实长度和容器容量，不是经验 worker 上限；更改应连底层格式一起改。 |
| `exp/offline_search/rounds/r15/descent/analyze.py:87`、`r16/prepare.py:224`、`r16/conductor.py:85` | pre-outcome 选择、真实 cell/task 覆盖；32 等在这里是已定义实验 cohort 的大小，不是吞吐保守上限。新 cohort 由建批 owner 参数化。 |
| `exp/offline_search/rounds/r15/pure_corr/build.py:101,123,125,129`、`r15/pure_corr/analyze.py:33,45` | corrector/metric/calibration 指向、实际数组、A pool、原种子和 complete/terminal 决定对照数字。 |
| `exp/offline_search/rounds/r15/qk/impl/audit.py:39,46`、`plans.py:48`、`prepare.py:51`、`exp/offline_search/rounds/r16/conductor_replay.py:151` | served artifact、representation、fit 记录间 identity/sha 字段比对是数据关联；与重新散列源代码不同。 |
| `exp/offline_search/rounds/r15/qk/astra/round2_finish.py:59,63`、`finalize.py:75,79`、`pilot_results.py:258,259` | 历史训练 donor/训练源/选中库与评估证据绑定；还没有可替代的逐局语义证明，不按字段名删掉唯一的无泄漏证据。 |
| `exp/offline_search/rounds/r17/xgpu_1017/a100_closedloop_cpucheck_1018.py:17,24`、`normalize_formal.py:79,105`、`paired_analysis.py:72` | 预登记计划、真实任务科学字段、benchmark/执行时钟和配对模型资产；跨卡比较中的数据配对不能混。 |
| `exp/offline_search/rounds/r17/scheduler/worker.py:1`、`autoscale_1017/reset_zero.py:85,143`、`filter_failed_knob/origins.py:22` | 活跃 job 的节点认领、zero-progress 重置语义以及历史完整源集合；拒绝漏局、重复派发、错误来源。 |
| `exp/offline_search/rounds/r17/survey_predict/audit_existing.py:1`、`horizon/analyze_arms.py:1`、`r16/cost.py:1` | 观察/分析入口中的实际模型、动作/步数、采样身份和数值口径检查按 B 处理；不把诊断 assert 当作生产吞吐门删除。 |

## 不确定 C：协调者需要裁定的问题

C 项当前保留，不宣称已经全拔。原则是把条件拆到真实约束后再删，不用一条“安全起见”概括全部。

| file:line | 要判断的问题 / 建议 |
|---|---|
| `exp/offline_search/rounds/r17/scheduler/defaults.py:20,25`、`filter_failed/timan1_gpu2.py:1`、`exp/r17_integration/ops/timan1_workers.py:23`、`worker_memory.py:61` | 固定 worker caps、保留 RAM/VRAM、每机 eligibility 与共享机器租约混用。协调者确认哪些仅是旧容量经验值；这些可去掉并直接用既有实时 admission，实际异批占用/租约不能一起清空。 |
| `exp/r17_integration/ops/library_staging.py:41,148,161,173,208`（W3）、`library_batch.py:66`、`library_salvage.py:125,154,160` | 511MiB/transfer reserve/capture quota 既有硬编码又有磁盘 staging 协议约束；是否已有动态可用空间+ACK背压替代？确认后拔静态上限，不能造成未回执 shard 被回收。 |
| `exp/r17_integration/ops/worker_lifetime.py:12`、`timan1_workers.py:117`、`exp/offline_search/rounds/r17/filter_failed/worker_resilience.py:153` | 32 局/1800 秒、T1 四局 recycle 是内存泄漏回收策略还是已经过时的经验门？若删除应同时改 recycle marker 与计数解释，不能把正常结束变成 terminal error。 |
| `exp/r17_integration/ops/worker_storage.py:62,66,73,123,173,193`、`incremental_sync.py:129,191,205,227,233`、`library_cleanup.py:44,53,55` | 源码/传输字节封条与“唯一可用副本已落地，允许删除远端”的证明混在一起。哪些已有独立 campaign+member+完整回执，允许只保留那些字段？暂保留删除前的证据链，不以 status=verified 字符串单独准许删除。 |
| `exp/r17_integration/ops/library_repair.py:180,184,195`、`library_salvage.py:227,229,231` | seal_pins 同时含模型/权重/norm/原始 episode 与 code；要明确删除 code 子项后历史 repair/salvage 的兼容格式。自哈希已拔，真实原始证据关联未拔。 |
| `exp/r17_integration/ops/formal_finalize.py:187,448,478,498,514`（W4）、`formal_queue.py:1` | startup/finalization deadlines 哪些只是旧保守值，哪些约束 owned child/lease？建议参数化等待窗口；父子交接、被取消后禁止再派发、证据关闭错误不得默默吞掉。 |
| `exp/r17_integration/ops/cache_control.py:52,67,99,110`、`formal_runtime.py:82,95,105,114,149`、`live_fleets.py:319`、`cache_resume.py:35` | 原代码还存在“非 SHA 条件为真但 pass”的关闭/资源/模型复合检查。本轮已恢复明确的完成局、Q3 和续跑任务检查；其余哪些字段是过时机器限制、哪些仍为数值绑定，请与 W4 launcher/runtime 语义一起裁定，不能整段删或整段恢复。 |
| `exp/offline_search/rounds/r15/qk/impl/runtime.py:66`、`qk/astra/round2_report.py:16`、`round2_finish.py:23`、`pilot_results.py:248,308` | 旧 QK 路径要求 guard_only，且 GATES/选择/训练记录混在同一冻结清单。它不是新 seeds 常驻路径；若还要重跑，应迁移到 Q3 后再去掉旧门门槛；不借本轮打开已退役服务端旧门。 |
| `exp/offline_search/rounds/r17/xgpu_1017/a100_closedloop_run_1018.py:165`、`a100_openpi_handoff.py:37`、`temporary_h100_stage.py:150` | 专用空卡实验（跨卡数值回放）、共用环境交还、rsync hosts allow 是实验隔离/共享资源或网络边界，不是普通只准某机器跑的门。协调者若取消这些独占实验前提，应一起更新比较口径与接管流程。 |

## 协调者要跑的命令（未在本机执行）

以下命令按顺序。时间/内存为静态估计，真实模型规模未在本机测量。CPU 验证在 timan1 的同步 checkout/环境执行，或等 weilandserver 明确交还；不要把已有 fa_* / ffk1015_* run 当临时目录。

### 1. 编译与 shell 语法

CPU；timan1 或交还后的 weilandserver；约 1–5 秒、<200MiB。在合并后的仓库根目录执行（以最终 diff 的 Python 文件为输入）：

```bash
git diff --name-only -- '*.py' | xargs -r python -m py_compile
bash -n exp/r17_integration/ops/pod/setup.sh
```

预期：退出 0。A2 本轮 72 个 Python 文件已逐文件 py_compile；shell 的 bash -n 未本地执行。

### 2. 不读实验数据的正反例

CPU；同上；约 1–10 秒、<300MiB；仅临时目录，含 fsync/os.sync，机器繁忙时可能更久。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src python - <<'PY'
import fcntl
import json
import tempfile
from pathlib import Path
from exp.r17_integration.common.batch_runtime import BatchConfig
from exp.r17_integration.ops.sync_back import snapshot_run, install_snapshot
from exp.r17_integration.ops.formal_resume import accepted_evidence
from exp.r17_integration.ops.formal_runtime import paired_lineage

def rejects(kind, call):
    try:
        call()
    except kind:
        return
    raise AssertionError("expected rejection: " + kind.__name__)

BatchConfig(max_batch=64, batch_wait_ms=100)
for args in (dict(max_batch=0), dict(max_batch=True), dict(batch_wait_ms=-1),
             dict(batch_wait_ms=float("nan"))):
    rejects(ValueError, lambda args=args: BatchConfig(**args))

with tempfile.TemporaryDirectory(prefix="a2_defenses_") as tmp:
    root = Path(tmp)
    src = root / "source"
    src.mkdir()
    (src / "manifest.json").write_text('{"campaign":"a2-only"}')
    (src / "data.json").write_text('{"value":1}')
    (src / "alias.json").symlink_to("data.json")
    archive = root / "snapshot.tar"
    snapshot_run(src, archive)
    dst = root / "destination"
    # 旧 expected SHA 参数仍兼容，但不是准入门。
    report = install_snapshot(archive, dst, expected_sha256="stale-source-pin")
    assert report["files"] == 3
    assert (dst / "alias.json").read_text() == (src / "data.json").read_text()
    (dst / "data.json").write_text('{"value":2}')
    rejects(ValueError, lambda: install_snapshot(archive, dst, expected_sha256="ignored"))
    with (src / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rejects(RuntimeError, lambda: snapshot_run(src, root / "busy.tar"))

    run = root / "run"
    driver = run / "driver"
    driver.mkdir(parents=True)
    task = dict(task_uid="u", bundle_id="b", orig_init_state_idx=0,
                extra=dict(task_name="task", num_steps=1))
    plan = dict(tasks=[task])
    terminal = dict(task_uid="u", attempt=1, accepted=True, status="done",
                    error=None, success=True)
    attempt = dict(terminal, n_steps=1)
    step = dict(task_uid="u", attempt=1, accepted=True, decision_seq=0,
                bundle_id="b", bundle_identity="bundle", task_name="task",
                original_init_idx=0, capture_enabled=False, source="policy",
                num_steps=1, actual_nfe=1, stage_calls=[1, 1, 1])
    def write(name, row):
        (driver / name).write_text(json.dumps(row) + "\n")
    write("journal.jsonl", terminal)
    write("attempts.jsonl", attempt)
    write("per_step.jsonl", step)
    assert set(accepted_evidence(run, plan, "bundle")["accepted"]) == {"u"}
    write("attempts.jsonl", dict(attempt, error="timeout"))
    rejects(ValueError, lambda: accepted_evidence(run, plan, "bundle"))
    write("attempts.jsonl", attempt)
    write("per_step.jsonl", dict(step, bundle_identity="another-batch"))
    rejects(ValueError, lambda: accepted_evidence(run, plan, "bundle"))
    (run / "RECEIPT.json").write_text(json.dumps(
        dict(status="FAIL", all_owned_closed=True)))
    rejects(ValueError, lambda: paired_lineage(
        dict(paired_control=str(run)), run, require_complete=True))
print("A2_CPU_REGRESSION_PASS")
PY
```

预期：放行旧硬上限与普通软链；拒绝无效数值、目标混批、忙 journal、错误终止回执、异批 decision 和不完整对照。

### 3. 首次真实运行前的 recipe 回归

CPU；优先 timan1，需同步新库、fit、Q3 与源码；约 2–20 分钟、4–32GiB RAM（取决于库与模型）。先设置 A2_C5_RUN_NAME / A2_C10_RUN_NAME，必须是 W1/W2 刚为新 seed 生成的独立定义名；先核对其根目录，禁止用旧批次替代。recipe_smoke 会使用自身临时目录并写指定报告，不启动 GPU。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:src python -m exp.offline_search.rounds.r17.longseg_1017.recipe_smoke \
  --name "$A2_C5_RUN_NAME" --name "$A2_C10_RUN_NAME" \
  --decisions 4 --out /tmp/A2_RECIPE_SMOKE.json
```

必须查看报告 status（脚本并非所有 FAIL 都非零退出）：PASS、cuda_initialized=false、每个 seed/size/segment 的 bundle 都出现。需要新建隔离负例并各自确认失败：错 store_digest、错 commit、错 fit.binding、缺 selection.fit_digest、Q3 绑定到另一 fit、RowIndexStore 缺一局的一部分。负例不能改已有 fit/YAML。两个 seed 的选择表和 C5/C10 YAML 文件不得同名覆盖；Cosmos/π0.5 MT50 不生成 DESIGN 禁止的长段变体。

### 4. 有空闲 GPU 后的小批次

GPU；由协调者选择与该参照相同卡型、相同步数 K 的已分配机器；4090 必须等 weilandserver 交还。每个 K/段长配置准备独立的 2 局 smoke，小批次预计 5–20 分钟，RAM 8–40GiB、VRAM 按对应模型现有配置预算，不能据此申请占用当前机器。

W4 先为上一步通过的定义生成新的 prepare-only smoke，并检查 episode plan 真的是 2 局。A2_SMOKE_RUN 设置为那个新目录，然后仅运行它：

```bash
PYTHONPATH=.:src python -m exp.offline_search.rounds.r17.filter_failed_knob.launch \
  --run "$A2_SMOKE_RUN"
```

预期：DP CUDA graph 自检通过；Q3 stall_only 和 C-specific fit 正确；RECEIPT/SUMMARY 完成局数为 2、terminal_errors=0、进程正常关闭；第二次对同目录派发被 LAUNCH 独占认领拒绝。失败时保留证据，不把部分局数当完整格。正式 500/2500 局与两步下降仍由原 owner 队列安排，A2 没有运行。

## 合并前还要对齐

- 上面的 C 项和 W1–W4 待拔项未由 A2 修改，不应在协调者总交付中写成“全仓已无防御”。
- 拔 stat/source 锁后，改文件不会自动让已加载模型/fit/cache 失效；本轮新 seed 使用新路径，修改产物后的进程重启/重新加载由既有建批流程负责。
- r16/k1ref/snapshot 不再 git-show 回填历史源码，构建使用当前源码；首次重跑必须完成上述语义/数值回归，不能把它称作旧 commit 的复现。
- 还保留不少 identity/hash 生成字段供旧消费者读取；需要最后协调者统一其说明，尤其是旧 *_unchanged / verified 字段与逐字节验证的区别。

