# Cache Trace 服务模式：全档常跑 + 按 verdict 选发 + 并发合批 + 全量落盘（实施计划）

> Status: **Verified — G2 R2 APPROVED（2026-09-23 18:12 CDT）→ §6 Verify 8 failed / 6049 passed（8 个均为 HEAD `54699e3` 同命令复现的既有失败，owner 明确豁免）→ 随本线提交入库**（G1 APPROVED 2026-09-21 23:13 CDT，Round 3；plan 经 Post-G1 polish）| Level: **L3**（框架层 orchestrator 流水线抽取 + 完整 twin 组件集 / interceptor / coordinator 核心拆分 + GR00T 并发 stage-3 合批 / 新数据子包 / 取代 `--collect`）| Authority: Execution | 2026-09-23
> 需求来源：owner 2026-09-21 会话口径（§1 逐条对照）。设计来源：12-agent 探索 workflow `wf_4caa4161-1f8`（5 读码 lane → 3 独立设计 → 3 lens 对抗审视 → 1 综合），综合稿经执行方逐条亲验源码后转写；G1 三轮（R1 NEEDS REVISION → r3；R2 NEEDS REVISION → owner 授权审查方直接修订 r4；R3 APPROVED），执行方复核 r4 后做 Post-G1 polish（§7.1 按 server 串行事实收窄）。§2 为 HEAD `54699e3` 的源码事实。
> 术语：**trace 模式** = 本计划新增的服务模式；**变体（variant）** = 同一决策下某一档位的 stage-3 计算；**执行档（executed arm）** = verdict 选中、真正发给 client 的那一档；**build-cache 模式** = trace 模式 + `record_noise_actions=True`，取代老 `--collect`；**twin（影子孪生件）** = 与真实组件同配置构造的第二套**完整** checkpoint 组件集（key_builder / gate / strategy / judge），只承接"每步都跑"的调用、只产生记录、永不进入真实决策流。
> 版本纪律（owner）：**只有一个版本**。不设 v1/v2、不留"后续 opt-in"、不把结构改动推给下一阶段。

---

## 0. 一句话

服务端新增可选 `TraceRuntime`：执行路径完全按 yaml（gate + verdict 决定发出什么），但**每步每个模块都运行**——key 必建、search 必跑、judge 必判（gate skip 步也一样）、stage1/2 必跑、stage3 全推理必跑（= MISS 档，附中间态）、每个可执行 warm 档从 top-1 的 x_t 各续跑一次、FULL_HIT = top-1 chunk——算完按 verdict 选一发出，其余全部落盘。**真实决策流逐调用不变**：`check()` 的流水线抽成一个对"组件集 + 状态"参数化的函数；真实件用 `force_search=False` 保持 HEAD 的语句顺序，twin 用同一函数强制检索；gate 结果和只读 replay_target 被记录，每个成功的 twin 判决只反馈一次。twin 缺 warm 快照时仅在影子流内显式降级并记录原因（§4.1），真实流仍按 HEAD 抛错。真实组件、session memo 和文件输出不接收 twin 调用。额外 stage-3 变体经 `BatchingCoordinator` 跨连接合批；coordinator 核心拆成 jax-free、模型无关，GR00T 同样接入并按 conditioning 形状分桶做 stage-3 跨连接合批。gate/judge/search_strategy/key_builder **零改动**（两 backend 各加逐 field 诊断方法 + facade 透传，owner D2 裁定）；trace 关闭时现有一切路径逐字节不变。

---

## 1. 需求（owner 原话整理，全部硬需求）

| # | 需求 | 落点（§） |
|---|---|---|
| R1 | 执行按 yaml；框架内每模块每步都跑（key/search/judge/stage1/2/3 全推理/每个 warm tier/FULL_HIT）；算完按 verdict 选一发出；不存在"shadow miss"，MISS 就是那次全推理 | §4（共享流水线 + 完整 twin 组件集）、§5 步 5–11 |
| R2 | 在框架层动刀；组件零改动；以后改策略采集代码不变 | §3、§11（`components/**` 零 diff；两 backend 的同一加法式诊断接口 + `CacheStorage` 一个透传 = D2） |
| R3 | 每步落盘：原始图像 ×N 相机、prompt 文本、robot_state；prefix token；降维后 query keys；top-k entry_id + 融合分 + **逐 field 分（top-k×field 矩阵）**；各档动作；gate 决策、verdict、executed；全推理 `noise_action_*`（**默认不收，只在 build-cache 模式打开**） | §7；逐 field 分的定义与覆盖见 §4.4 |
| R4 | warm tiers 以 yaml `warm_tiers` 为准；只对 top-1 算 | §5 步 5–8；可执行档集合定义见 §4.3 |
| R5 | 与 `--concurrent` 整合：变体进 coordinator stage-3 队列跨连接合批；`--non-concurrent` 也能跑；**GR00T 一并做结构改动，同样跨连接合批** | §6、§9 |
| R6 | 取代老 `--collect`；纯推理 yaml 下即建库采集；H5 是老 schema 超集，现有工具直接消费 | §8、§7 |
| R7 | π0.5 与 GR00T N1.5 都做；CP1 与 CP2 配置都做 | §5、§9 |
| R8 | 已接受：hit 步全推理也跑 ⇒ RNG 消耗与普通模式不同，分布同、不逐位可复现 | §5 步 4（默认把差异压到并发交错） |
| R9 | client/conductor 零改动；一集一文件；身份来自 `on_episode_start` 元数据 | §7；写失败终态协议 §7.4 在零改动约束内完成 |
| R10 | **屏蔽**：debug 模式下所有模块都跑，但只让合法的流通过真实组件（owner 对 memo 污染的裁定） | §4.1–4.2 twin 机制 |

---

## 2. 背景事实（执行方在 HEAD 亲验；I=`src/openpi/cache/interceptor.py`，O=`src/openpi/cache/orchestrator.py`，M=`src/openpi/models_pytorch/pi0_pytorch.py`，BC=`src/openpi/serving/batching_coordinator.py`，C=`src/openpi/cache/config.py`，S=`src/openpi/cache/groot/staged.py`，IMB=`src/openpi/cache/backends/in_memory_backend.py`）

- **π0.5 `infer()` 控制流**（I:928–1418）：`obs.pop("__gate_decision__")` → `_input_transform`（TokenizePrompt 会 pop `prompt`，故原始文本只在 `obs`）→ stage1 → `orchestrator.check(CP1, ...)` → **FULL_HIT 早返**（I:1043–1151，不跑 stage2/3、不调 CP3、`buffer_for_write` 无 intermediates）→ stage2 → cp2_only 时 `check(CP2, stage2=, ...)`（I:1199–1231）→ WARM_START `run_stage3_from(...)` / MISS `run_stage3(..., return_intermediates=True)`（并发下 noise 由 interceptor `model.sample_noise((1,H,D), dev)` 采样后进 coordinator，I:1283–1306）→ CP3 → `broadcast_action` → `buffer_for_write(...)` → `clear()`。
- **`CacheOrchestrator.check(self, checkpoint_id, *, request_context=None, **stage_outputs) -> CheckResult`**（O:580–586）。顺序：collect（O:619–620）→ gate（O:622–625）→ **build 无条件**（O:634）→ `_state_history.append`（O:643–647）→ `if not should_search`（O:649）：replay / skip 子分支各自 `strategy.record_query_keys(query_keys)`（O:667–668、O:697–698）后 `_feed_verdict_to_gate(..., searched=False)` 返回；searched 分支：`SearchContext` → `strategy.search(ctx)` → `StoragePayloadView(storage)` + `HistoryView(actions, states=list(self._state_history))`（O:739–743）→ `judge(...)`（O:745–771）→ 三条 return（O:802–881）。**`_feed_verdict_to_gate`（O:483）同时承担 gate `record_verdict` 与 judge `commit_verdict`（O:518–519，`searched and hasattr(judge, "commit_verdict")`）**。生命周期广播 `_broadcast_episode_start`（O:323–375）对 key_builder / strategies / gates / judges 逐一 `on_episode_start` 并 `storage.open_search_session(sid)`；`broadcast_action` 广播到 strategies/gates/judges；`on_episode_end` 广播并 `close_search_session`。`key_builder.collect` 先清自身缓存（`components/key_builder.py:124/:279`）。
- **CRD 判官**：`_propose` 在 `_pending` 非空时 `RuntimeError("new proposal while a proposal is still uncommitted")`（`components/crd_judge.py:229–235`）；唯一清除接口 `commit_verdict`（:292–301）⇒ 任何"只判不提交"的第二次调用必炸。
- **MlpRouterJudge 原生落盘**：`_build_judge` 把 `cfg.dump_dir` 直接透传（C:4504–4516），与 `cfg.dump`（DumpingJudge）无关；`on_episode_end/on_task_end` → `_finalize` → `_write_shard` 按 `run_id/batch_id/_shard_stem(identity)` 原子替换 `.bin/.jsonl` 并追加 manifest（`mlp_router_judge.py:916–951`）。JudgeConfig 中的外部路径字段：`dump`（C:333）、`state_log_dir`/`snapshot_every`（C:391–392，online_rit）、`dump_dir`（C:403，mlp_router）。
- **`required_warm_timesteps(config) -> frozenset[float]`**（C:3423–3479）是**库完整性**需求集合：threshold/failure_aware_gate/always_hit 取 `warm_tiers`；composite 取 composer 两个 t；online_rit 取 `tiers` **加每档的后继快照**（供参考更新，非执行档）；`dispatch_surface`（CRD 同类型）直接 `continue`（其 `start_t_ws` 在 artifact `SurfaceArtifact.start_t_ws`，`surface_judge.py:222`）。
- **逐 field 诊断**：`InMemoryBackend.search_with_diagnostics`（IMB:470–552）只在**单步 weighted_score_sum** 路径产出 `StepRetrievalFeatures{fused_topk, winner_per_field, field_own_margin, fused_margin, n_results}`（`storage_types.py:369–405`）；trajectory depth>1（含 legacy fallback）与 weighted_rrf 返回 `_diag_count_only`（IMB:54，字段为空）。单步 WSS 内部对全部候选保留 `per_field_masked[field] = (contribution, mask)`（IMB:1094–1100）。
- **`PI0Pytorch.run_stage3(self, stage2, *, noise=None, num_steps=10, return_intermediates=False, save_timesteps=(0.7, 0.5, 0.3))`**（M:644–651）；`_stage3_with_intermediates` 无条件 `for st in save_timesteps`（M:796）⇒ **传 `None` 即 `TypeError`**；`run_stage3_from`（M:704）；`sample_noise(self, shape, device, generator=None)`（M:316）。
- **Coordinator**（BC）：模块顶层 import `Stage1Output/Stage2Output/Stage3Output` 与 `stage_io`（BC:54–58）；`stage_io` 顶层 `import jax`（`stage_io.py:29`，`jax.tree.map` 用于 stage-1 obs 堆叠 :72）⇒ GR00T 岛不可 import。`Stage3MissPayload{stage2_out, noise [H,D], num_steps=10}`（BC:96–98）；`submit_to_stage`（BC:459–508）；pull-then-group（BC:667–681）；整批异常广播（BC:700–706）；`_sync_stage_stream` 在 stage-3 forward **之后**、回包之前调用（BC:787–795；:934/:949）；桶循环（BC:868–870）；`_group_stage3_requests` 按 `("miss", None, num_steps)` / `("warm_start", start_t, num_steps)` 分桶（BC:890–908）；`_run_stage3_bucket` MISS 分支不传 `save_timesteps`（BC:919–929）。
- **组件工厂**：`build_per_connection_components(config, shared_storage, *, yaml_id=None, quiet=False, online_registry=None) -> dict`（C:3889–3896）；`_build_judge(cfg, library_stats=None, *, yaml_id=None, schedule=PI05_V1, online_registry=None, library_sha256=None)`（C:4349）。
- **老 `--collect`**：`CollectionPolicy` 挂 4 个 forward hook（serve_policy.py:654–657；`_validate_collect_isolation` :712 拒绝并发）；`data_collector.py`：`InferenceEmbeddings`（:17）、`_METADATA_ATTR_ALLOWLIST`（:56）、`on_episode_end` 写 attrs + `step_%04d/{...}`（:118–189）。引用面：`tests/collect/test_collection_policy.py`、`tests/collect/test_init_noise_and_schedule_stamp.py`、`tests/robocasa365/test_pi05_stack_parity_manual.py`、`tests/exp/test_llm_layer_extract_self_check.py`、`tests/cache/groot/test_import_isolation.py`、`exp/common/build_in_memory_cache_artifact.py`、`exp/zixuan_proposal/build_dual_artifact.py`、`exp/robocasa365/groot_cache_collector.py`。
- **websocket server / client 边界**：每连接 handler 串行：`obs = await websocket.recv()`（`websocket_policy_server.py:631`）→ 处理该帧；`infer` 经 `await asyncio.to_thread(conn_policy.infer, obs)`（:977）在读取下一帧之前完成 ⇒ **同一连接上 `episode_start/episode_end/on_task_end` 不可能与在途 `infer` 重叠**（`ConnectionClosed` 也只在 infer 返回后的 send/recv 处浮出）。`on_episode_end` 同步返回后立即 ACK（:666–680）；普通 lifecycle 异常由外层发送 traceback 并关闭连接（:1030–1038）。LIBERO runner 的 `episode_end` 被 `contextlib.suppress(Exception)` 包住（`examples/libero/episode_runner.py:311–314`）。ACK 不证明异步落盘，下一集失败不能撤销上一集 accepted 记录；本方案不承诺自动重派写盘失败的原集（§7.4）。
- **config**：`_CONFIG_TYPES`（C:916；**`ShadowTeacherConfig`（C:737）未登记** —— HEAD 既有缺陷）；`CacheConfig`（C:757）；`validate_cache_config`（C:2023）；`_routing_errors`（C:3307）。
- **schedule**：`DenoiseSchedule.timesteps` / `snapshot_t(index)` / `timestep_set` / `snapshot_index`（`types.py:109–152`）；`PI05_V1`（:177）；`groot_n15_schedule(N)`（:180）。
- **GR00T**：`GrootStage1Output.input_embeds [B,N,C]`（S:130–144）；`GrootStage2Output{backbone_features [B,N,C], attention_mask [B,N], action_inputs}`（S:167–181）；`GrootStage3Output`（S:214–227）；`_head_inputs(stage2)` 每次重建 BatchFeature（S:669–679）；`run_stage3(self, stage2, *, noise=None)`：`noise is None` ⇒ 上游 `head.get_action` 逐字，否则转写 `denoise_loop(...)`（S:727–760）；`denoise_loop(..., on_step=None)`（S:923–975）；`run_stage3_from`（S:764）；`first_step_updates` 用 `vl.expand(k, ...)` 做同源批内扩展（S:831–884）；`live_schedule()`（S:417）；`run_stage2_llm` 与 `session()` 无 CUDA 同步。**上游动作头**（本机 `gr00t_n15/gr00t/model/action_head/flow_matching_action_head.py`，SHA 与 `S::UPSTREAM_ACTION_HEAD_SHA256` 一致）`vl_self_attention(backbone_features)` **不接 mask**（:263–268），`get_action` :394 与 `S::denoise_step` :915 均不向 DiT 传 conditioning mask ⇒ **异长补零合批会改变动作**；上游 `get_action` 的 randn dtype 是 `process_backbone_output` 之后的 `vl_embs.dtype`（:363–369）。`GrootCacheInterceptor.__init__(self, policy, runner, *, orchestrator=None, timer=None)`（`groot/interceptor.py:116–123`）。GR00T 并发现状：`_InferLockedPolicy` 进程锁串行整个 `infer`（`exp/libero_groot/serve_groot_libero.py:75–99`）。**HEAD 既有缺陷**：`serve_groot_libero.py:858–861` 非并发 yaml 分支 `GrootCacheInterceptor(policy, runner, **components)` ⇒ `TypeError`。
- **有状态判官**：`PercentileRollingCalibration.__call__` 每次 `buf.append`（`percentile_rolling.py:112`），`on_episode_start` 有意不重置（:118–124）。`online_rit` 依赖进程级 `CurveRegistry`（`online_state.py`）与 interceptor 侧续跑反馈。
- **RNG 隔离范式**：`shadow_teacher.stable_seed(task_uid, attempt, decision_idx)`（`shadow_teacher.py:50`）+ `sample_noise(generator=)`。
- **切片函数**：`key_builder._slice_cp1_fields`（`components/key_builder.py:181`）；`slice_groot_cp1_fields`（`groot/key_builder.py:79`）。
- **HEAD 既有测试失败**（`reference_preexisting_test_failures`）：GCS 网络访问一例与 `main.py` 源码锁（2≠3）一例；§12-A-1 要求在 Verify 日志中以 nodeid + 错误签名 + 同 HEAD 对照记录，不以"既有"自动豁免。

---

## 3. 设计总览

### 3.1 切点

```
WS episode_start → Interceptor.on_episode_start(experiment, task, episode_id, episode_name, extra)
                    ├─ [trace] sink.on_episode_start(EpisodeIdentity)   ← 在 I:592 `del experiment, episode_name` 之前；build 模式下 sink 有 sticky 写错误 ⇒ raise（§7.4）
                    └─ orchestrator.on_episode_start(...)  不变（内部同时广播到 twin 组件集，§4.2）
WS infer         → Interceptor.infer(obs, noise=)
                    ├─ self._trace is None → legacy 主体（I:928–1418）一字不改
                    └─ 否则 → _infer_traced(obs, noise)   §5
WS episode_end   → on_episode_end(success) → orchestrator 不变（含 twins）；[trace] sink.on_episode_end(success)（服务端串行保证此时无在途 infer，§2）
连接关闭/换 bundle → on_task_end → [trace] sink.on_task_end()（半集 success=False, trace_terminal=False）
```

每连接：`InferenceInterceptor` / `GrootCacheInterceptor`、`CacheOrchestrator`（真实组件集 + twin 组件集）、`H5TraceSink`。进程级：模型、`BatchingCoordinator`（π0.5 与 GR00T 各一实例，同一核心）、`BackendPool`、`TraceWriter`（按 out_dir 单例）。

### 3.2 trace 关闭时逐字节不变（C1/C2、WA §2.5/§3.1）

1. `infer()` / `_get_action_impl()` 仅首行判空分派；`on_episode_*` / `on_task_end` 各一个 `if self._trace is not None`。
2. `check()` 抽取为 `_check_impl(cset, state, ...)` 后，真实路径 = `_check_impl(self._real, self._real_state, ..., force_search=False)`，语句序列与 HEAD 相同；twins 只在 `trace_twins` 非 None 时构造、广播、运行；`CheckResult.trace` 末尾默认 `None`。
3. coordinator：核心拆分后 π0.5 batcher 的调用序列与 HEAD 相同（`submit_to_stage` = `wait_for(enqueue_to_stage(...))`；MISS 桶内无人设置 `save_timesteps` 时保持原调用形态；π0.5 桶键不变；per-bucket try/except 对单桶批可观测行为不变）。
4. `TraceConfig()` 默认 `enabled=False`；`build_trace_runtime` 返回 `None`。
5. `EpisodeDataCollector.on_episode_end` 改为调用抽出的 `write_step_group` 等函数；固定 timestamp 等身份字段后比较老键值、dtype、shape、压缩与 attrs，不把 HDF5 容器内部布局当成接口。
6. `GrootStagedRunner.run_stage3` 的 `noise is None` 分支零改动；GR00T `--non-concurrent` 路径不经 coordinator；trace 关闭时保留现有整个 infer 外层锁，trace 开启才用 §9 的 stage1/2 锁。
7. backend 新方法 `per_field_scores` 只被 trace 诊断调用；`StepRetrievalFeatures` 不改。
8. C2：`_trace_errors` 要求 `write_policy.type == "never"`；trace 与 twins 从不写 backend。
9. 机器证据：全量 pytest + orchestrator 50 步录制回放（真实件全字段 + 真实 session memo + 真实判官内部状态逐位）+ interceptor/coordinator spy 序列（§12-A）。

---

## 4. Orchestrator（O）：共享流水线 + 完整 twin 组件集

### 4.1 流水线抽取与所有权（R2-B1）

`_ComponentSet` 与 `_CheckState` 定义在 orchestrator；`CheckTrace` 等纯数据类型放 `trace/types.py`（不 import runtime/orchestrator，避免循环 import）。真实件为现有对象的视图，不能复制一套再让旧生命周期写到另一套。

| 归属 | 字段与约束 |
|---|---|
| `_ComponentSet` | key_builder、gates、judges、strategies、**独立 storage facade**、timer、library_stats/artifact_meta 的只读引用；各自按实际 judge 实例构造 `judge_wants_query_keys/step_features`；各自 state-history anchor |
| `_CheckState` | step_counter、miss_by_checkpoint、state_history、action_history、last_judge_commit、active search sessions、当前 episode/task 身份；容器类型/重置时机与 HEAD 相同（history 为 list）。真实旧属性读写必须通过兼容视图访问同一状态，无双份真值 |
| 真实 orchestrator 专属 | episode 写入缓冲、write_policy、offline_writers；twin 永不调用库写入/OfflineWriter。已由 `write_policy=never` 禁写，不让 twin 生命周期进入真实写回路径 |

```python
# 接口示意；_check_impl 返回 CheckResult，可接收只记录数据的 observer。
def check(self, checkpoint_id, *, request_context=None, trace=False, fetch_top1=False, **stage_outputs):
    observed = CheckObserver() if trace else None
    result = self._check_impl(self._real, self._real_state, checkpoint_id,
                             request_context, stage_outputs, force_search=False,
                             observer=observed)
    if trace and self._twins is not None:
        diagnostic = self.trace_check(checkpoint_id, request_context=request_context,
                                      fetch_top1=fetch_top1, **stage_outputs)
        result = dataclasses.replace(result, trace=diagnostic.with_real(observed))
    return result

def trace_check(self, checkpoint_id, *, request_context=None, fetch_top1=False, **stage_outputs):
    # 只访问 twin；CP3 的 FULL_HIT 步必须走此入口。
    return self._run_twin_check(checkpoint_id, request_context, stage_outputs,
                                fetch_top1=fetch_top1)
```

抽取 HEAD O:600–881，同时参数化其传递依赖：`_feed_verdict_to_gate`、`_with_judge_diag`、payload view、history view、artifact/schedule 查询、计时器与 session 生命周期。`trace=False` 时不构造 observer，不增加真实 hook 调用；observer 在已有语句旁读取局部结果，不再调用 gate/judge 或读会消费状态的诊断方法。两套 key_builder 的 model attachment 都在各自首次 build 前完成（I:417–423）；共享模型只读，临时字典/缓存独立。

强制检索的精确定义：twin collect → gate → build；记录 gate 返回值，并在其 skip 时只读调用 `replay_target()`（HEAD FollowWinner 是纯读）。随后跳过整个早返 replay/skip 分支，进入正常 searched 分支；**不执行 replay 的 fetch、record_query_keys、计数或 record_verdict**。strategy.search 自己记录 query；twin 最终 verdict 只调用一次 gate.record_verdict(searched=True)，有 commit 的 judge 只提交一次。真实流的 replay/skip 原样保留。CP3 仅增加 twin 调用次数，不声称两套 history 或计数始终相等。

**缺快照与异常**：真实 WARM 校验失败仍 raise（HEAD 无自动降级）。仅 twin 流在既有 payload/schedule 校验位置捕获明确的缺快照/非法档位 `KeyError/ValueError`，保留 `proposed_verdict` 与 `validation_error`，将**影子最终 verdict**改为 MISS 后再推进 twin 计数并反馈/commit 一次；CRD 已支持 WARM→MISS 的 `LEGAL_DOWNGRADES`，不伪报 WARM 执行成功。这使 D9 的无效 tier 可记录并跳过，CRD `_pending` 正常清除。judge/search/I/O 等其它异常不吞掉，不读取或直接篡改 `_pending`；终止该 trace episode、标为不完整、走正常 lifecycle 清理，不能继续对未提交实例发起下一步。该异常策略是 twin 的明确差异，真实分支不使用。

### 4.2 twin 装配、隔离与生命周期

`build_trace_twins(config, shared_storage, *, real_components, yaml_id, model) -> (cset, state)`：用配置副本调用现有 `build_per_connection_components`，**保留它返回的独立 facade/timer**。状态、session id 与模型相关 attachment 全部归该 cset。strip DumpingJudge 前读取真实 judge 的 `min_required_top_k`，将副本对应 strategy.top_k 提升到相同的有效下限，防止移除观测 wrapper 后缩小搜索宽度。签名探测按实际 twin judge 重新执行。

`_TWIN_STRIP`：`JudgeConfig.dump/dump_dir/state_log_dir/snapshot_every=None`；`TimerConfig.output_csv_dir=None` 且 twin timer disabled；关闭 collection/shadow_teacher 的外部写入。只读 artifact/weights/init_state 路径保留。递归扫描参与装配的 dataclass 字段名 `dir|path|file|dump|log`，每一命中必须进入“剥离”或有理由的“只读”表；不可只扫描 JudgeConfig 的顶层。`snapshot_every=None` 在现工厂会回落为 200，隔离依据是无持久化目录，而非频率为零。

online_rit 使用**进程级独立 twin CurveRegistry**（与真实 registry 相同的跨连接共享粒度，按 (yaml_id, library_sha256) 分区，并按现 registry 契约拒绝同 key 的不同 fingerprint；无 log_dir、require_persistence=False），只创建进程级单例，随服务进程结束释放；不为每连接建立被 atexit 强引用的 registry。真实与 twin registry 的条目/更新互不触达。连接上的 judge/pending state 仍独立；配置 fm0/fm1 不被 trace 改写。

生命周期 `on_task_begin/on_episode_start/broadcast_action/on_episode_end/on_task_end/clear` 分别作用于两套 cset/state，broadcast 的是**执行动作**，各 history 容器独立；`open/close_search_session` 使用各自 facade 和 sid。CP1 FULL_HIT 时真实 CP3 不调，单独 `trace_check(CP3, stage3=executed, ...)`；真实本来要走 CP3 时只调一次 `check(CP3, trace=True, ...)`，不得再补第二次 twin。CP2-only 不调用 CP1/CP3。未配置的 CP 不凭空造 handler。

所有现有合法 gate/judge/strategy 组合都支持 trace，无判官白名单；既有配置兼容性校验仍生效。payloadless student router 与外置 hit_executor 已由 routing/构造互斥拒绝；不能把它当成有缓存 payload 的 FULL_HIT。

### 4.3 可执行 warm 档集合

`executable_warm_tiers(config, *, assembled_judges) -> tuple[float, ...]` 与 `required_warm_timesteps` 分离：后者检查库快照完整性，不代表执行档。

- threshold/failure_aware_gate/always_hit：warm_tiers[*].start_t；always_warm_start：start_t。
- composite：composer.warm_start_t / warm_fallback_start_t 中非 None 者。
- dispatch_surface/CRD：读取已装配 judge（透明 wrapper 解包后）的 `artifact.start_t_ws`，**不是 storage.artifact_meta**。
- online_rit：已装配 continuation_spec.tiers 的 start_t，不加入后继参考快照。
- mlp_router/risk_router：无 warm 档；无 orchestrator 时为空。

按 live schedule 的 snapshot_index 验证、去重和排序（π0.5 DESC、GR00T ASC）；保存浮点值只用于显示，变体身份使用 `(entry_id, snapshot_index, schedule_id)`，防止格式化档位键冲突。任何真实 WARM_START 若未命中已算的 top-1 × tiers，另算 `warm_exec`（winner 不同或档位未枚举），真实执行档永远可选。真实 payload 必须通过原校验，不能用 D9 跳过真实执行。

### 4.4 逐 field 分：语义与 backend 范围（R2-B3）

两 backend 各新增只读 `per_field_scores(query_keys, ids, *, field_similarity, score_normalization, fusion_weights, fusion_method)`，`CacheStorage` 一处透明透传；不改已有 search 与 `StepRetrievalFeatures`。返回 `PerFieldTrace{fields, scores[k,F], present[k,F], kind_by_field, current_step_wss[k] | None}`，输入 ids 顺序即输出行序，F 按配置启用且存在于 query 的字段固定排列；空 top-k 返回 `[0,F]`，缺失字段值为 0 且 present=False，未知 id 抛错。调用不带 search_session_id、不写查询历史/session memo。

- **InMemory**：按现有 `_batch_field_scores` 与 `build_field_normalizers` 的公式/缺失 mask 复算。WSS 分数为 Layer-1 归一化后、乘 mask、未乘融合权重的分数；L2 normalizer 输入是现实现的正距离，不能先取负。`current_step_wss = Σ weight_f * score_f`。它是当前 query 的 WSS 复算，不能当作轨迹分。RRF 路径存原始 cosine 或 −L2（越大越相似），kind 标明 `raw_cosine/raw_neg_l2`，**不生成 current_step_wss**；不能把候选子集重排称为原 RRF 排名分。legacy fallback 保留其原 cosine 定义。
- **Qdrant**：同一 capability 逐 field/chunk 用与已有 search 相同的 `query_points(query=..., using=...)`，以候选 ids 的 HasIdCondition 过滤，limit=len(ids)，不返回 payload/vectors，返回每个候选的原生单向量分。采用现有 `_parse_point_id` 和 `_field_to_chunks`；分块 field 取 chunk 分的等权平均，kind=`qdrant_native_chunk_mean`，非分块为 `qdrant_native`（记录 collection 的距离类型）。缺 chunk 使该 field 的 present=False；未知 point 先只读 retrieve 检查并抛错。此值是原生逐 field 相似度诊断，**不是 RRF 分解**；额外记录 chunk 数，current_step_wss=None。不改变 Qdrant 既有检索与融合规则。

twin search 后（judge 前即可）复算最终候选的矩阵；原始 `results[j].score` 单独写 `twin_topk_scores`，trajectory 时亦作 `twin_topk_chain_scores`。`twin_topk_current_step_fused` 仅 WSS 有值并标记 `current_step_wss`，RRF/Qdrant 该 dataset 缺省且写明确 not_applicable 原因；逐 field 矩阵不能静默缺省。dual/dynamic_depth/text_ivf 复用各 checkpoint 配置的当前 query 度量；该矩阵不声称重现链中每层或正负库检索的中间分数。已有 depth1 WSS winner diagnostics 保留用于对照。

### 4.5 CheckTrace 与计算依赖

`CheckResult.trace` 末尾追加默认 None。`CheckTrace` 包括：真实/twin gate 决定（真实未调用时 None）、真实 results/judge_result（skip 时 None）、twin results/step_features/retrieval_signals、§4.4 PerFieldTrace、twin proposed/effective verdict 与 validation_error、只读 replay_target、top1_entry_id/top1_payload、checkpoint。真实 observer 只复制已取得的局部值；不为诊断再调用任何真实组件。

`fetch_top1=False` 只禁止额外的 trace top-1 fetch，不影响真实 fetch 或 twin judge 的 view.get。完整 trace 需要 FULL_HIT 变体，故有 configured primary checkpoint 时 **总是 fetch_top1=True**；record_search 仅控制检索诊断落盘，不控制动作计算。无库/空结果时 top1 为 None，`full_hit` 记 unavailable；CP3 诊断不做动作变体，可传 False。PayloadView 在单次 check 内复用，避免已读候选重复加载。

---

## 5. π0.5 `InferenceInterceptor._infer_traced(obs, *, noise)`

`__init__(..., trace=None)` 末尾追加；trace 开启时与 hit_executor/miss_executor、shadow_teacher、export_collect_meta、stage2/stage3 meta device 互斥。trace=None 仍执行原 infer 主体。主 checkpoint 来自已校验配置（CP1 或 CP2-only），无 orchestrator 表示建库纯推理。

1. pop request_context；变换前保存原始 images/prompt/state；对浅拷贝调用 input_transform 一次。批维、Observation、tokenized_prompt、input_images、client start_noise 与 legacy 相同。
2. stage1 必跑；π0.5 stage1/2 沿用 legacy 对 coordinator/本地 stage_fn 的选择，不能在 concurrent 下绕过既有调度直接并发调用共享模型。CP1：`check(CP1, trace=True, fetch_top1=True, ...)`，不早返；无 orchestrator 时 `cp=None`。
3. relocation/meta guard/stage2 按 legacy；CP2-only 用 run_stage2_capture，然后 `check(CP2, trace=True, fetch_top1=True, ...)`，不调用 CP1/CP3。
4. `effective_hit = cp.hit_type if cp else MISS`。client 给 noise 就用原值；否则 verdict_aware：MISS 用正常全局 sample_noise，FULL/WARM 用同 identity/attempt/step 的私有 generator；global 模式全用全局 RNG。
5. `trace = cp.trace if cp else None; top1 = trace.top1_payload if trace else None`。所有 top1、entry_id、query_keys 访问都显式可空。top1 无则每 tier=no_top1；有则按 schedule 校验各 tier，只有 snapshot 校验的已知失败记 no_snapshot 并跳过（D9）。FULL_HIT 变体在 top1 有 action_chunk 时记录，否则 unavailable。
6. 变体 full 永远存在：`Stage3MissPayload(stage2, z.squeeze(0), num_steps=10, save_timesteps=plan.save_timesteps)`；每个有效 top-1 tier 加 warm；真实 WARM 未命中该 `(entry,tier,schedule)` 时加 warm_exec。full_hit 是 top1.action_chunk，无计算；它与真实 FULL_HIT 的 winner 可以不同，落盘不得混淆。π0.5 trace 默认捕获模型原三档，build 捕获全 schedule。
7. 校验本连接 payload：stage2 结构/device 与 adapter 契约一致；noise/start_x 为 `[H,D]`、有限值，π0.5 为 float32；GROOT 噪声 dtype 另见 §9。验证后在生产 stream 记录 ready events，coordinator 才可读取。直调 full 在 save_timesteps=None 时**不传该 kwarg**，warm 用原 run_stage3_from；共享队列用 submit_many，回填统一 variant 结果。
8. 选发：MISS（含 cp=None）→full；WARM→匹配的 warm 或 warm_exec；FULL→**真实 cp.payload.action_chunk**，不使用 twin top1 代替真实 winner。三者统一为 legacy 的 batched Stage3Output 形态后输出变换。
9. 簿记按各 verdict 的 legacy 分支逐参数对齐：

   | verdict | 真实 CP3（CP1 配置且 CP3 已配置） | twin CP3 | broadcast / buffer |
   |---|---|---|---|
   | FULL_HIT | 不调用 | trace_check 一次 | 真实 cached_action；buffer 无 intermediates |
   | WARM_START | check(trace=True) 一次 | 随该 check 一次 | executed；buffer 无 intermediates |
   | MISS | check(trace=True) 一次 | 随该 check 一次 | executed；buffer 只传 π0.5 默认三档、10步、PI05_V1 stamp |

   无 orchestrator 则全部 cache 簿记跳过；CP2-only 无 CP3。CP3 实参是执行档，不是必算的 full 变体。所有出口用 finally clear（真实和 twin 各一次），异常使本集不完整，不能写成功 Close。
10. 组装 StepTrace：legacy.clean_action 为 full，action_executed 为实际发送动作；其余变体、可用性与错误原因各自保留。sink.record_step 只阻塞当前 infer 线程；sink 的 begin/finish_step 生命周期覆盖整个 infer，确保取消/断连时 Step→Close 顺序（§7.1）。
11. output_transform、`__hit_meta__` 的 legacy 字段按实际 cp 与 cp2_library_sha 等现有逻辑生成；无库为 MISS 元数据；只额外加 trace 子键，不丢原 provenance。私有 helpers 负责 variants、run/select、build_step_trace，不改变 trace-off 函数体。

“真实逐位不变”的证明对象是**同一录制 observation/action/stage-output 序列下的真实组件调用/状态**；新 GPU 批量 kernel 的误差另走数值门，不能承诺 trace 与普通模式的闭环轨迹逐位一致。R8 接受的 RNG/并发差异仍须记录。

---

## 6. Coordinator：jax-free 核心 + 多变体提交

### 6.1 模型边界与结果契约

新 `serving/batching_core.py` 搬入 StageRequest、队列、worker、pull-then-group、桶循环、指标、enqueue/wait/submit_many；不 import jax/openpi.models/stage_io。原 `batching_coordinator.py` 保留公开 BatchingCoordinator，用 Pi05StageBatcher 承接原逻辑与 stage_io。

`Stage3MissPayload{stage2_out, noise[H,D], num_steps, save_timesteps=None, ready_events=()}`；`Stage3WarmStartPayload{stage2_out, start_x[H,D], start_t, num_steps, capture_first_step=False, ready_events=()}`。核心结果 `Stage3VariantOutput{action[H,D], intermediates:dict|None, first_step_input[H,D]|None, first_step_x[H,D]|None}`，interceptor adapter 转回既有 Pi05/Groot 类型；老 submit_to_stage 调用仍收到原返回类型，不要求 legacy 调用方适配。

```python
class StageBatcher(Protocol):
    def bucket_key(self, payload) -> Hashable: ...
    def default_save_timesteps(self, payload) -> tuple[float, ...]: ...
    def order_timesteps(self, values, payload) -> tuple[float, ...]: ...
    def wait_ready(self, payloads) -> None: ...
    def run_stage1_batch(self, payloads) -> list: ...
    def run_stage2_batch(self, payloads, *, capture: bool) -> list: ...
    def stack_stage2(self, outs): ...
    def run_stage3_miss(self, stage2, noise, *, num_steps, save_timesteps) -> list[Stage3VariantOutput]: ...
    def run_stage3_warm(self, stage2, start_x, *, start_t, num_steps, capture_first_step) -> list[Stage3VariantOutput]: ...
    def sync(self) -> None: ...
```

save_timesteps=None 表示 adapter 默认，显式空 tuple 表示空请求，不能用 `requested or default` 混淆两者。同桶每行 effective_set 为显式请求或 adapter.default；全 None 保留 adapter 的原调用形态（Pi05 不传 kwarg）；否则只取 effective_set 并集，经 adapter 按 schedule 排序，回填按每行 effective_set 过滤。Pi05 默认 `(0.7,0.5,0.3)`（inspect.signature 守卫），Groot 默认空 tuple（legacy 无 intermediates）；build 使用各自完整 schedule，核心**不含任何模型时间常量**。warm 的 capture_first_step 按桶 OR，回填未请求者的辅助字段 None。

Pi05 桶键保留现有 mode/start_t/num_steps；旧请求无 ready event 则保持原路径，新 trace 请求必须带事件。Groot 桶键见下。每桶异常清理完成后给该桶回复错误，其他桶继续；不可在异步 kernel 尚未完成时 set reply/release 输入。submit_many 先全 enqueue 再按序 wait；失败后仍收割已入队请求；总 timeout 用同一 deadline，超时交由 worker 完成清理，不在调用线程释放最后的引用。未 enqueue 的剩余请求清楚标记 canceled，不能留下无人收割的 Future。

### 6.2 CUDA 就绪与所有权（π0.5 / GR00T 共用）

producer 在 stage2、noise、start_x 所有写入/搬运后，记录**实际生产 stream** 的 event；有多个独立 stream 时携带全部 events，不能假设当前/default stream 覆盖全部输入。worker 在读取/stack 前 wait_event；CPU 路径 events 为空。不能用 worker 的 torch.stack/to 代替跨 stream 同步。

StageRequest 持有输入、events、模型/执行上下文直至 worker 正常同步或异常 finally 同步完成；wait_for 超时只结束等待，不能释放 worker 所有权。异常后先做 stream 清理同步（或明确的 record_stream 等价生命周期管理），再发布失败。CUDA context 失效等无法确认完成的致命错误停止 worker 并报告服务失败，不继续复用缓冲。正常回包前 sync 保证输出可读。bundle unload / disconnect / drain 先停新请求，再等待该连接已入队请求终态，引用最后释放。

### 6.3 GR00T batcher

`cache/groot/batcher.py` 只提供 stage3，stage1/2 留在连接线程的 model 锁内。桶键含 mode/start_t/num_steps、schedule_id、实际 model 执行域、conditioning N/C、state 的完整非 batch shape、各输入 dtype/device；如实现按 embodiment 分桶，id 必须在 producer 准备为 host immutable 值，不能对 GPU tensor 做哈希或在 wait_ready 前 `.item()`。不同 bundle 若共享同一个模型/执行域且 shape 相同可合批。

仅同形张量沿 B 维 cat（features/mask/state/embodiment_id），不 padding。B=1 走同一 head loop 的独立等价门；cat 本身不是 view，不能以“cat 等于 view”证明数值相等。异长分桶；同任务同形连接仍合批，异任务同形也允许合批。

MISS 在 worker 自开 runner.session，以 denoise_loop(on_step=capture) 按 live_schedule.snapshot_t(i) 捕获 x_in；warm 调 run_stage3_from(..., schedule=live, capture_first_step=...)，拆分 action 与第一步辅助输出。π0.5 B=1/Groot B=1 对直调逐位；B>1 按各现有数值门容差，需包含不同 state/prompt 的同形输入。未设 OPENPI_STAGE3_BUCKET_FIRST=1 打 warning，不自动改最大批量配置。

---

## 7. 数据通路

### 7.1 组件、背压与事件顺序

TraceSink → 每连接 H5TraceSink → 按 out_dir 单例的 TraceWriter；一条写线程独占 h5py 与 sidecar handle。单 FIFO Queue，Step 使用 BoundedSemaphore(queue_steps)，Open/Close 不占 Step permit、只做短临界区登记和非阻塞入队。Step 持有 permit 直到写完或失败清理，在 finally release；writer 已死亡时 acquire/wait 必须检查健康状态并抛错，不能永久挂住。

sink 每个 episode 有独立 token 与状态 Open→Closing→Closed/Failed。infer 开始登记 begin_step，finally finish_step；on_episode_end/on_task_end 设置 close_requested（后者 terminal=False）并入队 Close。同连接的 Step→Close 顺序由服务端串行处理保证（§2：同连接 lifecycle ctrl 不与在途 infer 重叠），sink 只以**断言**守卫该不变量（Close 时 `in_flight_steps == 0`，否则 raise 并标 Failed），不实现延后回调或 Busy 状态机；orchestrator 的 episode/task-end 回调时机与 legacy 相同。前集 writer 仍在 Closing 时，可用新 token 开新集，不能改写旧任务身份。未知/重复 token、Close 后写 Step、重复 Close 均有明确定义并受测试约束；跨线程 producer 的提交顺序不靠“同一 FIFO”推导，靠 permit 与 token 状态机。

CPU 数据是不可变快照；禁止把仍会复用/修改的 numpy view 放队列。先取得 Step permit 再做归属于队列的 CPU 组装；在途模型输出与每连接当前 step 仍另占内存。内存账为 queue_steps × 单步字节 + 连接数 × 在途快照/模型输出 + 打开文件开销，不承诺总 RAM 仅 queue_steps × 4.4 MiB。

数据类型在 jax-free `trace/types.py`：

- `TracePlan`：model/live schedule/primary checkpoint（无库时 None）/warm_tiers；record_noise_actions、save_timesteps（build=全 schedule，否则 None）；record_prefix/raw/model/query/search/tokenized 开关；raw_image_keys、rng_isolation；yaml/bundle/connection/library 身份。fetch_top1=有已配置的 primary checkpoint，与 record_search 无关。
- `EpisodeIdentity`：experiment/task/episode_id/episode_name/extra_metadata/plan；task_uid、attempt 来自 client metadata，手工运行可回退，但 accepted-set 审计不允许以回退身份冒充 conductor 任务。
- `SearchTrace`：真实可空搜索/判决，twin 搜索/原始融合分、PerFieldTrace（含 present 与 kind）、chain_scores、可选 current_step_wss、step_features/retrieval_signals、twin proposed/effective verdict/error/replay_target、top1 id。
- `StepTrace`：legacy InferenceEmbeddings；raw/model images、prompt/raw_state/tokenized_prompt/query_keys；主 SearchTrace、CP3 trace；full_hit/warm/warm_exec/full_inference/executed 动作；executed_arm/verdict/tier_status/error_proxies/timing；真实与 twin continuation 诊断分别存放。
- `TraceRuntime`：plan/sink、twin 组件集引用（由 orchestrator 持有状态）；提供私有 noise_generator。runtime 在工厂内按需 import orchestrator，不从 trace/types 反向导入。

sink 接口：on_episode_start、begin_step/finish_step、record_step、on_episode_end、on_task_end、close；最后三者请求结束，不冒充已经落盘成功。writer 接口：get/open_episode/write_step/close_episode、drain(timeout)->DrainReport、errors()->list[TraceWriteError]；DrainReport 含未完成 episode、错误、超时与线程状态。pi05/groot trace adapter 负责原始观测、prefix 切片、中间态映射与 StepTrace 组装。

### 7.2 H5 schema（一集一文件；老 collect 的严格超集）

路径：`<out_dir>/<experiment>/<episode_name>.h5`（沿用 `resolve_episode_path` 逃逸守卫）；无 episode_name ⇒ `episode_{id:04d}_{ts}_p{pid}.h5`（`pid_suffix` 只在 trace 写器打开）。写 `.h5.tmp` → 成功 Close 后 rename `.h5`；**失败永不 rename**（§7.4）。sidecar 同目录 `<stem>.trace.jsonl`。（D10）

文件级 attrs：老的 `experiment_name, task, episode_id, num_steps, timestamp, success` + allowlist 身份 + `prompt` + `denoise_schedule_id / denoising_num_steps`（**总是写**，D4）；新增（全部 `trace_` 前缀）：`trace_schema_version=1, trace_noise_actions_recorded, trace_denoise_schedule_id, trace_denoising_num_steps, trace_model, trace_checkpoint, trace_concurrent, trace_rng_isolation, trace_warm_tiers(JSON), trace_bundle_id, trace_yaml_id, trace_yaml_sha256, trace_connection_id, trace_library_sha256, trace_key_builder_type, trace_terminal, trace_extra_metadata_json, trace_write_errors, trace_closed_ok`（最终成功文件中为 True；临时文件内的预备 True 不算凭证）。

每步组 `step_%04d/`（从 0 连续；顶层不新增其它 `step_` 前缀组）：

| 键 | dtype/形状 | π0.5 来源 | GR00T 来源 |
|---|---|---|---|
| `vision_{i}` (lzf) | fp16；π0.5 [256,2048]，GR00T 按实际切片维度 | `_slice_cp1_fields(prefix_embs, state, None)` | `slice_groot_cp1_fields(..., enabled=None, vision_fields=按 builder 相机数)` |
| `prompt_emb` (lzf) | fp16 [n_tok,C]，C 按模型 | prefix 非图像段 | 非图像 token |
| `robot_state` | f32 [32]/[8]/[20] | `Stage1Output.state[0]` | `state[0,-1][state_mask[0,-1]]` |
| `clean_action` | f32 [H,D] | **全推理** `full.action_chunk[0]` | `full.action_pred[0]` |
| `noise_action_0` | f32 [H,D] | 仅 `record_noise_actions`：z | `on_step` 第 0 步 x_in |
| `noise_action_1..N-1` | f32 [H,D] | 仅 `record_noise_actions`：`full.intermediates[PI05_V1.snapshot_t(i)]` | `on_step` 第 i 步 x_in |
| `input_images/{slot}` (lzf) | uint8 [224,224,3] | `extract_valid_images(inputs)` | 不写 |

`noise_action_*` 全写 0..N-1 或一个不写；`trace/` 内禁止匹配 `^noise_action_\d+$`。

新子组 `step_%04d/trace/`：attrs `executed_arm, hit_type, start_t, winner_id, cp1_score, score, searched(真实 gate), gate_twin_should_search, top1_entry_id, checkpoint, hit_override, tier_status_json, verdict_json(真实), twin_verdict_json, twin_replay_target, factor_outputs_json, router_outputs_json, twin_retrieval_signals_json, cp3_twin_json, error_proxies_json, timing_json, prompt, twin_topk_per_field_kind（逐 field JSON）, twin_validation_error, twin_proposed_verdict, real_continuation_json, twin_continuation_json`（任一 JSON attr > 60000 B ⇒ 改写为 vlen-str dataset `trace/json/<name>`，attr 置 `"@dataset"`）；datasets `trace/raw_images/<percent-encoded-wire-key>` uint8 lzf、`trace/raw_state` f32、`trace/model_images/<slot>`（opt-in）、`trace/tokenized_prompt` int64、`trace/query_keys/<field>` f32 [d]、`trace/search/real_topk_ids` vlen str（searched 步）、`trace/search/real_topk_scores` f32、`trace/search/twin_topk_ids` vlen str [k]、`trace/search/twin_topk_scores` f32 [k]、`trace/search/twin_topk_per_field/<field>` f32 [k]、`trace/search/twin_topk_per_field_present/<field>` bool [k]、`trace/search/twin_topk_chain_scores` f32 [k]（depth>1）、`trace/search/twin_topk_current_step_fused` f32 [k]（仅 WSS，kind=current_step_wss；RRF/Qdrant 不写并注明 not_applicable）、`trace/search/twin_winner_per_field/<field>` f32、`trace/search/twin_field_own_margin/<field>` f32、`trace/actions/{full_hit, warm_<snapshot_index>, warm_exec, full_inference, executed}` f32 [H,D]。

raw-image key 使用可逆 percent encoding（包括 `%` 与 `/`），同时保存 wire-key 映射；warm dataset 对应的 snapshot_index→start_t/schedule/entry_id 映射随该步记录；缺 full_hit/warm dataset 必须有 availability/tier_status 原因。

字节账（π0.5 估算，GR00T 以实际形状另计）：≈ **4.4 MiB/步**（vision 3.0 MiB + prompt_emb 0.78 + raw_images 0.29 + input_images 0.29 + 其余 < 60 KiB）；诊断跑 `record_prefix_tokens=false` ≈ 0.6 MiB/步（非超集，build 强制 True）。

### 7.3 消费者与准入边界

老键由同一 write_step_group 写出，InferenceEmbeddings 不变；legacy 读端不遍历 trace/。`h5_intermediates.episode_schedule/read_step_intermediates` 保持无 noise 时返回 None 的现契约；trace build 打开时必须有全套 0..N-1，不可部分缺失。

两审计器 `verify_collection_artifacts.py` / `verify_shadow_h5.py` 按文件 schema 分支：**无 trace_schema_version 完整保留旧校验**；version=1 要求 closed_ok=True、terminal=True、write_errors=0 及相关身份/步序一致，noise_actions_recorded 决定是否要求全部 noise_action；未知版本拒绝。诊断文件 noise 关闭可用于诊断，但不能冒充有 warm 快照的建库产物；建库消费者明确拒绝需要却未记录快照的输入。

conductor 采集用 run-plan + journal 选中的 **accepted 且 success、无执行错误的 task_uid/attempt/episode_name** 集合逐项对账，验证文件内部身份与所选 attempt 一致；成功任务的任何对应 `.h5` 缺失、未终态、失败或重复映射都令整批不准入。失败/未 accepted 的历史 attempt 的 `.tmp/.failed` 单独报告并排除，不能让有效的后续 retry 永久失效。若用户声称“全 run-plan 都完成”，还必须验证 run-plan 应有任务全部得到 accepted 成功结果，不能只审 journal 里碰巧存在的子集。旧手工单文件检查维持旧模式，并明确它不是 run-plan 完整性验收。

**建库 SOP 必经准入审计**，不能把 RPC ACK 或 conductor 的 accepted 数当作写盘成功。写器结束后审计才有终局结论；建库脚本只接受本次审计列出的确定文件清单（通过既有路径参数选择或新增服务端/离线清单参数，不改 client/conductor）。旧库构建输入仍按旧 schema 接受。离线产物只支持 IR 与开环动作误差代理，不能据此推断 SR。

### 7.4 写失败终态协议（client/conductor 零改动）

1. **路径预留**：writer 对规范化后的目标路径做唯一预留；writer 线程以目标路径对应的 O_CREAT|O_EXCL reservation 文件实施跨进程互斥（覆盖 replicas）；打开前发现既有最终文件、reservation 或其他 active token 占用则拒绝，绝不 os.replace 覆盖别集。重试应有独立 attempt/episode_name；同名重派需运营侧明确清理/隔离旧失败产物后重跑，不能自动删旧成功文件。临时文件带 episode token，路径逃逸检查仍沿用 resolve_episode_path。reservation 含 token/pid/episode 身份；只允许其 owner 释放，旧 crash 遗留需要明确恢复处理，不凭 pid 猜测自动夺锁。成功发布后的 reservation 清理失败仅报 housekeeping 错误，不撤销已完成的 Close；下次同名写入仍会被拒绝。
2. **提交顺序**：Open 建 `.h5.tmp` 与 `.trace.jsonl.tmp`；Step 只写临时文件；Close 验证步数/终态，写 attrs（closed_ok=True 只是临时文件内的预备标志），flush/fsync **H5 与已启用的 sidecar**、关闭 handles，先发布 sidecar，**最后以 H5 rename 作为提交点**。不在此之后安排任何会令该 Close 改判失败的附属 I/O；失败日志是独立尽力通道。只有完成该提交序列的 trace `.h5` 才能通过 §7.3，单独存在 sidecar 不算成功；无 sidecar 配置不要求该文件。
3. **失败状态**：Step/Close/sidecar 任一步失败，episode 进入 Failed；后续 Step 丢弃并释放 permit，Close 只清理、永不发布 H5。errors() 与 sink sticky 状态线程安全且不可清零；尽力改名 `.h5.failed`、写 `_trace_failures.jsonl`，失败则保留 tmp。审计依赖成功凭证/身份全集，不依赖失败日志一定写得出。非正常断连 terminal=False 的产物即使完成关闭也不准入成功集。
4. **暴露边界**：build 模式的 writer sticky 错误阻止后续 episode_start/record_step，入口收到失败信号后停止接纳新 build 请求并安排关闭/drain。已有 infer/lifecycle 错误按现协议传播；**不保证失败原集自动重派，不保证下一次调用一定存在，也不把下一个 episode 的失败记作原集已修复**。末步/Close 失败由最终 DrainReport、非零服务退出状态和 mandatory accepted-set 审计闭合；缺失集必须由既有重跑流程补齐后重新审计。诊断模式不使后续推理失败，但失败集仍不发布，错误仍报告。
5. **退出**：入口主控制流 try/finally 中先停止接收新请求，在共同 shutdown deadline 内等待在途 producer、终结未关闭集并 drain（总预算30s），再以 DrainReport 决定退出码：任何写错误/未完成/超时 ⇒ 3（其它原始非零退出码仍保留失败）。trace-on 使用可选 server stop event：writer 失败通过 loop.call_soon_threadsafe 设置事件，SIGTERM handler 同样只请求主循环退出；server.run(stop_event=None) 保留原 await serve_forever 路径，非 None 才进入停止接收/关闭连接的分支。deadline 到期将未完成 episode 标 Failed/禁止晚发布并退出3；默认线程池或 CUDA 清理卡住时入口的终止 watchdog 最终 os._exit(3)，不依赖无限 join。replica 子进程返回3，既有 supervisor 的 watchdog 汇总为非零1，同样判采集失败。atexit 仅 best-effort 兜底，**不以 atexit 的 SystemExit 设置退出码**。SIGKILL、解释器崩溃、机器掉电不能保证报告/退出码，仍靠缺失/临时/身份审计拒收；本协议不宣称跨掉电的事务持久性。
6. `__hit_meta__.trace.writer_error` 可作即时提示，不是成功/失败的唯一凭据。drain 必须在 artifact 审计前完成；服务退出 0、文件终态与 accepted 对账共同构成一次成功采集的证据。

---

## 8. 建库形态（R6）

默认 SOP（D11）：`serve_policy --cache --trace-out D --trace-build-cache`（无 yaml，`orchestrator=None`，走 serve_policy.py:637–648 分支）⇒ key/search/judge 为空（无 twins），prefix token 由 `_slice_cp1_fields` 切出（§12-J GPU parity 证明与老 hook 逐位同源）；verdict 恒 MISS、executed=full；`save_timesteps = schedule.timesteps`。也允许带 yaml + `trace.enabled` + `record_noise_actions`。老 `--collect` / `CollectionPolicy` / `_validate_collect_isolation`、GR00T `--collect-hdf5` / `GrootCacheCollector` 在 GPU parity 过门后于同一提交删除（D7）。

---

## 9. GR00T N1.5

1. **入口与锁**：只在接收 raw observation 的 `_get_action_impl` 首行分派到 `_get_action_traced`，一次 apply_transforms；已 normalized 的 `_get_action_cp2` 保留 legacy 路径，不再二次 dispatch。trace-off 保留现有整个 infer 的 `_InferLockedPolicy`；trace-on concurrent 改为共享 model 锁保护 stage1/2、需要模型的 KB 操作、dtype probe、noise/start_x 准备和 ready events，然后释放锁提交 §6 的 stage3。runner.session/autocast 是线程局部，producer 与 worker 分别进入。non-concurrent trace 在同线程直调；metadata/build 路径两者都支持。
2. **分支**：CP1：stage1 → session 外 check(trace=True) → stage2 必跑；CP2-only：stage1/stage2_llm/cp2 key source 准备完成后 session 外 check(CP2, trace=True)，无 CP1/CP3。check/model attachment 所需的只读模型操作遵守同一锁；生成存储张量经 `_to_storage_tensor` 在 session 外 clone。之后与 §5 同样可空 top1、full/warm/warm_exec、真实 winner 选发、broadcast/buffer/clear；GR00T 原 buffer 实参不额外塞入 π0.5 intermediates。输出只 unapply_transforms 一次，保留原 batch 维和 __hit_meta__（含 online_rit）。
   原始采集在一次性 batch 规范化后、apply_transforms 前复制 `video.*`、`state.*`、`annotation.*`；raw_image_keys 由 GR00T adapter 按服务输入相机映射生成，不沿用 π0.5 默认 observation/image 键。按原协议的 batch/time 维取当前帧（B=1 的 `[0,0]`），形状不符显式报错；prompt 从 annotation 取文本。prefix 用 slice_groot_cp1_fields，trace_vision_fields 与已配置 key_builder 或无库服务相机映射一致，不一致启动报错；tokenized_prompt 取 eagle_input_ids。禁止猜三相机用于 LIBERO 两相机。
3. **full 捕获**：runner.run_stage3(..., noise=None, on_step=None) 仅加法参数；noise=None 且 on_step 非 None 拒绝；noise=None 旧分支不变。trace 显式提供 noise，经 denoise_loop 回调，snapshot_t(i) 映射 x_in；build 断言 caps 恰有 N 个，caps[0] 是实际进入 loop 的 noise，clean_action=full.action_pred[0]。live_schedule 在装配/episode 边界核对；不把 DESC 的 .7/.5/.3 注入 ASC schedule。
4. **噪声 dtype**：`sample_noise(stage2, *, generator=None)` 的 shape/device 与 head 一致，dtype 必须等于 process_backbone_output 后的 vl.dtype。首次实际 stage2 到达时，在 model 锁和 runner.session 内用新建 `_head_inputs(stage2)` 做只读 probe；用 fork_rng 保存/恢复所涉及 CPU/CUDA RNG，模型须 eval；缓存键含 model/device/autocast/input dtype。构造期不能凭空 probe；不复用会被 process 原地改写的 BatchFeature。probe 与全局采样均在共享锁内，不能恢复覆盖其他 producer 的 RNG。worker head eval 无随机采样，变体 noise 均由 producer 准备。generator=None 走全局，私有 generator 不推进全局。
5. **continuation 闭合（R2-B4）**：真实与 twin 分别读取自己的 continuation_spec/pending_snapshot。真实流保留现 `_continuation_feedback` 的执行条件、candidate、feedback_mode、snapshot、n_rejected/fb_batch_size/invalid_reasons：WARM 从**真实执行变体**保留的 first_step_input/x 得到 executed pair，fm0 仅此；fm1 对其他档仍按原 helper 的 first_step_updates 调用和 batch 计数执行；MISS 按真实 candidate 取 payload 做原 fm1 side 更新；无候选/gate skip 维持原空 feedback；FULL 按原分支处理。不能把已算的 twin top1 更新塞给真实 registry，也不为真实 judge 多调用一次反馈。
6. **反馈调度与 twin feedback**：stage3 回包后，真实与 twin 的额外 first_step_updates 在共享 model 锁内、各自 runner.session 中串行执行；复用 worker 输出前须已完成 worker sync。用 twin pending candidate 和配置中的 fm0/fm1，假设执行的是 twin effective verdict，依同样反馈契约调用 twin record_continuation 一次；只对同 `(entry,tier,schedule)` 的有效第一步结果复用。所需候选不是诊断 top1 时，单独 side-evaluate 其 first_step_updates；pending=None 时反馈空行。缓存缺后继参考快照按原 feedback 校验记录 rejected，不能把不可比较的更新当成功。真实和 twin 回传诊断分别落盘。warm payload 的 capture_first_step 标志、batcher 拆分辅助结果、直调都必须实现；不得将结果缩成只有 Tensor。
7. **采样/loop 等价**：固定外供 noise 的 loop parity；同 seed 下 `run_stage3(noise=sample_noise(stage2))` vs `run_stage3(noise=None)` 的 sampling parity；私有 generator 的全局 RNG 前后不变；probe 首次与缓存命中分别测。同形 B>1 的容差门见 §6，不能用回调次数替代数值验证。
8. **serve 接入**：两个 GR00T 入口增加 --trace-out/--trace-build-cache，trace concurrent 构造进程级 coordinator；无 yaml 同样构造 TraceRuntime。--stage1-only/--compile-stage1/--diagnostic-seed/--rit-shadow-out/--loto-log-out 与 trace 互斥；修非并发 yaml 分支错误构造；GPU parity 后同提交删除旧 --collect-hdf5/GrootCacheCollector。load_guard 不改。
9. **import 隔离**：batching_core、trace/types/runtime/groot/h5_sink、groot/batcher/interceptor 不得 import openpi.models/openpi.policies/cache.interceptor/stage_io/jax；trace/pi05 仅由 π0.5 入口懒加载，GR00T import 链不得经过它。GUARDED_FILES 按各文件职责验证，不把模型特定 π0.5 adapter 误当通用运行时。

---

## 10. 配置与校验（C）

```python
@dataclass
class TraceConfig:                          # 置于 ShadowTeacherConfig 后、CacheConfig 前
    enabled: bool = False
    out_dir: str = ""
    record_noise_actions: bool = False
    record_prefix_tokens: bool = True
    record_raw_images: bool = True
    record_model_images: bool = False
    record_query_keys: bool = True
    record_search: bool = True
    record_tokenized_prompt: bool = True
    raw_image_keys: list[str] = field(default_factory=lambda: ["observation/image", "observation/wrist_image"])
    rng_isolation: str = "verdict_aware"    # "verdict_aware" | "global"
    queue_steps: int = 256
    sidecar_jsonl: bool = True

CacheConfig.trace: TraceConfig = field(default_factory=TraceConfig)
_CONFIG_TYPES["TraceConfig"] = TraceConfig   # 同位顺带登记 ShadowTeacherConfig（HEAD 既有缺陷，两行）
```

`_trace_errors(config) -> list[str]`（挂 `validate_cache_config` 末尾）：`enabled ⇒ out_dir 非空`；`write_policy.type == "never"`；`collection.export_collect_meta is False`；`shadow_teacher.enabled is False`；`routing is None`；`rng_isolation ∈ {...}`；`record_noise_actions ⇒ record_prefix_tokens`；`queue_steps ≥ 1`。**无判官白名单**。`validate_effective_trace(config | None, *, out_dir, build_cache) -> TraceConfig | None`。`_TWIN_STRIP`（§4.2）与递归字段守卫测试；沿用现有 gate/judge/strategy 合法性校验，两 backend 均须提供诊断 capability，不用禁用 record_search 绕过动作依赖。

serve_policy.py：`Args.trace_out`、`Args.trace_build_cache`；删 `collect/collect_dir`；`_resolve_trace` / `_build_trace_runtime` 在三条 `_wrap_policy` 分支注入 `trace=`（twins 用该连接的 shared_storage、真实已装配组件及 process twin registry；surface tiers 从 judge.artifact 读取）；`_validate_trace_isolation`；删除 `_validate_collect_isolation` 与 `CollectionPolicy` 包裹；主控制流 shutdown/finally + signal 请求退出 + atexit 尽力兜底（§7.4-5）。

---

## 11. 文件清单

**新增 `src/`**

- `src/openpi/serving/batching_core.py` — jax-free 核心 + `StageBatcher` 协议 + payload 类型（§6.1）
- `src/openpi/cache/groot/batcher.py` — `GrootStageBatcher`（§6.3）
- `src/openpi/cache/trace/{__init__,types,h5_sink,runtime,pi05,groot}.py` — §7.1（`runtime.py` 含 `build_trace_twins`、`executable_warm_tiers`、`_TWIN_STRIP`）

**修改 `src/` / `scripts/`**

- `src/openpi/serving/batching_coordinator.py` — 核心 + `Pi05StageBatcher`；`submit_many_to_stage`、`save_timesteps` 并集、per-bucket 隔离、`_MODEL_DEFAULT_SAVE_TIMESTEPS`
- `src/openpi/cache/backends/{in_memory_backend,qdrant_backend}.py` — 各加只读 `per_field_scores`（§4.4；不改现有搜索路径）；PerFieldTrace 数据类型在 trace/types 中，诊断调用时懒加载
- `src/openpi/cache/cache_storage.py` — `per_field_scores` 一行透传（**唯一**触碰 facade 之处）
- `src/openpi/cache/config.py` — `TraceConfig`、`CacheConfig.trace`、`_CONFIG_TYPES` 登记（+ ShadowTeacherConfig）、`_trace_errors`、`validate_effective_trace`
- `src/openpi/cache/orchestrator.py` — `_ComponentSet/_CheckState`、`_check_impl` 抽取、`CheckResult.trace`、`__init__(trace_twins=)`、`check(trace=, fetch_top1=)`/`trace_check`、所有传递状态依赖参数化、twins 生命周期广播与 attachment、按 cset 路由 continuation 接口
- `src/openpi/cache/interceptor.py` — `__init__(trace=)` + 互斥、`infer` 首行分派、`_infer_traced/_trace_variants/_run_trace_variants/_select_executed`（CP1 与 CP2-only；`warm_exec` 保底；B3 kwarg 规则）、`_stage3_via_coordinator` 透传、生命周期转发、探针、online_rit twin 反馈
- `src/openpi/collect/data_collector.py` — 行为保持抽取
- `src/openpi/collect/collection_policy.py` — **删除**（D7，GPU parity 过门后同一提交）
- `scripts/serve_policy.py` — §10 + §7.4-5（正常 shutdown 的 drain/退出状态）
- `src/openpi/cache/groot/staged.py` — `run_stage3(on_step=)`、`sample_noise(stage2, generator=)` + 首次真实 stage2 的 dtype 探针；warm 第一帧辅助结果沿用现有类型
- `src/openpi/cache/groot/interceptor.py` — §9-1/2/5/6（raw 入口、锁、真实/twin continuation）
- `src/openpi/policies/policy.py:240` — 仅更新旧 collect 注释
- `src/openpi/serving/websocket_policy_server.py` — 可选 run(stop_event) 的 trace shutdown 接口（§7.4-5），默认路径与 wire 协议不变；旧 collect 注释同步

**修改 `exp/`**

- `exp/libero_groot/serve_groot_libero.py`、`exp/robocasa365/serve_groot_n15.py` — §9-8 + §7.4-5
- `exp/robocasa365/groot_cache_collector.py` — **删除**（连带核查 `bench_groot_stages.py`、`groot_cp2_parity.py`）
- `exp/robocasa365/verify_collection_artifacts.py`、`exp/libero_groot/verify_shadow_h5.py` — §7.3（schema 分支、accepted attempt 身份与完整性）
- `exp/common/build_in_memory_cache_artifact.py`、`exp/zixuan_proposal/build_dual_artifact.py` — 核查 collection_policy 引用并更新，trace 建库按成功审计清单限定输入（§7.3），旧输入不加 trace attrs 要求

**测试**

- 新增 `tests/cache/trace/{test_config_trace, test_orchestrator_trace, test_twins_isolation, test_twins_stateful_judges, test_executable_tiers, test_per_field_scores, test_interceptor_traced, test_coordinator_core, test_coordinator_many, test_h5_schema_superset, test_writer_threading, test_writer_failure_protocol, test_trace_collect_parity_gpu}.py`、`tests/cache/groot/{test_trace_groot, test_groot_batcher, test_groot_stream_ready_gpu}.py`
- 修改 `tests/cache/test_serving_optimization.py`（`_StubModel` 经 π0.5 batcher，断言不变；**stub 的 `run_stage3` 对 `save_timesteps=None` 显式 raise**）、`tests/cache/groot/test_import_isolation.py`、`tests/collect/test_data_collector.py`、`tests/collect/test_init_noise_and_schedule_stamp.py`、`tests/robocasa365/test_groot_concurrent_serving.py`、`tests/robocasa365/test_pi05_stack_parity_manual.py`、`tests/exp/test_llm_layer_extract_self_check.py`、`tests/serving/test_policy_recorder_lifecycle.py`
- 删除 `tests/collect/test_collection_policy.py`、`tests/robocasa365/test_groot_cache_collector.py`、`tests/robocasa365/test_groot_cache_manual.py`（改写为 GR00T trace build 真件门）

**文档（WA §4 同 commit 同步索引）**

- `docs/data_collection/guide.md`、`docs/architecture/cache_system.md`（+ `.zh.md`）、`docs/experiments/conductor_tutorial.md`、`docs/cache/migration.md`、`docs/README.md`、`logs/README.md`

**源码范围说明**：owner D2 的 backend 放宽在 r4 落为两个现有 backend 的同一诊断 capability；新增 server stop_event 仅用于闭合原有异步写盘失败协议，不改 RPC/client/conductor。上述均属于本会话授权修订后的计划范围。

**不触碰**：`src/openpi/cache/components/**`、`backend_pool.py`、`groot/load_guard.py`、`pi0_pytorch.py`、任何 `exp/**/data/`。两 backend 与 cache_storage.py 仅新增 §4.4 的只读诊断 capability；不重写既有检索算法或融合语义。

---

## 12. 测试策略（实施后的验收门；G1 不冒充已跑）

**A. 真实路径回归**

1. 全量 `uv run pytest`；GPU 门控 CPU 基线使用 CUDA_VISIBLE_DEVICES=""。Verify 每个失败都列 nodeid、错误签名、同 HEAD `54699e3` 同环境对照；已知 GCS/源码锁失败不自动豁免，按 WA §6 标未完成或引用 owner 的明确测试豁免。
2. 使用生产 validator 接受的**具名 fixtures**分别覆盖：threshold + score_hysteresis / follow_winner；composite + 支持其 factors 的 WSS depth1/depth>1；CRD/surface + always_search + depth1 WSS 和匹配 artifact；mlp_router arms=tc + dump_dir 的合法配置；online_rit 的合法 GR00T/fm0 与 fm1 配置；failure_aware_gate + dual_retrieval；RRF/dynamic_depth/text_ivf 分别配支持它们的普通 judge。非法 CRD+非 always_search、surface+不合法检索等做独立拒绝测试，**不做全 Cartesian product**。
3. 每 fixture 50 步固定录制 observation/stage outputs/执行动作，对比 HEAD、重构 trace-off、重构 trace-on 的真实 CheckResult（去 trace）、真实 history/counters/judge state、strategy query history/session memo、gate/commit/continuation 调用实参；twin 必须独立。冻结 seed/clock/UUID 或按角色映射 session id 后再比较，不能把随机身份/时间戳误报回归。数据序列注入测试证明状态隔离；实机 GPU 数值用另门。
4. trace=None 的 interceptor stage/check/broadcast/buffer/clear/输出 spy；coordinator 老调用的返回类型、分桶与 None 不传 kwarg；老 collector 抽取后的 schema/data golden。原 serving optimization 测试断言不弱化；stub 显式拒绝 save_timesteps=None。

**B. twin 所有权与闭合**：两个 facade、timer、KB attachment、judge 签名表独立；拆掉 dump 后保持原有效 top-k 下限；真实旧属性和 `_real_state` 同一真值，生命周期只清一套。CRD FULL→WARM→MISS ≥3 步每成功判断 commit 一次；真实 payload 错误保持 raise；twin 缺快照明确 proposed WARM→effective MISS + reason，commit 合法降级且 pending 清空；非快照异常使 episode 失败并重建/重置生命周期，不能继续悬挂 proposal。FollowWinner skip 只读记录 replay_target，强制路径只 searched feedback 一次，真实 replay 行为不变。CP1 FULL×CP3 gate skip：twin CP3 collect/build/search/judge 各一次，真实 CP3 零次；其他 verdict 不能重复 twin CP3。CP2-only 无 CP3。递归 `_TWIN_STRIP`；真实 router 输出在固定身份/时钟下内容相同、twin 零文件；online_rit real/twin registry 互不污染，两个 twin 连接按 yaml/library 共享同一个 twin registry。

**C. 档位、空路径与选发**：六类 judge 的 executable tiers 与 required snapshots 分开测；surface 从已装配 artifact 取值、online 排除后继；DESC/ASC 与档位 id 无冲突。winner≠top1 与未枚举真实 WARM 均生成 warm_exec；缺诊断快照只跳该变体。无 orchestrator、空库、gate skip、record_search=False 且无 warm tiers 都能输出 full，top1 存在时仍记录 full_hit。真实 FULL 取真实 winner 而非 twin top1；真实 cache 簿记各 verdict 对齐，CP2 library provenance 保留。

**D. 逐 field 诊断**：InMemory depth1/depth>1/fallback/RRF/dynamic_depth/dual/text_ivf 和 Qdrant native/chunked 分别测矩阵 `[k,F]`、行序、kind、缺字段 present、零候选与未知 id。WSS 首行与现 winner_per_field 相等、current_step_wss 含相同权重；L2 正距离进入 normalizer；RRF 不伪造 fused 分。Qdrant mock 验证候选过滤/point id/chunk 平均，集成门用现有本地服务或测试实例核对已知向量数值；只调用读 API、不改真实 memo。无法运行集成门必须在 Verify 如实标明，不能靠 mock 代替该门。

**E. direct/coordinator 模型契约**：π0.5 × build on/off × direct/coordinator：off 不传 None，on 为完整 schedule；Groot N=4/N=10 ASC 同矩阵，不能出现 π0.5 .7/.3 默认。混合 None/空 tuple/显式请求的 union 与回填；warm first_step 辅助张量按请求正确拆分，旧返回类型不变。真实 online_rit 的 fm0/fm1 × WARM/MISS/无候选/skip 按 legacy 数值与调用/计数比对，候选≠twin top1；twin 更新独立且配置模式不被强制改为 fm1。

**F. 并发、GPU 所有权**：3 连接同形 multi-variant 可合桶，异形分桶；per-bucket 失败后下一桶仍完成。所有新增 trace 请求（两模型）GPU 人为延迟实际 producer stream，worker wait_ready 后读到正确数据；多 producer stream、timeout、forward 已排队后抛错、disconnect/unload、drain 均证明引用持有到 kernel 清理完成，再释放。CPU stub 验证顺序、deadline 和未入队取消；GPU 门验证真实 stream，不用立即就绪的 stub 冒充。致命 CUDA 状态测试以模拟错误证明 worker 停止，不在测试中破坏真实 context。

**G. 写器与失败协议**：Step 背压、writer 死亡唤醒、Open/Close 不阻塞 event loop；直接对 sink 注入“Close 时仍有在途 Step”⇒ 断言失败并标 Failed（服务端串行下该情形不出现，测试只守不变量）；两个 episode token 不串身份；跨进程同路径 reservation 竞争仅一方成功且不覆盖既有文件；CPU 快照不会被 producer 后续修改。注入末 Step、attrs、H5/sidecar flush/fsync/rename、失败日志不可写、drain timeout；均无成功 H5 发布、sticky/errors 可读、permit 释放、正常入口以子进程验证非零退出（drain 错误为3）。server stop_event=None 的默认路径回归；writer 失败/SIGTERM 均驱动正常 shutdown；子进程测试卡住 producer 的30s终止边界、禁止晚发布、replica 非零汇总，atexit 仅兜底。模拟客户端吞掉 episode_end 错误及最后一次调用后不再发请求，最终 accepted-set 审计仍拒收缺失集；**不要求 conductor 自动重派**。

**H. H5 与消费者**：旧 collector 文件没有 trace attrs 仍按旧规则接受；version1 noise on/off、未知版本、closed/terminal/errors 校验；accepted attempt 缺失/身份不符/重复映射拒绝，先失败后成功不同 attempt 允许成功 attempt 通过但报告残留。run-plan 任务覆盖检查与手工单文件模式区分。write_step_group 与 sink 老键值/dtype/shape/压缩相等；schedule/read_step_intermediates 两态；成功审计清单建小库并 warm 读取；sidecar 行数/终态对应，warm 名字以 snapshot_index 可唯一回读；raw wire key 包含 `%`、`/`、`__` 时编码无碰撞，GR00T raw/prompt/tokenized 与输入相机一致。

**I. GR00T 专项**：raw transforms/unapply 各一次，CP2 不重入 raw 分支；trace-off 外锁不变、trace-on 提交 stage3 前释放锁。worker 自开 session，storage clone 在 session 外。dtype probe 首次/缓存命中与私有 generator RNG 隔离；loop parity 与 sampling parity；on_step N 次、noise=None 回调拒绝；B=1 数值逐位，同形不同 prompt/state 的 B>1 按已定容差，异长不 padding。保留旧异长补零改变输出的反例。import isolation/GUARDED_FILES；重跑 G0-C 真件门。

**J. 删除老 collector 的 GPU 前置证据**：真 π0.5 固定 noise，用测试内老同源 hooks 与 trace build 比较 prefix/clean/noise 全字段；GR00T 对旧 collector 同源切片/shape/dtype 做对应证明；各产物通过现审计器、建 pkl 并作为库加载。证据拿不到不得删旧实现，整包仍不能提交为完成；不拆成另一版本。

**K. 端到端与 Verify**：π0.5 非并发无库建库2集；concurrent warm 2–4 client 合桶>1；同 seed 串行 MISS-only executed 对比；build→审计→pkl→warm trace；composite/CRD/tc-router 各1集状态隔离；GR00T LIBERO/RoboCasa concurrent 同形合桶+异长分桶；online_rit fm0/fm1 各1集真实/twin 反馈；吞吐/显存/队列峰值对表；诊断不可达 warm 档仍计算且按可用性标记；末步/Close 人为失败后服务退出与 accepted-set 审计拒收，补齐后重审成功。执行阶段填写命令、环境、artifact 路径与结果，不提前填 PASS。

---

## 13. 风险登记

1. twin 与真实的 gate/判官/历史会分叉；twin 是 forced-search 诊断，不是关闭 trace 时真实判决的反事实。明确 proposed/effective/error，影子降级不流入真实组件。
2. 抽取 `check` 涉及传递依赖及旧属性兼容，靠具名合法 fixtures 的录制回放、session/文件/continuation 隔离证明；不是简单替换八个 self 字段。
3. 计算代价为 full+各 warm+必要 warm_exec/online side 更新，另有 KB/CPU 检索；Qdrant 逐 field/chunk 多出网络读；记录吞吐不掩盖此成本。异长不能合批；只读模型仍需正确 producer/worker stream 所有权。
4. queue 背压仅约束已排队 step，不约束全部模型显存/在途 CPU 快照；按连接数另估。π0.5 约4.4 MiB/步，建库数百集可达数百 GiB，数据按仓库规范落 `/data`。
5. CUDA timeout 不是取消 kernel；任何异常释放输入前必须完成清理。context 致命错误不能在同 worker 中继续下一桶。
6. GR00T batch kernel 与串行可能存在数值差异；RNG dtype、首次 probe、loop 和 sampling 分开验证，不许用“cat 是 view”或“调用序列一样”代替数值门。
7. 异步 Close 错误可能在客户端已 accepted 后才发现；零 client/conductor 改动下无自动修复承诺，强制 drain/退出状态 + 身份全集审计与补采。SIGKILL/掉电靠残留/缺失拒收，不声称能返回退出码3。
8. 删除老采集器的连带引用、两种模型 schema、旧文件审计兼容必须同提交验证；同名最终文件禁止覆盖，失败 attempt 和成功 retry 不混淆。
9. `_infer_traced` 与 legacy 主体需各自维护；spy 和明确的调用契约约束漂移。HEAD 的构造/类型登记缺陷只做已授权局部修复。
10. H5 超大 JSON attr 转 dataset；schema 要有 present/availability，不能用缺 dataset 隐式伪装零分。原始图像转义名称需可逆且无碰撞，随 schema 保存 wire-key 映射。
11. transformers/pinned GR00T 源码的等价依赖靠真件门守卫；离线 IR/开环误差不能解释成闭环 SR。

---

## 14. owner 裁决项

| # | 项 | 裁定 / 默认 |
|---|---|---|
| D1 | 有状态判官 | **owner 裁定**：所有模块都跑、真实决策流不变、记录即可 ⇒ 完整 twin 组件集 + 共享流水线（§4），无白名单 |
| D2 | 逐 field 分粒度 | **owner 裁定**：放宽到 backend ⇒ 两 backend 的 per_field_scores + facade 透传（§4.4，逐 field 相似度；RRF 不伪造排名分解） |
| D3 | RNG 默认 | 默认 `verdict_aware`；备选 `global` |
| D4 | 噪声档关闭时的 stamp | 默认总是写 stamp + `trace_noise_actions_recorded` |
| D5 | GR00T 并发 | **owner 裁定**：结构一起改 ⇒ coordinator 核心 jax-free 拆分 + `GrootStageBatcher` 同形分桶跨连接合批 + trace-on 锁粒度 + Event/输入所有权契约 |
| D6 | CP2 臂 | **owner 裁定**：纳入 |
| D7 | 老机制删除 | 默认：GPU parity 过门后同一提交删除 |
| D8 | GR00T `--compile-stage1` 与 trace | 默认互斥 |
| D9 | top-1 缺某 tier 快照 | 默认 `tier_status="no_snapshot:<reason>"` 跳过 + 启动期 warning |
| D10 | 输出目录层级 | 默认 `<trace_out>/<experiment>/<episode_name>.h5` |
| D11 | 建库默认形态 | 默认无 yaml；同时允许带 yaml |
| D12 | 评审粒度 | **owner 裁定**：一次 G1、一次 G2、一个 commit |
| R10 | memo 屏蔽 | **owner 裁定** ⇒ twin 机制 |
| HEAD 缺陷 | :858–861、ShadowTeacherConfig 登记 | **owner 裁定**：顺带修 |

---

## 15. 实现顺序（仅排程，不改变一次 G2）

1. data_collector 抽取 + TraceConfig + trace 类型/写器/失败协议 + π0.5 无库 trace；保留旧 collector 作 parity 对照，先不删除。
2. orchestrator `_check_impl` 抽取（录制回放先绿）→ twins + `_TWIN_STRIP` + `executable_warm_tiers` + `per_field_scores` + warm 档 / `warm_exec` + CP2 分支 + CP3 twin。
3. coordinator 核心拆分 + `Pi05StageBatcher` + `submit_many/save_timesteps/per-bucket` + 并发实机冒烟。
4. GR00T 噪声/loop/first-step 契约 + 同形 batcher + stream 所有权 + trace-on 锁 + CP1/CP2 traced 入口 + serve 局部修复；两模型真件 parity 过门后，最终提交内删除老采集器。
5. 审计器 schema/accepted-attempt 完整性 + 建库输入清单 + 文档索引；G2 后按 WA §6 Verify 全量并附 GPU/端到端证据。

---

## 16. 探索过程记录

Workflow `wf_4caa4161-1f8`（2026-09-21 20:20–21:30 CDT，12 agent，0 失败）。三份独立设计的审视分（compliance / correctness / value）：A 7.5/6.5/7；B 8/7.5/7.5（骨架）；C 6.5/7/6.5。从 A 移植：`check(trace=)` kwarg + 单嵌套 `CheckResult.trace`、MISS 簿记 intermediates 过滤三键、桶内无 trace 请求不传 kwarg、`orchestrator=None` 建库形态。从 C 移植：`enqueue/wait_for/submit_many` 语义、`_MODEL_DEFAULT_SAVE_TIMESTEPS` + `inspect.signature` 单测、bucket-first 只 warning、`rng_isolation=verdict_aware`、`trace_` 前缀 stamp 副本、50 步录制回放、GR00T `caps[0]` 作 `noise_action_0`。三份共同遗漏、本 plan 补齐：per-bucket 故障隔离 + payload 预检、warm start_x `.to(stage3_device)`、`fetch_top1` 条件化、attr>60000B 转 dataset、sidecar、`tokenized_prompt/eagle_input_ids` 落盘、SR 反事实边界、WA §4 索引与连带测试清单、serve_groot_libero.py:858–861 缺陷。**r2**（owner 裁定）：twin 机制取代判官白名单；coordinator 核心拆分 + GR00T 合批；CP2 纳入；backend 加法导出；删一切 v1/v2/M4。**r3**（G1 R1）：twin 从"strategy+judge 影子"改为**完整组件集 + 共享流水线 `_check_impl`**（B1/B9）；GR00T 改**同形分桶**放弃补零（B2）；直调 `save_timesteps=None` 不传 kwarg（B3）；`executable_warm_tiers` 与 `warm_exec` 保底（B4）；`per_field_scores` 复算覆盖全部检索路径（B5）；`_TWIN_STRIP` 剥离一切原生落盘（B6）；CUDA Event 就绪契约（B7）；写失败终态协议 §7.4 + 审计器拒收（B8）；N1–N3 落入 §12-E/§9/§12-A-1。**r4**（owner 授权）：补齐 cset 依赖与独立 CP3 入口；模型特定档位与 first-step 反馈；两 backend 明确分数语义；CUDA 引用所有权；写盘凭证/旧 H5/accepted attempt 审计；GR00T 单入口与合法测试矩阵。既有各轮评审正文与执行方回应保留在下方，仅追加本轮回应和复核。

---

## 17. §4 Code 执行记录（执行方，2026-09-22）

按 §15 顺序实施，一次 G2、一个 commit（D12）。本节只记事实；GPU 门与端到端在 §17.3 按实际结果填写，未跑的项标 **pending**，不提前填 PASS。

### 17.1 落地清单（与 §11 对照）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | `data_collector.py` 抽取 `resolve_episode_path` / `write_episode_attrs` / `write_step_group`（老 writer 行为不变）；`TraceConfig` + `validate_effective_trace`；`cache/trace/{types,h5_sink,runtime,records,pi05}.py`（`TracePlan`/`StepTrace`/`TraceSink`/`TraceRuntime`；单写线程 `TraceWriter` + FIFO/permit 背压 + reservation + `.h5.tmp`→fsync→sidecar→rename 提交 + `.h5.failed` + build sticky）；π0.5 `_infer_traced`（无库建库形态） | done |
| M2 | orchestrator `_check_impl(cs, …, force_search, observer)` 抽取 + `_CheckState`/`TwinSet`/`_ComponentSet`；`check(trace=, fetch_top1=)` → `CheckResult.trace`；`trace_check`（FULL_HIT 后 CP3 twin-only）；`executable_warm_tiers`（六类 judge）+ `warm_exec` 保底；两 backend `per_field_scores` + facade 透传；CP2 分支；`TWIN_STRIP_FIELDS` / 只读路径豁免表 | done |
| M3 | `serving/batching_core.py`（jax-free 核心：`StageBatcher` 协议、`Stage3MissPayload.save_timesteps`、`ready_events`、`submit_many_to_stage`、per-bucket 故障隔离）+ `Pi05StageBatcher`（union/回填；全 None 不传 kwarg）；`BatchingCoordinator` 保持旧构造签名 | done |
| M4 | `groot/staged.py`：`run_stage3(on_step=)`（noise=None 拒绝回调）、`sample_noise(stage2, generator=)` + fork_rng dtype probe 缓存；`groot/batcher.py` `GrootStageBatcher`（同形分桶键含 schedule/exec_domain/embodiment/shape/dtype，cat 不补零；`run_miss/split_miss/run_warm/split_warm` 直调与合批共用）；`groot/interceptor.py` `_get_action_traced`（CP1 与 CP2-only、锁只覆盖 stage1/2+噪声+payload 准备、真实 `_continuation_feedback` 保留、twin 反馈复用变体 first-step、失败时双流 invalid）；`trace/groot.py` adapter（`video.*/state.*/annotation.*` 原始采集、`slice_groot_cp1_fields` 切片、`eagle_input_ids`、caps→`noise_action_*`）；`trace/runtime.py::build_groot_trace_runtime`；两 GR00T 入口 `--trace-out/--trace-build-cache` + 互斥 + 进程级 `BatchingCore(GrootStageBatcher)` + 非并发 yaml 分支 :858–861 修复；`GrootLiberoPolicyAdapter` 转发 `on_task_begin/end`；`serving/trace_serving.py`（共享退出协议，serve_policy 委托）；import 隔离 GUARDED/TRANSITIVE + pi05-adapter 不可达守卫 | done |
| M5 | 审计器 trace 分支（`verify_collection_artifacts._check_trace_attrs` + `expected_identity` + `unfinished_files`；`verify_shadow_h5.trace_problems`）；文档（`docs/data_collection/guide.md` 改为 trace 口径、`cache_system.md` §9.X Trace serving mode、`.zh.md` §9.6、`migration.md`、`conductor_tutorial.md`、`docs/README.md`）；`exp/common/trace_smoke_client.py`（§12-K 合成 client）；GPU parity 门测试 `tests/cache/trace/test_trace_collect_parity_gpu.py`（manual）与 `tests/robocasa365/test_groot_cache_manual.py` 改写为 trace build 真件门 | 代码 done；**GPU 门 pending（§17.3）**；老 collector 删除按 D7 等门后执行 |

### 17.2 CPU 测试证据（`CUDA_VISIBLE_DEVICES=""`）

- `tests/cache/trace/`（writer 线程/失败协议/schema 超集/config/interceptor traced/orchestrator trace/stateful twins/executable tiers/per-field/coordinator many/auditors）+ `tests/cache/groot/`（含新增 `test_groot_batcher.py` 12、`test_trace_groot.py` 15、`test_groot_stream_ready_gpu.py` 2[GPU]、import isolation 40）：**358 passed, 6 skipped**。
- `tests/serving/test_trace_serving_shutdown.py`：5 passed（build 写失败→request_stop→exit 3；SIGTERM 未完成集 exit 3；干净关闭 exit 0；watchdog `os._exit(3)` 有界；子进程跑真实 `WebsocketPolicyServer(stop_on_request)`）。
- 回归：`tests/cache tests/collect tests/serving tests/libero_groot tests/robocasa365/test_groot_concurrent_serving.py tests/robocasa365/test_groot_obs_adapter.py` → **2309 passed, 23 skipped**（deselect `test_frozen_commands_pass_the_new_guards`：HEAD 既有顺序依赖 `gr00t.__spec__ is None`，单独运行 2 passed）。
- 审计器相关：`tests/robocasa365/test_collection_artifacts.py`、`test_pnp_audit_and_build.py`、`tests/libero_groot/test_verify_shadow_h5.py`、`tests/cache/trace/test_auditors_trace_schema.py`（12）全绿。

- `tests/robocasa365 tests/exp`（忽略 `test_bench_groot_stages.py`：HEAD 既有收集错误 `bench.SCHEDULE_ID` 不存在）→ **2124 passed, 4 failed**；4 个失败在干净 HEAD worktree（`git worktree add … HEAD`，同 venv）同样失败，均非本线回归：`test_ws2_evidence_runner.py::test_hit_meta_rows_identical_across_runners[2]`（c0dac3a 给行加了 `start_t` 键，测试未更新）、`test_prebuilt_matrix_backend.py::test_cosine_fast_path_bit_identical / test_fast_path_robust_to_candidate_reordering`（fast path 与 parent 非逐位相等；`_compute_field_scores` 源码与 HEAD 逐字相同）。`tests/review_tests` 需 `REVIEW_SCRATCH` 环境变量（HEAD 既有），本轮未跑。

### 17.3 GPU / 真件 / 端到端门（§12-J、§12-K、§12-I G0-C）

| 门 | 命令 | 结果 |
|---|---|---|
| π0.5 trace build vs 老 hook 逐字段 parity + 审计 + 建 pkl 加载 | `uv run pytest tests/cache/trace/test_trace_collect_parity_gpu.py --run-manual -q`（2026-09-22 19:4x CDT，本机 4090，`/home/weiland/ckpt_pi05_robocasa_pytorch`） | **PASS** 2 passed：vision_0..2 / prompt_emb / noise_action_0..9 与老 forward-hook 捕获逐位相等；served action 逐位相等；clean_action ≤1e-5；文件过 `_check_h5_schema(require_schedule=True)`、建 1 条 pkl 并由 InMemoryBackend 加载（证据卡 `exp/robocasa365/analysis/trace_collect_parity.txt`） |
| GR00T 真件：trace build vs 老 hook 捕获 + 在线/离线 key parity + G0-C 重跑 | 岛 B `gr00t_n15_venv` `-m pytest tests/robocasa365/test_groot_cache_manual.py --run-manual -v`（RoboCasa target_posttrained ckpt-60000） | **PASS** 10/10：G0-C 两阶段逐位等价与负对照全过；`test_trace_build_matches_the_legacy_hook_capture`：同 seed 下 trace 的 `sample_noise`+转写 loop 与上游 `get_action` 内部 randn 的 `action_encoder` hook 捕获逐位相等（noise_action_0..3、vision/prompt/robot_state、clean_action、served action）；在线/离线 key parity PASS |
| π0.5 非并发无库建库 2 集 → 审计 → pkl → 带 yaml 并发 warm trace（3 client lockstep） | `e2e/run_pi05.sh`（端口 23180，产物 `/data/openpi_trace_e2e/pi05_20260922_200604/`） | **PASS**：build server SIGTERM 退出码 0；2 文件审计零问题、建 12 条 pkl；warm server `--concurrent` 退出码 0；3 连接 × 6 步全部 WARM_START，执行档为预算的 `warm_05`；文件含 twin top-k（2）、逐 field 分（3 field）、query keys、`full_hit/warm_03/warm_05/full_inference/executed`；stage1/2 跨连接批 size=3；诊断文件被审计器按用途正确分流（诊断接受 / 建库拒收） |
| GR00T LIBERO concurrent 同形合桶（无 yaml 建库形态） | `e2e/run_groot.sh`（端口 23185，`ckpt_n15_libero_spatial`，k8） | **PASS**：3 连接 × 6 步全部成功；3 文件 `groot_n15_k8_v1`、两相机 vision_0/1、noise_action_0..7、trace 提交态 + 审计零问题；server 退出码 0；pull-then-group 补跑的 stage-3 `[batch_done]`：8 批中 size=3 ×4、size=2 ×2（同形跨连接合进同一 forward） |

**门暴露并已修复的 4 个真缺陷**（均补了回归测试，CPU 套件复跑全绿）：
1. π0.5 `_precheck_variant_tensor` 以字符串比设备，`"cuda"` 与张量 `cuda:0` 误判 ⇒ 新增 `_same_device`（按 index/当前设备语义比较）+ 单测。
2. `scripts/serve_policy.py` 的 `--trace-build-cache` 声明为 `bool | None`，tyro 要求带值，文档里的开关用法解析失败 ⇒ 改为普通开关（关=沿用 yaml，开=强制建库）；同时 trace 生效判定改为按有效配置（yaml 单独打开 trace 也走 drain/退出码协议）。
3. GR00T traced 路径把 `apply_transforms` 放在共享锁外，3 连接并发首调时 einops 惰性后端注册竞态（`Tensor type unknown to einops`）⇒ `apply/unapply_transforms` 纳入共享锁（stage-3 仍在锁外合批）。
4. 生产入口只把 twin 组件集放进 `TraceRuntime.twins`，从未挂到 orchestrator（单测直接构造 `trace_twins=` 所以没暴露），导致无 twin 检索、无 top-1、执行档退化为 warm_exec ⇒ 新增 `CacheOrchestrator.attach_trace_twins`（同一集幂等、换集拒绝），两个 interceptor 构造时挂接 + 生产式装配回归测试。
| Qdrant `per_field_scores` 集成门 | qdrant-client 本地实例（`:memory:`）+ 生产 schema | **PASS**（§17.2） |

### 17.4 交 G2 前全量回归（§4 本地证据；§6 Verify 的正式裸跑在 G2 之后）

`CUDA_VISIBLE_DEVICES="" uv run pytest --ignore=tests/review_tests --ignore=tests/robocasa365/test_bench_groot_stages.py`（两处忽略是 HEAD 既有收集错误：`REVIEW_SCRATCH` 环境变量 / `bench.SCHEDULE_ID` 不存在），2026-09-22 20:13–21:02 CDT：**6001 passed / 9 failed / 84 skipped（49m20s，机器同时有他人负载）**。9 个失败全部非本线回归：
- 8 个与 2026-09-15 HEAD 基线清单逐条一致：`test_prebuilt_matrix_backend` ×2、`test_robocasa_policy_config` ×2（全量才出现的 tmp 隔离缺陷）、`test_ws2_evidence_runner::test_hit_meta_rows_identical_across_runners` ×2、`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2（`gr00t.__spec__ is None`，环境；参数 id 因本线把 `--collect-hdf5` 用例换成 `--trace-build-cache` 而改名，失败签名不变）。
- 1 个 `src/openpi/models/lora_test.py::test_lora_einsum_params_shape`：`CUDA_VISIBLE_DEVICES=""` 下 JAX 的 CUDA 插件 cuInit 失败（环境）；本线未改 `src/openpi/models/`（`git diff --stat HEAD -- src/openpi/models` 为空）。

以上归因引用的是旧日期基线，G2 R1 指出不能据此豁免；同 HEAD 同环境的重跑见 §17.5-C。

### 17.5 G2 R1 之后补齐的 §12 验收证据（执行方，2026-09-23）

§17.3 各行只证明其所列场景，不代表 §12 全部门通过。R1 的 B11 列出的缺口逐项补在这里；没跑成的项写明原因，不填 PASS。

**A. §12-A 具名合法配置 × 50 步 HEAD / trace-off / trace-on 全状态对照** — `tests/cache/trace/test_named_config_replay.py`（新增，CPU）

- 12 个具名配置，全部经生产 `load_cache_config`（validator）+ `build_shared_storage` / `build_per_connection_components` / `build_trace_twins` 装配：follow_winner、RRF、dynamic_depth、failure_aware_gate + dual_retrieval、composite × WSS depth1 / depth2、CRD（dispatch_surface + 按本 yaml 检索契约与库身份写的 artifact）、surface（SurfaceArtifact）、text_ivf（cp1_mean_pool 的 prompt_emb + robot_state + vision_0）、mlp_router arms=tc + dump_dir（每个 run 独立 dump 目录）、online_rit fm0 / fm1（`groot_n15_k8_v1`）。审查方的 threshold × {always, hysteresis} × {WSS, WSS depth2, RRF} 子集不重复。
- HEAD 臂直接从 `git show 54699e3` 加载老 `CacheOrchestrator`；三臂共用一个 backend，2 集 × 25 决策。每步比对：真实 `CheckResult`（去 `trace`）、orchestrator 历史与计数器、每个真实 gate / judge / strategy / key builder 的完整规范化状态、真实 session 的分数 memo、online judge 的学习状态 sha、`record_verdict` / `commit_verdict` / `record_continuation` 的全部实参。uuid 会话 id 按首次出现映射角色，墙钟冻结为常数（router 行时间戳），router 每 run 的 dump 路径做别名。另断言 twin 组件与真实组件是不同对象、有自己的 session 与 registry。
- 负对照：把 twin 的 gate / judge 换成真实对象（twin 集要防的缺陷），follow_winner / CRD / online_rit fm1 三个配置都必须报 `off vs on` 分歧，三个都报了。
- 结果（2026-09-23 10:54 CDT 复跑）：**15 passed**（12 对照 + 3 负对照），10.0s。

**B. §12-F 真实 disconnect / unload / drain 全生命周期** — `tests/serving/test_trace_connection_lifecycle.py`（新增）

- 真实并发 `WebsocketPolicyServer`（`concurrent=True`、动态 bundle 开启）+ 进程级 `BatchingCore(GrootStageBatcher)`，每个连接一套 `GrootCacheInterceptor`（trace build 模式，共享模型锁），stage-3 前向人为放慢。三个场景：客户端发出 infer 后立即断开；另一连接在途时，一个连接 rebind 到第二个 bundle（服务端结束旧栈）且在途连接所绑定的 registry 条目被替换；在途时收到停止请求（监听关闭、handler 先完成决策再 `on_task_end`、writer 排空）。断言：在途决策完成、落盘步数与 `trace_terminal` 按生命周期（断开/rebind 旧栈 = 非终态提交，审计不收；正常 `episode_end` = 终态）、core 无 fatal、无 `_FATAL_PAYLOADS`、writer drain ok；输入在 worker 入口和前向之后各算一次校验和（GPU 上后者排在 worker stream 前向之后、最后才读回），两者相等。
- CPU 与 CUDA 各跑一遍。CUDA 变体用 `torch.cuda._sleep` 真正拖住 worker stream（host 立即返回、kernel 仍在排队），`_publish_batch` 处记录回包那一刻 worker stream 是否已空闲。stub head 的 lazy 层首次前向会让 host 同步、掩盖缺失的 stream 同步，所以 harness 先在目标设备上预热一次。
- 负对照（不入库）：把 `BatchingCore._sync_stage_stream` 换成空操作，三个 CUDA 场景都失败，回包时刻记录为 `(1, False)`；正常代码下为 `(1, True)`。
- 结果：`uv run pytest tests/serving/test_trace_connection_lifecycle.py` → **6 passed**（CPU 3 + CUDA 3），约 14s，连跑 3 次稳定。

**C. §17.4 失败的同 HEAD 同环境归因**

- 做法：`git worktree add --detach /data/openpi_trace_e2e/head_wt 54699e3`；用主树同一个 venv 的解释器，`PYTHONPATH` 让 worktree 的 `src` / `packages/openpi-client/src` 优先（`openpi.__file__` 已核实落在 worktree，且 HEAD 下 `openpi.cache.trace` 不存在），执行与 §17.4 相同的 `CUDA_VISIBLE_DEVICES="" pytest --ignore=tests/review_tests --ignore=tests/robocasa365/test_bench_groot_stages.py`。2026-09-23 10:50–11:39 CDT：**12 failed / 5808 passed / 122 skipped**（48m43s）。日志 `/data/openpi_trace_e2e/full_pair/head_full.log`。
- §17.4 的 9 个失败在 HEAD 上全部复现，错误签名一致：`lora_test::test_lora_einsum_params_shape`（JAX CUDA 插件 cuInit）、`test_prebuilt_matrix_backend` ×2（`assert False`）、`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2（`gr00t.__spec__ is None`；HEAD 的参数 id 是 `collect-alone`，本线改成 `trace-build-alone`）、`test_robocasa_policy_config` ×2（全量下 tmp 路径 FileNotFoundError）、`test_ws2_evidence_runner::test_hit_meta_rows_identical_across_runners` ×2（`assert False`）。
- HEAD 多出的 3 个失败来自 worktree 本身，主树没有：`tests/dispatch_surface/test_rev2_phase0.py::test_task_manifests_match_split_assignments_and_known_atoms`（worktree 下缺 gitignored 数据文件）、`tests/libero_groot/test_gate_pareto_paths.py` ×2（worktree 没有 `.venv`）。
- 同一轮在当前工作树跑的同命令全量（10:50 起排队、11:39–12:31 CDT）：17 failed / 6023 passed / 88 skipped。除上面 9 个，多出的 8 个都是 R1 审查修复带进来、审查方回归范围（`tests/cache tests/collect tests/serving tests/scripts`）没覆盖到的，已修，见 Review Log 执行方回应的 B9 / B10：`tests/libero_groot/test_cp2_bundle_guards.py` 1 个（`bundle.config_path`），`tests/robocasa365/test_pnp_audit_and_build.py` 7 个（`build_manifest` 的无条件守卫）。最终树的全量结果见本节 E。

**D. §12-K 真模型 episode 与性能/队列对表**（2026-09-23 11:40–13:05 CDT，本机 4090，与另一会话的 step_diag GR00T server 共卡，按"空闲显存 ≥ 需求 + 余量"放行；产物 `/data/openpi_trace_e2e/k_20260923_104537/`，编排与对比脚本在 job tmp，不入库）

- 做法：每个具名配置在真 LIBERO 仿真里跑同一集两次（libero_spatial task 0、init 0、仿真 seed 7；client 复用 `examples/libero/main.py` 的 `_run_episode`，只把 episode_start 的身份补成 run_id / batch_id / task_uid / attempt 四件套），一次 trace-off、一次 trace-on（`--trace-out`），server 都是生产入口原样启动。逐步比对：服务动作块（逐位）、`__hit_meta__` 去掉 `trace` 字段后全等；再查 trace 文件（提交态、步数、每步判决 = 下发判决、twin 记录）和真实组件的原生输出（router dump 分片、online-RIT 状态日志，只屏蔽路径/墙钟/按臂命名的 yaml_id）。
- **同 seed 是必需前提**：这版 torch 每个进程的默认生成器种子都是随机的（两个独立进程的 `torch.initial_seed()` 与首个 `randn` 都不同），所以两个 server 进程的 MISS 噪声本来就不同。第一轮没固定种子，GR00T fm0/fm1 在各自第一个 MISS（第 6 步）起分叉，而且 off–off（fm0_off 对 fm1_off）和 on–on 之间也同样分叉，由此定位为进程 RNG，与 trace 无关（该轮存档在 `unseeded/`）。之后在启动包装里统一 `torch.manual_seed(20260923)` 再跑。
- 结果（off / on 各一集）：

| 配置 | 模型 · 服务形态 | 推理次数 | 判决 | 服务动作逐位相等 | hit_meta 不一致步 | trace 文件 | 真实组件原生输出 off vs on |
|---|---|---|---|---|---|---|---|
| online_rit fm0（正式 O-init 臂 yaml，S3 库） | GR00T · 并发单连接 | 17 | WARM 15 / MISS 2 | 17/17 | 0 | 17 步，终态提交，0 写错误，每步 twin top-k | 状态日志同名同数（4/4），终态快照与 feedback 逐字段相等；learning state 按臂名归一后重算哈希相等（15 次更新 / 15 条反馈） |
| online_rit fm1（同上，只改 feedback_mode） | GR00T · 并发单连接 | 17 | WARM 15 / MISS 2 | 17/17 | 0 | 同上 | 同上（15 次更新 / 45 条反馈） |
| composite（按生产 eval yaml 结构，warmup 标定） | π0.5 · 非并发 | 29 | FULL 9 / WARM 8 / MISS 12 | 29/29 | 0 | 29 步，终态，执行档 full_hit / warm_05 / full_inference | — |
| CRD（dispatch_surface，artifact 绑定本 yaml 检索契约与库身份） | π0.5 · 非并发 | 17 | WARM 16 / MISS 1 | 17/17 | 0 | 17 步，终态，执行档 warm_07 / full_inference | — |
| mlp_router arms=tc（固定种子权重，sample 模式，dump_dir） | π0.5 · 非并发 | 17 | FULL 16 / MISS 1 | 17/17 | 0 | 17 步，终态 | dump 分片 `.bin` 字节相同，jsonl 只差墙钟 `ts`；两臂都是 3 个文件，trace 目录下无 router 文件（twin 零文件） |

  所有配置都由生产 validator 与工厂装配校验（`prep_report.json`）；库为 π0.5 `cp1_mean_pool`（1018 条）与 GR00T w13 S3（1078 条）。composite 先用合成标定跑出全 MISS（记录在报告里），又用这一集的真实 jerk 原始值做 offline 标定重跑，三种判决才都覆盖到。fm0/fm1 各有 2 步 twin 判决与真实判决不同：该 yaml 用 `score_hysteresis` gate，twin 每步强制检索，这是 §13-1 预期的分叉，真实流不受影响。
- 性能 / 显存 / 队列对表（3 个 lockstep 客户端 × 40 步，`--concurrent`，生产入口 + 进程内采样线程每 20 ms 读 writer 队列、core 各 stage 队列与 CUDA 分配器峰值；nvidia-smi 每 200 ms 按 server PID 取进程显存；同卡有他人负载，吞吐只作 off/on 相对比较）：

| 运行 | 判决 / 执行档 | 决策/s | p50 / max ms | 分配器峰值 MiB | 进程显存峰值 MiB | writer 队列峰值 | stage-3 队列峰值 |
|---|---|---|---|---|---|---|---|
| GR00T off（online_rit fm1） | MISS×120 | 3.29 | 577 / 1922 | 5429 | 5996 | — | — |
| GR00T on | MISS×120 / full_inference | 3.99 | 565 / 1535 | 5503 | 6276 | 5 | 1 |
| π0.5 off（warm yaml，09-22 e2e 库） | WARM×120 | 2.60 | 1084 / 3819 | 7593 | 8448 | — | 0 |
| π0.5 on | WARM×120 / warm_05 | 1.60 | 1817 / 4273 | 7593 | 8452 | 6 | 2 |

  GR00T trace-on 反而更快：trace-off 是整次推理一把锁串行，trace-on 的 stage-3 变体走进程级 `BatchingCore` 跨连接合批（D5）。π0.5 trace-on 吞吐降约 38%，来自每个决策额外的 full 与各 warm 变体（§13-3）；两模型的显存峰值增量都在 0.3 GiB 以内，writer 排队峰值 5–6 步。
- **RoboCasa GR00T 并发"同形合桶 + 异长分桶"真件门**（§12-I / §12-K；`tests/robocasa365/test_groot_cache_manual.py::test_concurrent_trace_batches_same_shape_and_splits_lengths`，新增）：3 个连接在一个 `BatchingCore(GrootStageBatcher)` 上并发 trace build：连接 0/1 同 prompt、图像与 state 不同（同形不同内容），连接 2 是长 prompt。stage-3 调用序列为 `[[827], [814, 814], [814, 814], [827]]`（每个 payload 的 conditioning 长度）：同形两路每步合进一次前向，长 prompt 始终单独成桶、不补零。每次调用都按原组成与顺序直接重算，下发结果与之**逐位相等**（无串扰、拆分正确）；B=2 相对串行 batch-1 的漂移为 0.022–0.259（L2），同一 conditioning 换一次噪声的动作差约 18.4，漂移不超过其 1.4%，判据为 ≤ 10%（沿用 online-RIT 真件门的噪声地板规则）。第一版测试用了固定相对容差 1e-2（那是同 batch 的 eager/compiled 门），在 bf16 下 B=2 的真实漂移（1.9%）超出，且两条同形连接观测相同、测不出 conditioning 串扰，已按上面的设计重写。整文件：岛 B `gr00t_n15_venv` `-m pytest tests/robocasa365/test_groot_cache_manual.py --run-manual -v` → **11 passed**（G0-C 两阶段逐位等价与负对照、trace build 对老 hook 捕获、在线/离线 key parity、本并发门），36s。
- 跑门过程中暴露并已处理的问题，逐项写进了 Review Log 的执行方回应：GR00T LIBERO adapter 让 trace-off 断连也调用真实 `on_task_end`（改为只在 trace 模式下关闭 trace sink）；GR00T LIBERO 非并发入口不能服务 online_rit（HEAD 同样如此，没改，本门改用并发单连接）；审查修复里 `bundle.config_path` 与 `build_manifest` 守卫带来的两处回归；RoboCasa manual 测试要改成"审计后以 episode-list 建库"。

**E. 最终树全量回归（交 G2 前的本地证据；§6 正式 Verify 在 G2 之后）**：与 §17.4 相同命令 `CUDA_VISIBLE_DEVICES="" uv run pytest --ignore=tests/review_tests --ignore=tests/robocasa365/test_bench_groot_stages.py`，2026-09-23 13:05–13:55 CDT：**9 failed / 6036 passed / 88 skipped**（49m14s），日志 `/data/openpi_trace_e2e/full_pair/final_full.log`。9 个失败与 C 中 HEAD `54699e3` 同命令同环境复现的 9 个逐条相同（nodeid 与错误签名一致，frozen-commands 的参数 id 除外），C 里当前树多出的 8 个已消失。GPU 与真件部分另跑：π0.5 parity 门 `test_trace_collect_parity_gpu.py --run-manual` 2 passed；`tests/serving/test_trace_connection_lifecycle.py` + `tests/cache/groot/test_groot_stream_ready_gpu.py` 8 passed；RoboCasa 真件 manual 11 passed（D）；审查方独立套件 `tests/review_tests/cache_trace_g2` CPU 26 passed / 4 skipped，其 GPU 用例 `test_cuda_ownership.py` 4 passed（只运行、未读取）。

### 17.6 §6 Verify（G2 R2 APPROVED 之后，执行方，2026-09-23）

- 复核 R2 审查方修复（R2-B1 批次按全部 H5 判定、R2-B2 `DrainReport.fatal_error`、R2-B3 `close_trace_search_sessions`）：与交审备份 `/tmp/cache_trace_g2_r2_intake/` 逐文件对比，13 个文件的改动全部接受。审查方的独立探针不入库，所以把这三项的回归测试补进了仓库（只加测试，不改产品代码）：`tests/cache/trace/test_auditors_trace_schema.py::test_batch_classification_reads_every_file_and_legacy_tmp_headers`、`tests/cache/trace/test_writer_threading.py::test_fatal_failure_whose_error_record_fails_is_never_a_clean_drain`、`tests/cache/groot/test_trace_groot.py::test_close_trace_episode_releases_only_the_twin_search_sessions`（生产 `WeightedRrfKnnStrategy`，断连后只释放 twin session）。三者在 R2 修复前的代码上都会失败。
- 正式 Verify：裸 `uv run pytest --ignore=tests/review_tests --ignore=tests/robocasa365/test_bench_groot_stages.py`（两个 ignore 是 HEAD 既有的收集错误：前者是审查方未入库的本地目录、需 `REVIEW_SCRATCH`，后者引用 HEAD 上不存在的 `bench.SCHEDULE_ID`），GPU 可见（卡上无他人进程），2026-09-23 21:04–21:50 CDT：**8 failed / 6049 passed / 82 skipped**（46m35s），日志 `/data/openpi_trace_e2e/verify/verify_full.log`。GPU 可见后 `lora_test::test_lora_einsum_params_shape` 通过；进程里 36.9 GB 显存是 JAX 初始化 CUDA 后端时按默认 75% 预分配，不是某个测试在跑重活。
- 8 个失败与 §17.5-C 在 HEAD `54699e3` 同命令同环境复现的失败逐条相同（nodeid 与错误签名），都不在本线改动范围：`tests/exp/test_prebuilt_matrix_backend.py` ×2、`tests/robocasa365/test_groot_concurrent_serving.py::test_frozen_commands_pass_the_new_guards` ×2（`gr00t.__spec__ is None`）、`tests/robocasa365/test_robocasa_policy_config.py` ×2（全量下 tmp 隔离）、`tests/robocasa365/test_ws2_evidence_runner.py::test_hit_meta_rows_identical_across_runners` ×2。
- manual GPU 门（推理路径）在最终树上重跑：`tests/cache/trace/test_trace_collect_parity_gpu.py --run-manual` **2 passed**；岛 B `tests/robocasa365/test_groot_cache_manual.py --run-manual` **11 passed**。`tests/serving/test_trace_connection_lifecycle.py` 与 `tests/cache/groot/test_groot_stream_ready_gpu.py` 的 CUDA 用例已包含在上面 GPU 可见的全量里。
- 8 个既有失败的处理：按 §12-A-1 不自动豁免；owner 于 2026-09-23 22:10 CDT 明确豁免这 8 个 HEAD 既有失败（「豁免，继续推动流程」），据此进入 §7 Commit。Verify 结论：**PASS（附 owner 对 8 个 HEAD 既有失败的明确豁免）**。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-23 09:46 CDT

范围：HEAD `54699e3` 至本次工作树的 Cache trace 实现、关联测试、文档和本计划。`exp/step_diag/**`、`tests/exp/step_diag/**`、`logs/session_handoff.md` 及其产物属于另一实验线，排除。独立测试只放在 gitignored `tests/review_tests/cache_trace_g2/`，不进入暂存区。owner 在本会话已明确授权先暂存执行方版本，再直接修复至可放行，修复必须留在暂存区外；本轮按该授权继续，不因 Review Authority 的通常只读限制再次索要许可。

独立复跑：trace + GR00T + shutdown 共 363 passed / 6 skipped / 3 沙箱端口失败；在允许绑定 localhost 的环境重跑 shutdown 为 5 passed。新增 11 个定向探针，10 个暴露下列缺陷，1 个证明缺少 noise_action_0 已会被审计器拒绝（不列问题）。GPU 在沙箱内不可见；只读提权查询可见 4090 48 GiB，其他实验已占约 42 GiB，后续验证不得干扰它们。

- [Blocking] [Concern] **B1 / P1 — FULL_HIT 的 CP3 twin 检查了未执行动作** — reasoning: `interceptor._select_executed` 在 FULL_HIT 返回 None，`_infer_traced` 的 CP3 参数遂回落到 full；§4.2/§5.9 要求真实缓存 chunk。独立探针捕获 CP3 实参，与执行 chunk 的 1600 个元素全部不同。
- [Blocking] [Concern] **B2 / P1 — 剥离 DumpingJudge 后未保留检索宽度** — reasoning: `runtime.build_trace_twins` 比较 real/twin judge 的 hint 后直接 raise，没有先提升副本 search_strategy.top_k。真实 dump factor K=3、inner threshold hint=0 的合法配置启动失败；与 §4.2 的明确要求相反。
- [Blocking] [Concern] **B3 / P1 — CUDA 异常路径违反输入所有权** — reasoning: bucket-first 的 `_dispatch_stage3_bucket` 在 forward 抛错后直接回包并清 payload，没有同步失败 forward；`_sync_stage_stream` 又吞掉同步异常，成功回包并继续工作。独立模拟分别观测到 forward→reply（缺 sync）及同步错误后 reply_slot='unsafe'。§6.2 要求清理后回包，无法确认完成时停止 worker、保留引用并使服务失败。
- [Blocking] [Concern] **B4 / P1 — writer 的失败终态不闭合** — reasoning: `stop(timeout)` 只把 stop 排在已入队 Close 后，超时后仍会发布成功 H5；`_handle_open` 在注册 job 前 resolve/mkdir 出错只置 `_fatal`，`DrainReport.ok` 仍为 True。两个独立探针分别复现晚发布和 fatal 后空错误的成功 drain。退出超时必须禁止晚提交，未注册 token 的失败也必须形成错误报告。
- [Blocking] [Concern] **B5 / P1 — reservation 的检查顺序允许覆盖既有成功文件** — reasoning: `_handle_open` 在取得 O_EXCL reservation 之前检查 final.exists，取得后不再检查；另一进程在二者之间提交并释放 reservation，本进程就能覆盖其 H5。独立插入该交错后原文件字节被替换。唯一性判断必须在取得 reservation 后进行。
- [Blocking] [Concern] **B6 / P1 — shutdown watchdog 未覆盖卡住的 producer/server** — reasoning: watchdog 在 `serve_forever()` 返回后的 finally 才启动；WS `wait_closed()` 和 asyncio 默认线程池退出均可能等待 infer。独立子进程收到 SIGTERM 后仍卡住，0.2 秒 watchdog 配置亦未生效，8 秒后只能由探针 kill。共同 deadline/watchdog 必须从停止请求开始覆盖这一段。
- [Blocking] [Concern] **B7 / P1 — 私有 RNG 丢失 CUDA index** — reasoning: `TraceRuntime.noise_generator` 用 `torch.Generator(device=dev.type)`，请求 cuda:1 得到当前默认 CUDA generator；多卡 stage placement 的 hit 额外 full 将出现 generator/tensor device 不匹配。独立参数探针证实 cuda:1 被降为 cuda。
- [Blocking] [Concern] **B8 / P2 — dynamic/dual RRF 的逐 field 分被误标为 WSS** — reasoning: `_twin_per_field` 用 normalization 是否存在推断 fusion，而 DynamicDepth/Dual 的实际 fusion 由 `_base_fusion` 决定，RRF 允许携带未使用的 normalization。独立 dynamic RRF 探针得到 weighted_score_sum；这会伪造 current_step_wss，与 §4.4 的 RRF 语义冲突。
- [Blocking] [Concern] **B9 / P1 — 有效运行配置与入口 shutdown/provenance 不一致** — reasoning: Pi05 `_startup_trace_mode` 只读取启动 yaml，之后动态 bundle 可以开启 trace 而进程仍走普通 serve_forever；GR00T shutdown 的 build_mode 只看 CLI，不看 yaml record_noise_actions。两个 GR00T `_trace_runtime` 还总传 args.cache_config 的路径，动态 bundle 会落错 yaml hash。应绑定实际连接配置并确保动态开启 trace 也受失败监控/退出协议管理。
- [Blocking] [Concern] **B10 / P1 — 新 trace 建库仍可绕过准入清单** — reasoning: `build_in_memory_cache_artifact.resolve_h5_paths` 未作任何 trace 分支，未给 manifest/episode_list 时仍直接 rglob 所有 H5。已完成文件中的非 accepted/orphan attempt 可直接进库；这违反 §7.3 要求 trace 建库只读取审计选定清单。旧无 trace schema 的扫描输入应保持兼容。
- [Blocking] [Concern] **B11 / P1 — §12 关键验收证据尚未闭合** — reasoning: §17.3 四行通过不等于 §12 全部 GPU/真件/端到端门通过。现有 50 步用例只比当前 trace-off/on 的 threshold + WSS/RRF，没有规定的同 HEAD、多类具名配置全状态对照；GPU readiness 用例未覆盖 forward 后抛错/致命同步/timeout/disconnect；§12-K 的 composite/CRD/tc-router、GR00T online_rit fm0/fm1 真件 episode、accepted-set 故障→补齐重审链未给证据。必须补证或如实保留未完成，不能保留“全部 PASS”的总述。
- [Non-blocking] [Concern] **N1 — 文档与索引有残留矛盾** — reasoning: guide 同时说 trace 支持并发和 single-connection-only；logs 索引在 G2 行尾仍说“当前仅 G1 计划”；reference 将 collect 描述为 forward hooks，未列 trace/batching_core。应同步最终事实。
- [Non-blocking] [Concern] **N2 — 显式空 snapshot 请求仍额外捕获默认档** — reasoning: Pi05StageBatcher 对任意显式请求的 union 无条件加入三个默认值，§6.1 要求只在某行 None 时加入默认值；当前返回过滤正确，但做了未请求的额外 capture。

Checklist：计划一致性 FAIL（B1–B10）；测试覆盖与通过 FAIL（独立反例与 B11，既有通过计数不替代缺失门）；文档/索引 PARTIAL（N1）；无回归 FAIL（B2/B3/B7 的合法配置/执行路径）。尚不能 APPROVED。下一步先冻结执行方快照于 index，再按 owner 授权修复；不得暂存后续审查者修复或独立探针。

### G2 Round 1 — Owner 授权修复与复核 — NEEDS REVISION — 2026-09-23 10:10 CDT

按 owner 本会话明确授权完成直接修复；这是同轮修复复核，不冒充另一独立审查者的 R2。以上 R1 原文保持不变。执行方 83 个范围内文件及 R1 记录已冻结于 index，tree 为 `bb73464176ed898bcfd44d149e28e898ae78fc88`；HEAD 仍为 `54699e3`。以下所有修复、文档同步和本记录均留在暂存区外。81 个排除文件逐个 SHA256 与接手时一致，暂存区无范围外文件，无 `tests/review_tests/`。

修复复核：

| 项 | 最终修改与复核结果 |
|---|---|
| B1 | FULL_HIT 的 CP3 输入改为真实选中缓存 chunk，保持 batch 维和 stage-3 device；独立探针逐元素对账通过。 |
| B2 | 剥离 dump 前读取 real judge 的有效 top-k 下限，提升副本配置后再装配 twin；真实 DumpingJudge + K=3 启动探针通过。 |
| B3 | 两种 scheduler 的异常路径先清理实际 worker stream 再回包；同步失败停止 admission/worker、失败待处理队列、保留输入到进程退出。停止 core 仍保留 fatal 状态供服务监控，成功回包与 fatal 判定互斥。普通 stop 排空已接收请求；CPU 致命错误探针与真实 CUDA forward-error、多 producer stream、timeout 引用测试通过。 |
| B4 | writer 消息入队与停止标志串行化；共享停止 deadline，超时撤销提交权，晚完成 Close 不再发布；未注册 token 的 Open 异常也进入错误报告。Close 时有在途 Step 显式 Failed。修正原两项“允许发布半集”的测试断言，未放宽失败协议。 |
| B5 | 获得 O_EXCL reservation 后再次检查 H5/临时/失败/sidecar 文件；只清理自己取得的 reservation；模拟另一进程在检查间隙完成提交的探针通过，旧文件字节保持不变。 |
| B6 | 从首次 signal/failure 停止请求启动 30s deadline 和 60s watchdog，覆盖 server wait_closed 与 producer/executor 卡住；启动前 stop 请求持久化，ready callback 前建立 stop event；真实 WS 子进程及卡住 server 探针通过。 |
| B7 | 私有 generator 使用完整 torch.device，保留 CUDA index；参数探针通过。本机仅一张卡，未冒充已实测 cuda:1。 |
| B8 | dynamic/dual 优先使用实际 `_base_fusion`；零候选也输出 `[0,F]` 诊断，空矩阵 kind 保持具体分数语义；补齐 Qdrant 每 field 的 chunk_count 和 distance_by_chunk 及 H5 元数据。RRF 反例与 mock/本地 Qdrant 相关测试通过。 |
| B9 | Pi05 启动时冻结 trace shutdown 配置，拒绝在未启用该协议的服务中动态开启 trace（提示 CLI/startup yaml）；writer 标记有效 YAML build 模式，监控不再仅依赖 CLI；两个 GR00T 入口记录实际 bundle 的 yaml 路径/hash。入口/provenance/YAML-only build failure 探针通过。 |
| B10 | 新 trace 建库拒绝目录扫描，要求成功审计 manifest 或已审计 episode-list，并复检提交态、schedule、完整 loop inputs；manifest 校验声明的 task_uid/attempt；失败的全集审计禁止导出部分 manifest。旧无 trace attrs 的 H5 保留扫描路径。Close 失败→exit 3→拒收→attempt 2 成功→重审→manifest 解析全链通过。 |
| N1 / N2 | guide、reference、data_collection 索引、logs 索引与本页状态同步；Pi05 snapshot union 只在请求为 None 时加入默认值，显式空 tuple 不捕获默认档，独立断言通过。 |

独立证据与回归（均在本轮最终实现上；不同套件有重叠，不相加）：

- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/cache tests/collect tests/serving tests/scripts -v`：**2077 passed / 20 skipped**，105.10s；在允许 localhost socket 的环境运行。日志 `/tmp/cache_trace_g2_regression_unsandboxed.txt`。此前沙箱运行已有 2039 passed / 20 skipped、3 个 shutdown 监听权限失败，并在服务测试中停滞；只中断该次 pytest 后重跑，不能把前次算成完整通过。
- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/review_tests/cache_trace_g2 -q`：**26 passed / 4 skipped**，21.82s；4 个 GPU 用例在下项单独实跑。日志 `/tmp/cache_trace_g2_independent_final.txt`。其中 6 组 × 50 步对照直接加载 HEAD `54699e3` orchestrator，同当前 off/on 比较每步 CheckResult、history/counters、gate/judge 状态、query history/counter 与真实 session memo；覆盖 threshold × always/hysteresis × WSS/WSS depth2/RRF，**不等于 §12-A 全具名配置矩阵**。
- `UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/review_tests/cache_trace_g2/test_cuda_ownership.py tests/cache/groot/test_groot_stream_ready_gpu.py -q`：**6 passed**，3.17s，真实 CUDA；日志 `/tmp/cache_trace_g2_cuda_final.txt`。只用小张量，不加载模型、不影响其他实验。同步 context 致命错误另用 CPU 模拟，未破坏真实 CUDA context。
- `tests/review_tests/cache_trace_g2/test_entry_and_admission.py tests/cache/trace/test_writer_threading.py tests/robocasa365/test_collection_artifacts.py tests/libero_groot/test_dynamic_bundle_guards.py tests/robocasa365/test_groot_concurrent_serving.py`：**63 passed**；日志 `/tmp/cache_trace_g2_followup.txt`。
- `git diff --check` 通过；`git diff --cached bb73464176ed898bcfd44d149e28e898ae78fc88 --exit-code` 通过，证明修复没有污染冻结的执行方 index。

- [Blocking] [Concern] **B11 仍未完成 — 不予放行** — reasoning: 本轮补上真实 CUDA 异常/多流/timeout 门、同 HEAD 的 threshold 子集以及失败→补齐重审链，但还缺 §12-A 中 follow_winner、composite、CRD/surface、tc-router、online_rit fm0/fm1、failure-aware + dual、dynamic_depth/text_ivf 的生产 validator 具名配置 50 步完整状态对照；§12-F 的真实 disconnect/unload 全生命周期；§12-K 的 composite/CRD/tc-router 真模型 episode、GR00T online_rit fm0/fm1 真模型反馈、规定的数值/性能/队列对表。执行方 §17.3 的已有证据保留为所列场景的历史结果，不能据此宣布这些门已过。本机 GPU 查询时 48 GiB 中约 42 GiB 被其他实验占用，本轮未启动大模型验收。完整 Verify 亦未完成，§17.4 的全部失败仍需同 HEAD 同环境归因，不能由旧日期基线自动豁免。

最终 checklist：计划一致性 **PARTIAL**（B1–B10 的具体实现问题已修复；B11 要求未全部证明）；测试覆盖与通过 **FAIL**（上述已跑套件通过，指定验收矩阵仍缺项）；文档与索引 **PASS**（含动态 trace 启动边界与准入规则）；无回归 **已测范围 PASS / 全面结论待 B11**（2077 项与 6 组同 HEAD 对照通过）。owner 已授权的直接修复按其覆盖通常 Review-only 限制执行，不列为流程违规。**NEEDS REVISION: B11**。Code 修复已完成，G2 保持未放行；不能将本轮作为完整 Verify 或项目验收完成凭证。

### G2 Round 1 — Executor — 2026-09-23 13:12 CDT

按 execution_authority §10.2 逐条回应。审查方按 owner 授权做的直接修复已逐项复核；复核和补证过程中发现的回归一并列出、已修。执行方快照仍冻结在 index（tree `bb73464…`，本轮没有改动 index）；本轮所有改动与审查方修复一样都在暂存区外，暂存由 owner 决定。

- **B1** — Accepted — 复核审查方修复：FULL_HIT 的 CP3 输入为 `Stage3Output(action_chunk=cp.payload.action_chunk[None].to(self._stage3_device))`（`src/openpi/cache/interceptor.py:1210`），保留 batch 维与 stage-3 设备。真模型佐证：π0.5 tc-router 一集的 16 个 FULL_HIT 步执行档均为 `full_hit`，off/on 服务动作逐位相等（§17.5-D）。
- **B2** — Accepted — 复核：`build_trace_twins` 剥离 dump 前读取真实判官的 `min_required_top_k`，把副本 `search_strategy.top_k` 提到该下限再装配 twin（`src/openpi/cache/trace/runtime.py:193-196`）；§17.5-A 的 `router_tc_dump`（带 dump_dir）具名配置在 HEAD/off/on 三臂对照中通过。
- **B3** — Accepted — 复核：两种 scheduler 的异常路径先清理实际 worker stream 再回包；同步失败置 fatal、停 admission、保留输入（`src/openpi/serving/batching_core.py` `_fail_batch` / `_publish_batch`）。执行方另补真实服务器层证据：§17.5-B 的 CUDA 生命周期测试记录回包那一刻 worker stream 是否空闲，把 `_sync_stage_stream` 换成空操作后三个场景全部失败。
- **B4** — Accepted — 复核：停止与 abort 共享 deadline，过期后撤销提交权，晚完成的 Close 不再发布（`src/openpi/cache/trace/h5_sink.py:808`）；未注册 token 的 Open 异常也进入错误报告。另加一处加固：写线程 except 分支里的失败恢复本身再抛错时，只记日志并保持线程存活（`_fatal` / `_abort_reason` 已先置位，drain 报告和健康检查仍然看得到失败），避免一次恢复异常让写线程静默退出、后续队列无人处理。
- **B5** — Accepted — 复核：取得 O_EXCL reservation 之后重新检查 final / tmp / failed / sidecar 是否已存在（`h5_sink.py:671` 起），只释放自己持有的 reservation。
- **B6** — Accepted — 复核：第一次停止请求就启动共同 deadline 与 watchdog（`src/openpi/serving/trace_serving.py`），覆盖 `wait_closed` 和卡住的 producer；§17.5-B 的 drain 场景（决策在途时 `request_stop`）在真实 server 上通过，server 在 worker 交还决策之后才返回。
- **B7** — Accepted — 复核：私有生成器用完整 `torch.device` 构造（`src/openpi/cache/trace/types.py:347`）。同意审查方的说明：本机只有一张卡，cuda:1 未实测。
- **B8** — Accepted — 复核：`fusion_method = getattr(strategy, "_base_fusion", None)` 优先（`src/openpi/cache/orchestrator.py:959`），零候选输出 `[0,F]`；§17.5-A 的 dynamic_depth、failure_aware_gate + dual_retrieval 具名配置通过。
- **B9** — Accepted（附一处回归修复）— 复核 Pi05 启动时冻结 trace 停机配置、GR00T 记录实际 bundle 的 yaml 路径。审查方在两个 GR00T 入口里直接读 `bundle.config_path`，导致 `tests/libero_groot/test_cp2_bundle_guards.py::test_startup_factory_builds_a_cp2_only_interceptor_for_a_cp2_bundle` 失败：该测试的桩 bundle 没有这个字段（审查方的回归集不含 `tests/libero_groot`，所以没暴露）。生产的 `CurrentCacheBundle.config_path` 是必填字段，线上不受影响；改成与 `serve_policy` 一致的 `getattr(bundle, "config_path", None)`（`exp/libero_groot/serve_groot_libero.py:186`、`exp/robocasa365/serve_groot_n15.py:498`）。
- **B10** — Accepted（部分改写）— 接受"trace 建库拒绝目录扫描、只读审计得出的 manifest / episode-list"。改写 `build_manifest` 里"审计失败一律拒绝"的守卫：它让 `tests/robocasa365/test_pnp_audit_and_build.py` 的 7 个 HEAD 测试失败。旧 schema 流水线的契约是按 episode 拒收（被拒的剔除、合格的照常列出），§7.3 也明文要求无 `trace_schema_version` 的文件"完整保留旧校验"、旧库输入按旧 schema 接受；而生产上唯一发布 manifest 的 CLI 在 HEAD 就已经在审计失败时 `SystemExit(2)`。现在审计报告带 `trace_batch` 字段（本批有 trace 文件，或存在只有 trace 写器会留下的 `.h5.tmp` / `.h5.failed`），`build_manifest` 对失败的 trace 批次、以及没有声明批次形态的报告一律拒绝导出，只有审计器判定为旧 schema 的批次才保留逐 episode 语义；新增 `tests/cache/trace/test_auditors_trace_schema.py::test_failed_audit_manifest_is_whole_batch_for_trace_and_per_episode_for_legacy`（两种参数）。改写后 HEAD 的 7 个测试恢复通过，审查方独立套件 `tests/review_tests/cache_trace_g2` 为 26 passed / 4 skipped（只运行、未读取；4 个 GPU 用例另跑为 4 passed）。另外，`tests/robocasa365/test_groot_cache_manual.py::test_online_and_offline_keys_retrieve_the_same_entry` 仍用目录扫描建库，被 B10 的新规则拒绝；改为先过 `_check_h5_schema` 审计、再以 episode-list 建库，真件复跑通过。
- **B11（R1 原文）** — Accepted — 缺口逐项补证，全部写在 §17.5，不再保留"全部 PASS"的总述（§17.3 只证明所列场景）：A 12 个具名合法配置 × 50 步 HEAD/off/on 全状态对照，另加 3 个负对照（15 passed）；B 真实 `WebsocketPolicyServer` 上 disconnect / unload / drain 全生命周期，CPU + CUDA 6 passed，附去掉 stream 同步的负对照；C §17.4 的 9 个失败在干净 `54699e3` worktree、同一 venv、同一命令下全部复现，签名一致；D 真 LIBERO 仿真里同 seed 的 off/on 五组具名配置（GR00T online_rit fm0 / fm1，π0.5 composite / CRD / tc-router）服务动作逐位相等、hit_meta 零差异、真实组件原生输出一致，外加性能 / 显存 / 队列对表和 RoboCasa 并发"同形合桶 + 异长分桶"真件门；E 最终树全量回归。
- **B11（owner 授权修复复核记录）** — Accepted — 同上。§17.4 的失败已按同 HEAD 同环境归因（§17.5-C），最终树全量见 §17.5-E；§6 的正式 Verify 在 G2 APPROVED 之后按裸 `uv run pytest` 执行，本轮不冒充。
- **N1** — Accepted — 复核审查方的文档同步（guide、reference、data_collection 索引、logs 索引）无误。
- **N2** — Accepted — 复核：`Pi05StageBatcher` 只在某行请求为 `None` 时并入默认快照（`src/openpi/serving/batching_coordinator.py:163` 起），显式空 tuple 不再额外捕获默认档。

执行方在 B11-K 实跑中另外发现并处理的问题：

1. **GR00T LIBERO adapter 改变了 trace-off 行为（已修）**：本线之前给 `GrootLiberoPolicyAdapter` 无条件加了 `on_task_begin/end` 转发，HEAD 没有。结果是 trace-off 断连时也会调用真实 orchestrator 的 `on_task_end`，trace-on 的真实组件也比 trace-off 多一次生命周期调用（首轮 online_rit 实跑里 on 臂多出 task_end 快照，由此发现）。这违反"trace 关闭时现有路径逐字节不变"。现在 adapter 只在被包装的 interceptor 处于 trace 模式时暴露 `on_task_end`，并且只调用新增的 `GrootCacheInterceptor.close_trace_episode()`（以非终态关闭 trace episode，不碰真实组件）；新增 adapter 测试 2 个、interceptor 测试 1 个。RoboCasa 的 adapter 在 HEAD 就转发全部生命周期，两种模式一致，未改。
2. **进程 RNG**：这版 torch 每个进程的默认种子是随机的，所以闭环 off/on 对比必须在 harness 层固定种子，做法与证据见 §17.5-D。
3. **HEAD 既有限制（未改）**：GR00T LIBERO 入口的非并发分支用 `build_cache_components(config)`，不传进程级 `CurveRegistry`，因此不能服务 online_rit（HEAD `serve_groot_libero.py:858` 同样如此，ORT 正式跑都是并发）。这不在本线范围内，如实记录，是否另开由 owner 决定。

结论：B1–B10、N1、N2 的实现问题已修；复核中发现的 B9 / B10 两处回归已修；B11 所列证据已在 §17.5 补齐。重新进入 G2。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-09-23 18:03 CDT

复核目标：本计划执行方 R1 回应、§17.5 原始证据及相对 index `bb73464176ed898bcfd44d149e28e898ae78fc88` 的本线改动；其他实验继续排除。owner 既有授权继续适用。

对上轮回应的结论：接受 B9 的 `getattr` 测试桩兼容修复；接受 B10 保留已确认旧 schema 批次逐 episode manifest 语义的理由（本轮复跑 `test_pnp_audit_and_build` 已通过），但新批次识别有下述遗漏；LIBERO adapter 只结束 trace sink 的改法保持了 HEAD 的真实组件生命周期。B11 的新增具名配置与 CUDA 连接生命周期已独立复跑：相关 CPU 套件 483 passed / 6 skipped；真实 WebSocket + CUDA lifecycle/readiness 8 passed。直接读取五组真模型 off/on 原始 episode JSON 和 H5，逐步重算 actions 相等、身份归一后的真实 hit_meta 相等及 H5 判决相等，全部通过（fm0/fm1/CRD/router 各 17 步，composite_cal 29 步）。读取同 HEAD 和 final_full 原始全量日志，确认所列 9 个失败均复现；这证明本线归因，不豁免后续正式 Verify。

- [Blocking] [Concern] **R2-B1 / P1 — trace 批次分类依赖已准入候选，且把旧临时文件当作 trace 凭据** — reasoning: `verify_collection_artifacts.audit` 在唯一 accepted row 存在之后才读 H5 的 trace attr，missing journal / multiple accepted 两条提前退出路径不检查现有 trace 文件；混合目录中由此得到 `trace_batch=False`，失败全集仍可导出部分 manifest。反方向，旧 `EpisodeDataCollector` 同样写 `.h5.tmp`，所以 `bool(unfinished_files)` 不足以证明 trace。三个独立探针分别复现两个漏判和旧 schema 误判。应对完整文件集合判定 trace，包括未准入文件；临时文件按实际 schema 判定，不可读文件保守拒收。
- [Blocking] [Concern] **R2-B2 / P1 — nested recovery 失败仍可返回 clean drain** — reasoning: 新 except 守卫保持线程存活，但 `DrainReport.ok` 只读取 unfinished/errors/timed_out，未读取 `_fatal`。独立注入 Open 失败后 `_record_token_error` 再失败，得到 `healthy=False` 且 `DrainReport(unfinished=(), errors=(), timed_out=False).ok=True`；停机可能据此以 0 退出。应把 fatal 状态直接纳入 drain/stop 报告，不依赖错误记录器成功。

Checklist：计划一致性 PARTIAL（R2-B1/B2）；测试覆盖与通过 PARTIAL（既有新增门通过，4 个本轮独立反例失败）；文档/索引 PASS；无回归 PARTIAL（旧 schema 临时文件问题）。上轮 B11 的主要缺口已补证，本轮阻断变为以上两项；按 owner 授权继续直接修复，不提前宣告 APPROVED。

暂存边界：执行方独立增量及本轮审查记录入 index；上轮审查者源码修复仍留工作树。两个 GR00T serve 入口的 `getattr` 和 writer 的 nested recovery 与上轮未暂存修复重叠，三个文件整体保留未暂存，避免将审查者旧修复混入执行方暂存区。独立探针始终 gitignored。本轮交审工作树原样备份在 `/tmp/cache_trace_g2_r2_intake/`，排除文件摘要在 `/tmp/cache_trace_g2_r2_scope.json`。

### G2 Round 2 — Owner 授权修复与复核 — APPROVED — 2026-09-23 18:12 CDT

**code approved**。本结论覆盖当前完整工作树（含 R1/R2 审查者未暂存修复），不是只对 index 的结论。继续按 owner 的授权保持代码归属区分，不 commit、不混入其他实验。

本轮发现与关闭：

| 项 | 修复与独立复核 |
|---|---|
| R2-B1 — Closed | `audit` 对完整 H5 集合判定批次形态，包括未准入文件；读取临时文件 header，合法旧 `.h5.tmp` 不再误判为 trace；不可读 header 与 trace failure/reservation marker 采用整批准入。missing journal、重复 accepted、旧临时文件三个反例均转绿；旧 PnP 逐 episode manifest 契约仍通过。 |
| R2-B2 — Closed | `DrainReport` 加 `fatal_error`，drain/stop 的 `ok` 直接包含 fatal 状态；错误日志带出 fatal 原因。另复现 stop 分支的错误记录器异常会让线程退出且报告成功，已使 stop 捕获 fatal、继续清理其他 job，并用 finally 保证记录失败仍释放 H5/sidecar handle、清理临时文件。独立验证 nested recovery、stop cleanup、无错误记录的 fatal 仍 exit 3。 |
| R2-B3 / P2 — Closed（收尾补充发现） | LIBERO 的新 `close_trace_episode()` 只关 sink，生产 knn strategy 的 twin session 仍注册在共享 backend，断连后不释放。原测试用的 `_Strategy` 不创建 session，所以没覆盖。换生产 `WeightedRrfKnnStrategy` 的独立探针复现：关闭后 active set 仍含 real + twin。现加 `CacheOrchestrator.close_trace_search_sessions()`，由 trace-only close 的 finally 释放 twin session/memo，保持真实 hooks、真实 session 状态与 HEAD 一致；重复 close 幂等，探针通过。此修改不修订 HEAD 的真实 LIBERO 生命周期。 |

**B11 — Closed（G2 范围）**：独立阅读并复跑新增具名配置对照（包含负对照）；真实 WebSocket / CUDA 生命周期复跑通过；逐步读取五组真模型的原始 off/on JSON 与 H5 对账通过。fm0/fm1 的 `state_latest.json` 在归一 yaml 身份、进程 id、写入时钟并排除派生 hash 后，其完整学习状态结构相等，更新数均 15、反馈数分别 15 / 45；性能报告的 3 连接 × 40 步、耗时与判决已对照原始 client 输出（不能把其中夹带 uv warning 的文件直接当纯 JSON）。GR00T B>1 真件门沿用现有 online-RIT 噪声地板判据，另有同一 batch 组成的逐位回放约束，并非只有放宽容差。RoboCasa manual 11 项与 Pi05 parity 2 项采信执行方记录，未在本轮重复加载大模型。当前修复均未改变模型数值路径。

最终测试证据（各套件有重叠，不相加）：

- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/cache/trace tests/cache/groot tests/libero_groot/test_policy_adapter.py tests/libero_groot/test_cp2_bundle_guards.py tests/libero_groot/test_dynamic_bundle_guards.py tests/robocasa365/test_collection_artifacts.py tests/robocasa365/test_pnp_audit_and_build.py -q`：**483 passed / 6 skipped**，36.96s，`/tmp/cache_trace_g2_r2_cpu_final.txt`。
- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/review_tests/cache_trace_g2 -q`：**33 passed / 4 skipped**，17.27s，`/tmp/cache_trace_g2_r2_independent_final.txt`。7 个 R2 新探针；其中 6 个修复前反例有失败记录，另 1 个验证 fatal 退出码 3。4 个跳过为上轮的 GPU 专项，未冒充本轮 CPU 实跑。
- `UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/serving/test_trace_connection_lifecycle.py tests/cache/groot/test_groot_stream_ready_gpu.py -q`：**8 passed**，11.30s，真实 localhost socket 与 CUDA；`/tmp/cache_trace_g2_r2_lifecycle.txt`。
- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/openpi-g2-uv-cache uv run --no-sync pytest tests/serving/test_trace_serving_shutdown.py -q`：**5 passed**，14.34s，在允许 socket 的环境；`/tmp/cache_trace_g2_r2_shutdown_final.txt`。
- writer / audit / PnP 修复专项与旧独立探针合跑 **126 passed / 4 skipped**；最终 CPU 套件与 33 项独立检查又覆盖了之后的 twin 清理修复。
- `git diff --check` 通过。index 保留 86 个本线文件；上轮审查者源码修复及本轮修复均未暂存。index 条目（`git ls-files --stage -z`）SHA256 为 `9790335b617aa02ddb91ebc0c47a7d2bb42e432540d2de87277a95adaca9b837`，独立探针未入 index；81 个排除文件的内容 SHA256 与 R2 接手时一致。

最终 checklist：**计划一致性 PASS**（原 B11 补证与 R2 三项修复闭合）；**测试覆盖与通过 PASS（G2）**（上述独立运行、原始证据和负对照）；**文档与索引 PASS**（guide 与索引补充批次识别/fatal drain，plan/logs 索引同步）；**无本线新增回归 PASS**（执行方最终全量的 9 个失败，已从原始日志和 HEAD worktree `54699e3` 核对复现，同一 Python/pytest/plugin 版本）。owner 明确授权覆盖 Review 通常只读限制，未发现需要报告的流程违规。

**APPROVED — G2 放行；正式 Verify 待执行。** §17.5-E 的 6036 passed / 9 failed / 88 skipped 是执行方交审证据，非本次修复后的正式裸 `uv run pytest`；HEAD 既有失败的归因不构成豁免，也不表示 Verify 完成。
