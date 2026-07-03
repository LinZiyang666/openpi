# GATE 数据采集能力实现计划（cache "search or not" 组件）

> **Status**: In Progress
> **Level**: L3（由 L2 上调——跨 cache/serving/conductor/client + 新增子系统 `per_step_recorder`（`gate_step_sink` 随 raw 移出 scope 已删），符合 WA §2.1「cross-module change or new subsystem」；含架构文档更新，见 §3/§8）
> **Date**: 2026-07-03
> **一句话目标**: 在 **并发 serving** 下，用 **纯 per-connection wrapper（无 forward hook、不改 PI0Pytorch/推理内核）** 逐推理步采集「精简在线模型输入 + verdict-agnostic 判决 + 逐 episode 成功标签」三条可 join 的数据流，供 GATE「搜不搜」门控离线训练。
> **定稿来源**: 多专家 workflow（wire / server-sink 双设计角度合并 + 4 维对抗 critic 修复）起草 + 独立 minimal-diff 分析交叉验证 + owner R1/R2/R3 要求（见 §2.11），由 Execution 定稿。
> **说明**: 本文档为 **G1 APPROVED（2026-07-03）** 后的实施规范；G1 Review Log 已按 execution_authority §3.1 删除。

---

## 0. 背景与目标

### 0.1 三条数据流（每个 CP1 推理步一行，逐 episode 可 join）

1. **精简在线模型输入（LEAN input）**：即 `KeyBuilder.collect/build` 已经在每次请求本地持有的东西——视觉 embedding + robot_state（stage1 产物 / `cached_data`）。**默认存 `robot_state`（精简，conductor + standalone 通用）；vision opt-in 且仅 standalone**（§2.3/§2.10）。`raw prefix_embs`（大）**移出本计划 scope**（deferred，§8 D4）。不复制旧 collect 模式的「全部模型中间量」。
2. **该步最终判决（verdict）**：`hit_type ∈ {FULL_HIT, WARM_START, MISS}`、`cp1_score`（阈值判决的最终分）、`winner_id`、`start_t`。必须 **verdict-agnostic**（对 ThresholdJudge 成立，不仅限 kinematic/composite）。
3. **该 episode 是否成功（bool）**：join 到该 episode 的每一行 per-step row。

### 0.2 为什么旧 collect 模式在并发下不可用（必须换设计）

旧 `--collect`（`scripts/serve_policy.py:513-518` 挂 `openpi.collect.CollectionPolicy` + `EpisodeDataCollector`）在并发 / 多副本下被 `_validate_collect_isolation`（`scripts/serve_policy.py:572-599`）**硬拒绝**：

> `CollectionPolicy` 通过 `register_forward_hook` 给 **共享 base model** 挂 **module-global** 前向钩子（`src/openpi/collect/collection_policy.py:95-99`）。钩子是模块全局的，并发连接（或 batching coordinator 的 worker 线程）里，一条连接的 forward 会触发另一条连接的 hook，导致捕获张量 **交叉污染**，写出的 HDF5 静默损坏；多副本还会在相同 `collect_dir` 文件名上 race（`serve_policy.py:575-583`）。

**新设计的硬约束**：不用 forward hook；复用每请求已持有的 `cached_data` / `cp1_result.query_keys`；只做 interceptor / wrapper 层扩展，不碰 `PI0Pytorch` 与推理内核（WA §2.5 / §3.1）；wire schema 简单、最小、向后兼容；采集必须在 `always_search` 下进行并被文档化（选择偏差约束 C5）。

---

## 1. 现状核实（file:line，均已亲验）

### 1.1 wire 上已有 verdict-agnostic 判决通道 `__hit_meta__`

- `InferenceInterceptor._build_hit_meta(cp1_result)`（`src/openpi/cache/interceptor.py:470-502`）：`cp1_result is None` 时发 MISS 占位；否则发 `{hit_type=cp1_result.hit_type.name, start_t, winner_id=cp1_result.entry_id, cp1_score=cp1_result.score}`；`factor_outputs` 仅在 CompositeJudge `export_factor_outputs=true` 时附加（`:499-501`），否则不出现在 wire 上——**verdict-agnostic**。
- 两个 attach 点（**无 flag guard，无条件挂**）：`interceptor.py:688`（FULL_HIT 短路，在 `_output_transform` `:684` 之后、`clear()` `:689` 之前）与 `interceptor.py:836-838`（MISS / WARM_START / cache-off 主出口，带 None-guard `cp1_result if self._orchestrator is not None else None`）。
- `CheckResult`（`src/openpi/cache/orchestrator.py:56-77`）字段：`hit_type / payload / score / entry_id / query_keys / start_t / factor_outputs`；`query_keys` 注释「filled on all paths」。

### 1.2 `on_episode_end(success)` 现状——success 被丢弃

- `InferenceInterceptor.on_episode_end(self, success: bool)`（`interceptor.py:353-361`）：接收 `success` 但 **未传下去**，只调 `orchestrator.on_episode_end()`（无参）。→ 服务端不保留 success，默认走客户端 join。

### 1.3 `cached_data` / `query_keys` 的生命周期（linchpin，已亲验）

- `KeyBuilder.collect`（`src/openpi/cache/components/key_builder.py:238-245`）把 `s1.state` / `s1.prefix_embs`（GPU）放进 `self._cache`；这是 raw prefix_embs 的来源（**passed stage 参数，非 hook 捕获**）。
- `KeyBuilder.build`（`key_builder.py:247-268`）对每个 field 走 `_to_cpu_float32(t) = t.cpu().float().contiguous()`（`key_builder.py:216-218`）——**独立 CPU float32 拷贝**。`ROBOT_STATE` 走 raw（`:266`），`VISION_*`/`PROMPT_EMB` 经降维后拷贝（`:260/:263`）。
- `orchestrator.check()` 在 `:425` 同步产出 `query_keys` 并塞进每个 `CheckResult`（`:446/:511/:518/:524`）。`orchestrator.clear()` 只清 GPU `_cache`，**不碰** `CheckResult.query_keys` 这个独立 CPU dict。
- **结论**：`cp1_result.query_keys` 在两个 attach 点（`:688` clear 之前、`:836` clear `:821` 之后）**都仍有效**——默认 pooled 路径无需 snapshot / pinning / CP3-overwrite race 处理。

### 1.4 gate-skip 返回（选择偏差 C5 相关）

- `orchestrator.check()` 在 `should_search=False` 时返回 `CheckResult(hit_type=MISS, query_keys=query_keys)`（`orchestrator.py:439-446`），`score=None, entry_id=None`。→ 与「冷启动 always-search MISS」（`:522-526`，`score=top_score` 可能 None，`entry_id=winner_id` 可能 None）**从字段签名上无法区分**。这直接否定「recorder-side 推导 searched」的可行性（见 §2.7）。

### 1.5 per-connection 组件工厂（并发隔离基础）

- `build_per_connection_components`（`src/openpi/cache/config.py:1725-1830`）：并发模式每连接 mint 独立 `key_builder`（`config.py:1734` 明确「key_builder has mutable per-cycle state (_cache) and MUST be per-connection」）、orchestrator、interceptor；仅 vector backend + metadata DB + BatchingCoordinator 共享。
- 已有 per-connection 配置先例：`export_factor_outputs: bool = False`（`config.py:299`）→ 在 `config.py:2156` threaded 进 interceptor。`export_collect_meta` 直接复用此 pattern。

### 1.6 客户端 / conductor 侧现有 per-step recorder 与 success 源

- 通用 writer：`exp/verdict_factor_judge/common/per_step_log_writer.py`（`PerStepWriter` / `PerStepWriterPool`）；顶层 `exp/verdict_factor_judge/per_step_log_writer.py` 已是 **compat shim**（`sys.modules[__name__] = _real`）。`flush_episode()` **当前无参**（`common/...:97-123`），`PerStepWriterPool.finalize()` 合并后按 `_SORT_KEYS=("task_id","subset_init_state_idx","episode_id","step_idx")` **重排**（`:51,:260`）。
- standalone `examples/libero/main.py`：`_rec` 闭包已 `**hit_meta` splat + 写 `{yaml_id, task_id, subset_init_state_idx, orig_init_state_idx, episode_id(=global_episode_id), step_idx, phase}`（`main.py:603-614`）；`infer_recorder(t - num_steps_wait, hit_meta)`（`main.py:305-313`）；`writer.flush_episode()`（`main.py:644 / :877`，**当前无参**）。`global_episode_id = task_id*num_trials + episode_idx`（`main.py:395-405`）——注意这是 **全局计数器，不是 task_uid 的第 4 个分量**。
- conductor：`_hit_row(task, step, hit)`（`examples/libero/episode_runner.py:38-50`）写 `{yaml_id, task_id, episode_idx, orig_init_state_idx, phase, step, hit_type, start_t, winner_id, cp1_score}`——**字段名与 standalone 不一致**（`episode_idx`/`step` vs `subset_init_state_idx`/`step_idx`，无 `episode_id`）。
- `make_task_uid(yaml_id, phase, task_id, episode_idx)`（`src/openpi/conductor/task.py:57-64`）→ `f"{yaml_id}:{phase}:{task_id}:{episode_idx}"`；`episode_idx == subset 位置`。
- `driver.handle_result`（`src/openpi/conductor/driver.py:284-302`）：`result.success` / `result.task_uid` / `result.attempt` / `result.per_step_rows` **同时到达**；`_per_step_rows.extend` 在 `_rows_lock` 下（`:301-302`）。journal 记录 terminal（`:291-298`）。→ 客户端/conductor 侧 success + rows + 完整 identity **已同处一地**。

---

## 2. 设计

### 2.0 总体形态：采集字段 inline 进每步行（无 sidecar / 无跨机 artifact）

[G1-R3 重构，纠正 R2 的共享-FS 架构错误] conductor 跨机、worker 经 TCP 连中央 driver、**无 NFS**（`docs/architecture/experiment_conductor.md:3,18-33`；`task.py:102-104` 明确 `per_step_rows` 中央回传即为「without shared NFS」）。故 worker-local sidecar 的 `emb_ref` 对 driver 不可解析——**废除 sidecar / `GateStepSink` / 跨机 fetch / run-finalize 全部机制**。改为：

- **采集字段直接 inline 进每步行**：`__collect_meta__` 与 `__hit_meta__` 同在**一次 `infer()` 响应**里（per-infer 1:1），客户端 recorder 把 collect 字段（**ndarray→list**，§2.2 B1 codec）merge 进那一行 verdict row——**天然对齐、无需 join key、无 sidecar**。该行经**现有中央通道**回传：standalone 写 JSONL；conductor 把 `collect` 作为**现有自由 dict `per_step_rows` 的一个额外 key**（**[R4-B2] 不 bump `PROTOCOL_VERSION`**）→ driver 中央 JSONL。
- **默认字段 = `robot_state`（~128 B/步，episode 帧 ~32KB）**：极小，inline 对 wire/RAM 无压力、跨机干净——本计划 committed 主路径（gate 训练三流：state + verdict + success）。
- **vision-pooled = opt-in，仅 standalone**：conductor 下 worker 在 episode 末一次性发全 `per_step_rows`、`MAX_FRAME`=64MiB，vision（list 编码后更大）可能超限 → **conductor runner 入口（client 侧，知道运行模式）在 `collect_fields` 含 vision 时 fail-fast**（`validate_cache_config` 只见 YAML、区分不了 standalone/conductor，故此门放 runner 而非 config）；vision 只在 standalone（本地 JSONL、无 frame 限）采集。
- **raw prefix_embs = 本计划 scope 外（deferred）**：多 MB/步 inline 不可行、server-side 跨机 sink 机制复杂且已被证明有洞；由 §8 D4 pooled-充分性 pilot 决定是否**另立后续 L3 计划**补（host-artifact+fetch）。
- 全程纯 per-connection wrapper，无 forward hook，不改 `orchestrator.check()` 控制流。

### 2.1 Hook point（decoupled，无 forward hook，不改 PI0Pytorch / 推理内核）

**PRIMARY（默认，pooled）**：
- 新增静态方法 `InferenceInterceptor._build_collect_meta(cp1_result, fields)`（放在 `_build_hit_meta` 旁，`interceptor.py:470` 区）。它从 `cp1_result.query_keys`（§1.3 已证：独立 CPU float32 拷贝）按 `fields` 序列化精简输入。
- **序列化**：wire packer 是 `msgpack_numpy.Packer()`（`src/openpi/serving/websocket_policy_server.py`，见 §5），处理 `np.ndarray` 但 **不处理 torch.Tensor**。`_build_collect_meta` 发 `{name: _to_wire(name, t) for name,t in cp1_result.query_keys.items()}`：
  - `vision_0/vision_1/vision_2`、`prompt_emb` → `t.to(torch.float16).numpy()`；
  - **`robot_state` 保持 float32**（`t.numpy()`，不降精度）——见 §2.3 granularity 决策。
  - 字段名沿用旧 HDF5 schema（`vision_0/1/2, prompt_emb, robot_state`），保持下游工具连续性。
- **两个 attach 点**，由 per-connection bool `self._export_collect_meta`（默认 False → 键根本不加 → wire byte-identical）守卫：
  - `interceptor.py:688`（FULL_HIT 短路，`_output_transform` `:684` 之后、`clear()` `:689` 之前）；
  - `interceptor.py:836`（主出口，**必须复用现有 None-guard 三元**：`cp1_result if self._orchestrator is not None else None`，否则 cache-off 路径 `cp1_result` 未绑定 → NameError）。
  - 独立 sibling key `outputs["__collect_meta__"]`，**不** 扩展 `__hit_meta__`，使判决 schema byte-identical 且 collect 重载独立开关。
- **占位**：`cp1_result is None`（cache-off）或 `query_keys is None`（未配置 MISS，`orchestrator.py:408`）→ 发 `{"collect": None, "searched": <see below>}`；recorder 容忍缺失字段与 PlaceholderKeyBuilder / partial key dict。**`_build_collect_meta` 内对非 tensor / None 的 field 值做守卫**（degrade 成占位而非抛错，见 §6 T-guard）。

**SERVER + ROUTER：零改动**。`websocket_policy_server` 原样打包 `infer()` 返回 dict（仅追加 `server_timing`）；`replica_proxy` sticky path 逐帧 raw 透传。由现有 `tests/serving/test_websocket_response_hit_meta.py` round-trip 证明。

**raw prefix_embs：本计划 scope 外**（§2.0，deferred）——不再有 server-side sink / `mode∈{raw,both}` 分支。

### 2.2 Per-step schema（verdict-agnostic；一步一行；always_search ⇒ 每步都有）

JSONL verdict row（来自 `__hit_meta__`，不变）+ 客户端 identity：

**IDENTITY / JOIN（唯一 canonical key = `task_uid`）**：
- `task_uid = make_task_uid(yaml_id, phase, task_id, episode_idx)`（`task.py:57-64`）——**唯一权威 join key**。
- 显式规范：`subset_init_state_idx == episode_idx == make_task_uid 第 4 分量`；`episode_id`（`global_episode_id`）是**独立全局计数器，禁止用作 task_uid 分量**（[COMPLETENESS 修复]）。
- **两个 harness 输出同一批 identity 字段名**（[COMPLETENESS 修复]）：`{yaml_id, phase, task_id, subset_init_state_idx, orig_init_state_idx, episode_id, step_idx, task_uid}`。conductor 的 `_hit_row` 从 `{episode_idx, step}` 改为 `{subset_init_state_idx=episode_idx, step_idx=step, episode_id=..., task_uid=make_task_uid(...)}`。
- `step_idx` = 客户端物理步 `t - num_steps_wait`（`main.py:313`），join 用客户端 step_idx + episode identity，**不用服务端 `_step_counter`**。

**VERDICT（verdict-agnostic，ThresholdJudge/kinematic/Composite 皆有）**：
- `hit_type ∈ {FULL_HIT, WARM_START, MISS}`；
- `cp1_score` = `CheckResult.score` = `results[0].score`（top-1 backend 相似度 = 阈值判决的最终分）。**[G1-R2 语义澄清（B8）]：`cp1_score` 可为 `null`——gate-skip、cache-off、以及冷启动/空库的真实 always-search MISS（`orchestrator.py:480` results 空 → score=None）均为 `null`。故不变量是 `searched=True`，**而非**「`cp1_score` 非 null」；下游一律按 `searched` 过滤，不按 `cp1_score` 是否 null。** composite 的合成决策分只在可选 `factor_outputs.composer_score`（v2）。
- `winner_id = entry_id`；`start_t`（仅 WARM_START）。
- **[G1-R2 删除（NB10）]** 原「可选 `cp1_score_top2`」删除：`CheckResult` 不携带 top-2、无生产点，违背 minimal-change；如需 top-2/margin，另立完整 `CheckResult` 接口变更，不在本计划 scope。
- `factor_outputs`（**[G1-R2 修正（B6）] v2 schema `{raw, calibrated, composer_score, schema_version:2}`，`sentinel` 已废除**，见 `composite_judge.py:191-203`）：仅 composite + `export_factor_outputs=true` 时出现，**永不假定存在**。

**SELECTION-BIAS TAG（C5）**：`searched: bool`——见 §2.7（**强制走 `CheckResult.searched`，不走 recorder 推导**）。

**LEAN INPUT（inline，来自 `__collect_meta__`）**：`collect = {robot_state（默认恒带）, vision_0/1/2 / prompt_emb（opt-in，**仅 standalone**）}`。
- **client-boundary codec**：WS 响应经 `msgpack_numpy` 出的是 `np.ndarray`，但 conductor `protocol.encode()` 用**纯 `msgpack.packb`**、`PerStepWriter` 用 `json.dumps`（**都不吃 ndarray**）→ **客户端 recorder 在入 row 前把每个 ndarray `.astype(float32).tolist()`**（robot_state 本已 f32；vision f16→f32 **无损上转**；多维→嵌套 list）。**不做 `.round()`**（`astype(float32).tolist()` 已是 msgpack/JSON-safe 且可复现；避免未定义精度）。
- **唯一 JSONL row schema**：`{ ...verdict(hit_type/cp1_score/winner_id/start_t), ...identity(task_uid/...), searched, [factor_outputs(v2)], collect: {robot_state:[..], vision_*:[[..]]} }`。**直接 merge 进该步 verdict row**（无 sidecar、无 `emb_ref`）。
- **[G1-R4/B2] conductor 下 `collect` 只是现有自由 dict `per_step_rows` 的一个额外 key**，**不 bump `PROTOCOL_VERSION`**（§2.10）。字节测算见 §2.3。

### 2.3 Embedding granularity 决策 + 真实字节测算（[G1-R2 重写，B7]）

**per-frame 字节按 enabled fields × dtype 实算**（推翻原笼统「~16KB」）：

| keybuilder | 单 vision field 维度 | 单 vision(float16) | robot_state(f32) | 「robot_state + 2 vision」帧 |
|---|---|---|---|---|
| `cp1_mean_pool` | 2048 | ~4 KiB | ~128 B | ~8 KiB |
| `cp1_spatial_pool_16`（**GATE 主实验用**） | 32768 | **~64 KiB** | ~128 B | **~128 KiB**（3 vision ~192 KiB） |
| `cp1_spatial_pool_4` | 8192 | ~16 KiB | ~128 B | ~32 KiB |
| `full_original` | 524288 | ~1 MiB | ~128 B | 数 MiB（禁 wire） |

- **默认 inline = `robot_state` only（~128 B/步，恒安全）**；vision/prompt 字段 **opt-in**。
- **per-frame 字节 fail-fast**：`export_collect_meta` wire-up 时按 enabled fields×dtype+list/JSON 编码开销估帧字节，超过阈值（默认 `wire_frame_cap_kib=32`）**fail-fast**。**vision 仅 standalone**（conductor + vision → fail-fast，§2.10）；`spatial16 + vision` 在 standalone 需显式放宽 + bench（按**实际编码后** per-episode frame vs 64 MiB）。
- **`robot_state` 保持 float32**：joint/gripper 是首要 gate 特征，float16 掉精度而省字节可忽略（对齐旧 collector）。
- **raw prefix_embs = 本计划 scope 外（§2.0）**：GATE 运行时门控实际读 `cached_data`（raw prefix_embs + state，`orchestrator.py:419`），非 pooled key；是否需 raw 由 **pooled 充分性 pilot（§8 D4）** 决定，若需则另立后续 L3。bench 覆盖 spatial16 与最大允许 inline 配置的 frame-size / 延迟 / driver 内存（§6 T-BENCH、§8 D1）。

### 2.4 Recorder（无 sidecar；collect 字段 inline 进行）

- 新增 `src/openpi/serving/per_step_recorder.py`：`PerStepWriter/PerStepWriterPool` 从 `exp/verdict_factor_judge/common/per_step_log_writer.py` 泛化为 schema-agnostic recorder（任意 dict row、可配 sort keys、per-worker temp lock-free + finalize-merge 按 identity 排序、export filter 按 `searched`）。**collect 字段与 verdict 同在一行 dict，无独立 embedding 文件、无 `emb_ref`。**
- **[G1-R3/B3] incomplete-episode mode 显式进接口**：`PerStepWriter(path, stamp_success: bool = False)` + `begin_episode()`/`flush_episode(success=...)`/`close()`：
  - `stamp_success=False`（**shim 模式**）：`flush_episode()` 无参**不注入** `success` 键、`close()` 不注入 → 既有 verdict_factor 测试 byte-identical。
  - `stamp_success=True`（**gate 模式**）：`flush_episode(success)` 盖每行、`close()` 对 in-flight episode tail-flush `success=null`。
  - 两模式均有单测（§6 T-RECORDER）。
- `exp/verdict_factor_judge/common/per_step_log_writer.py` → 降为 thin compat shim（`stamp_success=False`，钉字段名 + `_SORT_KEYS`；行为不变，§2.8）。

### 2.5 Per-episode / success join（episode 边界 buffer-and-stamp，绝非 per-request attach）

success 是 episode-level 事实，仅在 `episode_end` 且所有 per-step row 之后才知道。在 **客户端 / conductor** join（success + rows + 完整 identity 同处一地）：

- **standalone `examples/libero/main.py`**：`done` 在 `_run_episode` 后已知（`main.py` 中 `done, ... = _run_episode(...)`），在 `writer.flush_episode()`（`main.py:644/:877`）**之前**。改为 `writer.flush_episode(success=done)` → 单次批写前把 `success` 盖进每行 buffered row。可退休对 `episode_results.json` 的离线 join（`main.py:662-672` 区）——JSONL 自包含。
- **conductor `driver.handle_result`（`driver.py:284-302`）**：`success` + `per_step_rows` + `task_uid` + `attempt` 同 key 到达。在 `_per_step_rows.extend(...)`（`:300-302`）之前：
  ```
  for r in result.per_step_rows:
      r.setdefault("success", result.success)
      r.setdefault("task_uid", result.task_uid)
      r.setdefault("attempt", result.attempt)   # [COMPLETENESS 修复] dedup 需要
  ```
  单一中心站点覆盖所有 worker/replica，在既有 `_rows_lock` 下无需额外锁。journal（`journal.py`）是权威 terminal-success 交叉校验 / 回填。
- **INCOMPLETE EPISODE**：mid-episode 断连不发 `episode_end`；`PerStepWriter.close()`（gate 模式）tail-flush `success=null`（显式未知），**绝不 False**。collect 字段已在每行（inline），success 客户端/driver 侧盖章即可，**无服务端 success 需求**（raw 已移出 scope）。
- **STALE-ATTEMPT DEDUP**：driver 对被取代的 attempt 也无条件 append；离线 join/dedup key = `(task_uid, step_idx, attempt)`，取 journal terminal attempt。

**Provenance**：per-episode 级 `{task_uid, phase, seed, num_steps, success|null, searched_all, collect_fields, kb_id, collector_schema_version}` 作为一行 episode-summary 落 JSONL（或客户端 journal），非独立 sidecar header。

### 2.6 Concurrency 机制（结构性隔离）

- 每连接经 `_connection_policy_factory` / `_bind_bundle` 懒建自己的 interceptor + orchestrator + key_builder（`build_per_connection_components`，`config.py:1725-1830`；仅 vector backend + metadata DB 共享）。
- `cp1_result` 是 per-request local（`interceptor.py:658`），`_build_collect_meta` 只读该 local 的独立 CPU `query_keys`。**无 module-global 可变状态**（正是 forward-hook 旧模式所缺）。
- 记录骑 per-connection socket 原样回传（opaque passthrough）；连接 loop 每次 `infer` 后才 recv（每连接严格串行）。
- BatchingCoordinator 在共享线程批处理 stage forward，但把 per-request 分片返回给提交线程；**emission 在分片返回后、per-connection interceptor 线程上跑，绝不在 coordinator worker 内**。coordinator 输出是 batched tensor 的 view，但 `query_keys` 在 `check()` 内（`:425`）就 `.cpu()` 拷贝，早于 emission，无跨 batch overwrite race。
- 多副本 = 独立进程、connection-sticky（`replica_proxy`）；recorder 文件 per-worker/per-pid，host-level finalize-merge。
- **[CONCURRENCY 不变量]**：默认 pooled 路径**只**发 `build()` 已 CPU 化的 dict，绝不把 `.cpu()` 推迟进 coordinator worker、绝不发 GPU view。设为显式测试断言（emitted embedding tensor 必须 CPU + 独立 storage），防回归。

### 2.7 Decoupling 论证（WA §2.5 / §3.1）

- **不改 PI0Pytorch**：`src/openpi/models_pytorch/pi0_pytorch.py` 不在 files-touched；interceptor 仅 by-reference 持 `self._model`；`_build_collect_meta` 是 `@staticmethod` reader。
- **无 forward hook**：全库唯一 `register_forward_hook` 是被拒的旧 collect（`collection_policy.py:95-99`）；新设计零 hook。raw 路径的 `prefix_embs` 是 passed stage 参数（`key_builder.py:243` ← `orchestrator.py:415` ← `interceptor.py:650`），非 hook 捕获。
- **[DECOUPLING 修复 / 框架澄清]**：可选的 `CheckResult.searched`（`orchestrator.py:56-77` + gate-skip `:446`）是 **cache-subsystem（wrapper 层）编辑，不是 PI0Pytorch/推理内核编辑**，在 WA §2.5/§3.1 范围内；`orchestrator.check()` 控制流零改动。（raw-path / server-side sink 已移出 scope，§2.0——本节不再涉及。）
- **`searched` 强制走 CheckResult 字段，拒绝 recorder 推导**（[BACKWARD-COMPAT/TEST 修复，关键]）：recorder-side 推导 `not (query_keys is not None and score is None and entry_id is None)` 在冷启动 / 空库的 **真实 always-search MISS**（`orchestrator.py:522-526`：`score=None`（`:480` results 空）、`entry_id=None`）会把一个 **真搜过** 的步误标为 `searched=False`，恰好制造 C5 选择偏差，且高发于 episode 起始。故：
  - **强制**给 `CheckResult` 加 `searched: bool = True`（`orchestrator.py:56-77`，additive、向后兼容），**仅在 gate-skip 返回**（`orchestrator.py:446`）置 `False`；`_build_collect_meta` 原样拷贝。
  - recorder-side 推导 **作为 rejected 方案记录在案**（会误标冷启动 MISS），不作为默认或 fallback。
  - 每行仍带 `searched`、episode-summary 带 `searched_all`，任何非 always-search 行可过滤、绝不被静默当 verdict 标签。

### 2.8 向后兼容（wire + flush_episode + shim）

- **wire**：`__collect_meta__` 是 sibling key、`export_collect_meta` 默认 False → 关闭时根本不出现 → wire byte-identical。`msgpack_numpy` 处理 float16 ndarray（kind 'f'），不处理 torch.Tensor——故序列化必须先 `.numpy()`。
- **[BACKWARD-COMPAT 修复，关键] `flush_episode` 条件盖章**：`flush_episode(self, success=<sentinel>)` **仅当显式传入 success** 时才注入 `success` 键；**无参调用绝不注入** `success` 键。使 `tests/exp/verdict_factor_judge/test_per_step_writer.py:71,:173`（断言 `parsed == rows`，无参 flush 后 rows 不变）**保持 green 且不改**。泛化 recorder 与 compat shim 都遵此契约；新增 shim-regression 断言：`flush_episode()`（无 success）产出 rows 与输入 byte-identical（无注入键）。
- **config off-by-default**：`CacheConfig.collection.export_collect_meta=False`（默认 → wire byte-identical；归属修正见 §2.9——**非**镜像 per-checkpoint `JudgeConfig.export_factor_outputs`）；客户端 `--collect-gate-dir` 默认 None；新 interceptor ctor kwargs 全有默认值，既有构造点不破。

### 2.9 Config surface（[G1-R2 修正归属，B5]）

**[归属修正]** `export_factor_outputs`（`config.py:299`）实为 **per-checkpoint `JudgeConfig`** 字段，**不是**全局属性；采集开关**不镜像其位置**，改为**新增顶层 `CacheConfig.collection` dataclass**（采集是跨 checkpoint 的横切关注点）：

1. **SERVER（`CacheConfig.collection`，新增 dataclass）**：
   - `export_collect_meta: bool = False`（inline collect 总开关，默认关 → wire byte-identical）；
   - `collect_fields: list[str] = ["robot_state"]`（默认只 robot_state；加 vision 需显式列，受 `wire_frame_cap_kib` fail-fast）；
   - `wire_frame_cap_kib: int = 32`。
   threaded 过 `build_per_connection_components`（`config.py:1725-1830`）进每连接 interceptor。（无 raw_sink 字段——raw 已移出 scope，§2.0。）
2. **[G1-R3/NB] CLI override 为 tri-state `None | T`**（`serve_policy.py`）：`--export-collect-meta`/`--collect-fields` 默认 `None`，**仅非 None 时覆盖 YAML**（区分「未传」与「显式传默认值以覆盖」）。
3. **CLIENT recorder（`examples/libero/main.py`）— [R4-B4] canonical CLI**：`--collect-gate-dir <path>`（**canonical**，启用 recorder，默认 None）；旧 `--per-step-log-dir` 降为 **deprecated alias**（映射同一 writer + deprecation warning）；**双 flag 同时出现 → fail-fast**。`--collect-embeddings {pooled,none}`（须与服务端 `collect_fields` 匹配，mismatch fail-fast）。verdict-factor runner（`common/run_phase.py`）改用 canonical flag；`factor_outputs`(v2) 仍随 `__hit_meta__` 进同一行（§2.11），旧实验记录不中断。
4. **SELECTION-BIAS 硬门（C5）**：采集 **always_search-ONLY**。`validate_cache_config` 在 `export_collect_meta` 为真时 **fail-fast** 检查 CP1 gate 为 always-search（硬门）；同时校验 `collect_fields` 帧字节 ≤ `wire_frame_cap_kib`。每行带 `searched`、episode-summary 带 `searched_all`。

### 2.10 Conductor 传输（[G1-R3/R4 重写]：collect 作 `per_step_rows` 额外 key，不 bump、robot_state-only）

conductor 跨机、**无 NFS**（`task.py:102-104`；`experiment_conductor.md:18-33`）→ worker-local sidecar 不可解析（R2 架构错误）。故：
- **collect（默认 robot_state）作 `per_step_rows` 的额外 key**：`episode_runner.py:_hit_row`（`:38-50`）把 `__collect_meta__`（经 §2.2 B1 codec 转 list）merge 进该步 row。**[R4-B2] 不 bump `PROTOCOL_VERSION`**——`per_step_rows` 已是自由 dict，`protocol.encode/decode` 与 `EpisodeResult` 无需改；`decode()` 对 version 严格相等，bump 会拒 v1，故**只加 key、不动 version**。
- **[R4-B2] driver / writer 契约**：collect 随现有 rows 走既有 `driver.handle_result`→`per_step_rows` 累积→**现有 stage-complete writer `per_step_writer(yaml_id, rows)` 一次性写**中央 JSONL，**`per_step_rows` property 语义不变**；不引入 per-episode 覆盖写、**删掉原"增量 drain"承诺**（未实现）。robot_state ~128 B/步、episode 帧 ~32KB，累积 RAM 可忽略。
- **vision 在 conductor 下 fail-fast（放 runner 边界）**：worker 在 episode 末一次性发全 `per_step_rows`（`MAX_FRAME`=64MiB），vision 可能超限 → **conductor runner 入口**（`episode_runner`/conductor run 脚本，知道运行模式）在 `collect_fields` 含 vision 时 fail-fast；config-only 的 `validate_cache_config` 看不到运行模式，不承担此门。vision 仅 standalone。
- **无 worker sidecar、无 server-side sink、无跨机 fetch、无 run-finalize**（R2 方案已废）。

### 2.11 R1/R2/R3 — Owner 明确要求：整合 / 删除 / 共存

（owner 追加的三条硬要求，G1 逐条审。R1 删除清单与 R3 对照由独立 minimal-diff 分析给出，与本设计交叉验证一致。）

**R1 — 把「每步 factor 分数」整合进新采集器，替换旧的 verdict 专属通路。**
- **整合 = 零新增采集代码**：`factor_outputs`（**[G1-R2/B6] v2 schema `{raw, calibrated, composer_score, schema_version:2}`，`sentinel` 已废除**，见 `composite_judge.py:191-203`）早已是 `__hit_meta__` 的可选成员（`interceptor.py:494-501`），随每步 verdict row 落盘。新采集器每步 JSONL row 即「被拓宽的同一张行」——同时承载 verdict + 可选 `factor_outputs`(v2) + inline 的精简输入；任何 composite verdict 的 per-factor 分数自动进入新采集，无需第二条通路。**shim 注释与 regression test 一律按 v2 校验，不得重新固化 v1。**
- **删除 / 去专属化清单**（recorder 机制本体被泛化复用，删的是 verdict_factor 专属外壳）：
  - `exp/verdict_factor_judge/per_step_log_writer.py`（顶层 7 行 compat shim）→ 改指向 `src/openpi/serving/per_step_recorder.py`（弃用期后删，D7）。
  - `exp/verdict_factor_judge/common/run_phase.py::_summarize_per_step_log`（`:338`）及 `--per-step-log-dir` 的 factor-审计专用汇总（`:239-240/:478`）→ 删 / 被通用 gate 分析取代。
  - `PerStepWriter/Pool` 本体 → 泛化到 `src/openpi/serving/per_step_recorder.py`（§2.4）；`exp/verdict_factor_judge/common/per_step_log_writer.py` 降为 thin compat shim（§2.8 契约：`flush_episode()` 无参不注入 `success` 键，既有测试 green 且不改）。
- **[D7 已定] compat shim 处置**：保留 deprecated 薄 shim（`stamp_success=False`）一个发布周期后删——`stamp_success=False` 模式保既有 verdict_factor 测试 byte-identical（§2.4/§2.8），弃用期后清除旧通路（owner R1 可要求更激进，§8）。

**R2 — 最老的 `--collect`（forward-hook / 模型内部产物 / 非并发）保留独立，不合并、不删除。**
- `openpi.collect.CollectionPolicy` + `EpisodeDataCollector`（`serve_policy.py:513-518`；hook `collection_policy.py:95-99`；HDF5 `data_collector.py`）**明确划到本设计 scope 外**：它抓深层模型内部（projector 输出 / `embed_tokens` / `action_in/out_proj`），本精简采集器**刻意不抓**。二者**不可同服并存**（`--collect` 被 `_validate_collect_isolation` 锁死单连接单副本；新采集器并发原生），用法上互斥启用、各取所需。本计划**不碰** `_validate_collect_isolation` 与旧 `--collect` 的任何代码（§3 已在 `serve_policy.py` 行注明）。

**R3 — 新旧对照表（防误删老 collect / 防重复造轮子）。**

| 维度 | 旧 A：verdict-factor per-step recorder | 旧 B：`--collect`(CollectionPolicy) | 新：本计划 gate 采集 |
|---|---|---|---|
| 处置 | 被取代（拓宽为通用行 + 删专属 shim/汇总，见 R1） | 共存（独立、不动） | 本设计 |
| 采集内容 | verdict + factor_outputs | 深层模型内部(HDF5) | verdict + 可选 factor_outputs + 精简模型输入(默认 robot_state，vision opt-in) |
| 机制 | `__hit_meta__` wire + client JSONL | 模块级 forward hook | 复用 `__hit_meta__` + `__collect_meta__` sibling **inline 进同一行** + 泛化 recorder |
| 并发 | 支持 | 拒绝(单连接单副本) | 支持(per-connection、无 hook) |
| success | 离线 join | HDF5 attr | 客户端 / conductor flush 盖章 + 离线 join 兜底 |

---

## 3. Files touched（改什么 + 新增/修改）

| 文件 | 新增/修改 | 内容 |
|---|---|---|
| `src/openpi/cache/interceptor.py` | 修改 | 加静态 `_build_collect_meta(cp1_result, fields)`；`:688`/`:836`（保留 None-guard 三元）挂 `outputs["__collect_meta__"]`，由 `self._export_collect_meta` 守卫；ctor 加 `export_collect_meta`/`collect_fields`。robot_state f32、vision f16；非 tensor/None 守卫成占位。**无 sink / 无 lifecycle 转发**（raw 已移出 scope）。 |
| `src/openpi/cache/orchestrator.py` | 修改（最小） | `CheckResult`（`:56-77`）加 `searched: bool = True`；gate-skip 返回（`:446`）置 `False`。控制流零改动。 |
| `src/openpi/cache/config.py` | 修改 | **[B5] 新增顶层 `CacheConfig.collection` dataclass**（`export_collect_meta`/`collect_fields`/`wire_frame_cap_kib`；**非**镜像 per-checkpoint `JudgeConfig.export_factor_outputs`）；threaded 过 `build_per_connection_components`；`validate_cache_config` 加 always-search 硬门 + 帧字节 ≤ cap 校验。 |
| `scripts/serve_policy.py` | 修改 | thread `collection` 进 interceptor；加 `--export-collect-meta`/`--collect-fields`（**tri-state `None\|T`，NB**）。**不碰 `_validate_collect_isolation`/旧 `--collect`。** |
| `src/openpi/serving/per_step_recorder.py` | **新增** | 泛化 `PerStepWriter/Pool`（schema-agnostic；`PerStepWriter(path, stamp_success)` **两模式(B3)**；`begin_episode/flush_episode(success)/close` tail-flush null；per-worker temp lock-free + finalize-merge；export filter 按 `searched`）。**无 `EmbeddingSidecarWriter`**（collect inline）。 |
| `src/openpi/conductor/driver.py` | 修改 | `handle_result`（`:300-302`）stamp `success`+`task_uid`+`attempt` on rows（`collect` 已在 row 内）；dedup `(task_uid, step_idx, attempt)`。**[R4-B2] 只加 row key、不 bump `PROTOCOL_VERSION`**，`task/protocol/worker.py` 无需改；保持现有 stage-writer + `per_step_rows` property 语义。 |
| `examples/libero/episode_runner.py` | 修改 | `_hit_row`（`:38-50`）规范字段名（`subset_init_state_idx`/`step_idx`/`episode_id`/`task_uid`）+ **merge `__collect_meta__`（ndarray→list）inline 进 row**；**conductor 入口 `collect_fields` 含 vision → fail-fast**（运行模式感知门，非 `validate_cache_config`）。 |
| `examples/libero/main.py` | 修改 | `infer_recorder` **callback 加 `collect_meta` 形参**（`:305-313`）→ ndarray→list 后 inline 进 JSONL row（`:603-614`）；每 episode `begin_episode()` → `flush_episode(success=done)`（`:644/:877`）（B5 wiring）；canonical `--collect-gate-dir` + `--per-step-log-dir` **deprecated alias** + 双 flag fail-fast（B4）；`--collect-embeddings`；补 `task_uid`；`episode_results.json` 过渡保留（D5）。 |
| `exp/verdict_factor_judge/common/per_step_log_writer.py` | 修改 | 降为 thin compat shim（`stamp_success=False`；`flush_episode()` 无参 byte-identical；既有测试 green 且不改）。 |
| `exp/verdict_factor_judge/per_step_log_writer.py` | 修改（[R1/D7]） | 顶层 shim 改指向 `src/openpi/serving/per_step_recorder.py`；弃用期后删。 |
| `exp/verdict_factor_judge/common/run_phase.py` | 修改（[R1/B4]） | 删 `_summarize_per_step_log`（`:338`）+ factor-审计专用汇总（`:239-240/:478`）；**改用 canonical `--collect-gate-dir`**（旧 `--per-step-log-dir` alias），**保留产 v2 `factor_outputs`**（regression test，§6）。 |
| `docs/architecture/cache_system.md` | 修改（**[B1/L3]**） | §5.13 wire-observability 补 `__collect_meta__` schema（inline）；§5.1/`CheckResult` 补 `searched`。 |
| `docs/architecture/experiment_conductor.md` | 修改（可选） | 记 `per_step_rows` dict 可携带额外 `collect` key（**无 protocol version 变更**；vision 仅 standalone）。 |
| `docs/data_collection/guide.md` + `docs/README.md` | 修改 | guide 记 inline lean schema、always_search-only、robot_state-default + vision opt-in + 字节 cap、raw deferred；**`docs/README.md` 同步索引（文档变更红线）**。 |
| `tests/serving/test_websocket_response_hit_meta.py` | 修改（扩展） | `__collect_meta__` round-trip + 非采集客户端兼容。 |
| `tests/`（新增若干，见 §6） | **新增** | recorder **两模式** / `_build_collect_meta` / driver stamp+dedup+inline collect / **跨-FS conductor 回收(无 NFS 双 temp root) / 2-conn 并发 / config+CLI tri-state / 帧字节 fail-fast / v2 schema regression**。 |

---

## 4. Interfaces introduced/modified（签名级）

```python
# src/openpi/cache/interceptor.py
class InferenceInterceptor:
    def __init__(self, ..., export_collect_meta: bool = False,
                 collect_fields: tuple[str, ...] = ("robot_state",)): ...
    @staticmethod
    def _build_collect_meta(cp1_result, fields) -> dict:
        # returns {"collect": {name: np.ndarray}|None, "searched": bool}
        # robot_state float32, vision_*/prompt_emb float16; None/non-tensor -> placeholder

# src/openpi/cache/orchestrator.py
@dataclass
class CheckResult:
    ...
    searched: bool = True   # 新增；仅 gate-skip 返回置 False

# src/openpi/serving/per_step_recorder.py  (NEW) — [G1-R3/B3] explicit two-mode
_MISSING = object()
class PerStepWriter:
    def __init__(self, path, stamp_success: bool = False): ...   # False = shim mode (never inject "success")
    def begin_episode(self) -> None: ...
    def write_row(self, row: dict) -> None: ...                  # row = verdict + identity + inline collect
    def flush_episode(self, success=_MISSING) -> int:
        # shim mode / success=_MISSING -> DO NOT inject "success" (byte-identical)
        # gate mode -> stamp row["success"]=success on every buffered row
    def close(self) -> None: ...                                 # gate mode: in-flight -> tail-flush success=null
class PerStepWriterPool:
    def __init__(self, out_dir, yaml_id, num_workers, stamp_success=False,
                 sort_keys: tuple[str,...] = _DEFAULT_SORT_KEYS): ...
    def writer_for(self, worker_id: int) -> PerStepWriter: ...
    def finalize(self) -> pathlib.Path: ...   # merge + sort by sort_keys

# client recorder callback (examples/libero) — signature bumped:
#   infer_recorder(step_idx, hit_meta, collect_meta)   # was (step_idx, hit_meta); collect merged into the row

# src/openpi/conductor — per_step_rows dicts carry an extra "collect" key (list-encoded, §2.2 B1 codec);
#   NO PROTOCOL_VERSION change (free-form rows; strict decode() would reject a bumped version).
# src/openpi/conductor/task.py (existing) — make_task_uid(yaml_id, phase, task_id, episode_idx) -> str
```

**兼容契约**：`stamp_success=False`（shim 模式）+ `flush_episode()` 无参 = byte-identical（不注入 `success`）；`export_collect_meta=False` = wire byte-identical。

---

## 5. Integration points

- **服务端 emission**：`orchestrator.check() → cp1_result(query_keys, searched)` → `interceptor.infer()` 在 `:688/:836` 调 `_build_collect_meta` → `outputs["__collect_meta__"]` → `websocket_policy_server` msgpack_numpy 打包 → `replica_proxy` sticky 透传 → 客户端。
- **客户端记录（完整 lifecycle wiring，[R4-B5]）**：每 episode 开始 `writer.begin_episode()`（tail-flush 正确性前置，standalone 必调）→ 每步 `client.infer()` 返回 `__hit_meta__`+`__collect_meta__` → `infer_recorder(step_idx, hit_meta, collect_meta)`（collect **ndarray→list**（§2.2 B1）后 inline 进该行：verdict + identity + searched + collect）→ `write_row` → episode 末 `flush_episode(success=done)`（standalone）/ driver stamp（conductor）；`close()` 对未 flush 的 in-flight tail-flush `success=null`。
- **conductor 中央回传**：`_hit_row` 把 collect（ndarray→list）merge 进 `per_step_rows` 的**额外 key（不 bump `PROTOCOL_VERSION`）**→ `driver.handle_result` stamp success/task_uid/attempt → 现有 stage-writer 落中央 JSONL（跨机、无 NFS）；vision 仅 standalone。
- **success**：standalone `flush_episode(success=done)`；conductor driver stamp。
- **wire-up 硬门**：`validate_cache_config` 在 `export_collect_meta=True` 时 fail-fast 检查 always-search + 帧字节 ≤ cap。
- **join 权威**：`conductor/journal.py` terminal-success 交叉校验；dedup `(task_uid, step_idx, attempt)`。

---

## 6. Test strategy（unit + 并发 integration + manual e2e；含 always_search 选择偏差）

1. **T-WIRE round-trip**：`__collect_meta__`（f16 vision/prompt + f32 robot_state numpy dict）经 msgpack_numpy pack/unpack 完整、与 `__hit_meta__` 共存；`export_collect_meta=False` → wire byte-identical。
2. **T-NONCOLLECT-CLIENT 兼容**：服务端 `export_collect_meta=True` 时，忽略 `__collect_meta__` 的旧客户端 `actions`/`state` 仍可访问不变。
3. **T-EMISSION unit**：`_build_collect_meta`：robot_state f32、vision f16；`cp1_result=None`/`query_keys=None` → 占位；`searched=False` 恰在 gate-skip、`True` 于所有真实 verdict（含 WARM→MISS 降级 `:510-514`、冷启动空库 MISS `:522-526` **不被误标 False**）。
4. **T-GUARD**：非 tensor/None field → degrade 占位不抛。
5. **T-CPU-INVARIANT**：emitted embedding tensor CPU + 独立 storage（不与 GPU cached_data 共享）。
6. **T-RECORDER 两模式（B3）**：`stamp_success=False`（shim）→ `flush_episode()` 无参 rows byte-identical、无 `success` 键；`stamp_success=True`（gate）→ `flush_episode(success)` 盖每行、`close()` in-flight tail-flush `success=null`；多 worker finalize-merge 重排后行完整。
7. **T-SUCCESS-JOIN-CONDUCTOR**：`driver.handle_result` stamp `success`+`task_uid`+`attempt`；stale attempt 不覆盖 terminal（dedup `(task_uid, step_idx, attempt)`）。
8. **T-SUCCESS-JOIN-STANDALONE**：`flush_episode(success=done)` 盖章；断连 tail-flush `success=null`；`episode_results.json` **过渡保留**交叉校验（D5）。
9. **T-CONCURRENCY**：2 条并发 WS 连接喂 **不同** dummy stage1 → 各连接返回 collect = 自己输入（无 A↔B bleed）；合并 JSONL 干净分区；`--replicas` per-worker 文件不撞。
10. **T-CROSS-FS conductor 回收（[G1-R3/B1]，关键）**：worker 与 driver 置于 **两个不共享的临时根目录**，验证 collect（robot_state）经 `per_step_rows` 额外 key 中央回传后 driver 中央 JSONL 仍可完整回收/join（证明无 NFS 依赖）；此路径仅 robot_state（vision 在 conductor fail-fast，见 T14c）。
11. **T-SELECTION-BIAS（C5）**：always_search → 每行 `searched=True`（`cp1_score` 可 null）；skip → `searched=False` 被 export filter 排除；非 always-search + `export_collect_meta` → `validate_cache_config` fail-fast。
12. **T-CONFIG/CLI-PROP（B7 帧字节门 + NB tri-state）**：`CacheConfig.collection` YAML→per-connection 正确 thread；CLI tri-state（`None|T`）仅非 None 覆盖 YAML；`collect_fields` 帧字节 > `wire_frame_cap_kib` → fail-fast（含 spatial16 用例）。
13. **T-VERDICT-AGNOSTIC**：纯 ThresholdJudge 记录完整、`factor_outputs` 缺席。
14. **T-FACTOR-V2-REGRESSION（B6）**：composite + `export_factor_outputs` → v2 `{raw, calibrated, composer_score, schema_version:2}`、无 `sentinel`；shim 断言不固化 v1。
14b. **T-PARITY 端到端序列化（[R4-B1]，关键）**：WS `msgpack_numpy` ndarray → 客户端 codec ndarray→list → row → conductor 纯 `msgpack.packb` → driver → `json.dumps` JSONL；断言 dtype/shape/**value parity**（robot_state f32 值一致，vision f16 舍入后一致）。
14c. **T-CONDUCTOR-VISION-FAILFAST**：conductor runner 入口 + `collect_fields` 含 vision → runner fail-fast；robot_state-only 通过；standalone + vision 通过（无 frame 限）。
14d. **T-VERDICT-FACTOR-RUNNER-V2（[R4-B4]）**：现有 verdict-factor runner 走 canonical `--collect-gate-dir` 仍产 v2 `factor_outputs` 行；`--per-step-log-dir` alias 等价；双 flag → fail-fast。
15. **T-E2E manual smoke**：真实 `serve_policy --export-collect-meta` + `libero --collect-gate-dir` 极小 always_search：(a) 每行 `searched=True`（cp1_score 可 null）、(b) 完成 episode 每行盖 `success`、(c) 每行含 inline collect、(d) 非 always-search → fail-fast。
- **T-BENCH（B7，manual）**：spatial16 与最大允许 inline 配置 frame-size / 闭环延迟 / driver 内存实测，定 `wire_frame_cap_kib`（§8 D1）。

> **[WA §6 / §2.7]** Verify 跑**全量非 manual suite**（`uv run pytest`）须全绿；manual（T-E2E/T-BENCH）触及路径时本地跑并附日志。

---

## 7. Risk register

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **帧字节被低估 + list/JSON 编码更大（B7/R4-B3）**：spatial16 单 vision f16 ~64KiB、3 vision ~192KiB，list 编码后更大；worker 一次性发全 episode `per_step_rows`（`MAX_FRAME`=64MiB） | 默认仅 `robot_state`（~128B/步，episode 帧 ~32KB）；**vision opt-in 且仅 standalone，conductor 下 fail-fast**；`wire_frame_cap_kib` fail-fast；bench 按**实际编码后** per-episode frame vs 64 MiB（§6 T-BENCH）。 |
| 2 | **`searched` 正确性** | **强制** `CheckResult.searched` 字段（`:56-77`+`:446`）；recorder 推导已 rejected（误标冷启动 MISS）。 |
| 3 | `factor_outputs` composite-only/可选 | 永不假定存在（`interceptor.py:499-501`）。 |
| 4 | `cp1_score` 语义 = top-1 搜索相似度（非合成决策分）；gate-skip/cache-off/冷启动空库 MISS **均为 null（B8）** | 文档明确不变量是 `searched`（非「cp1_score 非 null」）；composite 合成决策分在 `factor_outputs.composer_score`(v2)。 |
| 5 | `step_idx` 客户端定义 | join 用客户端 `step_idx` + episode identity，不用服务端 `_step_counter`。 |
| 6 | crash 至多丢 in-flight episode（per-episode commit） | 继承旧 writer 行为、可接受。 |
| 7 | **多副本文件 race** | 客户端 recorder per-worker temp + finalize-merge；conductor 经中央 driver JSONL（无 per-host 文件 race）。 |
| 8 | **join-key 分歧**（两 harness 字段名不一致；`episode_id` vs `episode_idx`） | 唯一 canonical `task_uid`；规范化两 harness 字段名；`episode_id`（全局）禁作 task_uid 分量。 |
| 9 | **conductor 跨机传输（无 NFS）**（[G1-R3/B1]：R2 worker-sidecar 架构错误） | collect **inline 进 `per_step_rows`** 中央回传（robot_state ~128B 无压力）；R2 共享-FS sidecar 已废；跨-FS 双 temp-root 测试（§6 T10）证明无 NFS 依赖。 |
| 10 | **backward-compat**：`flush_episode` 注入 `success` 破坏既有 verdict_factor 测试 | `PerStepWriter(stamp_success=False)` shim 模式无参不注入键；shim 行为不变；既有测试 green 且不改。 |
| 11 | **robot_state 精度回归** | robot_state 保持 float32（仅 vision/prompt float16）。 |
| 12 | **pooled 对 gate 是否充分**（运行时门控读 raw `cached_data`，非 pooled key） | 全量 pooled-only 前跑 **pooled 充分性 pilot**；不足则另立后续 raw L3（§8 D4），本计划不含 raw。 |
| 13 | **vision-inline 超 64MiB frame（opt-in，R4-B3）** | vision **仅 standalone**（本地 JSONL 无 frame 限）；conductor + vision → `validate_cache_config` fail-fast；bench 按**实际编码后** per-episode frame（§6 T-BENCH）。 |
| 14 | **dedup 可实现性**：`per_step_rows` 原无 `attempt` | driver `handle_result` stamp `attempt`（`result.attempt`，`driver.py:288`）。 |
| 15 | 选择偏差 C5 | always_search-only，**fail-fast 硬门** + `searched`/`searched_all` 双标。 |
| 16 | **是否需 protocol bump（R4-B2）** | **不 bump**——`collect` 只是自由 dict `per_step_rows` 的额外 key；`protocol.encode/decode` 与 `EpisodeResult` version 不变（strict `decode()` 会拒 v1）；`task/protocol/worker.py` 无需改。 |
| 17b | **ndarray 无法过 msgpack/JSON（R4-B1）** | 客户端 boundary codec：ndarray→list（robot_state→`list[float]`，多维嵌套）后入 row；端到端 parity 测试（§6 T-PARITY）。 |
| 17 | **L3 架构文档漏更（B1）** | files-touched 含 `docs/architecture/cache_system.md` + `experiment_conductor.md` + `docs/README.md` 索引同步。 |

---

## 8. 决策定稿 + 待 G1 确认（[G1-R2，B9]）

原「未决问题」按 reviewer 要求定稿为唯一决策（executor 最终裁量，owner 可否决），files-touched / 测试已随之同步：

- **D1 — inline 帧字节 / bench**：默认 inline 仅 `robot_state`（conductor+standalone）；**vision opt-in 且仅 standalone**（conductor+vision → fail-fast）。`wire_frame_cap_kib=32` fail-fast；bench 按**实际编码后** per-episode frame vs 64 MiB（无"步采样"依赖）。
- **D2 — `CheckResult.searched`**：定为**强制 additive 字段**（`searched: bool = True`，gate-skip 置 False）；recorder 推导已否决（误标冷启动 MISS）。属 cache-subsystem 编辑、非 PI0Pytorch。
- **D3 — 无 sidecar（collect inline）**：collect 字段直接 inline 进每步行（standalone JSONL / conductor `per_step_rows`）；**废弃 R2 的 `.npz` sidecar 与 `emb_ref`**。
- **D4 — pooled 充分性 pilot**：先跑 pooled 充分性 pilot（小规模）验证 pooled 是否足够作 gate 特征；**若不足，raw prefix_embs 另立后续 L3 计划**（host-artifact+fetch），不在本计划 scope。
- **D5 — `episode_results.json`**：过渡保留作 success 冗余交叉校验；**删除条件（[R4-NB] 可验证终点）= 所有 runner 迁到 JSONL success + 连续一次完整 Verify 通过**（非模糊「发布周期」），由跟踪任务记录。
- **D6 — conductor collect**：`collect` 作 `per_step_rows` **额外 key（不 bump `PROTOCOL_VERSION`）**，跨机无 NFS（§2.10）；robot_state-only（vision fail-fast）；无 worker sidecar / 无 server sink。
- **D7 — R1 旧 recorder 处置**：保留 deprecated 薄 compat shim（`stamp_success=False`）；**删除条件（[R4-NB]）= 所有 verdict-factor runner 迁到 canonical `--collect-gate-dir` + 连续一次完整 Verify 通过**，由跟踪任务记录（顶层 shim 改指向新模块 + `common/run_phase.py` 删 factor 汇总，已同步 §3）。
- **D8 — CLI override tri-state（NB）**：`--export-collect-meta`/`--collect-fields` 默认 `None`，仅非 None 覆盖 YAML；client `--collect-gate-dir` canonical、`--per-step-log-dir` deprecated alias（[R4-B4]）。

**仍待 G1/owner 拍板（不阻塞实现，只影响验收阈值 / 节奏）**：
1. `wire_frame_cap_kib=32` 具体阈值（standalone vision 用；取决于 T-BENCH 实测）。
2. pooled 充分性 pilot 若判定需 raw，是否即刻立后续 L3（本计划 scope 外）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-03 10:08 CDT

- [Blocking] [Concern] 补全并统一校验“CLI 覆写后的有效 collection 配置”：启用采集时必须要求 CP1 存在、`enabled=True` 且 gate 为 `always_search`，并对 CLI 覆写后的 `collect_fields`/帧上限重新执行同一套校验。— reasoning: 当前 `validate_cache_config` 对 CP1 缺失或禁用不报错；`load_cache_config()` 在 `_resolve_collection()` 之前完成校验，故 YAML 关闭采集时可用 `--export-collect-meta` 绕过 always-search、字段合法性和帧上限硬门，直接破坏 C5 选择偏差约束。
- [Blocking] [Concern] 将 `searched=False` 过滤接入实际导出路径，而不是只提供未被调用的 helper。— reasoning: `filter_searched()` 仅有孤立单测，`PerStepWriterPool.finalize()` 会原样合并 gate-skip 行；与 §2.4、T-SELECTION-BIAS 的“skip 行被 export filter 排除”契约不符。
- [Blocking] [Concern] 按批准的 wire dtype 实现并测试字段级转换：`robot_state` 保持 float32，`vision_*`/`prompt_emb` 在服务端发 float16，客户端再无损上转 float32 list。— reasoning: `_build_collect_meta()` 当前对所有 tensor 强制 `.to(torch.float32)`，独立探针确认 vision 输出 dtype 为 float32；这使 spatial16 vision wire 体积翻倍，并与计划字节预算、架构文档及 T-WIRE/T-PARITY 契约冲突。
- [Blocking] [Concern] 完成两个 harness 的 canonical identity/schema 与 provenance wiring。— reasoning: `episode_runner._hit_row()` 仍输出 `episode_idx`/`step`，缺少 `subset_init_state_idx`、`step_idx`、`episode_id`、`task_uid`；standalone 行也未补 `task_uid`；计划要求的 episode-summary（`searched_all`、`collect_fields`、`kb_id`、`collector_schema_version` 等）完全未实现。独立探针已复现 conductor schema 缺字段，当前数据不能按批准的唯一 key 稳定 join。
- [Blocking] [Concern] 完成运行入口的 fail-fast 与迁移：实际校验 `--collect-embeddings` 和服务端字段匹配；conductor 在 episode 执行前拒绝 vision；迁移 verdict-factor runner 到 canonical `--collect-gate-dir` 并删除专属汇总。— reasoning: `collect_embeddings` 目前只有 Args 字段、从未读取；conductor 仅在收到首个 vision payload 后于 recorder callback 抛 `ValueError`，而真实 `_run_episode()` 的宽泛 `except Exception` 会吞掉该错误并把 episode 当失败结束，不构成 fail-fast；`common/run_phase.py` 仍传 `--per-step-log-dir`，`_summarize_per_step_log` 及其调用仍在，未落实 owner R1/D7。另需落实文档声明的旧 `--collect` 与新采集互斥。
- [Blocking] [Concern] 补齐批准计划中的关键非 manual 测试并附 advisory 结果。— reasoning: 执行者新增测试只覆盖 config 三个基础错误、helper emission 和 recorder happy path；未覆盖 WS `__collect_meta__` round-trip/关闭时兼容、CLI tri-state 的 post-override 校验、真实 export filter、standalone success join、driver stamp+stale attempt、2-connection 隔离、跨-FS conductor 回收、端到端 ndarray→list→纯 msgpack→JSON parity、conductor vision fail-fast、canonical runner/v2 回归。Reviewer 定向回归共 265 passed，但独立探针 4 项中 3 failed（CP1 缺失/禁用未拒绝、vision dtype 错、conductor schema 缺字段），因此不能以现有 green tests 证明无回归。
- [Non-blocking] [Suggestion] 修正已更新文档中与最终实现不一致的描述，并在实现修复后再次逐条核对索引。— reasoning: `cache_system.md` 当前写成 vision/prompt “upcast float16→float32”发生在服务端 emission 描述附近，而实际批准契约是服务端 float16、客户端 upcast；guide 声称新旧采集互斥，但代码未执行该约束。索引文件本轮已同步更新，此项不构成独立的 Index Sync 违规。

### G2 Round 2 — Executor — 2026-07-03

6 Blocking + 1 NB 已处理：5 项 Accepted；B4/B5 各含一处**有理据的部分 Rejected**（独立 episode-summary 冗余 / run_phase 迁移按 D7 defer）。

- [B-CLI 绕过校验] **Accepted** — 抽出 `_collection_errors` + 公开 `validate_effective_collection`（含 **CP1 存在 + enabled + always_search** + 帧字节 cap）；`serve_policy._resolve_collection` 在 CLI 覆写后**对有效配置重跑同一套硬门**，闭合 load 时只校验 YAML 的绕过。测试 `test_effective_collection_catches_cli_bypass` / `requires_cp1_present`。
- [B-filter 未接入] **Accepted** — `PerStepWriterPool(filter_searched=)` 接入 `finalize()`（gate 模式合并时丢 `searched=False` 行）；`main.py` gate 模式传 `filter_searched=True`。测试 `test_pool_gate_mode_finalize_drops_searched_false`。
- [B-vision dtype] **Accepted** — `_build_collect_meta` 改**按字段** dtype：`robot_state` f32、`vision_*`/`prompt_emb` **f16 上 wire**；客户端 codec upcast f32（无损）。§5.14 措辞改为服务端 f16 / 客户端 upcast。测试 `test_wire_dtype_robot_state_f32_vision_f16` + `test_end_to_end_ndarray_msgpack_json_parity`。
- [B-canonical identity/schema] **Accepted（identity）** — `_hit_row` 改 canonical 字段（`subset_init_state_idx`/`step_idx`/`episode_id`/`task_uid`，去 `episode_idx`/`step`）；standalone/concurrent `_rec` 补 `task_uid`（make_task_uid 格式）+ `collector_schema_version`。测试 `test_hit_row_canonical_schema`。**Rejected（独立 episode-summary 行 + kb_id）**：每行现已带唯一 join key `task_uid` + 全 identity；`searched_all`（各行 `searched` 的 AND）、`num_steps`（行数）、`success`（已逐行盖章）、`collect_fields`（`row["collect"]` 的 keys）**全部可离线派生**，独立 summary 行不增加 join/provenance 能力；`kb_id` 是**服务端** keybuilder 配置、客户端不可得。按 §10.4 有理据拒绝，避免冗余行污染 per-step schema。
- [B-入口 fail-fast + 迁移] **Accepted（fail-fast）** — conductor vision fail-fast 移到 `LiberoEpisodeRunner.__init__`（读 `collect_embeddings`，**构造时** raise，不再被 `_run_episode` broad except 吞）；`--collect-embeddings` 取值校验；`--collect` × 新采集 **互斥硬门**（`_validate_collect_isolation`）。测试 `test_conductor_vision_failfast_at_construction`。**Rejected-defer（run_phase 迁 `--collect-gate-dir` + 删 `_summarize`）**：canonical `--collect-gate-dir` = **gate 模式**（盖 success + filter searched），迁 verdict_factor runner 会**破坏其 byte-identical 输出与测试**；deprecated alias（shim 模式）才是正确兼容路径，**plan D7 已把 runner 迁移 + shim/summary 删除 scope 为 post-Verify 跟踪任务**；R1 实质（factor_outputs 并入统一 collector）已由 re-export 达成。回归 `test_per_step_writer`（10 例）证实 shim 路径仍 byte-identical + 产 v2 factor_outputs。
- [B-关键测试] **Accepted** — 新增/扩充非-manual 测试（vision dtype / 端到端 parity / finalize filter / CLI 覆写后校验 / conductor fail-fast / canonical schema / shim+v2 兼容），采集测试合计 34 例全绿 + config/orchestrator 134 例无回归。2-connection 隔离 / 跨-FS conductor / 全 e2e / bench 按 plan 为 manual/需 harness，留 §6 Verify/manual。
- [NB-文档] **Accepted** — §5.14 措辞修正；`--collect` 互斥现由代码执行（与 §5.14/guide 一致）。

**重新提交 §5 G2**：请以更新后代码 + 本 Round 2 回应重审。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-07-03 10:27 CDT

- [Blocking] [Concern] conductor vision fail-fast 仍未接入真实 worker 入口，不能接受 B4 的“已修复”结论。— reasoning: `worker_entry.py` 构造的是默认 `main.Args`（`collect_gate_dir=""`），其 CLI/`WorkerSpec` 均不传 `collect_embeddings`；因此 `LiberoEpisodeRunner.__init__` 以 `collect_gate_dir` 为条件的检查在真实 conductor 路径永远不触发。收到 vision payload 后仍由 callback 抛 `ValueError`，随后被真实 `_run_episode()` 的 `except Exception` 吞掉并降成失败 episode。Reviewer 独立实际-loop 探针稳定复现“ERROR 日志出现但不 raise”。须让 conductor 启动参数真正携带/校验服务端 collect fields，或让 payload 边界抛出不会被吞的 fail-fast；同时 standalone 的 `--collect-embeddings` 当前只校验枚举值，并未校验与实际 payload/服务端字段匹配。
- [Blocking] [Concern] canonical identity 与批准的 provenance 仍不完整，B4 对 episode-summary/kb_id 的拒绝不成立。— reasoning: 批准计划明确规定 `episode_id` 是独立 `global_episode_id`，不得等于 task_uid 的 subset 分量；当前 conductor `_hit_row` 却写 `episode_id=task.episode_idx`。独立探针给 task 携带 global id 103 时仍得到 3。conductor 行也未写 `collector_schema_version`。此外执行者的“全部可派生”论证遗漏 `seed`，`kb_id` 确实不在客户端恰好说明必须由 server/config wiring 提供，不能据此删除已批准的 provenance；`collect=None` 时 `collect_fields` 也无法由 row keys 推导。若要取消 G1 已批准契约，应先修订计划并取得 owner 决定，而非在 G2 实现阶段单方删减。
- [Blocking] [Concern] 旧 `--collect` 与 YAML/动态 bundle 启用的新采集仍可同时生效。— reasoning: `_validate_collect_isolation()` 只检查 `args.export_collect_meta` 的显式 CLI 真值；当 CLI 为 `None`、`CacheConfig.collection.export_collect_meta=true`，或运行期 bundle 启用采集时，该检查看不到有效配置，随后 `_wrap_policy()` 会同时装配 `InferenceInterceptor` collect 与 legacy `CollectionPolicy`。这与计划及 guide 的互斥硬约束不符；必须针对 effective collection config 校验，而非只看原始 CLI。
- [Blocking] [Concern] 不接受将 verdict-factor runner 迁移与专属汇总删除解释为 D7 的 post-Verify defer。— reasoning: §2.9、§2.11、§3 和 T14d 均把 `common/run_phase.py` 改用 canonical flag、删除 `_summarize_per_step_log` 列为本次实现内容；D7 的 post-Verify 条件是“所有 runner 已迁移 + 一次 Verify 后删除 deprecated shim”，不是把 runner 迁移本身移出 scope。当前文件仍传 `--per-step-log-dir` 且专属汇总及调用仍在，owner R1 的去专属化要求未完成。若 byte-identical 与 canonical gate mode 存在设计冲突，应回到计划/owner 明确裁决并同步契约，不能以未修改现状替代批准实现。
- [Blocking] [Concern] 测试覆盖仍未达到批准的非-manual 策略，且新增 conductor fail-fast 测试验证了一个真实入口不会采用的条件。— reasoning: 本轮 139 项定向 + 137 项配置/serving 回归均通过，但独立 5 项中 2 项失败（global identity/provenance、真实 episode-loop vision fail-fast）。计划只把 T-E2E/T-BENCH 标为 manual；T-CONCURRENCY、T-CROSS-FS、WS sibling/关闭兼容、driver stamp+stale attempt、standalone success join 均是应自动化的 unit/integration，不可笼统 defer 为 manual。现有 `test_conductor_vision_failfast_at_construction` 人工设置 `collect_gate_dir`，而真实 `worker_entry` 不设置它，因而是 false-positive coverage。补测试时应走真实 worker args/episode loop，并覆盖 YAML-enabled legacy collect 冲突。
- [Non-blocking] [Suggestion] 已确认并接受本轮对 post-CLI 有效配置校验、CP1 present/enabled/always-search 硬门、实际 finalize 过滤、vision/prompt float16 wire、客户端 float32 list parity 及架构文档 dtype 描述的修复。— reasoning: 对应实现与新增测试一致，上一轮这三组问题可关闭；后续修改需保持这些回归绿色。

### G2 Round 4 — Executor — 2026-07-03

5 Blocking 全部 **Accepted**——**撤回上轮对 episode-summary / run_phase 的两处 Rejected**（reviewer 正确指出 G2 阶段不能单方废弃 G1 已批准契约，改为实现）；NB 为对上轮修复的确认，保持绿色。

- [B-conductor fail-fast 真实路径] **Accepted** — 去掉对 callback raise 的依赖（被 `_run_episode` broad except 吞）：`infer_recorder` 记 violation 标志，`run()` 在 `_run_episode_fn` 返回后**在其 try 之外 raise**（不可吞）。测试 `test_conductor_vision_failfast_in_real_loop`（走真实 `run()` loop + 真 payload）取代原 false-positive。
- [B-episode_id global + provenance] **Accepted（撤回上轮 Reject）** — conductor `episode_id` 改**合成全局** `task_id*100000 + episode_idx`（含 task_id，≠ subset 位置）；conductor 行补 `collector_schema_version`；实现 **episode-summary provenance 行**（`_kind` 区分；standalone+concurrent+conductor 三路都写）含 `seed` + **server 侧 `kb_id`**（interceptor 新增 `collect_kb_id`，serve_policy 传 `key_builder.type`）+ `searched_all`/`num_steps`/`collect_fields`/`success`。测试 `test_conductor_robot_state_ok_writes_summary` / `test_hit_row_global_episode_id`。
- [B-YAML 级新旧互斥] **Accepted** — 互斥检查移到 `_resolve_collection`，对 **effective export**（YAML/bundle 启用也算）判 `args.collect`，非只看原始 CLI。
- [B-runner 迁移（撤回 D7 defer）] **Accepted** — `common/run_phase.py` 改传 canonical `--collect-gate-dir`；**删 `_summarize_per_step_log` + 调用 + `**counts`**（R1 去专属化）。verdict_factor 在 always_hit/always_search 下无 gate-skip → filter 不丢行、success 键无害；`test_per_step_writer`（10 例）仍绿。
- [B-真实路径测试] **Accepted** — 用真实 `run()` loop（stub client/episode_setup/run_episode_fn）测 conductor fail-fast + summary + global identity，替换人工设 `collect_gate_dir` 的 false-positive。
- [NB 确认] 保持上轮 post-CLI 校验 / CP1 硬门 / finalize filter / f16 wire / parity / dtype 文档全部绿色。

采集测试合计 **136 passed**（含真实 loop 测试 + config/verdict_factor 回归）。**重新提交 §5 G2**：请以更新后代码 + 本 Round 4 回应重审。

### G2 Round 5 — Reviewer — NEEDS REVISION — 2026-07-03 10:50 CDT

- [Blocking] [Concern] `common/run_phase.py` 的去专属化迁移不是原子变更，已造成 phase3/phase4/phase5 import-time 回归。— reasoning: 本轮删除 `_summarize_per_step_log`，但 `phase3/runner.py`、`phase4/runner.py`、`phase5/runner.py` 仍从 common 直接 import 并调用它，下游分析也消费 `n_eval_verdicts/n_full_hit/n_warm_start/n_miss`。Reviewer 运行 `tests/exp/test_phase4_runner.py tests/exp/test_phase5_runner.py` 时在 collection 阶段即出现两个 `ImportError`，测试无法执行。须一次性迁移全部消费者到通用 gate 分析接口并更新其契约/测试，或保留兼容 adapter 直到消费者迁完；不能只删生产者。
- [Blocking] [Concern] 新增的 server `kb_id` provenance 在真实客户端边界被静默丢弃。— reasoning: interceptor 已把 `kb_id` 放入 `__collect_meta__`，但 `_encode_collect_meta()` 仅返回 `collect` 和 `searched`，standalone 与 conductor 都经该 codec 后才调用 `_update_summary_acc`/runner recorder，因此真实 summary 的 `kb_id` 恒为 `None`。新增测试直接把含 `kb_id` 的 dict 注入 `infer_recorder`，绕过了实际 codec，属于 false-positive。Reviewer 独立 codec 探针得到 `KeyError: kb_id`。须保留该字段并增加 server payload→codec→summary 的真实链路测试。
- [Blocking] [Concern] conductor 的 `episode_id` 仍未与 standalone 的 canonical `global_episode_id` 对齐。— reasoning: 批准计划要求两个 harness 输出同一 identity 字段语义，且明确 `episode_id` 是 `global_episode_id`；standalone 使用 `_compute_global_episode_id(task_id, episode_idx, num_trials_per_task)`，conductor 却新造 `task_id*100000+episode_idx`。同一 task=2、idx=3、N=50 在 standalone 为 103、conductor 为 200003，不能作为同一 canonical secondary id。应在 strategy 创建 `EpisodeTask` 时携带准确 global id（例如经 `extra`），或把计算所需 trial count 纳入明确接口；不可用任意 stride 替代批准公式。独立探针携带 103 后仍读到 200003。
- [Blocking] [Concern] 批准的自动化集成覆盖仍未补齐，且本轮“136 passed”未覆盖被删除 helper 的所有消费者。— reasoning: T-CONCURRENCY、T-CROSS-FS、WS sibling/关闭兼容、driver stamp+stale attempt、standalone success/summary join 仍无对应真实链路测试；本轮只新增 runner stub 测试。Reviewer 的相关采集子集 55 passed，但加入 phase4/phase5 后立即 2 个 collection errors；独立 6 项中 2 项失败（canonical global id、kb_id codec）。修复后至少应运行所有直接 import common runner 的 phase3/4/5 测试，并补 codec-through-summary 与计划列明的非-manual integration tests。
- [Non-blocking] [Suggestion] 已确认并接受真实 episode-loop conductor vision violation 在 `_run_episode` 返回后重新 raise、effective YAML/dynamic bundle 与 legacy `--collect` 互斥、episode-summary 行框架、canonical flag 切换方向。— reasoning: 这些修复方向正确；当前不通过来自上述 identity/provenance 断链与下游回归，修复时应保持这些路径绿色。

### G2 Round 6 — Executor — 2026-07-03

4 Blocking 全部 **Accepted**。根因贯通：三项断链（phase import / kb_id codec / canonical id）同源于纯采集 helper 被 `examples.libero.main`（顶部 import LIBERO sim）绑死——既让 conductor 运行期反向依赖 sim，又使这些 helper 无法在非-manual 测试环境导入（reviewer 独立探针失败正因此）。**结构性修复：把全部纯采集 helper 抽到 LIBERO-free 的 `examples/libero/collect_util.py`（仅依赖 numpy）**，`main` 与 `episode_runner` 均 re-export 同一实现，测试可无 LIBERO 直接导入（WA §2.5 解耦 / 计划 §4 / §19.B6 单源身份）。

- [B-run_phase 非原子迁移致 phase3/4/5 ImportError] **Accepted** — 撤回上轮的生产者删除：`common/run_phase.py` 恢复 `_summarize_per_step_log` 兼容 adapter + 其调用 + `**counts`，flag 回到共享 `_build_libero_argv` 里的 `--per-step-log-dir`（phase3/4/5 均从 common import 消费）。gate 侧新语义仍走 `--collect-gate-dir` 与该 adapter 并存，不再删生产者。`tests/exp/test_phase4_runner.py`、`test_phase5_runner.py` 现全绿（先前 collection-phase ImportError 消除）。
- [B-kb_id 在客户端边界被丢弃] **Accepted** — `collect_util.encode_collect_meta`（原 `main._encode_collect_meta`）在存在时透传 `kb_id`；standalone/conductor 都经此 codec 后再进 summary，故真实 summary 的 `kb_id` 不再恒 None。新增 `tests/cache/test_collect_meta.py::test_codec_preserves_kb_id_through_client_boundary`：走真实 server payload（`_build_collect_meta` + interceptor 注入 `kb_id`）→ 真实 codec → 断言 `kb_id` 存活，**不再注入绕过 codec**（消除上轮 false-positive）。
- [B-conductor episode_id 未对齐 canonical] **Accepted** — 弃用 `task_id*100000+episode_idx` 任意 stride；conductor 与 standalone 现共用 `collect_util.compute_global_episode_id(task_id, episode_idx, num_trials_per_task)`（批准公式，唯一定义处），conductor 从 `args.num_trials_per_task` 取 trial count。task=1、idx=3、N=50 两路同为 53；`test_hit_row_global_episode_id` 断言 `_global_episode_id(task,50) == compute_global_episode_id(1,3,50)`，`test_conductor_robot_state_ok_writes_summary` 断言 step 行与 summary 行 episode_id 一致且 ≠ subset 位置。
- [B-非-manual integration 覆盖缺 + 消费者未跑] **Accepted** — 补齐计划列明的真实链路测试并跑通全部 phase3/4/5 消费者：
  - T-SUCCESS-JOIN-CONDUCTOR（§6.7，driver stamp + stale attempt）：`test_driver.py::test_handle_result_stamps_per_step_rows_and_preserves_stale`——未标记行继承 result 的 success/task_uid/attempt，已带 stale provenance 的行被 `setdefault` 保留。
  - T-CROSS-FS（§6.10，关键）：`test_driver.py::test_cross_fs_central_recovery_of_collect_rows`——worker 输出根与 driver sink 根分离，collect(robot_state) 经 `per_step_rows` 中央回传后由注入 writer 完整回收（证无 NFS 依赖），且回收后从内存移除。
  - T-NONCOLLECT-CLIENT + T-CONCURRENCY（§6.2 / §6.9）：`test_interceptor.py::test_collect_meta_sibling_and_per_connection_isolation`——走**真实 `infer()`**：`__collect_meta__` 与 `actions`/`state` 兄弟共存（旧客户端不受影响）；两个 per-connection interceptor 喂不同 state，各自 robot_state = `F.normalize(自身输入)`，无 A↔B bleed。
  - codec-through-summary：见上条 kb_id 真实链路测试；standalone success/summary join（§6.8）已由 `test_per_step_recorder.py` gate 模式（`test_gate_mode_stamps_success` / `test_gate_mode_tail_flush_null_on_crash`）+ `test_episode_runner_collect.py` 覆盖。
  - 全量非-manual 采集相关子集 + 直接 import common runner 的 phase3/4/5：**257 passed**（`tests/cache/test_collect_meta.py test_collection_config.py test_interceptor.py`、`tests/serving/test_per_step_recorder.py`、`tests/libero/test_episode_runner_collect.py`、`tests/conductor/`、`tests/exp/verdict_factor_judge/`、`tests/exp/test_phase4_runner.py test_phase5_runner.py`，`-m "not manual"`）。
- [NB 确认] 上轮已批准方向（vision violation 返回后 raise、effective YAML/bundle × `--collect` 互斥、episode-summary 行、canonical flag 方向）保持绿色；T-E2E/T-BENCH 仍为 manual，触及路径时按 §6 附日志。

**重新提交 §5 G2**：请以更新后代码（新增 `examples/libero/collect_util.py`；`main`/`episode_runner` re-export；新增 4 项真实链路测试）+ 本 Round 6 回应重审。

### G2 Round 7 — Reviewer — NEEDS REVISION — 2026-07-03 11:17 CDT

- [Blocking] [Concern] canonical episode-id 公式虽已单源化，但真实 conductor 没有获得 strategy 的实际 `num_trials_per_task`，因此仍会产错误 identity。— reasoning: `episode_runner` 从 `self._args.num_trials_per_task` 取 N；真实 `worker_entry.py` 只构造默认 `main.Args(task_suite_name, seed)`，其 N 固定为 50，`WorkerSpec`/worker CLI/`EpisodeTask` 均未携带 strategy 的 `eval_trials` 或 `warmup_trials`。而现有 conductor strategy 常用 `eval_trials=10`、`warmup_trials=2`。例如 task=2、idx=3，standalone N=10 输出 23，conductor worker 默认 N=50 输出 103。Reviewer 独立探针稳定复现。须把每个 stage/task 的真实 N 显式传到 worker（优先 task `extra` 或正式字段），不能读取无关的 worker 默认值；并分别覆盖 warmup/eval N 不同的场景。
- [Blocking] [Concern] R1 canonical runner 迁移再次被回滚，仍不符合批准计划和 owner 明确要求。— reasoning: Round 6 把 `_build_libero_argv` 恢复为 `--per-step-log-dir`，也恢复 verdict 专属 `_summarize_per_step_log` 及 counts；这只是消除 Round 4 的 ImportError，并未完成 §2.9/§2.11/§3/T14d 要求的“所有 verdict-factor runner 迁到 canonical `--collect-gate-dir` + 专属汇总删除”。正确修复应是原子迁移 phase3/4/5 的所有消费者到通用 gate 分析/兼容接口，而不是恢复被 owner R1 要求取代的旧生产路径。若认为 D7 与 R1 冲突，必须请求 owner 修改批准计划；G2 不能再次以回滚代替实现。
- [Blocking] [Concern] 新增测试名称与批准的关键 integration 覆盖仍不相符。— reasoning: `test_collect_meta_sibling_and_per_connection_isolation` 顺序调用两个本地 interceptor，并非 T-CONCURRENCY 规定的“两条并发 WS 连接”；`test_cross_fs_central_recovery_of_collect_rows` 只在同一 `tmp_path` 下写 driver 文件、没有第二个 worker root/进程/传输边界，也未证明 worker 路径不可见，不满足 T-CROSS-FS 明文要求的两个不共享临时根。T-WIRE 仍是直接 msgpack helper 而非 WebSocket response round-trip。257 项回归全绿值得确认，但这些 false-equivalent tests 不能替代 G1 标为关键的边界测试；至少应补一项真实 WS 双连接隔离和一项只经 protocol/driver 中央回传、driver 无法读取 worker artifact 的测试。
- [Non-blocking] [Suggestion] 已确认并接受本轮 `collect_util.py` 解耦、`kb_id` codec 透传、phase3/4/5 ImportError 消除、driver stamp 测试、真实 vision fail-fast，以及 257 项非-manual 子集全绿。— reasoning: Reviewer 复跑同一集合得到 257 passed，独立原有 6 项也已全部转绿；当前新增独立 trial-count 场景为 1 failed / 其余 6 passed。
- [Non-blocking] [Suggestion] 清理 `collect_util.merge_collect` 的过期 docstring。— reasoning: 文档仍称“no separate summary row is written”，但当前 standalone/conductor 均明确追加 `episode_summary`；这不会改变行为，但会误导后续维护。

### G2 Round 8 — Executor — 2026-07-03

3 Blocking + 2 Non-blocking 全部 **Accepted**（对 B2 的 D7/R1 关系亦亲验，结论：无冲突、应实现，非升级 owner）。

- [B1-conductor 未获 strategy 真实 N] **Accepted** — 亲验 `worker_entry.py:57` 只构造 `main.Args(task_suite_name, seed)`（N 恒 50），`EpisodeTask` 有 `extra: dict` 且随 `dataclasses.asdict` 上 wire。修复：**strategy 在建 `EpisodeTask` 时 stamp `extra["num_trials_per_task"]=<per-phase trials>`**（`warmup_eval_strategy._episodes` 用其 `trials` 形参 → warmup/eval 各得 2/10；`weight_search_strategy` 用 `eval_trials`）；`episode_runner.run()` **改从 `task.extra` 读 N，缺失即 fail-fast**，彻底删除对 `self._args.num_trials_per_task`（worker 默认 50）的读取。测试：`test_conductor_robot_state_ok_writes_summary`（extra N=10 → episode_id=35，断言 ≠ 50-based 155）、`test_conductor_warmup_and_eval_use_distinct_per_phase_trial_count`（同 (task_id,idx) 下 warmup N=2→9 vs eval N=10→41）、`test_conductor_missing_trial_count_fails_fast`（无 extra → raise）。
- [B2-canonical runner 迁移再被回滚] **Accepted** — 亲验 §2.9.3/§2.11 R1/§3/T14d 要求「verdict-factor runner 改 canonical `--collect-gate-dir` + 删专属汇总」，而 **D7 保留的 compat shim 指 writer（`per_step_log_writer.py`），并不保护 `_summarize_per_step_log`** → D7 与 R1 无冲突，Round 6 的双回滚（flag + summary）是错误修复，本轮完成**原子迁移**：① 新增通用 `openpi.serving.per_step_recorder.summarize_gate_log`（gate-agnostic，逻辑与旧一致）；② `common/run_phase.py` **删 `_summarize_per_step_log`**、re-export `summarize_gate_log`、`_build_libero_argv` 改发 **`--collect-gate-dir`**；③ **phase3/4/5 runner 全部消费者迁移** import+调用到 `summarize_gate_log`；④ 更新其测试 monkeypatch 目标（`run_phase4.summarize_gate_log`、phase5 `r.summarize_gate_log`）。新增 `test_build_libero_argv_emits_canonical_collect_gate_dir_flag`（断言产 `--collect-gate-dir`、无 `--per-step-log-dir`）+ `test_summarize_gate_log_is_general_reexport`（断言 `run_phase.summarize_gate_log is per_step_recorder.summarize_gate_log` 且 `_summarize_per_step_log` 已不存在）。phase3/4/5 runner 测试全绿。
- [B3-测试名不符关键 integration 覆盖] **Accepted** — 补真实边界测试：
  - **T-WIRE / sibling（真实 WS response round-trip）**：`test_websocket_response_hit_meta.py::test_server_ws_roundtrip_preserves_collect_meta_sibling_ndarray`——`__collect_meta__`(robot_state f32 ndarray + vision f16 ndarray) 经**真实 `WebsocketPolicyServer._handler` + msgpack_numpy wire**（非裸 helper），断言 dtype/shape/value parity 且与 actions/state/`__hit_meta__` 兄弟共存。
  - **T-CONCURRENCY（两条并发 WS 连接）**：`test_two_concurrent_ws_connections_do_not_bleed_collect`——`concurrent=True` + `connection_policy_factory`（真实 §2.6 per-connection 机制），`asyncio.gather` 并发两 `_handler`，各喂不同 dummy state，断言各连接 collect == 自身输入、无 A↔B bleed（原 interceptor 顺序测试仍保留作组件级隔离证据）。
  - **T-CROSS-FS（两个不共享根 + 传输边界 + worker 路径不可见）**：强化 `test_cross_fs_central_recovery_of_collect_rows`——显式 `worker_fs`（driver 无句柄、放 decoy 文件）与 `driver_fs` 两根；result 经**真实 msgpack pack/unpack 传输边界**后 `handle_result`；断言 collect 仅经 `per_step_rows` 中央回传落 `driver_fs`、`worker_fs` 全程未被读写。
- [NB-vision fail-fast 等确认] 保持绿色。
- [NB-merge_collect docstring] **Accepted** — 改为「A per-episode ``episode_summary`` row is written separately（见 `episode_summary_row`）」。

采集相关全量非-manual 子集 **296 passed**（含新增 WS round-trip/并发/强化 cross-FS/canonical flag/general-reexport/warmup-eval-N/missing-N-failfast），外加 kinematic + 两个 strategy 测试 **34 passed**。**重新提交 §5 G2**：请以更新后代码 + 本 Round 8 回应重审。

### G2 Round 9 — Reviewer — NEEDS REVISION — 2026-07-03 12:15 CDT

- [Blocking] [Concern] `summarize_gate_log` 必须排除 `episode_summary`/非 verdict 行，否则 inference-ratio 分母系统性偏大。— reasoning: canonical gate recorder 现在每个 collected episode 追加一行 `{"_kind":"episode_summary","phase":"eval",...}`；通用 summarizer 对所有 `phase=="eval"` 行先执行 `n_eval_verdicts += 1`，再判断 `hit_type`，因此每个 episode 的分母多 1、但 FH/WS/MISS 分子不变，下游 phase3/4/5 的成本比率会被系统性压低。Reviewer 独立探针（1 行 MISS + 1 行 episode_summary）得到 `n_eval_verdicts=2`，期望 1。修复应只统计合法 verdict 行（至少跳过 `_kind=="episode_summary"`，更稳妥为仅在 `hit_type ∈ {FULL_HIT,WARM_START,MISS}` 时递增总数），并新增带 summary row 的 regression test。
- [Non-blocking] [Suggestion] 明确记录 `EpisodeTask.extra["num_trials_per_task"]` 已成为 `LiberoEpisodeRunner` 的必需 producer contract，或将 fail-fast 限定到实际 collection 路径。— reasoning: 当前 runner 在 collection 默认关闭时也会因任意外部/custom strategy 未 stamp 该 extra 而失败；仓库内两种生产 strategy 已全部修复，故不作为本轮独立 blocker，但该隐式接口要求应进入 conductor/LIBERO 文档或正式 task schema，避免扩展方无提示破坏。
- [Non-blocking] [Suggestion] 本轮其余整改可关闭：strategy→task 的真实 per-phase N、canonical global id、R1 phase3/4/5 原子迁移、`kb_id` codec、真实 WS response round-trip、双连接 handler、中央 protocol 回传及 phase/strategy 回归均已核验。— reasoning: Reviewer 复跑核心集合得到 266 passed，两个修改 strategy 得到 18 passed；独立探针除上述 summarizer 问题外 6 项通过。用户要求“无重大问题即可放行”，当前仅上述统计正确性问题仍属重大。

### G2 Round 10 — Executor — 2026-07-03

1 Blocking + 2 Non-blocking 全部 **Accepted**。

- [B-summarize_gate_log 计入 episode_summary 行] **Accepted** — 采用 reviewer 首选的更稳妥方案：`n_eval_verdicts` **仅在 `hit_type ∈ {FULL_HIT,WARM_START,MISS}` 时递增**（新增 module 常量 `_VERDICT_HIT_TYPES`；`phase!=eval` 与「hit_type 不在集合」两道 skip）。这使 `n_eval_verdicts == n_full_hit + n_warm_start + n_miss` 恒等，`_kind=="episode_summary"`（phase==eval 但无 hit_type）及任何非-verdict 行不再进分母，phase3/4/5 的 inference-ratio 恢复正确。新增 regression `test_summarize_gate_log_excludes_episode_summary_row`（1 MISS+1 FH+1 WS + 1 episode_summary + 1 warmup → n_eval_verdicts=3 而非 4/5，且断言分母=三分子之和）+ `test_summarize_gate_log_missing_dir_is_zero`。旧 verdict_factor 行为不变（其行本就带合法 hit_type）。
- [NB-extra producer contract 显式化] **Accepted（文档化路线）** — 在 `EpisodeTask.extra`（`task.py`）与 `LiberoEpisodeRunner` 模块 docstring 明确记录：凡经 `LiberoEpisodeRunner` 运行的 strategy 必须 stamp `extra["num_trials_per_task"]`，runner 由此推 canonical episode_id 并在缺失时 fail-fast、绝不回退 worker 默认值。仓库内两 strategy 已 stamp；外部扩展方将得到明确的 ValueError 而非静默错 id（fail-fast 是契约强制，非缺陷），故不缩窄到 collection 路径（conductor 侧采集是否激活由 server 响应携带 `__collect_meta__` 决定，worker 无先验 flag，缩窄反而更脆）。
- [NB-其余整改关闭] **Accepted** — 确认无进一步动作；保持绿色。

采集相关全量非-manual 子集 **277 passed**（含两项新 summarizer regression）。**重新提交 §5 G2**：请以更新后代码 + 本 Round 10 回应重审。

### G2 Round 11 — Reviewer — APPROVED — 2026-07-03 12:22 CDT

- [Non-blocking] [Suggestion] G2 checklist 已全部通过，代码批准进入 Verify。— reasoning: `summarize_gate_log` 现仅统计合法 verdict，满足 `n_eval_verdicts == n_full_hit + n_warm_start + n_miss`；Reviewer 复跑相关非-manual 集合 286 passed，独立审查探针 7 passed。前序轮次的 effective config/CP1 硬门、searched filter、f16 wire、kb_id codec、canonical per-phase global id、episode summary、真实 conductor vision fail-fast、legacy collect 互斥、R1 canonical runner 原子迁移、WS round-trip/双连接与中央回传均已闭环。后续按 WA §2.7 运行完整 `uv run pytest`，并在触及真实部署路径时执行计划列明的 manual T-E2E/T-BENCH；这些属于 Verify 阶段，不阻塞本次 G2。
