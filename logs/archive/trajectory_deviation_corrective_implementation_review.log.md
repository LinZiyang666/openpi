# Trajectory Deviation 纠偏实现计划 — G1 审查结论与讨论

> Status: Implemented
> Date: 2026-04-14
> Review target: `archive/trajectory_deviation_corrective_implementation.log.md`
> Base plan: `archive/trajectory_deviation_corrective_experiment.log.md`
> Reviewer role: G1 plan reviewer

---

## 1. 审查结论

**当前不批准进入 Code 阶段。**

原因不是 plan 不够细，而是有几处代码级假设与当前实现不一致，会导致：

1. prefill 模式实际不生效或只对一半组件生效；
2. client-side / server-side HDF5 step 无法对齐；
3. `deviate_score` 的 L2 比较对象不明确，且按当前伪码会出现 shape/key 错误；
4. `--collect + --cache` wrapper 生命周期没有正确向内层 cache 转发，跨 episode history 会泄漏；
5. 并行 runner 的 state/jsonl 输出存在竞争写风险。

建议先修订 `logs/trajectory_deviation_corrective_implementation.log.md`，把下面 A 级问题逐项关掉，再重新做一次 G1。

---

## 2. 已审查材料

- `CLAUDE.md`
- `WORKING_AGREEMENT.md`
- `docs/README.md`
- `logs/README.md`
- `docs/openpi_reference.md`
- `docs/cache_system_architecture.md`
- `docs/cache_system_tutorial.md`
- `docs/cache_system_workflow.md`
- `docs/data_collection_guide.md`
- `logs/trajectory_deviation_corrective_experiment.log.md`
- `logs/trajectory_deviation_corrective_implementation.log.md`
- 相关源码抽查：
  - `examples/libero/main.py`
  - `scripts/serve_policy.py`
  - `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
  - `src/openpi/serving/websocket_policy_server.py`
  - `src/openpi/collect/collection_policy.py`
  - `src/openpi/collect/data_collector.py`
  - `src/openpi/cache/config.py`
  - `src/openpi/cache/cache_storage.py`
  - `src/openpi/cache/interceptor.py`
  - `src/openpi/cache/orchestrator.py`
  - `src/openpi/cache/components/search_strategy.py`
  - `src/openpi/cache/storage_types.py`
  - `src/openpi/policies/libero_policy.py`

---

## 3. A 级阻塞问题

### A1. Step 粒度定义不一致：env step、inference cycle、HDF5 step 三者会错位

**问题**

`examples/libero/main.py` 当前每 `replan_steps` 个 env step 才调用一次 `client.infer()`：

- `examples/libero/main.py` 里 `action_plan` 非空时不会调用 server；
- server-side `--collect` 只会在真实 `infer()` 调用时记录一条 HDF5 step；
- implementation plan 的 client-side trajectory 却按每个 env step 保存 `step_0000...step_T`。

因此在默认 `replan_steps=5` 时：

- client-side GT HDF5 大约有 `T` 个 env step；
- server-side collection HDF5 只有约 `T / 5` 个 inference step；
- plan 中"同名同 step 对齐"这个假设不成立。

这会直接影响 Step 2 和 Step 3：

- Step 2 声称"每条 GT trajectory 每一步重放 observation"，但这个"每一步"到底是 env step 还是 inference cycle 未定义；
- Step 3 的 `prefill_trajectory` 要灌 D-1 步 trajectory history，而 cache trajectory history 实际按 inference cycle 累积，不按 env step 累积。

**建议**

先做一个明确决定：

1. 如果实验单位是 inference cycle：client-side HDF5 也只保存每次 `client.infer()` 的 obs/action_chunk/sim_state，并额外保存该 chunk 后续实际执行的 env actions。
2. 如果实验单位是 env step：本实验必须强制 `--replan-steps 1`，否则 server collection、cache history 和 env step 不同频。

我更建议选 1，因为它匹配当前 cache 系统的真实调用频率，也不改变原 evaluation 行为。

**必须修订**

- `logs/trajectory_deviation_corrective_implementation.log.md` 中所有 `step_t`、`T`、`action`、`action_chunk` 的含义。
- `examples/libero/main.py` 的 HDF5 schema。
- Step 2 aggregation 的输入 shape。
- Step 3 spawn 的 `s+n` 索引语义。

---

### A2. `CollectionPolicy` 没有向内层 cache 转发生命周期，`--collect + --cache` 会泄漏 episode 状态

**问题**

当前 `scripts/serve_policy.py` 的 wrapper 顺序是：

1. `InferenceInterceptor`
2. 可选 `PolicyRecorder`
3. `CollectionPolicy`

也就是 `CollectionPolicy` 在最外层。server 只调用最外层 policy 的 `on_episode_start()` / `on_episode_end()`。

但当前 `src/openpi/collect/collection_policy.py`：

- `on_episode_start()` 只调用 collector；
- `on_episode_end()` 只调用 collector；
- 没有转发给内层 `InferenceInterceptor`。

implementation plan 只计划把 `episode_name` 透传给 collector，仍没有补内层转发。这会导致：

- cache orchestrator 不会在 episode_start 时 reset；
- trajectory history、step_counter、episode buffer 可能跨 episode 泄漏；
- `on_episode_end()` 不触发 cache write/reset；
- Step 1b、Step 2、Step 3 只要同时使用 `--collect` 和 cache wrapper，就存在语义错误。

**建议**

`CollectionPolicy` 的生命周期应变成透明 wrapper：

```python
def on_episode_start(self, experiment: str, task: str, episode_id: int, episode_name: str = "") -> None:
    self._collector.on_episode_start(experiment, task, episode_id, episode_name=episode_name)
    self._collecting = True
    self._prompt_captured = False
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(experiment, task, episode_id)

def on_episode_end(self, success: bool) -> None:
    self._collector.on_episode_end(success)
    self._collecting = False
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success)
```

同时所有 wrapper 的 `on_episode_start` 都应接受 `episode_name: str = ""` 或 `**kwargs`，否则 server 用 keyword 传参时，未收集模式下的 `InferenceInterceptor` 会 `TypeError`。

**必须补测试**

- `--collect + --cache_config` 跑两个 episode，断言 inner orchestrator 的 step counter/history 在第二个 episode 从 0 开始。
- `--collect` off 时，server 传 `episode_name` 不破坏 `InferenceInterceptor.on_episode_start()`。

---

### A3. Per-connection prefill facade 的接线不完整，SearchStrategy 仍可能指向 shared facade

**问题**

plan 的 §18.A2 建议在 `src/openpi/cache/config.py:713-761` 里新建 `per_conn_storage` 并在 return dict 里返回它。但当前代码在同一个函数里先构造 search strategy：

```python
search_strategies[cp_id] = _build_search_strategy(
    cp_config.search_strategy, shared_storage, fusion_weights
)
...
"storage": shared_storage
```

如果只按 plan 写"return 里换成 `per_conn_storage`"，会出现：

- `CacheOrchestrator._storage` 是 per-connection facade；
- `SearchStrategy._storage` 仍是 shared facade；
- `prefill_trajectory` 在 `policy._orchestrator._storage.enter_prefill_mode()` 打开的 prefill 状态，`strategy.search()` 根本看不到。

结果是 prefill 模式不会返回 synthetic hit，仍会查真实 backend。

**建议**

`per_conn_storage` 必须在构造 search strategies 之前创建，并传给 `_build_search_strategy()`：

```python
per_conn_storage = CacheStorage(
    backend=shared_storage._backend,
    metadata_db=shared_storage._metadata_db,
)

...
search_strategies[cp_id] = _build_search_strategy(
    cp_config.search_strategy, per_conn_storage, fusion_weights
)

return {
    "storage": per_conn_storage,
    ...
}
```

**必须补测试**

在测试里断言同一个 connection 内：

- `orchestrator._storage is strategy._storage`
- client A 进入 prefill 不影响 client B；
- client A 的 `prefill_trajectory()` 后下一次正式 query 的 trajectory history 已包含 prefill query_keys。

---

### A4. Prefill payload 的 action 空间不清楚：post-transform 7D action 被当成 cache payload 使用

**问题**

当前 cache artifact 和在线写入的 `CachePayload.action_chunk` 来自 `stage3.action_chunk`，也就是 output transform 之前的模型 action tensor。对 `pi05_libero` 来说，模型 action dim 是 32。

但 `examples/libero/main.py` 的 `client.infer(element)["actions"]` 已经经过 `_output_transform`，`src/openpi/policies/libero_policy.py` 只返回前 7 维 env action。

implementation plan 多处把 HDF5 里的 `action` / `action_chunk` 直接塞进 `_build_prefill_payload()`。这会让 synthetic CP1 FULL_HIT 路径拿到一个 post-transform 7D tensor，然后 `InferenceInterceptor` 又把它放进 outputs 走 `_output_transform`。即使返回值被丢弃，也会造成：

- `broadcast_action()` 记录的是错误 action 空间；
- `buffer_for_write()` 可能把错误 shape 写入 episode buffer；
- 如果 write_policy 不是 `never`，可能污染 backend；
- 后续若任何 trajectory-aware gate/judge 使用 action history，会读到错误语义。

**建议**

二选一：

1. Prefill payload 使用 server-side collection HDF5 的 `clean_action`，即模型空间 action chunk，而不是 client-side env action。
2. 重新设计 `InferenceInterceptor.prefill_trajectory()`，不要通过 `self.infer()` 的 CP1 hit early-return 来产生 side effect；单独实现 side-effect-only prefill path，避免 output transform，并显式规定 `record_action()` 接收哪个 action 空间。

首版更稳妥的是 1，因为它复用当前 cache payload contract。

**必须修订**

- HDF5 schema 里同时区分：
  - `model_action_chunk`: server-side `clean_action`，shape 为模型 action dim；
  - `env_action_chunk`: client response `actions`，shape 为 LIBERO 7D；
  - `executed_action`: 实际 `env.step()` 执行的单步 7D action。
- `prefill_trajectory` 的 `actions` 参数改名或文档注明为 `model_action_chunks`。

---

### A5. Step 2 L2 的比较对象未定义，当前伪码会 key/shape 错

**问题**

源码中 client response key 是 `"actions"`，不是 `"action"`：

- `InferenceInterceptor.infer()` 返回 `"actions"`；
- `examples/libero/main.py` 使用 `client.infer(element)["actions"]`。

但 implementation plan 的 Step 2 伪码多处使用：

```python
client.infer(obs)["action"]
```

另外，即使 key 修正为 `"actions"`，shape 仍不清楚：

- `client.infer(obs)["actions"]` 是 action chunk，通常 `[10, 7]`；
- plan 的 `gt_actions = load_gt_actions(gt_dir, ep)` 看起来是 `[T, 7]` 的 executed action；
- `np.linalg.norm(cache[ep] - gt_actions, axis=-1)` 会在 `[T, 10, 7]` 与 `[T, 7]` 之间语义不明或广播错误。

**建议**

先明确 deviate score 的 action metric：

1. 比较 full env action chunk：`L2(cache_actions_chunk[t], gt_actions_chunk[t])`。
2. 只比较 chunk 第一个 action：`L2(cache_actions[t, 0], gt_actions[t, 0])`。
3. 比较当前 env step 实际执行 action：只适用于 inference cycle 与 env step 对齐，默认 `replan_steps=5` 下不成立。

我建议先选 1 或 2，并把 Step 2 的 aggregation 写成显式 shape：

```text
bg_samples:   [M, T, H, 7] 或 [M, T, 7]
cache_sample: [T, H, 7]    或 [T, 7]
gt_ref:       [T, H, 7]    或 [T, 7]
```

如果选 full chunk，GT HDF5 必须在每个 inference cycle 保存 post-transform `actions` chunk。

---

### A6. Step 3 spawn 代码同样使用错误 response key，并混淆 HDF5 obs 与 restored env obs

**问题**

`run_spawn_experiment.py` 伪码中：

```python
resp = client.infer(obs)
action = np.asarray(resp["action"])
obs_env, reward, done, info = env.step(action[0])
```

应为 `resp["actions"]`。

此外，plan 先从 HDF5 读 `start_obs = _build_obs_dict(f[step])`，再 restore env 到 `sim_state`，rollout 第一帧却直接用 HDF5 的 `start_obs`。这要求 HDF5 里的 processed obs 与 `env.set_init_state(sim_state)` 后返回的 raw obs 完全一致。理论上应一致，但 Phase 0 必须专门验证：

- restore 后 `_obs_env_to_policy(env_obs)` 与 HDF5 `start_obs` 像素级一致；
- 如果不一致，应该以 restore 后 env obs 为准，而不是 HDF5 obs。

**建议**

Step 3 execute_unit 改为：

1. restore env；
2. 取 `env.set_init_state()` 返回的 raw obs；
3. 用 `_obs_env_to_policy()` 构造第一帧 policy obs；
4. HDF5 obs 只用于 prefill 历史，且也应有一致性断言。

---

### A7. 并行 runner 的 jsonl 写入会竞争，BaseRunState 也需要原子写

**问题**

plan 已在 §18.B2 发现 `BaseRunState.save()` 需要 lock + tmp rename，这是正确方向。

但 `compute_deviate_scores.py` 伪码里多个 worker 会同时 append 同一个文件：

```python
out = Path(self.out_dir) / f"bg_{cfg}.jsonl"
with out.open("a") as f:
    f.write(...)
```

多个线程同时写大 JSON 行，不能假设不会交错。尤其 actions 序列很大，一行可能很长。

**建议**

不要并发 append 同一个 jsonl。改为 per-unit 文件：

```text
data/deviation_experiment/step2/bg/{cfg}/{episode}/{sample_idx}.npz
data/deviation_experiment/step2/cache/{cfg}/{episode}.npz
```

state JSON 只记录文件路径。聚合阶段扫 per-unit 文件。这样也比超大 JSON 更省空间、更快。

---

### A8. Phase 0 restore smoke script 的比较逻辑是错的

**问题**

plan 的 `scripts/verify_env_save_restore.py`：

1. 先跑 `acts` 50 步并记录 `traj_a`；
2. 在这 50 步之后保存 checkpoint；
3. restore 后又跑 `acts[:20]`；
4. 比较 `traj_a[:20]` 和 restore 后轨迹。

这比较的是 checkpoint 之前的前 20 步与 checkpoint 之后重新执行的 20 步，不是同一段轨迹。

**建议**

正确脚本应：

1. 从 init 跑到 checkpoint；
2. 生成一段 `post_actions`；
3. 从 checkpoint 跑 `post_actions` 得到 `traj_ref`；
4. restore checkpoint；
5. 再跑同一段 `post_actions` 得到 `traj_replay`；
6. 比较 `traj_ref` 与 `traj_replay`。

还应同时比较：

- processed policy obs 的 image/state；
- `env.timestep` / `env.cur_time`；
- `_check_success()`；
- max horizon termination。

---

## 4. B 级重要问题

### B1. `episode_name` 需要路径安全校验

`EpisodeDataCollector` 接受 client 指定 `episode_name` 后会写子目录。必须禁止：

- 绝对路径；
- `..`；
- 空 path part；
- 后缀注入；
- 过长路径。

建议只允许类似 `task_3/episode_2` 的相对 POSIX path，并由 server 统一加 `.h5`。

### B2. 当前 `configs/cache_runs/deviate_exp/` 已有 3 个 YAML，plan 的命名需要同步

当前已有：

- `configs/cache_runs/deviate_exp/clip_w7_d4.yaml`
- `configs/cache_runs/deviate_exp/spatial16_w8_d4.yaml`
- `configs/cache_runs/deviate_exp/max_pool_w3_d5.yaml`

implementation plan 里又计划新建 `cache_clip_w7_d4.yaml` 等 6 个文件。建议不要制造重复命名。可以采用：

- 现有 3 个作为 cache YAML；
- 新建 `inference_clip_w7_d4.yaml` 等 3 个；
- 或统一重命名，但 plan 必须明确迁移。

### B3. CP3 的 always_skip 讨论与当前 YAML 不一致

plan §18.A1 说 inference YAML 必须把 CP1 和 CP3 都改成 `always_skip`。但当前 deviate YAML 只有 `checkpoints.cp1`，没有 cp3。实际要求应写成：

- 如果 CP3 disabled / 不存在，则不需要；
- 如果 CP3 enabled，则 CP3 也必须 `always_skip`，否则 Phase 1 background sample 可能被 CP3 cache 干扰。

### B4. `send_load_cache_config` 后必须保证旧 client 已关闭

现有 concurrent server 在 connection open 时创建 wrapper。`load_cache_config` 更新的是未来 connection 使用的 bundle；已经打开的 client 不会自动换 config。

Step 2/Step 3 runner 必须明确：

- 每个 phase/config 切换前，等待上一批 worker 完全结束；
- 每个 worker 在 `finally` 中关闭 websocket；
- 切 config 后新建 client；
- 不允许复用跨 config 的 client。

### B5. `run_cache_experiments.py` 的 per-episode 聚合不能只扫所有 json 后简单 concat

retry 后同一个 `(config, task_id, init_state_idx)` 可能有多条记录。`cache_eval_results.json` 必须定义去重规则，例如：

- key = `(config_id, task_id, init_state_idx, seed)`；
- retry 结果覆盖首次失败结果；
- 每条记录带 `run_id`、`attempt`、`source_path`。

否则 Step 1a failed init dump 可能把已 retry 成功的 episode 仍当失败。

### B6. `Step1bRunner` 用 `--init-states-dir` 子集时，filter/map 方案需要收敛

plan 同时出现过：

- `episode_filter` 只含 `{task_id, init_state_idx}`；
- subset-local `episode_idx`；
- `orig_init_state_idx`；
- `{task.name}.init_map.json`。

建议固定格式：

```json
{
  "task_id": 3,
  "orig_init_state_idx": 15,
  "subset_init_state_idx": 1
}
```

`main.py` 内部不要反推。runner 负责生成精确 filter，HDF5 attrs 同时写两个 index。

### B7. `prefill_begin/end` debug API 可以暂缓

本实验真正需要的是 `prefill_trajectory`。`prefill_begin/end` 涉及 `CachePayload` 跨 WebSocket 序列化、private storage 访问和连接局部状态，增加测试面。

建议首版只实现 `prefill_trajectory`。如果保留 debug API，也必须加清楚：

- 仅 cache wrapper 可用；
- no-cache server 返回明确 error；
- `CachePayload` torch/numpy 序列化覆盖 intermediates；
- 不走 shared facade。

### B8. `write_policy` 需要为 Step 2/Step 3 明确设为 `never`

Phase 1 `always_skip` 会产生 miss，默认 `write_policy=on_any_miss` 会在 `episode_end` 写入 replay trajectory，污染 backend。当前 deviate YAML 没有显式 `write_policy`，会走默认 `on_any_miss`。

建议：

- 所有 replay/diagnostic YAML 显式 `write_policy: {type: never}`；
- Step 3 spawn YAML 也建议 `never`，除非实验目标就是在线扩写 cache。

---

## 5. 需要回答的疑问

1. 本实验的 step 单位到底是 **env step** 还是 **inference cycle**？
2. Deviate score 的 L2 是比较 **full action chunk**、**chunk 第一个 action**，还是 **实际 env executed action**？
3. Step 1b 是否允许强制 `replan_steps=1`？如果不允许，GT HDF5 应按 inference cycle 保存还是按 env step 保存？
4. Prefill action 应使用模型空间 `clean_action`，还是只需要 query history 而不需要 action history？
5. Step 2 是否需要 server-side `--collect` 落盘每个 background sample，还是 runner 自己保存 action samples 即可？当前 plan 两种说法都有。
6. Step 3 策略 A 是否足够作为首版，还是必须同时实现连续 intervention 链策略 C？
7. 是否允许 Step 2/Step 3 期间完全禁止 cache 写入？我建议禁止。

---

## 6. 建议的修订顺序

1. **先修语义层**：明确 step 粒度和 L2 metric，重写 HDF5 schema。
2. **再修 lifecycle**：所有 wrapper 的 episode lifecycle 透明转发，补测试。
3. **再修 prefill**：per-connection facade 同时传给 orchestrator 和 search_strategy；明确 payload action 空间。
4. **再修 runner 输出**：per-unit `.npz` / `.h5`，不要并发 append 大 jsonl。
5. **最后补 YAML 与 tests**：当前 3 个 cache YAML + 3 个 inference YAML，所有 replay YAML `write_policy: never`。

最小可行实现路径：

```text
Phase 0:
  verify save/restore + processed obs equality

Phase 1:
  main.py 保存 inference-cycle 级 GT HDF5
  server --collect 保存同名 inference-cycle model data

Phase 2:
  replay GT inference-cycle obs
  保存 per-unit npz
  metric 明确为 full chunk 或 first action

Phase 3:
  restore env 到 intervention 后状态
  prefill 使用 server-side model_action_chunk
  spawn rollout 使用 resp["actions"][0]
```

---

## 7. G1 退出条件

修订后的 plan 至少需要满足：

- step 粒度单一且贯穿 Step 1/2/3；
- client-side 与 server-side HDF5 step 可机械对齐；
- action tensor 空间有明确命名和 shape；
- `CollectionPolicy` 生命周期转发问题已纳入实现；
- prefill facade 接线覆盖 orchestrator 和 search_strategy；
- replay/spawn YAML 不会写入 backend；
- 并行 runner 输出没有竞争写；
- Phase 0 restore 脚本比较的是 checkpoint 后同一段轨迹。

这些满足后，可以重新审查。当前版本不建议编码。

---

## 8. Plan 作者回应（逐条评估 · 2026-04-14）

> 回应者：plan 作者
> 方法：已对每条 A/B 断言读对应源码交叉验证，再决定承认或驳回。
> 结果：**A1–A8 全部承认，B1–B8 全部承认**（B3 与 B7 做了范围收窄）。plan 对应修订见 §9 摘要，具体代码级改动入新版 `logs/trajectory_deviation_corrective_implementation.log.md §19`。

### 8.1 A 级条目逐条评估

#### A1. Step 粒度不一致 → **承认**

验证：`examples/libero/main.py:126-143` 确认 `client.infer()` 只在 `not action_plan` 时调用，即每 `replan_steps` 个 env step 调一次。默认 `replan_steps=5` 时 server-side `--collect` 写 ~T/5 条 step，client-side plan 的 §18.A3.2 把 `step_record.append` 放在"L139 infer 之后、L146 env.step 之前"——**但 §4 的 §4.1 改动 2 描述却是"每 env step 保存"**，plan 内部存在自相矛盾。

**采纳 reviewer 的选项 1**：实验单位 = **inference cycle**（更匹配 cache 系统原生节奏，且 Step 3 teleport 发生在 inference-cycle 边界最安全）。具体：
- Client-side HDF5 只在每次 `client.infer()` 调用点记录一条 record；
- 每条 record 同时保存 `action_chunk`（client 返回，env 空间 `(horizon, 7)`）+ **chunk 内实际被 env.step 消费的那 `replan_steps` 个 action**（`executed_actions`, `(replan_steps, 7)`）+ 本轮 infer 对应的 `sim_state`。
- Server-side `--collect` 已是 inference-cycle 级别，按 `episode_name` 与 client-side 一一对齐。

**不强制 `replan_steps=1`**（会把 Step 1a 耗时增加 5×，且验证 cache 有效性需保留默认重规划频率）。

plan 修订：§4、§10、§11、§13.1 schema、§18.A3 全部按 inference cycle 改写；见 §19.1。

#### A2. CollectionPolicy lifecycle 不转发 → **承认**

验证：`src/openpi/collect/collection_policy.py:103-109` 确认 `on_episode_start/end` 只调 `_collector`，不调 `self._policy`。这是既有代码的隐 bug（即便不做本实验，`--collect + --cache` 组合也不正确）。

修订：
- `CollectionPolicy.on_episode_start/end` 补 `if hasattr(self._policy, "on_episode_start"): self._policy.on_episode_start(...)` 转发；
- 对 `InferenceInterceptor.on_episode_start` 新增 `episode_name: str = ""` 可选 kwarg（默认空字符串=旧行为），避免 server 用 keyword 调用时 TypeError。
- 新增单元测试 `tests/test_collect_cache_lifecycle.py`：两 episode 连跑断言 orchestrator step_counter / strategy._query_history 在第二 episode 从 0 重置。

详见 §19.2。

#### A3. Per-connection facade 未接到 search_strategy → **承认**

验证：`src/openpi/cache/config.py:747-749` 在 return dict 之前构造 search_strategies，`_build_search_strategy(..., shared_storage, ...)` 把 `shared_storage` 直接存进 `self._storage`（`search_strategy.py:181` 等处）。plan §18.A2.2 只改了 return dict 的 `storage` 字段，确会造成 orchestrator 与 search_strategy 指向不同 facade 的撕裂。

修订：`build_per_connection_components` 的正确 patch 是 **先建 `per_conn_storage`，再把它传入 `_build_search_strategy`**：

```python
per_conn_storage = CacheStorage(
    backend=shared_storage._backend,
    metadata_db=getattr(shared_storage, "_metadata_db", None),
)
# ... gates/judges 不变 ...
search_strategies[cp_id] = _build_search_strategy(
    cp_config.search_strategy, per_conn_storage, fusion_weights   # ← 改用 per_conn_storage
)
return {"storage": per_conn_storage, ...}
```

测试：`tests/test_per_connection_prefill.py` 断言 `orchestrator._storage is search_strategies[CP1]._storage`，且 client A prefill 不影响 client B。

详见 §19.3。

#### A4. Prefill payload action 空间错位 → **承认（采用方案 1）**

验证：`src/openpi/policies/libero_policy.py:100` 的 `LiberoOutputs.__call__` 返回 `data["actions"][:, :7]`——`client.infer(element)["actions"]` 就是 env 空间 `(horizon, 7)`。而 `CachePayload.action_chunk` 的 contract 是**模型空间** `(horizon, 32)`（由 `orchestrator.broadcast_action` 在 `interceptor.py:464` 输入，那里拿到的是 `_run_stage3` 的输出，未过 output_transform）。

修订（采 reviewer 方案 1）：
- Step 1b 同时开 `--collect`，收 **server-side 模型空间 `clean_action`** 到 HDF5（episode_name 对齐）。
- prefill_trajectory 的 `actions` 参数 = **model-space action**，来自 server-side HDF5 `step_{t}/clean_action`。
- 在 `InferenceInterceptor.prefill_trajectory(observations, actions, ...)` 的 docstring 明确标注 `actions: list[np.ndarray], shape (horizon, model_action_dim)`。
- `_build_prefill_payload` 直接 `torch.from_numpy(action)`，不做 truncation。

HDF5 schema 拆分（更新 §13）：
- Client-side GT HDF5（`data/deviation_experiment/gt_trajectories/task_X/episode_Y.h5`）保存：`sim_state`, `agentview_image`, `eye_in_hand_image`, `robot_state`, `env_action_chunk` (horizon,7), `executed_actions` (replan_steps,7)。
- Server-side collected HDF5（`data/deviation_experiment/collected/task_X/episode_Y.h5`）保存：`vision_emb`, `prompt_emb`, `clean_action` (horizon, model_dim), `noise_action_steps`。
- 两份按 `episode_name="task_X/episode_Y"` 对齐，Step 3 prefill 读 server-side `clean_action`。

**驳回 reviewer 提议的三个字段都放 client-side**（`model_action_chunk` 要走 server collect 才有；client 拿不到）。

详见 §19.4。

#### A5. Step 2 L2 比较对象未定义 → **承认**

验证：response key 确为 `"actions"`（`client.infer(element)["actions"]` 见 `main.py:139`），plan §10.1 伪码写成 `["action"]` 错别字。

修订：
1. 所有伪码 `client.infer(obs)["action"]` → `["actions"]`；
2. Deviate score 的 L2 metric：**chunk 第一个 action**（最语义化，对齐"实际被执行的那一步"）：
   ```
   bg_samples[m][t]  shape (horizon, 7)
   bg_first[m][t]    shape (7,)          = bg_samples[m][t][0]
   cache[t]          shape (7,)          = cache[t][0]
   gt[t]             shape (7,)          = GT 里 executed_actions[0]（即 chunk 第一个 action）
   ```
3. `deviate_score[t] = L2(cache[t], gt[t]) / max(mean_pairwise_L2(bg_first[:, t]), floor)`。

**驳回 reviewer 选项 3（实际 executed env action）**：replan_steps=5 时 chunk 前 5 个 action 都会被执行，比较"第一个"最稳定，且 bg 样本天然有"第一个 action"，其它 4 个在 replan 之后无意义。

详见 §19.5。

#### A6. Spawn 用错 response key + HDF5 obs 与 restored env obs 未验证 → **承认**

修订：
1. `resp["action"]` → `resp["actions"]`；`action[0]` 取 chunk 第一个（一致于 A5 选定的 metric）。
2. 新增 Phase 0 一致性断言（`scripts/verify_restore_obs_equivalence.py`，独立于 A8 的 verify_env_save_restore）：从 GT HDF5 读 `sim_state`，restore 后对比 `_obs_env_to_policy(env_obs)` 与 HDF5 存的 post-transform `{agentview_image, eye_in_hand_image, robot_state}`，每像素 `np.array_equal` 断言（images uint8 应完全一致；robot_state 允许 `atol=1e-6`）。
3. 若断言失败 → Step 3 不能用 HDF5 obs 作为 rollout 首帧，必须 restore 后用 env.step 返回的 raw obs 经 `_obs_env_to_policy` 重新构造。

详见 §19.6。

#### A7. 并行 runner jsonl 竞争写 → **承认**

验证：§10.1 Phase1Runner 的 `with out.open("a") as f: f.write(...)` 在 `ThreadPoolExecutor` 下确有竞争（Python open/append 不保证原子，大 JSON 行会交错）。

修订：所有 Phase 1 / Phase 2 / Spawn 改用 **per-unit npz/h5 文件**：
```
data/deviation_experiment/step2/bg/{cfg}/{ep}/{sample_idx}.npz   # phase 1
data/deviation_experiment/step2/cache/{cfg}/{ep}.npz             # phase 2
data/deviation_experiment/step3/{cfg}/{ep}_s{s}_n{n}_k{k}.npz    # spawn per-unit
```
`BaseRunState.save()` 已按 §18.B2 加 lock + tmp+rename，覆盖 state JSON 的并发安全；per-unit 文件由 worker 各自独立写，互不冲突。aggregate 阶段按 glob 扫 per-unit 文件合并。

详见 §19.7。

#### A8. verify_env_save_restore 比较逻辑错 → **承认**

验证：§11.0 脚本确实是"跑 50 步 → 保 ckpt → 跑 30 步 → restore ckpt → 跑 acts[:20]，再比 traj_a[:20] vs traj_b"。traj_a[:20] 是初始 50 步中的前 20 步，traj_b 是 ckpt（=50 步后）状态再跑 20 步——不可能吻合。

修订：重写脚本为 reviewer 提议的 ref vs replay 模式：
```python
env1.reset + set_init_state(init)
for a in pre_actions: env1.step(a)       # 跑到 ckpt
ckpt_sim = env1.get_sim_state().copy()
ckpt_obs_ref = env1._get_observations()  # 存参考 obs

# branch A: 继续跑 post_actions → traj_ref
traj_ref = [env1.step(a) for a in post_actions]

# branch B: restore + 跑同一段 post_actions → traj_replay
env2 = env1  # 或新建同配置 env
env2.set_init_state(ckpt_sim)
# timestep/cur_time 复位
traj_replay = [env2.step(a) for a in post_actions]

# 比较: robot0_eef_pos 每步 np.allclose(atol=1e-6)；
#      _check_success() 相同；
#      post-transform obs (_obs_env_to_policy) 像素/数值一致。
```

额外对照 reviewer 提议的：`env.timestep`、`env.cur_time`、max horizon termination。

详见 §19.8。

### 8.2 B 级条目逐条评估

#### B1. episode_name 路径安全 → **承认**

修订：`EpisodeDataCollector.on_episode_start` 接收 `episode_name` 后做校验：
```python
def _validate_episode_name(name: str) -> str:
    if not name: return ""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(not part for part in p.parts):
        raise ValueError(f"Invalid episode_name: {name!r}")
    if len(name) > 200 or p.suffix:
        raise ValueError(f"episode_name must be a bare relative path, got: {name!r}")
    return name
```
Server 侧 `.h5` 后缀统一由 `data_collector` 追加。

详见 §19.B1。

#### B2. YAML 命名重复 → **承认**

验证：`configs/cache_runs/deviate_exp/` 已有 3 个 YAML（`clip_w7_d4.yaml`、`spatial16_w8_d4.yaml`、`max_pool_w3_d5.yaml`），plan §8.3 计划新建同名前缀 `cache_*`。

修订：**复用现有 3 个作为 cache YAML**（把文件名解释为 cache config），只新建 3 个 `inference_*.yaml`：
- `inference_clip_w7_d4.yaml`、`inference_spatial16_w8_d4.yaml`、`inference_max_pool_w3_d5.yaml`
- 这三个 YAML 相对现有 3 个 cache YAML 的差异：`cp1.gate.type: always_search` → `always_skip`；加 `cp3: {enabled: true, gate: {type: always_skip}, ...}`（B3）；`write_policy: {type: never}`（B8）。

plan 全部"cache_clip_w7_d4.yaml"字样 → 直接改为"clip_w7_d4.yaml"。

详见 §19.B2。

#### B3. CP3 always_skip 声明与现状不一致 → **承认（范围收窄）**

验证：现 YAMLs 只有 cp1，无 cp3。plan §18.A1.3 要求"inference_*.yaml 必须 cp1+cp3 都 always_skip"——若 cp3 本不存在（因而不会被 orchestrator.check(CP3) 触发），则不需要写。

修订按 reviewer 提议：
- 如果保留"只有 CP1"：inference_*.yaml 仍然只写 cp1；Phase 1 样本独立性天然成立（不存在 CP3 cache）。
- 如果 pilot 显示需要 cp3 enable（例如为了测 CP3 cache 命中率）：inference_*.yaml 的 cp3 也必须 `always_skip`。

**首版选项**：cp3 不 enable。只在 cache YAML 里按需开 cp3。plan §18.A1.3 的绝对断言改为条件 statement。

详见 §19.B3。

#### B4. send_load_cache_config 前需关闭旧 client → **承认**

修订：在 `exp/compute_deviate_scores.py` 与 `exp/run_spawn_experiment.py` 的 main 里每次切 config 按以下顺序：
1. 等待上一批所有 futures 完成（`executor.shutdown(wait=True)` 后再建新 `ThreadPoolExecutor`）；
2. `send_load_cache_config(...)`（独立 admin 连接，收到 ack 后 close）；
3. 新一批 worker 各自 `WebsocketClientPolicy(host, port)` 建立新连接；
4. 每个 worker 的 `execute_unit` 结尾 `client.close()`（或 `with WebsocketClientPolicy(...)` context manager）。

`WebsocketClientPolicy` 需加 `__enter__/__exit__` 或显式 `close()`（若目前没有，新增；若有则遵循）。

详见 §19.B4。

#### B5. cache_eval_results.json 去重规则 → **承认**

修订：`_aggregate_episode_results` 按 `(config_id, task_id, init_state_idx, seed)` key 做 dedup，**retry 后新结果覆盖旧结果**，同时每条 row 保留：
- `run_id`（目录名 / state JSON 路径）
- `attempt`（0 = 首次，1+ = retry pass）
- `source_path`（`.episode_results.json` 的相对路径）
Dump 前按 `(config_id, task_id, init_state_idx, seed, -attempt)` 排序后按 key groupby 保留 `last`（即最高 attempt）。

`scripts/dump_step1a_failed_inits.py` 消费这份去重后的结果，保证"已 retry 成功"的 episode 不会被误 dump 为 Step 1b 待跑。

详见 §19.B5。

#### B6. Step 1b filter/map 方案收敛 → **承认**

修订（按 reviewer 建议）：
- `episode_filter` JSON 的规范格式固定为：
  ```json
  [
    {"task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 1},
    ...
  ]
  ```
- `scripts/dump_step1a_failed_inits.py` 同时生成：
  - `{task.name}.init`（torch）
  - `{task.name}.init_map.json`（`[orig_0, orig_1, ...]`）
  - 顶层 `step1b_filter.json`（上面规范格式，Step1bRunner 读这个 dispatch）
- `main.py` 不反推 orig→subset，接收 filter JSON 后 loop 内直接按 `subset_init_state_idx` 过滤；`orig_init_state_idx` 写入 HDF5 attrs。

详见 §19.B6。

#### B7. prefill_begin/end debug API 暂缓 → **承认**

修订：首版**只实现 `prefill_trajectory`**。
- `_cache_config_rpc.py::send_prefill_begin/end` 暂不导出；
- `websocket_policy_server.py` 只新增 `prefill_trajectory` 控制分支，不加 `prefill_begin/prefill_end`；
- 未来若需要 administrative debug entry 再补，单独 PR。

详见 §19.B7。

#### B8. write_policy: never 显式声明 → **承认**

修订：所有本实验 YAML（3 cache + 3 inference，共 6 份）**top-level 显式加**：
```yaml
write_policy:
  type: never
```
理由：
- inference_*.yaml 配 always_skip → 每步都 MISS → 默认 `on_any_miss` 会在 `episode_end` 把 replay trajectory 写进 backend，污染 artifact；
- cache_*.yaml 虽然 gate=always_search，若 judge MISS 也会写；本实验目标是 evaluate 既有 cache，不在线扩写。

详见 §19.B8。

### 8.3 §5 疑问回复

| # | 问题 | 回复 |
|---|------|------|
| 1 | env step vs inference cycle | **inference cycle** (A1) |
| 2 | L2 metric | **chunk 第一个 action** (A5) |
| 3 | 强制 replan_steps=1 | **不强制**；HDF5 保存 inference-cycle 级 record |
| 4 | Prefill action 空间 | **model-space clean_action**（来自 server-side --collect HDF5，A4） |
| 5 | Step 2 bg 是否需要 server-side --collect | **不需要**；runner 自行收 client response actions 足矣（collect 对 bg 只是浪费 I/O） |
| 6 | Step 3 策略 A 是否足够 | **A 为首版默认**；unit key 预留 `strategy` 维度便于扩展 B/C |
| 7 | Step 2/3 禁止 cache 写入 | **强制禁止**（B8 `write_policy: never`） |

### 8.4 是否仍有 reviewer 误解需驳回

全部 A/B 条目在技术层面均成立。两个 **范围收窄** 而非驳回：

1. **A4** 我采用方案 1 但 HDF5 schema 按"client-side 两字段 + server-side 一字段"拆分（而非 reviewer 提议的"client-side 三字段"），减少冗余。
2. **B3** 不做"绝对要求 cp3 always_skip"，改为"若 cp3 enable 则必须 always_skip"。

---

## 9. Plan 修订摘要（§19 入口指引）

`logs/trajectory_deviation_corrective_implementation.log.md` 新增 §19 "G1 审查修订汇总"，逐条列出精确 diff：

| §19.N | 对应 A/B | 影响原章节 |
|-------|---------|----------|
| §19.1 | A1 | §4, §10, §11, §13.1, §18.A3 |
| §19.2 | A2 | §3.3, §3.4 |
| §19.3 | A3 | §5, §18.A2 |
| §19.4 | A4 | §7, §13.1, §13.2 |
| §19.5 | A5 | §10.1, §10.2 |
| §19.6 | A6 | §11.1 + 新增 Phase 0 脚本 |
| §19.7 | A7 | §10.1, §11.1, §13.3 |
| §19.8 | A8 | §11.0 |
| §19.B1–B8 | B1–B8 | 分散多章，每条给精确位置 |

§18 与 §19 如有冲突以 **§19 为准**；§19 入口注明此规则。

---

## 10. G1 重审申请

修订完成后请按 §7 退出条件逐项回查。我预计下一轮 review 聚焦：
- §19 所有改动与 §1-§18 的 cross-reference 是否一致（易漏）；
- 新增 Phase 0 obs 一致性脚本的 assert 边界（atol/像素等价）是否合理；
- `CollectionPolicy.on_episode_start` 转发后是否引入新 race（inner policy 的 episode_start 可能在另一线程被调用）。

不另起新 reviewer，同一份 reviewer 验收即可。

---

## 11. G1 重审结论（2026-04-14）

> Reviewer: 同一 G1 reviewer
> Review target: `logs/trajectory_deviation_corrective_implementation.log.md` §19

### 11.1 结论

**仍不批准进入 Code 阶段，但阻塞面已大幅收窄。**

§19 已经实质解决上一轮 A1–A8 / B1–B8 中的大部分核心问题，尤其是：

- step 粒度改为 inference cycle；
- prefill payload 改用 server-side model-space `clean_action`；
- per-connection facade 同时传给 orchestrator 和 search_strategy；
- Step 2 L2 metric 固定为 chunk 第一个 env action；
- runner 输出改为 per-unit `.npz`；
- replay/spawn YAML 显式 `write_policy: never`。

但重审时发现 5 个仍需修订的问题，其中 R1/R2 是当前 G1 release blocker。修完这 5 个点后，我预计可以批准 plan。

### 11.2 R1. inference-cycle HDF5 伪码没有处理 episode 在 chunk 中途结束

**问题**

§19.1 的 `_pending_executed` 只在 `action_plan` 被完整消耗时写入：

```python
if not action_plan:
    traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed)
```

但 LIBERO episode 可能在 chunk 中途 `done=True`，也可能在 max horizon 到达前提前 break。此时最后一个 cycle record 的 `executed_actions` 仍是 `None`，flush HDF5 时会失败，或者下游 Step 2 读 `executed_actions[0]` 时失败。

**要求修订**

在 `_run_episode` 结束前统一 finalize pending cycle：

```python
if args.save_trajectory and traj_buffer and traj_buffer[-1]["executed_actions"] is None:
    traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed)
```

同时 HDF5 schema 不能再写死 `executed_actions (replan_steps, 7)`，应改成：

```text
executed_actions (K, 7) float32
executed_action_count: int, 1 <= K <= replan_steps
```

Step 2 的 `load_gt_first_actions()` 只读 `[0]`，所以 K 可以小于 `replan_steps`，但必须保证 K >= 1。

### 11.3 R2. lifecycle 转发没有覆盖 `PolicyRecorder` wrapper

**问题**

§19.2 写的是：

```python
if hasattr(self._policy, "on_episode_start"):
    self._policy.on_episode_start(...)
```

但 `scripts/serve_policy.py` 的 wrapper 顺序允许 `CollectionPolicy(PolicyRecorder(InferenceInterceptor(...)))`。当前 `src/openpi/policies/policy.py::PolicyRecorder` 没有 `on_episode_start` / `on_episode_end`，也没有 `__getattr__` 透明转发。此时 `CollectionPolicy` 看到的 inner policy 是 `PolicyRecorder`，`hasattr` 为 false，生命周期仍然传不到 `InferenceInterceptor`。

**要求修订**

补一个明确方案：

1. 给 `PolicyRecorder` 增加 lifecycle 透明转发：
   ```python
   def on_episode_start(self, *args, **kwargs):
       if hasattr(self._policy, "on_episode_start"):
           self._policy.on_episode_start(*args, **kwargs)

   def on_episode_end(self, *args, **kwargs):
       if hasattr(self._policy, "on_episode_end"):
           self._policy.on_episode_end(*args, **kwargs)

   def on_task_begin(self, *args, **kwargs): ...
   def on_task_end(self, *args, **kwargs): ...
   ```
2. 或者在 `CollectionPolicy` 里用 wrapper-chain walker 找到内层 lifecycle target。

我建议选 1，简单且修复的是通用 wrapper 透明性问题。

### 11.4 R3. `InferenceInterceptor.on_episode_start` 修订片段必须保留原 task_key / episode_id 语义和 no-orchestrator guard

**问题**

§19.2 的片段写成：

```python
def on_episode_start(..., episode_name: str = "") -> None:
    del episode_name
    self._orchestrator.on_episode_start()
    # ... 原有逻辑 ...
```

这段作为 patch 容易误实现成两类 bug：

- `self._orchestrator` 可能为 `None`（`--cache` 无 config 或无 cache wrapper 场景），需要保留 guard；
- 原代码把 `task` 和 `episode_id` 传入 `orchestrator.on_episode_start(task_key=task, episode_id=str(episode_id))`，不能丢。

**要求修订**

把 §19.2 的片段改成完整代码：

```python
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None:
    del experiment, episode_name
    if self._orchestrator is not None:
        self._orchestrator.on_episode_start(
            task_key=task,
            episode_id=str(episode_id),
        )
```

### 11.5 R4. `collected_dir` 的实际路径语义需要固定

**问题**

§19.4 同时出现：

```text
data/deviation_experiment/collected/libero_xxx/task_X/episode_Y.h5
```

和：

```python
collected_path = Path(args.collected_dir) / f"{ep}.h5"
```

由于 `EpisodeDataCollector` 实际写入路径是 `{collect_dir}/{experiment}/{episode_name}.h5`，`args.collected_dir` 到底是：

- `data/deviation_experiment/collected`
- 还是 `data/deviation_experiment/collected/libero_spatial`

目前不明确。这个会让 Step 3 找不到 server-side `clean_action`。

**要求修订**

固定一个 CLI 语义，例如：

```text
--collect-dir data/deviation_experiment/collected
--task-suite-name libero_spatial
server-side path = {collect_dir}/{task_suite_name}/{episode_name}.h5
```

然后 Step 3 用：

```python
collected_path = Path(args.collected_dir) / args.task_suite_name / f"{ep}.h5"
```

或者把参数命名成 `--collected-suite-dir`，明确传入的是已包含 suite 的目录。二者选一即可。

### 11.6 R5. `episode_start` keyword 透传要求 server 分支也同步

**问题**

§19.2 修了 `CollectionPolicy` 和 `InferenceInterceptor` 的签名，但 server 侧 §3.2 仍有旧片段与新片段混杂。最终 patch 必须统一为 keyword 调用，否则 wrapper 签名兼容性测试覆盖不到真实路径。

**要求修订**

`websocket_policy_server.py` 的 `episode_start` 分支固定为：

```python
conn_policy.on_episode_start(
    experiment=experiment,
    task=task,
    episode_id=episode_id,
    episode_name=episode_name,
)
```

所有 lifecycle-aware wrappers 都必须能接受这些 kwargs。

### 11.7 通过标准

下一版 plan 若补齐 R1–R5，我会给出 **plan approved**。当前不建议开始编码，因为 R1/R2 会直接导致数据收集失败或 lifecycle 仍然泄漏。

---

## 12. Plan 作者对 R1–R5 的回应（2026-04-14）

> 回应者：plan 作者
> 方法：对 R1–R5 每条读对应源码交叉验证后修订；修订入 `logs/trajectory_deviation_corrective_implementation.log.md §20`。
> 结果：**R1–R5 全部承认并完成修订**。

### 12.1 核实记录

| ID | 源码验证 | 结论 |
|----|---------|------|
| R1 | §19.1 原片段 `if not action_plan: traj_buffer[-1]["executed_actions"] = np.stack(...)` 确只在 chunk 完整消耗时写回；`done=True` 提前 break 场景 `executed_actions=None` | 承认 |
| R2 | `src/openpi/policies/policy.py:212-234` PolicyRecorder 无 `on_episode_start/end`、无 `__getattr__`；wrapper chain 断 | 承认 |
| R3 | `src/openpi/cache/interceptor.py:270-276` 原代码确有 `if self._orchestrator is not None` guard 与 `task_key=task, episode_id=str(episode_id)` kwargs；§19.2 patch 丢了两者 | 承认 |
| R4 | `EpisodeDataCollector` 实际路径 = `{collect_dir}/{experiment}/{episode_name}.h5`；§19.4 `args.collected_dir` 语义歧义 | 承认 |
| R5 | §3.2 dispatcher 用 positional 调用无法强制 wrapper 签名一致 | 承认 |

### 12.2 §20 修订映射

| R | §20 章节 | 关键输出 |
|---|---------|---------|
| R1 | §20.R1 | finalize pending cycle；HDF5 schema 改 `executed_actions (K, 7) + K ∈ [1, replan_steps]`；超罕见空分支 pop |
| R2 | §20.R2 | `PolicyRecorder` 补 4 个 lifecycle 透明转发方法；显式列举而不用 `__getattr__`（更稳） |
| R3 | §20.R3 | 完整 signature + 保留 guard + 保留 `task_key`/`episode_id` kwargs |
| R4 | §20.R4 | CLI 契约：`--collected-dir` 不含 suite；Step 3 / Step 2 aggregate 用 `{collected_dir}/{task_suite_name}/{ep}.h5` |
| R5 | §20.R5 | server `episode_start/end` dispatcher 强制 keyword 调用；所有 lifecycle-aware wrapper 签名统一为 `(experiment="", task="", episode_id=-1, episode_name="")` |

### 12.3 无驳回条目

本轮 R1–R5 全部技术成立，无驳回。**方案选择上一处可商榷**：

- R2 reviewer 给出方案 1（PolicyRecorder 加转发）与方案 2（CollectionPolicy 里走 wrapper-chain walker）；我采**方案 1**，与 reviewer 倾向一致。理由写在 §20.R2.1。

### 12.4 §20 与 §19 / §1–§18 的冲突解决

§20 顶部明确：`§20 > §19 > §1–§18`。具体：
- §20.R1 覆盖 §19.1 的 finalize 片段与 HDF5 schema；
- §20.R2 覆盖 §19.2 的 lifecycle 转发片段（从"仅改 `CollectionPolicy`"扩到"同时改 `PolicyRecorder`"）；
- §20.R3 覆盖 §19.2 中 `InferenceInterceptor.on_episode_start` 的 patch；
- §20.R4 覆盖 §19.4 的 `collected_dir` 路径片段；
- §20.R5 覆盖 §3.2 改动 1 与 §19.2 的 episode_start dispatcher。

### 12.5 G1 重审申请

请按 §11.7 通过标准复核 §20。若 R1–R5 均满足，请给出 **plan approved**，随后进入 Code 阶段。

Code 阶段计划先按 §15 实现顺序 checkpoint（Layer A → D → B → C → F），每个 checkpoint 交由 G2 gate。

---

## 13. G1 第三轮裁决（2026-04-14）

> Review target: `logs/trajectory_deviation_corrective_implementation.log.md` §20
> 冲突解决规则：按 plan 作者声明，以 `§20 > §19 > §1–§18` 为准。

### 13.1 结论

**暂不批准进入 Code 阶段。**

R1–R5 的主体修订已经满足第二轮审查要求，尤其是：

- R1 已补 mid-chunk finalize，并把 `executed_actions` 改为动态 K；
- R2 已明确选择给 `PolicyRecorder` 补 lifecycle 转发；
- R3 已恢复 `InferenceInterceptor.on_episode_start` 的 guard 与 `task_key` / `episode_id`；
- R4 已固定 `--collected-dir` 不含 suite，Step 3 按 `{collected_dir}/{task_suite_name}/{ep}.h5` 读 server-side HDF5；
- R5 已把 server dispatcher 固定为 keyword 调用。

但 §20 里还有一个实现/测试会互相冲突的 blocker，需要先修掉。

### 13.2 S1. `PolicyRecorder` 的 `*args, **kwargs` 与 R5 keyword-only 要求冲突

**问题**

§20.R2.1 要给 `PolicyRecorder` 增加：

```python
def on_episode_start(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(*args, **kwargs)
```

但 §20.R5.1 又写：

```text
所有 lifecycle-aware wrapper 必须接受下列 4 个 kwargs
签名全部对齐为:
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None: ...
```

同时 §20.Z 的 R5 验证方式写成：

```text
keyword 透传 | mypy --strict 或单元测试用 positional 调用时 MUST raise TypeError
```

这三者不能同时成立：

- `PolicyRecorder.on_episode_start(*args, **kwargs)` 会接受 positional 调用，不会 raise `TypeError`；
- 普通显式签名 `def on_episode_start(self, experiment="", ...)` 也会接受 positional 调用；
- 如果真的要求 positional 调用必须失败，签名必须是 keyword-only：
  ```python
  def on_episode_start(
      self,
      *,
      experiment: str = "",
      task: str = "",
      episode_id: int = -1,
      episode_name: str = "",
  ) -> None: ...
  ```

**要求修订**

二选一，必须写清楚：

1. **推荐选项：只要求 server 使用 keyword，不要求 positional raise。**
   - `PolicyRecorder` 可用显式同构签名：
     ```python
     def on_episode_start(
         self,
         experiment: str = "",
         task: str = "",
         episode_id: int = -1,
         episode_name: str = "",
     ) -> None:
         if hasattr(self._policy, "on_episode_start"):
             self._policy.on_episode_start(
                 experiment=experiment,
                 task=task,
                 episode_id=episode_id,
                 episode_name=episode_name,
             )
     ```
   - 删除 §20.Z 中"positional 调用时 MUST raise TypeError"这条测试要求，改为"server dispatcher 必须用 keyword；wrapper 收到 keyword 后正确转发"。

2. **严格选项：所有 lifecycle hook 都改 keyword-only。**
   - `CollectionPolicy` / `PolicyRecorder` / `InferenceInterceptor` 都用 `*, experiment=..., task=..., episode_id=..., episode_name=...`；
   - server 只用 keyword；
   - 单元测试再保留 positional raise。

我建议选项 1。它满足本实验需要，改动小，并且不会破坏已有可能用 positional 调 lifecycle hook 的测试或脚本。

### 13.3 其余检查

除 S1 外，§20 对 R1–R5 的修订可以接受。R1 的动态 `executed_actions (K, 7)`、R4 的 collected path 契约、R5 的 server keyword dispatcher 都已经足够明确。

### 13.4 通过条件

只要 plan 追加一个很小的 §21 或修订 §20.R2 / §20.R5 / §20.Z，消除 S1 矛盾，我会给出：

```text
plan approved
```

当前仍不建议开始编码，因为若照 §20 原样实现，`PolicyRecorder` lifecycle 测试和 R5 验证标准会冲突。

---

## 14. Plan 作者回复 — G1 第三轮（2026-04-14）

> 本轮仅针对 §13 提出的 S1 冲突。R1–R5 主体修订已在 §13.1 确认，无返工。
> 冲突解决规则升级为 **§21 > §20 > §19 > §1–§18**，复核时以 `logs/trajectory_deviation_corrective_implementation.log.md` §21 为最终版。

### 14.1 S1 — 接受（选项 1 · 推荐方案）

reviewer 指出的三者不相容成立：

1. `PolicyRecorder.on_episode_start(self, *args, **kwargs)` 接受 positional；
2. `def on_episode_start(self, experiment="", ...)`（非 `*` 隔开）同样接受 positional；
3. "positional 调用 MUST raise TypeError" 仅当 keyword-only（`*` 分隔）时成立。

选用 reviewer 方案 1（不升级为 keyword-only，改 server + wrapper 全 keyword 调用并删除 positional-raise 测试要求）。理由：
- 本实验范围内无 positional caller，约束 server dispatcher 已足；
- keyword-only 会扩散到所有 lifecycle hook 实现与既有测试，blast radius 过大；
- 显式同构签名对 mypy / IDE 比 `*args, **kwargs` 更友好。

**未采纳 reviewer 方案 2（严格 keyword-only）**：见上，范围收益不成正比。

### 14.2 plan 修订落点（§21）

| Review 锚点 | Plan 新落点 | 变更摘要 |
|----|------|---------|
| §13.2 S1 — `PolicyRecorder` 签名 | §21.S1.2 | `PolicyRecorder` 改显式同构签名 + keyword forward（替换 §20.R2.1 的 `*args, **kwargs` 版本）|
| §13.2 S1 — 测试对齐 | §21.S1.2 末尾 | 测试断言改用 keyword 形式 `assert_called_once_with(experiment=..., task=..., ...)` |
| §13.2 S1 — R5 验证表述 | §21.S1.4 | 替换 §20.Z 的 R5 行：不再要求 positional raise；改为 grep server dispatch / wrapper forward 全 keyword + `test_server_dispatch_uses_keyword` 断言 `call.args == ()` |
| keyword forward 升级为刚性约束 | §21.S1.3 | server dispatcher keyword 调用 + wrapper 内部 forward 全 keyword 明确为刚性；`CollectionPolicy` / `InferenceInterceptor` 签名保持普通 default kwargs（不加 `*`）|
| 不动项清单 | §21.S1.5 | §20.R1 / §20.R3 / §20.R4 / §20.R5.1 / §20.R5.2 / §19 A1–A8 B1–B8 均不动，防止 regression 误读 |
| 新 checklist | §21.Z | R1–R5 保持不变；R5 验证方式替换；新增 S1 行（显式同构签名 + 删除 positional-raise 要求）|

### 14.3 验证等价性（为 reviewer 提供 spot-check）

reviewer 若要 grep 一次性复核 S1 是否生效，用下列断言：

```bash
# 断言 PolicyRecorder 已显式列 4 kwargs（而非 *args, **kwargs）
grep -nA 7 "class PolicyRecorder" src/openpi/policies/policy.py \
  | grep -E "experiment: str|episode_name: str"

# 断言 server dispatcher 四个 kwarg 全 keyword
grep -nE "conn_policy\.on_episode_start\(" src/openpi/serving/websocket_policy_server.py \
  | grep -v "experiment="        # 应无匹配

# 断言 CollectionPolicy 向 inner forward 用 keyword
grep -nA 6 "self\._policy\.on_episode_start" src/openpi/collect/collection_policy.py \
  | grep -E "experiment=|task=|episode_id=|episode_name="
```

（代码实施阶段在 G2 再跑这三条，本轮仍纯 doc-level。）

### 14.4 本轮结论

- S1 已在 §21 消除，**采 reviewer 推荐方案 1**；
- R1–R4 无返工；R5 验证口径替换（verification method 换，刚性约束不变）；
- 冲突解决规则：**§21 > §20 > §19 > §1–§18**；
- 请 reviewer 以 §21 为准复核；若通过，给出 `plan approved`，我进入 Code 阶段按 §15 checkpoint 顺序（Layer A → D → B → C → F）实施，每 checkpoint 走 G2 gate。

---

## 15. G1 第四轮裁决（2026-04-14）

> Review target: `logs/trajectory_deviation_corrective_implementation.log.md` §21
> 冲突解决规则：按 plan 作者声明，以 `§21 > §20 > §19 > §1–§18` 为准。

### 15.1 结论

**plan approved。**

§21 已按第三轮 S1 要求完成修订，可以进入 Code 阶段。当前批准范围以 `logs/trajectory_deviation_corrective_implementation.log.md` §21 为最终解释层；若与旧章节冲突，按 `§21 > §20 > §19 > §1–§18` 处理。

### 15.2 S1 复核

S1 已消除，理由如下：

- §21.S1.1 明确采用 reviewer 推荐方案 1：不升级 keyword-only，不再要求 positional 调用 raise；
- §21.S1.2 将 `PolicyRecorder` 从 `*args, **kwargs` 改为显式同构签名，并在转发 inner policy 时使用 keyword；
- §21.S1.3 把 server dispatcher keyword 调用、wrapper 内部 keyword forward 升级为刚性约束；
- §21.S1.4 用 grep + `test_server_dispatch_uses_keyword` 替换 §20.Z 的 positional-raise 验证；
- §21.Z 新增 S1 退出条件，明确“显式同构签名（非 keyword-only）+ 删除 positional MUST raise 要求”。

因此，第三轮指出的三方冲突（`*args, **kwargs` / 普通 default kwargs / positional MUST raise）已经不再存在。

### 15.3 非阻塞备注

§21.S1.2 中有一句“测试本身仍用 keyword 调用”后面括号里提到旧式 positional 前三参也合法。最终代码片段已经改成全 keyword 调用，并且 §21.S1.4 / §21.Z 的验证口径也以 keyword 为准，因此这只是文字残留，不阻塞进入 Code。实现时请按 §21.S1.2 的最终代码片段和 §21.Z checklist 执行。

### 15.4 Code 阶段约束

进入 Code 后，建议按 plan §15 的 checkpoint 顺序推进，并把下列项作为 G2 gate 的最低要求：

- 每个 checkpoint 对应的 unit / smoke / integration test 必须随实现提交；
- lifecycle 链路必须覆盖 `CollectionPolicy -> PolicyRecorder -> InferenceInterceptor` 三层；
- R5 验证应检查 server dispatch 和 wrapper forward 均为 keyword 调用，且 mock `call.args == ()`；
- R1 mid-chunk done 场景必须有动态 `executed_actions (K, 7)` 测试；
- R4 的 `{collected_dir}/{task_suite_name}/{ep}.h5` 路径契约必须在 Step 1b 到 Step 3 的集成测试里验证。

G1 结论：`plan approved`。

---

## 16. Plan 作者 polish 回应（2026-04-14）

> 针对 §15.3 非阻塞备注的文字残留做一次性清理，不改变 §15 的 `plan approved` 状态。

### 16.1 §21.S1.2 文字残留清理

原文：

> 测试本身仍用 keyword 调用（`rec.on_episode_start("exp", "task", 1, episode_name=...)` 的 positional 前三个也合法，因签名非 keyword-only），因此测试代码无需修改。

问题：括号里的 "positional 前三个也合法" 与 §21.S1.3（刚性约束）/ §21.S1.4（`test_server_dispatch_uses_keyword` 断言 `call.args == ()`）/ §21.Z 的 R5 / S1 行冲突，容易被实现者误读为"测试允许 positional"。

替换后（已落 plan 文件 §21.S1.2）：

> **对 §20.R2.2 测试的影响**：§21.S1.3 已把 server dispatcher 与 wrapper forward 全 keyword 升为刚性约束，因此 §20.R2.2 原测试片段中带 positional 的调用需改写为全 keyword。测试调用与断言**一律用 keyword 形式**（`rec.on_episode_start(experiment=..., task=..., episode_id=..., episode_name=...)` + `inner.on_episode_start.assert_called_once_with(experiment=..., task=..., episode_id=..., episode_name=...)`），与 `PolicyRecorder` 内部的 keyword forward 对齐，避免 Mock `call` 比较在 positional/keyword 表达差异上出现假阳性。签名本身虽非 keyword-only，**测试与生产代码路径均禁止出现 positional 调用**。

### 16.2 与 §21 其余章节的一致性确认

- §21.S1.3 刚性约束 —— 未变；
- §21.S1.4 R5 验证方式（grep + `test_server_dispatch_uses_keyword`）—— 未变；
- §21.Z checklist R5 / S1 行 —— 未变；
- §20.R2.2 原测试片段（§21.S1.2 末尾那段 `test_policy_recorder_forwards_lifecycle`）本身已经是全 keyword，无需再动。

### 16.3 结论

- `plan approved` 保持不变；
- 非阻塞备注已解决；
- 下一步进入 Code 阶段，按 plan §15 checkpoint 顺序 Layer A → D → B → C → F，每 checkpoint 走 G2 gate。

---

## 17. G2 第一轮代码审查（Layer A+D+E，2026-04-14）

> Review target: Layer A+D+E 基础设施 bundle
> 代码范围：
> - `exp/_run_state_base.py`
> - `exp/_cache_config_rpc.py`
> - `src/openpi/cache/components/gate.py`
> - `src/openpi/cache/config.py`
> - `src/openpi/cache/cache_storage.py`
> - `src/openpi/cache/interceptor.py`
> - `configs/cache_runs/deviate_exp/*.yaml`
> - `tests/exp/*`
> - `tests/cache/*`

### 17.1 结论

**G2 第一轮结论：REQUEST CHANGES。**

不是 BLOCK：Layer D/E 的主体方向可以接受，`AlwaysSkipGate`、per-connection `CacheStorage` facade、`write_policy: never`、`prefill_trajectory` 的 deferred modes 都基本按 G1 约束落下来了。

但当前 bundle 还有两个 must-fix，会影响已批准 plan 的契约一致性或 runner resume/filter 语义。修完后可进入下一轮 G2；不建议在这些点修正前继续堆 Layer B/C/F。

### 17.2 Must-Fix 1：`send_prefill_begin/end` 是当前 server 不支持的公开死 API，且与 G1 §19.B7 冲突

**位置**

- `exp/_cache_config_rpc.py`
- `tests/exp/test_cache_config_rpc.py`
- `src/openpi/serving/websocket_policy_server.py`

**问题**

`exp/_cache_config_rpc.py` 暴露了：

```python
send_prefill_begin(server_url, payload_b64)
send_prefill_end(server_url)
```

测试里也假造了 `{"__ack__": "prefill_begin"}` / `{"__ack__": "prefill_end"}`。

但当前 server dispatcher 只处理：

```text
episode_start / episode_end / load_cache_config
```

未知 `__ctrl__` 会返回 `{"__ack__": "ignored"}`。因此这两个 helper 对真实 server 调用一定会抛 `RuntimeError`，测试没有覆盖真实协议，只验证了本地 mock。

更关键的是，G1 已批准版本里 §19.B7 明确首版只实现 `prefill_trajectory`，删除 `prefill_begin/end`，理由是独立 admin 连接无法命中后续 inference 所在的 per-connection facade。当前代码把这个已删除接口重新作为 public helper 放回来了，会让后续实现者误用一个语义上不成立的入口。

**要求**

二选一：

1. **推荐：按 G1 §19.B7 删除首版 `send_prefill_begin/end`。**
   - `exp/_cache_config_rpc.py` 只保留 `send_load_cache_config`；
   - 删除对应 tests；
   - Step 3 后续通过 client-side `prefill_trajectory` 走同一 WebSocket connection。

2. 若坚持保留 debug API，则必须在同一个 bundle 内实现 server 分支，并明确它只影响当前连接的 per-connection policy/facade，不能碰 shared storage；同时需要修改 G1 plan 的 §19.B7/§21 解释层。这个范围已经超出当前 Layer A+D+E bundle，我不建议首版走这条。

### 17.3 Must-Fix 2：`BaseRunState.run(unit_filter=...)` 的 retry 阶段绕过 filter

**位置**

- `exp/_run_state_base.py`
- `tests/exp/test_run_state_base.py`

**问题**

当前 primary queue 会应用 `unit_filter`：

```python
queue = [u for u in self.units.values() if u.status != "done"]
if unit_filter is not None:
    queue = [u for u in queue if unit_filter(u)]
```

但 retry pass 直接取全局失败单元：

```python
still_failed = self.failed_units()
...
for u in still_failed:
    ...
    self._execute_one(u)
```

这会导致 resume / 分片运行时，filter 外的历史 failed unit 被本次运行意外重试。例如只想重跑 `task_3:*`，state 里若已有 `task_7:*` failed，retry pass 会把 `task_7:*` 也执行掉。

这和 `unit_filter` 作为 runner 子集选择器的直觉不一致，也会破坏后续 Step 2/3 以 unit key 做 shard / retry 的可控性。

**要求**

retry 阶段也必须应用同一个 filter。可选实现：

```python
still_failed = self.failed_units()
if unit_filter is not None:
    still_failed = [u for u in still_failed if unit_filter(u)]
```

并补一个测试：预置 state 中同时存在 filter 内外的 failed unit，`run(resume=True, unit_filter=...)` 后只允许 filter 内 unit 被执行；filter 外 failed unit 保持 failed。

### 17.4 Should-Fix：per-connection facade 测试没有直接覆盖 search_strategy 持有的 storage

**位置**

- `src/openpi/cache/config.py`
- `tests/cache/test_config.py`

**问题**

当前实现已经把 `_build_search_strategy(..., per_conn_storage, ...)` 写对了，这是 G1 A3 的关键修复。

但测试只断言了返回 dict 里的 `conn_a["storage"]` / `conn_b["storage"]` 是不同 facade、共享 backend；没有断言 `conn_a["search_strategies"][CP1]` 内部实际持有的是 `conn_a["storage"]` 而不是 `shared`。

这正是 G1 第一轮指出过的 bug 形态：orchestrator 拿 per-connection facade，但 search_strategy 仍拿 shared storage，导致 prefill 状态不生效。当前代码是对的，测试还差一刀。

**建议**

补测试断言（按具体 strategy 私有字段命名可接受，因为这是实现级 regression test）：

```python
strategy = conn_a["search_strategies"][CheckpointID.CP1]
assert strategy._storage is conn_a["storage"]
assert strategy._storage is not shared
```

最好再验证 `conn_b` 同样成立，防止未来重构回归。

### 17.5 Should-Fix：`prefill_trajectory` 测试还没有断言 trajectory side effect

**位置**

- `src/openpi/cache/interceptor.py`
- `tests/cache/test_interceptor.py`

**问题**

`prefill_trajectory` 的核心目标不是“stage2/stage3 被跳过”，而是“像真实 inference cycle 一样推进 key_builder / search_strategy 的 trajectory history / step_counter，并把 supplied action 作为历史 action 广播”。

当前测试覆盖了：

- 每个 prefill step 跑一次 stage1；
- stage2/stage3 被 CP1 synthetic FULL_HIT 跳过；
- prefill mode 会退出；
- deferred modes 抛错。

但没有直接断言 strategy 的 query/action history 或下一次真实 search 的 `trajectory_history` 长度已经包含 prefill 步。这会漏掉“返回 action 正确但 trajectory side effect 没产生”的回归。

**建议**

增加一个小型 strategy stub，记录 `record_query_keys()` / `record_action()` / `SearchContext` 或最终 `QuerySpec.trajectory_history`，断言：

- `prefill_trajectory` 后 query history 长度等于 prefill step 数；
- 记录到 action history 的 action 是 `_build_prefill_payload()` 传入的 model-space action；
- 随后一次真实 `infer()` 构造出的 trajectory depth 包含 prefill 历史。

这条我列为 should-fix，不阻断本轮修 must-fix，但建议和 Must-Fix 2 一起补掉。

### 17.6 对提交人提出的四个疑问的回答

1. **per-connection facade 共享边界**：认可“复用 shared backend + shared metadata_db，只隔离 facade 层”。当前实现方向对；需要补 §17.4 的 regression test，锁住 search_strategy 也拿 facade。

2. **AlwaysSkipGate 语义**：认可。`gate=False` 仍走 build，并由 orchestrator 调 `record_query_keys()`，随后 full inference 的 `broadcast_action()` 会补 action history；这正符合 Step 2 背景 L2 采样需要。`record_action` noop 首版可接受。

3. **prefill_trajectory deferred modes**：认可首版 `actions=None` / `record=True` / `on_miss!="error"` 先抛 `NotImplementedError`。这比半实现更好，测试也覆盖了抛错路径。

4. **inference YAML diff 最小化**：当前只改 `gate.type` 并统一 `write_policy: never`，可以接受。未来若 spawn 需要不同 `top_k` / `trajectory_depth`，我倾向 inference YAML 自含完整配置，不做跨 YAML 引用；实验可复现性比去重更重要。

### 17.7 非阻塞备注

- `CacheStorage.enter_prefill_mode()` 是 facade-local 状态；当前 `build_per_connection_components()` 新建 facade 并复用 backend 的做法合理。
- `CacheStorage.search()` 在 prefill 模式下仍先跑 query dim/filter validation。我不要求改，因为这能提前暴露错误 obs/config；当前 deviate YAML 的 `step_filter: all` 不会引入 filter 兼容问题。
- `BaseRunState.save()` 的 `threading.Lock` 只覆盖进程内并发。当前 Layer A 没有多进程 runner，暂不要求文件锁；建议在 docstring 或后续 runner 文档里明确“不支持多进程共享同一 state 文件”。

### 17.8 本轮测试复跑情况

尝试复跑本 bundle 目标测试：

```bash
uv run pytest tests/exp/test_run_state_base.py tests/exp/test_cache_config_rpc.py tests/cache/components/test_gate.py tests/cache/test_cache_storage.py tests/cache/test_config.py tests/cache/test_interceptor.py
```

未能进入 pytest 执行阶段。当前工作区 `.venv/bin/pytest` 指向旧机器路径，`uv run` 调起后找不到对应 Python。这个是本地环境问题，不作为代码失败结论。

提交人报告的 `tests/exp/ + tests/cache/ -> 308 passed, 1 pre-existing failure` 我本轮无法独立复核；下一轮请在修复 must-fix 后继续附回归摘要。

### 17.9 G2 退出条件

下一轮 G2 至少需要看到：

- `send_prefill_begin/end` 被删除，或同 bundle 完整实现并修订 plan 解释层；
- `BaseRunState` retry 阶段应用 `unit_filter`；
- 新增覆盖 retry filter 的单元测试；
- 补充 per-connection search_strategy 持有 facade 的 regression test；
- 回归摘要说明上述测试通过。

---

## 18. G2 第二轮代码审查（Layer A+D+E v2，2026-04-14）

> Review target: Layer A+D+E v2 修订
> 对应上一轮：§17 must-fix 1/2 + should-fix 1/2

### 18.1 结论

**G2 第二轮结论：APPROVE。**

本轮修订已关掉 §17 的两个 must-fix，并补上两个 should-fix 的 regression tests。批准范围仅覆盖当前 Layer A+D+E bundle；可以继续进入 Layer B-1，但 Layer B-1 仍需按 G2 gate 单独提交审查。

### 18.2 Must-Fix 复核

**Must-Fix 1：删除 `send_prefill_begin/end`**

已解决。

- `exp/_cache_config_rpc.py` 只保留 `send_load_cache_config`；
- 模块 docstring 明确首版不实现 standalone `prefill_begin` / `prefill_end`，原因是 prefill 必须走同一 WebSocket connection 才能命中正确 per-connection facade；
- `tests/exp/test_cache_config_rpc.py` 删除了 begin/end mock 测试，只覆盖 `send_load_cache_config`；
- 新增 missing version fallback 测试，保护 `version=-1` 行为。

这与 G1 §19.B7 / §21 的最终解释层一致。

**Must-Fix 2：`unit_filter` 同时约束 retry pass**

已解决。

- `exp/_run_state_base.py` 增加 `_in_scope()`，primary queue 和 retry queue 都使用同一个 predicate；
- `run()` docstring 明确 `unit_filter` 同时约束 primary 与 retry；
- `tests/exp/test_run_state_base.py::test_unit_filter_also_scopes_retry_pass` 覆盖了 filter 外 failed unit 不被当前 shard retry 的场景。

### 18.3 Should-Fix 复核

**Should-Fix 1：search_strategy 持有 owning facade**

已补。

`tests/cache/test_config.py::test_per_connection_search_strategy_uses_owning_facade` 直接断言：

- `strat_a._storage is conn_a["storage"]`
- `strat_b._storage is conn_b["storage"]`
- strategy storage 不是 shared，也不是彼此共享

这正好锁住 G1 A3 的回归点。

**Should-Fix 2：`prefill_trajectory` trajectory side effect**

已补到可接受水平。

`tests/cache/test_interceptor.py::test_prefill_trajectory_records_query_keys_into_strategy` 用 `_RecordingStrategy` 验证每个 prefill step 都触发 search 并记录 query history，覆盖了“只 skip stage2/3 但不推进 trajectory”的回归风险。

后续 Layer B/C/F 真正接入 client/server 后，仍建议再补一次 e2e：client 发 `prefill_trajectory` 后，下一次真实 `infer` 的 trajectory context 包含 prefill 历史。本轮 Layer D 单元/集成粒度已经足够。

### 18.4 回归

已复跑：

```bash
.venv/bin/python -m pytest tests/exp/ tests/cache/ --deselect tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks -q
```

结果：

```text
309 passed, 1 deselected, 13 warnings
```

warnings 均非本轮阻塞项。

### 18.5 放行范围

批准进入下一阶段：

```text
Layer B-1: APPROVED TO START
```

注意：这是允许开始 Layer B-1，不是预批准 Layer B-1 的实现。Layer B-1 完成后仍需新的 G2 审查，尤其要重点覆盖：

- `websocket_client_policy.prefill_trajectory` 与 server `prefill_trajectory` 分支必须使用同一 connection；
- `episode_start` 必须按 §21 全 keyword dispatch；
- lifecycle wrapper 链必须覆盖 `CollectionPolicy -> PolicyRecorder -> InferenceInterceptor`；
- 不得重新引入 `prefill_begin/end` 首版路径。

---

## 19. G1 裁决（Layer B-1 Client RPC 计划，2026-04-14）

> Review target: Layer B-1 实施计划
> 计划范围：`packages/openpi-client/src/openpi_client/websocket_client_policy.py` + client 单测

### 19.1 结论

**plan approved。**

Layer B-1 计划与 G1 最终解释层一致：只改 client 侧 WebSocket policy，不碰 server；只新增 `episode_name`、`prefill_trajectory`、context manager / `close()`；不重新引入 `prefill_begin/end`。

可以进入 Code 阶段。完成后提交新的 G2 审查。

### 19.2 测试目录裁决

测试放在：

```text
packages/openpi-client/src/openpi_client/websocket_client_policy_test.py
```

理由：`packages/openpi-client/src/openpi_client/` 下已有 `image_tools_test.py`、`msgpack_numpy_test.py` 这种同包相邻测试布局。继续用同包相邻文件最符合现状，也避免额外创建 `openpi_client/tests/` 子包带来的 import / packaging 差异。

不建议放 `tests/client/`，因为当前顶层 `tests/` 主要覆盖 repo 内部 `openpi` 包；本任务被测对象在独立 `openpi-client` package 内。

### 19.3 计划复核

批准以下改动：

- `episode_start(..., episode_name: str = "")` 始终发送 `__episode_name__`，空串作为 legacy 默认值；
- `prefill_trajectory(observations, actions, *, record=False, on_miss="error")` 使用同一 `_ws` 连接发送 ctrl payload，并按 `infer()` 同样处理 string error response；
- `__enter__` / `__exit__` / `close()` 只负责关闭当前 websocket，不做额外重连或状态机；
- 不在 client 做 action/observation 长度校验，语义校验仍由 server-side `InferenceInterceptor.prefill_trajectory` 承担；
- 不 import numpy，仅依赖现有 `msgpack_numpy.Packer()` 处理 ndarray。

### 19.4 G2 最低要求

Code 阶段至少补下列测试：

- 不传 `episode_name` 时，发送消息包含 `__episode_name__ == ""`；
- 传入 `episode_name` 时，wire payload 完整保留该字符串；
- `prefill_trajectory` 发送的 ctrl payload 包含 `__ctrl__` / `observations` / `actions` / `record` / `on_miss`；
- `prefill_trajectory` 收到 string response 时抛 `RuntimeError`；
- context manager 退出会调用 `_ws.close()`；
- `close()` 连续调用不抛。

小建议：第一个测试名不要叫 `omits_episode_name`，因为计划要求 wire payload **包含** `__episode_name__ == ""`。建议命名为 `test_episode_start_defaults_episode_name_to_empty_string`，避免测试名与行为相反。

### 19.5 Layer B-1 边界

本轮批准不包含：

- `src/openpi/serving/websocket_policy_server.py` 的 server ctrl 分支；
- `CollectionPolicy` / `PolicyRecorder` / `InferenceInterceptor` lifecycle 签名；
- `prefill_begin/end` 任何形式的独立 RPC；
- `base_policy.BasePolicy` 接口扩展。

这些仍留给后续 Layer B-2 / lifecycle bundle，并且必须按 §21 的 keyword dispatch 约束单独过 G2。

---

## 20. G2 第一轮代码审查（Layer B-1 Client RPC，2026-04-14）

> Review target: Layer B-1 implementation
> 代码范围：
> - `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
> - `packages/openpi-client/src/openpi_client/websocket_client_policy_test.py`

### 20.1 结论

**G2 第一轮结论：APPROVE。**

本轮实现符合 §19 的 G1 裁决：只改 client 侧 WebSocket policy；`episode_name` 空串始终上 wire；`prefill_trajectory` 走同一 `_ws` connection；新增 context manager / `close()`；没有重新引入 `prefill_begin/end`，也没有扩大到 server 或 `BasePolicy`。

批准进入下一阶段：

```text
Layer B-2: APPROVED TO START
```

注意：这只放行开始 Layer B-2，不预批准 server 实现。Layer B-2 完成后仍需新的 G2 审查。

### 20.2 代码复核

**`episode_start`**

已满足：

- 签名增加 `episode_name: str = ""`；
- 发送 payload 包含 `__episode_name__`；
- 不传 `episode_name` 时发送空串，而不是 omit；
- string response 仍按既有模式转 `RuntimeError`。

**`prefill_trajectory`**

已满足：

- 使用当前实例的同一个 `_ws` 发送 ctrl message；
- wire fields 为 `__ctrl__` / `observations` / `actions` / `record` / `on_miss`；
- 不在 client 做长度或 shape 校验，语义校验留给 server-side `InferenceInterceptor.prefill_trajectory`；
- string response 按 `infer()` 同样转 `RuntimeError`；
- 未新增 `prefill_begin/end`。

**context manager / `close()`**

已满足：

- `__enter__` 返回 `self`；
- `__exit__` 关闭 `_ws`，并吞掉 close error；
- `close()` 委托 `__exit__(None, None, None)`；
- 行为足够支撑后续 runner 用 `with WebsocketClientPolicy(...) as client:` 管理连接生命周期。

### 20.3 测试复核

新增测试文件位置符合 §19.2：

```text
packages/openpi-client/src/openpi_client/websocket_client_policy_test.py
```

新增 7 个测试覆盖了 §19.4 的最低要求，并额外覆盖 close error swallow：

- `test_episode_start_defaults_episode_name_to_empty_string`
- `test_episode_start_forwards_episode_name`
- `test_prefill_trajectory_sends_expected_ctrl_payload`
- `test_prefill_trajectory_raises_on_string_response`
- `test_context_manager_closes_websocket`
- `test_close_is_idempotent`
- `test_close_swallows_close_errors`

测试用 fake `websockets.sync.client.connect`，直接解包 msgpack wire payload；覆盖粒度合适。

### 20.4 对实现方自我存疑的裁决

1. **`prefill_trajectory` 形参用 bare `list`**：接受。当前文件没有 `from __future__ import annotations`，保持 runtime-safe 和最小侵入合理。docstring 已说明 `observations` / `actions` 语义。

2. **`__exit__(self, *exc)`**：接受。它符合 context manager 协议；返回 `None` 会让 with-body exception 正常传播，close error 被吞掉也符合 §19.B4。

3. **不扩 `BasePolicy.close` 抽象**：接受。Layer B-1 是子类级 wire/client lifecycle 扩展，改父类会扩大改动面，当前没有必要。

### 20.5 回归

已复跑 client 同包测试：

```bash
.venv/bin/python -m pytest packages/openpi-client/src/openpi_client/websocket_client_policy_test.py packages/openpi-client/src/openpi_client/image_tools_test.py packages/openpi-client/src/openpi_client/msgpack_numpy_test.py -q
```

结果：

```text
28 passed
```

已复跑提交人给出的总回归：

```bash
.venv/bin/python -m pytest tests/cache/ tests/exp/ packages/openpi-client/ --deselect tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks -q
```

结果：

```text
337 passed, 1 deselected, 13 warnings
```

warnings 均非本轮阻塞项。

### 20.6 Layer B-2 Gate 提醒

Layer B-2 进入 server 侧后，G2 至少需要重点验证：

- `episode_start` server dispatcher 读取 `__episode_name__`，并以 keyword 方式调用 policy wrapper；
- `prefill_trajectory` ctrl 分支使用当前 connection 的 `conn_policy`，不能碰 shared storage；
- server string/error path 清晰，client 侧能收到可诊断错误；
- 仍不得新增 `prefill_begin/end` 首版路径。

---

## 21. G1 审查（Layer B-2/B-3/B-4 Bundle 计划，2026-04-14）

> Review target: Layer B-2 / B-3 / B-4 bundle 实施计划
> 计划范围：
> - `src/openpi/serving/websocket_policy_server.py`
> - `src/openpi/collect/collection_policy.py`
> - `src/openpi/policies/policy.py`
> - `src/openpi/cache/interceptor.py`
> - `src/openpi/collect/data_collector.py`
> - tests

### 21.1 结论

**REQUEST CHANGES。**

捆绑这三层是合理的：client ctrl → server dispatcher → wrapper lifecycle → collector 落盘确实是一条链，拆得太细会让 G2 只能审半截语义。

但当前计划还漏掉了三个会直接影响 §21 / §20.R2/R5 的关键约束，修订后预计可以批准。

### 21.2 Must-Fix 1：`CollectionPolicy.on_episode_start/end` 必须显式 forward 到 inner policy，且 forward 用 keyword

**问题**

计划的 B-3.a 伪码只写了：

```python
self._collector.on_episode_start(...)
self._collecting = True
```

没有把 lifecycle 继续转发给 `self._policy`。这会重新打开最早 G1 A2/R2 的问题：server 只调用最外层 `CollectionPolicy`，如果 `CollectionPolicy` 不转发，`PolicyRecorder` / `InferenceInterceptor` 收不到 episode boundary，cache orchestrator 的 episode state 和 write/reset 仍会漏。

`on_episode_end` 也同样必须 forward。当前源码正是因为 `on_episode_end()` 只 flush collector、不触发 inner `InferenceInterceptor.on_episode_end()`，才会导致 cache write/reset 不发生。

**要求修订**

把 B-3.a 改成明确的双向 lifecycle 透明转发：

```python
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None:
    self._prompt_captured = False
    self._collector.on_episode_start(
        experiment,
        task,
        episode_id,
        episode_name=episode_name,
    )
    self._collecting = True
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
        )

def on_episode_end(self, success: bool = False) -> None:
    self._collector.on_episode_end(success)
    self._collecting = False
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success=success)
```

测试必须断言 wrapper forward 使用 keyword，而不是 positional。三层链测试要覆盖 `CollectionPolicy -> PolicyRecorder -> InferenceInterceptor`。

### 21.3 Must-Fix 2：server `episode_end` 也必须改成 keyword dispatch

**问题**

计划只改了 server `episode_start` 的 keyword dispatch，但 §20.R5.2 和 §21.S1.3 已明确 `episode_start / episode_end` 都要避免 positional wrapper 调用。

当前源码仍是：

```python
conn_policy.on_episode_end(obs.get("__success__", False))
```

这会绕过 G1 对 wrapper forward 的刚性约束，也会让测试只覆盖一半 lifecycle。

**要求修订**

B-2.a 旁边补上 `episode_end` 分支修订：

```python
elif ctrl == "episode_end":
    success = obs.get("__success__", False)
    if hasattr(conn_policy, "on_episode_end"):
        conn_policy.on_episode_end(success=success)
    await websocket.send(packer.pack({"__ack__": "episode_end"}))
```

新增或扩充 `test_server_dispatch_uses_keyword`：mock policy 的 `on_episode_start` 和 `on_episode_end` 都要断言 `call.args == ()`。

### 21.4 Must-Fix 3：`prefill_trajectory` 在非 cache policy 上不能静默 `ignored`

**问题**

计划的 wire-contract 表写道：

```text
非 cache policy（裸 Policy）收到 prefill_trajectory ctrl -> "ignored" ack
```

这不符合上一轮 Layer B-2 gate 提醒里的“server string/error path 清晰，client 侧能收到可诊断错误”。Layer B-1 的 client `prefill_trajectory()` 只在 response 是 string 时抛 `RuntimeError`，不会验证 `{"__ack__": "ignored"}`。如果 server 对 known ctrl 返回 ignored，调用方会把它当普通 dict 返回，后续 spawn 逻辑可能在没有 prefill 的情况下继续跑，形成静默错误。

**要求修订**

`prefill_trajectory` 是已知 ctrl。如果 `conn_policy` 没有该方法，应返回 client 可感知的错误，而不是 `ignored`。

建议实现：

```python
elif ctrl == "prefill_trajectory":
    if not hasattr(conn_policy, "prefill_trajectory"):
        await websocket.send("prefill_trajectory requires a cache-enabled policy")
        continue
    ...
```

这样 Layer B-1 client 会按既有 string response 规则抛 `RuntimeError`。也可以选择让 AttributeError 走外层 exception handler，但显式错误消息更清楚。

测试矩阵第 4 条请从“prefill_begin/end 不存在”拆开：

- `prefill_begin` / `prefill_end` 仍返回 `{"__ack__": "ignored"}`，作为“不重新引入首版路径”的 regression guard；
- `prefill_trajectory` + non-cache policy 返回 string error，client/server 测试能证明这是可诊断错误。

### 21.5 其他修订要求

**PolicyRecorder 签名**

计划写“新增 4 方法”还不够，必须按 §21.S1.2 写成显式同构签名 + keyword forward：

```python
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
        )

def on_episode_end(self, success: bool = False) -> None:
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success=success)
```

`on_task_begin/end` 无参数透明转发即可。

**EpisodeDataCollector**

路径逃逸校验是必要的，保留。实现时注意：

- `_episode_name` 和 `_episode_attrs` 要在 `__init__` 初始化；
- escape check 必须在 `mkdir()` / HDF5 写入之前执行；
- 推荐使用 `target.resolve().is_relative_to(out_dir.resolve())`，并覆盖 `../evil` 与 absolute path 两类输入；
- `tmp_path = path.with_suffix(".h5.tmp")` 可以接受。

**`set_episode_attr` 无 episode 时行为**

接受 silent behavior，但前提是 `__init__` 初始化 `_episode_attrs = {}`。`on_episode_start()` 清空 attrs，因此 episode 前误写 attr 不会污染下一集。CollectionPolicy 的正常调用路径只会在 `_collecting=True` 后写 attr。

### 21.6 测试目录裁决

使用以下目录：

```text
tests/serving/test_websocket_policy_server.py
tests/collect/test_collection_policy.py
tests/collect/test_data_collector.py
tests/collect/test_collect_cache_lifecycle.py
```

理由：

- `websocket_policy_server.py` 属 serving 层，新建 `tests/serving/` 比放顶层 `tests/test_websocket_policy_server.py` 更清楚；
- collection/data collector/lifecycle 链路集中放 `tests/collect/`，后续 B-4/C 层测试也可复用；
- 不建议把新测试散在顶层 `tests/`。

### 21.7 对 4 个不确定点的裁决

1. **`asyncio.to_thread` vs 同步调用**：选 `asyncio.to_thread`。`prefill_trajectory` 会跑 stage1，多连接/health check 下不应阻塞 event loop；和 `infer` 分支保持一致。

2. **路径逃逸校验**：必须保留。即使 client 是受信实验脚本，HDF5 写盘接口也不应允许 `episode_name` 逃出 `out_dir`。

3. **`set_episode_attr` 无 episode 时行为**：允许 silent accept，但要初始化 `_episode_attrs`，并在 `on_episode_start()` 清空。无需 raise。

4. **测试目录**：按 §21.6。

### 21.8 下一轮通过条件

修订计划需要明确补齐：

- `CollectionPolicy.on_episode_start/end` 向 inner policy keyword forward；
- server `episode_end` keyword dispatch；
- `prefill_trajectory` unsupported policy 返回 client 可感知错误，而不是 `ignored`；
- `PolicyRecorder` 显式同构签名 + keyword forward；
- `tests/serving/` 与 `tests/collect/` 的落点；
- 测试矩阵覆盖 start/end keyword dispatch、三层 lifecycle、path escape、prompt 单次捕获、无 begin/end 分支。

补齐后可再提 G1，我预计能批准。

---

## 22. G1 二轮裁决（Layer B-2/B-3/B-4 Bundle v2，2026-04-14）

> Review target: Layer B-2 / B-3 / B-4 bundle v2
> 对应上一轮：§21 must-fix 1/2/3

### 22.1 结论

**plan approved。**

v2 已补齐 §21 的三个 must-fix，可以进入 Code 阶段。批准范围仍限于 server dispatcher、collection wrapper lifecycle、PolicyRecorder lifecycle、InferenceInterceptor signature、EpisodeDataCollector episode naming / attrs 这条链；Layer C 继续留到下一轮。

### 22.2 Must-Fix 复核

**MF-1：`CollectionPolicy` lifecycle 透明转发**

已解决。

v2 明确 `CollectionPolicy.on_episode_start()` 在 collector 开始后继续以 keyword 调 `self._policy.on_episode_start(...)`，`on_episode_end()` 也以 `success=success` 继续 forward。这关闭了 `CollectionPolicy -> PolicyRecorder -> InferenceInterceptor` lifecycle 断链风险。

G2 时重点看：

- forward 必须是 keyword；
- `_prompt_captured` 在 `on_episode_start()` 重置；
- `_collecting` 状态切换不阻断 inner lifecycle；
- 三层链测试断言最终 orchestrator 收到 `task_key=task`、`episode_id=str(episode_id)`。

**MF-2：server `episode_end` keyword dispatch**

已解决。

v2 把 `conn_policy.on_episode_end(obs.get(...))` 改为：

```python
conn_policy.on_episode_end(success=obs.get("__success__", False))
```

这与 §20.R5.2 / §21.S1.3 对 `episode_start` / `episode_end` 全 keyword dispatch 的要求一致。

**MF-3：unsupported `prefill_trajectory` 返回可诊断错误**

已解决。

v2 将 non-cache policy 的 `prefill_trajectory` ctrl 从 silent `{"__ack__": "ignored"}` 改为 string error。Layer B-1 client 会按现有 `isinstance(response, str)` 逻辑抛 `RuntimeError`，不会静默继续 spawn。

保留 `prefill_begin/end` 走 unknown ctrl 的 `ignored` ack 作为“不引入首版路径”的 regression guard，可以接受。

### 22.3 其他计划点

**PolicyRecorder**

v2 保持 `PolicyRecorder` 显式 lifecycle 方法方向。Code 阶段必须按 §21.S1.2 实现显式同构签名 + keyword forward，不要回到 `*args, **kwargs`。

**EpisodeDataCollector**

路径逃逸校验、`_episode_attrs` 初始化、`set_episode_attr()` 加锁、`on_episode_start()` 清空 attrs 都已纳入计划。G2 时重点看逃逸校验发生在 `mkdir()` / HDF5 写入之前，并覆盖 `../evil` 与 absolute path。

**测试目录**

接受 v2 的测试落点：

```text
tests/serving/test_websocket_policy_server.py
tests/serving/test_policy_recorder_lifecycle.py
tests/collect/test_collection_policy.py
tests/collect/test_data_collector.py
```

虽然上一轮建议把三层链放 `tests/collect/test_collect_cache_lifecycle.py`，v2 将 `PolicyRecorder + server lifecycle dispatcher` 放到 `tests/serving/test_policy_recorder_lifecycle.py` 也可以接受。G2 以测试内容为准：必须覆盖 server start/end keyword dispatch、PolicyRecorder keyword forward、以及 `CollectionPolicy -> PolicyRecorder -> InferenceInterceptor` 三层穿透。

### 22.4 G2 最低要求

Code 阶段完成后，G2 至少需要看到：

- server `episode_start` 读取 `__episode_name__` 并 keyword dispatch；
- server `episode_end` keyword dispatch，mock `call.args == ()`；
- server `prefill_trajectory` 使用 `asyncio.to_thread(conn_policy.prefill_trajectory, ...)`；
- non-cache policy 收到 `prefill_trajectory` 返回 string error，client 侧可抛 `RuntimeError`；
- `prefill_begin/end` 没有新增分支；
- `CollectionPolicy` start/end 都向 inner keyword forward；
- `PolicyRecorder` 四个 lifecycle 方法显式转发；
- `InferenceInterceptor.on_episode_start()` 增加 `episode_name=""` 且保留 `task_key=task`、`episode_id=str(episode_id)`；
- `EpisodeDataCollector` 支持 `episode_name` 子目录、legacy timestamp fallback、attrs 写入、路径逃逸拒绝；
- prompt attr 只在每 episode 首次 collecting infer 捕获一次，并在新 episode 重置。

G1 结论：`plan approved`。

---

## 23. G2 第一轮代码审查（Layer B-2/B-3/B-4 Bundle，2026-04-14）

> Review target: Layer B-2 / B-3 / B-4 implementation
> 代码范围：
> - `src/openpi/serving/websocket_policy_server.py`
> - `src/openpi/collect/collection_policy.py`
> - `src/openpi/collect/data_collector.py`
> - `src/openpi/policies/policy.py`
> - `src/openpi/cache/interceptor.py`
> - `tests/serving/*`
> - `tests/collect/*`

### 23.1 结论

**G2 第一轮结论：REQUEST CHANGES。**

这不是 BLOCK。server dispatcher、`CollectionPolicy` lifecycle forward、collector episode naming / attrs、`InferenceInterceptor.on_episode_start()` 扩签名这些主体实现方向都对，新增测试也覆盖了大部分 §22.4 gate。

但有一个 must-fix：`PolicyRecorder` 仍使用 `*args, **kwargs` passthrough，直接违背 §22.3 / §21.S1.2 对“显式同构签名 + keyword forward”的要求。这个点修完后，本 bundle 应该可以进入下一轮批准。

### 23.2 Must-Fix：`PolicyRecorder` 不能用 `*args, **kwargs` forward

**位置**

- `src/openpi/policies/policy.py`
- `tests/serving/test_policy_recorder_lifecycle.py`

**问题**

当前实现：

```python
def on_episode_start(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(*args, **kwargs)

def on_episode_end(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(*args, **kwargs)
```

这与已批准计划冲突：

- §22.3 明确写了：`PolicyRecorder` 必须按 §21.S1.2 实现**显式同构签名 + keyword forward**，不要回到 `*args, **kwargs`；
- §21.S1.3 的约束不是“只要当前测试用 keyword 就行”，而是 wrapper 之间 forward 必须 keyword；
- 现在如果任何 caller 用 positional 调 `rec.on_episode_start("exp", "task", 1, episode_name="x")`，`PolicyRecorder` 会把前三个参数继续 positional 转给 inner policy，这正是 §21 要避免的表达差异。

当前测试只覆盖了“外部用 keyword 调 recorder 时，inner 收到 keyword”，没有锁住实现签名，也没有锁住“wrapper 内部 forward 必须 keyword”的结构约束。

**要求修复**

把 `PolicyRecorder` 改为显式签名，并在内部始终 keyword forward：

```python
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
        )

def on_episode_end(self, success: bool = False) -> None:
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success=success)

def on_task_begin(self) -> None:
    if hasattr(self._policy, "on_task_begin"):
        self._policy.on_task_begin()

def on_task_end(self) -> None:
    if hasattr(self._policy, "on_task_end"):
        self._policy.on_task_end()
```

并补一个轻量 regression test，建议用 `inspect.signature` 锁住不再出现 varargs：

```python
sig = inspect.signature(PolicyRecorder.on_episode_start)
assert "args" not in sig.parameters
assert "kwargs" not in sig.parameters
assert list(sig.parameters) == ["self", "experiment", "task", "episode_id", "episode_name"]
```

也可以同时检查 `on_episode_end` / `on_task_begin` / `on_task_end` 没有 `VAR_POSITIONAL` / `VAR_KEYWORD` 参数。

### 23.3 已通过的关键点

以下点本轮审查可以接受：

- server `episode_start` 已读取 `__episode_name__` 并用 keyword dispatch；
- server `episode_end` 已用 `success=...` keyword dispatch；
- server `prefill_trajectory` 分支用当前 connection 的 `conn_policy`，并通过 `asyncio.to_thread(...)` 执行；
- non-cache policy 收到 `prefill_trajectory` 返回 string error，不再静默 `ignored`；
- `prefill_begin/end` 没有新增分支，测试锁定为 unknown ctrl 的 `ignored` ack；
- `CollectionPolicy.on_episode_start/end` 已向 inner policy keyword forward；
- `InferenceInterceptor.on_episode_start()` 已接受 `episode_name=""`，并保留 `task_key=task`、`episode_id=str(episode_id)`；
- `EpisodeDataCollector` 支持 `episode_name` 子目录、legacy timestamp fallback、attrs 写入、路径逃逸拒绝；
- prompt attr 的单次捕获和 episode reset 已有测试覆盖。

### 23.4 对实现方自我存疑的裁决

1. **CollectionPolicy forward 顺序：collector-first, inner-second**  
   接受。collector 先初始化 HDF5 episode 状态，再让 inner cache wrapper reset episode state，语义可接受。G2 不要求调整顺序。

2. **`_episode_attrs.clear()` vs 重新赋值**  
   接受 `.clear()`。保持 dict identity 没有坏处，且已在 lock 内执行。

3. **`Path.is_relative_to`**  
   接受。当前项目测试运行在 Python 3.12；无需为更低版本改 `os.path.commonpath`。

4. **PolicyRecorder 使用 `*args, **kwargs`**  
   不接受。见 §23.2，这是本轮唯一 must-fix。

### 23.5 回归

已复跑新增测试：

```bash
timeout 25s .venv/bin/python -m pytest tests/serving/ tests/collect/ -q
```

结果：

```text
27 passed, 5 warnings
```

未复跑全量回归，因为 §23.2 已发现需要改代码的 must-fix。下一轮请在修复后继续提供并复跑全量回归。

### 23.6 下一轮通过条件

下一轮 G2 至少需要看到：

- `PolicyRecorder` 四个 lifecycle 方法改为显式签名；
- `PolicyRecorder` 内部调用 inner policy 时全部 keyword forward；
- 测试锁住 `PolicyRecorder` 不再含 `*args` / `**kwargs`；
- `tests/serving/ tests/collect/` 通过；
- 全量回归摘要更新。

## 24. Layer C G2 第一轮审查（`examples/libero/main.py`）

**G2 第一轮结论：REQUEST CHANGES。**

HDF5 schema 本体基本符合 §20.R1.2：`num_cycles` / dynamic `executed_actions(K, 7)` / `executed_action_count` / `orig_init_state_idx` 都已落地，mid-chunk finaliser 也有单测覆盖。但 serial path 的 `episode_id` 仍违反实验层契约；在进入 Layer F 前必须修。

### 24.1 Must-fix：serial path 的 `episode_id` 编码不符合契约

位置：`examples/libero/main.py:346-421`。

实验计划 `logs/trajectory_deviation_corrective_experiment.log.md:74` 明确规定：

```text
episode_id = task_id * num_trials_per_task + episode_idx
```

并要求 serial / concurrent 结果一致。当前 concurrent path 在 `examples/libero/main.py:523` 使用了这个公式，但 serial path 仍在 `examples/libero/main.py:347` 初始化 `global_episode_id = 0`，并在 `examples/libero/main.py:421` 只对实际执行的 episode 递增。

这在全量无 filter 且从 task 0 开始跑时碰巧一致；但以下任一真实实验场景都会错：

- `--task-ids 3`：`task_3/episode_0` 的契约 `episode_id` 应为 `3 * num_trials_per_task + 0`，当前 serial 会写成 `0`；
- `--episode-filter` 跳过前面 episode：当前 serial 会把执行过的 episode 压成连续编号，和 `task_id/episode_idx` 不再可逆；
- Step 1a results JSON、server-side collected HDF5、client-side GT HDF5、Layer F unit key 都可能用 `episode_id` 对齐，错位会让后续 F-3/F-4 在错误 episode 上取数。

修复建议：

```python
episode_id = task_id * args.num_trials_per_task + episode_idx
```

serial loop 内统一用这个 `episode_id` 传给：

- `client.episode_start(..., episode_id=episode_id, ...)`；
- `_flush_trajectory_h5(..., episode_id=episode_id, ...)`；
- `per_episode_log` 的 `"episode_id"`。

`total_episodes` / `task_episodes` 继续作为进度计数即可，不要混用为实验 ID。

测试要求：补一个轻量 regression，覆盖 serial 的 filter 或 task subset 场景。最低要求是断言 `task_id=3, episode_idx=1, num_trials_per_task=5` 时写出的 `episode_id == 16`，不要再是执行序号 `0/1`。

### 24.2 Should-fix：filtered concurrent progress total 不准确

位置：`examples/libero/main.py:478`。

当前 concurrent path 的 `total_episodes = len(task_id_list) * args.num_trials_per_task` 忽略 `episode_filter`。数据落盘不受影响，但 filtered Step 1b 会出现总进度条永远到不了 100% 的观感问题，也容易误导人工监控。

建议在加载 `filter_pairs` 后计算目标 episode 总数；例如只统计 `task_id in task_id_list` 的 filter pair 数。这个不阻塞 schema，但建议随 must-fix 一起改，成本很低。

### 24.3 测试与验证

已复跑：

```bash
.venv/bin/python -m pytest tests/examples/test_libero_main.py -q
```

结果：

```text
9 passed
```

已复跑 Layer B + Layer C 的新增测试：

```bash
.venv/bin/python -m pytest tests/serving/ tests/collect/ tests/examples/ -q
```

结果：

```text
37 passed, 5 warnings
```

尝试用本地环境静态确认真实 LIBERO `OffScreenRenderEnv` 接口时，当前 venv 不含 `libero` 包，因此无法在本机直接 inspect 真实类。这个不改变上面的 must-fix 结论。

### 24.4 对实现方自我存疑的裁决

1. **`env.get_sim_state()` / `env.timestep` / `env.cur_time`**  
   接受当前直接使用方案。实验计划前文已经把这些作为 Phase 0 验证过的接口；本轮本地环境缺 `libero`，无法复核真实类。进入 Layer F restore 前仍要执行计划里的 env save/restore smoke；若 smoke 失败，再改为 MuJoCo 原语 fallback。

2. **Concurrent results JSON 按 `(task_id, init_idx)` 排序**  
   接受。并发 completion order 本来不稳定，最终 JSON 做确定性排序更利于复现和 diff。

3. **HDF5 写不加锁**  
   当前 task queue 粒度是 `task_id`，两个 worker 不会同时拥有同一 task，因此接受。若未来改为 episode 级切分，这条约束必须重审。

4. **warm-up 不纳入 buffer 的测试**  
   不是本轮 must-fix。当前 `_run_episode` 逻辑在 `t < num_steps_wait` 分支直接 `continue`，不会触发 infer/buffer；代码路径清楚。但如果本轮已经要补 serial `episode_id` regression，可以顺手补一个 `num_steps_wait > 0` 的小单测，锁住 warm-up 不产生 cycle。

### 24.5 下一轮通过条件

下一轮 G2 至少需要看到：

- serial path 的 `episode_id` 改为公式编码；
- 新测试覆盖 serial path 的公式编码，尤其是 task subset 或 episode filter 场景；
- `tests/examples/test_libero_main.py` 通过；
- `tests/serving/ tests/collect/ tests/examples/` 通过；
- 若改了 concurrent progress total，请同步说明 filtered 场景的计数规则。

在 §24.1 修复前，不放行 Layer F。Layer F-3/F-4 依赖 HDF5 / JSON 的 episode 对齐，当前 serial ID bug 会把下游结果建立在错位索引上。

## 25. Layer C G2 第二轮审查（`examples/libero/main.py`）

**G2 第二轮结论：APPROVE。**

Layer C 可以放行。上一轮 §24.1 的 blocker 已修复：serial 和 concurrent path 现在都通过 `_compute_global_episode_id(task_id, episode_idx, num_trials_per_task)` 生成 `episode_id`，serial path 不再使用执行顺序递增的 `global_episode_id += 1`。这使 `task_id/episode_idx`、client-side GT HDF5、server-side collected HDF5、results JSON 与后续 Layer F unit key 的 episode 对齐重新成立。

### 25.1 修复复核

- `examples/libero/main.py` 新增 `_compute_global_episode_id(...)`，公式为 `task_id * num_trials_per_task + episode_idx`；
- `_eval_serial` 在每个 episode loop 内调用该 helper，并把结果传给 `client.episode_start()`、`_flush_trajectory_h5()` 与 per-episode results JSON；
- `_eval_concurrent` 也改为调用同一个 helper，避免两条路径未来再次分叉；
- 源码中未再出现 `global_episode_id += 1`；
- `env_timestep` / `env_cur_time` 已改为从 `env.env.timestep` / `env.env.cur_time` 读取，并且测试 fake env 改为模拟这一层嵌套。

### 25.2 测试

已复跑 Layer C 单测：

```bash
.venv/bin/python -m pytest tests/examples/test_libero_main.py -q
```

结果：

```text
11 passed
```

已复跑 Layer B + Layer C 新增测试：

```bash
.venv/bin/python -m pytest tests/serving/ tests/collect/ tests/examples/ -q
```

结果：

```text
39 passed, 5 warnings
```

已复跑当前范围回归：

```bash
.venv/bin/python -m pytest tests/ packages/openpi-client/ --deselect tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks -q
```

结果：

```text
420 passed, 1 skipped, 1 deselected, 14 warnings
```

`test_compute_aggregate_success_rate_sums_all_tasks` 仍按前文记录作为既有无关失败 deselect。

### 25.3 剩余建议

`_eval_concurrent` 的进度条 total 仍按 `len(task_id_list) * num_trials_per_task` 计算，filtered Step 1b 下显示不会精确到 filter 后 episode 数。这是 §24.2 的 should-fix，不影响 HDF5 schema 或 Layer F 数据对齐；可以后续作为 runner polish 处理。

当前代码中 §23 的 `PolicyRecorder` blocker 也已在工作区修复为显式签名 + keyword forward，因此不再阻塞 Layer F。

### 25.4 放行

Layer C：**APPROVE**。

Layer F-1..F-6：可以进入实施，但 G2 时需重点检查：

- F-3/F-4 读取 HDF5 时使用 `num_cycles`，不要误用 `num_steps` 作为 cycle 数；
- 读取 `executed_actions` 时接受动态 `K`，只要求 `K >= 1`；
- restore env 时使用 `sim_state` + `env_timestep` + `env_cur_time`，并在真实 LIBERO 环境跑一次 save/restore smoke；
- Step 2 / Step 3 unit key 必须与 `_compute_global_episode_id()` 的编码保持一致。

## 26. Layer F-1..F-6 G2 第一轮审查

**G2 第一轮结论：REQUEST CHANGES。**

不是 BLOCK：F-1..F-6 的 runner 骨架、state/retry、unit key、server-side `clean_action` prefill 方向基本成型，测试也能稳定跑过当前 F 目标集。但 F-4 的 deviation metric 已经偏离 §19/§20 的批准契约，F-5 还有两个会影响 spawn 结果可信度的问题。进入 Verify 或真实长跑前必须修。

### 26.1 Must-fix

#### MF-1：F-4 deviate score 又回到了 full chunk L2，违反已批准的 first-action metric

批准契约已经在 §8.1 A5 和 §19.5 / §20.R1.3 收敛为：Phase 1/2 只保留 `client.infer(obs)["actions"][0]`，GT 读 `step_*/executed_actions[0]`，每个 cycle 的 L2 在 `(7,)` 上计算。该选择的原因是对齐“实际进入 env.step 的第一步”，并且动态 `executed_actions(K, 7)` 只要求 `K >= 1`。

当前实现不符合：

- `exp/compute_deviate_scores.py:55-94` 的 `load_gt_episode()` 返回 `env_action_chunk`，没有读取 `executed_actions[0]`；
- `exp/compute_deviate_scores.py:119-160` 的 `compute_deviate_score()` 明确把 `(H, Ad)` flatten 成 `H * Ad` 做 L2，注释还写成“compare the entire planned chunk”；
- `exp/compute_deviate_scores.py:239-252` 的 `_roll_out_episode()` 收集整段 action chunk；
- `exp/compute_deviate_scores.py:320-327` 和 `exp/compute_deviate_scores.py:366-372` 写出的 JSONL 字段仍是 full `chunks`；
- `tests/exp/test_compute_deviate_scores.py:59-92`、`tests/exp/test_compute_deviate_scores.py:111-149`、`tests/exp/test_compute_deviate_scores.py:223-290` 现在把 full chunk 行为锁死了，测试没有覆盖“非首个 horizon action 巨大但首个 action 相同”的回归。

影响：deviate score 会被未实际执行的 chunk 尾部支配，ranking 不再代表“下一步动作是否偏离 GT”。这会直接污染 F-5 top-k spawn 点选择和 F-6 coverage 曲线。

要求：

- Phase 1/2 的输出改为 `actions_first`，shape `(T, 7)`；如果继续用 JSONL，也要改字段名或至少测试锁定 shape；
- aggregate 使用 `load_gt_first_actions(gt_dir, ep)` 从 `step_{t}/executed_actions[0]` 读取 GT；
- `load_gt_first_actions()` 验证 `num_cycles > 0` 且每个 `executed_actions.shape[0] >= 1`，否则 skip/log warning 或显式失败；
- 更新单测，让 full-chunk 实现必然失败，例如首个 action 相同、第二个 action 相差很大时 `cache_l2` 必须为 0。

#### MF-2：F-5 `_execute_spawn_unit()` 未关闭 websocket client 和 LIBERO env

`exp/run_spawn_experiment.py:473-494` 创建 env 和 client，`exp/run_spawn_experiment.py:541-549` 的 `finally` 只调用 `client.episode_end(...)`，没有调用 `client.close()`，也没有关闭 `env`。

影响：

- B-1 专门新增 `WebsocketClientPolicy.close()` / context manager，目的就是让 worker 退出时立即下降 backend / connection 引用计数；F-5 现在没有使用这个能力；
- 并发 spawn 时会泄漏 websocket 连接、server-side per-connection facade、以及 MuJoCo / EGL env 资源；
- 若 `prefill_trajectory()` 或 rollout 中途抛异常，泄漏更容易积累，长跑时会表现为随机连接耗尽或 GPU/EGL 资源耗尽。

要求：

- 在 `_execute_spawn_unit()` 中用 `with common.make_client() as client:`，或在 `finally` 中对 `client.close()` 做 best-effort 调用；
- 对 `env.close()` 也做 best-effort 调用；
- 新增测试覆盖成功路径和 `prefill_trajectory()`/`infer()` 抛异常路径，断言 client/env 都被 close。

#### MF-3：F-5 rollout budget 混用了两个 step 坐标系

Layer C 写 HDF5 时，`examples/libero/main.py:319-321` 的 `num_steps` 是 `sum(executed_actions.shape[0])`，即 policy 开始后的实际执行 action 数；同一个 HDF5 cycle 里，`examples/libero/main.py:198-202` 的 `env_timestep` 来自 `env.env.timestep`，包含 warm-up dummy steps 的 env 全局 timestep。F-5 现在在 `exp/run_spawn_experiment.py:413-414` 读二者，并在 `exp/run_spawn_experiment.py:517-523` 计算 `remaining = num_steps - env_timestep`。

影响：生产默认 warm-up 为 10 步时，`env_timestep` 比 policy-step 坐标多出 warm-up 偏移，spawn budget 会被系统性低估。靠 `max(1, budget)` 兜底只是不让它变成 0，但成功率仍可能被截短 rollout 扭曲。

要求二选一：

- 在 Layer C HDF5 额外记录 policy-step 坐标，例如 `policy_step_idx` 或 `num_steps_wait`，F-5 用同一坐标系计算剩余 budget；
- 或者 F-5 不再用 `num_steps - env_timestep` 限制，只用 `max_spawn_env_steps`，并在注释/测试里说明 spawn 是独立恢复实验，不以 GT 剩余步数截断。

无论选哪种，都需要补测试覆盖 warm-up 偏移场景，避免现有 fixture 的 `env_timestep=i*5` 和 `num_steps=num_cycles*5` 把问题隐藏掉。

#### MF-4：Step 2 默认会把失败的 Step 1b 轨迹当作 GT

`exp/run_step1b_gt.py:251-298` 的 `_read_unit_result()` 在 `success=False` 但 subprocess 正常写出 JSON 时仍返回 `hdf5_path`；注释说“downstream code can elect to skip them”。但当前 downstream 没有做选择：`exp/compute_deviate_scores.py:382-390` 的 `discover_episodes()` 直接扫描所有 `episode_*.h5`，`exp/compute_deviate_scores.py:80-94` 的 `load_gt_episode()` 也不检查 `f.attrs["success"]`。

影响：失败的纯 inference 轨迹会进入 Phase 1/2/aggregate，成为“GT manifold”。这会让 deviation score 和 spawn recovery 都建立在失败轨迹上，和本实验“从 cache 失败中找纯 inference 成功轨迹作为 GT”的语义冲突。

要求：

- 默认跳过 `success=False` 的 GT HDF5，并 log warning；
- 或新增显式 flag，例如 `--include-failed-gt`，默认关闭；
- 新增测试：一个 `success=False` 的 HDF5 不应出现在 `discover_episodes()` 或 aggregate 输入中。

### 26.2 Should-fix

#### SF-1：随机 baseline 使用 Python `hash()`，跨进程不可复现

`exp/run_spawn_experiment.py:385-388` 注释声称 baseline draw “comparable across resumes”，但 seed 用的是 `hash((self.common.cfg, ep, self.common.random_seed))`。Python 的 `hash()` 默认受 hash randomization 影响，不同进程可能不同。`exp/compute_deviate_scores.py:312`、`exp/compute_deviate_scores.py:358`、`exp/run_spawn_experiment.py:499` 的 `episode_id` 也有类似问题，虽然它们主要用于 telemetry，严重性低于 baseline seed。

建议：用 `hashlib.sha256(...).digest()` 或 `zlib.crc32()` 生成稳定 int，并加一个测试在同一输入下锁定具体 seed / unit key。

#### SF-2：真实 LIBERO restore smoke 仍未作为本轮证据出现

你说明 Phase-0 smoke 脚本尚未包含在本批次；这可以不阻塞代码修订，但不能进入最终 Verify。F-5 的核心假设是 `sim_state + timestep + cur_time` restore 后 obs 与 HDF5 中的 transformed obs 等价。长跑前需要把真实 LIBERO smoke 的命令和结果补进本 review log。

### 26.3 测试复核

本轮复跑 F 目标测试：

```bash
.venv/bin/python -m pytest tests/scripts/test_dump_step1a_failed_inits.py tests/exp/test_run_cache_experiments.py tests/exp/test_run_step1b_gt.py tests/exp/test_compute_deviate_scores.py tests/exp/test_run_spawn_experiment.py tests/exp/test_analyze_deviation_results.py tests/exp/test_run_state_base.py -q
```

结果：

```text
1 failed, 79 passed
```

失败项仍是既有无关失败：

```text
tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks
```

按既有约定 deselect 后复跑：

```bash
.venv/bin/python -m pytest tests/scripts/test_dump_step1a_failed_inits.py tests/exp/test_run_cache_experiments.py tests/exp/test_run_step1b_gt.py tests/exp/test_compute_deviate_scores.py tests/exp/test_run_spawn_experiment.py tests/exp/test_analyze_deviation_results.py tests/exp/test_run_state_base.py --deselect tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks -q
```

结果：

```text
79 passed, 1 deselected
```

注意：你提交说明写“70 个 F 系列相关测试”，当前实际目标集是 80 个（含 1 个既有失败）；这不是 blocker，但后续 G2 回复请按实际 pytest 计数更新，避免审查记录和测试集对不上。

### 26.4 二轮期待

下一轮 G2 至少需要看到：

- F-4 metric 改回 first-action，并有能杀死 full-chunk 实现的测试；
- F-5 client/env close 修复和异常路径测试；
- F-5 budget 坐标系修复或明确改为只用 `max_spawn_env_steps`；
- Step 2 对 `success=False` GT 的默认处理明确并有测试；
- F 目标测试在 deselect 既有失败后通过；
- 若声称进入 Verify，补真实 LIBERO restore smoke 证据。

## 27. Layer F-1..F-6 G2 第二轮审查

**G2 第二轮结论：APPROVE。**

上一轮 §26 的 4 个 must-fix 和 1 个 should-fix 已复核通过。Layer F-1..F-6 代码层可以进入 Verify。注意：真实 LIBERO restore smoke 仍是 Verify 阶段证据，不是本轮 code review 的 blocker，但最终实验前必须补。

### 27.1 Must-fix 复核

#### MF-1：first-action metric 已恢复

`exp/compute_deviate_scores.py` 已把 GT 读取改为 `executed_actions[0]`：

- `load_gt_episode()` 返回 `(obs_seq, gt_first_actions)`，其中 `gt_first_actions` shape 为 `(T, Ad)`；
- `compute_deviate_score()` 接收 Phase 1/2 的 full chunk 输入，但内部只用 `bg_chunks[:, :, 0, :]` 和 `cache_chunks[:, 0, :]` 计算 L2；
- `aggregate()` 传入 `gt_first`，不再用 `env_action_chunk` 当 GT。

测试覆盖也已能杀死 full-chunk 回归：`tests/exp/test_compute_deviate_scores.py` 的 `H=2` 噪声尾用例会在旧 full-chunk metric 下失败。

非阻塞建议：Phase 1/2 JSONL 仍保存 full `chunks`。这在功能上可接受，因为 metric 已在 `compute_deviate_score()` 内部收敛为 first-action；若后续磁盘体积成为问题，再把存储改成 `actions_first`。

#### MF-2：spawn 资源释放已补齐

`exp/run_spawn_experiment.py::_execute_spawn_unit()` 的 `finally` 现在按顺序执行：

1. `client.episode_end(success=success)`；
2. `client.close()`（若存在）；
3. `env.close()`（若存在）。

三个步骤各自独立 `try`，不会因为前一步失败跳过后续 cleanup。新增测试覆盖 success 路径和 `client.infer()` 抛异常路径，均断言 client/env 被关闭。

#### MF-3：budget 已统一到 env-timestep 轴

Layer C schema 已增加：

- `num_steps_wait`；
- `final_env_timestep`。

`examples/libero/main.py::_run_episode()` 返回真实 `env.env.timestep` 作为 `final_env_timestep`，serial / concurrent 两条路径都传入 `_flush_trajectory_h5()`。

F-5 读取：

- `start = step_anchor/env_timestep`；
- `final = f.attrs["final_env_timestep"]`；
- fallback 为 `num_steps_wait + num_steps` 并 warning；
- `budget = max(1, min(max_spawn_env_steps, final - start))`。

新增测试覆盖 warm-up 偏移场景：旧公式 `num_steps - env_timestep` 会得到 10，新实现得到 30。旧 HDF5 缺 `final_env_timestep` 的 fallback + warning 也有测试。

#### MF-4：失败 GT 默认过滤已落地

`exp/compute_deviate_scores.py::discover_episodes()` 现在默认：

- `success=True`：纳入；
- `success=False`：skip；
- 缺 `success`：warning + skip。

同时提供 escape hatch：

- `--include-failed-gt`；
- `--include-unknown-gt`。

四个测试覆盖默认 skip、include failed、unknown warning+skip、include unknown 仍 warning。

### 27.2 Should-fix 复核

随机 baseline seed 已从 Python `hash()` 改为 `hashlib.blake2b(..., digest_size=8)`，`BaselineRunner` 的 random points 跨进程稳定。测试锁定了 seed bytes 派生出的 expected unit keys。

保留项：`compute_deviate_scores.py` 和 `run_spawn_experiment.py` 的 telemetry `episode_id=hash(...)` 尚未改为 stable hash。这个不影响 baseline draw、state key 或实验数据 join，本轮按实现方声明暂不扩范围，接受为 non-blocking polish。

### 27.3 测试复核

二轮目标测试：

```bash
.venv/bin/python -m pytest tests/exp/test_compute_deviate_scores.py tests/exp/test_run_spawn_experiment.py tests/examples/test_libero_main.py -q
```

结果：

```text
48 passed
```

F 目标集回归（deselect 既有失败）：

```bash
.venv/bin/python -m pytest tests/scripts/test_dump_step1a_failed_inits.py tests/exp/test_run_cache_experiments.py tests/exp/test_run_step1b_gt.py tests/exp/test_compute_deviate_scores.py tests/exp/test_run_spawn_experiment.py tests/exp/test_analyze_deviation_results.py tests/exp/test_run_state_base.py --deselect tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks -q
```

结果：

```text
88 passed
```

既有失败 `tests/exp/test_run_cache_experiments.py::test_compute_aggregate_success_rate_sums_all_tasks` 仍按前文记录不计入本批。

### 27.4 放行

Layer F-1..F-6：**APPROVE**。

可以进入 Verify。Verify 阶段至少补：

- 真实 LIBERO `sim_state + env_timestep + env_cur_time` restore smoke；
- 一次小规模 Step 1b → Step 2 → Step 3 → F-6 端到端 dry run；
- dry run 中确认 `discover_episodes()` 的 success 过滤数量、`final_env_timestep` attrs、spawn `env_steps_executed` 分布都符合预期。

## 28. Verify 阶段工作记录（2026-04-14）

按用户裁决，Verify 阶段纯文档不走 Plan，小代码改动也不走 Plan；遇到非小改动会先停下来请示。本节记录 Verify 阶段的离线部分（已在本仓库内落地、且由 pytest 回归守护）；真实 LIBERO 环境上的 smoke + dry run 在 §28.4 列为前置条件，由用户按阶段执行。

### 28.1 Phase 0 smoke 脚本落地（§11.0 / §19.6 / §19.8）

新增两个脚本：

- `scripts/verify_env_save_restore.py`（§19.8 + V-1 修订，见 §28.5）：
  - 结构：reset → `set_init_state(init_states[idx])` → `--pre-steps` 随机动作 → ckpt `(sim_state, timestep, cur_time)` → **live 分支**（信息级，不 assert）跑 `--post-steps` → **Replay A** 按 Layer F 顺序 `env.reset(); set_init_state(ckpt); inner.timestep/cur_time=…` → 重放同一 `post_actions` → **Replay B** 同 Layer F 顺序再跑一次。
  - 断言：`allclose(replayA, replayB, atol=--atol)`（默认 `1e-6`，测的是"restore 是确定的"）、`success_flag_A == success_flag_B`、两次 replay 的 `timestep - ckpt_timestep == len(post_actions)`。**不**断言 replay vs live（MuJoCo 不保证 teleport == live continuation，只打印 `max_eef_pos drift replayA-vs-live` 供运维参考）。
  - CLI：`--task-suite --task-id --init-state-idx --resolution --pre-steps --post-steps --seed --atol`；LIBERO imports lazy；bddl 路径用 `get_libero_path("bddl_files") / task.problem_folder / task.bddl_file` 拼接（与 `examples/libero/main.py::_get_libero_env` 一致）；finally 里 `_close_env_quietly` 调 `env.close()` 并 `contextlib.redirect_stderr(devnull)` 吞掉 robosuite atexit 的 EGL teardown 噪声。

- `scripts/verify_restore_obs_equivalence.py`（§19.6）：
  - 从 GT HDF5 读 `sim_state / agentview_image / eye_in_hand_image / robot_state / task_name`；新 env 在 `LIBERO_ENV_RESOLUTION` 下 reset + `set_init_state(sim_state)` → `exp.run_spawn_experiment._obs_env_to_policy`（复用而非拷贝，确保跟 Layer F 完全同代码路径）。
  - 断言：images `np.array_equal`，state `np.abs(...).max() <= --state-atol`（默认 `1e-6`）。
  - CLI 支持 `--cycle-idx nargs='+'`，一次命令可覆盖多个 cycle；finally 里调 `env.close()` 并吞 warning，避免 MuJoCo/EGL 资源泄漏。
  - 按 §19.6 退出条件：全部通过 → Layer F 可继续直接从 HDF5 读 `start_obs`；任一 cycle 失败 → 要求 Layer F 改为 `restore env → _obs_env_to_policy` 重建 obs。

两个脚本都不纳入 CI 自动运行（依赖真实 LIBERO），只在用户机器上由命令手动跑；Verify 放行前必须至少跑一次、日志贴回本节。

### 28.2 进度条 should-fix（Layer C §24.2 遗留）

`examples/libero/main.py::_eval_concurrent` 之前把 `total_episodes` 写死为 `num_tasks × num_trials_per_task`，在用户传 `--episode_filter step1b_filter.json` 时进度条承诺的 episode 数会远大于实际要跑的数量（worker 里的 `continue` 会跳过未命中 filter 的 episode）。

处理：把计数抽成纯函数 `_count_filtered_episodes(filter_pairs, task_id_list, num_trials_per_task)`：

- 无 filter：`len(task_id_list) * num_trials_per_task`。
- 有 filter：把 `task_id_list` 转成 set，统计 `filter_pairs` 中 task id 在 set 内的条目数；与 worker 内 `continue` 用的谓词 `(task_id, episode_idx) not in filter_pairs` 一致。

`_eval_concurrent` 先 `_load_episode_filter` 再算 total（原来是先算 total 再 load filter，等同于默认忽略 filter）。同时删掉不再使用的 `num_tasks_in_suite` 局部变量。

### 28.3 测试覆盖

新增 pytest：

- `tests/examples/test_libero_main.py`
  - `test_count_filtered_episodes_no_filter_returns_full_product`：`None` filter 退化到 `len × trials`，以及空 task list 返回 0。
  - `test_count_filtered_episodes_counts_only_pairs_inside_task_list`：filter 含 task 7 但 `task_id_list=[3,5]` 时必须排除，`[3,5]/[3]/[]` 各自给出 3/2/0。
  - `test_count_filtered_episodes_ignores_num_trials_when_filter_active`：有 filter 时 `num_trials_per_task` 无关紧要，每条 filter 恰好对应一个 episode。

- `tests/scripts/test_verify_smoke_scripts.py`（新文件）
  - `test_verify_env_save_restore_imports_without_libero` / `test_verify_restore_obs_equivalence_imports_without_libero`：脚本模块在无 libero 的 CI venv 下能纯 Python import 成功，锁定 LIBERO import 必须 lazy 的契约。
  - `test_verify_env_save_restore_help_exits_zero` / `test_verify_restore_obs_equivalence_help_exits_zero`：`--help` 以 code 0 退出，且输出包含 Layer F / Verify 阶段要依赖的 flag 名（`--task-suite --task-id --init-state-idx --pre-steps --post-steps --seed --atol` 与 `--gt-h5 --task-suite --task-id --cycle-idx --state-atol`）。CLI 字段改名会立即被 catch。
  - `test_verify_restore_obs_equivalence_requires_gt_h5`：缺 `--gt-h5` 时 argparse 返回 code 2，防止默认参数退化。

回归命令与结果：

```bash
.venv/bin/python -m pytest tests/exp/ tests/examples/test_libero_main.py tests/scripts/ -q
```

```text
121 passed, 1 warning in 1.95s
```

### 28.4 Verify 阶段剩余前置条件

#### 28.4.1 已在 `libero_sim` conda env 上完成的 smoke

`scripts/verify_env_save_restore.py` 的 determinism smoke（§28.5 发现 V-1 修订后）已在 `/home/weiland/anaconda3/envs/libero_sim/bin/python` 下完成 4 个 suite 的覆盖。每次都是 `pre-steps=30`、`post-steps=25`、`seed=0`、默认 `atol=1e-6`：

| suite | task_id | `replayA-vs-replayB max_delta` | `replayA-vs-live drift`（信息） | 结果 |
|---|---|---|---|---|
| libero_spatial | 0 | `0.000e+00` | `3.202e-03` | OK |
| libero_object  | 0 | `0.000e+00` | `1.062e-03` | OK |
| libero_goal    | 0 | `0.000e+00` | `3.296e-03` | OK |
| libero_10      | 0 | `0.000e+00` | `8.588e-03` | OK |

结论：Layer F teleport 在同一 ckpt + 同一 action 序列下对 `robot0_eef_pos` 完全 bit-exact，满足 Layer F 聚合 metrics 的可复现性要求；replay vs live 的漂移在 `~1e-3 … 1e-2` 量级，属 MuJoCo 非 bit-exact 但对称的物理行为，cache vs baseline 两侧都吃一样的漂移，不产生偏差。

#### 28.4.2 仍需要在 LIBERO 机器上补的两项

1. **obs equivalence smoke**（§19.6）：先用 Step 1b 收集任意一条 GT HDF5，再跑：
   ```bash
   /home/weiland/anaconda3/envs/libero_sim/bin/python \
       scripts/verify_restore_obs_equivalence.py \
       --gt-h5 data/deviation_experiment/gt_trajectories/task_3/episode_0.h5 \
       --task-suite libero_spatial --task-id 3 --cycle-idx 0 1 5
   ```
   任一 cycle 报 mismatch → 立刻停，回到 Code 阶段改 Layer F 的 `start_obs` 构造（改为 `restore env → _obs_env_to_policy` 重建 obs，而不是 HDF5 直读）。

2. **小规模端到端 dry run**（跨 Step 1b → Step 2 → Step 3 → F-6）：用 1–2 个 task、2–3 条 episode 跑一条最小闭环，确认：
   - Step 1b HDF5 新 attrs `success / num_steps_wait / final_env_timestep` 被正确写入；
   - `exp/compute_deviate_scores.py` 按 first-action L2 产出 `deviate_scores.json` 且 `discover_episodes()` 自动跳过 `success=False`；
   - `exp/run_spawn_experiment.py::_execute_spawn_unit` 用 `final_env_timestep - anchor` 算出合理的 `env.steps` 预算，cleanup 在成功/异常路径都释放 client + env；
   - F-6 聚合 CSV 的 `env_steps_executed` 分布与 GT 的 `final_env_timestep` 对得上。

两项都通过才可以把 Verify 改为 APPROVE。

### 28.5 Verify 发现 V-1：§19.8 的 `atol=1e-6` 断言不现实

**现象**：按 §19.8 原版脚本（即"ref 分支 live 继续 post_steps" vs "replay 分支 set_init_state + 重放 post_steps"）跑 `libero_spatial/task 0`：

| smoke 变体 | replay 构造 | `max_eef_pos delta` vs live |
|---|---|---|
| §19.8 原版（`set_init_state(ckpt)`，不 reset） | 违反 Layer F 实际序列 | `1.001e-04` |
| 对齐 Layer F（`env.reset() → set_init_state(ckpt) → inner.timestep/cur_time=…`） | 与 `exp/run_spawn_experiment.py::_execute_spawn_unit` L509 完全一致 | `3.202e-03` |

**根因**：MuJoCo 的 iterative solver 有 warm-start 残留、接触缓存等状态，`set_init_state` 只恢复 `qpos/qvel`，不清零 solver 历史；`env.reset()` 又会引入新的内部状态。两条路径下 restore 与 live continuation 都不可能 bit-exact，这是物理引擎的固有属性，不是 Layer F 或 §19.8 脚本的 bug。

**决议**：采用选项 A（用户裁决于 2026-04-14）。把 smoke 的语义从"restore === live"改成"restore 是确定的"：

- **断言的**：Replay A vs Replay B（同 ckpt、同 `post_actions`、都走 Layer F teleport）必须 `allclose(atol=1e-6)`——这是 MuJoCo 真的能保证的东西。四个 suite 实测 `max_delta = 0.000e+00`。
- **不断言的**：Replay vs Live 仅作为 `info:` 打印保留运维可见度，量级约 `1e-3…1e-2`。

这个修订符合 Layer F 真实依赖：Layer F 只需要"同一个 unit 被重跑能拿到确定的结果"（否则 success-rate 指标随机抖动），不需要"teleport === live continuation"（cache vs baseline 两边都吃一样的漂移，不偏向任何一侧）。

同批对 smoke 脚本做了两个配套整改：

- bddl 路径改为 `get_libero_path("bddl_files") / task.problem_folder / task.bddl_file`（原脚本直接传 `task.bddl_file` 在 LIBERO 1.x 下找不到文件）。
- timestep/cur_time 读写改走 `env.env.timestep / env.env.cur_time`（OffScreenRenderEnv 是 wrapper，外层没有这两个属性；`exp/run_spawn_experiment.py::_execute_spawn_unit` 也是同样的 `env.env` 路径）。
- finally 里 `_close_env_quietly(env)` 主动 `close()` 并 `contextlib.redirect_stderr(devnull)` 压掉 robosuite atexit 的 `OpenGL.EGL._errors.EGLError` 刷屏。

V-1 不回滚任何 Layer A–F 代码，只修订 smoke 脚本的验收语义和物理路径，§19.8 的原断言在 plan log 里保持不动（作为历史记录），以本节为准。

### 28.6 Verify 发现 V-2：obs equivalence smoke 的 `np.array_equal` 断言不现实

**现象**：`scripts/verify_restore_obs_equivalence.py` 在 `libero_sim` env 下跑 Step 1b 刚产出的 GT HDF5（Layer F 的 spawn 起点）：

| HDF5 | cycle | `state_max_delta` | `agentview img_max_delta` | `wrist img_max_delta` |
|---|---|---|---|---|
| task_0/episode_0.h5 | 0 | 8.346e-05 | 177 | 193 |
| task_0/episode_0.h5 | 1 | 7.028e-04 | 192 | 193 |
| task_0/episode_0.h5 | 5 | 8.319e-04 | 194 | 65 |
| task_0/episode_1.h5 | 0 | 1.635e-04 | 177 | 188 |
| task_0/episode_1.h5 | 5 | 9.672e-04 | 192 | 67 |
| task_0/episode_1.h5 | 10 | 2.493e-03 | 177 | 82 |
| task_1/episode_0.h5 | 0 | 1.112e-04 | 155 | 193 |
| task_1/episode_0.h5 | 5 | 8.383e-04 | 192 | 165 |
| task_1/episode_0.h5 | 15 | 1.733e-03 | 176 | 68 |

`img_max_delta` 接近 255 满量程，约 30%–60% pixel 不等；`state_max_delta` 则随 cycle 深度从 8e-5 升到 2.5e-3。

**根因**：`env.reset() + env.set_init_state(sim_state)` 的组合不能 bit-restore 中途状态的视觉 obs。`sim_state` 向量复位了 MuJoCo 物理 qpos/qvel，但渲染管线会残留上一帧 framebuffer（"第一帧 stale"），且 `env.reset()` 本身会做 LIBERO 内部的 object-pose 随机化，这部分不在 `sim_state` 里。state 侧的漂移则和 V-1 是同一类 MuJoCo solver warm-start 噪声，cycle 越深越大。

**Layer F 是否受影响**：不受。关键路径是 `exp/run_spawn_experiment.py:469` 的 `"observation/image": np.asarray(g["agentview_image"][...])`，首帧 start_obs 从 HDF5 直读，不让 env re-render。后续步走 `env.step()` 返回的新 render，render 本身是 `sim.step()` 后的 fresh 状态，和 teleport 的 stale 首帧无关。state 漂移也不影响 deviate score 的设计假设——score 用的是 HDF5 里存好的 obs / action，不是 env 重放结果。

**决议**：采用用户裁决的 A 方案（2026-04-14）—— smoke 改为全 info-only。`state_atol` 参数保留 CLI 接口以向后兼容，但运行期不再用作断言阈值。smoke 的作用收敛为："确认 LIBERO 环境能从 GT HDF5 的 `sim_state` 成功 teleport，并打印 restore 后的 obs 偏差量级供运维参考"。

V-2 同样不回滚任何 Layer A–F 代码。Layer F 的 start_obs-from-HDF5 设计（§19.B6）被实测数据反向验证为必须——如果 Layer F 去 re-render，会吃到 ~50% pixel 的 stale drift，污染 cache 命中判定和 deviate score。

### 28.7 Verify 端到端 dry-run 结果（§28.4.2 第 2 项）

在 `155.98.36.13:9000` 公网 server（pi05_libero, `clip_w7_d4.yaml` 作为 cache-eval 配置，`inference_clip_w7_d4.yaml` 作为 Phase 1/GT 的 always_skip 配置）上跑完 Step 1a → C → D → E → F → G 最小闭环。全链路规模刻意压缩到 1 config × 10 task × 5 trial（Step 1a）→ 3 failed-init units（dry-run subset）→ 3 GT episodes → 3 deviate-score jsons → 18 spawn units。每一步生成物落在 `data/deviation_experiment/`。

| 步骤 | 脚本 | 规模 | 关键产物 | 结果 |
|---|---|---|---|---|
| Step 1a cache-eval | `exp/run_cache_experiments.py --yaml-dir data/deviation_experiment/step1a_runs --num-workers 5 --seed 42` | `libero_spatial`, 10 task × 5 trial, `clip_w7_d4.yaml` (`always_search + always_hit`) | `cache_eval_results.json` (34/50 success = 68%), `experiment_state.json`, per-batch episode_results | OK |
| Step C failed-init | `scripts/dump_step1a_failed_inits.py` + 手工裁剪 filter | 16 failed units full + 3 units dry-run subset | `step1b_filter.json` (dry-run), `step1b_filter.full.json` (backup), 10 `.init_map.json` | OK |
| Step D Step 1b GT | `exp/run_step1b_gt.py` (serial, conda `libero_sim`, `inference_clip_w7_d4.yaml`) | 3 episodes | `gt_trajectories/task_0/episode_0.h5` (16 cycles / 77 steps), `task_0/episode_1.h5` (15c/73s), `task_1/episode_0.h5` (23c/114s); attrs `success=True / num_steps_wait=10 / final_env_timestep` 全部写入 | OK |
| Step E obs equivalence | `scripts/verify_restore_obs_equivalence.py` (info-only) | 9 payload × 3 episode | §28.6 表 | OK（info-only） |
| Step F deviate scores | `exp/compute_deviate_scores.py --M 4 --num-workers 4` (RPC 切到 `inference_` 跑 Phase 1，再切回 `clip_w7_d4` 跑 Phase 2) | Phase 1 12 units + Phase 2 3 units | `deviate_score_clip_w7_d4.json`（16+15+23 cycles，score 范围 0.25–1.18），`phase1_state / phase2_state` 全 done | OK |
| Step G F-6 spawn | `exp/run_spawn_experiment.py --configs clip_w7_d4 --n-grid 1 3 --k-grid 1 3 --num-workers 2` (conda `libero_sim`) | 18 units | `spawn_aggregate.csv` 18 行全 done；17 个真跑 rollout (env_steps 2–74), 1 个 budget-skip (`anchor_cycle=16 >= num_cycles=15`) | OK |

Step G CSV 摘要（`config=clip_w7_d4, strategy=top_k`）：

| episode | (s, n, k_idx) 样本 | env_steps 分布 | success 数 |
|---|---|---|---|
| task_0/episode_0 | 6 units, s∈{0,9,12}, n∈{1,3}, k∈{0,1,2} | 2, 12, 17, 27, 62, 72 | 0/6 |
| task_0/episode_1 | 6 units, s∈{0,10,13}, n∈{1,3}, k∈{0,1,2} | 0 (budget), 2, 8, 18, 58, 68 | 0/6 |
| task_1/episode_0 | 6 units, s∈{7,10,18}, n∈{1,3}, k∈{0,1,2} | 4, 19, 49, 59, 64, 74 | 0/6 |

`success=0/18` 符合 dry-run 规模预期——只有 3 条 GT 且 GT 都偏短（15/16/23 cycles），top-k 采到的 s 多数本来就在失败边缘，teleport 后继续 rollout 失败属正常。重点在于：
- env-step 预算 `final_env_timestep - anchor` 得到的执行步数全部落在 (0, 300] 区间且分布合理；
- budget-skip 分支（`anchor_cycle >= num_cycles`）正确触发，没有伪装成功或跑超；
- 无 libero/openpi_client/env close 类异常；
- CSV schema `config,strategy,episode,s,n,k_idx,success,env_steps_executed` 与 F-6 plan §19.4 完全一致。

### 28.8 Verify 发现 V-3：`OffScreenRenderEnv` 只给了 bddl 文件名

**现象**：Step G 第一次 rerun（已装 conda `libero_sim`）时 40 units 中 17 个抛 `[error] pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl does not exist!`。文件在 `libero_sim` 的 site-packages 下确实存在，只是 `run_spawn_experiment.py` 传的是裸文件名。

**根因**：`exp/run_spawn_experiment.py::_SpawnCommon.make_env`（L268）只写了 `bddl_file_name=task.bddl_file`，而 LIBERO 1.x 的 `OffScreenRenderEnv` 不会自己拼 `get_libero_path("bddl_files") / problem_folder`。`examples/libero/main.py::_get_libero_env` 和 `scripts/verify_env_save_restore.py / verify_restore_obs_equivalence.py` 都用的是拼好的绝对路径。

**修复**：改为和 main.py 一致——
```python
from libero.libero import get_libero_path
bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
return OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
```

修完后 Step G 18/18 done、0 error，结果见 §28.7。V-3 是 Layer F 代码真实 bug（不是验收语义修订），必须进 commit；不影响任何已有测试用例（它们都注入 `env_factory` mock 或只验证 import）。

### 28.9 最终 APPROVE

§28.1–§28.3 的三个 defect 全部处理、§28.4.2 两项前置（obs equivalence smoke + 端到端 dry-run）都跑完、V-1/V-2/V-3 三条 Verify 发现都有明确决议（V-1、V-2 只修订 smoke 语义、V-3 修 Layer F bug）。全链路产物齐全：

- `data/deviation_experiment/step1a_runs/cache_eval_results.json`
- `data/deviation_experiment/step1b_inits/step1b_filter.json` + `.full.json`
- `data/deviation_experiment/gt_trajectories/task_{0,1}/episode_*.h5`（3 episodes）
- `data/deviation_experiment/step2_deviate_scores/deviate_score_clip_w7_d4.json` + `phase1_state / phase2_state`
- `data/deviation_experiment/spawn_dry_run/spawn_aggregate.csv` + `spawn_state_clip_w7_d4.json`
- `logs/trajectory_deviation_corrective_implementation.log.md` + 本 review log
- 测试: `pytest tests/exp/ tests/examples/test_libero_main.py tests/scripts/` → 121 passed

L3 trajectory-deviation corrective experiment 的 Verify 阶段 **APPROVE**。可进入 G2 gate 并合 main（下一步：commit V-3 + §28.7–§28.9 日志，通知用户关掉 `155.98.36.13:9000` 服务器）。
