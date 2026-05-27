# 全库代码审计 2026-05-26（51e364b → HEAD）

> Status: Validated（审计 + 修复完成并通过 §6 Verify；含 3 项 PO 决定"仅记录"的接受风险，未归档以保持可见）
> Date: 2026-05-26
> Level: L2/L3 审计 → 修复
> Authority: Execution（PO 行使独裁官权，WA §7 显式豁免本会话流程分离；审计 agent 全程只读取证，修复由 executor 逐条核对后亲自落地）

## 1. 范围与方法

- 基线 `51e364b`（2026-04-01）对比 HEAD（branch `Ziyang`），过滤 yaml/data/figure 等非代码文件，聚焦 `src/**/*.py`（~1.9 万行新增）+ `examples/libero/**` + `scripts/` + `exp/**/*.py` + `docs/` + `logs/`。
- 三轮共 16 路 agent 并行审计：Round 1（7 路，按子系统切分）、Round 2（6 路，独立复核 R1 + exp/ 全量 + config/interceptor 逐行 + 测试套件实跑 + 跨模块接缝 + 安全健壮性）、Round 3（3 路，深读最大三个文件 orchestrator.py / websocket_policy_server.py / main.py + 复核 CRITICAL RCE）。
- 所有 CRITICAL/MAJOR 经独立 agent + executor 二次核对，剔除被驳回项（如 search_strategy E402 非 pre-commit gate；M8 状态 desync 在当前部署下因 server 异常即拆连接而不成立）。

## 2. 已修复（本次 commit）

### 2.1 测试红线（WA §6/§2.7）
- `tests/conductor/test_journal.py`：`test_failed_status_removes_from_done` 在 HEAD 实测失败——断言旧 last-writer-wins，但 `journal.replay_done_uids` 已改 done/failed 双终态（commit f171394 的 anti-livelock 修复未同步测试）。重写为 `test_failed_status_is_terminal` 断言新语义；并修 `test_record_and_replay` 中 `u2` 的 `done+success=False` 矛盾数据为 `failed`。

### 2.2 逻辑/并发 bug（MAJOR，全部独立复核 CONFIRMED）
- **M1** `cache/components/llm_layer_key_builder.py` `attach_model`：删除对**共享模型** `language_model.config._attn_implementation = "eager"` 的写入——该写入会与并发 Stage 2 forward 竞态、并永久把 Stage 2 从 sdpa 偷偷退回 eager（与 PO"保留 sdpa"决定冲突）。改为 layer-N replay 跑在模型既有 backend（sdpa）上，parity 反而更一致；docstring 同步更正；测试改为断言"不改共享 config"。
- **M2** `cache/components/factors/composers/__init__.py` `WeightedSumZeroNanComposer._score_only`：权重和为 0（如 +w/−w 抵消）时 `ZeroDivisionError`。加 `weight_total == 0.0 → NaN` 守卫（路由 compose() 到 MISS），与兄弟类 `WeightedSumComposer` 一致。
- **M3** `serving/websocket_policy_server.py`：连接计数器/单连接标志在 pre-loop（`on_task_begin` + metadata send）失败时泄漏——单连接模式永久锁死、并发模式计数虚高。用 try/except 包裹 pre-loop 段，失败时 `decrement_connection()` + 复位标志后再抛。
- **M4** `serving/batching_coordinator.py`：批后 metrics 块在 try/except 之外，插桩异常会静默杀死 stage worker 线程→后续 `submit_to_stage(timeout=None)` 永久阻塞。metrics 块包 try/except（仅 log，不杀线程）。
- **M5** `serving/websocket_policy_server.py` `_bind_bundle`：bundle 切换时先建新栈、再 `on_task_end` 旧栈、**提交 conn_policy 后**再 `on_task_begin`——避免新栈 `on_task_begin` 异常时 `conn_policy` 残留指向已结束的旧栈。
- **M6** `serving/websocket_policy_server.py`：`load_cache_config` 的 `build_shared_storage`（artifact pickle.load，数十 MB）改 `await asyncio.to_thread(...)`，不再阻塞事件循环（含 /healthz 与所有其他连接）。
- **M7** `conductor/driver.py` `_handle_conn`：断连 requeue 原用 `attempt=None` 绕过 scheduler 代际栅栏，迟到断连可作废已重派给别 worker 的当前调度。改为 `inflight` 记录每 uid 的 dispatch generation，requeue 透传该 `attempt` 受栅栏保护；scheduler docstring 同步。
- **M9** `examples/libero/main.py`：并发模式 Ctrl+C 走 `os._exit(0)` 绕过 `eval_libero` 的 finally→per-step 日志丢失。在 `os._exit` 前显式 `per_step_pool.finalize()`；并修正误导注释（实为 per-episode `flush_episode`，非 `buffering=1`）。
- **M13** `conductor/driver.py` + `scheduler`：driver 从不发 `MSG_SHUTDOWN`，worker 在 driver 退出后孤儿无限重试。`handle_pull` 在 `all_done()` 时返回 `MSG_SHUTDOWN`，`_handle_conn` 改为尊重返回的消息类型，worker 干净退出（主部署本由 `WorkerAgent.stop()` SIGTERM 缓解；本改动闭合常态窗口）。
- **M10** `exp/weighted_sum/refine_round.py` `_cid`：权重→cid 百分比用 `int(round(x*100))`，与 `emit_yamls.py` 的截断 `int(x*100)` 不一致（0.375→"38" vs "37"），破坏 refine 对 baseline 最优点的 join/复算。改为与 emit 一致的截断。

### 2.3 死代码 / 注释漂移 / 合规（WA §3.1 / §3.2）
- 删死代码：`orchestrator.build_keys`（无任何调用方，已二次 grep 确认）、`orchestrator.py` 未用 `dataclasses.field` import、`dumping_judge.py` 未用 `HitType` import、`test_libero_main.py` 残留死变量 `full`/`subset`/`is_dummy`。
- 中文注释→英文（§3.2 硬违规）：`config.py:2263`、`dumping_judge.py`（4 处）、`exp/weighted_sum/{build_all_results,plot_results}.py`、`exp/.../v2_spec.py`、`tests/examples/test_libero_main.py`、`tests/exp/test_run_cache_experiments.py`。src/ 与 scripts/ 中文残留实测归零。
- 注释/docstring 准确性：`key_builder.py` 196/588→256/768 token 注释；`composers` `WeightedSumComposer` "B0 stub" docstring 改为描述已落地行为；`_descriptor_kernel.py` + `composers/__init__.py` 指向不存在模块（`source_window.py`/`runtime_continuity.py`）的 coupling map 改为真实 consumer（`online.py`/`offline.py`/`_descriptor_kernel.py`）；`timing.py` 多处 "Step 2 stub / 未来实现 / Do not read" 过期 docstring 更正为已实现的 `CpuMonitor`/`GpuMonitor` 现状；`search_strategy.py` logger/import 顺序（E402）。
- 文档索引同步（WA §4 宪法红线）：`docs/README.md` 死链（指向已 archive 的 `verdict_factor_judge_refactor.log.md`）；`logs/README.md` 补 4 个漏索引的活跃 log；`logs/concurrent_serving_optimization_plan.log.md` front-matter 指向已删 `concurrent_serving.md` → `conductor_tutorial.md`；`cache_system.md:607` 重复标题。
- 代码异味（审计窗口内）：`plot_results.py` E702 分号拆行。

## 3. PO 决定"仅记录、不改码"的接受风险

> 这三项经 AskUserQuestion 由 PO 明确决定。代码保持现状，在此登记追踪。

### 3.1 C1 — `load_cache_config` 远程未认证 RCE（CRITICAL，独立复核 CONFIRMED）
- **链路**：`concurrent` 默认开 → WebSocket `_handler` 无任何鉴权 → 客户端 YAML 的 `preload_path`（及 `calibration.samples_source.offline.path`）无根限制 → `InMemoryBackend.load_artifact` → `pickle.load(open(path))`，构造恶意 pkl 的 `__reduce__` 即以 server uid 任意执行。`_enforce_runtime_write_policy` 仅查 `write_policy.type`，不碰 `preload_path`；dump-root allowlist 仅约束 dump 路径。位于 conductor/sweep 正常 serve 路径，公网 frp 入口（155.98.36.13:9000）。
- **PO 决定**：暂只记录不改码（视为可信内网/临时部署）。
- **建议修复（未实施）**：`preload_path` / offline calibration path 加 artifact-root allowlist（默认 `exp/common/data/cache_artifacts`，复用 `_safe_resolve_under_root`），并把 `pickle.load` 换成只允许已知类的受限 unpickler；复核确认不破坏 conductor（其路径均在该 root 下）。已在 `docs/deployment/libero.md` 公网端口段补一句操作者警示。

### 3.2 M11 — eager→sdpa 改变非缓存 baseline 数值
- `pi0_pytorch.py` 把 Stage 2 attention backend 从 eager 改为 sdpa（实测非 bit-identical，bf16 ~2e-3），与 4 月前 eager 下测得的 baseline 不可直接比较；周围 docstring 仍称 "bit-identical"。
- **PO 决定**：保留 sdpa（吞吐 + 消除 config 竞态），修文档 + baseline 重测。
- **本次已做**：M1 已使 LLM-layer-extract keybuilder 不再把 Stage 2 偷偷退回 eager（现真正跑 sdpa）。
- **待 PO/研究侧**：在 sdpa 下重测所有 LIBERO reference baseline 后，方可与 HEAD 成功率对比；任何用 eager 下旧 pkl artifact 的 cache key 与在线 sdpa key 存在 ~1.6e-2 偏差（LLM-layer-extract keybuilder 已 shelved，影响面小）。

### 3.3 M12 — `stage_device_placement.py` CI 零覆盖
- `tests/models_pytorch/test_stage_device_placement.py` 整模块 `pytest.skip`（纯性能原因，~5min），297 行新模块在 CI 无任何执行覆盖（WA §6）。
- **PO 决定**：维持 skip 现状，性能优先。

## 4. 已识别但本次未改的 MINOR 建议项

> 多为增量型/部署相关，回归风险高于价值，留待单独评估，避免破坏现有 yaml/测试。

- `config.py` validator 缺口：`field_similarity.type` 未校验 `{cosine,l2}`；`top_k/rrf_k/candidate_multiplier/step_window/buffer_size` 无范围校验；未知 yaml key 仅 WARNING 静默丢弃（typo 致 cache 静默关闭）。建议加 fail-closed 校验（需配套测试，谨防误拒现有配置）。
- `interceptor.py` timer 在 `on_episode_end` + `on_task_end` 双重 finalize，污染 per-task 计时；`exp` phase4/phase5 runner `_extract_finite_factor_raw` 对 `None` 调 `float()` 会崩（fallback 路径）；`exp` 多 server 共享 summary 文件并发追加冲突 / `serving_benchmark/collect.py` fd 泄漏；qdrant backend 丢 `CachePayload.factors`；RRF vs cosine 对零相似度候选处理不一致。
- 部分活跃 log 的状态/G2 标注与 git 现实存疑（如 `backend_c2_autoguard_decouple` 称"待 commit"但已 c12a1e0 落地）——**归档/状态变更需 PO 确认（WA §5），未擅改**。

## 5. 验证（§6 Verify）

- `uv run pytest -m "not manual" --ignore=tests/review_tests`：**1717 passed / 1 skipped**（journal 测试已转绿，较审计前 +1），**0 回归**。
- 余 10 failed 全为 `@pytest.mark.env_dependent` 上游 JAX/GCS/网络测试（model_test ×4 / download_test ×2 / transforms_test ×2 / data_loader_test ×1 / train_test ×1），与本次改动无关、不触碰任何被改文件。
- `uv run ruff check <全部 27 个改动文件>`：All checks passed。
- src/ + scripts/ 中文注释残留：0。

## 6. 改动文件（27 个）

src: `cache/{config,orchestrator,timing}.py`、`cache/components/{dumping_judge,key_builder,llm_layer_key_builder,search_strategy}.py`、`cache/components/factors/{_descriptor_kernel.py,composers/__init__.py}`、`serving/{batching_coordinator,websocket_policy_server}.py`、`conductor/{driver,scheduler}.py`。
examples: `libero/main.py`。
exp: `weighted_sum/{build_all_results,plot_results,refine_round}.py`、`verdict_factor_judge/common/v2_spec.py`。
docs: `README.md`、`architecture/cache_system.md`、`deployment/libero.md`。
logs: `README.md`、`concurrent_serving_optimization_plan.log.md`、本文件。
tests: `cache/components/test_llm_layer_key_builder.py`、`conductor/{test_driver,test_journal}.py`、`examples/test_libero_main.py`、`exp/test_run_cache_experiments.py`。

## 7. 修后 /code-review 追加修复（9 角度 finder + 复核）

对本次改动跑了一遍 `/code-review`（recall 模式）。两个 CONFIRMED：

- **R1**（in-scope，本次引入）：`websocket_policy_server.py` 新增的 pre-loop `try/except`（M3）只 decrement 计数 + re-raise，未像两个 in-loop handler 那样调 `conn_policy.on_task_end()`——non-concurrent 模式下 `on_task_begin` 已跑、若 metadata send 失败则 policy 半启动（timing CSV 未 flush）。已补 guarded `on_task_end`。
- **R2**（M1 一致性后果）：`exp/common/build_llm_layer_matrix.py:432` 独立 force `eager`，而 M1 已让在线 keybuilder 走 sdpa（`build_in_memory_cache_artifact.py` 经 `attach_model` 已随之走 sdpa）。改为 `sdpa` 保离线/在线 keyspace 一致（cp1_llm_layer_extract keybuilder 已 shelved，影响面小）。+1 文件 → 共 28 文件。

清理类 finding（未强制重构，留作建议）：websocket 连接 teardown 现有 3-4 处重复（理想是单一 try/finally）；`refine_round._cid`/`emit_yamls` 的 `int(x*100)` 应抽共享 helper（当前靠注释约束一致）。`_score_only` 零权重旧码实际抛 ZeroDivisionError（solve_recipe 不接 → 未捕获崩溃），新返回 NaN 反而更稳——确认为正向修复。

## 8. 三轮全仓库复审（2026-05-27，11 路 agent，每轮即修即暂存）

PO 指令：分 3 轮复审全仓库 + 当前修改，多方向 agent，每轮立即修复并加入暂存区。

### Round 1（改动复核 / 实验完整性 / cache 核心 / 并发）→ 已修+暂存
- 改动复核：28 文件 change-set 独立判定**安全可提交**，所有 R1/R2/M1–M13 修复重验正确，无新 bug/契约破坏。
- **proxy 扇出无超时**（replica_proxy `_broadcast`/`_aggregate` 的 backend `recv()`）→ 包 `asyncio.wait_for(_FANOUT_TIMEOUT_S=30)`，一个卡死 replica 不再拖死所有广播。
- **batching 死线程→永久挂起**：补 `submit_to_stage` None→`_DEFAULT_SUBMIT_TIMEOUT_S=300` 有限兜底（M4 metrics-guard 之外的兜底）。
- **agent 僵尸/孤儿**：`supervise_once` reap 退出 Popen + `_default_spawn(start_new_session=True)` + `stop()` 用 `os.killpg` 杀进程组（穿透 conda-run wrapper）。
- stale docstring（test_llm "eager force"）。

### Round 2（安全 / 测试有效性 / 类型契约 / 依赖+上游+死代码）→ 已修+暂存
- **死代码删除**：`src/openpi/models/vit.py`（零导入方且自身 import 不存在的 `models/resnet`，必崩）。
- **上游 live-path 修复**：`gemma_pytorch.py` 删运行期 `import pytest` + `pytest.Cache` 注解（上游 e4429ad 已删；把 pytest 拖进核心模型运行期 import 链）。
- **DoS 加固**：WS server `max_size=None`→256MB 上限（未认证客户端不能用多 GB 帧 OOM）。
- 类型注解：`StageRequest.reply_slot: list|None`。
- **补测试**（R2-F 两个 HIGH 缺口）：batching `submit_to_stage` 超时兜底（None→有限）+ 坏 stage3 payload→TypeError。
- 📝 仅记录：**第 2 个 pickle RCE**（`calibration.samples_source.offline.path`，与 C1 同类，PO 已决记录）；buffer/batch-param 无界 DoS（可信内网）；依赖 CVE（torch 2.7.1 / transformers 4.53.2 硬钉 + transformers_replace 绑定，**pillow 12.1.1→12.2.0 图像解析 CVE 属你的环境升级决定**）；`requires-python <3.12` 与实际 3.12.7 不符；test 套件覆盖：batching 49% / stage_device 0%（模块级 skip，PO 决定保留）。

### Round 3（架构解耦 / 文档全量 / 统计+配置面）→ 已修+暂存
- 架构不变式 §2.5/C1/C2/机制-策略分离**全部 INTACT**（cache/serving/conductor 均在 staged API + wrapper 之后，C2 `__init_subclass__` 守卫覆盖全 backend，conductor 无实验语义）。
- **18 处 doc 正文死链**（指向 archive/ 内 log 缺前缀）→ 脚本安全补 `archive/`（仅对 archive 内 basename）。
- **config validator 安全补强**：搜索数值范围（top_k/rrf_k/candidate_multiplier/step_window）、`field_similarity.type∈{cosine,l2}`、percentile 缺 p5/p95——把运行期崩溃前移为启动期清晰报错（不误拒合法配置，101+2 测试 pass），+2 validator 测试。
- serve_policy "bit-identical" docstring → 改为 C1=结构=raw Policy、数值=当前 sdpa 模型（非 4 月前 eager baseline）。
- 📝 仅记录（属研究设计/有风险，不擅改）：**统计方法**——phase4/5 的 `_winner_rule_5pp` 在 n=100（SE≈3-4pp）下 5pp≈1σ 且无多重比较校正，**per-bucket "winner" 结论统计上不可靠**；episodes 实为配对（应用 McNemar）；大 15-35pp Pareto/baseline 结论稳健；500ep 的 random_periodic/warm_start/trajectory 扫描可信。**unknown-key 静默丢弃**（typo 致 cache 静默 OFF）建议改 fail-closed，但有误拒含额外键合法 YAML 风险，未改。worker_module LIBERO 默认（可配置）；empty_cache 基线 monkeypatch（数值中性，opt-out）。

### §6 Verify（终验）
- `uv run pytest -m "not manual" --ignore=tests/review_tests`：**1721 passed / 1 skipped / 0 新增失败**（较复审前 1717 多 4 个新测试）；余 10 failed 仍为 `@env_dependent` 上游 JAX/GCS。
- ruff（全改动文件）：剩 6 个**既有**问题（build_llm_layer_matrix 5×E402 有意 lazy-import；test_serving_optimization `coordinator` 占位 F841）——非本次引入、ruff 非 pre-commit 门禁、属 drive-by，保留记录。
- 每轮修复均已 `git add` 暂存；**未 commit**（待 PO 指示）。
