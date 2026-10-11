# R17 integration review A — P1 shared framework / P2 MetaWorld + conductor + π0.5-MW / P5 GR00T-MW / P6 tools

2026-10-09，checkout `/home/weiland/projects/openpi`（branch `Ziyang`，工作区未提交）。只读审查：未改代码、未用 GPU、未起服务、未碰运行中进程。复现脚本在 `review/repro_A/`，运行前缀统一为：

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python <script>
```

本次实跑的 CPU 测试（全部通过，说明现有测试**覆盖不到**下表问题）：
- `exp/r17_integration/tests/{common,benchmarks,conductor,models/pi05,debug,search,ops}`：187 passed
- `exp/r17_integration/tests/models/groot`：43 passed
- `exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/serving/test_router_bundle_rebind.py tests/cache/test_cache_storage.py`：213 passed

## 发现

| # | 严重度 | 位置 | 问题 | 证据 | 修复 |
|---|---|---|---|---|---|
| 1 | **blocker** | `models/groot/provider.py:68,72,157`；`exp/mw_finetune/groot/conventions.py:23-25` | GR00T-MW adapter 的 `prepare`/`extract_state_only` 只认规范键 `image`/`state`；P2 worker 用的原 kernel `exp/metaworld/env.py:87-93` 只发 `observation/image`/`observation/state`/`prompt`（`benchmarks/metaworld.py:169-170` 也按这组键校验）。A 的旧服务器由 `adapters.canonical_observation` 翻译，R17 通用 `ConnectionPolicy` 没有翻译层。结果：GR00T-MW 经 R17 conductor/worker 的**每一局第一次 infer 就 KeyError**，整条 MW 闭环跑不起来。P5 测试全部直接喂 `image/state`，所以 43 项通过。 | `repro_A/repro_groot_wire.py` 输出：`P2 kernel wire keys: ['observation/image', 'observation/state', 'prompt']` → `P2-runner obs -> KeyError 'state'`；改用 `image/state` 时 `actions (5, 4)`。 | 在 `GrootMetaWorldAdapter.prepare/extract_state_only` 里像 Fast-WAM `wire_policy._field(obs,"state","observation/state")`、DP `staged.py:102-119` 那样两套键都接受（或先调 `exp.mw_finetune.adapters.canonical_observation`）；加一个用 `exp.metaworld.env.observation()` 输出驱动 GR00T `ConnectionPolicy` 的测试。 |
| 2 | **major** | `src/openpi/cache/{config,types,storage_types}.py`、`src/openpi/serving/websocket_policy_server.py`（P1 共享改动） | 运行时默认关闭没问题，但这 4 个文件的哈希被 R16 `DESIGN_LOCK.json` 封存，且不在 `CONDUCTOR_LOCK.json` 的 amendment 里。R16 的封存校验（`analyze.verify`，dev/test/freeze/retest/retest_combined/retest_superseding 全部先调它）现在直接拒绝，R16 结果在本 checkout 上无法复核或重跑分析。 | `python -m exp.offline_search.rounds.r16.analyze verify` → `ValueError: Unregistered conductor source change`。逐文件比对：DESIGN_LOCK 的 3301 个封存文件里，未登记的漂移**只有**这 4 个 R17 改动文件（另 2 个 R16 文件已登记）。 | R17 不要改共享 checkout 里被封存的文件：把 hook 移到 `exp/r17_integration`（R17 自己的 loader/storage builder 加 WS 的 `bundle_validator` 注入、子类化），或者 R17 用独立 worktree/branch；若坚持改共享文件，需要 coordinator 显式给 R16 登记 amendment，不能静默让冻结校验失败。 |
| 3 | **major** | `models/pi05/serve.py:21-22,53`；`models/pi05/provider.py:20-37`；`common/provider.py:89`；`src/openpi/cache/config.py:1177,1190`；`conductor/manifest.py:24-27` | π-MW 五层是**进程级**配置，并且绑死在 K 上：`recipe_factory = recipe.fork() if num_steps != 10 else None`，库目录、guard/corrector、periodic gap 都只能通过 CLI（`--recipe-store/--periodic-gap`，guard/corrector 恒开）设置。YAML 没法携带这些设置：binding 键集合必须完全一致，严格解析会拒绝未知字段。后果：(a) 五层 + 满步 K10 MISS（R14/R15 满步逐任务下降的参照）**无法表达**；(b) 在带 `--recipe-store` 的 endpoint 上，plain K2/K1 bundle 会**静默变成检索 arm**；(c) `select_bundle` 换不了库大小、IR 档位或 guard/corrector，所以 R16 那种 ir45→ir40→ir32→ir25→off 的逐任务下降没法靠 bundle 分档（也没有 target_IR→gap 的换算）；(d) `EndpointSpec.accepts` 只比较 model/norm/layout/protocol/K，两个 recipe 或 gap 不同的 π endpoint 在队列看来完全一样，会把一个 arm 的局派到另一种配置上，结果混在一起。 | 代码路径：`serve.py:53` lambda 只看 `adapter_binding["num_steps"]`，忽略 bundle 内容；`validate_binding` 要求 `set(binding)==required`；`_dict_to_dataclass` 对带 binding 的 YAML 遇到未知键会抛错。P2 的 descent 测试（`tests/conductor/test_dispatch.py:314-354`）用假结果驱动，各档并没有真正对应不同的服务端配置。 | 给 bundle 加一个版本化、要校验的 `recipe` 段（store digest / fit receipt / corrector、guard 开关 / gap 或 target_IR / commit），由 `recipe_factory(config, storage)` 按这段构建，同一 store digest 的 fit 结果可以缓存；把 recipe/fit identity 写进 episode `extra` 和 `EndpointSpec`（或者 bundle identity），让兼容性判断能区分；K10 + recipe 要可用。 |
| 4 | **major** | `models/groot/serve.py:74-124`；`models/groot/provider.py:231-252`；`models/groot/manifest.py:166-180`；`common/provider.py:151-166` | GR00T-MW 的服务入口**完全没接五层**（`serve()`/`main()` 都不传 `recipe_factory`，只有测试里注入）。库的 binding 又和 endpoint 的 MISS K 绑在一起：ModelSpec identity 包含 `default_sampler`，`validate_library` 要求 `sampler_identity`/`denoise_schedule` 相等，通用 `validate_artifact` 也比较 `sampler_identity`。所以 K4 teacher 库在 K2/K1 endpoint 上会被拒，R16 的主配置（库 + guard/corrector/knob + MISS k1/2）在 GR00T-MW 上无法服务，只剩同 K 的 stock FULL_HIT。P5 handback 写的 “Hot YAML may change commit/recipe/library choices” 与代码不符。 | `repro_A/repro_groot_meta_k.py`：`K4 vs K2 ModelSpec identity equal: False`；`K4 teacher library on K2 endpoint -> library sampler_identity differs from served GR00T-MW endpoint`。 | 按 #3 的同一方案给 GR00T 接 recipe；binding 里把 `library_sampler_identity`（teacher）和 `miss_sampler_identity`（endpoint）分开；GR 的 artifact validator 比较 model spec 时不含 `default_sampler`。 |
| 5 | **major** | `models/groot/serve.py:79-81,103-124`；`conductor/worker_entry.py:83` | GR00T-MW 的 debug 采集部署不了。服务端：`main()` 没有 `--debug-directory`，`capture_sink` 恒为 None。worker 端：`--debug-directory` 从 server metadata 读 `["r17_model_spec"]`，但 GR00T 发出的 metadata 是 `manifest.to_dict()`，没有这个键，worker 启动时就 KeyError；而且 to_dict 里的 layout/feature_spec 结构和 ModelSpec 不同，就算拿来用，binding 也对不上。 | `repro_A/repro_groot_meta_k.py`：`served metadata has r17_model_spec: False`。 | GR00T metadata 加 `r17_model_spec: asdict(provider.model_spec)`；把 π 的 `make_capture_sink_factory`（core-aware）提到 common，GR00T `main()` 加 `--debug-directory`。 |
| 6 | **major** | `search/analyze.py:35-44`；`search/audit.py:34`；`benchmarks/metaworld.py:181-187`；`common/serving.py:300-306` | 真实运行时成本台账永远凑不齐：(a) 没有任何代码产出逐决策的 `cost_ms`（grep 只在测试 fixture 里出现），`server_timing.infer_ms` 和阶段计时都没写进 `__hit_meta__` 或 `cost_rows`；(b) 每局最后一次决策没有服务端执行 ack，因为 ack 是跟在下一次 infer 后面发的，`episode_end` 不带 ack，所以 `executed_count=None`。`summarize`/`audit_campaign` 的 cost status 对任何真实 run 都会是 `unsupported`，INVENTORY A01（配对成本）在 MW 上做不出来。另外 π 的 `extract_state_only`（`models/pi05/adapter.py:193-200`）在每次 tail 上都跑一遍完整输入变换（resize + tokenize），但 tail 记账是 `(0,0,0)`。 | 代码路径如左；P2 handback “Final control acknowledgment at episode termination” 一行也承认了 (b)。 | `ConnectionPolicy` 在 meta 里写入实测的阶段/总 ms，runner 把它和 `server_timing` 一起放进 `cost_rows`；`episode_end` 控制消息带上 `__execution__` 终局 ack（或者 reader 合并客户端的 `execution_ack` 行）；tail 只做 state normalization。 |
| 7 | minor | `common/serving.py:133-145,161-168` | 客户端没发 R17 envelope 时（`MetaWorldBenchmarkAdapter.make_episode_runner` 默认 `framework_context=False`，或者旧的 `MetaworldEpisodeRunner`），`ConnectionPolicy` 会静默补默认值：stock runner 发的是 `orig_init_state_idx`，代码读的是 `original_init_idx`/`orig_idx`，所以落到 0；`task_name` 变成 `str(task_id)`，campaign 变成 `"r17"`，也没有 explicit_seed。结果同一任务所有起点共用同一套噪声和 periodic coin，违反 A 的正式 RNG 规则，而且不报错。 | `repro_A/repro_identity_default.py`：idx 3 和 7 都得到 `original_init_idx=0 task_name='41' explicit_seed=None`，`distinct inits share noise: True`。 | 同时读 `orig_init_state_idx`；对绑定 MT50 protocol 的 bundle，缺 task_name、原始 idx 或 explicit_seed 时直接 fail closed；`framework_context` 的默认值改为 True。 |
| 8 | minor | `conductor/strategy.py:65-70,172-183` | 逐任务下降中，某个 (task, level) cohort 只要有一个终局 error，controller 就停在未决状态，但下一档的 stage 照常激活。`on_dispatch` 用的是 **controller 当前档**（旧档）的 key，新档第一次派发的 attempt=1 ≤ 旧 attempt，于是抛 “attempt must increase”。这些局要卡到 1800 s 超时重派之后才会运行，而 `accept` 会因档位不匹配丢弃它们的结果；freeze 永远做不了。 | 代码路径：`expect` 的 key 是 `(task, levels[current], idx)`；`on_stage_begin` 只在 `stopped` 时清空，不检查上一档是否已判定。 | `on_stage_begin` 里如果上一档 cohort 未判定，就让 stage 以带 `FatalEpisodeError` 标记的错误失败（级联），或者把该任务标成 error-stopped；`expect` 改用派发任务自己的 `extra["descent_level"]`。 |
| 9 | minor | `common/contracts.py:73-75`；`common/serving.py:264-267` | `validate_commit` 允许 commit 不是 replan 的整数倍（例如 GR H16 时 commit=7/12）。这样 tail 只返回 `commit-cursor<5` 行动作，kernel 报 “need (>=5, 4)”，整局记为 error。 | `repro_A/repro_groot_meta_k.py`：`commit=7 accepted by layout`；kernel 的检查在 `exp/metaworld/env.py:136-142`。 | 要求 `commit % replan_steps == 0`（或者 runner 允许最后一段 plan 不足 replan）。 |
| 10 | minor | `benchmarks/metaworld.py:142-144,180`；`conductor/worker_entry.py:83` | 物理采集和模型采集的身份可能对不上：`__request__` 的 `bundle_generation` 取 `t.bundle_id`，而物理侧 `_identity()` 取 `extra["bundle_generation"]`，`episode_tasks` 允许两者不同，一旦不同，`episode_identity` 就 join 不上。worker 的物理 binding 一律用 metadata 里的默认 sampler（π 是 K10），K2/K1 arm 的物理采集被归到 K10 binding 下。 | 代码路径如左（driver smoke 恰好两者相同，所以测试没暴露）。 | 两侧用同一字段；物理 binding 按 task 的 `num_steps`/sampler identity 逐任务选。 |
| 11 | minor | `search/audit.py:26` | tail 零调用审计只匹配 `"tail"`/`"policy_tail"`，而 `ConnectionPolicy` 还会产出 `library_tail`/`cache_tail`（`common/serving.py:223`），这两类行漏审。 | 代码路径。 | 改用 `source.endswith("_tail")`。 |
| 12 | minor | `models/groot/manifest.py:145-150` | GR00T ModelSpec 把 `trace`/`batch_stage3`/`native_resume`/`library`/`warm_reset_hook` 标为 `supported`，但通用执行器对 trace/warm-reset 直接拒绝（`common/provider.py` `validate_config`），也从不挂 batcher。capability 元数据高报，会误导 G3 审计。 | 代码路径。 | 服务路径接通之前，这些项标成 `implemented_unverified`/`unsupported`。 |
| 13 | minor | `conductor/strategy.py:26-39` | bundle 只在 driver 初始化时 preload 一次，`on_stage_begin`/`on_resume` 什么都不做；推理服务一重启，bundle 就丢了，后续局全部报 unknown bundle 并反复重试。R16 strategy 会在 stage 开始时重新加载。 | 代码路径。 | `on_stage_begin` 时对兼容 endpoint 幂等地重新 `load_cache_config`。 |

已核对、**没有问题**的部分：A 的噪声规则（`decision_seed` + PCG64 float32，与 `exp/mw_finetune/pi05.py` 一致，不含 arm/attempt/campaign）；MT1(task, seed=7)、idx 0..9 / 10–29 / 30–49 分区、settle 15、上限 160、replan 5、corner2 旋转 180°（与数据集一致，见 A camera parity）；π 的 key 取 prefix 前 256 个真实 base token，pool16 和 legacy `_spatial_pool_tokens` 先池化再转 float32 的顺序一致；建库用的 key 与在线服务用的 key 是同一个 `retrieval_fields`（DecisionSink rawkeys），归一化 state 两侧一致；π 和 GR 的动作逆变换各只做一次，夹爪不翻转；4D 动作；debug 开关只换成数值等价的 `run_stage2_capture`/`_stage3_with_intermediates`；连接/bundle/episode 之间状态不泄露（每次绑定新建 `ConnectionPolicy`，按 episode 重置 recipe/guard/key_builder）；dispatch 先 fsync 再派发、attempt fence、重试耗尽后转终局；共享改动在 hook 关闭时运行时字节等价（上面 213 项测试）。

## 分模型结论

- **π0.5-MW**：模型 hook 基本正确：原 transforms、原 staged stages、K10/2/1、自身视觉 key、逆变换一次、H5 下 commit=5 的边界。它能跑 G1 机械 smoke（full K10 vs retrieval K2），但还算不上“支持 π0.5-LIBERO 的全部功能”：五层是进程级配置且与 K 耦合（#3）、成本台账不全（#6）、没有阶段 batching/trace/warm/CP2，R16 式逐任务下降也没法正确分发。
- **GR00T-MW**：staged S1/S2/S3、one-camera key、state4/action4/H16、每个 endpoint 固定 K、原生 snapshot 这些 hook 在 CPU 上正确，但**端到端不可用**：P2 worker 一发请求就 KeyError（#1，blocker）；服务入口没有五层（#4）；库与 K 绑定，做不了库 + 减步；服务端和 worker 的 debug 采集都部署不了（#5）；也没有 GR00T 的 conductor 启动入口（`conductor/driver.py:main` 写死了 π）。
- **LIBERO legacy 不受影响？** 运行时：**是**，hook 关闭时字节等价，213 项 legacy/default-off 测试通过。R16 封存流水线：**否**（#2），`analyze verify` 及其后的 dev/test/retest 分析都被拒。

## INVENTORY 中这两个模型目前仍不支持的功能

两个模型共同缺：
- S04/S05：replica proxy 只有 plan，没有接线也没有验证。
- S06–S08：动态阶段 batching 和在线 batch 参数没挂上（`Pi05StageBatcher`、GR S3 batcher 都没用到）。
- S13：stage1-only、compile、stage 放置。
- P04：CP1 judge 只有 always_hit/always_miss/threshold；没有 warm tiers，没有 CP3，没有 Qdrant 验证。
- P05–P09：五层不能按 bundle 选择（π 是进程级且只在 K≠10 时生效；GR 完全没接），没有 target_IR→gap，R16 descent 的档位无法表达。
- P13：WARM_START。
- P14：warm reset / self-start。
- P15：CP2、online-RIT、shadow、routing。
- P16：H2/M2 和 camera 变体。通用执行器在加载 bundle 时直接拒绝。
- D01：PolicyRecorder `--record`。
- D02：DataCollector H5 embeddings。
- D03/D04：staged trace twins，`trace.enabled` 会被拒绝。
- D07：look recordings。
- D10：augmentation / oracle shadows。
- D12：warmup factor calibration，因为 judge 种类受限，实际不可用。
- A01：逐决策成本和配对定价（#6）。
- A04：物理语义分析，目前只有 reach 的 goal，predicates、restore、robot role 都是 unsupported。
- X01：split。
- C04：warmup/calibration DAG。

π0.5-MW 额外缺：
- P11：commit>5（H5 下按设计不适用，正确拒绝）。
- 五层 + 满步 MISS（#3a）。

GR00T-MW 额外缺：
- 整个 MW 闭环（#1）。
- 五层（#4）。
- 库 + 减步 MISS（#4）。
- D05/D06/D08/D09：服务端和 worker 的 debug/物理采集（#5）。
- conductor 启动入口。
- S1/S2 batching。
