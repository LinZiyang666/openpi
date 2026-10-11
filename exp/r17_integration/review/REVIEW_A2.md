# R17 integration review A — round 2 (after "Fix round 1")

2026-10-09，checkout `/home/weiland/projects/openpi`（branch `Ziyang`，未提交）。审查范围与第一轮相同：
- P1 framework；
- MetaWorld/LIBERO benchmark 与 conductor；
- π0.5-MW；
- GR00T-MW；
- 涉及这两个模型的 P6 工具。

只读审查：未改代码、未用 GPU、未起服务、未碰运行中进程。第一轮复现脚本在 `review/repro_A/`；本轮新脚本在 `review/repro_A2/`，运行前缀与第一轮相同：

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python <script>
```

本轮实跑结果：
- `git diff HEAD -- src scripts examples packages exp/metaworld exp/offline_search exp/libero_groot`：**无输出**，共享源码与 HEAD 一致。
- `python -m exp.offline_search.rounds.r16.analyze verify` → `DESIGN_VERIFIED; no outcome reads`。
- R17 测试 `tests/{common,benchmarks,conductor,models/pi05,debug,search,ops}`：**281 passed**。
- `tests/models/groot`：**61 passed**。
- legacy 测试共 **1007 passed、10 skipped、1 deselected**。范围：`test_default_off`，`tests/cache/{test_config,test_denoise_schedule_config,test_cache_storage}`，`tests/serving/{policy_identity,per_step_recorder,router_bundle_rebind}`，`tests/exp/metaworld/{test_env,test_runner}`，`tests/conductor`，`tests/cache/groot`，`tests/libero_groot`；deselect 的是真实 EGL 测试。
- 第一轮 3 个复现脚本重跑后，原报错都不再出现（详见下表）。

## 1. 第一轮发现的修复核对

| ID | 结论 | 证据 |
|---|---|---|
| A1 blocker：GR00T 只认 `image/state`，P2 worker 发的是 `observation/*` | **已修复** | `models/groot/wire_policy.py:9-24` 的 `observation_field` 两套键都接受，冲突时 fail closed；`prepare`/`extract_state_only` 都改用它（`provider.py:68,72,157`）。端到端复现 `repro_A2/r1_groot_p2_runner.py`：用真实 `R17MetaWorldEpisodeRunner` + 原 MW kernel + GR00T stub 后端（commit10）跑一局，输出 `success True`；三次决策依次为 `('policy',(1,1,4),0,10)`、`('policy_tail',(0,0,0),5,5)`、`('policy',(1,1,4),0,10)`；服务端终局 ack 为 `decision_seq 2, executed_count 2, terminal True`；所有行都有 `cost_ms`。 |
| A2 major：R16 封存校验失败 | **已修复** | 4 个共享文件与 HEAD 字节一致；`r16.analyze verify` 通过。hook 移到 `common/config.py`、`common/native.py`、`common/websocket_policy_server.py`（子类 + socket facade）。 |
| A3 major：π 五层是进程级、绑死 K、队列分不出配置 | **已修复**（参数语义问题见 N1、N2） | YAML 新增 `r17.recipe.v1` 段（`common/config.py:12-34`）；`make_recipe_factory` 按 bundle 构建，K 不再影响（`common/serving.py:99`）；CLI 的 `--recipe-store/--periodic-gap` 会直接报错（`models/pi05/serve.py:112-113`）。`bundle_identity`/`recipe_identity` 进入 episode extra、`EndpointSpec.bundle_identities`（`conductor/manifest.py:19-28`），preload 时核对服务端返回的 identity（`conductor/strategy.py:81-84`）；服务端 `on_episode_start` 也比较 bundle identity（`common/serving.py:167-168`）。 |
| A4 major：GR00T 无五层、库与 K 绑死 | **已修复** | `serve()` 传入 `make_recipe_factory`（`models/groot/serve.py:93-96`）。teacher sampler 与 MISS sampler 分开：`retrieval_model_identity` 去掉 `default_sampler`（`common/recipe.py:74-80`），`validate_library` 跳过 MISS sampler、单独校验 teacher（`models/groot/manifest.py:184-214`）。原 `repro_A/repro_groot_meta_k.py` 现在输出 `K4 teacher library accepted on K2 endpoint`。native resume 仍要求同 grid。 |
| A5 major：GR00T debug 采集部署不了 | **已修复** | `manifest.to_dict()` 带 `r17_model_spec`/`r17_samplers`（`manifest.py:163-168`）；GR `main()` 有 `--debug-directory`（`serve.py:142`），走 P6 的 `CoreCaptureFactory`；worker 用 `PhysicalCaptureFactory`，按任务的 K 选 sampler。 |
| A6 major：成本台账凑不齐 | **已修复** | 每次决策写 `cost_ms`/`stage_ms`/`pricing_eligible`（`common/serving.py:341-345`）；`cost_row` 在客户端落盘（`benchmarks/execution.py`）；`episode_end` 带终局 `__execution__`（`execution.py:9-20`，`common/websocket_policy_server.py:52-70`）。π 的 tail 只做 Normalize（`models/pi05/adapter.py:193-208`）。剩余的计价口径问题见 N5。 |
| A7 minor：缺上下文时静默补默认值 | **已修复** | `_request` 缺 campaign/task_name/original_init_idx/task_id/seed/explicit_seed 时直接报错（`common/serving.py:200-207`）；同时识别 `orig_init_state_idx` 别名（:161-164）；R17 runner 默认 `framework_context=True`。`repro_A2/r7_context_and_ordered_tasks.py (a)`：用 stock runner 的 extra 调用 → `missing R17 request context: ['campaign', 'explicit_seed']`，别名读出的 idx=3。 |
| A8 minor：下降中终局错误导致卡死 | **部分修复** → 见 N3 | 只处理了以 `FatalEpisodeError` 开头的错误（`conductor/strategy.py:257`）。其他非重试错误（`ConfigValidationError`/`SidecarError` 等）仍然卡死。 |
| A9 minor：commit 不是 replan 的整数倍 | **已修复** | `validate_commit` 默认要求整数倍（`common/contracts.py:73-75`；原脚本报 `commit must be a multiple of replan=5`）；MW runner 和 plan 也拒绝；只有 LIBERO 经显式协商才允许 partial。 |
| A10 minor：物理采集与模型采集身份/sampler 对不上 | **已修复** | `__request__` 和 `_identity()` 都改用 `extra.bundle_generation`（`benchmarks/metaworld.py:161,201`）；`_identity` 带 `num_steps`，worker 按任务 K 选 sampler（`conductor/worker_entry.py:151-158`）。 |
| A11 minor：audit 漏查 library/cache tail | **已修复** | 改为 `source.endswith("_tail")`（`search/audit.py:28`）。 |
| A12 minor：GR00T capability 高报 | **已修复** | 直连 hook 改标 `implemented_unverified`（`models/groot/manifest.py:146-156`）。 |
| A13 minor：服务重启后 bundle 丢失 | **已修复** | 每个 stage 开始时重新发布 bundle，派发前用 `ensure_bundle` 检查（`conductor/strategy.py:34-62`）。代价见 N4。 |

## 2. 本轮新发现

| # | 严重度 | 位置 | 问题 | 证据 | 修复 |
|---|---|---|---|---|---|
| N1 | **major** | `common/recipe.py:9-12,112` | 新加的 `recipe.target_ir` 用闭式 `gap=(1-t)/t` 换算，实际控制的是 **MISS 占决策的比例**，不是 R11/R16 的 owner IR，即 (.152·V + c(k)·M)/N。R11 的做法是在库上回放 schedule（含 guard 强制 MISS）、按模型系数二分求 setting（`exp/offline_search/rounds/r11/knob/calibration.py:138-153`）。写 `target_ir: 0.25` 的 bundle 达不到 R16 “ir25” 的成本水平，下降档位的标签和成本结论都会错。 | `repro_A2/r_target_ir.py`（π、MW 每局 32 次决策、不计 guard）：`target .45 → MISS/N .469 → IR(K10) .549 / IR(K2) .385`；`.25 → gap 3 → MISS/N .250 → IR(K10) .364 / IR(K2) .276`。P1 自己的样例 YAML `fake_recipe_full.yaml` 用的就是 `target_ir: 0.45`。 | 二选一：改名为 `miss_fraction`，只接受显式 `periodic_gap`；或者实现 R11 的库回放校准（模型/K 各自的 V/M 系数，计入 guard），并在 fit receipt 里记录预测 IR。修好之前 `target_ir` 应该拒绝加载。 |
| N2 | **major** | `common/recipe.py:29-30,105`；`search/fit.py:62-69`；`common/layers/recipe.py:245,277,301`；`common/serving.py:211-212` | recipe 的任务编号来自 YAML 的 `ordered_tasks`，但查询用的是 benchmark 的 `request.task_id`（MT50 canonical 或 LIBERO 全局 id）。两者之间没有任何校验，`ordered_tasks` 只检查非空、不重复。只要它不是 benchmark 的 canonical 表（顺序不同，例如按字母排序：canonical 里 `hand-insert` 排在 `drawer-*` 之前；或者只取部分任务、恰好编号重叠），检索、早期度量、corrector、task one-hot 都会**静默使用别的任务的库**。部分编号对不上时会 KeyError，那种情况是 fail closed。 | `repro_A2/r7_context_and_ordered_tasks.py (b)`：`ordered_tasks=[reach-v3, assembly-v3]`，`assembly-v3` 的请求（canonical id 0）→ `served from library task index 0 (=YAML 'reach-v3'); returned action value 5.0`，即 reach 的动作。 | 校验 `ordered_tasks` 与 `adapter_binding.benchmark_protocol` 对应的任务表（MW 的 `FORMAL_SPEC.tasks`，LIBERO 的 40 任务）完全相等；或者查询改为按 `request.task_name` 映射。 |
| N3 | minor（A8 剩余） | `conductor/strategy.py:118-127,213-222,253-273`；`conductor/driver.py:43-56` | 实时路径只对以 `FatalEpisodeError` 开头的错误停止该任务。其他终局错误，即 repr 中含 `ConfigValidationError`、`ConfigValidation`、`SidecarError` 的非重试错误，会让 cohort 停在未判定状态；下一档 stage 照常激活，`on_dispatch` 报 “dispatch level differs”。这些任务已写进 dispatch 日志但没有发出，要等 1800 s 超时，重派后还会再报错，`all_done` 永远不成立。resume 路径则对任何错误都停止，两条路径不一致。`ensure_bundle` 失败走的是 `mark_result(retriable)`，次数耗尽后变成 done_fail，既不写 journal 也不通知 controller。 | `repro_A2/r8_descent_nonfatal_error.py`：`stopped: set() errors: {}`；`next-level pull -> ValueError dispatch level differs from active descent level`；`dispatched-but-unsent k1 tasks: 1 | all_done: False`。 | 所有被接受的非重试终局错误都走 `stop_error`；`on_stage_begin` 发现上一档未判定时跳过并停止该任务；`on_dispatch` 失败时 requeue，与 `ensure_bundle` 同样处理；`ensure_bundle` 次数耗尽时写 journal 并通知 strategy。 |
| N4 | minor（性能，规模大时可能变成错误） | `common/recipe.py:83-101`；`common/serving.py:59-67,94,99`；`src/openpi/serving/websocket_policy_server.py:899-901`；`conductor/strategy.py:34-62`；`search/build_library.py:114` | 每次 bind 一个 recipe bundle，`make_bundle_validator` 和 `bundle_recipes` 各打开一次 `Store`，在查 fit 缓存**之前**就对 store 的所有 array 文件做全量 SHA-256，也就是每次 bind 哈希两遍。stock 的 `_bind_bundle` 在 asyncio 事件循环上同步执行，哈希期间该进程所有连接都停住。会触发 bind 的场合：driver 每次派发都在控制连接上调 `ensure_bundle → select_bundle`（arm 交替时每次都会重新 bind）；每个 stage 开始时都把所有 bundle 重新 `load_cache_config`，又各哈希两遍并重新加载 artifact。P6 的 teacher store 会把采集到的**全部数组**写进去（`payload = dict(values)`）；π 采集包含 prefix_embeddings/prefix_out，粗估每行约 16 MB，200 个 episode 约 100 GB。只存 key 也约 0.8 GB。 | `repro_A2/r_bind_store_rehash.py`：每次 bind 哈希 8 个文件，store 只有 4 个 array 文件，即每次全量哈希 2 遍。事件循环同步 bind 见 stock 第 899-901 行。 | 进程内按（path, digest, inode/mtime）缓存已验证的 Store，先查 fit 缓存；bind 验证放到线程里执行；只在 stage 开始时检查是否存在（`select`），不存在才重新 load；teacher store 只存 key/state/action。 |
| N5 | minor（计价口径） | `common/serving.py:341-345`；`models/groot/provider.py:76,91,107,125`；`search/analyze.py:77-81` | 新的 `cost_ms` 是 host wall 时间，包含 GR00T 共享 `model_lock` 的排队等待和并发 GPU 争用。`summarize` 用 host-wall 总和除以 full baseline 得出 `normalized_within_model`，只有在两个 arm 并发负载相同时才可比。这与 R16 的 owner IR（N/V/M × 实测阶段系数）不是同一口径。 | 代码路径如左；P1/P6 自己也标注了 host_wall、GPU 计价待定。 | 报告里以 N/V/M 加阶段系数为主口径；`cost_ms` 只在同并发、无采集的配对 run 中使用，并在结果里写明并发与锁等待；或者把锁等待单独计时并从 `stage_ms` 中扣除。 |

LIBERO bridge（`benchmarks/libero_bridge.py`）按本轮范围读过：kernel 复用原 `_run_episode`；replan 用 `_ArgsProxy` 做按次截断；终局 ack、全局/局部 task id、原 init pool。**没有发现影响结果的问题**。注意它只接受 wire 96/256，`wire_size=224` 会被拒绝，因此 π0.5-LIBERO 不能走这条 R17 路径。这不影响本次两个 MW 模型。

## 3. 分模型结论

- **π0.5-MW**：通过框架接入的完整路径在 CPU 上已经打通：
  - full/K10/2/1 采样；
  - 按 bundle 选五层，且与 K 无关；
  - endpoint 按 identity 绑定，conductor 按 episode 分发；
  - 服务端和 worker 两侧的 debug 采集；
  - 逐决策成本加终局 ack；commit=5（H5 的上限）。

  可以进入 G1 GPU 机械 smoke。上 GPU 前须注意两点：用 `periodic_gap` 而**不要用 `target_ir`**（N1）；`ordered_tasks` 必须逐字等于 `FORMAL_SPEC.tasks`（N2）。正式 run 前处理 N4 的哈希开销。

- **GR00T-MW**：上一轮的 blocker 已修复，已用真实 P2 runner 做了端到端复现。现在具备：
  - 五层与热切换 YAML；
  - K4 teacher 库可用于 K2/K1 endpoint；
  - debug 采集（服务端和 worker）；
  - 用 `--arm` 启动 conductor。

  N1、N2、N4 同样适用。S1/S2 batching 不支持，S3 batcher 没有接入服务路径。GPU 与真实 checkpoint 尚未验证。

- **LIBERO legacy 不受影响？** **是**。共享源码与 HEAD 一致，R16 封存校验通过，1007 项 legacy 测试通过。

## 4. INVENTORY 中这两个模型仍不支持的功能

两个模型共同缺：
- S04/S05：replica proxy 只有 plan。
- S06–S08：动态阶段 batching 和在线 batch 参数未接入；GR 的 S3 batcher 也未接。
- S13：stage1-only、compile、stage 放置。
- P04：CP1 judge 只有 always_hit/always_miss/threshold；没有 warm tiers，没有 CP3，Qdrant 未验证。
- P13–P16：WARM_START、warm reset/self-start、CP2/online-RIT/shadow/routing、H2/M2 和 camera 变体，通用执行器仍直接拒绝。
- D01：`--record`。
- D02：DataCollector H5。
- D03/D04：trace twins。
- D07：look recordings。
- D10：augmentation/oracle shadows。
- D12：factor calibration。
- R11 校准的 target-IR 档位（N1）。
- GPU 上的 owner-IR 计价；目前只有 host-wall 时间（N5）。
- A04：物理语义分析，目前只有 reach 的 goal；predicates、restore、robot role 都是 unsupported。
- X01：split。
- C04：warmup/calibration DAG。

π0.5-MW 额外：
- commit>5：H5 下按设计不适用。

GR00T-MW 额外：
- S1/S2 batching。
- 不同 K 之间的 native resume（按设计要求同 grid）。
