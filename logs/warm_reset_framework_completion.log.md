# Warm reset 一等公民框架补全：plain / full 臂、无库自产、GR00T 每 K 端点、钉物体 RoboCasa、注册钩子与分析适配器

> 状态：`Verified — 已提交`（代码与测试完成，CPU 全量 Verify 无新增失败、π0.5 MISS 步数 GPU 逐位对等通过；owner 2026-09-25 16:20 CDT 当次豁免 plan / G1，直接实现；owner 2026-09-25 19:57 CDT 指示提交推送）。
> Authority: Execution。上位：[`warm_continuation_first_class_plan.log.md`](warm_continuation_first_class_plan.log.md)（框架本体，dea5066）；差距清单：[`warm_reset_migration_study.log.md`](warm_reset_migration_study.log.md) §2–§8。
> 决定来源：owner 2026-09-25 16:20 CDT——全部在跑实验与新 MetaWorld benchmark 改走 warm reset 一等公民框架。
> 约束：不改 `exp/step_diag/**`、`tests/exp/step_diag/**`；不碰在跑队列 `sdq` 与操作方 GR00T LIBERO（`/data/openpi_wr` 冻结树、端口 23180–23189、tmux `wr_*`）；`exp/metaworld/**`、`src/openpi/policies/metaworld_policy.py`、`src/openpi/training/config.py` 属 MetaWorld 线，未改；未改 `logs/README.md` / `docs/README.md` 索引（由上级会话更新）。

## 1. 做了什么

| # | 需求 | 实现 |
|---|---|---|
| 1 | π0.5 按 bundle 钉 MISS 步数 | 新顶层 yaml 块 `miss: {num_steps, evidence_dir}`（`MissConfig`）；interceptor 新构造参数 `miss_num_steps`，MISS 分支（直连 `run_stage3(num_steps=)` 与 coordinator `Stage3MissPayload`）用它；缺块时在调用时读模块 `_NUM_STEPS`（step_diag 的进程级猴补丁照旧生效）。`__hit_meta__["miss_nfe"]` 加法字段（仅有块时出现）；证据 wrapper 对 plain/full 臂同样写决策 / finalize 行，决策行带 `miss_nfe`；准入新增 `MissSpec` 分支。 |
| 2 | GR00T full / plain_k | GR00T 的 MISS 步数是进程级 `--denoising-steps`：`miss_num_steps` 在 GR00T 上是断言（构造时与每次 MISS 前核对活 head 步数，不符即在任何 head 调用前抛错），load guard 同样核对；入口按 K 规划端点（`--k-servers k=host:port`），每臂冻结 `endpoints`，driver 只把臂派到对应端点，准入拒收派错端点的 episode。未做「infer-lock 下逐 bundle 改 head 步数」：那要改共享 head 的状态，不干净。 |
| 3 | 无库自产触发 | `warm_reset.trigger: always` + `warm_reset.start_t`（缺省 `verdict` = 现行为），仅允许 `start.source: self` 且不启用任何 checkpoint（判决结构上恒为 MISS，不检索）。两个 interceptor 调执行体新入口 `run_self_only`：与判决路径同一自产 + 续跑体，起点形状取自模型配置（π0.5 `[1,H,D]` float32、stage-3 设备；GR00T `[H,D]` float32 主机张量，与库快照同形同 dtype）。决策 `hit_type: SELF_ONLY`（wire 与证据），`start_t` = 块值；准入按 SELF_ONLY 分支核对，K+N 计价。 |
| 4 | 入口补齐 | `exp/warm_reset`：臂类型 `warm` / `self_only` / `miss`（`full`、`plain_k<k>`）；无库臂由无库模板生成，只含无库臂可省 `--base-yaml`；GR00T 每 K 端点；钉物体 RoboCasa（`--pinned-objects`：pin 表冻进 plan，EpisodeTask 带 pin 三键，agent 核对本机 manifest 的 `pin_id` 并传给 worker，受信身份与配对身份带 `pin_id`）；RoboCasa 稳定 experiment id（`--experiment-id`）；`plan.json` 记录 `init_pool_sha256`；环境注册表 `exp/warm_reset/envs.py` + 插件钩子 `exp/<包>/warm_reset_env.py`。 |
| 5 | 分析适配器 | `exp/warm_reset/analysis.py`：读一个或多个同环境运行目录，按配对身份组织逐集结局 / 实测 NFE / 决策数，调用 `warm_variants` 与 `success_length` 的统计函数，输出同形 JSON / MD；默认拒绝框架 cell 与 step_diag cell 配对，`--allow-cross-framework` 才算同臂差（标注、不进 macro）；不同环境拒绝合并。 |
| 6 | 文档 | `docs/cache/warm_reset_experiments.md`（臂类型表、GR00T 每 K 端点、钉物体与稳定 id、init pool、分析、注册钩子）；`docs/architecture/cache_system.md` §5.22.1；`docs/cache/tutorial.md`（`trigger: always` 与 `miss` 块）。 |

## 2. 改动文件

**src**
- `src/openpi/cache/config.py`：`WarmResetConfig.trigger/start_t`、`MissConfig`、`CacheConfig.miss`、`is_library_free()`、`_warm_reset_errors` 的 trigger 规则、`_miss_errors`、抽出 `_evidence_dir_errors`。
- `src/openpi/cache/warm_reset/types.py`：`WarmResetSpec.trigger/start_t/always`（缺省时 digest 与旧值逐位相同）、`MissSpec`、`HIT_SELF_ONLY` / `TRIGGER_*`。
- `src/openpi/cache/warm_reset/runtime.py`：`_build` 支持 `miss` 块（无执行体 + wrapper + `miss_num_steps`），执行体工厂收 `schedule=`；`WarmResetParts.miss_num_steps` / `interceptor_kwargs()`；`refuse_warm_reset` 同时拒 `miss` 块。
- `src/openpi/cache/warm_reset/pi05.py` / `groot.py`：执行体 `trigger_always`、`run_self_only`，公共体抽成 `_execute`（数值语句不变）；GR00T 执行体在装配时拿到配置 schedule。
- `src/openpi/cache/warm_reset/evidence.py`：决策行加 `miss_nfe`（仅响应带它时）；准入 `MissSpec` 分支、SELF_ONLY 期望 hit type、问题码 `miss_nfe_missing`。
- `src/openpi/cache/interceptor.py`：`miss_num_steps` 参数与拒绝组合、`_miss_steps()`、MISS 两条调用与写库步数改读它、自产-only 分支、hit meta 的 `miss_nfe` / `SELF_ONLY`。
- `src/openpi/cache/groot/interceptor.py`：`miss_num_steps`（断言 + `miss_nfe`）、自产-only 分支、拒绝组合。
- `src/openpi/cache/groot/load_guard.py`：无库配方放行空 checkpoint 集并跳过 artifact 身份，核对 MISS 步数 / 自产 schedule 与活 head。
- `src/openpi/cache/trace/runtime.py`：`MissConfig.evidence_dir` 登记为孪生只读路径字段（路径字段守卫测试要求）。

**serving 装配**
- `scripts/serve_policy.py`：两处 `InferenceInterceptor(..., miss_num_steps=)`。
- `exp/libero_groot/serve_groot_libero.py`、`exp/robocasa365/serve_groot_n15.py`：注入改为 `parts.interceptor_kwargs()`（warm 臂仍恰为 `warm_reset=executor`）；LIBERO key builder 检查对无库配方跳过。

**exp/warm_reset**
- `envs.py`（新）：`EntryEnv` / `BenchmarkAdapter` / `LiberoAdapter` / `RoboCasaAdapter`、`register_env` / `get_env` / `env_ids` / `entry_env_from_spec`、插件发现。
- `plan.py`：臂类型、无库模板、`miss_block`、每 K 端点、pin / experiment id / init pool；`read_plan` 按类型核对；旧计划缺省为 warm。
- `conductor.py`：strategy 用适配器生成 experiment / extra，钉端点；`worker_agent` 交给适配器。
- `admit.py`：按臂类型的 worker hit type、MISS 期望、适配器身份、端点核对。
- `run.py`：`--self-trigger`、`--k-servers`、`--experiment-id`、`--pinned-objects`（prepare 与 agent）、`--init-pool-sha256`，`--base-yaml` 可选，`tasks` 走适配器。
- `analysis.py`（新）：见 §1-5。

**tests**（新）
- `tests/cache/warm_reset/test_miss_and_self_only.py`：配置正反例、旧 digest 冻结值、π0.5 每 bundle MISS（直连 / coordinator / 双 bundle 同模型）、缺块读 `_NUM_STEPS`、MISS 证据准入正负例、无库自产 = 带库自产（逐位）、SELF_ONLY 准入。
- `tests/cache/warm_reset/test_miss_parity.py`：CPU 桩上每 bundle `plain_k`/`full` 与 step_diag `Pi05DiagInterceptor` plain 臂逐位相等（直连 + coordinator，k ∈ {1,2,3,10}）。
- `tests/cache/warm_reset/test_miss_parity_manual.py`（manual GPU）：真 checkpoint 上同一对等。
- `tests/cache/groot/test_warm_reset_groot_library_free.py`：GR00T MISS 断言、无库自产 = 带库自产、load guard。
- `tests/exp/warm_reset/test_entry_arms.py`：新臂类型、每 K 端点（含 GR00T 无库 yaml 过其端点的 load guard）、钉物体、稳定 id、init pool、注册表与插件发现、端到端准入、分析（合并分段、拒绝跨环境、跨框架标注）。

## 3. 新 yaml 字段与缺省

| 字段 | 缺省 | 含义 |
|---|---|---|
| `warm_reset.trigger` | `verdict` | `always` = 无库自产（仅 `source: self`、无启用 checkpoint） |
| `warm_reset.start_t` | 无（`null`） | `always` 必填（schedule 的可恢复点）；`verdict` 下禁止 |
| `miss`（顶层块） | 无 | 缺省 = 今天的 MISS |
| `miss.num_steps` | 必填 | π0.5 每 bundle 的 MISS 步数；GR00T 必须等于活 head 步数；≠ K 时需 `write_policy: never` |
| `miss.evidence_dir` | 必填 | 服务端证据目录 |

`miss` 与 `warm_reset` 互斥（一臂一条证据流），`miss` 拒 trace / shadow teacher / routing。缺这些字段时所有路径逐位不变：`WarmResetSpec` 摘要与旧值相同（测试冻结了 HEAD 值），interceptor 不注册新 probe、wire 无新键，`_wrap_policy` 返回类型不变，入口对只含旧臂类型的运行写出与之前同构的 `plan.json`（新键只在使用时写）且 warm 臂 yaml 文本不变。

## 4. 入口参数表达（摘要，详见 guide）

- `full` / `plain_k<k>`：`--arms full,plain_k2`（π0.5 同一 server）；GR00T 另加 `--k-servers "1=h:p1,2=h:p2"`，对应 server 以 `--denoising-steps k` 起。
- 缓存 reset / shoot：`--arms warmreset_t0.2,midreset_t0.75_n1 --base-yaml <库 yaml>`。
- 带库自产：`--arms selfresetfinal_t0.2 --base-yaml <库 yaml>`（`--self-trigger verdict`，缺省）。
- 无库自产：`--arms selfresetfinal_t0.2 --self-trigger always`（无需 `--base-yaml`）。
- 钉物体 RC：`prepare --pinned-objects exp/robocasa365/config/pnp_pinned_objects.json` + `agent --pinned-objects <本机副本>`；稳定 id：`prepare --experiment-id <id>`。
- 新 benchmark：`exp/<包>/warm_reset_env.py` 在导入时 `register_env(EntryEnv(..., adapter=<BenchmarkAdapter 子类>))`。

## 5. 设计取舍与偏差

- **`miss_nfe` 是「交给模型自身循环的步数」**，不是逐步计数：π0.5 的 MISS 走模型自己的 `range(num_steps)` 循环（直连 / coordinator 桶），要逐步计数只能改共享模型实例或再转写一份 MISS 循环（后者要新 payload 与新的逐位对等负担）；GR00T 用活 head 的 `num_inference_timesteps`，与 `GrootStage3Output.steps_run` 的既有口径一致。旧 server 忽略块时行里没有 `miss_nfe`，准入以 `miss_nfe_missing` 拒收；步数不符以 `steps_mismatch` 拒收。
- **GR00T 每 K 一个进程**（任务 2 允许的两种之一）。GR00T `plain_k1` 需 `--denoising-steps 1` 的 server；`groot_n15_schedule` 拒绝 K<2，本改动的 MISS / 无库路径不调用它，但 K=1 GR00T server 的整条生产启动路径本机未实跑（无 GPU 余量，见 §7）。
- **无库自产要求不启用任何 checkpoint**：否则 FULL_HIT / WARM_START 判决会抢先，`always` 名不副实。interceptor 仍调用 `orchestrator.check`（未配置 checkpoint 时立即返回 MISS，不检索、不建 key）。
- **env 注册表**：step_diag 六个环境按原 id 注册；插件按约定文件名发现（`exp/*/warm_reset_env.py`），失败只影响请求该 id 的调用。发现时 MetaWorld 线已在 `exp/metaworld/warm_reset_env.py` 放了一个同名数据模块（尚未调用 `register_env`），可正常导入，`pi05_metaworld` 待其接入钩子后可用；未改其文件。`BenchmarkAdapter.export_tasks` 返回完整任务清单（含 `init_indices`），便于 MetaWorld 这类「idx 选 train task」的 benchmark。
- **success_length 只输出决策数单位**：journal 不记录环境步数，适配器不伪造 `env_steps`。
- **init pool**：prepare 主机可读 pool 时计算 `sha256_tree`；worker 本地路径用 `--init-pool-sha256`；都没有时记 `null`（prepare 警告），分析以 `path:<目录>` 作 pool 身份。
- 为兼容 `tests/libero_groot/test_dynamic_bundle_guards.py` 对 `load_guard` 模块的整体替身，`is_library_free` 定义在 `openpi.cache.config`（通用：「不检索的配方」），`load_guard` 复用它。

## 6. 向后兼容核验

- 操作方的真实运行目录 `/data/wr_runs/smoke_groot_libero_spatial`（dea5066 冻结树 prepare / run / admit）复制到 `/tmp/wr_compat/` 后，用新代码 `admit`：`ok`、逐臂汇总与逐集报告与其 `admission.json` **完全相等**（含 replan 后的派发身份核对）；分析适配器在该副本上正常出表。
- 既有测试：`tests/cache/warm_reset`、`tests/cache/groot/test_warm_reset_groot.py`、`tests/exp/warm_reset/test_entry.py`、`tests/cache/test_interceptor.py`、`tests/libero_groot` 全部通过（未改其内容）。

## 7. 测试

新增测试文件与覆盖见 §2 tests。定向运行（CPU，`CUDA_VISIBLE_DEVICES=""`）：
`tests/cache/warm_reset/test_miss_and_self_only.py` 52、`test_miss_parity.py` 8、
`tests/cache/groot/test_warm_reset_groot_library_free.py` 21、`tests/exp/warm_reset/test_entry_arms.py` 34，全部通过；
既有 `tests/cache/warm_reset`、`tests/exp/warm_reset/test_entry.py`、`tests/cache/groot/test_warm_reset_groot.py`、
`tests/cache/test_interceptor.py`、`tests/libero_groot`、`tests/cache/trace/test_config_trace.py` 未改内容、全部通过。
「缺字段逐位不变」的测试保证：旧 `WarmResetSpec` 摘要冻结值（HEAD 代码算出）、缺块时 MISS 在调用时读 `_NUM_STEPS`
且 wire 无新键 / 无证据目录、`_wrap_policy` 返回类型不变、默认选项的 `plan.json` 键集与旧版相同、旧形态计划 replan 出与
旧 strategy 相同的 EpisodeTask 字段、混合运行里 warm 臂 yaml 文本与仅 warm 臂运行相同。

## 8. Verify / GPU 对等记录

- **manual GPU 对等**（2026-09-25 17:03–17:06 CDT，weilandserver 4090，运行前空闲 11.96 GB ≥ 需求约 8 GB + 4 GB；
  未动其他进程）：`SD_ENV_ID=pi05_libero_10 SD_CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
  SD_OBSERVATION=/tmp/sdiag/obs/obs_pi05_libero_10.npz pytest tests/cache/warm_reset/test_miss_parity_manual.py --run-manual`
  → **1 passed、0 skipped**（165 s）。k ∈ {1, 2, 3, 10}、同一噪声：每 bundle `miss` 的直连与 coordinator（桶内单请求）
  动作与 step_diag `Pi05DiagInterceptor` 进程级 plain / full 臂逐位相等，`miss_nfe == k`，四个 k 的动作互不相同。
  日志 `/tmp/wr_complete/miss_parity_manual2.log`（第一次运行因测试自身的临时目录未建而失败，修测试后重跑）。
- **未跑的 GPU 项**：GR00T 无库自产 / MISS 断言的真模型对等（CPU 桩已逐位；GR00T MISS 与 step_diag 同为进程级
  `--denoising-steps`，没有新数值路径）；π0.5 无库自产的真模型对等（与带库自产共用同一执行体 `_execute`，CPU 桩逐位相等，
  带库自产的真模型对等见上位计划 §15.4）；任何 serving / 模拟器冒烟（本轮未起 server）。
- **§6 CPU 全量 Verify**：见下（首轮 17:03–17:07 CDT 发现 1 个新失败：`tests/cache/trace/test_config_trace.py::
  test_every_path_like_config_field_is_classified`——新 `MissConfig.evidence_dir` 未登记到 trace 孪生路径表；已在
  `src/openpi/cache/trace/runtime.py` 的 `TWIN_READONLY_PATH_FIELDS` 登记（miss 与 trace 互斥，永不进孪生））。
- **§6 CPU 全量 Verify（终稿）**：标准命令（`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu ... pytest -n 24 --dist loadfile -q`，
  日志 `/tmp/wr_complete/verify2.log`）→ **7418 passed、95 skipped、21 failed、49 errors**。21 个失败恰为既有清单：
  `test_ws2_evidence_runner` ×2、`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2、
  `test_prebuilt_matrix_backend` ×2、`tests/review_tests` 15 个（与首轮逐项相同，未读其内容）；49 个错误来自既有的 3 个
  收集错误文件（`test_bench_groot_stages.py`、`test_review_cache_prune_g2.py`、`online_rit_g2/test_round2_boundaries.py`）。
  无新增失败。
