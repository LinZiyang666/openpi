# R17 integration review A — round 3 (verify Fix round 2, then fix the rest myself)

2026-10-09，checkout `/home/weiland/projects/openpi`（branch `Ziyang`，未提交）。本轮按 owner 指示，审查后由我直接修复剩余问题。

范围与写权限：
- 审查范围：P1 framework、MetaWorld/LIBERO benchmark 与 conductor、π0.5-MW、GR00T-MW、P6 工具。
- 我改过的目录只有以下几个：
  - `exp/r17_integration/{models/groot,debug,search}`；
  - 上述目录对应的 tests，外加 `tests/conductor` 中的一个 P2 测试；
  - `review/repro_A3/`。
- **未改动**：`common/`、`models/{fastwam,dp}`、`qualification/`、4 个封存共享源码、R16/harness、`exp/mw_finetune/`。
- 只用 CPU，未 commit/push，未碰任何运行中的进程。

基线：Fix round 2 之后、我修改之前：
- R17 cross：346 passed；
- `tests/models/groot`：**54 passed / 7 failed**。P1 收紧 canonical task table 后，GR00T 的测试 fixture 没有迁移；P5 本轮没有做 Fix round 2。

## 1. 第二轮发现（N1–N5）在当前代码上的核对

复现脚本：第二轮的在 `review/repro_A2/`，本轮新增的在 `review/repro_A3/`。

| ID | 结论 | 证据 |
|---|---|---|
| N1 major：`target_ir` 用 MISS 比例冒充 owner IR | **已修复（P1）** | `common/recipe.py:12-16` 去掉了 `(1-t)/t`。`common/ir.py` 实现了与 R11 相同的库回放二分（bracket 0–40、26 次迭代、setting 保留 4 位小数），计入 guard 和 commit tail，并要求绑定 MISS sampler、协议和来源的 `r17.ir_costs.v1` 系数；缺系数或目标不可达时拒绝。`repro_A2/r_target_ir.py` 现在报错 `no MISS-fraction shortcut`。P1 的 R11/R16 一致性测试包含在 cross 中，全部通过。我在 GR 热切换测试里也改为断言 calibration receipt：`periodic_gap == setting`，`pred_IR_rounded == owner_ir(N,V,M,costs)`，且 gap 不等于 `(1-t)/t`。 |
| N2 major：recipe 任务编号与 benchmark task_id 不校验 | **已修复（P1/P2），GR 侧由我补齐** | `common/benchmarks.py:46-54` 要求 `ordered_tasks` 与 task_table_identity 逐字等于 canonical 表；`common/serving.py:224-236` 在任何模型计算之前核对 `task_name`/`task_id`。`LayoutRecipe` 底层 API 仍用数字编号，但服务边界已经卡住。`repro_A3/r2_task_table_and_n5.py`：字母序表 → `recipe ordered_tasks differs from canonical benchmark task table`；reach-v3 的编号配 push-v3 的名字 → `request task_name/task_id differs ...`。GR 的 default config 原先不带任务表，现已补上，见 R3-2。 |
| N3 minor：下降中非 Fatal 前缀的终局错误导致卡死 | **已修复（P2）** | `conductor/strategy.py` 改用 driver 的 `is_retriable_error` 判定；派发前的失败统一走 `handle_result`。`repro_A3/r8_descent_nonfatal_error_v3.py`：`ConfigValidationError`/`SidecarError`/`FatalEpisodeError` 三种情况都是 `all_done True`，出错的任务停止且不派下一档，健康任务照常走到 k1，freeze 为 `diagnostic_error`。 |
| N4 minor：每次 bind 都全量重哈希 teacher store | **已修复（P1/P2）** | `common/recipe.py:99-138` 按（路径, digest）缓存已验证的 Store，用 inode/mtime/ctime 指纹检测变化；bind 改在 worker 线程中准备（`common/websocket_policy_server.py:45-66`）；stage 开始时只 select 已存在的 bundle，缺失的才重新 load。`repro_A2/r_bind_store_rehash.py`：3 次 bind，每次哈希 0 个文件、0 字节。 |
| N5 minor：计价口径（host wall 含锁等待） | **部分修复（P1/P2）→ 剩余部分由我完成** | P1 把计时拆成 `queue_wait_ms`/`service_ms`/`pricing_time_field`；π 提供了 `drain_queue_wait_ms`（返回 0，因为没有共享锁）。剩余问题：GR 没有原生 hook，所有决策都不可计价；P6 分析器仍用 host-wall `cost_ms` 求和计价，reader 还会丢掉 service 字段。见 R3-1、R3-3。 |

第一轮的 A1–A13 仍然有效，没有回退：`repro_A2/r1_groot_p2_runner.py` 仍输出 `success True`，决策序列为 policy/tail/policy，终局 ack 为 `executed_count 2, terminal True`。

## 2. 本轮发现并由我修复的问题

| # | 严重度 | 问题 | 修复（文件:行） | 回归测试 |
|---|---|---|---|---|
| R3-1 | minor（N5 剩余） | GR00T-MW 没有原生的共享 `model_lock` 等待计时。common 因此对每次 GR 决策给出 `service_ms=None, pricing_eligible=False`，无法区分排队和服务时间。 | `models/groot/provider.py:20-44` 新增 `_TimedLockView`：互斥锁仍是进程共享的同一把 RLock，只在每个连接上累计等待 acquire 的时间；`:80-82` 提供 `drain_queue_wait_ms`；8 处 `with self.model_lock` 改为走这个计时视图。`adapter.model_lock` 仍是共享锁，原有的同一性断言保持成立；P2 的 `instrument_model_lock` 遇到已有 hook 时直接返回，不会重复包装。 | `tests/models/groot/test_round3_review_a.py::test_round3_native_model_lock_wait_is_separated_from_service`：另一线程持锁 50 ms 时，`queue_wait_ms ≥ 30`，`service_ms = cost_ms − queue_wait_ms`，各 stage 的 queue+service 等于 wall；与无竞争的新连接相比动作字节完全相同；drain 之后归零。 |
| R3-2 | minor（N2 剩余，GR） | GR 自己的配置生成器 `default_config`（`serve.main` 不带 `--cache-config` 时使用）不带 canonical 任务表。P2 只能靠一个 bridge shim 补救，且有测试把“缺表”当作预期断言。 | `models/groot/serve.py:116-139`：默认 bundle 带一个禁用状态的 `RecipeConfig`，内含 50 个 canonical 任务和 `task_table_identity`，使每个请求在模型计算之前就核对 name/index。 | `test_round3_review_a.py::test_round3_default_config_carries_canonical_table_and_rejects_mismatched_task`（名字与编号不一致时，stage 计数不变就被拒绝）；`tests/conductor/test_p2_fix_round2.py::test_gr_default_yaml_emission_bridge...` 改为断言原生输出已带表、bridge 是幂等的空操作。 |
| R3-3 | minor（N5 剩余，P6） | 分析器计价仍用 `sum(cost_ms)`，即包含共享锁排队的 host wall；`normalized_within_model` 也是两个 wall 总和之比。reader 的 `cost_fields` 会丢掉 `queue_wait_ms`/`service_ms`/`pricing_time_field`，所以 P1/P2 发布的 service 口径到不了分析层。 | `search/analyze.py:92-113`：`pricing_absolute_ms = Σ service_ms`，只计入声明了 `pricing_time_field=service_ms`、无采集、可计价的行；baseline 必须给出同一字段的 `pricing_absolute_ms`；`absolute_ms` 保留为 wall 诊断值，新增 `queue_wait_total_ms`。`:12-26,121-127` 新增可选 `ir_costs`：用实际 stage 台账按 `(vision·V+miss·M)/N` 输出 `owner_ir`，并要求系数绑定同一个 MISS sampler。`debug/reader_profiles.py:25-87` 透传并校验 queue/service 字段：`service+queue` 必须等于 `cost_ms`；未拆分的行一律不可计价；`pricing_time_field` 只接受 `service_ms`。 | `tests/search/test_round3_review_a.py`（4 项）：wall 20/queue 12/service 8 → 计价 8，归一化 8/16 而不是 20/40；无拆分的行不可计价；owner IR 数值正确，sampler 不一致或缺 schema 时报错；计时字段优先级（R3-4）。现有 fixture 补齐 service 字段：`tests/debug/fixtures.py`、`test_analysis.py`、`tests/search/test_review_costs.py`。 |
| R3-4 | minor（本轮新发现） | 字段名冲突：osdebug 的 `BlockWriter.put` 会把**写入器的反压等待**写进 capture 记录顶层的 `queue_wait_ms`；episode metadata 里也可能带着无关的 cost 副本。原 `cost_fields` 按后者覆盖前者合并，顶层值会盖掉决策自己的锁等待和 `cost_ms`。修复后真实的 capture 路径（`test_review_runtime[True]`）确实触发了 `service+queue≠cost` 校验，暴露了这个冲突。 | `debug/reader_profiles.py:35-48`：同一决策的计时字段只取一个台账（diagnostics/meta 优先，取第一个非空值）；写入器的等待另存为 `writer_queue_wait_ms`。共享的 writer 不在写权限内，未改。 | `test_round3_writer_backpressure_and_episode_copies_do_not_replace_decision_timing`；`tests/debug/test_review_runtime.py` 中的 GR 装配 fixture 改用 reach-v3 的 canonical task_id。 |
| R3-5 | blocker for GR suite（fixture 迁移） | 7 个 GR 测试的 recipe helper 不带 `task_table_identity`，并且断言旧的 `(1-t)/t`。P1 的严格校验对它们是正确拒绝。 | `tests/models/groot/test_fix_round1.py`：`recipe_config` 改用 canonical 表和 identity；`target_ir` 配上标注为 CPU 诊断用、非实测的 `bound_costs`。`test_fix_round1_bundles.py`：改为断言 calibration receipt；服务来源必须与 knob/guard 的强制标志一致；至少有一个 recipe bundle 真正命中库。 | GR 套件 54/7 → **63 passed**。 |

修复的灵敏度证明见 `review/repro_A3/mutation_witness.py`：在内存中逐一放回旧机制（删掉 GR hook；default config 去表；分析器按 wall 计价），对应的新回归测试 **3 项全部变红**，恢复后 **3 项全部变绿**。

## 3. 最终 CPU 结果

所有命令都在以下环境下运行：`CUDA_VISIBLE_DEVICES=''`、JAX 只用 CPU、BLAS/OMP 单线程、不写 pyc/cache；产物在 `/tmp/r17_review_A3/final`。

| 运行 | 结果 |
|---|---|
| `python -m exp.offline_search.rounds.r16.analyze verify` | `DESIGN_VERIFIED; no outcome reads` |
| `git diff HEAD -- src scripts examples packages exp/metaworld exp/offline_search exp/libero_groot` | 空，共享源码与 HEAD 一致 |
| R17 cross：`tests/{common,benchmarks,conductor,models/pi05,debug,search,ops}` | **350 passed** |
| `tests/models/groot` | **63 passed** |
| `tests/models/fastwam`（.venv，只读核对） | **110 passed** |
| `tests/models/dp`（DP island 的 python，只读核对） | **125 passed** |
| Legacy 与 default-off（明细见下） | **1087 passed，16 skipped，1 deselected** |
| `tests/serving/test_replica_proxy.py -m env_dependent --run-manual` | **6 passed** |

Legacy 运行包含：
- `test_default_off`；
- cache 的 config、denoise_schedule、storage、phase3 测试；
- serving 的 identity、recorder、replica、router 测试；
- MW 的 env/runner、`exp/mw_finetune/tests`；
- `tests/conductor`、`tests/cache/groot`、`tests/libero_groot`；
- R16 的 conductor 与 groot_servers 测试。

16 个 skip 是：6 个 proxy socket 测试（上表已单独运行通过）、2 个需要 CUDA、8 个需要 `--run-manual` 的真实模型或环境测试。deselect 的 1 个是真实 EGL 测试。

## 4. 分模型结论

- **π0.5-MW：在 CPU 上判定范围内正确，可以进入 G1 GPU 机械 smoke。**
  - 已具备：K10/2/1 采样；按 YAML 选五层（与 K 无关）；canonical 任务表在服务边界强制；R11 式 owner-IR 校准；queue/service 拆分计时；终局 ack；debug 采集；conductor 按 identity 匹配 endpoint。
  - 注意：
    - MW 的 IR 系数尚未实测，在实测之前不能用 `target_ir`，只能用显式 `periodic_gap`，而且不得称作 IR 档位；
    - π 没有共享锁，线程间的 GPU 争用会计入 `service_ms`，只能在相同并发下比较。

- **GR00T-MW：在 CPU 上判定范围内正确，可以进入 G1 GPU 机械 smoke。**
  - 已具备：经真实 P2 runner 端到端跑通；五层与热切换 YAML；K4 teacher 库可用于 K2/K1 endpoint；原生锁等待计时；default config 带 canonical 表；服务端和 worker 的 debug 采集。
  - 不支持：S1/S2 batching；S3 batcher 没有接入服务路径。真实 checkpoint 与 GPU 尚未验证。

- **LIBERO legacy 不受影响：是。** 共享源码与 HEAD 一致，R16 封存校验通过，legacy 测试 1087 passed。

## 5. 这两个模型仍不支持的功能

两个模型共同缺：
- S04/S05：replica proxy 只有 plan。
- S06–S08：动态阶段 batching 和在线 batch 参数。
- S13：stage1-only、compile、stage 放置。
- P04：CP1 只支持 always_hit/always_miss/threshold；没有 warm tiers，没有 CP3，Qdrant 未验证。
- P13–P16：WARM_START、warm reset/self-start、CP2/online-RIT/shadow/routing、H2/M2 和 camera 变体，通用执行器仍直接拒绝。
- D01：`--record`。
- D02：DataCollector H5。
- D03/D04：trace twins。
- D07：look recordings。
- D10：augmentation/oracle shadows。
- D12：factor calibration。
- A04：物理语义分析，目前只有 reach 的 goal；predicates、restore、robot role 都是 unsupported。
- X01：split。
- C04：warmup/calibration DAG。
- 实测的 MW 模型 owner-IR 系数与 GPU 服务时间：`gpu_service_ms` 恒为 None。

π0.5-MW 额外：commit>5（H5 下按设计不适用）。

GR00T-MW 额外：S1/S2 batching；不同 K 之间的 native resume（按设计要求同 grid）。

## 6. 我改动的文件

- 产品代码：
  - `models/groot/provider.py`
  - `models/groot/serve.py`
  - `debug/reader_profiles.py`
  - `search/analyze.py`
- 测试，新增：
  - `tests/models/groot/test_round3_review_a.py`
  - `tests/search/test_round3_review_a.py`
- 测试，迁移：
  - `tests/models/groot/test_fix_round1.py`
  - `tests/models/groot/test_fix_round1_bundles.py`
  - `tests/debug/fixtures.py`
  - `tests/debug/test_analysis.py`
  - `tests/debug/test_review_runtime.py`
  - `tests/search/test_review_costs.py`
  - `tests/conductor/test_p2_fix_round2.py`
- 复现脚本：`review/repro_A3/{r2_task_table_and_n5.py, r8_descent_nonfatal_error_v3.py, mutation_witness.py}`
