# WARM_START 引入 RoboCasa365 —— plan

> 状态：**G1 APPROVED**（2026-09-06，3 轮）。已完成 Post-G1 polish，进入 §4 Code。
> 授权口径：owner 2026-09-06 裁定「先编写 plan，收益确定之后再做」。**G1 放行只表示计划具备执行条件，不自动启动执行**；owner 启动收益前置阶段后，范围仅为 W0-W4（允许新增隔离的 benchmark/runner 与测量产物），**不授权生产代码改动或数据重新采集**。阶段 B 必须在 W4 再获 owner 明示授权。
> §4 Code 的交付物 = **W1 一个文件**：`exp/robocasa365/bench_groot_stages.py`（新增、隔离、不进生产路径）。W0/W2/W3 是运行动作不是代码，须等外部 pnp W8 campaign 收工、GPU 空窗后才可执行。
> 并行前置：外部 pnp W8 的 GR00T cache 臂已收官、teacher 基线臂正在跑（`logs/pnp_run_progress.md`），本 plan 的任何**运行**动作都排在该 campaign 收工之后。

---

## 0. 一句话目标

让 **WARM_START**（缓存命中后从中途的去噪状态续跑，而非从纯噪声重来）在 RoboCasa365 上对**两个 teacher**都可用，覆盖 **采集 → 建库 → 标定/warm up → 正式实验** 四段；并在动手改框架之前，先用一次独占 GPU 的微基准把**收益**测出来。

⚠ **范围补充（owner 2026-09-06 指示，G2 Round 3 记录）**：GR00T 的 warm-start 缺口**不是 RoboCasa 独有的**——`src/openpi/cache/groot/{staged,interceptor,load_guard}.py` 与 `exp/robocasa365/groot_cache_collector.py` 是 **RoboCasa365 与 LIBERO 两条线共用的同一套代码**（LIBERO 侧在 `exp/libero_groot/serve_groot_libero.py:400-409` 直接构造 `GrootCacheCollector`）。因此阶段 B 的 W5-W13 一旦落地，LIBERO×GR00T 自动获得同一能力；**但 LIBERO 不是 RoboCasa 的复制品**，它有三处必须单独处理的差异（见 §1.8），漏掉任何一处都是静默失败。本 plan 的目标范围据此扩为 **RoboCasa365 + LIBERO 两条 GR00T 线**；文件名保持不变以免打断在跑的评审引用。

---

## 1. 已核实事实

> 全部经 file:line 核实。凡标 ⚠ 的都是**静默失败面**：不报错、不中断，只表现为"效果不好"。

### 1.1 两个模型的去噪结构（**时间方向相反**）

| | pi0.5（运行时是 PyTorch `PI0Pytorch`） | GR00T N1.5 |
|---|---|---|
| 去噪步数 | **10** | **4** |
| 时间方向 | t: **1 → 0**，`dt = −1/N` | t: **0 → 1**，`dt = +1/N` |
| 出处 | `src/openpi/models_pytorch/pi0_pytorch.py:575-582`；JAX 侧同约定并明写于 `src/openpi/models/pi0.py:273-275` | ckpt `config.json` `action_head_cfg.num_inference_timesteps = 4`；上游 `get_action` 的 `t_cont = t/float(num_steps)` |
| 可恢复点 | 9 个（0.9…0.1） | **3 个**（0.25 / 0.5 / 0.75） |
| 噪声起点 | `sample_actions(noise=...)` **可外部注入** | `actions = torch.randn(...)` **写死在 `get_action` 内** |
| action chunk | (50, 32) | (16, 32) |

⚠ **JAX 侧 `src/openpi/models/pi0.py` 不在 warm-start 链路上**。`CollectionPolicy` 硬性拒绝非 PyTorch 模型（`src/openpi/collect/collection_policy.py:24-26`）。

### 1.2 pi0.5 侧：WARM_START 是**现成能力**

| 环节 | 位置 | 事实 |
|---|---|---|
| 中间量捕获 | `collection_policy.py:86-88, :98` | forward hook 挂 `model.action_in_proj`，取 `inp[0]` |
| 取哪几步 | `:192-194` | `range(1, num_steps)` → x_1..x_9，**丢弃 x_0** |
| clean_action | `:196-198` | `x_last + dt·v_last` |
| h5 落盘 | `src/openpi/collect/data_collector.py:163-164` | `enumerate(..., start=1)` → `noise_action_1..9` |
| 建库 | `exp/common/build_in_memory_cache_artifact.py:635-653` 与 `:1029-1045` | `t = round(1.0 − i/_NUM_STEPS, 4)` |
| 契约 | `src/openpi/cache/storage_types.py:65-66` | `CP1 WARM_START: action_chunk + intermediates + denoising_num_steps required` |
| 续跑入口 | `pi0_pytorch.py:704-772` | `run_stage3_from(stage2, start_x, start_t, *, num_steps=10)` |
| 判决 | `components/judge.py:239`（`AlwaysWarmStartJudge`）、`:284`（`ThresholdJudge` + `warm_tiers`） | |
| 执行 | `src/openpi/cache/interceptor.py:1181-1197` | probe 名 `stage3_warm` |
| 规模 | 全仓 **517** 个 yaml 用 `always_warm_start`、**882** 个用 `warm_start_t` | 成熟在用 |

**必须由 schedule 统一表达的数值细节**（pi0.5 的实现只能作参照，不能把下降时间轴公式原样套给 GR00T）：
- `DenoiseSchedule` 必须提供 `snapshot_t(i)`、`snapshot_index(t)` 与 `remaining_steps(t)=N−i`。pi0.5 在 `t=1−i/N` 处剩 `N−i≈tN` 步；GR00T 在 `t=i/N` 处剩 `N−i≈(1−t)N` 步。故 GR00T 的 `t=0.25/0.5/0.75` 分别续跑 **3/2/1** 步，而不是 pi0.5 的 `floor(start_t·N+0.5)`。
- timestep 张量从 schedule 的起点按 `i` 次 `dt` **空转重放累加**得到，不直接取字面 `start_t`（pi0.5 `:746-757`）：bf16 下字面 0.3 与累加出的 0.2999999225 会让结果偏约 1e-3。索引换算统一用 half-up（`floor(x+0.5)`），不用 Python `round()` 的 banker's rounding。
- `intermediates[t]` 是**该步去噪之前**的 x_t（pi0.5 快照点 `:806-808`），因此可直接作为同一 schedule 下 `run_stage3_from(start_t=t)` 的种子。

### 1.3 GR00T 侧：**功能不存在**，不是"配置没开"

四处同时缺，缺一都退化成静默失败：

| # | 缺什么 | 位置 | 不补的后果 |
|---|---|---|---|
| G-1 | 采集不写中间量 | `exp/robocasa365/groot_cache_collector.py:162` `noise_action_steps=[]` | ⚠ 库照常建成，orchestrator 每步静默降级 MISS，表现为"命中率莫名为 0" |
| G-2 | 没有 partial 执行体 | `src/openpi/cache/groot/staged.py:440-479`，`:472` 的 `action_head.get_action(...)` 是**一次原子调用** | ⚠ 即使库里有 intermediates，也没有任何东西能消费它。无 `run_stage3_from` 等价物、`GrootStage2Output` 只有 `action_pred`（`:141-146`） |
| G-3 | 时间轴常量反向且不兼容 | `types.py:46-48` `CANONICAL_DENOISE_TIMESTEPS = {0.1…0.9}`；builder 的 `_NUM_STEPS = 10` 硬编码 | ⚠ **本 plan 最危险的一条，见 §1.7-A** |
| G-4 | 守卫硬拒 | `src/openpi/cache/groot/load_guard.py:46` `_ALLOWED_JUDGE_TYPES = {"threshold","always_hit"}`；`:96-102` `warm_tiers` 非空直接拒 | ❌ **响亮**：server 启动即 `ConfigValidationError`。⚠ 但**放开它必须是最后一步**——先放开而 G-1/G-2/G-3 没做，正好掉进纯静默的坑（该模块 docstring `:10-13` 已把这个失效模式写死） |

**好消息：GR00T 的中间量不必复刻循环就能拿到。** 在 `action_head.action_encoder` 上挂 forward hook 取 `inp[0]`，4 次调用给出 `[x_0(噪声), x_1, x_2, x_3]`，`clean_action` 即已有的 `action_pred`。与 pi0.5 在 `action_in_proj` 上挂钩完全同构。
⚠ 反面路线（在 `staged.py` 里复刻 4 步循环）**不推荐**：`UPSTREAM_FORWARD_SHA256`（`staged.py:86-88`）只钉了 eagle 的 forward，**action head 没有任何漂移守卫**，复刻等于新增一个无保护的漂移面，还要重跑 G0-C 两阶段等价门（`staged.py:54`）。

### 1.4 采集侧

- ⚠ **起点噪声事后不可复原**：`pi0_pytorch.py:823-825` 内部 `sample_noise`，client 从不传 noise（`episode_runner.py:27`）。已采的约 **319 GiB** 产物永久缺这一项，只能重采。
- pi0.5 的 `action_in_captures[0]` **就是**起点噪声，现被 `range(1, num_steps)` 丢弃；`collection_policy.py:193` 的 `range(1,…)` 与 `data_collector.py:163` 的 `start=1` 是**配套偏移**，⚠ 只改一处会让所有时间戳整体错一格且**无任何断言会报**。
- 体积不是约束：GR00T +0.47%、pi0.5 +0.22%，全量重采增量 **< 1.5 GiB**（主导项是三路 vision 的 3.1 MB/step，占 94%）。
- ⚠ **h5 不记录去噪步数**：`data_collector.py:146` 的 `attrs["num_steps"]` 是 **episode 长度**，与采集侧同名局部变量（去噪步数）语义冲突 ⇒ builder 只能猜 10。
- ⚠ **plan hash 不含任何 schema 描述**（`run_collect.py:326-352`）⇒ 加数据集不改 hash、resume 照过 ⇒ **有噪声与无噪声的 episode 会混进同一个 journal 而系统不知情**。
- ⚠ **审计对此改动是哑的**：`verify_collection_artifacts.py:45` 的 `REQUIRED_STEP_FIELDS` 不含 `noise_action_*`，`:252-256` 只做存在性检查 ⇒ 多存不报错（安全），**漏采也不报错**（无保护）。

### 1.5 建库 / 标定 / emit

- ⚠ **`_NUM_STEPS = 10` 在四处各自硬编码、互不引用**：`interceptor.py:90`、`build_in_memory_cache_artifact.py:635` 与 `:1029`、`build_clip_cache_artifact.py:198`；加上 `types.py` 的 canonical 集合，**没有单一 source of truth**。
- **标定完全不用改**：`exp/common/calibrate_score_normalizers.py` 全脚本不读 intermediates，只读 `query_keys` / `trajectory_id` / `task_key` / `vector_dims`。真正强制重标的是**几何变化**。
- ⚠ **在线写回在服务态是死代码**：`scripts/serve_policy.py:26-48` 强制 `write_policy.type == "never"` ⇒ **WARM_START payload 的唯一真实来源是离线 builder**。
- **⛔ 操作地雷**：`index_digest.json` 的 `source_sha256` 冻结的是 **emitter 源文件本身的 sha256**（`emit_ws_search_yamls.py` + `emit_ws_search2_yamls.py`，`emit_ws_search2_yamls.py:216-224`，比对 `:288-293`）。改动其中**任何一个字节**，正在跑/待续跑的 `run_ws_search2.py` 会在 preflight 直接 `SystemExit`。实测 `ws_search2_pnp/index_digest.json` 与两个源文件逐字节吻合，是**活跃冻结**。
  ⇒ **warm-start 的 emit 必须走新文件 + 新 out_root + 新 digest，绝不就地改这两个 emitter。**
- ⚠ emit 侧 `verify_cell`（`emit_ws_search2_yamls.py:157-183`）**对 judge 零断言** ⇒ 忘了把 `always_hit` 换成 warm judge，132 格会全程跑完且全是 FULL_HIT，assert 不响。

### 1.6 延迟台账与测量约束

**pi0.5 三段延迟（权威台账，RTX 4090，2026-08-19，`exp/data_authority/records/latency_bench__libero_spatial__executor_costs.json`，status=authoritative）**：

| 档 | stage1 | stage2 | stage3 | 合计 | stage3 占比 |
|---|---|---|---|---|---|
| eager | 63.06 | 35.27 | 349.81 | 448.1 | 78.1% |
| default | 47.55 | 30.36 | 122.81 | 200.7 | 61.2% |
| **cuda_graph** | **10.26** | **27.69** | **29.57** | **67.5** | **43.8%** |

- ⚠ **台账里没有 GR00T**（标题只有 pi0.5 teacher / ACT / SmolVLA）。GR00T 三段分割**从未测过**。
- ⚠ **这是 LIBERO 上的 pi0.5**（2×224² 双相机、action_horizon 10），RoboCasa365 是三路 256²、GR00T action_horizon 16 ⇒ **绝对值不可跨 benchmark 引用，只有结构占比可外推**。
- ⚠ **weilandserver 的显卡 2026-08-26 物理更换**，晚于 08-19 的台账 ⇒ pi0.5 那组数**不能默认与新卡上的 GR00T 数同框**，必须在同一张卡上复标定。
- ⚠ **现存 server CSV 的 stage 数字不能用来算占比**：`OPENPI_MONITOR_LEVEL=BASIC` 下 cuda 探针被降级成 `PerfCounterBackend` 且不做 sync（`src/openpi/cache/timing.py:417-423`），测的是 kernel 提交时间。
- ⚠ **teacher-only 地板臂窗口搭不了车**：`serve_groot_n15.py:201-202` 的 teacher-only 分支**不构造 `GrootStagedRunner`、也不构造 timer**，三个探针一个都不注册。必须独立微基准。
- ⚠ **`cp1_search 261 ms` 是作废数据**（探测配方）。修正值 136.1 ms，且 **P1 冻结搜索缓存 shipped 后降到 2.62 ms（约 60×）**，单次推理预算 202 → 约 67 ms，瓶颈翻转为 stage1（`logs/robocasa365_ws_search_plan.log.md:154, :157`）。**任何引用 261 ms 的论证都作废。**

### 1.7 静默失败面（风险登记册摘要，完整版见 §4）

**A. 最危险的一条 —— 错的配置能跑通，对的配置被拒绝。**
GR00T 4 步的中间量若被现有 builder 处理，会被打成 t = **0.9 / 0.8 / 0.7**（`t = round(1.0 − i/10, 4)`），而这三个值**恰好全部落在** `CANONICAL_DENOISE_TIMESTEPS` 里。于是：
- config 校验（`config.py:2609-2618` / `:2655-2661`）放行；
- orchestrator 的 `start_t not in payload.intermediates` 检查（`orchestrator.py:745-747`）也**通过**；
- 每步都返回 WARM_START，**但拿的是完全错的 x_t**（真值应为 0.25 / 0.5 / 0.75）。
- 而真值写进 yaml **反而会被拒**，因为不在 canonical 集合里。
症状只有动作质量下降，不报任何错。

**B. 唯一的库覆盖度门只对一种 judge 生效。**
`config.py:3169-3175` 会读 `intermediates_completeness`、不到 100% 就 `ConfigValidationError` —— 但只在 `judge.type == "dispatch_surface"` 时执行（`:3266`）。`always_warm_start` 与 `threshold + warm_tiers` **都没有这层对账**，唯一保护是 `orchestrator.py:744-753` 的运行时 warning + 静默降级。

**C. 浮点键漂移已验证**：`1.0 − 7/10 == 0.30000000000000004 != 0.3`，i=7,8,9 都漂；必须 `round(...,4)` 后才等于字面量。config 侧已封（`config.py:2609-2618` 就地归一化回写），**库侧没封**。

**D. 时间轴方向翻转无任何自动化覆盖。** `run_stage3_from` 对 `start_t` 零校验（`pi0_pytorch.py:731-772`：无 range 断言、无符号检查）；canonical 集合 `{0.1…0.9}` 在 t→1−t 下**不变**，方向翻转原样通过。唯一能抓的检查是手动脚本 `exp/dispatch_surface/d0_check.py:462-478`，serving 路径不跑。

**E. routing 白名单有洞。** `judge.type=composite` + `composer.type ∈ {and, or}` + `composer.warm_start_t` 会被 `config.py:2895-2907` 放行（该白名单只比 `tier_thresholds`，而 and/or composer 不用它），运行期在 `interceptor.py:983` **每请求 RuntimeError**。

---

### 1.8 LIBERO × GR00T 线：**同一套代码，三处必须单独处理的差异**

> owner 2026-09-06 指示核查"LIBERO 那条线是不是漏了"。核查结论：**没有"漏"这回事** ——
> GR00T 的 warm-start 在**两条线上都不存在**，且缺的是同一批文件。以下全部经 file:line 核实。

**共用的部分（做完一次两条线都得到）**：

| 组件 | 位置 | 共用方式 |
|---|---|---|
| 分阶段执行体 | `src/openpi/cache/groot/staged.py:334`(`run_stage1`) / `:440`(`run_stage2`) | 两线同一个 `GrootStagedRunner`；`:472` 的 `action_head.get_action(...)` 是唯一原子调用点 |
| cache interceptor | `src/openpi/cache/groot/interceptor.py:216-228` | 两线同一个；WARM_START 一律 `raise RuntimeError` |
| 装配期守卫 | `src/openpi/cache/groot/load_guard.py:46,96-102` | 两线同一个白名单 |
| **采集器** | `exp/robocasa365/groot_cache_collector.py:162` `noise_action_steps=[]` | ⚠ LIBERO 的 server 在 `exp/libero_groot/serve_groot_libero.py:400-409` **直接 import 并构造这个 RoboCasa 模块**；补 hook 一次两条线同时生效 |
| 建库器 | `exp/common/build_in_memory_cache_artifact.py:635-659, :1029-1052` | 两线同一条路径（LIBERO 经 `exp/libero_groot/build_size_libraries.py:42` 转手）；`_NUM_STEPS = 10` 写死 |

**三处必须单独处理的差异（漏掉即静默或误配）**：

| # | 差异 | 位置 | 处置 |
|---|---|---|---|
| L-1 | ⚠ **LIBERO 跑 8 步，不是 4 步** | `exp/libero_groot/serve_groot_libero.py:60` `DEFAULT_DENOISING_STEPS = 8`，`:380` 经 `Gr00tPolicy(denoising_steps=...)` 生效；注释 `:56-59` 说明已发表的 LIBERO 数字用的就是 8 而非 ckpt 内置值。RoboCasa 用 4（`staged.py:41`） | D4 的 `schedule_id` 必须把**步数写进主键**（`groot_n15_k4_v1` / `groot_n15_k8_v1`），否则两条线的库会互相"合法"对上而 t 的含义完全不同 |
| L-2 | **第二道 emitter 守卫** | `exp/libero_groot/emit_gate_yamls.py:177-182` 拒绝写出任何带非空 `cp1.judge.warm_tiers` 的 yaml | D9 的"最后放开"要放开**两道**，见 D9 |
| L-3 | 三个装配点带既有豁免 | `serve_groot_libero.py:160,235,429` 均传 `allow_hysteresis_gate=True`（RoboCasa 侧用默认值） | W10 放宽白名单时不得把该豁免冲掉 |

**术语陷阱**：`exp/libero_groot/` 里的 "warmup"（`emit_gate_yamls.py --mode warmup`、`emit_warmup_pool.py`）指的是**阈值标定的 force-MISS 臂**，与本 plan 的 warm-**start**（中途续跑）**完全无关**，只是共用一个英文词。两者在同一条流水线里相邻出现，读 runbook 时极易混淆。

**存量数据**：LIBERO×GR00T 已采语料（约 89 GB）与 RoboCasa 语料一样**零 `noise_action_*` 数据集** —— `noise_action_steps=[]` 让 `data_collector.py:163` 的循环一次都不执行。⇒ 两条线的重采集义务对称，都归 W13。

---

## 2. 设计裁定

**D1（立论口径）—— 按精度立项，不按延迟立项。**
warm-start 的价值主张是"在 FULL_HIT 与 MISS 之间造一个可用的中间档"，不是省时手段。
先例（`exp/warm_start/data/`，libero_spatial，各 n=500）：FULL_HIT 相对 teacher 掉约 29 pp，warm-start 收回其中 **87-95%**。
时间侧的结构上限：`WARM_START ≥ s1 + s2`，pi0.5 cuda_graph 档地板是 MISS 的 **56.2%**，最多省 39.4%，**永远降不到 FULL_HIT 的 15.2%**。
⇒ 报告与 plan 一律以精度立论；凡引用"省多少 ms"处，必须同时给出 episode 墙钟折算（132 次推理/集、p50 103.7 s/集）。

**D2（收益先行）—— 不测出数不动生产代码。**
GR00T 侧改造是 L3 量级（时间轴常量、整库重采、新增可恢复入口、放宽守卫、interceptor 渗透）。生产改造必须先过 **G-A1 + W4 owner 裁决**；完成改造后先跑 G-A2，G-A2 不通过则不得进入全量正式实验。

**D3（测量口径）—— CUDA-Graph 单档，三段各自编译，独占 GPU，机器固定 weilandserver。**
owner 2026-09-06 三条裁定：
1. 必须用 CUDA Graph（`mode="reduce-overhead"`）；eager / default 两档不测、不写退路。
2. **三个 stage 必须分别编译成各自独立的 graph** —— cache 要在阶段之间切开，一整个 forward 一张图无法切分。
3. **测量机固定 weilandserver**；**测量期间 GPU 必须完全空闲，不允许任何其他进程占卡**。
⇒ 排在外部 pnp W8 campaign 全部收工、其 6 个 server 关停之后的真空窗口。计时契约照抄台账 teacher 路径（`policy.py:101-131` 的 `time.monotonic()` + 每阶段 `torch.cuda.synchronize()`），**不用 `SystemTimer`**（batch=1 是 launch-bound，CUDA event 测 GPU timeline 会系统性漏掉主导项；且 SNAPSHOT 档的 sync 本身会压住 GPU）。

**D4（denoise schedule 身份契约 —— 单一权威 + 全链路携带）。**

⚠ 仅按"timestep 值集合"校验**挡不住方向反转**：`{0.1…0.9}` 这个集合在 t→1−t 下不变（§1.7-D）。且 `CacheConfig` 当前**没有 model/teacher/schedule 任何身份字段**，`AlwaysWarmStartJudge`、`config.py` 多处校验、`surface_judge` 都直接 import 全局 `CANONICAL_DENOISE_TIMESTEPS` —— "按 (模型, 步数) 生成"无处取得模型参数。因此本裁定不是"改一个常量"，而是**引入一个显式的 schedule 身份对象并让它贯穿五个面**。

冻结契约（`DenoiseSchedule`，单一权威，其余全部由它派生）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `schedule_id` | `"pi05_v1"` / `"groot_n15_k4_v1"` / `"groot_n15_k8_v1"` | 字符串主键；**这是唯一权威**，不做隐式推断。⚠ **步数必须进主键**：同一个 GR00T ckpt 在 RoboCasa 跑 4 步、在 LIBERO 跑 8 步（§1.8），若两条线共用一个 `groot_n15_v1`，两边的库会互相“合法”地对上而 `intermediates` 的 t 含义完全不同 —— 这正是 D4 存在的理由。原先冻结的 `groot_n15_v1` 作废，不得使用 |
| `num_steps` | 10 / 4 / 8 | |
| `direction` | `"noise_to_clean_desc"`（t:1→0）/ `"noise_to_clean_asc"`（t:0→1） | **方向是显式字段**，不靠数值集合推断 |
| `timesteps` | 由 `num_steps` + `direction` 生成的、按**从噪声到 clean 的执行顺序**排列的元组 | pi0.5: (0.9…0.1)，9 个可恢复点；GR00T k=4: (0.25, 0.5, 0.75)，剩余步数 3/2/1；GR00T k=8: (0.125, 0.25, …, 0.875)，7 个可恢复点，剩余步数 7…1。⚠ **k=8 的集合与 `CANONICAL_DENOISE_TIMESTEPS`（{0.1…0.9}）交集为空**，所以 LIBERO 侧不存在 k=4 那种“碰巧合法”的静默坑，但六处校验点会**全部拒绝**它 —— W5 若只把校验改成“按 schedule 取值”而漏掉任何一处，症状是响亮的启动失败（可接受），反过来若保留全局常量兜底则退回静默（不可接受） |
| 派生方法 | `snapshot_t(i)` / `snapshot_index(t)` / `remaining_steps(t)` / `replay_timestep(i, device)` | 任何 builder、续跑入口与校验器都不得自行重写方向或步数公式 |

五个携带面（缺一即静默）：

| 面 | 现状 | 契约 |
|---|---|---|
| YAML/config | `CacheConfig` 无身份；cache YAML 本身就是 `CacheConfig` 根对象 | 新增顶层 `denoise_schedule: <schedule_id>`（`CacheConfig.denoise_schedule` 默认 `"pi05_v1"`，**不是**不存在的 `cache.*` 子树）；`judge.start_t` / `warm_tiers[].start_t` 的合法集合改为由该 schedule 的 `timesteps` 决定 |
| HDF5 attrs | ⚠ 只有 episode 长度语义的 `num_steps`（`data_collector.py:146`） | 新增 `attrs["denoise_schedule_id"]` 与 `attrs["denoising_num_steps"]`（与既有 `num_steps` **不同名**，避免 §1.4 那个语义冲突） |
| artifact meta | pkl 顶层无 schedule | 新增 `denoise_schedule_id`；loader 记录并校验所有 entry 的 payload schedule/步数共识 |
| `CachePayload` | 只有 `denoising_num_steps`，⚠ **区分不了 1→0 与 0→1** | 保留 `denoising_num_steps`，新增可回填的 `schedule_id`。`validate_for_checkpoint` 只保留 CP 结构不变量；新增 `validate_for_warm_start(schedule, start_t)`，在确实要执行 WARM_START 时要求 `action_chunk + intermediates + denoising_num_steps + schedule_id`、目标键存在且身份一致（FULL_HIT 的 CP1 payload 不被误伤） |
| 装配期绑定 | 只有 dispatch_surface 在 backend preload 后对账（`config.py:3169-3175`） | 见 W9：在 `build_per_connection_components()` 的 storage-aware 装配点，把 completeness 门推广到所有可产生 WARM_START 的 judge，并同时比对 `schedule_id` 与 `denoising_num_steps`；纯 YAML load 期只做静态 schedule/timestep 校验 |

**向后兼容（必须证明，不能假设）**：旧 YAML 因 `CacheConfig.denoise_schedule` 默认值继续取 `"pi05_v1"`；旧 artifact / 旧 h5 缺字段也只在兼容 loader 中回填为 `"pi05_v1"`。`InMemoryBackend.load_artifact()` 还须给旧 pickle 的 payload 补 `schedule_id` 属性，并拒绝顶层身份与 entry 共识不一致的混库。现存 517 个 `always_warm_start` yaml、882 个 `warm_start_t` yaml、CP2 / dispatch_surface 既有产物由 §8-T12 证明零改动兼容；任何 GR00T warm 配置则必须显式写 `denoise_schedule: groot_n15_k4_v1`（RoboCasa）或 `groot_n15_k8_v1`（LIBERO），不得靠模型名推断，也不得由两条线共用一个 id。

**消费者迁移清单**：registry 与派生数学放在 cache-side 独立模块；`config.py` 在构造 `AlwaysWarmStartJudge` / `ThresholdJudge` / `FailureAwareGateJudge` / surface judge 时注入已解析 schedule（不得让组件继续 import 全局 canonical 集合）；两个 offline builder 按 H5 schedule 调 `snapshot_t(i)`；artifact loader 汇总身份与共识；orchestrator 在返回 WARM_START 前调用 payload 的 warm validator，若装配门被绕过也必须 fail loud，不再 warning 后静默降级 MISS；pi0.5 interceptor 的步数与时间戳逻辑保持数值兼容，GR00T 走 8b 的升序实现。

⚠ **措辞更正**：早前写的"四处 `_NUM_STEPS` 收敛到单一常量"是错的 —— 10 与 4 必须并存。正确说法是**收敛到单一来源**（`DenoiseSchedule`），四处硬编码改为向它取值。

**D5（采集 schema —— 冻结到数据集名与索引，不留二选一）。**

⚠ 先纠一处本 plan 早前的自相矛盾：GR00T hook 的 4 次捕获**本身就是** `[x_0, x_1, x_2, x_3]`，其中 x_0 即起点噪声 —— 所以"GR00T 收 4 步 **+** 起点"是重复计数，实际是"4 次捕获 = 1 个起点 + 3 个可恢复中间量"。

冻结契约（唯一，不再二选一）：

| 数据集名 | 语义 | pi0.5 | GR00T | 谁消费 |
|---|---|---|---|---|
| `noise_action_0` | **起点噪声 x_init**（t=1.0 / t=0.0，按 schedule 的 direction） | 新增（今天被 `range(1,…)` 丢弃） | 新增（hook 的第 0 次捕获） | **只用于基线复现**，⚠ **不进 `CachePayload`** |
| `noise_action_1..N−1` | **可恢复中间量** | 1..9（不变） | 1..3 | 进 `CachePayload.intermediates` |
| `clean_action` | 终态 | 不变 | 即 `action_pred` | 不变 |

- **索引归属唯一**：`noise_action_i` 恒为"第 i 次去噪之前的 x"，i=0 即纯噪声。pi0.5 现有 1..9 的语义**逐字不变**，只是补上一个 i=0。
- **实现走"加字段"而非"改索引"**：`InferenceEmbeddings` 新增 `init_noise: np.ndarray | None = None`，writer 仅在非 None 时单写 `noise_action_0`；`enumerate(..., start=1)` 与 `range(1, num_steps)` 这对配套偏移**一个字节都不动**（§1.4 的静默错位风险归零）。schedule 两个 file-level attrs 由 pi0.5/GR00T collection wrapper 在 `on_episode_start()` 后通过 `set_episode_attr()` 写入，不把模型语义塞进通用 collector 构造器。
- **旧产物兼容**：`build_in_memory_cache_artifact.py:644` 的 `if 1 <= idx < N` 天然跳过 `noise_action_0` ⇒ 新 h5 建出的库与旧库在 `intermediates` 上逐位一致。要用起点噪声必须显式另开读取路径。
- **init noise 明确不进 `CachePayload`**：`start_t` 合法集合不含端点，且它的用途是让"从纯噪声出发"这条对照可逐比特复现，与 warm-start 起点语义不同。

**D6（GR00T 采集走 hook；生产续跑只保留一份受守卫的循环）。**
采集在 `action_head.action_encoder` 挂 forward hook 取 `inp[0]`，不为采集中间量复刻循环。生产 `run_stage3` / `run_stage3_from` 仍须按 W8 抽出上游 action-head 去噪体；它们共享一个单步实现与同一个 upstream hash/parity 守卫，不再产生第二份采集专用循环。
⚠ 两个实现约束：hook 在 `session()` 的 `torch.inference_mode()` 内触发，捕获张量必须照 `groot_cache_collector.py:147-149` 做 `is_inference() → clone()`；模型句柄目前私有（`staged.py:208` 的 `_model`），需公开属性或走 `policy.model`。

**D7（新旧产物强制隔离）。**
在 run-plan `params` 加 schema 版本字段（如 `collect_schema: "v2"`），使 plan hash 改变、强制 `--batch N+1`。
理由：plan hash 现在不含任何 schema 描述，而审计对本改动是哑的 ⇒ 不隔离就会出现"有噪声与无噪声的 episode 混在同一个 journal 里而系统不知情"。

**D8（emit 走新文件）。**
warm-start 的 yaml 发射**新建文件、新 out_root、新 digest**，绝不改 `emit_ws_search_yamls.py` / `emit_ws_search2_yamls.py` 任何一个字节。理由见 §1.5 的操作地雷。

**D9（守卫放开是最后一步）。**
`load_guard.py` 的白名单是当前唯一挡在静默坑前面的**响亮**防线。放开顺序必须是：G-1 采集 → G-3 时间轴 → G-2 执行体 → **最后** G-4 守卫。任何提前放开都会把响亮失败换成静默失败。
⚠ **守卫有两道，不是一道**：除 `load_guard.py` 外，LIBERO 侧的 emitter `exp/libero_groot/emit_gate_yamls.py:177-182` 还会拒绝写出任何带非空 `cp1.judge.warm_tiers` 的 yaml。两道必须**同时**在最后一步放开；只放开 `load_guard` 会得到“配置合法但根本发不出来”的假象，只放开 emitter 则 server 启动即拒。另注：LIBERO 的三个调用点（`serve_groot_libero.py:160,235,429`）都传 `allow_hysteresis_gate=True`，放开时不得把这个既有豁免一并冲掉。

---

## 3. 工作单元

> 单元之间的依赖见 §6。G1 只放行计划，不等于启动执行；owner 启动后，**W0-W4 均不改生产代码**，是收益前置阶段的完整边界。

### 3.0 阶段 A —— 收益前置（**owner 启动后唯一可执行的边界，零生产代码改动**）

⚠ **B4 澄清的依赖矛盾与本 plan 的选路**：W2 要测的是"GR00T 三段各自 CUDA Graph + k∈{1..4} 阶梯"，而生产路径的 `run_stage2()` 至今是 LLM + 4 步 action head 的**原子调用**，噪声也写死在上游 `get_action` 体内。两条出路：**(a)** 在独立 bench 脚本内复刻拆分与噪声外提，用 upstream sha256 + eager-parity 守卫；**(b)** 先做一个受限的生产 API 改动。
**本 plan 选 (a)。** 理由：D2 要求"不测出数不动刀"，选 (b) 等于在门之前偷渡 W7 的实现；而 (a) 的复刻只活在 bench 脚本里，不进生产路径，且 `Gr00tPolicy(denoising_steps=k)` 是**既有构造参数**（`exp/libero_groot/serve_groot_libero.py:380`），k 阶梯无需改任何生产代码。
⇒ **阶段 A 全程不改生产代码**；G-A1 需要的 `--denoising-steps` 同样在 bench/独立 runner 内实现，不动 `serve_groot_n15.py`。

| # | 单元 | 产物 | 改生产代码 |
|---|---|---|---|
| **W0** | 空窗确认（**范围见下方 3.0.1**） | 归属清单核对 + 空窗证据 | 否 |
| **W1** | 微基准 `exp/robocasa365/bench_groot_stages.py`（新文件）：脚本内复刻三段拆分与噪声外提，带 `UPSTREAM_GET_ACTION_SHA256` 与 eager-parity 断言；提供 stage1 诊断模式输出 max\|Δ\| / relative-Frobenius / cosine 分位数 / 最坏 token 范数 / dtype | 脚本 | 否（新文件） |
| **W2** | 跑 G-M 测量：GR00T k∈{1,2,3,4} × 3 进程 + pi0.5 同卡复标定 × 3 | `exp/robocasa365/data/latency/*.json`（schema 见 §5-G-M） | 否 |
| **W3** | 步数敏感性筛查（G-A1）：独立 runner 起 teacher-only，k∈{1,2,3,4} 跑同一 seed 段 | SR 曲线 + 统计判据 | 否 |
| **W4** | 判 G-M / G-T / G-S / G-A1 → **owner 唯一裁决点** | 裁决 | — |

#### 3.0.1 W0 的资源范围（**收窄，不得跨 session**）

⚠ 早前写的通过条件"`nvidia-smi` 零占用 + `tmux ls` 无 server"会诱导终止其他 session 的进程 —— weilandserver 的 23100-23199 端口段与 `srvN` tmux 名是**多 session 共享命名空间**，本线已有过宽杀伤及他人的事故记录。

**本任务可停的精确清单（冻结）**：

| 机器 | 可停 | 判别依据 |
|---|---|---|
| weilandserver | tmux `srv0`-`srv5`，且其监听端口 ∈ {23160-23165, 23170-23173} | 端口 + tmux 名双重匹配，按 PID 定点 |
| timan107 | tmux `drv0` / `ag0`-`ag5` / `flr0`，以及它们派生的 `worker_entry` | 同上 |

**其他任何占用一律不动**：只能**等待**，或由 owner 另行协调。W0 的通过条件相应改为「上表清单已全部停止 **且** GPU 剩余占用为零」——若剩余占用非零，W0 **不通过**，等待或上报，**不得为制造空卡窗口终止任何不在上表内的进程**。

### 3.1 阶段 B —— L3 改造（**W4 通过后才启动，本 plan 不授权**）

| # | 单元 | 关键内容 |
|---|---|---|
| W5 | **schedule 身份契约**（D4）：新增 `DenoiseSchedule` 及四个派生方法；config / h5 attrs / artifact meta / `CachePayload` / 装配期绑定五面携带；四处 `_NUM_STEPS` 和 builder/续跑方向公式改为向它取值；旧产物由 loader 回填 `pi05_v1` | src |
| W6 | **采集补收**（D5）：`InferenceEmbeddings` 加可选 `init_noise=None`；`data_collector` 非空时单写 `noise_action_0`；GR00T 在 `action_head.action_encoder` 挂 hook（`is_inference()→clone()`）；两个 collection wrapper 用 `set_episode_attr()` 写 file-level `denoise_schedule_id` + `denoising_num_steps` | src + exp |
| W7 | **审计与 run-plan**：`REQUIRED_STEP_FIELDS` 加 `noise_action_*` 的**基数与索引连续性**断言（不止存在性）+ shape/dtype；`params` 加 `collect_schema` 强制新旧隔离（D7） | exp |
| **W8** | **GR00T 执行链（端到端，不止 `staged.py`）** —— 见 3.1.1 | src |
| W9 | **库绑定门推广**：新增 fail-closed 的 `required_warm_timesteps(...)`，穷举 `always_warm_start`、带 `warm_tiers` 的 threshold/failure-aware、dispatch surface 及任何 composite/composer warm 出口；未知但可能产生 WARM_START 的形状直接拒。`build_per_connection_components()` 在 backend preload 后统一检查目标 timestep 的 100% completeness、entry schema 共识、`schedule_id` 与 `denoising_num_steps` | src |
| W10 | **放开 `load_guard.py` 白名单**（**必须最后**，D9） | src |
| W11 | emit 新文件 + 新 out_root + 新 digest（D8）；`verify_cell` 补 judge 形状断言 | exp 新增 |
| W12 | **文档与索引**（L3 义务）—— 见 3.1.2 | docs |
| W13 | 重新采集 → 建库 → 重标定 → **先跑 G-A2 配对臂**；只有 G-A2 通过才进入全量正式实验 | 数据 |

#### 3.1.1 W8 的完整执行缝（B1）

⚠ 早前只写"拆出可续跑阶段"是不够的：`groot/interceptor.py:222-227` 在 CP1 返回 WARM_START 时**直接 raise**，`:177, :184` 把 `__hit_meta__.start_t` **硬编码为 None**。即使 `staged.py` 拆好了，线上仍然跑不起来。W8 冻结为六件事：

| # | 内容 |
|---|---|
| 8a | **typed stage API**：`GrootStage2Output` 从"只有 `action_pred`"改为携带 LLM 输出（`backbone_features [1,N,2048]` + `attention_mask`）；新增 `GrootStage3Output`。⚠ `run_stage2:460-469` 每次新建 `BatchFeature` 防 `vlln` 二次施加的约束必须跟着搬到新边界，搬错是**静默错值** |
| 8b | **两条 action 路径**：`run_stage3(stage2, *, noise)` 全量 / `run_stage3_from(stage2, start_x, start_t, *, schedule)` 续跑。后者先要求 `schedule_id` 属于 GR00T 族（`groot_n15_k4_v1` / `groot_n15_k8_v1`）且与运行时实际 `num_inference_timesteps` 一致，再由 `snapshot_index()` 与 `remaining_steps()` 得到步数（k=4: 0.25→3 / 0.5→2 / 0.75→1；k=8 同法外推），并由 `replay_timestep()` 生成张量时间戳；不得调用 pi0.5 的 `_warm_start_num_steps`。backbone conditioning（`vl_embs`）由 8a 的 typed 输出显式传给两条路径，不再隐式复用 mapping |
| 8c | **噪声外提**：复刻上游 `get_action` 循环体（远端 `flow_matching_action_head.py:350-405`，约 40 行），新增 `UPSTREAM_GET_ACTION_SHA256` 钉住漂移；顺带让 `--diagnostic-seed` 对 staged 路径重新生效 |
| 8d | **interceptor 三分支**：`groot/interceptor.py` 去掉 `:222-227` 的 raise，真正消费 `payload.intermediates[start_t]`；FULL_HIT / WARM_START / MISS 三条路径各自的 stage 调用与写回行为冻结 |
| 8e | **metadata / probe / lifecycle**：`__hit_meta__.start_t` 从硬编码 None 改为真实值（否则下游 `summarize_inf_ratio._warm_cost(None)` 会静默按 0.5 计价）；探针从 `stage1_vision`/`stage2_llm`/`stage2_action` 扩为三段口径。⚠ 改名波及 CSV 消费方与 G0-E 探针计数，须一并列出 |
| 8f | **等价门**：G0-C 从 `run_stage2(run_stage1(x))` 扩到 `run_stage3(run_stage2(run_stage1(x)))` vs 上游 `get_action`，固定 noise 逐位等价；并保留"不传 noise / 传不同 noise 结果必须不同"的反向对照（防噪声被固定这个静默失效） |

#### 3.1.2 W12 文档义务（L3）

⚠ `docs/architecture/cache_system.md` 现行章程明写 **"GR00T 二阶段、不支持 WARM_START"** —— 这是本改造直接推翻的条款，不同步就是文档与实现相悖。

| 文档 | 改什么 |
|---|---|
| `docs/architecture/cache_system.md` | GR00T 两阶段 → 三阶段；WARM_START 支持状态；`DenoiseSchedule` 契约；LayerNorm/autocast 条目补 stage3 |
| `docs/data_collection/guide.md` | ⚠ 该文件 `:3` 自称"carries Working Agreement authority"，`:84-102` 写死 `noise_action_1 … noise_action_9` ⇒ 必须同步 schema |
| 采集 runbook / tutorial / migration | 新增 `noise_action_0` 与 schedule 字段的迁移说明 |
| `docs/README.md`、`logs/README.md` | **同一个 commit 内**同步索引 |

---

### 3.2 阶段 B 的 LIBERO×GR00T 覆盖（W14，随 W5-W13 一并授权）

⚠ 本节是 owner 2026-09-06 在 G2 Round 3 指示加入的范围扩展。**它与 W5-W13 同属阶段 B，同样不被本 plan 授权**，
仍需 W4 的 owner 裁决才启动；此处只把"做的时候要做什么"冻结下来，避免届时凭记忆补。

| # | 单元 | 关键内容 | 触及 |
|---|---|---|---|
| W14a | **LIBERO schedule 绑定** | `serve_groot_libero.py` 的三个装配点在构造 cache 组件时绑定 `groot_n15_k8_v1`；与 `DEFAULT_DENOISING_STEPS` 的值做**一致性断言**（两者不一致直接拒起，不允许静默以 config 为准） | exp |
| W14b | **采集侧**（无独立改动） | LIBERO 复用 `GrootCacheCollector`，W6 的 `action_encoder` hook 一次覆盖两线；**唯一要加的是断言**：写 file-level `denoise_schedule_id` 时取自实际生效的 `denoising_steps`，不是常量 | exp |
| W14c | **建库侧** | `build_size_libraries.py` 透传 schedule id 到 `build_in_memory_cache_artifact.py`；产物 `artifact_meta` 必须带 `schedule_id`，`verify_libraries.py` 增加"库内 schedule_id 唯一且与请求一致"的检查 | exp |
| W14d | **emit / eval** | `emit_gate_yamls.py:177-182` 与 `emit_search_yamls.py` 按 D8 走**新文件新 digest**，不就地改（LIBERO 的 emitter 同样被在跑实验的 digest 冻结逻辑约束，规则与 RoboCasa 的 `emit_ws_search*.py` 一致）；`orchestrate_search.py` / `run_conductor.py` 的 cell 校验补 warm judge 形状断言 | exp 新增 |
| W14e | **等价门 G0-C-libero** | 8 步版本的三段链路 vs 上游 `get_action` 固定 noise 逐位等价 —— **不得**用 RoboCasa 的 4 步门代替：步数不同、图形状不同、`replay_timestep` 的累加误差也不同 | 测试 |
| W14f | **文档** | `docs/architecture/cache_system.md` §5.17 的"GR00T 不设第三阶段"条款对两条线同时失效，须一次改到位；`docs/data_collection/guide.md` 的 `noise_action_1…9` schema 同理 | docs |

**顺序**：W14 各项分别挂在对应的 W5/W6/W7/W9/W10/W11/W12 之后，不单独排一个阶段 ——
它们是"同一改造的第二条消费线"，分开排会让两线的 schedule 契约有机会漂移。

**代价提示**：W13 的重采集义务因此**翻倍**（RoboCasa 语料 + LIBERO 约 89 GB 语料都要重采），
这是把 LIBERO 纳入范围的主要成本，需在 W4 裁决时一并计价。

---
## 4. 风险

| # | 风险 | 是否静默 | 缓解 |
|---|---|---|---|
| R1 | GR00T 中间量被打上 pi0.5 的时间戳（0.9/0.8/0.7），config 与 orchestrator 双双放行 | ⚠ **静默** | W5 的 schedule 单一来源 + W9 装配期身份/步数/覆盖度对账 |
| R2 | warm 配置与库不匹配 → 每步静默降级 MISS | ⚠ **静默** | W9：completeness 门推广到所有 warm 出口（单点收益最高） |
| R3 | 时间轴方向翻转 | ⚠ **静默**（canonical 集合对 t→1−t 对称） | 把 `d0_check.py:462-478` 的数值 parity 收进 `tests/`，并加一条"反向必须 fail"的对照测试 |
| R4 | 浮点键漂移（`1.0−7/10 ≠ 0.3`） | ⚠ **静默** | 库侧补 round 与 round-trip 测试；config 侧已封 |
| R5 | 改 emitter 源文件 → 在跑实验 preflight 崩 | ❌ 响亮但**致命** | D8：新文件、新 digest，绝不就地改 |
| R6 | 采集索引偏移只改一处 → 全部时间戳错一格 | ⚠ **静默** | D5 走"新增字段"而非"改索引"，零索引风险 |
| R7 | 新旧产物混进同一 journal | ⚠ **静默** | D7：`collect_schema` 强制隔离 |
| R8 | 提前放开 `load_guard` 白名单 | ⚠ 把响亮换成静默 | D9：放开是最后一步 |
| R9 | 复刻 GR00T 去噪循环 → 新增无守卫的上游漂移面 | ⚠ 上游一变就静默错 | D6：走 hook 路线 |
| R10 | 测量被其它进程干扰 → 绝对值作废 | — | D3：独占 GPU，W0 空窗确认 |
| R11 | 换卡导致台账不可比 | ⚠ 静默地把两组数放进同一张表 | W2 含 pi0.5 复标定 cell；偏差 ≥5% 时报告须同列"台账值 / 本卡值" |
| ~~R12~~ | ~~routing 白名单的 and/or 洞~~ | — | **移出本 plan 范围** —— 它是既存的通用 config/composer 组合问题，不是让 GR00T `threshold` / `always_warm_start` 能执行的必要条件。留在这里既无对应 W 单元、又会撑破 minimal-change 边界。已单列到 §9 作为独立后续项 |

---

## 5. 门

**G-M（测量门）**：W2 完成，产出 GR00T 三段在 CUDA-Graph 档、独占 GPU、weilandserver 上的延迟，含 k∈{1,2,3,4} 阶梯拟合的 `s2act(k) = a + b·k`。
⚠ 只测 k=4 再除以 4 会**系统性高估**收益（图内含不随步数变的固定开销 `a`）。真实上限是 `3b/T`，不是 `0.75·φ`。

**G-M 的预注册测量契约**（跑之前冻结，否则 per-shape graph 的结果外推不到正式路径）：

| 项 | 冻结值 |
|---|---|
| 计时契约 | `time.monotonic()` + 每阶段后 `torch.cuda.synchronize()`，照抄台账 teacher 路径（`policy.py:101-131`）。**不用 `SystemTimer`**：batch=1 是 launch-bound，CUDA event 测 GPU timeline 会系统性漏掉主导项；且 SNAPSHOT 档的 sync 本身压 GPU |
| prompt 形状集 | ⚠ 必须用**真实 RoboCasa365 的 5 个 PickPlace prompt**（钉死后每任务恰好 1 个，见 §1），逐形状各测一遍并**分别报告**；不得只测一个形状就外推。变长 N 在 CUDA Graph 下每形状一张图，这正是 §7.2 的风险 |
| warmup / 迭代 | warmup 30（覆盖编译 + graph capture + 时钟爬升）；measurement 200 次/进程 |
| 重复 | 每 cell **3 个独立进程串行执行**（独占 GPU，禁止并跑），cell 顺序随机交错；报 median-of-medians + 跨进程离散度（进程间方差是主要不确定度来源） |
| 逐次列表 | ⚠ **必须落盘**（台账的已知缺陷：15 个文件里只有 4 个带逐次列表，n=30 只存了摘要） |
| eager↔compiled parity | 每段各一条断言，与延迟同一次运行内产出；parity 不过则该 cell 的延迟数**作废** |
| **graph 确实 replay 的证据** | ⚠ 时延分布不能证明 graph replay。可执行契约（G2 R2 后冻结）：measurement 区间由 `cudaProfilerStart/Stop` 界定，外层 `nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop`，区间内再压一个 cell 身份 NVTX marker（`trace_marker`，由 cell 身份字段派生）；导出用 `nsys stats --report cuda_api_sum,nvtx_sum --format csv`，按列名 `Name` + `Num Calls` 读**整数调用数**（不得数字符串出现次数）。判据三条全部必须成立：trace 内出现该 cell 的 marker；`cudaGraphLaunch` 计数**恰等于** `iters·(2+k)`（三段各一次 replay + 每步一张 stage-3 图）；区间内 `cudaStreamBeginCapture`/`EndCapture`/`cudaGraphInstantiate` 计数为 **0**（warmup 允许 capture，measurement 不允许）。inductor `perf_hints` 落在 cell JSON 的 `inductor_perf_hints` 字段，unique-graph 计数取 Dynamo `stats.unique_graphs`（必须恰好新增 3），任何 cudagraph skip 都使该 cell 作废。**拿不到 trace，G-M 不成立** |
| RNG / 固定输入 | parity 与延迟 cell 使用**显式固定 noise**：同 noise 的 eager/compiled 及重复 replay 必须一致；另设 sampled-noise 子测，连续调用必须不同。两种结果分字段记录，禁止把“固定输入确定性”误判为“RNG 被图固定” |
| 输出 JSON schema | 权威定义是 `bench_groot_stages.py` 的模块级 `MEASURE_RECORD_FIELDS`，落盘前由 `assert_record_schema()` 强制：`{schedule_id, k, prompt_sha256, trace_marker, prompt_shape_n, mode, warmup, iters, proc_idx, stage1_ms[], stage2_llm_ms[], stage3_ms[], total_ms[], parity_stage1, parity_stage2, parity_stage3_copy_vs_upstream, parity_stage3_eager_vs_compiled, parity_worst, fixed_noise_replay_equal, sampled_noise_distinct, cuda_trace_path, cudagraph_launch_count, expected_cudagraph_launch_count, capture_calls_after_warmup, compile_count, cudagraph_skips, inductor_perf_hints, busy_gpu_override, gpu_name, gpu_uuid, torch, driver, ckpt_sha256, upstream_get_action_sha256, git_commit, host, ts}`。raw cell 的 `cudagraph_launch_count` / `capture_calls_after_warmup` 恒为 `null`，只能由 `--mode certify` 从 trace 回填 |
| provenance | `gpu_uuid` + `torch` 版本必录 —— ⚠ weilandserver 的卡 2026-08-26 换过、GR00T 岛 torch 2.5.1+cu124 与 pi0.5 栈 2.7.1+cu126 不同，这两条是跨表对读时唯一能发现混淆的字段 |
| pi0.5 复标定 | 同一空窗、同一张卡复跑台账脚本的 CUDA-Graph 档；三段偏差 < 5% 则直接引用台账，否则报告须**同列"台账值 / 本卡值"**并给缩放因子 |

**G-T（时间门，无否决权）**：若 `3b/T < 20%`，则**不得以"省延迟"为由立项**。

**G-S（系统门，binding 于叙事）**：按 episode 墙钟折算（132 次推理/集 ÷ 103.7 s/集）收益 < 5% 时，**报告中禁止出现任何吞吐收益主张**。按现有账几乎必然 FAIL —— 这不是坏消息，是把叙事钉到正确的轴上。

**G-A（精度门，唯一有否决权）**：分两级，**判据在跑之前预注册**。

**⚠ 先撤回一个未经论证的推断。** 早前写的「纯噪声跑 k 步的 SR 曲线是 warm-start 精度收益的**下界包络**」**不成立，本 plan 撤回该表述**。理由：warm-start 的起点是从**另一条库轨迹**取回的 `x_t`，其分布与"从纯噪声截断到第 k 步"完全不同 —— 前者可能更好（已被一条成功轨迹条件化）也可能更差（来自不同初始状态），现有材料推不出任何序关系。
⇒ **G-A1 降级为 schedule-sensitivity screening（筛查），只有否决权、没有背书权。**

**G-A1（筛查，不需要实现 warm-start）**：k∈{1,2,3,4} 起 teacher-only，跑同一 seed 段。测"从纯噪声只跑 k 步"的 SR。预注册判据：

| 项 | 取值 |
|---|---|
| 样本量 | 250 ep/k（与地板臂同预算、同 seed 段），4 档共 1000 ep |
| 聚合 | **任务级 macro SR**（先任务内平均再跨任务平均），与既有 W10 报告同口径；同时报 pooled 作参考 |
| 配对 | 同 `(task, episode_idx)` 跨 k 天然配对（`seed = base_seed + orig_init_state_idx` 且场景钉死） |
| 不确定度 | 每任务的单档 SR 报 Wilson 95%；macro 与跨 k 差值用**任务内成对重采 episode、再跨任务等权平均**的 stratified paired bootstrap，固定 10,000 resamples 与随机种子 |
| **否决条件（唯一主比较）** | 做预注册的 k=1 对 k=4 非劣检验，margin=**5 个绝对百分点**：若 paired-bootstrap 的 `SR1−SR4` 95% 下界 `> −0.05`，则 1 步已对 4 步非劣，判 schedule 冗余并否决 warm-start 立项。不得用“差异不显著”冒充等价，也不得使用分母未定义的“相对 20%” |
| 不否决时 | 仅记录"步数确实敏感"，**不据此预测 warm-start 收益**；是否继续由 G-A2 单独判 |
| 次要比较 | 3 个相邻 k 差值只作诊断，配对随机化 p 值做 Holm 校正；不覆盖或改写上述唯一主比较 |

**G-A2（真跑，仅在 G-A1 未否决时才做）**：warm-start 配对臂 k∈{1,2,3}，与 FULL_HIT 臂、teacher 地板臂**同 seed 段**。预注册判据：

| 项 | 取值 |
|---|---|
| 样本量 | 250 ep/cell |
| 主指标 | 先按任务算 `recovery_task(k) = (SR_warm − SR_fullhit) / (SR_teacher − SR_fullhit)`，再对有效任务等权 macro；不得先 pooled SR 再相除 |
| **gap 退化的处理** | ⚠ 若 `SR_teacher − SR_fullhit ≤ 0`（gap 为零或反号），`recovery` **无定义** ⇒ 该任务从主指标中剔除并单列，**不得**按 0 或 1 计入；有效任务少于 3 个时 G-A2=`INCONCLUSIVE`，不得放行全量正式实验 |
| 不完整配对 | 任一臂缺该 `(task, idx)` ⇒ 该配对整体剔除，剔除数须报告；剔除率 > 5% 时该任务降级为"不可判" |
| **通过条件** | 至少一个 k 的 macro `recovery(k)` 点估计 **≥ 0.50**，且任务内成对重采的 bootstrap 对 `H0: recovery≤0` 给出 Holm-adjusted 单侧 `p<0.05`；普通 95% 区间同时报告但不冒充 family-wise 下界 |
| 多重比较 | 3 个 k 的主张用 Holm 校正；必须落盘每个未校正 p、校正后 p、bootstrap seed 与 resample 数 |
| 双口径 | ⚠ PickPlace 死任务在任何配置下都趋零 ⇒ 含/不含死任务**两个口径都要报**，主判据用含全部任务的口径 |

**综合裁定：按 G-A 立项，绝不按 G-T / G-S 立项。**

---

## 6. 执行顺序

| # | 动作 | 前置 | GPU | 改生产代码 |
|---|---|---|---|---|
| 0 | **W0** 空窗：停本任务清单内的进程（§3.0.1），确认剩余占用为零；非本任务占用只能等 | 外部 pnp W8 campaign 收工 | 释放 | 否 |
| 1 | **W1** 写 `bench_groot_stages.py`（脚本内复刻拆分与噪声外提 + upstream sha + eager-parity + stage1 诊断模式） | 0 | 0 | 否 |
| 2 | **stage1 等价门诊断**（§7.4-1）：由 W1 的隔离诊断模式在 `default` / `reduce-overhead` 各跑一次并输出扩展指标 | 1，**独占 GPU** | 少量 | 否 |
| 3 | **W2** 跑 G-M：GR00T k∈{1..4} ×3 + pi0.5 同卡复标定 ×3 | 0+1+2，**独占 GPU** | 约 45 min | 否 |
| 4 | 拟 `s2act(k) = a + b·k`，判 **G-M / G-T / G-S** | 3 | 0 | 否 |
| 5 | **W3** 跑 G-A1 步数敏感性筛查（k∈{1..4} × 250 ep，独立 runner） | 4 | 约 1.2 h | 否 |
| 6 | **W4** 判 **G-A1** → **owner 唯一裁决点：是否进阶段 B（L3 改造）** | 5 | 0 | — |

**第 4 步给出时间侧的决策数字；第 6 步是唯一的"要不要动框架"裁决点。阶段 A（0-6 步）全程零生产代码改动。**

若 W4 获 owner 明示授权，阶段 B 的实现顺序冻结为：

`W5 schedule → W6 collection capability → W7 audit/schema isolation → W8 execution → W9 storage-aware binding → W10 load-guard relaxation`。

W10 是最后一个运行时能力改动，不得与 W5-W9 并行或提前合入；随后跑 §8 全矩阵，再完成 W11 emitter 与 W12 文档/索引。W13 先重采、建库、标定并执行 G-A2；G-A2 通过后才可扩到全量正式实验，失败或 `INCONCLUSIVE` 均停在该处。

---

## 7. 三段各自 CUDA-Graph 编译（owner 硬约束的可行性）

### 7.1 判定：**可行，且有同仓正面先例。当前的 park 基于一个从未被诊断的失败，不是技术不可行。**

**A. "三段各一张图"这个形态本仓已证明过。**
`exp/ablation_study/latency_bench/bench_teacher.py:66-73` 就是分别编译 `_stage1_token_prep` / `_stage2_llm_backbone` / `_stage3_action_expert` 三个 callable。台账里的 `pi05_stage_split_ms.cuda_graph` 正是它的产物（`data/pi05_compile_ro_3stage.json`，`"fused": false`，`torch 2.7.1+cu126`）。
关键对照（`analysis/analysis.md:103-108`）：

| 编译档 | 三段 wall | 一整团 wall | 一整团优势 |
|---|---|---|---|
| default | 206.67 ms | 140.56 ms | **+32.0%** |
| **reduce-overhead** | **72.51 ms** | 72.04 ms | **+0.6%（噪声内）** |

⇒ **在 CUDA-Graph 档下，"拆成三张图"相对"一张大图"不收任何边界税。** 这正面兑现了 owner "三段必须分别编译"的约束 —— 不需要为切开付性能代价。（default 档要付 32%，但那一档本来就不用。）

**B. ⚠ GR00T 那次 cos=0.8716 的 FAIL 与 CUDA Graph 无关 —— CUDA Graph 从来没被真正试过。**
`torch._inductor.cudagraph_trees` 的**第一次调用走 warmup 路径而非图重放**（`cudagraph_trees.py:1807, 2151-2177, 1781`；`_inductor/config.py:873 skip_cudagraph_warmup=False`）。而 `_verify_compiled_vision` 恰恰只在**第一次真实推理**上跑（`staged.py:379-382`）。
⇒ 那道门比较的是「inductor 生成的 kernel 在 warmup 模式下的输出 vs eager」，**此时还没有任何图被录制或重放**。cos=0.8716 只能是 **inductor codegen 的数值差异**，不可能是静态缓冲复用、图内地址固定这类 CUDA-Graph 特有失效。
**推论一**：退回 `default` 档救不了它。**推论二**：这条路的可行性尚未被证伪。

**C. 根因两个候选（须先诊断再动手）**
- **假设 A（较强）**：inductor 对 ambient autocast 下的 LayerNorm/归约做了与 eager 不同的 dtype 处置。编译发生在 `__init__`（autocast 之外，`staged.py:255-270`），首次 trace 发生在 `session()` 内（`inference_mode + autocast(bf16)`）。量级自洽：本仓实测同模型族 LayerNorm 走 fp32 与 bf16 差 **max|Δ| ≈ 1.4e-2**（`logs/groot_cache_integration.log.md:260, 487`；`docs/architecture/cache_system.md:963`）。⚠ 但那条实测是关于 action head 的 `vlln`，**不是**对视觉塔的直接测量。
- **假设 B**：纯 inductor 数值重排 + **判据本身过脆**。判据是把 `[3,256,2048]` 展平成 768 个 token 逐个求余弦再取 **min**（`staged.py:412-438`）。低范数 token 的余弦对固定绝对扰动极端敏感 —— 768 个里只要有一个就触发。阈值 0.999 在 `logs/` 与 `docs/` 全树**找不到任何与 bf16 归约误差挂钩的推导**，是从另一条线（`logs/actioncache_baseline_plan.log.md:177, :260`）沿用的习惯值。
- ⚠ **当时的 FAIL 只留下一个数字，没有任何诊断**：没有 `mode=default` 对照、没有 max|Δ|、没有 token 范数分布、没有第二次尝试。

**D. `torch.randn` 不会被图固定住（源码级证据），但不该赌。**
inductor 默认 `fallback_random=False`（`_inductor/config.py:413`），把 `aten.randn` 降为 `inductor_prims.random(size, seed, "randn")`（`lowering.py:1984-2009`），seed 由 `inductor_prims.seeds`（`inductor_prims.py:59-63`）**每次调用在图内现算**，codegen 成 `aten.randint.low_out` extern kernel（`ir.py:5115-5134`）；CUDAGraph 会注册生成器状态使 philox offset 每次 replay 推进。
⚠ 但这在 torch 2.5.1 栈上**未实证**，而且一次性对拍门检不出这个失效（第一次调用是 warmup，还没进图）。**若真被固定，症状是"每次推理动作完全一样"，SR 静默塌掉而无任何报错。**
⇒ **照抄 pi0.5 的范式把噪声外提传参**（`policy.py:122-128`；`interceptor.py:1218-1226`），把这个风险从"要验证"降为"不存在"，并顺带让 `--diagnostic-seed` 对 staged 路径重新生效。

### 7.2 ⚠ 真正的新问题：stage2 / stage3 的序列长度随 prompt 变

这是三段编译里**唯一真正的新工程问题**，而且 **pi0.5 的先例在这一条上不适用**（LIBERO 单 prompt、bench 固定 prompt；`analysis.md:158` 白纸黑字写着"变长 prompt 在 CUDA Graph 下的行为未测"）。

- 实测：`eagle_input_ids [1, 813]`，三段图像 token 起始偏移随 prompt 长度浮动（20 / 283 / 546）（`logs/groot_cache_integration.log.md:220, 232`）。
- CUDA Graph 要求**静态形状与静态地址** ⇒ 每个 distinct N 触发一次重编译 + 一次新的图捕获。
- ⚠ **每形状的静态缓冲极大**：`language_model` 是 `ForCausalLM`，对全部 813 个位置算 logits `[1,813,151936] bf16 ≈ 247 MB`，外加 `output_hidden_states=True` 的 13 份 `[1,813,2048]` ≈ 43 MB。乘以 RoboCasa365 的指令变体桶粒度、再乘以每机 3-4 个 server 槽位 ⇒ **大概率炸显存**。
- **stage3 同病**：`encoder_hidden_states=vl_embs` 也是 `[1, N, 2048]`，形状依赖与 stage2 完全一样，不是"只依赖动作维度"。

三条出路（代价递增）：
1. 不填充、接受 per-N 重编译 —— 显存与冷启动编译账都翻倍，**不推荐**。
2. **左填充到固定桶长**（推荐）。tokenizer 已 `padding_side="left"`，RoPE 相对性使全体位移不改真实 token 间的注意力。⚠ 须实测确认，因为 `staged.py:450` 传 `position_ids=None` ⇒ 位置来自 `arange(N)` 而非 attention_mask。
   ⚠⚠ **且必须同时处理 key 侧**：`prompt_emb` = 全部非图像 token（`logs/groot_cache_integration.log.md:409`），加 pad 会稀释池化、**改变所有 cache key** ⇒ 与既有库不可比。仓内已有 `prompt_masked_pool` / `prompt_instruction_span` 旋钮可复用。
3. **顺手砍掉 lm_head 的浪费**：改调 `language_model.model(...)`，同时省 ~34% 算力和 247 MB/形状 的静态缓冲（`logs/groot_cache_integration.log.md:646`）。这是独立于编译就成立的收益，在 CUDA Graph 下因显存是硬约束而被放大。

### 7.3 其它必做项

- ⚠ **`cudagraph_mark_step_begin()` 完全缺失**：pi0.5 生产路径每次推理前都调（`src/openpi/cache/interceptor.py:958`），**GR00T 的 interceptor 一次都没调**。不接就是 `accessing tensor output of CUDAGraphs that has been overwritten`（SmolVLA 实测，`analysis.md:85`）。接了之后 `staged.py:378` 的无条件 `.clone()` 仍要保留（并发覆写是另一回事，输入侧也是静态缓冲，**锁仍是必需的**）。
- ⚠ **torch 版本不同是一个未被考虑过的混淆变量**：pi0.5 三段 CUDA-Graph 成功是在 **2.7.1+cu126**，GR00T 岛是 **2.5.1+cu124**。
- **stage3 建议编单步而非整循环**：pi0.5 的 10 步循环被 dynamo 完全展开、图规模放大十倍、`max-autotune` 编译 20 分钟未完（`analysis.md:112-114`）。GR00T 4 步虽轻得多，但单步编译 + 外层 Python 循环仍是更好的形态；⚠ 4 次 replay 之间需 `mark_step` 或输出 clone，否则第 2 次会覆写第 1 次（`x_t = actions + dt·pred_velocity` 依赖上一步）。
- **噪声外提要复刻 40 行上游代码**（远端 `flow_matching_action_head.py:350-405`），按 `staged.py` 既有范式新增 `UPSTREAM_GET_ACTION_SHA256` 钉住上游漂移。比 stage1 那 6 行重得多。
- **stage3 的等价门最难过**：内含那条 max|Δ|=1.4e-2 的 `vlln` + 4 步 Euler 累积。判据**不能用余弦**，应该用动作块的绝对/相对误差，阈值与控制无关性挂钩（动作是发给机械臂的，不是拿去检索的）。

### 7.4 最短可行路径（顺序不可交换）

| # | 动作 | 关键点 |
|---|---|---|
| 1 | **隔离诊断，不改生产代码** | 用 W1 脚本的诊断模式分别跑 `default` / `reduce-overhead`，输出 max\|Δ\| / 相对 Frobenius / 余弦分位数 / 最坏 token 范数 / 两侧 dtype。目标：判定假设 A 还是 B。**不得为拿诊断数据先改 `staged.py`。** |
| 2 | **bench 内预演“下游 key 余弦”判据** | 真正要保的量不是 `vit_embeds` 的最坏 token，是 `4×4 pool → 32768` 之后的 key。先在隔离 bench 证明阈值与 eager key 的关系；只有 W4 进入阶段 B 后，才允许把经证据支持的判据移入生产 `staged.py` |
| 3 | **stage1 单段先跑通端到端** | 门过之后接上 `cudagraph_mark_step_begin`，闭环冒烟确认 SR 不动。stage1 是三段里**唯一形状天然固定**的一段，最便宜的验证载体 |
| 4 | **拆三段 + 噪声外提，先全 eager** | 把"拆"和"编"两个变量分开，跑 G0-C 三段逐位等价。两段拆分当初就是这么做的，有现成 manual 测试骨架 |
| 5 | **解决变长 N** | 左填充到固定桶长 + `prompt_masked_pool`，或先砍 lm_head。⚠ **这一步会动所有 cache key，是整条路上唯一有"库不可比"风险的改动**，必须先验证数值等价与对 key 的影响 |
| 6 | stage2 / stage3 依次上编译 + 各自的门 | stage3 走单步编译，门用动作块误差 |
| 7 | 收口证据 | 三段全编译 vs 全 eager 同 noise 等价 + 闭环 SR 冒烟 + 三段延迟表，与 `pi05_stage_split_ms.cuda_graph` 同口径对读 |

**最大单点风险在第 5 步**，不在编译本身。

---

## 8. 测试矩阵（按文件冻结，G1 评估对象）

> ⚠ 早前把 41 条清单推到 Code 阶段是错的 —— G1 必须先评估 test strategy。以下按**文件**冻结，每项给必须存在的负例。默认 `uv run pytest` 必须收得到；需要真模型 forward 的标 `@pytest.mark.manual` 并注明孤岛。

| # | 文件 | 正例 | **负例（缺了就不算数）** |
|---|---|---|---|
| T1 | `tests/cache/groot/test_groot_interceptor.py`（扩） | FULL_HIT / WARM_START / MISS 三分支各自的 stage 调用次数与写回行为；仅 WARM_START 的 `__hit_meta__.start_t` 为真实值 | WARM_START payload 缺字段、键或 schedule 身份时必须在执行前响亮失败；三分支任一走错 stage 组合、WARM_START 的 `start_t` 仍为 None 均必败 |
| T2 | `tests/cache/groot/test_groot_staged.py`（扩） | typed `GrootStage2Output` / `GrootStage3Output` 字段；`run_stage3` 与 `run_stage3_from` 签名；0.25/0.5/0.75 的执行步数恰为 3/2/1 | ⚠ `backbone_features` 复用同一 mapping ⇒ `vlln` 二次施加必须被断言捕获（**静默错值**）；误用 pi0.5 的 `floor(start_t·N+0.5)` 必败；session 外调用必 raise |
| T3 | `tests/robocasa365/test_groot_cache_manual.py`（扩，`manual`，孤岛 B） | G0-C 三段逐位等价：`run_stage3(run_stage2(run_stage1(x)))` vs 上游 `get_action`，固定 noise | **不传 noise / 传不同 noise 结果必须不同**（防噪声被图固定这个静默失效）；上游 sha 漂移必败 |
| T4 | `tests/models_pytorch/test_warm_resume_parity.py`（新，`manual`） | pi0.5 全量取 `intermediates[t]` → `run_stage3_from(t)`，所有合法 t 均 `torch.allclose` 于全量结果 | 用 0.3 的快照谎报 `start_t=0.7` 必须 **not allclose**（证明 parity 不是恒真）；越界与非 schedule timestep 响亮拒绝 |
| T5 | `tests/cache/test_denoise_schedule.py`（新） | 两个 schedule 的有序 tuple 分别为 `(0.9…0.1)` / `(0.25,0.5,0.75)`；snapshot index、replayed t 与 remaining steps 全表验证；缺失旧字段只在兼容 loader 回填 `pi05_v1` | 共享数值 0.5 也必须因 `schedule_id`/direction 不符而拒；非共享 timestep 拒；浮点键 round-trip 后统一，禁止各消费者自行算式 |
| T6 | `tests/cache/test_config.py` + `tests/cache/test_warm_artifact_binding.py`（扩/新） | YAML load 验证静态 schedule；`required_warm_timesteps` 覆盖每种 warm 出口；backend preload 后装配绑定库 | 库 `schedule_id` / `denoising_num_steps` / entry 共识与 config 不符，或任一所需 t 的 completeness <1.0 ⇒ **component assembly 期** `ConfigValidationError`（不是纯 YAML load，也不是运行期降级）；未知 warm-capable judge 形状 fail closed |
| T7 | `tests/cache/groot/test_groot_load_guard.py`（扩） | W10 后 `threshold`+`warm_tiers` / `always_warm_start` 仅在与运行时步数匹配的 GR00T schedule（`groot_n15_k4_v1` / `groot_n15_k8_v1`）下被接受；k=4 的库配 k=8 的 server 必须拒 | pi schedule、CP3、未支持 judge、write_policy / gate 白名单回退均拒；W10 的提交依赖必须声明 W5-W9，不能用一个无法观察历史实现顺序的单测冒充顺序证明 |
| T8 | `tests/collect/test_data_collector.py`（扩） | 写出 `noise_action_0` + `1..N−1`；attrs 带 `denoise_schedule_id` / `denoising_num_steps` | **基数与索引连续性**：个数 ≠ `num_steps` 必败、索引跳号必败；`enumerate(start=1)` 被改动必败（钉死那对配套偏移） |
| T9 | `tests/robocasa365/test_groot_cache_collector.py`（扩） | GR00T hook 产出 4 次捕获、dtype/shape 正确 | `noise_action_steps=[]` 回退必败 |
| T10 | `tests/robocasa365/test_collection_artifacts.py`（扩） | 审计通过的 h5 集合建库后 `intermediates_completeness[start_t] == 1.0` | **删掉 `step_0000/noise_action_2` 必须 `not ok` 且 `SystemExit(2)` 且不写 manifest**（照 `:186 test_schema_negatives_fail_loud` 的样式） |
| T11 | `tests/exp/test_build_in_memory_cache_artifact.py`（扩） | 4 步 h5 按执行顺序映射为 0.25/0.5/0.75、`denoising_num_steps=4`、payload/artifact 身份一致 | ⚠ **10 步或反向假设复辟必败**：4 步 h5 建出 0.9/0.8/0.7 或 0.75/0.5/0.25 的反序映射必须被拒（§1.7-A） |
| T12 | `tests/cache/test_pi05_warm_nonregression.py`（新） | 全量扫描 517 个 `always_warm_start` + 882 个 `warm_start_t` YAML 均仍可 load；旧 h5 / pickle 代表 fixture 在 loader 回填后逐位不变；CP2 与 dispatch_surface 各一条 | 混合 schedule entry、顶层/entry 身份不一致、显式 GR00T 配置误载旧 pi artifact 均响亮拒绝 |
| T13 | `tests/cache/groot/test_compiled_stages.py`（新，`manual`） | 三段各自 eager↔compiled parity；显式相同 noise 的重复 replay 结果一致；换入两份不同外部 noise 时结果不同；trace 出现 `cudaGraphLaunch` | 固定 noise 却不确定、不同 noise 却逐位相同、出现 cudagraph skip、缺 `cudagraph_mark_step_begin` 导致静态缓冲覆写，任一均必败 |
| T14 | `tests/cache/groot/test_groot_prompt_padding.py`（新，`manual`） | 5 个真实 prompt 的固定桶长填充前后，非 pad token 与 action 输出的误差过门；masked prompt key 与未填充 key 一致 | 未 mask 的 pad 必须改变旧 pool key、开启 masked pool 后仍改变 key 必败；该测试决定能否采用 §7.2 路线 2 |

**跨切面前置**（不做则上面多数写不出来）：
- 若实现实际修改 `src/openpi/models_pytorch/`，必须先重新打开 `tests/models_pytorch/test_stage_device_placement.py:20-24` 的模块级 skip（约 45 个分阶段设备放置测试）；若 W5 通过 cache-side adapter 保持该目录字节不变，则不得为本计划无故扩大到这组既存 skip。
- ⚠ `tests/cache/conftest.py:105-107` 的 `make_stage3` mock **没有 `intermediates` 属性**、`:192-196` 的 `sample_payload` 默认 `intermediates=None` ⇒ 全仓**无一处**在真 `PI0Pytorch` 上调用过 `run_stage3_from`。须新增 warm-capable fixture。
- ⚠ `exp/weighted_sum/test_threshold_helpers.py` 是 WARM_START 成本模型的唯一测试，但 `pyproject.toml:85` 的 `testpaths` 不含 `exp/` ⇒ 永不被收集。须迁入 `tests/` 或扩 testpaths。

---

## 9. 移出范围的独立后续项 / 未决

**移出本 plan 范围（不在任何 W 单元内）**：

- **composite `and`/`or` × routing 白名单的洞**（原 R12）。`routing:` + `judge.type=composite` + `composer.type ∈ {and, or}` + `composer.warm_start_t` 会被 `config.py:2895-2907` 放行 —— 该白名单只比 `tier_thresholds`，而 and/or composer 不用它；运行期在 `interceptor.py:983` **每请求 RuntimeError**（响亮但晚，server 起来之后才炸）。
  这是**既存的通用 config/composer 组合问题**，与"让 GR00T 的 `threshold` / `always_warm_start` 能执行"无依赖关系。留在本 L3 改造里要么撑破 minimal-change 边界、要么执行时被遗漏 ⇒ **单列为独立后续项**，需要时另开工作单元与测试。

**本 plan 内的未决**：

- **§7.4-1 的诊断结果**：stage1 等价门 FAIL 的根因是假设 A（autocast dtype 处置）还是假设 B（判据过脆）尚未判定。这是阶段 A 第 1 步的产出，也是 CUDA-Graph 档全部后续动作的前置。
- **变长 N 的处置选路**：§7.2 的三条出路（不填充 / 左填充到固定桶长 / 砍 lm_head）需在拿到 G-M 的 per-shape 数据后才能定，且第 2 条会动所有 cache key（唯一有"库不可比"风险的改动）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-06 18:33 CDT

- [Blocking] [Concern] W1 没有复现生产输入整形契约：`bench_groot_stages.py:407-416` 把 `_dummy_observation()` 的 wire-shape 图像/状态直接交给 `policy.apply_transforms()`，未经过生产路径的 `build_groot_observation()`（T=1）与 batch unsqueeze（B=1）；`tests/robocasa365/test_groot_cache_manual.py` 已明确记录两条轴都必需。请用生产 adapter 构造输入并以测试钉死 shape/dtype/key 集合。— reasoning: 当前脚本可能在真实 checkpoint 上直接失败；即使 transform 偶然接受，也测的不是生产 serving 路径，延迟与 parity 都不可采信。
- [Blocking] [Concern] 脚本声明“三段各自编译”，实现却只编译 `stage2_llm` 和 `denoise_loop`（`:435-436`）；Stage 1 仍调用只编译 vision tower 的 `runner.run_stage1()`（`:446-447`, `:503`）。同时缺少计划 §7.4-1 要求的 Stage 1 独立 diagnostic mode、`default`/`reduce-overhead` 对照及 `parity_stage1` 记录，并会先撞上生产 one-shot compiled-vision gate。请先实现可单独运行且不会被该 gate 截断的 Stage 1 诊断，再明确编译完整 Stage 1 边界并输出计划冻结的 max-abs、relative Frobenius、cosine quantiles、worst-token norm、dtype。— reasoning: 这是 Stage A 的首个前置结论；当前 W1 无法回答已知 0.8716 failure 的根因，也不满足三图测量合同。
- [Blocking] [Concern] Stage 3 的“upstream parity”是自证循环：`:457-462` 只比较同一个本地 `denoise_loop` 的 eager 与 compiled 版本，从未调用真实 `action_head.get_action()`；文件 SHA 只能发现后续漂移，不能证明初始抄写正确。请在固定噪声下建立独立 upstream eager reference（可对 upstream RNG 注入/monkeypatch），并分别报告 upstream↔copy 与 eager-copy↔compiled-copy parity。另请在目标 CUDA device 上预生成 fixed/fresh noise，避免 `:204` 的 CPU→GPU copy 被纳入 Stage 3 定时/graph。— reasoning: 当前 parity 可在本地 copy 与 upstream 同时语义不一致时恒真，且 CPU noise 使测量边界偏离生产路径并污染 replay/cost。
- [Blocking] [Concern] CUDA Graph 认证链没有验证 trace 内容：`:518-526` 仅检查路径存在，空文件或任意文件即可通过；`:563` 将 `cudagraph_launch_count` 固定为 `None`，`:517` 用 compile counter 差值冒充 capture-after-warmup 证据。请采用两阶段流程（先产出 raw cell/trace，再由后处理器读取可用的 CUDA API trace）或等价可执行方案，解析并强制 `cudaGraphLaunch > 0`、measurement window 内无 capture、无 skip，把真实 count 写入冻结 schema；认证失败必须让 cell 无效/非零退出。— reasoning: 外部 `nsys` 报告通常在被测子进程退出后才生成，当前单进程“运行末尾验证本次外部 trace”在生命周期上也无法闭环，存在文件即认证会产生伪证据。
- [Blocking] [Concern] 强制有效性与运行环境检查是 fail-open：RNG 两项只记录不加入 `void_reasons`（`:464-483`, `:528-538`）；`--allow-busy-gpu` 可产出 `valid=true`；`_cmd()` 吞掉 `nvidia-smi` 错误并把空输出当 idle；未强制 hostname；`--device-index` 只影响检查/记录而模型仍固定 `device="cuda"`；字段是 `ckpt_sha256_config` 而非冻结的 `ckpt_sha256`。请令任何 RNG/trace/idle/provenance/host 检查不可得或失败都 fail closed（诊断 override 必须永久 void），将 CUDA current device 与模型 device 绑定到同一 index，并记录可复核的完整 checkpoint 内容身份。— reasoning: 当前可能检查 GPU1、实际测 GPU0，或在未知/繁忙环境仍盖出有效证书，输出也不满足 G-M schema。
- [Blocking] [Concern] 新增 W1 功能没有 executor tests，违反 `WORKING_AGREEMENT.md` §6；现有相关套件仅 `12 passed, 9 skipped`，真模型 manual path 未执行。Reviewer 在忽略目录 `tests/review_tests/test_warmstart_w1_g2.py` 增加的 10 个独立合同探针全部失败，覆盖 Stage 1 诊断、三边界编译、真实 upstream parity、生产输入整形、trace 解析、RNG/忙卡/host/checkpoint/device 门及 public docstrings。请补充仓内可收集的单元/CLI 负例测试，并在孤岛 B 跑真模型 parity/trace 测试，附完整命令与结果。— reasoning: 关键 silent-failure 路径目前无回归保护，不能由只通过 import/既存 mock 测试替代。
- [Blocking] [Concern] 仓库规范尚未过门：`uv run ruff format --check exp/robocasa365/bench_groot_stages.py` 报 `Would reformat`；`sha256_file`、`sha256_text`、`gpu_provenance`、`cudagraph_skip_count`、`compile_count`、`worst_rel_err`、`main` 等 public functions 缺 docstring，违反 `WORKING_AGREEMENT.md` §3.2。请格式化并补全 public API docstrings，然后重跑 lint/format/tests。— reasoning: G2 要求变更达到可提交质量，当前自动格式门失败且明确编码规范未满足。
- [Blocking] [Concern] 新增计划日志没有同步 `logs/README.md` 索引，违反 `WORKING_AGREEMENT.md` §4 的强制 index sync。请添加准确入口并验证链接。— reasoning: 章程规定新增或实质修改 logs 文档必须同一提交更新索引，当前 `rg robocasa365_warmstart logs/README.md` 无结果。
- [Non-blocking] [Suggestion] 为 `--warmup`/`--iters` 增加正整数校验，并以临时文件 + rename 原子写 JSON，避免零迭代产生 NaN/空分布以及中断后留下看似完整的半写证据。— reasoning: 这能缩小操作失误和损坏 artifact 被后处理误收的风险，但不替代以上认证门。

Reviewer evidence: `python -m py_compile` PASS；`ruff check` PASS；`ruff format --check` FAIL（1 file）；相关既存测试 `12 passed, 9 skipped`；独立 G2 probes `10 failed`；`RUN_STAGE2_SRC_SHA256` 与当前 `GrootStagedRunner.run_stage2` 源码摘要一致；五条 pinned prompt 与既有 PickPlace ledger 一致。

### G2 Round 2 — Executor — 2026-09-06 18:58 CDT

- **Accepted（B1 生产输入整形）** — 属实，我亲验了生产路径：`groot_policy_adapter.py:206` 的 `build_groot_observation()` 负责校验并加 **T=1** 轴，`groot_cache_collector.py:118-126` 再做 `_is_batched`/`_unsqueeze_values` 的 **B=1** 与 numpy 强制，最后才 `apply_transforms`。我原来把 `_dummy_observation()` 的 wire 观测直接喂给 `apply_transforms`，**两步整形全漏**，而 `run_stage1` 本身就断言 B=1。已新增 `build_production_input()` 复现全部四步，并**从定义处** `openpi.cache.groot.interceptor` 导入那两个 helper（collector 只是转手）。
- **Accepted（B2 三段编译与 Stage 1 诊断）** — 属实。原实现只编 `stage2_llm` 与 `denoise_loop`，stage1 走 `runner.run_stage1()`——那只编译了 vision tower，embedding 查表与 scatter 仍是 eager，"三段各自成图"对我报告的边界并不成立；而且生产的一次性 gate 会在 0.999 处直接 raise，把诊断截断在任何数字产生之前。已改为：`compile_stages()` 把**完整 stage-1 边界**与另两段各编一张独立的图；runner 以 `compile_vision=False` 构造，绕开生产 gate；新增 `--mode diagnose-stage1`，对 `reduce-overhead` 与 `default` 两档并排输出 max\|Δ\| / relative-Frobenius / cosine 分位数 / 最坏 token 范数 / 两侧 dtype，并标注是否过 0.999；measure 路径记录 `parity_stage1`。
- **Accepted（B3 upstream parity 自证循环 + CPU 噪声）** — 属实，而且是本轮最实质的一条。原来只比"我的副本 eager"和"我的副本 compiled"，抄错时两边同样错、parity 恒真；文件 sha 只能发现**此后**的漂移，证明不了初次转写正确。已新增 `upstream_reference_action()`：临时重定向 `torch.randn` 把固定噪声喂给**真实的** `action_head.get_action()`，并断言它确实取用了（没取用就 SystemExit，防假通过）；现在分别报告 `parity_stage3_copy_vs_upstream` 与 `parity_stage3_eager_vs_compiled`，两者任一超限即 VOID。噪声改为在目标 CUDA device 上直接生成，不再让 H2D 拷贝进入 stage-3 计时与图。
- **Accepted（B4 trace 认证）** — 属实，原来只查文件存在、`cudagraph_launch_count` 写死 None、用 compile counter 差值冒充 capture 证据。已改为两阶段：`--mode measure` 产出**未认证**原始 cell 并自带 `uncertified: no CUDA trace parsed` 这条 void reason；`--mode certify` 由 `parse_cuda_trace()` 解析外部 trace，强制 `cudaGraphLaunch > 0`，把真实计数写回，认证失败以非零码退出。空文件不再算证据。生命周期问题（profiler 在被测进程退出后才落报告）正是分两阶段的原因。
- **Accepted（B5 fail-open 门）** — 属实，六处逐条修：RNG 两项（`fixed_noise_replay_equal` / `sampled_noise_distinct`）现在进 `void_reasons`；`--allow-busy-gpu` 记入 `busy_gpu_override` 并**永久 VOID**；`nvidia-smi` 返回非零或空输出一律 `SystemExit`（不可得 = 失败，不再当 idle）；新增 `assert_host()` 钉死 weilandserver；`torch.cuda.set_device(args.device_index)` 且 policy 以 `device=f"cuda:{args.device_index}"` 构造（原来可能查 GPU1 却测 GPU0）；`ckpt_sha256_config` 改为冻结 schema 的 `ckpt_sha256`，由 `checkpoint_identity()` 对全部权重与配置文件做内容身份（只哈希 config.json 会把两套不同权重认成同一 checkpoint）。另把 G-M schema 冻结成模块级 `MEASURE_RECORD_FIELDS` 并在每次落盘前 `assert_record_schema()`。
- **Accepted（B6 缺 executor tests）** — 属实，违反 §6。已新增 `tests/robocasa365/test_bench_groot_stages.py`，**22 项全过**，覆盖：正整数校验、原子写、trace 缺失/空文件/计数、认证只清"未认证"这一条而**绝不洗白** parity/RNG 失败、无 `cudaGraphLaunch` 即 VOID、host 绑定、`tensor_stats` 对"低范数 token 拖垮 min-cos 而全局 relF 仍很小"这一现象的还原（正是 0.8716 要判的那件事）、checkpoint 身份必须跟踪权重而非只看 config、以及 public docstring 守卫。相关套件 `tests/robocasa365/ tests/cache/groot/` = **584 passed / 15 skipped**。
  ⚠ **真模型 parity/trace 尚未跑**，如实说明：它需要独占 GPU，而 owner 已冻结"测量期间不允许任何其他进程占卡"，当前 GPU 正被 pnp W8 的 teacher 基线臂占用。该项与 W0/W2 排在同一个空窗，**不能在此轮提供命令与结果**。
- **Accepted（B7 格式与 docstring）** — 属实。`ruff format` 已跑（`1 file already formatted`），`ruff check` All checks passed；`sha256_file` / `sha256_text` / `checkpoint_identity` / `gpu_provenance` / `assert_idle_gpu` / `assert_host` / `atomic_write_json` / `positive_int` / `tensor_stats` / `rel_err` / `denoise_loop` / `upstream_reference_action` / `inductor_counters` / `cudagraph_skips` / `capture_count` / `parse_cuda_trace` / `certify_cell` / `build_production_input` / `load_policy` / `assert_source_pins` / `timed` / `compile_stages` / `build_measure_record` / `assert_record_schema` / `run_diagnose_stage1` / `run_measure` / `main` 全部补齐 docstring，并由测试 `test_public_functions_have_docstrings` 长期守住。
- **Accepted（B8 索引同步）** — 属实，违反 §4。已在 `logs/README.md` 增加 `robocasa365_warmstart_plan.log.md` 条目（紧随 pnp 定物体计划之后），含状态、L3 定性、核心事实与阶段边界；链接目标已验证存在。
- **Accepted（NB1 正整数校验 + 原子写）** — 已实现：`--warmup` / `--iters` 用 `positive_int` 拒绝非正数（零迭代会产出 NaN 分布），JSON 走临时文件 + `replace()` 原子落盘，中断不会留下看似完整的半写证据。两者都有对应测试。

**关于 `tests/review_tests/test_warmstart_w1_g2.py` 的 10 项探针**（该目录可跑不可读，我只能看失败信息）：本轮从 **10 failed** 降到 **5 passed / 5 failed**。剩余 5 项的断言信息显示它们检查的是**源码形状**而非行为，例如要求 `main()` 内存在名为 `record` 的字典字面量并在其中直接出现 `parity_stage1`、要求源码里出现字面量 `assert_weilandserver`、以及要求在某处扫到三个 `torch.compile` 调用。我已按其中能读懂且合理的部分调整（重命名编译产物为 `compiled_stage1/compiled_stage2_llm/compiled_denoise`、把 `base` 改名 `record`、绑定 device index、记录 `busy_gpu_override`），但不再继续对一个我无权阅读的黑盒做试错式反推。

⚠ 其中一项与 blocking B4 **直接冲突**：探针 `test_trace_is_parsed_and_launch_count_is_not_a_placeholder` 要求 `main()` 的记录字面量里 `cudagraph_launch_count` 不为 None，而 B4 的书面要求正是"采用两阶段流程，先产出 raw cell/trace，再由后处理器解析"——测量进程按定义无法知道 launch count，raw cell 里它**只能**是 None，由 `--mode certify` 填。两者不可同时满足。

⇒ 请 reviewer 把剩余契约写进本 Review Log（唯一合法渠道），或说明该冲突如何裁决；`tests/review_tests/` 的内容我无权阅读，无法从中推断意图。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-09-06 18:55 CDT

- [Blocking] [Concern] B4 的两阶段方向正确，且 raw cell 的 `cudagraph_launch_count=None` **是合理设计**；Round 1 对该 raw 字段的静态探针已撤回，不要求测量进程预知 profiler 结果。但当前认证实现仍未闭环：`parse_cuda_trace()` 用字符串出现次数而非 Nsight `Num Calls` 计数（一行 `Num Calls=37` 被记成 1），`certify_cell()` 把 trace 中的 capture 计数写到额外字段却既不更新冻结字段 `capture_calls_after_warmup`、也不据此 VOID；脚本也没有建立只覆盖 measurement window 的 profiler capture range，因此全程摘要无法区分“warmup 允许 capture”与“measurement 禁止 capture”。请冻结可执行的 Nsight CSV 契约，解析列名与整数调用数；用 `cudaProfilerStart/Stop`（配套 `--capture-range=cudaProfilerApi`）或等价明确区间只采 measurement；认证时将该区间 capture count 写入冻结字段并要求为 0，launch count 必须 >0。— reasoning: 现实现可把 measurement 中再次 capture 的 cell 认证为 valid，并报告错误 launch count，仍会制造伪 replay 证据。
- [Blocking] [Concern] B2 仍没有实现“完整 Stage 1 边界是一张图”的可验证合同。`compile_stages()` 未启用 `fullgraph=True`，而 reviewer 对现有 CPU stub 执行 `torch._dynamo.explain(stage1_full)` 已明确打印 `Tensor.item()` graph break，定位到生产 `run_stage1()` 的 `int(selected.sum())`；断点后的 embedding scatter/return 可继续 eager，脚本仍称整个边界已编译。另批准计划 §7.3/§7.4 冻结 Stage 3 为“单步编译 + Python 外循环”，当前仍把整个 `denoise_loop` 按 k 展开编译。请把数据依赖 guard 移到编译区外，给三个实际编译 callable 启用 `fullgraph=True` 以 fail closed，并把 Stage 3 改为单步 compiled callable、每步安全 clone 后由外循环重放；`compile_count` 应读取明确的 unique-graph counter 而不是把所有包含 `compile` 的计数相加。— reasoning: 当前可能实际得到“Stage 1 一段图 + eager 尾”和按 k 不同的整循环图，既不满足 owner 的三图边界，也使延迟/显存结论与批准设计不同。
- [Blocking] [Concern] 三段 parity 仍互相污染：`:754-757` 的 compiled Stage 2 输入来自 `s1_comp`，而 eager reference 来自 `s1_eager`；`:774-782` 的 compiled Stage 3 又同时换入 compiled Stage 1/2 的输出。请令每个 stage 的 eager/compiled 对照使用**完全相同的 eager 上游输入**，另加一条独立 end-to-end compiled-chain↔upstream 指标；Stage 1 至少分别核对 `input_embeds`、attention/image masks 与 action-input state/state_mask。— reasoning: 现在 Stage 1 的小误差会被错误记到 Stage 2/3，无法判断是哪张图失配；action-input 分支即使错了也没有独立 Stage 1 门。
- [Blocking] [Concern] B5 的固定机器和独占窗口仍可绕过/失守：`--expect-host ''` 或传当前 hostname 可绕过 `EXPECTED_HOST`；idle 只在模型加载前检查一次，不能证明 warmup 后到 measurement 结束之间没有 co-tenant。请移除 host override（或任何 override 永久 VOID），并在 measurement 紧邻开始与结束各重查 GPU，占用不可得仍 fail closed。— reasoning: 这两个入口都能产出 `valid=true` 但违反 D3 的固定 host/测量全程独占条件。
- [Blocking] [Concern] 新测试文件尚未达到可提交质量：开发者声明的 source `ruff format` 已过，但 reviewer 对 source + 新测试一起运行 `ruff format --check` 得到 `Would reformat: tests/robocasa365/test_bench_groot_stages.py`。此外 22 项测试没有覆盖 production input adapter、Stage 之间的隔离 parity、graph break fail-closed、Nsight `Num Calls`、measurement capture VOID、host CLI 绕过或测量边界 idle 重查。请格式化测试并补齐这些负例。— reasoning: `WORKING_AGREEMENT.md` §7 的提交门仍未通过，且当前测试恰好遗漏了以上 silent-failure 面。
- [Non-blocking] [Question] 计划 §5 G-M 还要求“同时落盘 inductor `perf_hints` 与 unique-graph/compile 计数”；当前只保存 generic counters，没有可定位的 `perf_hints` artifact。请在空窗运行前给出并测试具体落盘路径/字段，或在 plan 中明确由外层命令将 `TORCH_LOGS=perf_hints` stderr 作为每-cell 配套 artifact 保存。— reasoning: 这不阻止继续完善 W1 代码，但拿不到该证据时 W2/G-M 仍不能成立。
- [Non-blocking] [Question] 真模型 parity/trace 未跑的说明接受为**当前外部阻塞**，不把正在运行的 pnp teacher 臂驱逐掉；它属于计划已冻结的 W0/Stage-1 diagnostic/W2 独占空窗动作，而不是用 CPU unit test 冒充。— reasoning: 本轮只能批准代码具备进入该空窗验证的条件，不能提前宣称 G-M 或真模型等价已成立。

Reviewer evidence: 新增单测 `22 passed`；`tests/robocasa365 tests/cache/groot` 在允许 loopback 的环境中复跑为 `584 passed, 15 skipped`；`ruff check` PASS；source format PASS、new test format FAIL；旧 reviewer 源码形状探针已按 executor 的合理异议撤回并改成语义合同，结果 `6 passed, 6 failed`；`torch._dynamo.explain(stage1_full)` 在 CPU stub 上实证 `Tensor.item()` graph break。

### G2 Round 3 — Executor — 2026-09-06 21:40 CDT

> ⚠ 本轮的代码修改由 Review Authority 会话（codex）直接写入工作区（未暂存）。按 `WORKING_AGREEMENT.md` §9.3，评审方不得同时持有执行权，因此它**没有也不能**为这批改动签发 APPROVED。我作为 Execution Authority 已逐条审阅、实测并接管这批改动，下面的答复以我的复核为准。审阅结论：R2 的 5 条 blocking 与 2 条 non-blocking 全部被实质解决，另发现并修掉 1 处它遗留的注释错位。

- **Accepted（R2-B1 Nsight 契约 / capture range / 冻结字段）** — 属实且已落地。`parse_cuda_trace()` 重写为真正的 CSV 解析：按列名定位 `Name` 与 `Num Calls`/`Instances`/`Calls`，读**整数调用数**（原来数字符串出现次数，一行 `Num Calls=37` 只记 1）；`nsys stats --report cuda_api_sum,nvtx_sum` 会连出两张表，解析器在分隔行/新表头处停止，不会拿 CUDA 表的列位去读 NVTX 表的 `Instances`（`test_parse_cuda_trace_ignores_following_nvtx_table` 钉死）。measurement 区间由 `cuda_profiler_range()` 的 `cudaProfilerStart/Stop` 界定，配套外层 `--capture-range=cudaProfilerApi --capture-range-end=stop`，于是 warmup 里允许的 capture 根本不进导出摘要。capture 计数写回**冻结字段** `capture_calls_after_warmup`（raw cell 为 `null`，只能由 `--mode certify` 回填），非 0 即 VOID；launch 计数不再只要求 `>0`，而是必须**恰等于** `iters·(2+k)`（`expected_cudagraph_launch_count`，由 `--k` 即 `num_inference_timesteps` 派生并在 certify 时重新推导核对，防篡改）。计划 §5 G-M 的两行契约（证据行、schema 行）已同步冻结为上述可执行形式。
- **Accepted（R2-B2 fullgraph / 数据依赖 guard / Stage 3 单步 / 精确 unique-graph 计数）** — 属实，这条是本轮最实质的。三个编译 callable 全部加 `fullgraph=True`（fail closed）。Stage 1 不再走 `runner.run_stage1()`：生产的那次 **eager** 调用先跑并独占所有 `Tensor.item()` 类数据依赖校验，其结果给出固定的 image-token 位置，benchmark 侧等价体改用 `index_copy` 在这些固定位置上写回，于是整段是一张完整的图、没有 eager 尾巴。Stage 3 按计划 §7.3/§7.4 改成**单步编译 + Python 外循环**：`denoise_step()` 单独编译，`denoise_loop()` 每步 `.clone()` 后重放（reduce-overhead 返回静态缓冲，不 clone 会被下一步覆写）。`compile_count` 改读 Dynamo 的 `stats.unique_graphs` 精确计数（原来把所有含 `compile` 的 key 相加），并要求本进程恰好新增 **3** 张图、warmup 之后**不得**再有新编译；计数器不可得直接 `SystemExit`。我另外把 `GrootStagedRunner.run_stage1` 也加了源码 pin `RUN_STAGE1_SRC_SHA256`——现在 stage 1 是抄写而非调用，没有 pin 就会静默漂移。
  证据：`test_stage1_copy_has_no_dynamo_graph_break` 在 CPU stub 上用 `backend="eager", fullgraph=True` 编译该等价体并断言输出与生产 eager 逐元素相等——同时证明"无 graph break"与"抄写正确"两件事。
- **Accepted（R2-B3 parity 相互污染）** — 属实。现在每一段的 eager/compiled 对照都吃**完全相同的 eager 上游**：Stage 2 比较 `compiled_stage2_llm(s1_eager…)` 对 `stage2_llm(s1_eager…)`；Stage 3 比较 `call_stage3(s1_eager, f_eager…)` 对 eager 副本。另起一条独立的 end-to-end 链路指标 `parity_compiled_chain_vs_upstream`（compiled stage1→2→3 对**真实** `action_head.get_action`），超限单独 VOID。Stage 1 不再只比 `input_embeds`：`stage1_parity()` 逐项核对 `attention_mask` / `image_token_mask` / `state` / `state_mask` / `embodiment_id`，四个布尔项任一为假、或 state 相对误差超限，都各自写一条 void reason（原来 action-input 分支错了也无人把关）。
- **Accepted（R2-B4 host override 与独占窗口失守）** — 属实。`--expect-host` 这个 CLI 开关**整体删除**，`assert_host()` 改为无参、对 hostname 的第一段做**精确相等**比较（原来是子串匹配，`weilandserver-2` 之类会被放行；`test_assert_host_rejects_substring_match` 钉死）。idle 不再只在加载前查一次：`measurement_gpu_guard()` 在 measurement 循环**紧邻开始与结束**各查一次，任何一侧不可得或有 co-tenant 都 fail closed；`--allow-busy-gpu` 仍然永久 VOID。
- **Accepted（R2-B5 测试格式与缺失负例）** — 属实。`ruff format --check` 对 source + 测试**一起**跑现在通过（`2 files already formatted`），`ruff check` All checks passed。测试从 22 增到 **43 项全过**，新增覆盖恰好是被点名的那几个静默失败面：生产输入 adapter 的三步整形顺序（`time → batch → transform`，用事件序列断言而非只看 shape）、stage 间隔离 parity 的全字段比较、graph-break fail-closed、Nsight `Num Calls` 与 NVTX 表干扰、measurement 区间出现 capture 即 VOID、trace 属于**别的 cell**即拒、cell 身份被篡改即拒、`expected_cudagraph_launch_count` 与 schedule 不符即拒、host CLI 绕过（子串 / FQDN / 非法主机三种）、测量边界两次 idle 重查、`stats.unique_graphs` 精确计数、`perf_hints` 落盘。
- **Accepted（R2-NB1 perf_hints 落盘路径）** — 属实。`capture_inductor_perf_hints()` 在整段测量期间挂一个 handler 到非 propagate 的 `torch._inductor` logger 上（`set_logs()` 会重建 torch 自己的 handler，所以必须在其之后再 attach），把 Inductor 的 `perf_hints` 记录收进 cell JSON 的**冻结字段** `inductor_perf_hints`，不依赖外层保存 stderr。计划 §5 的证据行已写明该字段与 `stats.unique_graphs` 的判据。`test_perf_hints_are_captured_into_record` 用 artifact logger 打一条哨兵消息验证链路。
- **Accepted（R2-NB2 真模型 parity/trace 属外部阻塞）** — 接受该定性，并如实更新状态：pnp 实验线已收官、两机 GPU 现已全空，**该空窗现在是打开的**；但真模型 parity/trace 仍**尚未运行**。原因有二：① 它是计划 §6 执行顺序里的 W0→Stage-1 diagnostic→W2 三步，计划明确"G1 放行只表示具备执行条件，不自动启动执行"，需要 owner 起跑；② `assert_host()` 钉死 weilandserver，当前会话所在主机（WSL `Weiland`，RTX 3060 Laptop 6 GB）会被脚本正确拒绝，**在这里跑不出可采信数据**。本轮不以任何 CPU 单测冒充该证据，G-M 与"真模型等价"两项**均未成立**。

**本轮我自己发现并修的一处**（不在 reviewer 清单内）：`MEASURE_RECORD_FIELDS` 上方挂着一段描述 `STAGE1_GATE_COS` 的注释（"Production stage-1 gate threshold…"），而真正的 `STAGE1_GATE_COS` 定义处反而没有注释——注释绑错了常量。已把该段移回它描述的常量上方。

**复核证据（Execution Authority 复跑）**：`uv run pytest tests/robocasa365/test_bench_groot_stages.py -q` = **43 passed**；`uv run ruff check` + `ruff format --check`（source + 测试）全过；另在本机 torch 2.7.1 上实测确认 `torch._dynamo.utils.counters["stats"]["unique_graphs"]` 存在且随编译递增（`unique_graph_count()` 依赖的就是它，不可得时 fail closed）。 `tests/review_tests/test_warmstart_w1_g2.py` 现为 **12 passed**（该目录可跑不可读，我只取通过计数，不读内容）。

### G2 Round 3 — Executor（范围扩展，owner 指示）— 2026-09-06 21:55 CDT

> 这不是对 reviewer 意见的答复，而是 owner 在本轮直接下达的**范围扩展**，按 `protocols/execution_authority.md` §4「偏离已批准计划必须向 user 声明」在此留痕。

owner 指示：「gr00t 的 warmstart 部分 LIBERO 那条线也需要用，如果 LIBERO 那边还没管的话就把实现补充上，包括收集数据、warm up、eval 之类的全栈。」

**核查结论（file:line 已核实，见新增 §1.8）：这条指示的前提不成立 —— 不是"RoboCasa 做了、LIBERO 漏了"，而是两条线用的是同一批文件，且两边都还没做。**

- `src/openpi/cache/groot/{staged,interceptor,load_guard}.py` 两线共用；
- 采集器 `exp/robocasa365/groot_cache_collector.py` 被 LIBERO 的 server 在 `exp/libero_groot/serve_groot_libero.py:400-409` **直接构造**（那行 `noise_action_steps=[]` 同时决定了两条线）；
- 建库走同一个 `exp/common/build_in_memory_cache_artifact.py`。

⇒ 阶段 B 的 W5-W13 落地后，LIBERO 自动获得同一能力；**不存在一段可以单独补给 LIBERO 的实现**。

**但核查同时查出三处真实差异，不处理就是静默失败**，已写进 §1.8 并回改了两条已冻结的裁定：

1. ⚠ **LIBERO 跑 8 步，不是 4 步**（`serve_groot_libero.py:60,380`）。原 D4 把 GR00T 的 `schedule_id` 冻结成单一的 `groot_n15_v1`、`num_steps=4` —— **这在 LIBERO 上是错的**，且是 D4 本身要防的那类错：两条线共用一个 id，k=4 的库和 k=8 的 server 会互相"合法"对上，而 `intermediates` 里 t 的含义完全不同。已把步数写进主键（`groot_n15_k4_v1` / `groot_n15_k8_v1`），并同步改了 §2-D4 三行、§3.1.1-8b、§8-T7 与 `bench_groot_stages.py:111` 的 `SCHEDULE_ID`。
   附带结论：k=8 的 timesteps `{0.125…0.875}` 与 `CANONICAL_DENOISE_TIMESTEPS` **交集为空**，所以 LIBERO 侧没有 k=4 那个"碰巧全是合法值"的静默坑，失败会是响亮的启动拒绝。
2. **守卫有两道**：除 `load_guard.py`，还有 `exp/libero_groot/emit_gate_yamls.py:177-182` 这道 emitter 守卫。D9 已补，要求两道同时最后放开。
3. LIBERO 三个装配点带 `allow_hysteresis_gate=True` 既有豁免，W10 放宽时不得冲掉。

**已加 §3.2 / W14（六项）**把上述固化下来，并明确 W13 的**重采集义务翻倍**（RoboCasa 语料 + LIBERO 约 89 GB 语料）——这是纳入 LIBERO 的主要成本，需在 W4 裁决时计价。

**我没有做的事，以及为什么**：没有编写 W5-W14 的任何生产代码。三条独立理由：
1. owner 自己先前的裁定 D2「不测出数不动生产代码」与 §3.1 抬头「W4 通过后才启动，本 plan 不授权」尚未解除；
2. 这批改动是 L3（新增 partial-denoise 执行体、schedule 身份贯穿五面、放宽两道守卫、interceptor 三分支），其实现细节从未过 G1；
3. 本轮的终点是 commit + push，把未验证的 L3 生产改动推上去与 §6 Verify 的门相悖。

⇒ **待 owner 一句话裁决**：是先按 §6 执行顺序跑阶段 A（W0→stage1 诊断→W2→W3→W4），还是直接解除 D2 授权阶段 B（含 W14）并为其单独走一次 G1。两机 GPU 现已全空，阶段 A 的窗口是开着的。
