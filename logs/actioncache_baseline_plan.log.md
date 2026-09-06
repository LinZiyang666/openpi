# ActionCache 式 post-backbone 基线：CP2 单 key 阈值臂（两 suite × 两库 regime）

> Level: **L3**（新 checkpoint 等级 CP2 接线：config / orchestrator / interceptor / judge / types / stage_io / batching_coordinator + `Stage2Output` 加一个可选字段与一个新的公开 staged 方法 + 新 KeyBuilder + 新 exp 目录 + 架构文档更新）
> 状态：`Done`（v0.6 冻结；G1 APPROVED R4；§4 Code 完成；G2 APPROVED R2 2026-09-04，owner-authorized self-remediation 见 D16；§6 Verify 全量 `uv run pytest` 2026-09-04 见下文 Verify 记录；§8 步骤 2–6（建库 / parity / shadow / 出臂 / rollout / 开销实测）为后续实验运行，不在本 commit）
> 上位文档：`docs/papers/actioncache_2607.06370v2.{pdf,txt}`（本地副本）、`docs/iclr/actioncache_response_plan.md` §5（四臂方案，本 plan 只做其中的 post-backbone 臂）、`docs/iclr/actioncache_positioning_plan_codex.md` §3（P2-AC 臂定义）
> 术语：不用 E/X/Arm 代号；实验一律写目录名 + 一句话。

---

## 0. Owner 决策记录（逐条、带日期；本节只增不改）

| # | 日期 | 决策 | 后果 |
|---|---|---|---|
| D1 | 2026-09-04 | 在 libero_spatial 与 libero_10 上复现 ActionCache 的实验形态 | 两 suite |
| D2 | 2026-09-04 | 他们的库构造（prefill 集数、init 来源）披露不清，**不复现其采集过程**，直接用我们现有的库 | 库 = 我们的 pkl 对应的 H5 |
| D3 | 2026-09-04 | 两个库规模：50 轨迹级（`exp/rit_pareto` 用的 `cp1_spatial_pool_16.pkl`）与 500 轨迹级（`exp/ablation_study/cache_size` 的 S6 pkl，含失败轨迹） | 2 suite × 2 规模 = **4 组** |
| D4 | 2026-09-04 | 对照库与我们的库**逐条一一对应**：我们 pkl 的每条 entry 在他们式的库里有且只有一条同 (trajectory, step) 的 entry，payload 相同，只换 key | 建库方式 = 复制 CP1 pkl 的 entries、只替换 `query_keys` |
| D5 | 2026-09-04 | H5 存的是 VLM backbone **之前**的数据；他们要 backbone **之后**的输出 ⇒ 用 π0.5 把 H5 的 Stage 1 数据跑过 backbone，再用 Random-Sparse-Ternary 矩阵降到 d=500 | 新 KeyBuilder 从 CP2 取数 + 离线建库脚本 |
| D6 | 2026-09-04 | 其余（backend / 检索 / judge / conductor / 成本口径）复用现有框架；用我们的术语他们 = **单模态 + cosine + 单阈值 + 单一固定档** | 零新机制，只加 CP2 接线 |
| D7 | 2026-09-04 | 结论前提：我们的方法效果优于他们（执行档上他们是我们的子集）。本实验的目的是**把这个前提变成同库、同 teacher、同硬件的实测行** | 不做"打不打得赢"的假设检验措辞，做 frontier 对照 |
| D8 | 2026-09-04 | 执行 horizon 用我们的 **5**（`examples/libero/main.py:56` `replan_steps=5`，与 upstream openpi 一致）。他们的 10 来自 lerobot 的 `n_action_steps=10` 约定（原文 §4.1 "we set the action execution horizon to 10"，Table 5），lerobot 文档称其"matching the original OpenPI implementation"与 openpi 源码不符 | 两侧臂同用 5；与他们论文绝对数字不直接比，作披露项 |
| D9 | 2026-09-04 | §10-Q1：出口方案 **A**（`Stage2Output.prefix_out` 可选字段） | 实施形态按 G1 R1 收敛为 **A′**（§3.1）：可选字段 + 新公开方法 `run_stage2_capture`，`_stage2_llm_backbone` / `run_stage2` 一字不改，不触发 WA §2.5/§3.1 override |
| D10 | 2026-09-04 | §10-Q2：档型按他们主实验 **N_hit ∈ {0,1}** | = FULL_HIT@CP2 / WARM@0.1@CP2 |
| D11 | 2026-09-04 | §10-Q3/Q4：**只做他们的实验，不补我们侧的 S6 阈值扫描**。但要画帕累托极限，所以在他们的基础上跑 warmup / cache-in-the-shadow 收集分数分布，映射到 IR 空间再取切点 —— 即我们 **GST K=1** 的口径；不用原始 cosine 网格 | 阈值 = shadow 分布上的 IR 寻址分位切；每档一条 IR 阶梯 |
| D12 | 2026-09-04 | §10-Q5：投影按他们的取（d=500，p=0.01，每行 ±1 各 floor(pD/2)=9,912 个，seed 冻结） | 元数据入 artifact |
| D13 | 2026-09-04 | §10-Q6 核实：他们的库**跟 suite 走**（LIBERO 每 suite 一个 10,000 条的库，spatial/l10 不混）；但 **suite 内不做 task 过滤**（key 里带语言 token，cosine top-1 在整个 suite 库上取），原文 "reuse ... across different episodes or even different tasks"。我们的检索是任务内作用域（`search_strategy.py:389`，`ctx.task_key` 来自 `orchestrator.py:632-636`）。**待 owner 裁**：忠实版（无 task 过滤）还是我们口径（有过滤）；建议忠实版，过滤版作为同库零成本消融 | 需要一个 `task_scoped` 开关（additive，默认 True） |
| D14 | 2026-09-04 | §10-Q6：**关掉 task 过滤**（忠实版，suite 内全库 top-1），**不做过滤版消融** | CP2 臂 `task_scoped: false`；主线只有他们的实验 |
| D13-证据 | 2026-09-04 | owner 质疑"从未启用过任务过滤"。实证：把 `exp/rit_pareto/data/runs/*/per_step.jsonl` 全部命中行的 `winner_id`（轨迹 stem）经本地 pkl 映到 `payload.task_key`，与 episode 的 `task_id`（`config/task_order_*.json`）比对：spatial 5 组 934,352 次命中、l10 4 组 2,005,866 次命中，**跨任务命中 0**，10 个 task 各只对应 1 个 winner task。过滤自 2026-04-06 首个 commit 起就在（`git log -S`），LIBERO 所有阈值实验都在任务内检索 | Q6 的差异是真实的 |
| D15 | 2026-09-04 | Owner Ziyang Lin 定向覆盖 WA §9.3：授权 G1 R3 reviewer 在同一会话原地修复其审查项并复核；执行方 v0.5 + R3 review 保留在 index，reviewer 的 v0.6 修订留在 working tree | R4 approval 明示为 owner-authorized self-remediation，不声称独立复审 |
| D16 | 2026-09-04 | Owner Ziyang Lin 定向覆盖 Review Authority §4–§6：授权 G2 R2 reviewer 先把执行方 R1 修订加入 index，再在同一会话修复至可放行；reviewer 修订与本条记录留在 working tree，不再请求另开 Execution / Review session | G2 R2 的 APPROVED 明示为 owner-authorized self-remediation；执行方快照与 reviewer 修订由 staged / unstaged 边界区分，不声称修复后另有独立审查者 |
| D17 | 2026-09-04 | Owner Ziyang Lin 实验启动指令：4 组各**扫 10 个点**（原 §3.9 的 ≤17 臂是上限）；拓扑 weilandserver 4 replica + timan107 48 worker；每臂 pruned-500 全集；spatial 先于 l10，不做完不停；20 min cron 巡检 + 条件触发 Monitor；跑完不画图 | 执行方解释：每档目标 IR {65,75,85,95}（4 臂）+ 每档参考臂 θ_raw=0.85（1 臂）= 10 臂/组；`export_arms --targets 65,75,85,95`；阶梯可由 owner 改 |

**执行方注记（非 owner 决策）**：D7 是 owner 的工作假设，记录在案；本实验的读数按 §1 预注册为可证伪的前沿对照，结果如何写以实测为准（G1 R1 B9）。

## 1. 目标与范围

- **问题**：在同一个库、同一个 teacher、同一套评测池上，ActionCache 式 post-backbone 单 key 阈值 cache 的 SR × 解析 model-forward IR 前沿在哪；它在两个库 regime（50 轨迹库 / S6 库）下各在哪。
- **两类组，读数不同**（D11：不补我们侧 S6 扫描）：
  - **50 轨迹库 × 2 suite**：同库对照。我们侧前沿引用 `exp/rit_pareto`（K=2 RIT/GST，同一 pkl 内容、同评测池）作参考线；读数 = 两条前沿在共同 IR 区间上的位置关系，预注册三种结果都可报（他们式更高 / 我们更高 / 噪声内不可分）。
  - **S6 库 × 2 suite**：**只有 ActionCache 式臂**，读数 = 同一臂在两个**库 regime**（50 库：2026-04 采集、5 init/task、仅成功轨迹；S6 库：2026-08 差集池采集、45 init/task、含失败轨迹）下的前沿差，**描述性 library-regime comparison**。两库非嵌套、来源/成功组成/容量同时不同，size 与 composition 不可分，文档与图注不得称 "library-size effect / scaling"；每库报告确切组成（§3.11）。
- **不做**：他们的在线续入库、LRU/LFU、他们的 ckpt、horizon 10（D8）、我们侧 S6 扫描（D11）、过滤版消融（D14）、端到端 latency 分位/吞吐 serving bench（§3.10）。全部作为披露项写进结果。

## 2. 可行性核查（全部亲验，file:line）

| 事项 | 结论 | 证据 |
|---|---|---|
| H5 有什么 | 两个库族的 H5 每步都有：`vision_0/1/2 float16[256,2048]`、`prompt_emb float16[200,2048]`、`robot_state float32[32]`、`clean_action[10,32]`、`noise_action_1..9[10,32]`、**`input_images/{base_0_rgb,left_wrist_0_rgb} uint8[224,224,3]`**；文件 attrs 含 `task`（S6 另有 `prompt`）、`success`、`num_steps` | 本地 `exp/common/data/db/libero_cache/<suite>/*.h5` 与 weilandserver `/data/openpi/ablation_study/cache_size/collect_h5/<suite>/task_N/episode_M.h5` 均已逐 key 列出 |
| 从 H5 重建 prefix 并跑 backbone | 已有通路：`_build_fake_stage1`（`exp/common/build_in_memory_cache_artifact.py:365-404`）；`_build_fake_stage1_with_masks` 重 tokenize 还原 lang mask + tokenizer self-check（`docs/cache/llm_layer_extract.md:208-211`） | 复用；跑到最后一层 + final norm |
| 原始图像可用于 parity | `input_images` 是 transform 后的 224×224 输入（`collection_policy.py:101,218`），可重建 Observation 走真 `run_stage1` | §3.8 parity 协议（离线、精确） |
| backbone 输出在哪 | prefix-only 分支 `prefix_output = ...last_hidden_state`（`gemma_pytorch.py:102-112`）；`_stage2_llm_backbone` 以 `_, past_key_values = ...` 丢弃（`pi0_pytorch.py:548-555`）；`Stage2Output` 只有 `stage1 + past_key_values`（`:74-91`） | §3.1 A′ |
| 维度 | 968 token × 2048 = **1,982,464** = 他们 Table 5 的 D（含 padding 位） | 一致 |
| Stage2 传输面 | `stage_io.stack_stage2_output/split_stage2_output` 只搬 `stage1 + past_key_values`（`stage_io.py:179-196`）；coordinator 在 `batching_coordinator.py:830-831` 调 `self._model.run_stage2` 后 split，`:896` stack 后进 stage3；serving 默认 concurrent，`--replicas>1` 强制 concurrent（`serve_policy.py:722-726, 758-759`）⇒ 正式拓扑（4 replica）**走 coordinator** | §3.1 必须覆盖 |
| compile | serving 入口 `_disable_compile_for_serving`（`serve_policy.py:209-220`）⇒ interceptor `compile_mode=None`（`interceptor.py:367-383`），staged 函数在 serving 下是 eager | A′ 不依赖 compile 行为 |
| CP2 现状 | `CheckpointID.CP2` Reserved（`types.py:58-61`）；`_VALID_CHECKPOINTS={"cp1","cp3"}`（`config.py:665`）；orchestrator 只遍历 `("cp1","cp3")`（`orchestrator.py:218`）；四处 `checkpoint_id == CheckpointID.CP1` 守卫（step counter / verdict 记录：`orchestrator.py:545-547, 600, 620, 698`）；`_state_history_anchor_cp = min(gates)`（`:90-92`）；`ThresholdJudge` 阈值表只有 CP1/CP3、warm_tiers 只对 CP1（`judge.py:299-303, 318`）；judge factory 只传 cp1/cp3 阈值（`config.py:3418-3421`）；interceptor 每步无条件 `check(CP1)`（`interceptor.py:925`），后处理与 wire 固定读 `cp1_result` | §3.3 |
| 真实 config schema | `KeysConfig` 五个固定字段（`config.py:74-80`），`_keys_iter`（`:716`）；`KeyBuilderConfig` 无通用 `params`，每种 builder 一个 typed 子配置（`:511-533`）；`SearchStrategyConfig.field_similarity: dict[str, FieldSimilarityConfig]`、`score_normalization: ScoreNormalizationConfig{type, fields}`（`:380-433`）；`BackendConfig.vector_dims` + `InMemoryConfig.preload_path`（`:469-481`）；未知 key 只告警忽略（`:848-850`） | §3.4 按真实 schema 写 |
| 单模态 cosine | `weighted_score_sum` 拒绝 `type:none`（`score_normalizers.py:407,541`）；`affine_clip(lo,hi)` 注册名 `"affine_clip"`（`:118-128`） | `per_field` 形式 |
| task 过滤 | `_build_step_filters` 用 `ctx.task_key`（`search_strategy.py:379-406`）；`text_ivf_knn` 在策略内抑制 task_key 的先例（`:524, 577`）；策略由 `_build_search_strategy` 构造（`config.py:3865-3935`） | §3.6 |
| artifact 绑定点 | `build_shared_storage` 是 storage 构造唯一 choke point，"covers both public assembly entries"（`config.py:2826-2832`）；`key_builder_type` / `prompt_pool` 的 expected-vs-loaded 检查在 `:2863-2890` | §3.7 |
| runner / wire | `run_gtp.validate_arms` 硬编码 `cp1`（`run_gtp.py:196-220`）；`__hit_meta__` = `{hit_type,start_t,winner_id,cp1_score,searched}`（`interceptor.py:676-702`）；client `_hit_row` 同字段（`episode_runner.py:58-79`） | §3.8 |
| 库与 H5 对应 | 50 轨迹：远端 pkl 普查 spatial 1,018 条/49 轨迹（一个 task 4 条）、l10 2,640/50，与本地同名 pkl 一致 ↔ 本地 `db/libero_cache/<suite>/*.h5`；S6：`cache_size_<suite>_all_S6.pkl`（spatial 9,813/439，l10 26,493/392）↔ `collect_h5/<suite>/task_N/episode_M.h5` + 本地 `lists_all/` | entry 带 `trajectory_id`（= h5 stem）与 `step_idx` |
| payload 支持他们的档 | `intermediates{0.1..0.9}`；`start_t=0.1` 合法（`types.py:44-46`）；`_warm_start_num_steps(0.1,10)=1` | N_hit=0 → FULL_HIT；N_hit=1 → WARM@0.1 |

## 3. 设计

### 3.1 backbone 输出出口 —— A′（additive 公开 staged 边界，private helper 不动）+ request-aware capture

- `Stage2Output` 增加 `prefix_out: torch.Tensor | None = None`（`[B, 968, 2048]`，backbone 最终层 + final norm 输出）；`Stage2Output.to()` 同步搬运该字段（None 保持 None）。
- 新公开方法 `PI0Pytorch.run_stage2_capture(stage1) -> Stage2Output`：与 `_stage2_llm_backbone` 完全相同的 `paligemma_with_expert.forward(...)` 调用（`pi0_pytorch.py:548-554` 同参），只是保留返回的 `prefix_output`（HF forward 本来就算出的 `last_hidden_state`，`gemma_pytorch.py:102-112`；capture 不增加任何计算，只是持有引用）。**`_stage2_llm_backbone` 与 `run_stage2` 一字不改**；非回归证据 = 单测：同一 `stage1` 下两者 KV 逐位相等。
- **传输面**：`stage_io.stack_stage2_output` / `split_stage2_output` 携带 `prefix_out`（全 None → None；全非 None → 沿 dim 0 cat / split；混合 → `ValueError`）。
- **request-aware capture（coordinator）**：stage-2 批按 stage 同质但可混 bundle（`batching_coordinator.py:788-790, 826-830`），且同一 `bundle_id` 可被热替换而已有连接仍持旧 wrapper（`websocket_policy_server.py:571-579, 798-814`），所以 capture capability 不能放在可覆盖的全局 bundle 注册表中：
  - `StageRequest` 增加不可变字段 `requires_stage2_capture: bool = False`；`submit_to_stage(..., requires_stage2_capture=False)` additive 透传。默认 False 保持所有现有 caller 行为。
  - `InferenceInterceptor.__init__` 从本 wrapper 实际持有的 orchestrator 冻结 `self._cp2_only = orchestrator is not None and orchestrator.has_checkpoint(CP2)`；提交 Stage 2 时传 `requires_stage2_capture=self._cp2_only`。该值属于连接 wrapper 的配置快照，不随同名 bundle 后续替换而改变。
  - `_run_batch(stage_id=2)`：`capture = any(r.requires_stage2_capture for r in batch)`；真则调 `run_stage2_capture`，否则调原 `run_stage2`。split 后所有 shard 都带 `prefix_out`，只有 CP2 wrapper 读取；混合批中的 CP1 请求只多持有一个 `[1,968,2048]` bf16 引用（约 4 MB/请求，随该请求 Stage2/Stage3 生命周期释放），不增加模型算子。
  - 同 id 热替换语义沿现有 wrapper snapshot：旧连接继续执行旧配置，新连接绑定替换后的配置；两者的请求各自携带真实 capability，in-flight 批也不受注册状态竞态影响。non-concurrent 直连仍按本 wrapper 配置绑定 `run_stage2_capture`。
  - 测试：无 CP2 startup config 后 hot-load CP2；CP1/CP2 同批；全 False 时精确调用 `run_stage2`；同一 bundle id 在旧连接存活时 CP2→CP1、CP1→CP2 两向替换，新旧连接同时发请求；用 barrier 固定替换瞬间的 in-flight 批，断言每个 CP2 请求均有 `prefix_out` 且 CP1 数值逐位不变。
- **CP2 × `StageDeviceConfig` 交叉校验的层级**：`config.py` 看不到 CLI 的 `StageDeviceConfig`，所以该规则**不在** R-CP2；放在同时看到两者的 serving 装配边界：`serve_policy.py` 连接策略工厂（`:467` 签名含 `stage_config`，两处调用 `:537-544` / `:599-606`）新增 `_validate_cp2_stage_placement(cache_config, stage_config)`：cp2 ⇒ `stage2 != "meta"` 且 `stage3 != "meta"`（MISS/WARM 需要 stage3），违反即 `ValueError`（启动期立即失败；hot-load 的 bundle 在首次 `_bind_bundle` 时失败并回错）；`InferenceInterceptor.__init__`（`interceptor.py:187, 242`）作第二道断言。
- 不用 hook（B）：理由同 R1。

### 3.2 新 KeyBuilder：`cp2_vlm_ternary`

- `collect(CP2, stage2=...)`：取 `stage2.prefix_out`（stage2 设备上）；`build(CP2)`：flatten `[1,982,464]` → 固定稀疏三值投影 → `{"vlm_out": [500] CPU float32}`。`collect(CP1|CP3, ...)` 抛 `ValueError`（本 builder 只服务 CP2；配置校验保证不会被调用）。
- 投影：`R ∈ {−1,0,+1}^{500×D}`，每行 `floor(pD/2)=9,912` 个 +1、9,912 个 −1（p=0.01；D12），位置由 `seed` 无放回均匀采样；实现为两个 `int32[500, 9912]` 索引表，数值契约固定为 `k = h[idx_pos].sum(-1, dtype=torch.float32) − h[idx_neg].sum(-1, dtype=torch.float32)`，即输入可为 bf16/fp16，但两侧均以 float32 累加并在 float32 中相减，随后只做一次 D2H。offline 建库、online builder 与 verifier 共用这一函数；`projection_meta() = {seed, d, p, D, nnz_per_sign, accumulation_dtype: "float32", digest(sha256 of both index tables)}`。
- 注册：`types.VLM_OUT="vlm_out"` 入 `CACHE_QUERY_FIELDS`；`config.py:1667` builder 类型白名单加 `cp2_vlm_ternary`；`_build_key_builder`（`:3223`）分支。

- **投影资源所有权**：`build_per_connection_components()` 为每个 WebSocket 连接新建 KeyBuilder（`config.py:2974-2986`），两张 `int32[500,9912]` 索引表约 39.6 MB，不能按连接复制。设计：`cp2_vlm_key_builder.py` 模块级 **不可变** `ProjectionSpec` 注册表，键 `(seed, d, p, D)`，值 = CPU 索引表 + digest，进程内只构造一次（锁保护）；按 `(spec_key, device)` 惰性缓存设备副本，所有连接的 builder 只持引用，从不写入；artifact 绑定的 expected digest 与在线 builder 用同一个 spec 对象计算。预算：每进程 CPU 39.6 MB + 每 CUDA 设备 39.6 MB（4 replica ⇒ ×4 进程），写入 §8 拓扑验收；生命周期 = 进程（无显式 teardown，注册表大小有界于不同 seed 数）。测试：同 seed 两个 builder 共享同一对象（`is`）；不同 seed 隔离；设备副本复用；digest 与离线建库一致。

### 3.3 CP2 单步生命周期（interceptor / orchestrator / config / judge）

**step counter 语义（保持所有既有配置逐字不变）**：不引入 `primary_checkpoint` 改写守卫。四处 `checkpoint_id == CheckpointID.CP1` 守卫（`orchestrator.py:545-547, 600, 620, 698`）改为 `checkpoint_id in (CheckpointID.CP1, CheckpointID.CP2)`（二者由校验互斥，不会双计）。interceptor 只在 **CP2-only** 配置下把每步无条件的 `check(CP1)` 换成 `check(CP2)`；CP1-only / CP1+CP3 / CP3-only 的调用序列一字不改，因此 CP3-only 依赖的"未配置 CP1 早递增"（`:543-547`）原样保留。冻结坐标表（cycle 1 / cycle 2；`current_step` = search 看到的 `SearchContext.current_step`）：

| 配置 | cycle 1 各检查点 search 看到 | cycle 1 结束计数 | cycle 2 看到 |
|---|---|---|---|
| CP1-only | CP1: 0 | 1 | CP1: 1 |
| CP1 + CP3 | CP1: 0，CP3: 1 | 1 | CP1: 1，CP3: 2 |
| CP3-only | CP3: 1（未配置 CP1 早递增） | 1 | CP3: 2 |
| CP2-only（新） | CP2: 0 | 1 | CP2: 1 |

测试对四种配置 × `step_filter ∈ {all, exact, window}` 断言 search 收到的 `current_step` 与候选集与冻结表一致；前三行与 HEAD 逐值相同。`_state_history_anchor_cp = min(gates)`（`:90-92`）不改，CP2-only 下自动为 CP2。

**config 校验（`_VALID_CHECKPOINTS` 加 `"cp2"`，新增规则组 R-CP2）**：
- `cp2` 与 `cp1` 互斥；`cp2` 与 `cp3` 互斥（CP2 builder 不实现 CP3 collect，且本线不需要）；
- `cp2` ⇒ `key_builder.type == cp2_vlm_ternary` 且 `keys` 只允许 `vlm_out.enabled=True`；反之 `cp2_vlm_ternary` ⇒ 必须且只能配置 `cp2`；
- `cp2` ⇒ `judge.type == threshold`（拒绝 mlp_router / risk_router / dispatch_surface / composite / dump）、`gate.type == always_search`、`routing is None`、`shadow_teacher` 未启用、`collection.export_collect_meta == False`、`write_policy.type == never`；
- （stage 设备规则不在此层：见 §3.1 交叉校验，放在 serving 装配边界。）

**interceptor.infer（`interceptor.py:827` 起）CP2-only 分支**，逐段：
1. Stage 1（不变）→ CP2-only 下 **不调用** `check(CP1)`（条件：orchestrator 配置集合恰为 `{CP2}`，由 `orchestrator.has_checkpoint(CP2)` 暴露）；relocation `stage1.to(stage2_device)`、stage2 meta guard 不变。
2. Stage 2：直连模式下 `self._stage2_fn` 绑定 `run_stage2_capture`；coordinator 模式下按 §3.1 的 bundle 注册决定 capture。
3. `cp_result = check(CP2, stage2=stage2, tokenized_prompt=...)`（key builder 在 stage2 设备上取 `prefix_out`，`build()` 做唯一 D2H）；若 `stage2.prefix_out is None`（coordinator 未注册 capture 的防御分支）⇒ `RuntimeError`，不静默 MISS。step counter 按上表在 `check(CP2)` 内递增一次。
4. 分支（与 CP1 分支同体，不含 hit_override / routing / shadow / collect_meta 路径，校验已排除）：
   - FULL_HIT：`cached_action = payload.action_chunk` → `broadcast_action(cached_action)` → `buffer_for_write`（write never ⇒ orchestrator 内 decline）→ `_unbatch_outputs` → `__hit_meta__` → `clear()` → return。跳过 Stage 3。
   - WARM_START：relocation `stage2.to(stage3_device)`、stage3 meta guard 不变 → `run_stage3_from(stage2, intermediates[0.1], 0.1)` → 后续同 MISS。
   - MISS：`run_stage3(stage2, noise, num_steps)`（intermediates 采集在 write never 下不需要，沿用现有 `return_intermediates` 逻辑不改）。
   - MISS/WARM 之后：`broadcast_action(stage3.action_chunk)`；**不做** CP3 check（cp3 已被互斥禁止）；`buffer_for_write`（decline）；`__hit_meta__`；`clear()`。
5. `__hit_meta__`（§3.8）与 per_step 行来自 `cp_result`；CP1 路径的 `cp1_result` 变量与语义不变（CP2 分支是并列分支，不重命名既有代码）。

**判定（judge）**：`ThresholdJudge` 阈值表加 `CheckpointID.CP2`（factory `config.py:3418` 传 `cp2_threshold=cfg.threshold`）；warm_tiers 对 `checkpoint_id in (CP1, CP2)` 生效（`judge.py:318`）。

### 3.4 真实 config schema 与 YAML（load-and-assert）

新增/修改 dataclass：`KeysConfig.vlm_out: KeyFieldConfig(enabled=False)`（`_keys_iter` 随之枚举）；`KeyBuilderConfig.cp2_vlm: CP2VlmKeyBuilderConfig(seed: int = 0, d: int = 500, p: float = 0.01)`；`SearchStrategyConfig.task_scoped: bool = True`（§3.6）；`JudgeConfig` 无新字段。

```yaml
key_builder:
  type: cp2_vlm_ternary
  cp2_vlm: {seed: 20260904, d: 500, p: 0.01}
keys:
  vlm_out: {enabled: true, weight: 1.0}
  robot_state: {enabled: false}
backend:
  type: in_memory
  vector_dims: {vlm_out: 500}
  in_memory: {preload_path: /abs/path/cp2_<suite>_<lib>.pkl}
checkpoints:
  cp2:
    enabled: true
    gate: {type: always_search}
    search_strategy:
      type: weighted_score_sum_knn
      top_k: 1
      step_filter: all
      task_scoped: false                       # D14
      field_similarity: {vlm_out: {type: cosine}}
      score_normalization:
        type: per_field
        fields: {vlm_out: {method: affine_clip, params: {lo: -1.0, hi: 1.0}}}
    judge: {type: threshold, threshold: 0.925}                      # N_hit=0：θ_norm=(0.85+1)/2
    # judge: {type: threshold, threshold: 1.5, warm_tiers: [{threshold: 0.925, start_t: 0.1}]}   # N_hit=1
write_policy: {type: never}
```

导出器对每个生成的 yaml 执行 `load_cache_config` → 断言：`checkpoints == {"cp2"}`、builder 类型、`keys` 仅 `vlm_out`、`vector_dims`、strategy 各字段、judge 阈值与 tier 与 export record 逐位一致。测试用真实 `load_cache_config` 做 round-trip，未知 key 只告警的行为由此被检测（断言字段值而不是不报错）。

### 3.5 N_hit 两档的 judge 契约

- N_hit=0 臂：`threshold = θ_norm`，无 `warm_tiers` ⇒ 只可能 FULL_HIT / MISS。
- N_hit=1 臂：`threshold = 1.5`（严格高于归一化分数上界 1.0，FULL 永不触发），`warm_tiers=[{threshold: θ_norm, start_t: 0.1}]` ⇒ 只可能 WARM_START / MISS。
- `θ_norm = (θ_raw + 1)/2`，`θ_raw` 是原始 cosine；export record 同时记录 `theta_raw`、`theta_norm`、`tier` 并断言换算；图与文档一律报 `theta_raw`。
- 档纯度 gate（事后，fail-closed）：N_hit=0 臂 per_step 中 WARM 行数必须为 0；N_hit=1 臂 FULL 行数必须为 0；违反即该臂作废。

### 3.6 `task_scoped`（策略层）

`SearchStrategyConfig.task_scoped: bool = True` → `_build_search_strategy` 透传给 `WeightedScoreSumKnnStrategy(task_scoped=...)`（其余策略不接收，保持签名不变）→ `_build_step_filters(step_filter, step_window, ctx, task_scoped)`：`task_scoped=False` 时三种 `step_filter` 只去掉 `task_key`，`step_range` 逻辑不变（`search_strategy.py:379-406`）。默认 True 时 `_build_step_filters` 输出与现在逐字段相等（测试断言）。orchestrator 不感知该开关。

### 3.7 artifact 身份与图结构

- **ID 策略：继承源 pkl 的 `id`**（`id_policy: "inherited_from_source"` 写进 artifact meta）。于是 `prev_ids/next_ids` 原样有效，一一对应由 `id` 直接给出；`CacheEntry.id` 的"query key 哈希"语义在本 artifact 上不成立，meta 显式声明，`load_artifact` 不重算 id（`in_memory_backend.py:244-352` 无此校验）。
- **`checkpoint_id` 必须改写为 `CheckpointID.CP2`**：backend 按 `entry.checkpoint_id == spec.checkpoint_id` 严格过滤（`in_memory_backend.py:539`，链遍历同样校验 `:1197, 1440, 1526`），保留 CP1 标签会把全库过滤成空、静默全 MISS。`checkpoint_id` 与 `query_keys` 是**仅有的两个**允许随 key 一起变化的 entry 字段。
- artifact meta：`key_builder_type`、`vector_dims`、`projection`（§3.2）、`id_policy`、`source_pkl_sha256`、`h5_manifest={files:[{path, sha256}], digest}`、`model={checkpoint_dir, weights_digest}`、`tokenizer={source, max_len}`、`build_git_commit`。
- 验证器 `verify_cp2_artifact.py`：(a) entry `id` 集合 == 源 pkl（无缺无重）；(a′) 每条 `checkpoint_id == CP2`，任一 CP1 标签即 fail（负例测试）；(a″) 用真实 `InMemoryBackend` 加载后以 CP2 `QuerySpec` 检索非空；(b) 每条 `payload` 张量逐位相等、`trajectory_id/step_idx/outcome/prev_ids/next_ids` 相等；(c) 所有 `prev_ids/next_ids` 指向本 artifact 内；(d) key `vlm_out` 为 finite float32 长 500；(e) `action_chunk` 形状 (10,32)，`intermediates` 含键 0.1 的比例 = 100%；(f) `vector_dims == {vlm_out: 500}`；(g) meta 各绑定字段非空。
- 加载绑定：在 `build_shared_storage`（`config.py:2826`，单一 choke point，覆盖 single 与 concurrent 两个装配入口）新增：配置为 `cp2_vlm_ternary` 时，artifact `projection` 必须与配置的 `{seed,d,p}`、固定的 `accumulation_dtype=float32` 及 builder 现算的 index digest 逐项相等，`id_policy` 必须为 `inherited_from_source`，否则 `ConfigValidationError`。

### 3.8 runner、wire 与 parity

- runner：`exp/gate_threshold_pareto/run_gtp.py` 增加 additive 参数 `--checkpoint {cp1,cp2}`（默认 `cp1`）；`validate_arms`（`:196-220`）改读 `cfg.checkpoints.get(checkpoint)`；`cp2` 时额外断言 §3.4/§3.5 契约。默认值下行为不变（测试）。
- wire：`_build_hit_meta`（`interceptor.py:676`）**additive** 增加 `checkpoint: "CP1"|"CP2"|None` 与 `score: float|None`：CP1 时 `score == cp1_score`，CP2 时 `cp1_score: None`，cache-off / `cp_result is None` 时两者均为 None；client `_hit_row`（`episode_runner.py:58`）additive 增加同名两列；`aggregate.py` 仅对 CP1/CP2 行按 `checkpoint` 选成本表，None 行按 full-inference anchor 处理。旧字段全部保留，旧消费者忽略新增字段。
- **parity 协议（离线、精确、无需 server）**：对每个库抽样 200 步：(i) 离线路径 = H5 `vision_*/prompt_emb` 重建 Stage 1 → `run_stage2_capture` → builder；(ii) 在线等价路径 = H5 `input_images` + `task` attr + `robot_state` 重建模型 `Observation`（tokenize 沿 `_build_fake_stage1_with_masks` 的 tokenizer 与 state 串规则）→ 真 `run_stage1` → `run_stage2_capture` → builder。断言 key 的 cosine ≥ 0.999（fp16 存储的 `vision_*` 为唯一差异源），并报告最大偏差；不达标则建库改用路径 (ii)。作为 manual GPU 测试与 §8 步骤 2 的验收门。

### 3.9 打分、shadow 分布与切点（D11：GST K=1，IR 寻址）

- 检索：单字段 cosine、`affine_clip` 归一（§3.4）；`task_scoped: false`（D14）。
- **shadow 表离线建**：`build_shadow_table.py` 用 `exp/rit_pareto` 同一 150 集 dev cohort（H5 在 weilandserver `/tmp/dsp_shared/rit_pareto/<suite>/h5/`，唯一副本）：每决策重建 Stage 1 → backbone → CP2 key → 对 CP2 库（同 `task_scoped:false`）取 top-1 cosine `s`。输出 `{suite, lib, rows:[{episode, step, s}], cohort_manifest_sha256, library_sha256}`。
- IR 寻址（K=1）：`IR_a(θ) = [n(s≥θ)·c_a + n(s<θ)·c_miss] / (N·c_miss)`；目标阶梯 {60,65,…,95}（N_hit=0 可达下界 56.2%，N_hit=1 60.6%）；每目标取最近可达分位切，记 `predicted_ir / ir_gap`（|gap|>1 pt 的目标省略）。
- 固定参考臂：他们的默认 `θ_raw = 0.85`（两档各一臂），替代不可复算的"肩部"。
- 每组臂数：N_hit=0 ≤ 8 + N_hit=1 ≤ 7 + 参考 2 = **≤ 17 臂 ≤ 8,500 集**。

### 3.10 成本口径

- 主轴 = **解析 model-forward IR**（与 `exp/rit_pareto` 同口径，逐步 verdict 计数 × 单价）；CP2 单价表（CUDA-Graph 档）：FULL@CP2 = s1+s2 = 37.95 ms、WARM@0.1 = 40.91、MISS = 67.52；同时给 eager 档单价表（E0 表两档均有）算第二组 IR，两档并列。文档不使用"实测 GPU 成本"称呼解析量。
- **CP2 开销实测列（新 harness）**：`exp/cache_latency_bench` 是"不加载模型、仅驱动 CP1、`cp1_build` 无 D2H"的回放基准（其 runbook 自述），不能驱动需要 `prefix_out` 的 CP2 builder。新建 `exp/actioncache_baseline/bench_cp2_overhead.py`：在 weilandserver 加载真模型，回放 shadow cohort H5（重建 Stage 1 → `run_stage2_capture` 得到 GPU 上的 `prefix_out`），随后以单连接入口 `components = build_cache_components(config)` 装配真实 `CacheOrchestrator`（直接使用该 dict 的 storage/key_builder/gates/judges/search_strategies；不再二次调用 `build_per_connection_components`）并执行 `check(CP2, stage2=...)`；计时边界 = `torch.cuda.synchronize()` 包住整个 `check()`，内部分段用 `SystemTimer` 探针 `cp2_collect / cp2_build（含稀疏投影 + 唯一 D2H）/ cp2_search（全库 filter + cosine，task_scoped:false）/ cp2_judge（归一化 + 阈值）`。输出契约：每决策 CSV（segment ms）+ `overhead.json {suite, library, library_sha256, config_digest, hardware {gpu, driver, torch, cuda}, n_decisions, cold: first 50 决策 {median, p95}, warm: 其余 {median, p95}, per_segment 同}`，4 个库各一份。验收/回退：warm total P95 ≤ 10 ms（≈ FULL@CP2 单价的 25%）⇒ 直接报告并给 "IR + 开销/决策" 敏感性；10–40 ms ⇒ 报告并在图注标注；> 40 ms（≥ FULL@CP2 单价）⇒ 停止发臂，先分析 `cp2_build/cp2_search` 分段；仅当 profile 证明 backend 为瓶颈且现有批量矩阵路径未覆盖该形状时才补 additive backend path，再重测。测试：假模型 + 3 条 entry 库的 dry-run 产出 schema 合法的 CSV/JSON。
- 端到端 latency 分位 / 吞吐 serving bench：不在本 plan（Review Log R1-B9，R2 已接受）。

### 3.11 统计与验收门（冻结）

- **随机性事实**：conductor 只固定 A-pool 的 episode 身份 `(task_id, init_idx)`；client 不发 `noise`（`packages/openpi-client/.../websocket_client_policy.py`、`examples/libero/episode_runner.py` 均无 `noise`），server 端 flow-matching 初始噪声由 `InferenceInterceptor.infer(noise=None)` 交给模型从**全局 RNG** 采样（`interceptor.py:900-905`），`serve_policy.py` 无 seed 选项，且并发调度改变采样顺序。⇒ **action noise 不配对、不可复现**；既有 `exp/rit_pareto` 结果与本实验无法共享逐集噪声实现。
- **配对声明**：两侧只在 `(task_id, init_idx)` 上 block/pair；噪声作为未配对随机效应处理。provenance 记录 server git commit、torch/CUDA 版本、replica 数与 worker 数。
- **CI**：每臂 SR Wilson 95%；臂间 ΔSR 与成本 IR 用按 task 分层的 episode 级 bootstrap（B=2000，两侧独立重采，不做逐集配对差）；IR 在每个 replicate 内按 ratio-of-sums 重算。
- **跨线对照规则（50 库组）**：定义 `ΔSR = SR_CP2 − SR_RIT-reference`。点估计：对每个 CP2 臂的实测 IR `x`，取 `exp/rit_pareto` K=2 no-gate 上凸包在 `x` 处的线性插值；`x` 超出观测参考支撑区间则不比较，只画点。区间：每个 bootstrap replicate 按 task 生成一套 CP2 episode 重采索引和一套独立的 RIT cohort 重采索引；后一套索引共同应用于全部 RIT 参考臂，以保留同 cohort 臂间相关性。随后重算 CP2 的 `(x_b, SR_b)`、RIT 各臂 `(IR_b, SR_b)` 与该 replicate 的上凸包，再在 `x_b` 插值得 `ΔSR_b`。若 `x_b` 不在该 replicate 的参考支撑区间，记 `support_miss`；miss 比例 >1% 时该臂不作三分裁决，只报 descriptive point/两侧 CI，否则以有效 replicates 的 percentile 95% CI 裁决：下界 >0 ⇒ ActionCache 式更高；上界 <0 ⇒ RIT 更高；其余 ⇒ 噪声内不可分。S6 组只作 regime 描述，不做跨线插值。该协议只重采既有 RIT 原始结果，不增加 rollout。
- **库组成表（每库必报）**：来源采集与 init 池、轨迹数、entries、成功/失败轨迹数、每 task 最少轨迹数、horizon；并声明 50 库与 S6 库非嵌套。
- 完整性 gate（每组，fail-closed）：journal 终态行数 = 唯一 uid = 臂数×500；0 dup；per_step attempt 集合 == journal attempt；`failed` 且 `client_timing.steps < 上限` 判截断 ⇒ 剔除补跑（沿 `audit_k3_group.py`）；档纯度（§3.5）；server 加载的 `library_sha256` == export record；预测 IR 与实测 IR 差逐臂报告。
- 自动测试：emitter（IR 阶梯 → θ 反解 → yaml → record 一致）、validator（§3.7 每条一个负例）、aggregate/frontier（非支配集、Wilson、两侧分层 bootstrap、RIT cohort 共享重采索引、逐 replicate hull、support-miss 与三分裁决）、shadow 表 schema、overhead bench schema。

## 4. 涉及文件

| 文件 | 改动 |
|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py` | `Stage2Output.prefix_out` 可选字段 + `.to()` 搬运；新公开 `run_stage2_capture`；`_stage2_llm_backbone` / `run_stage2` 不动 |
| `src/openpi/serving/stage_io.py` | `stack/split_stage2_output` 携带 `prefix_out` |
| `src/openpi/serving/batching_coordinator.py` | `StageRequest/submit_to_stage.requires_stage2_capture`；`_run_batch(stage 2)` 按批内 request capability 决定 capture（`:826-830`） |
| `scripts/serve_policy.py` | `_validate_cp2_stage_placement`（工厂 `:467`，调用点 `:537-544`/`:599-606`）；直连模式 `_stage2_fn` 绑定 |
| `src/openpi/cache/types.py` | `VLM_OUT` + 白名单；CP2 注释 |
| `src/openpi/cache/components/cp2_vlm_key_builder.py`（新） | `CP2VlmTernaryKeyBuilder` |
| `src/openpi/cache/config.py` | `KeysConfig.vlm_out`；`CP2VlmKeyBuilderConfig`；`SearchStrategyConfig.task_scoped`；`_VALID_CHECKPOINTS`；R-CP2 校验组；builder 白名单与 `_build_key_builder`；`_build_search_strategy` 透传；judge factory `cp2_threshold`；`build_shared_storage` projection/id_policy 绑定 |
| `src/openpi/cache/orchestrator.py` | 四处守卫改 `in (CP1, CP2)`；`has_checkpoint()`；`:218` 遍历 cp2；`check()` 文档 |
| `src/openpi/cache/components/judge.py` | 阈值表 CP2；warm_tiers 对 CP1/CP2 |
| `src/openpi/cache/components/search_strategy.py` | `_build_step_filters` `task_scoped`；`WeightedScoreSumKnnStrategy` 参数 |
| `src/openpi/cache/interceptor.py` | CP2-only 并列分支（不改 CP1 分支）；直连 `_stage2_fn` 绑定；`__init__` 的 CP2×stage 设备断言；`__hit_meta__` additive 字段；`cp2_check` 探针 |
| `examples/libero/episode_runner.py` | `_hit_row` additive `checkpoint`、`score` |
| `exp/gate_threshold_pareto/run_gtp.py` | `--checkpoint` additive |
| `exp/actioncache_baseline/`（新，按 `artifact_layout.md`） | `build_cp2_artifact.py`、`verify_cp2_artifact.py`、`parity_check.py`、`build_shadow_table.py`、`export_arms.py`、`bench_cp2_overhead.py`、`aggregate.py`、`config/`、`analysis/` |
| `tests/cache/`、`tests/serving/`、`tests/actioncache_baseline/` | §7 |
| `docs/architecture/cache_system.md` §3 CP2、`docs/experiments/actioncache_baseline.md`（新）、`docs/README.md`、`logs/README.md` | 索引同步 |

## 5. 接口

- `Stage2Output(stage1, past_key_values, prefix_out=None)`；`run_stage2_capture(stage1) -> Stage2Output`。
- `StageRequest(..., requires_stage2_capture=False)`；`BatchingCoordinator.submit_to_stage(..., requires_stage2_capture=False)`；`CacheOrchestrator.has_checkpoint(cp)`。
- `CP2VlmTernaryKeyBuilder(seed, d=500, p=0.01)`：`collect(CP2, stage2=...)`、`build(CP2) -> {"vlm_out": Tensor[500]}`、`projection_meta()`。
- `WeightedScoreSumKnnStrategy(..., task_scoped=True)`。
- `ThresholdJudge(cp1_threshold, cp3_threshold, warm_tiers, cp2_threshold=None)`（None ⇒ 沿用 cp1 值；factory 显式传）。
- `__hit_meta__` += `checkpoint`, `score`（cache-off 均为 None）；per_step 行 += 同名两列。
- YAML：§3.4。

## 6. 集成点

- server：`serve_policy.py --cache_config <cp2 yaml>`，conductor 按臂热切 bundle，`--replicas 4` concurrent 路径经 §3.1 传输面。
- 离线建库 / shadow 表 / parity 在 weilandserver（4090，pi05 ckpt、H5 均在本机）；S6 l10 26.5k 条 eager 约 45 min。
- 图：`exp/rit_pareto/build_figure.py` 新 figure spec（指针引用我们侧 50 库前沿）。

## 7. 测试策略

**单元**：投影索引表（确定性、每行 ±1 各 9,912、无重复、digest）；稀疏实现对低维 dense-float32 oracle 逐值一致，真实 bf16 输入仍以 float32 累加；offline/online/verifier 调用同一 projection 函数；`build()` 输出；`Stage2Output` 旧构造兼容 + `.to()`；`run_stage2` vs `run_stage2_capture` KV 逐位相等；`stack/split_stage2_output` 三态（全 None / 全有 / 混合 raise）+ 与 `prefix_out=None` 时 byte-identical；`_build_step_filters` `task_scoped` × {all, exact, window} 六格；config R-CP2 每条规则一个负例 + 正例 yaml round-trip 字段断言；judge 对 CP2 三态、N_hit=1 契约（threshold 1.5 永不 FULL）；orchestrator `check(CP2)` 端到端（假 storage）、step counter 恰增一次；`build_shared_storage` projection / id_policy mismatch fail-fast（两个装配入口各一）；ProjectionSpec 注册表共享 / 隔离 / 设备副本复用；artifact 任一 CP1 标签 entry 拒载（验证器负例）+ 真实 backend CP2 检索非空；step 坐标冻结表（§3.3 四配置 × 三 filter，前三配置与 HEAD 逐值相同）。
**inference path（staged API，非 GPU 假模型 / GPU manual 两版）**：request-aware coordinator capture（无 CP2 启动后 hot-load CP2；CP1/CP2 同批；全 False 时 `run_stage2` 精确调用；同 bundle id CP2→CP1 与 CP1→CP2、旧连接/新连接并存及 barrier 固定的 in-flight 批）；`_validate_cp2_stage_placement` 对 stage2/stage3 meta 各一负例；CP2-only 配置下 `infer()` FULL/WARM/MISS 三分支 × {non-concurrent 直连, BatchingCoordinator}：每步恰好一次 step increment / 一次 `clear()` / 一次 `broadcast_action`；不调用 `check(CP1)`；stage2/stage3 meta guard 行为；relocation 顺序；`__hit_meta__` 字段；cache-off 明确产出 `checkpoint=None, score=None` 且旧消费者可读；CP2+CP1、CP2+CP3、非法 judge/gate/routing 组合被拒；legacy CP1/CP3 配置的旧字段与 verdict 值非回归（对照现有 fixture 逐字段）。
**实验工具**：emitter / validator / parity 断言 / aggregate & frontier（Wilson、两侧分层 bootstrap、逐 replicate 重建 hull、support-miss 与三分裁决）/ shadow 表 schema / overhead bench dry-run schema / `run_gtp --checkpoint` 默认非回归。
**Verify**：`uv run pytest` 全量；GPU manual：parity 200 步 × 4 库、`run_stage2_capture` 数值对拍。

## 8. 执行顺序与预算

1. Code（§3.1–3.8）→ G2 → Verify。
2. 建 4 个 CP2 库 → `verify_cp2_artifact` 全过 → parity 门（§3.8）。
3. shadow 表离线建（4 组，零 rollout）→ `export_arms`（每组 ≤17 臂）→ yaml load-and-assert。
4. 4 组 ≤17 臂 × 500 集 ≤ 34,000 集；吞吐按 `exp/rit_pareto` 实测（spatial ≈ 95 集/min、l10 ≈ 30 集/min）⇒ 4 组串行 ≈ 13 h。每组跑完执行 §3.11 完整性 gate。
5. CP2 开销实测（`bench_cp2_overhead.py`，4 库各一次；按 §3.10 阈值决定是否先修 backend）；聚合、出图、结果文档 `exp/actioncache_baseline/analysis/`。
6. 拓扑验收：每 replica 进程投影资源 +39.6 MB CPU +39.6 MB GPU（§3.2），4 replica 合计 < 0.2 GB GPU，纳入 server 启动显存检查。

## 9. 风险登记

| 风险 | 影响 | 对策 |
|---|---|---|
| H5 `vision_*` 为 fp16 存储，离线重建与在线 Stage 1 有舍入差 | key 偏差 | §3.8 parity 门（cosine ≥ 0.999），不达标改用原始图像路径建库 |
| S6 l10 26.5k 条检索延迟 | 若 backend 路径退化，CP2 overhead 可吞掉模型收益 | 建库后先跑 `bench_cp2_overhead.py` 并看分段；仅在 profile 证明现有批量矩阵路径未覆盖时补 additive backend path |
| coordinator 路径漏传 `prefix_out`（尤其热替换/in-flight） | 4 replica 下 key 为空 | §3.1 request capability + interceptor 侧 `prefix_out is None ⇒ RuntimeError`（不静默 MISS）+ 同-id/in-flight 集成测试 |
| 投影索引按连接复制 | 每连接 39.6 MB × N | §3.2 进程级不可变注册表 + 设备副本缓存，预算入 §8 |
| action noise 不配对、不可复现 | 臂间差含噪声方差 | §3.11 按 task 分层 bootstrap，不做配对声明 |
| padding 位置 hidden state 进 key | 与他们一致，但对 prompt 长度敏感 | 记录；不做 mask 版 |
| `pD/2` 取整、种子 | 复现细节 | 元数据全记（D12） |
| horizon 5 vs 10（D8）、无在线续入库、任务内 vs 全库检索差异（D14 已对齐） | 与原文绝对数字不可比 | 只作本项目同 teacher / 同评测池下的内部对照，结果文档披露 |
| S6 两组无我们侧对照（D11）；50/S6 两库来源、成功组成、容量同时不同 | 不能称同库对照，也不能归因于库规模 | §1/§3.11：descriptive library-regime comparison + 每库组成表，声明 size × composition 不可分 |

## 10. 待 owner 决策

无。§10 六项已裁（D9–D14）。

## 11. 文档义务

`docs/architecture/cache_system.md` §3 CP2 节改写（active：post-backbone single-key；生命周期与传输面）；新 runbook `docs/experiments/actioncache_baseline.md`；`docs/README.md`、`logs/README.md` 同 commit 同步；`docs/iclr/actioncache_response_plan.md` §4 第 3 条弹药措辞更正（B.3 有总括不相交声明，"全文无一字"不成立）。

## 12. §6 Verify 记录（2026-09-04）

裸全量 `uv run pytest`（无 `--ignore`，含 `tests/review_tests/`）：**5034 passed, 60 skipped, 15 failed**，1013 s。15 条失败逐条溯源，无一属于本改动：

- 13 条在 HEAD 的 `git archive` 导出树上（`PYTHONPATH` 指向导出的 `src/`，`tests/review_tests/` 原样拷入）以相同错误复现，即 HEAD 既有：`tests/dispatch_surface/test_rit_pl.py::test_sonly_note_compiles`（缺 `exp/rit_pareto/analysis/analysis.md`，该文件由另一 session 删除 / 待其 commit）、`tests/exp/test_prebuilt_matrix_backend.py` 两条（`assert False`，与本线无关的快路径位等价）、`tests/review_tests/test_cache_size_g2.py` 三条（A-pool 数据 / `verify_launch_binding` 签名漂移）、`tests/review_tests/test_groot_robocasa_g2.py` 一条（无 `robocasa` 模块）、`tests/review_tests/test_rl_router_g2_contracts.py` 两条（CLI 子进程 rc=1；再导出子进程 `No module named 'exp'`）、`tests/review_tests/test_ws2_g2_round1.py` 四条（argparse `Namespace` 缺 `teacher` / `worker_home`）。
- 2 条（`tests/robocasa365/test_robocasa_policy_config.py` 的 `test_data_config_asset_id_and_quantile_norm` / `test_data_transform_chain_shape`，`FileNotFoundError: /tmp/pytest-of-…`）是本次并行启动第二个全量 pytest 造成的 basetemp 轮换竞争；单独复跑 6 passed。
- 本线 blast radius 内的包全部 green：`tests/actioncache_baseline` 78 passed；`tests/cache` 1543 passed；`tests/serving` 113 passed；`tests/gate_threshold_pareto` / `tests/conductor/test_episode_runner.py` / `tests/libero` / `tests/models_pytorch` / `tests/scripts` / `tests/ablation_study/test_router_hooks.py` / `tests/dispatch_surface/{test_block_variance_probe,test_precheck_analyzer}.py` 210 passed；ruff 在本线全部文件零告警（`in_memory_backend.py` 的 F401/E741 为 2026-04 既有，不在本 diff）。
- 未做：真模型 GPU parity（§8 步骤 2，200 步 × 4 库）与 §8 步骤 3–6 属实验运行，不在本 commit。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-04 18:49 CDT

- [Blocking] [Concern] CP2 的 WARM_START / MISS 尾部不得再调用 `check(CP3)`；把 post-inference CP3 分支限定为非 CP2 路径，并将 direct / coordinator 两条 CP2 测试都冻结为每决策只调用一次 `check(CP2)`。— reasoning: 冻结计划 §3.3 明确规定 CP2 与 CP3 互斥且“**不做** CP3 check”，架构文档也作同样声明；但 `interceptor.py:1238-1252` 对任意非空 orchestrator 无条件 probe CP3，`test_acb_interceptor_cp2.py:139` 还把 `[CP2, CP3]` 固化为期望值。真实 orchestrator 虽因 CP3 未配置而早退，仍产生计划外调用与 `cp3_sum` 计时污染。
- [Blocking] [Concern] 为 `run_gtp.validate_arms(..., checkpoint="cp2")` 实现计划 §3.4/§3.5 的完整 CP2 契约校验并逐字段补负例：至少 exact `{cp2}`、builder / 唯一 key / vector dims、`top_k=1`、`step_filter=all`、`task_scoped=false`、单字段 cosine + `affine_clip(-1,1)`、always-search、N_hit 两档 judge 形状与 `write_policy=never`；不能只把原 CP1 validator 的取值键改成 CP2。— reasoning: 独立反例把 emitter 生成的 n0 YAML 改成 `top_k: 2` 后，`validate_arms(..., checkpoint="cp2", eval_gate="always_search")` 仍返回成功；这会让手改、损坏或旧版 arm 在发出 rollout 前绕过预注册协议，而 runbook/架构文档均声称 runner 会断言该契约。
- [Blocking] [Concern] `export_arms` 必须机械执行每组 `N_hit=0 ≤ 8、N_hit=1 ≤ 7、参考臂 2、总数 ≤17` 的冻结预算，并在 record 中对因档下界或预算省略的目标给出原因；增加默认参数下的上限测试。— reasoning: 当前两档都遍历默认 `{60,65,…,95}` 并各追加参考臂，没有档级上限。独立使用 10,000 个均匀 shadow score 调默认 exporter 得到 18 臂（n0=9、n1=9、0 skipped），直接超过 §3.9/§8 的 ≤17 臂与 ≤8,500 episode/group 预算。
- [Blocking] [Concern] 闭合 `bench_cp2_overhead.py` 的可测量输出契约：harness 必须强制启用并注册真实 CP2 timer，CSV 必须逐 decision 写出各 segment ms，JSON 必须含 `suite`，并用 dry-run/schema 测试证明 `cp2_collect/build/search/judge` 均有非零样本；`>40 ms` 后先按 segment 定位，不能在无 profile 时预判 backend。— reasoning: emitter 固定写 `timer.enabled: false`（`export_arms.py:62`），bench 原样 `build_cache_components(config)` 且不覆盖该值，因此 `SystemTimer.measure()` 全是 no-op，`per_segment` 只能是 count=0/None；其 CSV 也只有 `episode,step_idx,total_ms`，record 没有计划要求的 `suite`。这使 §3.10 的四库开销列与回退判定事实上不可执行，而现有测试只测 `verdict_for()` 三个阈值，未运行 harness/schema。
- [Blocking] [Concern] 把 `stats.pareto_frontier` 改为冻结协议要求的参考**上凹包/upper concave hull**（同 cost 先保留最高 SR、去支配点、再按斜率单调性剔除低于弦的点），并让点估计和每个 bootstrap replicate 共用该实现；增加低于弦的非支配点反例。— reasoning: 当前函数只保留 cost 递增时 SR 创新高的非支配点。独立输入 `[(0,.5),(1,.51),(2,.9)]` 得到三点并在 x=1 插值为 .51，但真正上包络应剔除中点、在 x=1 为 .70；因此现有 `ΔSR` 点估计与 bootstrap 均可能系统性压低参考线，违反 §3.11 的跨线 estimand。
- [Blocking] [Concern] 实现 §3.11 的完整 fail-closed 聚合门，且 `--allow-partial` 只能放宽 episode 数量、不能绕过身份完整性：核对 export-record 的完整 arm 集、journal 终态总行数与唯一 uid、0 duplicate、每个 accepted journal attempt 与 per_step attempt 集合严格相等、截断 failed 的剔除/补跑、以及实际 server `library_sha256 == export_record`；每项各放一个负例。— reasoning: `load_episode_ledger` 用嵌套 dict 静默覆盖重复终态，aggregate 只遍历实际出现的 arm，未检查 attempt 集合、截断或 library digest。独立构造同一 accepted terminal row 重复两次、仅一组 per_step 且 export digest 明显错误的 run，`aggregate(..., expect_episodes=1)` 仍成功返回 n=1；损坏/混库/漏臂 rollout 会被当成完整结果发布。
- [Blocking] [Concern] 将 artifact / shadow 的模型与载荷 provenance 做成可证伪绑定：artifact 必须提供计划命名的 `model.weights_digest`，该摘要须覆盖全部权重内容；shadow/overhead（以及 parity 记录或显式 artifact 参数）须核对所加载 checkpoint 与 artifact model identity；verifier 还须严格断言 `action_chunk.shape == (10,32)`，不能只要求全库形状彼此一致。— reasoning: `checkpoint_identity()` 只哈希每文件首 1 MiB 并明确假设“swapped checkpoint changes sizes or heads”；独立修改 2 MiB 权重文件最后一个字节后 identity 完全不变，且当前 meta 根本没有 `weights_digest` 字段。`build_shadow_table` 只记录 checkpoint 路径、不与库的 `model` 比较；verifier 对任意一致的错误 shape 也会通过。这些都弱于 §3.7 的绑定与 fail-closed 约定，可能让库 key、shadow 阈值和运行模型来自不同权重却无告警。
- [Blocking] [Concern] 补齐 G1 冻结的核心 staged/integration 证据，而不只测试局部布尔位：用同一 mocked/真实 backbone forward 对拍 `run_stage2` 与 `run_stage2_capture` 的 KV 逐位相等；覆盖 WebSocket 同 bundle id CP2→CP1 / CP1→CP2 热替换时旧、新 wrapper 与 in-flight 请求；CP2 FULL/WARM/MISS × direct/coordinator 均断言一次 step increment / clear / broadcast、无 CP1/CP3 check。— reasoning: 现有 coordinator 测试只直接提交 `[True,False]` flag，并未经过 bundle 发布/连接绑定；现有 interceptor coordinator stub 只看 flag 是否透传；没有任何测试对拍两个真实 staged 方法的 KV。§3.1 与 §7 把这些场景列为 request-aware capture 能放行的关键非回归证据，当前 71 个目标测试未提供该证据。

### G2 Round 1 — Executor — 2026-09-04

- Accepted — CP2 尾部不再 probe CP3：`interceptor.py` post-inference 块以 `not self._cp2_only` 守卫 CP3 调用（含 `cp3_sum` 计时）；`test_acb_interceptor_cp2.py` 三档 + coordinator 路径全部冻结为 `checks == [CP2]`；新 `test_acb_staged_integration.py::test_cp2_decision_cycle_per_tier[direct|coordinator]` 用真 orchestrator + 真 CP2 组件断言每决策恰一次 `check(CP2)`、step counter +1、一次 broadcast / clear。
- Accepted — runner 完整契约：新 `libs.cp2_contract_problems(cfg)` / `cp2_tier_of_config(cfg)`（checkpoints=={cp2}、builder、唯一 key、vector_dims、`top_k=1`、`step_filter=all`、`task_scoped=false`、trajectory_depth 1、单字段 cosine、per_field `affine_clip(-1,1)`、always_search、threshold judge 两档形状、write never、无 routing/shadow/collect_meta）；`run_gtp.validate_arms(checkpoint="cp2")` 逐臂执行并核对臂名档位与 suite 标签；导出器 `assert_arm_yaml` 改为同一实现 + record 绑定。反例 9 条（`top_k=2`、`step_filter=exact`、`task_scoped=true`、l2、归一化区间、n0 带 tier、n1 FULL 可达、tier start_t=0.3、双 tier）+ 臂名/档位/suite 不一致 3 条，均 `SystemExit`。
- Accepted — 臂数预算机械化：`export_arms.plan_tier_targets` 执行 n0 ≤ 8 / n1 ≤ 7（+ 各 1 参考臂）与每组 ≤ 17 的硬上限，省略目标逐条记 `reason ∈ {below_tier_floor, no_cut_within_max_gap, duplicate_cut, tier_budget}`；低于档下界的目标不再被 `max_gap` 容差救回；record 增 `budget` 字段，超限 `SystemExit`。测试复现 R1 反例（10,000 均匀 shadow 分 + 默认参数）得 8+7+2=17，且 n1/60 以 `below_tier_floor` 省略；过密阶梯从低 IR 端按 `tier_budget` 丢弃。
- Accepted — overhead harness 闭合：`bench_cp2_overhead.build_orchestrator` 强制 `timer.enabled=True` 并把 monitor level 提到 BASIC（否则 `SystemTimer` 全 no-op），timer 仍禁用即中止；`run_decisions` 每决策 `on_task_begin` / `summary(task_only=True)` 读 orchestrator 探针，CSV 逐决策写 `total_ms` + 六段 `<segment>_ms`，四核心段任一缺记录即中止；JSON 含 `suite`（新必填 `--suite`）、库与模型绑定；verdict `> 40 ms` 改名 `halt_profile_segments`，文档改为"先按分段定位"。dry-run 测试（真组件 + 3 条 entry 库 + 假 stage2）断言六决策每段 count=6、CSV/JSON schema 完整。
- Accepted — 参考线改为上凹包：`stats.reference_hull`（同 cost 取最高 SR → 去支配 → 单调链弦检验剔除凹陷/共线点），点估计与每个 bootstrap replicate 共用；`pareto_frontier` 删除。测试含 R1 反例 `[(0,.5),(1,.51),(2,.9)]` → x=1 插值 .70。
- Accepted — 完整性门：`stats.load_episode_ledger` 遇重复终态行 `SystemExit`；新 `stats.audit_run`（镜像 `exp/rit_pareto/ops/audit_k3_group.py`）检查终态行数 == 唯一 uid、per_step `(uid, attempt)` == journal `(uid, attempt)`、`failed` 且 `client_timing.steps < 上限`（spatial 200 / l10 500）判截断、`failed` verdict 行 < 42/100 判短、臂集合 == export record、每条 verdict 行 `library_sha256` == export record（缺失即报）、每臂集数；`--allow-partial` 只放宽集数。为使"实际 server 库摘要"可证：`CacheOrchestrator.artifact_meta` 只读属性 + CP2 `__hit_meta__` 附加 `library_sha256`（非 CP2 路径不含该键，legacy wire 逐字节不变）+ `_hit_row` 同名列。反例 6 条（dup 终态、缺 per_step 对、多 per_step 对、截断 failed、短 failed、无摘要）+ record 臂集合缺/多 + 摘要不等 + 混库，均 `SystemExit`，且 `allow_partial=True` 下同样失败。
- Accepted — 可证伪 provenance：`libs.weights_digest` 对 checkpoint 目录全部字节做 sha256（替换首 1 MiB 的 `checkpoint_identity`），artifact `model.weights_digest` 写入并由 backend `artifact_meta["model"]` 暴露；`build_shadow_table` / `bench_cp2_overhead` 加载模型前 `libs.assert_model_binding` fail-closed，`parity_check --expect-weights-digest` 同理并记录本机 digest；verifier (g) 要求 64 位 hex digest，(e) 每条 `action_chunk` 严格 == `(10, 32)`（一致的错误形状不再通过）。测试：2 MiB 文件改最后一字节 digest 必变、绑定失败退出、旧 `sha256_head1mib` meta 被拒。
- Accepted — staged / integration 证据：`test_run_stage2_and_capture_issue_identical_forward_and_kv`（同一 mocked backbone forward，kwargs 逐项同一对象、KV 逐位相等、仅 capture 保留 prefix_out）；`test_same_bundle_id_hot_swap_with_in_flight_request[cp2_to_cp1|cp1_to_cp2]` 走真实 `wps._bundles` 注册表 → `serve_policy._wrap_policy` 装配，旧 wrapper 请求阻塞在 stage 2 时同 id 发布替换并绑定新连接，断言两 wrapper 各自冻结的 capture 能力、wire `checkpoint` 与 `library_sha256`；`test_cp2_decision_cycle_per_tier` 覆盖 FULL/WARM/MISS × direct/coordinator 的一次 step 增量 / clear / broadcast、无 CP1/CP3 check、coordinator 侧 warm 走 `Stage3WarmStartPayload`。

### G2 Round 2 — Reviewer — APPROVED — 2026-09-04 19:34 CDT

- 执行方对 G2 R1 八项 blocking 均已按 Accepted 说明落实：CP2 尾部不再 probe CP3；runner / exporter / overhead / upper hull / aggregate / provenance / staged integration 的实现与测试逐项相符。执行方 R1 修订已依 D16 加入 index，未纳入其他 session 的 RIT 图件、脚本或 handoff working diff。
- [Resolved under D16] [Concern] 独立反例发现共享 CP2 契约仍接受 `N_hit=1 threshold=2.0` 与 `keys.vlm_out.weight=2.0`，parity 也可省略 artifact model digest；reviewer 将 N_hit=1 FULL 阈值严格冻结为 `1.5`、要求唯一 key 权重严格为 `1.0`（并补 config / checkpoint enabled 与 brute-force 明示检查），同时令 parity CLI 与直接 `run()` 均强制做全权重 digest 绑定。原反例现返回两条契约错误，新增负例通过。
- [Resolved under D16] [Concern] `compare_to_reference.py` 原可绕过 §3.11 aggregate gate 直接从不完整或混库 raw run 生成可发布 ΔSR；reviewer 改为强制接收同一 `export_record.json`，计算前复用 aggregate 的终态 / attempt / arm-set / library digest / checkpoint / tier-purity 全门，并把 audit 写入 comparison record；错误库摘要反例在读取 reference 前即 fail-closed。
- 独立验证：`tests/actioncache_baseline` 78 passed；`tests/cache tests/serving` 1656 passed, 13 skipped；邻接 gate / dispatch / staged-model 测试 32 passed, 1 skipped；ActionCache 目录 Ruff 全绿；全文件 Ruff 仅命中 `in_memory_backend.py` 两个 2026-04 既有告警（unused `F`、变量名 `l`），不在本任务 diff。GPU 真模型 parity 200 步 × 4 库仍属于后续 Verify，不作为 G2 advisory 测试伪报已完成。
- D16 明确覆盖了 reviewer 不得改源文件、修复后须另开独立 Review、以及本轮全量 stage 的通常规则；因此本轮 verdict 是 owner-authorized self-remediation 的放行，不等同于独立第三方复审。审查者的 6 个实现/文档修订文件、2 个测试文件及本计划记录保持 unstaged，执行方快照保持 staged。
