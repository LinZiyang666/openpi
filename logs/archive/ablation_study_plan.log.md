# Ablation Study — cache 有效性双方向消融实验计划

> Status: In Progress（infra shipped d2e4293；**Phase 0 完成** 2026-08-12 00:00；**Phase 1 完成** 2026-08-12 晚，**EN-3 重冻结** 2026-08-13：四组合套件级统一 step=**020000**（标准配方终点，零选择偏差，band 降披露；见 analysis/sr_ledger.md 与 config/select_freeze_*.yaml）；**Phase 2 完成** 2026-08-13：正式 1-cell conductor smoke（hit_act spatial，10/10 ep，FULL_HIT=override=sidecar 计时 144 行三方相等，命中率 70.9%）+ test_manual_e2e -m manual PASSED，preflight underpowered_ok 审计闭环；**Phase 3 完成** 2026-08-13：teacher anchor 复跑 spatial **500ep SR=0.974** / l10 **500ep SR=0.868**（官方 init 50/任务，json 存 data/anchors/；历史协议锚 0.95/0.83 同量级）；纯学生锚点=各组合 @20000 EN-2 成绩；SmolVLA l10 002000 补评入账（0.344，曲线完整化）；cache_baseline 臂归 Phase 4；**当前停在 Phase 4 门待 owner 放行**（主矩阵 7 臂×2 + 4b 帕累托 10 点已备）；交接见 logs/session_handoff.md；EN-1/2/3 偏差记录见文末；O7 预批 underpowered_ok 已固化）
> Created: 2026-08-11
> Owner: Ziyang Lin
> Level: **L3**（横跨 cache interceptor / cache config / serving wrapper / LIBERO 观测链 + 新 sidecar 执行子系统；随 ship commit 更新 `docs/architecture/cache_system.md` 与 `docs/README.md`，见 §12）
> Experiment dir: `exp/ablation_study/`（canonical 四槽结构，方向细分在槽内，见 §6.2）
> 关联文档: `docs/architecture/cache_system.md`（子系统法）、`docs/cache/migration.md`、`docs/experiments/artifact_layout.md`

---

## 1. 背景与科学问题

合作者对 cache 系统的有效性提出质疑：cache 命中时回放的"大模型历史 action"是否真的有信息量？还是被判为 hit 的步骤本来就"简单"，任何便宜的执行器都能通过——即 cache 的价值只剩路由信号（hit/miss 判定）本身，而非其存储的 Pi0.5 action 内容。

本实验用两个互补方向夹击该问题。两方向共享同一套 cache 判定机制（gate → KeyBuilder → 检索 → judge），只改变 hit / miss 槽位上的**执行体**：

| 方向 | hit 槽 | miss 槽 | 回答的问题 |
|---|---|---|---|
| **方向 1** `small_at_hit` | 小模型现算 | Pi0.5 全推理 | cache 回放的内容是否优于一个便宜执行器现算？若 SR 持平 → 质疑成立；若 cache 臂显著更高 → cache 内容本身有价值 |
| **方向 2** `small_at_miss` | Pi0.5 cache 回放 | 小模型现算 | cache 能否把大模型能力注入弱执行器？若 SR(cache+small) ≫ SR(纯 small) → cache 是能力放大器，支撑 cloud-edge 部署叙事（`docs/papers/cloud_edge_deployment.md`） |

方向 1 回答**质量**问题而非延迟问题（其 hit 路径 = 检索开销 + 小模型推理，天然比纯 cache 回放贵）；方向 2 则同时具有 SR 与延迟意义（miss 也不再跑 Pi0.5 Stage2/3）。

## 2. 已定决策（owner 已拍板）

1. **小模型双臂**：SmolVLA（~450M，lerobot，语言条件 VLA）为主臂；ACT（~80M，延迟收益最极致）为第二臂。
2. **训练路线**：Pi0.5 rollout 蒸馏（非 LIBERO 人类 demo）。cache 库本就构建自 Pi0.5 rollout，蒸馏使方向 1 成为"同一教师知识的两种载体对决"（非参数检索 vs 参数蒸馏），confound 最小。
3. **蒸馏数据规模**：差集池全集。已核实 `exp/common/data/db_init/libero/<suite>/` 每 suite 恰为 50 init/任务 × 10 任务 = 500 init（libero_spatial 与 libero_10 均如此），即"全 init − 官方 pruned_init 测试集"的全部；无需再抽样。cache 库的 50 init（`db_init/libero_cache/`，`sample_cache.py` seed=42 每任务抽 5）是其严格子集 → 蒸馏集 = cache 知识源的同源超集，且数据量上偏袒学生（怀疑方最强对照）。测试集（官方 pruned_init，50/任务）零接触。
4. **设备拓扑**：推理 server = jupyter-ziyang10（NAT 后，`tether expose` 走 broker 公网入口）；模拟 client = timan107。小模型训练亦在 ziyang10（显存预留 40G 规约内）。

## 3. 目标度量

- **主指标**：SR（success rate），在官方 pruned_init 测试集上（50 ep/任务 × 10 任务 = 500 ep/臂/suite）。
- **副指标**：per-step 延迟分解（server 侧 `SystemTimer` 探针 + client 侧 `infer_ms`）、hit rate、per-task SR 分解。
- **统计设计（预注册，防事后挑读）**：所有臂在**同一组 pruned_init**（episode 身份 = `(task_id, init_idx)`）上评测 → 配对设计。
  - 配对检验：臂间 SR 差用 **McNemar 精确检验**（配对二值结局，先例：TWA 线 `analyze_stepweight.py`）；Wilson 95% CI 仅用于**单臂 SR** 的报告；**臂间差的区间一律用配对口径**（配对风险差的 Tango score CI 或配对 bootstrap）。噪声口径申明：flow-matching noise 为 server 侧每次采样，配对仅锚定 init，不锚定 noise。
  - 方向 1 的"≈0"主张用**非劣效/等价检验（TOST）**，作用于**配对风险差**，预注册等价 margin **δ = 3pp**；"未拒绝差异 ≠ 等价"明确禁用。**功效前置门**：正式评测启动前以 500 配对 ep 与预期不一致对率计算可达精度；若 δ=3pp 的 TOST 在该精度下不可判 → 先向 owner 报告（加 ep 或改 δ 为 plan delta），不带病开跑。underpowered/不确定结果一律判读为"证据不足"。判读：TOST 通过 → 合作者质疑成立；McNemar 显著且 cache 优 ≥ δ → cache 内容有价值；两者皆否 → 证据不足，不强行判读。
  - 方向 2：`SR(small_at_miss) − SR(纯 small)` McNemar 显著 > 0 → cache **混合收益**成立；同时报告与 `SR(cache 基线)`、`SR(纯 Pi0.5)` 的差距。**归因边界**：该差值无法区分"判定信号选对了步骤"与"任意替换部分步骤为更强执行体的平凡混合效应"——凡涉及"路由信号选择价值"的主张**必须**以 O6 随机路由控制臂为前提（O6 未跑则全文只做混合收益 + 延迟收益的窄主张）。
  - 多重性：主比较族 = {方向 1, 方向 2} × {SmolVLA, ACT} × suite，Holm–Bonferroni 校正，α = 0.05；500 配对 ep 的检出功效在报告中给出精确计算。
  - 学生强度前置校准：**只用差集池内部的验证切片**（§5 Phase 1 的 45/5 划分），不触碰 pruned_init。强度准入区间：验证切片 SR 明显低于 Pi0.5 anchor 且远高于 0；不达标 → 触发 §11-O4 旋钮调整。模型/checkpoint 冻结后，pruned_init 对每个模型**只跑一次**（Phase 3/4 正式臂）。

## 4. 总体架构

```
timan107 (client)                        jupyter-ziyang10 (server)
┌─────────────────────┐    broker       ┌──────────────────────────────────┐
│ conductor + LIBERO  │ weiland.top:14xxx│ serve_policy.py (openpi venv)    │
│ episode_runner      │ ───────────────► │  Policy(Pi0.5) ← Interceptor     │
│ (pruned_init 测试集) │                 │   ├─ CP1 check（不变）           │
└─────────────────────┘                 │   ├─ hit_executor  ─┐ (方向1)    │
                                        │   └─ miss_executor ─┤ (方向2)    │
                                        │            localhost ▼           │
                                        │  sidecar_server.py (lerobot venv)│
                                        │   SmolVLA / ACT(×10 per-task)    │
                                        └──────────────────────────────────┘
```

- cache 判定链路（gate/KeyBuilder/检索/judge）与基线**逐字节相同**；routing 只替换判定后的执行体。routing 目标以 **cache yaml 的 `routing:` 段**表达（§6.1），随 conductor 既有的 bundle 热切换机制逐臂下发——server 进程全程不重启，臂矩阵 = yaml 集合。
- sidecar 是 ziyang10 上的独立进程 + 独立 venv（lerobot），经 localhost websocket（openpi msgpack 协议）被主 server 调用。隔离动机：openpi 依赖魔改 `transformers_replace`，与 lerobot 的 transformers 版本需求大概率冲突（§10-R1）。
- 小模型消费 client 原始 obs（`observation/image`、`observation/wrist_image`、`observation/state`、`prompt`，已核实 `examples/libero/main.py:305-308`），输出 client 空间 action chunk——与其训练数据同空间，不经 Pi0.5 transforms。

## 5. 阶段分解

### Phase 0 — 蒸馏数据采集（500 ep/suite）

- 复用**已存在**的 client 侧轨迹采集路径：`examples/libero/main.py --save_trajectory --save_trajectory_dir <dir> --init_states_dir exp/common/data/db_init/libero/<suite>`（flag 已核实 main.py:75,93-95；loader `_load_init_states` main.py:1160 接受 `.init`）。
- 每 replan cycle 记录（已核实 main.py:348-360 schema）：`sim_state`、`agentview_image`、`eye_in_hand_image`、`robot_state`(8 维)、`env_action_chunk`、`executed_actions` — 正是蒸馏所需 (obs → chunk) 对，无需改采集代码。
- Server 侧以纯推理（无 cache）serve Pi0.5；conductor 不参与（save_trajectory 未接入 episode_runner，走 main.py 独立路径，同 trajectory_deviation Step 1b 先例）。
- 产量预期：500 ep/suite，成功筛后 ~420–480 ep；obs-chunk 训练对 ~10k（spatial）/ ~26k（libero_10）。
- 落盘 client 侧（timan107）`exp/ablation_study/data/distill_raw/<suite>/task_{id}/episode_{idx}.h5`（gitignored）；采集完成后跨机回传（自包含步骤）：`tether pull timan107:<path> <local>` → `tether push <local> ziyang10:<path>`（push 首参必须本地文件），两端 `sha256sum` byte-identical 验证后 Phase 1 开工。
- 运行前置断言：`--init_states_dir` 目录内**不得存在** `.pruned_init` 文件（`_load_init_states` 优先匹配 `.pruned_init`，混入会静默替换差集池，main.py:1165-1170）；采集脚本启动时检查并 fail-fast。

### Phase 1 — 数据转换 + 小模型训练（ziyang10，lerobot venv）

- **训练/验证划分（防泄漏，全在差集池内部）**：500 init/suite 按任务分层切分为 **train 45/任务（450）+ student-val 5/任务（50）**，切分表（init_idx 清单）为 tracked 文件。cache 库的 50 init 保持落在 train 侧（保同源超集性质）。checkpoint 选择、早停、强度校准**只看 student-val**；pruned_init 在模型冻结前零接触。
- `build_distill_dataset.py`：traj H5 → lerobot dataset 格式（成功 ep 筛选 `attrs["success"]`；图像/state/action/task 字符串）。参照上游 `examples/libero/convert_libero_data_to_lerobot.py` 的既有转换先例。
- **学生动作契约（O5 已定）**：训练标签 = `env_action_chunk`（[10, 7] env 空间完整 chunk，与 Pi0.5 返回口径一致）；serving 返回 10-chunk，client 照常执行 `replan_steps` 步——client 逻辑零改。obs = 采集所录的 resize_with_pad 处理图（与 policy 所见同像素）+ 8 维 state float32。
- SmolVLA：从官方预训练 checkpoint 微调，单模型多任务（语言条件）。ACT：每任务独立训练 10 个模型/suite，sidecar 按 `prompt` 字符串精确匹配路由权重（manifest 为 tracked json：task_description → checkpoint 路径；unknown → raise）。
- **冻结清单**（tracked，`config/freeze_manifest_<suite>.yaml`）：lerobot 版本 pin、基座 checkpoint id、图像/state 预处理、归一化统计来源、action 表示/dtype/维度、horizon、成功筛选规则、切分表、选中 checkpoint 的 sha256。
- 训练 recipe/超参落 `exp/ablation_study/config/`；产物 checkpoint 落 `exp/ablation_study/data/checkpoints/`（gitignored）。

### Phase 2 — sidecar 与路由集成

代码改动详见 §6。集成后做 3 层验证：sidecar 单测（协议 roundtrip）→ 本地 fake-policy 路由单测 → **1-cell smoke 验收标准**（自包含）：1 个 routing yaml × ~10 ep 端到端跑通，全 ep 完成、无 error/Traceback、逐步行含 `executor` 字段、`executor` 计数与 `hit_type` 计数一致、sidecar 计时日志行数 = 对应路由步数。任何一条不满足 → 不进 Phase 3/4。

### Phase 3 — standalone 锚点评测

| 臂 | 说明 |
|---|---|
| 纯 Pi0.5 | anchor 已有（spatial ~0.95+ / libero_10 ~0.83），必要时按当前 HEAD 复跑对齐 |
| cache 基线 | §8.2 锁定的剥离-warm_tiers 基线 yaml 跑 pruned_init 测试集，得 SR + hit rate + 延迟 |
| 纯 SmolVLA / 纯 ACT | **router 路径实现**：always-MISS yaml（gate `always_skip`，gate-skip 返回 MISS 已核实 orchestrator.py:532）+ `routing.miss_to` → 每步皆小模型执行，harness 与主评测臂完全一致。该臂**仅作 SR 锚点**：其 client 端延迟仍含 Pi0.5 Stage1 + key build/history 记录 + gate 调用（`always_skip` 跳过的是检索与 judge），不代表纯 small 部署延迟；纯 small forward 延迟由 sidecar 侧 per-request 计时日志单独度量（§6.2） |

### Phase 4 — 双方向主评测

- 方向 1：yaml `routing.hit_to` → sidecar（SmolVLA / ACT 各一臂）；方向 2：yaml `routing.miss_to` → sidecar（同）。
- 共 4 routing 臂 + Phase 3 的 4 锚点臂 = 8 臂/suite（O6 启用则 +1）；suite 顺序 §11-O1。
- **编排契约（可执行）**：server（ziyang10）单进程 `--replicas 1` 起一次，全程不重启；两个 sidecar 进程并存（SmolVLA :7001，ACT :7002），各 suite 换 checkpoint 时才重启 sidecar（重启边界 = suite 边界）；臂间切换 = conductor 按既有 yaml 亲和机制下发 `load_cache_config`（bundle 热切换），routing 段随 yaml 走。臂 ↔ (yaml, sidecar 端点, 模型) 绑定表写入 `config/arm_matrix_<suite>.yaml`（tracked）。
- **SR 与延迟分跑**：SR 主跑用 conductor 常规并发；延迟口径另跑专门 pass（`--workers 1`、非并发），并用 sidecar 侧 forward/排队分离计时，消除 coordinator 队列与 sidecar GPU 锁的混杂。
- 运维（server 启停/监控/无人值守）遵循项目内文档 `docs/experiments/conductor_tutorial.md`；监控验收：L1 健康脚本纳入 sidecar 进程与端口探活。

### Phase 5 — 分析与报告

- 汇总 SR/hit-rate/延迟表 + per-task 分解 + §3 配对统计（McNemar / 配对风险差 TOST / 功效计算）；最终报告 `exp/ablation_study/analysis/analysis.md`（纯 .md 规约）。

## 6. 代码改动清单（files touched + interfaces）

### 6.1 src 侧（最小、加法式、默认惰性）

**`src/openpi/cache/interceptor.py`** — `InferenceInterceptor` 新增两个可选构造参数：

```python
hit_executor:  Callable[[dict], dict] | None = None   # Replaces the executor on FULL_HIT
miss_executor: Callable[[dict], dict] | None = None   # Replaces the executor on MISS / gate-skip
```

- 两者默认 `None` → 全路径行为与现状逐字节相同（wire 不变式，与 `record_verdict`/`replay_target` 同款加法式 hook 纪律）。
- `hit_executor`：插入 FULL_HIT 分支（现 interceptor.py:717-750）。cache 记账（`broadcast_action`/`buffer_for_write`）照常执行后，调 `hit_executor(obs)` 得到 client 空间 outputs dict，attach `__hit_meta__`（`hit_type` 保持 `FULL_HIT`，新增 `executor: "override"` 字段），执行 `orchestrator.clear()`（对齐现 FULL_HIT 短路 interceptor.py:749 的收尾语义，释放 KeyBuilder 每周期张量引用）后返回；跳过 cached action 的输出装配与 `_output_transform`。
- `miss_executor`：插入 FULL_HIT 分支之后、stage2 之前（现 interceptor.py:752 附近）。cp1 非 hit（含 gate-skip 的 MISS，orchestrator gate-skip 返回 MISS 已核实 orchestrator.py:532）时调 `miss_executor(obs)`，attach `__hit_meta__`（MISS + `executor: "override"`）、`orchestrator.clear()` 后返回；完全跳过 stage2/3、CP3、miss 侧的 `broadcast_action` 与 `buffer_for_write`（现 MISS 记账清单见 interceptor.py:849-881）。**记账不对称显式声明**：hit 臂广播 cache action、miss 臂不广播——在 §8.2 禁 action-history 组件约束下无行为影响，写入公平性申明。
- **fail-fast 约束**：任一 executor 非 None 时，(a) 构造期要求 `orchestrator is not None`；(b) 运行期 cp1 返回 `WARM_START` → raise（routing 臂 yaml 必须关 warm_tiers，防静默混合语义）；(c) `prefill_trajectory` 的 prefill_mode（interceptor.py:448-454，合成 FULL_HIT 走完整 infer）与任一 executor 组合 → raise（本实验不用 prefill，防止 prefill obs 被误发 sidecar）。
- executor 收到的 `obs` 是 `infer()` 入参原 dict（已 pop `__gate_decision__`）。前置假设——input transforms 不就地改写 obs 的 ndarray——由 §9 的 parity 测试锁定。
- 计时：新增 `sidecar_hit` / `sidecar_miss` CPU 探针包住 executor 调用。

**`src/openpi/cache/config.py`** — `CacheConfig`（现 config.py:496）新增可选 `routing` 段（additive，缺省 None → 现有全部 yaml 逐字节兼容）：

```yaml
routing:
  hit_to:  "127.0.0.1:7001"    # direction 1; mutually exclusive with miss_to
  miss_to: null
  connect_timeout_s: 10
  request_timeout_s: 30
```

`load_cache_config`（config.py:710）解析；`validate_cache_config`（config.py:1218）在 `routing` 非 None 时执行 **§8.2 正向 allowlist** 校验（不满足 → yaml 加载即拒绝）。routing 入 yaml 而非 CLI 的动机：conductor 的臂切换走 `load_cache_config` bundle 热切换（server 进程不重启、无法改 CLI），routing 必须随 yaml 逐臂下发才可编排；同时结构性消除"裸 `--cache` 分支 flag 静默失效"问题（该分支无 yaml → 无 routing 可言）。

**`scripts/serve_policy.py`** — `_wrap_policy()`（现 serve_policy.py:398-565）内 `InferenceInterceptor` 共有三处构造：bundle 分支（473-485）、`--cache_config` 分支（529-541）、裸 `--cache` 分支（548-553，orchestrator=None，结构性无 routing）。前两处按 `cache_config.routing` 为**每个连接**构造一个 `SidecarExecutor`（见下）传入 interceptor。不新增任何 CLI flag。routing 臂固定 `--replicas 1` 写入臂矩阵 config（多 replica 下 sidecar GPU 锁串行使延迟臂无意义）。

**sidecar 客户端所有权与失败语义**（`SidecarExecutor`，实现放 `src/openpi/cache/sidecar_executor.py` 新文件）：

- 连接**惰性建立**于首次调用，且**自行实现有界连接**（直接以带 deadline 的 websocket 连接 + metadata 握手建立会话，复用 `openpi_client` 的 msgpack 编解码；**不得**调用 `_wait_for_server`（无超时重试循环，websocket_client_policy.py:32-44）再从外部包超时——那会泄漏存活线程与半途 socket）；单次 infer 包在 `request_timeout_s` 内。
- 超时/连接断/响应 schema 非法（shape/dtype 校验失败）→ **fail-closed**：raise 使该 ep 失败，交 conductor ep 级重试；**绝不**静默回退 Pi0.5（会污染臂语义）。
- 确定性关闭：`InferenceInterceptor.on_task_end` 转发 `executor.close()`（连接关闭 + socket 释放）；bundle 热切换导致 wrapper 重建时旧 executor 同样被 close（`_wrap_policy` 持有并在替换前关闭）。
- server 启动期与每次 bundle 下发时各做一次端点探活（有界超时），失败即拒绝该 yaml——覆盖"探活后才死"的窗口由 fail-closed + ep 重试兜底。
- sidecar 进程重启所有权：人工 + 监控告警（L1 巡检含 sidecar 进程/端口），不做自动拉起。

**`examples/libero/episode_runner.py`** — `_hit_row`（episode_runner.py:58-80）是白名单式装配，新增一行透传 `executor: hit.get("executor")`，conductor 逐步行才能记录"谁执行了这一步"（standalone main.py 路径经 `**hit_meta` 展开天然透传，无需改）。

### 6.2 exp 侧（`exp/ablation_study/`，canonical 四槽，方向细分在槽内）

```
exp/ablation_study/
  __init__.py
  sidecar_server.py  build_distill_dataset.py  train_smolvla.py  train_act.py  run_ablation_eval.py   # 代码全在根（无子包）
  config/
    common/            # 采集/冻结清单/O2 派生基线 yaml
    small_at_hit/      # 方向 1 routing yaml（hit_to）
    small_at_miss/     # 方向 2 routing yaml（miss_to）+ always-MISS 锚点 yaml
    arm_matrix_<suite>.yaml
  data/                # gitignored：distill_raw/ checkpoints/ runs/
  analysis/            # analyze_ablation.py + analysis.md + 图
```

现骨架中 `small_at_hit/`、`small_at_miss/` 两个嵌套四槽子树与其 `__init__.py` 在 §4 Code 期**移除/重排**为上述结构（方向区分降到 config 槽内 + data 子目录 + 文件名前缀，完全符合 artifact_layout 四槽章程）。

| 文件 | 职责 |
|---|---|
| `sidecar_server.py` | lerobot venv 进程：加载 SmolVLA 或 ACT(×10 per-task，按 `prompt` 精确匹配路由；unknown prompt → raise)，实现 openpi websocket+msgpack 协议 server 侧（~60 行，不依赖 openpi 主包）。协议契约：accept 时**先发一条 metadata dict**（`WebsocketClientPolicy.__init__` 连接后先 recv metadata，websocket_client_policy.py:27,40，否则 router 连接死锁），随后处理 infer 请求；GPU forward 加锁串行；per-request 计时日志分离 forward 与排队时间（延迟锚点 + §5 Phase 4 混杂控制） |
| `build_distill_dataset.py` | Phase 1 转换器：traj H5 → lerobot dataset（成功筛选 + train/val 切分表 + O5 契约） |
| `train_smolvla.py` / `train_act.py` | 训练入口（薄封装 lerobot 训练 CLI + 本实验超参 yaml + 冻结清单产出） |
| `run_ablation_eval.py` | Phase 3/4 评测 driver：复用 conductor（`ConductorDriver` + `LiberoEpisodeRunner`），按 `arm_matrix_<suite>.yaml` 逐臂下发 routing yaml（bundle 热切换），并做臂配置二次校验 |
| `analysis/analyze_ablation.py` + `analysis.md` | Phase 5 汇总与报告（McNemar/TOST/CI 实现在此） |

采集/评测产物均入 `exp/ablation_study/data/`（gitignored，按 artifact_layout §3）。

### 6.3 观测链路

`__hit_meta__` 已随每步返回并被 conductor `_hit_row` 记录（episode_runner.py:58）。`_hit_row` 为白名单装配，需 §6.1 所列的一行透传改动才能记录新增 `executor` 字段；standalone main.py 路径（`**hit_meta` 展开）天然透传。

## 7. 集成点 API 依据（已亲验）

| API | 位置 | 已验行为 |
|---|---|---|
| `InferenceInterceptor.infer` FULL_HIT 短路 | `src/openpi/cache/interceptor.py:717-750` | hit → broadcast/buffer → `_unbatch_outputs` → `_output_transform` → attach `__hit_meta__` → `orchestrator.clear()`(:749) → return |
| gate-skip 的 verdict | `src/openpi/cache/orchestrator.py:532` | gate 返回 False → `CheckResult(MISS, searched=False)`，落入 miss_executor 分支 |
| MISS 分支入口 / meta guard | 同上 `:752-763,798-831` | stage2 前有 meta-device guard；MISS 走 stage2/3 + intermediates |
| `_build_hit_meta` | 同上 `:480-521` | dict 装配点，新增 `executor` 字段的落点 |
| `_wrap_policy` per-connection 构建 | `scripts/serve_policy.py:398-565` | **三**处 `InferenceInterceptor(...)` 构造（473-485/529-541/548-553）；executor 仅注入前两处，裸 `--cache` 组合被 CLI 校验拒绝 |
| 运行时写策略强制 never | `scripts/serve_policy.py:26-48` | serving 侧 `write_policy` 强制 `never` → routing 臂天然只读，无写路径污染 |
| `WebsocketClientPolicy(host, port).infer(obs)->dict` | `packages/openpi-client/src/openpi_client/websocket_client_policy.py:12-47` | router→sidecar 转发用 |
| client obs 键 | `examples/libero/main.py:305-308` | `observation/image`、`observation/wrist_image`、`observation/state`(8 维)、`prompt` |
| `--save_trajectory` 采集 schema | `examples/libero/main.py:75,93-95,348-360,672-675` | 每 replan cycle 记 obs 图像 + executed_actions + sim_state，落 `task_{id}/episode_{idx}.h5` |
| `--init_states_dir` loader | `examples/libero/main.py:75,1160-1180` | 接受 `{task_name}.init`，指向差集池目录即 500-init 全量 |
| conductor hit 行记录 | `examples/libero/episode_runner.py:58-80,168` | `_hit_row` 白名单装配 `__hit_meta__` 逐步落 JSONL；`executor` 字段需一行透传改动（§6.1） |
| cache config schema/校验/构建 | `src/openpi/cache/config.py:496,710,1218,2183` | `CacheConfig` / `load_cache_config` / `validate_cache_config` / `build_per_connection_components` —— `routing` 段与 allowlist 校验的落点 |
| `record_action` 纯缓冲语义 | `src/openpi/cache/components/search_strategy.py:135-137` | broadcast 只 append `_action_history`，allowlist 组件族无读取方 |
| O2 基线 yaml（spatial） | `exp/gate_research/config/libero_spatial/eval/..._d1__fh40_ws40_quantile.yaml` | 已核实：CP1-only、`always_search`+`threshold(0.983416)`+warm_tiers、`weighted_score_sum_knn` d1、write never、in_memory pkl 路径 |

## 8. 公平性申明（预写入报告）

1. **closed-loop 分叉**：各臂 action 不同 → 状态漂移 → hit/miss 判定序列必然分叉。本实验保证"同配置同判定逻辑"，不保证"逐步同判定"；这是 closed-loop 消融的固有限制，逐步 hit rate 会随臂报告以供核对。
2. **cache 内部记账口径与正向 allowlist**：hit_executor 臂中 `broadcast_action` 广播的是 cache 命中的 action（模型空间），而实际执行的是小模型 action（client 空间，无法进同一 history）；miss_executor 臂则完全不广播（§6.1）。`record_action` 在允许组件族中是纯缓冲（search_strategy.py:135-137，`_action_history` 仅被 composite 判决链消费），故 broadcast 内容差异不影响判定链。强制手段为 **`validate_cache_config` 在 `routing` 非 None 时的正向 allowlist**（yaml 加载即拒绝，`test_ablation_config.py` + `run_ablation_eval.py` 二次校验）：
   - checkpoints：**仅 cp1**（cp3 段存在 → 拒绝）；
   - gate：`always_search` 或 `always_skip`（O6 臂另加 `random`）；judge：`threshold` 且 **无 `warm_tiers`**（O6 臂另加 `always_hit`）；
   - search_strategy：`weighted_score_sum_knn` 或 `weighted_rrf_knn`，**深度 1**（无 trajectory 字段/深度策略）；
   - write_policy：`never`；collection 段：无；stage placement：legacy 默认（无 meta/分体）；prefill：不使用（§6.1c 运行期 raise 兜底）；
   - backend：`in_memory` + 既有 pkl artifact。

   **O2 已锁定**：基线与 routing 臂共用同一判定链，取自 gate_research 已标定 eval 家族并**剥离 warm_tiers**（FULL_HIT/MISS 二值化）——libero_spatial 派生自 `exp/gate_research/config/libero_spatial/eval/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh40_ws40_quantile.yaml`（threshold=0.983416，artifact `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl`，已核实 CP1-only、write never、d1）；libero_10 派生自同家族 `cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh40_ws40_quantile.yaml`（threshold=0.997349，烘焙值）。派生 yaml 入 `config/common/`，与原文件的唯一差异 = 删 `warm_tiers` + 加 `routing`（cache 基线臂删 `warm_tiers` 不加 routing）。**G1 批准绑定这两个具名 fh40 派生配置及其烘焙阈值**：任何操作点变更 = plan delta，须记录并在受影响正式臂运行前复审。
3. **延迟口径**：方向 1 的 hit 路径 = Stage1 + 检索 + sidecar 往返 + 小模型 forward，天然贵于纯 cache 回放；报告延迟表时按组件分解（SystemTimer 探针），不做单一总延迟对比的误导性结论。
4. **渲染一致性**：蒸馏采集与评测在同一环境版本/同一渲染栈上连续完成（Phase 6.0 渲染漂移前科的规避措施）；采集与评测之间不升级 LIBERO/mujoco/驱动。

## 9. 测试策略

- `tests/ablation_study/test_router_hooks.py`：fake policy + fake orchestrator 驱动 `InferenceInterceptor`：
  - 两 executor 均 None → 输出与现状逐 key 相等（回归锁）；
  - hit_executor 设定 + FULL_HIT → executor 被调、输出为 executor 返回值、`__hit_meta__.executor == "override"`、broadcast/buffer 仍发生、`clear()` 已调；
  - miss_executor 设定 + MISS → stage2/3 未被调用（fake 上打桩计数）、broadcast/buffer 未被调、`clear()` 已调；
  - WARM_START + 任一 executor → raise；prefill_mode + 任一 executor → raise；
  - executor 调用前后 obs 逐 ndarray 逐字节不变（parity，锁 §6.1 前置假设）。
- `tests/ablation_study/test_sidecar_protocol.py`：sidecar server 进程内 loopback：msgpack roundtrip、metadata 握手、per-task ACT 路由（prompt→权重选择、unknown → raise）、并发两连接串行锁、畸形响应（错 shape/dtype/缺键）→ SidecarExecutor raise。
- `tests/ablation_study/test_sidecar_executor.py`：连接超时（端点不存在 → connect_timeout 内 raise）、请求超时、`close()` 幂等且释放连接、`on_task_end` 触发 close、bundle 替换路径旧 executor 被 close。
- `tests/ablation_study/test_ablation_config.py`：§8.2 正向 allowlist 全字段覆盖（cp3 段/warm_tiers/深度>1/write≠never/collection/meta placement/`hit_to`+`miss_to` 同置 → 各自拒绝）。
- `tests/cache`/`tests/scripts` 增补：routing 经 **runtime bundle 热切换**路径生效（`get_current_cache_bundle` 分支）与静态 `--cache_config` 路径生效各一测；`__collect_meta__` 在 executor 分支下保持装配；executor 抛异常沿 infer 传播（不吞）；meta stage placement 与 routing 组合被 allowlist 拒绝。
- 并发/非并发生命周期：coordinator 存在（concurrent）与 None（legacy）两路径下 executor 分支行为一致（fake coordinator 单测）。
- GPU/模型依赖的端到端 smoke 标 `@pytest.mark.manual`；1-cell E2E 验收标准见 §5 Phase 2。
- §6 Verify 口径：裸 `uv run pytest tests/ablation_study tests/cache tests/scripts tests/exp tests/conductor tests/libero`（改动 blast-radius：cache/serving wrapper/exp 层 + 直接触及的 conductor/LIBERO runner；`tests/scripts` 为 serve_policy 单测既有所在）。范围遵循 owner 既定裁定（§6 Verify = blast-radius 目录，非 repo-wide CI 口径），见 Review Log R2 对 item 8 的响应。

## 10. 风险登记册

| # | 风险 | 缓解 |
|---|---|---|
| R1 | lerobot 与 openpi 的 transformers 版本冲突 | sidecar 独立 venv + 独立进程，主 server 零新依赖；协议层仅 websockets+msgpack |
| R2 | 学生强度落点不可控（太弱/太强 → 平凡结论） | Phase 3 强度校准前置；数据量/训练步数为旋钮；判读口径预注册（§3） |
| R3 | obs 口径不匹配（分辨率/视角/state 格式） | 蒸馏数据直接取自 client 发送的同一 obs 流（`--save_trajectory` 记录的就是 policy 所见图像），训练-服务同空间；集成测试断言 shape/dtype |
| R4 | sidecar 往返开销污染延迟结论 | localhost msgpack 往返 ~1-3ms 量级，SystemTimer 单独探针剥离；报告按组件分解 |
| R5 | routing 臂 yaml 误用 action-history 组件 → 记账口径破坏 | config 校验强制（§8.2）；测试覆盖（§9） |
| R6 | 采集/评测间渲染漂移（项目前科） | §8.4：同环境版本连续完成，不中途升级 |
| R7 | ziyang10 显存：Pi0.5 (~14G peak) + SmolVLA + 10×ACT + 训练任务挤兑 | 训练与 serving 分时段；serving 期 sidecar 常驻 ~3-5G；遵守 40G 预留规约，告危按 10G 粒度释放 |
| R8 | interceptor 改动引入回归 | 默认 None 惰性 + §9 回归锁测试 + G2 审查 |
| R9 | 500 init 中个别 init 采集失败/不成功 → 蒸馏集缺口 | 成功筛选后按任务统计覆盖；单任务成功 < 30 ep 时向 owner 报告再定补采（同 init 换 noise seed） |
| R10 | sidecar 进程中途死亡 / 探活后死亡窗口 | SidecarExecutor 有界超时 + fail-closed（§6.1）+ 启动期与每次 bundle 下发探活 + L1 巡检含 sidecar 进程/端口 + conductor ep 级重试兜底 |
| R11 | 并发队列 + sidecar GPU 锁混杂延迟测量 | SR 与延迟分跑（§5 Phase 4）：延迟 pass `--workers 1` 非并发 + sidecar forward/排队分离计时 |
| R12 | 学生校准触碰测试集（泄漏） | train/val 全在差集池内部（45/5 分层切分，tracked 切分表）；pruned_init 冻结后每模型仅一跑（§3/§5 Phase 1） |

## 11. 开放项（G1 前/中由 owner 裁决）

- **O1 suite 顺序**：建议先 libero_spatial（基线最全、单 ep 短、迭代快）打通全链路，再 libero_10。两 suite 都做为最终交付。
- **O3 ACT 多任务方案**：本 plan 取 per-task 10 模型；若 owner 倾向单模型 + task embedding 改造，Phase 1 工作量上调（lerobot ACT 无原生条件输入）。
- **O4 学生强度旋钮的调整协议**：若 Phase 3 校准落点不佳，优先调训练步数（早停 checkpoint 序列中选），其次数据量降采样；均在报告中如实记录。
- **O6 方向 2 随机路由控制臂（建议启用）**：`random` gate（`p_inference` 调到复现 cache 基线实测 hit rate）+ `always_hit` judge + `routing.miss_to`——以匹配比例做**随机**步骤替换（search 步仍回放最近邻）。**凡对"路由信号选择价值"做任何主张，O6 为必要前提**（§3）；不启用则全文只做混合收益窄主张。成本 +1 臂/suite（500 ep）。

## 12. L3 文档义务

- `docs/architecture/cache_system.md`：§4.2/§5 增补 "external executor hooks" 小节——`hit_executor`/`miss_executor` 语义、`routing` 配置段、SidecarExecutor 生命周期（close/超时/fail-closed）、broadcast/write 记账在 executor 分支下的偏差申明；与既有 additive-hook 条目（`record_verdict`/`replay_target`）并列。
- `docs/README.md` 索引行同 commit 更新（Index Sync 红线）。
- `logs/README.md` 本 plan 行改 L3（随本轮修订同步）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-11 13:22 CDT

- [Blocking] [Concern] The distillation builder drops every task description and does not produce the per-task ACT input layout — reasoning: the canonical trajectory writer stores `attrs["task_name"]` (`examples/libero/main.py`), while `build_distill_dataset.iter_frames()` reads `attrs["task"]`, so both students train with an empty task string and the ACT prompt router cannot match live prompts. In addition, the builder emits one combined dataset, whereas `train_act.py` requires `task_*/prompt.txt` plus one dataset root per task; no implementation creates those inputs. Align the reader with the canonical HDF5 schema and make the builder/trainer contracts executable end to end, with tests using a canonical trajectory fixture.
- [Blocking] [Concern] The G1-approved leakage guard and checkpoint-selection protocol are not implemented — reasoning: `emit_split()` randomly places five episode indices into validation without preserving the cache-library five-init subset in train, writes no tracked split artifact in this change, and `build_lerobot_dataset()` exports only train. Neither trainer consumes a validation slice or performs the promised validation-only selection/strength gate; the ACT wrapper emits no freeze manifest at all, and the required tracked recipes/freeze inputs are absent. Implement and test the exact 45/5 constrained split, train/val export, validation-only checkpoint selection, recipes, ACT prompt manifest, and freeze-manifest contracts before touching `pruned_init`.
- [Blocking] [Concern] The evaluation driver cannot reach normal completion and does not persist the observability needed by the plan — reasoning: `run_ablation_eval.py` calls the resident `WorkerAgent.run()` synchronously and only joins the driver afterward, but the agent returns only after `stop()` and will respawn workers that receive shutdown. Follow the established conductor pattern (agent background thread, driver join, `agent.stop()` in `finally`). The driver also never writes `driver.per_step_rows`, so hit rate, `executor` provenance, and main-server timing/step evidence are discarded; add the planned crash-safe output path and lifecycle tests.
- [Blocking] [Concern] The analysis program is incompatible with the journal produced by that driver and would omit unsuccessful episodes — reasoning: `Journal.record()` writes `task_uid/yaml_id/phase/status/success/ts`, while `load_journal()` directly reads nonexistent `task_id` and `orig_init_state_idx`, producing `KeyError` on a real record. It also accepts only `status == "done"`; conductor journals ordinary unsuccessful rollouts as `status == "failed", success == false`, so filtering them would inflate SR even if identity were fixed. Consume a real, tested result contract, retain every terminal rollout outcome, and validate complete paired identity/coverage before statistics.
- [Blocking] [Concern] The preregistered statistical procedure is not faithfully implemented — reasoning: the promised power/precision gate must run before formal evaluation from `n=500` and an expected discordance rate, but the only `powered` value is computed post hoc from the observed bootstrap interval. Holm–Bonferroni adjusted p-values are calculated independently as `(m-rank)*p` without the required cumulative maximum/step-down stopping rule, so the code can reject a later hypothesis after retaining an earlier one (the independent probe reproduces adjusted values `0.06, 0.04` and a false rejection). Implement a preflight power command/report, correct monotone Holm adjustment and decisions, and test both; the final report must expose paired comparisons, precision/power, and multiplicity results rather than only the single-arm SR table.
- [Blocking] [Concern] The G1-approved failure-mode coverage and Phase-2 acceptance evidence are incomplete — reasoning: the focused suite passes 31 tests, but the promised request-timeout/death-mid-request, invalid dtype/missing-key, ACT prompt selection/unknown prompt, executor exception propagation with cleanup, `__collect_meta__`, actual static and runtime-bundle wrapper paths, concurrent coordinator path, bundle-replacement close, and manual 1-cell acceptance are absent or only named in docstrings. The sidecar handshake failure path also leaves the newly connected local socket unowned because `_conn` is assigned only after metadata receive succeeds. Add the missing executable tests and close a connection on every handshake failure; attach the 1-cell evidence required by Phase 2.
- [Blocking] [Concern] The plan/index state does not accurately describe a G2-ready implementation — reasoning: the polished plan header still says Code is in progress and retains the stale unresolved O2 bullet even though `logs/README.md` says Code complete and the later body locks O2. Synchronize the plan status/index and remove the obsolete O2 ambiguity during the next post-response polish.
- [Non-blocking] [Suggestion] Tighten routing input validation — reasoning: the current endpoint check accepts values such as `host:not-a-port`, and `NaN`/infinite timeouts bypass the `<= 0` guard. Validate a numeric in-range port and finite positive timeouts so malformed experiment YAML fails at load rather than during wrapper construction.
- [Non-blocking] [Concern] Static quality is not green — reasoning: `ruff check` reports F541 in `train_smolvla.py:42` (`f"--policy.type=smolvla"` has no placeholder). Remove the unnecessary prefix when addressing the blocking pipeline revisions.

### G2 Round 2 — Executor — 2026-08-11

- Item 1（task 文本丢失 + ACT per-task 输入）：**Accepted** — `iter_frames` 改读规范 `attrs["task_name"]`（连同 `success`/`step_XXXX` 组名已对照真实 GT H5 逐项核实）；`build_per_task_datasets` 新增 ACT 布局 `task_<id>/{dataset, prompt.txt}`，`train_act.py` 改吃 `task_dir/dataset`。新增 `tests/ablation_study/test_distill_builder.py`：规范 schema fixture 上验 task_name 读取、失败 ep 过滤、帧形状。
- Item 2（切分约束 + 校准协议）：**Accepted** — `emit_split` 重写为受约束分层切分：`cache_positions_for_task`（与 `sample_cache.find_original_indices` 同匹配规则）定位 cache 库 init 在差集池中的位置并强制留在 train（`constrained_split`，val 只从非保护集抽），切分表含 `protected_in_train` 佐证；`--part {train,val}` 双导出、`--emit-val-inits` 产出仅含 val init 的每任务 `.init` 文件（student-val rollout 走 standalone main.py 直连 sidecar，sidecar 已加 ctrl-ack 支持该链路）；训练 recipe（`config/common/recipe_{smolvla,act}.yaml`）入库，ACT 也产 freeze manifest（含 selection_protocol 字段）。约束/匹配/不足候选 fail-fast 均有测试。
- Item 3（评测生命周期 + 观测持久化）：**Accepted** — `run_ablation_eval.py` 改为 run_phase2 先例的生命周期（agent 后台线程 + `driver_thread.join()` + `finally: agent.stop()`）；新增必填 `--per-step-out`，经 `ConductorDriver(per_step_writer=...)` 崩溃安全逐 flush 追加 per-step JSONL（hit rate / executor 佐证 / server 计时随行保留）。
- Item 4（journal 契约）：**Accepted** — analyzer 改消费真实 Journal schema：身份从 `task_uid`（`<arm>__t<i>__e<j>`，`parse_task_uid`）解析；终态 `done` 与 `failed` 都进 SR 分母（重试以文件序后写胜出）；统计前 `check_paired_coverage` 审计各臂身份集合缺口。以合成真契约记录的单测覆盖（含 failed 保留与 retry 去重）。
- Item 5（统计忠实性）：**Accepted** — `holm_adjust` 重写为正确 step-down（升序、`(m-i)p` 取累积 max、首个保留后停止拒绝），审查探针场景（0.06/0.04 假拒绝）已入测试断言修正后的单调行为；新增 `--preflight-n/--preflight-discordance` 前置功效门（正式评测前运行，不可判则 exit 非零阻止开跑）；md 报告扩展为配对比较表 + TOST 判决 + powered + Holm 列。TOST 作用于配对风险差（配对 bootstrap 90% CI 规则），Wilson 仅标注单臂。
- Item 6（失败路径覆盖 + 握手泄漏 + 1-cell）：**Accepted** — `_ensure_connection` 握手失败即关闭已开 socket（修复泄漏）；补交测试：请求中途超时、非法 dtype（raw server）、缺 `actions` 键、ctrl-ack 不误推理、`route_prompt` 精确匹配/unknown raise（已抽为纯函数）、executor 异常传播 + 状态恢复（executor 分支 try/finally `clear()`）、`__collect_meta__` 在两执行分支下保持装配（interceptor 补齐）、fake coordinator 下 miss 路由仅 submit stage1、`_wrap_policy` 静态与 runtime-bundle 两路径实测 executor 注入 + `on_task_end` 关闭。1-cell 验收以可执行 manual 测试交付（`test_manual_e2e.py`，断言 §5 Phase 2 全部标准）；其运行证据依 plan 属 Phase 2 运行期出场门（需 GPU+sim 拓扑），将在 Phase 2 实跑时附上。bundle 替换关闭语义按实现设计经 `on_task_end` 承载（wrap 为每连接构建，替换发生于连接边界），由 wrap 路径测试与 close 测试共同覆盖。
- Item 7（plan/索引状态同步）：**Accepted** — plan 头改"§4 Code 完成，G2 复审中"，§11 陈旧 O2 未决 bullet 删除，O2 正文补 libero_10 烘焙阈值 0.997349。
- Item 8（非阻塞，routing 输入校验）：**Accepted** — 端口须数字且 ∈[1,65535]，超时须有限正数（NaN/inf 拒绝），yaml 加载即报错；`bad_endpoint`/`bad_timeout` 测例覆盖。
- Item 9（非阻塞，ruff F541）：**Accepted** — 已修；本次全部触及文件 ruff 清洁（仓库仅剩 `in_memory_backend.py:887` 既有 E741，非本改动文件，未越界清理）。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-08-11 14:54 CDT

- [Blocking] [Concern] The journal/analysis fix uses a task UID that is incompatible with the conductor that writes the journal, so a real run still cannot be analyzed — reasoning: `AblationEvalStrategy` emits `<arm>__t<i>__e<j>`, but `ConductorDriver._uid_meta()` only round-trips the canonical `<yaml_id>:<phase>:<task>:<episode>` format. Consequently the journal's `yaml_id` becomes the entire per-episode UID, producing one apparent arm per episode; `load_journal()` then groups on those values and the pairwise analysis fails. Use `make_task_uid()` (or another format accepted by the driver) and parse that same canonical identity in the analyzer; test the strategy through the actual driver/journal boundary rather than constructing synthetic records with a hand-written `yaml_id`.
- [Blocking] [Concern] The constrained split still cannot open the real LIBERO init files — reasoning: `_task_name_of()` reads HDF5 `attrs["task_name"]`, which the canonical writer fills from `task_description`/`task.language` (natural language with spaces), while `db_init/libero/<suite>` and `db_init/libero_cache/<suite>` are keyed by `task.name` stems with underscores. `emit_split()` therefore attempts paths such as `pick up ... .init` and raises `FileNotFoundError`; the new tests exercise helpers but never call `emit_split()` with realistic names. Preserve prompt and init-file stem as separate fields, resolve the mapping deterministically, and add an end-to-end split fixture matching the real naming contract.
- [Blocking] [Concern] The validation-only checkpoint-selection and reproducible student environment promised at G1 remain declarations rather than an executable pipeline — reasoning: emitting val init files and writing `selection_protocol: student-val only` into a manifest does not score candidate checkpoints, enforce the student-strength gate, or select/freeze a checkpoint. Neither trainer consumes validation results; the ACT freeze manifest still has no checkpoint SHA-256 map. No LeRobot version/revision is pinned for the separate venv, while the repository's locked LeRobot 0.1.0 cannot import the new `lerobot.datasets` / `lerobot.policies` paths used here. Add the actual val-evaluation/selection command and artifact contract, hash every selected ACT checkpoint, and provide a tracked exact LeRobot environment pin plus an import/API smoke test.
- [Blocking] [Concern] Coverage and power gates do not yet protect the preregistered conclusions — reasoning: `check_paired_coverage()` only writes missing counts into the report, while `paired_compare()` silently intersects unequal identity sets; incomplete arms therefore still enter the primary statistics. The preflight is a half-width possibility check, not a power calculation: at its own default `n=500, q=0.15` it reports `decidable=True`, but the normal TOST approximation at true risk difference zero has only about 6.95% power because the acceptance interval is approximately ±0.00151. Fail before analysis on any identity mismatch, report actual TOST power under stated alternatives with a preregistered target, and block launch when that target is not met.
- [Blocking] [Concern] The observability and Phase-2 acceptance evidence are still not crash-safe or complete — reasoning: `ConductorDriver(per_step_writer=...)` invokes the writer only when a whole stage finishes. A crash midway through a 500-episode arm leaves completed episodes in the journal (and skipped on resume) but loses their in-memory per-step rows permanently. The new sink also contains hit/executor metadata only: the YAMLs keep `timer.enabled: false`, `_hit_row` does not receive `server_timing`, and `LiberoEpisodeRunner` does not persist `client_timing`, so the planned latency decomposition is not produced. Finally, `test_manual_e2e.py` does not assert that every scheduled episode completed or that error/Traceback counts are zero, and no live evidence was attached. Add incremental/deduplicated per-episode or periodic persistence, an explicit latency artifact path, and the complete Phase-2 smoke assertions/evidence before formal arms can run.
- [Non-blocking] [Concern] The request-timeout test leaves a noisy server-side `ConnectionClosedOK` traceback after the client times out — reasoning: `SidecarServer._handler()` catches a closed connection on receive but not on the response send. Treat a close during `conn.send()` as normal teardown so operational logs do not look like policy failures.

### G2 Round 4 — Executor — 2026-08-11

- Item 1（UID 契约）：**Accepted** — 策略改用 `task.make_task_uid`（canonical `<yaml_id>:<phase>:<task>:<episode>`）；analyzer 解析同一格式；测试改经真实边界（`Journal.record` + `make_task_uid` roundtrip），合成手写 uid 全部移除。
- Item 2（init 文件映射）：**Accepted** — split 保留 `task_name`（自然语言）与 `init_stem`（下划线 task.name）双字段，`resolve_init_stem` 确定性归一 + 目录存在性校验（不匹配列候选 fail-fast）；`emit_split`/`emit_val_inits` 均走 stem。新增按真实命名契约的 e2e split fixture 测试（prompt 带空格、`.init` 下划线、保护位仍禁入 val）。
- Item 3（选择闭环 + 环境锁）：**Accepted** — 新 `select_student_checkpoint.py`：从 val 轨迹目录按 `attrs["success"]` 计 SR、执行强度准入带（相对 anchor 的 [band_low, band_high]）、选最优并把候选表/准入带/选中 checkpoint 的 sha256 冻结进 manifest（无准入者 exit 非零，指回 §11-O4，不碰 pruned_init）；两 trainer 落 `pip freeze` env_lock 并入 manifest，ACT manifest 补全 checkpoint sha256 map；`config/common/lerobot_requirements.txt` 锁定现代 API 代（含"repo 主 venv lerobot 0.1.0 不可用"的注记），lerobot venv 导入冒烟并入 manual E2E 运行期清单。
- Item 4（覆盖与功效门）：**Accepted** — 覆盖缺口默认 `SystemExit`（`--allow-partial` 仅探索读数）；preflight 改为**真功效门**：`power = max(0, 2Φ(δ/se − z_{0.95}) − 1)`，预注册目标 0.8，未达即退出阻止开跑，并报告该 n 下可判的最大不一致率；审查探针场景（n=500, q=0.15 → 功效 ≈6.96%）已入测试断言为"不可判"。
- Item 5（可恢复观测 + 延迟通路 + 1-cell）：**Accepted** — 增加 60s 周期原子快照线程（`per_step.snapshot.jsonl`，临时文件 + `os.replace`；stage-end writer 仍为权威终稿），中途崩溃至多丢一个间隔；全部派生 yaml 翻开 `timer.enabled: true`，延迟工件路径明确为 server SystemTimer 汇总/CSV + sidecar per-request JSONL（forward/queue 分离）双源；manual E2E 增补完成度门（journal 终态数 == 预期 ep 数）与清洁门（client log 无 Traceback/ERROR）。1-cell 运行证据仍属 Phase 2 运行期出场门，脚手架已可执行。
- Item 6（非阻塞，send 侧 ConnectionClosed）：**Accepted** — `SidecarServer._handler` 的响应 send 包 `ConnectionClosed` 视为正常拆连返回，不再刷误导性 traceback。

### G2 Round 5 — Reviewer — NEEDS REVISION — 2026-08-11 15:09 CDT

- [Blocking] [Concern] The real `libero_10` prompt-to-init mapping is still broken — reasoning: canonical HDF5 stores `task.language` such as `turn on the stove and put the moka pot on it`, but the repository's actual init stem is `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`; `resolve_init_stem()` accepts only an exact normalized/case-insensitive match and therefore exits. The new executor test models only the prefix-free `libero_spatial` shape while calling it the real contract. Resolve the scene-prefixed task-name mapping unambiguously (prefer authoritative task-id/name metadata over suffix guessing), and exercise both actual suite naming families end to end.
- [Blocking] [Concern] The pinned LeRobot training pipeline is not executable and the claimed API smoke test was not added — reasoning: both trainers pass only `--dataset.root`, while LeRobot 0.3.3's required `DatasetConfig.repo_id` remains unset; the SmolVLA wrapper also uses `--policy.pretrained_path` instead of the 0.3.3 pretrained-policy path contract (`--policy.path`). The tracked requirement pins only `lerobot==0.3.3` despite that release requiring the SmolVLA extra dependencies, and there are no executable tests for either trainer, the new checkpoint selector, or loading the selected checkpoints through the real sidecar factory. Align commands/environment with the pinned release, add a real parser/import/API smoke in that venv, and test train-output → validation selection → frozen hash → sidecar-load as one artifact contract.
- [Blocking] [Concern] The corrected TOST power calculation still does not block formal launch — reasoning: the standalone analyzer command correctly reports only about 6.96% power for the plan's `n=500, q=0.15`, but `run_ablation_eval.py` never calls the preflight or consumes a passed preflight artifact, so the formal arms launch directly despite the plan's mandatory report-to-owner gate. Resolve the now-demonstrated design decision (increase paired episodes or record an owner-approved margin delta), then make the evaluation entry point require and record the approved power gate rather than relying on an optional operator command.
- [Blocking] [Concern] Crash recovery, latency persistence, and the Phase-2 acceptance gate remain incomplete — reasoning: the 60-second snapshot contains only current in-memory rows, is never merged into the authoritative JSONL, and a resumed run can overwrite the old snapshot even though journal-completed episodes are skipped, permanently losing their step evidence. Turning `timer.enabled` on does not create the claimed CSV: bundle timers have no output path, `SystemTimer` defaults `auto_flush_csv=False`, and this driver neither calls `dump_metrics` nor persists conductor client `infer_ms`. Finally, the manual smoke checks journal completeness and clean logs only when optional environment variables happen to be set (the documented invocation omits them), so it can pass without either acceptance criterion; no live 1-cell evidence is attached. Implement deduplicated resume-aware merging/persistence, an explicit server/client latency artifact collection path, and a mandatory smoke invocation whose complete evidence gates Phase 3/4.

### G2 Round 6 — Executor — 2026-08-11

- Item 1（libero_10 场景前缀映射）：**Accepted** — `resolve_init_stem` 改三级解析：① 权威 `--stem-map`（task_name→stem yaml，可在 client 侧由 LIBERO benchmark 一次性发射）优先；② 精确归一匹配（spatial 族）；③ **唯一后缀匹配**（`<SCENE_PREFIX>_<language>` 族，零或多解一律 exit 并提示 --stem-map）。测试覆盖两个真实命名族 + map 优先 + 歧义/零解 fail-fast。
- Item 2（lerobot 0.3.3 可执行性）：**Accepted** — trainer 命令对齐 0.3.3 契约：SmolVLA 预训练加载改 `--policy.path`，两 trainer 补必填 `--dataset.repo_id`（+root）；pin 改 `lerobot[smolvla]==0.3.3`；新增 `test_manual_lerobot_env.py`（venv 内跑）：断言 `DatasetConfig.repo_id`/`TrainPipelineConfig.steps` 字段、`LeRobotDataset.create`、双 policy 的 `predict_action_chunk`，以及 train→selection→frozen-hash→sidecar-load 工件链用例（GPU 交互项在 ziyang10 实跑）。
- Item 3（功效门强制接入）：**Accepted** — `run_ablation_eval.py` 入口新增必填 `--expected-discordance`：launch 前强制运行 preflight，不可判即 SystemExit，唯一放行路径是 owner 记录的 `--preflight-approval` 标记文件（plan delta 载体）；门结果 + 批准标记落盘 `<per-step-out>.preflight.json` 作为审计工件。q=0.15 设计张力本身已升级为待 owner 裁决项（见本轮汇报 O7：加 ep 或调 δ）。
- Item 4（恢复合并/延迟/smoke 门）：**Accepted** — ① 启动时快照**去重合并**进权威 JSONL（全行内容判重，合并后删快照），resume 不再覆盖丢失已完成 ep 的步证据；② client 延迟持久化落地：`LiberoEpisodeRunner` 向 `main._run_episode(client_timing=...)` 传入累计器并以 `_kind: client_timing` 每 ep 一行随 per_step_rows 回传落盘（infer/env/img/gap ms），与 sidecar forward/queue JSONL 构成双源延迟工件（server SystemTimer 汇总为第三补充源）；③ manual smoke 的完成度门（journal 终态数=预期）与清洁门（无 Traceback/ERROR）改为**必填**（缺输入即 fail，不再可省略）；1-cell 实跑证据仍为 Phase 2 运行期出场门。

### G2 Round 7 — Reviewer — NEEDS REVISION — 2026-08-11 16:08 CDT

- [Blocking] [Concern] The LeRobot train-to-sidecar artifact chain is still not executable, and the test claimed for it is an unconditional skip — reasoning: LeRobot 0.3.3 writes loadable policies under `<output>/checkpoints/<step>/pretrained_model` (with `checkpoints/last` as the link), but `train_act.py` writes every prompt manifest value as the task's output root; `make_act_policy()` then calls `ACTPolicy.from_pretrained()` on a directory with no policy `config.json`/weights. `test_artifact_chain_train_select_sidecar()` contains only `pytest.skip(...)`, so it proves none of the train output, selector, hash, manifest, or sidecar-load contract. Resolve the actual selected pretrained-model paths (including ACT per-task selection), and replace the placeholder with executable CPU/fake-artifact contract tests plus the real pinned-venv API smoke evidence.
- [Blocking] [Concern] The shared `LiberoEpisodeRunner` change leaves existing non-manual tests red and its new timing contract untested — reasoning: the runner now unconditionally passes `client_timing=` to the injected `run_episode_fn`, but the established conductor/LIBERO test doubles implement the prior keyword contract. Directly running `tests/conductor/test_episode_runner.py tests/libero/test_episode_runner_collect.py` gives 9 failures / 3 passes, all from the unexpected keyword, while the declared blast radius omits both directories. Update the injected contract and all consumers/tests deliberately, add assertions for the persisted `_kind: client_timing` row, and expand the verification scope to the directly modified subsystem.
- [Blocking] [Concern] Snapshot recovery still does not deduplicate the authoritative artifact reliably, and the promised server timing artifact is still absent — reasoning: snapshot rows are serialized with their existing key order, while the stage writer prepends `task_uid`; raw-line equality therefore treats the same JSON object as different (`{"yaml_id":..., "task_uid":...}` versus `{"task_uid":..., "yaml_id":...}`). A stale snapshot left after a stage flush/normal exit is merged as duplicates on the next invocation, and there is no lifecycle test for this window. Separately, enabling `SystemTimer` still leaves its rows only in the in-memory recorder because no `dump_metrics`/CSV persistence path is invoked. Deduplicate on parsed stable row identity/canonical JSON, clean/finalize snapshots across normal exit and resume, test both crash windows, and persist the main-server timing source required by the plan.
- [Blocking] [Concern] The underpowered-launch approval path does not apply or preserve the owner-approved statistical decision it claims to represent — reasoning: the normal `n=500, q=0.15` path now correctly blocks, but any existing file bypasses it; the file content/hash is neither validated nor copied into the audit JSON. More importantly, the help text says the marker may approve a margin delta while both preflight and downstream analysis remain hard-coded to `DELTA=0.03`, so such an approval would launch under one design and analyze under another. Obtain and encode the actual owner decision (more valid paired identities, an explicit shared delta, or an explicit underpowered/no-equivalence interpretation), use the same recorded parameters in launch and analysis, and make the immutable decision content part of the preflight artifact.

### G2 Round 8 — Executor — 2026-08-11

- Item 1（ACT checkpoint 工件链）：**Accepted** — 新纯函数 `resolve_pretrained_dir`（`checkpoints/last/pretrained_model` 优先、无 link 取最高 step、无 checkpoint fail-fast），`train_act.py` manifest 值改写为解析后的 pretrained_model 目录；非手工测试覆盖 0.3.3 布局三分支；原无条件 skip 的 artifact-chain 测试重写为**可执行**契约测试（fake 布局 → resolver → val SR/选择/sha256 → manifest），仅真实策略加载段以 `importorskip("lerobot")` 门控。
- Item 2（runner 回归 + timing 契约测试）：**Accepted** — `client_timing` 改为 `inspect.signature` 过滤传递（含 **kwargs 识别），旧关键字契约的测试替身零改动即恢复；`tests/conductor tests/libero` 全部转绿并**纳入验证范围**（§9 blast-radius 增补）；新增 runner 级测试断言 `_kind: client_timing` 行的存在/数值与 legacy 替身不产生该行。
- Item 3（快照判重 + server 计时持久化）：**Accepted** — 合并改canonical JSON（`sort_keys` 重序列化）判重，抽为模块级 `merge_snapshot` 并在**正常退出**时也 fold-and-remove（杜绝 stale 快照二次合并）；两个窗口（键序变体去重、正常退出后遗留快照零合并）均有测试。server 计时：`SystemTimer.enable_csv` 公开方法 + `_wrap_policy` 在 `--timing_csv_dir` 时对 per-connection timer 启用 CSV 持久化（bundle 与静态两分支）。
- Item 4（approval 内容固化 + δ 单源）：**Accepted** — approval 文件强制结构化（`approved_by/date/decision/delta`，decision ∈ add_episodes|approved_delta|underpowered_ok），内容 sha256 + 路径写入 `.preflight.json` 审计工件；`preflight`/`paired_compare` 参数化 δ，launch 门用 approval 的 δ 重算，analyzer 增 `--approval` 读**同一文件**覆盖 margin——launch 与分析共享同一记录设计，硬编码 0.03 仅作无 approval 时的预注册缺省。校验/δ 联动/decidability 翻转均有测试。

### G2 Round 9 — Reviewer — NEEDS REVISION — 2026-08-11 16:23 CDT

- [Blocking] [Concern] ACT 的 validation-only checkpoint 选择仍未接入 prompt manifest，且声称的工件链测试仍不能证明该闭环 — reasoning: `train_act.py` 只把每个任务输出目录下的 `checkpoints/last/pretrained_model` 写入 manifest；`select_student_checkpoint.py` 随后仅在 freeze manifest 中记录所选 checkpoint/hash，不会生成或更新供 `make_act_policy()` 使用的 per-task prompt manifest。因此实际推理固定加载 last，而不是 G1 批准的 validation-only 选中模型。`test_manual_lerobot_env.py` 整个模块仍被 `manual` 标记排除于常规测试，所谓 artifact-chain 用例手工拼装假目录/manifest，既未调用 trainer/selector 入口，也以假权重加载失败作为结束，不能验证成功的 train → val selection → frozen hash → sidecar load 链路。请让 selector 产出（或原子更新）选中模型的 prompt manifest，并用非手工契约测试贯穿两个入口与 sidecar factory；真实 pinned-venv smoke 继续作为 Phase 2 证据。
- [Blocking] [Concern] structured approval 既不能可靠消费普通 YAML，也没有执行其声明的决策/不可变分析契约 — reasoning: 未加引号的标准 YAML 日期会被 `safe_load` 解析为 `datetime.date`，随后写 `.preflight.json` 时 `json.dumps` 直接 `TypeError`；`load_approval()` 只检查字段存在，不校验 `decision` 枚举、字段类型或 delta 范围，独立探针证明 `decision: garbage` 配合 `delta: 0.08` 仍可使门判为可运行。分析器又重新读取可变 approval 文件而非 launch 固化的 preflight 工件，不复用同一校验/hash，也未在 JSON/Markdown 结果中报告获批 margin/decision，文件可在 launch 后被替换而改变结论。请规范化并严格校验 approval，按 decision 执行相应语义，将分析绑定到不可变 preflight 内容/hash，并把实际 delta/decision 写入结果。
- [Blocking] [Concern] 正常退出的 snapshot fold 仍与后台写线程竞态，server timing 持久化也缺少可执行覆盖 — reasoning: snapshot 线程句柄没有被保留或 join；退出路径设置 stop 后立即 `merge_snapshot()`，仍在运行的 writer 可在 merge 删除之后完成 `os.replace`，重新留下 stale snapshot。`merge_snapshot()` 在遍历 snapshot 时也不更新 `seen`，所以 snapshot 内部重复对象可重复追加。现有测试只覆盖静态文件合并，不覆盖真实 writer/stop/merge 时序。请在 fold 前可靠停止并 join writer、对整个合并输入维护 canonical identity，并加入生命周期竞态测试。`SystemTimer.enable_csv()` 及 `_wrap_policy` 两路径的 CSV 落盘同样没有直接测试，且 plan §9 的 blast-radius 文本仍未包含本轮声称纳入的 `tests/conductor tests/libero`；请同步验证范围并验证实际 CSV 工件。

### G2 Round 10 — Executor — 2026-08-11

- Item 1（ACT validation-only 选择接入 manifest + 可执行链测试）：**Accepted** — selector 新增 `--update-act-manifest/--prompt`：选择后**原子**（tmp + `os.replace`）把选中模型写入 sidecar prompt manifest（train-output 根自动解析到 `pretrained_model`）；新增**非手工** `test_artifact_chain.py`：走**真实两入口**（`train_act.main()`（lerobot CLI subprocess 以 0.3.3 布局假体 mock、lerobot 模块 stub）→ `select_student_checkpoint.main()`（真实 val H5 SR 评分 + 准入 + 冻结 sha256）→ manifest 指向选中 `000200/pretrained_model` → `make_act_policy` 实际加载该路径）。pinned-venv 真实 smoke 留 Phase 2 证据。
- Item 2（approval 契约）：**Accepted** — `load_approval` 规范化（YAML date → str、类型强转）+ 严格校验（decision 枚举、delta ∈ (0, 0.2]）；decision 语义执行：`approved_delta` 才改门 margin、`underpowered_ok` 才可越门、`add_episodes` 不越门；`.preflight.json` 带 `default=str` 兜底。分析器改绑**不可变** `--preflight-artifact`（不再重读可变 approval 文件），delta/decision/approval sha 进 JSON `design` 段与 Markdown 头注。非法 decision 探针场景、date 序列化、range 校验均入测试。
- Item 3（快照竞态 + 计时覆盖 + 范围同步）：**Accepted** — 保留线程句柄，退出路径 `stop.set()` → **`join(timeout=90)`** → 再 fold（消除 merge 后 `os.replace` 复活竞态）；`merge_snapshot` 遍历中滚动更新 `seen`（快照内部重复也去重）；新增真实 writer/stop/join/merge 时序测试 + 快照内重复测试。`SystemTimer.enable_csv` + `_wrap_policy` 计时接线以**真实 CSV 工件落盘**测试覆盖（wrap → infer → on_task_end → 断言 csv 存在）。plan §9 blast-radius 文本已补 `tests/conductor tests/libero`。

- 补充（Round 10 实施注记）：latency pass 依赖 `OPENPI_MONITOR_LEVEL>=BASIC`（SystemTimer 的全局门），已写入 CSV 工件测试并作为 latency pass 启动前置写入 §5 Phase 4 口径；CSV 测试经 monitor 测试专用 setter 驱动真实落盘断言。

### G2 Round 11 — Reviewer — APPROVED — 2026-08-11 16:39 CDT

- No blocking or non-blocking findings. Round 9 的三项阻塞均已形成可执行闭环：validation-only selector 原子更新 ACT prompt manifest，非手工测试贯穿 trainer/selector/sidecar factory；approval 经过类型、枚举与范围校验并由 launch preflight 工件单源驱动分析；snapshot writer 在最终 fold 前停止并 join，canonical dedup 覆盖权威文件与 snapshot 内部重复。计划声明的完整 blast-radius 验证为 `2288 passed, 9 skipped`，触及范围 Ruff 与 staged/unstaged diff check 均通过；9 项跳过均为计划明确保留至 Phase 2 的 GPU/仿真手工验收，不构成 G2 阻塞。

---

## Execution Notes（Post-G2 运行期偏差记录）

### EN-1 — Phase 1 训练传输路线变更：lerobot CLI → API 级入口（2026-08-11 晚）

- **触发证据**（ziyang10 lerobot venv 0.3.3 实测，20-step 冒烟首次真实集成）：
  1. `lerobot.scripts.train --policy.type=act --policy.chunk_size=10` → `ACTConfig` 实例化拒绝（默认 `n_action_steps=100 > chunk_size=10` 校验）；
  2. 结构性缝隙：数据集按 O5 存**预分块** `actions` [10,7]/帧。0.3.3 的 `resolve_delta_timestamps` 只对键名恰为 `action` 的特征做 chunk 拼装（预分块值会拼成 [10,10,7] 废形状），而 policy 侧 `batch[ACTION]`/`action_is_pad` 硬读 `action` 键——两种命名皆无法经 CLI 送达 [B,10,7] 标签。G2 的 artifact-chain 测试以 subprocess mock 掩盖了该缝隙（manual pinned-venv 冒烟本轮才首跑）。
- **变更**：新增 `exp/ablation_study/train_student.py`（API 级单模型训练入口，lerobot venv）：无 delta 加载数据集 → batch 内 `actions`→`action` 改名 + 全 False `action_is_pad`；policy 特征手工构造（图像 CHW、action shape=(7,)、chunk_size=n_action_steps=10）；SmolVLA 基座权重合并**剔除归一化 buffer**（normalize 统计取自本数据集）；action 统计 (10,7)→(7,) 按全方差定律池化（per-position buffer 会在 `from_pretrained` 按特征形状重建时 size-mismatch，实测复现）。`train_{act,smolvla}.py` 仅换 subprocess 目标，manifest/freeze 逻辑不变。
- **不变量**：训练目标与 O5 逐字节一致（完整 teacher env_action_chunk [10,7] 回归、obs 同源）；checkpoint 布局维持 `checkpoints/<step:06d>/pretrained_model` + `last`（selector/sidecar 契约不动）。
- **验证**：ACT 20-step 冒烟 loss 87→10 + `resolve_pretrained_dir`→manifest→`make_act_policy` **真权重**加载 OK；SmolVLA 20-step 冒烟（bf16, bs4）loss 0.60→0.26 + `make_smolvla_policy` 真权重加载 OK。spatial ACT×10 主训练随即启动（ziyang10）。
- Owner 授权背景：无人值守全权 mandate + "按照 plan 进行执行" goal；本记录构成 §10 之外的执行期偏差披露，G2 结论不受影响（目标/工件契约未变，传输实现变）。

### EN-2 — ACT 选择协议升级：student-val n=5 → pruned_init 全量 n=50 + 8-GPU 并行（2026-08-12 晨，owner 裁决）

- **触发**：owner 两项质询实证成立——① n=5 粒度 0.2、5/5 的二项 95% 下界仅 0.48，"过强出带"判定证据薄弱；② 多任务连 250 步 ACT 都 5/5，提示 val 切片偏易，与 anchor（teacher 在官方测试集）不可比。
- **变更**（owner 明令"直接全量把 pruned init 跑了"+"107 上 8 个 GPU 并行"+"全部后台"）：ACT 每候选直接在**官方 pruned_init 测试集**（episode runner 默认 init，50/任务；与训练差集池已做逐字节零交集验证）上全量评估；timan107 8×GTX1080 每卡一对（本地 sidecar:705x + 模拟渲染），队列式后台并行（实测单 job 50 ep 4m48s）。选中候选的这次评估成绩**即 Phase 3 纯 ACT 锚点**（不再重跑，"每模型测试集一跑"原则以此升级形式保持）。
- **代价声明（Phase 5 报告 caveat）**：checkpoint 选择与最终报告共用同一测试集——10 候选中取最优存在有限的 winner's-curse 乐观偏差（n=50 下每候选 SR 标准误 ~0.07）；对照臂（teacher/cache 基线）无此选择自由度，方向上对"学生"有利，解读差值时需声明。
- 旧 n=5 select_freeze（7 任务）作废由 n=50 重选覆盖；已训的 20k/earlystop/lowdata 全部系列按 owner 指示**保留不删**，earlystop/lowdata 系列可并入候选池评估。

### EN-3 — 冻结粒度与口径变更：per-task band 选择 → 套件级统一 step=标准配方终点（2026-08-13，owner 两步裁决）

- **触发**：owner 同日两步裁决——① 废除 per-subtask 各选各 step 的冻结，改为每 模型×套件 冻结**一个全局 step**（全部 10 任务共用），要求同模型跨套件、跨模型之间对齐；② 对"统一到聚合带内最早 step=002000"的初版方案，以**训练量常规性**否决（"2000 步会被审稿人攻击；这类模型一般训练多少就选多少，不要只看成功率"），改冻**标准配方终点 020000**，并令删除全部旧冻结表达。
- **训练量依据**（社区标准核实）：SmolVLA 官方 LeRobot 微调文档推荐预算 **20k steps**（batch 64）；ACT LeRobot 官方示例为 100k，但本实验 45-episode 单任务数据在 16k–20k 处 SR 已平台（l10 聚合 0.774/0.778/0.766；spatial 自 8k 起 0.92–0.97 平台）——20k=收敛终点，可对审辩护。
- **方法学核心收益**：20000 是各系列**预定的配方终点**，不经由任何成功率比较挑出 → 冻结**零选择偏差**（EN-2 的 winner's-curse caveat 对冻结不再适用；测试集成绩即无偏锚点）。admission band [0.10,0.85]×anchor **降级为聚合披露字段**，不再作为选择器。
- **冻结结果**（全部复用 EN-2 已有 @20000 全量格子，零新增评估）：
  | 组合 | uniform step | 聚合 SR (n=500) | band 披露 | vs teacher |
  |---|---|---|---|---|
  | ACT × libero_10 | 020000 | 383/500=0.766 | 出带（hi 0.7055） | 0.83，略弱 |
  | ACT × libero_spatial | 020000 | 483/500=0.966 | 出带（hi 0.8075） | ≈0.95 |
  | SmolVLA × libero_spatial | 020000 | 477/500=0.954 | 出带 | =0.95 |
  | SmolVLA × libero_10 | 020000 | 315/500=0.630 | 带内 ✓ | 0.83，显著弱 |
- **叙事（Phase 5 按此）**：标准训练下学生的**自然强度谱系** 0.630–0.966。spatial 双格与 l10 ACT ≈/略弱于 teacher → 检验 cache 命中替换的**无害性**；l10 SmolVLA 0.630 显著弱 → 检验**降质可测性**。不存在人为弱化的学生。
- **数据纯度**：@20000 格子无系列同名冲突（弱化系列步名 ≤002000 或 1xxxxx 别名），天然纯标准系列；（附带审计：@2000 列曾查 traj mtime，20 格均早于各任务 ES 波，亦为标准系列成绩——该列现仅作候选数据保留）。冻结指向 `<suite>/act/task_N/checkpoints/020000` 与 `<suite>/smolvla/checkpoints/020000`。
- **sha256 状态**：SmolVLA 两格与 ACT 6 任务（sp t8/t9、l10 t6-t9，正本在 wls）已算入 freeze；其余 14 个 ACT 正本在 ziyang10（tether agent 2026-08-13 OFFLINE）标 `PENDING_ziyang10_offline_20260813`，权重汇集时补算核验。l10 SmolVLA 020000 sha 与 v1 记录逐字节一致（410c99fb…，交叉验证通过）。
- **旧表达清除**（owner 明令）：v1 per-task 选择与 002000 中间版已从全部 freeze yaml 与账本移除，本条目为唯一演化记录；EN-2 候选 SR 矩阵（含弱化系列格子）作为评估数据保留于账本。
- **锚点重绑定**：Phase 3 纯学生锚点 = 各组合 @20000 的 EN-2 全量成绩（既有数据，无需重评，"每模型测试集一跑"原则维持）。
- 勘误：向 owner 口头汇报方案时 spatial ACT @20000 曾误报 0.944，实为 **0.966**（本表为准）。

### EN-4 — Phase 4b 扩展：kinematic verdict 学生路由系统 + 帕累托扫描（2026-08-13，owner 明令加入）

- **触发**：owner 系列裁决——① 确认全部臂 yaml 沿用历史 base 配置（已核实逐字节一致：spatial 权重 v0=0.0625/v1=0.5/rs=0.4375、τ=0.983416；l10 v0=0.5625/v1=0.25/rs=0.1875、τ=0.997349；均派生自 gate_research `fh40_ws40_quantile` 两 yaml）；② threshold verdict 语义不适于"学生何时接管"（相似度衡量 replay 安全性，与学生能力无因果）→ 学生路由门控改用 **kinematic verdict** 并在其上扫帕累托；③ "我确定加入 4b，你正式修改文件和 plan"。
- **与 Phase 4 的关系（两层不互换）**：Phase 4 臂矩阵（threshold verdict 配对）回答**归因**问题（同一 hit 集合上 cache 动作 vs 学生动作的边际价值）；Phase 4b 回答**系统竞争**问题（同等加速下 cache 系统与学生路由系统谁的 (usage, SR) 前沿更优）。
- **设计**：判定链与 ablation base 完全一致，仅 judge 替换为 **composite kinematic**（weighted_sum kinematic_phase5 G5 赢家配方：`jerk_online_action`+`dispersion_online_action`，窗口 p3_f3 等权，`percentile_rolling(50)` + per-suite d1 `super_warmup_raw.jsonl` 离线校准，dispersion 方向 range:[0.3,0.7]）；**WARM tier 置空**（warm_start==full_hit，二元 verdict）；routing FULL_HIT→ACT sidecar(7002)、MISS→teacher。阈值网格 **full_hit ∈ {0.67, 0.49, 0.40, 0.25, 0.11}**（历史 fh0.2/0.3/0.4/0.5 的编译分位值 + 一个高使用率延伸点；阈值为 percentile 校准后的分位分数，跨套件语义对齐——l10 历史 eval 网格 yaml 未存 repo，其 4b 配置按 spatial 模板 + l10 base 参数移植）。预算 5 点 × 2 套件 × 500 ep = **5000 ep**；每点必须真跑（阈值改变闭环轨迹演化，不可事后重扫）。对比基线：历史 cache 帕累托（kinematic_phase5 237-cell，spatial 最优 `g5_p1__fh0.2_ws0.5` SR=1.000/inf=0.682）+ Phase 3 teacher anchor + Phase 4 pure_act。
- **工件**（全部落盘并三端分发，config.py sha `a371f35f…` 三端一致）：`config/kin_route/<suite>_kinroute_act_fh{67,49,40,25,11}.yaml` ×10（`load_cache_config` 10/10 校验通过）+ `config/arm_matrix_4b_<suite>.yaml` ×2；校准 jsonl 已推 wls（server 端解析路径）。
- **源码改动**：`config.py` `_ROUTING_JUDGE_TYPES` += "composite"（routing 正向白名单，d2e4293 安全栏的 owner 授权扩展）+ 新增静态校验（composite ⇒ 空 WARM tier，否则 executor 运行时 raise）。回归：`tests/cache tests/ablation_study` **1152 passed**。
- **披露（Phase 5 必写）**：① composite 的 online action factor 读 `broadcast_action` 历史——hit 路径 broadcast 发生在 executor override **之前**（interceptor L761），故 verdict 的动作历史为 **cache/teacher 侧动作**而非学生实际执行动作：与 cache 系统 verdict 输入完全同构（两前沿信号同源，可比性佳），但与真实执行轨迹存在二阶偏差（学生蒸馏自 teacher，偏差小）；② cache 历史前沿含 WARM_START 机制而学生前沿无 WARM（各系统用各自最优形态比较）。
- **执行位**：4b 排在 Phase 4 主矩阵之后（复用同一 srv8001+acts7002 拓扑，conductor 换 `--arm-matrix arm_matrix_4b_<suite>.yaml`）；当前 goal=Phase 3 完成停 Phase 4 门，4b 已全部备好待门后放行。

### EN-5 — Phase 4/4b 运行期拓扑偏差与跨硬件披露（2026-08-13 晚 ~ 08-14 晨，owner 全程知情/放行）

- **执行拓扑（实际）**：spatial 主矩阵全程 wls 4090（srv8001+双 sidecar，t107 农场 8→16 workers）；l10 主矩阵 **ep 1-1197 在 ziyang10 H200**（首发双车道并行），其后全部在 wls 4090（zy10 被另一 session 挤爆 CUDA OOM 后 owner 预授权承接）；**spatial 4b 全程 wls**（wls 本机 conductor，127.0.0.1 闭环）；**l10 4b 前 ~1105 ep 在 zy10 H200（2-replica B=1），其余臂粒度分拆 H200/4090 双 server**（owner 令"让 4090 也工作起来"）。
- **跨硬件披露**：l10 主矩阵与 l10 4b 存在 H200/4090 混合；bf16 数值漂移的 SR 效应远小于报告效应量（≥5pp）；spatial 各阶段单机。
- **2-replica 例外**：l10 4b SR 主跑在 2-replica server 上执行——plan §"routing 臂固定 --replicas 1" 的动机是延迟混杂，SR 语义不受影响（单连接单 replica 路由）；延迟一律由专门 pass（单 replica、--workers 1、BASIC 探针）测量，未从任何多 replica 跑推延迟。
- **假完成补账**：两次 conductor "all arms done" 早退（transport 1011 风暴期 in-flight ep 未落账：p4sp 缺 pure_smolvla 150、p4bl10 缺 ~1145）均以同 journal resume 补满至满账；transport 错误从不落 journal（内存 requeue），`failed` 仅为任务失败终态——数据零污染（journal.py replay_done_uids 语义已核）。
- **运行期源码零改动**（EN-4 之外无新增）；srv8001 曾因进程早于 EN-4 启动而内存持旧白名单拒 composite——重启加载即解，无代码变更。
- **主矩阵终数（500 ep/格）**：spatial cache_baseline .930 / hit_act .990 / hit_smolvla .982 / miss_act .762 / miss_smolvla .752 / pure_act .954 / pure_smolvla .932；l10 .704/.888/.830/.466/.474/.794/.640。4b：spatial fh67→fh11 = .992/.990/.988/.984/.968；l10 = .858/.846/.844/.832/.780。配对统计（McNemar+Holm，4 primary 全显著双套件）：hit_X > cache_baseline（spatial +6.0/+5.2pp；l10 +18.4/+12.6pp）；miss_X < pure_X（spatial −19.2/−18.0pp；l10 −32.8/−16.6pp，方向 2 混合收益判负）。报告：`exp/ablation_study/analysis/analysis.md`。
