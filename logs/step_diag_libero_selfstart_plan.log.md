# step_diag：LIBERO 上复现 warm reset / 自产起点消融 — 计划

> 状态：`In Progress`（2026-09-24 立项）。级别 L2（`exp/step_diag` 多文件 + 新驱动路径）。**G1 由 owner 当次豁免**（2026-09-24「免除 G1，但要停在 G2 接收外审」）；本文件是 G2 审查的对照依据。实现由子代理完成、执行方验收。
> 上位：`exp/step_diag/analysis/step_vs_warmstart.md` §6（RoboCasa365 warm reset 各变体）与同日 RoboCasa 自产起点消融 `sdiag_self13`（代码已在工作树，owner 当次裁定 L1 不审查、正在跑）。

## 1. 目的

在 LIBERO spatial 与 libero_10 上，对 π0.5 与 GR00T N1.5 复现 RoboCasa365 macro-13 页面上的全部配置：full、纯减步、我们的 warm start（精确续跑）、缓存起点的 warm reset 各变体、自产起点（当场直接推理，不用 cache）的同一组 warm reset 变体。回答：RoboCasa 上「reset 式 warm reset 提升 π0.5 成功率、缩短成功集、且自产起点同样有效」在 LIBERO 上是否成立。

## 2. 记法与环境

- 记法同网页：T = 起点动作所处的 t（0 = 最终动作），N = 实际执行的去噪步数，t = 传给模型的 t（π0.5 记法：1 噪声、0 干净；GR00T 原生时间 = 1 − t）。
- 套件：`libero_spatial`、`libero_10`；初始状态池 = 官方 pruned A 池（每任务 50，10 任务，**每臂每套件 500 集**），与既有 LIBERO shadow（`run_libero_diag.py`）同一池、同一 seed 约定。
- 环境：`pi05_libero_{spatial,10}`（K=10）、`groot_libero_{spatial,10}`（K=8，`groot_n15_k8_v1`）。缓存库沿用 shadow yaml 的库（π0.5 快照 t∈{0.1,0.2,0.3}；GR00T 原生 {0.875,0.75,0.5}）。
- 实验 id：`sdiag_libero_self`；数据根独立（驱动 `data/libero_self`，server `data/server_libero_self`），不与既有 LIBERO shadow 混。

## 3. 臂（每套件）

### 3.1 π0.5（K=10，9 臂，与 RoboCasa 逐项相同）

| 臂 | 起点 | T | N | t 序列 |
|---|---|---|---|---|
| `full` | 噪声 | — | 10 | 1, 0.9, …, 0.1 |
| `plain_k2` | 噪声 | — | 2 | 1, 0.5 |
| `warm_t0.2`（ours，精确续跑） | 缓存快照 | 0.2 | 2 | 0.2, 0.1 |
| `warmreset_t0.2` / `selfwarmreset_t0.2` | 缓存 / 自产快照 | 0.2 | 2 | 1, 0.5 |
| `resetfinal_t0.2` / `selfresetfinal_t0.2` | 缓存 / 自产最终动作 | 0 | 2 | 1, 0.5 |
| `midfinal_t0.2` / `selfmidfinal_t0.2` | 缓存 / 自产最终动作 | 0 | 2 | 0.9, 0.45 |

### 3.2 GR00T（K=8，29 臂）

owner 裁定：warm reset 各臂的 (T, N, t) 与 RoboCasa（K=4）**逐项相同**；K=8 网格上有 T=0.25（原生 0.75）与 T=0.5（原生 0.5）的快照，因此需要把**步数 N 与快照 t 解耦**（RoboCasa 里 N 由 `remaining_steps(start_t)` 推出）。精确续跑（ours）只能在原生 8 步网格上走，取 T=0.125/N=1 与 T=0.25/N=2。

| 臂 | 起点 | T | N | t 序列（π0.5 记法） |
|---|---|---|---|---|
| `full` | 噪声 | — | 8 | 1, 0.875, …, 0.125 |
| `plain_k1` / `plain_k2` | 噪声 | — | 1 / 2 | 1 / 1, 0.5 |
| `warm_t0.875`（ours） | 缓存快照 | 0.125 | 1 | 0.125 |
| `warm_t0.75`（ours） | 缓存快照 | 0.25 | 2 | 0.25, 0.125 |
| N=1，快照 T=0.25（缓存 / 自产各一） | 快照 | 0.25 | 1 | 1 ／ 0.75 ／ 0.5 |
| N=1，最终动作 T=0（缓存 / 自产各一） | 最终动作 | 0 | 1 | 1 ／ 0.75 ／ 0.5 |
| N=2，快照 T=0.5（缓存 / 自产各一） | 快照 | 0.5 | 2 | 1, 0.5 ／ 0.75, 0.375 ／ 0.5, 0.25 |
| N=2，最终动作 T=0（缓存 / 自产各一） | 最终动作 | 0 | 2 | 1, 0.5 ／ 0.75, 0.375 ／ 0.5, 0.25 |

合计 5 + 12 缓存 + 12 自产 = 29 臂。说明：K=8 上 T=0.5 在 t=0.5 喂入走 2 步（0.5, 0.25）**不是**精确续跑（精确续跑要 4 步 dt=1/8），因此作为 warm reset 臂保留（RoboCasa 上它与 ours 等价、未单列自产臂）。

臂名：GR00T LIBERO 的 warm reset 臂写成 `<mode>_t<原生 start_t>_n<N>`（例 `midreset_t0.75_n1`、`selfresetfinal_t0.5_n2`），mode 沿用 `envs.WARM_VARIANT_MODES` / `SELF_VARIANT_MODES`；最终动作臂的 start_t 只决定检索哪一格快照的条目（取其 payload 的最终动作），与同 N 的快照臂共用 yaml。ours 臂沿用 `warm_t<start_t>`。π0.5 臂名与 RoboCasa 相同。

## 4. 实现

1. **envs**：LIBERO 实验 id、两 policy 的臂表；`validate_arm` 放行 LIBERO warm / variant / self 臂（t 属于该 env 的 `warm_ts`，judge = `always_warm_start start_t`）；GR00T LIBERO 臂名解析出 (mode, start_t, N)。RoboCasa 行为不变（`sdiag_self13` 正在跑，其臂名与语义不得改变）。
2. **yaml**：LIBERO warm yaml = 同 env 的 `shadow.yaml` 仅把 `checkpoints.cp1.judge` 换成 `{type: always_warm_start, start_t: <t>}`（与 RoboCasa `warm_t*.yaml` 相对 `shadow.yaml` 的差异完全相同）；`emit_arms` 放行 LIBERO warm cell 并校验库里有该 t 的快照。
3. **GR00T 步数解耦**：`groot_warm_variant_stage3` / `install_warm_variant` 接受显式 `num_steps`（缺省 = `remaining_steps(start_t)`，RoboCasa 行为不变）；自产起点同样取 K=8 直接推理在原生 start_t 的快照。证据行 `executed_steps = N`。
4. **服务端**：π0.5 LIBERO 走现有 `serve_diag_pi05`（单连接）。GR00T LIBERO：`serve_diag_groot --benchmark libero` 目前只接 shadow；扩成 full / plain（`--exec-steps`）/ warm / 变体 / 自产，与 RoboCasa 相同的 `GrootEvidencePolicy` + `GrootCacheInterceptor` + `install_warm_variant` 组合，每连接一个 DiagSession；`ops/serve_{pi05,groot}.sh` 放行新臂。
5. **驱动**：`run_libero_diag.py` 从 shadow-only 泛化为任意臂：launch 清单（期望身份 = 任务 × 池内 init_idx）、journal、per_step `episode_summary`、`--run-prefix`（同 RoboCasa，避免同槽位顺序任务撞 run_id）、按任务子集运行（队列按 (臂, 套件, 任务) 切成 50 集的 cell）。
6. **分析**：`warm_variants.py` 与 `success_length.py` 支持 LIBERO env（任务 = LIBERO 任务，配对身份 = (env, 任务, 初始状态池 SHA, init_idx, env_seed)），输出 self vs 缓存臂 / full / 纯减步 的配对差与成功集推理调用次数。
7. **测试**：GR00T K=8 各臂的 (快照, t 序列, dt, N)；解耦 N 缺省时与旧行为逐位一致；臂表 / 校验；yaml 生成；驱动参数与期望身份；分析在合成数据上的配对与宏观。

## 5. 运行（G2 通过后）

两对机器 h100 ↔ timan108、weilandserver ↔ timan107（timan107 避开 GPU3），调度器同 `ops/self13_queue.py` 的槽位 / 预算 / 续跑机制；RoboCasa `sdiag_self13` 未跑完的部分优先保留其算力，LIBERO 随其释放接上。约 38 臂 × 2 套件 × 500 ≈ 3.8 万集。

## 6. 不做的事

- 不改 RoboCasa 已跑与在跑的臂语义；不同步到任何远端 / 岛树（G2 前）。
- 不沿用 nfe_baseline 阶梯的 LIBERO full / plain 数据（驱动不同，配对身份不一致），全部在本驱动下重跑。

## 7. 实现记录（G2 交付）

**变更集**（工作树，未暂存、未提交；审查范围 = 下列全部 `exp/step_diag` / `tests/exp/step_diag` 改动）：

| 文件 | 内容 |
|---|---|
| `envs.py` | LIBERO 轮常量（`LIBERO_SELF_EXPERIMENT_ID`、套件、seed 7、50 集）与臂表（π0.5 9、GR00T 29）；`split_warm_steps` / `warm_steps_of` / `executed_steps_of`（`_n<N>` 后缀）；`validate_arm` 放行 LIBERO warm / 变体 / 自产臂（t ∈ env `warm_ts`，GR00T 变体须带 `_n<N>`，精确续跑与 π0.5 不得带）；RoboCasa 自产起点（`SELF_VARIANT_MODES`、`SELF13_*`）与 GR00T shoot 消融（`GROOT_SHOOT_MODES`、`SELF_SHOOT_MODES`、`SHOOT_ENTRY_T`、`is_self_mode`，仅 RoboCasa） |
| `groot.py` | `groot_self_start`（自产起点）、`_SELF`；`groot_warm_variant_stage3` / `install_warm_variant` 的 `num_steps`（缺省 = `remaining_steps(start_t)`，RoboCasa 不变）；`shoot_denoise_loop` 与 shoot 分派 |
| `pi05.py`、`recorder.py`、`serve_diag_pi05.py` | π0.5 自产起点（`_self_start`、`DiagSession.self_start_seed`）；LIBERO 无需代码改动 |
| `serve_diag_groot.py` | LIBERO 全模式（full / plain / warm / 变体 / 自产）：每连接 `GrootEvidencePolicy(GrootCacheInterceptor)` + `install_warm_variant(num_steps=…)`；`--denoising-steps` 与臂一致性校验；shoot / 自产模式表 |
| `emit_arms.py`、`config/arms/index.json` + 12 份 LIBERO `warm_t*.yaml` | LIBERO warm cell（shadow.yaml 仅替换 judge），库快照校验；yaml 被 `.gitignore` 覆盖，入库需 `git add -f` |
| `run_libero_diag.py` | 任意臂：launch 清单 / journal / `episode_summary`、`--arm-id` `--tasks` `--episodes` `--run-prefix`、`check_cell`；shadow 行为与旧 run id 不变 |
| `run_diag.py`、`ops/run_rc_cell.sh` | RoboCasa：`sdiag_self13` 准入分支、`--run-prefix` / `SD_RUN_PREFIX` |
| `ops/serve_{pi05,groot}.sh` | 新模式；LIBERO 非 shadow 输出 `<root>/<teacher>/<env_id>/<arm_id>` |
| `ops/self13_queue.py`（新） | RoboCasa `sdiag_self13` 调度器（在跑） |
| `analysis/aggregate_arms.py`、`warm_variants.py`、`success_length.py`（新） | LIBERO 分析（`analyze_libero` 等）、按 env 取 K；self vs 缓存 / full / plain 配对；成功集推理调用次数 |
| 测试：`test_libero_selfstart.py`（新，82）、`test_self_start.py`（新）、`test_groot_shoot.py`（新）；`test_emit_arms.py`、`test_serve_parse.py`、`test_groot_warm_variants.py`（按新规则更新） | |

**计划符合性**：LIBERO 部分完全按本计划 §2–§4、§7 实现，偏差如下（均为执行方裁定、未改变实验语义）：
1. LIBERO warm yaml 对 env 的全部 `warm_ts` 生成（π0.5 0.1/0.2/0.3，GR00T 0.875/0.75/0.5），多于臂实际用到的 t。
2. `validate_arm` 只做结构校验；本轮 (t, N) 组合的精确清单由驱动按 `LIBERO_SELF_ARMS_BY_POLICY` 强制。
3. 驱动另接受含 `smoke` 的实验 id（任意臂、1–50 集）供冒烟；其他实验 id 只允许 shadow。
4. §5 的 LIBERO cell 启动脚本与调度器属运行阶段，未实现（G2 后做）。
5. 同一变更集里包含本线另两块已在跑、owner 当次裁定 L1 不审查的 RoboCasa 改动：自产起点消融（2026-09-24）与 GR00T shoot 消融（2026-09-24，仅 RoboCasa，`validate_arm` 拒绝其用于 LIBERO）；它们与 LIBERO 共用上述文件，一并提交审查。

**本地测试（advisory，非 §6 Verify）**：`CUDA_VISIBLE_DEVICES="" uv run pytest tests/exp/step_diag -q` → `242 passed, 1 skipped`（2026-09-24 19:40 CDT）。另：`warm_variants` 重构前后对合成 RoboCasa 数据的 JSON / markdown 逐字节一致；`success_length` RoboCasa 路径复现既有 `data/analysis/success_length.json` 全部数值。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-24 19:54 CDT

**审查身份与范围**：Authority: Review；L2 / G2。接受 owner Ziyang Lin 当次 G1 豁免，本计划正文为设计对照，不补做 G1。基线为分支 `Ziyang`、HEAD `2dc90d0`；Round 1 无前轮暂存基线。按 §7 审查当前全部 `exp/step_diag` / `tests/exp/step_diag` 变更及 12 份 LIBERO warm YAML，包括共用文件中的 RoboCasa self / shoot 实现；尊重其 L1 裁定，不将既有 RoboCasa 运行认定为本次 LIBERO 越过 G2。`exp/trace_dual/` 与 `logs/session_handoff.md` 的独立任务改动不纳入本轮快照。未改源代码，未启动 LIBERO rollout。

**上下文**：已阅读 `CLAUDE.md`、`WORKING_AGREEMENT.md`、`protocols/review_authority.md`，从 docs/logs 索引阅读相关项目参考、cache 系统架构（含 GR00T）、conductor、数据采集及实验产物约定；结合原 step-vs-warmstart 计划、本计划、结果报告 §6、全部交付代码与测试，沿 LIBERO 驱动 → worker → 每连接 evidence policy/interceptor → journal/server rows → 准入与分析核查。没有阅读执行法。

**Constitutional Violation / 交付合规缺口**：

- `WORKING_AGREEMENT.md` §3.2 要求公共函数有 docstring；本轮新增的若干公共函数没有，见 B6。
- §5 Index Sync Rule 要求 logs 文档与 `logs/README.md` 同提交更新。新计划尚未登记，而本轮交付没有索引变更，见 B6。当前没有提交，因此这是待补齐的 G2 交付缺口，不是声称已经发生了违规提交。

**G2 checklist**：

| 项目 | 判定与依据 |
|---|---|
| 与计划一致 | **未通过**。9/29 臂、LIBERO K=8、显式 N 解耦、默认旧行为、YAML 与隔离路径符合设计；但正式 50 集队列完整性、同一初始状态配对及自产证据未被分析端可靠约束（B1–B3），成功长度也未可靠绑定 accepted attempt（B5）。 |
| 测试覆盖与通过 | **未通过**。执行方专项测试已独立复跑通过；新增独立检查 13 项中 3 个正控通过、10 个负例断言失败，稳定复现 B1–B5。现有 happy-path 测试不能替代拒收测试。 |
| 文档与索引 | **未通过**。计划、ops 与报告已有说明，但新计划缺 logs 索引、新公共函数缺 docstring（B6）；self 的 continuation N 与实际 K+N 成本亦须明确区分（B4）。 |
| 无回归 | **所测旧路径通过，整体暂不放行**。专项和相邻模块测试未发现旧缓存/GR00T/LIBERO/conductor 路径回归；新分析路径存在可复现的数据误准入与计数问题。未进行真实 GPU 模型 parity 或 §6 Verify，不能据 CPU 测试声称这部分完成。 |

**可执行修订项**：

- [Blocking] [Concern] **B1 — 跨臂配对丢失 LIBERO 初始状态池身份。** `analysis/aggregate_arms.py:347` 的 `pair_identity` 不含 `init_pool_sha256`；`analysis/warm_variants.py:212` 的跨臂门只比较模型/环境契约；`analysis/success_length.py:105` 更压缩为 `(task, init_idx)`。— reasoning: 在两个任务、每任务 50 集的有效合成数据上，仅令一个臂的 launch、terminal、worker/server 证据一致地换成另一池 SHA，逐臂检查仍通过，warm 分析仍产出进入 macro 的配对差，长度分析仍给出 100 对且无问题。同一 `init_idx` 在不同池不代表同一初始状态。修订要求：在两条分析路径保存并校验完整 LIBERO 配对身份（至少 env/suite、task、pool SHA、init_idx、env_seed），明确报跨池/跨 seed 不匹配并阻止其进入正式配对统计；补同池正控及异池负例。

- [Blocking] [Concern] **B2 — 正式轮的完整性由输入清单自行缩小。** `analysis/aggregate_arms.py:323` 仅核对已加载 launch 声明的集合；`analysis/warm_variants.py:195` 及 `analyze_libero` 没有针对 `sdiag_libero_self` 核验冻结的 50 身份集合。— reasoning: 两任务每臂仅声明、完成 4 集，仍同时得到 `complete=True`、`equal_nfe=True` 并进入 macro；这不满足本计划 §2 的每任务 50 集。修订要求：正式分析校验 experiment、seed 约定、每任务 `init_idx=0..49` 的完整集合及跨臂集合一致性，拒绝截断清单/混入 smoke；任务子集可作运行进度，但须明确区分局部完整与每套件 10×50 的正式完成状态，不能把局部统计当作正式全量结论。

- [Blocking] [Concern] **B3 — self 臂未要求任何自产推理证据。** `analysis/aggregate_arms.py:295` 起的决策准入只检查公共 schedule、hit、steps 和 stage3 次数，没有验证 `self_start`、`self_seed`、`self_direct_nfe`。— reasoning: self 臂分别缺失全部自产字段、标成 `self_start=False`、或将 `self_direct_nfe` 改成 0，三个负例均仍被标为 `equal_nfe=True` 且无准入问题，无法证明观察到的是本实验要测的当场自产起点。修订要求：按 arm 类型核验这些字段，要求自产标记、符合会话/决策约定的 seed 和该环境 K 次 direct inference；缺失或不一致应明确拒收；补 self/cache 错标及字段缺失测试。这里要求的是证据准入，不要求把 direct inference 加入现有 continuation `n_stage3_calls`。

- [Blocking] [Concern] **B4 — `episode_total_nfe` 对 self 臂漏算 K 步起点生成。** `analysis/aggregate_arms.py:319`–`:332` 只累加 continuation `executed_steps`；`warm_variants._task_cells` 也未输出自产成本，新增 LIBERO 分析文档仍称 equal-NFE panel。— reasoning: GR00T LIBERO K=8、N=1、3 次决策且每行已记录 `self_direct_nfe=8` 的正当 self 数据，当前 `episode_total_nfe` 是 3，实际动作头前向为 27。同 N 的消融设计可以保留，但不能由此把 self 与 cache 描述为总计算量相同。修订要求：分别暴露 continuation NFE、自生产起点 NFE、真实总 NFE，或将旧字段明确重命名为 continuation 并新增总量；在面板/说明中解释 self 为 K+N。补 π0.5 K=10 和 GR00T K=8 的计数检查，保持现有 N 与 stage3 调用语义不变。

- [Blocking] [Concern] **B5 — 成功长度分析会采纳无效摘要/错误终局。** `analysis/success_length.py:79`–`:105` 收集所有 summary，按 launch 的 driver run 筛选后直接采用长度与 journal success，没有检查 summary 的 accepted、success/error 与 terminal 的一致性，也没有完整传递 manifest 准入问题。— reasoning: 将 summary 置为 `accepted=False`、令 summary.success 与成功 terminal 相反、或为成功 terminal 加入 error，三个负例仍输出全部 100 集成功长度且 `problems=[]`。这会污染成功集和配对长度，违背该文件自身“accepted attempt”口径。修订要求：长度只能来自身份/run/attempt/arm/config 一致的有效 accepted terminal 和 accepted summary，显式拒收 error、结果冲突及 manifest 问题；补失败/重试混杂的负例，适用于共用的 RoboCasa 与 LIBERO 路径。

- [Blocking] [Concern] **B6 — 补齐交付文档规范。** 新 `logs/step_diag_libero_selfstart_plan.log.md` 没有登记到 `logs/README.md`；新公共函数缺少 docstring，例如 `run_libero_diag.build_parser`、`warm_variants.markdown_libero`、`success_length.success_length/main` 及新 `ops/self13_queue.py` 多个公共函数。— reasoning: 分别不满足 G2 文档/索引项及 `WORKING_AGREEMENT.md` §3.2、§5。由执行方将索引与计划同批交付，并补新增公共 API 说明；审查者不能越权修改这些文件。

- [Non-blocking] [Concern] **C1 — 真实服务验证仍属于后续交付。** — reasoning: 本轮没有运行 GPU 权重或闭环 rollout；已记录的 LIBERO launcher/调度器延至运行阶段与本计划 §7 偏差说明一致，不据此增加本轮阻断项。G2 修订通过后，仍需按项目流程完成 Verify 与真实服务 parity/冒烟，再启动正式 LIBERO 轮；本报告的 CPU 测试结果不替代这些证据。

**独立验证记录**：

- `CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/codex-review-uv-cache uv run --offline --no-sync pytest tests/exp/step_diag -q` → **242 passed, 1 skipped**。
- 相邻模块：`tests/cache/groot/{test_groot_staged,test_groot_interceptor,test_groot_load_guard}.py`、`tests/cache/test_interceptor.py`、`tests/libero_groot`、`tests/libero/test_episode_runner_collect.py`、`tests/conductor/test_integration.py` → 初跑 **306 passed, 5 skipped, 3 failed**；3 个失败全部是沙箱禁止建立本地 TCP socket。仅将 conductor 3 项在允许该操作的环境重跑 → **3 passed**，因此该组最终 **309 passed, 5 skipped**，没有将环境失败归为产品回归。
- 独立准入/统计探针 → **10 failed, 3 passed**：失败对应 B1（2）、B2（1）、B3（3）、B4（1）、B5（3）；正控覆盖匹配的 50 集准入、错误 continuation 步数拒收、真实工作树 9/29 臂与 YAML/index SHA 一致性。独立测试仅存于忽略的 `tests/review_tests/`，不进入共享索引；执行方应依据上述可复现条件补自身测试，不读取审查测试目录。
- `git diff --check` 通过。上述结果均为 G2 advisory / independent runs，不是声称执行方的 §6 Verify 已完成。

**裁决**：NEEDS REVISION：完成 B1–B6 并提供对应拒收/计数测试后再次申请 G2。按审查法 §5 将本计划（仅追加本段）及 §7 交付代码/测试/YAML 暂存为 Round 1 快照；不提交、不推送，独立审查测试不暂存。

**条款编号勘误（本轮）**：上述 Index Sync Rule 的准确出处是 `WORKING_AGREEMENT.md` **§4 Documentation**，不是 §5；合规缺口与 B6 的修订要求不变。

### G2 Round 1 — Executor — 2026-09-24 21:05 CDT

修订由执行方委派子代理实现、执行方逐项复核；未读 `tests/review_tests/`，未改 git 状态（保留审查方暂存基线），未触及部署树 / 在跑实验。复核结果：`CUDA_VISIBLE_DEVICES="" uv run pytest tests/exp/step_diag -q` → **283 passed, 1 skipped**（原 242）；RoboCasa 回归：`success_length` 对本地数据复现既有 `data/analysis/success_length.json` 全部 46 行 0 差异，`warm_variants`（π0.5 t=0.2、GR00T t=0.75 / 0.5，13 任务）与 Q-B `aggregate` 修订前后数值 0 差异、markdown 逐字节一致（只新增 NFE 字段）。

- **B1** — Accepted — `aggregate_arms.pair_identity` 改用 `PAIR_IDENTITY_KEYS`（增 `init_pool_sha256`、`env_id`，`cell_admission` 给每个 outcome 打上所准入的 env）；`warm_variants.cross_identity_problems` 为新跨臂门（`init_pool_mismatch` / `env_seed_mismatch`），命中的任务退出 macro、逐任务配对 n=0 并在 markdown 列出；`success_length.identity_key` 同样含 (task, init_idx, env_seed, pool, env)，`identity_mismatches` 报告不一致。RoboCasa 无池（pool=None），配对结果不变（见上回归）。计划 §4.6 的配对身份描述已同步改为 (env, 任务, 池 SHA, init_idx, env_seed)。测试：`test_libero_pairing_identity_includes_the_initial_state_pool_and_env_seed`（同池正例、异池 / 异 seed 负例）、`test_success_length_pairs_only_the_same_pool_and_reports_a_mismatch`。
- **B2** — Accepted — `warm_variants.libero_formal_problems` 按冻结设计核验：实验 id = `sdiag_libero_self`（smoke 拒）、launch 的 env 与 seed 7、每任务 init_idx 0..49 恰好各一次 × 10 任务、单一池；臂 / cell / 任务标 `formal` 或 `partial`，`macro` 只收正式臂正式配对，在跑数据另出 `macro_partial` 并在 JSON / markdown 显式标 partial；`success_length` 的 LIBERO 路径同样给出 per-arm / per-panel `status`，CLI 打印 `[PARTIAL]`。测试：4 集 cell 断言 partial；`test_libero_formal_status_needs_the_frozen_design`（smoke、他实验、49 / 4 集、9 任务、seed 8、双池、错 env）；9 臂 × 10 × 50 正例；`test_libero_smoke_arm_is_never_formal`；`test_libero_success_length_in_flight_is_partial`。
- **B3** — Accepted — `aggregate_arms.expected_self_seed` / `self_start_problems` 在 `cell_admission` 逐决策调用：自产臂要求 `self_start is True`、`self_seed` 等于由期望身份（launch 的实验 id、journal 的 attempt、行的 decision_idx、env_seed / init_idx / pool）按 `recorder.noise_seed(..., "self")` 重算的值、`self_direct_nfe == K`（π0.5 10、GR00T RC 4、LIBERO 8）；非自产臂带任一自产字段即 `self_start_on_cache_arm`；任何问题使 `equal_nfe=False`。`executed_steps` / `n_stage3_calls` 语义未变。种子重算所需字段行内齐全，无缺口；并用真实 `DiagSession.self_start_seed()` 钉住（LIBERO 含池、RoboCasa 无池）。**真数据核验**：对正在跑的 RoboCasa `selfresetfinal_t0.2`（wls server 行 + timan107 journal / per_step）按新准入运行，已完成的 4 个任务 cell 全部 `complete=True`、`equal_nfe=True`、`problems={}`、`mean_self_direct_nfe=10.0`，在跑 cell 仅 `missing_terminal`。测试：参数化负例（缺字段、`self_start=False`、`self_direct_nfe` 0 / 4、seed+1、下一决策的 seed、他池 seed、cache 臂带自产字段）、plain 臂带自产字段、RoboCasa 自产臂准入。
- **B4** — Accepted — `cell_admission` 输出 `episode_continuation_nfe`（原含义）、`episode_self_start_nfe`、`episode_total_nfe`（保留键名，改为真实总量；非自产臂与原值相同，故 RoboCasa 既有 JSON 数值不变），并给出 `self_start`、`k_self`、`mean_self_direct_nfe`；`warm_variants._task_cells` 增 `nfe_per_decision`、`episode_nfe_mean`，markdown 步数列对自产臂写成 `1.0 + 8 self` 并注明自产臂每决策 K+m；「equal-NFE panel」措辞改为「equal continuation NFE (self = K+m)」（docstring、LIBERO markdown、`ops/README.md`）。真数据：π0.5 自产臂 `episode_total_nfe` = 决策数 × 12（例 180 决策 → 2160）。测试：GR00T K=8、N=1、3 决策 → 3 / 24 / 27（cache 臂 3 / 0 / 3），π0.5 K=10、N=2 → 6 / 30 / 36；多余 stage-3 调用仍拒；缺 `self_direct_nfe` 时总量为 None。
- **B5** — Accepted — `success_length.arm_episodes` 只从「accepted、无 error、success 为布尔的 journal 终局」与「同 task_uid / attempt / driver run 且恰一条 accepted 的 `episode_summary`」取长度；`summary_problems` 核验 arm / experiment / config_sha 与 launch 一致、summary 与终局 success 一致、无 error、身份字段（init_idx、task_id、seed、pool）一致、计数为整数；manifest 问题、冲突、launch env 不符均上报；RoboCasa 与 LIBERO 共用。测试：11 个参数化拒收（未 accepted、success 冲突、summary error、他 run、他 arm、他 config、他池、缺计数、终局 error、仅旧 attempt 的 summary、两条 accepted summary）、重试混合（正确取 attempt 2、忽略被隔离的重复）、manifest 问题传递、RoboCasa 路径显式测试。
- **B6** — Accepted — `logs/README.md` 在「Diffusion Policy 减步实验」段按既有行格式新增本计划一行，该文件其他内容未改（相对暂存区仅此一行）；补 docstring：`run_libero_diag.build_parser`、`warm_variants.markdown_libero` / `markdown` / `main`、`success_length.success_length` / `main`、`ops/self13_queue.py` 全部公共函数（仅增 docstring）；AST 扫描确认变更文件中无缺 docstring 的新增公共函数。
- **C1** — Accepted — 不改代码；G2 通过后先做 §6 Verify，再以 smoke 实验 id 做真实服务冒烟与准入核验（两模型各一 LIBERO 自产臂 + 一缓存臂），之后才启动正式 LIBERO 轮。

变更文件（本轮）：`analysis/aggregate_arms.py`、`analysis/warm_variants.py`、`analysis/success_length.py`、`ops/README.md`、`ops/self13_queue.py`、`run_libero_diag.py`、`logs/README.md`、`tests/exp/step_diag/{test_libero_selfstart,test_self_start,test_aggregate}.py`；计划 §4.6 一句（配对身份）。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-09-24 21:00 CDT

Authority: Review，L2；按 Round 1 暂存索引复核执行方全部增量与逐项回复，保留 G1 owner 豁免。已读本计划及前轮完整交换、全部本轮代码/测试差异与相关调用方；沿用 Round 1 已取得的架构上下文。没有修改执行方代码。

**前轮处理**：B1、B2、B3、B4、B6 接受关闭：跨池/seed 不再配对；正式 10×50 与进度视图区分；self seed/K 证据被核验；K+N 成本拆分；索引与新增公共函数说明补齐。此前 13 项独立探针全部通过（正控按新 `macro_partial` / `formal` 接口调整，self 正控改用真实私有 seed）。B5 的 accepted/结果/error 检查已修，但仍存在下列漏口。C1 保留，真实 GPU/Verify 未运行，不新增这方面的阻断。

**G2 checklist**：计划一致性 **未通过**（长度证据仍可错绑）；测试 **未通过**（专项 283 passed / 1 skipped，独立检查合计 14 passed / 13 failed）；文档/索引 **部分通过**（B6 已关闭，但执行方将队列文件称为“仅增 docstring”，实际还改变了退出判定、重启挂接并新增 `reap_orphans`，需补准确记录与测试）；无回归 **未通过**（新退出判定把传输失败当作 cell 失败）。本轮未发现新的宪法违规，上一轮规范缺口已修复。

- [Blocking] [Concern] **R2-1 / B5 续 — 长度证据需真正拒收而非仅报告。** `success_length.arm_episodes` 遇 manifest 冲突或错误 env 后仍追加长度；不核对目录所请求的 arm 与 launch arm；同 driver run/launch 匹配后不核对该 uid 属于该 launch 的 expected 集合；LIBERO 摘要缺 pool/seed/task_id/init_idx 也被接纳。— reasoning: 4 个 launch 错绑负例与 4 个缺字段负例均复现被错误收录的数据（错 arm 甚至 `problems=[]`）。需令有问题的目录/身份不贡献长度，按实际 worker schema 验证必需身份字段，保留合法重试的正例。缺 settle 字段的原样例已被现有长度范围检查拒收，但仍应明确其类型与非负性，避免偶然依赖计数余数。

- [Blocking] [Concern] **R2-2 — 未配对长度受 full 进度截断。** `success_length.analyse` 只遍历 full 的任务；另一臂有 a、b 两任务、full 只有 a 时，其 `n_success` 少计 b，所谓 own-success macro 也只算 a。— reasoning: 独立例中本应 2 成功集、平均 6 次调用，输出 1 集、平均 4 次；本轮新增 formal/partial 状态可能将一个完整臂的这种截断统计标为 formal。需按各臂自己的任务集合算未配对统计，配对仍只取共同成功身份，并令配对结果的完整性同时依赖两个臂。

- [Blocking] [Concern] **R2-3 — 新队列退出判定把传输异常当成任务失败。** `self13_queue.cell_exit` 仅把 rc=99 视作传输异常，其他非零 rc 或不完整输出均落到 exit=1。— reasoning: rc=1/124/255 及 rc=0 空输出四例均返回 1；连接中断/超时会消耗任务的 MAX_TRIES，甚至令仍在运行的任务被标 failed。需只有明确读到 cell exit，或成功完成远端探测且确认 session 已结束时才返回终局；异常/不完整探测保持 unknown，补运行中/成功/失败/传输异常及恢复挂接测试。

**验证**：`CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/codex-review-uv-cache uv run --offline --no-sync pytest tests/exp/step_diag -q` → 283 passed, 1 skipped。独立 R1/R2 探针 → 14 passed, 13 failed，失败分布 R2-1 八项、R2-2 一项、R2-3 四项；探针仍只在忽略目录，不暂存。没有重跑未改动的 GPU/缓存路径，Round 1 相邻模块结果仍只代表其所测范围。

**裁决与 owner 当次流程覆盖**：开发者当前快照 NEEDS REVISION（R2-1–R2-3）。本轮用户明确授权“审查结束后把他的修改加入暂存，直接修改直到可以放行，自己的修改留在暂存区外”。据此先冻结此开发者交付与本轮 Review Log，再直接修复上述问题并验证；后续代码、测试和记录改动均留在暂存区外，不再更新此快照。该后续阶段是 owner 授权的直接修复，不能表述为对本人修改的独立第三方复审。无需再次请求权限；不提交、不推送、不部署。

### Owner-authorized remediation — APPROVED (G2 code scope) — 2026-09-24 21:17 CDT

本段为按 owner 当次授权完成直接修复后的代码就绪判断，不是新的独立第三方审查。Round 2 对开发者快照的 NEEDS REVISION 结论保留；本段评价的是该快照加以下未暂存修复后的工作树。

**快照隔离**：先暂存开发者交付与 Round 2 审查记录，共 42 文件；之后没有再次 `git add`。冻结的 `git diff --cached --binary` SHA256 为 `60ae6adc76d3f91d3ba3da458d09af46b9c86c29bec53a6e237f5d62f6eedd98`。本段、索引更新、源代码、ops 说明及新增测试全部留在暂存区外。其他任务的 `logs/session_handoff.md` 与 `exp/trace_dual/` 未修改。

**修复与关闭**：

- **R2-1 / B5 关闭**：`success_length.py` 遇 manifest 冲突、错误 env 或目录/launch arm 不一致时剔除该目录；摘要必须属于其 launch 声明的完整 expected identity；LIBERO 的 task/index/seed/pool 字段及非负整数 settle 计数必需，决策和环境步计数须为正整数。有效的其他身份、合法重试仍被保留。
- **R2-2 关闭**：任务列表改为已加载各臂的任务并集，不再由 full 的进度裁剪其他臂的成功数；保持完整任务集合上的未配对 macro 口径。配对统计单独标 formal/partial，只有自身与 reference 都正式完整才标 formal，CLI 同步显示。
- **R2-3 关闭**：队列用明确的远端探测完成标记、session 状态及有效计数判断终局；所有传输非零 rc、不完整输出、仍存活的 session 均保持 unknown，不消耗重试。补 shell 参数引用；停止等待时保留可恢复的 running 状态，不把 unknown 当失败去轮转日志或清理 worker。没有连接实验机器或执行真实进程清理。
- 新增共享测试 `test_success_length_admission.py`、`test_self13_queue.py`，共 37 项，覆盖错绑/缺字段、部分任务统计、formal 配对状态、真实 bash 探测逻辑（tmux 私有桩）、恢复挂接与停止等待。补齐 ops 说明，明确开发者本轮队列变动包含退出/恢复/孤儿清理逻辑，修正“仅增 docstring”的交付描述。

**最终 G2 checklist**：

| 项目 | 判断与依据 |
|---|---|
| 与计划一致 | **通过**。前轮 B1–B4/B6 的修复保留，B5 及本轮 R2-1–R2-3 已补齐；臂、预算、池/seed、服务推理语义均未另改。 |
| 测试覆盖与通过 | **本次范围通过**。最终专项 320 passed / 1 skipped，加独立探针 27 passed，总计 347 passed / 1 skipped；新失败模式都有回归测试。完整仓库 Verify 的限制另列如下，未声称已通过。 |
| 文档与索引 | **通过**。ops 行为及证据准入说明、此记录与 logs 索引同步；改动公共 API 有 docstring；后续新增记录按 owner 要求保持未暂存。 |
| 无回归 | **所测范围通过**。本地 RoboCasa 历史长度结果 46 行逐项一致，三个 panel 均无准入问题；旧专项和新增测试均通过。未进行真实 GPU/闭环验证。 |

**验证记录与限度**：

- 最终：`CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/codex-review-uv-cache uv run --offline --no-sync pytest tests/exp/step_diag tests/review_tests/test_libero_selfstart_g2_r1.py tests/review_tests/test_libero_selfstart_g2_r2.py -q --tb=short` → **347 passed, 1 skipped**（39.48s）。输出 `/tmp/openpi-g2-target-final.log`。独立探针仍未进入共享索引。
- `success_length --out /tmp/openpi-g2-success-length-after.json` 与既有 `exp/step_diag/data/analysis/success_length.json` 的 **46 行 0 差异**，每个 panel `problems=[]`。
- 默认全仓测试（排除私有审查目录）在 `tests/robocasa365/test_bench_groot_stages.py` 收集失败：被测模块没有 `SCHEDULE_ID`。相关测试和模块均与 HEAD 字节相同，单独收集亦复现。日志 `/tmp/openpi-g2-verify-20260924.log`、`/tmp/openpi-g2-bench-collection.log`。
- 排除该既有收集失败文件的补充运行，在无关 dispatch_surface 统计设计测试长时间执行后主动 SIGINT 收尾：**2954 passed, 41 skipped, 1 failed，剩余未运行**（482.08s）；失败为未改动的 JAX LoRA 测试在隐藏 GPU 时初始化 CUDA backend，显式 `JAX_PLATFORMS=cpu` 单独重跑 **1 passed**。完整记录 `/tmp/openpi-g2-verify-rest-20260924.log` 与 `/tmp/openpi-g2-lora-cpu.log`。不把此次未完成的补充运行算作全仓通过。
- `git diff --check` 通过；冻结暂存快照保持不变。未 commit、push 或部署。

**结论**：按 owner 授权，当前工作树的本次 **G2 code approved / APPROVED**，上述已知代码阻断全部关闭。全仓 §6 Verify 仍未完成，需先处理既有 benchmark 收集问题并在合适环境执行；C1 的真实 GPU parity/两模型 self+cache 冒烟仍是正式 LIBERO rollout 前的后续步骤，本次未授予 rollout 已验证通过的结论。

### G2 Round 2 — Executor — 2026-09-24 22:15 CDT

owner 当次指示（21:35 CDT）：审查方按授权直接修改到可放行，修改留在暂存区外；执行方不得全盘接收，须自行检查、修改后推进到 commit / push。执行方逐 hunk 复核了审查方的未暂存修改（`analysis/success_length.py`、`ops/self13_queue.py`、`ops/README.md`、两份新测试、本日志与索引），未读 `tests/review_tests/`。

- **R2-1** — Accepted — 审查方修复保留不改：manifest 问题 / 错 env / 目录与 launch arm 不一致时整目录不贡献长度；摘要须属于其 launch 声明的 expected 身份；LIBERO 必需 task_id / init_idx / seed / pool 与非负整数 settle 计数，计数须为正整数。执行方另在真数据上核验「整目录剔除」不会误伤：对在跑的 RoboCasa `sdiag_self13` 两台 worker 的 `rc_self13` 快照（π0.5 3 臂、GR00T 10 臂，含重试过的 cell）逐臂跑新 `arm_episodes`，13 个目录 **0 个 problems**，已完成任务均 50 条身份。
- **R2-2** — Accepted — 任务集合取已加载各臂的并集、未配对 macro 仅在该臂覆盖全部任务时给出、配对统计单独标 formal / partial 且须两臂皆 formal，均保留。
- **R2-3** — Accepted — 新 `cell_exit`（远端探测须以 `CELL_PROBE_DONE` 收尾、session 精确名 `=<name>` 已结束才判终局，传输失败 / 输出不完整一律 unknown、不耗重试）保留。执行方另对真机只读探测：timan107 / timan108 上在跑 cell 返回 None、已结束 cell 返回 0；并核对 cell 会话在写出 `SDCELL_EXIT` 后立即 `exit`（`run_rc_cell.sh`），故「session 仍在即 unknown」只会推迟一个轮询。在跑的 `sdq` 队列进程仍是旧代码，未重启（新判定只更保守，下次重启生效）。

**§6 Verify**（owner 当次裁定：全量不碰 GPU 测试，按 weilandserver 核数并行）：`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 uv run --no-sync --with pytest-xdist --with pytest==9.0.2 pytest -n 24 --dist loadfile`（忽略两份既有收集错误的审查探针与 HEAD 上即收集失败的 `tests/robocasa365/test_bench_groot_stages.py`）→ **6553 passed, 21 failed, 92 skipped，墙钟 197 s**。21 项全部在本变更范围外：
- 非审查测试 6 项 = 既有基线：`test_ws2_evidence_runner::test_hit_meta_rows_identical_across_runners` ×2 与 `test_prebuilt_matrix_backend` ×2 在干净 HEAD 导出树上同样失败；`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2 在当前工作树单独跑 16/16 通过，仅全量时因同一 worker 先装入 `gr00t` 桩模块而失败（测试隔离，既有）。
- `tests/review_tests/` 下他线旧探针 15 项（rl_router、keybuilder、x0、warmstart_w1、ws2、groot_robocasa、cache_size、n1_serverside、tracer_phase7），报错均指向各自线的模块；仓内 `exp/step_diag` 之外无任何代码 import `exp.step_diag`。执行方不得阅读其源码，失败 id 交 Review Authority。
- 本线：`tests/exp/step_diag` + `tests/review_tests/test_libero_selfstart_g2_r{1,2}.py` → **357 passed, 1 skipped**（审查方 347 + 执行方新增 LIBERO 队列测试 10）。

**owner 同时裁定**：删除 `tests/dispatch_surface/test_rev2_confirmation.py`（43 项；其模块级 fixture 以冻结的 10,000 次 bootstrap 跑正式 outcome design，单测 15 分钟以上、纯 CPU；所测 Rev 2 confirmation 链在他线零引用，计划自 2026-08-29 停在待 G2），单独提交。

**运行阶段增补（计划 §5，L1 ops，非 G2 范围）**：`ops/run_lib_cell.sh`（单任务 cell 启动器）、`ops/libero_queue.py`（排在 RoboCasa 队列之后、与其共用主机预算）与 `tests/exp/step_diag/test_libero_queue.py`（10 项离线测试）；LIBERO 使用独立部署树，不改在跑的 RoboCasa 部署树。GPU 冒烟（C1）在正式轮之前做。
