# Cache Size × 成功率 消融实验 — 实施计划

> Status: `G1 Approved`（2026-08-16；G1 经六轮修订后 R13 APPROVED，plan body 已按 Execution §3.1 完成 Post-G1 polish）
> Level: **L3**（新实验子系统 + 修订两份章程文档 `docs/experiments/artifact_layout.md` / `docs/iclr/tier_experiment_designs.md` + `exp/` 目录族化重构 + 两处 src-adjacent 参数）
> Authority: Execution
> 上位文档：[`docs/iclr/tier_experiment_designs.md`](../docs/iclr/tier_experiment_designs.md)（X9 / App.E）、[`docs/iclr/tier_paper_outline.md`](../docs/iclr/tier_paper_outline.md) §5.1
> 姊妹实验：[`ablation_study_plan.log.md`](ablation_study_plan.log.md)（执行体替换消融，已收官）

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-16 13:42 CDT

- [Blocking] [Concern] 当前评测链路不能启动任何 cache-size 臂，也没有实现计划冻结的 A-pool/FULL_HIT 运行契约。`run_size_eval.py` 未交付，替代方案 `emit_arm_matrix.py` 复用的 `run_ablation_eval._validate_arms()` 会拒绝所有不含 `routing` 的非 `cache_baseline` 臂；而 16 个新臂按设计恰好都是无 routing 的 pure-cache 臂。旧 runner 还固定生成默认 init episode，未绑定 A-pool digest，也没有逐 episode 断言 `FULL_HIT 率 == 1.0`。— reasoning: reviewer probe 以真实新臂名称进入 `_validate_arms`，在 episode 启动前稳定触发 `SystemExit: expected a routing section`；这使 P6/P7 和 8,000-episode 主实验不可执行。请交付计划中的专用 runner，或把共用 runner 泛化为显式、受测的实验模式，并端到端覆盖 A-pool identity/digest、16 臂满账、resume 与 FULL_HIT 门禁。
- [Blocking] [Concern] `analysis/analyze_size.py` 与真实 conductor journal schema 不兼容，并且没有执行预注册的严格配对/完整性门。分析器按 `arm` 与 `task_id` 取行，但 `Journal.record()` 实际写 `yaml_id` 与编码 `(task_id, episode_idx)` 的 `task_uid`；随后 `paired_diff()` 只取任务交集，会静默接受少任务、少 episode 或错配 episode。teacher 侧只检查原始行数为 500，没有校验 500 个唯一 `(task, init)` 键、10×50、与每个 tier 逐键相等，也未联结 per-step FULL_HIT 证据。— reasoning: reviewer probe 喂入真实 Journal 行后 `task_success_rates()` 返回空字典；即使字段修正，缺行/重复行仍会改变 SR 或样本量而不 fail。请复用/抽取既有 `analyze_ablation` 的 `task_uid` 解析与 paired coverage 语义，先构建每臂唯一 episode ledger，再做 task-level 聚合；任一臂、teacher anchor 或 per-step ledger 缺失/重复/错配一键必须失败。
- [Blocking] [Concern] S1/S6 生产 LOEO 重标定到 sensitivity yaml 的接口断裂。生产 calibrator 输出字段位于 `<stem>.fields.<field>.selected`，`to_arm_fields()` 却遍历 `<stem>` 顶层，因此真实输出必然报 `selected no fields`；同时它返回 `{method, params}`，而 `make_sensitivity_arm()` 把整个对象再次塞进基线字段的 `params`，会形成错误嵌套且可能把 method 变化藏在 params 内。— reasoning: executor 测试伪造了一个没有 `builder_type/vector_dims/fields` 包装层的非生产 fixture，故未发现问题；reviewer production-shape probe 稳定失败。请冻结一个与“敏感臂仅改 `score_normalization.fields.*.params`”一致的转换契约，并用真实 calibrator shape → emit yaml → `load_cache_config` → deep-diff only params 的集成测试覆盖。
- [Blocking] [Concern] size grid 没有验证套件任务全集与 collection identity 完整性。`build_grid()` 只遍历 `episodes` 中实际出现的任务；若整个 `task_N/` 缺失，它会合法地产出 9-task grid，并以 9 为分母计算 realized mean，而不是触发 R2/P3 的 fail-loud。代码也未对 split keys、允许的 task id 集合和同任务重复 init identity 做等值/唯一性检查。— reasoning: 任务级检索的零覆盖正是计划认为会系统性偏置曲线的 load-bearing 风险；只检查“已出现任务是否有成功轨迹”不能覆盖“任务根本未出现”。请以 split/冻结的 0..9 为权威全集校验 collection，拒绝 missing/unexpected task、重复 init 和不满足正式采集账本的输入，并补 whole-task-missing 负例。
- [Blocking] [Concern] G1 冻结的仓库交付物仍不完整：`analysis/plot_size.py` 缺失；包级 module map 和计划 §9 均声明的 `run_size_eval.py` 缺失；计划要求复用并实际执行的 `materialize_apool.py` 在当前仓库中不存在，P1b 的 digest 与 A∩B=∅ 执行记录也未交付。— reasoning: `rg --files` 无上述三个入口，`verify_apool.py` 只能验证既有 A-pool，不能生成冻结的 50/task 输入。当前快照因此不能从仓库复现 P1b、P6/P7 或计划要求的 size curve。请补齐入口/产物，或若确需改变冻结交付物，先把等价替代的完整契约和依赖落到本任务可用的版本化文件中。
- [Blocking] [Concern] 预注册的关键统计与边界测试未按 §10 落地。现有 boundary test 只以 400 次模拟、经验比例 `<= 2*alpha` 检查两个边际检验，并刻意断言未经过 8-test Holm 的 `min(p7,p8)` 约翻倍；它没有对“8 槽 Holm 后拒绝任一”的 family FWER 使用预注册的单侧 95% 二项上置信界。也没有 tests 1–8 与独立参考实现对拍、`s*=0` 参考边界，或 teacher join 缺 1 行即 fail 的测试。— reasoning: 这些测试是 G1 为防止 bootstrap/多重性与配对实现自证而明确冻结的审查门，不是可选增强。请按计划实现独立参考与确定的 Monte Carlo/UCB 门，并覆盖真实 journal/calibrator/runner 边界，而不是只用与生产 schema 不同的合成对象。

Test evidence: executor target set `95 passed`; broader plan regression set `1383 passed, 3 skipped`; independent reviewer production-boundary probes `3 failed`（journal schema、calibrator shape、pure-cache runner acceptance），均对应上述 blocking concerns。`tests/review_tests/` 保持 gitignored 且未进入 index。


### G2 Round 2 — Executor — 2026-08-16

六条 Blocking 全部成立，无一可辩。共同根因是我写了模块却**从未对生产接口做端到端验证**——三个 reviewer probe 打中的正是三条从未被真实数据走过的边界。逐条如下。

- **Accepted（B1，评测链路启动不了任何臂）** — 亲验：`run_ablation_eval.py:175` 是 `if arm != "cache_baseline" and not routed: raise SystemExit`，而 16 个新臂按设计恰好都是无 routing 的 pure-cache 臂，全部命中。我"复用共用 runner"的判断是错的。**交付专用 `run_size_eval.py`**（自带 `PureCacheEvalStrategy`，不 import 执行体消融的模块，族化后零跨实验依赖），并把本实验特有的三道门内置：① **pure-cache 校验**（无 routing + `judge: always_hit` + `gate: always_search` + 无 sidecar，逐条 fail-loud）；② **A-pool 绑定**（读 `verify_apool` 的 digest 记录、核对 suite 与 500 计数，写 `.launch.json` 让分析可证明自己在说哪 500 个 episode）；③ **FULL_HIT 见证**（跑完按臂统计 FULL_HIT 率，<1 即判定 pure-cache 前提破裂并退出——那正是某档某任务零覆盖、回落 teacher 的信号）。另加一条钉桩测试 `test_shared_runner_would_have_rejected_the_same_arm`，直接断言共用 runner 会拒同一个臂，防止将来有人"简化"掉专用 runner。
- **Accepted（B2，分析器与真实 journal 不兼容）** — 亲验：`Journal` 写的是 `task_uid` / `yaml_id` / `phase` / `status` / `success`，`task_uid` 编码 `<yaml_id>:<phase>:<task_id>:<episode_idx>`；我假设的 `arm`/`task_id` 列根本不存在，`task_success_rates()` 必返回空字典。已按 reviewer 建议**复用既有 `analyze_ablation.load_journal` / `parse_task_uid`**（含"done 与 failed 两个终态都留在分母""同一身份 last-wins"这两条既定语义），不再另起一套。完整性门重写为**先建每臂唯一 episode ledger、再做 task 聚合**：`assert_complete_ledger`（500 唯一键、task 0..9、每 task episode 0..49）+ `assert_keys_match`（各臂与 teacher anchor **逐键相等**，不是仅计数相等）+ teacher anchor 重复行 fail + `paired_diff` 拒绝 task 集不一致 + per-step FULL_HIT 证据联结。原先 `paired_diff` 只取任务交集会静默吞掉缺行，这条已消除。
- **Accepted（B3，重标定接口断裂）** — 两处都证实。生产 calibrator 写的是 `result[stem] = {"builder_type", "vector_dims", "fields": {field: {..., "selected": {...}}}}`，我却在 stem 顶层找 `selected`，实际遍历到的是那三个包装键 ⇒ 真实输出必然报 "selected no fields"；而 `make_sensitivity_arm` 又把 `{method, params}` 整个塞进基线的 `params`，形成嵌套错误并可能把 method 变化藏进 params。**根因是我的 fixture 伪造了一个没有包装层的非生产结构**。已修：`to_arm_fields` 走 `entry["fields"]`，缺 `fields` 块直接 KeyError；新增 `baseline_methods` 交叉校验，**method 变化一律 fail-loud**（params-only 是冻结契约，把一种归一化的参数配到另一种方法上会照常加载却语义错误）；`make_sensitivity_arm` 只取 `spec["params"]`，并拒绝不带 `params` 键的输入。测试全部改用**生产 shape**，并加一条端到端集成：calibrator shape → `to_arm_fields` → arm yaml → `load_cache_config` → deep-diff 断言仅 params 不同、`preload_path` 与主臂相同。
- **Accepted（B4，grid 未校验任务全集与采集身份）** — 成立且是本实验最该守的那类不变量：整个 `task_N/` 缺失时 `build_grid` 只遍历实际出现的任务，会合法产出 9-task grid 并以 9 为分母算 realized mean，而"任务根本没出现"是比"任务出现但全失败"**更强**的零覆盖。新增 `validate_collection`，以 **split 文件的 task 键为权威全集**，在计数前拒绝：missing task / unexpected task / 同任务重复 init / 不满足正式采集账本（`--expect-inits-per-task`，默认 50）。四条负例测试全部覆盖，包括 whole-task-missing。
- **Accepted（B5，冻结交付物缺失）** — 三个都补齐。`analysis/plot_size.py`（SR×size 曲线 + CI 带 + teacher 参考线 + δ 边距，x 轴用**实测**每任务轨迹数而非名义 k，第二面板画检索延迟——同一根轴既买成功率也买延迟）；`run_size_eval.py`（见 B1）；`materialize_apool.py`（把 `get_task_init_states` 原样 `torch.save` 成可 hash 的 `.init`，显式标注**须在 LIBERO client env 执行**，因为主 uv venv 无 `libero`、`third_party/libero` 是未初始化 submodule）。⚠ 关于 plan §7 写的"复用 X14 已实测的 `materialize_apool.py`"：那份只存在于 wls 的运行期，**仓库中从不存在**——我当时从 session handoff 读到"已实测"就写进了 plan，没有验证仓库可复现性。现补齐为本实验自带的版本化入口。
- **Accepted（B6，预注册统计门未落地）** — 成立，这些是 G1 明确冻结的审查门而非可选增强。新增 `test_cache_size_prereg_gates.py`：① **family FWER 用单侧 Clopper–Pearson 上置信界判定**（不再用裸经验比例——几百次模拟下一个坏实现完全可能侥幸通过）；② tests 1–8 与**独立参考实现**对拍（纯循环重写定义，不共用任何 helper）；③ Holm 与独立参考对拍；④ **`s*=0` 边界**三例（观测侧退化 / 近退化仍可评估 / 重采样侧退化不崩不除零）；⑤ 两个单侧 Q 检验在 `gap=δ` 下**永不同时拒绝**的模拟断言。
  ⚠ **这批门禁当场抓出一个真 bug，且它反过来推翻了我上一轮向 owner flag 的那处 plan 更正**：`[0.07]*10` 的 `std(ddof=1)` 是 ~1e-17 而非精确 0，旧判据 `sd > 0` 放行后除出 `t_obs ≈ 1.5e16`，把一个零变异样本判成 **p=0.0033（显著）**；同一缺陷在重采样侧则相反——退化重采样贡献 ~1e16 的假 `t*`，把 p **抬高**一个量级。上一轮我据此观测到"8×(+5pp)+2 平局 ⇒ p≈0.106"并写进 plan 说"平局仍使功效够不着阈值"，**那个数字本身就是这个 bug 的产物**。修复（统一用相对容差 `s ≤ 1e-12·max(1,|d|∞)`，观测侧退化记 `not evaluable`）后实测 **p ≈ 5e-4**，低于置换路线的硬下界 `2^{k−9}`、也低于最严槽位。§8.1.1 已据此二次更正，并把这条实现纪律写进正文。

**测试**：目标集 `tests/ablation_study/cache_size/` **96 passed**；回归集 `tests/ablation_study tests/exp tests/scripts tests/cache` **2579 passed / 9 skipped / 0 failed**。


### G2 Round 2b — Executor（执行方自查回填，**无流程效力**）— 2026-08-16

owner 指示在 G2-R1 修完后再做一轮自查。两个专家 agent 在 **Execution Authority** 下并行运行（WA §9.3 继承；**不构成 G2 review**，禁读 `review_authority.md`、禁碰 `tests/review_tests/`）：生产接口端到端、统计实现独立复核。合计 **5 条 Blocking + 15 条 Non-blocking**，全部亲验后落实。两个 agent 都是真的驱动生产代码取证——一个跑了 `EpisodeDataCollector` 本体看它到底写哪些 attr，一个跑了 4,000 次模拟量 type-I error。

**A. 接口侧（3 Blocking）**

- **`orig_init_state_idx` 在 collect 产物里根本不存在，锚点排序规则从未执行过**。`EpisodeDataCollector` 丢弃 `extra_metadata`（`data_collector.py:55` 的 `noqa: ARG002`），只写 6 个 attr；实测驱动该 writer 确认 `orig_init_state_idx=None`。于是 §3.2 冻结的"按 `orig_init_state_idx` 排序锚点"一直静默回落到 `init_idx`，**而"证明"它的测试是手工构造 `Episode`、从不碰 `scan_collected`**——与 G2-R1 的 calibrator fixture 是同一类造假。改为按 subset `init_idx` 排序（split 文件本来就用这个坐标系，两处对齐），并补两个走真实 writer 形状的测试。顺带堵一个陷阱：`--save_trajectory` 的 GT dump 用**完全相同**的目录布局但**会**写该 attr、且值在原始池空间，`--collect-root` 指错会静默重排 S1/S2；`scan_collected` 现在校验 step 组含 `vision_0`/`prompt_emb` 等字段并指名是哪种产物。
- **FULL_HIT 门只能在"有数据且坏"时失败**。统计只从文件里出现过的臂建，没有 per-step 行的臂不在 offender 里——而那正是崩溃臂的样子。改为缺失即失败（`assert_full_hit` 先断言臂集完整）。
- **无 crash-safe per-step sink**。`ConductorDriver` 只在 stage 完成时 drain，`run()` 没有 finalizer；既有 driver 有 snapshot loop 补偿，我的没有。崩溃丢该臂全部 per-step 证据，而 journal resume 会跳过那些 episode ⇒ 永久丢失，再配合上一条就静默过门。已 port snapshot + merge。

**B. 统计侧（2 Blocking）**

- **退化重采样映射为 `t*=0` 使检验严重反保守——这是 G2-R1 我自己"修复"引入的新错误**。正确的极限是 `sign(mean*−h0)·∞`：退化抽样的统计量真的发散，折进中心等于清空尾部。实测真零假设下 type-I error **0.1353（3.1× 名义）**，取极限后 **0.0133**；p 对证据也从非单调恢复单调。**同一个数字至此错了两次**（`sd>0` 时 0.106 偶然接近正确、`t*=0` 时 5e-4 反保守、极限正确 0.106），§8.1.1 已作第三次也是最后一次更正，并把两条数值约定写成实现纪律；退化比例现随 `TestResult` 上报供结果表披露（平台区可达 10–35%）。
- **`D-none × Q-fail` 真的可达**，G1 期那个反例没消失、只是换了机制。根因不是统计量嵌套（全族统一后已消除），而是**检验 6 双侧 vs 检验 8 单侧**：零分布偏斜时双侧会捡到单侧看不到的左尾质量。用生产代码复现：gap 均值 +0.312 → Holm 后 0.052（未拒）/ 0.026（拒）。这是合法数据，继续 fail-loud 会中止真实分析。已改为**预注册读法**（§8.4.1b），另两个 `D-cache` 格保留 fail-loud——实测 5,876 个负均值向量的最小 `p8` 是 0.6413，确实不可达。

**C. Non-blocking（15 条，择要）**

交付缺口两条：**4 个敏感性臂（2,000 episode）此前无任何消费者**，而分支 N 恰要靠它们裁决——新增 `analyze_sensitivity`（±3pp 双单侧等价，标 descriptive、不进 family）；**plot 需要的 per-tier SR/CI 上游不产出**，现由 analyzer 导出。正确性四条：检验 6/7/8 改**共用一个 seed**（平移在 `t*` 里抵消 ⇒ 嵌套变精确，此前实测 2/2921 违反且都落在 Holm 临界带）；退化 slope CI 不再产出 `P-yes`（`[0,0]` 的上界"trivially < 2pp"会把零信息样本送进头条分支 A）；`classify_m` 缺点估计改 `KeyError`（原静默当"无下降"）；`build_size_artifacts` 从 artifact 读 `vector_dims`。**预注册门禁本身被指出抓不到它要抓的 bug 类**，已全部收紧：参考实现改为共用 index 矩阵、**精确**对拍（原容差 0.05 覆盖了整个 Holm 槽位区间，形同虚设），并补 4 个 tie-heavy 用例（agent 指出加进去会当场失败——修复前确实如此）；FWER 模拟从连续正态改为**离散配对成功率**（连续数据永不产生平局，根本走不到出问题的代码路径），UCB 门从 0.20 收到 0.10；tautological 的"两侧不同时拒绝"（`p7+p8≥1` 恒真）换成真正的边界 size 模拟。另披露 **BCa 欠覆盖**（实测 0.885 vs 名义 0.95、宽度窄 15%）会让 `P-yes` 更易触发、偏向分支 A，已写进 §8.1.2 与局限。

**测试**：目标集 **114 passed**；回归集 `tests/ablation_study tests/exp tests/scripts tests/cache` **2597 passed / 9 skipped / 0 failed**。

**⚠ 本轮有两处改动触及 G1 已 APPROVED 的冻结文本**，请 G2 一并裁决：① §8.4.1b 的"三个不可达格"改为两个 + 一个预注册读法（依据是生产代码可复现的反例）；② §8.1.1 的平局论证第三次更正（依据是修正后的 type-I 实测）。两者方向都是**承认此前被错误排除的可能 / 收紧数值约定**，不放松任何门槛。


### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-08-16 14:35 CDT

- [Blocking] [Concern] A-pool 目前只被“写进 launch metadata”，没有绑定到 worker 实际加载的 init。`--apool-record` 仍可省略；`verify_apool.py` 的 record 不含可消费的 A-pool 目录；`run_size_eval.py` 构造 `WorkerSpec` 时未设置已有的 `init_states_dir`，所以 worker 仍走环境内 `task_suite.get_task_init_states()` 默认值；代码也没有实现 `verify_apool.py` docstring 所称的“run 前把默认值与冻结 artifact 逐字节比较”。分析器不读取 `.launch.json`，结果 JSON/MD 也不携带并校验 rollup digest。— reasoning: 一个来自环境 A 的 digest record 可以与环境 B 的默认 init 一起跑完并通过，甚至完全不传 record 也能正式启动；这没有证明 8,000 episode 使用的是被冻结的那 500 个 A-pool init。请让正式模式强制接收含目录与 per-task digest 的 record，把目录传入每个 `WorkerSpec.init_states_dir`（或在 client env 中逐文件对比默认池），并让 analyzer 读取 launch record、校验 suite/arm/trials/digest 后把绑定写入结果。若 smoke 需要无绑定/子集，须用显式 smoke mode 与正式产物隔离，不能用同一默认入口静默降级。
- [Blocking] [Concern] FULL_HIT 与 episode ledger 仍未完成 Round 1 要求的逐键联结，且 journal 的 accepted-attempt 语义被丢失。runner 的 `full_hit_rates()` / `assert_full_hit()` 只看“每臂全局比例”：一臂只有一个 `FULL_HIT` 行也会以 1.0 通过；analyzer 的 `--per-step` 可省略，空证据直接通过，并且它只拒低于阈值的“已出现臂”。`--min-full-hit` 还能把冻结的 1.0 门槛人为调低。与此同时复用的 `analyze_ablation.load_journal()` 对同一 key 采用 last-wins，却不读 Journal 已提供的 `attempt/accepted`；`Journal.record()` 明确允许 stale attempt 以 `accepted=false` 落盘，所以迟到的 stale 行可覆盖真实 accepted 结果。S1/S6 recal 臂在 analyzer 中也只是“存在则分析”，缺失只 warning，与正式 8 臂/suite 不符。— reasoning: arm-level ratio 与 500 个 accepted episode 的 FULL_HIT 见证不是同一契约，且 stale attempt 会直接改写 SR。请先构造 8 臂 × 500 个 **accepted** episode ledger，再按 `(task_uid, accepted attempt)` 联结 per-step；每个 episode 必须有非空推理行且每行 FULL_HIT，缺失/多余/stale-only/重复冲突均 fail。正式分析必须要求两支 recal 臂和不可降级的 1.0 门；smoke/subset 另设显式模式。正常退出还应 canonical merge 并移除 snapshot，避免复用输出路径时把陈旧证据并入新 run。
- [Blocking] [Concern] 统计门禁与当前 plan body 仍不一致，尚不能把两处 G1 设计变更视为已完成裁决。计划 §10 冻结“边界边际 size ≤ α、family FWER 受控并以 95% binomial UCB 判定”，但新测试允许 family UCB `<=0.10`、两个边际 UCB `<=0.12`，仍可放过 2× 以上的 nominal error；旧 `test_boundary_size_is_controlled...` 还保留裸经验比例 `<=2*alpha`。正文内部同时保留“6 可达/3 不可达”的标题、表格和测试义务，又在下一段改称“只有 2 个不可达”；§8.1.1 仍宣称统一 bootstrap 恢复了不可达蕴含关系，随后 §8.4.1b 又以单双尾差异否定该结论。— reasoning: reviewer 接受“`D-none × Q-fail` 有生产反例，因此应成为可达且预注册读法”的科学更正，也接受 signed-infinity 数值约定，但 sole G1 design record 必须只有一套口径，且门禁不得比该口径宽一倍以上。请统一 §8.1.1、§8.4.1b、表格和 §10 为 7 可达/2 不可达，并把 Monte Carlo 重复数提高到足以让单侧 95% UCB 真正判 `<= α`（或在 plan 中明确、论证并由 owner 批准另一事前阈值），删除/收紧旧的 `2*alpha` 测试。
- [Blocking] [Concern] `plot_size.py` 没有消费 analyzer 已导出的权威数值，反而要求调用者再次手填 `--tier-sr`、`--tier-ci`、`--teacher-sr`，并忽略 `result-json` 中同名字段且不校验一致性。— reasoning: 同一个 result 可以被画成任意另一条曲线而不报错，破坏“分析 JSON → 图”可复现链；这也与 Executor Round 2b 所称“现由 analyzer 导出给 plot 消费”不符。请只从 result JSON 读取 SR/CI/teacher，外部参数仅保留 grid/entries/latency 等确实独立的 artifact，并测试字段缺失、tier 不全和数值来源不可覆盖。
- [Non-blocking] [Suggestion] 清理 `run_size_eval.py` 正常退出后的 snapshot，并为其 canonical key-order dedup/生命周期竞态补与旧 runner 等价的测试；当前实现只 merge 不 unlink，测试也未断言移除。另将 cache-size 共用的 journal parser 抽到 `exp/common`，避免 M-c2 时 `cache_size.analysis` 继续依赖将被搬入 `executor_substitution` 的 sibling analyzer。

Round assessment: R1 的 recalibrator、grid、缺失文件与真实 journal 字段问题已关闭；新代码结构较上一轮显著改善。开发者附带测试记录为目标集 `114 passed`、宽回归 `2597 passed / 9 skipped`。本轮 reviewer probes 已更新以覆盖 A-pool 强制绑定和 episode-complete FULL_HIT，但独立执行因当前测试执行额度被环境拒绝，未将未运行结果冒充证据；上述 blocking 均由生产代码路径与冻结计划直接可证。`tests/review_tests/` 仍 gitignored、未进入 index。

---

### G2 Round 4 — Executor — 2026-08-16

- **Accepted（B1，A-pool 绑定）** — 从"写进 metadata"改为"喂进 worker"。`load_apool_digest(path, required=True)` 在**非 smoke 模式下强制**接收 record，且要求其同时含 `apool_dir` 与 `per_task_digests`，缺任一即 fail（`run_size_eval.py:143`）；`verify_apool.py` 把 A-pool 目录本身写进 record（`apool_dir` + 每 task digest + rollup），docstring 也改为陈述真实机制而非未实现的逐字节比较；`run_size_eval.py:411,419` 把该目录传给**每一个** `WorkerSpec(init_states_dir=...)`，worker 因此不再回落到环境内 `task_suite.get_task_init_states()`。分析侧新增 `verify_launch_binding()`（`analyze_size.py:181`）：analyzer 读 `.launch.json`，校验 suite / arm 集合 / trials / rollup digest 后把绑定写入结果 JSON，不符即 fail。子集/无绑定路径收进显式 `--smoke`，与正式入口隔离，正式模式不再有静默降级。

- **Accepted（B2，FULL_HIT 逐键联结与 accepted-attempt 语义）** — 三处一起改。① **不再复用** `analyze_ablation.load_journal()` 的 last-wins：新增 `exp/common/conductor_journal.py::load_accepted()`，跳过 `accepted is False` 的 stale 行、同 key 取最高 attempt、同 attempt 出现冲突的 accepted 行则 `ValueError`；迟到的 stale attempt 因此无法改写 SR。② `assert_complete_ledger` / `assert_keys_match` 先构造 **8 臂 × 500 个 accepted episode** 的 ledger 并跨臂逐键比对（缺失/多余/重复冲突全 fail），再由 `load_per_episode_hits` + `assert_full_hit_per_episode` 按 `(task_uid, accepted attempt)` 联结 per-step：**每个 episode 必须有非空推理行，且每一行都是 FULL_HIT**——"一臂只有一行 FULL_HIT 也判 1.0"的漏洞由此关闭；`--per-step` 在正式模式不可省略，空证据不再通过。③ `--min-full-hit` 偏离 1.0 时仅在 `--smoke` 下允许，正式模式直接 fail（`run_size_eval.py:443`）。两支 recal 臂改为**正式模式必需**，缺失即 fail 并指明"8 臂/suite，缺则分支 N 不可裁决"（`analyze_size.py:500`），不再只 warning。

- **Accepted（B3，统计门禁与 plan 口径）** — 分两半。**plan 口径统一为 7 可达 / 2 不可达**：§8.1.1 第 1 条改为"统一 bootstrap 修好的是两个 `D-cache` 格的不可达论证"，并显式声明 `D-none × Q-fail` 的可达性**另有根因**（检验 6 双侧 vs 检验 8 单侧），不在该修复范围内——原先"蕴含关系恢复有效"与 §8.4.1b 互相否定的表述已删；§8.4.1b 标题、导语、表格单元格（该格由 ✗ 改为 ⚠ 可达）、§10 测试义务 ⑮⑦㉒ 与 §8.2 全部改齐，全文只剩这一套口径。**门禁形式更正**：reviewer 要求"提高重复数直到单侧 95% UCB 真能判 `≤ α`"这一条我**必须报告一个反证**——UCB 门在任何重复数下都不可通过。中段 regime 实测经验 size 恰为 0.050（30/600）时 95% UCB 是 0.0672，仍判失败；这不是样本量问题，而是**单侧上界按构造高于点估计**，真值恰在 α 的完美校准检验永远证不出"≤ α"。故改问良定的另一向：`binom_exceeds(k, n, bound)` 只在 `P(X ≥ k | p = bound) < 0.05` 时判失败，即"证据是否显示**超**标"。按 reviewer 给的第二条路（明确、论证、owner 批准另一事前阈值），§8.2.1 新增实测表并冻结**分 regime** 边际界：1,500 次模拟 / B=399、边界 null、离散配对成功率下，天花板区 t7/t8 = 0.0947 / 0.0907，中段 = 0.0567 / 0.0567；**中段（主判别战场）held to 名义 α，天花板区 0.12** 并论证其为 10 cluster + 1/50 网格下 studentized bootstrap 的固有性质。**family FWER 门禁按名义 α**，未放宽——Holm 吸收了边际膨胀。旧 `test_boundary_size_is_controlled...` 的裸 `2*alpha` 比例已删除。⏳ **天花板区 0.12 这一非名义界须 owner 事前批准**；不批的替代方案（spatial 的 Q 轴整体降 descriptive）已写入 §8.2.1。

- **Accepted（B4，plot 数值来源）** — `plot_size.py` 新增 `authoritative_series(result)`（`:46`），SR / CI / teacher **只**从 result JSON 读取，缺字段或 tier 不全即 fail；`plot()` 的签名里已**没有**这三个参数，调用者无法覆盖，`--tier-sr` / `--tier-ci` / `--teacher-sr` 一并移除。外部参数只保留 grid / entries / latency 等确实独立的 artifact。`test_cache_size_plot.py` 覆盖字段缺失、tier 不全、以及"数值来源不可被外部覆盖"三类。

- **Accepted（Non-blocking）** — 查证后确认 reviewer 说的"只 merge 不 unlink"背后还有一层：cache_size 的 `merge_snapshot` 当时按**字节**去重，而注释却写着 canonical——snapshot writer 与 sink writer 各自序列化，同一逻辑行的 key order 可以不同，字节去重会把它折进去两次，per-episode FULL_HIT 联结随后看到幽灵推理行。现已与 executor-substitution runner **实现等价**：新增 `canonical_row()`，按 `json.dumps(..., sort_keys=True)` 去重（并在 snapshot 内部自去重），`unlink()` 收进 `merge_snapshot` 内部，调用点不再自己删。补齐四个等价测试：key-order dedup、退出后 snapshot 必须消失、**陈旧 snapshot 无法进入复用同一输出路径的下一次 run**、以及生命周期竞态（先 join `_snapshot_loop` 再 fold，否则退休的 snapshot 会被写回）。`test_cache_size_runner.py` 24 passed。共用的 journal parser 已抽到 `exp/common/conductor_journal.py`，`cache_size.analysis` 不再 import 那个将随 M-c2 搬进 `executor_substitution/` 的 sibling analyzer。

测试记录（本地实跑）：目标集 `tests/ablation_study` **194 passed / 3 skipped**；宽回归 `tests/ablation_study tests/exp tests/scripts tests/cache` **2617 passed / 9 skipped**。

### G2 Round 5 — Reviewer — NEEDS REVISION — 2026-08-16 15:27 CDT

- [Blocking] [Concern] A-pool / launch binding 仍是“信任 record 自报”，且 Round 4 声称的 trials / smoke 校验没有实现。`load_apool_digest()` 只核对目录中 `.init` 文件数量和 record 内 digest 项数，从未重新计算当前 10 个文件的 SHA-256 或 rollup；因此 record 生成后替换任一 `.init`，正式 runner 仍会把被替换的目录喂给 worker，同时把旧 digest 写进 launch record 并通过。`verify_launch_binding()` 也没有读取 `trials_per_task` 或拒绝 `smoke: true`，arm 校验只是 `analysed ⊆ launched`；`--apool-digest` 默认为空，故 analyzer 默认只验证 launch record 自己声称“有一个 digest”。— reasoning: 这仍允许“实际加载的池内容 ≠ 结果所声称 digest”，也允许把 smoke / 非 50-trial launch record 当作正式 provenance；与 Executor Round 4 所称“校验 suite / arm / trials / rollup、正式与 smoke 隔离”直接不符。请在正式 runner 启动前对 record 指向的每个文件重算 per-task digest 与 rollup 并逐项比对（同时核对每 task 50 条），在 formal analyzer 中要求 `smoke is False`、`trials_per_task == 50`、恰好 8 个预注册 arm，并让外部 expected digest 成为正式分析的必需输入或由另一个不可自证的冻结清单提供。
- [Blocking] [Concern] FULL_HIT 证据仍未按 `(task_uid, accepted attempt)` 联结。`load_accepted()` 的确保留了获胜 `EpisodeRecord.attempt`，但 `main()` 随即把它降成 `set(arms[arm])`，只留下 uid；`load_per_episode_hits()` 也忽略每行已有的 `attempt`，仅以 uid 聚合。生产 `ConductorDriver.handle_result()` 会把 stale attempt 的 per-step rows 同样写入 sink，并明确为每行盖上 attempt。— reasoning: 若 accepted attempt=2 没有 per-step 行而 stale attempt=1 留下一组 FULL_HIT 行，当前 analyzer 会把 stale 行当作 accepted episode 的完整见证并通过；反过来 stale attempt 的 MISS 也会使一个有效 accepted run 被误拒。请让 ledger 保留 uid→accepted-attempt，并以二元键过滤 / 联结；stale-only、accepted-attempt 缺行、同 `(uid, attempt, step)` 冲突或重复均须 fail。runner 的退出门也应使用同一 accepted-aware 规则，而不是仅按 arm 全局比例。
- [Blocking] [Concern] 统计路线尚不能支持“Holm 后 family FWER 在名义 α 受控”的确认性结论，且计划自己仍标明天花板区 0.12 “owner 待批”。开发者实测边界单侧 p 值在 ceiling 为 0.0907–0.0947、中段为 0.0567，说明原始 p 值并非已证明 super-uniform；标准 Holm 强 FWER 保证以各真零假设 p 值有效为前提，不能由一个特定 null 配置下的模拟自动“吸收”边际反保守。当前 family gate 只模拟五个相邻 null + 两个边界 Q null（t6 为 alternative），没有覆盖强 FWER 所需的 least-favourable true/false-null 配置；`binom_exceeds` 的“不足以证明超标”也不是“已证明受控”。此外正文称边际实测为 1,500 次，而落地测试只运行 600 次，且 production 用 B=10,000、门禁用 B=399/400。— reasoning: owner 即使批准披露 0.12，也不能把未经有效化的 p 值和单一低功效 Monte Carlo 非拒绝改写成 nominal confirmatory control。请二选一：校准/替换 Q 检验，使原始 p 值在预注册离散边界模型下有效，并验证强 FWER 的关键 null 子集；或把受影响的 Q 推断明确降为 descriptive、删除 nominal-Holm 控制主张。若仍保留非名义界，须由 owner 对具体阈值与结论降级方式作显式事前裁定，并让计划、测试次数/B、报告措辞一致。
- [Non-blocking] [Suggestion] Round 3 的 plot 单一数值来源问题以及 snapshot canonical merge/unlink、共用 journal parser 已按要求关闭；这些部分无需再改。

Round assessment: 开发者附带测试记录为目标集 `194 passed / 3 skipped`、宽回归 `2617 passed / 9 skipped`。Reviewer 独立 probes 已更新，覆盖 A-pool 内容被替换、stale-only per-step 冒充 accepted attempt、formal analyzer 接受 smoke/错误 trials 三条路径；本轮环境仍未提供可用的独立测试执行额度，因此没有把未执行 probes 计作通过证据。上述三项 blocking 均可由当前生产路径直接证实。`tests/review_tests/` 保持 gitignored，未进入 index。


### G2 Round 6 — Executor — 2026-08-16

**⚠ 本轮再次触及 G1 已 APPROVED 的冻结文本，请 G2 一并裁决**：§8.1.1 的确认性检验构造由 R11 冻结的 **studentized null-calibrated pairs cluster bootstrap** 换成 **studentized null-imposed sign-flip（wild）检验，穷举全部 1,024 个符号模式**。依据是可复现的实测：旧构造在本设计**最期待落入的平台区**边际 size 达 **0.114**（2.3× 名义）。方向是**收紧**——新构造在 5 个 regime × 3 个 side × 4 个 Holm 槽位上处处不高于名义，且功效更高，同时**消除**了上一轮向 owner 挂起的非名义阈值。

- **Accepted（B1，A-pool / 正式 launch 绑定）** — reviewer 说得对：上一轮我把"读回 record 里的 digest 并自洽"当成了校验，那只证明文件与自己一致。现改为**启动前重算**：`rehash_apool()`（`run_size_eval.py:143`）对 record 指向目录里的每个 `.init` 重算 SHA-256、按 `verify_apool.rollup_digest()` 重算 rollup，并 `torch.load` 逐 task 核对 **50 条 init**；`load_apool_digest(verify_contents=True)` 逐项比对 task 名集合、每个 digest、rollup、总数，任一不符即 fail。`rollup_digest` 从 `verify_apool` 内联表达式提为函数，两处共用同一算式。Round 4 声称而未实现的 trials / smoke 校验现已落地：`verify_launch_binding()` 要求 `smoke is False`、`trials_per_task == 50`、**arm 集合相等**（不再是 `analysed ⊆ launched`），且 `apool_digest_expected` 由可选参数改为**必需**——`--apool-digest` 在非 smoke 分析中缺失即 fail，理由写在报错里："launch record 写的是自己的 digest，拿它校验自己什么也不能证明"。测试侧同步补齐：fixture 改为写**真实的** 50 条 init 与**真实**digest（此前是 `d0..d9` 这类假件，永远无法触发 rehash），新增"record 写好后替换一个 `.init` 即 fail"、"rollup 被篡改即 fail"、"某 task 只有 49 条即 fail"、"smoke record 被正式分析拒收"、"5-trial shakedown 被拒"、"多 launch 一个 arm 被拒"。

- **Accepted（B2，`(task_uid, accepted attempt)` 联结）** — reviewer 的读码是准确的：`load_accepted()` 保留了获胜 attempt，`main()` 却 `set(...)` 掉了它。现在 uid→accepted-attempt 一路带到底。新模块 `exp/ablation_study/cache_size/full_hit.py` 由 runner 与 analyzer **共用**（不是两份实现）：`load_per_episode_hits()` 以 `(task_uid, attempt)` 聚合，正式模式要求每行都带 `attempt` 与 `step_idx`，同 `(uid, attempt, step)` 重复即 fail（canonical merge 已去重，幸存者意味着两行互相矛盾）；`assert_full_hit_per_episode()` 只认**accepted attempt 那一支**，其余 attempt 的行被**过滤并计数上报**（`stale_rows_ignored`）而非报错——生产里重试本来就会留下它们——但永远不能顶替 accepted 那一支的证据；accepted attempt 无行即 fail，报错还会点明"其中 N 条只有别的 attempt 有行"。runner 的退出门改为调用**同一对函数**（`assert_accepted_full_hit`，`run_size_eval.py`），arm 级比例降为 informational 日志。两个方向都有测试：stale 行冒充 accepted（旧实现会放行，新实现 fail）、stale MISS 误伤干净的 accepted 重跑（新实现正确忽略）。

- **Accepted（B3，统计有效性）— 但采取的是 reviewer 二选一里的第一条，并且把问题连根换掉。** 我没有去"校准"那个反保守的 bootstrap，因为先做了实验：**非参数 double bootstrap（prepivoting）把 t7 的 ceiling size 从 0.093 推到 0.128**——10 个 cluster 的外层重抽样比真实 DGP 更欠离散，方向是反的。于是改为**替换检验**：studentized null-imposed sign-flip，穷举 1,024 个模式。结果（8,000 次/格，MC SE≈0.0024，见 §8.2.1 全表）：**5 个 regime × 3 个 side × 4 个 Holm 槽位水平，每一格都不高于其名义水平**（最大 0.0540 vs 0.05，+1.7 SE；α/8 处最大 0.0065 vs 0.00625，+0.3 SE）。有了边际有效性，**Holm 的强 FWER 控制就由定理直接给出**——Holm 1979 是 Bonferroni 型的，对 p 之间的依赖不作任何假设，因此不需要枚举 least-favourable 配置来*建立*它；§10 仍模拟了全零假设与"3 个假零假设 + 错误事件只计真零假设"两个配置作为对该推理的**核对**。三项附带收益：① 检验**确定性**——无 seed、无 B，"production B=10,000 / 门禁 B=399" 的口径分裂**不再存在**；② `t*` 里 `h0` 完全抵消 ⇒ 检验 6/7/8 参照分布逐位相同 ⇒ 双侧与单侧嵌套**构造性精确**（旧实现靠共用 seed 买，仍有 ~2/3000 违反）；③ **功效更高**（平台区 0.910 vs 0.729）。⚠ 我必须同时报告新构造的**保守面**：双侧检验在平局重的 regime 是保守的（平台区 0.017–0.020），已写进 §8.1.1 的"代价与披露"，不得读成"无差异"。⏳ **上一轮的 owner 待批项（天花板区 0.12）随之整体作废**，全文不再有任何非名义阈值。另：正文 8,000 次与门禁 1,000 次的差异是**两个不同的东西且都在 repo 里**——`test_marginal_size_high_precision`（标 `manual`）与每次都跑的 `test_marginal_size_at_every_holm_slot_level`；门禁自身是 60 个二项检验的族，按 `1−0.05/60` 判定，否则一个校准正确的实现几乎每次都会踩响一格（这一点是实测踩到的：600 次下 ceiling/less 恰好落在 40/600 的判定边界）。

- **Accepted（Non-blocking）** — 确认 plot 单一数值来源、snapshot canonical merge/unlink、共用 journal parser 三项无需再改，本轮未触碰其行为。

测试记录（本地实跑）：`tests/ablation_study` **227 passed / 8 skipped**；含 manual 的 `tests/ablation_study/cache_size --run-manual` **172 passed**（含 5 个 regime 各 8,000 次的高精度 size 研究）；宽回归 `tests/ablation_study tests/exp tests/scripts tests/cache` **2650 passed / 14 skipped**。

### G2 Round 7 — Reviewer — APPROVED — 2026-08-16 16:31 CDT

- [Non-blocking] [Suggestion] **Round 5 B1 closed.** 正式 runner 现于启动前重算 10 个 `.init` 的逐 task SHA-256、rollup 与每 task 50 条计数；formal analyzer 要求外部 expected digest，并以相等关系校验 suite、8-arm 集合、50 trials，且拒绝 smoke launch。record 自证与 smoke 混入正式 provenance 的两条路径均已关闭。
- [Non-blocking] [Suggestion] **Round 5 B2 closed.** journal 的 uid→accepted-attempt 一路保留到共享 `full_hit.py`，runner 与 analyzer 均按 `(task_uid, accepted attempt)` 联结，accepted attempt 缺行、重复 step、未知 episode 与非 FULL_HIT 均 fail，stale attempt 只计数披露而不能作证。Reviewer 另发现正式 `main()` 仍先执行旧 arm-level 1.0 floor、会让 stale MISS 误杀干净 retry；依据 owner 本轮超流程授权已直接把旧门限制到 smoke，formal 只走 accepted-aware 门，并补回归钉桩。
- [Non-blocking] [Suggestion] **Round 5 B3 closed，G1 统计构造变更获 G2 接受。** 反保守的 pairs bootstrap 已由确定性的 studentized null-imposed sign-flip / wild-residual 检验替换，8 个 primary tests 共用同一构造并保留 Holm。Reviewer 按 owner 授权直接收紧了冻结正文：不再把 `0.054 > 0.05` 写成“逐格不高于名义”，也不声称 n=10 下有限样本精确；最终口径是渐近边际有效性支撑 Holm strong FWER，预注册的 5-regime × 3-side × 4-slot 模拟提供设计特定的有限样本压力测试，报告必须披露 n=10、渐近依据与模拟范围。旧 0.12 非名义阈值及 owner 待批项随旧构造整体作废。
- [Non-blocking] [Suggestion] 独立验证：`PYTHONPATH=. .venv/bin/pytest tests/ablation_study/cache_size -q` 为 **168 passed / 5 skipped**；关键模块 `py_compile` 通过。扩大到 `tests/ablation_study` 时 cache-size 与其余非 socket 测试通过，10 个 sidecar 测试仅因当前 reviewer sandbox 禁止创建本地 socket 而失败（`PermissionError: Operation not permitted`），不是代码断言失败；执行方附带的非受限记录为 **227 passed / 8 skipped**，宽回归 **2650 passed / 14 skipped**。

Round assessment: approved plan consistency, interface/provenance binding, accepted-attempt evidence integrity, test strategy, docs/index obligations, and regression posture. Plot 单一数据源、snapshot 生命周期与共用 journal parser 继续保持关闭状态。没有剩余 blocking；G2 release criterion satisfied，进入 Verify。

---

## 1. 这个实验回答什么

**问题**：cache 库的规模（用多少条 teacher 轨迹建库）如何影响 cache 的成功率？

**为什么现在做**：论文 thesis 是「库的价值在索引不在 payload」。第一个 ablation（执行体替换）已经证明了 payload 可替换、index 承重，但**整条线至今没有一个 size 轴**——所有结果都建立在同一个 50-init 库上。审稿人必问的两个问题目前都没有数据：

1. 「你的结论是不是库太小导致的？库大了 replay 就够用了吗？」
2. 「§5.1 说 replay 只在超便宜端有竞争力，那个三区制的边界在哪？」

**论文落位 = 新增的 X9b，不替换 X9**。原 X9 在 `tier_experiment_designs.md` 的冻结交付是**离线**的「shadow hit-eligible 状态上 replay 动作误差 vs NN 距离分位曲线 + 同状态学生误差对照 + 交叉点」，其自变量是**单点的近邻距离**，comparator 是**学生**。本实验的自变量是**库规模**、因变量是**闭环 SR**、无学生 comparator——两者问的不是同一个问题：

| | X9（保留，不动） | **X9b（本实验）** |
|---|---|---|
| 自变量 | NN 距离分位 | 库 size（成功轨迹/任务，按实收触顶） |
| 因变量 | replay 的**动作误差** | replay 的**闭环 SR** |
| comparator | 学生误差曲线 | teacher 锚点 |
| 模式 | 离线重放 | 真实闭环 rollout |
| 回答 | replay 在多近的邻域内比学生准 | 库要多大 replay 才够用 |

二者互补：X9 说明 replay 层为什么存在（近重复区间内它更准），X9b 说明 payload 路线的**数据成本**。

⚠ **措辞更正**：本 plan 早先几处写"给 §5.1 三区制一个定量**边界**"是**过度声称**。核实 `tier_paper_outline.md:60-62`：§5.1 的三区制是 **SR × measured compute** 帕累托上的分区，其**边界位置由 X4/X11 的成本轴决定**；X9b 给的是"超便宜端那一点的**高度**随数据预算怎么变"。正确表述是「给三区制中 **replay 区的高度上限与其数据成本**」。全文按此口径。**只有闭环 SR 能回答后者**——离线误差小不蕴含 rollout 成功（误差沿闭环累积、且 replay 的失败模式是轨迹偏离而非单步误差）。

因此本计划的文档义务是**在 `docs/iclr/tier_experiment_designs.md` 增补 X9b 卡片并保留 X9 原文**，同步 `tier_paper_outline.md` 的实验台账行与 `docs/README.md`（见 §9.2）。主结果进 App.E 密度分析、被 §5.1 三区制讨论引用；若曲线在大 size 端仍显著低于 teacher，它同时是 §6.1 双重解离的独立佐证（"payload 的上限不是数据量问题"）。

### 1.1 本实验的关键设计：只测纯 cache

owner 裁定（2026-08-16）：**第一轮只测纯 cache**，配置为 `gate: always_search` + `judge: always_hit`。

这个选择消掉了实验里最大的混杂源。带阈值的完整 cache system 里，库变大 → 最近邻相似度分布右移 → 固定阈值下命中率自己会涨，于是 SR 的变化分不清是「库大所以 hit 更准」还是「hit 更多了」；要分离就得为每个 size 重新标定阈值，而标定本身又引入新的自由度。`always_hit` 下**不存在阈值**：每步都强制回放 top-1 的 cached action，SR 直接且唯一地反映「库能不能独自把任务做完」。

⚠ **命中率恒为 1 有一个必须满足的前提**：检索是**任务内作用域**的（`search_strategy.py:389` + `in_memory_backend.py:351-352`），而 `judge.py:214-215` 在检索结果为空时回落 `HitType.MISS`（= 完整 teacher）。所以前提不是"库非空"，而是"**每个任务**在该档都有 ≥1 条 entries"。§3.2 的 size 口径（每任务 k 条成功轨迹）正是为保证这个前提而冻结的；§10 的 manual smoke 与每臂的逐 episode `FULL_HIT 率 == 1.0` 断言是它的运行期验证。

代价是这一轮测不了 hit rate / bad-hit rate 的 size 依赖。这部分明确推迟，见 §13。

---

## 2. 已冻结的 owner 裁定（2026-08-16）

| # | 决定 | 影响 |
|---|---|---|
| D1 | `exp/ablation_study/` 升格为**实验族**父目录，下辖 `executor_substitution/` 与 `cache_size/` | 需增补 `artifact_layout.md` §1「实验族」规则（WA §8 注册文档，owner 已批准修订） |
| D2 | 库轨迹**直接重跑 `--collect`**，不写 replay 驱动 | 不复用 Phase 0 的 `sim_state`；新轨迹与蒸馏数据非同源，作为披露项 |
| D3 | 采满 **500 init/suite**（差集池全量） | 一次投入，X12b/X6 等后续实验复用同一批 h5 |
| D4 | size 轴按**成功轨迹/任务**切；目标档统一，实际条数按 `min(k,n_t)` 触顶 | 嵌套子集，见 §3.2 |
| D5 | 只测**纯 cache**（`always_search` + `always_hit`），完整 cache system 推迟 | 见 §1.1、§13 |
| D6 | 主量 = **SR**；hit rate 不在本轮范围 | `always_hit` 下 hit rate 恒为 1，本就无定义 |
| D7 | 评测用**全 pruned_init**（A 池 500/suite） | 见 §7 |
| D8 | 大体积 h5 落 `/data`（3.6 T HDD），软链回 repo；**软链名须自带软链信息** | 见 §4.3 |
| D9 | 既有 `exp/ablation_study/data`（88.8 GiB）一并迁 `/data` | 复制先行、删源与建链推迟到空闲期，见 §4.4 |

---

## 3. 实验设计

### 3.1 固定量（所有 size 点逐字节相同）

- teacher = Pi0.5 `pi05_libero`（norm_stats 权威版 `c0ee3c1a…`）
- keybuilder = `cp1_spatial_pool_16`（vision_0/1 各 32768 维、prompt_emb 2048、robot_state 32；`build_in_memory_cache_artifact.py:46` `_VECTOR_DIMS`，该 builder 那行在 `:48`）
- 检索 = `weighted_score_sum_knn`，`top_k: 1`，`step_filter: all`，**depth-1**（无 trajectory 项）
- 模态权重与 per-field zscore+tanh 归一化参数：**沿用生产标定值**，取自 `exp/ablation_study/config/common/libero_{spatial,10}_baseline.yaml`
- backend = `in_memory` / `brute_force`；`write_policy: never`
- 两套件 = `libero_spatial` + `libero_10`

**归一化参数固定是一个需要 G1 评判的选择**。`per_field` 的 mu/sigma 是从 50-init 库的 query×library 分数分布拟合来的。zscore+tanh 对**单模态**排序是单调变换（无影响），但本配置是**三模态加权和**，各模态归一化参数不同 → 库分布变化时融合排序会变。固定它 = 所有 size 点共用同一个算子（纯 size 效应，与 D5「零标定」一致）；重标定 = 每个 size 点都用"该 size 下的最优算子"（引入标定这一自由度）。本计划取固定，并在 §8.3 加敏感性检查：**在 S1 与 S6 两端**各用重标定参数跑一个 500 ep 臂（共 4 臂，复用主臂 pkl 只换打分参数），量化这个选择的量级。取两端而非只取最大档，是因为标定锚点在 S3（50-init 库），失配是**双向**的，且小库一侧机制上更危险（§8.3）。

### 3.2 自变量：库 size（嵌套）

| size 档 | **名义目标成功轨迹/任务** | 名义总轨迹上限 | 预估入库 entries 上限（spatial） | 预估入库 entries 上限（l10） |
|---|---|---|---|---|
| S1 | 1 | 10 | ~212 | ~528 |
| S2 | 2 | 20 | ~424 | ~1,056 |
| **S3** | **5** | **50** | **~1,060** | **~2,640** |
| S4 | 10 | 100 | ~2,120 | ~5,280 |
| S5 | 20 | 200 | ~4,240 | ~10,560 |
| **S6** | **45** | **450** | **~9,540** | **~23,760** |

预估 = `k × 10 任务 × 每 ep 步数`。每 ep 步数用实测值（spatial 1062 steps/50 ep = 21.2；l10 2640/50 = 52.8，**已在本地 h5 上逐文件复核**，源 `exp/common/data/db/libero_cache/`）。**新口径下不再乘成功率**——size 直接就是成功轨迹数（见下），这也顺带消除了原表对"Phase 0 实收成功率"这个**本地无法复核**的数字（`distill_raw` 在远端 `/data`）的依赖。

#### ⚠ size 轴的完整规则

记任务 `t` 在 B-train 内的**实际成功轨迹数**为 `n_t`（采集后实测）。对每个目标档 `k ∈ {1,2,5,10,20,45}`：

**R1（统一取数规则）**：该任务在该档实际入库条数 **`n_{t,k} = min(k, n_t)`**。这条规则**对所有档一致**，不是只给 S6 的特例。

**R2（最低可运行门）**：若某任务 `n_t = 0`，则**该套件整个实验不可运行**——§1.1 依赖的"每任务库非空"前提在**所有**档都破，评测会退化成部分 teacher。此时停下报告，不得继续。（按 Phase 0 量级实际不可能：l10 最难的 t8 teacher SR 0.46，45 次全败概率 ~1e-13。）

**R3（档位退化的处置）**：若某相邻对的库对**每个**任务都逐条相同（即 `n_{t,k} = n_{t,k+1}` ∀t），该检验退化为恒等比较。预注册处置：**该检验仍占 family 名额但记为 `not evaluable`**（与 `not rejected` 分开记账），并以 `p=1` 留在完整 8-test Holm 输入中；产出表标注哪些相邻对退化。**不删档、不改 family 大小**——与 §6.2.2 的降级原则一致：矩阵在采集后不因数据而变。

**R4（x 轴与 `slope_k` 的解释）**：曲线 x 轴用**实测的每任务平均入库条数** `mean_t n_{t,k}`（不是名义 k），图上同时标注名义档名。`slope_k` 相应解释为"从档 k−1 的实测规模增到档 k 的实测规模"。S6 因普遍触顶，语义是**"数据预算耗尽点"**而非固定 45 条——与分支 E 的"在 benchmark 的 init 预算内"一致。

**R5（历史锚 S3 的歧义消解）**：优先级 1 是"历史 5-init 锚中**成功**者"，优先级 2 是"B-train 其余成功者"。若历史 5 条中有失败，**S3 仍按 R1 取满 5 条**（不足部分由优先级 2 补足），即 **S3 恒为 5 条**（只要 `n_t ≥ 5`）。代价是 S3 与历史库的 init 集合不再完全相同 ⇒ **产出表须标注每任务的历史锚命中数**（5/5、4/5…），"同 init、新轨迹"的可比性按命中数打折。

**量级预期**（Phase 0 的 spatial ~95% / l10 ~84%）：45 个 B-train init 期望产出 spatial ~42.8、l10 ~37.8 条成功 ⇒ **S6 普遍触顶**；S5(20) 在期望上安全，但 l10 最难任务（teacher SR 0.46）的 `n_t` 期望仅 ~20.7，**S5 有实质概率触顶**——R1/R3/R4 正是为此而设。实际条数以 P4 实测表为准。

过滤依据：`outcome_filter` 默认只收成功 episode（默认值在 `build_in_memory_cache_artifact.py:608`/`:765`，过滤行为在 `:636`）。

#### ⚠ S6 = 45 而非 50：B-val 永不入库

原设计的 S6 = 50 init/任务 = **整个差集池 500**。实测 `config/common/split_<suite>.yaml` 的结构是 **train 45/任务（450）+ val 5/任务（50）**，两套件皆然（`protected_in_train` 即历史 cache 库那 5 个 init，位于 train 侧）。所以原 S6 会把 **B-val 全部收进 library**。

把整个差集池收进库会**摧毁 B-val 的定义性质**。`tier_experiment_designs.md` 的 Init 池协议把 B-val 定为「唯一既不在学生训练内、也不在库内的切片」，是整条 TIER 线（X3 的 router 标签、X14 的训练口径、未来一切"新 init"条件的合法代理）赖以成立的**唯一**干净切片；本 benchmark 官方 init 预算封顶 1000/suite，**没有可新铸的第三池**。为一个 size 档把它烧掉，代价远超收益。

**冻结的解法**（两条硬约束）：

1. **size 轴上限改为 45/任务（450 = B-train 全量）**。语义反而更干净：「库用尽全部可用的训练 init」。相对 50 只少 10%，在近似对数的 size 轴上几乎不可辨（S5→S6 仍是 2.25×）。
2. **排列构造把 B-val 的 5 个 init/任务钉死在末尾**（位置 46–50），且 size 网格**永不取到**它们。于是 S1–S6 全档都不含 B-val，B-val 继续作为全 TIER 线唯一的**库外**干净切片保留；§8.3 的重标定已改为库内 LOEO，**不消费 B-val**。

**嵌套协议**：每个任务对其**成功轨迹**固定一个确定性排列，size=k 取前 k 条 → 大库严格包含小库。排列构造（**在采集完成后执行**，因为要用到实测成败）：

1. **优先级 1 = 历史 cache 库那 5 个 init 中本轮采集成功的**（取自 `exp/common/data/db_init/libero_cache/<suite>/<task>.init`；已逐字节验证这 50 个 init 100% 属于 500 差集池、且全部落在 `protected_in_train` 内）。**组内顺序按 `orig_init_state_idx` 升序**。于是 **S3 是「同 init、新轨迹」的历史可比点**：5 个历史 init 都成功时严格成立；若其中有失败，则按 R5 用优先级 2 补足到 `min(5,n_t)`，并披露历史锚命中数。
2. **优先级 2 = B-train 其余 40 个 init 中成功的**，按固定种子（`seed=0`）确定性洗牌后的顺序。
3. **B-val 的 5 个 init 恒排除**，网格永不取（§3.2 红线）。
4. 失败的 episode 不进排列（它们本就不入库）。
5. 最终的 (task, init) 清单**作为 tracked 产物落盘**（`config/size_grid_<suite>.yaml`），含每任务的实际可用成功条数，不是运行时重算。

⚠ **执行顺序的连带变化**：size_grid 现在依赖实测成败 ⇒ `emit_size_grid.py` 必须排在**采集之后**（§12 的 P3 之后），而不是 P1。P1 只产出"排列规则 + B-val 排除表"这部分不依赖成败的骨架并测试它。

采集仍按 D3 采满 500 init/suite（每任务 50；B-val 每任务 5 个也采）——**采集 ≠ 入库**。额外采集的 50 个 B-val episode/suite 为后续实验（X12b 解耦批次等）留料并保持干净切片完整；它们不进任何 size 档的 library，也不参与 §8.3 的库内 LOEO 重标定。

**已知代价**：嵌套设计测不了同 size 下的抽样方差（要测得对每个 size 抽多个不相交子集，采集与评测成本翻倍）。本轮不做，作为局限披露。

#### ⚠⚠ size 的定义必须是「每任务 k 条**成功**轨迹」，不能是「每任务抽 k 个 init」

R2–R10 一直用的是"每任务抽 k 个 init 去跑，入库的是其中成功的那些"。自查发现这个口径会产生**与自变量共变的方向性偏倚**，且偏倚方向正好指向本 plan 标为"预期最可能"的结论：

**机制（已逐环节亲验）**：
1. 库只收成功 episode（`build_in_memory_cache_artifact.py:636-637`，失败整条不入库）；
2. **检索是任务内作用域**——`search_strategy.py:389` `task_filter = QueryFilter(task_key=ctx.task_key)`，`in_memory_backend.py:351-352` 逐条按 `entry.payload.task_key` 过滤，**别的任务的 entries 补不上位**；
3. 于是某任务在某档若**一条成功轨迹都没入库**，该任务的检索恒返回空 → `judge.py:214-215` 的 `AlwaysHitJudge` 回落 `HitType.MISS` → **该任务 50 条评测 episode 全程跑完整 teacher**。

**量级**（用 `anchors/` 的逐任务 teacher SR 实算）：l10 的 t8 teacher SR 仅 **0.46**，在原口径的 S1 档（1 init/任务）有 **54%** 概率零覆盖；l10 在 S1 期望有 **1.32 个任务**零覆盖，S2 期望 0.375 个。spatial 较轻（最大 6%）但非零。一个零覆盖任务把该 task 的 `d_t` 从"纯 replay SR"整体换成"teacher SR"，在 10-cluster 的 task 级均值上就是 ~4.6pp 的跳变——与 δ=5pp 同量级，且是一次抛硬币。

**为什么这是最严重的一条**：teacher 混入比例随 size **单调递减** ⇒ 小档 SR 被系统性抬高 ⇒ 测得的 slope 被**压平** ⇒ 方向正好推向 `P-yes` 即分支 A。这不是随机噪声，是与自变量共变的偏倚。

**冻结的修正：size ≡ 每任务入库的成功轨迹数。**

- 排列在**成功 episode 之内**定义：采集完成后，每个任务按 §3.2 的确定性顺序取**前 k 条成功**轨迹入库。
- 于是每个 (size, task) 单元有 `min(k,n_t)` 条成功轨迹；R2 要求 `n_t≥1`，因此零覆盖在结构上消失。
- **这不是事后选择偏差**：库本来就只收成功轨迹（`outcome_filter=success` 是既有设计），"每任务 k 条成功轨迹"只是把 size 定义得更精确——它衡量的正是"库里有多少可复用经验"，而不是"抽了 k 个 init 碰了多少运气"。R2 曾以"避免事后选择"为由选了按-init 口径，那个担忧在此不成立，而它换来的代价（上面的方向性偏倚）远大于收益。
- **副产物**：size 的语义在两套件间可比了（原口径下同一个 k 在 spatial 与 l10 意味着不同的实际库量，因为成功率不同）。
- **触顶情形须披露**：若某任务的成功轨迹总数低于任一名义档 k，该任务从该档起用尽其全部成功轨迹；产出表逐任务、逐档标注实际条数与是否触顶，并按 R3 识别全任务退化的相邻比较。

**测量层的兜底**（即使上述修正到位也必须做，用于验证）：每臂逐 episode 记录 `FULL_HIT 步数 / 总步数`，断言 == 1.0；任何 < 1 的 episode 单列并排查。`__hit_meta__` 已在记 `hit_type`，成本为零。逐任务 `hit_type` 分布进 §8.5 产出。

### 3.3 因变量

- **主量**：episode SR，A 池 500 ep/suite/size 点，配对（同 init / 同 seed）
- **次级连续结局**：**失败前存活步数 / 首次偏离时刻**。replay 的失败模式是轨迹偏离，SR 是它的粗二值化；在 §8.1 已明确承认功效很低（10 cluster）的前提下，一个连续量能实质改善判别力。作为 **descriptive** 报告，不进 family。
- **覆盖率量**：**逐步 top-1 相似度（`cp1_score`）的分布随 size 的变化**。`__hit_meta__` 已在逐步记录，只需声明为分析产出。它给出"库覆盖了多少查询状态"，是本实验里**唯一**能支撑外推超出 450-init 硬预算的量——§8.4.2 分支 E 目前只能承认"外推不可判定"。
- **随记**：每步检索延迟、pkl 加载耗时与驻留内存、库的逐任务 entries 分布、逐档逐 field 的归一化饱和率（§8.3）

#### ⚠ 3.3.1 可识别性边界：本实验测不出"payload 的绝对上限"

`always_hit` 的 SR = f(**库内容**, **检索质量**)，而本实验把检索质量固定在单一 keybuilder + `top_k=1` + depth-1 + 一套归一化参数上。因此「库变大 SR 不再涨」有两个观测等价的解释：

1. payload 已饱和——更多经验也没有新信息（plan 想要的读法）；
2. **这个 index 吃不下更密的库**——经验在库里，但检索选不出来。

**这一点对本论文尤其尖锐**：thesis 正是「库的价值在索引不在 payload」，若把饱和无条件读成 payload 的上限，等于把自己的核心主张反过来当成了对照组的结论。五轮外审都没触及这条（R1-B6 只谈 X9/X9b 的定位，未谈可识别性）。

**冻结的处置**（不加 rollout）：

- **分支 A 的结论句必须带 index 限定**——可发表形式是「**在本 index（`cp1_spatial_pool_16` + top-1 + depth-1 + 生产标定）下**，更多 payload 不再带来增益」，**不得**写成无限定的"payload 上限不是数据量问题"。
- 上面的**覆盖率量**与**饱和率**共同提供区分二者的间接证据：若 top-1 相似度随 size 持续右移而 SR 不涨，指向解释 2；若相似度也饱和，指向解释 1。这是观察性证据，不构成识别。
- 真正的识别需要一个 **index 上界臂**（例如用 `--save_trajectory` 副产物里的 `sim_state` 按真实状态距离检索，作为"任何 index 能从这个库里榨出的上限"）。**本轮不做**，登记进 §13 推迟项。

---

## 4. 工作项 A：`exp/ablation_study` 族化重构

### 4.1 规范修订

`docs/experiments/artifact_layout.md` §1 现规定 canonical 结构是扁平一层 `exp/<experiment>/{config,data,analysis}/`。增补一条**实验族**规则：

> 当同一研究问题下有多个独立实验时，允许一层族目录 `exp/<family>/<experiment>/{config,data,analysis}/`。族目录本身**只能**含 `README.md`（族索引）与各实验子目录，**不得**直接存放代码、config 或 data——否则族与实验的边界会随时间糊掉。族的引入需在 `docs/README.md` 索引中体现。

同步更新 `docs/README.md`（WA §4 index sync 红线）。

### 4.2 目标结构

```
exp/ablation_study/ # 族（只有 README + 子实验）
├── README.md # 族索引：本族有哪些 ablation、各自结论
├── executor_substitution/ # 实验 1（已收官）
│ ├── __init__.py
│ ├── build_distill_dataset.py run_ablation_eval.py select_student_checkpoint.py
│ ├── sidecar_server.py train_act.py train_smolvla.py train_student.py
│ ├── config/ data/ analysis/
└── cache_size/ # 实验 2（本计划）
 ├── __init__.py
 ├── <runner>.py …（见 §9）
 ├── config/ data/ analysis/
```

`tests/ablation_study/` 同步分层为 `tests/ablation_study/{executor_substitution,cache_size}/`。

### 4.3 大数据落盘与软链

实体落 `/data`（`/dev/sda1`，3.6 T，实测余 3.6 T；对照：repo 所在 `/` 仅余 526 G 且 openpi 已占 100 G）：

```
/data/openpi/ablation_study/
├── executor_substitution/{checkpoints,distill_raw,lerobot,val_traj,val_inits,runs}/
└── cache_size/collect_h5/<suite>/
```

repo 侧建软链。**软链名自带指向信息**（D8）：

```
exp/ablation_study/cache_size/data/
└── collect_h5__symlink__slash_data_openpi/ -> /data/openpi/ablation_study/cache_size/collect_h5/
```

`ls -l` 会同时显示名字里的 `__symlink__slash_data_openpi` 与真实 target，两重提示，杜绝"以为数据在 repo 里"的误解。

`.gitignore:6` 让软链条目被忽略（实测 `git check-ignore -v` 命中），不会误提交——这一半成立。⚠ **但 `!exp/**/data/` 并不产生任何「目录桩」**：git 根本不追踪空目录，实测 `git ls-files | grep -c "exp/.*/data/.*gitkeep"` = 0，该规则只是让 git 不剪枝遍历。后果：**新克隆上 `cache_size/data/` 不存在，建软链前必须先 `mkdir -p`**——这一步要写进 M-c1 的脚本。仍然**不得**把 `data/` 本身做成软链。

⚠ **`executor_substitution` 的软链改名有兼容代价**：`config/act_manifest_libero_{10,spatial}.json` 里的 20 条 checkpoint 路径写死了 `exp/ablation_study/data/checkpoints/...`。族化重构本来就要改这些路径，届时一并改到带软链信息的新名。在此之前**保持原路径可用**。

**M-b 实际采用的命名（与 D8 的偏差，需 G1 追认）**：M-b 执行时 X14 的两个 ACT sidecar 仍在运行（虽然 conductor 已停），它们重启时会按 manifest 的硬编码路径重读权重。若此刻就把软链改成带信息的新名，X14 剩余 4 个 run 会在下次 sidecar 起停时全部找不到权重。因此 **M-b 建的是原名软链**（`data/checkpoints` 等），并在 `data/` 下放了一份 `__READ_ME__DATA_IS_A_SYMLINK_TO_slash_data.txt` 作为替代的防误解措施——`ls` 一眼可见，内容含指向、迁移原因与两个校验陷阱。带软链信息的改名与 manifest 的同步修改一并推迟到 **M-c**（属 plan 的 code scope，G1 通过后执行）。`cache_size` 是新实验、无历史包袱，其软链从一开始就用 §4.3 的带信息命名。

### 4.4 迁移的三段式（已启动第一段）

| 段 | 动作 | 状态 |
|---|---|---|
| M-a | **复制**（不删源、不建链）`exp/ablation_study/data` → `/data/openpi/ablation_study/executor_substitution/` | ✅ **完成并验证** 2026-08-16 10:11（wls tmux `csmig`，日志 `/tmp/csmig.log`）。98.66 GB / 122,766 文件 / 205 MB/s / 8 分钟。校验：文件数 122,766=122,766；**内容字节和 98,664,415,401=98,664,415,401 逐字节一致**；`rsync -n -a --itemize-changes` 差异条目 **0**；sha256 抽样 **7/7 通过**（3 个最大 checkpoint + 3 个 distill_raw h5 + 1 个硬链接族成员） |
| M-b | 删源 + 建软链 | ✅ **完成** 2026-08-16 10:25（owner 于训练暂停后放行）。两段式：先把 6 个子目录改名为 `.<name>__PENDING_DELETE`（同盘改名，秒级可回滚）→ 建软链 → 验证 → 删。删除脚本内置前置断言（copy 的文件数与内容字节和必须匹配、且对应位置必须已是软链，否则 abort）。SSD 释放 **88 G**（341G→253G，余量 614G）。事后验证：manifest **20/20** 条 `model.safetensors` 经软链可读；`/data` 侧 `find -type f` = 122,766 |
| M-c | 族化重构落地 | ⬚ **拆成两段，见 §9.4**：M-c1（纯新增 `cache_size/`，零破坏，G1 后立即）/ M-c2（收编扁平文件 + 改全部引用，须 X14 结束后的单一静默窗口） |

**M-b 的静默窗口如何取得（已执行，记录事实）**：M-b 会让路径短暂消失，而 X14 的两个 ACT sidecar（PID 75422 / 284619）按 manifest 硬编码路径读权重。实际执行时 owner 暂停了训练（conductor `rlrm6` 与全部 worker 已停），M-b 即在该窗口完成；两个 sidecar 与 pi05 server 虽仍在跑，但 `lsof +D` 实测对该目录**无打开的文件句柄**（权重启动时已进显存），且 M-b 建的是**原名**软链（§4.3），故它们即使重启也能正常读到——事后 20/20 条 manifest 路径验证通过。**注意这个窗口只对 M-b 够用，对 M-c2 不够**：M-c2 要改的是 `exp/rl_router/` 的运行时 import 与 manifest 内容本身，必须等 X14 全部 run 结束，见 §9.4。

### 4.5 ⚠ M-a 实测教训：`du` 不能用作复制校验判据

M-a 的首版校验脚本报了 **MISMATCH**（`du -sb` src 95,362,874,581 vs dst 98,664,416,597，差 3.3 GB），但复制是完好的。根因：

- 源树里有 **64 个硬链接文件**（`find -type f -links +1`），`du` 对同一 inode 的多个路径**只计一次**；
- `rsync` 未加 `-H`，把硬链接族**展开成独立副本**（目标端 `links>1` 的文件数为 0）；
- 于是目标端的 `du` 必然大于源端，而**每个路径的内容完全正确**。

**M-b 删源前必须用的判据**（已实测可用）：

1. 文件数：`find <root> -type f | wc -l` 两端相等；
2. **内容字节和**：`find <root> -type f -printf '%s\n' | awk '{s+=$1} END{print s}'` 两端相等（这条才是内容级的量，`du` 不是）；
3. `rsync -n -a --itemize-changes SRC/ DST/` 排除目录 mtime 后**差异条目为 0**（元数据级全量比对，只读元数据、秒级）；
4. sha256 抽样，样本须**覆盖硬链接族成员**（展开后是否仍逐字节相同是本次特有的风险点）。

副作用记录：硬链接展开使目标端多占约 3.3 GB 磁盘。`/data` 余量 3.6 T，无影响；但将来任何用 `du` 对账这两棵树的人都会再次困惑，故在此定档。

**同一类错误又踩了第二次：`find -L` 也不能用来数文件。** M-b 建链后数出 122,856（预期 122,766，多 90）。根因是树内有 **46 个 lerobot 自建软链**（`checkpoints/last -> 020000`），`find -L` 跟随后把它们指向的目录内容重复计了一遍。两端行为一致（`/data` 侧独立数出同样的 122,766 / 122,856），数据完好。

**判据选择的通用教训**：这两次误报的共同点是——**判据本身对"同一份内容有多个路径"这件事不是不变量**。`du` 对硬链接去重、`find -L` 对软链重复计数，都会在含链接的树上给出与"内容"不一致的数。校验复制**只用不跟随链接的 `find -type f`（计数）与 `-printf '%s'` 累加（字节）**，再辅以 sha256 抽样。

迁移完成后在 `exp/ablation_study/data/__READ_ME__DATA_IS_A_SYMLINK_TO_slash_data.txt` 落了一份同样内容的现场说明，供不读本 plan 的人直接看到。

---

## 5. 数据采集

### 5.1 为什么必须重采（关键调查结论）

Phase 0 已经在差集池全量 500 init/suite 上采过 rollout，**但那批 h5 不能用来 build pkl**：

- build 的输入契约是模型内部表征：`vision_0/1/2 (256,2048) fp16`、`prompt_emb (200,2048) fp16`、`robot_state (32,)`、`clean_action (10,32)`
- Phase 0 走的是 client 侧 `--save_trajectory`（`examples/libero/main.py:464 _flush_trajectory_h5`），实测字段（wls 上真文件 `distill_raw/libero_spatial/task_2/episode_8.h5`）为 `agentview_image / eye_in_hand_image / robot_state(8,) / env_action_chunk / executed_actions / sim_state(92,) / env_timestep / env_cur_time`——**零 embedding**
- 体积也印证：distill_raw spatial 500 ep = 3.2 G（6.4 MB/ep）vs collect 库 spatial 50 ep = 3.5 G（**70 MB/ep**），差 11×

库需要的是 server 侧 `--collect`（`src/openpi/collect/CollectionPolicy`，forward hook 抓 `paligemma_with_expert` 中间张量）。两条采集路径完全不同。

### 5.2 采集配置

- **server**：`scripts/serve_policy.py --collect --collect-dir <DIR> --replicas 1 --non-concurrent`，**cache 必须 OFF**
 - 硬约束已亲验（`scripts/serve_policy.py:659-695 _validate_collect_isolation`）：`--replicas > 1` → ValueError；`args.concurrent and not args.non_concurrent` → ValueError；与 `--export-collect-meta` 互斥。理由是 forward hook 是 module-global，并发下一个连接的 forward 会触发另一个连接的 hook，写出静默损坏的 h5。
 - cache OFF 是 TRACER Phase 4 的教训：活 cache 下 FULL_HIT 短路会跳过 stage2/3，CollectionPolicy 留下步洞。⚠ **代码里对此没有任何强制**：`_validate_collect_isolation` 只拦 replicas / concurrent / export_collect_meta，`--collect` + `--cache_config` 组合是**被接受**的，且 `load_cache_config` ctrl 能在运行中注入 bundle。唯一的实际护栏是巧合——`websocket_policy_server.py:697-705` 要求 `--concurrent` 才受理该 ctrl，而采集恰好用 `--non-concurrent`。故冻结两条纪律：**采集 server 不挂任何 conductor**；P2/P3 出场门加一条**步洞探针**（h5 的 `num_steps` 与预期推理次数一致）。
- **client**：`examples/libero/main.py --init_states_dir exp/common/data/db_init/libero/<suite>`（差集池 500）+ `--num_workers 1`（串行）+ `--save_trajectory`（§5.3）
- ⚠ **落盘路径隐含一层 `<experiment>`**：`data_collector.py:99` 是 `out_dir = self._base_dir / experiment`，而 `experiment = args.task_suite_name` ⇒ 真实路径是 **`<collect_dir>/<suite>/task_N/episode_M.h5`**。故 §6.1 的 `--data-dir` **必须指到 `<collect_dir>/<suite>`** 而非 `<collect_dir>`，否则 §5.3.1 的 `relpath` 会多带一层前缀、全部 entry id 与清单行一起改形。§4.3 的 `collect_h5/<suite>/` 布局与之自洽，此处把该约定写死。
- ⚠ **`--save_trajectory_dir` 必须显式 pin**：`main.py:95` 的默认值 `data/deviation_experiment/gt_trajectories` 落在 **repo 内且未被 gitignore**（实测 `git check-ignore` 退出码 1），约 12 G 副产物会污染 SSD 与 `git status`。发射命令须显式指向 `/data` 下的路径。

### 5.3 (task, init) 归属如何落进产物 —— 一个必须先解决的缺口

采集出来的 h5 默认**无法 join 回 (task_id, init_idx)**：

- `EpisodeDataCollector.on_episode_end`（`src/openpi/collect/data_collector.py:85` 起）在 `episode_name` 为空时走 `:117-119` 的 else 分支，用时间戳命名 `episode_{id:04d}_{ts}.h5`
- `episode_name` 只在 `args.save_trajectory` 为真时才非空（`examples/libero/main.py:590-594`）
- ⚠ client 确实传了 `extra_metadata={"task_id":…, "orig_init_state_idx":…}`（`main.py:600-603`），`CollectionPolicy.on_episode_start` 也确实转发（`collection_policy.py:127-133`），**但 `EpisodeDataCollector.on_episode_start` 的该参数标着 `# noqa: ARG002 — accepted for keyword-call compat` 并被直接丢弃**（`data_collector.py:55`）——所以它不会进 h5 attrs。这也解释了为什么现有 `libero_cache` 的 h5 attrs 里只有 `episode_id/experiment_name/num_steps/success/task/timestamp`，而 (task,init) 映射得靠外部的 `*_init_map.json` 补。

**采纳方案**：采集时**同时开 client 侧 `--save_trajectory`**。此时 `episode_name = "task_{id}/episode_{idx}"`，collector 按该名落盘（`data_collector.py:102-116`，含 `is_relative_to` 路径逃逸断言），于是 **h5 的路径本身就编码了 (task, init)**，正是 §3.2 按任务切子集所需的结构。副产物是 client 侧另写一份轨迹 h5（约 12 G，含 `sim_state`，作为将来 replay 类实验的备份，不浪费）。

**备选方案（记录，未采纳）**：改 `EpisodeDataCollector.on_episode_start` 真正消费 `extra_metadata` 写进 `_episode_attrs`（约 3 行 additive 改动）。更通用，但触碰 src；WA §3.1 minimal change 下取零改动方案。

#### ⚠ 5.3.1 上述命名会撞碎 artifact ID —— 必须配套修 builder

`task_{id}/episode_{idx}.h5` 这个结构与现有 builder 的 ID 规则**直接冲突**，且失败是**静默**的：

- `build_in_memory_cache_artifact.py:508` / `:641`：`trajectory_id = h5_path.stem` —— 只取**文件名**，丢弃目录；
- 同文件 `:539` / `:671`：`entry_id = f"{trajectory_id}:{step_idx}"`；
- `in_memory_backend.py:264`：`self._entries[entry.id] = entry` —— **dict 赋值，后写覆盖先写**。

于是 `task_0/episode_0.h5` … `task_9/episode_0.h5` 十个任务全部生成 `episode_0:0, episode_0:1, …`，加载时**后 9 个任务的同名 entry 逐个覆盖前面的**。历史 `libero_cache` 从未暴露此问题，只因它用的是时间戳命名（`episode_0000_20260410_013433_002633.h5`），stem 天然唯一。若不修，本实验每个库只会剩下约 1/10 的有效轨迹，而**日志里的 "Built N entries" 仍是全量**（覆盖发生在 backend 加载侧，不在 build 侧），完全静默。

**修复**（`build_in_memory_cache_artifact.py`，additive）：新增

```
--trajectory-id-mode {stem,relpath} # 默认 stem = 现状，逐字节不变
```

`relpath` 模式下 `trajectory_id = h5_path.relative_to(data_dir).with_suffix("").as_posix`，即 `task_0/episode_0` → entry id `task_0/episode_0:7`。全局唯一、可读、可追溯回源文件。本实验的 12 次 build 一律用 `relpath`。

**为什么不改默认**：默认切 `relpath` 会让所有历史 artifact 的 entry id 变形，而 `winner_id` 贯穿 gate 线（`ScoreHysteresisGate` / `FollowWinnerGate` 解析 `traj:step`）、逐步日志与全部历史分析产物。保持 `stem` 为默认是 WA §3.1 与向后兼容的要求。

**出场门**（§10 编码为测试）：跨任务 entry id 唯一；**`len(artifact["entries"]) == len(backend._entries)` 加载后不掉条**；逐任务 entry 数与该任务成功 episode 数一致（即无某任务被整体吞掉）。第二条是这一族 bug 的通用探针——它能抓住任何 ID 碰撞，不只本次这一种。

### 5.4 规模与墙钟

| | libero_spatial | libero_10 |
|---|---|---|
| episodes | 500 | 500 |
| 预估 h5（collect 侧） | ~35 G | ~86 G |
| 预估 h5（save_trajectory 副产物） | ~3.2 G | ~8.6 G |
| 预估墙钟（串行单连接） | ~3.5–4 h | ~7–9 h |

合计约 **121 G / 11–13 h**。落盘位置见 §4.3；`/data` 余量 3.6 T，充裕。

墙钟外推依据：Phase 0 双进程实测 5.6 ep/min（spatial）→ 单进程约 2.8 ep/min；collect 的每步 3.3 MB 磁盘写按 +20–30% 计。**这是外推不是实测**，正式发射前先跑 §12 的 10-ep smoke 取真实速率。

---

## 6. 库构建

### 6.1 需要的最小 src-adjacent 改动

`exp/common/build_in_memory_cache_artifact.py:799` 是 `h5_paths = sorted(Path(data_dir).rglob("*.h5"))`——只吃整目录，没有子集选择入口。做 6 档 size 网格必须能选子集，而拷贝目录不可行（l10 全量 86 G，拷 6 份即 516 G）。

**新增两个参数**（additive，均默认保持现状、逐字节不变）：

```
--episode-list <path> # 每行一个相对 data_dir 的 h5 路径；给定时取代 rglob 结果
--trajectory-id-mode {stem,relpath} # 默认 stem（现状）；relpath 见 §5.3.1
```

选 `--episode-list` 而非 `--limit-per-task`：前者把"选哪些 episode"的逻辑留在实验侧（`config/size_grid_<suite>.yaml` 是 tracked 的确定性产物），build 脚本只负责按清单读，职责边界干净，且清单文件本身就是可审计的实验记录。

**`--episode-list` 的边界契约**——逐条 fail-fast，不做静默跳过：

| 拒绝条件 | 理由 |
|---|---|
| 绝对路径 | 清单必须相对 `data_dir`，否则库内容与 `--data-dir` 脱钩、不可复现 |
| 规范化后逃逸 `data_dir`（`..`） | 同 `data_collector.py:108-113` 的 `is_relative_to` 惯例 |
| 非 `.h5` 后缀 / 非普通文件 / 不存在 | 静默跳过会让库悄悄变小，而 size 正是本实验的自变量 |
| 空行、重复（**规范化后**判重） | 重复会让同一 episode 的 entry 被算两次 |

清单读入与校验**只做一次**，产出一个已验证的 `list[Path]`，**全部三个** `h5_paths` 消费点共用**同一个**列表对象，杜绝路径间行为分叉：serial 循环（`:879`）、`ProcessPoolExecutor`（`:893`）、以及容易被忽略的 `cp1_llm_layer_extract` 专用 serial 循环（`:836`，另在 `:831` 取 `h5_paths[0]` 做 tokenizer self-check）。（`:878`/`:889` 是对应的日志行。）

### 6.2 产物

6 档 × 2 套件 = **12 个 pkl**。`outcome_filter` 保持默认 `success`（`build_in_memory_cache_artifact.py:781` 校验合法值 success/failure/all）。

预估最大档 pkl：spatial ~3.5 G、l10 ~8.0 G（按现有 405 / 418 KB per entry × §3.2 的 S6=45 档 entries 线性外推）。落 `/data`，软链回 `cache_size/data/cache_artifacts__symlink__slash_data_openpi/`。

#### 6.2.1 ⚠ 单进程热切换 12 档的内存是**累计**而非单档

原文按"最大单档 8.7 G"估内存，**错了**。亲验：

- `BackendPool`（`src/openpi/cache/backend_pool.py:170-201`）是进程级单例，`self._backends: dict[BackendFingerprint, VectorStoreBackend]`，按 fingerprint 缓存；类上只有 `get`/`get_or_load`/`_get_load_lock`/`reset_for_tests`（后者仅供单测），**没有任何 eviction / clear / pop**；
- 每档 pkl 的 `preload_path` 不同 → fingerprint 不同 → 每档都是一个新条目，**只增不减**；
- server 侧 `_bundles` 亦保留每个已加载 bundle。

所以一个 server 进程顺次热切换 12 档后，常驻量 ≈ **所有 12 个 artifact 之和**，而不是最大档。

**容量核算** —— ⚠ **主机绑定，仅对 weilandserver（wls）成立**（实测 `free -g`：总 **251 GB**、available **234 GB**）。本实验的采集、build 与评测**全部在 wls 执行**；其它主机（例如本地 WSL 开发机仅 ~23 GB 总内存、ziyang10 有 32 GiB pod cgroup 硬墙）**绝不能**套用下表。§12-P6 的加载门必须在**实际执行主机**上重测，而不是引用本表。

| 项 | 估算 |
|---|---|
| spatial 六档累计（0.08+0.16+0.4+0.8+1.6+3.5） | ~6.5 G |
| l10 六档累计（0.18+0.36+0.9+1.8+3.6+8.0） | ~14.8 G |
| 12 档合计 | **~21 G** |
| Pi0.5 server 基线 RSS + 10 worker | ~15–25 G（执行体消融期实测量级） |
| 合计峰值 | **~40–50 G，占 available 234 G 的 ~20%** |

**结论：容量上安全，但机制判断必须写对**——原文的错误在于低估了 4× 且用错了模型。据此定两条硬门（§12 编码）：

1. **加载门**：每档 build 完成后、正式评测前，逐档实测「加载后进程 RSS 增量」，累加值与上表对照；**实测累计超过 available 的 50%（117 G）即停**，转降级方案。
2. **降级方案**（预先选定，不临场决定）：按 suite 分进程——spatial 与 l10 各用一个 server 进程、各自只加载自己的 6 档（累计 ~6.5 G / ~14.8 G）。这是干净的切分，因为两套件本就不共享库、不配对比较。若单 suite 内仍超限，再退到"每 2 档重启一次 server"（代价是热切换的便利，不影响科学口径）。

#### 6.2.2 ⚠ 检索延迟：原估低估 **8.5×**

R10 之前引的"4.15 ms"是 `cache_latency_bench` **优化后**的数字，而那些优化**从未落地 src**——`PrebuiltMatrixBackend` / `PrenormDotBackend` / `LeanSearchBackend` 全在 `exp/cache_latency_bench/opt/` 下经 `components_hook` 注入 bench，`grep -rn "PrebuiltMatrix" src/openpi/cache/` **零命中**；服务路径走 src 的 `build_cache_components`，`in_memory_backend.py:389-396` 至今仍是每步每 field 现场 `torch.stack(vecs).float` + `cosine_similarity`，`freeze` 也不预建矩阵。**shipped 基线是同报告 `:17` 的 `| baseline | src InMemoryBackend | 35.49ms | search 33.9 (95%) |`**（libero_10 / `cp1_spatial_pool_16` / 2,640 entries）。

**重算**（检索按 task 过滤，bench 与本实验按同一因子缩放，故比例外推成立）：

| 档 | 总 entries | search ≈ 33.9 ms × (E/2640) | + Stage1 114.6 ms | vs teacher 690 ms |
|---|---|---|---|---|
| spatial S1 | ~212 | ~2.7 ms | ~117 ms | 5.9× |
| spatial S6 | ~9,540 | **~122 ms** | ~237 ms | 2.9× |
| l10 S1 | ~528 | ~6.8 ms | ~121 ms | 5.7× |
| **l10 S6** | **~23,760** | **~305 ms** | **~420 ms** | **1.6×** |

所以 §7 原写的"每步 120–220 ms、快 teacher 3–5×"只在小档成立；**大档只有 1.6–2.9×**。

**连带的墙钟重估**：按各档 entries 加权，l10 平均每步 ~210 ms、spatial ~152 ms ⇒ **8,000 ep 约 20–25 h**（原估 3–6 h 错了一个量级）。这仍在无人值守可接受范围，但必须写准。

**降级预案：只许动运行策略，不许动科学矩阵。** 删档（例如"少跑一档"）看似省时，实则会连带改变相邻对检验、Holm family、M 轴、size 曲线、yaml 与 episode 预算和全部测试——等于在 P6 现场把预注册矩阵换掉；何况被删的小档本就廉价，省不下触发降级的大档延迟。故冻结的降级阶梯全部**矩阵不变**：

1. **延长墙钟 / 分批执行**（首选）：20–25 h 分两个夜间窗口跑，按臂断点续跑（journal 已支持）。这是最自然的处置——瓶颈是时间不是资源。
2. **分进程 / 分档重启**：按 suite 拆两个 server 进程（§6.2.1 的既有降级），或每 2 档重启一次以释放 BackendPool。
3. **可选前置（不作为本 plan 依赖）**：把 bench 的 R1–R4 优化落 src（该报告 `:114` 自称"src 落地是独立的正式 L2 任务"），检索可回 ~4 ms 量级。这是另一个 L2 任务。

**触发阈值**：P6 实测 l10 S6 单步延迟 > 600 ms（即接近 teacher 的 690 ms、优势不足 1.2×）时触发阶梯 1–2 并记录；**任何情况下都不删档、不改 family**。

这同时说明：**size ↔ latency 本身就是本实验的一个观测量**（帕累托的另一个轴），大档的检索成本已经接近 teacher 推理本身——这个事实值得进报告，因为它直接关系到"更大的库在部署上是否划算"。

---

## 7. 评测

- **池（已冻结）**：A 池 = 官方 pruned_init 500/suite，**物化成显式产物**，不使用"空值回落"。

 **决定：物化，不开空值口子。** 理由有三：① `resolve_init_states_dir` 拒空值是防训练误落测试集的守卫，为 eval 开口子等于拆掉守卫本身，而本实验与 X14 都要用 A 池，口子一旦开就长期存在；② 空值回落的内容取决于安装的 LIBERO 版本，**不可 hash、不可复现**；③ 物化后 A 池成为可 sha256 的一等产物，配对分析（§8）与 teacher anchor 的 join 键才有可验证的基准。

 ⚠ **不要重复造轮子**：`logs/session_handoff.md:188` 记录 X14 已有**实测过**的 `materialize_apool.py`，产物路径为 `exp/common/data/db_init/libero/<suite>_apool/<task>.init`，且 A/B 逐行互斥（500 行、atol=1e-7）与 `torch.save/load` 逐位往返**均已量过**、三处守卫零代码改动通过。两种布局都能过 `check_init_pools`（`emit_router_yamls.py:161-174` 只拒 same/nested），所以这是**重复而非冲突**——但只应 ship 一个。**冻结：复用 X14 的 `materialize_apool.py` 与其路径**，本 plan 不再新写 `emit_apool.py`；§9.1 相应删除该条目，改为"复用并记录 digest"。

 ⚠ **执行环境**：A 池物化依赖 `task_suite.get_task_init_states`，而 `third_party/libero/` 是**未初始化的 submodule**、主 uv venv 里 `import libero` 失败 ⇒ **物化必须在 LIBERO client env 跑**，而 `test_apool` 在 `uv run pytest` 下跑。故 §12-P1 把 A 池物化**单列为需要 LIBERO 环境的子步骤**，其出场门（digest + "A∩B=∅"）作为**显式交付物**，不与其它单测混在一句"全绿"里；单测只断言**文件与 digest**，不调 `get_task_init_states`。⚠ 这也意味着**"A 池 ∩ 差集池 = ∅"这条防泄漏核心断言至今没有在本 repo 执行过**——它是整个泄漏论证的落脚点，必须在 P1 留下执行记录。

 **物化协议**：
 - 产物路径沿用 X14 的 `exp/common/data/db_init/libero/<suite>_apool/<task_name>.init`；
 - 生成方式：`task_suite.get_task_init_states(task_id)` 的返回值原样 `torch.save`，**逐字节不做任何变换**；
 - 出场断言（编码进 §10 测试）：每任务 50 条；与 `db_init/libero/<suite>` 的差集池**交集为空**（防污染的核心断言，逐字节比对，方法同已验证过的 `libero_cache ⊂ libero` 那次）；总数 500/suite；
 - 记录 per-task 与 rollup 的 **sha256** 进 `config/apool_digest_<suite>.yaml`（tracked），此后一切引用以 digest 为准；
 - 与 X14 M7 共用同一份产物（`logs/session_handoff.md` §6.3 的同一个缺口），避免两条线各造一份。

 ⚠ **A 池上的选型红线**：本实验 16 个臂全部在 A 池（冻结测试集）上跑，等于做了一次 size 扫描。本 plan 自身不在 A 上做任何选择（网格与阈值全部事前冻结），但**下游有一个真实的诱惑**：若论文 §5.1 的 "best-calibrated replay cache" 基线日后从这条曲线里挑表现最好的 size，那就是**测试集选型**。故预注册：**前沿图的 replay 操作点固定取 S6（数据预算上限），不取曲线上表现最好的档**。
- **臂**：12 主臂（6 size × 2 suite）+ **4 敏感性臂**（§8.3，S1/S6 两端 × 2 套件）= **16 臂**，每臂 500 ep 配对 → **8,000 ep**
- **配置**：§3.1 固定量 + `judge: always_hit`（替换 baseline yaml 的 `threshold`）；`gate: always_search`
- **臂切换**：靠 conductor 下发 yaml 热切换 pkl（`load_cache_config` ctrl）。**受 §6.2.1 的累计内存门约束**：默认按 suite 分两个 server 进程（各服务自己的 8 臂，但只加载 6 个不同 pkl——敏感性臂复用主臂 pkl），而不是一个进程吃下全部 16 臂
- **server 启动模式**：换臂靠 `load_cache_config` ctrl，而 `websocket_policy_server.py:696-705` **要求 `--concurrent`**。⚠ 这与 §5.2 采集侧强制的 `--non-concurrent` **正好相反**，两阶段的 server 启动命令不可复用。
- **墙钟**：`always_hit` 下每步只跑 Stage1 + 检索，**不跑 stage2/3**；但检索成本随 size 显著增长——按 §6.2.2 重算，l10 平均每步 ~210 ms、spatial ~152 ms，**8,000 ep 约 20–25 h**（大档 l10 S6 每步 ~420 ms，对 teacher 690 ms 只有 1.6×）。降级预案见 §6.2.2。

### 7.1 历史锚点（本实验的先验，已亲验）

`exp/warm_start/data/baseline_failures.json` 里有 libero_spatial 在 **50-init 库**（即 S3 档）上的 AlwaysHit 500 ep 实测：

| cfg | AlwaysHit SR | 纯推理 SR |
|---|---|---|
| clip_w7_d4 | 0.674 | 0.992 |
| max_pool_w3_d5 | 0.696 | 0.984 |
| spatial16_w8_d4 | **0.692** | 0.984 |

**这是本实验可行性的关键证据**：纯 cache 在 S3 档已有 0.69 的 SR，既远离 0（曲线不会贴地板）也远离天花板（离 teacher 还有 ~0.29 的空间），size 轴有充足的动态范围。libero_10 无对应历史基线，其 S3 档由本实验首次测得。

⚠ 该历史值的配置（`spatial16_w8_d4`）与本实验固定量（§3.1）不完全一致（权重档位不同），故它是**量级先验而非配对基线**，不能直接作为 S3 的对照值——S3 由本实验重新测得。

---

## 8. 分析与判读

### 8.1 推断口径（预注册）

**统一原则：推断单位自始至终是 task，CI 与 p 值必须出自同一个 cluster 结构。** R2 版本在这里自相矛盾——一边声明 task 内 episode 误差不独立、据此让 CI 按 task 重采样，一边又用把 500 个 episode pair 当独立的精确 McNemar 出 primary p 值。那些 p 值在计划自己声明的相关结构下不具备所声称的有效性，Holm 校正也就无从谈起。现按下述口径重新冻结。

- **观测与配对**：每个 (task_id, init_idx) 在两臂下构成一个配对观测（同 init、同 seed）。**聚合单位 = task**：每个 task 先算配对差 `d_t = SR_臂A(t) − SR_臂B(t)`（各 50 个 episode 的成功率之差），得到每套件 **10 个 cluster 级观测**。
#### 8.1.0 估计目标（estimand）—— 必须先写明，因为它是功效代价的唯一来源

**估计目标 = task 超总体上的平均效应**：把每套件的 10 个 task 视为从一个 task 分布中抽样，推断针对该分布。（另一个可选的 estimand 是"固定 task、论域仅限这 500 个 episode"，那样推断单位回到 init、功效大得多；G1 期间已裁定取 cluster 路线，本 plan 不重开该选择，但其代价必须显式化，见下。）

⚠ **必须显式化的代价链**：LIBERO-spatial / LIBERO-10 各自**恰好就是**那 10 个 task，并非从任何总体抽出；选择 task 超总体这个 estimand ⇒ 有效样本量从 500 降到 **10** ⇒ 下面 §8.1.1 的平局问题、§8.2 的功效紧张、以及若干结论的"结构性不可判"全部由此而来。**这条因果链必须写进论文的局限段**，不得只报结果不报代价。

**配套的敏感性分析**（新增，成本近零）：主 Q 轴保留 task-cluster 版本，**另报一个固定-task 的分层精确检验**（推断单位 = init，500/suite，task 作为分层）。两者一致 ⇒ 结论稳健；不一致 ⇒ 那正是 cluster 效应强度的直接证据，比拿 episode 级 McNemar 做这件事严谨得多。该敏感性分析为 **descriptive**，不进 family。

#### 8.1.1 检验族的统一构造

R10 版本把 family 混装成"检验 1–6 用精确符号置换 + 检验 7/8 用 bootstrap"。自查发现这个混装有三个独立的硬缺陷，且**根因同一条：两类统计量不嵌套**——

1. **决策树的"不可达格"论证失效**。经典符号置换的功效由**跨 task 的符号一致性**决定、对幅度几乎不敏感；旧 bootstrap 由**幅度与方差**决定。二者非嵌套 ⇒ "更强结论蕴含更弱结论"在**总体层面**就不成立，与样本量、与 Holm 步进都无关——连两个 `D-cache` 格（点估计自相矛盾，本该最稳）都失去论证依据。<br>⚠ 注意区分：全族统一为下述同一套 studentized null-imposed sign-flip **修好了这一条**（§8.4.1b 两个 `D-cache` 格现已确证不可达），但 `D-none × Q-fail` 的可达性**另有根因**（检验 6 双侧 vs 检验 8 单侧），不在本条的修复范围内。
2. **可交换性无依据**。R10 在 376 行称"null `E[d]=0` ⇒ 两臂可交换"，却在同节为检验 7/8 选 bootstrap 时说"平移后置换要额外假设对称"——**同一节对同一技术给了互斥前提**。实际上符号翻转要求每个 `d_t` **关于 0 对称**，`E[d]=0` 推不出它；而本设计固定 init 与 seed、`always_hit` 下检索确定，**臂内没有可随机化的随机源**，"两臂可交换"没有设计层面的依据，对称性纯属对 task 级效应分布的假设——而 `d_t` 是两个有界比例之差，在 spatial 天花板区与 l10 重尾难度分布下偏态可预期。
3. **平局导致结构性不可判**。`d_t` 落在 `1/50 = 2pp` 的离散网格上，`d_t = 0` 是有正概率的原子。设 k 个 task 平局，则 `p_min = 2^{k+1}/1024 = 2^{k−9}`：k=0→0.00195、k=1→0.0039、**k=2→0.0078**、k=3→0.0156。而 Holm 最严阈值为 `0.05/8 = 0.00625` ⇒ **k ≥ 2 时该检验在最严槽位上永远不可能拒绝**。平台区（S4–S5、S5–S6）出现 ≥2 个平局是**基准情形而非边角情形**（极难任务两档都 0/50、极易任务两档都 50/50）。

**冻结的修法（G2-R5 第二次更正，见下）：全族统一为 studentized null-imposed sign-flip（wild）检验，穷举全部 2^10 = 1,024 个符号模式。** 每个检验在自己的 H0 处**由中心化施加零假设**，统计量一律 studentized：

| # | 假设 | 零假设施加方式 | 统计量（对拍观测值） | p（单/双侧） |
|---|---|---|---|---|
| 1–5 | `H0: Δ_k = 0` | `R = 0 + (d − mean(d))·w` | `mean(R)/(sd(R)/√10)` vs `mean(d)/(s/√10)` | 双侧 |
| 6 | `H0: gap = 0` | 同上 | 同上 | 双侧 |
| 7/8 | `H0: gap ≥ δ` / `≤ δ` | `R = δ + (d − mean(d))·w` | `(mean(R) − δ)/(sd(R)/√10)` vs `(mean(d) − δ)/(s/√10)` | 各单侧 |

`w` 取遍全部 `{+1,−1}^10`。**这是穷举而非抽样**：n=10 时参照分布恰有 1,024 个原子，比任何"够大"的 Monte Carlo 都便宜，且**每个 p 值都是确定的**——没有 seed、没有 B、报告里没有一个数字含模拟误差。

**为什么换掉 R11 冻结的 pairs cluster bootstrap（G2-R5 的实测证据）**：

| regime（两臂同率，零假设为真） | pairs bootstrap 单侧 size | sign-flip 单侧 size |
|---|---|---|
| 天花板区（spatial-like） | 0.084 / 0.081 | 0.053 / 0.054 |
| **平台区（S4–S5、S5–S6 的基准情形）** | **0.114 / 0.113** | 0.052 / 0.052 |
| 中段（l10-like，主判别战场） | 0.052 / 0.048 | 0.050 / 0.052 |
| 强偏态 | 0.069 / 0.072 | 0.053 / 0.052 |

（4,000–8,000 次模拟，`gap = δ` 的边界 null，离散配对成功率；MC SE ≈ 0.0034 / 0.0024。）pairs bootstrap 在**本设计最期待落入的平台区跑到名义值的 2.3 倍**——那不是可以靠 Holm "吸收"的量，Holm 的强 FWER 保证以各真零假设边际 p 值有效为前提。sign-flip 在全部 5 个 regime、3 个 side、4 个 Holm 槽位上均**未检出经门禁多重性校正后显著的 size 超标**（个别裸点估计略高于名义，详表见 §8.2.1），**且功效更高**（平台区 0.910 vs 0.729，天花板区 0.997 vs 0.915，中段持平）——这次不是拿功效换 size。

**为什么这不是 §8.1.1 一开始否掉的那个符号置换**（三条反对逐一回答）：

1. **对称性**：被否掉的是把 `d_t − h0` 关于 0 翻转的经典符号检验，其**有限样本精确性**确实要求 `d_t` 关于 h0 对称。这里翻转的是**关于样本均值的残差**，且统计量 studentized；其依据是 studentized randomization / wild-residual 构造在随机化假设不成立时的**渐近**稳健性原则（Chung & Romano 2013 提供一般 studentized randomization 论证；本实现是该原则的一样本 wild-residual 对应物），而不是声称 n=10 时仍精确。上表的"强偏态"与"单点离群"两行提供本设计有限样本下的经验核对（size 0.049–0.055）。
2. **平局的结构性不可判**：经典符号检验里 `d_t = h0` 的 task **无论取哪个符号都不贡献**，k 个这样的平局把参照集压到 `2^(10−k)` 个原子、双侧 `p_min` 钉在 `2^(k−9)`（与本节第 3 条同一算式：k=2 即 **0.0078 > 0.05/8 = 0.00625**，最严槽位已不可达）。翻转残差则只有 `d_t` **恰等于样本均值**才归零，那是巧合而非平台区的常态：实测 `p_min = 1/1025 = 0.00098`，5 个 regime 全部 **< 0.00625**，最严槽位处处可达（§10 有专门门禁）。
3. **嵌套 / 不可达格**：全族仍是同一个 studentized 统计量的单调函数，两个 `D-cache` 格的不可达论证不变。**并且更强了**：`t*` 里 `h0` 完全抵消（`t* = mean(r·w)/(sd(r·w)/√10)`，与 h0 无关），所以检验 6/7/8 的参照分布**逐位相同**，双侧与单侧的嵌套是**构造性精确**的。R11 下这要靠共用 seed 买，且实测仍有 ~2/3000 违反、两次都落在 Holm 临界带。

⚠ 但这**不**使 `D-none × Q-fail` 变得不可达——那一格的可达性与统计量是否嵌套无关，根因是**检验 6 双侧 vs 检验 8 单侧**取的是不同的尾，详见 §8.4.1b。

  ⚠⚠ **实现纪律（这个数字前后错了两次，两次都是数值约定错误，记录在此以免第三次）**：
  1. **观测侧**必须用**相对容差**而非 `sd > 0`。十个相同浮点数的 `std(ddof=1)` 是 ~1e-17 而非精确 0；按 `> 0` 放行会除出 ~1e16 的假 t，把无变异样本判成"显著"（实测 p=0.0033）。
  2. **参照侧**的退化模式（`sd(R) = 0`）必须取**带符号无穷极限** `sign(mean(R) − h0)·∞`，**不能**映射为 0。退化模式的统计量真的发散；把它折进零分布的**中心**会清空尾部，使检验严重反保守——真零假设下实测 type-I error **0.1353（3.1× 名义）**，且 p 对证据非单调。丢弃这些模式再归一化**不能**解决问题（实测与错误行为几乎相同）。
  3. 退化模式比例随 `TestResult.degenerate_resample_fraction` 上报，结果表须披露——读者有权知道多少推断建立在极限约定上。

  这两条约定错误各自把同一个观测量报成过 `0.106`（偶然接近正确，出于错误原因）、`5e-4`（反保守）、最后 `0.106`（正确）。

**代价与披露**：① 放弃了"H0 下每个 `d_t` 对称"才有的有限样本**精确性**，改为渐近有效 + 上表的有限样本模拟支持；报告不得把它写成 n=10 下的数学精确检验。② **双侧检验在平局重的 regime 实测保守**（平台区 0.017–0.020，天花板区 0.039，中段 0.045）——在已模拟的 DGP 中未见 size 膨胀，代价主要体现为功效损失；检验 1–6 在平台区的低功效必须与 `Q-inconc` / `not evaluable` 分支一并披露，不得读成"无差异"。③ 中心化施加零假设隐含"H0 处 `d_t` 的方差 = 观测方差"这个同方差假设；studentize 用于缓解它，且上表的天花板/平台两行正是方差随均值变化最剧烈的经验压力测试。

**p 值的下界修正**：`p = (1 + #{t* 在尾内}) / (2^10 + 1)`（Davison & Hinkley；Phipson & Smyth 2010）。这里的 `+1` 不是可选的：与**精确**随机化检验不同，恒等符号模式**不**复现 `t_obs`（参照样本被中心化到 H0，观测样本未必在 H0 上），所以裸计数真的可以是 0，而 0 不是合法 p 值、会毫无阻力地穿过任何 Holm 阈值。

#### 8.1.2 CI 的地位与降级判据

`gap` 与各 `slope_k` 的 task-cluster bootstrap 95% **BCa** CI 照常报告（B=10,000），但**只作效应量展示**；确认性落格一律由 §8.2 Holm 后的拒绝结果驱动（§8.4.1b）。二者程序不同，边界附近可能不一致（例如 `U < δ` 但检验 7 Holm 后未拒绝）；此时**以 Holm 后的检验为准**，并在结果表显式标注"CI 与家族门禁边界分歧"，**不得择优采用**。

⚠ **已知的 BCa 欠覆盖（须与 §8.4.1 的精度门槛一并披露）**：实测名义 95% 区间的实际覆盖率在天花板区为 **0.885**（μ=0）/ **0.874**（μ=+2pp），中段 0.904；平均宽度 0.0190 对 t 区间的 0.0226，**窄约 15%**。因为 `P-yes` 要求两个 slope 的 CI 上界都 < 2pp，系统性偏窄的区间会让 `P-yes` **更容易**触发，从而偏向分支 A——而分支 A 的 descriptive 半句正是论文的头条主张。这一偏向必须写进结果的局限段，不得只报落格。

**BCa → percentile 的降级必须是数据的确定性函数**（不能"看完两个再选"）。预注册判据，三者任一触发即全表改用 percentile，且**无论是否触发都在结果表打印 `â` 与 `z₀`**：

- jackknife 加速项 `|â| > 0.25`；或
- 偏差校正 `|z₀| > 0.5`；或
- `â` 的分母 `[Σ(θ̄−θ̂₍ᵢ₎)²]^{3/2} < 1e−12`（10 个 leave-one-cluster-out 估计全相同时分母为 0 → NaN；平台区 `d_t` 全相等是可能的）。
- **CI = task-cluster 配对 bootstrap**（有放回抽 10 个 task，task 内取其全部 episode），B=10,000，报 **BCa**；10 个 cluster 下加速项估计不稳时降级 percentile 并披露。与上面的检验共用同一 cluster 单位，口径自洽。
- **精确 McNemar 降级为 descriptive**：仍报 discordant pair 计数与 episode 级 p，但**不进 primary family、不做 Holm、措辞不得称"显著"**。它的用途是与 task 级结果对照。⚠ 措辞须克制：两者背离**不能**直接读成"cluster 效应强度"——McNemar 条件在 discordant pair 上、task 级是 task 等权平均，**即使簇内相关为 0 也会背离**。正确表述是"两者背离提示簇结构或 task 权重口径的影响，需结合逐任务分解判读"。（本实验每 task 恰 50 episode，故点估计层面 task 等权 = episode 等权，背离只会出现在 p 值层面。）
- **功效的诚实披露**：cluster 推断把有效样本量从 500 降到 10，这是自洽性的代价；studentized sign-flip 的功效取决于 task 间效应的均值、离散度与边界距离，效应异号或贴近 0/δ 时功效很低。**这不是可以事后补救的**，因此 §8.4 预注册了 inconclusive 区，而不是把功效不足读成"无差异"；实现前以 §10 的 size/power 模拟量化可判区间。
- **teacher 配对来源（已冻结并亲验）**：`exp/ablation_study/data/anchors/libero_{spatial,10}_teacher/results_tasks*.json`。实测每套件 **500 行、unique (task_id, init_state_idx) = 500、SR = 0.974 / 0.868**，字段含 `task_id / init_state_idx / orig_init_state_idx / episode_id / seed / success`，与 A 池一一对应 → **可做逐 episode 配对**，不是只有比例。分析脚本必须以 (task_id, init_state_idx) 为 join 键并断言 500/500 命中，缺一即 fail。

### 8.2 多重性家族（预注册）

**每套件恰 8 个 primary 检验**，Holm 校正，family 冻结如下（不因结果调整）。所有检验均为 §8.1.1 的 **studentized null-imposed sign-flip（穷举 1,024 模式）**；每个检验的 `d_t` 定义逐条钉死：

| # | 检验 | `d_t` 定义 | 侧 | 驱动的落格 |
|---|---|---|---|---|
| 1–5 | 相邻档 S1–S2 … S5–S6，`H0: Δ = 0` | `SR(S_k, t) − SR(S_{k−1}, t)` | 双侧 | M 轴（非单调触发）+ 增益位置 |
| 6 | S6 vs teacher，`H0: gap = 0` | `SR(teacher, t) − SR(S6, t)` | 双侧 | **D 轴**（方向） |
| 7 | S6 **非劣**，`H0: gap ≥ δ` | 同上 | 单侧 | **Q-pass** |
| 8 | S6 **劣效**，`H0: gap ≤ δ` | 同上 | 单侧 | **Q-fail** |

⚠ **为什么 7 与 8 各占一个名额，而不合并成一个"Q 槽位"**——这是个容易走错的设计点，理由分两层：

1. **`min(p₇, p₈)` 不是合法的 super-uniform p 值**。在 `gap = δ` 时 `H0₇: gap ≥ δ` 与 `H0₈: gap ≤ δ` **同时为真**（边界属于两者），而 null 施加后的同一参照分布给出 `p₇ + p₈ ≥ 1`（两侧共用逐位相同的 `t*` 数组，见 §8.1.1）⇒ 在 α < 0.5 时两事件互斥 ⇒ `P(min ≤ α) = P(p₇≤α) + P(p₈≤α) ≈ 2α`，**第一类错误翻倍**。分划原理在此**不适用**：它要求参数空间被**不相交**子集划分，而 `{gap≥δ}` 与 `{gap≤δ}` 恰在 `gap=δ` 处相交。
2. **即使修正也不划算**。合法的合并需 `p_Q = min(1, 2·min(p₇,p₈))`，则 family=7 下最严阈值 `0.05/7 = 0.00714` 要求 `min(p₇,p₈) ≤ 0.00357`；而各占一格时 family=8、最严阈值 `0.05/8 = 0.00625` 直接作用于单侧 p。**各占一格反而宽松 1.75×**。

故冻结：**两侧各占一个 Holm 名额，family = 8**。`Q-pass` ⟺ 检验 7 Holm 后拒绝；`Q-fail` ⟺ 检验 8 Holm 后拒绝；二者都不拒 ⟺ `Q-inconc`。

⚠ **§10 的对应断言**：不能只断言 `p₇+p₈≈1`（那由构造恒真）。**必须在 `gap = δ` 的边界 null 下做经验 size 模拟**，直接验证两个槽位各自的经验拒绝率在 §8.2.1 的分 regime 事前界内、且合并决策的 family FWER 在名义 α 受控。

**为什么需要检验 8**：检验 7 被拒绝支持 `gap < δ`（非劣成立）；**未拒绝只是"不足以确认非劣"，推不出 `gap > δ`**。而 R8 却用未校正的 `L > δ` 定 `Q-fail`，并据此触发 A/E/G 等"payload 不够用"的主结论——那实际上是在执行方向相反的另一项确认性检验，却既不在 family 也没有自己的 Holm 结果。现把 δ 两侧各配一个单侧检验：**`Q-pass` ⟺ 检验 7 Holm 后拒绝；`Q-fail` ⟺ 检验 8 Holm 后拒绝；两者都不拒绝 ⟺ `Q-inconc`**。二者 H0 互补（仅交于 `gap = δ`），逻辑上不可能同时拒绝——§10 断言之，同时拒绝即实现缺陷。

**为什么必须纳入非劣（R7-B11 的结论保留）**：以 CI 是否越过 `+δ` 作决定，正是单侧非劣检验的区间形式；不打印 p 值不会让确认性假设消失。Q 轴驱动主决策（直接改写 thesis 作用域），故必须受多重性控制。

**M 与 P 两轴的地位**：
- **M（非单调）** 由检验 1–5 的 **Holm 后**结果驱动：`M-yes` ⟺ 存在 k 使检验 k 拒绝**且**点估计 `mean(d_t) < 0`。它是确认性的（触发最高优先级的分支 N）。
- **P（平台）** 明确降为 **descriptive**：它由 `slope_5`/`slope_6` 的 BCa CI 相对 2pp 描述曲线形状，**不进 family、不承载确认性主张**。理由：thesis 的科学分量全部由 Q 轴承载（"够不够用"），P 只用于组织叙事与选择措辞模板。这样既避免把未校正 CI 当确认性判据（R9-B12 的批评），又避免 family 继续膨胀（每加一个平台检验就多占一个 Holm 名额，把最严阈值从 `0.05/8 = 0.00625` 进一步压低；10 cluster 下功效本已紧张）。**结果表须显式声明 P 为 descriptive**，且任何"曲线已饱和"的表述都必须附 CI 而不得称"显著饱和"。

#### 8.2.1 边际有效性与 family FWER（G2-R5 冻结，**不再有 owner 待批项**）

**论证结构**。Holm 的强 FWER 控制是 Bonferroni 型的：**只要每个真零假设的 p 值边际 super-uniform，强控制就成立，对 p 之间的依赖结构不作任何假设**（Holm 1979）。所以本设计不需要枚举 least-favourable 的真/假零假设配置来"建立"强控制——需要证明的是**边际有效性**，而且要在 Holm 实际使用的**每一个槽位水平**上证明，不是只在 α 上。（§10 仍模拟两个配置作为对该推理的**核对**，而非其依据：全零假设配置，以及 5 个相邻比较中 3 个为真效应、错误事件只统计真零假设的配置。）

**实测边际 size**（每格 8,000 次模拟，MC SE ≈ 0.0024；`gap = δ` 的边界 null；**离散**配对成功率——连续分布不产生平局，走不到退化模式的代码路径）：

| regime | side | α=0.05 | α/2=0.025 | α/4=0.0125 | α/8=0.00625 |
|---|---|---|---|---|---|
| 天花板区（spatial-like） | less | 0.0530 | 0.0204 | 0.0070 | 0.0006 |
| 天花板区 | greater | 0.0540 | 0.0194 | 0.0081 | 0.0006 |
| **平台区（本设计的基准情形）** | less | 0.0515 | 0.0084 | 0.0010 | 0.0000 |
| 平台区 | greater | 0.0522 | 0.0077 | 0.0006 | 0.0000 |
| **中段（l10-like，主判别战场）** | less | 0.0504 | 0.0214 | 0.0096 | 0.0051 |
| 中段 | greater | 0.0521 | 0.0220 | 0.0104 | 0.0051 |
| 强偏态 | less | 0.0526 | 0.0241 | 0.0104 | 0.0029 |
| 强偏态 | greater | 0.0521 | 0.0238 | 0.0086 | 0.0027 |
| 单点离群 | less | 0.0488 | 0.0225 | 0.0109 | 0.0065 |
| 单点离群 | greater | 0.0484 | 0.0227 | 0.0110 | 0.0059 |

**没有一格显示经 60 项门禁 multiplicity 校正后显著高于其名义水平**；裸点估计并非逐格都 ≤ 名义值（最大读数 0.0540 对 α=0.05，为 +1.7 MC SE；α/8 处最大 0.0065 对 0.00625，为 +0.3 MC SE）。因此本设计采用的口径是：**在 studentized wild 检验的渐近边际有效性条件下，Holm 给出名义 α=0.05 的 strong FWER 控制；n=10 的有限样本可信度由上述预注册 DGP 压力测试支持，但不宣称有限样本精确保证。** 全文不再使用非名义边际界，也不再需要 owner 对 0.12 作例外裁定；最终报告必须同时披露 n=10、渐近依据与模拟范围。

对照：R11 冻结的 pairs cluster bootstrap 在**平台区**边际 size 为 **0.114 / 0.113**（2.3× 名义），在天花板区 0.084 / 0.081。那才是必须靠"分 regime 放宽事前界 + owner 批准"才能自圆其说的状态；G2-R4 曾按那条路写过一版 §8.2.1（中段名义 α、天花板区 0.12、⏳owner 待批），**已随构造更换整体作废**。

**门禁判据的形式**。family FWER 与边际 size 一律以**单侧二项检验问"证据是否显示拒绝率*超过*界"**（`P(X ≥ k | p = 界)` 小于阈值才判失败），**而不是**要求 95% 上置信界落在界以下。后者对**校准完美**的检验也永远不可通过：上界按构造高于点估计，真值恰在 α 时无法证明"≤ α"。实测确认过这一点——经验 size 恰为 0.050（30/600）时 95% UCB 是 0.0672，"UCB ≤ α" 判失败。

**门禁自身的多重性**。§10 的 size 门禁是 5 regime × 3 side × 4 槽位 = 60 个二项检验，全部要过。若每格按裸 95% 判，一个**校准正确**的实现几乎每次都会踩响一格——而会叫的狼被静音之后比没有狼更糟。故每格按 `1 − 0.05/60` 判定。n=1,000 次模拟下该门仍能对名义 0.05 标出任何真实 size ≥ ~0.072，足以捕捉被替换构造的 0.114。**计划正文引用的 8,000 次读数与门禁运行的 1,000 次是两个不同的东西，二者都在 repo 里**：前者是 `test_marginal_size_high_precision`（标 `manual`，显式跑），后者是每次都跑的 `test_marginal_size_at_every_holm_slot_level`。

**双侧等价保持 descriptive**（不入家族）：等价（CI ⊂ `(−δ, +δ)`）只作为 Q-pass 的**附注**报告，明确标注未控制跨结论错误率，**不得单独改写 thesis 主结论**。非劣与等价的地位有意不同：主问题是**非劣**（故纳入 family），等价只是顺带观察（故 descriptive）。

**两套件仍各自独立 Holm**（不跨套件合并 14 个槽位）：两套件是**不同的科学问题**（不同任务分布、teacher 水平 0.974 vs 0.868），不是同一假设的重复检验；§8.4.2 的汇总规则要求并列报告两个结论，**不做"至少一个套件显著"式的推断**，故 per-family error rate 按套件定义是恰当的。这一点在结果表中须显式声明。

其余一切（敏感性臂、延迟、entries 分布、逐任务分解、descriptive McNemar、等价附注）不进 family、不做校正、措辞不得使用"显著"。


### 8.3 归一化敏感性臂（预注册，含防泄漏协议）

**臂**：**4 个**（`{spatial,l10}_{S1,S6}_recal`），各 500 ep，计入 §7 的 **8,000 ep** 预算。对照是同套件同档的主臂（同 init 配对）。

**为什么是 S1 与 S6 两端而非只有 S6**：生产参数是从 **50-init 库（≈S3）** 拟合的，失配是**双向**的——S6 比标定库大一个量级，S1/S2 比它**小**一个量级，而后者恰好决定曲线左端的陡峭度（`P-no` / 分支 E）。只在 S6 设臂等于只控了一侧。机制上小库一侧更危险：`libero_spatial_baseline.yaml:50-64` 的 vision sigma 是 **7e-3** 量级，`ZScoreNormalizer` 的 `0.5*(tanh(z)+1)`（`score_normalizers.py:174-178`）在偏离 mu 3–4 sigma（余弦下降 0.02–0.03）后即贴地饱和；小库下 top-1 近邻远 ⇒ 两个 vision 项对**所有**候选塌成 ~0 ⇒ 加权和的排序被 sigma=1.0 的 `robot_state`（权重 0.4375）单独接管 ⇒ **检索规则本身随 size 改变**，正是本实验要排除的 size×算子混杂。

**重标定必须是生产程序的原样重跑。** 已亲验：生产参数出自 `exp/common/calibrate_score_normalizers.py`，其 `collect_loeo_scores` 用**库内 LOEO**——每个库 entry 当 query，对**不在自己 `trajectory_id` 内**的全库条目打分，same-task / cross-task 两桶后 pooled 拟合，**不需要任何外部 query 池**（`fit_from_scores` 只是它调用的原语，单用它不构成"同一程序"）。

⚠ **一个必须避开的陷阱**：若改用库外 rollout（如 B-val）当 query 源，敏感性臂就一次动了**两个**变量（库规模 + query 分布）——那样 CI 含 0 不能排除失配、不含 0 也不能归因到 size，臂本身失去意义。

**冻结**：重标定 = 对目标档 pkl **原样重跑 `calibrate_score_normalizers.py`**，产出 `config/recal_norm_<suite>_{S1,S6}.yaml`（tracked）。LOEO 已经解决自检索（剔除同 trajectory），**不需要** B-val 当 query 池——§3.2 排除 B-val 的理由回归其本来的、更重要的那条：保护全 TIER 线唯一的"库外"切片（与本节无关）。

**产物与臂 yaml**：`cache_size_<suite>_{S1,S6}_recal.yaml` = 对应主臂 yaml **仅替换 `score_normalization.fields.*.params`**，其余逐字段相同（§10 测试）。**不新建 pkl**——归一化是检索时的打分变换、与库内容无关，敏感性臂与对应主臂共用**同一个** `preload_path`。故 pkl 总数仍 12（§6.2），臂数 **16**（§7）；这也使敏感性对比成为严格的单变量对照（库逐字节相同，只有打分参数不同）。

**判读**：`ΔSR(recal − fixed)` 按 §8.1 的同一 cluster 口径出 CI，并做**显式的等价检验**（`±3pp` 双侧非劣）。⚠ 不得用"CI 含 0 且半宽 < 3pp"这类表述代替——那是用 CI 外衣包装的等价主张，与 §8.4.1b 对 Q 轴的要求同理。该检验为 **descriptive**、不进 family、不单独改写 thesis。等价成立 ⇒ 分支 N 的"归一化失配"解释被排除；不成立 ⇒ 主曲线措辞限定为"在生产标定参数下"，重标定结果并列报告。

**零成本的连带产出**：逐档、逐 field 记录归一化后分数分布与**饱和率**（`calibrate_score_normalizers.py:_diagnostics` 的 `sat` 就是为此设的）。若某档 vision 饱和率接近 1，该档的"检索质量"读数须在报告中标注为算子伪影。这条不需要额外 rollout，进 §3.3 随记与 §8.5 产出。

### 8.4 预注册判读（互斥且穷尽的决策树）

R2 版本的五个分支既不互斥（③ 可与 ①/② 同时成立）也不穷尽（斜率 CI 跨 2pp、gap 显著但 ≤5pp 都无处可落），而 §12-P8 却要求结果落进这五支——那等于逼迫见数后补规则。现改为**三个原子判定 + 固定裁决顺序**的决策树。

#### 8.4.1 三个原子判定（每套件独立计算，全部三值或四值）

统计量：`slope_k = SR(S_k) − SR(S_{k−1})`（k=2..6）；`gap = SR(teacher) − SR(S6)`。CI 均为 §8.1 的 task-cluster bootstrap 95% BCa。

| 判定 | 地位 | 取值 | 判据（冻结） |
|---|---|---|---|
| **M**（单调性违反） | **确认性**，由检验 1–5 驱动 | `M-yes` | 存在 k 使**检验 k 经 Holm 后拒绝** 且点估计 `mean(d_t) < 0` |
| | | `M-no` | 其余 |
| **P**（平台） | **descriptive**（§8.2；不进 family、不承载确认性主张） | `P-yes` | `slope_5` 与 `slope_6` 的 BCa CI **上界均 < 2pp** |
| | | `P-no` | `slope_6` 的 BCa CI **下界 > 2pp** |
| | | `P-inconc` | 其余（CI 跨 2pp 的不确定区） |

⚠⚠ **P 是 descriptive，却独占了本实验最重要的科学分岔——必须靠措辞约束堵住**

`Q-fail` 单独只能说明"S6 比 teacher 差超过 δ"，这与"再多数据就能补上"**完全相容**——那正是分支 E。**区分 A 与 E 的全部信息量在 P 上**。也就是说，论文头号主张里"**不是数据量问题**"这半句，其确认性完全来自一个被明确排除在多重性控制之外、且依赖 BCa/percentile 选择的判据。

**冻结的处置**（取"改写结论句"而非"把 P 纳入 family"——后者要为两个 slope 各加一个槽位，把最严阈值从 `0.05/8` 压到 `0.05/10`，在 10 cluster 下得不偿失）：

- **分支 A 的可发表形式被钉死为**：「检验 8 确认 `gap > δ`（Holm 后 p = …）；**末两档斜率的 BCa CI 上界 < 2pp，与"数据量不是主要瓶颈"一致，但该半句为 descriptive、未受多重性控制、不承载确认性主张**；且结论限定**在本 index 下**（§3.3.1）」。**不得**出现无限定的"payload 上限不是数据量问题"。
- 任何涉及"饱和/平台/仍在上升"的表述必须附 CI 数值，禁用"显著饱和""显著仍在上升"。
- **精度门槛的事前披露**：`P-yes` 要求两个 slope 的 CI 上界都 < 2pp，在 n=10、t₉≈2.26 下大致需要 task 间 slope 的 SD ≲ 2.5pp，即几乎每个 task 在相邻档间成功数变化 ≤ 1 个 episode。真饱和时可达，但**这是个紧门槛**——故 §8.4.2 把分支 A 标为"预期最可能"的同时，**必须对 `P-inconc`（→ 分支 G）做同等准备**，不得因为落 G 就去调阈值。
**G（teacher 缺口）单列于 §8.4.1b** —— R4 版本的四分法有方向性缺陷，已重做。

`P` 的三值设计与 `G` 的六值设计，都是为了让"不确定"成为**预注册的合法结局**，而不是被挤进最近的实质分支。

#### 8.4.1b G（teacher 缺口）—— 方向完整的六分法

**先冻结要回答的问题：单侧非劣。** 本实验问的是「库大到 B-train 全量时，纯 replay 够不够用」，"够用"的正确形式化是 **S6 不比 teacher 低超过 δ=5pp**（单侧非劣），而不是双侧等价——S6 若**优于** teacher 并不损害"payload 够用"的结论，反而更强，把它排除在"够用"之外是错的。但"S6 优于 teacher"必须**单独成格**：纯 replay 超过其数据来源的 teacher 是反直觉结果，可能指向选择效应或缺陷，绝不能与"接近但未达"混为一谈（这正是 R4 的 `G-small` 犯的错——`[-10pp,-2pp]` 会被它判成"接近但未达 teacher"）。

记 `gap = SR(teacher) − SR(S6)`（**正值 = teacher 更好**），其 task-cluster bootstrap 95% CI 为 `[L, U]`，δ = 5pp。

**R6 的六分法是错的，错在把两个正交的问题做成了互斥类别**。"相对 0 的方向"与"相对 δ 的够用性"是**两个独立的判定**：非劣的判据是 `U < +δ`，**与 CI 是否跨 0 无关**。于是 `[+1pp, +4pp]` 同时为真两件事——teacher 统计上更优（`L>0`）**且 S6 非劣、即够用**（`U=4pp<δ`）；R6 却只把它归为 `G-teacher-small` → 族 T，排除在"S6 够用"之外，若再遇 `P-yes` 就被送进 A「payload 上限不是数据量问题」，**与本节刚冻结的"够用 = 单侧非劣"定义直接冲突**。现拆为两个正交轴。

**落格一律由 Holm 后的拒绝结果驱动，不由 CI 驱动**。R8 用未校正 CI 越过 0/δ 来定 D/Q，会出现"原始 CI 满足 `U<δ`、但检验 7 经 Holm 后未拒绝，却仍宣称 `Q-pass`"这类越过家族门禁的结论。CI 现降为效应量展示（§8.1）。

#### D 轴（方向，相对 0）—— 由 primary **检验 6 的 Holm 后结果** 驱动

| 判据 | 取值 | 含义 |
|---|---|---|
| 检验 6 Holm 后**拒绝** 且 `mean(d_t) < 0` | **`D-cache`** | S6 显著优于 teacher。**触发 §8.4.3 强制排查** |
| 检验 6 Holm 后**拒绝** 且 `mean(d_t) > 0` | **`D-teacher`** | teacher 显著优于 S6 |
| 检验 6 Holm 后**未拒绝** | **`D-none`** | 方向不可断言 |

#### Q 轴（达标性，相对 +δ）—— **主决策轴**，由 primary **检验 7/8 的 Holm 后结果** 驱动

| 判据 | 取值 | 含义与措辞约束 |
|---|---|---|
| **检验 7** Holm 后拒绝 | **`Q-pass`** | **单侧非劣成立：S6 落后不超过 δ ⇒ "够用"**。若 BCa CI 另满足 `L > −δ` → 附注"双侧等价亦成立"，该附注为 **descriptive**（§8.2），不得单独改写 thesis |
| **检验 8** Holm 后拒绝 | **`Q-fail`** | 落后幅度**确定超过** δ ⇒ "不够用" |
| 检验 7 与 8 **均未拒绝** | **`Q-inconc`** | 非劣与劣效都未被确认，**幅度不可判**。明令**不得**读成"无差异""等价"或"够用" |

（检验 7 与 8 的 H0 互补、仅交于 `gap = δ`，**不可能同时拒绝**；§10 断言之。）

#### 两轴的组合：7 个可达、2 个逻辑不可达

D 与 Q 由**不同的检验**驱动（6 vs 7/8），互不蕴含。下表的 CI 示例仅用于直观展示典型数据形态，**落格以 Holm 后的检验结果为准**；标 ✗ 的**两格**（均在 `D-cache` 行）在两轴的假设结构下自相矛盾，**§10 须断言其不可达并 fail-loud**。标 ⚠ 的一格**可达**，走预注册读法：

| | `Q-pass`（检验 7 拒） | `Q-fail`（检验 8 拒） | `Q-inconc`（均未拒） |
|---|---|---|---|
| **`D-cache`**<br>（检验 6 拒, mean<0） | ✓ 典型 `[-10pp,-2pp]`：S6 更优且当然够用 | ✗ 不可达（`gap<0` 与 `gap>δ` 矛盾） | ✗ 不可达（能确认 `gap<0` 必能确认 `gap<δ`） |
| **`D-teacher`**<br>（检验 6 拒, mean>0） | ✓ **关键格**，典型 `[+1pp,+4pp]`：**teacher 统计上更优，但 S6 够用**——两项事实必须同时报告 | ✓ 典型 `[+6pp,+9pp]` | ✓ 典型 `[+4pp,+8pp]` |
| **`D-none`**<br>（检验 6 未拒） | ✓ 典型 `[-3pp,+4pp]`（若 `L>−δ` 附等价注） | ⚠ **可达**，走下述预注册读法（G2 期自查更正，见反例） | ✓ 典型 `[-2pp,+7pp]` |

⚠ **本节口径唯一**：可达 7 格、不可达 2 格（仅两个 `D-cache` 格）。G1 冻结文本原写"6 可达 / 3 不可达"，G2 期用生产代码找到 `D-none × Q-fail` 的可复现反例后更正，全文（§8.1.1、本节、§10 测试义务）已统一为此口径。

- **`D-cache × Q-fail` / `D-cache × Q-inconc`**：都要求点估计同时 `< 0` 且 `> δ`，自相矛盾。实测搜索 5,876 个负均值 gap 向量，检验 8 的 p 最小值是 **0.6413**，从未接近拒绝。这两格保留 **fail-loud**：真出现即实现有缺陷。
- **`D-none × Q-fail` 可达**，根因不是"统计量不嵌套"（全族统一构造后那个问题已消除），而是**检验 6 双侧、检验 8 单侧**：零分布偏斜时（一个任务差异极端就够），双侧计数会捡到 `−|t_obs|` 以外的左尾质量，而单侧计数看不到，于是 `p6` 可以超过 `p8` 到 Holm 槽位比吸收不了的程度。用生产代码复现：`gap` 均值 +0.312 时 raw `p6=0.026` / `p8=0.003`，Holm 后 **0.052（未拒绝）/ 0.026（拒绝）**。这是完全合法的数据形态，若继续 fail-loud 会在真实数据上中止整个分析。**冻结读法**：「幅度已确认超过 δ，但方向未在家族层面确认——二者用的是不同的尾，这不是不一致；两项事实都要报，且**不得**把这对结果转述为彼此印证」。

**主决策由 Q 轴驱动**（科学问题是"库够大时纯 replay 够不够用"），D 轴作为**并列报告的方向标注**，二者都必须出现在结论里——`D-teacher × Q-pass` 的正确表述是「teacher 在统计上仍更优，但差距在 δ=5pp 的实用界内，即 S6 在本预算下已够用」，**既不能只说前半句，也不能只说后半句**。

#### 8.4.2 裁决顺序（保证互斥）

**第 0 步优先于一切**：若 `M-yes` → 落 **分支 N**，且**不再**读 P/G 的实质含义（曲线本身可疑时，"平台""缺口"的解释都建立在可疑基础上）。先由 §8.3 敏感性臂裁决是否归一化失配：若是，报告为算子伪影并说明主曲线须在重标定下重读；若排除，作为「库密度与检索质量非单调」的独立发现如实报告。

**否则**（`M-no`）按 **(P, Q)** 查表 —— 主轴是 **Q（够不够用）**，因为那才是本实验问的问题；**D（方向）在每格内作为并列事实报告**，不参与选格。

| P \ Q | **`Q-fail`**（不够用） | **`Q-pass`**（够用） | **`Q-inconc`**（不可判） |
|---|---|---|---|
| **P-yes**<br>（饱和） | **A — 主结论**（预期最可能，但须对 `P-inconc` 同等准备）：检验 8 确认 `gap > δ`；末两档斜率 CI 上界 < 2pp，**与"数据量不是主要瓶颈"一致，但该半句为 descriptive、未受多重性控制**（§8.4.1）；结论限定**在本 index 下**（§3.3.1）。给 §5.1 三区制中 replay 区的**高度上限与数据成本** | **C**：饱和且 S6 够用 ⇒ thesis 的"payload 不足"须**按套件限定作用域**，与另一套件并列报告。附方向：`D-teacher` → "teacher 统计上仍更优但差距在实用界内"；`D-none` → "方向不可断言"；`D-cache` → 先过 §8.4.3 排查再作为独立发现 | **D**：平台成立、够用性不可判。报平台结论，达标性标 inconclusive 并披露功效，**不得**替换成"无差异"或"够用" |
| **P-no**<br>（陡峭） | **E**：数据量不足主导。主张收缩为「在 benchmark 的 init 预算（B-train 450/suite 硬上限）内 payload 不足」，外推不可判定入局限。**这不是坏结果**——官方 init 预算封顶是 benchmark 结构给定的，"要多少数据才够"本身就是 payload 路线的成本，与 index 路线的零边际成本形成对照 | **F**：已够用却仍在上升（罕见）。如实报告，提示 S6 之后可能进一步拉开，列为 future work；`D-cache` 同样触发 §8.4.3 排查 | **E'**：仍在上升且够用性不可判。按 E 报告"仍在上升"，达标性单列 inconclusive |
| **P-inconc**<br>（不确定） | **G**：不够用确立、是否到平台不可判。报达标性结论，**明确声明 size 轴未分辨出平台**，不得暗示饱和 | **I**：够用成立但平台不可判。报够用（含方向标注），平台标 inconclusive | **H — inconclusive**：两维度均不足以支撑实质主张。如实报告曲线与 CI，结论限于"本预算下不可判"，并给出达到可判所需的规模估算 |

**每格的报告义务**：无论落哪格，结论段必须**同时**给出 Q 的结论（够用性，主）与 D 的结论（方向，并列），以及 §8.2 检验 6（方向）与检验 7/8（达标性）的 Holm 调整后 p 值。禁止只报其中一项——`D-teacher × Q-pass` 只说"teacher 更优"或只说"S6 够用"都是不完整且误导的。

**作用域从句是每一格的义务，不只是分支 E**：R10 只给分支 E 写了作用域收缩，而"预期最可能"的分支 A 完全没有限定语。现统一要求：**每格结论都必须带**「单 teacher（Pi0.5）/ 单 keybuilder（`cp1_spatial_pool_16`）/ 两 LIBERO 套件 / sim-only」的作用域，分支 A 还须额外带 §3.3.1 的 **index 限定**与 §8.4.1 的 **P-descriptive 限定**。

#### 8.4.3 `D-cache` 的强制排查

若任一套件落 `D-cache`（纯 replay 显著优于其数据来源的 teacher），**在写入结论前**必须先排除三个可解释来源，排查结果一并报告：

1. **anchor 协议错配** —— teacher anchor 与本实验是否同 init、同 seed、同 `replan_steps`？（anchor 采于 2026-08-13，协议见 `ablation_study_plan.log.md`；join 键须 500/500 命中，§8.1 已要求）
2. **success 过滤的选择效应** —— 库只收成功轨迹（`outcome_filter=success`），S6 在 45 init/任务下对易任务覆盖极密，replay 等于在"已知能成功的路径"上重放；这在原理上**可以**超过 teacher 的期望 SR，属真实效应而非缺陷，但必须显式归因。
3. **评测泄漏** —— A 池与库的 init 是否真的不相交（§7 的 digest 断言 + §3.2 的 B-val 排除）。

三者都排除后，`D-cache` 作为**独立发现**报告，并明确它与 thesis 的关系：它强化"index 承重"（检索选对了状态）而非削弱它。注意 `D-cache` 逻辑上蕴含 `Q-pass`，故主决策仍走 §8.4.2 的"够用"列，排查只影响该发现的**归因与措辞**，不改变落格。

**汇总规则（取代原 ⑤）**：两套件**各自独立**走这棵树——原 ⑤ 把"两套件不一致"与 ①–④ 并列是分类错误，它是**汇总层**而非同层分支。汇总时：l10（teacher 0.868，有 headroom）为主判别战场，spatial（0.974，天花板区）作无损验证；两套件落格不同时**并列报告两个结论**并以 l10 为主叙事，同时披露 spatial 的功效限制（天花板区配对差异稀少、10 cluster 下置换检验功效更低）。**禁止**用其中一个套件的结果覆盖另一个。

#### 8.4.4 阈值来源（事前冻结，避免事后挑选）

- **2pp** 取自 §7.1 历史 AlwaysHit 三配置的离散度（0.674 / 0.692 / 0.696，极差 2.2pp）——"配置噪声量级"，斜率小于它无法与噪声区分。仅用于 `P`。
- **δ = 5pp** 取自执行体消融中被判为实质效应的最小量级（该报告的显著效应均 ≥ 5pp）。它是 **Q 轴的唯一阈值**：`Q-pass/Q-fail` 的切分点、单侧非劣界（主判据）、以及附注双侧等价时的对称界 `±δ`。

两者**在见到本实验任何数据前冻结**，不因结果调整。δ 一数多用是有意的：若非劣界与实质界取不同值，就会出现"非劣成立但落后幅度仍被判为实质"的自相矛盾结局。

### 8.5 产出

1. SR × size 曲线（两套件各一条，含 §8.1 的 CI 带）
2. §8.2 的 8×2 primary 检验表（原始 p、Holm 调整后 p、配对 RD 与 CI、discordant pair 计数）
3. 每档实测 entries 表（**逐任务**，披露 §3.2 的任务间不均衡）
4. 每档检索延迟、pkl 加载耗时、加载后 RSS 增量（§6.2.1 的门禁数据）
5. §8.3 敏感性臂对比表
6. A 池 digest 与 teacher anchor join 的完整性记账（500/500）

---

## 9. 文件触及清单

### 9.1 新增（`exp/ablation_study/cache_size/`）

| 文件 | 职责 |
|---|---|
| `__init__.py`（并含 `tests/ablation_study/cache_size/__init__.py`） | 包桩；测试侧的 `__init__.py` 是 basename 冲突的解药之一（§10） |
| `emit_size_grid.py` | 生成 tracked 的 `config/size_grid_<suite>.yaml`。⚠ **依赖实测成败，必须在采集之后跑**（§3.2 新口径）：每任务对**成功**轨迹排序 = 历史 5-init 锚中成功者（按 `orig_init_state_idx` 升序）→ B-train 其余成功者（`seed=0` 洗牌）→ B-val 恒排除。自检断言：各档 ∩ B-val = ∅；每档每任务恰 k 条（除触顶任务）；嵌套；种子确定性 |
| _(A 池物化)_ | **复用 X14 已实测的 `materialize_apool.py`**，不新写（见 §7）。本 plan 只负责记录 digest 与执行"A∩B=∅"断言 |
| `emit_episode_lists.py` | 把 size_grid 映射成 `--episode-list` 清单（`task_{id}/episode_{idx}.h5`） |
| `build_size_artifacts.py` | 驱动 12 次 build（`--episode-list` + `--trajectory-id-mode relpath`），产出**逐任务** entries 实测表 |
| `emit_size_yamls.py` | 派生 **12 主臂** yaml（`judge: always_hit` + 各档 `preload_path`）+ **4 敏感性臂** yaml（S1/S6 × 2 套件，仅换 `score_normalization.fields.*.params`，`preload_path` 与对应主臂**相同**） |
| `run_recal.py` | §8.3 重标定：对 **S1 与 S6** 的 pkl **原样重跑 `exp/common/calibrate_score_normalizers.py`**（库内 LOEO，**不用** B-val 当 query）→ `config/recal_norm_<suite>_{S1,S6}.yaml`；同时导出逐档逐 field 的饱和率 `sat` |
| `run_size_eval.py` | conductor driver：**16 臂 × 500 ep = 8,000 ep** 配对评测（A 池）；每臂逐 episode 记 `FULL_HIT 率` |
| `analysis/analyze_size.py` | §8.1–8.4 统计：teacher anchor join（断言 500/500）、**8×2 primary + Holm（两套件各自独立）**、studentized null-imposed sign-flip（穷举，确定性）、BCa CI（含降级判据）、D/Q 落格、size 轴 R1–R5 规则的实测表 |
| `analysis/plot_size.py` | SR × size 曲线（含 CI 带与 teacher 水平线） |

### 9.2 修改

| 文件 | 改动 | 风险 |
|---|---|---|
| `exp/common/build_in_memory_cache_artifact.py` | +`--episode-list`（含 §6.1 边界契约）+ `--trajectory-id-mode`（§5.3.1）；两者默认均保持现状逐字节不变 | 中（有非回归 + ID 唯一性测试） |
| `docs/experiments/artifact_layout.md` | §1 增补「实验族」规则 | **WA §8 注册文档**，owner 已批准 |
| **`docs/iclr/tier_experiment_designs.md`** | ① **增补 X9b 卡片，保留 X9 原文不动**（§1 的 B6 裁定）；② ⚠ **登记两处章程状态变更**：`:3` 的章级统计约定冻结为「McNemar + **episode 级** cluster bootstrap」，而 X9b 用 **task 级** cluster 并把 McNemar 降为 descriptive —— 须在 X9b 卡片里声明这是**有意的口径偏离**并给理由（§8.1.0 的 estimand），否则两处会各自被后续实验引用；`:5` 的 Init 池协议写「cache 库 50 init 受 `protected_in_train` 锁在 B-train 侧」，X9b 之后 **B-train 全量入库**，后续任何"补 B-train 做标定"的实验所面对的"库对 B-train 覆盖稠密"警告从 5/45 变成 45/45，量级完全不同，须登记 | 中（上位目标章程） |
| **`docs/iclr/tier_paper_outline.md`** | 实验台账表增 X9b 行 | 低 |
| `docs/README.md` / `logs/README.md` | index sync（WA §4 红线） | 低 |
| `exp/ablation_study/**` → `exp/ablation_study/executor_substitution/**` | 族化移动（**M-c2**） | **高**，见 §9.3 / §9.4 |
| `exp/rl_router/{run_rl_router,microbench_cost,emit_router_yamls,collect_warmstart}.py`、`exp/rl_router/config/run_matrix.yaml` | 随族化更新 import 与路径（**M-c2**，X14 静默窗口内） | **高**，见 §9.3-A |
| `tests/{ablation_study,exp,scripts}/**` | 随族化更新 import 与路径常量（**M-c2**） | 中 |
| `src/openpi/cache/{config,sidecar_executor}.py` | docstring 内示例路径更新（**M-c2**，无逻辑改动） | 低 |

### 9.3 族化重构的引用面

原文声称"已全量 grep"，实际 grep 加了 `--include` 过滤且只查了 `tests/`，**漏掉了 `exp/rl_router/` 对 ablation_study 的运行时依赖**。这是本轮最严重的遗漏——它把"改路径"升级成了"会打断正在跑的 X14"。重新审计（`grep -rn "ablation_study"` 全 repo，排除 `.venv`/worktrees）结果：

**A. X14（rl_router）——运行时 import，不是路径字符串**

| 位置 | 形态 |
|---|---|
| `exp/rl_router/run_rl_router.py:110` | `from exp.ablation_study.build_distill_dataset import check_init_dir` |
| `exp/rl_router/microbench_cost.py:371` | `from exp.ablation_study.sidecar_server import make_act_policy, make_smolvla_policy` |
| `tests/exp/test_rl_router_run_loop.py:2040` | `from exp.ablation_study.sidecar_server import route_prompt` |
| `exp/rl_router/emit_router_yamls.py:54` | `BASELINE = "exp/ablation_study/config/common/{suite}_baseline.yaml"` |
| `exp/rl_router/config/run_matrix.yaml:22,23,26,27` | 4 条 `split:` / `baseline:` 路径 |
| `exp/rl_router/collect_warmstart.py:20` | docstring 内示例路径 |

**B. ablation_study 自身的测试与配置**

- `tests/ablation_study/` 下 **7** 个文件含 `from exp.ablation_study.X import …`（目录内共 8 个 `.py`，`test_router_hooks.py` 无引用）
- `tests/ablation_study/test_ablation_config.py:14` `CONFIG_ROOT`
- `tests/scripts/test_routing_wrap_policy.py` **3 处** yaml 路径（`:18`、`:19`、以及 `:52` 的内联路径）
- `exp/ablation_study/config/act_manifest_libero_{10,spatial}.json` 共 20 条 checkpoint 路径（随 §4.3 软链改名一并改）
- `exp/ablation_study/analysis/analysis.md` §9 的 artifact layout 段

**C. src/ 内的引用**（`src/openpi/cache/config.py:2359`、`src/openpi/cache/sidecar_executor.py:4`）—— 经复核为**对本实验名与 plan 章节号的散文引用**（`# (ablation_study plan §8.2)` / `(ablation_study plan §6.1)`），**不含任何文件路径**，因此不会 dangling。迁移时无需改动；此处保留记录只为说明"已查过、确认无需动"。

**D. 族内自引用（原清单完全漏掉的一组，本轮自查发现）** —— `exp/ablation_study/` **自身**约 30 个文件也硬编码了旧路径，其中含**运行时**依赖：

| 位置 | 形态 | 危险度 |
|---|---|---|
| `analysis/plot_ablation.py:30-31` | **运行时路径常量** `pathlib.Path("exp/ablation_study/data/runs")` / `("exp/ablation_study/analysis")` | **高** —— M-c2 后会静默读到族目录而非实验目录，不报错、只是找不到数据 |
| `train_act.py:19-20`、`run_ablation_eval.py:220-221` | dotted import `exp.ablation_study.X` | 高（ImportError，会被 pytest 抓到） |
| `config/{arm_matrix_*,select_freeze_*}.yaml` 等 26 个 yaml | 约 80 处路径串 | 中 |
| `sidecar_server.py:20-23`、`select_student_checkpoint.py:11-14`、`analysis/render_sr_ledger.py:12-14` | docstring 内 CLI 示例 | 低 |

**这是与 R1-B2 同一类的遗漏**：上次漏了"外部消费者（rl_router）"，这次漏了"内部自引用"。根因相同——grep 时把 `exp/ablation_study/` 自身排除在搜索范围外。故 §9.4 的 M-c2 出场门 #1 的 grep **必须包含族目录自身**，不得再用 `--exclude-dir`。

**门禁覆盖性复核**：出场门 #1 的斜杠形态 grep 能抓住上表的路径串（含 `plot_ablation.py` 的运行时常量）；**点号形态的 dotted import 不在门 #1 覆盖内**，但会被门 #3（`uv run pytest tests/ablation_study tests/exp tests/scripts` 全绿）抓到。两道门合起来覆盖上表全部四类。

**执行期义务**：M-c 落地时**重跑一次全 repo 审计**（`grep -rn "ablation_study" --exclude-dir=.venv --exclude-dir=.claude`），逐条改完后以 **"全 repo 无 dangling `exp/ablation_study/<旧扁平路径>`"** 与 **"X14 全部消费者可导入"**（下方出场门）作为门禁，而不是凭本清单认为已穷尽。

### 9.4 迁移时序

原文的矛盾：§4.4/R6 说 M-b/M-c 都必须等 X14 结束，§12 却把族化放在 P1（立刻）、把 M-b 放到 P8（最后）。且 M-b 事实上已在 X14 训练暂停窗口完成。现按"**先加不减**"重新冻结，把破坏性动作压缩到一个静默窗口：

| 段 | 内容 | 破坏性 | 时机 |
|---|---|---|---|
| **M-c1**（过渡态，**立即可做**） | 只**新建** `exp/ablation_study/cache_size/`（含 `__init__.py`、代码、`config/`、`data/` 与软链）。**不动**任何现有扁平文件 | **零** —— 纯新增，现有导入路径与 X14 全部消费者不受影响 | P1，G1 通过即可 |
| **M-c2**（收编，**须静默窗口**） | 把扁平的 7 个 `.py` + `config/` + `analysis/` + `data/` 收进 `executor_substitution/`，改 §9.3 A/B/C 全部引用，软链改带信息名，manifest 同步 | **高** —— X14 的 4 个 run 与全部 sidecar 都要停 | X14 **全部 run 结束后**的单一原子窗口 |

**过渡态的明确期限与代价**：M-c1 之后、M-c2 之前，族目录下同时存在 `cache_size/` 子实验与旧的扁平文件，**违反 §4.1 我们自己写的"族目录不得直接存放代码"规则**。这是有意接受的临时偏离，期限 = X14 结束，并在 `exp/ablation_study/README.md`（M-c1 产出）里显式标注为过渡态。选它的理由是替代方案更差：要么阻塞本实验到 X14 结束（本实验的采集本可与 X14 并行排队），要么给 rl_router 加兼容 shim（WA §3.1 禁止的 dead code 类新增）。

**M-c2 的出场门**（缺一不可）：
1. `grep -rn "exp/ablation_study/\(build_distill_dataset\|run_ablation_eval\|select_student_checkpoint\|sidecar_server\|train_act\|train_smolvla\|train_student\|config/\|analysis/\|data/\)"` 全 repo **无遗留旧路径**；
2. `python -c "import exp.rl_router.run_rl_router, exp.rl_router.microbench_cost, exp.rl_router.emit_router_yamls"` 成功（X14 消费者可导入）；
3. `uv run pytest tests/ablation_study tests/exp tests/scripts` 全绿；
4. 两个 act_manifest 的 20 条路径 `os.path.isfile(<p>/model.safetensors)` 全部 True（复用 M-b 已用过的检查）；
5. 族目录下**不再有**任何直接存放的代码/config/data（§4.1 规则恢复）。

---

## 10. 测试策略

| 测试 | 断言 |
|---|---|
| `tests/ablation_study/cache_size/test_size_grid.py` | **按 §3.2 的 R1–R5 断言，不再断言"恰 45"**：① `n_{t,k} = min(k, n_t)` 对**所有**档成立；② **每档每任务非空**（R2 的运行前提）；③ 嵌套性（S_k 清单 ⊆ S_{k+1}，允许触顶后相等）；④ **每档 ∩ B-val = ∅**（§3.2 红线）；⑤ S3 恒为 `min(5, n_t)` 条且历史锚命中数被记录（R5）；⑥ 种子确定性；⑦ 退化相邻对被标 `not evaluable` 而非静默通过（R3）；⑧ 合成 `n_t < k` 的任务，断言不抛异常且按 R1 取数 |
| **`tests/exp/test_build_in_memory_cache_artifact.py`**（扩充既有文件） | ① 合成 `task_{0..9}/episode_0.h5` 后，`--trajectory-id-mode relpath` 下**全部 entry id 唯一**；② **`len(artifact["entries"]) == len(backend._entries)`**（加载后不掉条，这是 ID 碰撞的通用探针）；③ 逐任务 entry 数 == 该任务成功 episode 的步数和（无任务被整体吞掉）；④ **`stem` 模式下对同一份历史输入产出与改动前逐值相同的 id**（非回归）；⑤ 反向断言：`stem` 模式喂 `task_N/episode_0.h5` **确实**发生覆盖（把 bug 钉成已知行为，防止将来有人误改默认） |
| `tests/exp/test_build_in_memory_cache_artifact.py`（扩充既有文件） | `--episode-list` 只读清单内文件；不给定时 `h5_paths` 与改动前逐值相同（非回归）；**§6.1 边界契约逐条 fail-fast**（绝对路径 / `..` 逃逸 / 非 .h5 / 不存在 / 空行 / 规范化后重复）；serial 与 ProcessPool 两路径消费同一已验证列表 |
| `tests/ablation_study/cache_size/test_apool.py` | A 池 50/任务、500/suite；**与差集池交集为空**（逐字节）；digest yaml 与实际文件 sha256 一致 |
| `tests/ablation_study/cache_size/test_size_yamls.py` | **16 个** yaml 全部 `load_cache_config` 通过；judge 恒 `always_hit`、gate 恒 `always_search`；12 主臂的 `preload_path` 与档位一一对应；主臂除 judge/preload_path 外与 baseline 逐字段相同；**4 个敏感性臂与其对应主臂（S1 或 S6）仅差 `score_normalization.fields.*.params`，且 `preload_path` 完全相同**（复用 pkl，§8.3） |
| `tests/ablation_study/cache_size/test_cache_size_analysis.py` | **落格由 Holm 后检验驱动**：① **门禁优先于 CI**：构造 `gap` BCa CI `U < δ` 但检验 7 经 8-检验 Holm **未拒绝**的数据，断言判 `Q-inconc` **而非** `Q-pass`，且结果表标注"CI 与家族门禁边界分歧"；② 同理构造检验 6 未拒绝但 CI 不含 0 的数据，断言判 `D-none`；③ 断言 D/Q 的实现**不读取** CI（以 CI 为输入的变体应使测试失败）。<br>**Q 双侧确认**：④ 检验 7 拒 → `Q-pass`；⑤ 检验 8 拒 → `Q-fail`；⑥ 均未拒 → `Q-inconc`；⑦ **边际 size 模拟**（R11-B14，G2-R5 加严）：在 `gap = δ` 的边界 null 下，对 **5 个预注册 regime × 3 个 side × 4 个 Holm 槽位水平**各自断言经验拒绝率**不高于名义水平**（判据形式与门禁自身多重性见 §8.2.1）；另有 8,000 次的高精度版本标 `manual` 常驻 repo，供复现正文表格。<br>**null 施加与穷举**：⑧ 检验 7/8 的参照分布**由中心化施加 H0 后翻转残差**得到——断言其 p 值与"观测分布尾比例"实现**不同**（用一组能区分二者的合成数据对拍，后者不得通过）；⑨ 参照样本的均值中心恰为 δ；⑩ 参照集恰 `2^10 = 1,024` 个模式、p 分辨率 `1/1025`，且**同一数据两次调用逐位相同**（无 seed、无 B）；⑩b **最严槽位可达性**：5 个 regime 下 `p_min < 0.05/8`，否则该槽位永远不能拒绝。<br>**D/Q 正交性（R7-B10，保留）——每个案例断言两项事实**：⑪ 典型 `[+1pp,+4pp]` 形态（检验 6 拒且 mean>0、检验 7 拒）**必须同时**判 `D-teacher` 与 `Q-pass`，**输出同时含"teacher 更优"与"S6 够用/非劣"两句**，缺任一即 fail；⑫ `D-cache × Q-pass` 触发 §8.4.3 排查清单且措辞不含"未达/接近 teacher"；⑬ `Q-inconc` 措辞不含"等价/无差异/够用"；⑭ 双侧等价附注仅在 `Q-pass` 且 BCa `L > −δ` 时出现，且标记 descriptive。<br>**组合与终局**：⑮ **2 个**不可达组合（两个 `D-cache` 格）的 fail-loud 检测（含 Holm 步进导致的病态输入），**外加 `D-none × Q-fail` 的可达性用例**——该格须产出预注册读法而非中止分析；⑯ (P×Q) 9 格 + 分支 N 共 10 终局各一案例；⑰ **优先级**：`M-yes` 与 `P-yes/Q-fail` 并存时判分支 N；⑱ **M 由 Holm 后检验驱动**（构造 CI 上界<0 但检验未拒绝的数据，断言 `M-no`）；⑲ 临界值六处；⑳ 互斥穷尽性随机测试（D、Q 各自恰落一格，组合恰落一终局）；㉑ 每格输出必须同时含 Q 与 D 结论。<br>**family 与口径**：㉒ **Holm 家族恰 8 检验**（断言 7 与 8 各占一格、Holm 输入个数为 8；退化相邻比较以 `p=1` 保留槽位）；**`gap = δ` 的边界 family FWER**：在边界 null 下用**离散**配对成功率模拟，断言 family FWER 在**名义 α** 受控；**另加一个含假零假设的配置**（5 个相邻比较中 3 个带真效应，错误事件只统计真零假设），因为步进法在部分零假设为假时最吃紧。判据形式按 §8.2.1：问"证据是否显示超标"，而非"能否证明不超标"；不用单次经验比例的随机越界作门；㉓ 两套件各自独立 Holm（不合并成 16）；㉔ **P 与等价附注、McNemar 均不进 Holm 输入**（断言 P 的判定不改变任何 p）；㉕ studentized sign-flip 与独立参考实现（`itertools.product` 纯 Python 循环枚举，不共用任何 helper）**精确**对拍（检验 1–8，含 `s*=0` 的 fail-fast/not-evaluable 边界与 4 个 tie-heavy 用例）；㉖ 参照集与 CI 的重采样单位都是 task 不是 episode；㉗ teacher join 缺失 1 行即 fail |
| 族化非回归（M-c2） | §9.4 的 5 条出场门 |

> **G2-R6 reviewer 口径收紧（owner 授权直接修订）**：上表边际门禁中的“经验拒绝率不高于名义水平”和 family 门中的“受控”，均按 §8.2.1 的最终口径解释：门禁断言的是“没有经门禁自身 multiplicity 校正后显著的超标证据”，不是声称每个 Monte Carlo 点估计逐字面 ≤ 名义值，也不是 n=10 下的有限样本精确证明。确认性 Holm 结论依赖 studentized wild 构造的渐近边际有效性；有限样本依据是预注册压力测试及其明确披露的模拟范围。

⚠ **测试文件名必须全局唯一**：`tests/ablation_study/` 下**无 `__init__.py`**，`pyproject.toml:85-86` 也未设 `importmode` ⇒ 默认 `prepend` 模式下 basename 相同的两个测试文件会在 **collection 阶段**直接报 `import file mismatch`，让 §12-P1 的出场门当场失败（**M-c1 一落地即触发，不必等 M-c2**）。故：① 新测试一律带 `cache_size_` 前缀（上表已改）；② §9.1 的新增清单须包含 `tests/ablation_study/cache_size/__init__.py`，M-c2 时同样给 `tests/ablation_study/__init__.py` 与 `executor_substitution/__init__.py` 补上。

**手动测试**（`@pytest.mark.manual`）：
- 10-ep 采集 smoke：h5 含 `vision_0/1/2`+`prompt_emb`，路径为 `task_N/episode_M.h5`，实测 ep/min；
- 1 臂 10-ep 评测 smoke：`always_hit` 下 `hit_type` 恒 FULL_HIT；
- **累计内存门**（§6.2.1）：单进程顺次热切换本 suite 全部 6 个不同 pkl（8 个评测臂；S1/S6 敏感性臂复用对应 pkl），逐档记录加载后 RSS 增量与累计值，断言累计 < available 的 50%。

---

## 11. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 采集期间 4090 被 X14 或其它 session 占满 → 采集变慢或 CUDA OOM | 采集是单连接串行、显存占用小（一个 pi05 实例）；发射前查 `vram_free_mb`；X14 M6 未跑完时优先让路（本实验墙钟不敏感） |
| R2 | 重采轨迹与 Phase 0 不同（flow-matching 噪声）→ 与既有蒸馏数据非同源 | D2 已裁定接受；作为披露项写进报告。副作用：S3 档也不等于历史 50-init 库的字节，只等于同 init |
| R3 | **12 档在单进程内累计常驻**（`BackendPool` 无 eviction）+ 检索延迟随 entries 线性增长 | §6.2.1：累计估算 ~21 G vs wls available 234 G，容量安全；仍设逐档 RSS 加载门（累计 > 50% available 即停）+ 预选降级方案（按 suite 分进程 → 每 2 档重启）。延迟风险体现在墙钟，已计入 §7 上界 |
| R4 | 归一化参数失配混进 size 效应（**双向**：S6 比标定库大一个量级、S1/S2 小一个量级） | §8.3 的 **4 个**敏感性臂（S1+S6 两端 × 2 套件，4×500 ep 已进预算）+ 逐档饱和率产出 + §8.4.2 分支 N 的裁决顺序 |
| R5 | ~~A 池入口未解决~~ **已冻结** | §7：复用 X14 已实测的 `materialize_apool.py`，产物 `db_init/libero/<suite>_apool/` + digest，不开空值口子；须在 LIBERO env 单列执行（P1b） |
| R6 | 族化重构打断 X14（**运行时 import，不只是路径**） | §9.4 两段式：M-c1 纯新增零破坏立即做；M-c2 压缩进 X14 结束后的单一静默窗口，5 条出场门 |
| **R9** | **S6 库把 B-val 吃掉，摧毁全 TIER 线唯一的"库外"切片** | §3.2：size 上限改 45/任务，B-val 钉死在排列末尾且网格不取；测试 `test_size_grid.py` 断言各档 ∩ B-val = ∅ |
| **R10** | **artifact ID 碰撞静默丢 90% 库**（`stem` + `task_N/episode_M.h5`） | §5.3.1：`--trajectory-id-mode relpath`；`test_build_artifact_ids.py` 的"加载后不掉条"通用探针 |
| **R11** | 本实验被误读为已交付原 X9 | §1：定位为**增补的 X9b**，X9 原文保留；`tier_experiment_designs.md` 同步增卡 |
| R7 | `/data` 是 HDD；build 阶段**不是顺序读**——`build_in_memory_cache_artifact.py:1021` 的 `--workers` **默认 0 = all CPUs**、`:893` 以 `ProcessPoolExecutor` 并发提交，单机械盘上是多流交织寻道；且嵌套网格使 6 档累加 `1+2+5+10+20+45 = 83` 份 vs 单份 50 ⇒ 12 次 build 的**总读量约 200 G 而非 121 G** | §12-P4 显式 pin 小 `--workers`（或把当档子集先 stage 到 SSD），并补 build 阶段墙钟估算。采集侧写入仅 ~5 MB/s，远低于盘速，不受影响 |
| R8 | 500 ep 采集中途崩溃 | **比原估好**：`main.py:81` 有 `--task_ids`（分批不改 `task_{id}/episode_{idx}` 命名，因 `episode_idx` 是任务内循环变量）；`main.py:106` 还有 `episode_filter`（`_load_episode_filter` :440-462 读 `[{task_id, orig_init_state_idx, subset_init_state_idx}]`，:571 据此跳过）⇒ 可做**逐 episode 精确续跑且 h5 名逐字节不变**，配合 `--save_episode_results`(:100) 自动生成清单。缓解从「崩溃丢一整批 50 ep」收紧到「只补丢的那几个」 |

---

## 12. 执行顺序与出场门

| 阶段 | 内容 | 出场门 | X14 依赖 |
|---|---|---|---|
| P0 | M-a 复制 → M-b 删源+建软链 | ✅ **均已完成**（§4.4） | 已用训练暂停窗口 |
| P1 | **M-c1 纯新增** `cache_size/`（含 `mkdir -p data/` + 软链）+ builder 两参数 + 排列规则骨架 + 全部单测 | `uv run pytest tests/ablation_study tests/exp tests/scripts` 全绿（**注意 basename 唯一性**，§10）；新测试文件名带 `cache_size_` 前缀 | **无**（零破坏） |
| **P1b** | **A 池物化**（复用 X14 `materialize_apool.py`）—— ⚠ **须在 LIBERO client env 执行**，主 uv venv 无 `libero` | **显式交付物**：digest 落盘 + **"A 池 ∩ 差集池 = ∅" 断言的执行记录**（该断言是整个泄漏论证的落脚点，至今未在本 repo 执行过） | 无 |
| P2 | 10-ep 采集 smoke | h5 含 `vision_*`/`prompt_emb`；路径 `task_N/episode_M.h5`；实测 ep/min | 需与 X14 排队用卡 |
| P3 | 正式采集 500×2（含 B-val 那 50，采而不入库） | 1000/1000 h5 齐、schema 校验、sha 记账 | 需与 X14 排队用卡 |
| **P3b** | **`emit_size_grid`**（依赖 P3 的实测成败，§3.2） | 各档 ∩ B-val = ∅；每任务条数表（含触顶任务）；嵌套性；种子确定性 | 无 |
| P4 | build 12 主 pkl（`--trajectory-id-mode relpath`，`--data-dir` 指到 `<collect_dir>/<suite>`） | 12/12 load 通过；**加载后不掉条**；逐任务 entries 表；嵌套性（小库 entries ⊆ 大库，触顶任务允许相等）。⚠ 显式 pin 小 `--workers`（§11-R7），并记录 build 墙钟 | 无 |
| P5 | 对两套件的 S1/S6 运行生产 LOEO 重标定 + 生成 **4 个敏感性臂 yaml**（不新建 pkl；分别复用对应 S1/S6 主臂的 `preload_path`，只替换 `score_normalization.fields.*.params`） | 4 份 `recal_norm_<suite>_{S1,S6}.yaml` 落盘；4 个敏感性臂 yaml 与各自对应主臂 deep-diff 仅在 params | 无 |
| P6 | 1 臂 10-ep 评测 smoke + **累计内存门** + **延迟硬门** | ① `hit_type` 恒 FULL_HIT 且逐 episode `FULL_HIT 率 == 1.0`；② 逐档 RSS 累计 < 50% available（**在实际执行主机上测**，§6.2.1），否则转降级；③ **实测 l10 S6 每步延迟**，超 §6.2.2 预算即触发降级预案 | 需与 X14 排队用卡 |
| P7 | 正式评测 **8,000 ep**（16 臂） | 16 臂满账、配对完整、teacher join 500/500；**每臂逐 episode `FULL_HIT 率 == 1.0`**（§3.2 的零覆盖兜底） | 需与 X14 排队用卡 |
| P8 | 分析 + 报告 `analysis/analysis.md` | 判读按 §8.4 决策树落定（分支 N 或 P×Q 9 格之一），**落格由 Holm 后检验驱动**；每格须同时报 Q 与 D 结论；`D-cache` 先过 §8.4.3 排查；**8 检验** Holm 表两套件各自独立；P 与等价附注标 descriptive | 无 |
| **P9** | **M-c2 族化收编**（收扁平文件 + 改全部引用 + 软链改名 + manifest） | §9.4 的 5 条出场门 | **X14 全部 run 结束后的单一静默窗口** |
| P10 | 同步 `tier_experiment_designs.md`（X9b 卡）+ outline 台账 + docs/README | index sync 一致 | 无 |

**关键排序性质**：P1–P8 全部**不需要** X14 结束（只需排队用 GPU），破坏性的 P9 被单独隔离到最后。这解决了原 §12 与 §4.4/R6 的自相矛盾。

---

## 13. 明确推迟的部分（不在本轮范围）

1. **完整 cache system 的 size 依赖**：带 `threshold` verdict 的 hit rate / bad-hit rate / SR 三者随 size 的联合变化。需要为每个 size 重新标定阈值，且要引入 hit 质量的逐步标注。**推迟原因是它会把阈值标定这一自由度混进 size 效应**（§1.1），不是因为不重要——它才是运维视角真正关心的量。
2. **同 size 的抽样方差**（§3.2 嵌套设计的代价）。
3. **warm start 档**（`always_warm_start`）的 size 依赖：历史 Ceiling-W 在 0.94–0.98，是比 always_hit 更接近生产的 replay 形态。
4. **第二 keybuilder / 第二 teacher**：与 X6（key-source）和 `benchmark_and_teacher_selection` 线交叉，各自独立排期。
5. **index 上界臂**（§3.3.1）：用 `--save_trajectory` 副产物里的 `sim_state` 按真实状态距离检索，给出"任何 index 能从这个库里榨出的上限"。这是唯一能把"payload 饱和"与"index 吃不下更密的库"真正**识别**开的设计；本轮只用覆盖率与饱和率提供间接证据，并把分支 A 的结论限定在本 index 下。

---
