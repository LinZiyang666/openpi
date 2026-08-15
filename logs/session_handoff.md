# Session Handoff — X14 在线 RL Router 基线（TIER 论文线）

> 更新：2026-08-15 18:00 CDT（§4 Code 完成；**G2 Round 12 已在 Ziyang Lin 显式 owner override 下 APPROVED**，Round 11 最后两项 pilot 阻断已修复并由 269 条任务测试、30 条独立审查探针及 1229 条扩展测试验证。**§6 Verify 尚未执行**。本文覆盖了旧 ablation_study handoff——owner 确认其已过时；那条线的记录存于 `logs/ablation_study_plan.log.md` EN 注记 + `exp/ablation_study/analysis/` + memory）
> 交接对象：接管本工作流的下一个 session。
> 唯一实施权威：[`rl_router_baseline_plan.log.md`](rl_router_baseline_plan.log.md)——**先完整读它**，本文只是导航与上下文，不重复 plan 细节；两者冲突以 plan 为准。

---

## 0. 接手第一步（不可跳）

1. 按 `CLAUDE.md` 会话初始化协议：读 `WORKING_AGREEMENT.md` → 声明 **Execution Authority** → 读 `protocols/execution_authority.md`（**绝不读 review_authority.md**，也绝不碰 `tests/review_tests/`——宪法级封印）→ 出状态卡。
2. 完整读 `logs/rl_router_baseline_plan.log.md`（204 行，自包含：需求、17+ 亲验代码锚点、全部冻结契约、测试策略、风险、里程碑、owner 裁决）。
3. 读本文件其余部分补上下文。

当前状态卡（接手时点）：

```
WORKFLOW STATUS | Authority: Execution | Task: X14 在线 RL router 基线实现 | Level: L3
Understand ✅ → Plan ✅ → G1 ✅(APPROVED R5) → Code ✅(M1+M2) → G2 ✅(APPROVED R12, owner override) → Verify ⬚
```

§6 Verify 仍未执行：按 Execution §5 必须等 G2 APPROVED。

## 1. 我们在干什么（大图景）

**论文线**：TIER（experience-tiered inference）投 ICLR 2027。thesis =「经验库的价值在索引不在 payload」；系统 = 检索相似度 + 双阈值把每个控制步派给 teacher（Pi0.5）/ student（蒸馏 ACT/SmolVLA）/ cache（FULL_HIT 直接回放 clean action）三层执行体。论文工作文档全在 `docs/iclr/`（提纲中英双语 + 实验设计卡 X1–X14，提纲文末有 Q&A rebuttal 弹药库）。重要口径：论文拆两篇（Markov 继承线独立成文，本篇零 history/trajectory 内容）；相关 memory：`project_iclr_paper_scoping`、`reference_libero_init_budget`。

**本任务（X14）**：审稿人/导师必问"为什么不训一个 router 而用检索"。我们的回答：监督学习对逐步闭环路由在语义上不可用（反事实标签/分布错位/Bellman 耦合三层论证，见提纲 Q&A Q1），唯一语义正确的训练路线是**在线 RL**——所以真跑它当基线：3 个 MLP router（R_ts / R_tc / R_tsc），batch on-policy REINFORCE 嵌入现有 cache 框架（MLP 作为新 verdict 层 judge `mlp_router`，对它屏蔽一切库侧信息），用 interaction-efficiency 曲线给"训 router 的取得成本"标价，冻结权重在 A 池与 TIER 配对对决。

## 2. G1 历程（为什么 plan 长这样）

G1 走了 5 轮（R1–R4 各 NEEDS REVISION，R5 APPROVED），审稿人共提 35 条 blocking 意见、**全部 Accepted**。每一轮都逼出了实质设计升级，接手者务必理解这些是**冻结契约**而非可选建议（Review Log 已按 Execution §3.1 在 polish 时删除，但契约全部内化在 plan 正文；G2 会重新开一节 Review Log）：

- R1：特征源从 raw cached_data 改为 `build()` 后 `query_keys`；三层动作记录；身份四元组；算法冻结雏形。
- R2：orchestrator payloadless FULL_HIT 加法分支（现状 winner=None 会落 MISS）；interceptor 三态互斥状态机（False 必须强制 replay）；维度实测 65,568；scheduler-accepted 贯通；per-episode RNG；warm-start 同架构 MLP 头；λ pilot 必须真训练（固定策略下臂分布与 λ 无关）。
- R3：DumpingJudge 条件转发；fp16 量化入 encoder（parity 按构造精确）；MISS 路径也带 router_outputs；repair 用新 task_uid + training_accepted_manifest；constant_arm 入 schema；多种子聚合 seed-0 primary；交互总账含 warm-start+pilot。
- R4：服务器权威 `decision_idx`（client step_idx 是物理步、随 replan_steps 跨步，不可作 join 键）；attempt 权威 = dispatch 时顶层 `task.attempt`；分片 = 内存 buffer + 终结一次性 headerless `.bin`（finalize 广播必须在 `on_episode_end` 的 **finally**——write-never 配置走 decline early-return）；repair = 每轮新建 TaskGraph+Driver + wls packager 权威；核验 = CPU fp32 单线程逐位相等。

## 3. 接手者接下来做什么（G2 已放行，进入 §6 Verify）

M1+M2 已实现。G2 的全部 blocking 已关闭；Round 12 在 owner 明示超越流程、授权 reviewer 直接完成最后两项 pilot 修复后放行，完整披露见 plan `## Review Log`。接手者下一步固定为 §6 Verify：

- 运行 `uv run pytest --ignore=tests/review_tests` **全量必须全绿** + staged API tests；遇到与本改动无关的失败 → 停下来交 owner 当场逐项裁决（plan 明令禁止预豁免）。已知 HEAD 既有失败 1 条（`tests/examples/test_libero_main.py::test_eval_paths_use_shared_episode_id_helper_source`，grep 的是本次未改的 `examples/libero/main.py`），**不得预豁免**，须当场提交裁决。
- 本轮 reviewer 已按 owner override 修改并暂存最后修复；后续 Execution Authority 不要重复改写或重新暂存其他 session 的文件。

**已落地的实现地图**（改动位置速查）：

- src：`cache/components/mlp_router_judge.py`（新，judge+encoder+权重+分片状态机）、`judge.py`（`judge_accepts_query_keys` + `JudgeResult.hit_override/router_outputs`）、`orchestrator.py`（payloadless FULL_HIT 分支 + 条件注入 + `_unique_judges` + episode/task-end finally 广播）、`interceptor.py`（三态分派 + `arm_executed` 回写）、`config.py`（`mlp_router` 类型/字段/工厂/校验，**含禁 `miss_to`、CP1-only、feature_fields 须 enabled**）、`conductor/{scheduler,driver,journal}.py`（accepted/error 贯通，**`SidecarError` 归 fatal**）、`examples/libero/episode_runner.py`（身份覆盖 + `router_outputs` 列）。
- exp：`exp/rl_router/`（`batch_package`=三源五键 join + Local/Ssh transport + round-scoped package + ledger 栅栏回收、`train_router`=REINFORCE + full-N admission + sidecar 身份/连续性/digest 校验 + export_meta/state/rejected 摘要 + `--export-only` 幂等重导、`fit_warmstart`=(uid,attempt) admission + **δ₀ 在"实际 ship 的那个 head"的 held-out fold 上按均值 realized rate 二分求解**（最终 head 只训 4 折，第 0 折留作标定；测试直接断言 grafted 参数的部署态 student rate=0.5）+ 强制 folds 清单、`collect_warmstart`、`run_rl_router`=`RemoteRun` 远端命名空间 + 两类 repair 状态机（**generation 与 quarantine 持久化到 `repair_state.json`，崩溃后按原 generation 续跑**）+ `_ResultRowPersister` 逐 result 落盘 + 容量探测 fail-closed、`microbench_cost`=逐臂**进程内** CUDA-event（ACT 按 manifest 真实 prompt 逐任务测再取均值；`in_process` 为假则 gate 拒）、`pilot_lambda`=λ 闭环 + split yaml 产出、`emit_router_yamls`=arm yaml + 预注册守卫、`launch_gates`=G-launch（复算 pilot 逐候选 manifest/digest、arm yaml 字段、episode budget、M4 绑定证据）+M4 五断言+run manifest；**M4 报告由 `run_rl_router.py --smoke` 真实跑 20ep×2 批产出**（规模**机械固定**为 `SMOKE_EPISODES × SMOKE_BATCHES`，绝不取自 matrix；bootstrap 是唯一可在无容量报告时启动的模式，有 main 级 golden 锁调用次数与批大小）、`config/run_matrix.yaml`）。
- tests：`tests/{cache,serving,conductor,libero,exp}/` 共 8 个新文件，含 `tests/exp/test_rl_router_run_loop.py` 的 **`run_rl_router.main()` 双隔离 filesystem golden**（远端树与本地树分离，任何把远端 artifact 当本地读的写法都会在此暴露）。

**M4 起进入运行期**（G2 APPROVED + §6 Verify 之后）：20-ep smoke（`launch_gates.py smoke` 五断言机器出场门）→ M5a cost microbench（**逐臂在各自 host 上跑 `microbench_cost.py arm`，再 `combine`**）→ M5b warm-start 采集（B-train 450 ep×2 套件，constant_arm 模式，**必带 `--init-states-dir`**）→ M5c λ pilot（`pilot_lambda.py plan → run`）→ `launch_gates.py check` 双门 → M6 五 run 正式训练 → M7 A 池一次性评测 → M8 spatial 确认 → M9 analysis。**M4 起每次 launch 前与 owner 确认**；运行期操作先加载 `experiment-lifecycle` skill（tether/监控/无人值守纪律都在里面）。

**实现时最容易踩的十个契约**（全文详见 plan，此处速查）：① MLP 只见 query_keys、决策函数签名不含任何库侧量（score 置换不变性测试锁）；② student 臂 = FULL_HIT + winner_id=None + hit_override=True（零 fetch）；③ interceptor 三态：True→executor、False→强制 replay（即使配了 hit_executor）、None→现行为逐字节不变（golden）；④ 三源 join 只认 `decision_idx`；⑤ 身份五元组 `(run_id, batch_id, task_uid, attempt, weights_version)` 入分片路径，attempt 从顶层 `task.attempt` 强制覆盖；⑥ 分片 finalize 广播必须放 `on_episode_end`/`on_task_end` 的 **finally**；批完整性权威 = shard manifest（不是 journal）；⑦ encoder 算子次序 `normalize(raw)→Q`，RL dump = Q 输出 fp16 字节，核验 = CPU 单线程逐位 `==`；⑧ 每批恰一次 Adam step，多 epoch 禁止；⑨ full N training_selected 封定前绝不更新；⑩ `constant_arm` 与 `weights_path` 恰一互斥。

## 4. Owner 已裁决事项（全部生效，勿再问）

- **D1–D8 全按建议**（plan §9 有逐条记录）：λ 网格 {0.05,0.2,0.5} + pilot 协议；批 100 ep；仅 B-train；cost = M5a 实测 GPU-time；旗舰全曲线其余终点；ACT 主臂；旗舰双训练种子（共五 run ≈ 20k 训练 ep）；X14 卡三处偏离已批准且上位卡已同步。
- **不用 cache warm-start**（实验配置层）：所有 cache hit = FULL_HIT 直接回放 clean action；系统代码 WARM_START 能力保留不删。"warm-start" 在本 plan 里只指 RL 权重热启动。
- 全局规矩：commit 信息英文、无 Co-Authored-By；多 agent 编排每阶段 agent 数静态固定；无人值守期间禁 run_in_background 后台任务（用 L3 cron）；共享机禁宽 pkill、按 PID 定点。

## 5. Git 与文件状态

- **G2 Round 12 时点本任务快照由 reviewer 暂存**，包含 owner override 下的最后两项 pilot 修复与审计记录；其他 session 的未暂存内容保持原状。
- **本任务改动范围**：`src/openpi/cache/`、`src/openpi/conductor/`、`examples/libero/episode_runner.py`、`exp/rl_router/`、`tests/{cache,serving,conductor,libero,exp}/`、`docs/{README,architecture/cache_system,cache/tutorial}.md`、`logs/README.md`、本 handoff、plan 的 `## Review Log`。
- **其他 session 的改动**（勿动、勿暂存）：`docs/iclr/`、`logs/benchmark_selection.log.md` 等属于并行工作流。
- Commit 时注意宪法级 Index Sync：docs/logs 的文件与对应 README 必须同 commit。

## 6. 设备与拓扑事实（勘察于 2026-08-15 14:00，launch 前须复核）

- **weilandserver**：4090 **49140 MiB**，勘察时显存 0 占用、无 tmux、8000/70xx 无监听——干净（旧 ablation handoff 曾记录存活服务，实测已不在）。规划角色：pi05 routed server :8000 + ACT sidecar :7002 (+SmolVLA :7001) + trainer + packager。
- **timan107**：48 核 + 8×GTX1080 8G，存量 tmux（w0-w11、c1-c3）待按名处置。角色：conductor + 8 LIBERO worker。
- tether：只查看未挂载；现存 expose `wls-ssh :14024`（batch package scp 通道）、`t107-ssh :14010`；执行期需新开 `rlr-srv`（wls:8000）——**launch 前经 owner 确认**。jupyter-ziyang10 OFFLINE。
- B 池数据事实：官方 init 1000/套件 = pruned_init 500（A 池，只测量）+ 差集池 500（B-train 450 / B-val 50，tracked split yaml）；无可新铸第三池（memory `reference_libero_init_budget`）。

## 7. 关键文档/记忆指针

- Plan（唯一实施权威）：`logs/rl_router_baseline_plan.log.md`
- 论文语境：`docs/iclr/tier_paper_outline.md`（+`.zh.md`；Q&A Q1 = 本任务的存在理由）、`docs/iclr/tier_experiment_designs.md`（X14 卡）
- 框架文档：`docs/architecture/cache_system.md`、`docs/cache/tutorial.md`（实现后两者都要按 plan §5 更新）、`docs/experiments/conductor_tutorial.md`
- Memory：`project_iclr_paper_scoping`、`reference_libero_init_budget`、`feedback_no_unsolicited_git_add`、`feedback_review_cycle_protocol`、`reference_pytest_manual_skip`（⚠ 其 Verify 口径已被本 plan G1 裁决覆盖：本任务用全量 `--ignore=tests/review_tests`）、`feedback_no_background_tasks_unattended`、`reference_preexisting_test_failures`（背景参考，但本任务不得预豁免、遇失败须 owner 当场裁决）。

## 8. 本任务特有陷阱

- G2 审稿人与 G1 同样严格（本线 G1 打了 5 轮）；每条意见逐项响应、Accepted/Rejected 必须有实质理由，anti-rubber-stamping 双向适用。
- 实现偏离 plan 任何一条冻结契约都须先向 owner 亮牌再动（Execution §4 禁止静默偏离）。
- t107 的 conda run 块缓冲、pkill 自匹配、tether HOME 等运维坑见 `experiment-lifecycle` skill 与 memory 陷阱条目。
- 不实时更新本 handoff——只在关键节点（stage DONE / 重大决策 / 拓扑变更）同步（memory `feedback_handoff_no_realtime_update`）。
