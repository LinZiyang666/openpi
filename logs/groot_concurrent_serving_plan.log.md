# GR00T 并发评测服务改造 — fast-track plan

**Process**: owner 裁定（/goal 2026-08-21 原文：「现在开始修补缺口，使用正常快速流程免去G1，你草拟plan之后用agent专家自我审查，之后编码推进，于外审门G2停止」）：免 G1，plan + 3 静态 agent 专家自审代之；编码后停**外审 G2**。**工作级别 L2**（多文件+契约面）；免 G1 授权仅限本 plan。
**Status**: `Shipped` — G2 **APPROVED**（Round 4，2026-08-21）；正式 §6 Verify 过。
**Scope**: 修补 plan F6 预记缺口——`serve_groot_n15.py` 评测/teacher 服务路径的 per-connection 并发化。采集路径（D-L 冻结）不动。

## 1. 缺口与已验证事实（file:line 全部亲验 2026-08-21）

- 现状：`serve_groot_n15.py:321-339` 一次构建单 `GrootPolicyAdapter` 直塞 `WebsocketPolicyServer`，未传 `concurrent`/`connection_policy_factory` → 构造器默认 `concurrent=False`（`websocket_policy_server.py:470`），单连接闸 `_has_active_connection`（`:480,:522`）拒第二连接。
- 参照范式（pi05 已并发）：`serve_policy.py:865-881` `_connection_policy_factory` + `concurrent=True`；server 侧 `_bind_bundle` 懒调 `factory(self._policy, bundle_id)`（`websocket_policy_server.py:549-575`），bundle 换绑语义现成。
- 可变状态盘点（必须 per-connection）：
  - key_builder `cp1_groot_*`：可变 `_cache`/`_state_index`（F6 记录）；
  - `GrootCacheInterceptor`（`interceptor.py:119-163`）：orchestrator/timer/lifecycle 转发；
  - `CacheOrchestrator`：pi05 同例 per-connection；
  - `GrootStagedRunner`（`staged.py:200-244`）：持模型引用 + timer + probe 注册，无跨集状态 → per-conn 实例共享底层 model；
  - `GrootPolicyAdapter`（`groot_policy_adapter.py:186-249`）：仅持 `_policy` + hooks 转发。
- 共享面：`Gr00tPolicy`（模型权重）与 storage backend——`build_per_connection_components` 语义（`config.py:2483-2497`）：仅 backend 共享，`CacheStorage` facade per-connection；其 key_builder 走同一 `_build_key_builder`（`config.py:2720`），`cp1_groot_*` 在支（`:2770+`）✓。

## 2. 设计

- **D1 CLI（Rev2，R3-F1）**：`--concurrent` **默认 False、显式 opt-in**——保证 G0-E 已批形态与全部既有命令（裸 `--cache-config` / 采集 runbook）语义零变；`--concurrent` + `--collect-hdf5` 同给 = `parser.error`；`--collect-hdf5` 单独给恒单连接（**有意偏离** pi05 的响亮报错约定 `serve_policy.py:690-694`——照搬会让冻结的采集 runbook 命令开始报错，R3-F3 记账）。
- **D2 cache 路径工厂（Rev2）**：启动期一次 `load/validate config` + **`build_shared_storage(config)`**（`config.py:2457-2462`，pi05 并发同款 `serve_policy.py:829-834`；R2-NB1/R1-NB6：免建即弃组件）+ `validate_artifact_identity`（对启动期临时 facade 校验一次）；每连接 `build_per_connection_components(config, shared_storage, quiet=True)` → `CacheOrchestrator` + `GrootStagedRunner(policy.model, timer=conn_timer)` + `GrootCacheInterceptor` → `GrootPolicyAdapter` → 推理锁包装。**bundle 契约（R1-B1/R3-F4）**：工厂对 `bundle_id != "default"` fail-fast `raise`（server 现有 error-ack 路径 `websocket_policy_server.py:624-630/:818-824/:908-916` 回给客户端），杜绝「select_bundle ack 成功但仍服务 CLI yaml」的静默错配；yaml 热切换支持是后续项不是本 plan。**CSV 隔离（R1-B2）**：per-conn timer 的 `enable_csv` 指向 `output_csv_dir/conn_<uuid8>/` 连接唯一子目录（`timing.py:661-666` 文件名仅 task 序数+秒，并发必互覆）。
- **D3 teacher-only 工厂**：每连接 `GrootPolicyAdapter(shared_policy)` + 推理锁包装；同 D2 的 bundle fail-fast 契约。
- **D4 推理锁（v1 并发模型，Rev2）**：模块级 `threading.Lock` 包整个 `infer()`——GPU 严格串行、sim/网络重叠取收益（**注**：FULL_HIT 亦每步跑 stage1（`interceptor.py:212-213`），且 CP1 搜索/前后处理同被串行，收益打相应折扣——R1-NB2 预期管理）。**锁包装必须透明转发全部 lifecycle hooks 且 hasattr 面与内层等价**（不虚假暴露 `prefill_trajectory`；server 以 hasattr 探测 `:584/:635/:647/:667/:943`——包装不透传则 CSV 永不 flush、episode 不复位，R3-F5）。**共享 backend 无锁的安全前提（R1-NB1，显式记账）**：`InMemoryBackend` 读路径依赖 CPython GIL 原子性 + 每连接 uuid session 键不相交；写路径被 `load_guard.py:83-86` 的 `write_policy=never` 结构性堵死（`batch_insert` 不可达）——换 backend 或收窄锁粒度时此前提失效须重审。
- **D5 metadata（Rev2，R3-F2）**：仅并发分支增 `"concurrent": True`（pi05 同约定 `serve_policy.py:872`）；非并发/采集模式 metadata 字节不动（provenance 跨批 diff 零噪音）。
- **D6 风险验证项 — ✅ 已验证（2026-08-21，远端 /home/weiland/gr00t_n15 源码）**：transforms **无跨调用时序状态**（state_action/concat 调用期零自写；base 的 training/dataset_metadata 为一次性 setup）。唯一调用期自写 = `video.py:get_transform` 的 `self.width/height` 缓存——被 D4 整段 infer 锁串行化，且本部署各连接分辨率恒同（幂等写）。结论：无需 per-connection transform 手术，共享 policy + 整段锁安全。
- **D7 回归面（Rev2）**：默认（不带 `--concurrent`）路径 byte-fidelity 保持现行为——D1 默认 False 后该承诺对**所有既有命令**成立，不再限于显式 flag。**有意不引入** pi05 的 `empty_cache` no-op patch（`serve_policy.py:810`）：断连清理的 `gc+empty_cache` 停顿可接受，引入 patch 反而扩 blast radius（R1-NB4 记账）。
- **D8 unapply 无原地写 — ✅ 已验证（R1-NB3，2026-08-21 远端源码）**：`state_action.py:262-272` unapply 为 `data[key]=value.numpy()/astype` 替换、`concat.py:160-170` 为切片视图——均不改张量内容；FULL_HIT 回放共享库张量在 server 侧无写点，msgpack 序列化即拷贝。

## 3. 测试（非 manual，`tests/robocasa365/test_groot_concurrent_serving.py`）

- T1 工厂隔离：两次 factory 产物 interceptor/orchestrator/key_builder 两两 `is not`，storage backend `is` 同一。
- T2a 显式冲突：`--concurrent` + `--collect-hdf5` → parser error。
- T2b D-L 正向：仅 `--collect-hdf5` → server 收 `concurrent=False`。
- T2c 回归路径：默认（无 `--concurrent`）+ `--cache-config` → 旧单次构建结构（单 orchestrator、无工厂无锁包装），D7 兑现（R3-F6）。
- T3 锁串行：fake policy 记录临界区并发度，2 线程 × N infer 断言恒 ≤1。
- T4 teacher-only 工厂：per-conn adapter 异实例、底层 policy 同一。
- T5 lifecycle 隔离：conn A 的 episode_start/end 不落到 conn B 的 orchestrator。
- T6 bundle 契约：`bundle_id != "default"` → raise；default 幂等（R1-B1/R3-F4）。
- T7 锁包装透明性：逐 hook 转发到内层 + hasattr 面等价（无假 `prefill_trajectory`）（R3-F5）。
- T8 CSV 隔离：两连接 timer 的 csv 目录互异（R1-B2）。
- T9 metadata：并发模式含 `concurrent:True`，默认模式无此 key（R3-F8）。
- T10 共享 backend 压力：conn A infer 循环 × conn B lifecycle 钩子并发轰击，无异常、session/memo 隔离（R1-NB1）。
- （manual/真机：双连接 cache eval 冒烟 → 外审后真机段）

## 4. Verify

裸 `uv run pytest tests/robocasa365 tests/cache`（§6 纪律）。实现若被迫触及 `src/openpi/serving` 或 `scripts/`，Verify 面相应扩大并在 G2 说明（当前设计纯 exp/ + 新测试，R3-F11 亲验 server 侧机制全现成）。

## 5. 停点

外审 G2：展示 working tree diff，等 owner 审查/暂存。**不 git add。**

## Review Log

### Round 1 — 3 静态专家 agent 自审（2026-08-21）

- **R2 API 保真：APPROVE**。七组 file:line 全数亲验一致；facade/backend 类型链逐环验证无错配（`per_connection_facade()` 产平级 facade、backend 恒单例）。NB×4 已采纳：build_shared_storage（→D2）、CSV 目录风险（→D2/R1-B2 合并）、行号微漂移（§1 保留原引用+此处备注：serve_groot_n15 构建段实为 :323-339；adapter 类体至 :251）、上游 docstring 过时（非本 plan 项）。
- **R1 并发正确性：NEEDS-REVISION**。B-1 bundle 控制面静默错配（→D2 fail-fast + T6）；B-2 CSV 并发互覆（→D2 conn 子目录 + T8）。NB：backend 无锁前提显式化（→D4 + T10）、FULL_HIT 锁代价预期管理（→D4 注）、unapply 原地写验证（→D8 已验证闭合）、断连清理两弱点记账（→D7）、工厂在 event-loop 线程执行的停顿沿袭 pi05 既有模式（接受）、build_shared_storage（→D2）。
- **R3 冻结契约/回归面：NEEDS-REVISION**。F1 默认值翻转破坏 G0-E byte-fidelity（→D1 改默认 False 显式 opt-in）；F4 = R1-B1（→D2/T6）；F5 锁包装 lifecycle 透传（→D4 + T7）。NB：metadata 仅并发分支（→D5 + T9）、collect 静默单连接偏差记账（→D1）、T2 三拆（→T2a/b/c）、流程头补 L2+裁定原文（→Process 行）、D6 门可执行性（已由实测证据闭合，见 D6/D8）、Verify 条件句（→§4）。

**Rev2 处置**：5 项 blocking 全部吸收进 D1/D2/D4 与 T6/T7/T8；测试面 T1-T10。进入编码。

### 实现记录（2026-08-21，Executor）

- **改动面（纯 exp/ + 新测试，Verify 面无需扩大）**：
  - `exp/robocasa365/serve_groot_n15.py`：+`_InferLockedPolicy`（锁包 infer、`__getattr__` 委托保 hasattr 面等价）；+`_require_default_bundle`（bundle fail-fast，经 server error-ack 回客户端）；+`_build_concurrent_factory`（teacher-only 与 cache 双路：启动期一次 `build_shared_storage`+`validate_artifact_identity`，每连接 `build_per_connection_components(quiet=True)`+9 键 orchestrator+`GrootStagedRunner(model, timer=conn)`+interceptor+adapter+锁包装；CSV per-conn `conn_<uuid8>/` 子目录）；+`--concurrent` flag（默认 False）；`--concurrent`×`--collect-hdf5` = parser.error；main() 并发分支传 `concurrent=True`+factory+`metadata["concurrent"]=True`，默认分支逐字节保留原构建路径。
  - `tests/robocasa365/test_groot_concurrent_serving.py`（新，14 用例）：T1/T2a/T2b/T2c/T3(+异常释放)/T4/T5/T6(双路)/T7/T8/T9(源码 pin)/T10（真 `InMemoryBackend` lifecycle×search 双线程压力）。
- **pre-G2 advisory 测试跑**（非正式 Verify——法定顺序 Code→G2→Verify，正式 §6 Verify 在 G2 APPROVED 后重跑）：裸 `uv run pytest tests/robocasa365 tests/cache` = 1417 passed / 21 skipped / 0 failed（2026-08-21）。
- 状态：**停外审 G2**。工作树未暂存（owner 纪律），diff 待审。

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-21 11:57 CDT

- [Blocking] [Concern] `bundle_id="default"` 绕过了“固定 CLI 配置、拒绝 YAML 热切换”的契约：generic server 的 `load_cache_config` 可把任意新 YAML 以 default id 登记并返回 success，随后 `select_bundle("default")` 也被 `_require_default_bundle` 接受，但 factory 仍构造启动时 CLI 栈（teacher-only 启动甚至继续无 cache）——reasoning: 独立 `tests/review_tests/test_groot_concurrent_g2.py` 端到端驱动真实 `_handler`，实际收到连续两个成功 ack；这正是 D2/T6 声称消除的 provenance 静默错配。必须让此 entrypoint 对全部 hot-load（包括 default id）响亮拒绝，或让 factory 真正绑定已加载 bundle，并补 default-id 覆盖测试。
- [Blocking] [Concern] T10 没有兑现计划承诺的 session/memo 并发压力：测试构造的 `QuerySpec` 使用空 `trajectory_history`/`trajectory_weights`，因此 backend 必走 single-step 分支、从不触及 `_score_memo`；且两个线程无 barrier，`lifecycle_churn` 可先置 `stop`，使 search 循环零次执行——reasoning: 共享 backend 的 lock-free session/memo 安全是 D4 的显式前提，也是前轮测试盲区 blocking 的闭环项；测试须强制重叠、断言至少一次 search，并提供 depth>1 + `search_session_id` + `trajectory_query_ids` 以真实覆盖 memo 创建/读写/清理及跨 sid 隔离。
- [Blocking] [Concern] `logs/` 索引与交接文档未同步：新建的本 plan 在 `logs/README.md` 无条目，已修改的 `session_handoff_robocasa365.md` 对应索引仍写 `Ready-to-collect`/T5 待开跑；handoff 本身又先写“T6 建库完成”，随后仍列“建库→评测（T6 build artifacts）”待办——reasoning: 这违反 Working Agreement §4 的 Index Sync Rule，并使本次包含在审查范围内的操作交接状态互相矛盾；须在同一轮更新索引并把待办收敛到真实的评测下一步。
- [Non-blocking] [Concern] plan/status 把 G2 前的 1417/21/0 运行标成已经完成的“§6 Verify”，而法定顺序是 Code → G2 → Verify——reasoning: 本轮仅把该输出视作 G2 advisory evidence；请把记录改称 pre-G2 advisory，并在 G2 通过后重新执行、记录正式 Verify，避免流程状态虚报。

### G2 Round 1 — Executor — 2026-08-21

- [Blocking 1，default-id 热切换绕过] **Accepted** — 采纳「对全部 hot-load 响亮拒绝」路线（真正绑定已加载 bundle = 热切换实现，仍留后续项）：`WebsocketPolicyServer` 新增构造参数 `allow_dynamic_bundles: bool = True`（默认 True，pi05/LIBERO 语义零变）；为 False 时 `load_cache_config` 与 `select_bundle`（**含 default id**）一律 error-ack——select_bundle 整体关死是因为正常服务从不需要显式 select（首次 infer/episode_start 懒绑定），任何成功 ack 都在为不会发生的切换背书。`serve_groot_n15` 并发分支传 `allow_dynamic_bundles=False`；工厂内 `_require_default_bundle` 保留为纵深防线。新增 2 条 handler 级测试（真 `_handler` + scripted websocket）：关死态两 ctrl 全 error-ack 且 factory 零调用；默认态 legacy 流原样（missing yaml / unknown bundle 消息可达）。**改动触及 `src/openpi/serving/`**：按 plan §4 条件句，Verify 面相应扩大——serving 侧覆盖 = 上述 2 条 handler 用例（置于 tests/robocasa365；`tests/serving` 目录按既有纪律不跑）。
- [Blocking 2，T10 未兑现 session/memo 压力] **Accepted** — T10 重写：depth-3 `trajectory_history` + `search_session_id` + `trajectory_query_ids`（真实进入 `_score_memo` 分支，运行中断言 `sid in backend._score_memo`）；`threading.Barrier(2)` 强制两线程同刻起跑；断言 search 次数 ≥10（禁零搜索退化）且结果与单线程基线逐 id 相等；收尾断言 close 后本 sid 与全部 churn sid 的 memo 桶清空（跨 sid 隔离）。
- [Blocking 3，logs 索引与 handoff 状态未同步] **Accepted** — `logs/README.md`：本 plan 登记为独立条目（`Awaiting external G2`）、`session_handoff_robocasa365.md` 行由 `Ready-to-collect` 改 `Eval-prep`（T5/T6 收官事实入摘要）；handoff §6：矛盾的「建库→评测（T6 …）」待办改写为 **T7 评测**下一步（并发改造过 G2 → eval driver → L0 先行），T4b/收尾 commit 两条过期待办按事实闭合，最近提交行更新（本地 `32d291e` 未 push / origin `ef9d9cc`）。
- [Non-blocking，Verify 时序标注] **Accepted** — 实现记录中 1417/21/0 已改标 **pre-G2 advisory**；正式 §6 Verify 将在 G2 APPROVED 后按扩大面重跑并记录于此。

整改后聚焦回归：`tests/robocasa365/test_groot_concurrent_serving.py` 16 passed（T10 重写版 + 2 条 handler 用例含在内）。重入 G2。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-08-21 12:12 CDT

- [Blocking] [Concern] `allow_dynamic_bundles=False` 把正常 worker 必需的 default 绑定也关死了：`RobocasaEpisodeRunner._ensure_client()` 在每条新连接首次使用前无条件执行 `client.select_bundle(task.bundle_id)`，而本线 task 的 id 固定为 `"default"`；当前 handler 在调用 factory 前直接回 error，因此 T7 eval worker 无法抵达首次 infer——reasoning: 独立 handler 探针用固定 server 发送真实 `select_bundle("default")`，实际得到 disabled error，factory 调用数为 0；仓内 runner 契约见 `episode_runner.py:270,314-320`。应拒绝所有 `load_cache_config` 和非 default select，但允许 default select 完成首次懒绑定（或同步改造并验证真实 runner 协议），不能把“禁止热切换”与“禁止启动配置的幂等绑定”混为一谈。
- [Blocking] [Concern] 重写后的 T10 仍未强制并发操作真正重叠：`Barrier(2)` 只同时释放两个线程，`lifecycle_churn` 的 200 次纯 Python add/discard 仍可能在另线程第一次 search 前全部结束；`search_count >= 10` 只保证 stop 后补跑 search，不证明 memo 读写与 churn 同时发生——reasoning: Round 1 要求的是 shared-backend concurrent pressure，不是两个线程都曾运行。须用 Event/第二道 barrier 等握手让 churn 等到 `conn-a` memo 已创建，并让 search 等到 churn 进入活跃区，再断言活跃区内至少发生一次（最好多次）search。
- [Blocking] [Concern] 新测试当前不能通过仓库 lint：`test_shared_backend_survives_lifecycle_vs_search_pressure` 在 `tests/robocasa365/test_groot_concurrent_serving.py:450` 导入但未使用 `InMemoryBackend`，`uv run ruff check ...` 报 F401——reasoning: Working Agreement §7 要求 pre-commit 通过；删除死 import 并重跑检查即可。
- [Non-blocking] [Suggestion] 正式 post-G2 Verify 的扩大面应显式包含 `tests/serving`，而不是仅把两条 handler 用例放进 `tests/robocasa365` 后声明 serving 目录不跑——reasoning: 本轮直接修改了公共 `src/openpi/serving/websocket_policy_server.py`；Reviewer 独立运行 `tests/serving` 已绿，但正式 Verify 仍应覆盖该目录并记录命令。
- [Non-blocking] [Suggestion] 在 `WebsocketPolicyServer` 公共类 docstring 的 Args 中补记 `allow_dynamic_bundles` 语义——reasoning: 构造器已成为可复用公共接口，现有 Args 列表停在 `connection_policy_factory`，使用方看不到新安全开关及默认兼容行为。

### G2 Round 2 — Executor — 2026-08-21

- [Blocking 1，default 绑定被误关] **Accepted** — 混淆「禁热切换」与「禁启动配置幂等绑定」属实：`select_bundle` 门改为只拒**非 default** id；`select_bundle("default")` 放行走既有懒绑定（`load_cache_config` 仍整体拒绝 ⇒ "default" 槽永远无法被注册覆盖，default 绑定恒等于启动配置，幂等）。与 `episode_runner.py:314-320` 的 runner 契约（每连接首次使用前 `select_bundle(task.bundle_id)`，本线恒 "default"）恢复兼容。handler 测试改写为四拍：load_cache_config → error(disabled)；select_bundle("other") → error(disabled)；select_bundle("default") ×2 → 两次成功 ack 且 factory 恰调一次（重复 select 为 server 侧 no-op）。
- [Blocking 2，T10 重叠未证明] **Accepted** — 增加双向握手与窗口内计数：churn 线程先等 `memo_ready`（搜索线程在 `sid in _score_memo` 后置位）才进入活跃区；活跃区内每周期 1ms 步调（真实 lifecycle 事件本就稀疏，紧循环只会抢 GIL 饿死对手而非与之竞速）；退出条件 = churn ≥200 周期 **且** 活跃区内 search 增量 ≥5（10s 截止失败）；断言改为 `searches_inside_churn >= 5` + `churn_iters >= 200` + memo 曾建 + 基线等值 + 收尾桶清空。重叠由「搜索计数严格落在 churn 活跃窗口内」直接见证。
- [Blocking 3，Ruff F401] **Accepted** — 删除 T10 重写遗留的未用 `InMemoryBackend` import；`uv run ruff check`（本线三文件）All checks passed。
- [Non-blocking，正式 Verify 面含 tests/serving] **Accepted** — 正式 post-G2 Verify 命令定为：裸 `uv run pytest tests/robocasa365 tests/cache tests/serving`。本轮 advisory 已按此面预跑 = **1527 passed / 27 skipped / 0 failed**（2026-08-21）。
- [Non-blocking，docstring] **Accepted** — `WebsocketPolicyServer` 类 docstring Args 补 `allow_dynamic_bundles` 条目（语义、default-slot 例外、默认兼容行为）。

重入 G2。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-08-21 12:23 CDT

- [Blocking] [Concern] Round 2 恢复的 default 幂等绑定语义尚未同步到同批文档，当前仍有三处明确写成相反契约：`logs/README.md` 本 plan 条目仍称 `load_cache_config`/`select_bundle` “一律 error-ack 含 default id”且状态停在 Round 1 整改中；`WebsocketPolicyServer.__init__` 旁注仍称两类 ctrl message 都会被拒绝；handler 测试 docstring 仍称“default id included”也应 error-ack——reasoning: 实现与测试现在正确地允许 `select_bundle("default")`，这是 `RobocasaEpisodeRunner._ensure_client()` 抵达首次 infer 的必要协议；相反说明会让后续运维/复用者再次破坏已经修复的 runner 契约，并违反 Working Agreement 的 Index Sync Rule。请仅把三处文字同步为“拒绝全部 load、拒绝非 default select、允许 default 启动配置幂等绑定”，并把 README 状态推进到真实 G2 轮次；代码行为无需再改。
- [Evidence] Round 2 五项整改的实现面均已亲验通过：default 两次 select 成功且 factory 只调用一次；T10 在 memo 建立后进入 churn 窗口并以窗口内 search 增量证明重叠；`uv run ruff check src/openpi/serving/websocket_policy_server.py exp/robocasa365/serve_groot_n15.py tests/robocasa365/test_groot_concurrent_serving.py` = All checks passed；Reviewer 独立运行 `uv run pytest -q tests/robocasa365 tests/cache tests/serving tests/review_tests/test_groot_concurrent_g2.py` = **1529 passed / 27 skipped / 0 failed**（其中 2 条为 ignored review-only 独立探针）；`git diff --check` 与 `git diff --cached --check` 均通过。

### G2 Round 3 — Executor — 2026-08-21

- [Blocking，三处文档措辞与实现相反] **Accepted** — 统一为「拒全部 `load_cache_config`、拒非 default `select_bundle`、允许 default 启动配置幂等绑定」：① `websocket_policy_server.py` 构造器注释重写（含 runner 每连接首发 default select 的依据，标注 G2 R1/R2）；② handler 测试 docstring 改为准确描述四拍语义；③ `logs/README.md:70` 行措辞更正并把状态摘要推进至当前轮次（R1/R2 共 6 blocking 已整改、R3 措辞项闭合、advisory 1527/27/0、待 APPROVED 跑正式 Verify）。代码行为零改动；`ruff check` 过、目标套件 16/16。

重入 G2。

### G2 Round 4 — Reviewer — APPROVED — 2026-08-21 12:30 CDT

- [Verdict] **APPROVED** — Round 3 唯一 blocking 已逐处闭合：公共 server 构造器注释、handler 测试 docstring 与 `logs/README.md` 索引现均准确表述固定配置契约（拒绝全部 `load_cache_config`、拒绝非 default `select_bundle`、允许 `select_bundle("default")` 作为启动配置幂等绑定）；README 状态也已推进到当前审查轮次。工作树相对 Round 3 索引仅有上述文字与 Review Log 变更，代码行为零改动；无新增 blocking/non-blocking concern。
- [Evidence] Reviewer 亲验 `uv run ruff check src/openpi/serving/websocket_policy_server.py exp/robocasa365/serve_groot_n15.py tests/robocasa365/test_groot_concurrent_serving.py` = All checks passed；`uv run pytest -q tests/robocasa365/test_groot_concurrent_serving.py tests/review_tests/test_groot_concurrent_g2.py` = **18 passed / 0 failed**（16 条交付测试 + 2 条 ignored review-only 独立探针）；`git diff --check` 与 `git diff --cached --check` 均通过。Round 3 已独立跑过扩大面 `tests/robocasa365 tests/cache tests/serving` + review-only 探针 = **1529 passed / 27 skipped / 0 failed**。
- [Gate] **G2 APPROVED**。下一步按法定顺序执行正式 Verify：裸 `uv run pytest tests/robocasa365 tests/cache tests/serving`，并把正式结果登记到本 plan；manual 真机双连接冒烟在其后进入真机段。

### 正式 §6 Verify（G2 APPROVED 后，2026-08-21）

裸 `uv run pytest tests/robocasa365 tests/cache tests/serving` = **1527 passed / 27 skipped / 0 failed**。与 pre-G2 advisory 及 Reviewer 独立运行一致（Reviewer 侧 1529 含其 2 条 review-only 探针）。manual 真机双连接冒烟随 T7 评测准备进入真机段。
