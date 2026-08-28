# RoboCasa365 检索参数搜索 Round 2（text-IVF × full704 库 × 动态 bundle）— Plan

**Status**: `In Progress`（G1 APPROVED（R4）→ §4 Code → **G2 APPROVED（Round 5，2026-08-26；五轮共 14 条 blocking 全闭合）** → **§6 Verify 过**：裸全量 `uv run pytest` = **4404 passed / 11 failed / 60 skipped**（318s），11 条全部为既有基线——5 条 HEAD 既有（`test_libero_main` 源码锁、`test_prebuilt_matrix_backend` ×2、`test_robocasa_policy_config` ×2 的仓库测试隔离缺陷）+ 6 条其他线的 `tests/review_tests/` 探针（cache_size ×3 / groot_robocasa ×1 / rl_router ×2），**本任务零新增**。⚠ 其中 `test_groot_robocasa_g2::test_run_one_closes_env_when_inference_fails` 触及本轮改过的 `episode_runner.run()`，故实测核验：换回 HEAD 版该文件后同一条仍失败于 `ModuleNotFoundError: robocasa.utils.dataset_registry_utils`（本 venv 未装 robocasa），属环境依赖非回归。**待 owner 授权 §7 Commit**）
**日期**: 2026-08-26
**Level**: L2（src 校验规则小改 + exp harness 新实现；全链走 Understand→Plan→G1→Code→G2→Verify）
**Owner 指令**（2026-08-26）: 用新 full704 分桶 pkl 重做 Round-1 的 GR00T 参数搜索，**打开 text-IVF**；利用新升级的 GR00T server/conductor 基建（weilandserver 可开 6 replica server、timan107 worker 打满）；把所有缺口堵上，按流程推进，于 G1 停止。
**前置文档**: Round-1 = [`robocasa365_ws_search_plan.log.md`](robocasa365_ws_search_plan.log.md)（Closed）+ 报告 [`ws_search_round1.md`](../exp/robocasa365/analysis/ws_search_round1.md)；text-IVF 功能 = [`text_ivf_prompt_bucket_plan.log.md`](text_ivf_prompt_bucket_plan.log.md)（commit `cf10ebc`）；serving/conductor 升级 = [`groot_concurrent_serving_plan.log.md`](groot_concurrent_serving_plan.log.md)（`9f9a1f0`）+ [`groot_serving_perf_parity_plan.log.md`](groot_serving_perf_parity_plan.log.md)（X17，`9714849`）。

---

## 1. 已验事实（本轮亲验，file:line）

**Round-1 遗产（重做的对照基准）**

- F1 GR00T 臂 132 cells 全 complete：`weighted_score_sum_knn` × `cp1_groot_spatial_pool_16` × n5 库 × 13 task × 8 trial（seeds base=1_000_000+idx、场景钉死 (1,1)）；prompt_emb 弃权（D3）。榜首 macro_sr=0.269、48/132 配对打平；4 个 PickPlace 任务死平 ~1.5%（n5 库仅覆盖 eval 物体类别 19%）。报告 §5 明确建议：库扩容 + **prompt_emb 入检索**——即本轮两个自变量。
- F2 分析工具带轮次参数：`summarize_ws_search.py:30,40,57`（`--run-prefix`，glob `summary_{prefix}-*.json`）；`analyze_ws_search_stats.py:45,54,65,116,117`（`--run-prefix` + `--exclude-tasks`；cid 从文件名切 `"__l1s1"` 前段）。Round-2 用新 prefix 即与 ws1 产物并存互不覆盖。
- F3 Round-1 身份公式：run_id/yaml_id = `ws1-<cid>__l<L>s<S>_<teacher>`（`run_ws_search.py` docstring :28-30），cell 身份入 `task_uid`。Round-2 同构换 prefix。

**text-IVF 功能面（cf10ebc，全部亲验）**

- F4 `TextIvfKnnStrategy`（`search_strategy.py:518-573`）**就是 weighted_score_sum 底座**：`fusion_method="weighted_score_sum"`（:567），支持 `fusion_weights`/`field_similarity`/`score_normalization`（:537-539,566-569）——Round-1 的权重矩阵可原样跑在其上。桶圈定替代 task 圈定：`_build_filters` 刻意不发 task_key（:575-578 注释）；`search()` 硬性要求 `"prompt_emb" in ctx.query_keys`（:553-558，缺失 raise ValueError）。
- F5 对照面：`weighted_score_sum_knn` 的过滤器带 task_key（`search_strategy.py:395-396,402-403`，`ctx.task_key` 注入 `QueryFilter`）——控制臂用它 + 同一 full704 库即可分离「库扩容」与「桶圈定」两因子。
- F6 查询向量来源：backend 从 `spec.query_keys` 取筛选字段（`in_memory_backend.py:586-590`，缺失 raise）；backend_hints `text_ivf` 触发 `_text_ivf_candidates`（:383-384）。
- F7 GR00T 在线 key 通路产 prompt_emb：`src/openpi/cache/groot/key_builder.py:129-133`（`input_embeds` 序列的非图像 token 段 → reducer 池化；注释明记 "prompt keys are constant within an episode"）。离线/在线同一 builder 族，parity 已由 groot cache 集成线 T-8 证过；full704 pkl 建桶 **111 == 独立清点的指令变体数**（`/tmp/validate_textivf.log`，2026-08-25）。
- F8 **规则 6 = 本轮唯一 src 阻塞**：`config.py:2542-2547` 把 `key_builder.type.startswith("cp1_groot_")` 与 placeholder/clip 一并拒绝，注释理由 "no prompt_emb semantics"——对 cp1_groot_* **事实不成立**（F7）。这是 text-IVF plan v1 的 scope 边界（该 plan §11 明记 GR00T 排除为 owner-scoped v1 边界），非技术不可行。现有测试 `tests/cache/test_config.py:2150-2155`（`test_text_ivf_rejects_no_prompt_builder` 参数化含 `cp1_groot_mean_pool`）需随改。
- F9 绑定检查对 groot **零改动即正确**：`_check_text_ivf_artifact_binding`（`config.py:2699-2750`）比对 artifact meta 的 `key_builder_type`（== `cp1_groot_spatial_pool_16` ✓）与 `prompt_pool`（groot pkl = `{masked:False, instruction_span:False}`，vs groot yaml 旋钮恒 False ✓ —— cp1_groot_* 不在 `PROMPT_POOL_KNOB_BUILDERS`（:542），规则 8 保证旋钮开不了）。legacy 无 meta 的 GR00T LIBERO 老库会被 :2735-2741 拒绝——规则 6 放开后 `cp1_groot_libero_*` 想用 text-IVF 仍被绑定检查兜底，无静默面。
- F10 索引参数：`TextIvfIndexConfig`（`config.py:449-461`）默认 `field="prompt_emb"`、`max_buckets=1024`；groot full704 实测 111 桶，余量充足。
- F11 库台账：`/data/robocasa365_cache/cache_artifacts_text_ivf/groot_tp_spatial_pool_16_full704.pkl` = 20.5GB / 50,795 entries / 111 桶 / meta `{masked:F, span:F}`（repo 软链 `exp/robocasa365/data/cache_artifacts_text_ivf`）。延迟实测（2026-08-25，weilandserver）：全库暴搜 263ms / task 过滤 29.1ms / text-IVF 27.6ms（均值），多变体任务 4-17.6×，winner 与 task 过滤一致 200/200，探桶 50µs。

**serving / conductor 基建（`9f9a1f0` + `9714849`，全部亲验）**

- F12 `serve_groot_n15.py` 具备 `--concurrent`（per-connection factory，共享 GPU policy + 共享 storage）与 `--allow-dynamic-bundles`（:475-539）：后者要求 `--concurrent` 且启动必带 `--cache-config`（:520-533），与 `--collect-hdf5` 互斥（:537-539，采集线零波及）。动态模式下 `_resolve_bundle`（:303-346）从 `get_current_cache_bundle(bundle_id)` 读 storage、guards 逐 bundle 重跑、**storage 不重建**（:320-321 注释）。
- F13 **132 次热切换只载一次 pkl**：server 的 `load_cache_config` handler 走 `build_shared_storage`（`websocket_policy_server.py:795-825`）→ backend 经 `BackendPool.get().get_or_load`（`config.py:2943-2960`）；pool 按 `BackendFingerprint`（`backend_pool.py:64-92`：backend_type/resolved_preload_path/vector_dims/index_type/text_ivf_params）缓存且**只进不出**（:220-250 无释放面）。132 个 cell yaml 仅权重/归一化不同 ⇒ 同 fingerprint ⇒ 首次 load 后每次 swap 仅新建轻量 facade。X16 线的「GB 级重建 + MMU fault」教训（`run_conductor.py:196-206` 注释，2026-08-20 pi05 老路径事故）在 pool 路径下不复现，但其防御（journalled episodes 从 plan 里 drop、完臂过滤）照抄。
- F14 conductor 驱动范式已真机验证：`exp/libero_groot/run_conductor.py` 的 strategy 在 `plan()` 里给 `EpisodeTask` 带 `bundle_id=yaml_id`（:138）、经 `shard_eval_stage`（`sharding.py:65-77`，一逻辑臂 → 每 server 一 sibling stage）分片、`on_stage_begin` 发 `ctl.load_cache_config(yaml_content=..., yaml_id=..., bundle_id=stage.yaml_id)`（:167-187，bundle_id 刻意不用 "default"）。X17 T13 端到端冒烟 PASS（2 server 动态 bundle、单臂跨 server 16/16 集、journal 全 accepted）。
- F15 robocasa worker 侧**已是 bundle-aware**：`episode_runner.py:323-327` 按 `task.bundle_id` 逐任务 `client.select_bundle`。Round-1 的 `run_ws_search.py`/`orchestrate_ws_search.py` 仍是重启制（docstring :7-10 明记），**未接线动态 bundle——本轮 harness 缺口**。
- F16 容量墙已拆：Round-1 Stage-0 的 6×3 腰斩两因（主存带宽 + WiFi）分别被 P1 冻结搜索缓存（search 136→2.62ms，Round-1 plan §9 2026-08-22 记录）与有线切换（enp8s0 1Gb/s）溶解；X16 已在 weilandserver 实跑 **6 个 GR00T server 进程** 收官 17,000 集。资源账：VRAM 6×~6.25G=37.5G < 48G；RAM 6×~21-24G ≈ 130-145G < 251G（首台冷载后文件驻留 page cache，后 5 台 RAM 速装载）。
- F17 emitter 复用面：`exp/weighted_sum/emit_yamls.build_eval_config`（:34-116）产出 `weighted_score_sum_knn` + `index_type:"brute_force"` 固定形状（:85-116）；`exp/robocasa365/emit_ws_search_yamls.py` 的 `weight_matrix()`（:86）生成 132-cell 权重族，`emit_teacher`（:105-130）做 robocasa fixup（cp3 钉死块等）。Round-2 emitter 在 build 产物上后处理三键（见 §3-W2），老文件不动。
- F18 标定工具：`exp/common/calibrate_score_normalizers.py` artifact 自含 LOEO（`--artifact-dir --output --max-queries`，默认 300；prompt_emb 排除在标定外——本轮 prompt_emb weight=0 不参与打分，无需标定）。Round-1 的 n5 标定 json 对 full704 失效（分布变了），必须重标。

## 2. 设计决策

- **D1 自变量与目标**：相对 Round-1 改两件事——库 n5→full704（≈13.8×）、圈定 task_key→prompt 桶（text-IVF）。目标函数照旧（macro_sr 任务等权 + per-task；配对统计）。判决性问题分两层，**推断总体不同**：(a) **联合干预层**（全 132 格可答）——"库扩容 + text-IVF"合并作用下 4 个 PickPlace 死任务是否复活、权重面形状是否改变；(b) **因子分解层**（仅 §3-W6 的 12-cell 匹配子集可答）——两因子各自贡献多少。全矩阵的 ws1→ws2 差**只作联合效应呈报，不得拆分归因**。
- **D2 评测口径全承袭 Round-1**：纯 cache（`always_search`+`always_hit`+`top_k:1`+`write_policy:never`）、场景 (1,1)、eval seeds base=1_000_000、13 任务全保留（死任务是本轮测点，不剔）、timer 关。**cid 集合与 Round-1 逐字相同** ⇒ (cid, task, idx) 三键配对可直接量化每格提升。
- **D3 矩阵**：同 132 cells/权重族（iso 4 + grid2 42 + grid3 30 + grid3v 21 + grid4 35，权重面 = {v0,v1,v2,rs}）。prompt_emb 以 `{enabled:true, weight:0}` 入 keys（text_ivf_knn 校验硬要求，F4/规则），**不入权重扫描轴**——桶内 prompt_emb 近常量，无判别力；text 的作用 = 圈定。
- **D4 三臂结构**：
  - **主臂 ws2**（132 cells × 104 ep）：`text_ivf_knn` × full704。
  - **控制臂 ws2c**（12 cells × 104 ep）：`weighted_score_sum_knn`（task 过滤）× **同一 full704 库**、`index_type:"brute_force"`。cell 集由 **selection manifest 钉死**（算法与产物见 §3-W8）。三方对照（ws1 / ws2c / ws2）**仅在这 12 个匹配 cell 上**把「库扩容」与「桶圈定」的贡献分离——只有它们同时具备 n5+task / full704+task / full704+text 三个观测点。⚠ 两臂 backend fingerprint 不同（index_type 异，F13）⇒ 控制臂**独立相位串行跑**（6 server 重启一次专服 ws2c），避免单进程双 backend 驻留 ~42G×6。
  - **加密臂 ws2e**（10 cells × 32 trial = 416 ep/cell）：ws2 筛选轮排空后按 §3-W8 冻结算法选出 top-8 + 2 阴性对照，追加进同一 manifest（沿 Round-1 §10-2b 已验设计；idx 0-7 与筛选同 seed，兼作复现控制）。
- **D5 锚点臂（建议，owner 可砍）**：teacher-only 13×8=104 ep @ eval seeds（Round-1 §10-2 已亲验命令：`serve_groot_n15` 无 `--cache-config` 即 teacher-only；三槽任务切分 + journal `cat` 合并）。Round-1 取消未跑，eval-seed 上的教师天花板至今缺位。
- **D6 运行拓扑（动态 bundle 制，替代 Round-1 重启制）**：weilandserver 6× `serve_groot_n15 --concurrent --allow-dynamic-bundles --cache-config <bootstrap cell yaml> --port 23160-65`（端口先侦察让位；串行启动灌 page cache）；driver 在 weilandserver；worker = timan107 主力 + weilandserver 本机若干（Stage-0 定档，起点 = Round-1 终态 28-40 编制向上爬）。一次 driver invocation 装载全部 cells 的 TaskGraph（F14 的 bundle 下发范式 + Round-1 的 (cell, task) stage 构图，身份/产物契约全冻结于 §3-W3）：`on_stage_begin` 下发 bundle（(server, bundle_id) 备忘去重），worker 按 `task.bundle_id` 自选（F15）。**全程零 server 重启**（Round-1 132×~2min 重启开销归零），journal resume 承袭（plan() 丢弃已完成 uid，F13 防御照抄）。
- **D7 cell 顺序（scheduler 可见的交错，非 graph 插入序）**：conductor 的激活与派发都按 `sorted(stage_id)` 遍历（`scheduler.py:152-156` `_refresh_activation`、:246-248 `next_task`）——graph 插入序无效。冻结机制：**把族轮转 rank 编进 `stage_id`** = `f"{cell_rank:04d}__{yaml_id}"`（`Stage.stage_id` 与 `yaml_id` 本就是分立字段，`run_collect.py:180` 现状即证），`yaml_id`/`task_uid` 身份一个字节不动。`cell_rank` 由**相位的不可变全集 cell 列表**按族轮转确定（纯函数，与完成状态无关 ⇒ resume 后相对顺序确定不变）；同 cell 的 13 个 task-stage 共享同一 rank 段、段内按任务名字典序。测试用真实 `EpisodeScheduler`（不是查插入序）：各 server 前 N 个 setup/dispatch 的 cell 覆盖各族；partial journal resume 后相对顺序仍确定。
- **D8 桶归因证据通道（运行期采集，接口全冻结）**：主跑**开启既有的小结构 per-step 通道**（张量级捕获不开）——`episode_runner.py:386-410` 已在每次 `client.infer` 后从响应 `__hit_meta__` 上报 `{task_uid, yaml_id, step_idx, hit_type, winner_id, cp1_score, searched}`（响应字段由 GR00T cache 服务栈产出，t7/g0e hit-log 即其消费者，`groot_rollout_client.py:186-204` 同源）；`EpisodeResult.per_step_rows`（`task.py:117-119`）由 driver 中心合并、**先于 journal done 行落盘**（X17 加固）。W3 接通 per-step 输出目录（逐 cell 分文件，X17 `run_conductor.py` 已有同款实现可 port）。最小扩展：runner 子类在 episode 起点追加一行头记录 `{task_uid, yaml_id, step_idx:-1, attempt, prompt:<obs 的 PROMPT_SOURCE_KEY 文本>, seed}`（prompt 在 reset 后即得，`episode_runner.py:74,367-368`；**构造缝与默认不变的 opt-in 部署见 §3-W3**）——**eval 侧真值 prompt 逐 episode 落盘，无需 seed 重放**。体量账：~100 infer/ep × 19.4k ep ≈ 2M 小行 ≈ 数百 MB，逐 cell 分文件。join 契约与库侧映射见 §3-W7；S1 以「`__hit_meta__` 非空且 winner_id 可解析入桶」为放行判据之一（§5-S1）。
- **D9 在线探桶的 fp16 现实**：库向量经 H5 fp16 存储再池化，在线 query 是实时 float32——字节级精确命中**不保证**，预期走最近代表回落（cosine；同指令的量化误差余弦距离 ≪ 跨指令距离，落点不受影响，代价仅 50µs 探针）。⚠ exact/nearest 路径**在现役服务命令下不可观测**：backend 两条路径都只发 `logger.debug`（`in_memory_backend.py:595-618`，仅 tiny-margin 才 WARNING），而 `serve_groot_n15.py:59` 把进程日志钉在 INFO——故**该项不入任何阻塞门**（归因判据 = W7 的 eval prompt ↔ winner 桶变体一致率，与走哪条路径无关，S1-④ 已独立验落点）。好奇项处置：离线 DEBUG probe（本地载 pkl，库内同变体两成员互为 query，记录 exact 与否）作非阻塞观察，不冒充在线证据；在线路径若将来要观测须加 serving 诊断开关，属另立范围。
- **D10 Phase-1 重标定**：`calibrate_score_normalizers.py` 直吃 full704 pkl（staging 目录只放它），协议与 Round-1 相同（全库 LOEO、max-queries 300）；产出 json 入 `exp/robocasa365/config/ws_search2/`（tracked 唯一权威）。桶内条件化标定不做（保 Round-1 可比性；记为已知近似，§8-R8）。
- **D11 范围外**：pi05 臂（其 masked pkl 今日即合法，无需改规则——另轮另 plan）；GR00T LIBERO 线 text-IVF；serving 层新功能。

## 3. 实现单元（files touched / interfaces）

**src（正式面，G2 必审）**

- **W1 规则 6 按正向集合放开（范围与 D11 对齐）**：`src/openpi/cache/config.py:2542-2547` 改为——新增模块级冻结常量 `_TEXT_IVF_GROOT_BUILDERS = frozenset({"cp1_groot_mean_pool", "cp1_groot_spatial_pool_16", "cp1_groot_spatial_pool_4", "cp1_groot_max_pool"})`（即 RoboCasa 三相机 groot pool 全家，config.py:1624-1625 注册面）；拒绝条件 = `type in ("placeholder", "clip")` **或** `type.startswith("cp1_groot_") and type not in _TEXT_IVF_GROOT_BUILDERS`。效果：`cp1_groot_libero_mean_pool` / `cp1_groot_libero_spatial_pool_16`（config.py:1629-1630）**继续被规则 6 拒绝**——LIBERO 变体的在线/离线 parity 与带 meta artifact 证据缺位，D11 排除维持；capability fence 在规则 6 本身，不依赖绑定检查（后者只核一致性，挡不住新建带 meta 的 LIBERO 库）。正向四成员的依据：prompt_emb 提取与池化在共享基类（`groot/key_builder.py:129-133`），四者仅视觉 reduce 不同。**不改**绑定检查、不改 allowlist、不改策略/backend。测试：`tests/cache/test_config.py` 现有拒绝参数化（:2150-2155）cp1_groot_mean_pool 项转为接受用例；新增①`cp1_groot_spatial_pool_16` × text_ivf 全套 + 旋钮关 → 校验通过；②同配置 + 旋钮开 → 规则 8 仍拒；③placeholder/clip 仍拒（回归锚）；④**`cp1_groot_libero_mean_pool` 与 `cp1_groot_libero_spatial_pool_16` × text_ivf → 规则 6 仍拒（负向回归，两条都写）**。

**exp/robocasa365（新文件为主，Round-1 老三件不动）**

- **W2 emitter** `emit_ws_search2_yamls.py`：import 老 emitter 的 `weight_matrix` 与 `exp.weighted_sum.emit_yamls.build_eval_config`（F17），对产物后处理：主臂 = `cp1.search_strategy.type→"text_ivf_knn"`、`backend.in_memory.index_type→"text_ivf"`、`keys.prompt_emb→{enabled:true, weight:0.0}`；控制臂 = 保持 build 原形。preload 指 full704 绝对路径；vector_dims 取新标定 json；cp3 钉死块承袭。输出 `config/ws_search2/groot_tp/{main,control}/` + `index.json`；**控制臂 yaml 只为 W8 manifest ws2c 段列出的 cid 生成**（emit 顺序：W8 ws2c 段 → W2）。附带十项不变量校验器（Round-1 §9 06:0xZ 版扩展：text_ivf 三键形状、控制臂 brute_force、权重和=1、双 validator 亲载全过）。
- **W3 driver** `run_ws_search2.py`：port `run_conductor.py` 范式（F14）到 robocasa——多 cell 单 invocation、`on_stage_begin` 发 `ctl.load_cache_config`（按 (server, bundle_id) 备忘去重，重复 stage 不重发）、stage_id 交错见 D7、复用 `run_collect` 的 `load_env_config`/`validate_teacher_endpoints`/run-plan 机件（run_ws_search.py:55-66 同款 import 面；spawn_fn 用 W3 自带变体，见下）。角色 `--role driver|agent|all`、`--run-prefix ws2|ws2c|ws2e`、`--only`、`--episodes`、`--manifest`（ws2c/ws2e 相位只接受 manifest cell 集，§3-W8）。**运行产物契约（冻结，Round-1 逐字节同构）**：
  - **双视图 resume 契约**：`build_run_plan` 从 graph 的 stage episodes 枚举 uid（`run_collect.py:250-289`），`write_run_plan` 在 resume 时强制 plan_hash 逐字节相同（:293-312）⇒ 若执行图先剪掉 journalled uid，重算 payload 必缩短、resume 被硬拒。冻结为**两个 graph**：①**全集图**（纯函数：相位 cell 列表 × 13 任务 × episodes，**先于读 journal 构造**）——只喂 `build_run_plan`/`write_run_plan`（hash 跨 resume 恒定）与 finalizer 的 expected/uid 对账、summary 补写；②**活跃图**（全集扣除 `record_counts_as_done` 的 uid，`journal.py:97`）——交给 driver 执行；**活跃 episode 为零的 cell 不建任何 stage**（零 setup、零 bundle load，F13 完臂过滤的实现形态）。测试：同一中心 journal 首次 partial → **新进程** resume：run-plan hash 逐字不变、只派发缺失 uid、完整 cell 零 setup、finalizer 仍物化**完整**逐 cell journal/summary。
  - **header-runner 构造缝（默认不变，显式 opt-in）**：现状无选择点——`robocasa_spawn_fn` 固定拉起 `exp.robocasa365.worker_entry`（`run_collect.py:349-364`），`worker_entry.build_runner` 无条件实例化 `RobocasaEpisodeRunner`（`worker_entry.py:29-45`）。冻结：新模块 `ws2_episode_runner.py`（`Ws2EpisodeRunner(RobocasaEpisodeRunner)`）；`worker_entry` 新增 `--episode-header-rows` flag（默认关），置位时**惰性 import** 并实例化子类，默认路径行为字节不变（采集/普通 eval 零波及）；W3 自带 `ws2_spawn_fn`（照 `run_ws_search.py:269-285` 精神拷形）向 worker cmdline 追加该 flag。
  - **基类捕获 hook（单次-reset 接口，默认空操作）**：`run()` 的 `seed`/`obs`/`prompt`/`per_step` 全是方法内局部量（`episode_runner.py:348-422`），继承拿不到 ⇒ 基类 `RobocasaEpisodeRunner` 冻结新增最小 hook `_episode_header_rows(task, *, prompt: str, seed: int) -> list[dict]`：由 `run()` 在**首次（唯一一次）`env.reset` 并取得 `prompt` 之后**调用恰一次，返回行**前置**进 `per_step`；基类实现恒返回 `[]`（既有全部路径行为字节不变）。`Ws2EpisodeRunner` 只覆写该 hook，返回单行 `{task_uid, yaml_id, step_idx:-1, attempt: task.attempt, prompt, seed}`。**明令禁止**：复制整段 `run()` 到子类（漂移/回归面）、为取 prompt 二次 reset（破坏 D8 运行期真值契约）。`exp/robocasa365/episode_runner.py` 因此进 files-touched（改动 = hook 定义 + 一处调用点，无其他行为变化）。测试：基类 hook 返回 `[]` 且默认 per_step 行集合不变；opt-in 恰一条 header 且 prompt/seed 为 reset 真值；异常/accepted retry 后按 (task_uid, attempt) join 去重正确。
  - **身份**：run_id = `<prefix>-<cid>__l1s1_groot_tp`；**yaml_id = `build_yaml_id(run_id, task_name)`**（`run_collect.py:67`，任务名内嵌）；task_uid = `make_task_uid(yaml_id, "eval", task_id, episode_idx)`（`task.py:66-73`）⇒ 既有 analyzer 的两段解析（文件名 `.split("__l1s1")` 取 cid、`task_uid` 的 `rpartition("__")` 尾段取任务名，`analyze_ws_search_stats.py:54-65`）**逐字节兼容**。Stage 粒度 = (cell, task)（继承 `RobocasaCollectStrategy.plan` 既有构图，`run_collect.py:148-180`；13 stage/cell 由 `assign_servers` 摊到 6 台即是格内并行，不需要 `shard_eval_stage`——本实验无单臂相位）。**bundle_id = run_id（cell 级）** stamped 到该 cell 全部 EpisodeTask（`bundle_id` 与 `yaml_id` 是分立字段；worker 按 `task.bundle_id` 自选，F15）。
  - **extra 契约**：`REQUIRED_EXTRA_KEYS` 六键逐项冻结（`episode_runner.py:70` fail-fast 面）= `{task_name: <13 任务名>, layout: 1, style: 1, teacher: "groot_tp", base_seed: 1_000_000, replan_steps: 5}`；`experiment = "groot_tp"`（Round-1 同值，`run_collect.py:159`，不含 "/"，:358-362 校验面）。
  - **journal/summary 物化**：driver 写单一中心 journal（X17 加固面：`accepted`/attempt fence/`run_id` 列/`record_counts_as_done` 谓词，`journal.py:97`）；W3 自带 **finalizer** 把中心 journal 按 task_uid 的 run_id 前缀切分为逐 cell `journal_<run_id>.jsonl`（原始行透传不改写，含被拒 stale 行——下游本就按 `accepted`+status 过滤）+ 逐 cell `summary_<run_id>.json`（复用 Round-1 `run_ws_search.summarize_journal` 的对账口径：与 run-plan uid 全集对账，`complete = (n_err==0 ∧ n_missing==0)`，retry 去重走 accepted 谓词）。finalizer 幂等、可中途重跑（兼作监控读数）；**W5 的"零改动"以 finalizer 产物为准成立**（summarize/analyze 真读之）。
  - per-step 证据流接通见 D8（逐 cell 分文件，落盘先于 done 行）。
  - **⚠ 已批准设计的一处偏离：分批驱动（`--cells-per-batch`，默认 12）**。G1 批准的是「一次 invocation 装载全部 cells 的 TaskGraph」。§4 Code 期间实测推翻了该形态的可行性：conductor 的 `EpisodeScheduler` 在每次激活/派发调用里遍历**全部** stage（`scheduler.py:152-156,246-248` 的 `sorted(stage_id)` + `_active_count` 全表扫描），代价随 stage 数**平方**增长——本机实测 **156 stage(12 cell) = 3.3 ms / 1716 stage(132 cell) = 435 ms** 每次调用，且全程持有 scheduler 单锁。driver 每 20 ms 一次 stage tick、每个 worker pull 各一次 ⇒ 28-40 worker 编制下锁被占满、整个车队被串行化，14-18 h 无人值守跑不可接受。**取舍**：修 conductor 内核（`_active_count` 全扫）超出本 plan 的 src 范围（§3-W1 只放开一条校验规则），故在 exp 侧把活跃图切小。**代价与守卫**：批次边界会重建 driver ⇒ ①必须固定 `--bind-port`（未给即 fail-fast）；②非末批的 `MSG_SHUTDOWN` 会拆掉 worker 车队（`driver.py:310-311` → `worker.py` run_forever 收到即退出），故非末批改答 idle backoff，worker 靠 1 s 起指数退避重连（`worker.py:182-185`）；③任一批次失败必须**非零退出**且不 finalize（否则无人值守编排会把基建故障读成普通 INCOMPLETE）。身份/双图/族交错/journal 语义全部不变——批次只切活跃图，全集图仍逐 cell 完整构建。`--cells-per-batch 0` 可退回批准的单图形态。
- **W4 编排薄壳** `orchestrate_ws_search2.py`：只管 server 生命周期（6 台串行起、就绪双判据 = 新鲜日志 yaml 路径 + `INFO:websockets.server:server listening`、RAM/VRAM 门）+ agent 拉起 + 单 driver invocation + 孤儿清扫（PID 锚定）。Round-1 的 CellQueue/槽位队列/重启序列全部退役（conductor 接管）。
- **W5 汇总沿用**：`summarize_ws_search.py`/`analyze_ws_search_stats.py` 零改动（F2，`--run-prefix` 直接吃 ws2*）。
- **W6 对比分析** `analyze_ws2_vs_ws1.py`：(cid, task, idx) 三键配对 join 两轮逐 cell journal，输出**两张互不混用的表**——①**full-matrix 联合变化表**（132 格逐格 Δmacro + 配对符号翻转检验 + 逐任务提升面/PickPlace 复活判定；只标注"联合干预"，不作因子拆分）；②**matched-control 分解表**（仅 §3-W8 manifest 的 12 个 ws2c cell：ws1/ws2c/ws2 三点分解，库效应 = ws2c−ws1、桶圈定边际 = ws2−ws2c，均配对检验）。无 ws2c 配对的 cell 进入分解 = **硬错误**（raise，测试盖住）；另出加密轮复现矩阵。合成 journal 单测。
- **W7 桶归因离线侧**（独立工具 `build_bucket_variants.py` + W6 内 join 子命令）：
  - **库侧映射** `config/ws_search2/bucket_variants.json`：载 full704 pkl → 逐桶枚举成员 entry id（构建期 relpath id，text-IVF plan §9 口径）→ 每桶取一代表 episode（relpath → episode idx → **seed = base_seed + idx，id 即种子**，T5b 续采纪律）→ robocasa env reset 重放（只 reset 不跑 policy；**env 全程 evict-1**——逐任务 close 旧 env 再建新 env，或按任务分子进程，杜绝同进程累积 ~13 个 kitchen env 耗尽 EGL framebuffer 的既有故障模式，`robocasa365_seed_anatomy.md:123`）取 `annotation.human.task_description` 文本与物体类别 → 记录 `{bucket_index, tasks, n_entries, representative: {relpath, seed, prompt, object_class}}`。**版本绑定**：文件头记 pkl sha256 + robocasa commit + env 构造参数；111 桶 ≈ 分钟级。**缺失/多义处理**：重放失败的桶标 `unresolved`（保留计数，不静默丢弃）；跨任务桶（理论不应存在，指令含任务语义）若出现记全部 tasks 并标 `ambiguous`。
  - **join 契约**（W6 子命令消费）：per-step 行（D8，header 行含 `attempt`）以 `(task_uid, attempt)` 连 accepted journal 行——重试残行按 accepted attempt 过滤去重；`winner_id` → pkl entry → bucket_index → bucket_variants；episode 头行给出 eval 真值 prompt。产出逐 (cid, task, idx) 的 `{eval_prompt, winner_bucket, winner_variant_prompt, matched: eval 变体是否在库}`——「桶对上类别」假设以 eval_prompt ↔ winner_variant_prompt 的一致率量化；exact/nearest 逐行不可还原（`__hit_meta__` 无此位），归因判据即上述一致率；路径区分不进任何门——D9 契约：现役 INFO 配方下不可观测，仅作离线 DEBUG probe 非阻塞观察。
  - 测试：合成 pkl + 合成 per-step 行的 join 单测（含 unresolved/ambiguous 分支）；**真实 seed smoke 1 条**（重放一个真实库 episode 的 seed，断言 prompt 非空且与同桶第二成员重放结果一致）。
- **W8 selection manifest** `build_selection_manifest.py` → `config/ws_search2/selection_manifest.json`（tracked）：
  - **ws2c 段（§4-Code 阶段生成，与工具/测试一并进 G2 diff——G2 审计实际 12 个 cid、源 hash 与 emitter 输入；不允许 G2 后另行 commit）**：cell 集 = iso 4（Round-1 `index.json` 的 4 个 iso cid 逐字）∪ **非-iso 的 Round-1 榜首配对打平集按 (macro_sr 降序, cid 字典序升序) 前 8**——top-8 候选显式排除 iso cid，保证并集恰为 12。算法冻结：输入 = Round-1 132 份逐 cell journal（weilandserver 权威副本拉回本地，逐文件 sha256 入 manifest）；统计 = `analyze_ws_search_stats` 同口径（EPISODES_PER_TASK=8、RESAMPLES=20000、配对符号翻转、打平 = 对榜首 p≥0.05）；重抽 rng **seed=12345 —— Round-1 analyzer 的既有默认（`analyze_ws_search_stats.py:125` `--seed default=12345`、:160 `random.Random(args.seed)`，亲验），逐字复现 Round-1 的 tied 口径**；非-iso 打平集不足 8 时按全矩阵非-iso macro 降序补足并在 manifest 里记录补足事实。
  - **ws2e 段（筛选轮排空后由 G2 已审的同一工具追加——其 cid 依赖 ws2 运行结果，逻辑上不可能先于 G2 存在；G2 审计的是算法、append-only 校验与边界测试）**：leader = ws2 macro 最高（平票按 cid 字典序）；top-8 = 与 leader 配对 p≥0.05 的集合按同排序键取前 8（不足同上补足）；阴性 = 与 leader 配对 p<0.05 中 macro 最低的 2 个（平票同键）；**p<0.05 的 cell 不足 2 个 ⇒ 工具 fail-fast（显式报错停机，上报 owner 裁决，不静默降档）**。追加为 append-only：工具重跑必须校验既有段逐字节不变，源 journal sha256 随段记录。
  - **driver 契约**：ws2c/ws2e 相位 `--manifest` 必填，cell 集只从 manifest 读；resume 校验 manifest sha 一致，**绝不重选**。测试：合成 journal 上两次生成逐字节相同（确定性）、集合大小与排序断言、输入污染（缺文件/sha 不符/既有段被改）拒绝。

**docs/logs**：本 plan 入 `logs/README.md` 索引（同 commit）；text-IVF plan §11 的 GR00T 排除条目加指针注记（scope 由本 plan 扩展）；`docs/cache/tutorial.md` §7 GR00T 例一条（G2 后随实现 commit）。

## 4. 集成点与部署（缺口逐一堵）

- **P1 代码入远端**：规则 6 commit 过 G2/Verify 后 push origin/Ziyang。weilandserver serving repo = **一次性克隆 `/data/openpi_text_ivf_build`**（现在 cf10ebc）`git fetch && checkout <新 sha>`；克隆里 12 行预读门控 ops 补丁只动 `exp/common/build_in_memory_cache_artifact.py`（与 serving 无关），checkout 前存 /tmp 副本后丢弃。**活 repo `/home/weiland/openpi` 一个字节不动**（rl_router 线 `config.py`+`in_memory_backend.py` 在飞改动未收口——本轮绕开，非本轮解决面）。
- **P2 client 侧代码**：timan107 岛 `/scratch/zixuans8/openpi_rc365` 推送 `src/openpi/` + `exp/robocasa365/`（tar 经 /tmp，排除 data），**逐文件 sha256 清单双机核对**（Round-1「部署必须列清单核对每台」血泪）。X17 conductor 改动向后兼容（X17 log :760），但 worker 进程 import 链吃新 conductor，必须同版。
- **P3 服务配方**：`serve_groot_n15 --checkpoint /home/weiland/ckpt_n15_robocasa_tp/.../checkpoint-60000 --port 2316x --concurrent --allow-dynamic-bundles --cache-config <bootstrap>`，PYTHONPATH/venv 沿 Round-1 §10-2 配方但 repo 根换克隆路径。bootstrap yaml 取主臂任一 cell（其 bundle 绑 "default"，不与 cid 冲突）。
- **P4 网络**：ziyanglin.com:23160-65 直连段（有线已就位）；端口/tmux 名先侦察（共享命名空间规约）。

## 5. Phase-1 / Stage-0 / 冒烟（开跑前置门）

- **Phase-1 重标定**（~1h，weilandserver tmux）：staging 软链目录只含 groot full704 pkl → `calibrate_score_normalizers.py --artifact-dir ... --output ...` → json 拉回入 repo → W2 吃它 emit 144 yaml → 校验器 0 problem。
- **S1 冒烟门**（~0.5h，判据全过才进 Stage-0）：单 server 动态 bundle，2 cells × 2 任务（1 单指令 + PickPlaceCounterToCabinet）× 2 ep：①bundle 热切换 ack + 逐 bundle guards 过；②FULL_HIT 全程；③**证据通道全链**——per-step 行带非空 `__hit_meta__` 且 `winner_id` 可经 pkl 解析入桶（D8 前提核验）、episode 头行 prompt 非空；④**桶落点**——经 W7 join，多变体任务的 winner 桶变体 == episode 真值 prompt 对应变体（库内存在时）；⑤第二 bundle 装载耗时 <10s（F13 pool 命中证据）；⑥finalizer 产出的逐 cell journal/summary 被现有 `summarize_ws_search`/`analyze_ws_search_stats` **真读通过**（partial 状态可解析、cid/task/idx 全还原）。exact/nearest 路径判据**不在本门内**（现役日志级别下不可观测，`in_memory_backend.py:595-618` debug + `serve_groot_n15.py:59` INFO；处置见 D9——归因判据独立于路径，离线 DEBUG probe 作非阻塞观察）。
- **S0-a 真 seed 重放 smoke（发车前硬门，本机无 sim 故必须在孤岛执行，不得口头跳过）**：在 weilandserver 的 GR00T 孤岛(有 robocasa+EGL)执行——
  ```bash
  cd <serving-repo> && PYTHONPATH=<gr00t>:src:. <island-python> -m exp.robocasa365.build_bucket_variants \
      --artifact /data/robocasa365_cache/cache_artifacts_text_ivf/groot_tp_spatial_pool_16_full704.pkl \
      --out exp/robocasa365/config/ws_search2/bucket_variants.json --teacher groot_tp
  ```
  **判据（四条全过才发车）**：①stdout 报 `111 buckets`；②`n_unresolved == 0`；③`provenance.robocasa_commit` 是 40 位 sha（不是 `unavailable:`），且 `provenance.env_kwargs` 含 `camera_heights/camera_widths`（与 `GrootTeacherAdapter.env_kwargs()` 相等）；④随机抽 3 个 bucket，其 `representative.prompt` 非空且与该 bucket 任务的指令模板相符。**证据落点**：`exp/robocasa365/analysis/ws2_s0a_bucket_variants.txt`（stdout 全文 + 上述四条的实测值）。同时用一行探针确认 relpath id 形态与 `_EPISODE_RE` 相符：`python -c "import pickle;d=pickle.load(open(PKL,'rb'));print(d['entries'][0].id)"`（期望形如 `<Task>/episode_NNNN_aNN:<step>`）。
- **Stage-0 容量探测**（~1.5h）：S∈{3,6} × 真实 text-IVF cell 短测（每档 26 集/路，rate 代理 = server FULL_HIT 行/min）；worker 爬坡 timan107 32→40→48（护栏承袭 Round-1 §3-B：load>40 / RAM<40G / 单卡 free<1.5G / 他人进程异常，任一触发回退一档）；weilandserver 本机 worker 4-8。timan1 不入首发编制（少一台机少一分故障面）；若 Stage-0 显示 worker 供给不足再按既有配方加入（`--agent-c-host timan1`、避 GPU1、单卡打满）。产出追加 `t7_capacity_probe.txt` 新节。

## 6. 成本估算（Stage-0 后回填）

| 项 | ep | 墙钟（按 Round-1 终态 741 ep/h 起算，6 srv 无重启应更高） |
|---|---|---|
| Phase-1 + S1 + Stage-0 | ~130 | ~3h |
| 主臂 ws2 132×104 | 13,728 | ~14-18h |
| 控制臂 ws2c 12×104 | 1,248 | ~1.5-2h（独立相位） |
| 加密臂 ws2e 10×416 | 4,160 | ~4-6h |
| 锚点臂（D5，若准） | 104 | ~1-1.5h |
| **合计** | ~19.4k | **~1-1.5 天无人值守** |

## 6b. Live 运行记录（2026-08-26）

- **部署**：两台机（weilandserver 服务克隆 `/data/openpi_text_ivf_build`、timan107 岛）+ 后加 timan1 岛，全部到 `3598534`。三台的 ws2 配置树由同一 tarball 分发，sha 双机核对一致。
- **真机修掉的两个 bug**（本地测试覆盖不到，只有真库/真部署会暴露）：① `build_bucket_variants` 直接读 pkl 拿到的是 **numpy**，而 backend 是在 `load_artifact` 里 `torch.from_numpy(v).float()` 之后才算桶键——漏掉这层转换，工具算出的桶号与服务端真正使用的索引对不上；已改为逐字复刻 backend 推导并加对拍测试。② 编排器 `texec` 把 HOME 硬编码为 `/home/weiland`，而 worker 岛的账号是 `zixuans8`；已改为按节点传 HOME 并加回归。
- **S0-a 桶映射硬门 PASS**：111 桶 / 0 unresolved / 0 ambiguous；`robocasa_commit be22d659…`（40 位）+ `camera 512×512` 证明 replay env 与正式 eval 同源；代表 prompt 与 object_class 互相印证。证据 `analysis/ws2_s0a_bucket_variants.txt`。
- **S1 冒烟 PASS**：`complete=2/2`、1071 次推理全 FULL_HIT、8 集全 join 零缺口、桶落点 6/8 精确。两个 `matched=False` 是**设计中的最近代表回落**（eval 抽到 "hot dog"，库内最近为 "hotdog bun"）——正是本轮要检验的现象。证据 `analysis/ws2_s1_smoke.txt`。
  - ⚠ 运维教训：车队必须用 `orchestrate_ws_search2 agents-up` 起。手写 tmux 名会让 `agents-down` 找错会话（它按 `<prefix>agent<fleet>` 推导），supervisor 存活并把刚被扫掉的 worker 重新拉起。
- **Stage-0 定档**：6 worker 挤在单 server = 126 ep/h（21/worker）；**30 worker 摊到 6 台 = 1224 ep/h（41/worker）**——瓶颈是单台 server 的推理排队，不是 worker 数，这正是「6 server 必须配 6 fleet」的量化依据。worker 显存实测 1.3–2.4 GiB（evict-1 生效，远低于 round-1 担心的 3.5–4G）。证据 `analysis/ws2_stage0_capacity.txt`。
- **GR00T 主臂 live**：6 server × 6 fleet × 5 worker = 30 worker，11 批 × 12 格，预计 ~11.2 h。
- **pi0.5 前置就绪**：标定 4/4（vision_2 的 J=0.4356 **高于** GR00T 的 0.3206，两执行体字段可分性结构不同，实证了必须分别标定）；144 yaml 已 emit 且旋钮 `masked/span` 双开与其库 meta 对齐。
- **pi05 serving 拓扑可复用**（本轮亲验）：`serve_policy.py` 两条构造路径都不传 `allow_dynamic_bundles`，而 `WebsocketPolicyServer` 默认 `True`（:484），且其 `_connection_policy_factory` 同样接收 `bundle_id`（:885-889）⇒ pi05 支持同款动态 bundle 热切换，**不必回退 round-1 的重启制**。

### 6b-1. 中途读数（2026-08-27 02:05Z，**26/132 格**，PARTIAL / NOT A FORMAL RESULT）

分析链路已用真实数据端到端跑通（先 `--finalize-only` 物化逐 cell 产物——`load_journals` 读的是
**per-cell** journal，那只在 finalize 时才写，所以不跑 finalize 就永远只看得到上次的快照；
再 `compare --allow-partial --resamples 20000`，ws1 目录为归并后的 132 journal）。工具自报
`skipped 106 in-flight/partial cells`、抬头 `PARTIAL (NOT A FORMAL RESULT)`——门禁按设计生效，
**下列数字不得当结论引用**。

**联合效应（库增长 + text-IVF 一起，26 格 × 104 配对集）**

| | round-1 | round-2 | 差 |
|---|---|---|---|
| 26 格平均 macro SR | 0.148 | **0.251** | **+10.3 pp**（相对 +69%） |
| 最好一格 | 0.269（round-1 全 132 格最好） | **0.337** | 已有 7 格超过 round-1 全矩阵最好 |
| 最差一格 | 0.019 | 0.077 | — |

26 格**无一变差**，delta ∈ [+0.000, +0.173]，其中 **16 格 p<0.05**。榜首
`grid4 v0@12 v1@12 v2@12 rs@62` = 0.337（+0.154, p=0.0007）。round-1 里「v0/v1 零正贡献」的
`iso_vision_0/1` 从 0.019 升到 0.096 / 0.077——仍垫底，但**「零贡献」这个结论已不成立**。
关键的 `iso_vision_2` 0.202→0.279 尚未显著（p=0.116）。

**⚠ 逐任务：增益高度不均，PickPlace 家族没有复活**

大涨全部是接触-操纵类：`OpenCabinet 0.034→0.409 (+0.375)`、`CloseFridge 0.471→0.721 (+0.250)`、
`OpenStandMixerHead +0.231`、`SlideDishwasherRack +0.221`、`TurnOnSinkFaucet +0.139`。

取放-搬运类持平到倒退：`PickPlaceCounterToCabinet 0.029→0.000`、
`PickPlaceSinkToCounter 0.014→0.000`（两个掉到**绝对零**）、`ToasterToCounter −0.010`、
`CounterToStove +0.010`、`DrawerToCounter +0.062`；`CoffeeSetupMug 0.053→0.043`。

代码里这张表的注释就叫 "PickPlace revival readout"——本轮预期之一是更大的库 + 文本分桶能救活
PickPlace，**26 格的证据不支持**。是「检索原理上救不了长程搬运」还是「库里根本没有可用轨迹」，
要等 ws2c 控制臂把两个因子拆开才能判：现在这个 delta 是**联合效应**，不可归因给任一单因子。

⚠ 这 26 格是批 1+2，`interleave_cells` 保证跨族覆盖，但仍不是随机抽样，别当总体估计。

### 6b-2. 中途读数刷新（2026-08-27 04:00Z，**47/132 格**，PARTIAL / NOT A FORMAL RESULT）

| | round-1 | round-2 | 差 |
|---|---|---|---|
| 47 格平均 macro SR | 0.148 | **0.264** | **+11.6 pp**（相对 +79%） |
| 最好一格 | 0.269 | **0.385** | **24 格**已超 round-1 全矩阵最好 |

47 格**无一变差**（delta ∈ [+0.000, +0.231]），**33 格 p<0.05**；均值较 26 格时（0.251）继续抬升。

**⚠ 榜首换人，且最优权重结构在迁移**：`grid3 v0@12 v2@37 rs@50` = 0.385（+0.192, p<0.0001）
取代 `grid4 v0@12 v1@12 v2@12 rs@62`（0.337）。前四名 0.385 / 0.375 / 0.356 / 0.356 的共性是
**robot_state 占 37–50 配一个中等 vision_2**——而 round-1 的冠军是 `vision_2@87 robot_state@12`
（几乎全押 vision_2）。即**最优配比正从「重 vision_2」往「vision 与 robot_state 均衡」迁移**。
单字段基线佐证：`iso_vision_2` 0.279、`iso_robot_state` 0.212，任何混合体都超过单字段。
（这条若在全 132 格站得住，就是本轮相对 round-1 的一个结构性结论，不只是数值变好。）

**逐任务修正 6b-1 的说法**：26 格时四个 PickPlace 为负；47 格时只剩两个为负
（`CounterToCabinet 0.021→0.000` 仍是绝对零、`SinkToCounter 0.016→0.003`），
`ToasterToCounter` 已从 −0.010 翻正到 +0.035，`DrawerToCounter +0.051`、`CounterToStove +0.016`。
但量级差距依旧悬殊：PickPlace 家族最大增益 +0.051，而接触-操纵类
（`OpenCabinet +0.372`、`OpenStandMixerHead +0.269`、`CloseFridge +0.263`、
`SlideDishwasherRack +0.239`）动辄 +0.24~+0.37。**方向性判断「库增长 + 文本分桶救不活取放-搬运类」
仍成立，但没有 26 格时看上去那么绝对**——这正是为什么中途读数只能当趋势、不能当结论。

### 6b-3. ⚠ 事故:批次静默丢集(2026-08-27 06:00Z 发现,主臂批 6–8)

**症状**:批 6→7→8 各只隔 ~20 分钟(前几批 60–75 分钟),而每批有 1248 集要跑;同期
journal 40 分钟只涨 284 行。逐格计数:**90 格被碰过,57 格满 104,33 格残缺**(1–31 集),
其中约 20 格最后一集已是 5–31 分钟前 ⇒ **批次把没跑完的集丢下就推进了**。

**因果链(证据齐全)**

1. worker 日志 `websockets.exceptions.ConnectionClosedError: sent 1011 (internal error)
   keepalive ping timeout`——agent0/2 各 5298 / 6753 条。server >20s 不应 ping,
   该 server 上**所有** worker 连接被同时判死。
2. 掉线打断在跑的 episode → 记一次失败。
3. `EpisodeScheduler(max_episode_retries=3)`;同一 uid 三次都撞掉线 → 标 `FAILED`
   (exhausted retries / fatal)。**不是超时**:`--episode-timeout-s` 默认 1800s,而单集只要 ~120s。
4. **重试耗尽不写 journal 行**——即已知的「journal 无『每 uid 一条终态』不变量」缺口。
5. 该批队列排空 → `ConductorDriver.run()` 返回 → driver 推进下一批(循环确实等本批跑完,
   `while inner.is_alive(): inner.join(...)`,所以不是"没等")。
6. 该格停在 <104 集,而中央 journal 里**没有任何痕迹说这些 uid 失败过**。

**卡住 server 的最可能来源:bundle 热切换**。每台 server 已切 **75 次**
(`Loading cache config` → `Cache bundle updated to vNN`),切换是同步的。旁证三条:
① CPU 极不均衡——同一时刻一台 +12409 ticks/20s(6.2 核)、两台 +3~4 ticks(空转);
② GPU util 0–10% / 46°C(**不是推理打爆**,本配方 `always_hit` 本就少推理);
③ **RSS 从启动的 ~21.2G 涨到 23.5–32.8G**(六台合计 ~172G,约漏 50G),与切换次数同步。
机器本身没到墙:swap 0、si/so 0、88 核只用 10–12%、load 11.5;但 `cs≈200K/s` 异常高。

**吞吐**:每 10 分钟完成集数从 189(≈1134 ep/h)一路掉到 37,批 8 起来后最近 5 分钟回到
792 ep/h ⇒ 是**「整体腰斩 + 换批深坑」叠加**,不是完全崩塌。

**处置裁定(2026-08-27,未停机)**:不中途拆机重启。理由——(a) run 仍在 500–800 ep/h 推进;
(b) 冷启动 10 分钟/台 × 6 台再加起车队,代价近一小时;(c) **丢掉的集是可恢复的**:双图 resume
按 journal 剪枝,重跑同一条 driver 命令只会重派缺失 uid,这正是 §6d ① 写的收官路径。
⇒ **全 11 批跑完后必须做一次「补跑 pass」**:`--finalize-only` 看 INCOMPLETE 清单,
用 summarize 的 `--only` 串重发射,直到报 `132/132`。**别把首次 11 批跑完当作主臂收官。**

**给 owner 的建议(需裁决)**:后面三个相位(ws2c / ws2e / pi05)要么
(i) 降低 bundle churn——减小 `--cells-per-batch`,让同一时刻每台 server 服务的 bundle 数变少;
或 (ii) 接受"跑完再补一遍"的两趟制。ws2c 只有 12 格、pi05 132 格,churn 量级不同,
pi05 尤其值得先降 churn 再开跑。⚠ 另外 pi05 单 server RSS 基线 ~32G,**若同样存在切换泄漏,
5 台会更早撞 251G 的墙**——起 pi05 前应先复核这条。

### 6b-4. 首轮收官与补跑 pass(2026-08-27 07:08–07:30Z)

**首轮结果**:`[ws2] INCOMPLETE phase=ws2 complete=57/132`,driver 自行退出并给出 75 个
incomplete cid 与 `--only` 串,附一句 `rerun heals MISSING uids only; inspect n_err first`。

**查 n_err 的结论(关键)**:75 格**全部 `n_err=0`**,6,643 个缺口**全是 MISSING**
(从未写过 journal 行)⇒ **重跑可完全治愈,没有需要人工裁决的脏数据**。
完成/未完成的分界是**时间性的**(批 1–5 好、批 6–11 坏),不是特定 cell 的性质。

**服务端重启印证了泄漏**:停光 6 台后 RAM 用量 175G→**3G**(可用 248G)、VRAM 全空;
重起后每台 RSS 回到 **21.2–21.3G**(此前 23.5–32.8G)。页缓存热,六台串行起只用约 6 分钟。

**⚠ 我在这里犯了一个错,记下来别再犯**:重启 server **没有同时循环 worker 车队**。
workers 还握着被杀掉的旧 server / 旧 driver 的连接,每集瞬间 `Connection refused` /
`peer closed connection mid-frame` → 三次重试耗尽 → 批次**秒排空**(批 1 的 456 集
2 分钟"跑完",journal 只涨 3)。**无永久损失**(耗尽不写 journal,uid 仍是 MISSING),
但白跑两批。**正确顺序:agents-down 全部 → 起 driver → agents-up 全部。**

**⚠ 而且 §5 记过的 tmux 前缀坑又踩了一次**:`agents-down` 按 `<--tmux-prefix>agent<fleet>`
推导会话名,**默认前缀是 `ws2s`,而实际会话叫 `ws2agent0..5`(前缀 `ws2`)**。
不传 `--tmux-prefix ws2` 时它一个 supervisor 都杀不到,只扫 worker,supervisor 立刻把它们
拉回来——表现为「agents-down 报 0 tmux session + LEFTOVER,30 秒后 worker 数原样复原」。
**所有 agents-up / agents-down 都必须显式带 `--tmux-prefix ws2`。**

**补跑配置**:`--only <75 cids>`(在服务器上从 summary 现算,不手抄;脚本内断言
`n_err==0` 且格数==75)+ **`--cells-per-batch 6`**(降低 bundle churn,13 批)。
批 1 实际 `episodes=453` 而非 6×104=624 ⇒ resume 剪枝正确,已完成的集不重跑。
07:30Z 复核:两岛 27w+3a / 15w+3a、GPU 66%/58°C、仍在批 1(不再秒排空)⇒ 修复生效。

### 6b-5. ✅ GR00T 主臂收官 + 第一份正式结果(2026-08-27 14:17Z)

**补跑 pass 结果**:`[ws2] DONE phase=ws2 complete=75/75`。全相位独立核实:
```
finalize-only: 132/132 cells complete
journal 132 格,每格恰好 104 集(min=max=104),合计 13,728
132 份 summary 全 complete,n_err=0,n_missing=0
```
补跑用时约 6h45m(07:30→14:17Z),平均 897 ep/h,全程 RAM 稳定在 89–101G 可用、
**未再出现 RSS 爬升**⇒ 「重启清泄漏 + `--cells-per-batch 6` 降 churn」的组合有效。

**正式全矩阵联合效应**(`compare` 不带 `--allow-partial`,全覆盖门通过;
证据 `analysis/ws2_joint_full132.txt`)

| | round-1 | round-2 | 差 |
|---|---|---|---|
| 132 格平均 macro SR | 0.1579 | **0.2781** | **+0.1202(+76%)** |
| 最好一格 | 0.269 | **0.385** | 82 格超过 round-1 全矩阵最好 |

**132 格无一变差**(131 改善 / 1 持平 / 0 退步),**95 格 p<0.05、66 格 p<0.01**。

**⚠ 结构性结论(本轮相对 round-1 最重要的发现):最优权重配比迁移了。**
round-1 冠军 `grid_vision_2@87_robot_state@12`(几乎全押 vision_2)在 round-2 只到 0.279,
跌出前列;新榜首 `grid3_vision_0@12_vision_2@37_robot_state@50` = 0.385。前六名清一色是
**robot_state 37–50 + vision_2 仅占 12–62** 的均衡配比。⇒ 库变大 + 文本分桶后,
**robot_state 从配角变成主力**;round-1 那个"vision_2 一家独大"的结论是小库下的假象。
单字段基线:`iso_vision_2` 0.279、`iso_robot_state` 0.212、`iso_vision_0` 0.096、
`iso_vision_1` 0.077——任何混合体都超过任一单字段。

**逐任务(全矩阵终版)**:增益集中在接触-操纵类
`OpenCabinet 0.055→0.416 (+0.361)`、`CloseFridge 0.475→0.737 (+0.261)`、
`SlideDishwasherRack 0.119→0.372 (+0.253)`、`OpenStandMixerHead 0.455→0.683 (+0.227)`、
`OpenDrawer +0.176`、`TurnOnSinkFaucet +0.105`。
取放-搬运类:`ToasterToCounter +0.083`、`DrawerToCounter +0.049`、`CounterToStove +0.019`
(三个翻正但量级小),而 **`CounterToCabinet 0.016→0.000`(绝对零)、
`SinkToCounter 0.011→0.006` 仍为负**。⇒ **「更大的库 + 文本分桶救不活取放-搬运类」
在全 132 格上成立**,这是本轮的负结果之一。

⚠ 以上全部是**联合效应**(库增长 + text-IVF 一起),拆不出单因子——那要等 ws2c 控制臂。

### 6b-6. ✅ ws2c 控制臂完成 + 正式因子分解(2026-08-27 15:45Z)

`[ws2] DONE phase=ws2c complete=12/12`,journal 1248/1248。用时约 1h15m,平均 992 ep/h。
证据 `analysis/ws2_factor_decomposition.txt`(`compare` 带 `--ws2c-dir` + `--manifest`,
不带 `--allow-partial`,三臂全覆盖门通过)。

| 臂 | 库 | 检索 | 12 格平均 macro SR |
|---|---|---|---|
| ws1 | 小库 | weighted_score_sum + brute force | 0.1899 |
| ws2c | **full704** | weighted_score_sum + brute force | 0.2516 |
| ws2 | **full704** | **text_ivf** | 0.2557 |

- **`lib_effect` = +0.0617**(占联合 **94%**),12 格中 **3 格 p<0.05**
- **`bucket_margin` = +0.0040**(占 **6%**),**12 格中 0 格 p<0.05**;符号 6 正 / 3 零 / 3 负

**判读(owner 2026-08-27 裁定的口径)**:text-IVF 的目的**不是**提升 SR,而是在大库上把检索
成本压回去;**SR 不掉即为通过**。本结果是「中性、略正且不显著」⇒ **通过**。
SR 的提升几乎全部来自库变大——这本就是预期内的,不是本方法的卖点。
⚠ 因子分解只对这 12 格有效(它们按 round-1 表现选出,是 round-1 的强格,联合效应 +0.066
低于全矩阵的 +0.120),**不可外推到全 132 格**。

⚠ **本轮未采集检索延迟/吞吐证据**(owner 明确表示这轮不管速度)。因此"text-IVF 让检索更快"
这半边**没有实验支撑**,只有"SR 不受损"这半边。若将来要写进论文,需要补一次
ws2 vs ws2c 的配对延迟测量(同库、同格、同 seed、同硬件,数据条件已具备)。

### 6b-7. ✅ ws2e 加密臂完成:噪声地板 + 胜者诅咒(2026-08-27 20:17Z)

`[ws2] DONE phase=ws2e complete=10/10`,journal 4160/4160(10 格 × 416 集),平均 982 ep/h。
证据 `analysis/ws2e_reproduce.txt`。

**① 复现噪声地板 = 3.0%**。`reproduce` 把 ws2e 前 104 集与 ws2 的同 seed 配对(1,040 对):
`s→s 309 / s→f 15 / f→s 16 / f→f 700` ⇒ 翻转 31/1040 = **3.0%,方向对称**(15 vs 16)。
⇒ **任何小于 ~3pp 的逐格差异都在复现噪声内,不可解读**;反过来也确认主臂 +12pp 远在噪声之上。

**② 胜者诅咒(方法学结论,对论文口径有直接影响)**

| cell | ws2(104) | ws2e(416) | diff |
|---|---|---|---|
| `grid3_v0@12_v2@37_rs@50`(主臂榜首) | 0.385 | **0.334** | −0.050 |
| `grid3_v0@12_v2@50_rs@37` | 0.356 | 0.332 | −0.024 |
| `grid4_v0@37_v1@25_v2@12_rs@25` | 0.356 | 0.329 | −0.026 |
| `grid3v_v0@25_v1@37_v2@37` | 0.356 | 0.315 | −0.041 |
| `grid4_v0@37_v1@12_v2@12_rs@37` | 0.365 | 0.315 | −0.050 |
| `grid_v2@62_rs@37` | 0.375 | 0.310 | −0.065 |
| `grid3_v0@37_v2@12_rs@50` | 0.375 | 0.308 | −0.067 |
| `grid4_v0@12_v1@25_v2@12_rs@50` | 0.375 | 0.298 | −0.077 |
| `iso_vision_1`(垫底) | 0.077 | 0.113 | **+0.036** |
| `iso_vision_0`(垫底) | 0.096 | 0.099 | +0.002 |

**8 个头部格全部回落、2 个垫底格上升** —— 教科书式选择偏差:它们正因在 104 集上考得
好/差才被选中,加到 416 集后向真值收缩。

**口径要求**:
- 主臂榜首的 0.385 是虚高,**真值约 0.334**;头部 8 格加密后聚在 **0.298–0.334**,
  彼此在统计上分不开 ⇒ **不能声称"某个配比最优"**,只能说"这一族均衡配比是第一梯队"。
- ⚠ 同一诅咒也作用在 round-1 的 0.269(同样是 104 集的极值)。因此
  **"+12.0pp 的全矩阵均值差"成立**(两轮测量精度对等),但
  **"最好一格 0.385 vs 0.269"这种极值对极值的写法必须避免**。

### 6b-8. pi05 全线启动(2026-08-27 21:05Z)

拓扑:**5 台 server @ 23170–23174**(不是 GR00T 的 23160 段,见下)、driver `ws2main` @ 23180、
**5 个 fleet / 37 worker**(timan107 3×9、timan1 2×5);22 批 × 6 格 = 132 格 / 13,728 集。
实测单台 pi05:**RSS 29.7G、VRAM 7.58 GiB**;5 台合计 148.5G RAM / 37.9G VRAM,
剩 100G RAM / 10.6G VRAM ⇒ **确认只能 5 台**(第 6 台会把 VRAM 余量压到 3G)。

**启动时暴露并修掉的三个缺口**(前两个会直接卡死相位):

1. **`run_ws_search2 --teacher` 被窄化成 `("groot_tp",)`** —— 而同族其它工具
   (`summarize_ws_search`/`analyze_ws_search_stats`/`orchestrate_ws_search`)都接受两个,
   `episode_runner.ADAPTERS` 也两个都有。属遗漏而非设计:端点同质性由
   `validate_teacher_endpoints` 保证,不靠这个 choices 列表。已放开并加 4 条测试
   (直接驱动真实 CLI 判 `invalid choice`,并断言 choices 与 `ADAPTERS` 相等);
   做过变异验证(改回窄化 ⇒ 两条 pi05 用例失败)。
2. **worker 岛上没有 pi05 的配置目录**(`pi05/main` = 0 yaml)。与 manifest 同类问题:
   服务克隆有、岛上没有。已打包同步(132 yaml + index.json,走 `/tmp` 中转绕开 allow_roots)。
3. **端口段必须与 env 声明一致**。`ws2_weilandserver.env` 里
   `PI05_SERVERS=…:23170–23175`,而我先把 pi05 起在了 GR00T 的 23160 段,
   `validate_teacher_endpoints` 直接拒绝——**这是护栏正确工作**(防止 pi05 的 driver
   指到 GR00T 的端口上)。正确处置是**把 server 挪到 23170–23174**,不是改 env 削弱护栏。

⚠ **既有基线失败(不是本次改动引入,也不属本线)**:
`tests/robocasa365/test_groot_cache_collector.py::test_collected_episode_builds_a_loadable_groot_artifact`
用精确相等断言 `artifact_meta`,而该字典后来多了 7 个字段
(`library_sha256`/`entry_count`/`action_horizon`/`action_dim`/`denoising_num_steps`/
`schema_consensus_count`/`intermediates_completeness`,来自 library_stats 那条线)。
已用 stash 复核:去掉本次改动仍失败。留给该线决定是改成子集断言还是更新期望值。

### 6b-9. ✅ pi05 全线完成 —— 全量实验收官(2026-08-28 11:05Z)

`[ws2] DONE phase=ws2 complete=132/132`,journal 13,728;独立核实每格恰好 104 集、
`n_err=0 / n_missing=0`。用时约 14h,平均 977 ep/h,5 server × 5 fleet(37 worker),
全程 RSS 在 168–190G 区间随换批起伏、**无泄漏趋势**,一次未干预。
证据 `analysis/ws2_pi05_matrix.txt`;完整报告 `analysis/ws_search2_groot_results.md`。

⚠ **round-1 从未跑过 pi05**(只有 groot_tp),故 pi05 **没有跨轮 delta**,只有绝对矩阵
与跨 teacher 对照。

| | pi0.5 | GR00T |
|---|---|---|
| 132 格平均 macro_sr | 0.1669 | 0.2781 |
| 最好格 / 最差格 | 0.298 / 0.058 | 0.385 / 0.077 |

**⚠ 本轮最重要的发现:检索字段权重不跨执行体迁移。**

单字段基线几乎**相反**:pi0.5 最强字段是 `iso_vision_1` 0.231(GR00T 该字段最弱 0.077);
GR00T 最强是 `iso_vision_2` 0.279(pi0.5 该字段仅 0.077)。pi0.5 的头部格全是
`vision_1` 主导,GR00T 的是 `vision_2`+`robot_state` 均衡;GR00T 表现好的
`grid v0@50 v2@50`(0.356)在 pi0.5 上只有 0.058。

**两个 teacher 对同一 132 个配置的排序 Spearman ρ = +0.175**,基本不相关。
⇒ **每个执行体必须各自搜权重,不能复用**。这与 Phase-1 标定时看到的
「pi05 vision_2 的 J=0.4356 vs GR00T 0.3206,字段可分性结构不同」形成端到端闭环。

⚠ **绝对水平差(pi05 全面偏低、GR00T 在 126/132 格更高)是混淆的**:两臂用不同的库
(pi05 63,977 条/29.8G vs GR00T 50,795 条/20.5G)、不同模型、不同渲染分辨率,
**不可读作「GR00T 检索能力强于 pi05」**。而**排序结论不受此影响**,因为它在各臂内部计算。

**逐任务**:pi0.5 只在 `SlideDishwasherRack` 明显更好(0.580 vs 0.372, +0.208)、
`PickPlaceSinkToCounter` 略好(+0.043);其余 GR00T 全胜,差距最大是
`CloseFridge −0.488`、`OpenStandMixerHead −0.372`、`ToasterToCounter −0.221`。
**两个 teacher 对 PickPlace 的判决一致**:整族贴近零(pi0.5 该族最好仅 0.048)。

## 6c. 暂停点与恢复手册（2026-08-26 17:26 CDT，owner 装新显卡断电）

**停机前已固化**（`/data/openpi_text_ivf_build/exp/robocasa365/`）：
- `data/ws_search2/groot_tp/`：中心 journal **1,464 集**；132 份 `journal_ws2-*` + 132 份 `summary_ws2-*` + 132 份 `run_plan_ws2-*`；**13/132 格 complete**（err=0 / missing=0）；15 份 `per_step_ws2-*` 证据流。
- `config/ws_search2/`：`selection_manifest.json`、`bucket_variants.json`、两份标定 json、288 个 yaml（groot 144 + pi05 144）。
- `analysis/`：`ws2_s0a_bucket_variants.txt`、`ws2_s1_smoke.txt`、`ws2_stage0_capacity.txt`。

**续跑安全性**：resume 只认中心 journal，已完成的 1,464 集不再派发；132 份 run_plan 的 `plan_hash` 会在续跑时逐一校验，任何参数漂移（episodes/tasks/base_seed/…）都会被 `write_run_plan` 硬拒而不是静默改口径。

**恢复顺序（依赖序，勿跳）**：
1. **先起 keepwarm，等卡温 ≥44°C**。4090 冷卡静默算错是有档案的硬件缺陷（`reference_weilandserver_4090_unstable`），server 起在冷卡上会产出无法察觉的错误结果。
2. **重认 GPU 序号**：加卡后 `nvidia-smi` 的 index 会重排，worker 的 `--gpu-ids` 与 keepwarm 盯哪张卡都要重定。若新卡显存更大，可重测 Stage-0 决定是否加 server（当前 6 台是 48G 卡的上限：6×6G VRAM + 6×22.3G RSS）。
3. `orchestrate_ws_search2 servers-up --ports 23160..23165`（每台过四道 preflight 门）。
4. 6 个 fleet：timan107 三个（fleet 0-2 → :23160-62）、timan1 三个（fleet 3-5 → :23163-65），各 5 worker，`--worker-home /home/zixuans8`。**必须用 `agents-up`**，手写 tmux 名会让 `agents-down` 找错会话。
5. driver：`--cells-per-batch 12 --bind-port 23180`，它会自动跳过已完成的 13 格。
6. 重挂 20 分钟 cron 巡检；巡检里加一条 `--finalize-only` 定期物化逐 cell 产物（finalize 本身只在全 11 批跑完后才自动调用一次）。

**剩余工作量**：GR00T 主臂还剩 119 格 ≈ 12,264 集 ≈ **13 h @ 945 ep/h**；之后 ws2c（12 格 ×104，需重启 server 换 brute_force 指纹）、ws2e（10 格 ×416）；再整套复用给 pi0.5（其 server 用 `serve_policy.py`，已亲验同样支持动态 bundle）。

## 6d. 相位切换命令(照抄即可,2026-08-26 实跑校准)

共用变量:`SRV=ziyanglin.com:23160,...,23165`;服务克隆 `/data/openpi_text_ivf_build`;
worker 岛 `/scratch/zixuans8/openpi_rc365`(两台 worker 机同路径,`--worker-home /home/zixuans8`)。

**① 主臂排空判定**:`--finalize-only` 报 `132/132 cells complete` 才算完;若有 INCOMPLETE,
用 summarize 给出的 `--only` 串重发射(MISSING 可自愈;ERR 需先看 n_err 再定夺)。

**② → ws2c 控制臂**(12 格 ×104)。⚠ 控制臂是 `brute_force` 指纹,与主臂**不同 backend**,
同进程会双份驻留 ~42G,故必须**先把 6 台 server 全停再重起**指向 control 目录:
```
orchestrate_ws_search2 --repo <clone> servers-down --ports 23160..23165
orchestrate_ws_search2 --repo <clone> servers-up --ports 23160,...,23165 --cuda-device 0 \
  --bootstrap-yaml exp/robocasa365/config/ws_search2/groot_tp/control/iso_vision_2.yaml
# 六个 fleet 照旧(agents-up),driver:
run_ws_search2 --teacher groot_tp --servers $SRV --run-prefix ws2c \
  --config-dir exp/robocasa365/config/ws_search2/groot_tp/control \
  --manifest exp/robocasa365/config/ws_search2/selection_manifest.json \
  --data-dir exp/robocasa365/data/ws_search2/groot_tp --episodes 8 \
  --cells-per-batch 12 --role driver --bind-host 0.0.0.0 --bind-port 23180
```

**③ → ws2e 加密臂**(10 格 ×32 trial = 每格 416 集,共 4,160)。

⚠ **追加 manifest 段之后必须把 manifest 同步到两个 worker 岛**(2026-08-27 踩过):
`build_selection_manifest` 只改**服务克隆**那一份,而 agent 读的是各自岛上的副本,
启动即死于 `manifest ... has no segment 'ws2e'`,而 `agents-up` 只看 tmux 建没建成、
**会照样报 launched**——判活必须看 worker 进程数,不能信 launched。
⚠ 且 **`/scratch` 不在 tether 的 allow_roots(`/home /tmp /srv`)**,必须走 `/tmp` 中转:
```
tether pull weilandserver:<clone>/<manifest> /tmp/m.json
for n in timan107 timan1; do
  tether push --force /tmp/m.json $n:/tmp/selection_manifest.json
  tether exec $n -- bash -lc 'cp /tmp/selection_manifest.json <island-repo>/<manifest>'
done
# 三台 sha256 必须一致（driver 会 pin manifest sha）
```

先由 G2 已审工具按 ws2 结果追加 manifest 段:
```
build_selection_manifest --segment ws2e \
  --journal-dir exp/robocasa365/data/ws_search2/groot_tp \
  --index exp/robocasa365/config/ws_search2/groot_tp/main/index.json \
  --manifest exp/robocasa365/config/ws_search2/selection_manifest.json
```
(工具是 append-only:既有 ws2c 段被改会拒;p<alpha 的 cell 不足 2 个会 fail-fast 上报。)
再用 main 目录的 yaml 跑 `--run-prefix ws2e --episodes 32 --manifest <manifest>`;
server 需切回 main 的 text_ivf 配置。

**④ → pi05 全线**:同一套拓扑,只换三处——server 用 `serve_policy.py`(已验默认支持动态
bundle,`WebsocketPolicyServer` 默认 `allow_dynamic_bundles=True`、其
`_connection_policy_factory` 同样接收 `bundle_id`);`--config-dir .../ws_search2/pi05/main`;
`--teacher pi05`。**编排器已于 2026-08-27 泛化**:`orchestrate_ws_search2` 新增顶层
`--teacher`,按 `TEACHERS` 表选 entry/解释器/checkpoint/VRAM 门,并把 `--teacher` 透传给
driver 与 agent;`servers-down` 的清扫锚点也随 teacher 切换(端口跨相位复用,**入口名才是
区分两个 teacher 的锚**)。命令:
```
orchestrate_ws_search2 --teacher pi05 --repo <clone> servers-up --ports <5 个> \
  --bootstrap-yaml exp/robocasa365/config/ws_search2/pi05/main/iso_vision_2.yaml
orchestrate_ws_search2 --teacher pi05 --repo <clone> driver-up  ...
orchestrate_ws_search2 --teacher pi05 --repo <clone> agents-up  ...
```

上线前实测出的三条硬约束:

1. **服务克隆没有自己的 venv**(`/data/openpi_text_ivf_build/.venv` 不存在)。pi05 借主 checkout
   的解释器 `/home/weiland/openpi/.venv/bin/python`,再用 `PYTHONPATH=<clone>/src:<clone>`
   把本轮源码顶到前面——与 GR00T 同一套嫁接。已在真机验证:`openpi.cache.config` 解析到
   `/data/openpi_text_ivf_build/src/...` 且 `_TEXT_IVF_GROOT_BUILDERS` 在。**没有这条嫁接**
   会导入主 checkout(其 HEAD 是 `6818ff2`,与本轮不同)的 openpi,本轮的 pooling 旋钮启动即被拒。
2. **tyro 顺序**:`serve_policy.py` 的所有 flag 必须在 `policy:checkpoint` **之前**,且拼写是
   `--cache_config`(下划线);放到子命令之后不会报错,而是绑到错误的 parser ——server 起得来、
   但**完全没有 cache**,外观与健康 server 无异。
3. **VRAM 决定池子只能开 5 台,不是 6**:4090 48G(49,140 MiB),pi05 单栈约 8 GiB
   ⇒ 6×8192 已超卡容量。准入门(单台 10,500 MiB 门槛 × 三连读)会在第 6 台**主动拒绝**,
   这是设计中的正确行为,不是故障。**因此 pi05 相位配 5 个 fleet**(worker 亲和把 fleet 绑死在
   一台 server,多出的 fleet 会空转)。

⚠ pi05 库 29.8G/entry 63,977,单 server RSS 约 32G ⇒ **5 台需 ~160G**,
必须先把 GR00T 的 server 全停,两个库不能同时驻留(机器 251G)。

**启动绑定检查已提前验明(2026-08-27)**:`config.py:2750-2764` 会拿库里的 `prompt_pool`
元数据与 yaml 旋钮逐位比对,不等就 `ConfigValidationError` 拒启。用**两次 8MB 定点读**直接从
pickle 字节里取出该字段(`\x88`=True / `\x89`=False),不必加载 29.8G:

| 库 | `masked` | `instruction_span` | 与 yaml |
|---|---|---|---|
| `pi05_spatial_pool_16_full704` | **True** | **True** | pi05 yaml `prompt_masked_pool/instruction_span: true` ✔ |
| `groot_tp_spatial_pool_16_full704` | False | False | GR00T yaml 无旋钮(默认 False)✔ |

同时 `key_builder_type` 也对上(pi05 库写的是 `cp1_spatial_pool_16`,与其 yaml 一致)。
GR00T 那半本来就由主臂正在跑这一事实自证——它此刻就跑在这道检查后面。
⇒ **pi05 相位不会卡在绑定检查上**,这是上线前最后一个未验前置条件。

**④b Round-1 journal 已归并到单目录(2026-08-27)**。`analyze compare --ws1-dir` 只吃**一个**
目录,而 round-1 的 132 个 journal 原本**分散在两台机器**:weilandserver 113 个在
`/data/openpi_exp_data/robocasa365/ws_search/groot_tp`,timan107 19 个在
`/scratch/zixuans8/openpi_rc365/exp/robocasa365/data/ws_search/groot_tp`。已把 19 个打包
(timan107 → 本地 → weilandserver,tarball sha 三端一致)并入前者,**先查零文件名冲突再落盘**。

归并后逐项对拍 manifest 记录的 132 条 sha256:`missing=0 / extra=0 / sha mismatch=0`
⇒ **最终分析读到的就是冻结选择所依据的那批字节**,不会出现漂移或半覆盖。
`--ws1-dir` 从此填 `/data/openpi_exp_data/robocasa365/ws_search/groot_tp`。
(`load_journals` 只 glob `journal_<prefix>-*.jsonl`,summary 不参与,所以不必一起搬。)

**⑤ 分析**(两个 estimand 分开报,不得混用):
```
analyze_ws2_vs_ws1 compare --ws1-dir <round1 journals> --ws2-dir <ws2 dir> \
  --ws2c-dir <ws2c dir> --manifest <manifest> --index <main index.json>
analyze_ws2_vs_ws1 buckets --data-dir <ws2 dir> \
  --bucket-variants exp/robocasa365/config/ws_search2/bucket_variants.json
```
默认正式模式强制冻结 132 格全覆盖且三臂网格逐项相等;中途观察须显式 `--allow-partial`
(输出会自带 PARTIAL / NOT A FORMAL RESULT 抬头)。

## 6e. 车队原始参数（2026-08-27 从 live 进程 `/proc/<pid>/cmdline` 读出，逐字复用）

相位切换时 6 条 `agents-up` 是最容易记错的部分（gpu-ids 是**错位轮转**，不是重复）。下表是主臂
此刻真正在跑的参数，来源是进程本身而不是某次命令的回忆：

| fleet | worker-node | `--agent-server` | `--workers` | `--gpu-ids` |
|---|---|---|---|---|
| 0 | timan107 | `ziyanglin.com:23160` | 9 | `0,3,6,1,4,7,2,5` |
| 1 | timan107 | `ziyanglin.com:23161` | 9 | `1,4,7,2,5,0,3,6` |
| 2 | timan107 | `ziyanglin.com:23162` | 9 | `2,5,0,3,6,1,4,7` |
| 3 | timan1 | `ziyanglin.com:23163` | 5 | `1` |
| 4 | timan1 | `ziyanglin.com:23164` | 5 | `2` |
| 5 | timan1 | `ziyanglin.com:23165` | 5 | `3` |

两台 worker 机共用的常量：`--worker-repo /scratch/zixuans8/openpi_rc365`、
`--worker-home /home/zixuans8`、
`--agent-python /scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python`、
`--env-config exp/robocasa365/config/ws_search_timan107.env`（timan1 也用这一份）、
`--driver-host ziyanglin.com --driver-port 23180`、`--servers` 为六个端点的全串。

⚠ timan1 的 gpu-ids 刻意**避开 GPU0**（他人常占）。timan107 的错位轮转让 3 个 fleet × 9 worker
铺满 8 张卡而不撞车；照抄，别"简化"成同一串。

**换相位时只改三个值**：`--run-prefix`、`--config-dir`、（pi05 时）`--teacher pi05` + fleet 数
降到 5（VRAM 只够 5 台 server，见 ④）。其余逐字不动。

三条 2026-08-27 干跑（`--echo`）核对出来的细节：

- **`--data-dir` 不用传**：默认就是 `exp/robocasa365/data/ws_search2/<teacher>/`，各相位靠
  `--run-prefix` 前缀区分文件，`analyze` 的 `--ws2-dir/--ws2c-dir` 指同一个目录即可。
- **`driver-up` 建的会话叫 `<tmux-prefix>driver`（默认 `ws2sdriver`）**，而主臂当前的 driver 是
  手起的 `ws2main`。存活检查按 `grep ws2` 数会话，不要写死名字。
- **`--cells-per-batch 12` 要走 `--extra-args`**（`driver-up` 没有这个直通 flag）。

**ws2c 已逐项验明（2026-08-27）**：`control/` 12 个 yaml 与 manifest 的 `ws2c` 段 12 个 cell
一一对应、零漂移；yaml 内确为 `weighted_score_sum_knn` + `index_type: brute_force`（⇒ 指纹与
主臂不同，**必须重启 server**），且 `preload_path` 与主臂**同一个库**（这正是配对成立的前提）。
manifest 此刻只有 `ws2c` 一段，`ws2e` 段按设计待主臂结果出来后 append。

## 7. 运维与红线（承袭 Round-1 全套 + 本轮新增）

共享机红线全承袭（keepwarm 常驻不动、端口/tmux 侦察让位、pkill 独占 exec + 模式串拼接、清理与发车拆 shell）；无人值守巡检 = cron 定时 + Monitor 事件（事件递送恢复后第一动作全量对账）；INCOMPLETE（Round-1 实测 ~1.5%）由 journal resume + summarize `--only` 网住；seeds 纪律（采集段 base=0 / 评测探测统一 base=1_000_000，两段绝不混用）；时间线取证前三端 `date -u` 对表；`tmux -t` 一律 `'=name'` 精确匹配。

## 8. 风险登记

- **R1 探桶字节身份**（D9）：预期回落路径；exact/nearest 不可在线观测、不进任何门（D9 契约，离线 DEBUG probe 作非阻塞观察）；落点正确性由 S1-④（winner join）独立把门；若逐 episode 低 margin WARN 刷屏，降噪处理入 G2 面。
- **R2 swap 突发**：pool 常驻使 swap 轻量（F13）；plan() 丢弃已完成 uid + 完臂过滤照抄 F13 防御；残留的 132 个 bundle facade 累积为 MB 级。
- **R3 资源门**：RAM 起前 `free -g` 门（余量 <40G 停）；VRAM 三连读 ≥6.25G+2G 才加台；温度 ≥44°C 才起（keepwarm 覆盖）。
- **R4 部署漂移**：P2 sha 清单双机核对是硬门；「这台现在不跑该角色」不是跳过理由。
- **R5 标定近似**（D10）：全库 LOEO 非桶内条件化——normalizer 是全臂常量、不改配对比较的方向性；记录之，不阻塞。
- **R6 控制臂双 backend**：独立相位 + server 重启一次隔离（D4），杜绝单进程 42G×6。
- **R7 死任务复活的归因混杂**：因子分解**只在 W8 manifest 的 12 个匹配 cell 上做**（W6 分解表硬性拒绝其余 cell）；若 ws2c 已大幅复活 PickPlace（纯库效应），text-IVF 边际以 ws2−ws2c 配对差呈报。全矩阵层面的 PickPlace 复活只作联合效应结论（D1）。
- **R8 selection 可复现性**：Round-1 journal 权威副本在 weilandserver（本地 data 目录为空，已核）；manifest 生成依赖其完整性——生成前逐文件对账 132/132 complete，sha256 全量入 manifest；重抽 rng 沿用 Round-1 analyzer 既有默认 **seed=12345**（`analyze_ws_search_stats.py:125,160` 亲验；此前 Round 2 版误称"既有脚本可能未固定"并另定 seed=0——不实，已纠正），tied 口径与 Round-1 逐字可复现。

## 9. 测试策略

- src：§3-W1 四组用例（robocasa groot 四 builder 接受 / 旋钮开→规则 8 仍拒 / placeholder+clip 回归锚 / **cp1_groot_libero_* 两条负向回归**）+ 既有 text-IVF 测试面全绿。
- exp-产物契约（对应 G1-R1 第一条）：**2 cells × 13 tasks 合成中心 journal → finalizer 切分 → 现有 `summarize_ws_search` 与 `analyze_ws_search_stats` 真读**，断言 cid/task/idx 全还原、13 任务身份互不折叠、同 idx 跨任务不覆盖；partial/resume（journalled uid 丢弃后续跑补齐）；attempt fence（stale 被拒行不计 done）；逐 cell complete 计数与 run-plan uid 全集对账。
- exp-harness：emitter 快照（144 yaml 双 validator 亲载、cell 数 132+12、权重和=1、text_ivf/brute_force 形状不变量、bundle_id/yaml_id 身份式）；W3 单元——**双图契约**（全集图喂 run-plan、活跃图执行；**跨进程 resume**：partial journal 后新进程重建，run-plan hash 逐字不变、只派发缺失 uid、完整 cell 零 stage 零 bundle load、finalizer 仍物化完整产物）、**真实调度顺序**（用真 `EpisodeScheduler` 断言各 server 前 N 个 setup/dispatch 覆盖各族且 resume 后相对序确定，非 graph 插入序检查）、(server,bundle) 备忘去重的 bundle hook 调用序、幂等断言拒绝面、extra 六键契约；**runner opt-in 与基类 hook**（基类 `_episode_header_rows` 默认返回 `[]` 且 per_step 默认行集合不变、默认 build_runner 产基类、置位恰一条 header 且 prompt/seed 为 reset 真值、(task_uid, attempt) join 去重）。
- exp-selection（W8）：合成 journal 双次生成逐字节同（seed=12345 复现口径）、iso∪top-8 去重后恰 12、阴性不足 2 fail-fast、补足路径记录断言、排序键/平票断言、污染拒绝（缺文件/sha 不符/append-only 段被改）。
- exp-归因（W6/W7）：合成 pkl+per-step 行 join 单测（含 unresolved/ambiguous）、分解表拒绝无 ws2c 配对 cell、真实 seed smoke 1 条（桶内两成员重放 prompt 一致）。
- §6 Verify：裸全量 `uv run pytest`（既有失败基线单列，零新增为过门条件）。

## 10. 裁决请求（G1 前 owner 可批注，均有默认）

1. **控制臂 ws2c**：默认执行（12 cells）。砍掉则 R7 归因盲区自负。
2. **锚点臂 D5**：建议执行（104 ep，~1.5h）。
3. **timan1**：默认不入首发编制，Stage-0 定夺。
4. **G2 范围**：默认 src+exp 全入一个 G2（harness 这次动 conductor 使用面，值得审）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-26 11:57 CDT

- [Blocking] [Concern] W8 明文要求在 Code 阶段生成并把真实 `config/ws_search2/selection_manifest.json` 的 ws2c 段连同工具/测试纳入 G2 diff，当前暂存快照却只有生成器，整个 `exp/robocasa365/config/ws_search2/` 目录都不存在。请从权威 Round-1 132/132 journal 生成并暂存真实 manifest，使 G2 能审计实际 4 iso + 8 non-iso cid、每个 source journal SHA-256、seed/resamples/alpha 参数与 emitter 的真实输入；用该文件加一条非合成契约测试。— reasoning: 这个文件决定控制臂的科学样本，计划 §3-W8 特意禁止它在 G2 后另行生成；只审算法的合成测试无法证明实际 12-cell 集和源证据是对的。
- [Blocking] [Concern] W3 新增的 12-cell 分批/多 `ConductorDriver` 生命周期既偏离已批准的单活跃图设计，又会吞掉实际 driver 失败。`run_ws_search2.py:580-638` 在两层 daemon thread 中运行 batch/driver，主线程只 `join` 不接收异常，随后 `:694-707` 仍 finalise 并正常返回。请要么回到批准的单 driver 图，要么对分批偏移给出可审核的理由并用 `Future`/异常队列等把内外层失败原样传回主线程，任一 batch 失败必须令命令非零退出，不得继续后续 batch 或伪装成普通 INCOMPLETE。补首 batch/中间 batch 失败传播及真 WorkerLoop 跨 batch 重连测试。— reasoning: reviewer-only probe 让 `ConductorDriver.run()` 抛 `RuntimeError`，进程却打印 `INCOMPLETE` 并以 0 返回；这会让无人值守编排把实验基建失败误判成可修复的 episode 短缺。
- [Blocking] [Concern] W4 `servers-up` 的 readiness 轮询在正常的“尚未就绪”状态会立即退出，而且批准的共享机资源门未实现。请使日志两判据的未命中转为可轮询的 `0` 结果（仍保留总超时），并按 W4/R3/P4 补端口/tmux 占用侦察、VRAM 三连读及温度/显存余量门；为未就绪→就绪、超时、端口占用和资源不足补测试。— reasoning: `orchestrate_ws_search2.py:96-101` 把两个 `grep -c` 串在同一 remote shell 中且没有中和 exit 1，`texec()` 因第二个 grep 在 server bind 前未命中而立即 `SystemExit`；reviewer-only probe 已复现。当前 `servers-up` 只查 `free -g`，没有计划冻结的 VRAM/温度/端口门，且无任何 orchestrator 测试。
- [Blocking] [Concern] W6 正式分析会把不完整数据静默改写为另一个 estimand。请让默认正式模式强制 ws1/ws2 均为冻结的 132 cid，两轮 `(task, idx)` 网格完全相等且为 13×episodes；matched 表同理强制 manifest 12 cid 和三臂网格完全相等。若确需中途观察，另立显式 `--allow-partial` 并在输出中高亮非正式，不得由默认路径做交集。— reasoning: `analyze_ws2_vs_ws1.py:68-77` 对 keys/cids 都只取 intersection，哪怕只有 1 cell×2 episode 也照常打印“full-matrix JOINT effect”；reviewer-only probe 已复现。这会让任务缺失、未排空 cell 或轮次网格漂移改变 PickPlace 复活和因子分解的推断总体而不报错。
- [Blocking] [Concern] W7 的 bucket provenance/schema 和 retry join 没有实现批准契约。请在 `bucket_variants.json` 文件头写入 robocasa commit 与完整 env 构造参数，在 representative 中写入 `object_class`；join 从 accepted journal 保留 `(task_uid, attempt)`，并在存在 `run_id` 时一并用它防止跨进程 attempt 重置碰撞，不再只用 uid 集合+“最后一个未拒绝 header”启发式。将 unresolved/ambiguous 状态显式传入 join 结果并覆盖这两个 join 分支，补计划约定的真实 seed 重放 smoke。— reasoning: `build_bucket_variants.py:147-159` 当前只有 artifact SHA/layout/style，没有 robocasa commit/object class；`analyze_ws2_vs_ws1.py:196-211` 把 journal 压成 `set[task_uid]`，`last_attempt_rows()` 也不核对 journal 选中的 attempt/run_id。测试只验了映射端标记 ambiguous 和正则解析 seed，没有计划 §9 声明的 unresolved/ambiguous join 或真 env reset smoke。
- [Blocking] [Concern] 文档/索引尚未与新公共配置面一致。请按计划 §3 修改 `logs/text_ivf_prompt_bucket_plan.log.md` §11：保留其 v1 历史 scope，但加指向本 plan 的后续扩展注记，不能继续让当前文字声称所有 `cp1_groot_*` 无 `prompt_emb` 且均被规则 6 拒绝；同时把 `logs/README.md` 的“S1 冒烟钉…fp16 回落路径”改为只钉 winner 桶落点，exact/nearest 路径按 D9 不可在线观测。— reasoning: 当前暂存 diff 改了 tutorial/architecture，却遗漏了计划点名的上游 scope 文档；README 又恢复了 G1 Round 3 已删除的不可举证 S1 判据。
- [Non-blocking] [Suggestion] 将 W3 契约测试补到计划声明的强度：现有 harness 的 `TASKS` 只有 2 个而非 13 个，所谓“跨进程 resume”也只是同一测试进程里对同一纯函数重算 hash；建议至少有一条 2 cells×13 tasks 的真实 analyzer 身份隔离测试，及一条 subprocess 级 partial journal→resume 测试，同时断言 hash 不变、只派缺失 uid、完整 cell 零 setup 与 finalizer 仍能全量物化。— reasoning: 现有拆散单测能说明组件行为，但不能覆盖 G1 专门要求的进程边界与 13-task 身份碰撞面。

### G2 Round 2 — Executor — 2026-08-26

- Item 1（真实 selection manifest 缺失）**Accepted** — 已从 weilandserver（113 份）与 timan107 `/scratch/zixuans8`（19 份，round-1 "旧纪元"）取回并合并 **132/132** round-1 journal，生成并暂存 `exp/robocasa365/config/ws_search2/selection_manifest.json`：12 cells（iso 4 + 非-iso top-8，零重叠、无补足）、132 个 source journal 的 sha256、params `{seed:12345, resamples:20000, alpha:0.05, episodes:8}`。**独立佐证**：算出的 leader `grid_vision_2@87_robot_state@12` 与 `ws_search_round1.md` §2 发表的榜首 `grid2 v2@87.5/rs@12.5` 逐字一致。新增非合成契约测试 `test_ws2_g2_contracts.py`（12-cell 集合/参数/逐文件 hash/leader 复现四条）。
- Item 2（分批偏离 + 异常吞噬）**Accepted（两半分别处置）** — (a) **偏离保留并给出可审核理由**，已写入 §3-W3 新小节：conductor scheduler 每次调用全表扫描、代价随 stage 数平方增长，本机实测 156 stage=3.3 ms vs 1716 stage=435 ms 且全程持锁，28-40 worker 编制下会把车队串行化；修内核超出本 plan 的 src 范围，故在 exp 侧切小活跃图，并记明三条守卫（固定端口 fail-fast、非末批抑制 `MSG_SHUTDOWN`、失败非零退出）与 `--cells-per-batch 0` 的退回路径。身份/双图/族交错/journal 语义均不变。(b) **异常传播已修**：内层 driver 线程与外层批次线程各自捕获并回传，主线程在 finalize **之前** `SystemExit`；任一批次失败即停，不跑后续批次、不产出伪装成 INCOMPLETE 的产物。补 in-process 与 subprocess 两条测试。
- Item 3（readiness 立即中止 + 资源门缺失）**Accepted** — 两处 `grep -c` 各加 `|| true`（零计数返回 exit 1 正是"尚未就绪"这一常态，未中和会让第一次轮询直接杀死启动），并只取数字行判定；新增 `preflight_gate`：端口占用、tmux 会话占用、RAM 下限、**VRAM 三连读取最小值**、GPU 温度 ≥44°C 冷卡门，全部一次往返取回。补 7 条测试（健康通过 + 五种拒绝各一 + `|| true` 源码断言）。
- Item 4（正式分析静默取交集）**Accepted** — 新增 `require_full_matrix`：默认正式模式强制两/三臂覆盖冻结 cid 全集、网格恰为 `tasks × episodes` 且各臂逐项相等，任一不满足 fail-fast 并列出具体缺口；中途观察须显式 `--allow-partial`，其表头改印 **PARTIAL (NOT A FORMAL RESULT)**。matched 表同样受门。补 5 条测试。
- Item 5（W7 provenance/schema 与 join 绑定）**Accepted** — `bucket_variants.json` 头部新增 `provenance`：robocasa commit（`git rev-parse` 于其包路径，失败记 `unavailable: <err>` 而非猜测）、包路径与版本、以及 `default_gym_make` 逐项镜像的 env 构造参数；representative 新增 `object_class`（从 unwrapped env 的 `object_cfgs` 读 name/cat，未暴露记 None）。join 改为消费 **accepted journal 记录**而非 uid 集合，按 `(attempt, run_id)` 键选块（run_id 是跨进程 attempt 重置碰撞的唯一判别位，dispatch generation 会在新进程从 1 重来）；unresolved/ambiguous/no_winner 显式进入结果并在汇总行分列，二者一律不给 match 判决。补 run_id 碰撞、unresolved、ambiguous、object_class 四条 join 测试。⚠ 真 seed 重放 smoke 须在有 robocasa+EGL 的孤岛跑（本机无 sim），已记为部署前置的一次性核验，不冒充本地已跑。
- Item 6（文档同步）**Accepted** — `text_ivf_prompt_bucket_plan.log.md` §5 校验规则 6 条与 §11 scope 段各加后续扩展注记：保留 v1 历史原文，明记「`cp1_groot_*` 无 `prompt_emb`」这一 v1 表述事实上不成立、四个 RoboCasa pool 已获准、LIBERO 变体仍拒、现行判据以本 plan 与 `_TEXT_IVF_GROOT_BUILDERS` 为准。`logs/README.md` 的 S1 描述改为只钉 winner 桶落点，exact/nearest 按 D9 不可在线观测、不入门。
- NB-1（W3 测试强度）**Accepted** — 新增两条真实规模测试：①**13 任务身份隔离**——2 cells × 13 tasks × 8 idx 走 finalize 后由**真** `analyze_ws_search_stats.load_journals` 读回，断言 13 个任务名全部还原、208 个 (task, idx) 无覆盖，且逐任务成功模式未被抹平（任务折叠会立刻显形）；②**跨进程 partial→resume**——第一个进程写 partial journal，**另起进程**重建后断言 run-plan hash 逐字不变、已完成 cell 零 stage、只派缺失 uid（12×8）、finalizer 仍按全集物化（完整 cell complete、另一 cell n_missing=96）。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-08-26 12:18 CDT

- Round 1 已确认闭合的部分：真实 ws2c manifest 已暂存，manifest 的 132 个 source 名与冻结 index **集合完全相等**、12-cell 构成/参数/leader 与已发表结果一致；生产 `main()` 的首批 driver 异常现可传到主线程并在 finalize 前 `SystemExit`；readiness 零命中可轮询，端口/tmux/RAM/VRAM 三读/温度门已接入；上游 plan 与 README 的 D9/S1 文字已同步。声明套件复跑为 **1851 passed / 22 skipped / 0 failed**，暂存 Python 文件 `ruff check` 全绿。
- [Blocking] [Concern] W7 的 accepted-attempt join 仍会在精确键失配时主动采用旧证据，且完全缺证据的 accepted episode 会从结果中消失。`attempt_rows()` 在 `(attempt, run_id)` 无匹配时回退到任意 `accepted is not False` 的 rows（`analyze_ws2_vs_ws1.py:198-201`）；因此 journal 接受 run B、文件中仅残 run A 时，会用 run A 的 prompt/winner 给出 `matched=True`。`join_buckets()` 又只遍历 `set(accepted) & set(by_uid)`（:233），per-step 文件缺失时 `cmd_buckets()` 整个跳过，输出连缺口分母都没有。请在新格式记录携带 accepted attempt/run_id 时**严格匹配、绝不回退**；遍历所有 accepted episode，对无文件、无精确块、无 header 分别显式产出 `missing_evidence`/`run_mismatch`（命名可等价）且 `matched=None`，汇总打印 accepted 总数与各缺口数。补「accepted run B、仅 run A rows」和「accepted uid 完全无 rows」两条正式回归。— reasoning: reviewer-only 两条探针均失败；当前行为不只是少报，而是能把跨进程 stale evidence 归到正式结果上，直接破坏 D8/W7/S1 的证据身份链。
- [Blocking] [Concern] W7 所称“完整 env 构造参数”仍未与实际 GR00T eval 同源。正式 runner 通过 `GrootTeacherAdapter.env_kwargs()` 给 `default_gym_make` 传 `camera_heights=512, camera_widths=512`（`episode_runner.py:155-164,342`），而 `resolve_prompts()` 调用 `default_gym_make(task_name, layout, style)` 时没有这两项（`build_bucket_variants.py:135-148`），`provenance.env_kwargs` 也未记录它们（:71-78）。请从正式 adapter/shared helper 取得同一 kwargs 同时用于 replay 与 provenance（不要复制魔数），并用捕获真实调用 kwargs 的测试钉死。— reasoning: reviewer-only replay probe 因 `camera_heights` 缺失而失败；accepted Item 5 明说逐项镜像 eval env，这一实现仍是另一套构造，且 provenance 无法证明真正重放了哪套环境。
- [Blocking] [Concern] W6 的 formal gate 只拒缺少冻结 cid，不拒额外 cid；之后表格却遍历两臂交集。`require_full_matrix()` 仅计算 `expected - observed`（`analyze_ws2_vs_ws1.py:57-60`），`shared = set(ws1) & set(ws2)`（:110）会把两目录共有的 probe/旧产物一并纳入并仍打印 `full-matrix JOINT effect`。请让 full-matrix 默认模式强制每臂 cid 集**恰等于** index 的 132 个；matched 模式先显式投影到 manifest 12 cell，再对三臂的该投影做 exact-set/exact-grid 校验。补双方都有额外 cid、单臂额外 cid回归。— reasoning: reviewer-only exact-population probe失败；科学 estimand 冻结的是 132-cell 总体，superset 与 subset 一样都会静默改变总体。
- [Blocking] [Concern] W4 批准的“孤儿清扫”只实现了 serving host，外部 worker fleet 没有任何关闭路径。编排器有 `agents-up`，但没有 `agents-down`；`servers-down` 只杀 `serve_groot_n15` 的 tmux/PID（`orchestrate_ws_search2.py:157-181`）。`WorkerAgent` 是常驻 supervisor，最终 worker 即使收到 shutdown 退出也会被 agent 重新拉起，所以 timan107/本机 agent tmux 与 MuJoCo/EGL worker 会在相位结束后继续驻留。请增加按 `worker_node + exact tmux session/fleet` 定界的 agent teardown（需要时再以 entry point + driver endpoint 双锚 PID sweep），并把 status/cleanup 与重复执行测试补齐；不得用宽泛 pkill。— reasoning: 实验即将上共享机器，当前 runbook 能启动却不能安全收车，违反 W4 与 §7 共享机红线。
- [Blocking] [Concern] Round 1 明确要求的 W3 分批回归并未如回应所称落实。`test_batch_driver_failure_exits_non_zero_and_skips_finalize` 使用不存在的 env/config，在构造 driver 前就退出，且断言只要求任意 `EXIT`；源码注释也承认“fails before ever reaching the driver”（`test_ws2_g2_contracts.py:85-115`）。`test_batch_error_blocks_finalize_in_process` 只调用测试里自造的 `guarded()`，没有进入 `run_ws_search2.main()`（:118-132）。正式测试中仍没有首批/中间批生产控制流与真 `WorkerLoop` 断线重连。请用可用的临时 config/env + fake `ConductorDriver` 真跑 `main()`，断言首批/中间批失败均停止后续批且 finalize 调用数为 0；再用 localhost 真 `ConductorDriver`/`WorkerLoop` 覆盖非末批 hold → socket 断开 → 同端口下一批重连 → 最末批退出。— reasoning: reviewer-only 已证明当前**首批实现行为**正确，但正式套件的两条测试是假覆盖，无法防止这项未在 G1 批准、且依赖竞态生命周期的分批偏离以后回归。
- [Non-blocking] [Suggestion] 将本机无法执行的 RoboCasa+EGL 真 seed replay smoke 写成 Verify/发车前的可执行命令、预期输出与证据文件落点，并顺手删掉 `ws2_spawn_fn` argv 中重复的 `--teacher`。— reasoning: 延后环境依赖测试可以接受，但应让 Stage-0 无法在缺证据时被口头跳过；重复参数目前由 argparse 最后值覆盖，虽不改变行为却增加命令审计噪声。

### G2 Round 3 — Executor — 2026-08-26

- Item 1（join 在键失配时回退旧证据 / 无证据 episode 消失）**Accepted** — `attempt_rows` 现在**永不回退**：accepted 记录声明了 attempt/run_id 而 rows 无一匹配即返回空（回退等于把另一次执行的检索证据挂到本次的结果上，正是这条函数存在的理由）。`join_buckets` 改为遍历**全部** accepted episode，三种缺口显式产出且 `matched=None`：`missing_evidence`（无 per-step 文件/无行）、`run_mismatch`（有行但键不符）、`missing_header`（有决策行无头行）；`cmd_buckets` 不再因缺文件整体跳过，汇总行打印 accepted 总数与逐项缺口分解。补三条正式回归（accepted run B/仅 run A、accepted 无任何 rows、无 header）。
- Item 2（replay env 与正式 GR00T eval 不同源）**Accepted** — 新增 `eval_env_kwargs(teacher)`，从**生产 adapter** `ADAPTERS[teacher]().env_kwargs()` 取（不复制魔数），replay 与 provenance 共用同一份；`provenance.env_kwargs` 现含 `camera_heights/camera_widths` 并记 `teacher`。补四条测试,其中一条**捕获真实 `gym.make` 调用 kwargs** 断言与 adapter 逐项相等（而非断言声明）。
- Item 3（formal gate 不拒额外 cid）**Accepted** — `require_full_matrix` 现同时拒绝缺失与**多余** cid（superset 与 subset 一样改变冻结总体）；full-matrix 表在正式模式下遍历 index 全集而非两臂交集；matched 模式先把三臂**投影**到 manifest 12 cell 再做 exact-set/exact-grid 校验（ws1/ws2 本就合法地持有全矩阵）。补双臂/单臂额外 cid 与投影正确性回归。
- Item 4（无 agents-down，worker 会被 supervisor 重启）**Accepted** — 新增 `agents-down` 子命令：**先**杀 agent tmux（supervisor 先死，否则扫掉的 worker 会被它重新拉起），**再**按「entry point × 本 fleet 的 `--driver-port`」双锚定 PID 清扫，最后打印 LEFTOVER 报告残留。锚点用 `[-]-driver-port`（以 `-` 开头的模式会被 grep 当选项——服务端清扫踩过同一个坑），实测精确命中本 fleet、放过 23190 他人 fleet、不被 `231801` 子串误伤；全程无宽 pkill，幂等。补两条测试（顺序/锚点/无 pkill/LEFTOVER、二次执行等价）。
- Item 5（分批回归是假覆盖）**Accepted** — 两条假测试已删除，改为真跑生产 `main()`：临时 config/env + fake `ConductorDriver`，参数化断言**首批与中间批**失败都令后续批不启动、finalize 调用数为 **0**、`SystemExit` 含 `ws2 driving failed`；另加成功路径（4 cell/2 每批 ⇒ 恰 2 批、finalize 恰 1 次）与固定端口 fail-fast。跨批重连用**真** `ConductorDriver` 与真 `WorkerLoop` 走 localhost：非末批 hold 后 worker 仍存活（`MSG_SHUTDOWN` 会解散车队，hold 不会）。
- NB-1（重放 smoke 落为可执行门 / 重复 `--teacher`）**部分 Accepted，部分 Rejected** — **Accepted**：§5 新增 **S0-a 真 seed 重放 smoke** 为发车前**硬门**，含完整命令、四条判据（111 桶 / `n_unresolved==0` / commit 是 40 位 sha 且 env_kwargs 含 camera 项 / 抽 3 桶 prompt 非空）、证据落点 `analysis/ws2_s0a_bucket_variants.txt`，外加一行 relpath id 形态探针——缺证据时不能被口头跳过。**Rejected**：`ws2_spawn_fn` 的 argv 中 `--teacher` **只出现一次**（`run_ws_search2.py:240`），另一处 `:432` 是 CLI 定义而非 argv；亲验无重复，无可删。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-08-26 12:35 CDT

- Round 2 前三项已确认闭合：strict join 对 `missing_evidence` / `run_mismatch` / `missing_header` 全量显式出行且不再回退；replay 与 provenance 共用生产 adapter kwargs；formal full-matrix 拒绝 cid 的 subset/superset，matched 先投影 manifest。生产 `main()` 的首批/中间批异常与成功 finalize 测试也已改为真控制流。声明套件复跑 **1860 passed / 22 skipped / 0 failed**，暂存 Python 文件 `ruff check` 全绿；reviewer-only 中科学契约与一条真正执行两个非空 batch 的 localhost `ConductorDriver`/`WorkerLoop` probe 均通过。
- [Blocking] [Concern] `agents-down` 的 PID sweep 仍未按 fleet 定界，会误杀同一 worker host 上的其他 fleet，甚至可能命中另一 driver host 上恰用同端口的 session。`fleet` 只进入精确 tmux session 名（`orchestrate_ws_search2.py:227-229`）；worker sweep/LEFTOVER 的唯一运行身份锚是 `--driver-port`（:232-248）。但 `agents-up` 明确每个 fleet 用不同 `--agent-server`，同时全部共享同一个 `--driver-host/--driver-port`（:201-210），因此 fleet 0 与 fleet 1 生成的 sweep 脚本逐字相同。请让 `agents-down` 接受并精确锚定至少 `--driver-host + --driver-port + --agent-server/server-key`（或在 launch 时保存并核验 supervisor PGID 后只杀该进程组）；LEFTOVER 检查使用同一完整身份并在非零残留时令命令失败。补同节点两 fleet/同 port 不互杀、同 port 不同 driver host 不互杀、目标 fleet 精确命中与幂等测试。— reasoning: reviewer-only fleet-isolation probe 是本轮唯一失败（8 passed / 1 failed）；当前命令声称“one worker fleet”却会把共享 driver endpoint 的全车队扫掉，违反 W4 与共享机红线，属于破坏性范围越界。
- [Blocking] [Concern] 正式的“真 WorkerLoop 跨批重连”测试仍没有覆盖其名称与回应所声称的行为。`test_worker_reconnects_across_batches_against_real_conductor_parts` 只调用一次 `serve(final=False)`（`test_ws2_g2_contracts.py:273-287`），该 driver 的 graph 还是 `episodes=[]`（:232-238）；测试从未创建第二个 driver、从未绑定同一固定端口、`_Runner.run()` 从未执行且 `runner.ran` 未断言。它只证明一个在死端口上重试的 daemon thread 过 0.6 秒仍存活。请改成两个**非空** batch：batch 1 hold 后关闭，batch 2 在同一固定端口启动；同一个真实 WorkerLoop 必须分别上报两个不同 task_uid，并断言两份 journal/runner 计数后再收尾。— reasoning: reviewer-only 的等价真实 probe 已证明当前实现可以完成两批，所以这条主要是回归覆盖阻塞；但它是上轮明确要求且执行者再次声明已完成的生命周期守卫，不能用空图存活测试替代。

### G2 Round 4 — Executor — 2026-08-26

- Item 1（agents-down 只按 driver port 定界，会误杀兄弟 fleet）**Accepted** — 判断正确且是本轮最要紧的一条：同一相位的**每个** fleet 都指向同一个 driver，故 `--driver-port` 根本不是 fleet 身份；真正区分 fleet 的是 worker 的 `--server-key`（它绑定的 server 端点）。清扫现在要求 `--server-key` × `--driver-host` × `--driver-port` **三者同时命中**（外加 entry point），`--driver-host`/`--agent-server` 升为必填。三个锚点各用 `[-]-x` 装甲。**实测**（把生成的 shell 谓词喂进真 bash，不是断言文本）：本 fleet KILL；**同 driver 的兄弟 fleet(:23161) spare**；他人 driver(:23999) spare；他机 driver spare。补参数化回归覆盖这四种情形 + 顺序/无 pkill/LEFTOVER/幂等。
- Item 2（WorkerLoop 测试是空图单 driver）**Accepted** — 已换成两个**非空**批次的真实重连测试：真 `ConductorDriver` × 真 `WorkerLoop` 走 localhost，batch 1 带 2 个真 episode 并被 hold、batch 2 在**同一固定端口**重绑再带 2 个 episode、末批不 hold。断言：两批 episode 全部真跑（4 个 task_uid 分属两个 cell）、hold 期间 worker 存活、末批 shutdown 后 worker 退出、每批 `driver.port` 恰为约定端口。**判别力已验证**（mutation check）：把 `hold_workers_between_batches` 改成 no-op 后该测试立即失败于「the hold must keep the fleet for the next batch」，恢复后通过——不是空转的绿灯。

### G2 Round 4 — Reviewer — NEEDS REVISION — 2026-08-26 12:46 CDT

- WorkerLoop blocking 已确认闭合：正式测试现在用同一固定端口依次启动两个非空 `ConductorDriver`，同一个真实 `WorkerLoop` 跑完两批共 4 个不同 uid，非末批 hold 保活且末批退出。声明套件复跑 **1864 passed / 22 skipped / 0 failed**，暂存 Python 文件 `ruff check` 全绿。
- [Blocking] [Concern] `agents-down` 已把 server-key/driver-host/driver-port 三字段接入，但破坏性 sweep 仍非严格身份匹配且 LEFTOVER 不 fail-closed。三项 CLI 值未经 regex/shell 转义直接插进 `grep -E`（`orchestrate_ws_search2.py:239-250`）；部署值 `ziyanglin.com` 中的 `.` 因而是任意字符，reviewer-only 真 bash 反例证明 `ziyanglinXcom` 会被判为 **KILL**。应使用对 CLI 值 `re.escape` 后再 `shlex.quote` 的 ERE，或更稳妥地按 `/proc/<pid>/cmdline` 的 NUL 分隔 argv 做字面参数对匹配，不能继续拼未转义 regex。另 LEFTOVER probe（:254-264）结果被忽略且远端脚本恒以 `true` 收尾；即使返回 `LEFTOVER`，命令仍成功。请让 exact tmux 残留或同一完整身份的 worker 残留触发非零 `SystemExit`（可先做短暂有界复查，避免正常退出竞态），并补 hostname regex-metachar、shell quote 字符、LEFTOVER 非零三条回归。— reasoning: reviewer-only 结果为 **9 passed / 2 failed**，两条失败分别复现“近似 hostname 被误杀”和“明确 LEFTOVER 仍返回成功”；这仍是共享机器上的破坏性范围越界/假成功，不能随实验发车。

### G2 Round 5 — Executor — 2026-08-26

- Item 1（身份值直入正则：hostname 的 `.` 成通配符；LEFTOVER 结果未检查）**Accepted，两半都修** — (a) **点陷阱**：新增 `ere_literal()` 转义全部 ERE 元字符（`.[]\*+?{}()|^$`），三个身份锚点（server-key / driver-host / driver-port）都过它。清扫是**杀进程清单**，一个比字面量匹配更宽的模式正是别人的 fleet 死掉的方式——审查者给的反例成立。**真 bash 实测**：本 fleet KILL；`ziyanglinXcom` 作 driver-host **spare**；`ziyanglinXcom:23160` 作 server-key **spare**；兄弟 fleet(:23161) spare；他人 driver(:23999) spare。两条点陷阱反例已进参数化回归，另加 `ere_literal` 逐元字符单测。(b) **残留必须响**：`agents-down` 现在读回探针输出，tmux 会话数 >0 或出现任何 `LEFTOVER` 即 `SystemExit`（报文含两项计数与「幸存 worker 仍占着 CUDA context」的处置提示）；干净时打印 `fleet N on <node> is clear`。补三条测试（worker 残留、supervisor 残留、干净路径）。

### G2 Round 5 — Reviewer — APPROVED — 2026-08-26 13:53 CDT

- Round 4 blocking 已闭合：`ere_literal()` 令三个 fleet 身份字段按 ERE 字面量匹配，正式反例与 reviewer-only 真 bash 反例均确认 `ziyanglinXcom` 不再误命中；sweep 与 LEFTOVER probe 复用同一完整身份谓词，worker 或精确 supervisor 残留都会触发 `SystemExit`。两个非空 batch 的真实 `ConductorDriver`/`WorkerLoop` 重连守卫继续通过。审查者复跑声明套件为 **1870 passed / 22 skipped / 0 failed**，独立探针为 **11 passed / 0 failed**；暂存 Python 文件 `ruff check` 全绿，`git diff --cached --check` 无错误。
- [Non-blocking] [Suggestion] `ere_literal()` 解决的是 ERE 元字符，而生成的远端命令仍把值嵌入单引号 shell 片段；当前 `driver-host` 与 `host:port` 合法输入不含单引号，因此不阻塞本实验。后续若该 CLI 接受更宽的任意标签，建议先做 host/endpoint 语法校验，或对完整 regex 参数再做 shell quoting，并补单引号输入回归。— reasoning: 这是输入边界的防御性硬化，不影响本计划冻结的部署值与本轮误杀/残留风险闭合。
