# RoboCasa365 纯 cache 检索参数搜索（weighted_sum × spatial_pool_16 × n5 库）— Plan

**Status**: `Live — GR00T 臂主跑中`（commit `4cfdfa1` 已 push 部署；Stage 0 ✅、P2 门 ✅、env-cache OOM 热修 ✅（§9）；边际吞吐 239 ep/h；owner 已授权无人值守独断专行，队列=GR00T 臂→锚点→pi05 探测→pi05 臂→锚点→汇总→终局 commit）
**日期**: 2026-08-21
**Owner 指令**（2026-08-21）: T7 第一步 = 纯 cache 下搜索参数配置；search strategy 固定 `weighted_sum`（即 `weighted_score_sum_knn`），query builder 固定 spatial 16（`cp1_spatial_pool_16` / `cp1_groot_spatial_pool_16`），双 teacher 均用 per-task-5 的 n5 pkl；对标 LIBERO 最早那轮参数搜索的形制，但**不再搜 query builder**。开跑前先实验确定 weilandserver 可开的 concurrent server 数与 timan107 可开的 conductor worker 数（107 有他人实验，不得影响），记录后正式开始。

---

## 1. 已验事实（file:line 亲验）

- **LIBERO 最早搜索的形制**（`exp/weighted_sum/`，RESULTS.md 头部）：pi05_libero × judge=`always_hit` 纯缓存重放隔离检索质量；**每配置 100 ep（10 task × 10 held-out trial）**；基线 **136 配置 = (iso 3 + grid2 21 + grid3 10) × 4 keybuilder**；两层打分 = Layer-1 per-field 归一化（Phase-1 离线标定选定）+ Layer-2 加权和。
- **权重族生成器**（`exp/weighted_sum/emit_yamls.py:119-170`）：`isolation_weight_configs`（单字段 1.0）；`grid_weight_configs`（两字段对 × 7 档 `{0.125..0.875}`，和恒 1）；`grid3_weight_configs`（含 dominant 字段的三字段单纯形，step 0.125、dom_min 0.375，LIBERO 取 dominant=robot_state）。
- **yaml 结构**（`emit_yamls.py:34-117` `build_eval_config`）：`weighted_score_sum_knn` + `top_k:1` + `step_filter:all` + `score_normalization:{type:per_field, fields:...}`；judge=`always_hit`、gate=`always_search`、timer 关、`write_policy:never`；backend.vector_dims 必须等于 artifact 全字段集（weight-0 字段仍留维度、prompt_emb 以 `keys.enabled:false` 屏蔽）；已原生支持 vision_2（`WEIGHTED_FIELDS` 含之，:28）。
- **Phase-1 标定工具**（`exp/common/calibrate_score_normalizers.py:1-60,159-164`）：**artifact 自含 LOEO**（每 entry 作 query、对非同 trajectory 的库 entry 打分），拟合候选 normalizer、按 J 排序出 selected+shortlist；CLI = `--artifact-dir --output --max-queries(300)`；`_FIELD_SIM_TYPE` 已含 vision_2:cosine、robot_state:l2；prompt_emb 排除。**无需额外标定池**——直接吃 n5 pkl。
- **schema 现状**（`src/openpi/cache/config.py`）：`weighted_score_sum_knn` 合法类型（:1493）；要求 backend=in_memory（:1821）；强制非空 `score_normalization`（:1920-1926）。
- **robocasa eval 机件**（conductor 复用面）：`EpisodeTask`（`src/openpi/conductor/task.py:73-93`）带 `yaml_id/task_id/episode_idx/orig_init_state_idx/bundle_id/extra`；`RobocasaEpisodeRunner`（`exp/robocasa365/episode_runner.py`）`REQUIRED_EXTRA_KEYS=("task_name","layout","style","teacher","base_seed","replan_steps")`（:70），seed=`base_seed+orig_init_state_idx`（:25,354），场景由 extra.layout/style 控制（:228 只允许 kitchen 变）。`run_collect.py` 的 driver/agent/env-config 骨架可整体沿用（`collect_weilandserver.env` 声明 server 组/worker 解释器/ROBOCASA_CWD/EGL）。
- **GR00T 并发 server 真机已证**（`exp/robocasa365/analysis/t7_concurrent_smoke.txt`，2026-08-21 PASS）：`serve_groot_n15 --concurrent --cache-config` + n5 库，双连接真交错、FULL_HIT、零 stage2、零串扰；纯 cache 单步 stage1 仅 42-72ms；episode 墙钟 114-206s（sim 主导）。
- **GR00T server 不支持 bundle 热切换**：`--concurrent` 下 `allow_dynamic_bundles=False` 拒全部 `load_cache_config`（G2 冻结契约）；单连接模式的 bundle 机件也不会重建 GR00T 栈 ⇒ **换配置必须重启 server 进程**。
- ~~⚠ pi05 侧 3 相机在线 key 通路未验证~~ **✅ P2 门 PASS（2026-08-21，[`t7_p2_gate_pi05.txt`](../exp/robocasa365/analysis/t7_p2_gate_pi05.txt)）**：结构半程（三相机 yaml + n5 库 5901 entries 全维度接受）与在线半程（56 条逐步 FULL_HIT、winner 步序与执行同步滑动、OpenDrawer 单集 **sr=1.0**）双证；源码根因=`key_builder.py` 本就 `_NUM_IMAGES=3` 固定布局且库采自同一前向 hook ⇒ 在线/离线同构。pi05 臂解锁，容量探测排 GR00T 臂后。
- **T6 库**：`/data/robocasa365_cache/cache_artifacts_l1s1/{pi05,groot_tp}_spatial_pool_16_n5.pkl`（5901/4794 entries，2.8G/1.9G），identity 校验已在冒烟中通过（groot 侧）。

## 2. 设计决策

- **D1 目标函数**：每配置 pooled SR（13 task 等权均值），辅以 per-task SR；判据=选 top-10 进后续 refine 轮（对标 LIBERO emit_top10 流）。本轮不测延迟/inference_ratio（timer 关）。
- **D2 评测口径**：纯 cache = gate `always_search` + judge `always_hit` + `top_k:1`（与 LIBERO 基线和 cache_size 消融同口径）。场景固定 **(1,1)**（同分布，库=l1s1）；**eval seeds base=1_000_000**（与采集/gate 段不相交），**全配置/全 teacher 共用同一 seed 集 ⇒ 配置间成对可比**。
- **D2b 场景×种子语义（owner 2026-08-21 令记录）**：RoboCasa365 每次 reset 由 env 的 seeded RNG 从 `layout_and_style_ids` **抽取** (layout,style)（`robocasa .../kitchen.py:595` `rng.choice(...)`，timan107 源码亲验）——即不钉列表时 **layout 由随机种子决定**。本实验全线传单元素列表 `[(1,1)]`（`episode_runner.py` `default_gym_make` / rollout client 同款）把抽取退化为确定值：**场景控制走显式钉死，绝不走挑 seed**；seed 只决定钉死场景内的物体实例与连续位姿。推论照旧：同 seed 跨场景初始状态不同 ⇒ 配对在 task/config 层，不在 episode 层。
- **D3 字段集**：4 权重字段 {vision_0, vision_1, vision_2, robot_state}；prompt_emb 弃权（LIBERO D7 同理：任务常量；`keys.prompt_emb.enabled:false` 屏蔽，backend dims 保留 2048）。
- **D4 矩阵**（每 teacher 76 cells，镜像 LIBERO 三族，keybuilder 不搜）：
  - iso ×4：单字段 1.0；
  - grid2 ×42：C(4,2)=6 对 × 7 档 {0.125..0.875}；
  - grid3 ×30：rs-dominant 三字段单纯形（{v0,v1,rs}/{v0,v2,rs}/{v1,v2,rs} 各 10）。⚠ dominant=rs 承袭的是 LIBERO 结论，robocasa 未知——保留成本仅 30 cells，作对冲；若 owner 不认可可推迟到 round 2 由 grid2 信号定 dominant。
- **D5 每配置集数**：**13 task × 8 trial = 104 ep**（对标 LIBERO ~100）；降档选项 5 trial=65 ep（省 ~38% 墙钟，排序噪声增大）。
- **D6 运行拓扑（统一静态槽位设计，双 teacher 同构）**：weilandserver 起 S 个 pure-cache server（S 由 Stage 0 探出；GR00T 用 `serve_groot_n15 --concurrent`，pi05 用 serve_policy 对应配方），端口取 231xx 空闲段；76 配置静态均分为 S 条队列，每槽位串行：`起 server(--cache-config yaml_i) → 本配置 104 ep 跑完 → 关 server → 下一配置`。**不依赖热切换**（GR00T 不可能、pi05 P2 未验，统一重启制最稳）；重启开销 76×~2min÷S ≈ 每 teacher 半小时级。
- **D7 worker 侧**：timan107 车队 W 个 worker（W 由 Stage 0 探出），按槽位分组连各自 server 公网口（`ziyanglin.com:231xx`）；复用 conductor（driver 每配置一次 invocation：run-plan JSON + journal + resume 纪律照旧）。**ctl 保持 no-op**：run_collect docstring 的「eval 须真 ctl」前提是热切换拓扑（driver 要下发 load_cache_config/preload）；静态配置制下 driver 无 bundle 操作，worker 的 `select_bundle("default")` 即启动配置幂等绑定（G2 已证）。此偏离在此明示，审查请专门看这条。
- **D8 顺序**：GR00T 臂先跑（通路全验）；pi05 臂在 **P2 接线门 PASS 后**入队（门=单配置 G0-E 式冒烟：3 slot 在线 key dims 匹配 + FULL_HIT hit log + episode 完整）。
- **D9 锚点臂**：每 teacher 加 1 个 teacher-only（无 cache）臂 @ 同 seeds/scene ×104 ep，作纯 cache SR 的天花板/上下文（+208 全推理 ep，约 1.5h）。
- **D10 Phase-1**：`calibrate_score_normalizers.py` 分别吃两个 n5 pkl（staging 目录只放这两个，避免 6 件套全被标定）；weilandserver 上 tmux 跑（数据本地、251G RAM）；产出 `calibration_normalizers.json` 进 repo（tracked config）。selected normalizer 直接用；`__norm2` 对照本轮不做。

## 3. Stage 0 — 容量探测（owner 2026-08-21 指令，先测后跑）

**A. weilandserver concurrent server 数（S）**
1. 基线记录：keepwarm 温度/显存、他 session 占用、231xx 占用表。
2. 逐个加 server（GR00T pure-cache smoke yaml，n5）：每加一个前显存**连读 3 次 ≥ 单 server 实测 6.25G + 2G 余量**才起；记录每 server 稳态 VRAM/RAM。
3. 每档 S∈{1,2,3,...} 用固定 8-worker 小车队跑 2 task × 2 seed 短测，记 aggregate ep/h 与 stage1 延迟 P50/P95；**边际吞吐 < 70% 线性或显存余量不足即停**。
4. pi05 侧同法抽测 1 档（server ~8G，S_pi05 可能更小）。
- 预判（供证伪）：VRAM 上限 ~6 个 GR00T server；实际 S 可能受 CPU/锁竞争先封顶在 3-4。

**B. timan107 conductor worker 数（W）**
1. 基线记录：load avg（当前 6.9/48 核）、RAM available（160G）、8 卡各自 free VRAM、他人进程清单（yurenh2 训练 + zixuans8 LIBERO 车队，**只读不动**）。
2. 对 Stage-0A 定下的 S 个 server 做 worker 爬坡 W∈{8,16,24,32,(40)}：每档跑 10min，记 ep/h、load avg、RAM、per-GPU VRAM（worker 按 8 卡 round-robin 摊，单卡预留 ≥1.5G 给他人）。
3. **护栏（任一触发即回退一档定格）**：load avg > 40（48 核留 ~8）；RAM available < 40G；任一 GPU free < 1.5G；他人进程出现异常（重点盯训练进程存活）。
- 产出：`exp/robocasa365/analysis/t7_capacity_probe.txt`（S、W、每档吞吐表、护栏读数、最终配比 S×(W/S)），并回填本 plan §5 成本表。**记录完成后才正式开跑。**

**探测工作量**：~1.5-2.5h；探测 seeds 同用 **eval 段 base=1_000_000**（owner 2026-08-21：测试统一用 eval 随机种子；探测产出不入正式评测记录，无泄漏面）。

**✅ Stage 0 GR00T 侧完成（2026-08-21，记录 [`t7_capacity_probe.txt`](../exp/robocasa365/analysis/t7_capacity_probe.txt)）**：三档实测（1×4 / 3×4 / 6×3）。**运行点 = S=3 server × 4 worker（12 流），聚合 8.7 infer/s、S=3 线性、延迟不涨**；6×3 反而腰斩（4.58 infer/s）。**两堵墙**：① 主存带宽——单次 cp1_search 扫 ~1.9GB 向量，3 路并行 ~22GB/s 已近 DDR 上限，再加 server 全场互踩（load 33、单进程 2.4-4 核白烧）；② weilandserver 默认路由是 **WiFi 网卡**（实测 rx 8.3MB/s），18 路 obs 上传流触发空口争用崩溃，12 路健康。probe 配方是最坏情形（3 vision 全扫）；真实 cell 扫 1-2 字段应更快。timan107 护栏全程未触（load 峰 19.4/48、RAM 余 ≥104G、单卡 free ≥3.3G）。**pi05 侧探测分开做**（owner 裁定），待 P2 门后同法跑。

## 4. 实现单元

- **W-1 emitter** `exp/robocasa365/emit_ws_search_yamls.py`：import `exp.weighted_sum.emit_yamls.build_eval_config`（不复制实现），按 §2-D3/D4 生成 76 yaml/teacher 至 `exp/robocasa365/config/ws_search/{pi05,groot_tp}/`；vector_dims 取标定 json；preload 指 n5 绝对路径。
- **W-2 Phase-1 驱动**：staging 软链目录 + 两次 `calibrate_score_normalizers.py` 调用（weilandserver tmux），json 拉回入 repo。
- **W-3 搜索调度器** `exp/robocasa365/run_ws_search.py`：槽位队列 + server 起停（含就绪等待/端口侦察/显存三连读）+ 每配置 conductor driver invocation（eval extra：layout=1,style=1,base_seed=1_000_000,replan_steps=5）+ journal/审计 + 汇总 CSV。
- **W-4 汇总分析** `exp/robocasa365/analysis/summarize_ws_search.py`：journal → per-config/per-task SR 表 + top-10；最终报告 `exp/robocasa365/analysis/ws_search_round1.md`。
- **W-5 pi05 P2 接线门**：serve_policy + pi05 n5 yaml 单配置冒烟，证据卡入 analysis/。
- 测试：emitter 快照测试（cell 数 76、权重和=1、schema 过 validator）、调度器单元（队列切分/重启序列/失败重试）、汇总器（合成 journal）。

## 5. 成本估算（Stage 0 后按实测 S/W 回填修正）

**Stage 0 实测修正**：吞吐上限不由 worker 数定，而由 weilandserver 主存带宽 + WiFi 空口定：S=3×4 聚合 ≥8.7 infer/s（最坏配方）→ **≥237 ep/h**，真实 cell（少扫字段）预计更高。GR00T 臂 7904 ep ≤33h（乐观 ~1 天）。原表按 500 ep/h 的估算作废：

| 项 | ep 数 | 墙钟 |
|---|---|---|
| Stage 0 探测 | ~100 | 1.5-2.5h |
| Phase-1 标定 ×2 | — | ~1h |
| GR00T 臂 76×104 | 7904 | ~16h + 重启 ~0.9h |
| pi05 P2 门 | ~4 | 0.5h |
| pi05 臂 76×104 | 7904 | ~16h |
| 锚点臂 ×2 | 208 | ~1.5h（全推理慢些） |
| **合计** | ~16.1k | **~1.5-2 天无人值守**（5-trial 降档则 ~1 天） |

## 6. 风险与红线

- 共享机红线全承袭（keepwarm 不关、他 session 端口/进程/tmux 不动、pgrep 锚定、清理与重启异 shell、tether 10min → tmux+tee）。
- timan107 护栏见 §3-B3；探测与正式跑期间挂 cron 巡检（Monitor 管事件、cron 管按时）。
- serve_policy 挂死档案（futex 签名）在 eval 模式是否复现未知——巡检沿用 KA≥4 弹换 + resume。
- 76×2 次 server 重启 × 冷卡风险：keepwarm 常驻即覆盖（起 server 前查温度 ≥44°C）。
- seeds 纪律（权威出处 = `robocasa365_framework_integration.log.md` **§D-E**）：采集/建库段 base=0；**评测与探测统一 eval 段 base=1_000_000**（trials=8 ⇒ 每 task 1_000_000…1_000_007）；两段绝不混用——库内轨迹全部来自 0 段种子 ⇒ eval 初始状态与库零重合。

## 7. 裁决记录（2026-08-21）

1. **trial 数 = 8**（owner 明裁："8/task"）→ 104 ep/config。
2. **grid3 rs-dominant**：保留（owner 未反对，按推荐默认执行）。
3. **锚点臂**：加（按推荐默认执行）。
4. **流程形式**：owner 令「记录好后开始实验」= 直接开工；harness 代码全部落 `exp/robocasa365/`（不动 src/ 老实现），在本会话内自审——沿用 owner 对 cp1_search 调优的既有独裁令（搜索调优 override G1/G2、exp 内实现、会话内审查）。
5. **eval 种子**（owner 两次强调）：D-E 段 base=1_000_000，探测与评测统一用之；layout-by-seed 语义记录于 §2-D2b。

## 8. Review Log（fast-track 会话内自审，静态 2 agent，2026-08-21）

**Agent-A（harness 正确性）**：3 BLOCKING + 8 ADVISORY，全部处置——
- B1 重试耗尽 episode 无 journal 终态会从统计蒸发 → `summarize_journal` 与 run-plan uid 全集对账（missing/err 单列、不进 SR 分母），`complete = (n_err==0 ∧ n_missing==0)`；编排器只认 complete 才 skip/DONE，半死 cell resume 自动重跑。
- B2 port-up 不能证明身份（共享 231xx 段 + 垂死前任占口）→ 先杀前任并等端口真释放，再以**本次新鲜日志**（tee 截断）中「本 cell yaml 路径 + SERVER-LISTENING 横幅」双证为就绪判据；超时杀半启动 server。
- B3 STUCK 杀 driver 孤儿化 worker（start_new_session）→ run_ws_search 捕获 SIGHUP/SIGTERM 走 finally 收割 worker 组 + 编排器按「openpi_rc365 路径 × 槽端口」双锚定孤儿清扫。
- A4 假 `--capacity` 移除（scheduler 无 per-server 派发闸，并发=worker 数）；A5 server 启动全局互斥 + VRAM 门 6 次退避（放弃 cell 不弃队列）；A6 VRAM 解析取全卡最小值+容错；A7 tether 超时返回 unknown、driver 判死需连续两次 GONE；A8 STUCK 措辞改「留待 resume」；A9 身份超时 kill 半启动 server；A10 `pooled_sr`→`macro_sr`（任务等权宏平均，plan D1 原意）；A11 GPU 轮排参数化（--gpu-count，步长=workers-per-slot）。

**Agent-B（emitter/schema 保真）**：0 BLOCKING + 5 ADVISORY——152 yaml 双 validator 全过、字节可复现、cell 数/权重和/rs-dominant 判据全对。处置：①cp3 陷阱注释按实证机制改写（真陷阱=块**存在**但缺 `search_strategy.type` 时缺省 qdrant 且 disabled 也被 type-check；省略整块其实也合法；保留实载验证过的钉死形）；②标定 json 双源消歧（删 gitignored data/ 副本，config/ 为唯一权威，emitter 用法示例改正）；③强制 enabled 的 weight-0 字段实证不改变检索结果（`_iter_active_fields` 丢弃 w≤0），仅 key-build 侧小浪费，属 validator 硬要求；④`to_similarity` tau 在 per_field score-sum 路径 inert，已在 emitter docstring 注明；⑤cell 名百分数截断（12+87=99）为 LIBERO 既有命名惯例，保留以便跨实验 join。

处置后验证：`tests/robocasa365/test_ws_search_emitter.py` 7/7、ruff 全绿、`load_cache_config`+`validate_groot_cache_config` 76/76 亲验、re-emit 字节可复现。

## 9. Live 运行记录

- **2026-08-21 20:00Z GR00T 臂开跑**（commit `4cfdfa1` 部署双机；单 cell 试点 PASS——iso_robot_state 下 OpenDrawer 纯 cache 成功）。边际吞吐实测 **239 ep/h**（8min 窗，41→73 ep），与 Stage 0 预测 ≥237 吻合；全臂 ≈33h。
- **P2 接线门提前完成 PASS**（20:1xZ，见 §2 与 `t7_p2_gate_pi05.txt`）：pi05 三相机在线通路全通，OpenDrawer 纯 cache 单集 sr=1.0。
- **⚠ 实踩新坑 + 热修（20:44Z）**：`RobocasaEpisodeRunner` 的 kitchen env 缓存**无逐出**（`episode_runner.py` `_envs` 按 (task,layout,style) 只进不出）——T5 采集在 48G 卡上 13 个 kitchen 放得下，eval 车队在 timan107 8G 卡上每 worker 实测 ~3.5-4G 且随任务轮转持续增长，GPU2 一度只剩 50MiB。修复=新增 `max_cached_envs` 参数（默认 None 保采集字节兼容；eval 链固定 1，任务切换即 close 旧 env），贯穿 episode_runner/worker_entry/robocasa_spawn_fn/run_ws_search 四文件 + 2 条单测（27/27 绿）。处置：停 wsorch → tar 经 /tmp 部署（⚠ /scratch 不在 tether allow_roots，直推被拒）→ 杀旧 driver/worker（锚定清扫）→ 重启 wsorch 幂等续跑（73 ep journal 保留）。

## 10. 排空时刻 Runbook（GR00T 臂 "ALL SLOTS DRAINED" 后按序执行）

> **⚠ 2026-08-22 12:0x owner 终点改令（/goal）**："pi 0.5 stage0 运行拓扑做完之后可以停止，我们之后需要重新构造 cache"。⇒ 本轮执行链截断为：**①排空核对 132/132 → ①b 死任务终裁 → ①c pi05 Stage-0 容量探测（终点实验）→ 报告/收口/终局 commit → 停**。步骤 2（锚点臂）、2b（加密轮）、4（pi05 全臂）、5（pi05 锚点臂）**取消**——cache 将重构，老 n=5 库上的加密/锚点/pi05 铺量作废；其设计（配对检验、run-prefix、共享队列、9 任务集）保留给重构后的下一轮。timan1 供给（§10-8）照做（服务下一轮）。ws_search_round1.md 以"筛选轮 + 机理解剖"定稿，明记截断原因。以下原文保留作历史与下一轮参考。

1. **汇总核对（双源）**：summary/journal 分居两机——weilandserver `~/openpi/exp/robocasa365/data/ws_search/groot_tp/`（新纪元）与 timan107 `/scratch/zixuans8/openpi_rc365/...`（旧纪元 19 份，⚠ 注意是 `/scratch` 不是 tether 默认 HOME）。两侧各 `tar -czf` 后拉回本地同一目录合并（已验：两源 cid **零重叠**、13 任务集一致），再 `uv run python exp/robocasa365/summarize_ws_search.py --teacher groot_tp` 须 **132/132** complete；不完整的 cid 用编排器 `--only` 补跑。
2. **GR00T 锚点臂（~1.5h）**——命令已亲验（`serve_groot_n15.py:201/313` 无 `--cache-config` 即走 "teacher-only"；`run_ws_search` 全程不按 cid 找 yaml，cid 纯身份，故 `anchor_slot<N>` 无需产 yaml）：

   **① weilandserver 起 3 个 teacher-only server**（须先确认搜索车队已停、端口真空）：
   ```bash
   for p in 23160 23161 23162; do tmux new -s wsanc$p -d "export HOME=/home/weiland; cd /home/weiland/openpi && \
     OPENPI_MONITOR_LEVEL=BASIC PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \
     /home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/serve_groot_n15.py \
     --checkpoint /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000 \
     --port $p --concurrent 2>&1 | tee /tmp/wsanc$p.log"; done
   ```
   就绪判据（两条都要，⚠ 不看 CLI banner）：`serving stack: concurrent teacher-only (no cache)` + `INFO:websockets.server:server listening on 0.0.0.0:<p>`。

   **② timan107 三路 client**（`--role all`，任务 5/4/4 切分，各 8 worker）：
   ```
   slot0 :23160  CloseBlenderLid,CloseFridge,CoffeeSetupMug,OpenCabinet,OpenDrawer
   slot1 :23161  OpenStandMixerHead,PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter
   slot2 :23162  PickPlaceSinkToCounter,PickPlaceToasterToCounter,SlideDishwasherRack,TurnOnSinkFaucet
   ```
   `run_ws_search --role all --teacher groot_tp --server ziyanglin.com:2316N --cid anchor_slot<N> --tasks <上表> --episodes 8 --workers 8 --gpu-ids <2N,2N+1,...> --env-config exp/robocasa365/config/ws_search_timan107.env`（岛内解释器与 PYTHONPATH 同编排器 agent 配方）。
   ⚠ 教师直推每步全量推理，比 cache 档慢，单槽用满 8 worker。

   **③ 分析前必须合三为一**：三份 journal 各只覆盖任务子集，`analyze_ws_search_stats.py` 的全网格判据会把它们全部丢弃。合并就是 `cat` —— 逐条记录的任务名来自 `task_uid`（与 cid 无关），三份任务集互斥，所以 `cat journal_ws1-anchor_slot*.jsonl > journal_ws1-anchor__l1s1_groot_tp.jsonl` 正好拼成 13×8 全格，cid 记作 `anchor`。锚点 cid 不在 index 内 ⇒ 汇总器 weights 列空，属预期。完毕撤 server。
1b. **死任务终裁**（owner 已裁"跑完再弄"）：用 132/132 终版数据复核 4 个 PickPlace 的判定（856→1056 集/任务），落地四点方案（分析层双口径 / 加密轮 9 任务 / 锚点臂保留 13 / pi05 筛选保留 13）。
1c. **pi05 Stage-0 容量探测（owner 2026-08-22 直令：本组做完最先做这个）**：目标 **weilandserver ≥4 个 pi05 replica（至少 3，探测定档）**；timan107 worker 照 GR00T 拓扑打满。方法同原 §10-3（rate 代理=server log FULL_HIT 行/min；显存门 10.5G 三连读；先 1×N 档再多槽线性验证）。4 槽通过则 pi05 臂用 `--slots 23170,23171,23172,23173 --pull-port-base 23184`。
2b. **GR00T 加密轮（~4.5h，§9 2026-08-22 裁决新增）**：先 `analyze_ws_search_stats.py --journal-dir <合并目录> --csv <out>` 取"与榜首配对打平"的集合，按 macro_sr 取 top-8 组成 `--only` 列表。**部署前置**：把 `run_ws_search.py` 推到 weilandserver + timan107 并双机 sha 核对（否则 `--run-prefix` 被 argparse 拒）。再 `orchestrate_ws_search.py --teacher groot_tp --slots 23160,23161,23162,23163 --timan-workers 8 --weiland-workers 2 --run-prefix ws2 --episodes 32 --only <top8>`。产物落 `ws2-*`，与 round-1 并存互不覆盖；分析用 `--run-prefix ws2`。idx 0-7 与 round-1 同 seed ⇒ 同时是复现控制。**另带 2 个 round-1 判为显著劣于榜首的 cell 作阴性对照**（+1.1h），用来验证筛选轮的分离判断本身可复现——否则加密轮只能收窄区间，无法证伪筛选。
3. **pi05 容量探测（~1h，需 GR00T 臂已停、机器干净）**：方法同 Stage 0——1×4 档（serve_policy 配方 @23170 + 真实搜索 cell yaml + `run_ws_search --cid probe_cap --episodes 2 --journal-dir /tmp/pi05_probe` 隔离）测单 server 速率（速率代理=server log 每分钟 FULL_HIT 行数，pi05 无 CSV）；再 3×4 档验线性；显存门槛 10.5G/server 三连读。产出追加进 `t7_capacity_probe.txt` pi05 节 + 回填 §5。
4. **pi05 臂**：`orchestrate_ws_search.py --teacher pi05 --slots 23170,23171,23172 --pull-port-base 23184 --timan-workers <探测定> --weiland-workers <探测定>`（编排器已含 pi05 配方与身份 grep；本地 tmux `wsorch`，log `/tmp/wsorch_pi05.progress.log`——巡检 prompt 里的路径按此替换）。筛选完成后同 2b 做 pi05 加密轮。
5. **pi05 锚点臂**：同步骤 2，serve_policy 无 cache_config（teacher-only），端口 23170-72，cid anchor_slot<N>。
6. **双侧汇总与报告**：两侧 `summarize_ws_search` + `analyze_ws_search_stats`；最终报告 `exp/robocasa365/analysis/ws_search_round1.md`（feedback_analysis_md_location：报告放 analysis/ 纯 md）——含 top-10 表、**配对打平集合（不得只报点估计）**、初始状态支配诊断、iso 族模态诊断、grid2 权重面、rs-dominant 假设检验、加密轮前后对照与同 seed 复现率、锚点对照、与 LIBERO 结论的对读。
7. **收口**：plan/handoff/logs README 状态推进；数据侧 journal/summary 归档说明；**终局 commit push**（含本 runbook 执行期间产生的全部四文件热修与新证据；English message、无 AI 署名）。
8. **timan1 供给（owner 2026-08-22 直令，并行进行中）**：timan1.cs.illinois.edu = 48 核/503G RAM/4×A6000 48G/共享机（**GPU1 他人占用避开**；dehaowu2 等用户在场，共享机纪律全套生效），/scratch 余 923G，**EGL 用户态齐全**（timan108 缺的 10_nvidia.json 这台在位）。步骤：①33G 岛 nc 管道克隆（107 tmux `t1send` → timan1 tmux `t1recv` 端口 29411，含 `.local/share/uv/python` 解释器）——**进行中**；②文件数核对（期望 ~170722+解释器）+ CUDA smoke（GPU0 matmul）+ **EGL smoke（MuJoCo 离屏渲染，这是 108 倒下的那关）**；③编排器接线：TIMAN2 常量泛化为 `--agent-c-host`（默认 timan108）以便指到 timan1；④首发单卡（GPU0）小编制试跑，再按 48 核余量扩。用途：pi05 臂 +N worker，及以后的模拟环境需求。
- **VRAM 振荡定性（20:41Z）**：带逐出后 minfree 仍偶见 ~400MiB 深谷——两点采样证明是**振荡非棘轮**（0/1 卡 150s 内回收 3.2-3.7G、7 号卡整卡清空）：双栈卡两 worker 同持 env + kitchen 构建瞬时双份分配的叠峰。处置=维持 12 worker + 巡检盯 journal error 计数（OOM→respawn→retry 自愈链在位），error 频发才降 3/slot。
- **20:55Z 假警报闭案（时区鬼影）**：一度误判「journal 停摆 + worker 集体重生」，根因=把 cron 本地触发时刻当 UTC 与远端 `ps etime` 混算。三端 `date -u` 对表全同步，`lstart` 证实 12 worker 自 20:25Z 起从未重启。真实吞吐 ~150-300 ep/h 随任务 horizon 混波动（evict-1 的 env 重建随任务切换计入）。**巡检纪律追加：任何时间线取证前先三端 `date -u` 对表**。
- **01:0x-01:4xZ 网线切换 + 本机重启双事件**：owner 重启本机（编排器/监控丢失）并给 weilandserver 换 WiFi→有线（enp8s0 1000Mb/s 全双工）。远端数据面全程未断（driver 自主收官 15/76、NAT 转发无缝、tether agent 3min 自愈）。恢复走零浪费路径（等 driver 收官再重挂 wsorch）。
- **旁路实测裁决瓶颈归因（owner 令：不得无测量下断言）**：23160 空槽搭 timer 实验室（真实 score_sum cell 配方 + 真 episode 159 次推理）：**cp1_search 136.1ms（67%）/ stage1_vision 35.7ms（18%）/ total 202.5ms**——检索主导结论成立，但此前引用的 261ms/74% 是探测配方（rrf 三字段全扫）旧数，不可外推，已修正。⚠ 实验室首轮无效教训：serve_groot_n15 的 `SERVER-LISTENING` banner **先于真实 bind 打印**，旧 slot server 残留占口时 banner 检查会假阳——就绪判据必须用 `INFO:websockets.server:server listening`（编排器因有「等端口真空」前置门不受此影响）。
- **01:42Z 起 4 槽 A/B**（有线解掉 WiFi 墙后重探 S=4）：wsorch 以 23160-63 × 4 worker 重挂，观察聚合吞吐 vs 3 槽的 ~300 ep/h；若无 membw 恶化迹象（per-cell 周期 <70min）则保持。
- **两项工程补丁立项（owner 2026-08-22 直令）**：① GR00T 检索合批——多并发 query 共享一次库扫描（直击 136ms/67% 主项与内存带宽墙，预期单 server ~2×）；② stage1 CUDA graph 编译 + 编译缓存（36ms launch-bound 段）。实现走 fast-track，部署放安全边界（SR 不受延迟影响，臂内可比性无扰）。
- **P1 冻结搜索缓存 shipped（2026-08-22 0x:xxZ）**：`in_memory_backend` 双缓存（过滤结果按指纹 / 字段矩阵+行范数按列表对象弱引用守卫）+ 手写 cosine（`F.cosine_similarity` 广播核在 [369×32768] 实测 43ms/调用是真凶）。真 pkl 认证：**search 136-166ms → 2.62ms（warm，~60×）**；等价性 9 测试 + 全波及面 1545/27/0；失效面=insert/delete 全清。已滚动部署 weilandserver（槽位轮转自然升级；结果逐位等价故臂内可比性无扰）。**推理预算 202→~67ms**，瓶颈翻转：worker 供给不足 + stage1 36ms 串行段。裁决：下一轮界 wsorch 提档 4 槽×5 worker；**P2 合批实证降级**（动机已被 P1 溶解，n20 亦仅 ~10ms）；P3 stage1 CUDA graph 继续（owner 直令 + 现为主项）。
- **02:1x-03:2xZ 提速三连**：① wsorch 4 槽×5 worker 短暂运行；② **P3 compile-stage1 真机等价门 FAIL**（worst token cos=0.8716<0.999，server 拒服务自杀——门按设计工作；实现+门已 ship 但**不启用**，mode 旋钮 `OPENPI_STAGE1_COMPILE_MODE` 留诊断；因 P1 后 server 供给已过剩、瓶颈在 worker 侧，性价比归零，park）；③ **跨机 worker 池 shipped**（owner 令 weilandserver 也跑 sim；WSL 被 owner 否决——23G RAM 养控制面）：run_ws_search 恢复 run_collect 式 --role driver/agent/all + 固定 --bind-port；driver 迁 weilandserver（拉取口 23180-83 公网段，journal/summary 归一到 weilandserver，completeness 双主机查询兼容旧 15 份）；编排器每 cell 三进程（wssrv+wsdrvD@weiland、wsdrvA@timan 5 worker、wsdrvB@weiland 2 worker）=28 worker 总量；孤儿清扫双机执行。island 路径/拉取口/解释器三机验证全绿。
- **03:0xZ owner 纠偏：矩阵补全三相机覆盖（D4 修订）**：我镜像 LIBERO 形制时只搬了「三选三含 rs」的皮——LIBERO 3 字段的 grid3 本就是全字段单纯形，4 字段的忠实移植必须覆盖全相机面。补 **grid3v ×21**（{v0,v1,v2} 三相机单纯形，step .125）+ **grid4 ×35**（四字段全单纯形）→ 矩阵 **76→132 cells/teacher**。全部 264 yaml 双 validator 亲验、8/8 测试绿、已部署 weilandserver（sha 对齐）；wsorch 下一轮界重启自动纳入。成本 +56 cells/teacher ≈ +8h/臂（28 worker 速率下）。
- **02:5xZ 跨机拓扑首链修复**：agent B 全体启动即崩——根因=weilandserver 的 run_collect.py 旧版缺 `max_cached_envs` 形参（此前修复只推了 timan107）。补推三件套后 agent B×4 复活，本机 8 worker 就位，28 worker 满编。⚠ 教训：多机部署的文件推送必须列清单核对每台，"这台现在不跑该角色"不是跳过理由。
- **05:2x-06:0xZ 中期统计读数：8 trials/task 分辨不出榜首（方法学发现 + 计划追加）**。排空期用 41/132 已完成 cell 做了正式统计读出（新脚本 `analyze_ws_search_stats.py`，读 journal 而非 summary 以保留逐 episode 结果）：
  - **配对是合法且必须的**：`episode_runner.py:367` `seed = base_seed + orig_init_state_idx` 且场景钉死 (1,1) ⇒ 同 `(task, idx)` 在**每个 cell 都是同一初始状态**，cell 间比较天然配对。
  - **初始状态支配结果**：**49% 的初始状态在所有 cell 上判决一致**（全成或全败），跨状态方差是独立零模型的 **22×**。检索权重能动的只是剩下那一半。
  - **分辨力不足**：配对符号翻转检验（20000 次重抽）下，**15/41 个 cell 与榜首统计打平**（榜首 `grid3 v1@37/v2@25/rs@37` macro_sr=0.260，第 2-11 名 p 值 0.06-0.55）。非配对口径更差（19/40 打平）。⇒ **round-1 的 8 trials/task 只能筛掉差配置，选不出冠军**；报告不得宣称"某配置最优"。
  - 测量面本就窄：13 任务里 4 个 PickPlace 族在所有 cell 上恒为 ~0.02，spread 全部来自 CloseFridge/OpenDrawer/OpenStandMixerHead/TurnOnSinkFaucet 等 5-6 个任务。
  - **裁决（无人值守独断，owner 醒后可推翻）**：采用**筛选—加密两段式**。GR00T 锚点臂之后插入 **加密轮**：对配对打平的 top-8 用 `--episodes 32` 重跑（8×13×32=3328 ep ≈ **4.5h** @741 ep/h），顺带白得一个**同 seed 复现控制**（idx 0-7 重跑，用来量化"49% 一致"里有多少是确定性的）。pi05 臂同构（132×8 筛选 → top-8×32 加密）。**不追单一冠军**：把 top-3 的 0.03 差距做显著需 ~80 trials/task，科学价值不抵成本。备选（不加密）代价=报告只能给"区域"不能给点估计，且 pi05 臂大概率复现同样的平局。
  - **加密轮档位用实测不一致对外推验证（非拍脑袋）**：榜首与前 11 名之间，104 个配对 episode 里判决不同的只有 **7-13 个**（≈10%）——**权重配置大约每十个 episode 才改变一次结果**，这是本轮真实的效应量。按 z ∝ √n 外推：**32 trials/task 把打平集合从 10 收到 ~3**（第 4 名起 p<0.012），64 trials 才可能收到 1。⚠ 该外推假设"观测差=真差"，而这假设恰在 top-2/3（z≈0.9）上最不可信：若真差为零，再多 episode 也分不开，且那是正确结论。故取 32 档、**不追单一冠军**。
  - **配套代码（本地改完，测试绿，远端未部署）**：编排器的 `summary_complete()` 按 `ws1-<cid>` 键 ⇒ 加密轮会被当"已完成"跳过。新增 `--run-prefix`（默认 `ws1` 行为逐字不变）贯穿 run_ws_search/orchestrate/summarize/analyze；⚠ **编排器只在非默认时才拼出该 flag**——否则中途重启会用新编排器去驱动仍是旧 `run_ws_search.py` 的远端，argparse 直接拒。远端部署推迟到加密轮启动前（届时须双机 sha 核对）。新增 `tests/robocasa365/test_ws_search_stats.py` 8 测（含网格对齐回归：一个在飞 cell 恰好完成整数个任务会骗过"自身任务数×episodes"判据），全波及面 **209 passed/15 skipped**、ruff 净。
- **06:1xZ 队列改为分层交错（修既有教训的根因）+ 零浪费界重挂**。核对中期榜时发现：已完成的 44 个 cell **全部来自 grid3/grid3v 两族**——`sorted(index)` 是按族分块的（且 `'3' < '_'` 使 `grid3_*` 排在 `grid_*`(=grid2) 之前），于是 grid2/grid4/iso 三族一个未测，**iso 四个单字段对照（解释整个权重空间的锚）被排到最末**。此前记忆里"部分结果+字母序会误判"正是此因。处置=编排器新增 `stratify_by_family()` 与 `--cid-order stratified|sorted`（默认 stratified）：按族轮转，任何前缀都是均衡样本（实测前 24 个覆盖 5/5/5/5/4，而字母序前 24 个 100% 是 grid3）；族内保持排序故结果确定。切换按**零浪费界**：停本地 wsorch → 4 个在飞 cell 由远端 driver 自主收官（summary 28→32）→ 用新序重挂（已完成 cell 由 summary 双源检查跳过）。⚠ 同时提醒：**05:5xZ 那份中期统计读数（榜首 0.260 / 15-way 打平）是 grid3+grid3v 偏样本上的结论**，全矩阵排空后必须重算，不得直接引用。测试 211 passed/15 skipped，ruff 净（`baselines/pi05_step0b_client_ORIGINAL.py` 的 8 条属逐字存档，有意不改）。
- **06:0xZ 后续阶段可行性预检（全绿，均本地/只读）**：① 锚点臂命令亲验并写进 §10-2（`serve_groot_n15.py:201/313` 无 `--cache-config` 即 teacher-only；`run_ws_search` 不按 cid 找 yaml，`anchor_slot<N>` 无需 yaml；三份任务子集 journal 用 `cat` 即可拼成全格，因任务名来自 `task_uid` 与 cid 无关）。② pi05 就绪判据复核：`start_server` 要求"端口在听 **且** 日志含本 cell 的 yaml 路径"合取，且启动前有端口真空门 ⇒ pi05 只有一条 grep pattern 也安全，**不加 listening pattern**（该行未必出现，加了反而会让整臂 6min 超时失败）。③ 两侧各 132 个 yaml 对 index 做十项不变量校验（权重与 index 一致且和为 1、field_similarity/score_normalization 恰好覆盖正权重字段、strategy/key_builder/rs 维/preload 路径、cp3 关、timer 关、gate/judge、write_policy never）——**0 problem**。④ 两个 n5 pkl 在 weilandserver 就位（groot 1.94G / pi05 2.75G，+42%），/data 余 2.8T。⚠ 首次校验脚本对全部 264 个 yaml 报 NO_WEIGHTS，是我按错误 schema 找 `weights`（真实结构是顶层 `keys.<field>.weight`）——诊断工具先跑控制组的教训第二次生效。
- **09:52Z 首例 INCOMPLETE（孤例，已排队重跑）**：`grid_vision_1@50_robot_state@50` 21min 早退，n_err=0/n_missing=50（后 7 任务只各跑了 1 集）——driver 正常返回但 50 个 uid 无终态（重试耗尽形状）。排查：timan107 无新 OOM（最近为 8/21）、worker 32 满编、GPU 余量健康、server 温度正常；每 cell tee 覆盖日志故事发窗已灭失，无法定根因。**1/64 发生率（1.6%）判偶发**，complete=false 使其自动落入 --only 重跑网。⚠ 排查中两次险被误导且都是已记录过的坑：timan107 `ls`/`tmux` 显示本地时（UTC-5，"5 小时前的日志"其实是刚写的）；`wsdrvA23160` 会话"消失"只是撞进 47s 的 cell 过渡窗。配套：summarize_ws_search 的总计行从"文件数/132"改为 **complete 数/132** 并列出 RERUN 清单 + 现成 `--only` 串（文件数会把 INCOMPLETE 报成全绿）。
- **11:50-14:31Z 静态分片尾部空转事故（-2.5h）+ 共享队列修复**：23161/23162 两槽在 11:50/12:06 排空退场，**20/40 worker 空转 2.5h**（剩余 25 cell 全押在另两槽）。三重根因：① `cids[i::4]` 静态分片只均衡**计数**不均衡**工作量**（片里混着秒级跳过的已完成 cell）；② 排空的槽无法从忙槽取活；③ "queue drained" 通知与 cron 巡检恰在该窗口双双静默——**已定性：本地 wsorch 全程在写日志（10-14 点每小时 10-19 行，无 >20min 断档），冻结的只是会话事件递送**（cron 触发与 Monitor 通知被排队，14:28 一并补送）。实验数据面独立于会话零损失；损失的是介入时机。纪律追加：**会话事件递送不可尽信，递送恢复后第一动作=全量对账（summary 计数 + 各槽状态），不要只处理刚送达的那条事件**。止血=用 summarize 的 `--only` 清单（25 cell）显式重发射，四槽即刻全忙。治本=编排器改**共享工作队列**（`CellQueue.pop()` 槽空即取，deque+锁，2 条并发不重不漏测试），未来发射（锚点/加密/pi05）自动受益；当前在跑臂不弹跳（--only 片已工作均衡，弹跳收益≈0）。测试 213 passed/15 skipped，ruff 净。
- **14:4xZ 死任务识别 + owner 裁决（先跑完再弄）**：107 cell 全量证据（856 集/任务）判定 4 个 PickPlace 任务在任何权重配置下均 ~1.5% 地板率（PickPlace{SinkToCounter 11/856, CounterToStove 13, CounterToCabinet 14, DrawerToCounter 14}成功），零分辨信号；CoffeeSetupMug 3.7%/OpenCabinet 5.4% 仍有梯度不剔。**owner 令：132 格照跑不动，剔除工作全部推迟到排空之后**。届时按四点方案执行（分析层过滤双口径 / 加密轮 9 任务省 31% / 锚点臂保留 13 作教师判据 / pi05 筛选保留 13 跨教师可比）——排空后用 132 格终版数据复核这 4 个任务的判定再落地。`analyze_ws_search_stats.py --exclude-tasks` 已实现（分析侧惰性改动，不影响运行面），冒烟与 §10 措辞更新同样推迟。
- **12:44 本地 GR00T 筛选轮收官：132/132 全 complete**（12:29 首排空 → 唯一 INCOMPLETE `grid_v2@50/rs@50` 单槽 resume 14min 补完 → 12:44 终核 0 incomplete/0 never-run）。第二例 INCOMPLETE 与第一例同形（~50 missing、0 err、16-21min 短退），全靠 journal resume 无损补齐；两例发生率 2/133 起跑 ≈1.5%，未再深挖（发生窗日志被下一 cell tee 覆盖）。**终版统计（132 journal 配对）**：榜首换为 `grid2 v2@87.5/rs@12.5`=**0.269**，top-9 几乎全为 v2+rs 系；**打平集合 48/132**（α=0.05 配对符号翻转）——8 trials 只能划出"好区域"；**权重边际（全平衡设计）**：v0/v1 随权重单调降（0.172→0.019）、v2 单调升（0 权 0.117→高权 0.18-0.20）、rs 有>0 即饱和（0.120→~0.17-0.19 平台）；初始状态支配：38% 状态全 cell 同判、跨状态方差=独立零模型 62×。**①b 死任务终裁（1056 集/任务终数）**：四 PickPlace mean 1.1-1.7%、总成功 12-18/1056——判定成立，落地口径="纯 cache 不可迁移任务"（teacher 0.4-1.0），从搜索指标剔除双口径呈报。机群清场核验：timan107 零残留；weilandserver 残留 2 个脱 tmux 的 serve_groot_n15 按 PID 定点回收（显存回满 48G）。
- **13:2x 本地 pi05 Stage-0 探测完成（终点实验，owner 目标 PASS）**：GR00T 清场后逐档 1→3→4 server（:23170-73，每档每 server 26 集真实短评测 @8 worker/客户端，隔离 journal /tmp/pi05_probe）。**三档 wall 几乎不变（~9.5/9.5/10 min）⇒ 4 replica 线性成立**：165 → 490（3.0×）→ **625 ep/h（3.8×）**；全程 0 err/0 missing；驻留 7.9G/server（4 台 31.5G/余 17G，第 5 台过 VRAM 门未实测）；GPU 57°C keepwarm 全程在岗。**裁决：weilandserver 4 pi05 replica PASS（≥4 目标达成）**，timan107 32 worker 无客户端瓶颈。证据 `t7_capacity_probe.txt` pi05 节 + round1 报告 §6。探测机群已全撤（显存回 48G）。
- **12:3x 本地 X15 暂存污染摘除**：RL Router 线 G2 NEEDS REVISION 中一条涉本线——X15 的 `git add` 把本线未 commit 的 `in_memory_backend.py`（P1）扫进其暂存快照。已 `git restore --staged` 摘除（暂存内容与本线工作树逐字节同，工作树未动分毫）；其余 24 个 X15 暂存文件未触碰。跨 session 共享 checkout 的教训：stage 必须按显式路径清单。
- **11:3x-11:5x 本地 timan1 供给完成（owner 直令，§10-8 ①②③全过）**：timan1.cs.illinois.edu＝48 核/503G/4×A6000 48G/共享机（GPU1 他人占用避开），/scratch 余 923G，**EGL 用户态原生齐全**。33G 岛 nc 克隆（29411 口，tmux 两端）文件数 **174630 双端一致**（含 uv 解释器）；CUDA smoke ✓；EGL 离屏渲染 ✓（退出时 `__del__` EGLError 为清理噪声，功能无碍——**与 timan108 的真死区分开**：那台连 context 都建不出）；**完整 robocasa kitchen 建env+reset 验收 ✓**（OpenDrawer@(1,1) seed 1M，三相机 512² 出图）。编排器接线：`--agent-c-host`（默认 timan108，可指 timan1）已落地（`global TIMAN2` 覆写；⚠ 第一版两个语法/语义坑：先读后 global 声明会 SyntaxError、无 global 的赋值只建局部变量——已修正），213 测试绿 + 空队列 smoke 过。timan1 即插即用：`--timan2-workers N --agent-c-host timan1`（首发单卡 --timan2-gpus 0，避 GPU1）。
- **04:2x-05:1xZ timan108 扩容战役（半胜收兵）**：发现整机空闲的 timan108（48核/251G/3×A5000 24G 全空），nc 管道克隆 33G 仿真岛（文件数 170722 双端一致 + uv 解释器补传）。三重坑逐一定位：① ssh 集群锁 pubkey → nc 裸管道绕过；② **坏卡幽灵**（/proc 见 4 张 A5000，PCI c1:00.0/UUID 7637f928 驱动初始化失败被 smi 隐藏）→ 设备号空洞使 CUDA 全灭 → **userns shim（unshare -rm + bind /dev/nvidia1→0）救活 CUDA**（matmul 实测 OK，无 root）；③ **EGL 用户态库压根未安装**（纯计算节点）→ 从同驱动版本的 107 拷 GL 库族 + 自定义 vendor json——但 EGL context 创建仍败（/proc 死卡掩蔽亦无效，根因在更深层）。**结论：CUDA-only 任务可用（shim），sim worker（需 EGL）不可用；彻底修复需管理员重启（顺带治愈设备号空洞）**。33G 岛已留在 /scratch/zixuans8 备将来。CPU 渲染（osmesa）路线证伪：库缺失且性能上不可行。⚠ 教训：egldev 枚举 probe 在健康机上同样返回 0——诊断工具必须先跑控制组，本次为此浪费两轮推理。
