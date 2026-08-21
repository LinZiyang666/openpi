# RoboCasa365 纯 cache 检索参数搜索（weighted_sum × spatial_pool_16 × n5 库）— Plan

**Status**: `In Progress — harness 就绪，待里程碑 commit 授权后部署开跑`（Stage 0 GR00T 侧 ✅；harness 5 件 + 测试建成并过双 agent 自审（§8）；owner 裁定：trials=8/task、layout-by-seed 记录 §2-D2b、eval 种子=D-E 段 base 1_000_000）
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
- **⚠ pi05 侧 3 相机在线 key 通路未验证**（groot_cache_integration P2 遗留）：库按 `--vision-slots 3` 建，serve_policy `--cache_config` 在 pi05_robocasa 上的在线 key 构建须先过接线门。
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
