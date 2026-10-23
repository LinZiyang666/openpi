# W1：R17 线新种子产物与合并建批（DP/Fast-WAM/Cosmos LIBERO + 全部 MT50）

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守）和 seeds_1023/DESIGN.md。

## 范围
1. **LIBERO（DP、Fast-WAM、Cosmos）**，新子集种子 20261003、20261004，库大小 50/100/200/300/400（这三个模型有 6 档，500 是全集不做）。B 池父库不变，不重采。
2. **MT50（GR00T、DP、Fast-WAM、π0.5）**，seed 9 / seed 10 新库（由 W3 采集并 finish 出 READY.json、`subsets/size_{5..50}`、`fits/size_N`；库根 `/data/r17_library/seed{9,10}/<m>_mw_mt50/bpool`），库大小 250–2500 全做。
3. 变体：全步/减步 × 长执行段开（C10）/关（C5）。**Cosmos（C16）和 π0.5 MT50（C5）没有长执行段，只做默认段长的全步/减步两组，不改它们的模型代码。**

## 每个新库/子集要产出的东西（按现有流程，按新库重新标定）
- 子集视图（LIBERO：每任务 `default_rng(种子+任务号).permutation` 取前 N/10，嵌套；用 W3 给 `teacher_library.nested_views` 加的输出根参数写到新根，不经过 `library_alignment.audit_alignment`）。
- C10 拟合 + C10 段头 Q3（长执行段开，`longseg_1017/recalibrate.py` 流程；`labels()` :58-65 改成按种子传入保留局）、C5 拟合 + C5 Q3（长执行段关，`longseg_off_1022/fit500.py` 的 fit_cell 参数化 view 目录与 commit；单线程 OMP=1 才逐位复现）、Cosmos C16。
- 成功过滤：每格 kept = 视图局 ∩ 成功局（`filter_failed/audit_counts.summarize`），按 fit digest 写 selection。
- gap：长执行段开按新库 `recalibrate.solve`；长执行段关沿用同种子开的 gap（owner 10-22 裁定关不重标）；Cosmos `cache_knob.calibrate`。
- 每个新根的 manifest（`longseg_1017/staging.py:46` 要），新根实目录放 /nvme，/data 下放软链（照 `fit500.py:23`）。

## 建批（常驻复用，owner 10-23）
- 第一步（旋钮关）按「模型 × benchmark × 步数 K × 卡」合并成一批：两个种子 × 全部库大小 × C5/C10 的旋钮关格子放进同一个 campaign，一次加载模型，按 YAML 热切换。YAML 文件名、group 名、分析键、evidence 键都必须带种子和段长（`longseg_off_1022/build.py:163,168`、`formal_a_analysis.py:86`、`knob/common.py:99-118`、`evidence.py:16-28`），否则互相覆盖。每个库的 cache_inputs 前缀要不同（`cache_study.py:437`）。
- 第二步（各档）：旋钮关完成后，只对「不完整或旋钮关成功数 < 同卡同步数纯推理」的格子逐档下降（LIBERO 用 `longseg_off_1022/workflow.py` build_levels 的 `formal_a_queue.classify` 规则；MT50 用同样的两步：现有 MT50 判格规则见 `knob/descent.py:50-51`，按 owner 规则以同卡同步数纯推理为参照，写清你采用的判据），各档也按同样方式合并成批。
- 参照直接复用（`longseg_off_1022/REFERENCES.json` 等，见 DESIGN.md），不新跑。
- 部署：沿用 `longseg_off_1022/workflow.deployment()`（DP/FW 在 H100 级机器、Fast-WAM campaign 带 `fastwam_text_table`）；生成 `filter_failed_knob/commands/{prepare,launch}_<名>.sh`。批次名建议 `fa_sd_<模型>_<r|f>_<l|m>_<步骤>`，自己定但要短且不与现有名字冲突。

## 必须改成报错或带种子的（DESIGN.md「会静默出错的地方」）
- `filter_failed_knob/runtime.py:22-24,82`：selection 只在常驻启动时读、缺 digest 悄悄退回不过滤 → 改成每个 fit 加载时从它旁边的 selection 文件（按 fit digest）惰性读取，缺了直接报错。这样任何活常驻装新 YAML 都能正确过滤（W4 的跨批接管依赖这一点）。
- `filter_failed_knob/build.py:156-167` cells() 只按 (模型, suite, size) → 加种子。
- 已存在就复用的地方（`longseg_off_1022/build.py:80-83` 等）→ 路径带种子。
- `longseg_off_1022/workflow.py:79-80` 查不到 C5 fit 悄悄丢格 → 报错。

## 文件归属（只改这些，新代码放 seeds_1023/）
新模块 `exp/offline_search/rounds/r17/seeds_1023/`（inputs.py、build.py、workflow.py 等）；最小钩子改动允许：`exp/r17_integration/ops/cache_study.py`、`ops/cache_knob.py`、`filter_failed_knob/runtime.py`、`filter_failed_knob/build.py`、`filter_failed_knob/common.py`、`filter_failed/audit_counts.py`、`filter_failed/stage.py`（拔 DESIGN.md 列的门）、`longseg_1017/recalibrate.py`、`longseg_1017/staging.py`、`longseg_1017/sync_back.py`、`longseg_off_1022/build.py`、`longseg_off_1022/fit500.py`、`longseg_off_1022/workflow.py`、`exp/r17_integration/ops/formal_a_analysis.py`、`knob/common.py`、`evidence.py`（按实际路径）。不改：W3 的 teacher_library/cache_recipe/library_campaign，W2 的 r10/q3_port_1020/formal_a_queue，W4 的 formal_eval/formal_finalize/formal_prestage。

## 交付
按 COMMON.md。协调者命令要分：LIBERO 产物（可立即在别的 CPU 机器跑：说明需要同步哪些输入）、MT50 产物（等 W3 采库 finish 后）、建批、两步流程。回归检查：用旧种子 20261002 / seed 8 走一遍新代码，生成的 fit/Q3/selection 与现有产物逐位比对（列出比对哪几个文件）。
