# Plan: weighted_sum 系统化实验 4 阶段在 libero_10 上全量复刻

**Status**: `In Progress`（G1 APPROVED R3 / Post-G1 polish / §4 Code 完成 / pre-G2 自审已修 / **G2 APPROVED R2 2026-05-29（tests/exp 616 passed）**）
**Level**: **L2**。代码足迹（全在 `exp/` 内、向后兼容、零改 `src/`）：
- ④ Stage4 kinematic `preload_pkl_override`+`cfg_id` 双透传（spec/super_warmup/runner 3 builder, ~60-80 行）+ libero_10 `CFG_SPECS` 条目；
- ② `emit_trajectory_yamls` 的 `_EXPECT_BASE`/`_EXPECT_OVERLAP` 断言放宽；
- Stage1 `refine_round.py` 加 `rank_keybuilder(results, stem)`（支撑 winner 平局时双 keybuilder 各自 refine）；
- 新 `emit_top10.py` 小脚本 + 全部配套测试。
其余为数据预收集 + config 生成 + 评测运行，走 experiment-lifecycle skill。
**Authority**: Execution
**Date**: 2026-05-29
**Owner**: Ziyang Lin
**前序实验（libero_spatial 原版）**:
- `logs/weighted_sum_two_layer_refactor.log.md`（Stage 1 权重搜索）
- `logs/weighted_sum_trajectory_search.log.md` + `logs/weighted_sum_trajectory_weight_research.log.md`（Stage 2）
- `logs/weighted_sum_threshold_pareto.log.md`（Stage 3）
- `logs/weighted_sum_kinematic_phase5_replication.log.md`（Stage 4）
- `exp/weighted_sum/RESULTS.md`（libero_spatial 结果总览）

**同类移植先例**: `logs/verdict_phase5_libero10_systematic_sweep.log.md`（verdict phase5 → libero_10，已踩平 v2_spec preload 硬编码 / pkl enrich / 共享 warmup race 等坑）

---

## 0. 一句话

把 libero_spatial 上跑过的 weighted_sum 4 阶段系统化实验（① 两层校准+权重搜索 → ② trajectory → ③ threshold-pareto → ④ kinematic phase5 复刻）**原样移植到 `libero_10` task suite**。代码范围：**①③ 零代码；② `emit_trajectory_yamls` 断言放宽；Stage1 `refine_round.py` 加 `rank_keybuilder`；④ ~60-80 行（双透传 + CFG_SPECS 条目）；新增 `emit_top10.py`**——全 `exp/` 内、零改 `src/`。真正的工作量在**数据预收集 / warmup 阶段**（init_map / calibration / Stage3 warmup 分布 / Stage4 super warmup raw，见 §4）+ **运行纪律**（§2.3 决策阶段单 server / §4.0 server 数据双推 / §8.6 路径 override）。

---

## 1. 背景与目标

### 1.1 目标
在 `libero_10`（10 个长程任务，区别于 libero_spatial 的 10 个空间任务）上完整复刻 weighted_sum 系列 4 阶段，得到 **libero_10 自身内部可比**的结果。底层检索机制（两层 `weighted_score_sum_knn`：Layer-1 per-field 归一化 + Layer-2 加权和）、judge、防泄漏门、分析管线全部沿用，**仅切换 task_suite + 重新拟合/搜索 libero_10 自己的参数**。

> ⚠ **不 over-claim 跨 suite 可比**（自审 E2-OC 修订）：libero_spatial 版锁单 GPU，本次双 H200，RESULTS.md §7 实测跨 GPU SR 差 ~7pp（最大 18pp）≫ 5pp 决策边界。故本实验**只声称 libero_10 内部可比**，绝对 SR **不**与单 GPU 的 libero_spatial 数值直接对齐。且 §8 kinematic 的 factor recipe/权重是从 phase5 **移植的固定仪器**（未在 libero_10 重新调），Option B 是"**检索层**忠实复刻"，非整条链重调。

### 1.2 非目标
- 不改任何 `src/openpi/` 代码。
- 不重构两层架构 / config schema / backend 契约 / conductor 核心。
- 不引入 libero_object 或其它 suite。
- 不新增检索算法 / 归一化方法（沿用现有候选池）。

### 1.3 关键审计结论（基础设施已就绪度）

| 阶段 | 代码改动 | 依据 |
|---|---|---|
| ① 权重搜索 | **零**（`run_phase2 --task-suite` 参数化、`emit_yamls`/`calibrate` 全 CLI）| `run_phase2.py:53/119/129`、`weight_search_strategy.py:30`、`calibrate_score_normalizers.py` 任务无关 |
| ② trajectory | **近零（一处断言放宽）**：`select_base_configs` 从 `--results-csv` 动态推导 base，但 `emit_trajectory_yamls.py:63-64` 的 `_EXPECT_BASE=18`/`_EXPECT_OVERLAP=4` 是 **libero_spatial 算术常量**，libero_10 上 overlap 几乎必不等于 4 → `:154-155` `AssertionError`。须放宽为下界/按数据重算（自审 E2-GAP）| `emit_trajectory_yamls.py:63-64/154-155` |
| ③ threshold | **零**（`emit_threshold_yamls` base 走 `--base-yaml`；`run_phase2 --per-step-out` + `examples/libero/episode_runner.py:49` `cp1_score` 已落地）| `emit_threshold_yamls.py`、`solve_thresholds.py`、`summarize_inf_ratio.py` |
| ④ kinematic | **exp/ 改动（比"一处"大，自审 E4-VR）**：(a) `--preload-pkl-override`（runner.py:730 CLI 已存在）透传进 3 个 builder——但只换 pkl**不换权重/μσ**；(b) Option B 还需把 **`cfg_id` 透传**进 `build_super_warmup_yaml`/`_build_always_warm_yaml`（二者硬编码 `CFG_ID_DEFAULT`/`"spatial16_ws_d1_best"`、**无 cfg_id 形参**；改 `CFG_ID_DEFAULT` 被禁=破坏 libero_spatial 复现）+ 新 CLI `--cfg-id`；(c) 新 `CFG_SPECS` 条目 | `kinematic/spec.py:293`、`super_warmup.py:137/173`、`runner.py:96/244/287/308/330` 直吃 `cfg["preload_pkl"]`/`CFG_ID_DEFAULT`（硬编码 `v2_spec.py:155`）|

---

## 2. 设备拓扑 + libero_10 数据资产现状

### 2.1 设备拓扑（owner 已定 2026-05-29）
- **server #1**: jupyter-ziyang10（H200 NVL 140G / cgroup 10C·32G / NAT→`tether expose` name `ziyang-srv`）
- **server #2**: jupyter-xuanlel2（H200 NVL 140G / cgroup 10C·32G / NAT→`xl-srv` / 无系统 tmux·fuser / NFS uid 坑 → dump root 用 `/tmp/xl_*`）
- **client**: timan107（48 logical CPU / 8×GTX1080 仅跑 sim·conductor / NAT）
- broker: pc732 (`weiland.top`)；a100 OFFLINE（如需 failover 兜底需 owner 先恢复）
- ⚠ **两块 H200 为不同物理 GPU** → 跨机混跑结果须标注 "different physical GPU"。libero_spatial 版**刻意锁单 GPU**（RESULTS.md §7：跨架构 SR 差 ~7pp ≫ 同机噪声 0–0.6pp）；本次双 jupyter 由 owner 拍板接受混跑可比性代价（§13-Q5）。

### 2.3 ★ 阶段内单 server 钉死规则（自审 E2/E3 BLOCKER）

跨 GPU ~7pp ≫ 各处决策边界（Stage1 winner SE ~4.5pp / Stage4 5pp 门 / Pareto 5pp dominance）。conductor 的 capacity-aware `assign_servers` 会把**同阶段**的不同 yaml 分到两块 GPU → 一个 config 落 ziyang10 还是 xuanlel2 能使其 SR 漂移 > 选取边界 → **翻转 §3.1 的"确定性" winner 选取**。`always_hit` 纯回放无 offline-calib 吸收，漂移直接落在 SR 上。

**规则**：**任何"内部比较决定 winner/决策"的阶段必须钉死在单一 server**：
- **Stage 1（选 winner keybuilder + top10 + spatial16-d1-best）**：整个 base-136 + 3 refine **跑在同一台**（如 xuanlel2，replica 多、吞吐高）；另一台只做与选取无关的吞吐复制或留给后续阶段。
- **Stage 2a trajectory**：depth-Δ 对照其 d1 基线（来自 Stage1）——必须与产出该 d1 基线的**同一台** GPU。
- **Stage 3 / Stage 4**：Pareto cell / kinematic cell 的内部比较同样单 server（kinematic offline-calib 只吸收检索分漂移，**不**吸收 SR 漂移）。

跨 server 仅用于**独立阶段之间**或纯吞吐，绝不用于决策边界 ≤5pp 的阶段内比较。代价：双 server 的并行吞吐优势在选取敏感阶段不可用（owner 已接受混跑，此规则是为保住 §3.1 选取有效性的下限约束）。

### 2.2 libero_10 数据资产

| 资产 | 状态 | 处理 |
|---|---|---|
| 6 个 CP1 库 pkl（mean/max/spatial16/spatial64 + 2 CLIP）| ✅ 已存在 `exp/common/data/cache_artifacts/libero_10/` | 直接复用 |
| spatial_16.pkl 的 `payload.factors`（64 keys）+ `library_stats` | ✅ 已为 verdict phase5 enrich（有 `.pre_phase5.bak.pkl`）| Stage 4 需**验证** super_warmup_declared_keys ⊆ 64 keys（§8.4）|
| H5 原始轨迹（50 文件 / 8.6 GB / 44 唯一 ep）| ⏳ 后台从 ziyang10 拉取中（`bxfk4kmqb`）| init_map 生成用；见 §4.1 |
| **`libero_10_init_map.json`** | ❌ 缺失（ziyang10 也无）| **远端生成**（§4.1）|
| **libero_10 `calibration_normalizers.json`** | ❌ 缺失（现有是 libero_spatial）| Stage 1 Phase-1 重跑（§4.2）|

---

## 3. 阶段依赖 DAG（移植硬顺序）

```
Stage 0 (前置数据): init_map + Phase-1 calibration  ─┐
                                                     │
Stage 1 权重搜索 (base 136 + refine R1/R2/R3) ───────┤  产出 libero_10 winner / all_results.csv / top10
                                                     │     │
                              ┌──────────────────────┘     │
                              ▼                             ▼
   Stage 2 trajectory (N_base×depth + 78×depth wsweep)    Stage 3 threshold (base=Stage1/2 winner)
                                                            │
   Stage 4 kinematic (base = Stage1 winner-kb d1-best)  ──┘
```
**铁律**：Stage 1 必须先在 libero_10 上跑出 winner，②③④ 才能确定各自 base。Stage 2/3/4 之间相互独立、可并行。

### 3.1 ★ 阶段间 handoff 筛选准则（确定性规则；具体数值各 stage 中期临时定）

> owner 拍板：plan 只把**怎么筛**写死成确定性规则，**筛出来是哪个/哪些**等结果出来后中期临时决策。下列每条 handoff 的"准则"是 plan 锁定的，"值"是占位/待定。

| Handoff | 确定性筛选准则（plan 锁定）| 值（中期临时定）|
|---|---|---|
| **Stage1 内：winner keybuilder** | `refine_round.pick_best_keybuilder`（n≥50, per-kb max SR）取最强；**⚠ 平局条款（自审 E2-1a）**：`max()` 在并列时按 dict 插入序静默裁决——不可接受。**规则：若 top-2 keybuilder 在 1 SE（n=100 时 ≈±4.5pp）内并列，则两个都 refine、都带入下游**（复刻 libero_spatial 对 spatial_16≈max_pool 的处理），不让插入序决定 | 看 libero_10 base+refine SR |
| **Stage1 → 全实验 top10** | 全部 config（base+3 refine，按 yaml_id 聚合 SR 均值）**SR 降序前 10** → 落 `config/top10/libero_10/` | 待 Stage1 跑完 |
| **Stage1 → Stage2 base（动态 N_base 个，G1 R2 Item3）** | **完全由 `emit_trajectory_yamls.select_base_configs` 自动算**：per-keybuilder「top1 + top2 + 倒数第二（A 口径：仅正规权重网格、同 zscore、排除 `__norm2`/`iso_`）」∪ 全实验 top10（恰 10），去重 → `N_base = len(base_ids)`。⚠ **libero_spatial 恰为 18（overlap=4）是其结果的算术巧合；libero_10 的 overlap 几乎必不等于 4 → N_base ∈ ~[14,22]**。故 `_EXPECT_BASE=18`/`_EXPECT_OVERLAP=4` 旧断言**必须放宽**（§1.3②），下游 yaml/ep 预算按 `N_base` 实算（§6） | `N_base` 从 `all_results.csv` 确定性导出，**无需人工**；值待 Stage1 完成 |
| **Stage1 → Stage4 base（kinematic d1-best）** | **whichever keybuilder 赢 Stage1，取其 SR 最高的 depth-1 config**（自审 E2-1c：不预设 spatial_16 必赢）。若 spatial_16 不是 winner，二选一并写明：(a) kinematic base 换成 winner 的 pkl/权重；(b) 为与 phase5 跨 suite 对照，**固定 keybuilder=spatial_16**（接受其可能非 libero_10 最优）。其权重 + libero_10 calibration 对应 keybuilder 的 `selected` μ/σ → 填新 `CFG_SPECS` 条目（Option B）| 待 Stage1 结果 + owner 选 (a)/(b) |
| **Stage1/2 → Stage3 base（threshold）** | **Stage1/2 全实验 mean SR 最高的 config**（n≥50）；若某 base 的 trajectory(depth>1) 版 SR 高于其 d1 则用 trajectory 版，否则 d1 winner。个数 1（owner 可加几个）。⚠ Phase-A 分布**展度**是 fail-fast **门**（threshold_pareto §3），**不是**选取键——不要把"聚合分展度"当 base 选取依据（自审 E2-1b）| **唯一真·临时决策**：看 Stage1/2 SR 表后定 |

> 即：除 Stage3 base 是看结果临时拍板外，其余 handoff 都从 `all_results.csv` 按确定性规则导出。**两个自审补漏**：(E2-GAP1) **top10 无现成生成脚本**——libero_spatial 的 `config/top10/` 是人工挑的；libero_10 需写一个 "SR 降序前 10 + rank-10/11 SE 并列处理" 的小生成步骤（或人工挑并记录判据）。(E2-GAP2) 18-base 的 `_EXPECT_BASE=18`/`_EXPECT_OVERLAP=4` 是 libero_spatial 常量，须放宽（见 §1.3 ②）。

---

## 4. 数据预收集 / warmup 统一清单（★ owner 重点关注）

所有"主跑前必须提前到位"的数据，按依赖顺序：

### 4.0 ★ server 端数据同步（自审 E3 BLOCKER — 当前未满足）

offline calibration 在 server **bind 时读本机文件**（`config.py:2241 _load_calibration_samples_offline` → `Path(off.path).is_file()`），且 `SUPER_WARMUP_RAW_RELPATH` 与 `preload_path` 以**相对路径**写死进每个 yaml，**无 per-server override**。故每个用到的资产必须在**两台 server 同一相对路径**各有一份且 byte-identical。实测现状：

- **ziyang10 的 `cp1_spatial_pool_16.pkl` 与本地 sha256 不一致**（ziyang10 `f13517…` 1103155631 B vs 本地 `de5173…` 1103155623 B，差 8 B）——独立 enrich 出来的。**须先定 canonical 副本，再同步，三处 sha256 必须相等**才能开 Stage 1。
- **xuanlel2 根本没有 libero_10 enriched pkl** → 首次 `load_cache_config` 直接 `raise library_stats is None` 崩掉该 server 整片。
- 同理 Stage 4 的 `super_warmup_raw.jsonl`（run-warmup 后只 fetch 回 client）须 **push 到两台 server 同一相对路径**。

**须加入 §4 runbook 的显式步骤**：(1) 选定 canonical enriched pkl → 重新同步到两台 + 三处 sha256 断言相等；(2) Stage 4 run-warmup 后 `tether push` super raw 到两台同路径 + sha256 验；(3) `tether push remoteA:src remoteB:dst` 非法（push 首参须 local）→ 走 pull→local→push 或 local 起源分别 push 每台；(4) 同步后重启 server 确保重读。

### 4.1 init_map（Stage 0，所有阶段防泄漏硬门）
- **生成器**: `scripts/replay_libero_cache_trajectory.py map`
- **去重要点（已亲核）**: libero_10 的 6 个重复 episode H5 文件名带完整时间戳 → **不同 stem**，`build_mapping` 的 stem-dict（`_sources_from_h5_dir`）**不会**合并它们；纯 `--h5-dir` 会生成 50 条记录（6 ep 各 2 条）。**对策**：用 `--artifact <pkl>`（pkl 携带 canonical trajectory 集）为主源生成 → 取建库实际用的那一份；`--full-init-states-dir` 做 `orig_init_state_idx` 匹配。即便有重复，下游 `init_holdout.load_used_inits` 是 `{task_id: set(orig_init_idx)}`，set 自动去重，防泄漏无害。
- **远端执行**（owner 指示：init_map 走远端，H5 本地继续拉作备份）：在 ziyang10 上跑（repo+pkl+init-states 都在），只拉回小 JSON：
```bash
tether exec jupyter-ziyang10 -- bash -lc 'export HOME=/home/ziyang10; cd /home/ziyang10/openpi && \
  /home/ziyang10/.local/bin/uv run scripts/replay_libero_cache_trajectory.py map \
    --suite libero_10 \
    --artifact exp/common/data/cache_artifacts/libero_10/cp1_mean_pool.pkl \
    --h5-dir   exp/common/data/db/libero_cache/libero_10 \
    --init-states-dir      exp/common/data/db_init/libero_cache/libero_10 \
    --full-init-states-dir exp/common/data/db_init/libero/libero_10 \
    --num-trials-per-task 5 \
    --out exp/common/data/db/libero_cache/libero_10_init_map.json'
tether pull jupyter-ziyang10:/home/ziyang10/openpi/exp/common/data/db/libero_cache/libero_10_init_map.json \
            exp/common/data/db/libero_cache/libero_10_init_map.json
```
- **验收（自审 E3-GAP 强化）**: (a) 记录数、`orig_init_state_idx` 全解析成功（无 None）；(b) **核实 libero_10 每 task 的真实 init-state 总数**——`held_out_inits` 默认 `total_inits_per_task=50`，但 libero_10/LIBERO-LONG 未必是 50；若 ≠50 必须 `run_phase2 --total-inits <N>`（否则 `range(50)` 越界→env 报错或漏算）；(c) **打印每 task 的 held-out 直方图**，确认每 task held-out ≥ eval-trials(10)（`weight_search_strategy.py:44` 运行时会 fail-fast，但 Stage 0 就该挡）。⚠ 缺号 ep（28/33/36/37/43/48）+ 重复 ep → 建库未必恰好 5/task，"45 held-out" 是期望非已验证事实，须实测。

### 4.2 Phase-1 calibration（Stage 1）
- 离线、无 GPU。在每个 libero_10 库 pkl 上 LOEO 拟合 per-(模态,keybuilder) normalizer：
```bash
uv run exp/common/calibrate_score_normalizers.py \
  --artifact-dir exp/common/data/cache_artifacts/libero_10 \
  --output exp/weighted_sum/data/libero_10/phase1/calibration_normalizers.json \
  --max-queries 300
```
- 产出按 stem 分组的 `{sim_type, shortlist(top2), selected}`。可本地跑（pkl 已在本地）或远端跑。

### 4.3 Stage 3 warmup：cp1_score 分布（Phase A）
- base yaml（Stage1/2 winner）→ `emit_threshold_yamls --mode warmup`（judge `threshold=2.0` 强制全 MISS，机器人走**真实 policy 轨迹**，search 照常算 `cp1_score` 经 `--per-step-out` 落盘）。前置门：先看分布展度，过窄则 fail-fast。

### 4.4 Stage 4 super warmup raw（最大单项预收集）
- `kinematic/runner.py --mode run-warmup`：单 server 跑 super warmup yaml（`AlwaysWarmStart(0.7)` + `DumpingJudge(deferred)`），server 侧 `--warmup_dump_root` 落 dump，client `fetch_dump(ws_d1_kin_super_warmup)` 拉回 → 抽 finite → `super_warmup_raw.jsonl`。1 份 raw 喂全 237 cell（offline calibration source，绕开 WarmupPool LRU）。
- ⚠ ziyang10 启动需加 `--warmup_dump_root /tmp/<unique>_dumps`（NFS uid 坑用 /tmp）；须先 agentchat 通知 owner 释放 GPU。

### 4.5 Stage 4 ceiling/baseline
- **always-WARM ceiling**（本实验自测锚点）：3 yaml（start_t 0.3/0.5/0.7）× 100 ep，d1 base。
- **random/periodic baseline**：libero_spatial 版**不可迁移**（task-specific），`exp/random_periodic_gate/` 与 `exp/verdict_factor_judge/data/` **无 libero_10 版本**。Stage 4 范围内只跑 always-WARM ceiling；r/p baseline 是否重建 = §13-Q4 待定（libero_spatial 版 kinematic 选择跳过重建、Q3=b 只看 SR）。

---

## 5. Stage ① — 两层校准 + 权重搜索

**依赖**: §4.1 init_map + §4.2 calibration。
**Phase 2 config 规模（每 keybuilder 34 = iso 3 + grid2 21 + grid3 10；4 keybuilder = 136 base）**，复刻 libero_spatial：

```bash
# emit 136 base yaml（4 keybuilder 各跑一次）
for stem in cp1_mean_pool cp1_max_pool cp1_spatial_pool_16 cp1_spatial_pool_64; do
  uv run exp/weighted_sum/emit_yamls.py \
    --calibration exp/weighted_sum/data/libero_10/phase1/calibration_normalizers.json \
    --stem $stem \
    --preload-path exp/common/data/cache_artifacts/libero_10/${stem}.pkl \
    --output-dir exp/weighted_sum/config/phase2/libero_10 --mode both
done

# 评测（★ §2.3 单 server 钉死 — Stage1 选 winner/top10/d1-best，整批 base+refine 跑同一台，
#   禁止 capacity 跨 GPU 分派；client=timan107；每 yaml 100 ep = 10 task × eval-trials 10）
PYTHONPATH=. <uv> run exp/weighted_sum/run_phase2.py \
  --yaml-dir exp/weighted_sum/config/phase2/libero_10 \
  --init-map exp/common/data/db/libero_cache/libero_10_init_map.json \
  --journal  exp/weighted_sum/data/libero_10/phase2/journal.jsonl \
  --servers <xl-srv>:<port> \
  --task-ids 0-9 --task-suite libero_10 --eval-trials 10 \
  --workers <N> --server-workers <N>      # 单 endpoint；N = 该 server 实际 worker 数（如 xuanlel2 r3=48）
# ⚠ 另一台 server（ziyang10）本阶段绝不并入此 journal——只可跑独立阶段或与选取无关的吞吐复制

# 聚合本轮 → results.json（summarize.py 只产 per-yaml SR JSON，不产 csv）
uv run exp/weighted_sum/summarize.py --journal .../libero_10/phase2/journal.jsonl --out .../libero_10/phase2/results.json
```

- **3 轮 refine（串行循环，自审 E4-M2）**：`refine_round.pick_best_keybuilder` 吃上一轮 `results.json`，每轮 **emit → run（单 server）→ summarize → 喂下一轮**：base→…→R3。围绕 winner keybuilder 最优区加密。
- **★ winner 平局可执行闭环（G1 R1 Item2）**：base+R0 summarize 后，先算每 keybuilder 的 max-SR 与 1-SE 带（`SE = sqrt(p(1−p)/100)`，p=该 kb 的 max SR；n=100 时 ≈±4.5pp@p=0.5）。
  - **无平局**（top-1 kb 领先 top-2 kb > 1 SE）：照常 `refine_round.py`（默认选 max kb），单 stem 走 3 轮。
  - **平局**（top-2 kb 在 1 SE 内，复刻 libero_spatial spatial_16≈max_pool）：对每个并列 kb 各跑一条 refine 链 `refine_round.py --stem <kb> --output-dir round_{r}/libero_10/<kb>`。⚠ **这需要代码改动（G1 R2 Item1，纠正 R2 的"无新代码"误判）**：现 `refine_round.py:165` `pick_best_keybuilder` 返回的是**全局 best stem 的 `ranked`**，`--stem`(:166-167) 只换 `stem`/`entry`(calib/preload)，但 center 模式(:180) 与 norm2(:202) 仍用 `ranked`=全局 best 的 top weights → 强制第二个 kb 时会围绕**错** keybuilder 的最优区 refine、norm2 套错权重。**修**：新增 `rank_keybuilder(results, stem)` 返回该 stem 自己的 sorted configs，`main()` 改 `ranked = rank_keybuilder(results, args.stem) if args.stem else pick_best_keybuilder(results)[1]`（纳入 §9 gated code + 测试）。两链 results.json 全喂 `build_all_results.py`（多 `r1=… r1b=…` 合并到同一 csv）。
  - **下游纳入两 kb**：top10 从合并 csv 取（自然含两 kb）；Stage2 `select_base_configs` 本就 per-keybuilder；Stage4 base 取**两 kb 中 d1-SR 最高的单个**（或 owner 按 §3.1 (a)/(b)）。
- **`all_results.csv` 由 `build_all_results.py` 合并各轮 results.json 产出（不是 `summarize.py`）**（自审 E4-M2 关键修正）：
```bash
uv run exp/weighted_sum/build_all_results.py --out exp/weighted_sum/data/libero_10/phase2/all_results.csv \
  baseline=.../libero_10/phase2/results.json r1=.../libero_10/phase2/round_1_results.json \
  r2=.../libero_10/phase2/round_2_results.json r3=.../libero_10/phase2/round_3_results.json
```
此 csv 是 Stage 2 `select_base_configs` 的输入；缺它 Stage 2 无法选 base 集。
- **top10 生成（可复现命令，G1 R1 Item4；恰好 10，G1 R2 Item2 选 Option A）**：libero_spatial 的 `config/top10/` 是人工挑的、无脚本 → **新增小脚本 `exp/weighted_sum/emit_top10.py`**（纳入 §9 gated code + 测试），把"人工挑"收敛为确定性命令：
```bash
uv run exp/weighted_sum/emit_top10.py \
  --results-csv exp/weighted_sum/data/libero_10/phase2/all_results.csv \
  --src-config-dirs exp/weighted_sum/config/{phase2,round_1,round_2,round_3}/libero_10 \
  --out-dir exp/weighted_sum/config/top10/libero_10
```
  规则写死：按 yaml_id 聚合 mean SR 降序，**恰取前 10**；**rank-10/11 在 1 SE 内时用确定性 tie-break（SR 降序、再 yaml_id 升序）取够 10 个，不全收**（保 `len(top10)==10`，使 Stage2 base 计数稳定；SE 内的 #10/#11 对"强 base 集"目的等价，确定性即可复现）。`top10_manifest.json` 记录 10 项 yaml_id+SR+n **及被 tie-break 挤掉的 #11**（透明）。**测试**：合成 csv → 断言恰 10 个、降序、tie-break 顺序、manifest 字段（§9）。
- **Episode 预算**: base 136×100 + refine（R1~106 + R2~90 + R3~86）×100 ≈ **41,800 ep**（**100 ep/config = 10 task × eval-trials 10，owner 确认**）。
- **产出**: `all_results.csv`、`config/top10/libero_10/`、libero_10 d1-best winner（spatial_16 上 SR 最高的 d1 配置 + 其权重 + 归一化参数）→ 喂 Stage 2/3/4。

---

## 6. Stage ② — Trajectory（search + weight research）

**依赖**: Stage 1 的 `all_results.csv` + `top10/libero_10/`。

### 6a. Trajectory search（N_base × depth{3,4,5,6} yaml；N_base 动态，libero_spatial 曾为 18→72）—— base 由 §3.1 准则自动导出
`emit_trajectory_yamls.py` 的 `select_base_configs()` 从 `--results-csv` 动态推导 base（per-kb top1+top2+倒数第二 ∪ top10）。⚠ **`_EXPECT_BASE=18`/`_EXPECT_OVERLAP=4` 是 libero_spatial 常量，须先按 §1.3② 放宽**（否则 libero_10 overlap≠4 直接 AssertionError）；放宽后把 `--results-csv` / `--artifact-dir` / `--top10-dir` 指向 libero_10：
```bash
uv run exp/weighted_sum/emit_trajectory_yamls.py \
  --calibration exp/weighted_sum/data/libero_10/phase1/calibration_normalizers.json \
  --artifact-dir exp/common/data/cache_artifacts/libero_10 \
  --results-csv  exp/weighted_sum/data/libero_10/phase2/all_results.csv \
  --top10-dir    exp/weighted_sum/config/top10/libero_10 \
  --output-dir   exp/weighted_sum/config/trajectory/libero_10 --depths 3,4,5,6
# run_phase2 同 §5：★ §2.3 单 server 钉死 + 必须用与 Stage1 d1 基线同一台 GPU
#   （trajectory 测的是 depth-Δ vs d1 基线，跨 GPU 会污染 Δ）。--servers <同 Stage1 那台>:<port> 单 endpoint
```
**预算（动态，G1 R2 Item2）**: `N_base × 4 depth × 100 ep`（N_base 由 emit 时实算；libero_spatial 曾 18×4×100=7,200，libero_10 待定，~[5.6k, 8.8k]）。

### 6b. Trajectory weight research（78 配置 × depth{1,3,4,5} = 312 yaml）—— ✅ owner 确认纳入
`emit_trajectory_weight_sweep.py`（grid3 step 0.0625 ~78 weight，含 d1 重测），仅 spatial_16，分离「权重对 d1 过拟合」vs「trajectory 本质拖累」。
**预算**: 312×100 = **31,200 ep**（大头，可选）。

---

## 7. Stage ③ — Threshold-pareto（SR × inference_ratio）

**依赖**: Stage 1/2 winner base yaml（owner 定）。零代码（`episode_runner.py:49` cp1_score + `run_phase2 --per-step-out` 已就绪）。

```bash
# Phase A warmup（强制 MISS 收 cp1_score 分布）
uv run exp/weighted_sum/emit_threshold_yamls.py --base-yaml <winner>.yaml --mode warmup --output-dir exp/weighted_sum/config/threshold_pareto/libero_10
uv run exp/weighted_sum/run_phase2.py --yaml-dir .../threshold_pareto/libero_10 --init-map .../libero_10_init_map.json \
  --per-step-out .../warmup_per_step.jsonl --task-suite libero_10 --eval-trials <W> ...
# 前置门：分布展度不足 → fail-fast（换信号/换归一化）
# Phase B solve（分位 + zscore 两条对照；退化 cell T_fh−T_ws<eps 跳过）
# Phase C eval grid：fh_ratios×ws_ratios 受 fh+ws≤0.9 约束 ≈ 49 cell（emit_threshold_yamls 当前默认 11×5；注：libero_spatial plan 写的"16-cell"是旧值）× 100 ep
# + anchor（threshold=2.0 全 MISS，inf_ratio=1 锚点）
# Phase D：summarize_inf_ratio（FH 0 / WS@0.5 0.75 / MISS 1）→ Pareto
```
**预算**: warmup ~200 ep + eval 49×100 + anchor 100 ≈ **5,200 ep**。

---

## 8. Stage ④ — Kinematic phase5 复刻（唯一含代码改动）

**依赖**: Stage 1 libero_10 spatial_16 **d1 winner**（base）+ §4.4 super warmup raw + §8.4 pkl factors 验证。

### 8.1 流程（runner.py 8 个 mode）
`emit-warmup → run-warmup → verify-raw（7 检 hard-gate）→ emit-eval-yamls（237 cell，per-cell threshold 解算）→ run-eval（★ §2.3 **单 server** 钉死，`run_phase2 --strategy kinematic`，单 endpoint）→ aggregate-summary → analyze（5-group 5pp 决策 + Pareto overlay）`。237 cell = G1-G4 各 48（192）+ G5 45（`fh+ws≤0.9` 三角约束删退化）。
> ⚠ 为何单 server（G1 R1 Item1）：237 cell 的 5pp 决策门 + Pareto dominance 是**阶段内比较**；offline-calib 只把检索分漂移吸进 percentile rank，**不吸收 SR 漂移**，跨 GPU ~7pp 会直接翻转 cell 间排序。整批 237 cell + always-WARM 跑同一台。

### 8.2 代码改动（★ 唯一 gated code，纯 exp/、向后兼容、零改 src/）
把 **已存在**的 `--preload-pkl-override`（`runner.py:730`，当前只流向 run-eval 的 libero client args，未进 emit builder）透传进 3 个 yaml builder。镜像 `v2_spec.py:409-412/451-464` 已验证的模式：

**两类改动叠加**（自审 E4-VR 修正：原"仅 override pkl"不足以实现 Option B，因 override 不换权重/μσ）：

**(I) `preload_pkl_override` 透传**（换 pkl 路径）：
| 文件 | 锚点 | 改动 |
|---|---|---|
| `kinematic/spec.py` | `build_eval_yaml_for_cell` (:216 / :293) | 加 keyword-only `preload_pkl_override=None`；`preload_path = override or cfg["preload_pkl"]`；:293 改用 `preload_path` |
| `kinematic/super_warmup.py` | `build_super_warmup_yaml` (:126 / :173) | 同上 |
| `kinematic/runner.py` | `_build_always_warm_yaml` (:283 / :308) | 同上（加形参）|
| `kinematic/runner.py` | :96 / :244 / :330 调用点 | 传 `preload_pkl_override=args.preload_pkl_override or None` |

**(II) `cfg_id` 透传**（换权重/归一化/keys = Option B 真正需要的）：
| 文件 | 锚点 | 改动 |
|---|---|---|
| `kinematic/super_warmup.py` | `build_super_warmup_yaml` (:137 硬编码 `CFG_SPECS[CFG_ID_DEFAULT]`) | 加 `cfg_id=CFG_ID_DEFAULT` 形参 |
| `kinematic/runner.py` | `_build_always_warm_yaml` (:287 硬编码 `CFG_SPECS["spatial16_ws_d1_best"]`) | 加 `cfg_id` 形参 |
| `kinematic/spec.py` | `build_eval_yaml_for_cell` (:222 已有 `cfg_id` 形参 ✅) | 无需改 |
| `kinematic/runner.py` | argparse + 调用点 | 加 CLI `--cfg-id`（default `spatial16_ws_d1_best`），各调用点传 `cfg_id=args.cfg_id` |

> **绝不改 `CFG_ID_DEFAULT`**（自审 E4-VR + WA §3.1）——那会静默把 libero_spatial Stage4 路径重定向、破坏其复现。Option B 走"新 CFG_SPECS key + `--cfg-id` 指向它"。

净增约 **~60-80 行**（非原估 30-40）。**测试见 §9**（须复刻 precedent `tests/exp/test_phase5_libero10_path_override.py` 的 ~17 测试 4 桶结构，非单测）。

### 8.3 ★ d1-best base 来源 —— ✅ owner 确认 Option B
libero_spatial 版 kinematic 用 `CFG_SPECS["spatial16_ws_d1_best"]`（权重 v0@6 v1@50 rs@43 + libero_spatial 拟合的 zscore μ/σ + libero_spatial pkl）。libero_10 复刻有两条路：
- **Option A（最小改）**：复用 libero_spatial 的权重+归一化，仅 `--preload-pkl-override` 换 libero_10 pkl。**缺陷**：权重/归一化是 libero_spatial 拟合的，套到 libero_10 数据是**混淆变量**，不是"在 libero_10 重做实验"。
- **Option B（忠实复刻，推荐）**：在 `v2_spec.py` 新增 `CFG_SPECS["spatial16_ws_d1_best_libero10"]`，其 `keys.weight` = Stage1 libero_10 spatial_16 d1 winner 权重、`score_normalization` = libero_10 calibration 的 spatial_16 selected μ/σ、`preload_pkl` = libero_10 pkl；kinematic 用 `--cfg-id`/改默认指向它。**Stage 4 因此依赖 Stage 1 winner**。

> **owner 确认走 Option B**（检索层忠实"在 libero_10 重做"；kinematic factor 仪器仍移植自 phase5，见 §1.1 caveat）。
> ⚠ **gate-vs-runtime 拆分（自审 E4）**：CFG_SPECS 新条目的**结构**（key 存在、`weighted_score_sum_knn`、无 `trajectory_depth`、字段齐全）在 §4 Code/G1 时落地，但**权重 + μ/σ 真值**要等 Stage 1 winner 出来才中期填入——故条目**先以占位值入库过 G1/G2，真值 mid-run 填**。
> **〔2026-05-31 OWNER OVERRIDE — 依 WA line 7「Project Owner: Ziyang Lin. Holds absolute authority over this Working Agreement and all project matters. May override any process at will.」〕** 原表述"真值填入是再一次代码改动、**需重过独立 G2**"的强制要求，由 owner 行使 WA line 7 的保留 override 权**予以免除**（删除）。**替代核验机制 =「有条件的预决策」（owner 2026-05-31 定）**：
> 1. **决策条件化于 Stage 3 客观结果**：Stage 3 跑完 d1/d3/d4/d5 四 base 的 threshold-pareto → 各算 per-cell (inference_ratio, SR)（`summarize_inf_ratio.py` FH→0/WS@t→1−0.5(1−t)/MISS→1 + `summarize.py` SR）→ 画 4 条 Pareto 前沿。
> 2. **winner 选取规则**：在 inference_ratio(x) 整轴扫描，取「SR(y) 高于其他三条前沿的 **x-区间最长**」的那条 = winner base（owner 预判大概率 d1——SR 最高 0.520 + 检索分窄=早停多=低 x 端强）。
> 3. **Stage 4 base 导出（owner 2026-05-31 纠正：用赢家的 depth，改代码实现）**：取 winner base 的**完整检索配置** = `keys.weight`(v0/v1/robot_state) + **winner 的 `trajectory_depth`** + libero_10 calibration 的 spatial_16 selected μ/σ → 填 `CFG_SPECS["spatial16_ws_d1_best_libero10"]`。理由：**depth 与 kinematic verdict 正交**（depth 是检索层、verdict 是判定层，candidate 的运动学 factor 不随检索看几步历史而变），故 Stage 4 继承 winner 的 depth 不冲突、且忠实复刻 winner。**当前 CFG_SPECS 的 `search_strategy` 是单步 `weighted_score_sum_knn`（无 `trajectory_depth` 字段）→ 若 winner 非 d1，需改代码让该结构支持 winner 的 trajectory_depth**（owner 原话："如果 CFG 那个参数只支持 depth=1 那就是他的问题，我们修改代码实现"）。⚠ **此代码改动（加 trajectory_depth 支持）大于"填 9 个值"——其核验/把关方式待 owner 确认，agent 不自行决定。**
> 4. **结构性核验（替代 G2 的防错保证）**：这 9 个根值**非手填**，而是程序化从 `winner_yaml.keys.weight` + `calibration_normalizers.json` 读出后 `assert CFG_SPECS.weight == winner_yaml.weight` 且 `μ/σ == calibration` → 把 G2 本要防的「手填错值/取错来源/对应错」静默错误在导出+断言流程里结构性消除。执行时落一个小校验脚本打印这 9 个值的来源与等式，附在 Stage 4 启动记录里供事后追溯。
> §8.2(I) override 透传仍做（灵活性 + 测试覆盖；与 (II) cfg_id 正交）。

### 8.4 pkl factors 验证（不是假设）
libero_10/cp1_spatial_pool_16.pkl 已为 verdict phase5 enrich 64 keys。**须验证** `kinematic.super_warmup.super_warmup_declared_keys()`（237 cell declared_keys 并集）⊆ 这 64 keys。验证脚本（一次性）：算并集 → 与 pkl `payload.factors` 键集比对。若有缺 → 从 `.pre_phase5.bak.pkl` 用 `exp/common/factor_postprocess.py enrich-existing-pkl` 按 kinematic 的 factor 列表重 enrich。

### 8.6 ★ 路径/suite override 全集（自审 E4-VR — 防覆盖 libero_spatial Stage4 产物）
runner 默认全部烘焙 `kinematic_phase5`（data/config/`--super-raw-relpath`，runner.py:78-82/736）+ `--task-suite libero_spatial`（:721）+ always-warm echo 里硬编码 `libero_spatial_init_map.json`（:359）。libero_10 跑必须**全套 override 到独立目录**，否则覆盖/误读既有 libero_spatial Stage4 raw/journal：
- `--task-suite libero_10`、`--init-map .../libero_10_init_map.json`、`--task-ids 0-9`
- `--data-dir .../data/libero_10/kinematic_phase5/`、`--eval-dir .../config/kinematic_phase5/libero_10/eval`、`--thresholds-dir`/`--always-warm-dir`/`--journal`/`--per-step-dir`/`--summary` 按 data/config 各自指向 `data/libero_10/kinematic_phase5/` 或 `config/kinematic_phase5/libero_10/`
- `--super-raw-relpath` 用 libero_10 专属相对路径（**server 端两 suite 的 super raw 不可同路径相撞**）
- `--cfg-id spatial16_ws_d1_best_libero10`、`--preload-pkl-override .../libero_10/cp1_spatial_pool_16.pkl`

### 8.5 预算
super warmup ~150 ep + 237×100 = 23,700 ep + always-WARM 3×100 = **~24,000 ep**。

---

## 9. 代码改动汇总 + 测试策略

- **gated 代码（共 5 处，全 `exp/`，零改 `src/`，向后兼容）**：
  1. §8.2(I)+(II) kinematic 的 `preload_pkl_override` + `cfg_id` 透传（spec/super_warmup/runner）+ `--cfg-id` CLI。
  2. §8.3 `v2_spec.CFG_SPECS["spatial16_ws_d1_best_libero10"]` 新条目（占位值入库，真值 mid-run）。
  3. **②的 `emit_trajectory_yamls.py:63-64/154-155` 断言放宽**（libero_spatial 常量 18/4 → 放宽为下界/按 `N_base` 实算）——这使 ② 不再"零代码"。
  4. **新 `exp/weighted_sum/emit_top10.py`**（G1 R1 Item4）：吃 `all_results.csv` → SR 降序**恰前 10**（rank-10/11 1-SE 内确定性 tie-break，不全收）→ 复制对应 yaml 到 `config/top10/libero_10/` + 产 `top10_manifest.json`。
  5. **`refine_round.py` 加 `rank_keybuilder(results, stem)`**（G1 R2 Item1）：让 forced `--stem` 取该 stem 自己的 ranked（修 center/norm2 用错 keybuilder top weights 的 bug）。`main()`：`ranked = rank_keybuilder(results, args.stem) if args.stem else pick_best_keybuilder(results)[1]`。仅在 winner 平局双链 refine 时触发，但代码改动恒在。
- **测试（自审 E4-WEAK：须复刻 precedent `tests/exp/test_phase5_libero10_path_override.py` 的 ~17 测试 4 桶结构，非"单测"）**：
  - (a) override-used + **default-unchanged**（向后兼容）× `build_eval_yaml_for_cell` / `build_super_warmup_yaml` / `_build_always_warm_yaml` 三 builder；
  - (b) `cfg_id` 透传 × 三 builder + default 仍 `CFG_ID_DEFAULT`；
  - (c) CLI parse（`--preload-pkl-override`/`--cfg-id` set + default-empty）；
  - (d) runner mode 级 e2e（emit-eval-yamls / emit-warmup / run-always-warm 产出的 yaml 都带对的 pkl + cfg）——monkeypatch write_yaml 断言；
  - (e) 新 CFG_SPECS 条目**结构** test（非 manual：key 在、`weighted_score_sum_knn`、无 `trajectory_depth`、字段齐）+ `@pytest.mark.manual` 的 `load_cache_config` validator（**注**：`validate_cache_config` 仅在 `load_cache_config` 内跑、要加载 ~1GB pkl=manual，**CI 不跑** → §9(b) emit 自检不计入 green-CI claim）；
  - (f) 断言 libero_spatial 原条目/路径未动（复用 `test_phase5_orig_cfg_untouched`）。
  - (g) **`emit_top10.py`**（G1 R1 Item4 / R2 Item2-A）：合成 all_results.csv → 断言**恰 10 个**、SR 降序、rank-10/11 1-SE 内的确定性 tie-break 顺序（SR desc → yaml_id asc）、`top10_manifest.json` 字段（10 项 + 被挤掉的 #11）+ 复制的 yaml 数==10。
  - (h) **`rank_keybuilder`**（G1 R2 Item1）：合成多 keybuilder results → 断言 forced stem 返回该 stem 自己的 ranked（非全局 best）；center/norm2 用的 top weights 来自 forced stem；不传 stem 时与 `pick_best_keybuilder` 一致（向后兼容）。
- **§6 Verify**: `uv run pytest`（非 manual）全绿；manual 的 `load_cache_config`/validator 由 executor 本地跑（触及 GPU/大 pkl 路径时）。
- 其余阶段（①③）无代码改动；② 有 (3)、refine 有 (5)。本 **plan 整体** L2 → 需 G1（plan review）。

---

## 10. Episode 预算汇总（100 ep/config 口径）

| 阶段 | episode | 备注 |
|---|---|---|
| Stage 0 (init_map+calib) | 0 | 离线/远端 |
| Stage 1 (base+3 refine) | ~41,800 | 418 config × 100 |
| Stage 2a trajectory | N_base×4×100（动态）| libero_spatial 曾 72×100=7,200；libero_10 按 N_base 实算 |
| Stage 2b weight research | 31,200 | 312×100（可选, §13-Q3）|
| Stage 3 threshold | ~5,200 | warmup+49 cell+anchor |
| Stage 4 kinematic | ~24,000 | super warmup + 237×100 + always-WARM |
| **合计** | **≈109k（含 2b）/ ≈78k（不含 2b）±动态差额** | Stage2a 随 N_base 浮动（libero_spatial 18→7.2k 为参照）；见下 wall-clock |

**Wall-clock（自审 E3 + G1 R1 Item1 修正——原"数十小时"过于乐观）**：双 jupyter 实测 ~1.9 ep/s（verdict_phase5_libero10；非 6-server 的 6-7h，不可套用）。**但 §2.3 要求所有决策阶段单 server 钉死** → 这些阶段只有 ~单台 H200 吞吐（约 ~1 ep/s，取该机 replica 数而定），并行吞吐不可用：
- Stage1（~41.8k ep，单 server 决策）、Stage2a（N_base×4×100，参照 18→7.2k，单 server）、Stage3（~5.2k，单 server）、Stage4（~24k，单 server）几乎全是决策阶段 → 按 ~1 ep/s 估 ≈ **~78k ep / 1 ep/s ≈ 多日纯 eval**；含 Stage2b（31.2k，spatial_16-only，可在另一台并行不占决策机）。
- 唯一能吃双机的是**独立阶段之间并行**（如 Stage2/3/4 用不同机各自单 server 同时跑）+ Stage2b。
**叠加**：Stage1→2/3/4 串行 barrier（§3 DAG）、各阶段 server 重启/§4.0 数据 re-sync、8.6 GB H5 / 1 GB pkl 传输、warmup 收集、threshold 解算、潜在 failover → **端到端为多日级**。owner 据此规划无人值守窗口；不要按"数十小时"安排。

---

## 11. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | init_map 从重复 H5 生成出错/重复 | 从 `--artifact <pkl>` canonical 集生成；leak guard set 去重无害；验收查 None/缺号 |
| R2 | libero_10 calibration 选出的 normalizer/分布与 spatial 差异大致 Phase2 行为变 | 正常（实验本意就是重拟合）；Phase1 诊断图先看分离度 |
| R3 | Stage 3 总分展度不足，阈值分不开三档 | Phase A 前置门 fail-fast + zscore 变体对照 |
| R4 | Stage 4 pkl 64 keys 不覆盖 kinematic declared_keys | §8.4 先验证，缺则从 bak 重 enrich |
| R5 | 双 H200 跨物理 GPU 混跑 → 结果不可与单 GPU 对齐 | 标注 "different physical GPU"；owner 已接受（§13-Q5）|
| R6 | 共享 jupyter cgroup 10C/32G → server OOM / `--replica-spawn-batch` | 按 skill §3.7.1 限 spawn batch；先 N=1 试显存 |
| R7 | server-side pkl/yaml/raw 与本地不同步 | 主跑前 sha256 byte-identical 验（skill §3.2）；server 重启确保读到新 pkl |
| R8 | a100 离线，jupyter 组崩溃无 failover 目标 | precedent: phase5_libero10 曾转 a100。**failover 前提 = §4.0 双 server 已同 sync**（存活机要有相同 pkl/super raw 才能接盘，R8↔R7/§4.0 联动，自审 E3）；a100(40G) 接盘会重引入跨 GPU → **该阶段须重跑而非续接**（mid-stage 换 GPU 破坏阶段内可比）|
| R9 | 跨 GPU 漂移翻转 winner 选取（~7pp>5pp）| §2.3 阶段内单 server 钉死 |
| R10 | server 缺/错 libero_10 pkl 致 `load_cache_config` 崩 | §4.0 canonical + 双推 + sha256 三处相等，先于 Stage1 |

### 11.1 各 stage abort / go-no-go 门（自审 E4-MISSING）
- **Stage 0**: init_map 任一 task held-out < eval-trials(10) → abort，回补建库/调 eval-trials；pkl 三处 sha256 不等 → abort，先 §4.0 同步。
- **Stage 1**: winner SR 低于某地板（owner 定，如 < always-WARM ceiling 无意义）→ 与 owner 确认是否值得启 2/3/4。
- **Stage 3**: Phase-A 分布展度不足 → fail-fast（不浪费 eval）。
- **Stage 4**: `verify-raw` 7 检任一 FAIL（尤其 finite-per-key 地板）→ **不进 emit-eval-yamls**，加大 `trials_per_task` 重收 super warmup（libero_10 长程 ep 步数与 libero_spatial 不同，phase5 的 21.7 verdict/ep 不可直接套，须按 libero_10 实测）。

---

## 12. Commit 边界（artifact_layout.md §3）

- **gitignore 边界（自审 E1 校正措辞）**: `exp/**/data/**` 在 `.gitignore:6`（仓库级）；config 规则是 `exp/weighted_sum/config/**`（`.gitignore:11`，**非**仓库级 `exp/**/config/**`）。两者都使本 plan 的 emit yaml / journal / per_step / pkl / raw **不入库**（本地保留 + tar 离线归档）。
- **入库**: 代码改动（§8.2/§8.3 + ②断言放宽 + 新单测）、`logs/` 本 plan、各阶段 `exp/weighted_sum/analysis/**` 下 results.md + 图副本（`analysis/` 未被 ignore）。
- pkl/H5 历来不入库（libero_10 pkl 本地 + 两 server 各一份，sha256 一致）。
- **docs 决策（自审 E4-MISSING）**: `docs/experiments/weighted_sum.md` 现为 libero_spatial-only runbook。**本 plan 取向：libero_10 的运行记录以本 plan log 为准，不改该 runbook**；若后续 owner 要 runbook 覆盖 libero_10，则那是单独一次带 `docs/README.md` 同步的改动。
- **Index Sync**: `logs/README.md` 已加本 plan 条目（与 plan 同 commit 落地，WA §4）✅。
- commit message 英文、无 Co-Authored-By、author=LinZiyang666；owner 偏好单 commit + multi-section body。

---

## 13. 已确认决策（owner 2026-05-29）

| # | 问题 | 决策 |
|---|---|---|
| Q1 | 每 config ep 数 | ✅ **100 ep（10 task × eval-trials 10）**，镜像 libero_spatial 口径 |
| Q2 | Stage 4 kinematic base | ✅ **Option B**：libero_10 自己的 Stage1 spatial_16 d1 winner + libero_10 calibration → 新 `CFG_SPECS["spatial16_ws_d1_best_libero10"]` |
| Q3 | Stage 2b weight research（+31,200 ep）| ✅ **纳入**（完整复刻）|
| Q4 | Stage 4 random/periodic baseline | 默认**跳过重建**（同 libero_spatial Q3=b，只看 SR）；中期如需再议 |
| Q5 | 双 jupyter 混跑可比性 | ✅ owner 选双 jupyter，接受 "different physical GPU" 标注，不强求与单 GPU bit/SR 对齐 |
| Q6 | 执行节奏 | ✅ **plan 把筛选准则写死（§3.1），具体值各 stage 中期临时决策**；执行先 Stage 0+1，出 winner 再启 2/3/4 |

---

## 14. Stage 0 Execution Log（2026-05-29，真实实验开始）

> 实验正式开始。owner 设 /goal「严格按 plan + experiment skill 跑完整个实验，不做完不停」。owner 补充三条运行参数：① ziyang10 = 3 replica / 48 worker、xuanlel2 = 2 replica / 32 worker；② 多 server 独立阶段任务分配 **3:2**；③ **不用 CLIP**（2 个 clip pkl 不参与任何阶段）。

### 14.1 init_map（✅ 完成 + 全验证）— ★ 偏离 §4.1 的方法（已记录理由）
- **§4.1 的 `replay...map` 方式对本 libero_10 数据是坏的**：实测 `--num-trials-per-task 5` 的位置式 `episode//5` task 分配**不成立**——H5 按 episode 号 //5 分桶后 bucket 0/2/3 各含多个不同 prompt（采集非 task-major、交错），且本地与 ziyang10 的 openpi uv env **都没有 `libero` 模块** → map fallback 用裸 prompt 匹配 scene-前缀 init 文件名，全部 50 条 unresolved / `orig=None` → leak-guard 空集 = 泄漏。
- **改用的正确法**（在本地 `libero_sim` conda env，含 libero+torch+h5py）：① 用 LIBERO benchmark 取权威 `task_id↔name`（10 prompt 与 10 task language 一一对应）；② 对每 task，把 cache-子集 init（`db_init/libero_cache/<name>.init`，5 行）逐行在 full init（`db_init/libero/<name>.init`，50 行）中 `np.isclose` 唯一匹配 → `orig_init_state_idx`；③ emit 50 records（5 子集行 × 10 task）。这是 trajectory-无关的 used-set ground truth（docstring 定义："library 从每 task 一小 subset init 建"），绕开坏的位置式。
- **决策：本地生成（非 §4.1 远端）**。owner「走远端」前提（ziyang10 上有 repo+pkl+**init-states**）已破裂：ziyang10 **整个 `db_init` 缺失** + repo 落后 1 commit（8275c9f，且 replay 脚本两 SHA 间未改）；两边都无 libero → 产物字节等价 → 本地生成最简、无 GPU、可逆。
- **验证**：`init_holdout` 回读 → 全 10 task used=5 / held_out=**45** ≥ eval-trials 10 ✓；`orig_init_state_idx` 全 int 无 None；H5 交叉验证每 task 恰 5 轨迹、共 50、**无未映射 prompt**；**`total_inits_per_task=50`**（全 task 一致）→ run_phase2 **无需 `--total-inits` 覆盖**。
- 产物：`exp/common/data/db/libero_cache/libero_10_init_map.json`（50 records，gitignored）。

### 14.2 calibration（✅ 完成）
- 命令偏离：`--artifact-dir` 用只含 4 个 cp1 pkl 的符号链接临时目录（`/tmp/cal_libero10_cp1`），以**排除 CLIP**（脚本 `glob("*.pkl")` 否则会算 2 个 clip）。`--max-queries 300`。
- 产物 `exp/weighted_sum/data/libero_10/phase1/calibration_normalizers.json`：4 stem（mean/max/spatial16/spatial64），每个 3 字段 vision_0(cosine)/vision_1(cosine)/robot_state(l2)，**全选 zscore**，sat=0、J 为正（健康分离）。注：cp1 的 `vector_dims` 不含 vision_2（与 libero_spatial 设计一致，检索只用这 3 字段 + prompt_emb 被 `_EXCLUDED_FIELDS` 排除）。

### 14.3 §4.0 server 数据同步（部分完成）
- **三方 sha256**：ziyang10 的 cp1_mean/max/spatial64 与本地**逐字节相同**；**spatial_16 sha 不同**（local de51731d vs ziyang10 f13517ad，差 8B，别的实验独立 enrich）。
- **关键判定**：local 与 ziyang10 的 spatial_16 **query_keys content-hash 完全相同**（`ae88008207b6…`，2640 entries）→ 差异**只在 factors（Stage 4 用）**，Stage 1 检索（只用 query_keys）在 ziyang10 现有 pkl 上**逐字节等价** → **Stage 1 无需推 1.1GB、不覆盖他人文件**。spatial_16 factor canonical 同步**推迟到 Stage 4 前**（届时覆盖 ziyang10 需先备份 + 与 owner 确认）。
- **xuanlel2**：repo 旧（250292a）+ 4 cp1 pkl **全缺** → Stage 2/独立阶段前须 git 同步 + 推全 4 pkl（+ 验 sha）。不挡 Stage 1（单 server 钉死 ziyang10）。

### 14.4 Stage 1 base yaml（✅ emit）
- `exp/weighted_sum/config/phase2/libero_10/` = **136 yaml**（每 stem 34 = iso3+grid2×21+grid3×10）。preload-path baked = libero_10 cp1 pkl 相对路径。

### 14.5 ★ Stage 1 改为**双 server**（owner WA §7 决策，2026-05-29，证据支撑——修正 §3.4）
- **§3.4「决策阶段单 server 钉死」的前提被证伪**：RESULTS.md §7 的 ~7pp 漂移**专指 H200(Hopper) vs A100(Ampere) 跨架构** bf16 累加差异；**同 GPU run-to-run 噪声仅 0–0.6pp**（基本确定性）。
- 实测两台 server **完全同构**：`NVIDIA H200 NVL` / 驱动 `570.211.01` / compute_cap `9.0`（逐项相同）→ 不存在跨架构漂移 → 同一 yaml 的 SR 与落哪台 H200 无关 → **winner 选取不会被翻转**。owner 据此拍板（WA §7）：Stage 1 用**两台 H200 并行**，~2× 吞吐。
- ⚠ 但「双 server 有效」**额外要求两台服务端代码逐字节一致**（否则代码差异引入非-GPU 混淆）。

### 14.6 ★ 三台设备代码同步到 canonical `9519a79`（skill §3.1，2026-05-29）
- 发现三台都跑**旧代码**且彼此不一致：ziyang10 `8275c9f`（落后 7 commit，早于 weighted_sum 基线 + 全部 audit 修复）、xuanlel2 `250292a`、timan107 `099f491`。`src/openpi` 服务端差异大（cache/config、search_strategy、key_builder、serving/batching_coordinator+192、websocket_policy_server、gemma_pytorch、orchestrator…）→ 直接影响检索打分/推理。
- **已 `git stash -u`（保留可恢复）+ `git pull --ff-only origin Ziyang` 把三台全部同步到 `9519a79`**（owner 明确指示 git pull 同步）。stash 内容：ziyang10=untracked data/configs/tar、xuanle=`M orchestrator.py`、timan107=`M super_warmup.py`+untracked kinematic md（均在各机 `git stash list` 可恢复）。
- `transformers_replace` overlay 两区间**未变** → 旧 venv overlay 仍有效，无需重做。
- ignored 的 pkl 不受 stash 影响（保留）。**两台 server 现均 @9519a79、代码逐字节一致**。

### 14.7 Stage 1 eval 拓扑（双 server）
- **server**：ziyang10（3 replica，48 worker）+ xuanlel2（2 replica，32 worker），均 @9519a79，`OPENPI_SERVER_GPU_MEMORY_LOCK=0`，`--cache_config /tmp/stage1_placeholder.yaml` 占位、conductor 逐 yaml 热切。
- **数据同步**：4 cp1 pkl —— ziyang10 原有（mean/max/sp64 与本地逐字节同；spatial_16 query_keys 与本地同，factor 异不影响 Stage1）；xuanlel2 已推本地 4 pkl（sha 校验）。init_map+136 yaml 在 timan107:/tmp。
- **client**：timan107 @9519a79，`--servers ziyang:14000,xuanle:<port> --server-workers 48,32`（3:2 owner 指定，sum 80 worker；timan107 48 CPU 超额订阅但 worker 多为等推理 I/O-bound，监控 OOM/throttle），`--gpus 8 --conda-env libero_sim --eval-trials 10 --task-suite libero_10 --eval-concurrency 2`，tmux 内跑 + tee + journal 落 /tmp 后拉回 summarize。

---

### 14.8 ★ BUG 发现 + 修复：worker 未收到 task_suite（2026-05-29，run 期间 hotfix）
- **症状**：首次双 server eval，server log 全 `cp1 judge: MISS (top_score=None, winner=None)` → 检索返回 0 候选，退化成纯 live 推理，Stage 1 SR 无意义（journal 仍 success:true，但那是 libero_spatial 的 live 推理结果，对本实验全错）。
- **诊断**：server 端临时加 `DBGEMPTY` 日志实测 → `candidates=0`，`live_tk='pick up the black bowl between the plate and the ramekin and place it on the plate'`（= **libero_spatial task0**）≠ 库的 libero_10 裸 prompt（checkpoint CP1==CP1 ✓、qfields ✓）→ task_key 精确匹配失败（`in_memory_backend._filter_entries` line 348 `entry.payload.task_key != spec.filters.task_key`）。
- **根因**：`src/openpi/conductor/agent.py:_default_spawn` 构造 worker 命令时**从不转发 `--task-suite-name`**；`examples/libero/worker_entry.py:49` 默认 `libero_spatial`。run_phase2 的 `task_suite_name=args.task_suite` 只喂 strategy（生成 episode 的 task_id），没进 WorkerSpec → worker 一直跑 libero_spatial 任务。libero_spatial 实验时该默认恰好对 → **潜伏 infra bug，到 libero_10 才暴露**。
- **修复（3 行，client 侧）**：(a) `agent.py` `WorkerSpec` 加 `task_suite_name: str="libero_spatial"`；(b) `_default_spawn` base_cmd 加 `--task-suite-name spec.task_suite_name`；(c) `run_phase2.py` WorkerSpec 传 `task_suite_name=args.task_suite`。
- ⚠ **改了 `src/openpi/conductor/agent.py`（共享 conductor infra）+ `exp/weighted_sum/run_phase2.py`**——run 期间 hotfix，**需提交 + 理应补 G1/G2 review**（owner 待定是否补正式 gate）。
- **验证**：修复后微型 test（libero_10 task0/1ep）→ ziyang10 judge 全 `FULL_HIT`（top_score 0.94，winner=库轨迹 `episode_0001_*:57`），0 新 DBGEMPTY → 检索正常命中。
- **已应用位置**：本地 canonical（agent.py + run_phase2.py，**未提交**）+ timan107 repo（已 cp 进 `/scratch/zixuans8/openpi/`）。server 端（ziyang10/xuanle）无需此 fix（不跑 agent/run_phase2）。`--init-states-dir` 仍用 `""`（LIBERO 内建全集，沿用 libero_spatial 先例，orig_init_state_idx 0-49 一致）。
- **附带**：诊断期临时给 ziyang10 的 `in_memory_backend.py` 加过 DBGEMPTY 日志，已 `git checkout` revert（ziyang10 @9519a79 clean）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-29

- [Blocking] [Concern] `kinematic/runner.py --mode run-eval` 仍会默认输出双-server命令，违反 G1 §2.3 单 server 钉死规则 — reasoning: `_mode_run_eval()` 已打印"keep --servers to a SINGLE endpoint"（`exp/weighted_sum/kinematic/runner.py:370-373`），但实际命令仍直接使用 `args.servers`（`:382`），而 CLI 默认 `--servers` 仍是 `"weiland.top:14000,weiland.top:14001"`（`:770-771`）。默认运行会输出两个 endpoint，同时只输出 `--server-workers 48`（`:384`），要么被 `run_phase2.py` 的长度校验拒绝，要么操作者手动补成 `48,48` 后把 237 cell 分摊到两块 GPU，正好重犯 G1 R1 的 blocker。请把 run-eval 默认改成单 endpoint，或在 `_mode_run_eval` 中对 comma-separated `args.servers` fail-fast，并加测试断言输出命令只有一个 server 且 `--server-workers` 与之匹配。
- [Blocking] [Concern] top10 产物不是幂等的，且 Stage2 没有校验"恰好 10 个 yaml" — reasoning: G1 R3 选择 Option A 后，`top10/libero_10/` 必须恰好 10 个 yaml；但 `emit_top10.main()` 只 `mkdir` 然后 copy 新 top10（`exp/weighted_sum/emit_top10.py:134-137`），不会清理 out-dir 里旧的 `*.yaml`。如果重跑、换 results.csv、或目录里已有手工/旧产物，旧 yaml 会保留；随后 `emit_trajectory_yamls.select_base_configs()` 用 `top10_dir.glob("*.yaml")` 全量读取，main 里也只检查 `len(top10_ids) >= 1`（`exp/weighted_sum/emit_trajectory_yamls.py:147,155-166`），会静默把 11+ 或 <10 个 top10 纳入 Stage2 base，破坏 "top10 恰 10 + N_base 可解释" 的 handoff。请让 `emit_top10` 在成功解析并确认源 yaml 全存在后清理/重建 out-dir 中旧 `*.yaml`，并在 `emit_trajectory_yamls` assert `len(top10_ids) == 10`；补测试覆盖 stale output dir 和 top10 dir miscount。
- [Verified] `uv run pytest tests/exp/test_weighted_sum_libero10.py -q` → 22 passed；`uv run pytest tests/test_kinematic_super_warmup.py -q` → 18 passed。上述两个 blocker 是测试未覆盖的执行路径问题。

### G2 Round 2 — Executor — 2026-05-29

- **Accepted（Item 1, Blocking）** — `_mode_run_eval` 现对逗号分隔 `args.servers` **fail-fast**：`if "," in args.servers: raise SystemExit(...)`（`runner.py` run-eval 开头，引 §2.3 单 server 理由）。CLI 默认 `--servers` 仍是双 endpoint，故默认运行直接报错、强制操作者传单 endpoint，绝不会再输出双-server 命令；echo 的 `--server-workers 48` 与单 server 匹配。补测试 `test_run_eval_rejects_dual_server`（默认双 → SystemExit）+ `test_run_eval_single_server_ok`（单 endpoint → 输出只含一个 server + `--server-workers 48`，无第二 endpoint）。
- **Accepted（Item 2, Blocking）** — top10 幂等 + Stage2 恰-10 校验：(a) `emit_top10.main()` 在确认全部源 yaml 存在后、copy 前 **清空 out-dir 旧 `*.yaml`**（`for stale in out_dir.glob("*.yaml"): stale.unlink()`），保证重跑/换 csv/旧产物后目录恰为 10 个；(b) `emit_trajectory_yamls.select_base_configs` 现 **`if len(top10_ids) != 10: raise ValueError`**（Option A 不变量；main() 冗余的 `>=1` 断言移除）。补测试 `test_emit_top10_idempotent_clears_stale`（预置 stale yaml → 跑后恰 10、stale 清除）+ `test_select_base_configs_top10_miscount_raises`（top10=9 与 11 → ValueError）。
- **Verify**: `tests/exp/test_weighted_sum_libero10.py` 22→**27 passed**（+5）；`tests/exp/` 全量 611→**616 passed** 无回归；改动代码 0 汉字。

### G2 Round 2 — Reviewer — APPROVED — 2026-05-29

- [Approved] G2 R1 两个 blocker 均已关闭：`kinematic/runner.py --mode run-eval` 现在对多 endpoint `--servers` fail-fast，默认双 endpoint 不会再输出可误跑命令；`emit_top10.py` 现在清理 stale `*.yaml` 后再复制 top10，`emit_trajectory_yamls.select_base_configs` 强制 top10 目录恰好 10 个 yaml。
- [Verified] `uv run pytest tests/exp/test_weighted_sum_libero10.py tests/test_kinematic_super_warmup.py -q` → 45 passed；`uv run pytest tests/exp -q` → 616 passed。未发现新的 blocking / non-blocking findings。
