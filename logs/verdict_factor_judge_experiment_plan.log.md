# Verdict Factor Judge — Experiment Plan

> **Status**: `Plan`
> **Level**: **L2**（新增 1 个实验子目录 + N 个 YAML 配置；新增 `DumpingJudge` 包装类 + `JudgeConfig.dump` schema；扩 `episode_start.extra_metadata` 通道 5 处 wrapper 签名；可能新增 1–2 个 Composer / Normalizer 类；扩展 `analyze_cache_results.py` / 新写 calibration analysis 脚本）
> **Authority**: Execution
> **Owner**: Ziyang Lin
> **Drafted**: 2026-04-27
> **Depends on**: `logs/verdict_factor_judge.log.md`（B0）+ `logs/verdict_factor_judge_b1_b2.log.md`（B1+B2 Implemented）+ `logs/libero_spatial_factor_artifact_rebuild.log.md`（artifact 已 enrich 完毕）

---

## §inf_ratio — Inference-ratio 计算公式（项目唯一约定）

> **此公式所有 Pareto 分析、画图脚本、决策门必须严格遵守。改这个段落必须同步更新 `exp/verdict_factor_judge/analysis/plot_pareto.py` 头部 docstring。**

每 verdict 的 inference 成本贡献：

| verdict 类型 | inference_ratio 单次贡献 |
|---|---|
| `MISS` | `1.0` （full denoising path, t=1→0）|
| `FULL_HIT` | `0.0` （cache verbatim, no denoising）|
| `WARM_START @ start_t = t` | **`1 - 0.5 × (1 - t)`** |

flow matching 约定（`src/openpi/models/pi0.py` line 273 注释明确）：**t=1 是 noise，t=0 是 target**。`dt = -1/num_steps` 从 1 降到 0。`start_t = t` 表示从 noise level t 开始降，跑 (1-t) 比例的 denoising 步数。warm-start 的 0.5 系数 = action expert 大约是 full inference 的一半（KV cache 复用 + 不重跑 vision/language expert），所以 warm 单次成本 = 1 − 0.5×(1−t)。

具体值：

| start_t | warm 单次 inf 贡献 |
|---|---|
| 0.7 | **0.85** |
| 0.5 | **0.75** |
| 0.3 | **0.65** |

跨 yaml 聚合：

```
inf_ratio_yaml = (n_full_hit*0.0 + n_warm*warm_cost(t) + n_miss*1.0) / n_total
```

> **常踩坑（不要写）**：
> - ❌ `inf = 1 - start_t`
> - ❌ `inf = start_t`
> - ✅ `inf = 1 - 0.5 * (1 - start_t)`

baseline always-WARM @ start_t 直接 = 同 inf 行的 SR（每 verdict 都是 warm@t）：

| cfg | warm@0.3 (inf=0.65) | warm@0.5 (inf=0.75) | warm@0.7 (inf=0.85) |
|---|---|---|---|
| clip | 0.940 | 0.960 | 0.980 |
| max_pool | 0.946 | 0.966 | 0.964 |
| spatial16 | 0.942 | 0.952 | 0.976 |

Phase 2 Layer 1 yaml 的 warm 全部 fire 在 `start_t=0.7`（all_nan_fallback 默认值），所以每 warm verdict 贡献 **0.85** 到 inf_ratio。

---

## 0. TL;DR

CompositeJudge 框架（5 因子 + 3 composer + 1 normalizer）已经上线，但**没有任何"实战配置"被验证过**。本 plan 把所有候选 judge 设计方向收敛成 **7 个实验阶段**（Phase 0–5 + Phase 7；Phase 6 cross-keybuilder 取消，已内嵌每 phase × 3 cfg），按"先建对照、再单因子、再组合、再 composer 类型、再 calibration、再跨任务"的顺序推进；早期阶段决定后续阶段是否值得做。

实验底盘统一：
- 数据集：`libero_spatial`（10 task × 50 init = 500 ep 总集；本 plan 每 run 取 **100 ep** 子集 — 见 §2.4）；artifact 已带 168 keys × entry 的 F1b factors（commit `66a341f`）。
- **KeyBuilder + Search Strategy 锁定为 warm_start 实验同款 3 套**（详见 §3.7）：
  - **CFG-CLIP** — `clip_w7_d4`：keybuilder=`clip`, weights `(v0=0.1, v1=0.1, state=0.8)`, trajectory_depth=4
  - **CFG-MAX** — `max_pool_w3_d5`：keybuilder=`cp1_max_pool`, weights `(v0=0.5, state=0.5)`, trajectory_depth=5
  - **CFG-SP16** — `spatial16_w8_d4`：keybuilder=`cp1_spatial_pool_16`, weights `(v0=0.5, v1=0.25, state=0.25)`, trajectory_depth=4
- 3 套共用：`weighted_rrf_knn` (rrf_k=60, top_k=1, step_filter=all)；`field_similarity` 4 字段（vision/prompt cosine，robot_state l2→exp tau=0.334717）；`gate=always_search`；`backend=in_memory`；`write_policy=never`。
- 选这 3 套是因为它们的 baseline 已经齐全：pure inference / AlwaysHit / AlwaysWarmStart × 3 start_t 都已有 500 ep 数据可直接 join。
- Backend：`in_memory`（composite 唯一支持）。
- Runner：`exp.common.run_cache_experiments`（已有 cache eval pipeline）。

明确**不做**：
- 用 `ThresholdJudge` 当 baseline（项目内已确定无信号，见 memory）。
- 改 inference pipeline / Orchestrator / Storage。
- 论文级 ablation（本 plan 服务于"验出可上线 judge"，不为论文起草服务）。
- **不重跑已有 baseline**（pure inference / AlwaysHit / AlwaysWarmStart × spatial16_w8_d4 / max_pool_w3_d5 / clip_w7_d4，500 ep 都现成；详见 §3.8 复用清单）；唯一新增的"baseline 类"工作是 3 cfg × `AlwaysHit` 各 1 run，yaml 含 `judge.dump: {path, config_id, factors}` 字段触发 server-side `DumpingJudge` wrapper 写 calibration JSONL（详见 §4 Phase 0 / §6.2）。

---

## 1. 背景与 Intuition

### 1.1 Judge 在 cache 系统的定位

```
KeyBuilder → Gate (要不要查) → SearchStrategy (查+融合排序) → JUDGE (信不信第1名) → WritePolicy
```

Judge 的边界条件：
- 输入：search 后排好序的 `results: list[SearchResultLite]` + `view`（按需 fetch payload）+ `history`（已执行 K 步 action / state）
- 输出：`JudgeResult(MISS / WARM_START / FULL_HIT, winner_id, start_t)`
- 三档语义对应 trade-off：FULL_HIT 省最多但坏单价最高；WARM_START 是不确定性的"减压阀"；MISS 是安全网。
- Purity：只读 storage，不能写。

### 1.2 核心 Intuition

Cache 命中要可靠，必须**两件事同时成立**：

1. **候选自身可靠** — 来自一段稳定运行的源轨迹（F1b-A / F1b-T 测）。
2. **候选与当下可衔接** — 接上去 robot 的速度 / 加速度不会跳变（F1a-A / F1a-T 测）。

外加一条独立的 "retrieval 可信度" 信号：

3. **Top-K 候选互相之间不分歧** — 分歧大说明 retrieval 在猜（F2 测）。

3 类信号的正交性正是本框架的设计核心；它们不可互相替代，但单独使用也都未必足够。具体哪类信号 / 哪种组合在 LIBERO 上"真正能区分成败"是经验问题，本 plan 用实验回答。

### 1.3 上线后该解决的问题

- Q1：多大 cache hit rate 时 success_rate 不会跌穿 pure inference baseline？
- Q2：FULL_HIT 与 WARM_START 之间的"信任边界"在哪？哪些因子值组合应当被推进到 WARM_START 而不是 FULL_HIT？
- Q3：单一 cache pkl 上调好的 judge 配置，能跨 KeyBuilder（mean_pool / clip / llm_layer）移植吗？
- Q4：跨任务 (`libero_spatial` → `libero_10`) 不重调能保多少 success_rate？

---

## 2. 目标与衡量

### 2.1 主指标

| 指标 | 定义 | 角色 |
|------|------|------|
| `success_rate` | LIBERO `_check_success()` 在 episode 终止前达成的 episode 比例 | **硬约束**：不能比 pure inference floor 跌超过 `δ_success`（待 Phase 0 后定，建议初值 −2pp） |
| `inference_time_saved_ratio` | per-cycle 加权和：`MISS=0`, `WARM_START=0.5×(1−start_t)`, `FULL_HIT=1.0`，再除以 `total_cycles` | **优化目标**：在 success 约束下最大化 |

**warm_start 收益的 0.5 系数**：WARM_START 只跳过 Stage 3（action expert flow matching denoise）的一部分；Stage 1（vision）+ Stage 2（LLM）仍要跑。Stage 3 在 Pi0.5 上**约占整个 inference 时间的 50%**（剩下 50% 是 S1+S2，按经验各占约 25%）。所以 `start_t=0.3` 的 WARM_START 实际节省 `0.5 × (1 − 0.3) = 35%` inference 时间，`start_t=0.5` 节省 25%，`start_t=0.7` 节省 15%。FULL_HIT 跳所有 stage，节省 100%。

记号上不再叫 `cycles_saved_ratio` —— "cycle 数"的语义会让"WARM_START 算半 cycle 还是整 cycle"含糊，统一记为 **inference time** 维度更清晰。Phase 0 跑完后可以从 timer probe（`SystemTimer` 已实现）实测 stage-wise 时间占比并校正这个 0.5 系数；当前 plan 的所有 sweep 均按 0.5 估算。

### 2.2 辅指标

| 指标 | 定义 | 用途 |
|------|------|------|
| `cp1_full_hit_rate` | FULL_HIT verdict / total verdict | 看 verdict 在哪一档 |
| `cp1_warm_start_rate` | WARM_START verdict / total verdict | 同上 |
| `mean_start_t` | WARM_START 平均 start_t | start_t 越大越省 |
| `verdict_factor_log` | 每 verdict 的 (5 因子 raw, factor norm, tier, winner_id, episode_outcome) 三元组（仅 ablation 用） | calibration 分析的原料；Phase 0 起跑 |
| `cold_start_force_miss_rate` | cold-start 期间被强制 MISS 的 verdict 比例 | 评估 normalizer 选型 |

### 2.3 Baseline 与对照（**绝无 ThresholdJudge**）

| Tier | Judge | 含义 | 用途 | 数据来源 |
|------|-------|------|------|---------|
| **Floor** | `MISS-only`（关 cache，纯 inference）| 100% 算力、100% 正确 | success_rate 下界参照 | **复用** `exp/warm_start/data/baseline_failures.json` `stats.inference` 行（500 ep × 3 cfg，平均 0.987）|
| **Ceiling-A** | `AlwaysHit`（top-1 全 FULL_HIT）| 不挑、有就用 | success_rate **上界**；同时是 calibration 数据的**主源**（每 verdict 都被记录为 hit，可标 outcome） | **复用** `baseline_failures.json` `stats.always_hit` 行（500 ep × 3 cfg，0.674–0.696）；calibration JSONL 需新跑 3 cfg 的 server-side dump 各一次（`AlwaysHit + judge.dump:{...}`，详见 §4 Phase 0） |
| **Ceiling-W** | `AlwaysWarmStart(start_t)` × {0.3, 0.5, 0.7} | 不挑，全 WARM_START | start_t 选择参照；推断"中等信任档"的安全余量 | **复用** `exp/warm_start/data/{clip,max_pool,spatial16}/cache_eval_results.json`（9 cfg × 500 ep）|

`AlwaysHit`/`AlwaysWarmStart` 已在 codebase 中（`AlwaysHitJudge` / `AlwaysWarmStartJudge`），无需新代码；详见 §3.8 复用清单。

### 2.4 单元 + 统计有效性

- **每 run = 100 ep**（10 task × 10 固定 init），所有 yaml 共用同一组 `(task_id, orig_init_state_idx)` 切分（字段命名对齐 `cache_eval_results.json` row schema）。
- **固定 init 切分（复用已有 mechanism）**：`examples/libero/main.py` 的 `Args.episode_filter` (`main.py:88`) **已支持** JSON filter `[{task_id, orig_init_state_idx, subset_init_state_idx}, ...]`，trajectory_deviation Step 1b/3 已用过。本 plan 选取 `orig_init_state_idx ∈ {0..9}` × 10 task = 100 ep 子集，写到 `exp/verdict_factor_judge/config/init_subset.json`，作 single source of truth。**Runner 端只需让 `run_cache_experiments.py` 透传 `--episode-filter PATH` 给 `main.py`**（不需要在 main.py 新增 flag）。
- **现有 baseline 复用要 paired sub-aggregate**：warm_start baselines 是 500 ep 全集；analysis 在该 100 ep 子集上**重新 aggregate** baseline outcomes，join key 用 `(config_id, task_id, orig_init_state_idx)` —— 这是 `cache_eval_results.json` 行的实有字段（`row 0 = {task_id, init_state_idx, orig_init_state_idx, episode_id, seed, success, config_id, attempt, source_path}`）。不直接对比 500 ep 平均与本 plan 100 ep 平均（避免子集难度混淆 judge 质量）。
- **决策点统计协议**：所有 success_rate 比较改用 paired test：
  - **Wilson 95% CI** 给单个 run 的 success_rate
  - **McNemar test** 给两 run 的 paired success diff（同 init 上的对比）
  - "改善 ≥ 1pp" 之类的阈值改成 "McNemar p < 0.10 且 Wilson 区间不重叠"
  - n=100 时 Wilson 区间宽度 ≈ ±5pp（p≈0.7），决策需要充分容忍噪声
- 时间估算不在本 plan 范围（取决于 server 数 / worker 数 / 当前负载）。

---

## 3. 设计空间（实验维度）

每个维度独立；后续 Phase 在不同维度切片做 ablation。

### 3.1 Composer 维度（5 候选）

| 代号 | Composer | 已实现？ | 适用 |
|---|---|---|---|
| C-WS | `WeightedSumComposer`（S1）| ✅ | 因子加权和 + tier threshold；最易 sweep |
| C-AND | `AndGateComposer`（S2）| ✅ | 每因子独立阈值 ∀ 通过；保守，适用 FULL_HIT |
| C-OR | `OrGateComposer`（S3）| ✅ | 任一因子通过；激进，适用 WARM_START fallback |
| C-HIER | `HierarchicalComposer`（**新**）| ❌ | "F2 一票否决 → F1a-A 一票否决 → 剩下 weighted_sum" 三段式 |
| C-2STAGE | 串联 (`AndGate` for FULL_HIT) + (`WeightedSum` for WARM_START) | ❌ | 把"高信任档"和"中等档"用不同 composer 实现 |

C-HIER 与 C-2STAGE 仅在 Phase 3 ablation 显示标准 composer 不足时才落地实现；Phase 1–2 用 C-WS + C-AND + C-OR 即可。

### 3.2 因子集维度（6 候选 × 因子组合）

「因子」≠「描述子」：本表只锁定**激活哪些因子**；每个 F1a / F1b 因子内部默认产 4 个描述子（jerk / dir / curv_radius / cum_disp）× 1 或 N 个窗口 = 多个 descriptor key。描述子级别的启用 / 权重 / direction 见 §3.4b。

| 代号 | 因子集 | 默认 key 数（4 desc 全开 + W-MIX 3 win）| Hypothesis |
|---|---|---|---|
| F-MIN-A | `f1a_a` only | 4 keys（`f1a_a_{jerk,dir,curv_radius,cum_disp}`）| splice 衔接是单一最强信号？ |
| F-MIN-CONS | `f2` only | 1 key（`f2_var`）| retrieval 自我怀疑足够吗？ |
| F-SPLICE | `f1a_a` + `f1a_t` | 4+4 = 8 keys | A/T 互补能补全 splice 信号？ |
| F-INTRINSIC | `f1b_a` + `f1b_t` | 4×3×2 = 24 keys（W-MIX 3 win）| candidate 自身质量足够，无需 splice？ |
| F-A-ONLY | `f1a_a` + `f1b_a` + `f2` | 4 + 4×3 + 1 = 17 keys | 不依赖 state（最 robust 上线集）？ |
| F-FULL | 5 因子全开 | 4 + 4 + 4×3×2 + 1 = 33 keys | 上限设定 |

`f1a_t` 需要 chain walk + state 信号质量未知，列入但优先级低于 A 系列。

### 3.3 F1b 窗口维度（已 build 21 windows）

artifact 已含每 entry 168 keys（4 desc × 21 win × 2 family），window 是 yaml-side 选哪些 key 进 verdict，**不需要 rebuild artifact**。

**实测 trajectory 长度**（3 cfg 全部一致，commit `66a341f` artifact pkl）：49 trajectory / 1018 entries / T 范围 14–27 / median 21（**不是几百步**）。窗口集设计 + 默认推荐都要按这个尺度选。

| 代号 | 窗口集 | 各窗口 NaN%（实测）| 综合 NaN% | 用途 |
|---|---|---|---|---|
| **W-SHORT** | `(0,3) (1,1) (3,0)` | 14.4 / 9.6 / 14.4 | ~13% | "最稳" — NaN 比例最低；Phase 1–3 默认 |
| **W-MIX** | `(0,3) (1,1) (3,0) (0,5) (5,0)` | + 24.1 / 24.1 | ~17% | 短 + 中等未来/过去；Phase 4 主推 |
| W-FUT | `(0,3) (0,5) (0,7)` | 14.4 / 24.1 / 33.7 | ~24% | "纯未来" — chain-walk 入口与 multi-step skip 偏好 |
| W-PAST | `(3,0) (5,0) (7,0)` | 14.4 / 24.1 / 33.7 | ~24% | "稳定阶段判定" — entry 是否来自稳定区段 |
| W-SYM-S | `(1,1) (3,3)` | 9.6 / 28.9 | ~19% | "短对称" — 邻域综合质量 |
| **W-LONG-RISK** | `(5,5) (7,7)` | 48.1 / 67.4 | **~58%** | ⚠ NaN 占多数；不入默认，仅 Phase 4 描述子启用 sweep 评估 |

**重要**：曾经的 v3 默认 W-MIX `(0,5) (3,3) (5,5)` 综合 NaN ~34%，且含 (5,5) 48% 单窗口 — **已弃用**。新 W-MIX 改为 5 个短/中窗口，最长仅 (0,5) / (5,0)。

`(7,7)` 在 T=14 的 trajectory 上 100% NaN（窗口宽度 15 > 14），在 T=27 的也只覆盖 13/27 个 entry — 接近不可用。

### 3.3b F1b 长窗口边界 NaN 兜底策略（必须设计）

**事实**：F1b OfflineWriter 的边界规则是"窗口越界 → 该窗口所有描述子写 NaN"（`source_window.py:241-247`，`if lo < 0 or hi >= T:` 后赋 NaN 并 `continue`）。给定 trajectory 长度 $T$，窗口 $(p, f)$ 让头 $p$ 个 + 尾 $f$ 个 entry 全 NaN，valid 数 = $\max(0, T-p-f)$。

**实测 NaN 比例**（基于 commit `66a341f` 的 libero_spatial pkl，3 cfg 全相同：49 trajs / 1018 entries / T 范围 14–27 / median 21）：

| 窗口 | total | valid | NaN | NaN% | 等级 |
|---|---|---|---|---|---|
| `(0, 1)` | 1018 | 969 | 49 | **4.8%** | 极低 |
| `(0, 3)` / `(3, 0)` | 1018 | 871 | 147 | **14.4%** | 低 |
| `(1, 1)` | 1018 | 920 | 98 | **9.6%** | 低 |
| `(0, 5)` / `(5, 0)` | 1018 | 773 | 245 | **24.1%** | 中 |
| `(3, 3)` | 1018 | 724 | 294 | **28.9%** | 中 |
| `(0, 7)` / `(7, 0)` | 1018 | 675 | 343 | **33.7%** | 高 |
| `(5, 5)` | 1018 | 528 | 490 | **48.1%** | ⚠ 极高 |
| `(7, 7)` | 1018 | 332 | 686 | **67.4%** | ⚠ 不可用（T=14 的 traj 直接 100% NaN） |

注：上面是**全 entry 均匀采样**下的 NaN%。Verdict-time NaN% 取决于 retrieval winner 是否偏向中间或两端 entry — 待 §3.3b B-1 instrumentation 实测。

**问题**：T median=21 之下，"长窗口" `(5,5) (7,7)` 的 NaN% 是 48%/67%，winner 落到 NaN 区域是常态，不是边缘情况。把它们配进 yaml 给非零 weight，CompositeJudge 会**因 NaN 比例过高被动退化为 MISS**，污染 ablation 结论。

**兜底策略（plan 层面强制）**：

- **B-1（必做）— Phase 0 NaN 比例 instrumentation**：server-side `DumpingJudge` 写 JSONL 时含 `factor_nan: {key → bool}` 矩阵（schema 见 §4 Phase 0）；analysis 脚本算 **per-key NaN% (winner-conditional)**，与上表"全 entry"NaN% 比较，看 retrieval 是否系统偏向某段。Phase 1 启动前必须有这份数据。
- **B-2（必做）— 短-长窗口配对**：任何包含**高 NaN 窗口**（NaN% ≥ 25%，即 `(0,5) (5,0) (3,3) (0,7) (7,0) (5,5) (7,7)`）的 yaml，**MUST 同时含至少一个低 NaN 窗口 key**（NaN% < 15%，即 `(0,3) (3,0) (1,1)` 之一），并保证低 NaN key 在 weighted_sum 里 weight 总和 ≥ 高 NaN key weight 总和。
- **B-3（极高 NaN 窗口禁用清单）**：`(7, 7)` 在 T=14 的 trajectory 上 100% NaN，**整个 plan 默认不入任何 yaml**（仅在 Phase 4 描述子启用 sweep 中作为单独评估对象）；`(5, 5)` 仅在 Phase 4 W-LONG-RISK ablation 评估，不入 Phase 1/2/3/5/7 默认配置。
- **B-4（默认隐式兜底）— Composer NaN-skip**：现有 `WeightedSumComposer` 对 NaN 跳过求和与权重；当其它 key 占满全部 weight 时退化成"低 NaN 窗口加权和"。cold-start sentinel（all-NaN → MISS）保证完全无信号时安全 MISS，不会强行命中。
- **B-5（条件触发）— OfflineWriter 边界算法升级**：若 Phase 0 winner-conditional NaN% > 25% **且** Phase 4 W-LONG-RISK ablation 显示长窗口确有不可替代信号 → 触发 OfflineWriter 算法升级（边界 entry 用截断窗口做 best-effort 估计），列入后续 plan，不在本 plan 内。
- **B-6（否决）— 在线侧拉邻居**：通过 `view.walk_next/prev` 在 winner 边界时取邻居 entry 的同窗口 key 替代。否决理由：会破坏"winner-local"语义（描述子讲的就是这个 entry 的邻域），且带来跨 entry 数据污染。如未来需要，独立 plan 重新论证。

**对 §3.4b directions 的影响**：non_monotonic 描述子的 NaN 在 `WeightedSumComposer` / `OrGateComposer` 都是跳过；但 `AndGateComposer` NaN = not passed，会让 AND-gate 在边界 entry 上几乎必 MISS。Phase 3 ablation C-AND 必须**只配低 NaN 窗口 key**（W-SHORT 集合内），否则 AND-gate 评估失真。

每个 F1a / F1b 因子（`f1a_a` / `f1a_t` / `f1b_a` / `f1b_t`）内部产**同一组 4 个描述子**：

| 描述子 | orientation | 直觉 | F1a 单 key | F1b N 窗口 key 模板 |
|---|---|---|---|---|
| `jerk` | risky | 加速度跳变幅度，越大越 risky | `f1a_<a/t>_jerk` | `f1b_<a/t>_jerk__p<p>_f<f>` |
| `dir` | safe | 方向一致性 ∈[−1,1]，越大越平滑 | `f1a_<a/t>_dir` | `f1b_<a/t>_dir__p<p>_f<f>` |
| `curv_radius` | non_monotonic | 几何弥散度（停滞≈0，圆弧中等，大幅直线大）| `f1a_<a/t>_curv_radius` | `f1b_<a/t>_curv_radius__p<p>_f<f>` |
| `cum_disp` | non_monotonic | 累计路径长度（小+低 jerk=静止；大=快速移动）| `f1a_<a/t>_cum_disp` | `f1b_<a/t>_cum_disp__p<p>_f<f>` |

`f2` 是例外：只产 1 key (`f2_var`, risky)，无 4-描述子展开。

**单因子内 N 个 desc 的处理选项**（描述子级 ablation）：

| 代号 | 描述子启用集 | 何时用 |
|---|---|---|
| **D-ALL** | 4 个全开（默认）| Phase 1–4 默认；让 weighted_sum 自己决定权重 |
| **D-SAFE-RISKY** | 仅 `jerk` + `dir`（避 non_monotonic）| 排除 direction 配置不确定性；最简启动配置 |
| **D-JERK** | 仅 `jerk` 单 key | 单描述子隔离 — 验"最强一项是不是已经够了" |
| **D-DIR** | 仅 `dir` 单 key | 同上 |
| **D-CURV** | 仅 `curv_radius` 单 key | 同上；需在 yaml 显式给 `direction` |
| **D-CUM** | 仅 `cum_disp` 单 key | 同上 |

**non_monotonic direction 配置规则**（curv_radius / cum_disp 必须显式 specify，否则 validator 拒收）：

| direction 形式 | 含义 |
|---|---|
| `"high"` | 数值越大越偏 hit |
| `"low"`  | 数值越小越偏 hit |
| `"range:[lo,hi]"` | 落在 [lo, hi] 区间内偏 hit（如 `curv_radius range:[0.3, 1.0]` 喜欢"中等弥散度"） |

**directions 的 input space 取决于 normalizer**：

`_apply_direction(direction, v)` 收到的 `v` 是 **Composer 实际输入空间** 的值，不是 raw factor。两种模式下取值范围完全不同：

| Normalizer | Composer 收到的 `v` 范围 | range:[lo,hi] 的语义 |
|---|---|---|
| `N-PCT` / `N-PCT-LEN` (默认) | `v ∈ [0, 1]` (percentile rank) | "落在 P_lo–P_hi 区间偏 hit" |
| `N-PASS` (passthrough) | `v ∈ raw / z-score scale` | "原始 / z-score 量级落在 [lo, hi] 偏 hit" |

**默认本 plan 取值（N-PCT 模式，覆盖 Phase 1–3）**：所有 `range:[lo, hi]` 锚定 percentile **中段**（避两端极值）。

```yaml
directions:
  # F1a 单窗口（N-PCT）
  f1a_a_curv_radius:        range:[0.3, 0.7]   # 中等百分位 → 中等几何弥散度
  f1a_a_cum_disp:           high               # percentile 越高 → 累计路径越长 → 偏 hit
  f1a_t_curv_radius:        range:[0.3, 0.7]
  f1a_t_cum_disp:           high
  # F1b 多窗口 W-MIX = (0,3) (1,1) (3,0) (0,5) (5,0)；每窗口 mirror
  f1b_a_curv_radius__p0_f3: range:[0.3, 0.7]
  f1b_a_cum_disp__p0_f3:    high
  f1b_a_curv_radius__p1_f1: range:[0.3, 0.7]
  f1b_a_cum_disp__p1_f1:    high
  f1b_a_curv_radius__p3_f0: range:[0.3, 0.7]
  f1b_a_cum_disp__p3_f0:    high
  f1b_a_curv_radius__p0_f5: range:[0.3, 0.7]
  f1b_a_cum_disp__p0_f5:    high
  f1b_a_curv_radius__p5_f0: range:[0.3, 0.7]
  f1b_a_cum_disp__p5_f0:    high
  # f1b_t_* 同样镜像
```

**N-PASS 模式 anchor**（仅 Phase 4 N-PASS sweep yaml 用）：`range:[0.3, 1.5]` —— z-score 后的"中等几何弥散度"经验区间（P30–P95 of jerk-active subspace）；Phase 0 calibration JSONL 跑完后用每 cfg 实测分布校正这两套 anchor。

**自检规则**：`generate_yamls.py` 必须根据当前 yaml 的 `normalizer.type` 选对 anchor 表，**不允许混用**。Phase 4 N-PASS yaml 用 N-PASS 表；其它 yaml 用 N-PCT 表。

**与 §3.2 因子集 / §3.3 F1b 窗口的正交关系**：

- F-MIN-A × D-ALL = 4 keys
- F-MIN-A × D-JERK = 1 key
- F-INTRINSIC × W-MIX × D-ALL = 4×3×2 = 24 keys
- F-INTRINSIC × W-MIX × D-SAFE-RISKY = 2×3×2 = 12 keys（避 non_monotonic 配置）
- F-FULL × W-MIX × D-ALL = 33 keys（§3.2 表）

**Phase 1 / 2 / 4 中描述子级别 ablation 的位置**：
- Phase 1 单因子 default = D-ALL；额外加 1 套 D-JERK + 1 套 D-DIR 单 desc 对照 + 1 套 F-FULL × T-DUAL_07 WARM_START 探针（合 **8 yaml × 3 cfg = 24 run**）
- Phase 4 加一组 "描述子启用集" sweep（D-ALL / D-SAFE-RISKY / D-JERK + best non-mono / 等），看是否值得砍 keys

### 3.4 Normalizer 维度（4 候选）

| 代号 | Normalizer | 已实现？ | 用途 |
|---|---|---|---|
| **N-PCT** | `PercentileRollingNormalizer(window=50, force_miss)` | ✅ | **默认**（window 200→50 是 Phase 1 实证修正，见 §3.4b）|
| N-PCT-LEN | 同上 cold_start_strategy=`lenient` | ✅（参数） | 减少 cold-start MISS |
| N-PASS | passthrough（不归一化）| ✅（参数） | 直接用 raw 值 — 看 raw scale 自身是否可用 |
| N-PRELOAD | **新**：用 build 时已知的 entry-level factor 分布预填 buffer | ❌ | 消除冷启动；Phase 4 实现 |

### 3.4b ⚠ Multi-worker normalizer cold-start 放大（Phase 1 实证）

**事实**：Server 端 `build_per_connection_components` 给每个 ws connection 一份独立的 `CompositeJudge` + 独立的 normalizer。LIBERO eval 的 `--num-workers N` 开 N 个 main.py 子进程 → N 个独立 ws → N 个独立 normalizer。**buffer 不跨 worker 共享**，每个 worker 都要自己暖一遍。

**实测（idx 4 = F-FULL × T-DUAL_07，100 ep × `--num-workers=5`，window_size=200）**：

```
per-worker verdict 预算   = 100 ep / 5 workers × ~21 verdict/ep ≈ 420
最慢 key NaN 率（F1b）   ≈ 44%   → buffer 填满速率 56%
per-worker cold-start    = 200 / 0.56 ≈ 357 verdict (85% 浪费)
5-worker 总 sentinel firing ≈ 1785 / 2100 = 89.5%
```

没 P0 `all_nan_fallback=warm_start` 救援的话，这 89.5% 全是 MISS → success 接近 Floor → 整个 Phase 1 8 个 yaml 都看不到因子作用。**这是 Phase 1 单因子 yaml "全 MISS" 现象的真因**（不是 plan §8 风险登记里假设的 "winner-tail dead loop"）。

**修法**（按改动量）：
1. **N-PCT window 200→50**（已应用于全部 24 Phase 1 yaml）：cold-start 降到 21%，percentile 分辨率 0.5%→2%，是 Phase 1 默认。
2. `--num-workers 1`：cold-start 占比降到 ~17%（单 worker 拿全 100 ep 预算），但 wall-clock × 5。
3. **N-PRELOAD**：用 Phase 0 calibration JSONL 预填 buffer，**0 cold-start**，Phase 4 实施。

**自检**（任何新 cache 实验 plan 都必看）：跑前确认

```
per_worker_verdict_budget × (1 - max_factor_nan_rate)  >  window_size
```

否则 normalizer 永远暖不完。verdict_factor_judge Phase 1 的实证教训也写入 `docs/architecture/cache_system.md` §5.12，做为通用 cache 实验的 operational gotcha。

### 3.5 Tier 维度（3 候选）

| 代号 | Tier | 描述 |
|---|---|---|
| T-FULL | 单 FULL_HIT tier | 最简：要么命中要么 MISS |
| T-DUAL | FULL_HIT + 单 WARM_START（`start_t` ∈ {0.3, 0.5, 0.7} 各一组）| 中等档减压阀 |
| T-MULTI | FULL_HIT + 多 WARM_START tier（不同 score 段映到不同 start_t）| 现有 composer 不直接支持 — 需要 C-2STAGE 或扩 WeightedSum |

### 3.6 参数搜索维度（5 候选）

| 代号 | 方法 | 成本 | 何时用 |
|---|---|---|---|
| S-THEORY | z-score 后 P50/P95 直接当 anchor | 0 | 给 sweep 初值 |
| S-GRID-COARSE | 手动网格 ~20 组 | ~10 h × 1 keybuilder | Phase 1–3 主推 |
| S-CALIB | AlwaysHit run 收集 (因子, outcome)，事后 logistic / decision tree 学 weights | 1 baseline run + analysis | Phase 5 |
| S-BO | Bayesian opt / CMA-ES on (weights, thresholds) | 100+ episode | S-GRID 不收敛时 |
| S-LEARNED | 神经网络 / GBDT learned composer (S4) | 大 | 论文 ablation 才考虑；本 plan 不上 |

### 3.7 锁定的 KeyBuilder × Search Strategy 三套

本 plan **不引入 warm_start 之外的 keybuilder/search strategy 组合**。所有 phase 默认在下表 3 套上跑（中间 phase 可由 owner 决定收紧到 1 套以省算力，见 §4 phase 标注）。

| 代号 | run_id 词根 | KeyBuilder | 字段权重 | trajectory_depth / weights | vector_dims | artifact pkl |
|---|---|---|---|---|---|---|
| **CFG-CLIP** | `clip_w7_d4` | `clip` | v0=0.1 / v1=0.1 / prompt=0.0 / state=0.8 | 4 / [0.4, 0.3, 0.2, 0.1] | v0=512 / v1=512 / prompt=2048 / state=32 | `exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl` |
| **CFG-MAX** | `max_pool_w3_d5` | `cp1_max_pool` | v0=0.5 / v1=0.0 / prompt=0.0 / state=0.5 | 5 / [0.35, 0.25, 0.2, 0.12, 0.08] | v0=2048 / v1=2048 / prompt=2048 / state=32 | `exp/common/data/cache_artifacts/libero_spatial/cp1_max_pool.pkl` |
| **CFG-SP16** | `spatial16_w8_d4` | `cp1_spatial_pool_16` | v0=0.5 / v1=0.25 / prompt=0.0 / state=0.25 | 4 / [0.4, 0.3, 0.2, 0.1] | v0=32768 / v1=32768 / prompt=2048 / state=32 | `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` |

**3 套共用**（所有 verdict yaml 直接复制即可）：
```yaml
gate: { type: always_search }
search_strategy:
  type: weighted_rrf_knn
  top_k: 1
  step_filter: all
  rrf_k: 60
  field_similarity:
    vision_0:    { type: cosine }
    vision_1:    { type: cosine }
    prompt_emb:  { type: cosine }
    robot_state: { type: l2, to_similarity: { type: exp, tau: 0.334717 } }
backend:
  type: in_memory
  in_memory: { index_type: brute_force }   # preload_path 按上表 cfg 替换
write_policy: { type: never }
```

**artifact 路径迁移注意**：warm_start yaml 原指向 `exp/common/data/cache_artifacts/libero_spatial_warm/`，该目录在 commit `66a341f` 已删，重建到 `exp/common/data/cache_artifacts/libero_spatial/`（带 168 keys F1b factors enrich）；新 yaml 一律用新路径。

**不引入**：`cp1_mean_pool` / `cp1_spatial_pool_64` / `cp1_llm_layer_extract` —— 锁定 3 套就是为了 baseline 齐全且跨 cfg 可对照。

### 3.8 已有 Baseline 数据复用清单

| 文件 | 内容 | 用作 |
|---|---|---|
| `exp/warm_start/data/baseline_failures.json` `stats.inference` | pure inference × {clip_w7_d4, max_pool_w3_d5, spatial16_w8_d4} × 500 ep | Floor |
| `exp/warm_start/data/baseline_failures.json` `stats.always_hit` | AlwaysHit × {clip_w7_d4, max_pool_w3_d5, spatial16_w8_d4} × 500 ep | Ceiling-A |
| `exp/warm_start/data/{clip,max_pool,spatial16}/cache_eval_results.json` | AlwaysWarmStart(start_t ∈ {0.3,0.5,0.7}) × 同上 3 cfg × 500 ep | Ceiling-W |
| `exp/common/data/phase1/libero_spatial/experiment_state.json` | AlwaysHit × {cp1_mean_pool, cp1_spatial_pool_16, cp1_spatial_pool_64, cp1_max_pool, clip} × 8 weight × **50 ep**（task=10×ep_per_task=5）| weights pattern hints / 但 episode 数不足以做 Phase 5 calibration 主源 |

**忽略**：`exp/trajectory_deviation/`、`exp/temporal_prune/`、`exp/qdrant_step_knn/`、`exp/random_periodic_gate/`（owner 已确认与 verdict factor judge 实验无关）。

---

## 4. 阶段实施

每个 Phase 有：**目标 / 配置数 / 决策点 / 进入下一 Phase 的条件**。

### 4.0 Yaml 编写时机（execution model）

**不预先批量生成所有 phase 的 yaml**。后续 phase 的 yaml depend on 前一 phase 的决策点取舍 — 提前写就是浪费 + 容易写出脱离实测的配置。

| Phase | yaml 何时能写 | 依赖 |
|---|---|---|
| Phase 0 | **现在可写**（3 cfg × 1 yaml = 3 个）| 无依赖 |
| Phase 1 | **现在可写**（8 yaml × 3 cfg = 24 个，含 D-JERK / D-DIR + F-FULL × T-DUAL_07 探针）| 因子集和描述子启用集（D-ALL / D-JERK / D-DIR）+ T-DUAL_07 探针 yaml 已在 §3.2 / §3.4b / §4 Phase 1 锁定 |
| Phase 2 | Phase 1 完成后 | 按 Phase 1 R5 Pareto 决策（`success >= Floor − δ AND inference_time_saved_ratio > 0` + T-DUAL carry-forward）保留的因子 / tier 组合作 F-SPLICE / F-INTRINSIC / F-A-ONLY / F-FULL 的成员 |
| Phase 3 | Phase 2 完成后 | 因子集 = Phase 2 winner |
| Phase 4 | Phase 3 完成后 + Phase 0 NaN% 数据 | composer = Phase 3 winner；窗口集是否含 W-LONG-RISK 看 Phase 0 winner-conditional NaN%（§3.3b B-3）；描述子启用 sweep 是否做看 Phase 1 决策点 |
| Phase 5 | Phase 4 完成后 + Phase 0 calibration JSONL 完整 | 结构（因子集 / composer / 窗口 / normalizer / tier）= Phase 4 winner；weights 由 calibration_fit.py 学出（per-cfg + xcfg 共 4 套）|
| Phase 7 | Phase 5 完成后 + libero_10 artifact prerequisite | yaml = Phase 5 default × 3，artifact path 替换为 `libero_10/...pkl`；F1b 窗口集若依赖 libero_10 不同 T 分布需重新审视（§3.3b memo） |

**实务流程**（每 phase 一轮）：
1. 写 yaml（`generate_yamls.py` 从 phase-spec + cfg-spec 笛卡尔展开 — Phase 0/1 的 spec 现在写，后续 phase 的 spec 跟着 phase 决策写）
2. `validate_cache_config` 过 7 项 composite 静态校验
3. `run_cache_experiments` 跑 3 cfg × N yaml × 100 ep（3 server 并行）
4. analysis 脚本（`summarize_phase.py` / `factor_correlation.py` / `plot_pareto.py`）出 1-page md 到 `analysis/<phase>_summary.md`
5. 据决策点取舍 → 写下一 phase 的 spec

**例外**：如 Phase 0 决策点判定"3 cfg 因子分布 + winner-conditional NaN% 高度一致"，owner 可以提前批写 Phase 1–4 全部 yaml（spec 已稳定），减少串行等待。但**默认顺序执行**，避免基于错误假设批量生成被弃用的 yaml。

### Phase 0 — Baseline 复用 + Calibration Dump（cfg ×3）

- **目标**：从已有 baseline 数据 join 出 success_rate 上下界（无需重跑），并跑 3 套 cfg 的 AlwaysHit + dump verdict-factor 拿 calibration 原料。
- **复用（不跑新 episode）**：§3.8 列的 `baseline_failures.json` + `warm_start/data/{clip,max_pool,spatial16}/`；analysis 直接 join。
- **新跑**：
  - **3 × `AlwaysHit` × {CFG-CLIP, CFG-MAX, CFG-SP16} × 100 ep**：dump 在 server 端启用（不通过 runner CLI），每个 yaml 含 `judge.dump: {path, config_id, factors}` 字段；server 构造 judge 时自动包 `DumpingJudge` wrapper（详见 §6.2），每 verdict 的 5 因子 raw 值 + factor_nan + winner_id + cp1_score + (task_id, orig_init_state_idx, step_idx) → JSONL 到 `exp/verdict_factor_judge/data/calibration/{cfg}.jsonl`。
  - **复跑** AlwaysHit 而非任何新 judge —— calibration data 必须保证"每 verdict 都拿了 hit 后的 outcome"，CompositeJudge 一旦给 MISS 就丢了对应数据点。
  - **dump-mode top_k override**：`search_strategy.top_k` 的语义是 **search 返回给 judge 的候选池大小**，与 judge / 因子的"top_k"概念无直接绑定关系（参考 `src/openpi/cache/config.py:1694` `_build_search_strategy` 的 `effective_top_k = max(cfg.top_k, min_top_k_hint)`）。
    - 默认 yaml `top_k=1`：search 返回 1 个候选。Judge（不论 AlwaysHit 还是 CompositeJudge）拿到该 list 决定 verdict。
    - F2 旁路 dump 用同一 results list 算 variance，`consensus.py:85-89` 的 `K_eff = min(self.K, len(results)); if K_eff < 2: return {"f2_var": float("nan")}` 在 results 不足 2 时返回 NaN；`AlwaysHitJudge` (`judge.py:114-137`) 不暴露 `min_required_top_k` 属性 → `getattr(judges[cp_id], "min_required_top_k", 0)` (`config.py:1264`) 返回 0 → effective_top_k = max(1, 0) = 1 → F2 dump 必 NaN。
    - **Phase 0 dump yaml 直接把 `search_strategy.top_k` 写成 5**（=本 plan 启用的最大因子 K，目前是 F2 默认 K=5）。这让 search 返回 5 个候选；AlwaysHit verdict 行为不变（仍用 results[0]），dump 旁路的 F2 OnlineExtractor 拿到 5 个候选能正常算 variance。
    - Phase 1+ 走 CompositeJudge 时 yaml 仍写 `top_k=1`，由 `min_top_k_hint` 自动撑到 5（`config.py:1264`）—— 不需要在 verdict yaml 里手动 override。
- **配置数**：3 yaml × 100 ep = 300 ep
- **决策点**：
  - **预期决策**（基于已有 baseline 数据）：3 cfg 的 always_hit 都跌到 ~0.69（vs pure 0.984），中间 ~30pp 是 verdict judge 的可救空间，方向稳定。
  - `AlwaysWarmStart(0.7)` 在 3 cfg 上都接近 pure inference（0.96–0.98），说明 start_t=0.7 的 WARM_START 几乎不损 success；这一档应当作为 verdict judge 的"安全 fallback"档优先采用。
  - 跨 cfg calibration JSONL 比对：若 3 套 cfg 的因子值分布形态接近，后续 Phase 可只在 1 cfg 上 sweep 然后用另 2 cfg 验证；若分布差异显著，Phase 1–4 必须 3 cfg 全跑。
- **新代码（server-side 入口）**：
  - 验真：`exp/common/run_cache_experiments.py:84-100` `_send_cache_config` 仅发送 `yaml_content` 给 server，再 `subprocess` 启 `examples/libero/main.py`；它不接触 `Orchestrator.check()` 内部 locals (`results`/`view`/`history`/`cached_data`)。
  - 验真：`scripts/serve_policy.py:267-277` 构造 `CacheOrchestrator` 时无 dump hook 参数；`Orchestrator.__init__` 也无 observer / hook 字段。
  - 验真：verdict 真实调用点在 `src/openpi/cache/orchestrator.py` 的 `view = StoragePayloadView(...)` + `history = HistoryView(...)` + `judge_result = judge(results, checkpoint_id, cached_data, view=view, history=history)` 一段 server-side。
  - **方案**：新增 `DumpingJudge` wrapper 类（落在 `src/openpi/cache/components/judge.py` 或新文件 `dump_judge.py`），`__call__(results, checkpoint_id, cached_data, *, view, history)` 内：(1) 先调 inner judge 拿 result；(2) 把同一组 args 喂一组配好的 OnlineExtractor 拿 raw + nan dict；(3) append 一行到 JSONL；(4) 返回 inner result（**透明包装，不改 verdict**）。
  - **YAML 入口**：`JudgeConfig` schema 加可选字段 `dump: {path: str, config_id: str, factors: list[FactorConfig]}`。`config_id` 必填，命名约定 = `dump.config_id == yaml stem`，由 generate_yamls.py 强制；§6.1 三 cfg 子目录的 yaml 文件名 stem 全局唯一（cfg-prefixed naming，例 `clip_w7_d4_phase0_always_hit_dump.yaml` stem == `clip_w7_d4_phase0_always_hit_dump`），与 `run_cache_experiments.py:414` `run_id=yf.stem` 决定的 `cache_eval_results.json.config_id` 命名空间一致 → JSONL config_id == cache_eval_results config_id 保证 join 通。`_build_judge` 检测到 `dump` 字段时先 build inner judge，再包 `DumpingJudge(inner, factors_extractors, path, config_id, identity_fields=["task_id", "orig_init_state_idx", "step_idx"])`。
  - **identity 字段来源 — episode_start `extra_metadata` 通道**：

    **现状链路 grep 验真**：
    - `examples/libero/main.py` 算 `task_id` / `orig_init_state_idx` 但 `client.episode_start(experiment, task, episode_id, episode_name)` 只传 4 字段。
    - `WebsocketClientPolicy.episode_start` (`packages/openpi-client/src/openpi_client/websocket_client_policy.py:56-80`) wire 序列化只有 `__experiment__/__task__/__episode_id__/__episode_name__`。
    - server-side `websocket_policy_server.py:212-217` 从 obs 4 字段 read 后 `conn_policy.on_episode_start(experiment, task, episode_id, episode_name)`。
    - `InferenceInterceptor.on_episode_start` (`interceptor.py:279-301`) 显式 `del experiment, episode_name`，只 forward `task_key=task, episode_id=str(episode_id)` 给 Orchestrator。
    - `Orchestrator.on_episode_start(task_key, episode_id)` (`orchestrator.py:219`) 只这 2 字段；`_broadcast_episode_start()` 给 judge lifecycle 不带 kwargs。
    - **结论**：`orig_init_state_idx` 当前无任何路径可达 server-side judge，必须扩通道。

    **方案：扩 `episode_start` 的 `extra_metadata: dict | None` 字段，全链路透传 — 混合 dispatch**：conn_policy wrapper 链显式扩 `extra_metadata: dict | None = None` kwarg（wire/server/wrapper 都改）；judge lifecycle 走 `_safe_call_lifecycle` 已有的 `inspect.signature` filtering 路径不动现有 judge。改动列入 §6.2，链路：

    1. `WebsocketClientPolicy.episode_start` 加 `extra_metadata: dict | None = None` kwarg → 序列化字段 `__extra__: {...}` (空 dict 时省略保 wire backward compat)
    2. `examples/libero/main.py` 两处 `client.episode_start(...)` 加 `extra_metadata={"task_id": task_id, "orig_init_state_idx": orig_init_state_idx}` — 必含 `task_id` int（`task` 字段是 `str(task_description)` 不能反推；main.py loop 已有 `task_id` int 变量直接取）
    3. `websocket_policy_server.py:212-222` `conn_policy.on_episode_start(...)` 5 字段 keyword 调用：加 `extra_metadata=obs.get("__extra__", {})`
    4. `src/openpi/policies/policy.py:253-261` `Policy.on_episode_start` 加 `extra_metadata: dict | None = None` kwarg + 透传 inner policy
    5. `src/openpi/collect/collection_policy.py:115-125` `CollectionPolicy.on_episode_start` 加 `extra_metadata: dict | None = None` kwarg（仅 store-and-forward，不用）+ 透传 inner policy
    6. `src/openpi/collect/data_collector.py:48` `DataCollector.on_episode_start` 同样加 kwarg（与 CollectionPolicy 同款）— 保 server keyword call 5 字段对所有 conn_policy 类型不报错
    7. `src/openpi/cache/interceptor.py:279-301` `InferenceInterceptor.on_episode_start(experiment, task, episode_id, episode_name, extra_metadata=None)` + forward `self._orchestrator.on_episode_start(task_key=task, episode_id=str(episode_id), extra_metadata=extra_metadata)`
    8. `src/openpi/cache/orchestrator.py:219` `Orchestrator.on_episode_start(task_key, episode_id, extra_metadata=None)`：`self._current_episode_extra: dict = dict(extra_metadata or {})` 字段；`on_episode_end` 时 `self._current_episode_extra = {}`
    9. `_broadcast_episode_start()` 给 judge lifecycle 调用走 `_safe_call_lifecycle` 路径：`self._safe_call_lifecycle(judge, "on_episode_start", extra_metadata=self._current_episode_extra)` —— `inspect.signature` 自动 filter，不接受 `extra_metadata` 的 judge（4 个现有：AlwaysHit/AlwaysWarmStart/Threshold/CompositeJudge）自动 swallow；DumpingJudge 显式接 `extra_metadata=None` 拿到。**所有现有 judge / gate / strategy / key_builder / normalizer 完全不动**。
    10. `DumpingJudge.on_episode_start(self, extra_metadata=None)`：`self._current_extra = dict(extra_metadata or {})`；同时 `_safe_call_lifecycle(self._inner, "on_episode_start", extra_metadata=extra_metadata)` 转发给 inner（同款 filtering，inner 是 CompositeJudge 时其 normalizer reset 仍正常调）；verdict 调用时从 `self._current_extra` 读 identity 写 JSONL

    **per-connection isolation**：每 ws connection 已有独立 `InferenceInterceptor` + `CacheOrchestrator` + per-conn judges（`build_per_connection_components` 路径），`_current_episode_extra` 是 per-Orchestrator 实例字段，concurrent worker 之间天然 isolate。

    **on_episode_end / 下一 episode 清空**：在 `_current_episode_extra = {}` 后任何 verdict 都拿空 dict（DumpingJudge 写 JSONL 时缺字段标 `null`）—— 防止跨 episode identity 串。

    **混合 dispatch 的 layered 理由**：
    - conn_policy 链（wire → server → wrapper → InferenceInterceptor → Orchestrator）是显式 keyword call 路径，在 server `websocket_policy_server.py:212` 直接 `conn_policy.on_episode_start(experiment=..., task=..., episode_id=..., episode_name=..., extra_metadata=...)` 调；wrapper 不扩签名会直接 raise TypeError → 显式扩 5 wrapper 签名。
    - judge / gate / strategy / key_builder / normalizer lifecycle 是 `_safe_call_lifecycle` filtering 路径（`orchestrator.py:281-293` `inspect.signature` 已有保护），扩 kwarg 不会破坏旧 component。
    - 这样 conn_policy 链 5 处显式扩签名 + judge/gate/strategy/key_builder/normalizer 0 处改动，最小 blast radius。
  - **runner 不动**：`run_cache_experiments.py` 不加 dump CLI；它已能透传 yaml content 与 episode filter。

  JSONL schema（每行 = 1 verdict，append-only，不含 `episode_outcome` — outcome 由 analysis 阶段 join 拿）：
  ```json
  {
    "config_id": "spatial16_w8_d4_phase0_always_hit_dump",
    "task_id": 3, "orig_init_state_idx": 7, "step_idx": 42,
    "winner_id": "...", "cp1_score": 0.81,
    "factor_raw": {"f1a_a_jerk": 0.34, "f1b_a_jerk__p0_f5": 0.21, ...},
    "factor_nan": {"f1a_a_jerk": false, "f1b_a_jerk__p7_f7": true, ...}
  }
  ```
  写时 append-only，无任何 in-place mutation。

  **字段来源**：
  - `config_id`：DumpingJudge 构造时从 `judge.dump.config_id` 取（必填，§6.2）；与 `cache_eval_results.json.config_id` 命名空间一致（runner `run_cache_experiments.py:414` 用 `run_id=yf.stem`，generate_yamls.py 强制 `dump.config_id == yaml stem`）
  - `task_id` / `orig_init_state_idx`：DumpingJudge 从 `self._current_extra` 读，由 `extra_metadata` 通道（§4 Phase 0 链路 8）注入；`task_id` 是 int（不是 `str(task_description)`）
  - `step_idx`：DumpingJudge 内部计数器；与 episode 级 reset 对齐
  - `winner_id` / `cp1_score`：从 verdict `results[0]` 取
  - `factor_raw` / `factor_nan`：dump extractor 调用产出

  `episode_outcome` 由 `analysis/calibration_fit.py` 用 `(config_id, task_id, orig_init_state_idx)` join `cache_eval_results.json` 的 success 列拿到（row 0 实有字段：`task_id / init_state_idx / orig_init_state_idx / config_id / success / ...`）。`factor_nan` 是 §3.3b B-1 要求的 per-key NaN flag —— Phase 1 启动前 analysis 脚本统计 `per-key NaN%` + `per-window NaN%`，作为 §3.3b B-4 / B-2 决策依据。

  本 JSONL 行同样使用 `orig_init_state_idx`（不用 `subset_init_state_idx`，避免子集化时 init 索引漂移），与 baseline join key 完全一致。

> **Phase 1–4 cfg 数说明**：默认 3 cfg 全跑（保 cross-cfg 一致性观察）；如 Phase 0 决策点判定"3 cfg 因子分布相似"，可由 owner 收紧到 1 cfg（默认 CFG-SP16）以省 2/3 算力，Phase 5 / 7 仍按 3 cfg 验证。下面表格按 3 cfg 估算。

### Phase 1 — 单因子 Ablation（cfg ×3）

- **目标**：
  - **(a)** 每个因子单独使用（D-ALL = 4 描述子全开）的 success / inference_time_saved Pareto 点；
  - **(b)** 在 F1a-A 上额外加单描述子 ablation（D-JERK / D-DIR）—— 看 4 描述子里是否有"一项就够"的最强信号，决定 Phase 4 是否值得 sweep 描述子启用集。
  - **(c)** WARM_START 信号探测：在 F-FULL × D-ALL × T-DUAL(0.7) 上跑一组，避免因子在 T-FULL 下"不显著"被错误剪掉但在 WARM_START 下其实有用。
- **维度切片**：
  - Composer = C-WS（单 key 时退化为单维门）
  - 因子集 × 描述子启用 × tier：
    - {F-MIN-A, F-MIN-CONS, `f1b_a only`, `f1b_t only`, `f1a_t only`} × **D-ALL × T-FULL**（5 yaml）
    - F-MIN-A × **D-JERK × T-FULL**（1 yaml）
    - F-MIN-A × **D-DIR × T-FULL**（1 yaml）
    - F-FULL × D-ALL × **T-DUAL(start_t=0.7)**（1 yaml，**WARM_START 探针**）
  - 窗口 = W-MIX（F1b 系列）
  - Normalizer = N-PCT
- **配置数**：8 yaml × 3 cfg × 100 ep = 2,400 ep
- **决策点（Pareto rule）**：
  - **保留**满足 `success >= Floor − δ_success`（默认 δ = 2pp，Phase 0 后定）**且** `inference_time_saved_ratio > 0`（不是 trivially-MISS 的 zero-saving 配置）的因子。
  - **同时保留**：因子在 T-DUAL(0.7) WARM_START 探针 yaml 里**贡献正信号**（按 verdict_factor_log 看 mean_start_t > 0 且对应 verdict 在该子集 success ≥ paired_floor − δ）的项 — 即"FULL_HIT 不过但 WARM_START 有用" carry forward 到 Phase 2 测 T-DUAL。
  - 决策用 §2.4 协议：Wilson CI + paired McNemar，不用 raw delta。
  - 若 F-MIN-A × D-JERK ≥ F-MIN-A × D-ALL（McNemar 不显著差）→ "单 jerk 足够"信号成立，Phase 4 必须扫描描述子启用集；反之描述子启用集维度可在 Phase 4 砍。

### Phase 2 — 双层探索：因子内部 + 因子组合（cfg ×3，Pareto-first redesign）

> **注意**：原始 Phase 2 设计是"4 yaml 组合 ablation"，但 Phase 1 数据揭示 (a) 没破 Pareto 前沿，(b) 5 个候选因子的内部 descriptor / window 结构未充分探索（Phase 1 只跑 F1a-A 3 个 desc，其他因子内部全是 all-4 × W-MIX 单点）。Phase 2 重设计为双层：**Layer 1 内部探索 → Layer 2 组合 + tier 升级 → Layer 3 winner 大样本**。优化目标改为 (SR, inf_ratio) **Pareto 前沿外推**，单 SR 不够。

#### Phase 2 Layer 1 — 因子内部 descriptor / window 探索（已完成）

- **状态**：✅ Implemented + Validated（2026-04-28 跑完，6-server 并跑）
- **数据**：`exp/verdict_factor_judge/data/phase2_layer1/{cfg}/`（78 yaml × 100 ep × 3 cfg）
- **分析**：[`exp/verdict_factor_judge/analysis/phase2_layer1_results.md`](../exp/verdict_factor_judge/analysis/phase2_layer1_results.md) + [Pareto 图](../exp/verdict_factor_judge/analysis/phase2_layer1_pareto.png)

**26 yaml stem × 3 cfg = 78 unique runs**（split 13/13 进 `phase2_layer1_a/b` 给 6-server batch1-6）：

| 子层 | yaml 数 | 内容 |
|---|---:|---|
| 1.A | 3 | F1a-A close-out: curv / cum 单 desc + jerk+curv pair |
| 1.B | 5 | F1a-T desc sweep: 4 单 desc + jerk+dir pair |
| 1.C | 7 | F1b-A × {W-SHORT/PAST/FUT/SYM-S × all-4} + W-SHORT × 单 desc |
| 1.D | 7 | F1b-T 同 1.C 结构 |
| 1.E | 4 | W-LONG-RISK ((5,5)+(7,7)) × {F1b-A, F1b-T} × {all, jerk} |

全 T-FULL，warm 仅来自 `all_nan_fallback@0.7`。

**Layer 1 实测 winner**（跨 ≥ 2 cfg strict-Pareto-positive vs random_periodic + always-WARM 全 baseline）：

| yaml | clip / max_pool / spatial16 | 主信号 |
|---|---|---|
| **f1b_t_w_long_risk_d_jerk** | ★/★/★ (跨 3 cfg) | NaN-fallback warm + jerk-driven hit, SR 0.96-0.98 @ inf 0.67 |
| f1b_a_w_long_risk_d_jerk | ★ / / ★ | 同上，state→action |
| f1b_t_w_fut_d_all | / / ★ | spatial16 上 SR 0.98 @ inf 0.60，**接近纯 inference 质量** |
| f1b_t_w_short_d_jerk | ★ / / ★ | 短窗 jerk |
| f1a_t_d_jerk_only | ★ / ★ / | F1a-T jerk 单 desc |

Layer 2 直接复用这些 winner。已被淘汰的方向：F1a-A 三个 close-out yaml 全 dominated；W-PAST / W-SYM-S 单独价值有限。

#### Phase 2 Layer 2 — 跨因子组合 + tier 升级（next，待 Layer 1 winner 确认后生成）

- **目标**：用 Layer 1 winner 做组合 + 切到 **T-DUAL_07** 让 composer 主动产 warm tier（Layer 1 全 T-FULL，warm 全靠 NaN fallback；T-DUAL 让 composer 用因子分数 [0.3, 0.5) 主动判 warm）。
- **设计原则**：每 yaml 必含 ≤ 3 因子（避免 F-FULL 33-key 稀释问题），每因子用 Layer 1 选出的 winner desc / window。
- **6 候选 yaml**（每 yaml × 3 cfg × 100 ep）：

| Layer 2.A yaml stem | 因子组合 | tier | 假设 |
|---|---|---|---|
| `f1bt_LR_jerk_t_dual_07` | F1b-T × W-LONG-RISK × jerk | T-DUAL_07 | 跨 3 cfg winner + 主动 warm |
| `f1bt_LR_jerk_t_dual_05` | 同上 | T-DUAL_05 (start_t=0.5) | warm 单次成本 0.85→0.75 |
| `f1bt_w_fut_jerk_t_dual_07` | F1b-T × W-FUT × jerk | T-DUAL_07 | spatial16 信号 + jerk 单 desc |
| `splice_jerk_t_dual_07` | F1a-T.jerk + F1b-T.W-LONG-RISK.jerk | T-DUAL_07 | 双 state 因子互补 |
| `intrinsic_jerk_t_dual_07` | F1b-A.W-LONG-RISK.jerk + F1b-T.W-LONG-RISK.jerk | T-DUAL_07 | A+T 互补 |
| `full_lite_t_dual_07` | F1a-T.jerk + F1b-T.W-LONG-RISK.jerk + F2 | T-DUAL_07 | 3 因子精简组合 |

- **配置数**：6 yaml × 3 cfg × 100 ep = **1,800 ep ≈ 30 min wall-clock（6 server 并跑）**。
- **生成方式**：写 `phase2_layer2_spec.py`（mirror `phase2_spec.py`），落到 `config/{cfg}/phase2_layer2/`。
- **决策点（Pareto-first）**：每 yaml 算 (SR, inf_ratio) → 看是否在某 inf 段 strict-positive vs Layer 1 winner + 全 baseline。**至少 2 cfg 同时 strict-positive 才能进 Layer 3**。
- **失败兜底**：若 Layer 2.A 全部被 dominated → 启发权重路径（每个因子按 Layer 1 单 desc SR 反推 weight ∈ {0.5, 1.0, 1.5}），再 6 yaml × 3 cfg = 1,800 ep。

#### Phase 2 Layer 3 — Winner 大样本 stat power（破噪声）

- **目标**：100 ep 标准误 ±2-3pp 把所有 SR Δ ≤ 3pp 的"看似 strict-positive"全 cover 掉。Layer 3 用 1000 ep × 1 seed 把噪声压到 ±1pp，下真正的 Pareto 结论。
- **候选**：Layer 2 选 **3 个不同 inf 段的 strict-positive winner**（低 inf / 中 inf / 高 SR 各 1 个）。
- **配置数**：3 winner × 3 cfg × 1000 ep × 1 seed = **9,000 ep ≈ 1.5 h（6 server 并跑）**。
- **决策点**：哪个 winner 在 1000 ep 上仍 strict-positive vs full baseline → 进 Phase 3 当 default 因子集。
- **可选 Layer 3.B**：若 1 seed 的 SR 仍跌进 baseline 区间 → 再做 3 seed × 500 ep 看 seed-variance（成本同 Layer 3）。

#### Phase 2 总成本汇总

| 子层 | 状态 | yaml × cfg × ep | 总 ep | wall-clock (6 server) |
|---|---|---|---:|---|
| Layer 1 | ✅ done | 78 × 100 | 7,800 | ~2 h |
| Layer 2.A | next | 18 × 100 | 1,800 | ~30 min |
| Layer 2.B (兜底，可能不跑) | conditional | 18 × 100 | 1,800 | ~30 min |
| Layer 3 | after L2 | 9 × 1000 | 9,000 | ~1.5 h |
| **Phase 2 v2 总** | | | **18,600-20,400** | **~4-4.5 h** |

vs 原始 Phase 2 设计 = 2,400 ep — Phase 2 v2 多 ~8x 但能真正回答 "什么因子组合 + tier + start_t 能突破 Pareto 前沿"。

#### Phase 2 v2 → Phase 3 carry-forward

- 因子集 default = Layer 3 winner（最大概率是 `f1b_t_w_long_risk_d_jerk` 单因子或某个 splice/intrinsic 双因子组合）
- 描述子启用集 default = `[jerk]`（Layer 1 全部维度都证实 jerk 是主信号）
- 窗口 default = `W-LONG-RISK` 或 `W-FUT`（看 Layer 2 哪个 winner 进 Layer 3）
- Tier default = T-DUAL_07（Layer 2 主推方向，让 composer 三档判决）
- start_t default = 0.7 vs 0.5（待 Layer 2 `t_dual_05` 数据决定）

### Phase 3 — Composer 类型 Ablation（cfg ×3）

- **prerequisite — config builder 修复**：当前 `src/openpi/cache/config.py:_build_composer` 仅给 `WeightedSumComposer` 传 `directions`，而 `AndGateComposer` / `OrGateComposer` 构造时 `directions=None → self._directions={}`，使 `bind_orientations` 在含 non_monotonic key 时直接 raise。**Phase 3 启动前必须修这个 bug**：在 `_build_composer` 的 `and` / `or` 分支同样传 `directions=cfg.directions`，并补 `tests/cache/components/factors/` 一组 unit test cover non_monotonic key + AND/OR + directions 绑定。该改动列入 §6.2。
- **目标**：在 Phase 2 选定的因子集上比较 C-WS / C-AND / C-OR；如果都不令人满意，触发 C-HIER / C-2STAGE 实现。
- **维度切片**：
  - Composer ∈ {C-WS, C-AND, C-OR}（uniform threshold = 0.5）
  - 因子集 = Phase 2 winner（**若 builder 修复未完成 → C-AND/C-OR yaml 临时限定 D-SAFE-RISKY，避免 non_monotonic 触发未修 bug**）
  - Tier = T-FULL（先单档）
- **配置数**：3 yaml × 3 cfg × 100 ep = 900 ep
- **决策点**：
  - 若 C-AND success 高但 hit_rate 极低 + C-WS hit_rate 高但 success 跌 → 触发 C-2STAGE（FULL_HIT 用 AndGate / WARM_START 用 WeightedSum）。
  - 若 C-OR success 显著低 + C-AND 显著保守 → 不需要 C-HIER。
  - 若三者都难看 → 触发 C-HIER（cheap factor veto + 余下 weighted）。

### Phase 4 — 窗口 / Normalizer / Tier / 描述子启用 Ablation（cfg ×3）

- **目标**：在 Phase 2-3 选定的因子集 + composer 上做 4 个次维度 sweep；定 default config。
- **维度切片**：
  - 窗口 ∈ {W-SHORT, W-MIX, W-FUT, W-PAST, W-SYM-S, **W-LONG-RISK**}（6 yaml；W-LONG-RISK = `(5,5) (7,7)` 仅 ablation 评估，**§3.3b B-3 禁用清单的延伸验证**：实测长窗口在 NaN 占多数下能否仍贡献信号）
  - Normalizer ∈ {N-PCT, N-PCT-LEN, N-PASS}（3 yaml）；如冷启动 MISS rate > 30% → 触发 N-PRELOAD 实现
  - Tier ∈ {T-FULL, T-DUAL × 3 start_t}（4 yaml）
  - **描述子启用 ∈ {D-ALL, D-SAFE-RISKY, D-JERK + best non-mono}（3 yaml）** —— 仅在 Phase 1 决策点判定"单描述子有信号"时启用；否则砍掉这一组
- **配置数**：(6 + 3 + 4 + 3) × 3 cfg × 100 ep = 4,800 ep；如砍描述子启用维度，回到 3,900 ep
- **决策点**：每维选 cross-cfg 一致 winner，固化进 3 份 default yaml（CFG-CLIP / CFG-MAX / CFG-SP16 各一份）。

### Phase 5 — Calibration-driven Weight Search（S-CALIB，cfg ×3）

- **目标**：用 Phase 0 的 AlwaysHit calibration data（3 cfg 各一份 JSONL），事后学 weights，看是否优于 S-GRID。
- **流程**（outcome 通过 join 拿，决策用 paired stats）：
  1. 加载 Phase 0 落的 3 份 verdict-factor JSONL（每行：`config_id, task_id, orig_init_state_idx, step_idx, winner_id, cp1_score, factor_raw, factor_nan` — append-only，**不含 episode_outcome**）
  2. **Outcome join**：用 `(config_id, task_id, orig_init_state_idx)` join 对应 `cache_eval_results.json` 的 success 列（schema 在 `exp/warm_start/data/clip/cache_eval_results.json` 已实测），得到每 verdict 行的 `episode_success ∈ {0, 1}`；可选更细 weight schema（cycle 内距 episode end 的步数加权）作 ablation
  3. 训 logistic regression / GBDT（在 join 后的 `(factor_raw, episode_success)` 表上），输出 5 因子权重 + 截距 — **3 cfg 各训一组 + 1 组 cross-cfg 平均**
  4. 4 套 weights 写成 yaml，每套 × 对应 cfg 跑 100 ep 验证（cross-cfg avg × 3 cfg + per-cfg × 各自 cfg = 6 run）
- **配置数**：cross-cfg avg yaml × 3 cfg = 3 run + per-cfg yaml × 各自 cfg = 3 run，合计 **6 run × 100 ep = 600 ep**。
- **决策点**：按 §2.4 协议，对每 cfg 跑 paired McNemar test 比较 S-CALIB winner 与 S-GRID winner 的 success：若任一 cfg McNemar p < 0.10 **且** Wilson 95% CI 不重叠（S-CALIB 高于 S-GRID）→ 推 S-CALIB 为 default；否则保留 S-GRID winner。

### Phase 6 — 取消（cross-keybuilder 已内嵌于每 phase）

cross-keybuilder generalization 已内嵌于每 phase × 3 cfg 的决策点（§3.7 锁定 3 套 keybuilder，每 phase 都跑这 3 套），不再单立 Phase。

### Phase 7 — Cross-task Generalization（`libero_spatial` → `libero_10`，cfg ×3）

- **目标**：把 Phase 5 default yaml × 3 cfg 迁到 `libero_10`，验跨任务移植性。
- **prerequisite**：`libero_10` 的 3 套 cache_artifact pkl 需先 enrich F1b factors（参考 `logs/libero_10_cache_artifact_build_plan.log.md` + 加 `--factors-yaml` flag）。本 plan **不**囊括此 enrich 步骤；按 prerequisite 列入。
- **跑什么**：default yaml × 3 cfg × `libero_10` × 100 ep
- **配置数**：3 yaml × 100 ep = 300 ep
- **决策点**：跨任务 success drop < 3pp（任一 cfg）→ 移植成功，本 plan 完结进 archive；否则记录 cross-task gap，列后续 calibration work。

---

## 5. 配置 / Episode 总量估算

每 run = 100 ep（§ 2.4）。每 phase 默认 ×3 cfg。时间估算不在本 plan 范围。

| Phase | 配置数（× 3 cfg）| episode 数 |
|---|---|---|
| Phase 0 | 1 yaml × 3 cfg = 3 run | 300 |
| Phase 1 | 8 × 3 = 24 run（含 D-JERK / D-DIR + T-DUAL_07 探针）| 2,400 |
| Phase 2 | 4 × 3 × 2 tier = 24 run（T-FULL + T-DUAL_07 并跑）| 2,400 |
| Phase 3 | 3 × 3 = 9 run | 900 |
| Phase 4 | 16 × 3 = 48 run（含 W-LONG-RISK + 描述子启用 sweep）| 4,800 |
| Phase 5 | 6 run（cross-cfg avg × 3 cfg + 3 per-cfg × 各自 cfg）| 600 |
| Phase 7 | 1 × 3 = 3 run | 300 |
| **合计** | **117 run** | **11,700 ep** |

收紧分支：
- Phase 0 决策点判定"3 cfg 因子分布相似"→ owner 收紧 Phase 1–4 到单 cfg，episode 数 → ~4,400
- Phase 1 决策点判定"描述子启用维度无意义"→ Phase 4 砍 3 yaml × 3 cfg = 900 ep
- Phase 0 winner-conditional NaN% 显示 W-LONG-RISK 必然不可用 → Phase 4 砍 1 yaml × 3 cfg = 300 ep
- Phase 1 探针显示 T-DUAL_07 与 T-FULL 决策一致 → Phase 2 不并跑 T-DUAL，砍 1,200 ep

不含 calibration 分析 / 新 composer 实现的工作量。新 composer 实现（如 C-HIER / C-2STAGE）每个 ~1 工作日（含 unit test + 接进 `_build_composer`）。

---

## 6. 文件 / 接口变化

### 6.1 新建（plan 落地阶段）

**yaml 命名约定**：`{cfg_id}_{phase_descriptor}.yaml`。`run_cache_experiments.py:414/429` 用 `run_id=yf.stem` 决定 `cache_eval_results.json.config_id`，DumpingJudge 写 JSONL 用 `dump.config_id`，generate_yamls.py 强制 `dump.config_id == yaml stem`，三者全局唯一。例：`clip_w7_d4_phase0_always_hit_dump.yaml` stem 是 `clip_w7_d4_phase0_always_hit_dump`，与 `dump.config_id` 同值，与 `cache_eval_results.json.config_id` 同值 → Phase 5 三键 join 通。

```
exp/verdict_factor_judge/
  __init__.py                                       # 1-line docstring
  config/
    clip_w7_d4/                                     # CFG-CLIP yaml 全集（cfg-prefixed stem）
      clip_w7_d4_phase0_always_hit_dump.yaml
      clip_w7_d4_phase1_f1a_a_d_all.yaml
      clip_w7_d4_phase1_f2_only.yaml
      clip_w7_d4_phase1_f1b_a_only.yaml
      clip_w7_d4_phase1_f1b_t_only.yaml
      clip_w7_d4_phase1_f1a_t_only.yaml
      clip_w7_d4_phase1_f1a_a_d_jerk.yaml            # D-JERK 单 desc 对照
      clip_w7_d4_phase1_f1a_a_d_dir.yaml             # D-DIR 单 desc 对照
      clip_w7_d4_phase1_f_full_t_dual_07.yaml        # T-DUAL_07 WARM_START 探针
      clip_w7_d4_phase2_splice_{t_full,t_dual_07}.yaml
      clip_w7_d4_phase2_intrinsic_{t_full,t_dual_07}.yaml
      clip_w7_d4_phase2_a_only_{t_full,t_dual_07}.yaml
      clip_w7_d4_phase2_full_{t_full,t_dual_07}.yaml
      clip_w7_d4_phase3_and_gate.yaml
      clip_w7_d4_phase3_or_gate.yaml
      clip_w7_d4_phase3_weighted_sum.yaml
      clip_w7_d4_phase4_window_{short,mix,fut,past,sym_s,long_risk}.yaml
      clip_w7_d4_phase4_normalizer_{pct,lenient,passthrough}.yaml
      clip_w7_d4_phase4_tier_{full,dual_03,dual_05,dual_07}.yaml
      clip_w7_d4_phase4_descriptor_{d_all,d_safe_risky,d_jerk_plus_best_nm}.yaml
      clip_w7_d4_phase5_calibrated_per_cfg.yaml
      clip_w7_d4_phase5_calibrated_xcfg.yaml         # cross-cfg avg weights, applied to clip_w7_d4
      clip_w7_d4_phase7_libero_10.yaml
    max_pool_w3_d5/                                 # 同名 mirror，prefix 改成 max_pool_w3_d5_*
      max_pool_w3_d5_phase0_always_hit_dump.yaml
      ...
    spatial16_w8_d4/                                # 同名 mirror，prefix 改成 spatial16_w8_d4_*
      spatial16_w8_d4_phase0_always_hit_dump.yaml
      ...
    # Floor / Ceiling-A / Ceiling-W: 不出 yaml，analysis 直接 join 现有 baseline（§3.8）
  data/
    calibration/
      {clip_w7_d4,max_pool_w3_d5,spatial16_w8_d4}.jsonl   # Phase 0 dump per cfg
    runs/
      <phase>/<cfg>/<config>/cache_eval_results.json
      <phase>/<cfg>/<config>/<task_id>_<init>.episode_results.json
  analysis/
    summarize_phase.py                              # 跨 cfg 对比
    calibration_fit.py                              # logistic / GBDT fit (per-cfg + xcfg)
    plot_pareto.py                                  # success vs inference_time_saved
    factor_correlation.py                           # (factor → outcome) correlation
```

**yaml 模板生成**：因为同 phase 的 3 cfg yaml 只在 `keys / key_builder / search_strategy.trajectory_depth / search_strategy.trajectory_weights / backend.vector_dims / backend.in_memory.preload_path` 几段不同，建议加一个 generator 脚本 `exp/verdict_factor_judge/generate_yamls.py`（类似 `exp/random_periodic_gate/generate_batches.py`）从一份 phase-spec + 3 份 cfg-spec 笛卡尔展开，保证 3 cfg 之间 verdict 配置完全一致。

### 6.2 修改

- `exp/common/run_cache_experiments.py`（dump 入口在 server-side `DumpingJudge`，不在 runner）：
  - 加 `--episode-filter PATH` 透传选项 — `examples/libero/main.py:88 Args.episode_filter` 已实现 episode-level subset filter（trajectory_deviation Step 1b/3 已用过），runner 端只需透传，main.py 端无需改动。
- **`src/openpi/cache/components/judge.py`（新增 `DumpingJudge` wrapper）**：
  - 透明包装任意 SimilarityJudge：`__call__(results, checkpoint_id, cached_data, *, view, history)` 先 forward 到 inner judge 拿 `JudgeResult`，再用同一组 args 喂配好的 OnlineExtractor list 拿 raw + nan dict，append 一行 JSONL（含 `config_id, task_id, orig_init_state_idx, step_idx, winner_id, cp1_score, factor_raw, factor_nan`），最后返回 inner `JudgeResult`。
  - **必须保留 inner judge 完整 surface**：
    - `min_required_top_k` **属性**（`config.py:1264` `getattr(judges[cp_id], "min_required_top_k", 0)` 读它喂 search strategy）：DumpingJudge 实现为 `@property` 返回 `max(getattr(inner, "min_required_top_k", 0), max((ext.required_top_k for ext in self._dump_extractors), default=0))` —— 同时 cover inner CompositeJudge 的 hint 与 dump 侧（如配 F2 dump 需 K=5）需求
    - `on_episode_start(extra_metadata=None)` lifecycle **必转发，必走 filtered dispatch**（inner=AlwaysHit/Threshold/AlwaysWarmStart/CompositeJudge 4 个 `on_episode_start(self)` 无 kwargs，直 call `inner(**kw)` 会 TypeError）。DumpingJudge 实现：

      ```python
      def on_episode_start(self, *, extra_metadata=None):
          self._current_extra = dict(extra_metadata or {})
          # filtered forwarding — 同 Orchestrator._safe_call_lifecycle 同款
          # inspect.signature filter，inner 不接 extra_metadata 自动 swallow
          from openpi.cache.orchestrator import CacheOrchestrator
          CacheOrchestrator._safe_call_lifecycle(
              self._inner, "on_episode_start", extra_metadata=extra_metadata,
          )
      ```

      （或在 DumpingJudge 内复制一份 `_safe_call_lifecycle` 静态 helper，避免反向 import；具体放哪由实现期决定，但**必须**走 inspect.signature filter，禁止 `inner(**kw)` 直 call）
    - `record_action(action_chunk)` lifecycle **必转发**（`orchestrator.py:321` `judge.record_action(action_chunk)`）：DumpingJudge 实现 `def record_action(self, action_chunk): hook = getattr(self._inner, "record_action", None); hook is not None and hook(action_chunk)` —— 不缓存自己的 history。`record_action` 签名在 4 个现有 judge (`judge.py:135,180,227,357`) 全都是 `(self, action_chunk)` 一致，无 kwargs 失配风险，无需 filtered dispatch；如未来扩签名需对齐 `_safe_call_lifecycle` 路径。
    - **避免 inner 还有其它属性被 hidden**：用 `__getattr__` fallback `def __getattr__(self, name): return getattr(self._inner, name)` 把所有未显式 override 的 attribute access 透明转发给 inner（`__init__` 阶段需把 `self._inner` / `self._dump_extractors` / `self._dump_path` / `self._current_extra` 直接 set 到 instance dict，避免触发 `__getattr__` 递归）
  - **dump extractor 走同 capability flag injection**：`_build_judge` 在构造 dump 段 OnlineExtractor 时复用 composite 路径同款 builder：`requires_library_stats=True` 的 dump factor（如 F1a / F1b）自动注入 `library_stats=library_stats` kwarg；`requires_chain_walk=True` 的 dump factor 同 composite validator 强制 `backend.type=in_memory`。这一逻辑落在 `_build_judge` 的新 `_build_dump_extractors(dump_cfg, library_stats)` helper。
  - identity 字段拿法见 §4 Phase 0 中"identity 字段来源"段：DumpingJudge 通过 `on_episode_start(extra_metadata=...)` lifecycle 接收并 stash 到 `self._current_extra`；`__call__` 时取**两个 identity 字段** (`task_id` + `orig_init_state_idx`) 写入 JSONL：

    ```python
    row["task_id"] = self._current_extra.get("task_id")              # int or None
    row["orig_init_state_idx"] = self._current_extra.get("orig_init_state_idx")  # int or None
    ```

    缺失字段以 `None` (JSON `null`) 写入 — 上游 episode_start 未传 / 子集化时 main.py 端无对应 field / on_episode_end 后清空状态 都安全降级；analysis 阶段 `calibration_fit.py` 对 `null` identity 行 fail-loud（要求 100% 覆盖，§7.3 sanity check 强制）。**没有反向依赖 Orchestrator 的 layering 违规**。
  - unit test 覆盖（已落 §7.2）：AlwaysHit / Threshold / **CompositeJudge** 三种 inner 的等价性 + JSONL schema + extractor 抛错隔离 + identity 注入 + lifecycle (`on_episode_start` / `record_action`) 转发 + `min_required_top_k` 合并（inner hint + dump hint 取 max）+ `__getattr__` fallback 不递归。
- **`src/openpi/cache/config.py`**：
  - `_build_composer` 的 `and` / `or` 分支补传 `directions=cfg.directions`（与 `weighted_sum` 同款），并补 `tests/cache/components/factors/` 一组 unit test cover non_monotonic key + AND/OR + directions 绑定。这是 Phase 3 prerequisite。
  - `JudgeConfig` schema 加可选字段 `dump: {path: str, config_id: str, factors: list[FactorConfig]}`：`config_id` 必填（server `load_cache_config` 路径不接收 yaml 文件名，必须 yaml 自己 declare）；命名空间须与 `cache_eval_results.json.config_id`（runner `run_cache_experiments.py:414` `run_id=yf.stem`）一致 — 该 invariant 由 `generate_yamls.py` 在 yaml 生成期强制（`assert dump.config_id == yaml stem`），不要求 `validate_cache_config` 持有 yaml 路径。`_build_judge` 检测到 `dump` 字段时：先 build inner judge → 用 `factors` list 实例化 OnlineExtractor（走 `_build_dump_extractors` helper 同 capability flag injection）→ 包 `DumpingJudge(inner, extractors, path, config_id, identity_provider)`；validator 检查 `dump.path` 父目录存在 / `dump.config_id` 非空 / `dump.factors` 与现有 factor registry 验证 / capability vs backend 同 composite 校验。
- **episode_start `extra_metadata` 通道扩展** — 改动点：
  - `packages/openpi-client/src/openpi_client/websocket_client_policy.py:56-80` `episode_start` 加 `extra_metadata: dict | None = None` kwarg + wire 序列化字段 `__extra__`（空 dict 省略保 wire backward compat）
  - `examples/libero/main.py` 两处 `client.episode_start(...)` 加 `extra_metadata={"task_id": task_id, "orig_init_state_idx": orig_init_state_idx}`（必含 `task_id` int）
  - `src/openpi/serving/websocket_policy_server.py:212-222` `conn_policy.on_episode_start(...)` keyword 调用扩 1 字段：`extra_metadata=obs.get("__extra__", {})`
  - `src/openpi/policies/policy.py:253-261` `Policy.on_episode_start` 加 `extra_metadata: dict | None = None` kwarg + 透传 inner
  - `src/openpi/collect/collection_policy.py:115-125` `CollectionPolicy.on_episode_start` 加 `extra_metadata: dict | None = None` kwarg（不用，仅 forward）
  - `src/openpi/collect/data_collector.py:48` `DataCollector.on_episode_start` 同款（保所有 conn_policy 类型对扩字段 keyword call 不报错）
  - `src/openpi/cache/interceptor.py:279-301` `InferenceInterceptor.on_episode_start` 加 `extra_metadata: dict | None = None` 参数 + forward 给 Orchestrator
  - `src/openpi/cache/orchestrator.py:219` `Orchestrator.on_episode_start(task_key, episode_id, extra_metadata=None)` + `self._current_episode_extra: dict` 字段 + `on_episode_end` 清空
  - `_broadcast_episode_start()` 调 `self._safe_call_lifecycle(judge, "on_episode_start", extra_metadata=self._current_episode_extra)` — 保留 filtered dispatch；现有 4 judge `on_episode_start(self)` 无 kwargs 自动被 `inspect.signature` filter swallow，DumpingJudge 显式接收。**gate / strategy / key_builder / normalizer 等 lifecycle 路径完全不动**
  - 测试 `tests/cache/test_episode_extra_propagation.py` end-to-end smoke：mock client 发 `extra_metadata={"task_id": 3, "orig_init_state_idx": 7}` → assert `Orchestrator._current_episode_extra == {"task_id": 3, "orig_init_state_idx": 7}` after `_broadcast_episode_start()`；现有 `AlwaysHitJudge` / `CompositeJudge` 收 `_safe_call_lifecycle` 不报错；下一 episode 重置为空
- `src/openpi/cache/components/factors/composers/__init__.py`：若 Phase 3 决策触发，新增 `HierarchicalComposer` 与 `TwoStageComposer`，注册到 `_build_composer`。
- `src/openpi/cache/components/factors/normalizers/__init__.py`：若 Phase 4 决策触发，新增 `PreloadedPercentileNormalizer`（用 build pkl 时已知分布预填 buffer）。

### 6.3 不动

- `src/openpi/cache/storage_types.py` — 框架已支持本 plan 全部需求。
- `src/openpi/cache/orchestrator.py`：除 §6.2 列的 `Orchestrator.on_episode_start(..., extra_metadata=None)` 签名扩 + `self._current_episode_extra` 字段 + `on_episode_end` 清空 + `_broadcast_episode_start()` 改用 `_safe_call_lifecycle(..., extra_metadata=...)` 外不动；`_safe_call_lifecycle` 实现自身不变。
- `src/openpi/cache/components/judge.py` — 仅 §6.2 新增 `DumpingJudge` 类，原有 `AlwaysHitJudge` / `AlwaysWarmStartJudge` / `ThresholdJudge` / `CompositeJudge` 不动。
- `src/openpi/cache/config.py`：除 §6.2 directions 绑定修复 + JudgeConfig.dump 字段 + DumpingJudge wrap 分支 + 新 composer / normalizer 注册外不动。
- `scripts/serve_policy.py:267-277` `CacheOrchestrator` 构造段不动 — DumpingJudge 通过 `_build_judge` 包装，serve_policy.py 透明无感。

---

## 7. Test 策略

### 7.1 配置层

- 每个 yaml 在 push 前过 `validate_cache_config`（已有 7 项 composite 静态校验）。
- `JudgeConfig.dump` 字段加 schema parse + validator test（dump.path 父目录存在 / dump.config_id 非空 / dump.factors registry 校验 / capability vs backend 同 composite 5/6 项校验）。
- 新 composer / normalizer 加 unit test 到 `tests/cache/components/factors/`。
- `_build_composer` 的 `and` / `or` 分支补 `directions` 后加 unit test cover non_monotonic key 绑定（Phase 3 prerequisite）。

### 7.2 单元层

- **DumpingJudge wrapper test**（`tests/cache/components/test_dumping_judge.py`）：
  - **等价性 ×3 inner**：dump 开启 vs 关闭，inner judge 输出 `JudgeResult` 字节级一致 — 分别用 `AlwaysHitJudge` / `ThresholdJudge` / **`CompositeJudge`**（防 wrapper 隐藏 inner 行为）
  - **JSONL schema**：每行含 `config_id, task_id, orig_init_state_idx, step_idx, winner_id, cp1_score, factor_raw, factor_nan` 8 字段，类型对
  - **抛错隔离**：单个 OnlineExtractor `extract` 抛错时 dump path 吞掉异常 + 写 nan，verdict 仍正常返回
  - **identity 注入（双字段）**：通过 `on_episode_start(extra_metadata={"task_id": 3, "orig_init_state_idx": 7})` 喂的 identity → JSONL 行 `task_id == 3` (int) **且** `orig_init_state_idx == 7` (int) 同时正确出现；下一 episode 调 `on_episode_start(extra_metadata={})` 重置后两字段均为 `null`；只传一字段（`extra_metadata={"task_id": 3}`）时另一字段为 `null` 不报错
  - **lifecycle 转发**：包 `CompositeJudge` 时调 `wrapper.on_episode_start(...)` 后 inner CompositeJudge 的 `_normalizer.on_episode_start` 已被调（用 mock 验调用）；调 `wrapper.record_action(chunk)` 同样转发到 inner
  - **`min_required_top_k` 合并**：包含 F2(K=5) dump extractor + 包 inner `CompositeJudge`（含 F2(K=3) extractor）时 `wrapper.min_required_top_k == 5`（取 max）；inner 是 `AlwaysHitJudge`（无属性）+ dump 含 F2(K=5) 时 `wrapper.min_required_top_k == 5`；inner 是 `AlwaysHitJudge` + 无 dump factor 时 `wrapper.min_required_top_k == 0`
  - **`__getattr__` fallback 不递归**：`wrapper._inner` / `wrapper._dump_extractors` 访问不触发 `__getattr__`；不存在的 attribute 透传到 inner（用 mock inner 加自定义属性验）
- **Runner `--episode-filter` 透传 test**（`tests/exp/common/test_run_cache_experiments_filter.py`）：
  - mock `subprocess.Popen`，断言 main.py 命令行含 `--episode-filter <path>` 当 runner 收到对应 flag
  - 100-ep `init_subset.json` × 10 task × `orig_init_state_idx ∈ {0..9}` 文件结构 round-trip
- **Phase 0 dump-mode `top_k=5` 行为 test**（`tests/cache/test_dump_top_k_widening.py`）：
  - yaml `search_strategy.top_k=5` + `judge.type=always_hit` + `judge.dump.factors=[f2]` 构造的 `Orchestrator.check()` 在 mock backend (≥5 entries) 下 `len(results) == 5`，F2 dump 行 `factor_raw["f2_var"]` 非 NaN
  - yaml `search_strategy.top_k=1` + 同 judge + 同 factors，F2 dump 行 `factor_raw["f2_var"]` 是 NaN（控制组）
  - `judge.type=always_hit` + dump factors 含 F2(K=5)，wrapper.min_required_top_k 反向喂 search strategy → effective_top_k = max(yaml_top_k, 5)（避免 yaml 漏写 top_k=5 时仍能 dump）
- **episode_start `extra_metadata` 透传 test**（`tests/cache/test_episode_extra_propagation.py`）：
  - mock client 调 `episode_start(extra_metadata={"task_id": 3, "orig_init_state_idx": 7})` → assert wire 含 `__extra__: {"task_id": 3, "orig_init_state_idx": 7}` → server `InferenceInterceptor.on_episode_start` 收到 `extra_metadata` kwarg → `Orchestrator._current_episode_extra == {"task_id": 3, "orig_init_state_idx": 7}` after `_broadcast_episode_start()` — **assert 含 task_id 与 orig_init_state_idx 两字段**
  - JSONL 写出后 assert 行内 `config_id` / `task_id` / `orig_init_state_idx` 三字段都正确（task_id 是 int 而不是 task description string）
  - **lifecycle filtered dispatch**：assert `_safe_call_lifecycle(AlwaysHitJudge(), "on_episode_start", extra_metadata={...})` 不报错（`inspect.signature` 自动 filter `extra_metadata`）；同款测 `CompositeJudge` / `AlwaysWarmStartJudge` / `ThresholdJudge`；assert `DumpingJudge(inner=AlwaysHitJudge()).on_episode_start(extra_metadata={"task_id":3,"orig_init_state_idx":7})` 后 `_current_extra` 含两字段
  - 下一 `episode_start` (无 extra) 后 `_current_episode_extra == {}`
  - 旧 client 不传 `extra_metadata` 时全链路向后兼容（`InferenceInterceptor.on_episode_start` 无 kwarg 调 → `_current_episode_extra == {}` 无报错）
- **`dump.config_id` source test**（`tests/cache/test_dump_config_id.py`）：
  - assert `JudgeConfig.dump.config_id` 必填，缺失时 `validate_cache_config` raise `ConfigValidationError`
  - assert `dump.config_id == yaml stem` 由 `generate_yamls.py` 在生成期强制（test 在 `tests/exp/verdict_factor_judge/test_generate_yamls.py` 验：手改 yaml 内 `dump.config_id` 与 stem 不一致时 generator dry-run check raise）
  - assert DumpingJudge 写出的 JSONL 行 `config_id` 字段 == yaml `dump.config_id` == yaml stem（与 `Args.task_suite_name` / `task` string 等其它 yaml 字段无关）
  - **跨 cfg sanity**：3 个 cfg 子目录各自 cfg-prefixed yaml stem 不同 — `clip_w7_d4_phase0_always_hit_dump.yaml` / `max_pool_w3_d5_phase0_always_hit_dump.yaml` / `spatial16_w8_d4_phase0_always_hit_dump.yaml`，每份 yaml 内 `dump.config_id == yaml stem`；写出 3 份 JSONL 的 `config_id` 字段分别 `*_phase0_always_hit_dump`；模拟 join `cache_eval_results.json`（runner `run_id=yf.stem` 决定的 `config_id` 字段）应**精确字符串相等命中** row（不是部分匹配 / cfg substring 匹配）

### 7.3 集成层

- Phase 0 跑完后 sanity check：`AlwaysHit` 的 verdict 数 = total cycle 数 - cold-start MISS 数（应为 0，因为 AlwaysHit 不走 normalizer）。
- Phase 0 dump JSONL 行数 sanity check：行数 = 总 verdict cycle 数；`(config_id, task_id, orig_init_state_idx)` 三键对 join 到 `cache_eval_results.json` 应 100% 覆盖（与 Phase 5 join key 一致；显式 verify config_id 命名一致性，否则 Phase 5 join 会静默丢数据）。
- 每 Phase 完成后产出 1-page analysis md 到 `exp/verdict_factor_judge/analysis/<phase>_summary.md`，含：success / inference_time_saved 表 + 决策点结论（含 Wilson CI / McNemar p）+ 进 / 退下一 Phase 的判定。

### 7.4 不引入

- 不引入新的 pytest marker；已有 `@pytest.mark.manual` 已覆盖需要 GPU 的端到端 test。
- §7.2 列出的 unit test 全部走 `pytest`，不需 GPU / server。

---

## 8. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| AlwaysHit success_rate ≈ pure inference → verdict 边际收益小 | **不会**（已知 spatial16 always_hit 0.692 vs pure 0.984，30pp gap） | — | Phase 0 直接 join 现有数据消解此风险 |
| Cold-start force_miss 占比过高（>30%）| 中 | 中 | Phase 4 触发 N-PRELOAD 实现；或 Phase 4 选 N-PCT-LEN |
| Calibration data 标签噪声大（episode-level outcome 难归因到 single verdict）| 高 | 中 | Phase 5 备选标 (距 episode end 步数加权 / step-level critic)；若 S-CALIB 学不到清晰边界 → 退回 S-GRID winner |
| 跨 keybuilder / 跨任务严重劣化 | 中 | 中 | Phase 1–4 各 phase 决策点（cross-cfg 一致性内嵌）+ Phase 7 cross-task 决策点：若任一处劣化 > 阈值，本 plan 完结，记录 cross-domain gap 列后续 plan |
| 新 composer / normalizer 实现工作量超预算 | 低 | 中 | C-HIER / C-2STAGE / N-PRELOAD 仅在前 Phase 决策触发时实施，不预实现 |
| F1b artifact 不带 `library_stats` 老 pkl 兼容性 | 低 | 低 | `InMemoryBackend.load_artifact` 已有 fallback；Phase 0 验证一次即可 |
| F1b 长窗口在 entry 链两端 NaN（实测：T median=21，`(5,5)`=48% NaN, `(7,7)`=67% NaN），winner 命中 NaN entry 时 verdict 被动退化为 MISS，污染 ablation | **高（实测确认，已是常态）** | 中 | §3.3b B-1 Phase 0 winner-conditional NaN% instrumentation 强制；B-2 yaml 规则 low-NaN 配对；B-3 `(7,7)` 全禁、`(5,5)` 仅 Phase 4 ablation；C-AND 仅配 W-SHORT；W-MIX 默认从 v3 的 `(0,5)(3,3)(5,5)` 改为 v4 的 `(0,3)(1,1)(3,0)(0,5)(5,0)` |
| `DumpingJudge` wrapper 改变 inner judge 行为 | 低 | 低 | YAML-driven server-side wrapping（`judge.dump:{path,config_id,factors}` → `_build_judge` 自动包）；wrapper `__call__` 先 forward inner 取 `JudgeResult`，再用同 args 喂 dump extractor 写 JSONL，最后 return inner result（透明）；`__getattr__` fallback 保留 inner attribute；`min_required_top_k` / `on_episode_start` / `record_action` 显式 forward。§7.2 unit test 覆盖 AlwaysHit + Threshold + CompositeJudge 三种 inner 的等价性 + lifecycle 转发 + `min_required_top_k` 合并 |
| Stage 3 占 inference 50% 估算偏差 → §2.1 `inference_time_saved_ratio` 失真 | 中 | 低（不影响 sweep 排序，只影响绝对值）| Phase 0 用 `SystemTimer` probe 实测，事后校正系数；sweep 期间相对排序稳定 |

---

## 9. Open Questions（设计决策点，待 Phase 0 / 1 数据后决）

- **Q-A**：F1a-T 的 chain walk 在 LIBERO state 信号质量上是否有用？Phase 1 的 `f1a_t only` 行直接给答案。
- **Q-B**：`warm_start_t` 的"最优值"是否依赖 keybuilder？Phase 0 已有的 3 cfg × 3 start_t Ceiling-W baseline + 各 phase ×3 cfg 跑出来的 warm_start 命中率 / 平均 start_t 一起回答。
- **Q-C**：calibration 的标签该用 episode-level 还是 cycle-level？Phase 0 数据 + Phase 5 的 ablation 决定。
- **Q-D**：F1b 窗口集如果选 W-FUT-L 远胜 W-PAST，说明 candidate 内禀质量信号偏向"未来段"——这给 chain-walk 设计反向 hint，是否值得后续 plan？
- **Q-E**：Composer 的 NaN 处理在 LIBERO 边界（episode 早期）会不会主导 MISS rate？Phase 0 verdict-factor JSONL 给数据后再决定是否需要"NaN 容忍"模式。
- **Q-F**：Tier 的 multi-WARM_START 真的有用吗？还是单 WARM_START tier 已足够？Phase 4 给答案。
- **Q-G**：3 套 cfg 的因子值分布是否一致？Phase 0 calibration JSONL 比对决定 Phase 1–4 是否收紧到 1 cfg 跑（§4 Phase 1–4 cfg 数说明）。
- **Q-H**：Stage 3 占整 inference 50% 这个估算够准吗？Phase 0 后用 `SystemTimer` probe 实测 stage-wise 分布并校正 §2.1 的 0.5 系数。
- **Q-I**：3 cfg 的"最优 weights"如果各不相同（Phase 5 per-cfg vs xcfg 对比给答案），上线策略是 per-cfg yaml 还是 xcfg averaged yaml？per-cfg 的代价是 calibration 不可移植到新 keybuilder。
- **Q-J**：Phase 0 NaN% instrumentation 出来后，长窗口（`(5,5)` `(7,7)`）在 winner 上的 NaN% 若 > 8% 是否值得后续 plan 改 OfflineWriter 边界算法（截尾窗口估计）？还是直接砍长窗口、让短窗口承担 F1b 信号？取决于 Phase 1 / 4 是否证实"长窗口能区分而短窗口不能"。

---

## 10. 工作流挂钩

- Plan G1 已 APPROVED（Post-G1 polish 完成，§ 3.1 Step 2 Review Log 已删）。下一步进 §4 Code，按 §6.2 列出的 server-side `DumpingJudge` + episode_extra propagation + `_build_composer` directions 修复 + runner `--episode-filter` 透传等改动落地，再进 §5 G2。
- Phase 0 yaml 由 `exp/verdict_factor_judge/generate_yamls.py` 生成（强制 `dump.config_id == yaml stem` invariant）→ runner 跑 3 cfg × 100 ep AlwaysHit + dump → analysis 算 winner-conditional NaN% + paired baseline subset → 决策是否收紧 Phase 1–4 cfg 数。
- Phase 1+ yaml spec 在前 phase 决策点取舍后写（§4.0）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-04-27 12:55 CDT

- [Blocking] [Concern] The `episode_start.extra_metadata` signature expansion is not reflected in existing lifecycle tests, so the G2 test subset is currently red — reasoning: the implemented chain correctly adds `extra_metadata=None` to `PolicyRecorder.on_episode_start` and `CollectionPolicy.on_episode_start`, but `tests/serving/test_policy_recorder_lifecycle.py` and `tests/collect/test_collection_policy.py` still assert the old 4-keyword call shape and old explicit signature. The review command `PYTHONPATH=. uv run pytest tests/cache/components/test_dumping_judge.py tests/cache/test_dump_config.py tests/cache/test_episode_extra_propagation.py tests/cache/test_config_factor.py tests/exp/common/test_run_cache_experiments_filter.py tests/exp/verdict_factor_judge/test_generate_yamls.py tests/serving/test_policy_recorder_lifecycle.py tests/collect/test_collection_policy.py tests/collect/test_data_collector.py -q` produced 109 passed / 4 failed. Update the legacy lifecycle assertions/signature checks to the approved 5-field contract, including the default `extra_metadata=None` path and at least one non-empty metadata forwarding assertion.
- [Non-blocking] [Suggestion] Add one integration-level assertion for dump-side top-k widening through `build_cache_components`, not only the `DumpingJudge.min_required_top_k` property — reasoning: the implementation exposes the wrapper property and `build_cache_components` forwards `getattr(judge, "min_required_top_k", 0)` into `_build_search_strategy`, which matches the approved design. A small config-builder test with `judge.type=always_hit` + `judge.dump.factors=[f2 K=5]` would lock the Phase 0 calibration path against future regressions where the wrapper still reports K=5 but the built search strategy silently keeps `top_k=1`.

### G2 Round 1 — Executor — 2026-04-27

按 §10.2 / §10.3 验真后修。reviewer 跑的 G2 subset 给出 109/4 失败，全部源于 `episode_start.extra_metadata` 5-字段签名扩展未同步到既有 lifecycle 测试。2 项全部 Accepted。

- **Item 1 (Blocking, legacy lifecycle 测试断言旧 4-keyword shape)** — Accepted — 3 处既有测试更新到 5-字段 contract：
  - `tests/serving/test_policy_recorder_lifecycle.py`：`test_on_episode_start_forwards_kwargs` 期望 inner 调用含 `extra_metadata=None`；新加 `test_on_episode_start_forwards_extra_metadata` 验证非空 metadata 透传到 inner；签名锁定 test 从 4 字段 `["experiment", "task", "episode_id", "episode_name"]` 改为 5 字段加 `"extra_metadata"`，仍禁 *args/**kwargs（保留显式签名约束）。
  - `tests/collect/test_collection_policy.py`：`test_on_episode_start_forwards_to_collector_and_inner_policy` 期望 collector + inner 调用都含 `extra_metadata=None`；新加 `test_on_episode_start_forwards_extra_metadata` 验证非空 metadata 同时透传给 collector 与 inner；`test_on_episode_start_default_episode_name_is_empty_string` 同步加 `extra_metadata=None`。
  - `tests/collect/test_data_collector.py`：未触发改动（已存在测试不 assert 5-字段签名，DataCollector 自身用 `extra_metadata: dict | None = None  # noqa: ARG002` 接受但不用）。
  - reviewer 完整 G2 subset 命令 `PYTHONPATH=. uv run pytest tests/cache/components/test_dumping_judge.py tests/cache/test_dump_config.py tests/cache/test_episode_extra_propagation.py tests/cache/test_config_factor.py tests/exp/common/test_run_cache_experiments_filter.py tests/exp/verdict_factor_judge/test_generate_yamls.py tests/serving/test_policy_recorder_lifecycle.py tests/collect/test_collection_policy.py tests/collect/test_data_collector.py -q` 现 117 passed / 0 failed（之前 109/4）。
- **Item 2 (Non-blocking, dump-side top-k widening integration assertion)** — Accepted — `tests/cache/test_dump_config.py` 加 2 cases：
  - `test_build_cache_components_widens_top_k_for_dump_factors`：`AlwaysHit + dump.factors=[f2 K=5] + yaml top_k=1` → `judge.min_required_top_k == 5` **且** built search strategy `_top_k == 5`（验整条 Phase 0 calibration 路径，锁住"wrapper 报 5 但 strategy 仍 1"的隐藏失配场景）。
  - `test_build_cache_components_no_dump_keeps_yaml_top_k`（控制组）：`AlwaysHit` 无 dump → strategy `_top_k == 1`（确保 widening 不是无条件的）。

**未发现新 blocking issue**。Code 实现 + 既有测试同步完成；G2 subset 全 pass。

### G2 Round 2 — Reviewer — APPROVED — 2026-04-27 13:01 CDT

- [Decision] G2 approved. Round 1's blocking test-contract issue is closed: `PolicyRecorder` and `CollectionPolicy` lifecycle tests now assert the approved five-field `on_episode_start(..., extra_metadata=None)` default path and include non-empty `extra_metadata` forwarding cases. This matches the implemented wire/wrapper/orchestrator chain and the approved G1 identity propagation design.
- [Decision] The non-blocking top-k widening suggestion was addressed with useful coverage: `AlwaysHit + judge.dump.factors=[f2 K=5] + yaml top_k=1` now verifies both `DumpingJudge.min_required_top_k == 5` and the built search strategy effective `_top_k == 5`, with a no-dump control preserving `_top_k == 1`.
- [Verification] Reviewer reran the same G2 subset: `PYTHONPATH=. uv run pytest tests/cache/components/test_dumping_judge.py tests/cache/test_dump_config.py tests/cache/test_episode_extra_propagation.py tests/cache/test_config_factor.py tests/exp/common/test_run_cache_experiments_filter.py tests/exp/verdict_factor_judge/test_generate_yamls.py tests/serving/test_policy_recorder_lifecycle.py tests/collect/test_collection_policy.py tests/collect/test_data_collector.py -q` → 117 passed, 3 warnings.
- [Residual] Full repository verification remains a separate Verify-stage responsibility under the Working Agreement; G2 judgment is based on the reviewed diff and targeted G2 subset.
