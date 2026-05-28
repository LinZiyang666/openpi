# Weighted-Sum 阈值控制 FULL_HIT/WARM_START/MISS — SR × inference_ratio 帕累托

> **Status**: `Plan`（G1 owner-APPROVED 2026-05-27 R2 / §4 Code 进行中 / 运行待 wsweep winner + owner 下令）
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-05-27
> **关联**: 续 weighted_sum 系列。前序 always_hit 纯检索给出 (inf_ratio≈0, SR≈74%)；本实验用**检索总分阈值**自适应分配算力，扫出 SR×inference_ratio 帕累托。方法学移植 verdict factor judge Phase 3 的 data-driven threshold sweep（`exp/verdict_factor_judge/phase3/`），但阈值打在 weighted_sum 的**最终聚合分**上。

---

## 1. 动机

weighted_sum 的 `always_hit` 是「全 FULL_HIT、零推理、SR≈74%」一个点；不用 cache（全 MISS）则是「inf_ratio=1、原始 pi05 SR（>74%）」另一个点。中间地带 = **自适应算力**：对高置信请求重放（FULL_HIT，省算力）、低置信请求才真推理（MISS，保 SR）、中间用 WARM_START（省一半剩余推理）。本实验量化这条 **SR vs inference_ratio 帕累托前沿**，找「SR 崩塌前 inference_ratio 能压多低」的拐点。

## 2. 机制（全部现成，src 零改）

### 2.1 判分信号 = 最终聚合总分

阈值打在 `results[0].score`，对 `weighted_score_sum_knn` 即 **模态聚合 + trajectory 加权后的 [0,1] 总分**（不是单模态/单 step 分）。该分每请求经 `__hit_meta__.cp1_score`（`interceptor.py:492 = cp1_result.score`）暴露——**且与 judge verdict 无关**：gate=always_search 下 search 照常算出 top-1 score，即便 judge 判 MISS，`cp1_score` 仍为该 search 分。故 warmup 阶段即使强制 MISS 也能采到分布。

### 2.2 ThresholdJudge（`judge.py:199-237`）

```
score ≥ T_fh                  → FULL_HIT        (0 推理)
T_fh > score ≥ T_ws           → WARM_START@0.5  (部分推理)
score < T_ws                  → MISS            (全推理)
```

- **两个自由阈值** `T_fh`（full_hit 切点，亦即 warm 上界）、`T_ws`（warm 下界）。「warm_start 上下界」= `[T_ws, T_fh)`。
- `start_t` 固定 **0.5**。
- YAML 键：`judge:{type:threshold, threshold:T_fh, warm_tiers:[{threshold:T_ws, start_t:0.5}]}`。⚠ `cp1_threshold` 仅是 `ThresholdJudge` 的 Python 构造参数（`_build_inner_judge` 把 `JudgeConfig.threshold`→`ThresholdJudge(cp1_threshold=...)`）；YAML 未知键被 `_dict_to_dataclass` 静默忽略，故**必须**用 `threshold`/`warm_tiers`，不能写 `cp1_threshold`。
- `validate_cache_config`（config.py §1644-1666）要求 `warm_tiers[i].threshold` **严格 <** `judge.threshold`，warm_tiers 仅 CP1。

### 2.3 inference_ratio 公式（沿用 verdict）

```
FULL_HIT→0 ; WARM_START@t→1−0.5·(1−t) ; MISS→1
inf_ratio = (0·n_FH + 0.75·n_WS + 1·n_MISS) / N        (t=0.5 → warm cost 0.75)
```

## 3. 实验设计（Phase A→D，复用 verdict phase3 流程）

### Phase A — warmup 收集总分分布（沿真实 policy 轨迹）
- **base yaml**：weighted_sum 系列最优配置。**具体选哪个/几个 = 等当前 wsweep 跑完用其 winner 定**（owner 定；个数后定）。占位先用 spatial_16 top-1（v0@6 v1@44 rs@50, 74%）。
- warmup judge = **`threshold` judge 设 `threshold=2.0`、无 warm_tiers** → 强制全 MISS：search 照常算出 `cp1_score` 并经 `--per-step-out` 落盘，但 judge 返 MISS → robot 走**真实 policy 推理轨迹**（非 cache 重放轨迹），分布采在「阈值要 gate 的那条真实轨迹」上，消除重放偏置。held-out init，跑 W 个 ep。
- **前置门（最大风险）**：先看分布**展度**。weighted_score_sum 经 zscore(tanh)+加权和后可能聚得很窄（方差小）→ 阈值分不开三档。展度不足则该信号不适合做阈值（需换信号/换归一化），**实验在此 fail-fast 不浪费 eval**。

### Phase B — 反解阈值
- 每个目标 (fh_ratio, ws_ratio)：`T_fh = quantile(scores, 1−fh_ratio)`、`T_ws = quantile(scores, 1−fh_ratio−ws_ratio)`。复用 `exp/verdict_factor_judge/phase3/threshold_solver.py`（简化：weighted_sum 分已是最终分，无 calibration 层）。
- **退化 cell 处理**：窄/量化分布可能使两分位相等（`T_ws ≥ T_fh`），违反 §2.2 严格序。规则：**emit 前若 `T_fh − T_ws < ε`（ε=1e-6）跳过该 cell + 记 warning**（不静默、不强 nudge）；故所有 emit 的 yaml 必过 `load_cache_config`。
- **阈值法两条都跑对照**：① 经验分位（主）；② **zscore 参数化**（用 warmup `μ,σ` 设 `T=μ+k·σ`），验证「简单 zscore 阈值」是否够用（因总分本就 zscore 系归一）。

### Phase C — eval 扫格
- emit eval yaml（§2.2 正确键），扫 (fh_ratio, ws_ratio) 网格 = phase3 16-cell（各 ∈ {0.2,0.3,0.4,0.5}），需要时加密。
- 每 cell held-out init 100 ep，记 SR + **实际** inf_ratio（按真实 FH/WS/MISS 计数，非目标值）。
- **per-step 落盘 + 聚合**：`episode_runner.py:46/136` 已把每 step `hit_type` 收进 `EpisodeResult.per_step_rows`；扩 `_hit_row` 同时带 `cp1_score`（warmup 用），`run_phase2 --per-step-out` 把 driver 的 `per_step_rows` 落 jsonl，新 `summarize_inf_ratio.py` 按 hit_type 计数算 inf_ratio。**零改 src/ 与 conductor 核心，仅加 exp 侧 per-step 落盘 + episode_runner 行扩字段**。

### Phase D — 帕累托
- (inf_ratio, SR) 平面，inf_ratio 从高扫到低，找前沿 + 拐点。复用 phase3 `plot_pareto`。两端锚点：always_hit (≈0, 74%) 与 **raw-policy 全 MISS**——后者用 **`threshold=2.0`（> [0,1] 上界，`score==1.0` 亦走 MISS，避免 `≥` 含等号误判 FULL_HIT）** 或 cache-disabled baseline，干净测 inf_ratio=1 的原始 pi05 SR 锚点。

## 4. 代码改动（零改 src/ 与 conductor 核心）

| 路径 | 改动 |
|---|---|
| `exp/weighted_sum/emit_threshold_yamls.py` | **新**：emit ① warmup yaml（threshold judge `threshold=2.0` 强制 MISS）② eval yaml（`threshold:T_fh + warm_tiers:[{threshold:T_ws, start_t:0.5}]`）。复用 `build_eval_config`，emit 即 `load_cache_config` 自检 |
| `exp/weighted_sum/solve_thresholds.py` | **新**：吃 warmup per-step jsonl（cp1_score），分位（复用 phase3 threshold_solver）+ zscore 变体反解 (T_fh,T_ws)；退化 cell（T_fh−T_ws<ε）跳过+warn |
| `exp/weighted_sum/run_phase2.py` | **改（exp 非 src）**：加 `--per-step-out` 落 driver `per_step_rows` jsonl |
| `examples/libero/episode_runner.py` | **改（exp 侧执行层）**：`_hit_row` 增 `cp1_score` 字段（已收 hit_type，加一字段）|
| `exp/weighted_sum/summarize_inf_ratio.py` | **新**：per-step jsonl 按 hit_type 计 inf_ratio（FH 0 / WS@0.5 0.75 / MISS 1）|
| `exp/weighted_sum/analysis/plot_threshold_pareto.py` | **新**：(inf_ratio, SR) 帕累托（移植 phase3 plot_pareto）|
| 复用零改 | `ThresholdJudge`（src）、conductor 核心 / server、phase3 threshold_solver |

## 4b. 测试策略

- **emit 自检**：所有 warmup + eval yaml 过 `load_cache_config`（`judge.threshold` 生效、`warm_tiers[i].threshold` 严格 < `threshold`、warm_tiers 仅 CP1）。
- **solver 单测**：分位/zscore 边界——窄分布致 `T_ws≥T_fh` 必触发跳过；空/全等分布；`cp1_score` 为 `None`/缺失过滤；分位单调。
- **inf_ratio 聚合单测**：hit_type 计数→ratio（混合，WS@0.5=0.75；全 MISS=1.0；全 FH=0.0）。
- **§6 Verify**：`uv run pytest` 全绿（零改 src/ 应无回归）。

## 5. 锁定参数 / 待 wsweep 定

- **2-cut**（warm 上界 = T_fh）；3-cut（warm 上界独立 + gap）暂不做（需 ≤3 行 judge 变体，按需再议）。
- start_t = **0.5** 固定。
- 网格 = phase3 **16-cell**（(fh_ratio, ws_ratio) ∈ {0.2,0.3,0.4,0.5}²）起步，可加密。
- 阈值法：**分位 + zscore 两条对照**。
- **base yaml（哪个/几个）= 等当前 wsweep 跑完用 winner 定**（唯一遗留待定项，owner 拍板）。

## 6. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **总分判别力不足**（zscore 后聚太窄，阈值分不开三档） | Phase A 前置门先看分布展度，不足即 fail-fast；zscore 变体对照 |
| R2 | WARM_START 对 weighted_sum 是否真省算力又保 SR 未知 | inf_ratio 用模型化 0.75；Phase C 实测 warm cell 真实 SR；start_t 固定 0.5 先 |
| R3 | warmup 分布与 eval 分布漂移（held-out init 不同段） | warmup 与 eval 同源 held-out init；分位反解对小漂移鲁棒 |
| R4 | 原始 pi05 SR（inf_ratio=1 锚点）未在同机测过 | Phase C 含 all-MISS cell（`threshold=2.0`，含 score==1.0 亦 MISS）或 cache-disabled 直接测锚点 |
