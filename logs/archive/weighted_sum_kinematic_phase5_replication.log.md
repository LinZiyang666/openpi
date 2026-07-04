# Plan: weighted_sum × kinematic factor — phase5 237-cell replication with **single super warmup (offline-calibration mode)**

**Status**: **G1 APPROVED 2026-05-28 12:42 CDT** (Round 4 verdict; Rounds 1-3 NEEDS REVISION 各条已 Accepted) / §3.1 Post-G1 Polish 完成 / 进 §4 Code
**Level**: **L2**（多文件改动：新增 `exp/weighted_sum/kinematic/{spec,super_warmup,runner,strategy}.py` + 共享 `v2_spec.py` 加 cfg + ziyang10 server 重启加 `--warmup_dump_root` + `src/openpi/conductor/driver.py` +~14 行 driver-internal per_step flush [G1 R2 B1 mandated — strategy hook 签名不动以保持 5 个 override 兼容]）
**Date**: 2026-05-28
**Authority**: Execution
**Owner**: Ziyang Lin
**前序实验**: `logs/weighted_sum_threshold_pareto.log.md`、`logs/verdict_phase5_libero10_systematic_sweep.log.md`、`exp/verdict_factor_judge/analysis/phase5/results.md`

---

## 0. 一句话

在 weighted_sum 系列 **d1 best yaml**（`vision_0@6_vision_1@50_robot_state@43__d1`，SR=74%）的两层 `weighted_score_sum_knn` 检索之上，照搬 phase5 的 5-group × 237-cell（G5 grid 经 `fh+ws ≤ 0.9` 三角约束删退化 cell 后；phase5 原版 240）× kinematic factor 设计，**用 1 份 super warmup + offline calibration source 替代原 148 份 per-cell warmup + WarmupPool 链路**，eval 走 dual-server / 16+48 worker。判分用 `CompositeJudge` 的 `weighted_sum_zero_nan` composer + `tier_thresholds={full_hit: T_fh, warm_start: T_ws}`（注：本项目无 `ThresholdJudge` class，**判分入口是 composer 的 tier_thresholds 字段**）。

---

## 1. 目标与范围

### 1.1 目标
在 `libero_spatial` 上，把 verdict-phase5 的 (5 group × 48 cell × 100 ep = 24000 ep) 系统化 kinematic 扫描完整复刻一次，**但底层检索机制换成 weighted_sum 系列的两层 `weighted_score_sum_knn` (with score_normalizers)**，并使用 weighted_sum spatial16 上 SR 最高的 d1 配置作为 base。

### 1.2 主假设（**R1 R1B M5 补**）

| Hypothesis | 验证条件 | 失败回退 |
|---|---|---|
| **H1**: 在 d1 weighted_sum 检索（窄分布）下，G5 (FH, WS) ratio sweep 仍是最强信号轴（同 phase5 d4 rrf 结论） | G5 Pareto frontier 长度 ≥ 4 且最高 SR > 任何 G1-G4 cell | 若不成立：G5 在新检索下被某 G1-G4 dominate → 单独立结论值 |
| **H2**: G1-G4 在窄分布 d1 下**仍 inconclusive**（Δ ≤ 5pp）—— 重蹈 phase5 d4 覆辙 | 20 bucket 中 ≤ 2 个 decidable | 若 ≥ 3 decidable → 替代假设：检索几何切实改变 kinematic factor 信噪比 |
| **H3**: super warmup 抹平的 per-cell warmup-sampling noise 显著影响 G1-G4 decidability（vs phase5）| 与 phase5 同 cell 的 SR Δ 中位 < 2pp，但 bucket 内部 Δ rank 排序不同 | 仅在 H1+H2 都已知后判 |

### 1.3 与已完成实验的对照
| # | 实验 | 检索 | judge | 结论 |
|---|---|---|---|---|
| ① | trajectory_search | `weighted_score_sum_knn` d1/3/4/5 | `always_hit` | 救弱不救强 |
| ② | wsweep | 同 + 78 weight × 4 depth | `always_hit` | d1 ceiling=74% |
| ③ | threshold_pareto | 同 (4 best base) × 83 (FH,WS) | composite + `tier_thresholds` on **cp1_score** | SR 92% / inf 0.5 |
| ④ verdict phase5 | `weighted_rrf_knn` d4 | composite + tier_thresholds on **kinematic composer score** | G5 dominant; G1-G4 inconclusive |
| ⑤ **本 plan** | **`weighted_score_sum_knn` d1-best** | composite + tier_thresholds on **kinematic composer score** | 待跑 |

threshold_pareto 用 cp1_score 直接当判分；**本实验把 cp1_score 仅当检索质量，判分由 kinematic factor 决定**。两者底层检索完全同形，可在同一 Pareto 平面对比 (③ vs ⑤)。

### 1.4 不在范围内
- 任何 `src/openpi/cache/` 改动
- libero_object / libero_10
- weighted_sum 的 d3/d4/d5
- d1 best 3 选 1 中"换 yaml"扩展（仅 `@6_@50_@43_d1`；G1 可换）
- 重写 r/p baseline（直接复用 `exp/random_periodic_gate/analysis/aggregate.csv`，与检索无关，是 gate-level floor，安全复用）

### 1.5 半范围内：always-WARM ceiling（R1 R1B DESIGN-CRITICAL 修复）
phase5 results.md §5.4 复用的 always-WARM (0.30/0.50/0.70 → 0.942/0.952/0.976) 来自 **spatial16_w8_d4 / weighted_rrf_knn**，与本实验底层不一致。**本 plan 强制小型 always-WARM 跑量**：在 d1 base 上 emit 3 个 always-warm yaml × 100 ep = 300 ep ≈ +10 min on single server，作为本实验自己的 ceiling 锚点。

---

## 2. 前置事实（**代码级核实，每条都引用 file:line**）

### 2.1 d1 best yaml 三选一与分布形状（**R1 R1B DESIGN-MAJOR M7 + R1A MINOR 加注**）

`exp/weighted_sum/data/trajectory_wsweep/results.json` SR=74% 三个并列 d1：

```
A. cp1_spatial_pool_16__grid3_vision_0@62_vision_1@18_robot_state@18__d1   # vision-heavy
B. cp1_spatial_pool_16__grid3_vision_0@6_vision_1@43_robot_state@50__d1     # robot_state-heaviest
C. cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1     # robot_state 第二重 ← 本 plan
```

选 C 的理由：
- threshold_pareto d1 base 已用它跑过 83 cell（warmup raw 在 `exp/weighted_sum/data/threshold_pareto/warmup_split/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1.jsonl`，**已实测存在**），方便横向对比
- threshold_pareto §3.1 显示其 cp1_score 分布 std=0.023（窄但可用）

owner 若想换 A/B，只在 §1.1 line 192-226 swap CFG_SPECS 字段；C 与 A/B 的分布形状差异未量化，G1 可作 risk note。

### 2.2 检索 / 判决 / dump 三层解耦（**super warmup 可行性根因**）

`src/openpi/cache/orchestrator.py` + `interceptor.py` + `dumping_judge.py` 已核：

| Layer | 输入 | 输出 | 决定因素 |
|---|---|---|---|
| Retrieve | query_keys (vision_0/vision_1/robot_state) | `list[SearchResultLite]` with `.score` | `keys.{f}.weight` + `field_similarity` + `score_normalization` + `trajectory_depth` |
| Inner judge | results | JudgeResult | judge config |
| Dump (旁路) | results + view + history | per-step `factor_raw` jsonl row | DumpingJudge.dump_factors |

**dumping_judge.py:142-159**：

```python
def __call__(self, results, ...):
    judge_result = self._inner(...)           # inner 透传
    self._write_dump_row(results, view, history, judge_result)   # 旁路
    return judge_result
```

- inner = `always_warm_start(start_t=0.7)` 仅保证流水线全跑（gate=always_search）
- DumpingJudge 自带 `ZScoreNormalization(library_stats)`（`config.py:2310`），不依赖 inner
- factor.extract 之间相互独立（`dumping_judge.py:181-194` per-factor try/except）

### 2.3 ⚠ deferred dump **实际写盘**（R1 R1A MAJOR 修正）
- `dumping_judge.py:217-229` `_append_jsonl` lazy-open + write
- `websocket_policy_server.py:201-211` `_fill_deferred_dump_paths` 把 `dump.deferred=True` 解析成 `dump.path = warmup_dump_root / yaml_id.jsonl`
- `fetch_dump` (`websocket_policy_server.py:214-230`) 从那个磁盘文件读回客户端
- **所以 deferred=True 不是"内存模式"，是"path 延迟到 server side 决定"**
- ⚠ **server 必须以 `--warmup_dump_root <path>` 启动**（`scripts/serve_policy.py:174`），否则 fetch_dump 拒收

### 2.4 pkl 已含 phase5 用的 offline keys（pytest 实证，**R1A 核实**）

```
exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl
entries: 1018
payload.factors keys: 232
phase5-style offline_state keys: 32  (8 windows × 4 desc)
windows present: {p0_f3, p0_f5, p1_f1, p2_f2, p3_f3, p5_f5, p7_f7, p3_f0}
library_stats: present (LibraryStats)
```

→ phase5 G1-G5 offline declared windows 完全在 pkl 覆盖内（offline 只用 `{(0,3), (0,5), (3,3)}` ⊆ pkl）。

### 2.5 `(5,0)` 窗口仅 G2 online 用，**不需 enrich pkl**
- `phase5/spec.py:99` `(((0,5),(5,0)), "fut05+past50")` → online_action / online_state
- `online.py:151-199` `_OnlineActionBase._build_splice` 走 `history_actions + walk_next` 现场算，不查 pkl
- 但**前 max(P, F) step 的 history 不足 → factor_raw = NaN**：(5,0) 需要 history ≥ 5，每 episode 前 5 step NaN；(7,7) 需 history ≥ 7 且 walk_next ≥ 7（chain end fail），更稀疏

### 2.6 weighted_score_sum_knn d1 支持
- `search_strategy.py:336-380` `WeightedScoreSumKnnStrategy(TrajectoryMixin)`
- 默认 `trajectory_depth=1, trajectory_weights=None` → d1 yaml 不写 trajectory 字段
- 返回 list[SearchResultLite]，`results[0].score` = layer1 zscore tanh × layer2 weighted sum（与 weighted_rrf_knn 同接口）

### 2.7 calibration `samples_source.type=offline + format=jsonl` 完整支持（**R1 R1C BLOCKER 修复路径**）

`config.py:_load_calibration_samples_offline`:

```python
if off.format == "jsonl":
    per_key: dict[str, list[float]] = {}
    with path.open("r") as fh:
        for line in fh:
            row = json.loads(line)
            raw = row.get("factor_raw") or {}
            for k, v in raw.items():
                per_key.setdefault(k, []).append(float(v))
    return CalibrationSamples(per_key)
```

- JSONL 行格式 `{factor_raw: {key: v, ...}}` **与 super warmup raw 输出格式完全一致**
- 文件路径必须 **server 端可访问**（server 仓库根的相对路径）
- 每次 `build_per_connection_components` 重新读盘构造 `CalibrationSamples`（IO ~10MB / 解析 ~1s）— 比 WarmupPool 慢但**零内存累积、零驱逐风险**

**→ 本 plan 用 offline 路径替代 warmup 路径，完全绕开 WarmupPool LRU 问题**（R1 R1A B5 + R1C BLOCKER §3 解决方案）。

### 2.8 `PercentileRollingCalibration` "extras silently ignored"
- `percentile_rolling.py:31, 86` "samples.samples may carry more keys than bind_keys declares; extras are silently ignored. `for v in non_nan[-window_size:]: buf.append(v)`"
- **每 key 必须 ≥ 50 non-NaN samples**（line 79-82 fail-fast）→ super warmup 跑量必须保证这条对 (7,7) state 这种最稀疏 key 也成立

### 2.9 strategy hook 是正路（**R1 R1A B6 + R1C BLOCKER §1 修复路径**）
- `src/openpi/conductor/strategy.py:84-106` `ExperimentStrategy.on_stage_begin(stage, ctl, ctx)` 文档明确："eval stage: ctl.preload_normalizer_buffer(eval_id, ctx.get(calib_id)) THEN ctl.load_cache_config(eval_yaml)"
- `exp/weighted_sum/weight_search_strategy.py:84-89` `WeightSearchStrategy.on_stage_begin` 当前只 `ctl.load_cache_config(...)` — 单跳，不带 preload
- **本 plan**（R4 修订）：写 `KinematicSearchStrategy(WeightSearchStrategy)`，不 override `on_stage_begin / on_stage_complete`（继承基类）。原因：(a) offline calibration（§2.7）不需要任何 preload；(b) per_step 增量 flush 由 **driver 内部** 在 `_complete_stage` 末尾自动触发（G1 R2 B1 mandated），不通过 strategy hook。strategy 类只暴露 `_write_per_step(yaml_id, rows)` 作为 writer callable，由 `run_phase2.py` 注入 `ConductorDriver(per_step_writer=...)`。详见 §4.1。
- **`run_phase2.py` 完全不动**（R1A B6 + R1C 共识修正）

### 2.10 真实 declared_keys 并集 = 50 keys（**R1A BLOCKER §1 核实**）

Round 1 agent 实测 `set().union(*[c.declared_keys for c in p5.generate_all_cells()])` = **50 keys**：

- online: `ONLINE_DESCS = ("jerk", "dispersion")` × 2 channel × 8 unique windows = 32 keys
- offline: 4 desc × 2 channel × 3 unique windows = 24 keys，但 G3/G4 共用某些 window，实际并集 = 18 keys
- 总 50 keys（**不是 plan v1 的 77**；topk_action_variance 没有任何 cell 用）

⚠ G5 cells declared_keys 来自 `RECIPES_PHASE4["p1_state_fut_online_act"]` / `p2_action_fut_online_act` / `PHASE3_RECIPES["g6_..."]` —— 这些 declared_keys 必须**子集于** super_warmup factor list（Stage 3 verify check #5 是 ground truth）。

→ super warmup factor list **直接由 `set().union(*[c.declared_keys for c in generate_all_cells()])` 驱动**，不再手算（R1A BLOCKER §1 强制要求）。

### 2.11 per_step.jsonl 内存累积（**R1C BLOCKER §2 风险**）

- `src/openpi/conductor/driver.py:149, 250-252, 324-326` `_per_step_rows: list[dict] = []`
- `run_phase2.py:170-177` `driver_thread.join()` 后才写盘
- 237 cell × 100 ep × ~25 step ≈ 59 万行 × ~250 bytes ≈ 148 MB 常驻
- **3h 主跑中途任一 crash (expose drop / agent restart / pkill self-match / OOM) → 全失**
- session_handoff §15 #13 历史踩坑已确认

**→ 本 plan**（R4）：`ConductorDriver._complete_stage` 在 `strategy.on_stage_complete` 调用之后**driver 内部**自动 call `self._flush_per_step_for_stage(stage.yaml_id)`，flush 该 yaml 的 per_step_rows 到磁盘并从 driver 内存清出（§4.4）。strategy hook 签名 0 改动（G1 R2 B1 mandated — 保兼容既有 5 个 `on_stage_complete(stage, ctl, ctx)` override）。

### 2.12 G5 grid 中 fh+ws = 1.0 的 cell 退化（**G1 R1 Reviewer B1 修复 — 算术更正**）

`phase5/spec.py:173-175` 原 G5_THRESHOLD_GRID = 16 pairs。`derive_thresholds` (`phase3/threshold_solver.py:200-204`)：

- `i_ws = max(0, min(n-1, int((fh+ws) * n) - 1))`
- 当 `fh + ws == 1.0` → `i_ws = n - 1` → **T_ws = min(scores)**（bottom tier 塌缩）
- 当 `fh + ws < 1.0` → `i_ws < n - 1` → T_ws 在 sample 分布内部，OK

**实测算术** (`python3` verified, G1 R1 Reviewer B1)：

```python
grid_full = [(fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)]
filt = [(fh, ws) for (fh, ws) in grid_full if fh + ws <= 0.9 + 1e-9]
len(filt) == 15      # 仅 (0.5, 0.5) 被排除
# kept = [(0.2,0.2..0.5), (0.3,0.2..0.5), (0.4,0.2..0.5), (0.5,0.2..0.4)]
```

threshold_pareto §2.2 用 `max_total = 0.9`（含等号），本 plan 沿用：

- filter rule: `fh + ws ≤ 0.9` (inclusive)
- per recipe: **15 pairs** (not v2 的 11)
- 3 recipe (p1/p2/g6) × 15 = **45 G5 cell** (not v2 的 33)
- 总 cell: G1+G2+G3+G4 + G5 = 48×4 + 45 = **237** (not v2 的 225)

⚠ 边界情况 `fh+ws = 0.9`：T_fh = arr_desc[int(fh×n)−1] ≥ arr_desc[int(0.9×n)−1] = T_ws（降序数组前面 ≥ 后面），严格 T_fh ≥ T_ws。当 fh = 0.5, ws = 0.4 → T_fh = arr_desc[int(0.5n)−1], T_ws = arr_desc[int(0.9n)−1]；若 fh+ws < 1.0 则 i_fh < i_ws，T_fh > T_ws（严格大于）✓ ThresholdJudge 接受。

剩余 corner: 若 sample distribution 有 plateau 使 arr_desc[i_fh] == arr_desc[i_ws]，则 T_fh == T_ws，server 拒；§4.2 emit-eval try/except 兜底 skip（标 `__SKIP.json` reason="threshold tie"）。

### 2.13 trials_per_task 实测 verdict/episode 比（**R1B DESIGN-CRITICAL #1 修复**）

- threshold_pareto warmup `warmup_per_step.jsonl` = 8695 rows / 400 ep = **21.7 verdict/ep**（实测）
- phase5 per-cell warmup raw = 349 rows / 20 ep = 17.5 verdict/ep（实测）
- plan v1 写 "5 trials/task = 50 ep ≈ 5500 verdict" 错（≈ 1085 实际）

**→ 本 plan**：`trials_per_task = 15`（10 task × 15 = 150 ep × 21.7 ≈ 3250 verdict），保证最稀疏 key (7,7)_state 也 ≥ 50 finite（前 7 step NaN，150 ep × 14 finite ≈ 2100 finite，远超 50）。

跑时间估算：150 ep × ~5 s/ep（单 server 1 replica） ≈ 12-15 min。

---

## 3. 产物目录布局

```
exp/weighted_sum/kinematic/                       ← 新模块
  __init__.py
  spec.py                  # 237 cell 生成器（复用 phase5 G1-G4 + 本地 G5 generator with fh+ws<=0.9 filter）
  super_warmup.py          # super warmup yaml 构造器 + driver + 5-check verify
  strategy.py              # KinematicSearchStrategy(WeightSearchStrategy)
  runner.py                # 4-mode CLI 入口 (emit-warmup, run-warmup, verify-raw, emit-eval-yamls, run-eval, analyze)
  analysis/
    plot_pareto_overlay.py # 4-frontier overlay (r/p, threshold_pareto, this, phase5 d4)

exp/weighted_sum/config/kinematic_phase5/         ← 配置
  super_warmup.yaml        # 1 份
  eval/                    # 237 cell (G1+G2+G3+G4=192 + G5=45)
                           # ⚠ 不入 git（emit 产物，由 super raw + spec 重生）
  always_warm/             # 3 个 always-warm yaml（§1.5）

exp/weighted_sum/data/kinematic_phase5/           ← 数据（不入 git）
  super_warmup_raw.jsonl                          # ~3250 row super raw
  super_warmup_raw_dump.jsonl                     # 原始 fetch_dump 拉回
  per_step/<yaml_id>.jsonl                        # 增量落盘（§4.4）
  episode_results/                                # 237 cell × episode_results.json
  per_yaml_summary.jsonl                          # 237 row final
  thresholds/<yaml_id>.json                       # 237 cell threshold snapshot + bootstrap CI
  g{1..5}_decision.json
  always_warm_results.json                        # 3 cell × 100 ep ceiling
```

**入 git 边界**（artifact_layout.md §3 + R1D MINOR）：
- `spec.py / super_warmup.py / strategy.py / runner.py / analysis/*.py` 入库
- `super_warmup.yaml` 入库（source-of-truth）
- `eval/*.yaml` **不入库**（237 个 emit 产物，由 super raw + spec 确定性重生）
- `data/**` 不入库
- 跑完后写英文 `exp/weighted_sum/kinematic/analysis/results.md` 入库

---

## 4. 执行步骤

> **Stage 依赖（strict）**：0 (server 重启) → 1 (spec + cfg) → 2 (super warmup yaml + run + offline-mode tests) → 3 (5-check verify, **HARD GATE**) → 4 (emit 237 eval yamls + push to servers + smoke test) → 5 (run-eval) → 6 (always-WARM 3-cell) → 7 (analyze).

### Stage 0 — 环境前置（**R1D MAJOR §16.4 红线 + BLOCKER #4 修复**）

**0.1 喊 owner 释放显存 + 重启 ziyang10 server 加 `--warmup_dump_root`**

session_handoff §6 当前模板**没有** `--warmup_dump_root` 参数 → fetch_dump 会被 server 拒收。super warmup 必须能 fetch_dump，所以需要重启 ziyang10 server 一次。

agentchat 通知（heredoc + tether exec 顺序，安全 pkill 见 session_handoff §4 #5）：

```bash
# 1. 通知 owner
agentchat send 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --file - <<'EOF'
准备启动 phase5 kinematic on weighted_sum 实验。Stage 0：需要重启 ziyang10 server 加 --warmup_dump_root 给 super warmup 用。
GPU free 当前 9.5GB；起 1 replica 占 ~8GB 应可，但若 free 不够请释放。
约 30 秒内重启完成。
EOF

# 2. 等 owner ack 后重启
tether exec jupyter-ziyang10 -- bash -lc '
export HOME=/home/ziyang10
cd /home/ziyang10/openpi
mkdir -p /home/ziyang10/.warmup_dumps  # owner uid 700 enforce 由 serve_policy 自动处理
# 用 char-class + 拆 2 条命令规避 self-match
tmux kill-session -t srv0 2>/dev/null
fuser -k 8000/tcp 2>/dev/null
sleep 3
'
tether exec jupyter-ziyang10 -- bash -lc '
export HOME=/home/ziyang10
cd /home/ziyang10/openpi
CFG=exp/weighted_sum/config/kinematic_phase5/super_warmup.yaml  # 已 push
tmux new -s srv0 -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 1 --replica-spawn-batch 1 --port 8000 --warmup_dump_root /home/ziyang10/.warmup_dumps --cache_config $CFG policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/ws_kin_srv.log"
'
tether expose jupyter-ziyang10 --local 8000 --name ziyang-srv   # 复用旧 name，或换 kin-srv
```

⚠ tyro 顺序：`--warmup_dump_root` 是顶层参数，与 `--port`/`--cache_config` 同位，**必须在 `policy:checkpoint` 之前**（session_handoff §15 #1）。

**0.2 xuanle server 不重启**：xuanle 仅作 eval, eval 时**不需要** fetch_dump（eval yaml 是 offline calibration 模式，super raw 通过文件挂载读，不走 fetch_dump 链路）。维持现有 3-replica 不动。

**0.3 ziyang10 重启后 ready 探测**：参 session_handoff §6 watcher。

### Stage 1 — `kinematic/spec.py` + `v2_spec.py` 加 cfg

#### 1.1 `exp/verdict_factor_judge/common/v2_spec.py` 加 cfg（+25 行）

在 `CFG_SPECS` 字典追加：

```python
CFG_SPECS["spatial16_ws_d1_best"] = {
    "key_builder_type": "cp1_spatial_pool_16",
    "vector_dims": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
    # ↓ 从 wsweep d1 best yaml line 4-19 直拷
    "keys": {
        "vision_0":   {"enabled": True,  "weight": 0.0625},
        "vision_1":   {"enabled": True,  "weight": 0.5},
        "vision_2":   {"enabled": False, "weight": 0.0},
        "prompt_emb": {"enabled": False, "weight": 0.0},
        "robot_state": {"enabled": True, "weight": 0.4375},
    },
    "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
    # ↓ d1 best yaml 直拷，删 trajectory_*
    "search_strategy": {
        "type": "weighted_score_sum_knn",
        "top_k": 1,
        "step_filter": "all",
        "field_similarity": {
            "vision_0": {"type": "cosine"},
            "vision_1": {"type": "cosine"},
            "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
        },
        "score_normalization": {
            "type": "per_field",
            "fields": {
                "vision_0":   {"method": "zscore", "params": {"mu": 0.977693693699334, "sigma": 0.00699373921570407, "squash": "tanh"}},
                "vision_1":   {"method": "zscore", "params": {"mu": 0.9691840492031897, "sigma": 0.007853951498497307, "squash": "tanh"}},
                "robot_state": {"method": "zscore", "params": {"mu": -1.8439531429434792, "sigma": 1.0018373754826044, "squash": "tanh"}},
            },
        },
    },
}
```

⚠ **不要改 spatial16_w8_d4 cfg**（原 phase5 还在用）。phase5 G5 cell 通过 `RECIPES_PHASE4["..."]` 引用 phase4 recipes 的 declared_keys，phase4 spec 仍读 spatial16_w8_d4 — 这个引用关系**与 cfg_id 无关**（recipe 只是 declared_keys + factors list + weights，cfg_id 仅决定 keys/search_strategy/preload）。

`_FIELD_SIM_DEFAULT` / `_KEYS_DEFAULT` 等 module-level 常量仅在不显式提供时作默认，本 cfg 全字段显式声明，不受其变化影响。

#### 1.2 `exp/weighted_sum/kinematic/spec.py`（~150 行，**G1 R1 Reviewer B3 修复 — 不 monkey-patch phase5**）

不动 `exp.verdict_factor_judge.phase5.spec` 任何模块级状态。本地写 G5 generator 复用 phase5 的 `_g5_recipe_meta` 和 `_g5_fixed_weights`（这俩是纯函数，无 module-level mutation）；G1-G4 复用 phase5 generator（它们不读 G5_THRESHOLD_GRID）。

```python
import dataclasses
from exp.verdict_factor_judge.phase5 import spec as p5
from exp.verdict_factor_judge.phase5.spec import Cell, LOCKED_FH, LOCKED_WS

CFG_ID_DEFAULT = "spatial16_ws_d1_best"
YAML_PREFIX = "ws_d1_kin"
SUPER_WARMUP_ID = "ws_d1_kin_super_warmup"
G5_RECIPES = p5.G5_RECIPES   # ("p1_state_fut_online_act", "p2_action_fut_online_act", "g6_f1a_a_d_jerk_curv_pair")


def _g5_grid_filtered():
    """Triangular grid: fh + ws <= 0.9 inclusive (mirror threshold_pareto §2.2).
    
    Implementation note: monotonically reproducible; result = 15 pairs (verified).
    """
    return tuple(
        (fh, ws) for fh in (0.2, 0.3, 0.4, 0.5)
                 for ws in (0.2, 0.3, 0.4, 0.5)
                 if fh + ws <= 0.9 + 1e-9
    )


def _generate_g5_cells_local(cfg_id=CFG_ID_DEFAULT):
    """G5 generator with local grid, NO mutation of p5 module state.
    
    Mirrors `phase5.spec.generate_g5_cells` body but iterates _g5_grid_filtered()
    instead of `p5.G5_THRESHOLD_GRID`. Reuses `p5._g5_recipe_meta` and
    `p5._g5_fixed_weights` (both pure functions, safe to call).
    """
    cells = []
    for recipe_id in G5_RECIPES:
        meta = p5._g5_recipe_meta(recipe_id)
        factors = tuple(meta["factors"])
        declared = meta["declared_keys"]
        weights = p5._g5_fixed_weights(recipe_id, declared)
        short = meta["short"]
        for (fh, ws) in _g5_grid_filtered():
            axis_tag = f"{short}__fh{fh}_ws{ws}"
            yaml_id = f"{cfg_id}_phase5_g5_{axis_tag}"
            cells.append(Cell(
                yaml_id=yaml_id,
                group="g5",
                base_recipe=short,
                axis_tag=axis_tag,
                factors=factors,
                weights=weights,
                fh_ratio=fh,
                ws_ratio=ws,
                declared_keys=declared,
                warmup_yaml_id=meta["warmup_yaml_id"],   # placeholder, rewritten below
            ))
    return cells


def generate_all_cells():
    """Compose G1-G4 (phase5 native) + G5 (local-filtered grid), then rename.
    
    No phase5 module-level mutation. `p5.G5_THRESHOLD_GRID` remains
    untouched, so any subsequent `p5.generate_all_cells()` call in this
    process still returns the original 240-cell phase5 spec.
    """
    g1_g4 = (
        p5.generate_g1_cells(cfg_id=CFG_ID_DEFAULT)
        + p5.generate_g2_cells(cfg_id=CFG_ID_DEFAULT)
        + p5.generate_g3_cells(cfg_id=CFG_ID_DEFAULT)
        + p5.generate_g4_cells(cfg_id=CFG_ID_DEFAULT)
    )
    g5 = _generate_g5_cells_local(cfg_id=CFG_ID_DEFAULT)
    cells = g1_g4 + g5
    cells.sort(key=lambda c: c.yaml_id)
    
    renamed = []
    for c in cells:
        new_yaml_id = c.yaml_id.replace(f"{CFG_ID_DEFAULT}_phase5", YAML_PREFIX, 1)
        # G3 原本 share warmup_yaml_id by (base, channel) — 统一替换为 super
        # G5 原本指向 phase3/phase4 warmup_yaml_id — 统一指向 super
        renamed.append(dataclasses.replace(c, yaml_id=new_yaml_id, warmup_yaml_id=SUPER_WARMUP_ID))
    return renamed


def super_warmup_declared_keys():
    """The single source of truth for what super warmup MUST dump."""
    return set().union(*[set(c.declared_keys) for c in generate_all_cells()])

# Total: G1=48 + G2=48 + G3=48 + G4=48 + G5=45 = 237 cell
```

⚠ phase5 G5 cells 原本 `warmup_yaml_id` 指向 `spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup`（历史 warmup） — 本 plan 统一替换为 super warmup，意味着 G5 cell.declared_keys 必须**子集于** super_warmup_declared_keys。Stage 3 verify check #5 验证（237 cell 全 bind_keys 试一遍）。

实测：phase4 p1/p2 recipes 各 declared 8 offline keys (state/action × 4 desc × W_FUT(0,3)+(0,5))，phase3 g6 recipe 含 2 keys（`jerk_online_action_p3_f3, dispersion_online_action_p3_f3`），**全部已在 phase5 G1-G4 declared 并集内** ✓（pending check #5 confirm）。

#### 1.3 `build_eval_yaml_for_cell` — offline calibration mode

```python
def build_eval_yaml_for_cell(cell, fh_thr, ws_thr, super_raw_relpath):
    """Build eval yaml using offline calibration mode (samples_source.type=offline).
    
    super_raw_relpath: server-side relative path to super_warmup_raw.jsonl
                      (e.g. "exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl")
    """
    from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS, normalization_offline
    from exp.verdict_factor_judge.phase5.spec import _build_composer
    from exp.verdict_factor_judge.phase3.spec import _directions_for
    
    cfg = CFG_SPECS[CFG_ID_DEFAULT]
    composer = _build_composer(cell, fh_thr, ws_thr)  # 复用 phase5
    judge = {
        "type": "composite",
        "normalization": normalization_offline(),  # 跟 phase5 同
        "factors": list(cell.factors),
        "calibration": {
            "type": "percentile_rolling",
            "params": {"window_size": 50},
            "samples_source": {
                "type": "offline",
                "offline": {"path": super_raw_relpath, "format": "jsonl"},
            },
        },
        "composer": composer,
        "export_factor_outputs": True,
    }
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": dict(cfg["keys"]),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {"cp1": {
            "enabled": True,
            "gate": {"type": "always_search"},
            "judge": judge,
            "search_strategy": dict(cfg["search_strategy"]),
        }},
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {"preload_path": cfg["preload_pkl"], "index_type": "brute_force"},
        },
        "write_policy": {"type": "never"},
    }
```

**关键差异 vs phase5**：`calibration.samples_source.type = "offline"`（不是 `"warmup"`）→ eval 启动时 server 自己读盘构 CalibrationSamples，**完全绕开 WarmupPool + preload_normalizer_buffer 链路**。

### Stage 2 — super warmup yaml + 跑一次

#### 2.1 dump factor list 由 **真实 declared_keys 并集**驱动

```python
def build_super_warmup_yaml():
    from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS, factor
    from exp.weighted_sum.kinematic.spec import super_warmup_declared_keys
    
    cfg = CFG_SPECS["spatial16_ws_d1_best"]
    needed_keys = super_warmup_declared_keys()  # 50 keys, ground truth
    
    # Group keys back into factor blocks: f"{desc}_{src}_{ch}__p{P}_f{F}"
    # → factor type = f"{desc}_{src}_{ch}", windows={(P, F)}
    factor_windows = {}  # (type_name) -> set of (P, F)
    for k in needed_keys:
        head, win = k.split("__p")
        p, f = win.split("_f")
        factor_windows.setdefault(head, set()).add((int(p), int(f)))
    
    factors = [
        factor(t, windows=[{"past": p, "future": f} for (p, f) in sorted(wins)])
        for t, wins in sorted(factor_windows.items())
    ]
    # NB: topk_action_variance 未使用，不加入
    
    ss = dict(cfg["search_strategy"])
    ss["top_k"] = 5  # 给 future-extensible factor 留余量 (phase5 traditional)
    
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": dict(cfg["keys"]),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {"cp1": {
            "enabled": True,
            "gate": {"type": "always_search"},
            "judge": {
                "type": "always_warm_start",
                "start_t": 0.7,
                "dump": {"deferred": True, "config_id": "ws_d1_kin_super_warmup", "factors": factors},
            },
            "search_strategy": ss,
        }},
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {"preload_path": cfg["preload_pkl"], "index_type": "brute_force"},
        },
        "write_policy": {"type": "never"},
    }
```

#### 2.2 driver: 起 server + 跑 warmup（trials_per_task=15）

```python
def run_super_warmup(host="weiland.top", port=14000, num_workers=8, trials_per_task=15):
    """trials_per_task=15 means 150 ep ≈ 3250 verdict (per §2.13 实测 21.7 verdict/ep)."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    yaml_dict = build_super_warmup_yaml()
    yaml_path = Path("exp/weighted_sum/config/kinematic_phase5/super_warmup.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(yaml_path, yaml_dict)
    raw_dst = Path("exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl")
    raw_dump_dst = raw_dst.parent / "super_warmup_raw_dump.jsonl"
    
    # R1D MAJOR fix: 严格 row-count cache 判定，防止 crash 后 partial raw 被 skip
    if raw_dst.exists() and _line_count(raw_dst) >= 2500:  # safety floor < target 3250
        print(f"[super_warmup] cached ({_line_count(raw_dst)} rows), skip")
        return raw_dst
    if raw_dst.exists():
        print(f"[super_warmup] partial raw ({_line_count(raw_dst)} rows) → re-running")
        raw_dst.unlink()
        raw_dump_dst.unlink(missing_ok=True)
    
    with WebsocketClientPolicy(host=host, port=port) as ctl:
        ctl.load_cache_config(yaml_content=yaml_path.read_text(), yaml_id="ws_d1_kin_super_warmup")
        cmd, env = _build_libero_argv(
            yaml_id="ws_d1_kin_super_warmup", phase="warmup",
            num_trials_per_task=trials_per_task, ...
        )
        subprocess.run(cmd, env=env, check=True)
        content = ctl.fetch_dump("ws_d1_kin_super_warmup")
        _save_dump_jsonl(content, raw_dump_dst)
        needed_keys = list(super_warmup_declared_keys())
        _extract_finite_factor_raw(raw_dump_dst, raw_dst, needed_keys)
    return raw_dst
```

跑量参数 **基于 §2.13 实测**：
- `trials_per_task=15` × 10 task = **150 ep** ≈ **3250 verdict**
- 估算 finite for 大 window：
  - `(7,7)_state`: ep 前 7 step + chain-end fail → 每 ep ~5 finite → 150 ep ~750 finite ✓
  - `(5,0)_state`: ep 前 5 step → 每 ep ~17 finite → 2500 finite ✓
- 单 server 1 replica 15-20 min

#### 2.3 数据流转（**R1C MAJOR §K 修复**）

```
ziyang10 server (1 replica) [tmpfs warmup_dump_root]
    ↓ fetch_dump (WebSocket bytes)
timan107 driver [exp/weighted_sum/data/kinematic_phase5/super_warmup_raw_dump.jsonl]
    ↓ _extract_finite_factor_raw(needed_keys=union of 50 declared)
timan107 driver [exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl]    ← ground truth
    ↓ tether push (/tmp/super_warmup_raw.jsonl 中转，allow_roots 见 session_handoff §4 #2)
ziyang10 + xuanle server [exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl]
    ↓ samples_source.type=offline 启动时读盘
server-side CalibrationSamples（per-connection，每次 bind 时重读 ~1s）
```

```bash
# Push to both servers
for node in jupyter-ziyang10 jupyter-xuanlel2; do
  tether push exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl $node:/tmp/super_raw.jsonl
  tether exec $node -- bash -lc "
    export HOME=/home/$(case $node in *ziyang10*) echo ziyang10 ;; *xuanlel2*) echo xuanlel2 ;; esac)
    cd /home/\$(id -un)/openpi  # whichever
    mkdir -p exp/weighted_sum/data/kinematic_phase5
    mv /tmp/super_raw.jsonl exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl
  "
done
```

具体路径修正：driver 端用 `/home/weiland/projects/openpi/exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl`；server 端是 `/home/<user>/openpi/exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl`（ziyang10 ziyang10 / xuanle xuanlel2）。eval yaml 写 **相对路径** `exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl`，server 按自己 CWD 解析。

### Stage 3 — Verify HARD GATE（**R1A/B/C/D 共同要求加强**）

完成 super warmup raw 后，**必须**通过以下 7 个 check 才能进 Stage 4。check 用 try/except 收集失败列表后 assert（R1A MINOR 修复）。

```python
# kinematic/super_warmup.py: verify_super_raw(raw_path)
def verify(raw_path):
    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    
    # Check 1: row count
    assert len(rows) >= 2500, f"too few rows: {len(rows)} < 2500"
    
    # Check 2: cp1_score non-null ≥ 99%
    n_null = sum(1 for r in rows if r.get("cp1_score") is None)
    assert n_null / len(rows) < 0.01, f"cp1_score null rate {n_null}/{len(rows)}"
    
    # Check 3: all declared keys present + finite ≥ 50, with per-window detail
    from exp.weighted_sum.kinematic.spec import super_warmup_declared_keys
    needed = super_warmup_declared_keys()
    per_key_finite = {k: 0 for k in needed}
    for row in rows:
        raw = row.get("factor_raw", {})
        for k in needed:
            v = raw.get(k)
            if v is not None and not math.isnan(float(v)):
                per_key_finite[k] += 1
    bad = {k: n for k, n in per_key_finite.items() if n < 50}
    if bad:
        # specifically check (5,0) past-only and (7,7) wide-window (R1D MINOR)
        critical = [k for k in bad if "p5_f0" in k or "p7_f7" in k]
        print(f"WARN bad keys: {len(bad)} total, {len(critical)} are (5,0)/(7,7) variants")
        print(f"  sample bad: {list(bad.items())[:10]}")
        assert not bad, f"{len(bad)} keys have <50 finite (run more warmup trials)"
    
    # Check 4: per-cell reconstruct_scores succeeds for 5 sample cells × 5 groups
    from exp.weighted_sum.kinematic.spec import generate_all_cells
    from exp.verdict_factor_judge.phase3.threshold_solver import reconstruct_scores
    from exp.verdict_factor_judge.phase5.spec import cell_to_solver_recipe
    fails = []
    for g in ("g1", "g2", "g3", "g4", "g5"):
        c = next(c for c in generate_all_cells() if c.group == g)
        try:
            recipe = cell_to_solver_recipe(c)
            scores = reconstruct_scores(raw_path, recipe, composer_weights=c.weights)
            finite = [s for s in scores if not math.isnan(s)]
            assert len(finite) >= 500, f"{c.yaml_id}: only {len(finite)} finite scores"
        except Exception as e:
            fails.append((g, c.yaml_id, str(e)))
    assert not fails, f"reconstruct_scores failed: {fails}"
    
    # Check 5: derive_thresholds monotone+non-degenerate for 5 (fh, ws) ∈ filtered grid
    from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds
    from exp.weighted_sum.kinematic.spec import _g5_grid_filtered
    g5_cell = next(c for c in generate_all_cells() if c.group == "g5" and c.base_recipe == "p1")
    scores = reconstruct_scores(raw_path, cell_to_solver_recipe(g5_cell), composer_weights=g5_cell.weights)
    for (fh, ws) in _g5_grid_filtered()[:5]:
        t_fh, t_ws = derive_thresholds(scores, fh, ws)
        # R1A MAJOR fix: allow equality (tie-breaks valid); R1D BLOCKER #1 the strict-less was the bug
        assert 0.0 <= t_ws <= t_fh <= 1.0, f"({fh},{ws}): t_ws={t_ws} > t_fh={t_fh}"
        # R1B DESIGN-CRITICAL: also check non-degenerate (T_ws > min)
        assert t_ws > min(scores) + 1e-6 or (fh + ws) > 0.85, \
            f"({fh},{ws}): T_ws degenerate to min(scores), shouldn't happen with grid filter"
    
    # Check 6: ALL 237 cells bind_keys passes (catches any G5 declared_keys ⊄ super)
    from exp.verdict_factor_judge.phase3.threshold_solver import load_per_key_finite_history
    from openpi.cache.components.factors.calibrations.percentile_rolling import PercentileRollingCalibration
    from openpi.cache.components.factors.base import CalibrationSamples
    super_buffer = load_per_key_finite_history(raw_path, list(needed))
    bind_fails = []
    for c in generate_all_cells():
        try:
            cal = PercentileRollingCalibration(CalibrationSamples(super_buffer), window_size=50)
            cal.bind_keys(list(c.declared_keys))
        except Exception as e:
            bind_fails.append((c.group, c.yaml_id, str(e)))
    assert not bind_fails, f"bind_keys failed for {len(bind_fails)}/237 cells: {bind_fails[:5]}"
    
    # Check 7: bootstrap CI of (T_fh, T_ws) for 3 cells — sanity for thin distribution (R1B SUGGESTION)
    import numpy as np
    sample_cells = [g5_cell] + [next(c for c in generate_all_cells() if c.group == g and c.base_recipe == "p1") for g in ("g1", "g3")]
    for c in sample_cells:
        scores = reconstruct_scores(raw_path, cell_to_solver_recipe(c), composer_weights=c.weights)
        scores_arr = np.array([s for s in scores if not math.isnan(s)])
        t_fh_boot, t_ws_boot = [], []
        rng = np.random.default_rng(42)
        for _ in range(200):
            bs = rng.choice(scores_arr, size=len(scores_arr), replace=True)
            arr_desc = np.sort(bs)[::-1]
            i_fh = int(0.3 * len(arr_desc)) - 1  # fh_ratio=0.3 reference cell
            i_ws = int(0.6 * len(arr_desc)) - 1
            t_fh_boot.append(arr_desc[max(0, i_fh)])
            t_ws_boot.append(arr_desc[max(0, i_ws)])
        ci_fh = np.percentile(t_fh_boot, [2.5, 97.5])
        ci_ws = np.percentile(t_ws_boot, [2.5, 97.5])
        print(f"[CI] {c.yaml_id} @ fh=0.3 ws=0.3: T_fh CI={ci_fh}, T_ws CI={ci_ws}")
        # Sanity: CI width < 0.1 (else super warmup raw too noisy)
        assert (ci_fh[1] - ci_fh[0]) < 0.15, f"T_fh CI too wide: {ci_fh[1]-ci_fh[0]:.3f}"
    
    print("OK 7/7 verify checks")
```

**fallback**：若 check 3 失败（某 key <50 finite），自动 `trials_per_task += 5` 重跑（最多 3 次到 trials=25）；超过则 agentchat 通知 owner 决定。

### Stage 4 — `kinematic/strategy.py` + `runner.py`

#### 4.1 `ConductorDriver` 内部 flush per_step（**G1 R2 Reviewer B1 修复 — 改走 driver 内部路径，不改 strategy hook 签名**）

G1 R1 R3 提议的"on_stage_complete 加 `driver=None` kwarg + strategy 走 `driver._flush_...`"路径在 G1 R2 被否决：reviewer 实测 repo 中已有 **5 个具体 override** 都用旧 3-arg 签名：

- `exp/verdict_factor_judge/strategies/warmup_eval_strategy.py:169`
- `src/openpi/conductor/strategy.py:108` (base default — 这个本来就要改，但其他 5 个不能改)
- `tests/conductor/test_driver.py:118, 249`
- `tests/conductor/conftest.py:104`

向 `on_stage_complete(stage, ctl, ctx, driver=self)` 传 kwarg 会让上述任一 override 抛 `TypeError: got an unexpected keyword argument 'driver'`。这是接口回退。

按 reviewer 指示 (b) 路径：**让 ConductorDriver 在 `strategy.on_stage_complete` 调用之后自己 flush**，strategy hook 签名完全不动。

**driver.py 改动**（实测 `_complete_stage` line 182-186）：

```python
# src/openpi/conductor/driver.py 增量 patch

import threading   # 新 import

class ConductorDriver:
    def __init__(
        self,
        ...,
        per_step_writer: Callable[[str, list[dict]], None] | None = None,   # NEW
    ):
        ...
        self._per_step_writer = per_step_writer
        self._per_step_lock = threading.RLock()   # 包既有 _per_step_rows 写入路径
    
    def _flush_per_step_for_stage(self, yaml_id: str) -> None:
        """Drain _per_step_rows for `yaml_id` to writer, then remove from buffer.
        
        Called internally at the end of _complete_stage (NOT exposed to strategy
        — strategy hook signature unchanged, per G1 R2 Reviewer B1 mandate).
        """
        if self._per_step_writer is None:
            return
        with self._per_step_lock:
            this_yaml_rows = [r for r in self._per_step_rows if r.get("yaml_id") == yaml_id]
            if not this_yaml_rows:
                return
            try:
                self._per_step_writer(yaml_id, this_yaml_rows)
            except Exception:
                logger.exception("per_step_writer failed for yaml_id=%s; rows retained in memory", yaml_id)
                return   # don't drop from buffer on failure — final dump at run() end will catch them
            self._per_step_rows = [r for r in self._per_step_rows if r.get("yaml_id") != yaml_id]
    
    def _complete_stage(self, stage: _task.Stage) -> None:
        self._scheduler.mark_complete_running(stage.stage_id)
        ctl = self._ctl(stage.server)
        # G1 R2 B1: strategy hook signature unchanged — drain happens AFTER strategy callback
        self._strategy.on_stage_complete(stage, ctl, self._ctx)
        if stage.phase == "eval" and self._per_step_writer is not None:
            self._flush_per_step_for_stage(stage.yaml_id)
        self._scheduler.mark_complete_done(stage.stage_id)
```

**strategy.py base 改动：0 行**（hook 签名完全不动 — 这是 G1 R2 B1 强制约束）。

**5 个已有 override (warmup_eval_strategy.py / 4 个 test/conftest)**：0 改动。

**`kinematic/strategy.py`**（~45 行，简化版）：

```python
import json
from pathlib import Path
from exp.weighted_sum.weight_search_strategy import WeightSearchStrategy

class KinematicSearchStrategy(WeightSearchStrategy):
    """Override only to:
    
    1. offline calibration mode → no preload_normalizer_buffer needed
       (super_warmup_raw.jsonl pushed to server filesystem; eval yaml uses
       samples_source.type=offline so server 启动时各自读盘构 CalibrationSamples)
    
    Per-step incremental flush is driver-internal (driver._flush_per_step_for_stage
    called at end of _complete_stage); strategy hooks unchanged.
    """
    def __init__(self, *args, per_step_out_dir: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._per_step_out_dir = per_step_out_dir
        if per_step_out_dir is not None:
            per_step_out_dir.mkdir(parents=True, exist_ok=True)
    
    def _write_per_step(self, yaml_id: str, rows: list[dict]) -> None:
        """Bound method to inject as ConductorDriver(per_step_writer=...).
        
        driver internally invokes us at stage completion; we own only the
        file format / path decision.
        """
        if self._per_step_out_dir is None:
            return
        out = self._per_step_out_dir / f"{yaml_id}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    
    # on_stage_begin / on_stage_complete: NOT overridden — inherits base default
    # (per WeightSearchStrategy which only loads yaml; per_step flush is driver job)
```

**`src/openpi/` 改动汇总**（G1 R2 修订）：
- `driver.py` +~14 行（per_step_writer 注入 + RLock + _flush_per_step_for_stage + _complete_stage 末尾一行 call）
- `strategy.py` **0 行**
- 既有 5 个 override **0 行**
- 总计 **~14 行**（比 R1 R3 的 15 行减少 1，且零接口风险）

#### 4.2 `runner.py`（~250 行）— 4-mode CLI

```python
VALID_MODES = ("emit-warmup", "run-warmup", "verify-raw", "emit-eval-yamls", 
               "run-eval", "run-always-warm", "analyze")

def _mode_emit_warmup(args): ...
def _mode_run_warmup(args):
    raw = run_super_warmup(host=args.host, port=args.port, ...)
    verify(raw)

def _mode_emit_eval_yamls(args):
    super_raw_relpath = "exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl"
    n_ok, n_skip = 0, 0
    fails = []
    for cell in generate_all_cells():
        try:
            recipe = cell_to_solver_recipe(cell)
            scores = reconstruct_scores(args.super_raw_path, recipe, composer_weights=cell.weights)
            fh_thr, ws_thr = derive_thresholds(scores, cell.fh_ratio, cell.ws_ratio)
        except (ValueError, RuntimeError, KeyError) as e:
            # R1A BLOCKER #5 + R1D BLOCKER #2 fix: per-cell try/except, skip-not-crash
            fails.append((cell.yaml_id, str(e)))
            n_skip += 1
            (args.thresholds_dir / f"{cell.yaml_id}__SKIP.json").write_text(json.dumps({
                "yaml_id": cell.yaml_id, "error": str(e),
            }))
            continue
        yaml_dict = build_eval_yaml_for_cell(cell, fh_thr, ws_thr, super_raw_relpath)
        write_yaml(args.eval_dir / f"{cell.yaml_id}.yaml", yaml_dict)
        # Snapshot thresholds
        (args.thresholds_dir / f"{cell.yaml_id}.json").write_text(json.dumps({
            "yaml_id": cell.yaml_id, "fh_ratio": cell.fh_ratio, "ws_ratio": cell.ws_ratio,
            "fh_thr": fh_thr, "ws_thr": ws_thr,
            "n_finite_scores": sum(1 for s in scores if not math.isnan(s)),
        }))
        n_ok += 1
    print(f"[emit-eval] {n_ok} ok, {n_skip} skip; fails: {fails[:10]}")

def _mode_run_eval(args):
    """Dispatch to run_phase2 with KinematicSearchStrategy + dual-server topology."""
    # Just construct args and call run_phase2 main with --strategy=kinematic flag
    # (alternative: build ConductorDriver inline)
    ...
```

#### 4.3 `run_phase2.py` change（**G1 R2 Reviewer B2 修复 — wire per_step_writer**）

R3 v1 仅写了 strategy 切换但**没把** `per_step_writer=strategy._write_per_step` 传入 `ConductorDriver` → `driver._per_step_writer` 留 None → `_flush_per_step_for_stage` 早退 → 增量 flush 失效。G1 R2 B2 指出此漏洞。

R3 修订（~10 行）：

```python
# exp/weighted_sum/run_phase2.py（增量 patch）

parser.add_argument("--strategy", choices=("weight", "kinematic"), default="weight")
...

per_step_writer = None   # default: behavior unchanged for weight strategy
if args.strategy == "kinematic":
    from exp.weighted_sum.kinematic.strategy import KinematicSearchStrategy
    per_step_dir = (
        Path(args.per_step_out).parent / "per_step"
        if args.per_step_out else None
    )
    strategy = KinematicSearchStrategy(
        ...,
        per_step_out_dir=per_step_dir,
    )
    per_step_writer = strategy._write_per_step   # <--- G1 R2 B2 fix: 必须 wire 到 driver
else:
    strategy = WeightSearchStrategy(...)

driver = ConductorDriver(
    ...,
    strategy=strategy,
    per_step_writer=per_step_writer,   # <--- new kwarg; None for weight strategy = behavior unchanged
    ...
)
```

不加 `--pre-eval-hook`（R1A/C/D 三 agent 都否决）。`weight` 模式 `per_step_writer=None` → driver `_flush_per_step_for_stage` 自动早退 → 既有 wsweep / threshold_pareto 行为完全不变。

#### 4.4 per_step 增量落盘契约

- driver run 期间 `ConductorDriver._complete_stage` 末尾**driver 内部**调 `_flush_per_step_for_stage(stage.yaml_id)`，每个 yaml 完成时通过注入的 `per_step_writer` 写 `data/kinematic_phase5/per_step/<yaml_id>.jsonl`
- driver.run() 结束时**再**写一次 dedup 防止 strategy 漏拉（兜底）
- 分析阶段先扫 per_step/ 目录 merge 出全集

```bash
# 启动
tether exec timan107 -- bash -lc '
tmux kill-session -t run0 2>/dev/null
tmux new -s run0 -d "cd /scratch/zixuans8/openpi && PYTHONPATH=. /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py \
  --strategy kinematic \
  --yaml-dir exp/weighted_sum/config/kinematic_phase5/eval \
  --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
  --journal exp/weighted_sum/data/kinematic_phase5/journal.jsonl \
  --servers weiland.top:14000,weiland.top:14001 \
  --task-ids 0-9 --eval-trials 10 \
  --workers 64 --server-workers 16,48 \
  --gpus 8 --conda-env /scratch/zixuans8/libero_sim \
  --eval-concurrency 2 \
  --per-step-out exp/weighted_sum/data/kinematic_phase5/per_step.jsonl 2>&1 | tee /tmp/kin_run.log"
'
```

### Stage 5 — run-eval（dual-server 16/48）

`run_phase2 --strategy kinematic` 跑 237 cell × 100 ep ≈ 23700 ep；预估 dual-server 16/48 worker / GTX1080 client → **~3.0 h**。

⚠ 跨 GPU caveat（**R1C BLOCKER #4 + R1D BLOCKER #3 + 红线 §16.6**）：
- super warmup 在 ziyang10 H200 NVL（UUID 6eaa816f...）跑得 raw
- eval 60 cell 在 ziyang10 + 165 cell 在 xuanle H200 NVL（UUID 5c049b56...）
- 两块 H200 NVL 同型号但不同物理 → bf16 累加微差，retrieve score 差异 ~<1%
- 后果：**eval-side calibration 仍读同一份 super raw（offline 模式），threshold 一致**；但 retrieve cp1_score 在 xuanle 上数值略不同 → 进入 calibration buffer 后被 percentile rank 吸收，影响极小
- **report 必须按红线 §16.6 注明**这条 caveat

### Stage 6 — always-WARM 3-cell ceiling（**R1B DESIGN-CRITICAL #3 修复**）

```python
# kinematic/runner.py: _mode_run_always_warm
def _mode_run_always_warm(args):
    """Drop-in d1 base ceiling at start_t ∈ {0.3, 0.5, 0.7}.
    
    NOT phase5-rrf reused values; this is d1-ws self-measured ceiling.
    """
    for start_t in (0.3, 0.5, 0.7):
        yaml_dict = build_always_warm_yaml(start_t=start_t)
        # run 100 ep × 10 task on single server
        ...
```

预算：3 yaml × 100 ep = 300 ep ≈ 10 min on single server。

### Stage 7 — Analyze

#### 7.1 decision gate 沿用 phase5（5pp rule）

phase5/runner.py:_dump_decision_gate_table_phase5 复用（针对 G3 12 pattern → bucket by (base, channel) 跑 5pp top1-top2 Δ；G5 出 Pareto frontier）。

R1B DESIGN-MAJOR #6: frontier decision rule **明文化**：在最终报告中按以下规则判 dominate：
- "本实验 frontier dominate phase5 d4" 要求：(a) 至少 3 个 inf-bucket 上本实验 SR ≥ phase5 SR + 5pp，且 (b) 没有任何 bucket 反向 ≥ 5pp 落败
- "frontier 重叠"：所有 bucket ΔSR ∈ [-5pp, +5pp]
- 中间状态：mixed dominance，单独描述

#### 7.2 `plot_pareto_overlay.py` —— 4 frontier 同图
- r/p baseline (gray dashed)（与检索无关 ✓）
- threshold_pareto d1 envelope (teal solid)（同检索同 base）
- **本实验 237 cell 5 group 散点 + per-group frontier + 总 frontier (red)**
- phase5 d4 (purple, reference only — 不同检索注明)
- 本实验自家 always-WARM 3 点 (red star, NEW)
- phase5 d4 always-WARM 3 点 (purple star, **明确标 phase5-d4-only**)

---

## 5. 关键代码改动汇总

| 文件 | 类型 | 行数 | 说明 |
|---|---|---:|---|
| `exp/verdict_factor_judge/common/v2_spec.py` | edit | +25 | 加 `CFG_SPECS["spatial16_ws_d1_best"]` |
| `exp/weighted_sum/kinematic/__init__.py` | new | 1 | |
| `exp/weighted_sum/kinematic/spec.py` | new | ~120 | rename + cfg swap + G5 grid filter + super_warmup_declared_keys() |
| `exp/weighted_sum/kinematic/super_warmup.py` | new | ~250 | yaml 构造（factor list 由 declared union 驱动）+ driver + verify (7 check + bootstrap CI) |
| `exp/weighted_sum/kinematic/strategy.py` | new | ~45 | KinematicSearchStrategy (inherits WeightSearchStrategy hooks unchanged; exposes `_write_per_step` as writer callable for driver injection) |
| `exp/weighted_sum/kinematic/runner.py` | new | ~280 | 7-mode CLI |
| `exp/weighted_sum/kinematic/analysis/__init__.py` | new | 1 | |
| `exp/weighted_sum/kinematic/analysis/plot_pareto_overlay.py` | new | ~150 | 4 frontier + always-warm anchor |
| `exp/weighted_sum/run_phase2.py` | edit | +5 | `--strategy={weight,kinematic}` switch |
| `tests/test_kinematic_super_warmup.py` | new | ~100 | regression (含 phase5 原 spec 不破坏) |
| `docs/experiments/weighted_sum.md` | edit | +30 | 加 §4 kinematic phase5 对照 |
| **`src/openpi/conductor/driver.py`** | **edit** | **+~14** | **per_step_writer ctor 注入 + RLock + `_flush_per_step_for_stage` + `_complete_stage` 末尾 internal call（G1 R2 B1 mandated driver-internal path）** |
| ~~`src/openpi/conductor/strategy.py`~~ | ~~edit~~ | **0** | **G1 R2 B1：strategy hook 签名完全不动**（避免破坏既有 5 个 override） |

**src/openpi/ 改动**：~**14 行**（仅 driver.py，G1 R2 Reviewer B1 强制 driver-internal flush 路径，零接口风险；G1 R1 R3 的 strategy 签名变化已撤销）。
**既有 5 个 `on_stage_complete(self, stage, ctl, ctx)` override**（`warmup_eval_strategy.py:169` + `tests/conductor/{test_driver.py:118,249, conftest.py:104}`）：**0 改动**。

---

## 6. 风险与缓解（R1 全部 BLOCKER/MAJOR 应对）

| # | 风险 | 严重性 | 缓解 |
|---|---|---|---|
| R1 | super warmup 某 key < 50 finite | High | Stage 3 check #3 + 自动 trials+5 重跑 fallback ≤ 3 次 → agentchat 通知 owner |
| R2 | WarmupPool LRU 失败 | ~~High~~ **不存在** | **改用 offline calibration source 完全绕开**（§2.7） |
| R3 | super warmup 跑 ziyang，eval 跨 ziyang+xuanle GPU 差异 | Medium | 红线 §16.6 报告中注明；offline calibration 吸收大部分（cp1_score 进 percentile 后 normalized） |
| R4 | 77 factor extract per-step server CPU/RAM 压力 | Low→Medium | 实际 50 factor extract，单 server 1 replica，预 OOM 风险低；session_handoff §15 #11 的 force-MISS 48-worker OOM 不适用（本实验 super warmup 用 always_warm_start + 8 worker） |
| R5 | super warmup mid-crash → partial raw 卡 cache | Medium | Stage 2.2 严格 `_line_count ≥ 2500 才 skip`，否则 delete + 重跑 |
| R6 | eval 期间 agent restart 炸全部 worker | High | 红线 §16.8 **eval 中绝不重启 agent**；增量 flush per_step（§4.4）使 crash 不丢已 done yaml |
| R7 | d1 best 3 选 1 实际选择偏差 | Low | §2.1 注释；G1 owner 可换 |
| R8 | G5 declared_keys ⊄ super union | High | Stage 3 check #6 强制验证 237 cell bind_keys 全过 |
| R9 | per_step.jsonl 内存累积 + driver crash 丢失 | ~~High~~ Low | §4.4 增量 flush 解决 |
| R10 | expose tunnel 长闲掉线 (~3h eval) | Medium | 监控脚本含 `tether ps -a | grep "ziyang-srv\|xl-srv"` 健康检查；掉线即 re-expose（不重启 server） |
| R11 | G1-G4 全 inconclusive（重蹈 phase5）| Low（设计可接受）| H2 假设接受这是合法结论；§7.4 fallback "挑 Δ 最大 bucket 升 500 ep 复测"（owner 可选） |
| R12 | quantile estimator 在 ~3250 verdict 上 CI 宽 | Low | Stage 3 check #7 bootstrap CI 显式给出；分析阶段在 frontier 点画 error bar |
| R13 | pkill self-match / HOME 陷阱 / allow_roots | Medium | session_handoff §4 #5 §4 #1 §4 #2 铁律 — 监控/清理脚本严格遵循 |
| R14 | tyro 顺序（仅 Stage 0 server 重启时）| Low | §session_handoff §15 #1 + Stage 0 命令模板已遵守 |
| R15 | xuanle replica_proxy 在 eval 时 fetch_dump 拒收 | N/A | eval 用 offline calibration，**不调用 fetch_dump** |

---

## 7. 测试 / 验收清单

### 7.1 单测（pre-flight，**G1 R1 B1+B3+B4 修复**）

```python
# tests/test_kinematic_super_warmup.py
def test_237_cells_generated():
    """After G5 filter (fh+ws<=0.9), total = 48*4 + 15*3 = 237 (verified arithmetic)."""
    from exp.weighted_sum.kinematic.spec import generate_all_cells
    cells = generate_all_cells()
    groups = Counter(c.group for c in cells)
    assert groups == {"g1": 48, "g2": 48, "g3": 48, "g4": 48, "g5": 45}, groups
    assert len(cells) == 237

def test_warmup_yaml_id_unified():
    for c in generate_all_cells():
        assert c.warmup_yaml_id == "ws_d1_kin_super_warmup"

def test_super_factor_superset_covers_all_declared_keys():
    sup = super_warmup_declared_keys()
    miss = set()
    for c in generate_all_cells():
        miss |= set(c.declared_keys) - sup
    assert not miss, f"{len(miss)} keys not covered: {sorted(miss)[:10]}"

def test_g5_grid_filtered():
    """G1 R1 Reviewer B1 verified arithmetic: 15 pairs, not 11."""
    from exp.weighted_sum.kinematic.spec import _g5_grid_filtered
    grid = _g5_grid_filtered()
    for fh, ws in grid:
        assert fh + ws <= 0.9 + 1e-9
    assert len(grid) == 15   # exactly (0.5, 0.5) excluded
    assert (0.4, 0.5) in grid and (0.5, 0.4) in grid   # boundary kept
    assert (0.5, 0.5) not in grid

def test_phase5_orig_spec_still_works_REGARDLESS_OF_ORDER():
    """G1 R1 Reviewer B3 mandate: no module-level mutation of phase5.
    
    Call kinematic.generate_all_cells() FIRST (which would have monkey-
    patched in v2), then call phase5 native — phase5 must still emit 240
    cells with original G5_THRESHOLD_GRID intact.
    """
    from exp.weighted_sum.kinematic.spec import generate_all_cells as kin_gen
    from exp.verdict_factor_judge.phase5 import spec as p5_mod
    
    # First: run kinematic generator (would mutate p5 in v2 design)
    _ = kin_gen()
    
    # Then: verify p5 still has original 16-pair grid
    assert len(p5_mod.G5_THRESHOLD_GRID) == 16, \
        "kinematic.spec mutated phase5.G5_THRESHOLD_GRID — BLOCKER B3 regression"
    
    # And p5.generate_all_cells still emits 240
    cells = p5_mod.generate_all_cells()
    assert len(cells) == 240
    assert all("spatial16_w8_d4" in c.yaml_id for c in cells)
    g5_count = sum(1 for c in cells if c.group == "g5")
    assert g5_count == 48, f"phase5 G5 count drifted from 48 to {g5_count}"


@pytest.mark.manual
def test_super_warmup_yaml_validates_against_config(tmp_path):
    """G1 R1 Reviewer B4 fix: use real load_cache_config(path), not non-existent from_dict.
    
    Marked @manual because needs cp1_spatial_pool_16.pkl (630 MB) in workspace.
    Locally: pytest -m manual tests/test_kinematic_super_warmup.py
    """
    from openpi.cache.config import load_cache_config
    from exp.weighted_sum.kinematic.super_warmup import build_super_warmup_yaml
    import yaml as _yaml
    
    yaml_dict = build_super_warmup_yaml()
    yaml_path = tmp_path / "super_warmup.yaml"
    yaml_path.write_text(_yaml.safe_dump(yaml_dict))
    cfg = load_cache_config(yaml_path)   # real validator path
    assert cfg.checkpoints["cp1"].judge.type == "always_warm_start"


@pytest.mark.manual
def test_eval_yaml_validates(tmp_path):
    """Manual: requires cp1_spatial_pool_16.pkl in workspace."""
    from openpi.cache.config import load_cache_config
    from exp.weighted_sum.kinematic.spec import generate_all_cells, build_eval_yaml_for_cell
    import yaml as _yaml
    
    cell = next(c for c in generate_all_cells() if c.group == "g5")
    yaml_dict = build_eval_yaml_for_cell(
        cell, fh_thr=0.5, ws_thr=0.1,
        super_raw_relpath="exp/weighted_sum/data/kinematic_phase5/super_warmup_raw.jsonl",
    )
    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(_yaml.safe_dump(yaml_dict))
    cfg = load_cache_config(yaml_path)
    assert cfg.checkpoints["cp1"].judge.calibration.samples_source.type == "offline"
```

### 7.2 Stage 3 HARD GATE (7 check, §4 Stage 3)

### 7.3 Smoke test（Stage 4 完成后，Stage 5 前 1 cell × 5 ep）

```bash
PYTHONPATH=. uv run exp/weighted_sum/kinematic/runner.py \
    --mode run-eval --cell-ids ws_d1_kin_g5_p1__fh0.2_ws0.5 \
    --servers weiland.top:14000 --eval-trials 5 \
    --strategy kinematic
```

预期：journal 出 1 行 success_rate；per_step/<yaml_id>.jsonl 出 ~5×35 = 175 行带 cp1_score + factor_outputs.calibrated；threshold yaml 内含 `samples_source.type=offline`。

### 7.4 主跑后 SR sanity

- 237 cell 全跑完，0 个 error（除合法 SKIP 通过 §4.2 fail-skip）
- per_yaml_summary 237 行
- G5 SR median ≥ 0.80（参 phase5 d4 G5 median 0.96，d1 检索预期略低）
- 若 G1-G4 ≥ 3 bucket decidable → H1/H2 共同验证；若全 null → H2 confirm，需在报告中明记并执行 fallback（§7.5）

### 7.5 fallback 计划（**R1B DESIGN-MAJOR #4 修复**）

如果 §7.4 显示 G1-G4 全 inconclusive：
- 挑 Δ 最大的 3 个 bucket 升 500 ep 复测 ≈ 1500 ep × ~3 ep/s ≈ 8 min/bucket × 3 = 24 min
- 若仍 null → 主结论之一："窄分布 d1 检索同样不让 kinematic factor inner-axis decidable"，与 phase5 d4 outer-axis G1-G4 全 null 互补证据

### 7.6 Pareto overlay 图最终判定

按 §7.1 frontier decision rule 写文字结论。

---

## 8. Commit / Push 收尾（**由 owner 指示时执行；agent 不擅自 git add**）

主跑完成 + 分析 + 报告**全部完成后**，由 owner 在 chat 指示是否 commit。本节为 commit 拆分**提议**，不是自主行动。

⚠ 前提：**确认前序未提交的 4 个 commit（orchestrator MISS-score / capacity-aware driver / run_phase2 --server-workers / threshold_pareto 分析）已先 land**，否则本 plan commit 边界混淆。

提议拆分（英文 message、无 Co-Authored-By Claude、author=`LinZiyang666 <3177267975@qq.com>` only）：

1. `feat(verdict): add spatial16_ws_d1_best CFG_SPECS for kinematic-on-weighted_sum`
2. `feat(weighted_sum): kinematic phase5 237-cell replication with super warmup (offline calibration mode)`
3. `feat(weighted_sum): KinematicSearchStrategy with per_step incremental flush`
4. `feat(run_phase2): --strategy switch (weight|kinematic)`
5. `feat(analysis): kinematic phase5 Pareto overlay + always-WARM anchor`
6. `test(kinematic): unit tests for spec / G5 grid filter / super factor coverage / phase5 regression`
7. `docs(weighted_sum): §4 kinematic phase5 comparison`
8. `docs(conductor_tutorial): document --strategy switch`（若 §4.3 path 选）
9. `exp(kinematic): results.md`

config yamls：仅 `super_warmup.yaml` 入库；`eval/*.yaml` 237 个**不入库**（emit 产物）。data/ 不入库。

---

## 9. 估算 & 时间表

| Stage | 内容 | 估时 |
|---|---|---|
| 0 | ziyang10 server 重启 + agentchat 通知 + ready watcher | 5 min |
| 1 | spec.py + v2_spec.py edit + 单测 | 1 h |
| 2 | super_warmup.py + emit + 跑 super warmup | 30 min + 15 min run |
| 3 | verify 7 check（offline pure python + bootstrap CI）| 5 min |
| 4 | strategy.py + runner.py + run_phase2 --strategy + emit 237 eval | 1 h |
| 4-smoke | 1 cell × 5 ep | 5 min |
| 5 | 237 cell × 100 ep dual-server 16/48 | ~3.0 h |
| 6 | always-WARM 3 yaml × 100 ep | 10 min |
| 7 | decision gate + 4 frontier overlay + bootstrap CI + report | 1 h |
| **总** |  | **~6.7 h** |

实验全程 **不重启 server**（除 Stage 0 一次性）。eval 全程**不重启 agent**（§16.8 红线）。

---

## 10. G1 已 APPROVED 的参数（2026-05-28 12:42 CDT）

以下 12 条参数在 G1 review 中被 reviewer 默许（没有相反 NEEDS REVISION 意见），作为本实验的执行契约。Code 阶段若需偏离需重启 G1。

1. **d1 best yaml 选择**：3 个并列 SR=74%，本 plan 选 C `vision_0@6_vision_1@50_robot_state@43__d1`。owner 可换 A/B（仅 1 字段 swap）。
2. **trials_per_task=15** （super warmup ≈ 3250 verdict，§2.13 实测口径）；fallback：≤ 3 次 +5 重跑；超 25 trial 仍失败时通知 owner。
3. **G5 grid 加 `fh+ws ≤ 0.9` 三角约束**（与 threshold_pareto §2.2 完全对齐）：16 → **15** (FH, WS)，仅排除 (0.5, 0.5)（fh+ws=1.0 唯一退化点）。每 recipe 15 cell × 3 recipe = 45 G5 cell。**总 cell 数 = 237**。owner 可拒（保留 (0.5, 0.5) 接受 T_ws=min(scores) 全 WS）或要求更严约束 `fh+ws < 0.9` (= 13 pairs)。
4. **decision gate 5pp rule** 沿用 phase5。若 owner 想升 3pp，需把 ep 升到 500 以达统计 power。
5. **always-WARM ceiling**：本 plan 自跑 3 cell × 100 ep（300 ep ≈ 10 min）。owner 可拒（直接引 phase5 d4 数据 + 报告显式注明 cross-retrieval caveat）。
6. **super warmup 跑 server**：本 plan 选 ziyang10 1 replica（fetch_dump 简单，无 cross-replica merge）；xuanle 3-replica 不参与 super warmup。
7. **eval 拓扑**：dual-server 16/48 跑 237 cell，跨 H200 NVL 两块（不同物理 GPU）；offline calibration 吸收大部分差异，红线 §16.6 报告中注明。owner 可改单 server eval（速度减半，~6h）。
8. **per_step flush 实现路径**：~~v2: strategy hack~~（G1 R1 B2 否决：`ctx.driver` AttributeError） → ~~v3: `on_stage_complete(driver=None)` kwarg~~（G1 R2 B1 否决：破坏 5 个已有 3-arg override） → **R4: driver 内部 flush**：`_complete_stage` 在 `strategy.on_stage_complete` 调用后自己调 `self._flush_per_step_for_stage(stage.yaml_id)`，strategy hook 签名 0 改动。src/openpi/ 共 ~14 行（仅 driver.py）。**owner 无替代选项**。
9. **commit 时机**：实验全做完 + 报告齐 + 与 owner chat 确认后才 commit；agent 不主动 `git add`。
10. **报告语言**：英文（与 threshold_pareto results.md 一致）。
11. **agentchat 通知机制**：本 plan 通知点：(a) Stage 0 重启 server 前；(b) Stage 3 verify 失败或 fallback 重跑触发；(c) Stage 5 主跑 milestone (25/50/75%) + DONE；(d) eval 异常 cell 失败率超阈值时。
12. **plan G1 触发流程**：本 plan §0+§1+§6+§10 摘要发到 trajectory 房间 + 文件链接 → owner 在 chat APPROVE 或 NEEDS REVISION。

---

## 11. R1 自审响应表（**新增，response to R1 4-agent peer review**）

| # | R1 Finding | 严重度 | Response |
|---|---|---|---|
| R1A-B1 | dump factor 算术错（77 实为 50）；ONLINE_DESCS 只有 2 | BLOCKER | §2.10 + §4 Stage 2.1 **由 `set().union(*[c.declared_keys ...])` 驱动**，不手算 |
| R1A-B2 | G5 declared_keys 必须 ⊆ super union 未验证 | BLOCKER | Stage 3 check #6 强制 237 cell bind_keys 验证 |
| R1A-B3 | `collect_all_declared_keys_from_spec` 函数不存在 + extract 阶段不应预过滤 | BLOCKER | 改函数名为 `super_warmup_declared_keys()` 与 spec 同源；extract 用此 union（§2.10） |
| R1A-B4 | `_mode_emit_eval_yamls` 缺 try/except | BLOCKER | §4.2 显式 per-cell try/except + SKIP json |
| R1A-B5 | WarmupPool process-global，per-connection rebuild 行为不被 §2.7 论证捕获 | BLOCKER | §2.7 **改用 offline calibration source 完全绕开 WarmupPool 链路** |
| R1A-M1 | strategy.on_stage_begin 才是正路 hook | MAJOR | ~~§2.9 + §4.1 改 `KinematicSearchStrategy` override~~ → R4: hook 完全不 override；per_step flush 由 driver 内部触发 (G1 R2 B1) |
| R1A-M2 | deferred dump 实际写盘 | MAJOR | §2.3 修正描述 + Stage 0 加 `--warmup_dump_root` |
| R1A-M3 | check #4 严格 `t_ws < t_fh` 在 tie 时假阳性 | MAJOR | Stage 3 check #5 改 `t_ws <= t_fh` |
| R1A-M4 | top_k=5 (warmup) vs top_k=1 (eval) score 行为 | MAJOR | §2.2 注释 winner.score 不受 top_k 影响 |
| R1A-m5 | `ThresholdJudge` 措辞 → composer.tier_thresholds | MINOR | §0 + §1.3 改正 |
| R1A-m6 | v2_spec 全局 CFG_SPECS 副作用 | MINOR | §1.1 注 _FIELD_SIM_DEFAULT 不影响 + §7.1 加 regression test |
| R1A-m7 | (5,0)(7,7) boundary NaN 没显式验 | MINOR | Stage 3 check #3 加 critical key 分类报警 |
| R1B-DC1 | trials_per_task=5 verdict 数错 5× | DESIGN-CRITICAL | §2.13 实测 21.7 verdict/ep → trials_per_task=15 |
| R1B-DC2 | G5 (fh+ws≥1) 退化 cell | DESIGN-CRITICAL | §2.12 + §3.1 加 `fh+ws ≤ 0.9` filter → 15 cell/recipe (G1 R1 B1 算术更正后；v2 错算 11) |
| R1B-DC3 | always-WARM 跨检索复用错 | DESIGN-CRITICAL | §1.5 + Stage 6 自跑 3-cell ceiling |
| R1B-DM4 | super warmup 共享导致 sampling-noise 跨 cell 相关 | DESIGN-MAJOR | §1.2 H3 假设 + Stage 3 check #7 bootstrap CI |
| R1B-DM5 | G1-G4 hypothesis 未明 | DESIGN-MAJOR | §1.2 H1/H2/H3 表格 + §7.5 fallback |
| R1B-DM6 | frontier 比较 decision rule 没量化 | DESIGN-MAJOR | §7.1 明文规则 |
| R1B-DM7 | 16-cell vs 83-cell grid 不对称 | DESIGN-MAJOR | §7.1 frontier rule 不依赖 grid 密度；overlay 图同 cell 数策略 |
| R1B-DM8 | d1 best 3 选 1 没看分布形状 | DESIGN-MAJOR | §2.1 注释 + G1 owner 可换 |
| R1B-Dm9 | warmup 协议 force-MISS vs always_warm | DESIGN-MINOR | §1.5 与 threshold_pareto 协议差异 caveat |
| R1B-Dm10 | super warmup 消除 warmup-noise confound 是方法论收益 | DESIGN-MINOR | §1.2 H3 体现 |
| R1B-Dm11 | 100 ep 5pp barely conclusive | DESIGN-MINOR | §10 #4 升 3pp 选项 |
| R1B-Sug1 | dense G6 grid | SUGGESTION | §10 #6 owner 可选扩 |
| R1B-Sug2 | bootstrap CI 产物 | SUGGESTION | Stage 3 check #7 + §7.6 |
| R1C-B1 | hook 错位 run_phase2 → strategy | BLOCKER | §4.1 KinematicSearchStrategy |
| R1C-B2 | per_step 内存累积 | BLOCKER | ~~§4.1 on_stage_complete flush~~ → R4: `driver._flush_per_step_for_stage` 内部触发 + run() end 兜底 (G1 R2 B1) |
| R1C-B3 | WarmupPool LRU 必撞 | BLOCKER | §2.7 offline calibration 路径完全绕开 |
| R1C-B4 | dual-server super warmup + fetch_dump 未配 + 跨 GPU | BLOCKER | §1.5(自跑 ceiling) + Stage 0(--warmup_dump_root) + R3 caveat |
| R1C-M1 | super_warmup_raw 流转 | MAJOR | §2.3 flow chart 明文 |
| R1C-M2 | v2_spec CFG_SPECS 全局回归 | MAJOR | §7.1 加 regression test |
| R1C-M3 | G5 declared_keys 显式核对 | MAJOR | §2.10 + Stage 3 check #6 |
| R1C-M4 | monitor/cron 三件套 yaml prefix | MAJOR | §11 监控脚本以 `ws_d1_kin_` 前缀重写 |
| R1C-M5 | tether 操作链 / 端口管理 | MAJOR | §2.3 数据流转 + Stage 0 命令模板 |
| R1C-M6 | yaml_id naming 与 summarize 兼容 | MINOR | summarize 工具 yaml_id-agnostic ✓ |
| R1C-M7 | super_buffer 不经 worker 路径 | MINOR | §2.7 + §4.1 driver-side only |
| R1C-M8 | super warmup 内存量 | MINOR | §6 R4 + 实测 50 factor × 3250 verdict ≈ ~15MB / 1 episode |
| R1D-B1 | check #4 与 G5 grid 自相矛盾（删退化 cell）| BLOCKER | §2.12 G5 grid filter + Stage 3 check #5 改 `t_ws <= t_fh` |
| R1D-B2 | G3 jerk-only/disp-only pattern derive_thresholds 抛 | BLOCKER | §4.2 emit-eval try/except skip |
| R1D-B3 | 跨 GPU caveat 漏 | BLOCKER | §6 R3 + §5 stage 5 + 红线 §16.6 引用 |
| R1D-M1 | 红线 §16.3 agentchat 必回 plan 缺 | MAJOR | §10 #11 通知机制 |
| R1D-M2 | 红线 §16.4 起 server 喊 owner | MAJOR | Stage 0.1 agentchat 通知模板 |
| R1D-M3 | 红线 §16.5 commit 表述像 self-commit | MAJOR | §8 改 "由 owner 指示时执行" + 前提条款 |
| R1D-M4 | 红线 §16.6 跨 GPU 注明 | MAJOR | §5 + §6 R3 |
| R1D-M5 | super warmup server 选择缺 G1 项 | MAJOR | §10 #6 |
| R1D-M6 | WarmupPool LRU bind lazy/eager 验证 | MAJOR | 改 offline 路径已绕开 |
| R1D-M7 | super warmup partial raw 卡 cache | MAJOR | §4.2 `_line_count ≥ 2500` strict |
| R1D-M8 | R4 OOM 评级偏低 | MAJOR | §6 R4 升 Low→Medium + 实际 50 factor 非 77 |
| R1D-M9 | dump factor 算 24（实测 18）| MAJOR | §2.10 + §4.2 用 declared_keys union 不手算 |
| R1D-Mn1 | (5,0) past-only finite verify | MINOR | Stage 3 check #3 |
| R1D-Mn2 | §10 不完整 | MINOR | §10 补 #6-#12 |
| R1D-Mn3 | config/eval *.yaml 不入 git | MINOR | §3 注 + §8 commit list 修正 |
| R1D-Mn4 | docs/conductor_tutorial 同步 | MINOR | §5 + §8 commit #8 |
| R1D-Mn5 | unit test fixture pkl 路径 | MINOR | §7.1 `@pytest.mark.manual` |
| R1D-Mn6 | §1.3 d1 3 选 1 锁定 | MINOR | §1.3 + §10 #1 |
| R1D-Mn7 | transformers_replace overlay 隐含依赖 | MINOR | Stage 0 注：限 ziyang10+xuanle (已配) |
| R1D-G1 | tyro 顺序 | GAP | Stage 0 引用 §session_handoff §15 #1 |
| R1D-G2 | pkill 自匹配铁律 | GAP | Stage 0 + §11 引用 §session_handoff §4 #5 |
| R1D-G3 | expose keepalive timeout | GAP | §6 R10 |
| R1D-G4 | G3 共享 warmup 自然吸收 | GAP | §1.2 注释 |
| R1D-G5 | G5 yaml_id rename 正确性 | GAP | §7.1 unit test |
| R1D-G6 | warmup verify fail 自动 fallback | GAP | §10 #2 + §6 R1 |
| R1D-G7 | agentchat plan 摘要发布 | GAP | §10 #12 |

---

## 12. tether / 设备 / 监控操作（**R1C MAJOR + R1D GAP 补齐**）

详见 session_handoff §3-§8（不重复）。本实验特定：

### 12.1 端口管理
- ziyang10:8000 ↔ weiland.top:14000（expose name=`ziyang-srv`）
- xuanle:8000 ↔ weiland.top:14001（expose name=`xl-srv`）
- 全程**只 Stage 0 重启 ziyang10 server 一次**（加 `--warmup_dump_root`）；其他时刻 server 不重启

### 12.2 监控三件套（本实验定制版）

健康脚本路径 `/tmp/kin_health.sh` on timan107：
```bash
#!/bin/bash
J=/scratch/zixuans8/openpi/exp/weighted_sum/data/kinematic_phase5/journal.jsonl
TOTAL=237
done=$(grep -cE '"status"' "$J" 2>/dev/null || echo 0)
ok=$(grep -cE '"success": ?true' "$J" 2>/dev/null || echo 0)
cond=$(pgrep -fc "[r]un_phase2.*kinematic" 2>/dev/null || echo 0)
workers=$(pgrep -fc "[l]ibero_sim" 2>/dev/null || echo 0)
s1=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14000" 2>/dev/null && echo UP || echo DOWN)
s2=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14001" 2>/dev/null && echo UP || echo DOWN)
# expose tunnel keepalive check (R1D GAP #3)
exp_check=$(tether ps -a 2>/dev/null | grep -E "ziyang-srv|xl-srv" | wc -l)
errs=$(grep -ciE "traceback|connection refused|out of memory|EGL|CUDA error" /tmp/kin_run.log 2>/dev/null || echo 0)
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
echo "[$(date '+%H:%M:%S')] progress=$done/$TOTAL (${pct}%) success=$ok cond=$cond workers=$workers s1=$s1 s2=$s2 expose=$exp_check err=$errs"
[ "$done" -ge "$TOTAL" ] && echo "EVAL DONE"
[ "$s1" = DOWN ] && [ "$s2" = DOWN ] && echo "ALERT both servers DOWN"
[ "$exp_check" -lt 2 ] && echo "ALERT expose tunnel dropped (run: tether expose ...)"
[ "$cond" -eq 0 ] && [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ] && echo "ALERT conductor dead"
```

- 条件触发 Monitor：milestone (~25/50/75%) + ALERT 即唤醒
- cron 静默版：每 10min 巡检，progress≥TOTAL 自删
- agentchat Monitor：保持现存 b1e2lo9ge（owner 消息必回）

### 12.3 pkill 铁律（§session_handoff §4 #5）

清 conductor 必用 char-class + 两条 exec：
```bash
tether exec timan107 -- bash -lc '
  tmux kill-session -t run0 2>/dev/null
  pkill -f "[r]un_phase2" 2>/dev/null
  pkill -f "[w]orker_entry" 2>/dev/null
  pkill -f "[l]ibero_sim" 2>/dev/null
  sleep 4
  echo "cond=$(pgrep -fc \"[r]un_phase2\") workers=$(pgrep -fc \"[l]ibero_sim\")"
'
```

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-28 12:59 CDT

- [Blocking] [Constitutional] The working-tree plan file deleted the prior Review Log section and all recorded G1 review history. — reasoning: the staged version contains `## Review Log` plus G1 R1-R4 entries, but the current working tree ends at §12.3 and removes that section entirely. Review history is part of the audit trail; G2 cannot approve a code-phase package that drops the existing gate record.
- [Blocking] [Concern] `kinematic.runner --mode run-warmup` builds an invalid LIBERO command from default `--task-ids`. — reasoning: `exp/weighted_sum/kinematic/runner.py:503` stores task ids as the string `"0,1,2,3,4,5,6,7,8,9"`, `_build_libero_args_from_cli()` passes the argparse namespace through unchanged, and `exp/verdict_factor_judge/common/run_phase.py:241-243` iterates `args.task_ids`. Reproduction with `uv run python` produced `--task-ids 0 , 1 , 2`, so Stage 2 super warmup defaults are not executable. Parse to `tuple[int, ...]` before calling `_build_libero_argv`, matching phase5 runner's parser.
- [Non-blocking] [Suggestion] Add a focused unit test for the kinematic runner LIBERO argv adapter. — reasoning: the existing `tests/test_kinematic_super_warmup.py` covers cell generation and yaml structure, while this CLI handoff bug sits between argparse and the shared run_phase helper.

Validation:
- `PYTHONPATH=. uv run pytest tests/test_kinematic_super_warmup.py tests/conductor/test_driver.py -q -m "not manual"` passed: 26 passed, 2 deselected.
- `PYTHONPATH=. uv run python -c ..._build_libero_argv...` reproduced the malformed `--task-ids 0 , 1 , 2` command.

### G2 Round 1 — Executor — 2026-05-28

> [Blocking] [Constitutional] The working-tree plan file deleted the prior Review Log section and all recorded G1 review history. ...

- **Rejected** — `protocols/execution_authority.md` explicitly mandates G1 Review Log deletion at the post-G1 polish step; this is constitutional protocol compliance, not a violation. Cited verbatim:
  - **§3.1 step 2**: "Delete the entire `## Review Log` section — every G1 reviewer entry, every G1 executor response, every G1 round header."
  - **§10.1 line 269**: "The Log is append-only within each gate's rounds. Deleting, rewriting, or reordering existing entries is a violation — **with one exception: the Post-G1 polish (§3.1) deletes the entire G1 Review Log section en bloc before §4 Code begins.**"
  - **§10.1 line 270**: "G2 opens a fresh `## Review Log` section on the polished plan and its contents persist in the plan file after G2 APPROVED — there is no post-G2 polish." — exactly the section this G2 Round 1 entry was appended to.
  
  Additionally, the factual claim "the staged version contains `## Review Log` plus G1 R1-R4 entries" was independently verified false via git:
  - `git show :logs/weighted_sum_kinematic_phase5_replication.log.md | grep -cE "^### G1 Round"` → **0** (staged copy has 0 G1 Round entries)
  - `grep -cE "^### G1 Round"` on the working tree → **0** (working tree also has 0)
  - `git ls-tree HEAD logs/weighted_sum_kinematic_phase5_replication.log.md` → empty (file not yet in HEAD)
  
  Both index and working-tree copies match the polish requirement. Reviewer's stated premise contradicts both the protocol text and the actual git state; rejecting per §10.2 step 4 with reasoning grounded in the cited §3.1 / §10.1 lines. If the WA / protocol designers intend G1 audit-trail to persist, that is a protocol-change discussion separate from this gate, not a G2 blocker against the present diff.

> [Blocking] [Concern] `kinematic.runner --mode run-warmup` builds an invalid LIBERO command from default `--task-ids`. ...

- **Accepted** — Bug verified by reading `exp/verdict_factor_judge/common/run_phase.py:243-244` (`for t in args.task_ids: str(t)` iterates the string character-by-character → `--task-ids 0 , 1 , 2 ...`). Fix in `exp/weighted_sum/kinematic/runner.py`:
  1. Added pure helper `_parse_task_ids(raw: str) -> tuple[int, ...]` mirroring `phase5/runner._parse_args` line 156-157.
  2. Rewrote `_build_libero_args_from_cli` to wrap the argparse Namespace in a `types.SimpleNamespace` clone with `task_ids` already parsed to `tuple[int, ...]`. Chosen over in-place `args` mutation to avoid subtle aliasing if a caller re-uses the Namespace after the LIBERO command builds.
  3. Forwards only the fields `run_phase._build_libero_argv` actually reads (host, port, task_suite, num_workers, task_ids, per_step_log_dir, init_states_dir, episode_filter, cuda_visible_devices, conda_env, preload_pkl_override), so a future schema drift in argparse defaults can't leak through unnoticed.

> [Non-blocking] [Suggestion] Add a focused unit test for the kinematic runner LIBERO argv adapter. ...

- **Accepted** — Added two unit tests in `tests/test_kinematic_super_warmup.py`:
  - `test_runner_parse_task_ids_csv` — pure helper: default CSV → `(0, 1, ..., 9)`; empty → `()`; whitespace-tolerant; all elements `int`.
  - `test_runner_libero_args_emits_well_formed_task_ids` — end-to-end through `_parse → _build_libero_args_from_cli` plus a reproduction of the `run_phase.py:244` argv emit loop, asserting the tokens are exactly `["--task-ids", "0", "1", ..., "9"]` (no comma tokens). Includes a negative regression sentinel confirming the OLD raw-string path WOULD yield comma tokens.
  
  Local pytest: `28 passed, 2 deselected` (up from 26 passed by adding these two tests; full suite `tests/test_kinematic_super_warmup.py` + `tests/conductor/test_driver.py` re-ran clean).

### G2 Round 2 — Reviewer — APPROVED — 2026-05-28 13:07 CDT

- [Resolved] G2 R1 task-id blocker is fixed. `_build_libero_args_from_cli` now adapts the argparse namespace and converts `task_ids` to `tuple[int, ...]`; a direct `_build_libero_argv` reproduction emits `--task-ids 0 1 2 3 4 5 6 7 8 9`, not comma tokens.
- [Resolved] The G1 Review Log removal is accepted as post-G1 polish under the cited protocol exception; the active G2 Review Log is present and append-only from this point.
- [Checklist] Plan consistency: matches the G1-approved 237-cell kinematic-on-weighted_sum implementation path, including local G5 filtering, super warmup offline calibration, driver-internal per-step flush, and `run_phase2 --strategy kinematic` writer wiring.
- [Checklist] Test coverage: kinematic spec/yaml/runner adapter tests plus conductor driver regression tests pass.
- [Checklist] Docs/index: `logs/README.md` and the plan status reflect the G1-approved code phase; no G2-blocking stale operator instruction found.
- [Checklist] Regression risk: no strategy hook signature change; legacy `weight` path keeps `per_step_writer=None`; the new runner adapter is scoped to warmup subprocess argv construction.

Validation:
- `PYTHONPATH=. uv run pytest tests/test_kinematic_super_warmup.py tests/conductor/test_driver.py -q -m "not manual"` passed: 28 passed, 2 deselected.
- `PYTHONPATH=. uv run python -c ..._build_libero_argv...` confirmed well-formed task-id argv.
