# Plan: Phase 5 Systematic Sweep on libero_10

**Status**: Plan (G1 APPROVED 2026-05-22 / §4 Code 完成 / G2 Round 1 NEEDS REVISION → Executor R1 applied → 待 Reviewer Round 2)
**Level**: **L2**（需改 `exp/verdict_factor_judge/common/v2_spec.py` + `phase5/spec.py` + `phase5/runner.py`，并新增一个 G5-warmup driver 脚本；`src/openpi/` 不动）
**Date**: 2026-05-22
**Owner**: Ziyang Lin（执行由 Claude Execution Authority 辅助）

> `v2_spec.py:80` 把 `preload_pkl` 写死成 `libero_spatial/cp1_spatial_pool_16.pkl`，task_suite 切换无法仅靠 CLI 完成——必须在 `v2_spec.build_*_yaml` / `phase5.spec.build_*_for_cell` / `phase5.runner` 三层引入 `preload_pkl_override` keyword-only 透传。按 [`WORKING_AGREEMENT.md` §2.4/§2.6](../WORKING_AGREEMENT.md) 走 G1/G2 强制 gate。

---

## 1. 目标与范围

在 **`libero_10` task_suite** 上复现 phase5 systematic sweep — 5 group × 48 cell × 100 ep × 1 seed = **240 cell / 24 000 episode**，除 task_suite 切换外所有实验参数与 libero_spatial 版本（`logs/verdict_phase5_systematic_sweep.log.md` / `data/phase5_systematic/`）严格一致。

**用户决定**（2026-05-22）：
- **Q1 = (a) 完整重建 G5 依赖的 phase3/phase4 warmup raw on libero_10**（narrow 到 3 个 warmup yaml，并非 phase3 全 11 个 + phase4 全 2 个）
- **Q2 = (a′) 新数据目录 + 在 spec / runner 引入 `--preload-pkl-override` CLI knob**：把 `preload_path` 注入到 emit 出的所有 yaml 里。`task_suite` 与所有数据落盘路径走 CLI override。该修订源自 `v2_spec.py:80` 的 `preload_pkl` hardcoded 到 libero_spatial，纯 CLI override 不可达 yaml 内容字段（详见 §2.6）。
- **Q3 = (b) 跳过 always-warm / pure-inference / random_periodic baseline 重建**（只看 240 cell 的 SR；不画 Pareto，不画 heatmap）

**不在范围内**：
- `src/openpi/cache/...` 任何改动（仍坚持 zero src/ change；只动 `exp/verdict_factor_judge/` 下的 spec / runner / common）
- libero_10 上的 baseline_failures.json / always-WARM / random_periodic gate sweep 重建
- `analysis/phase5/plot_pareto.py` / `plot_heatmaps.py` 任何调用（缺 baseline 画不出图）
- 其他 5 个 pkl builder（`cp1_max_pool` / `cp1_mean_pool` / `cp1_spatial_pool_64` / `clip_vit_b_32` / `clip_vit_l_14`）的 enrich — phase5 spec 只读 `cp1_spatial_pool_16`，其他保持原状

---

## 2. 前置事实（调研结论）

### 2.1 libero_10 pkl 当前状态

| 字段 | libero_10 现状（Apr 21 build） | phase5 需求 |
|---|---|---|
| `entries` 数量 | 2640（50 trajectory × ~53 step） | ✅ 够用 |
| `entries[i].payload.action_chunk` | tensor `[10, 32]` | ✅ |
| `entries[i].payload.intermediates` | dict (8 个 timestep) | ✅ |
| `entries[i].payload.task_key` | str | ✅ |
| **`entries[i].payload.factors`** | **None** | ❌ 需要 64 个 offline factor keys |
| **`library_stats`** | **缺（key 不存在）** | ❌ `state_active_mask`/`action_active_mask` 在 `cache/config.py:1759` 是强制校验 |

> phase5 所有 cell 都含 state/action offline factor，`cache/config.py:702/774/1752/1759` 在 yaml load 阶段就会因 `library_stats is None` 抛错——**不补这两块 phase5 一行都跑不起来**。

### 2.2 enrich-existing-pkl 子命令可救场

`exp/common/build_in_memory_cache_artifact.py:840` 已有 `enrich-existing-pkl` 子命令：读现有 pkl entries → 跑 OfflineWriter 算 `payload.factors` → 重算 `library_stats` → 写回 pkl。**完全不需要重跑 LIBERO 仿真采集 h5**。phase3 在 libero_spatial 上就是这条路（见 `archive/libero_spatial_factor_artifact_rebuild.log.md` 和 `verdict_phase3_run_commands.log.md` §0）。

### 2.3 G5 历史 warmup raw 依赖

`phase5/spec.py:472-480` 把 G5 三个 recipe 的 warmup_yaml_id 写死指向 phase3/phase4：

| Recipe | 来源 | warmup_yaml_id |
|---|---|---|
| `p1_state_fut_online_act` | phase4 | `spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup` |
| `p2_action_fut_online_act` | phase4 | `spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup` |
| `g6_f1a_a_d_jerk_curv_pair` | phase3 | `spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup` |

**只需重跑这 3 个**（不是 phase3 全部 11 个 + phase4 全部 2 个）。warmup yaml 文件本身与 task_suite 无关（只描述 cp1_spatial_pool_16 + 因子 + AlwaysWarmStartJudge），可直接复用 libero_spatial 的；变的是 `--task-suite libero_10` 与输出 raw 路径。

### 2.4 runner CLI 已部分支持 path override，但 preload_pkl 是缺口

`phase5/runner.py:_parse_args` 已暴露以下参数（默认值全部写死 libero_spatial 数据位置，**路径**可通过 CLI 完整 override）：
- `--task-suite` → `libero_10`
- `--warmup-jsonl-dir` → phase5 自身 warmup raw 输出位
- `--phase3-warmup-dir` / `--phase4-warmup-raw-dir` → G5 raw 源
- `--summary-out` / `--episode-results-dir` / `--per-step-log-dir`
- `--eval-yaml-dir` / `--warmup-yaml-dir`

**但 yaml 内容里的 `backend.in_memory.preload_path` 字段是 spec 注入、CLI 未覆盖**——这正是核心缺口（见 §2.6）。

### 2.5 G1-G4 warmup 是 phase5 lazy 创建的，**但 G3 共享 warmup 有并发 race**

`phase5/runner.py:_ensure_warmup_for_cell` (line 473) 在 `--mode run-eval` 时若发现 cell 的 factor_raw 不存在会自动 lazy-emit + run warmup。但实现没有 inter-process lock（line 489-491 只是 `if raw_path.exists() and size > 0: return`，否则直接调 `_run_one_warmup`）。

- **G1 / G2 / G4**：1 cell ↔ 1 distinct `warmup_yaml_id`，`allocate_to_servers` 按 sorted cell_id 切片，每 server 内 warmup 独占 → 无 race
- **G3**：spec.py:382 `warmup_yaml_id = f"{cfg_id}_phase5_g3_{base}_{channel}__warmup"` — 12 patterns × 2 base × 2 channel 中，每 (base, channel) bucket 内 12 个 cell 共享 1 个 warmup。allocate_to_servers 不感知共享，**12 个共享 cell 可能被切到不同 server，并发 lazy-emit 同一 warmup yaml + 写同一 factor_raw jsonl**

→ 修复见 §4 Stage 1.5（新增）：把 G3 共享 warmup **预跑**，run-eval 阶段 G3 cell 永远走 file-exists fast path。

### 2.6 preload_pkl hardcoded 到 libero_spatial（核心 code-change 触发点）

`exp/verdict_factor_judge/common/v2_spec.py:80`：

```python
CFG_SPECS["spatial16_w8_d4"] = {
    ...
    "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
    ...
}
```

`v2_spec.build_eval_yaml` (line 368) / `build_warmup_yaml` (line 426) 都把 `cfg["preload_pkl"]` 写到 emit yaml 的 `backend.in_memory.preload_path`。**phase5/phase4/phase3 所有 emit yaml 都指向 libero_spatial pkl**——即使 enrich 了 libero_10 pkl，server load 的还是 libero_spatial（因为 yaml 这么说）。

**修复（属于 Code 阶段，§4 Stage 0.5）**：
- `common/v2_spec.py`：`build_eval_yaml` / `build_warmup_yaml` 接 keyword-only `preload_pkl_override: str | None = None`，None 时取 `cfg["preload_pkl"]` (向后兼容)，否则用 override 值
- `phase5/spec.py`：`build_warmup_yaml_for_cell` / `build_eval_yaml_for_cell` 透传 `preload_pkl_override`
- `phase5/runner.py`：新增 `--preload-pkl-override` CLI；`Args` 加同名字段；`build_*_for_cell` 调用处透传
- 同样的 override 也用于 §4 Stage 1 的 G5-warmup driver（独立 driver 也需要 patch yaml）

### 2.7 phase3 / phase4 现成 runner 不能直接复用于 G5 warmup 重建

| # | 事实 | 影响 |
|---|---|---|
| a | `common/run_phase._iter_eval_yamls` 只扫直接子项 + 跳过 `*__warmup`，`config/spatial16/phase3/*.yaml` 直接子项 = 0 | 用 `common.run_phase --phase-dir phase3` 跑 g6 warmup → 发现 0 eval yaml，整轮 no-op |
| b | `phase4/runner.py:151` `--round` required choices=[1,2,3,4]；`--recipe` 单值 | `--recipe p1,p2` + 无 `--round` 直接被 argparse 拒掉 |
| c | `phase4/runner._run_warmup_for_recipe` line 569：`raw_dst = _warmup_raw_dir() / f"{recipe_id}.jsonl"`，`_warmup_raw_dir()` 返回 hardcoded `data/phase4/warmup_factor_raw`，**无 CLI override**；并且 line 570-572：`if raw_dst.exists() and size > 0: return` — 已有 libero_spatial raw 会让 libero_10 直接 skip | `--warmup-jsonl-dir` 控制的是 fetch_dump 的中间产物路径，不是最终 factor_raw 落盘位；现状下根本写不到 `data/phase4_libero10/warmup_factor_raw/` |

**修复**：放弃复用 phase3 / phase4 runner，写一个 **`exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py`** 统一处理 3 个 G5 warmup yaml。Driver 内联调用 `phase5/runner._run_one_warmup` 的核心步骤（server load → spawn libero workers → fetch_dump → `_extract_finite_factor_raw`），但路径由 CLI 完全控制。

---

## 3. 产物目录布局

新建以下目录（与 libero_spatial 完全分流，避免覆盖）：

```
exp/verdict_factor_judge/
  config/spatial16/phase5_libero10/
    warmup/                                # phase5 lazy-emit 出的 G1-G4 warmup yaml（~148）
    eval/                                  # phase5 lazy-emit 出的 240 个 eval yaml
  data/
    phase3_libero10/
      warmup/                              # phase3 g6 warmup raw on libero_10（1 个 jsonl）
    phase4_libero10/
      warmup_factor_raw/                   # phase4 p1 / p2 warmup raw on libero_10（2 个 jsonl）
    phase5_libero10_systematic/
      warmup_factor_raw/                   # G1-G4 warmup raw on libero_10
      per_step/                            # 240 cell × per-step jsonl
      episode_results/                     # 240 cell × episode_results.json
      per_yaml_summary.jsonl               # 240 行最终 summary
      g{1..5}_decision.json                # decision-gate dump
      thresholds/                          # 240 cell 阈值快照
exp/common/data/cache_artifacts/libero_10/
  cp1_spatial_pool_16.pre_phase5.bak.pkl   # Stage 0 备份
  cp1_spatial_pool_16.pkl                  # Stage 0 enrich 后的新版本
```

**Tracking 策略**：
- `data/**` 全部走 `.gitignore`（参 `docs/experiments/artifact_layout.md` §3）。
- pkl 不 commit（libero_10 pkl 历来不入库；Stage 0 产物保留在本地）。
- `config/spatial16/phase5_libero10/{warmup,eval}/*.yaml` 可入库（参 libero_spatial 版本已 commit），但因为是 lazy-emit，每次跑都会重新生成；commit 时建议跑完 Stage 2 后一次性纳入。

---

## 4. 执行步骤

> **Stage 依赖顺序（strict）**：Stage 0（pkl enrich，独立可先做）+ Stage 0.5（code 改动，必须 G2 APPROVED）→ Stage 1（G5 warmup driver）→ Stage 1.5（G3 共享 warmup）→ Stage 2（240 cell run-eval）→ Stage 3（stats）。Stage 1 起的所有仿真都依赖 Stage 0.5 已 land 且 G2 APPROVED；先开仿真后 land code 会让 emit yaml 仍指向 libero_spatial pkl，整轮报废。

### Stage 0 — pkl enrich on libero_10/cp1_spatial_pool_16.pkl（~5 min CPU，与 Stage 0.5 可并行）

> 完全 mirror phase3 在 libero_spatial 上做的事情（见 `verdict_phase3_run_commands.log.md` §0）。
>
> **Server-side pkl 同步**：如果远端 server 与本机不共享文件系统，enrich 完成后必须把 `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl` 同步到 server 端的同路径（`rsync` / scp），并确认 server 进程能读到（必要时重启 server）。否则 Stage 1 一开始 `load_cache_config` 就会 raise `library_stats is None`。见 §6 R10。

```bash
# 0.1 备份原 pkl（Apr 21 build，无 library_stats / payload.factors）
cp exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
   exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pre_phase5.bak.pkl

# 0.2 写 8 offline factor × 8 window = 64 keys 的 enrich yaml
# 注意 YAML flow-mapping `:` 后必须有空格（phase3 踩过的坑）
cat > /tmp/phase5_libero10_offline_factors.yaml <<'YAML'
factors:
  - type: jerk_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: jerk_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: direction_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: direction_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: dispersion_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: dispersion_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: path_length_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: path_length_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
YAML

# 0.3 in-place enrich
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --factors-yaml /tmp/phase5_libero10_offline_factors.yaml \
  --output exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl
```

**Smoke gate（必跑）**：

```bash
uv run python -c "
import pickle
d = pickle.load(open('exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl','rb'))
e0 = d['entries'][0]
fac = e0.payload.factors
expected = {f'{desc}_offline_{ch}__p{p}_f{f}'
            for desc in ('jerk','direction','dispersion','path_length')
            for ch in ('action','state')
            for (p,f) in [(0,3),(0,5),(1,1),(2,2),(3,3),(5,5),(7,7),(3,0)]}
missing = expected - set(fac.keys())
ls = d.get('library_stats')
print(f'entries: {len(d[\"entries\"])} (expect 2640)')
print(f'factors per entry: {len(fac)} (expect >= 64)')
print(f'new keys missing: {missing or \"none\"}')
print(f'library_stats: {type(ls).__name__ if ls else \"None\"}')
print(f'  state_active_mask non-zero: {(ls.state_active_mask.sum().item() if ls else 0)}')
print(f'  action_active_mask non-zero: {(ls.action_active_mask.sum().item() if ls else 0)}')
"
# 期望:
#   entries: 2640
#   factors per entry: 64
#   new keys missing: none
#   library_stats: LibraryStats
#   state_active_mask non-zero: > 0
#   action_active_mask non-zero: > 0
```

### Stage 0.5 — Code 改动（§2.6 + §2.7 修复，L2 升级的根因）

> 这是 §2.6 + §2.7 揭示的 hardcoded preload_pkl / runner-API 缺口的修复，必须在 Stage 1 之前 land 并通过 G1/G2。改动总量 ≤ 150 行，向后兼容（不传新参数时行为与 libero_spatial 一致）。

**文件 1：`exp/verdict_factor_judge/common/v2_spec.py`**

- `build_warmup_yaml(cfg_id, eval_yaml_id, eval_factors, *, preload_pkl_override: str | None = None)`：若 override 非 None，写到 emit yaml 的 `backend.in_memory.preload_path`；否则 fallback `cfg["preload_pkl"]`
- `build_eval_yaml(cfg_id, factors, composer, ..., *, preload_pkl_override: str | None = None)`：同上
- 不动 `CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]` 默认值（libero_spatial 路径），保证现存 phase3/phase4/phase5 libero_spatial yaml emit 行为零回归

**文件 2：`exp/verdict_factor_judge/phase5/spec.py`**

- `build_warmup_yaml_for_cell(cfg_id, cell, *, preload_pkl_override=None)`：透传到 `v2_build_warmup_yaml`
- `build_eval_yaml_for_cell(cfg_id, cell, fh_thr, ws_thr, *, preload_pkl_override=None)`：透传到 `v2_build_eval_yaml`

**文件 3：`exp/verdict_factor_judge/phase5/runner.py`**

- `Args` dataclass 加 `preload_pkl_override: str = ""`
- `_parse_args` 加 `p.add_argument("--preload-pkl-override", default="")`
- `_mode_emit_warmup_yamls` / `_ensure_warmup_for_cell` / `_run_one_warmup` 内部调用 `build_warmup_yaml_for_cell(... preload_pkl_override=args.preload_pkl_override or None)`
- `_mode_emit_eval_yamls` / `_ensure_eval_yaml_for_cell` 同样透传
- 不传 / 传空串 → 默认走 libero_spatial 路径

**文件 4：新建 `exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py`**

```python
"""G5 historical warmup raw rebuild on libero_10.

Phase 3 / phase 4 runners can't be reused here:
  - common.run_phase._iter_eval_yamls 只扫直接子项, phase3/ 顶层没 .yaml
  - phase4.runner --round required, --recipe 单值, raw_dst hardcoded

This driver loads the 3 historical warmup yaml files (phase3 g6 +
phase4 p1/p2), patches `backend.in_memory.preload_path` to point at
the libero_10 pkl, runs each warmup against --task-suite libero_10,
fetches the dump, and extracts finite-only factor_raw to caller-
supplied output paths.

CLI (complete flag set — must match Stage 1 invocation in §4):
    uv run python -m exp.verdict_factor_judge.phase5.g5_warmup_libero10_driver \
      --host ...  --port ...  --warmup-trials 2  --num-workers 5  --task-suite libero_10 \
      --preload-pkl       exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
      --phase3-warmup-in  exp/verdict_factor_judge/config/spatial16/phase3/warmup \
      --phase4-warmup-in  exp/verdict_factor_judge/config/spatial16/phase4/warmup \
      --phase3-warmup-out exp/verdict_factor_judge/data/phase3_libero10/warmup \
      --phase4-warmup-out exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw

Naming note: this driver uses `--preload-pkl` (no fallback semantics needed
— driver always patches the path), while `phase5/runner.py` uses
`--preload-pkl-override` (override of an existing default). The two flag
names are intentional and not interchangeable.
"""
```

实现要点：
- 直接 `from exp.verdict_factor_judge.phase5.runner import _extract_finite_factor_raw`（私有 helper，**当前不在 `phase5/runner.py.__all__` 列表里**——driver 走 module-import 直接拿，不在本 plan 范围内动 `__all__`，避免无关接口扩面）
- 同样 `from exp.verdict_factor_judge.phase3.runner import _save_dump_jsonl`（同上私有 helper，已被 phase5/runner.py 引用过，沿用同一模式）
- 复用 `exp.verdict_factor_judge.common.run_phase._build_libero_argv` 构造 LIBERO worker 命令。**注意**：该 helper 除 `host/port/task_suite/num_workers/warmup_trials/eval_trials` 之外还会读 `task_ids` / `init_states_dir` / `episode_filter` / `cuda_visible_devices` / `conda_env` / `per_step_log_dir` 六个可选字段。Driver 不要自己手搓裸 dataclass——直接复用 `phase5.runner.Args` (或一个 mirror 子集) 并对未暴露的 CLI 字段提供安全默认（empty string / empty tuple），保证 driver smoke test 走的就是真实 builder 路径，不会因属性缺失 raise `AttributeError`
- yaml patch 用 `yaml.safe_load(...) → dict["backend"]["in_memory"]["preload_path"] = override → yaml.safe_dump`，写到 `/tmp/<warmup_id>__libero10.yaml`，server load 这个临时文件
- **declared_keys resolver**：driver 启动时调 `phase5.spec.generate_g5_cells()` → 按 `Cell.warmup_yaml_id` dedup 出 3 个 `{warmup_yaml_id: declared_keys}` 映射（实测 phase4 p1/p2 各 10 keys、phase3 g6 共 2 keys）→ 对每个 input warmup yaml 文件按 stem 查映射拿 declared_keys → 传给 `_extract_finite_factor_raw(src, dst, declared_keys=mapping[stem])`。**不从 warmup yaml 的 `dump.factors` 解析**——`common.v2_spec._build_dump_factor_superset()` 故意输出 superset，与 phase5 G5 阈值 solver 实际用的 `Cell.declared_keys` 不一致；`_extract_finite_factor_raw` 按错误的 key 集过滤会保留太多/太少行。
- driver 启动 sanity check：3 个 input warmup yaml stem 必须全部命中映射，否则 raise（防止 phase3/4 改名后悄悄 fall-back）
- 整个 driver 单进程串行跑 3 个 warmup（30-60 min wall-clock 可接受）

**测试**（L2 强制；新增 `tests/exp/test_phase5_libero10_path_override.py`）：

eval 侧 override 透传：
- `v2_spec.build_eval_yaml(preload_pkl_override="X")` → emit yaml 的 `preload_path == "X"`
- `v2_spec.build_eval_yaml()`（不传 override）→ emit yaml 的 `preload_path == CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]`（libero_spatial）
- `phase5.spec.build_eval_yaml_for_cell(cell, ..., preload_pkl_override="X")` → emit yaml 同上

**warmup 侧 override 透传**（Stage 1.5 与 G1-G4 lazy 路径都依赖这条线）：
- `v2_spec.build_warmup_yaml(preload_pkl_override="X")` → emit yaml 的 `preload_path == "X"`
- `v2_spec.build_warmup_yaml()`（不传 override）→ emit yaml 的 `preload_path == CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]`（libero_spatial）
- `phase5.spec.build_warmup_yaml_for_cell(cell, preload_pkl_override="X")` → emit yaml 同上
- `phase5.runner._mode_emit_warmup_yamls(args)`（`args.preload_pkl_override="X"`，monkeypatch `write_yaml` 收集 emit dict）→ 所有写出的 warmup yaml 都满足 `preload_path == "X"`
- `phase5.runner._ensure_warmup_for_cell(... args.preload_pkl_override="X")` lazy 路径 → 调用 `build_warmup_yaml_for_cell` 时透传 override（用 monkeypatch 替代真 server 调用，仅验 yaml 内容）

CLI parse：
- `phase5.runner._parse_args(["--mode", "run-eval", "--preload-pkl-override", "X"])` → `args.preload_pkl_override == "X"`
- `phase5.runner._parse_args(["--mode", "run-eval"])` → `args.preload_pkl_override == ""`（默认空串触发 None 透传）

G5 driver：
- declared_keys 映射断言：对 3 个 historical warmup_yaml_id 各跑一次映射，期望集分别匹配实测值
  - `spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup` → 10 keys 含 `{jerk,direction,dispersion,path_length}_offline_state__p0_f{3,5}` + `{jerk,dispersion}_online_action__p3_f3`
  - `spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup` → 10 keys 同上但 offline_state→offline_action（base channel 改成 action）+ `{jerk,dispersion}_online_action__p3_f3`
  - `spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup` → 2 keys `{jerk,dispersion}_online_action__p3_f3`
- unknown warmup yaml stem → driver raise `KeyError`
- smoke: 给一个 fixture warmup yaml + monkeypatch LIBERO subprocess + 假 fetch_dump → 验输出 jsonl 路径与 finite-only 行数

### Stage 1 — G5 历史 warmup raw 重建 on libero_10（用新 driver，~30-60 min total）

**输入复用**（不改原文件）：
- `exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.yaml`
- `exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml`
- `exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml`

> warmup yaml 描述 KeyBuilder + 因子 + AlwaysWarmStartJudge + DumpingJudge，与 task_suite 正交。但 `backend.in_memory.preload_path` 指向 libero_spatial pkl，driver 会在 server load 前 patch 到 libero_10 路径（见 Stage 0.5 文件 4）。

**Driver 调用**：

```bash
mkdir -p exp/verdict_factor_judge/data/phase3_libero10/warmup \
         exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw

uv run python -m exp.verdict_factor_judge.phase5.g5_warmup_libero10_driver \
  --host 155.98.36.32 --port 9000 \
  --num-workers 5  --warmup-trials 2 \
  --preload-pkl       exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --phase3-warmup-in  exp/verdict_factor_judge/config/spatial16/phase3/warmup \
  --phase4-warmup-in  exp/verdict_factor_judge/config/spatial16/phase4/warmup \
  --phase3-warmup-out exp/verdict_factor_judge/data/phase3_libero10/warmup \
  --phase4-warmup-out exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw
```

**期望产出**：
- `exp/verdict_factor_judge/data/phase3_libero10/warmup/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.jsonl`
- `exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw/p1_state_fut_online_act.jsonl`
- `exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw/p2_action_fut_online_act.jsonl`

> 文件名形态严格 mirror `phase5/spec.warmup_raw_path_for_cell` 的查找规则：phase3 g6 → `<warmup_yaml_id>.jsonl`；phase4 p1/p2 → `<recipe_id>.jsonl`（line 651-661）。

**Smoke gate**：

```bash
for p in exp/verdict_factor_judge/data/phase3_libero10/warmup/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.jsonl \
         exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw/p1_state_fut_online_act.jsonl \
         exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw/p2_action_fut_online_act.jsonl; do
  [ -s "$p" ] && echo "OK: $p ($(wc -l < $p) lines)" || echo "MISSING: $p"
done
# 期望: 3 行 OK, 每个 jsonl ≥ 100 lines (warmup 2 trial × 10 task × ~5-10 step/trial)
```

### Stage 1.5 — G3 共享 warmup 预跑（消除 §2.5 的并发 race）

G3 12 patterns 共享 1 个 warmup per `(base, channel)` → 共 4 个 distinct warmup_yaml_id。若不预跑，Stage 2 6-server 并发 run-eval 时这 4 个 warmup 在多 server 上撞车。

**预跑命令**：

```bash
uv run python -m exp.verdict_factor_judge.phase5.runner \
  --mode run-warmup \
  --groups g3 \
  --task-suite libero_10 \
  --host 155.98.36.32 --port 9000 \
  --num-workers 5  --warmup-trials 2 \
  --preload-pkl-override   exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5_libero10/warmup \
  --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_libero10_systematic/warmup_factor_raw
```

`_mode_run_warmup` 内部按 `warmup_yaml_id` dedup（line 350-357），4 个 distinct warmup 各跑一次。**单机串行**消除 race。

**Smoke gate**：

```bash
ls exp/verdict_factor_judge/data/phase5_libero10_systematic/warmup_factor_raw/spatial16_w8_d4_phase5_g3_*__warmup.jsonl | wc -l
# 期望: 4 (= 2 base × 2 channel)
```

> G1 / G2 / G4 各 cell 是 1:1 warmup，依赖 allocate_to_servers 切片后单 server 独占，可继续依赖 Stage 2 lazy 流程，不需要预跑。

### Stage 2 — phase5 sweep on libero_10（240 cell × 100 ep，主体）

> **`_libero10_` config dir 重用清理**：`_run_one_warmup` / `_ensure_eval_yaml_for_cell` 的 lazy-emit 路径**只在 yaml 文件缺失时**才应用 `--preload-pkl-override`；若 `config/spatial16/phase5_libero10/{warmup,eval}/` 已存在留有 libero_spatial 残留 yaml（比如 G1 早期 dry-run 跑了几次），runner 会复用现有文件而**不重写**。开 Stage 2 前必须确认目标 config 目录干净：`find exp/verdict_factor_judge/config/spatial16/phase5_libero10 -name '*.yaml' -delete`（首次跑 / 中途重置时执行；半路 retry 不要 delete 已成功 emit 的 yaml）。

**一条命令端到端**（lazy-emit warmup + lazy-emit eval + run-eval）：

```bash
mkdir -p exp/verdict_factor_judge/{config/spatial16/phase5_libero10/{warmup,eval},data/phase5_libero10_systematic/{warmup_factor_raw,per_step,episode_results,thresholds}}

uv run python -m exp.verdict_factor_judge.phase5.runner \
  --mode run-eval \
  --task-suite libero_10 \
  --host 155.98.36.32 --port 9000 \
  --num-workers 5  --warmup-trials 2  --eval-trials 10 \
  --preload-pkl-override   exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5_libero10/warmup \
  --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_libero10_systematic/warmup_factor_raw \
  --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3_libero10/warmup \
  --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw \
  --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5_libero10/eval \
  --summary-out            exp/verdict_factor_judge/data/phase5_libero10_systematic/per_yaml_summary.jsonl \
  --episode-results-dir    exp/verdict_factor_judge/data/phase5_libero10_systematic/episode_results \
  --per-step-log-dir       exp/verdict_factor_judge/data/phase5_libero10_systematic/per_step
```

> **顺序提醒**：必须先 Stage 1.5（G3 共享 warmup）跑完，再开 6-server run-eval；否则 G3 cell 仍可能 race。

**6-server 分片**（同 libero_spatial 版本，4:4:4:3:3:3 = 48/48/45/33/33/33）：

```bash
# 在执行端先把 240 cell 分桶（一次性，用 spec.py 的 allocate_to_servers）
uv run python -c "
from exp.verdict_factor_judge.phase5.spec import generate_all_cells, allocate_to_servers
buckets = allocate_to_servers(generate_all_cells())
for sid, cells in buckets.items():
    open(f'/tmp/{sid}_libero10.txt','w').write(' '.join(c.yaml_id for c in cells))
    print(f'{sid}: {len(cells)} cells')
"

# 每个 server 跑 (替换 S<N>):
uv run python -m exp.verdict_factor_judge.phase5.runner \
  --mode run-eval  --task-suite libero_10 \
  --cell-ids $(cat /tmp/S<N>_libero10.txt) \
  --host <server-S<N>-host>  --port <server-S<N>-port> \
  --num-workers 5  --warmup-trials 2  --eval-trials 10 \
  --preload-pkl-override   exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5_libero10/warmup \
  --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_libero10_systematic/warmup_factor_raw \
  --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3_libero10/warmup \
  --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw \
  --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5_libero10/eval \
  --summary-out            exp/verdict_factor_judge/data/phase5_libero10_systematic/per_yaml_summary.S<N>.jsonl \
  --episode-results-dir    exp/verdict_factor_judge/data/phase5_libero10_systematic/episode_results \
  --per-step-log-dir       exp/verdict_factor_judge/data/phase5_libero10_systematic/per_step
# 跑完 6 server 后 cat 合并: cat per_yaml_summary.S{1..6}.jsonl > per_yaml_summary.jsonl
```

> **`--preload-pkl-override` 不是 path placeholder 之一**——它是 §2.6 hardcoded `preload_pkl` 的核心修复 flag，每个 server 命令都必须显式列出，否则会回退到 libero_spatial pkl（v2_spec default），整个 sweep 报废。

> **服务器端口拓扑**：以你机器实际配置为准。memory 里记录的远程入口是 `155.98.36.13:9000`，runner default 是 `155.98.36.32`，看哪个能通。

**Smoke gate（开跑前先跑 1 个 cell 验证全链路）**：

```bash
uv run python -m exp.verdict_factor_judge.phase5.runner \
  --mode run-eval  --task-suite libero_10 \
  --cell-ids spatial16_w8_d4_phase5_g1_p1_action_jerk__win-3-3 \
  --eval-trials 1  --warmup-trials 1 \
  --host 155.98.36.32 --port 9000 \
  ... (同上 path overrides)
# 期望 summary jsonl 多 1 行，含 success_rate / n_full_hit / n_warm_start / n_miss / fh_thr / ws_thr
```

### Stage 3 — SR-only stats（轻量分析，不画图）

直接复用 `analyze_phase5_*` 的统计部分（Pareto / heatmap 不调用，因为缺 baseline）：

```bash
mkdir -p exp/verdict_factor_judge/analysis/phase5_libero10

uv run python << 'PY' > exp/verdict_factor_judge/analysis/phase5_libero10/sr_stats.md
import json, pathlib, statistics
rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase5_libero10_systematic/per_yaml_summary.jsonl') if l.strip()]
print('# Phase 5 libero_10 Systematic Sweep — SR Stats\n')
print(f'**Total cells**: {len(rows)} (expect 240)\n')
print('## Per-group SR distribution\n')
print('| Group | n | Min | Median | Max | Mean |')
print('|---|---:|---:|---:|---:|---:|')
for g in ('g1','g2','g3','g4','g5'):
    sub = [r['success_rate'] for r in rows if r.get('group')==g and r.get('success_rate') is not None]
    if not sub: continue
    print(f'| {g.upper()} | {len(sub)} | {min(sub):.2f} | {statistics.median(sub):.2f} | {max(sub):.2f} | {statistics.mean(sub):.3f} |')
print(f'\nCells with SR >= 0.95: {sum(1 for r in rows if (r.get("success_rate") or 0) >= 0.95)}/{len(rows)}')
print(f'Cells with SR <  0.85: {sum(1 for r in rows if (r.get("success_rate") or 1) <  0.85)}/{len(rows)}')
print('\n## Top-10 cells by SR\n')
print('| # | yaml_id | group | SR | inf |')
print('|--:|---|---|---:|---:|')
def _inf(r):
    n = r.get('n_eval_verdicts') or 0
    return ((r.get('n_warm_start',0)*0.75 + r.get('n_miss',0)*1.0)/n) if n else 0
ranked = sorted(rows, key=lambda r: -(r.get('success_rate') or 0))[:10]
for i,r in enumerate(ranked,1):
    print(f'| {i} | `{r["yaml_id"]}` | {r["group"]} | {r["success_rate"]:.2f} | {_inf(r):.3f} |')
PY
cat exp/verdict_factor_judge/analysis/phase5_libero10/sr_stats.md
```

**Decision gate dump**（沿用 phase5 runner 内置逻辑，写 5 个 `g{1..5}_decision.json`）：

```bash
uv run python -c "
from exp.verdict_factor_judge.phase5.runner import _dump_decision_gate_table_phase5
from pathlib import Path
_dump_decision_gate_table_phase5(Path('exp/verdict_factor_judge/data/phase5_libero10_systematic/per_yaml_summary.jsonl'))
"
ls exp/verdict_factor_judge/data/phase5_libero10_systematic/g{1..5}_decision.json
```

---

## 5. 工作量估算

| Stage | 仿真量 | 估时（按 phase5 libero_spatial 实测 ~12-15 ep/min/server 推） |
|---|---|---|
| 0 | 0 ep（仅 CPU enrich） | ~5 min |
| 1 | 3 yaml × 2 trial × 10 task × 1 server | ~30-60 min |
| 2 — G1-G4 warmup lazy | 148 distinct warmup × 2 trial × 10 task | ~2-3 h（按 6 server 并行） |
| 2 — eval 主体 | 240 cell × 10 trial × 10 task = 24 000 ep | ~3-4 h（按 6 server 并行） |
| 3 | 0 ep（仅 IO） | ~1 min |
| **合计** | **~24 060 episode 仿真** | **~6-7 h wall-clock（6-server）** |

---

## 6. 风险与缓解

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | enrich yaml 缺空格（`{past:0, future:3}`）→ PyYAML 把 key 当 plain scalar，`int(w["past"])` raise `KeyError: 'past'` | Stage 0 enrich fail | 严格按 §4 Stage 0 0.2 模板写；smoke gate 检查 `factors per entry == 64` |
| R2 | libero_10 entries 缺 vision/action 等关键字段（pkl 是旧 schema）→ OfflineWriter 算因子时 raise | Stage 0 enrich fail | 已 dry-run 验过 entry 0 的 payload 字段齐全（见 §2.1 表）；如果中间某个 entry 字段不齐，enrich 内部会 propagate 错误，不会静默 |
| R3 | runner 默认 `--host 155.98.36.32` 与你实际服务器入口（memory: `155.98.36.13:9000`）不一致 | Stage 1/2 连不上 | 所有命令显式 `--host` / `--port`，按你机器配置覆盖 |
| R4 | G5 cell 的 yaml_id（`phase5_g5_p1__fh0.2_ws0.5`）与 libero_spatial 版本同名 → 数据目录撞名 | summary jsonl 混 / 阈值串味 | Q2=(a′) 通过新 `_libero10_` 后缀目录 + `--preload-pkl-override` 双重分流；所有 emit yaml 内部 `preload_path` 指向 libero_10 pkl |
| R10 | server 端 cp1_spatial_pool_16 pkl 文件必须是 enrich 后的版本（含 `library_stats` + 64 keys/entry），否则 `cache/config.py:1759` 在 `load_cache_config` 时 raise；远端 server 必须先 sync libero_10 pkl 到本地 fs | Stage 1 / 1.5 / 2 第一个 cell load yaml 即崩 | Stage 0 enrich 后必须 (1) 本地 smoke gate 通过；(2) 若 server 在远端机器，`rsync exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl <server>:<same path>` 推送；(3) server 重启或确认 `--collect_dir` 与 pkl 路径不冲突 |
| R11 | Stage 1 / 1.5 / 2 全部依赖 Stage 0.5 的 code 改动已 land 并通过 G2 APPROVED；若先开仿真后 land code，emit yaml 仍会指向 libero_spatial pkl | 报废整轮仿真 | §4 Stage 编号是 strict prerequisite 顺序：Stage 0（pkl enrich，独立）+ Stage 0.5（code，必须 G2 APPROVED） → Stage 1（G5 warmup driver）→ Stage 1.5（G3 共享 warmup）→ Stage 2（240 cell run-eval）→ Stage 3（stats）。Stage 0 与 0.5 可并行，但 Stage 1 起必须等 0.5 G2 APPROVED |
| R6 | phase5 lazy-emit eval yaml 时阈值 solver 用的 raw 是 phase5_libero10_systematic 自己的 warmup raw → 与 libero_spatial 阈值会**不同**（因为 task 分布不同） | 不算 risk，符合预期 | 这正是切 task_suite 的实验意义；阈值会自动重算 |
| R7 | 服务器 GPU 加载 libero_10 pkl 时发现 schema 不匹配（如 LibraryStats tensor dtype 与 server 期待不符） | Stage 2 加载阶段 fail | Stage 0 smoke gate 已验证 `state_active_mask` / `action_active_mask` 都是非空 tensor；如果 server 端额外检查 dtype，Stage 2 第一个 cell 就会 fail，可早发现 |
| R8 | per_yaml_summary.jsonl 并发写入（6 server 同写同一文件） | 行损坏 / 缺行 | 按 phase3 / phase5 libero_spatial 经验：每 server 自己的 summary file，跑完 cat 合并；见 §4 Stage 2 6-server 拓扑注释 |
| R9 | libero_10 task_suite 在 `examples/libero/main.py` 实际行为与 libero_spatial 不一样（如 episode 长度上限、init state 数量）→ verdict 计数与 libero_spatial 不可比 | SR 数值需要单独解读，不能横向对比 | 这是已知事实（用户也认可）；分析只看 libero_10 自身分布与 group 内部对比 |

---

## 7. 测试 / 验收清单

- [ ] **Stage 0**：smoke gate 通过（64 factors / library_stats 非 None / state_active_mask 非零）
- [ ] **Stage 1**：3 个 jsonl 都生成，size > 0，且每行能被 `_load_per_key_finite_history` 解析
- [ ] **Stage 2 dry-run**：单 cell（`g1_p1_action_jerk__win-3-3` + `--eval-trials 1`）端到端通过，summary 新增 1 行
- [ ] **Stage 2 batch1 启动后 30 min checkpoint**：summary 已累积 ≥ 1 行/server，没有 traceback
- [ ] **Stage 2 完成**：summary jsonl 240 行，0 异常（按 phase5 libero_spatial 的 `runner.py:530` exception handler，单 cell 失败不会 abort，但收尾要核对 240/240）
- [ ] **Stage 3**：`sr_stats.md` 生成，5 个 `g{1..5}_decision.json` 生成

---

## 8. Commit / Push 收尾

按 [`docs/experiments/artifact_layout.md` §3](../docs/experiments/artifact_layout.md) 已注册的 tracking policy：`exp/**/data/**` 默认全部 ignored；任何 `data/**` 下要 tracked 的文件必须先在 `.gitignore` 加白名单例外。

**本 plan 的 commit 边界**（保守，避免触发 §3 例外审批流程）：

1. **入库**：
   - `exp/verdict_factor_judge/common/v2_spec.py` / `phase5/spec.py` / `phase5/runner.py`（Stage 0.5 code 改动）
   - `exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py`（新文件）
   - `tests/exp/test_phase5_libero10_path_override.py`（新测试）
   - `exp/verdict_factor_judge/config/spatial16/phase5_libero10/{warmup,eval}/*.yaml`（lazy-emit 出的 ~388 个 yaml，不在 `data/**` 下）
   - `exp/verdict_factor_judge/analysis/phase5_libero10/sr_stats.md`（不在 `data/**` 下）
   - `logs/verdict_phase5_libero10_systematic_sweep.log.md`（本 plan 文件；G2 round 在 §4 Code 完成后开启，并把 G2 Review Log 保留在文件中作为 code-review 永久记录）

2. **不入库（保留本地 + 单独打包备份）**：
   - `exp/verdict_factor_judge/data/phase{3,4}_libero10/...` 全部
   - `exp/verdict_factor_judge/data/phase5_libero10_systematic/...` 全部（包括 `per_yaml_summary.jsonl` / `g{1..5}_decision.json` / `thresholds/` / `per_step/` / `episode_results/` / `warmup_factor_raw/`）
   - 跑完用 `tar czf phase5_libero10_systematic_<date>.tar.gz exp/verdict_factor_judge/data/phase{3,4,5}_libero10*` 离线归档

> 若后续决定 `per_yaml_summary.jsonl` / 5 个 `g{1..5}_decision.json` 需要跨机器 reproducible，必须独立提一个 PR 修改 `.gitignore` 加白名单，并对照 `artifact_layout.md` §3 的现有 24-JSON 例外列表追加。这不在本 plan 范围内。
>
> commit message English-only，不加 Co-Authored-By（per global instruction）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-22 11:27 CDT

- [Blocking] [Concern] The new Stage 1 files do not pass the repository's configured ruff lint, so the code is not pre-commit-ready. — reasoning: `uv run ruff check exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py tests/exp/test_phase5_libero10_path_override.py` fails on the new files with import ordering (`I001`), `Optional[...]` annotations (`UP045`), an ambiguous multiplication sign in a new test comment (`RUF003`), and private-member access in the new tests (`SLF001`). `pyproject.toml` selects these rules and `.pre-commit-config.yaml` runs ruff; either make the new files lint-clean or add explicit targeted `noqa`/file-level rationale for intentional private-helper tests.
- [Blocking] [Concern] The official test file does not include the G5 driver success-path smoke that the approved plan requires. — reasoning: Stage 0.5's test plan calls for a G5 driver smoke with a fixture warmup yaml, monkeypatched LIBERO subprocess, fake `fetch_dump`, and assertions on output JSONL path plus finite-only rows. The current `tests/exp/test_phase5_libero10_path_override.py` covers declared-key mapping, YAML patching, output-path shape, and runner-args fields, but it never invokes `_run_one_historical_warmup()` or `main()` on a successful fake warmup. I ran that probe independently during review and it passed, but the regression guard needs to live in the committed test suite.
- [Blocking] [Concern] The log index and plan status metadata still describe the work as waiting for code, even though Stage 1 code is now present and under G2 review. — reasoning: `logs/README.md` indexes this plan as `Plan (G1 APPROVED 2026-05-22 / 待 §4 Code)`, and the plan header says `待 §4 Code → G2`. Since this G2 round is reviewing actual code changes (`v2_spec.py`, `phase5/spec.py`, `phase5/runner.py`, new driver, and new tests), the active log/index status should be updated in the executor response to reflect code complete / G2 Round 1 state. `WORKING_AGREEMENT.md` §4 requires logs index sync for log modifications.
- [Non-blocking] [Suggestion] Keep the existing lazy-emission skip behavior explicit in the runbook when reusing non-empty `phase5_libero10` config dirs. — reasoning: `_run_one_warmup()` and `_ensure_eval_yaml_for_cell()` only apply `--preload-pkl-override` when they emit missing YAMLs; if a reused output directory already contains stale libero_spatial YAMLs, the runner will load those existing files. The current plan uses fresh `_libero10_` config dirs, so this is not a correctness blocker for the intended run, but a short cleanup/retry note would reduce operator error.

### G2 Round 1 — Executor — 2026-05-22

- **Accepted (Item 1, Blocking)** — `uv run ruff check` 在 driver + test 两个新文件上 20 个违规已经全部清掉：I001 / UP045 自动 fix；RUF003 把测试注释里的 `×` 改成 ASCII `x`；SLF001 在测试文件顶部加 `# ruff: noqa: SLF001  # white-box tests intentionally exercise _-prefixed helpers`（与 `tests/exp/test_phase5_runner.py` 等已有 phase5 测试访问 `r._parse_args` / `spec._helper` 的项目惯例一致）；额外 ruff 自动 fix 后新 smoke 又引入 UP012（`.encode()` → bytes literal）+ PLC0415（function-local import），两条同步修：`b"..."` 多行 bytes literal、`json` / `math` 升 top-level import。最终 `uv run ruff check exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py tests/exp/test_phase5_libero10_path_override.py` 输出 "All checks passed!"。三个已存在文件（v2_spec.py 8、phase5/spec.py 35、phase5/runner.py 14 = 57 errors）经 `git stash` 前后对比确认全部是 pre-existing 违规，与本次 Stage 0.5 改动无关——按 WA §3.1 "Minimal change" 原则不在本 plan 范围内修。
- **Accepted (Item 2, Blocking)** — `test_phase5_libero10_path_override.py` 新增 `test_driver_run_one_historical_warmup_smoke`：fixture warmup yaml on tmp_path（含 libero_spatial preload）→ monkeypatch `drv.subprocess.run` 为 `MagicMock`、fake `ctl.fetch_dump` 返回手搓的 4-row per-step JSONL（行 0/1 含 finite，行 2 全 out-of-declared，行 3 全 NaN）→ 调 `_build_runner_args` 构造 RunnerArgs → 调 `_run_one_historical_warmup` end-to-end → assert: (a) `ctl.load_cache_config` 被调一次且其 yaml_content 解析后 `preload_path == libero_10 pkl`（验证 patch 生效）；(b) `subprocess.run` 调一次（LIBERO worker spawn）；(c) 输出 jsonl 在期望路径；(d) 含 2 行（finite-only 过滤后），row 0 完整保留，row 1 含 NaN k_a + finite k_b（按 `_extract_finite_factor_raw` 实际语义：保留 declared_keys 整子集，只要 ≥1 个 finite 就 keep row）。该 case 直接命中 `_run_one_historical_warmup` 主成功路径，正是 reviewer 独立 probe 通过但缺 regression guard 的部分。完整 case 数 17 → 18，全 PASS。
- **Accepted (Item 3, Blocking)** — `logs/README.md` 索引行从 `Plan (G1 APPROVED 2026-05-22 / 待 §4 Code)` 改为 `Plan (G1 APPROVED 2026-05-22 / §4 Code 完成 / G2 R1 NEEDS REVISION → Executor R1 applied / 待 Reviewer R2)`；plan header Status 行从 `Plan (G1 APPROVED 2026-05-22 — 待 §4 Code → G2)` 改为 `Plan (G1 APPROVED 2026-05-22 / §4 Code 完成 / G2 Round 1 NEEDS REVISION → Executor R1 applied → 待 Reviewer Round 2)`。两处同步，遵守 WA §4 Index Sync 红线。
- **Accepted (Item 4, Non-blocking)** — §4 Stage 2 顶部加引文块说明 lazy-emit skip 语义：明确 `_run_one_warmup` / `_ensure_eval_yaml_for_cell` 只在 yaml 缺失时应用 override，若目标 `config/spatial16/phase5_libero10/` 已有 libero_spatial 残留，runner 会复用而**不重写**；建议首次跑 / reset 时 `find ... -name '*.yaml' -delete`，半路 retry 时不要删已 emit 的 yaml。这是 operator hygiene 提醒，code-level fail-safe 不在本 plan 范围（要 land-time 的"重写已存在 yaml 模式"等于改 runner 行为，scope creep）。

### G2 Round 2 — Reviewer — APPROVED — 2026-05-22 11:43 CDT

- [Result] APPROVED. — reasoning: Round 1 的三个 blocking items 均已关闭：新 driver/test 目标 ruff 通过；官方测试已加入 `_run_one_historical_warmup()` 成功路径 smoke；`logs/README.md` 与 plan header 已同步到 Executor R1 / Reviewer R2 状态。非阻塞的 lazy-emit stale-yaml 风险也已写入 Stage 2 runbook。
- [Verification] `uv run ruff check exp/verdict_factor_judge/phase5/g5_warmup_libero10_driver.py tests/exp/test_phase5_libero10_path_override.py` PASS；`PYTHONPATH=. uv run pytest tests/exp/test_phase5_libero10_path_override.py tests/exp/test_phase5_spec.py tests/exp/test_phase5_runner.py tests/exp/verdict_factor_judge/test_v2_spec.py tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py -q` PASS (`97 passed, 1 warning in 3.47s` after sandbox uv cache failure was rerun with approved escalation); `git diff --check` PASS。
- [Non-blocking] [Note] Executor 记录的 broader legacy ruff debt remains out of this Stage 1 G2 scope. — reasoning: 本轮针对新增 driver/test 的 lint gate 已清零；`v2_spec.py` / `phase5/spec.py` / `phase5/runner.py` 的既有 ruff 债务若被后续更宽的提交门禁要求覆盖，应单独处理，不能阻塞本次 G2。
