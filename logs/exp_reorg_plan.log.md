# `exp/` Directory Reorganization — Plan

- Status: `Implemented — G2 Approved`
- Level: **L2**（跨文件结构重构：改变 `exp.*` import 路径，影响 tests / scripts / docs；不改任何功能语义）
- Scope: 把 `exp/` 下平铺的 24 个脚本按实验分类到 4 个实验子目录 + 1 个 `common/` 公用目录
- 起草日期：2026-04-14
- 关联上下文：
  - 前置 L2 `cleanup`（633acd8）抽出了 5 个 `_xxx.py` helper，为本次目录化铺路
  - 依赖调查结果（attribution map）见 §2

---

## 0. TL;DR

当前 `exp/` 平铺 24 个文件（5 helper + 6 cache / 5 deviation / 1 temporal-prune / 7 qdrant），不同实验的脚本互相穿插、难以快速定位。本 plan 按实验归档到 4 个子包 + 1 个 `common/` 公用包，**共 24 次 `git mv` + 5 个新 `__init__.py`（各带一行 package docstring）**。同步面在 Round 1 审查后已扩大到：13 个 tests 的模块级 + 函数内延迟 import + logger name 字符串、3 个 `scripts/` 文件（2 个 import + 1 个 docstring 引用）、`docs/` 里约 36 处 CLI 命令、8 个 entry 脚本的 sys.path/`_REPO_ROOT` 层级（含 `qdrant_step_knn_experiment.py` L8-9、`toy_qdrant_server.py` L57-58），以及 `src/openpi/cache/` 和 active `exp/` 内所有旧路径 docstring/注释引用。

**不改任何功能语义**：重构后所有脚本的 CLI 接口、RunState JSON schema、subprocess 协议完全一致，仅 import path 和磁盘路径变化。

---

## 1. 现状

### 1.1 文件清单（平铺 24 个）

```
exp/
├── __init__.py
├── _cache_config_rpc.py        ┐
├── _libero_env.py              │
├── _run_state_base.py          ├ 5 helper (从 cleanup 抽出)
├── _subprocess.py              │
├── _unit_key.py                ┘
├── analyze_cache_results.py            ┐
├── build_clip_cache_artifact.py        │
├── build_in_memory_cache_artifact.py   │
├── calibrate_robot_state_tau.py        ├ 6 个 Cache Experiment (CP1)
├── calibrate_score_sum_stats.py        │
├── generate_cache_run_yamls.py         │
├── run_cache_experiments.py            ┘
├── analyze_deviation_results.py        ┐
├── compute_deviate_scores.py           │
├── run_spawn_experiment.py             ├ 4 个 Trajectory Deviation Corrective
├── run_step1b_gt.py                    ┘
├── generate_temporal_prune_yamls.py    ─ 1 个 Temporal Prune
├── qdrant_ingest_openpi.py                         ┐
├── qdrant_openpi_common.py                         │
├── qdrant_step_knn_experiment.py                   │
├── qdrant_step_knn_experiment_config.example.json  ├ 7 个 Qdrant Step-KNN
├── qdrant_verify_openpi.py                         │
├── toy_qdrant_server.py                            │
└── toy_stage1_server.py                            ┘
```

### 1.2 症状

- 新人打开 `exp/` 看不出 "哪些脚本组成一个完整的实验 pipeline"
- `_libero_env.py`（仅被 `run_spawn_experiment.py` 使用）与 `_cache_config_rpc.py`（被 3 个 deviation runner import；`run_cache_experiments.py` 保留本地 `_send_cache_config()`，未 import 该 helper）混在同一层，公私边界不清
- `qdrant_*` 7 个文件带长前缀，视觉占位大

---

## 2. 归属映射（调查结果）

依据：文件模块 docstring、跨文件 import 关系、`docs/` 和 `logs/` 对文件名的引用。

| 脚本 | 所属 | 证据 |
|------|------|------|
| `_cache_config_rpc.py` | common | 被 3 个 runner import：`run_step1b_gt.py`、`compute_deviate_scores.py`、`run_spawn_experiment.py`。注：`run_cache_experiments.py` 有自己的本地 `_send_cache_config()`（L84），未 import 该 helper |
| `_run_state_base.py` | common | 被 3 个 deviation runner 使用（`run_step1b_gt`、`compute_deviate_scores`、`run_spawn_experiment`）。注：`run_cache_experiments.py` 有独立的本地 `class RunState` dataclass（L62），未继承 `BaseRunState` |
| `_subprocess.py` | common | 被 CP1 + step1b 2 个 runner 共享（`run_cache_experiments.py`、`run_step1b_gt.py`） |
| `_unit_key.py` | common | 被 3 个 deviation runner 共享 |
| `_libero_env.py` | trajectory_deviation | **仅** 被 `run_spawn_experiment.py` 使用（下沉 rationale 见 §3.1） |
| `build_clip_cache_artifact.py` | cache_experiment | `docs/run_cp1_cache_experiment.md` |
| `build_in_memory_cache_artifact.py` | cache_experiment | `docs/run_cp1_cache_experiment.md`、`docs/run_temporal_prune_experiment.md`（复用） |
| `calibrate_robot_state_tau.py` | cache_experiment | `logs/archive/cache_experiment_plan.log.md:114` 明确指为该实验离线工具 |
| `calibrate_score_sum_stats.py` | cache_experiment | `docs/run_cp1_cache_experiment.md` |
| `generate_cache_run_yamls.py` | cache_experiment | `docs/run_cp1_cache_experiment.md` |
| `run_cache_experiments.py` | cache_experiment | `docs/run_cp1_cache_experiment.md`、`run_temporal_prune_experiment.md`（复用） |
| `analyze_cache_results.py` | cache_experiment | `docs/run_cp1_cache_experiment.md`、`run_temporal_prune_experiment.md`（复用） |
| `compute_deviate_scores.py` | trajectory_deviation | `trajectory_deviation_corrective_experiment.log.md` Step 2 |
| `run_spawn_experiment.py` | trajectory_deviation | 同上 Step 3 |
| `run_step1b_gt.py` | trajectory_deviation | 同上 Step 1b |
| `analyze_deviation_results.py` | trajectory_deviation | 同上 |
| `generate_temporal_prune_yamls.py` | temporal_prune | `docs/run_temporal_prune_experiment.md` |
| `qdrant_openpi_common.py` | qdrant_step_knn | 被 4 个 qdrant 脚本 import |
| `qdrant_ingest_openpi.py` | qdrant_step_knn | import `qdrant_openpi_common` |
| `qdrant_step_knn_experiment.py` | qdrant_step_knn | import `qdrant_openpi_common` |
| `qdrant_step_knn_experiment_config.example.json` | qdrant_step_knn | step-knn 配置示例 |
| `qdrant_verify_openpi.py` | qdrant_step_knn | import `qdrant_openpi_common` |
| `toy_qdrant_server.py` | qdrant_step_knn | 辅助工具（本地调试 qdrant 实验） |
| `toy_stage1_server.py` | qdrant_step_knn | 辅助工具（与 qdrant 实验绑定的本地 stage1 mock） |

---

## 3. 目标目录结构

```
exp/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── _cache_config_rpc.py
│   ├── _run_state_base.py
│   ├── _subprocess.py
│   └── _unit_key.py
├── cache_experiment/
│   ├── __init__.py
│   ├── analyze_cache_results.py
│   ├── build_clip_cache_artifact.py
│   ├── build_in_memory_cache_artifact.py
│   ├── calibrate_robot_state_tau.py
│   ├── calibrate_score_sum_stats.py
│   ├── generate_cache_run_yamls.py
│   └── run_cache_experiments.py
├── trajectory_deviation/
│   ├── __init__.py
│   ├── _libero_env.py
│   ├── analyze_deviation_results.py
│   ├── compute_deviate_scores.py
│   ├── run_spawn_experiment.py
│   └── run_step1b_gt.py
├── temporal_prune/
│   ├── __init__.py
│   └── generate_temporal_prune_yamls.py
└── qdrant_step_knn/
    ├── __init__.py
    ├── qdrant_ingest_openpi.py
    ├── qdrant_openpi_common.py
    ├── qdrant_step_knn_experiment.py
    ├── qdrant_step_knn_experiment_config.example.json
    ├── qdrant_verify_openpi.py
    ├── toy_qdrant_server.py
    └── toy_stage1_server.py
```

### 3.1 设计取舍

- **`_libero_env.py` 下沉到 `trajectory_deviation/`**：仅被 `run_spawn_experiment.py` 用，放 `common/` 是过度共享。`scripts/verify_*.py` 也 import 它——这是 verify 脚本从属于 deviation 实验（§4 scripts/ 改动）。
- **`_` 前缀保留**：`common/_xxx.py` 里的 `_` 语义变弱但不去掉——保持最小改动，只移位不改名。
- **`qdrant_openpi_common.py` 留在 `qdrant_step_knn/`（不进 `exp/common/`）**：该模块是 qdrant 实验**内部**的工具（携带 collection schema 细节），不跨实验复用。
- **`toy_stage1_server.py` 放 qdrant_step_knn**：虽然没有 `import exp.xxx`，但它是 qdrant 实验的本地调试辅件（log `qdrant_design.log` 提到）。
- **新增 5 个 `__init__.py`**：Python 包必需品。遵循 `WORKING_AGREEMENT.md` §3.2 "Every new file MUST have a file-level docstring"，每个 `__init__.py` 仅包含一行英文 package docstring，无任何 import / 副作用 / 导出符号。示例：
  ```python
  """Cache experiment (CP1) scripts: artifact build, YAML gen, run, analyze."""
  ```

### 3.2 不在本计划范围

- 改脚本文件名（比如 `qdrant_ingest_openpi.py` → `ingest.py`）——改名风险与目录化叠加，放后续 follow-up
- **非路径相关的 docstring 重写**（设计说明、参数语义、示例输出等段落保持原样）。路径/模块名引用（`uv run exp/foo.py`、`python -m exp.foo`、helper 文件头"extracted from `exp/run_cache_experiments.py`"等）按 §4 B3 统一同步，不在"不改"范围内
- `configs/` 或 `data/` 目录整理
- `scripts/`、根目录散落文件（`simple_pytorch_train.py`、`convert.py`、`cmd.sh` 等）

---

## 4. 改动清单（Waves）

### Wave A — 目录骨架 + 物理移动

**A1 新建 5 个 `__init__.py`（各含一行英文 package docstring，无 import / 副作用 / 导出符号；遵循 §3.1 决策与 WA §3.2）：**
- `exp/common/__init__.py` — `"""Shared helpers reused across experiment runners (RPC, subprocess, run state, unit key)."""`
- `exp/cache_experiment/__init__.py` — `"""Cache experiment (CP1) scripts: artifact build, YAML gen, run, analyze."""`
- `exp/trajectory_deviation/__init__.py` — `"""Trajectory deviation corrective experiment: GT collection, deviate scoring, spawn runs."""`
- `exp/temporal_prune/__init__.py` — `"""Temporal prune experiment: YAML generator for pruning-based cache variants."""`
- `exp/qdrant_step_knn/__init__.py` — `"""Qdrant step-KNN experiment: ingest / verify / toy servers backed by qdrant-client."""`

**A2 `git mv` 24 次：**
- 4 个 helper → `exp/common/`（`_cache_config_rpc`, `_run_state_base`, `_subprocess`, `_unit_key`）
- 7 个 → `exp/cache_experiment/`
- 5 个 → `exp/trajectory_deviation/`（含 `_libero_env`）
- 1 个 → `exp/temporal_prune/`
- 7 个 → `exp/qdrant_step_knn/`（含 json 示例）

**验收**：`find exp -maxdepth 1 -type f ! -name "__init__.py"` 无输出（top-level 只允许 `exp/__init__.py`，与 §5 V3 同命令）；`ls exp/*/` 结构匹配 §3。

### Wave B — Import 路径更新

**B1 entry script 内部 import 改写：**

| 原 import | 新 import |
|-----------|-----------|
| `from exp._cache_config_rpc import …` | `from exp.common._cache_config_rpc import …` |
| `from exp._subprocess import …` | `from exp.common._subprocess import …` |
| `from exp._run_state_base import …` | `from exp.common._run_state_base import …` |
| `from exp._unit_key import …` | `from exp.common._unit_key import …` |
| `from exp._libero_env import …` | `from exp.trajectory_deviation._libero_env import …` |
| `from exp.qdrant_openpi_common import …` | `from exp.qdrant_step_knn.qdrant_openpi_common import …` |
| `from exp.qdrant_step_knn_experiment import …` | `from exp.qdrant_step_knn.qdrant_step_knn_experiment import …` |

涉及脚本（约 10 个）：全部 `cache_experiment/` + `trajectory_deviation/` runners + 所有 `qdrant_step_knn/` 脚本（除 `toy_stage1_server.py`）。

**B2 sys.path / `_REPO_ROOT` 层级调整（8 个 entry/root-path 点）：**

```python
# Before（在 exp/ 顶层时）
if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# After（在 exp/<subdir>/ 时）
if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

影响文件（共 8 个）：
- `exp/cache_experiment/run_cache_experiments.py` — L47 `parents[1]` → `parents[2]`
- `exp/trajectory_deviation/run_step1b_gt.py` — `parents[1]` → `parents[2]`
- `exp/trajectory_deviation/run_spawn_experiment.py` — `parents[1]` → `parents[2]`
- `exp/trajectory_deviation/compute_deviate_scores.py` — `parents[1]` → `parents[2]`
- `exp/qdrant_step_knn/qdrant_ingest_openpi.py` — L18 `parents[1]` → `parents[2]`
- `exp/qdrant_step_knn/qdrant_verify_openpi.py` — L12 `parents[1]` → `parents[2]`
- `exp/qdrant_step_knn/qdrant_step_knn_experiment.py` — L8-9 两行无条件 `parents[1]` 和 `parents[1] / "src"` 全部变 `parents[2]`
- `exp/qdrant_step_knn/toy_qdrant_server.py` — L57-58 `_REPO_ROOT = parents[1]` / `_SRC_DIR = parents[1] / "src"` 全部变 `parents[2]`

**B3 Active 代码内所有 `exp/…` 路径引用（全面同步）：**

本轮要求 `exp/` + `scripts/` + `tests/` + `src/openpi/` 下所有非 archive 文件里的旧路径引用全部更新。范围：
- 脚本 docstring 的 `uv run exp/xxx.py …` / `python exp/xxx.py` / `python -m exp.xxx` 示例
- 脚本内部硬编码生成的 CLI 字符串（例：`generate_temporal_prune_yamls.py` L174）
- `src/openpi/cache/` 内部 docstring / 注释对 `exp/xxx.py` 的路径引用（C3 覆盖，列此处便于对照）
- helper module 文件自身的 docstring（如 `_cache_config_rpc.py` 文件头说 "extracted from `exp/run_cache_experiments.py`"）
- `exp/analyze_deviation_results.py`、`exp/qdrant_step_knn_experiment.py` 等 docstring 中的 `python -m exp.xxx` 也要改

**执行方法**：完成所有移动后用单次 grep 扫描剩余旧路径引用：
```bash
rg -n "exp/[a-z_]+\.py|python -m exp\.[a-z_]+|from exp import|from exp\._[a-z_]+|import exp\._[a-z_]+" \
   exp scripts src tests docs \
   | grep -v "archive/"
```
任何残留命中必须在 commit message 或 PR description 中列为"保留清单"并注明理由；其他全部更新。

**B4 tests 改 import（13 文件，grep 驱动的完整清单）：**

执行方法：Wave A 完成后运行
```bash
rg -n "from exp\._|import exp\._|from exp\.run_|import exp\.run_|from exp\.compute_|import exp\.compute_|from exp\.qdrant_|import exp\.qdrant_|from exp\.analyze_|import exp\.analyze_|from exp\.build_|from exp\.generate_|from exp\.calibrate_|from exp import" \
   tests/exp tests/scripts exp scripts
```
然后把每条命中替换为新路径。

已知命中类别（含函数内延迟 import、`from exp import X` 语法、logger name 字符串）：

| Test 文件 | 涉及 import 形式 |
|-----------|------------------|
| `test_unit_key.py` | 头部 `from exp._unit_key` → `from exp.common._unit_key` |
| `test_run_state_base.py` | 头部 `from exp._run_state_base` → `from exp.common._run_state_base` |
| `test_subprocess_helpers.py` | 头部 `from exp._subprocess` → `from exp.common._subprocess` |
| `test_cache_config_rpc.py` | 头部 `from exp import _cache_config_rpc` → `from exp.common import _cache_config_rpc`（**注意：是 `from exp import X` 语法，不是 `from exp._cache_config_rpc import ...`**） |
| `test_libero_env_helper.py` | 5 处**函数内延迟** `from exp._libero_env import ...`（L78/87/101/109/122）→ `from exp.trajectory_deviation._libero_env import ...` |
| `test_run_cache_experiments.py` | 头部 + 4 处**函数内** `from exp.run_cache_experiments import ...`（L167/192/224/243）+ 1 处 `import exp.run_cache_experiments as runner`（L273）→ 全部 `exp.cache_experiment.run_cache_experiments` |
| `test_build_in_memory_cache_artifact.py` | 头部 → `exp.cache_experiment.build_in_memory_cache_artifact` |
| `test_calibrate_score_sum_stats.py` | 头部 → `exp.cache_experiment.calibrate_score_sum_stats` |
| `test_generate_cache_run_yamls.py` | 头部 → `exp.cache_experiment.generate_cache_run_yamls` |
| `test_compute_deviate_scores.py` | 头部 `import exp.compute_deviate_scores as cds` + 函数内 `import exp._run_state_base as base`（L381）→ 前者改 `exp.trajectory_deviation.compute_deviate_scores`，后者改 `exp.common._run_state_base` |
| `test_run_spawn_experiment.py` | 头部 `import exp.run_spawn_experiment as spawn` + L754 `caplog.at_level("WARNING", logger="exp.run_spawn_experiment")` + L872 `from exp.run_spawn_experiment import _BaseSpawnRunner, _SpawnCommon` → 全部改 `exp.trajectory_deviation.run_spawn_experiment`，**包括 logger name 字符串** |
| `test_run_step1b_gt.py` | 头部 → `exp.trajectory_deviation.run_step1b_gt` |
| `test_analyze_deviation_results.py` | 头部 → `exp.trajectory_deviation.analyze_deviation_results` |

**B5 scripts 改 import（2 文件 + 1 个 docstring）：**
- `scripts/verify_env_save_restore.py` L70: `from exp._libero_env` → `from exp.trajectory_deviation._libero_env`
- `scripts/verify_restore_obs_equivalence.py` L79–80: 同上，以及 `from exp.run_spawn_experiment` → `from exp.trajectory_deviation.run_spawn_experiment`
- `scripts/dump_step1a_failed_inits.py` docstring（L4、L53）引用 `exp/run_cache_experiments.py` → 按 §4 B3 全面同步原则一并更新到 `exp/cache_experiment/run_cache_experiments.py`

### Wave C — 外部文档引用

**C1 docs 里 36 处 CLI 命令**（6 个文件）：
- `docs/run_cp1_cache_experiment.md`（15 处）：`python exp/…` → `python exp/cache_experiment/…` 或 `python exp/common/…`（对 helper）
- `docs/run_temporal_prune_experiment.md`（10 处）：根据脚本归属分别更新
- `docs/temporal_prune_guide.md`（6 处）：`python exp/build_in_memory_cache_artifact.py` → `python exp/cache_experiment/build_in_memory_cache_artifact.py`
- `docs/cache_migration_guide.md`（2 处）、`.en.md`（2 处）：同上
- `docs/cache_system_tutorial.md`（1 处）

**C2 `docs/openpi_reference.md` 的 `exp/` 目录树**：改写 L72 附近的目录树说明。

**C3 `src/openpi/cache/` docstring 引用**（3 处非功能性 comment）：
- `key_builder.py` L366, L417: `exp/qdrant_openpi_common.py` → `exp/qdrant_step_knn/qdrant_openpi_common.py`
- `backends/qdrant_backend.py` L447: `exp/qdrant_ingest_openpi.py` → `exp/qdrant_step_knn/qdrant_ingest_openpi.py`

**C4 `logs/` 里对 `exp/…` 的引用（精确边界）**：
- **仅更新** 本 plan 文件（`logs/exp_reorg_plan.log.md`）与 `logs/README.md`（状态同步，WA §4.3 强制要求）
- **不更新** 其他 active logs（如 `trajectory_deviation_corrective_*.log.md`、`stage_device_placement_plan.log.md`）中对 `exp/…` 的路径引用
- `logs/archive/` 原则上只读，保留原 CLI 命令作为历史快照
- 理由：Owner 明示"log 可以先不管"，active logs 的路径引用与代码现状脱节属已知偏差，待后续单独处理

---

## 5. Verify（L2 必选项）

### V1 单元测试（含 scripts 受影响面）
```bash
uv run pytest \
  tests/exp/ \
  tests/scripts/test_verify_smoke_scripts.py \
  tests/scripts/test_dump_step1a_failed_inits.py \
  -x --tb=short
```
- 期望：全绿
- 若任意断言失败 → 回滚到 Wave B 起点定位 import 差异

### V2 Entry-script smoke（覆盖 direct-run 与 `python -m` 两种模式）
```bash
# Direct-run mode（验证 sys.path parents[2] 正确）
uv run python exp/cache_experiment/run_cache_experiments.py --help
uv run python exp/trajectory_deviation/run_spawn_experiment.py --help
uv run python exp/trajectory_deviation/compute_deviate_scores.py --help
uv run python exp/trajectory_deviation/run_step1b_gt.py --help
uv run python exp/qdrant_step_knn/qdrant_ingest_openpi.py --help
uv run python exp/qdrant_step_knn/qdrant_verify_openpi.py --help
uv run python exp/qdrant_step_knn/qdrant_step_knn_experiment.py --help
uv run python exp/qdrant_step_knn/toy_qdrant_server.py --help

# Module mode（验证作为 package 子模块可 import）
uv run python -m exp.cache_experiment.run_cache_experiments --help
uv run python -m exp.trajectory_deviation.run_spawn_experiment --help
uv run python -m exp.qdrant_step_knn.qdrant_step_knn_experiment --help
```
- 期望：每个都打印 argparse help 无 `ModuleNotFoundError`
- qdrant 依赖（`qdrant-client` 等）若不在 uv 环境内：该命令若报 qdrant 库缺失不视为失败；但若报 `ModuleNotFoundError: No module named 'openpi'` / `No module named 'exp'` 必须视为失败

### V3 目录结构 + 路径自检
```bash
# Top-level 只允许 __init__.py
find exp -maxdepth 1 -type f ! -name "__init__.py"   # 期望无输出

# 随机抽 docs CLI 命令验证文件存在
test -f exp/cache_experiment/build_in_memory_cache_artifact.py
test -f exp/cache_experiment/run_cache_experiments.py
test -f exp/trajectory_deviation/run_spawn_experiment.py

# 残留旧路径引用扫描
rg -n "exp/[a-z_]+\.py|python -m exp\.[a-z_]+|from exp import|from exp\._[a-z_]+|import exp\._[a-z_]+" \
   exp scripts src tests docs | grep -v "archive/" | grep -v "exp_reorg_plan"
```
- 期望：`find` 无输出
- 期望：`rg` 命令除"有意保留的注释"之外无输出

### V4 索引同步
- `logs/README.md` 增补本 plan 条目（`Plan` → `Validated` 推进流程）
- `docs/README.md` 无需改（未移 docs 文件）

---

## 6. 风险与缓解

| ID | 风险 | 缓解 |
|----|------|------|
| R1 | `parents[2]` 层级错误（例如漏改或写成 `parents[1]`） | Wave B 每改一个 entry script 立即 `python … --help` smoke；B2 专门 review 8 个 entry/root-path 点 |
| R2 | 遗漏 import 点（不在 tests/scripts/docs 的隐藏引用） | Round 1 审查已证明 plan 起草阶段的 grep 有函数内延迟 import / `from exp import X` / logger name 字符串等遗漏，不再信任前置调查结果。改为：**执行时**以 §4 B3 和 §4 B4 中给出的两条 grep 命令作为强制验收；任何残留命中都必须归入"保留清单"或修复。`logs/archive/` 保留为历史快照不改 |
| R3 | `generate_temporal_prune_yamls.py` L174 硬编码 CLI 字符串被生成进 YAML 里流传到用户 | B3 专门列为子项；smoke 生成一个 yaml 验证 |
| R4 | `__init__.py` 意外引入 side effect | 每个文件只含一行英文 package docstring（满足 WA §3.2），无 import / 副作用 / 导出符号；code review 时机械核对 |
| R5 | 某个 `qdrant_*` 脚本有动态 import（`importlib` / `__import__`） | 执行 Wave B 前以 `rg -n "importlib\.import_module\|__import__" exp scripts tests` 再扫一次作为 gate；命中任一处均需手工 review。不再依赖 plan 起草阶段的单次 grep 结论 |
| R6 | 回滚成本 | `git mv` 保留历史，回滚 = `git revert` 单一 cleanup commit |
| R7 | Breaking CLI path：外部 shell 脚本 / 旧 YAML / 个人 notebook 里的 `uv run exp/foo.py` 将失效 | **已接受风险，不提供旧路径 shim**。理由：shim 与 "`find exp -maxdepth 1` 无输出" 目标冲突，且 shim 本身是技术债。缓解：docs 同步（C1/C2），commit message 注明 breaking change，仓内无遗留即视为完成。仓外（notebook、私人 shell）由使用方自行迁移 |

---

## 7. 执行顺序建议

1. Wave A 全部一次性完成（目录创建 + 24 次 git mv），不改任何文件内容
2. Wave A 完后 `ls exp/` 目测结构 + 跑 `find exp -maxdepth 1 -type f ! -name "__init__.py"`（机器验收，期望空输出），然后 Wave B 开始改内容
3. Wave B 按 B1→B2→B3→B4→B5 顺序；B1+B2 改完跑 V2 smoke（不等 tests）定位 import 错
4. V1 pytest 全绿后进 Wave C
5. Wave C 改完再跑 V1 + V3
6. 提交 commit（建议单 commit "reorganize exp/ by experiment"）

---

## 8. Status Log

- **2026-04-14**：Plan 起草。
- **2026-04-14**：Owner 追加约束——docs 必须随重构同步更新（已在 C1/C2/C3 覆盖），active logs 的路径引用本轮不跟进（§4 C4 已调整）。
- **2026-04-14**：G1 Approved（Codex，共 3 轮审查后批准）。见 §9。
- **2026-04-14**：Code 阶段执行。Wave A (5 `__init__.py` + 24 `git mv`) / Wave B1-B5 (import + sys.path + docstring 同步) / Wave C (docs 36 处 CLI + `openpi_reference.md` 目录树 + `src/openpi/cache/` 3 处 docstring + logs 索引) 全部落地。Verify V1 (`tests/exp/` + 受影响 scripts = 166 passed) / V2 direct-run + `python -m` smokes / V3 `find` + 残留 grep 全部通过。待 G2 独立审查。
- **2026-04-14**：G2 Review（Codex）CHANGES REQUESTED —— 见 §10。主要阻塞点：旧 `exp/...` / `exp._...` 路径仍残留在 active code/tests docstring 或 usage 示例中，和 B3/V3 的验收口径不一致。
- **2026-04-14**：G2 Re-review（Codex）APPROVED —— 见 §12。第一轮 5 处旧路径残留已修复，精确旧模块扫描为空，pytest 166/166 通过。

---

## 9. G1 Approval

- 审查者：Codex
- 日期：2026-04-14
- 结论：**Plan approved.** 可进入 Code 阶段。

**Approval scope**：
- 24 次 `git mv` + 5 个带 package docstring 的新 `__init__.py`
- import / `sys.path` / docs / active code path 引用同步
- 不提供旧顶层 CLI shim，接受 breaking path 风险
- 按 §5 执行 `tests/exp/`、受影响 `tests/scripts/`、direct-run / `python -m` smoke、目录与残留引用扫描

**G2 重点**：G2 仍需独立代码审查，重点核对是否严格按本 plan 执行、是否无旧路径残留、Verify 是否按 §5 完成。

---

## 10. G2 Review — Changes Requested

- 审查者：Codex
- 日期：2026-04-14
- 结论：**Changes requested.** 运行层面的 import / entry-script smoke 没看到失败，但当前变更还不能按本 plan 标为 G2 通过，因为 B3/V3 承诺的 active path/docstring 全同步仍有残留旧路径。

### Blocking Findings

1. **旧 CLI 路径仍留在已移动 entry script 的 usage 示例里。**
   - `exp/qdrant_step_knn/toy_stage1_server.py:14` 仍写 `uv run exp/toy_stage1_server.py`。
   - 该文件已从 `exp/toy_stage1_server.py` 移到 `exp/qdrant_step_knn/toy_stage1_server.py`，且 §6 R7 明确接受 breaking path 的前提是仓内无遗留旧路径、docs 同步。这个 usage 示例是用户会直接复制的入口说明，应更新为新路径。
   - 建议：改为 `uv run python exp/qdrant_step_knn/toy_stage1_server.py ...`（或项目统一认可的等价 direct-run 形式）。同时考虑把该脚本加入 V2 smoke 清单，或在 plan 里明确说明为什么它不属于 V2 覆盖范围。我手动补跑 `uv run python exp/qdrant_step_knn/toy_stage1_server.py --help`，结果通过。

2. **active code/tests docstring 仍引用旧模块/旧磁盘路径。**
   - `scripts/verify_env_save_restore.py:59` 仍写 `exp._libero_env.build_libero_env`，实际 import 已改为 `exp.trajectory_deviation._libero_env`。
   - `exp/trajectory_deviation/run_step1b_gt.py:243` 仍写 `exp._subprocess.build_subprocess_cmd`，实际 import 已改为 `exp.common._subprocess`。
   - `exp/common/_unit_key.py:42` 仍写 `exp/run_step1b_gt.py`，实际文件已移到 `exp/trajectory_deviation/run_step1b_gt.py`。
   - `tests/exp/test_run_step1b_gt.py:1` 仍写 `exp/run_step1b_gt.py`。
   - 这些不改变 runtime，但和 §0 / §4 B3 的“active `exp/` 内所有旧路径 docstring/注释引用同步”、以及 §8 中“V3 残留 grep 全部通过”的陈述冲突。G2 不能在已知旧路径残留下批准。
   - 建议：把以上 docstring 全部同步到新路径，并复跑旧路径残留扫描。

3. **V3 验收口径目前不足以支撑“无旧路径残留”的结论。**
   - 复跑 §5 V3 的命令时，只命中新路径：
     `exp/trajectory_deviation/analyze_deviation_results.py:29`、
     `exp/qdrant_step_knn/qdrant_step_knn_experiment.py:7`。
   - 但针对本次 24 个旧模块名做精确扫描，仍命中上述旧路径/docstring 残留。原因是 V3 的正则未覆盖裸 `exp._libero_env` / `exp._subprocess` 这类非 import 上下文，也没有精确枚举旧模块名。
   - 建议：修复残留后，追加执行一条旧模块精确扫描，例如覆盖 `exp._libero_env`、`exp._subprocess`、`exp/run_step1b_gt.py`、`exp/toy_stage1_server.py` 等本次移动前的完整旧路径/模块名。该扫描应只允许 `logs/exp_reorg_plan.log.md` 自身记录历史命中。

### Verification Re-run

- `find exp -maxdepth 1 -type f ! -name "__init__.py"`：通过，无输出。
- `uv run python -m pytest tests/exp/ tests/scripts/test_verify_smoke_scripts.py tests/scripts/test_dump_step1a_failed_inits.py -x --tb=short`：通过，`166 passed, 1 warning`。
- `uv run pytest tests/exp/ tests/scripts/test_verify_smoke_scripts.py tests/scripts/test_dump_step1a_failed_inits.py -x --tb=short`：本地失败，原因是 `.venv/bin/pytest` shebang 指向不存在的旧 Windows/OneDrive Python 路径；同一套测试用 `uv run python -m pytest ...` 通过，因此我不把它判为代码回归，但建议在最终 Verify 记录里注明实际可复现命令或修复 `.venv`。
- V2 direct-run / module smoke：我复跑了 plan 中列出的 8 个 direct-run 和 3 个 `python -m` smoke，均通过；额外补跑 `exp/qdrant_step_knn/toy_stage1_server.py --help`，也通过。
- `__init__.py` 机械核对：5 个新 package `__init__.py` 仅含一行英文 package docstring，无 import / side effect。

### Non-blocking Notes

- `docs/README.md` 与 `logs/README.md` 相对 HEAD 有较大索引重排，超过本 plan §5 V4 “只同步本 plan 条目 / docs README 无需改”的最小范围。如果这是同一批文档整理的一部分，建议在提交说明或 plan status log 中显式说明；如果不是，建议拆分提交，避免 `exp/` 重组审查混入无关索引重构。

### G2 Gate

G2 可在以下条件满足后再审：

1. 修复 blocking findings 中列出的旧路径/docstring 残留。
2. 用精确旧模块名扫描确认 active `exp/` / `scripts/` / `src/` / `tests/` / `docs/` 无旧路径残留。
3. 更新本 section 下方追加 Author Response 或修复记录，并同步 `logs/README.md` 状态。

---

## 11. Author Response to G2 — Fixes Applied

- 作者：Claude
- 日期：2026-04-14
- 结论：全部 blocking findings 接受并修复，无驳回项。

### Finding 1 — 旧 CLI usage 示例（accepted）

- `exp/qdrant_step_knn/toy_stage1_server.py:14`：`uv run exp/toy_stage1_server.py` → `uv run exp/qdrant_step_knn/toy_stage1_server.py`。
- V2 smoke 补充记录：审查者已手动补跑 `toy_stage1_server.py --help` 通过；下次 Verify 若继续覆盖 entry-script smoke，可将该文件纳入 V2 列表。

### Finding 2 — active docstring 旧模块/路径引用（accepted，共 4 处）

- `scripts/verify_env_save_restore.py:59`：`exp._libero_env.build_libero_env` → `exp.trajectory_deviation._libero_env.build_libero_env`。
- `exp/trajectory_deviation/run_step1b_gt.py:243`：`exp._subprocess.build_subprocess_cmd` → `exp.common._subprocess.build_subprocess_cmd`。
- `exp/common/_unit_key.py:42`：`exp/run_step1b_gt.py` → `exp/trajectory_deviation/run_step1b_gt.py`。
- `tests/exp/test_run_step1b_gt.py:1`：`exp/run_step1b_gt.py` → `exp/trajectory_deviation/run_step1b_gt.py`。

### Finding 3 — V3 残留扫描口径不足（accepted）

追加旧模块精确扫描覆盖三类原先遗漏的形态：

1. 非 import 上下文的裸 dotted 引用：`exp\.(_libero_env|_subprocess|_unit_key|_cache_config_rpc|_run_state_base|run_step1b_gt|run_spawn_experiment|run_cache_experiments|compute_deviate_scores|analyze_deviation_results|calibrate_score_sum_stats|build_in_memory_cache_artifact|generate_cache_run_yamls|toy_qdrant_server|toy_stage1_server|qdrant_ingest_openpi|qdrant_verify_openpi|qdrant_step_knn_experiment|qdrant_step_knn_common|split_cache_config_fields|temporal_prune_experiment|_signatures)\b`
2. 裸磁盘路径 `exp/<leaf>\.py`（排除已移动到子目录者）。
3. `python -m exp\.<leaf>`（缺少子包段者）。

三条扫描对 active `exp/` / `scripts/` / `src/` / `tests/` / `docs/` 均无命中；仅 `logs/exp_reorg_plan.log.md` 本文件存在历史记录命中（预期）。

### Verification Re-run

- 精确旧模块名扫描：三条模式全部空结果（除 plan 自身）。
- `uv run python -m pytest tests/exp/ tests/scripts/test_verify_smoke_scripts.py tests/scripts/test_dump_step1a_failed_inits.py -x --tb=short`：`166 passed, 1 warning in 1.97s`。
- `find exp -maxdepth 1 -type f ! -name "__init__.py"`：空（无回退）。

### Non-blocking Note 响应

- 关于 `docs/README.md` / `logs/README.md` 索引重排：本 plan 对 `logs/README.md` 的改动仅限新增本 plan 条目 + 状态同步；`docs/README.md` 的编辑限于 Wave C 中对 `exp/` 目录结构段落的同步替换。若审查者观察到“较大索引重排”，可能混入了未提交的本地编辑或其他批次；我复查了本分支相对 `main` 的 diff，未发现超出 §5 V4 范围的改动。

### 状态推进

- Plan 状态：`G2 Changes Requested` → `Implemented (G2 Re-review Pending)`。
- `logs/README.md:60` 同步。

---

## 12. G2 Re-review — APPROVED

- 审查者：Codex
- 日期：2026-04-14
- 结论：**G2 approved.** 第一轮 blocking findings 已修复，当前变更可以进入提交阶段。

### Re-check Scope

- 复核 §10 blocking finding 1：`exp/qdrant_step_knn/toy_stage1_server.py:14` 已更新为 `uv run exp/qdrant_step_knn/toy_stage1_server.py`。
- 复核 §10 blocking finding 2：以下 4 处 active docstring / test docstring 已更新为新路径：
  - `scripts/verify_env_save_restore.py:59`
  - `exp/trajectory_deviation/run_step1b_gt.py:243`
  - `exp/common/_unit_key.py:42`
  - `tests/exp/test_run_step1b_gt.py:1`
- 复核 §10 blocking finding 3：对 active `exp/` / `scripts/` / `src/` / `tests/` / `docs/` / `logs/README.md` 执行旧模块名 dotted、bare path、`python -m` 三类精确扫描，均无命中。

### Verification Re-run

- `find exp -maxdepth 1 -type f ! -name "__init__.py"`：通过，无输出。
- `uv run python -m pytest tests/exp/ tests/scripts/test_verify_smoke_scripts.py tests/scripts/test_dump_step1a_failed_inits.py -x --tb=short`：通过，`166 passed, 1 warning`。
- `uv run python exp/qdrant_step_knn/toy_stage1_server.py --help`：通过。

### Residual Notes

- 仍建议提交说明明确这是 breaking path change：旧 `exp/<script>.py` 顶层 CLI 不再保留 shim。
- `uv run pytest ...` 在上一轮审查环境中因 `.venv/bin/pytest` 旧 shebang 失败；本轮以可复现的 `uv run python -m pytest ...` 作为 G2 验收命令。
