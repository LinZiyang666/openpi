# Trajectory 子实验化 + analysis 按子实验重组（libero_spatial / libero_10）

- **Status**: `Plan`
- **Level**: L1（实际代码变动仅两个 `exp/` plot 脚本的路径修改 + 新增常量模块；其余均为 `git mv` 搬运。按 WA §2.1 "exp/ 脚本"条归 L1）
- **Authority**: Execution
- **Process**: Code → Verify → Commit（L1 不过 G1/G2）

---

## 1. 背景

Phase1 实验已经按子实验划分成 `config/phase1/libero_spatial/`（40 yaml）与 `config/phase1/libero_10/batch{1,2,3}/`（60 yaml，libero_10 的 Phase1 即将跑完，batch1/2/3 的 run log 与 `experiment_state.json` 随后会同步落盘）。

现在需要把 **trajectory 实验** 以及对应的 **analysis 目录** 做同样的子实验化拆分，并把两个 `plot_results.py` 里重复的常量抽出公共模块。`data/` 层尽量同步对齐，避免 `config / data / analysis` 三层结构错位。

libero_10 的 trajectory 权重档位（top1 / top2 / 2nd-worst）必须基于 libero_10 **自己的 Phase1 排名**重新选取，而 Phase1 libero_10 尚未出榜 —— 因此本计划**只建 libero_10 trajectory 的目录骨架（含 batch1/2/3 三个空目录），不生成 YAML**，待 Phase1 libero_10 完成后在后续 L1 任务里生成。

## 2. Scope

### 2.1 In scope

1. 目录重组（纯 `git mv`）：
   - `exp/common/config/trajectory/*.yaml`（60 份） → `exp/common/config/trajectory/libero_spatial/`
   - `exp/common/data/trajectory/*.log`、`experiment_state.json`（61 项） → `exp/common/data/trajectory/libero_spatial/`
   - `exp/common/data/phase1/*.log`、`experiment_state.json`（41 项） → `exp/common/data/phase1/libero_spatial/`
   - `exp/common/analysis/phase1/{phase1_analysis.md, phase1_analysis.pdf, phase1_results.pdf, phase1_results.png, plot_results.py}`（5 份） → `exp/common/analysis/phase1/libero_spatial/`
   - `exp/common/analysis/trajectory/{trajectory_analysis.md, trajectory_analysis.pdf, trajectory_results.{pdf,png}, trajectory_results_facets.{pdf,png}, plot_results.py}`（7 份） → `exp/common/analysis/trajectory/libero_spatial/`

2. 新建 libero_10 目录骨架（空目录 + `.gitkeep`）：
   - `exp/common/config/trajectory/libero_10/batch{1,2,3}/`
   - `exp/common/data/phase1/libero_10/`
   - `exp/common/data/trajectory/libero_10/`
   - `exp/common/analysis/phase1/libero_10/`
   - `exp/common/analysis/trajectory/libero_10/`

3. 抽公共 analysis 模块 `exp/common/analysis/plot_common.py`：
   - 常量：`KEY_BUILDER_ORDER`、`KEY_BUILDER_LABELS`、`WEIGHT_IDS`、`WEIGHT_LABELS`、`KB_COLORS`、`BAR_COLORS`、`ROLE_MARKERS`、`ROLE_LABELS`
   - 子模块化两个 plot 脚本，使其 `from exp.common.analysis.plot_common import ...`

4. 修正跨目录引用：
   - `analysis/trajectory/libero_spatial/plot_results.py` 中 `PHASE1_STATE_FILE = SCRIPT_DIR.parent / "phase1" / "experiment_state.json"` 改为指向 `exp/common/data/phase1/libero_spatial/experiment_state.json`（路径解析从 `exp/common/data/` 起算；用 `Path` 往上拼）。
   - 两个 `plot_results.py` 中 `STATE_FILE = SCRIPT_DIR / "experiment_state.json"` 改为读 `exp/common/data/{phase1|trajectory}/libero_spatial/experiment_state.json`，**彻底消除"要把 state.json 手动 cp 进 analysis 目录才能跑 plot"的隐性依赖**。

5. 索引同步：
   - `logs/README.md` 新增本计划条目。

### 2.2 Out of scope

- **不生成 libero_10 trajectory YAML**：等 Phase1 libero_10 结果出榜后另起 L1 任务。
- **不动 `docs/experiments/cp1_cache.md`** 里的旧路径示例（libero_spatial 平铺时代的 `exp/common/config/phase1/*.yaml`）。该文档过期问题独立于本次重组，如需修复另起 L0/L1 doc 维护任务。
- **不修改 `exp/common/analyze_cache_results.py` / `generate_cache_run_yamls.py`**：它们对路径是完全参数化的，docstring 里的示例命令可以先留着；是否顺手刷新 docstring 留给 Verify 阶段判断。
- **不处理 `data/trajectory/experiment_state.json` 中包含 `traj_d5_043_d_rrf_w6` 崩溃后重跑条目**等数据语义问题；本次只做路径搬运。

## 3. Files Touched

### 3.1 移动（`git mv`）

| 源 | 目标 |
|---|---|
| `exp/common/config/trajectory/traj_*.yaml` × 60 | `exp/common/config/trajectory/libero_spatial/` |
| `exp/common/data/trajectory/{traj_*.log, experiment_state.json}` × 61 | `exp/common/data/trajectory/libero_spatial/` |
| `exp/common/data/phase1/{phase1_run_*.log, experiment_state.json}` × 41 | `exp/common/data/phase1/libero_spatial/` |
| `exp/common/analysis/phase1/{phase1_analysis.md, phase1_analysis.pdf, phase1_results.pdf, phase1_results.png, plot_results.py}` | `exp/common/analysis/phase1/libero_spatial/` |
| `exp/common/analysis/trajectory/{trajectory_analysis.md, trajectory_analysis.pdf, trajectory_results.pdf, trajectory_results.png, trajectory_results_facets.pdf, trajectory_results_facets.png, plot_results.py}` | `exp/common/analysis/trajectory/libero_spatial/` |

### 3.2 新增

- `exp/common/analysis/plot_common.py` — 公共常量
- `exp/common/config/trajectory/libero_10/batch{1,2,3}/.gitkeep`（3 个）
- `exp/common/data/phase1/libero_10/.gitkeep`
- `exp/common/data/trajectory/libero_10/.gitkeep`
- `exp/common/analysis/phase1/libero_10/.gitkeep`
- `exp/common/analysis/trajectory/libero_10/.gitkeep`

### 3.3 修改（内容，非移动）

| 文件 | 修改点 |
|---|---|
| `exp/common/analysis/phase1/libero_spatial/plot_results.py` | `STATE_FILE` 重新指向 `exp/common/data/phase1/libero_spatial/experiment_state.json`；`from exp.common.analysis.plot_common import ...` 抽出常量 |
| `exp/common/analysis/trajectory/libero_spatial/plot_results.py` | `STATE_FILE` + `PHASE1_STATE_FILE` 两条路径重指；`from exp.common.analysis.plot_common import ...` 抽出常量 |
| `logs/README.md` | 在 "Phase1 Experiments" 节后新增 "Trajectory Experiments" 小节，链到本 plan |

## 4. Interfaces

### 4.1 新增公共模块 `exp/common/analysis/plot_common.py`

纯常量模块，无副作用：

```python
"""Shared constants for phase1/trajectory plotting scripts."""

import matplotlib.pyplot as plt
import numpy as np

KEY_BUILDER_ORDER = ["a", "b1", "b2", "c", "d"]
KEY_BUILDER_LABELS = {
    "a":  "cp1_mean_pool",
    "b1": "cp1_spatial_pool_16",
    "b2": "cp1_spatial_pool_64",
    "c":  "cp1_max_pool",
    "d":  "clip",
}

WEIGHT_IDS = [f"w{i}" for i in range(1, 9)]
WEIGHT_LABELS = {
    "w1": "v0=1.0",
    "w2": "rs=1.0",
    "w3": "v0=.5 rs=.5",
    "w4": "v0=.25 rs=.75",
    "w5": "v0=.25 v1=.25 rs=.5",
    "w6": "v0=.15 v1=.1 rs=.75",
    "w7": "v0=.1 v1=.1 rs=.8",
    "w8": "v0=.5 v1=.25 rs=.25",
}

BAR_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
KB_COLORS = {
    kb: c for kb, c in zip(KEY_BUILDER_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))
}

ROLE_MARKERS = {"top1": "*", "top2": "s", "2nd_worst": "o"}
ROLE_LABELS = {
    "top1": "phase1 top-1",
    "top2": "phase1 top-2",
    "2nd_worst": "phase1 2nd-worst",
}
```

注意：`plt.cm.tab10(...)` 在 import 时就会被执行，这本身就是 phase1 / trajectory 两个老脚本当前的行为（两个脚本都在 module top-level 做同样的颜色表计算），不引入新副作用。

### 4.2 路径重定位约定

`exp/common/analysis/{phase}/{subexp}/plot_results.py` 读数据时：

```python
REPO_ROOT = Path(__file__).resolve().parents[4]  # analysis/<phase>/<subexp>/plot_results.py → repo root
STATE_FILE   = REPO_ROOT / "exp/common/data" / phase_name / subexp_name / "experiment_state.json"
```

trajectory 版还需要 phase1 baseline：

```python
PHASE1_STATE_FILE = REPO_ROOT / "exp/common/data/phase1" / subexp_name / "experiment_state.json"
```

`phase_name / subexp_name` 用脚本所在目录推断，避免复制粘贴时漏改。

## 5. Integration

- 无推理路径 / 无 server 路径改动。
- `plot_common.py` 仅被 analysis 脚本 import，不进入线上链路。
- 移动后的 YAML 内部 `preload_path` 完全不变（仍指向 `exp/common/data/cache_artifacts/libero_spatial/*.pkl`），与 runner 的契约不破。
- `run_cache_experiments.py` 等 runner 接受 `--yaml-dir` / `--state-path` 作为参数，本次不改；用户后续跑 libero_10 trajectory 时直接传新路径即可。

## 6. Test Strategy

本次是**目录重组 + 路径字符串修改**，没有新业务逻辑。

1. **Smoke：plot 脚本在新结构下跑通**
   - `cd exp/common/analysis/phase1/libero_spatial && uv run python plot_results.py`
   - `cd exp/common/analysis/trajectory/libero_spatial && uv run python plot_results.py`
   - 期望：读到 `experiment_state.json` 且在当前目录重新生成 `.png` / `.pdf`，与旧产物（git 历史里的 png）视觉一致（文件大小同量级即可）。

2. **`uv run pytest`**（Verify §6 要求）：预期无任何回归，因为本次不触碰 `src/` / 测试套件。

3. **路径残留检查**：`grep -rn "analysis/phase1/\|analysis/trajectory/\|config/trajectory/" exp/ logs/README.md` 确认没有遗漏引用。

## 7. Risk Register

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| `data/phase1/experiment_state.json` 搬移时被 libero_10 Phase1 后续 batch 同步任务写坏 | 中 | 高 | 迁移前与用户确认：libero_10 的后续 batch 数据同步应直接落到新路径 `data/phase1/libero_10/`；不会再追加写入 `data/phase1/experiment_state.json` |
| `plot_results.py` 路径修改后 baseline 找不到，图出错 | 中 | 低 | Smoke test 覆盖；失败时脚本 early-exit 而非生成错图 |
| Git 认为是删除+新增而非 mv | 低 | 低 | 使用 `git mv`；commit 后用 `git log --follow` 确认 |
| 公共模块常量漂移 | 低 | 低 | 常量均为字符串/tuple，被 import 即共享，不做二次拷贝 |
| libero_10 骨架目录被 git 视为空（不 tracked） | 高 | 低 | 每个新建空目录放 `.gitkeep` |
| 改动触发 pre-commit 格式化 `plot_results.py` 导致附带噪声 diff | 低 | 低 | 本次同时做路径修改，噪声可接受；commit 前 `ruff --check` |

## 8. Rollback

单 commit 完成所有移动 + 公共模块，回滚直接 `git revert <sha>`。

## 9. Stages & Estimates

| 阶段 | 内容 | 估时 |
|---|---|---|
| §4 Code | `git mv` + 建空目录骨架 + 写 `plot_common.py` + 改两个 `plot_results.py` 的路径与 import + 更新 `logs/README.md` | ~30 min |
| §6 Verify | Smoke plot 两脚本 + `uv run pytest` + grep 残留 | ~10 min |
| §7 Commit | 按"按子实验划分 trajectory 与 analysis + 抽公共 plot 模块"单 commit | ~2 min |
