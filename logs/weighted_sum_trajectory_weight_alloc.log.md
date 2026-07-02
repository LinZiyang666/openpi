# Weighted-Sum Trajectory 每步权重筛选搜索（libero_spatial）

> **Status**: `Implemented`（G1/G2 APPROVED / code de7c499 / 主跑完成 2026-07-02：17,100 ep，aggregate SR 0.674，0 err/0 ALERT / 分析出 results.md+decision.json）
> **结论（screening，非裁决）**：171 权重形状中**无一显著超 incumbent**（ΔSR>0 且 McNemar p<0.05 = 空），d3/d4/d5 最优 SR 0.73/0.72/0.72 **均 < d1 prior 0.74**；形状排序 peak≈decreasing > increasing/other > trough > uniform(最差)。→ **合作者假说（固定递减权重不合理致 d1>d3/d4/d5）不被支持**；退化更可能源于 trajectory(depth>1) 本身。详见 `analysis/libero_spatial/trajectory_weight_alloc/results.md`。裁决留 top-k 确认性重跑。
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-07-02
> **关联**: 续 weighted_sum 系列。前序 trajectory-search / threshold-pareto 发现 **d1(无 trajectory) SR 高于 d3/d4/d5**（mean SR 93.8% > 92.6% > 91.8% > 89.9%，见 `analysis/libero_spatial/threshold_pareto/threshold_pareto_results.md` §5.1）。合作者怀疑该退化源于 **trajectory 每步权重 `trajectory_weights` 分配不合理**（历史固定递减方案，从没按 depth 重搜）。本实验放弃任何预设假说，对每步权重做**筛选式（screening）搜索**：覆盖含边界在内的各种形状，产出**带不确定度的候选排名**，不做单阶段裁决性结论。

---

## 1. 动机与判据

### 1.1 现状
4 个 base（d1/d3/d4/d5）各是对应 depth 的 wsweep/trajectory winner，共用 `cp1_spatial_pool_16` + `weighted_score_sum_knn` + `always_hit`。其中 `trajectory_weights`（每步权重）是从老 trajectory 实验**直接复用的固定递减模板**，从未按 depth 重新搜索：

| depth | trajectory_weights（newest-first） | 模态权重 v0/v1/rs（精确 1/16 值） | 形状 |
|---|---|---|---|
| d1 | 无（单步） | 0.0625/0.5/0.4375 | — |
| d3 | `[0.5, 0.3, 0.2]` | 0.3125/0.125/0.5625 | 递减 |
| d4 | `[0.4, 0.3, 0.2, 0.1]` | 0.5625/0.1875/0.25 | 等差递减 |
| d5 | `[0.35, 0.25, 0.2, 0.12, 0.08]` | 0.3125/0.0625/0.625 | 递减 |

### 1.2 机制（代码亲验）
- `search_strategy.py:166-176`：`trajectory_weights` 为 **newest-first**（`weights_newest_first = self._trajectory_weights[:actual_depth]`，`history_newest_first = reversed(...)`），故 `weights[0]` = 当前步、`[1]` = 上一步…递减 = 当前步权重最大。
- `in_memory_backend.py:_score_trajectory`（:1035-1050）：最终 trajectory 分 = **各步相似度的加权和**（`accumulated_sim += weights[idx] * step_score`，`idx=0` 当前步），跨分支取 max。
- **`always_hit` 下 judge 恒 FULL_HIT、重放 top-1**（`emit_yamls.build_eval_config:105` 固定 `judge.type=always_hit`），故 **只有排名（argmax，选哪条 entry 重放）影响 SR**；分值绝对大小与归一化系数不影响排名。⇒ 本实验搜索的是**权重的相对"形状"**。
- **约束（`config.py:1581-1599 validate_cache_config`，亲验）**：`trajectory_depth>1` 时 `trajectory_weights` 必须**长度==depth、非负、和>0**（不要求单调 / 和为1 / 无 0；qdrant+depth>1 fail-fast）。本实验用 in_memory、全正权重，天然满足。

### 1.3 判据（screening，无预设假说）
本阶段是**筛选**，不是裁决。覆盖各种权重形状后：
1. **产出**：每 depth 内所有形状的 SR + **配对不确定度**（同 held-out init 的 paired 比较，见 §4.3）。
2. **排名**：按 SR 排序，报告候选 top-k（不宣称"追平/超过"为定论）。
3. **观察**：最优形状偏 递减 / 递增 / 均匀 / 当前步主导（near-boundary）/ 非单调？是否随 depth 漂移？incumbent 落在分布何处？
4. **不做的**：不据 100-ep 单阶段结果**裁决** "trajectory 能否用权重恢复 d1"——该判定留给候选 top-k 的**确认性重跑（更大样本，推荐后续，非本阶段）**。d1 天花板作**非裁决性 prior 参考**（§6 R5）。

**不预设"均匀更好"或"递减不合理"**——用覆盖含边界的搜索让数据说话。

---

## 2. 搜索矩阵（覆盖含边界，screening）

### 2.1 原理与不变量
- 固定：**每个 depth 复用它自己的 base**（模态权重 + Layer-1 zscore 归一化 + keybuilder + always_hit + write_policy=never 全不动），**只搜 `trajectory_weights`**。
- 权重 newest-first、和为 1、**全部 > 0**（无 `current_only`=d1 已有数据；无末尾 0=不退化 depth）。
- 搜索空间**必须覆盖 near-boundary（当前步主导，趋近 d1）区域**——若 trajectory 拖累强索引，最优区极可能就在那里。故不用"每维 ≥step"的截断单纯形（会把当前步上限卡在 0.6/0.625，系统性排除边界），改为三组并集。

### 2.2 三组权重集（并集去重，确定性可审计）

权重 newest-first、和为 1、全部 > 0。**canonicalization**：每分量 `round(w_i, 6)`，以该 6 位元组为去重身份。

**S1 — 当前步主导 × 尾形梯度（覆盖边界，all depths）**
- 当前步权重 `c ∈ {0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}`（8 档，**c=0.9 直抵 near-d1 边界**）。
- 尾部 `m = depth−1` 个老步按几何比 `q ∈ {0.5(递减尾), 1.0(平尾), 2.0(递增尾)}` 分配 `1−c`：**公式** `w[0]=c`；`w[1+j] = (1−c)·q^j / Σ_{k=0}^{m−1} q^k`（j=0..m−1，j=0 紧邻当前步）。全正（c=0.9 尾极小仍 >0，validator 合法）。
- 8×3=**24/depth**（同 depth 内无 c/q canonical 碰撞）。提供当前步主导陡→缓梯度（0.20 老步重 → 0.90 近乎单步）+ 尾形 3 态。这是本实验最重要的轴。
- **均匀点归属**：d5 uniform `.2×5` 由 **S1** 提供（`c=0.2, q=1`）；**d4 uniform `.25×4` 不在 S1**（`c=0.25` 不在 c 网格）→ 由 **S2** 提供；d3 的 `1/3` 两处都不落 ⇒ **d3 无精确 uniform**（最近似 `.4/.3/.3`∈S2）。

**S2 — 形状格点（内部形状覆盖：递增 / 单峰 / U / 平衡内部）**
- 均匀单纯形格点，每维 ≥ step，`w = parts/n`（`compositions(n, depth)` 正整数拆分）：**d3 @step0.1 (n=10) = 36**、**d4 @step0.125 (n=8) = 35**、**d5 @step0.125 (n=8) = 35**（= C(n−1,depth−1)）。
- 承担 **单调递增族**（S1 不干净覆盖）、d4 uniform(`.25×4`∈S2)、单峰、U 形、平衡内部。**注**：d5 uniform `.2×5` **不落** 8-into-5 格点（由 S1 提供，见上）；S2 是**内部形状覆盖**（非无偏穷举）。
- 递增族数（= partitions）：d3 8 / d4 4 / d5 3；配 S1 低-c 递增尾补充。

**S3 — incumbent 同批锚点**
- 取 tracked `emit_trajectory_yamls.DEPTH_WEIGHTS`：d3 `.5/.3/.2`（∈S2 d3 格点，自动含）、d4 `.4/.3/.2/.1`（off eighths → +1）、d5 `.35/.25/.2/.12/.08`（off → +1）。保证 3 个现有递减方案与新形状**同 server、同批、同 held-out init** 可比。

**重合 label 优先级**（cosmetic；分析读真实权重、不解析 ID）：同一 canonical 向量若属多组，label 取 **S3 incumbent > S1 > S2**；去重后保留 1 个 config。

### 2.3 精确规模（本会话脚本实算，emitter + 测试锁死）

| depth | \|S1\| | \|S2\| | \|S1∩S2\| | incumbent∈S2 | **union（实算）** |
|---|---|---|---|---|---|
| d3 | 24 | 36 | 8 | 是 | **52** |
| d4 | 24 | 35 | 0 | 否(+1) | **60** |
| d5 | 24 | 35 | 1 | 否(+1) | **59** |
| **合计** | | | | | **171** |

- 数值经 throwaway 脚本实算确认：d3=52 / d4=60 / d5=59。**171 config × 100 ep = 17,100 ep**。
- 各 depth 均衡 52–60；每 depth 递减/边界密（S1 8 档 c × 递减尾），递增/单峰/U（S2），均匀（S1 for d4/d5；d3 无精确 uniform）。
- **测试锁死** 每 depth `|union|` 与总 171（§5）；emitter emit 后断言 out-dir 恰为该 171 ID 集。

---

## 3. 代码改动（零改 src/ 与 conductor 核心）

| 路径 | 改动 |
|---|---|
| `exp/weighted_sum/emit_traj_weight_alloc.py` | **新**：per depth 生成 S1∪S2∪S3 的 trajectory_weights，用 tracked `calibration_normalizers.json` + winner cid（经 `grid3_weight_configs`）+ `build_eval_config` 重建 base（不依赖 gitignored base yaml），overlay `trajectory_weights`，写 `config/trajectory_weight_alloc/libero_spatial/eval/`，逐个 `load_cache_config` 自检。**仅 unlink out-dir 的 `*.yaml`，emit 后断言目录恰为预期 171 ID 集** |
| `tests/exp/test_traj_weight_alloc.py` | **新**（pytest 测试须在 `tests/<exp>/`）：见 §5 |
| `exp/weighted_sum/analysis/analyze_stepweight.py` | **新**（脚本 flat 放 `analysis/`，输出落 `analysis/libero_spatial/trajectory_weight_alloc/`——对齐 repo 约定，见 §3.2 偏离说明）：从 **eval YAML 读真实 trajectory_weights**（不解析 ID 串）；互斥形状分类（uniform/递减/递增/单峰/U/other + 正交 current_dominant）；从 journal 恢复配对（先 `task_uid` latest-ts 去重）算 **McNemar 精确 p + 固定seed bootstrap delta-SR CI** vs incumbent；完整性门**对齐 emitter `expected_ids()`**；出带图例的 per-depth SR×形状图 + top-k 候选表(`results.md`) + **全 config `decision.json`(data/，含 d1 天花板 prior)** |
| `exp/weighted_sum/analysis/__init__.py` | **新**：使 analysis 成为可 import 子包（对齐 `exp/trajectory_deviation/analysis/__init__.py` 先例，便于 `tests/exp` import 纯函数）|
| `.gitignore` | **不改**（calibration JSON 与 all_results.csv 已 tracked via 例外 :47-59；base 由 emitter 从 tracked 输入确定性重建，config/** 保持 gitignore）|
| `logs/README.md` | **同 commit 同步索引**（WA §4 Index Sync）|
| 复用零改 | `run_phase2.py`（`--strategy weight`）、`summarize.py`、`build_eval_config`、`grid3_weight_configs`、`init_holdout`、`ThresholdJudge`/conductor/server 全不碰 |

### 3.1 emitter 设计要点（精确接口 + 可复现 + 防污染）
- **精确权重来源（零手抄）**：模态权重**不硬抄两位小数**（会出 0.31-vs-真实-0.3125 且和=0.99 的错），而是从 tracked `emit_yamls.grid3_weight_configs(['vision_0','vision_1','robot_state'], step=0.0625, dom_min=0.1875)` 按 winner cid 取精确值：
  ```
  WINNER_CID = {3:'grid3_vision_0@31_vision_1@12_robot_state@56',   # → 0.3125 / 0.125  / 0.5625
                4:'grid3_vision_0@56_vision_1@18_robot_state@25',   # → 0.5625 / 0.1875 / 0.25
                5:'grid3_vision_0@31_vision_1@6_robot_state@62'}    # → 0.3125 / 0.0625 / 0.625
  ```
  三组实算和均 = 1.0（本会话已核）。incumbent tw 取 tracked `emit_trajectory_yamls.DEPTH_WEIGHTS[d]`。
- **base 重建（完整键名 + dict 访问）**：`entry = calib['cp1_spatial_pool_16']`（dict）；`build_eval_config(builder_type=entry['builder_type'], vector_dims=entry['vector_dims'], preload_path=..., weights=grid3_cfgs[WINNER_CID[d]], fields_calib=entry['fields'], trajectory_depth=d, trajectory_weights=<swept>)`。**必须用完整键 `vision_0/vision_1/robot_state`**（build_eval_config 按 `WEIGHTED_FIELDS` 全名取值；传 `v0/v1/rs` → 全零 → `no weighted fields enabled` fail）；`entry` 是 dict，用 `entry['fields']`（非 `.fields`）。
- **重建正确性 = 可重复测试（非人工 diff）**：`tests/exp` 断言 每个 `WINNER_CID[d]` ∈ grid3 输出、其 weights 和==1.0、重建的 incumbent config（tw=DEPTH_WEIGHTS[d]）过 `validate_cache_config` 且结构合法。**不依赖 gitignored base yaml**。
- **防 stale（安全删除）**：**不 `rmtree(out_dir)`**；改为 assert `out_dir` 解析后位于 `exp/weighted_sum/config/` 之下 + 仅 `unlink` 其中 `*.yaml`；emit 后 `assert {p.stem for p in out_dir.glob('*.yaml')} == expected_171_ids` 且无非-yaml stray。`run_phase2 --yaml-dir` glob 到的即恰本批 171。
- yaml_id = `cp1_spatial_pool_16__<winner_cid>__d{d}__<label>`，label ∈ `s1_c{cc}_q{qq}` / `s2_{parts}` / `incumbent`（唯一标签；真实权重由分析侧从 YAML 读，不解析 ID）。
- 全 config 过 `load_cache_config`→`validate_cache_config`（trajectory_weights 长度==depth、非负、和>0、in_memory、write_policy=never）。**d1 不进 eval**（emitter 断言 depth∈{3,4,5}）。

### 3.2 §4 Code 偏离说明（analyze 脚本位置）
G1-approved plan 原写 analyze 脚本路径为 `analysis/libero_spatial/trajectory_weight_alloc/analyze_stepweight.py`（嵌套）。**Code 阶段发现该路径与 repo 既有约定不符**：`exp/weighted_sum/analysis/` 下所有分析脚本一律 **flat 放**（`plot_threshold_pareto*.py` / `plot_trajectory_results.py` …），`analysis/libero_spatial/<exp>/` 子目录**只放输出**（results.md/png/pdf/csv）。故脚本改放 flat `exp/weighted_sum/analysis/analyze_stepweight.py` + 新增 `analysis/__init__.py`（对齐 `exp/trajectory_deviation/analysis/__init__.py`，使 `tests/exp` 可 import 纯函数），输出仍落 `analysis/libero_spatial/trajectory_weight_alloc/`。纯位置对齐，无逻辑/范围变化；G2 plan-conformance 声明此偏离。

---

## 4. 运行拓扑与流程（experiment-lifecycle skill §3–§4）

### 4.1 设备（仅 2 台）
- **Server = jupyter-ziyang10**（H200 NVL，cgroup 10C/32G）：`serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --cache_config <any eval yaml> policy:checkpoint --policy.config=pi05_libero --policy.dir=<pi05_libero_pytorch>`。NAT 后 → `tether expose ziyang10 --local 8000` → `weiland.top:14xxx`。**单 server 钉死**（决策实验，杜绝跨 GPU 污染）。
- **Client = timan107**（8×GTX1080 / 48C / 220G）：`run_phase2.py ... --workers 48 --gpus 8`（单 server → 48 worker 全给 ziyang10，无 `--server-workers`）。worker conda env `/scratch/zixuans8/libero_sim`。

### 4.2 主跑命令
```bash
PYTHONPATH=. uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/trajectory_weight_alloc/libero_spatial/eval \
    --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal  exp/weighted_sum/data/libero_spatial/trajectory_weight_alloc/journal.jsonl \
    --servers  weiland.top:<port> --task-ids 0-9 --eval-trials 10 \
    --workers 48 --gpus 8 --strategy weight --task-suite libero_spatial \
    --conda-env /scratch/zixuans8/libero_sim
```
- `--strategy weight`（`WeightSearchStrategy`，always_hit 纯回放，只出 SR）；**不需要** `--per-step-out`。
- 防泄漏硬门：`--init-map`（✅ `exp/common/data/db/libero_cache/libero_spatial_init_map.json`），仅取 held-out init，缺失 fail-fast。

### 4.3 聚合 + 配对分析
```bash
uv run exp/weighted_sum/summarize.py --journal <journal.jsonl> --out <results.json>   # 分母=done+failed
PYTHONPATH=. uv run exp/weighted_sum/analysis/analyze_stepweight.py \
    --journal <journal.jsonl> --yaml-dir <eval-dir> \
    --out-dir exp/weighted_sum/analysis/libero_spatial/trajectory_weight_alloc \
    --decision-out exp/weighted_sum/data/libero_spatial/trajectory_weight_alloc/decision.json
```
- 配对分析：从 journal 解析 `task_uid=yaml_id:eval:task_id:episode_idx`。**先按 `task_uid` 取最新 `ts` 的终态去重（与 `summarize.py` 一致，防 timeout/late-result 重复行误配对），再**建 `(task_id,episode_idx)` 配对键 + 校验每 config 恰 100 完整配对；同 `(task_id,episode_idx)` 跨 config 是同一初始状态（`held_out_inits` 确定性）→ paired binary outcomes → 每 config vs 该 depth incumbent 的 **McNemar 精确 p + 固定 seed paired-bootstrap delta-SR CI**。**全部 config 的完整 paired 统计写入机器可读 `decision.json`**（`results.md` 仅 top-k 摘要）；`--decision-out` **required 且落 `analysis/` 即 fail-fast**。报告候选（不宣称定论）。
- d1 天花板 SR 从 `phase2/all_results.csv` **稳健读取**（best regular-grid zscore `cp1_spatial_pool_16` mean SR = 0.74，不依赖脆弱精确 id），**标注为非裁决性 prior-run 参考**（同 jupyter-ziyang10 H200 系列、仅**非同批 prior run**，**非跨 GPU**）。

### 4.4 产物落位（遵 `docs/experiments/artifact_layout.md`）
- config: `config/trajectory_weight_alloc/libero_spatial/eval/` — **gitignore（`.gitignore:11`），但由 emitter + tracked calibration 完全可重建**（非入库）。
- data: `data/libero_spatial/trajectory_weight_alloc/{journal.jsonl, results.json, decision.json}`（gitignore，tar 归档）。**机器可读 `decision.json` 落 `data/`**（artifact_layout §2 — JSON 属 data/，非 analysis/）。
- analysis: `analysis/libero_spatial/trajectory_weight_alloc/{results.md, *.png}`（入库）；**人读决策表写进 tracked `results.md`**。

### 4.5 主跑前硬门 checklist（skill §3.13）
launch 前逐项 PASS：① ziyang10 git SHA == 本地；② `cp1_spatial_pool_16.pkl` 在 ziyang10 且 sha256 == 本地；③ eval yaml 在 server 侧（远端 emit 或同步）+ 数量/ID == 预期 171 集；④ server ready（skill §3.8 exit0）+ client→server 链路三层通；⑤ 1-cell smoke（~10 ep 单 yaml 全完成无 Traceback）；⑥ L1/L2/L3 监控就位。任一 ❌ 停。

---

## 5. 测试策略（`tests/exp/test_traj_weight_alloc.py` + emit 自检）

- **emit 自检（硬）**：所有 eval yaml emit 后过 `load_cache_config`→`validate_cache_config`（trajectory_weights 长度==depth、非负、和>0、in_memory、write_policy=never）。
- **生成器计数锁死**：`|S1|=24/depth`、`|S2|`= d3 36/d4 35/d5 35、**`|union|`= d3 52 / d4 60 / d5 59 / 总 171**（断言精确值，非区间）；每向量 长度==depth、全 >0、∑≈1（容差）；canonical 去重无重复；**S1 含 c=0.9 边界点**；d4/d5 uniform ∈ union、d3 无精确 uniform。
- **winner 重建正确性**：每 `WINNER_CID[d]` ∈ `grid3_weight_configs(...)`、weights 和==1.0、完整键名、重建 config 过 `validate_cache_config`。
- **派生-YAML 契约测试**：每 emit 出的 yaml 与其 depth 的重建 base **仅 `search_strategy.trajectory_weights` 不同**（deep-diff）、`trajectory_depth` 正确、ID 唯一、总数==171、**d1 不在 eval**。
- **形状分类器互斥单测**：precedence **uniform → 单调递减 → 单调递增 → 严格内部单峰 → 严格内部 U → other** 首次命中（互斥）；**严格/唯一极值**：plateau 峰 `[.1,.4,.4,.1]` / 谷 `[.4,.1,.1,.4]` → **other**（非 peak/trough），真单峰 `[.1,.3,.5,.3,.1]`→peak；`current_dominant`（`w[0]≥0.6`）正交布尔标签。
- **配对分析单测**：合成 journal 含 timeout/late 重复行 → **先按 `task_uid` latest-`ts` 去重**再配对；McNemar 计数正确。
- **bootstrap CI 单测**：固定 seed → 可复现（`r1==r2`）、`lo≤point≤hi`、空交集→(0,0,0)。
- **acceptance-gate + main 完整性**：`acceptance_check` **对齐 emitter `expected_ids()`（独立源，非输入自洽）** + 每 config paired-key 集恰等 `task_ids×range(trials)`（拒错 episode_idx / 缺 episode）；main 级：截断 yaml 批(≠171)→SystemExit、`--decision-out` 落 `analysis/`→SystemExit。
- **§6 Verify**：`uv run pytest` 全绿（零改 src/，预期无回归）。

---

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **100 ep 统计功效**：目标效应小（~2pp），单格 CI≈±9pp | **本阶段定义为 screening**（§1.3）：产出候选+配对不确定度，不裁决；paired McNemar/bootstrap 比 aggregate 更敏感；裁决留 top-k 确认性重跑（推荐后续）。用户已定 100 ep/单阶段 |
| R2 | **多重比较 / winner's curse**（171 次选择）| 报告候选 top-k 而非单一 winner 定论；配对 CI + 明示未做多重比较校正；确认性重跑独立验证（后续）|
| R3 | ~~d5 预算畸重~~（**已由 §2 重设计化解**：S1 承担 d5 递减/边界密度，各 depth 52–60 均衡）| — |
| R4 | **模态权重混淆**：4 base 模态权重各异 | within-depth tw 扫描干净；跨 depth / vs-d1 标注 caveat；d1 SR 作非裁决性天花板 |
| R5 | **d1 天花板为 prior-run 值**（同 jupyter-ziyang10 H200、**非同批 prior run**、**非跨 GPU**）| 分析明确标注为**非裁决性参考**；可选加 1 个同批 d1 anchor（用户明确"d1 已有数据"，默认不加）|
| R6 | **可复现 / 同步**：config gitignore | emitter 从 **tracked** calibration(:50)+winner cid 确定性重建；跑前核 pkl+config 的 sha256/数量 local↔ziyang10（§4.5）|
| R7 | **stale yaml 污染** | emitter 仅 unlink `*.yaml`+断言 out-dir==预期 171 ID 集（§3.1）；run 前 checklist ④ 再核数量/ID |
| R8 | **ziyang10 `tether push` 禁用** | 远端 git pull 代码后**直接 emit**（脚本入库、calibration tracked → 远端可重建 yaml）；pkl 确认已在（build 时放）；跑前定，不阻塞写代码 |
| R9 | **单 server 吞吐/walltime** | 3 rep + 48 worker；~1.0–1.5 ep/s → 17,100 ep ≈ 4–6 h；L1/L2/L3 监控守 err/进度 |
| R10 | **spawn OOM**（3 rep × ~14GB > 32GB cgroup） | `--replica-spawn-batch 2`（先 2 后 1 错峰，skill §3.7.1 已验）|

---

## 7. 锁定决策（G1 APPROVED）与主跑前遗留

- **搜索矩阵**：171 config（d3=52 / d4=60 / d5=59，实算锁死）× 100 ep = 17,100 ep。
- **d1**：不加同批锚点，用既有 SR（0.74）作非裁决性 prior-run 参考。
- **定位**：screening（只报候选 + 配对不确定度）；确认性重跑（top-k 更大样本）为推荐后续、非本阶段。
- **codename**：`trajectory_weight_alloc`；config `eval/` gitignore-but-regenerable。
- **主跑前遗留（不阻塞 Code）**：ziyang10 数据同步通道（R8）——远端 git pull 后直接 emit + 确认 pkl 在位，主跑前定。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-02 04:23 CDT

- [Blocking] [Concern] `analyze_stepweight.py` 未实现 G1 锁定的核心分析交付：没有 paired bootstrap CI、没有读取/展示 d1 prior baseline，并且只把 top-k（默认 5）写入 `decision.json`/`results.md`，其余 166 个配置的权重、SR 与 paired 统计被丢弃。— reasoning: 代码仅输出 McNemar p-value（p-value 不是 CI），全文件无 bootstrap 实现或 baseline/all_results 输入；而计划 §1.3 明确要求“所有形状的 SR + 配对不确定度”，§4.3 明确写 McNemar/paired bootstrap CI 和 d1 prior。应输出每个 config 的完整 paired 统计（机器可读 data artifact），实现固定 seed 的 paired bootstrap delta-SR CI 并测试复现性，接入/标注 d1 prior；top-k 只能是完整结果上的摘要。当前图的 shape 颜色也没有 legend/colorbar，无法解释“SR×形状”，应补可读图例。
- [Blocking] [Concern] 运行完整性硬门可被截断批次绕过：`main()` 把 YAML 目录自身的 stem 集合作为 `expected_ids`，因此只传 1 个 YAML + 对应 1 条 journal 也能通过并产出“结果”；`acceptance_check` 也只核每 task 的数量，不核 episode_idx 必须恰为 `0..trials-1`。— reasoning: 独立测试 `test_main_rejects_truncated_yaml_batch` 与 `test_acceptance_rejects_wrong_episode_indices` 均失败；这违背批准计划的 config 数==171、精确 100 配对键 fail-fast 门。分析器必须与 emitter 的 `expected_ids()`/锁定 per-depth 集独立交叉核验，并比较每 YAML 的 key set 是否恰等于 `set(product(task_ids, range(trials)))`，不能只验证两个不完整输入彼此一致。
- [Blocking] [Concern] “严格内部单峰/U”分类契约未实现：平台峰 `[.1,.4,.4,.1]` 被分类为 `peak`，平台谷 `[.4,.1,.1,.4]` 被分类为 `trough`。— reasoning: 当前实现的 `>= -tol` 只保证容差单调，未保证唯一 extrema/峰谷两侧存在严格变化；独立审查两项 probe 均失败。请明确并实现计划所写 strict/unique interior extremum（或若确实要 plateau 类，先修订分类名称与契约），并补提交测试覆盖平台和多极值边界。
- [Blocking] [Concern] `--decision-out` 的默认行为违反 artifact layout：省略参数时把 `decision.json` 写进 `out_dir`（analysis），独立截断批次 probe 已实际生成该违规路径。— reasoning: 批准计划要求机器 JSON 只能落 `data/`；将 `--decision-out` 改为 required，或提供确定的 data/ 默认路径并对 analysis 路径 fail-fast，避免正确性依赖调用者记住可选 flag。
- [Blocking] [Concern] 变更尚未达到项目 lint/pre-commit 基线。— reasoning: 独立 `ruff check` 报 `analyze_stepweight.py:221` 的 E741（变量 `l`）和 `emit_traj_weight_alloc.py:40` 的 F401（未使用 `combinations`）。提交测试虽为 23 passed，但独立审查测试为 4 failed；修复后需同时重跑提交测试、独立 probes、ruff，并附结果。
- [Non-blocking] [Suggestion] 去掉 `test_derived_config_only_tw_differs` 对 tracked calibration 的 `skipif`。— reasoning: calibration 是 emitter 的 source-of-truth；缺失时 CI 应硬失败而不是跳过最关键的 base 重建测试。
- [Non-blocking] [Concern] 计划头仍写“§4 Code 进行中”，且 emitter module docstring 仍误称 d4 uniform 由 S1 提供（S1 没有 c=.25）。— reasoning: 前者与 `logs/README.md` 的“Code 完成/待 G2”不一致，后者是 G1 R3 已记录的文字问题；在下一轮前同步状态并清理即可。

### G2 Round 1 — Executor — 2026-07-02

- **G2-1（分析交付缺失：无 bootstrap CI / 无 d1 prior / 只写 top-k / 图无图例）— Accepted**。属实。`analyze_stepweight.py` 已加：① `paired_bootstrap_ci(base, cfg, n_boot, seed, alpha)` 固定 seed 可复现的 delta-SR CI（单测 `r1==r2` + `lo≤point≤hi` + 空交集）；② `read_d1_prior` 稳健读 d1 天花板（best regular-grid zscore `cp1_spatial_pool_16` mean SR = 0.74，写入 `decision.json.d1_prior_sr` + note）；③ **全部 171 config 的完整 paired 统计**（delta_sr + McNemar + bootstrap CI）落 `decision.json.by_depth[d].configs`，`results.md` 仅 top-k 摘要；④ 图加 shape 图例 + incumbent 虚线。全量 171-batch 端到端验证：52/60/59 全 config 落盘、d1_prior=0.74、bootstrap CI 出。
- **G2-2（完整性门可被截断批次绕过 + 不校 episode_idx）— Accepted**。属实（独立 probe 2 失败）。`acceptance_check` 重构为**对齐 emitter 独立 `expected_ids()`**（不再拿 yaml-dir 自身当 expected）：yaml-dir 集与 journal 集都须 == 锁定集；且每 config paired-key 集须**恰等** `{(t,e) for t in task_ids for e in range(trials)}`（拒 episode_idx 越界/缺失）。`main()` 传 `expected_ids(depths)`。新增 main 级单测 `test_main_rejects_truncated_batch`（2 yaml≠171→SystemExit）+ `acceptance` 单测覆盖截断/错 episode。
- **G2-3（严格内部单峰/U 未实现，plateau 误判）— Accepted**。属实（`[.1,.4,.4,.1]`→peak / `[.4,.1,.1,.4]`→trough）。`classify_shape` 的 peak/trough 改为要求**唯一内部极值**（`len(near_max within tol)==1`）**且严格越过两侧邻居**（`w[k]-w[k±1]>tol`）；plateau/多极值 → `other`。新增单测：两 plateau→other、真单峰 `[.1,.3,.5,.3,.1]`→peak。
- **G2-4（`--decision-out` 默认落 analysis 违规）— Accepted**。`--decision-out` 改为 **required**，且 `"analysis" in path.parts` → fail-fast（在读任何文件前）。§4.3 命令已显式给 `data/` 路径。新增单测 `test_main_rejects_analysis_decision_path`。
- **G2-5（lint 未过：E741 / F401）— Accepted**。`analyze_stepweight.py` 的 `l` 循环变量改 `x`（E741）；`emit_traj_weight_alloc.py` 删未用 `from itertools import combinations`（F401）。`ruff check` 三文件 **All checks passed**；提交测试 **29 passed**（原 23 + bootstrap/plateau/截断/episode/decision-out 6 新）。
- **G2-6（去掉 calibration 的 skipif）— Accepted**。`test_derived_config_only_tw_differs` 移除 `skipif`，改为 `assert CALIB.exists()` 硬失败（calibration 是 emitter source-of-truth，CI 缺失应红不应跳）。
- **G2-7（plan 头 + emitter docstring d4-uniform 文字）— Accepted**。plan 头改为“§4 Code 完成 / G2 R1 …”与 README 同步；§2.2 与 emitter docstring 均修正：**d5 uniform 由 S1（c=0.2）提供、d4 uniform 由 S2 提供（c=0.25 不在 c 网格）、d3 无精确 uniform**。

### G2 Round 2 — Reviewer — APPROVED — 2026-07-02 04:42 CDT

- [Non-blocking] [Suggestion] `ruff format --check` 当前会重排 `analyze_stepweight.py`、`emit_traj_weight_alloc.py` 和 `test_traj_weight_alloc.py`。— reasoning: 仓库 `.pre-commit-config.yaml` 仅配置 `uv-lock`，CI 未启用 ruff format，因此不构成现行合入门；若执行格式化，应在本轮快照/提交前完成，避免落地后再产生无审查差异。
- [Non-blocking] [Suggestion] `paired_bootstrap_ci` 可在未来增加 `n_boot>0`、`0<alpha<1` 参数校验。— reasoning: 当前主跑使用安全默认值 `n_boot=2000, alpha=.05` 且测试覆盖复现性，不影响本实验；校验只用于提高通用 CLI 的错误诊断。

### G2 Round 2 — Executor — 2026-07-02

- **ruff format 建议 — Acknowledged（不执行）**。保持与本轮审查一致的代码快照落地：repo `.pre-commit-config.yaml` 仅 `uv-lock`、CI 未启用 `ruff format`，非合入门；`ruff check` 已全过。按审查者"format 须提交前完成、否则不做"的二选一，选**不格式化**以免引入未经审查的差异。
- **`paired_bootstrap_ci` 参数校验 — Deferred**。本实验固定安全默认值（`n_boot=2000, alpha=.05`）且测试覆盖复现性，不影响正确性；作为通用 CLI 加固列为可选后续，不在本次范围。
