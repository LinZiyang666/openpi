# Weighted-Sum 之上的 Trajectory Search 实验

> **Status**: `Plan`（G1 APPROVED 2026-05-27 / G2 owner-waived per WA §7 / 待 §4 Code → §6 Verify）
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-05-27
> **关联**: 在第三个实验 [`weighted_sum_two_layer_refactor.log.md`](weighted_sum_two_layer_refactor.log.md) 的最优配置之上叠加 trajectory search；方法学对照老实验 [`archive/trajectory_*`](archive/) + `exp/common/analysis/trajectory/libero_spatial`。

---

## 1. 背景与目标

weighted_sum 实验（两层 `weighted_score_sum_knn` 检索：Layer-1 per-field `zscore(tanh)`
归一化 + Layer-2 加权和，`judge=always_hit` 纯回放）在 libero_spatial 上跑出 SR 天花板
~74%（spatial_16 ≈ max_pool ≫ mean_pool ≈ spatial_64），最优权重区
`v0@0.06–0.31 / v1@0.44–0.50 / rs@0.44–0.50`，全程锁单一 jupyter H200（杜绝跨 GPU 污染）。

本实验复刻老 trajectory 实验（`exp/common/analysis/trajectory/libero_spatial`）对 Phase 1
的做法，但**基底换成 weighted_sum 的两层检索**：在选定的 base 配置上叠加多步 query
历史聚合（`trajectory_depth > 1`），测纯检索 SR 能否被 trajectory 进一步抬高。核心问题
与老实验对称：**trajectory 对弱 base 的「救弱」效果 vs 对强 base 的「几乎不动/倒退」**。

**关键复用**：trajectory 机制在 src 中已完整实现且经测试（见 §4），本实验**不改任何 src/
代码、不改评测基础设施**，只新增 exp/ 生成器 + 分析脚本。

---

## 2. 实验设计

### 2.1 两组 base 配置（同时跑、互不影响、去重）

数据源：`exp/weighted_sum/data/phase2/all_results.csv`（baseline + 3 轮共 418 配置；
同 yaml 多次出现按 SR 均值聚合）。

**第一组（per-keybuilder，复刻老实验选法）**：4 个 keybuilder 各取「**top-1 + top-2 +
倒数第二**」。倒数第二按 **A 口径**（用户拍板）：仅正规权重网格、同 `zscore` 归一化，
**排除 `__norm2`（换归一化层）与 `iso_`（单模态）**，避免引入混淆变量。

**第二组**：全实验 SR 前 10（`exp/weighted_sum/config/top10/`，8 spatial_16 + 2 max_pool，全 zscore）。

**12 个第一组逻辑条目（标注是否与 top10 物理重叠）**：

| keybuilder | 角色 | SR | 与 top10 | yaml_id |
|---|---|---|---|---|
| spatial_16 | top1 | 74% | **DUP** | `cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50` |
| spatial_16 | top2 | 74% | **DUP** | `cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@44` |
| spatial_16 | 2nd-worst | 60% | NEW | `cp1_spatial_pool_16__grid_vision_0@25_robot_state@75` |
| max_pool | top1 | 73% | **DUP** | `cp1_max_pool__grid3_vision_0@31_vision_1@25_robot_state@44` |
| max_pool | top2 | 73% | **DUP** | `cp1_max_pool__grid3_vision_0@6_vision_1@25_robot_state@69` |
| max_pool | 2nd-worst | 56% | NEW | `cp1_max_pool__grid_vision_0@25_vision_1@75` |
| mean_pool | top1 | 67% | NEW | `cp1_mean_pool__grid_vision_0@50_robot_state@50` |
| mean_pool | top2 | 66% | NEW | `cp1_mean_pool__grid3_vision_0@12_vision_1@12_robot_state@75` |
| mean_pool | 2nd-worst | 51% | NEW | `cp1_mean_pool__grid_vision_0@87_robot_state@12` |
| spatial_64 | top1 | 67% | NEW | `cp1_spatial_pool_64__grid3_vision_0@12_vision_1@12_robot_state@75` |
| spatial_64 | top2 | 67% | NEW | `cp1_spatial_pool_64__grid3_vision_0@12_vision_1@50_robot_state@37` |
| spatial_64 | 2nd-worst | 54% | NEW | `cp1_spatial_pool_64__grid_vision_0@62_vision_1@37` |

**去重结果**：spatial_16/max_pool 的 top1+top2（共 4 个）已在 top10 内 → 第一组独有 8 个。
**合并去重 base 总数 = 10 (top10) + 8 (第一组独有) = 18 个**。物理上每个 (base, depth) 只
生成一份 yaml、只跑一次；4 个双重身份配置的结果在分析时被两组各自引用，**不重复跑 episode**。

### 2.2 Trajectory 维度

- **depth ∈ {3, 4, 5, 6}**（用户拍板：照搬老实验 4 个 depth 做完整复现）。
- **`trajectory_weights`**：复用老实验固定递减方案（newest-first，和=1）：

  | depth | trajectory_weights |
  |---|---|
  | 3 | `[0.5, 0.3, 0.2]` |
  | 4 | `[0.4, 0.3, 0.2, 0.1]` |
  | 5 | `[0.35, 0.25, 0.2, 0.12, 0.08]` |
  | 6 | `[0.3, 0.25, 0.2, 0.12, 0.08, 0.05]` |

- **depth-1 基线**：复用 weighted_sum 已有 SR（18 base 配置的 depth-1 SR 已在同一 jupyter
  H200 上测过，同机可比），不重跑。

### 2.3 规模与不变量

- 配置数：**18 base × 4 depth = 72 个 trajectory yaml**。
- 每配置 **100 episode**（10 task × 10 held-out trial，`--eval-trials 10`，与 weighted_sum 基线一致）。
- 总量：**72 × 100 = 7200 episode**。
- 检索不变量（全部继承自 weighted_sum，逐字不变）：
  `search_strategy.type=weighted_score_sum_knn`、`top_k=1`、`step_filter=all`、
  Layer-1 `score_normalization.type=per_field` 三字段 `zscore`、
  `field_similarity`：vision_0/1=cosine、robot_state=l2(exp→sim)、
  `keys.prompt_emb.enabled=false`、`judge.type=always_hit`、
  `backend.type=in_memory`（trajectory 只支持 InMemoryBackend）、
  `write_policy.type=never`（C2 write-frozen）。
- **防泄漏硬门**：`run_phase2` 经 `init_holdout` 从 `libero_spatial_init_map.json` 读已用
  `orig_init_state_idx`，只从剩余 held-out init 取 episode；init_map 缺失即 fail-fast。

---

## 3. 文件改动

### 3.1 新增（exp/，L1 范畴）

| 路径 | 作用 |
|---|---|
| `exp/weighted_sum/emit_trajectory_yamls.py` | **新生成器**：硬编码 18 个 (keybuilder, weights) 选择 + depth→weights 表，`import build_eval_config`（来自 `emit_yamls.py`，**不改原文件**）逐 (base × depth) 生成 72 份 yaml 到 `config/trajectory/`。yaml_id = `<base_id>__d{depth}`。生成后对每份调 `validate_cache_config` 自检（trajectory_depth/weights 合法性 + load 兼容）。 |
| `exp/weighted_sum/analysis/plot_trajectory_results.py` | **新分析**：合并 depth-1 基线（从 `all_results.csv` 取 18 base SR）+ trajectory depths（从本实验 journal 聚合），绘 SR × depth（按 keybuilder / 按组）、算 Δ vs depth-1，复刻老 `trajectory_analysis` 风格（per-keybuilder facet + 救弱/强基线分层）。 |
| `exp/weighted_sum/config/trajectory/` | 72 份 trajectory yaml（新目录）。`data/**` 与生成 yaml 的入库策略按 `artifact_layout.md` §3。 |
| `exp/weighted_sum/data/trajectory/` | journal.jsonl + results.json + 分析中间产物（本地保留，按 artifact_layout 归档）。 |

### 3.2 复用（零改动）

| 复用项 | 说明 |
|---|---|
| `src/openpi/**`（trajectory 机制 + 推理 + cache） | 零改动，见 §4 |
| `exp/weighted_sum/emit_yamls.py::build_eval_config` | 已含 `trajectory_depth`/`trajectory_weights` 形参（行 41-94），直接调用 |
| `exp/weighted_sum/run_phase2.py` | 通用：glob yaml-dir 跑 conductor，trajectory 对其透明，零改动 |
| `exp/weighted_sum/init_holdout.py` / `weight_search_strategy.py` | 防泄漏 + conductor 策略，零改动 |
| `exp/weighted_sum/summarize.py` | journal → per-yaml SR，按 yaml_id 聚合，零改动 |
| `src/openpi/conductor/**` + `replica_proxy` + 并发 server | 编排/路由对 trajectory 透明，零改动 |

### 3.3 文档

- 在 `docs/experiments/weighted_sum.md` 末尾增一节「Trajectory 扩展」（生成命令 + 跑法 + 分析），
  并同步 `docs/README.md` 该行描述（WA §4 索引同步红线，同 commit）。

---

## 4. 接口与机制核实（已亲验）

- `weighted_score_sum_knn` + `trajectory_depth>1` 在 `in_memory_backend.py:289-317`（dispatch）
  + `:893-938`（legacy DAG 路径，含 `_MultiBranchSentinel` 多分支回退）已实现，注释明确
  「reuses per-step fusion (RRF / **score_sum**)」且为 score_sum 路径单独 `build_field_normalizers`。
- config 校验 `config.py:1576-1607`：`trajectory_depth>1` 必须配等长 `trajectory_weights`、非负、和>0；
  Qdrant + trajectory 直接 fail-fast（本实验用 in_memory，满足）。
- `serve_policy.py:97/105/573-606`：`--replicas N` + `--replica-spawn-batch B` 已实现；
  `--replicas>1` 强制 concurrent 模式。`replica_proxy` 对 conductor 透明（infer sticky /
  bundle·preload·unload broadcast / fetch_dump aggregate，见 conductor_tutorial §1.3）。

---

## 5. 基础设施 / 配置方案

**拓扑：2 台设备（a100 不用）。单一 jupyter server = 与 weighted_sum 基线同机，depth 对比无跨 GPU 污染。**
**（replica 数：临时为 2，用户 2026-05-27 指示；原方案 3。）**

### 5.1 Server — jupyter-ziyang10（H200 NVL，cgroup 10C/32G）

```bash
# 经 tether 在 jupyter 上起 server（tmux srv0）。⚠ 必须 export HOME=/home/ziyang10
# （tether exec 默认 HOME=/home/ziyang10/.tether-agent，否则找不到 checkpoint/tokenizer）
tether exec jupyter-ziyang10 -- bash -lc '
  export HOME=/home/ziyang10
  tmux has -t srv0 2>/dev/null || tmux new -s srv0 -d "
    cd /home/ziyang10/openpi && export HOME=/home/ziyang10 &&
    /home/ziyang10/.local/bin/uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi05_libero \
      --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
      --replicas 2 --replica-spawn-batch 2 \
      --port 8000 \
      --cache_config exp/weighted_sum/config/trajectory/<任一>.yaml
  "'
```

- `--replicas 2 --replica-spawn-batch 2`（**临时：用户 2026-05-27 指示 replica 数暂改为 2**，
  原方案 3）：2 副本单公共端口，2 个一批一起启动（「2 台一组开机」）。2 副本对 32GB
  host-RAM 压力更低，OOM 风险小。
- `--cache_config` 仅作启动占位；conductor 经 ctrl 平面按 yaml 逐个热切 bundle（broadcast 到全 replica）。

**暴露到公网**（jupyter 在 NAT 后；broker=pc732/weiland.top）：

```bash
tether expose jupyter-ziyang10 --local 8000 --name traj-srv
# → exposed: http://weiland.top:14xxx → jupyter-ziyang10:8000
```

### 5.2 Client — timan107（8×GTX1080 EGL / 48 logical CPU / 220 GiB）

```bash
tether exec timan107 -- bash -lc '
  cd /scratch/zixuans8/openpi &&
  /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir   exp/weighted_sum/config/trajectory \
    --init-map   exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal    exp/weighted_sum/data/trajectory/journal.jsonl \
    --servers    weiland.top:14xxx \
    --task-ids 0-9 --eval-trials 10 \
    --workers 48 --gpus 8 \
    --conda-env /scratch/zixuans8/libero_sim \
    --eval-concurrency 2'
```

> `--eval-trials 10` × 10 task = **100 episode/yaml**（`weight_search_strategy._episodes`：
> `for task_id in 10 tasks: for ep in range(eval_trials)`），与 weighted_sum 基线逐字一致。

**Worker 数**：`--workers 48 --gpus 8`（run_phase2 默认，也是 weighted_sum 跑通用的配置）。
`run_phase2` 起 48 个**独立 worker 进程**（非线程，绕开单进程 EGL cap 15），round-robin 到
8 个 GTX1080 EGL slot（~6 env/GPU，渲染轻量），48 logical CPU 各 ~1 worker。`always_hit`
缓存重放 + server 端 BatchingCoordinator **跨连接 batch**，靠大量并发 worker 填 batch 窗口，
故 worker 数不按 replica 数配而取满 client 容量。conductor 中央队列 worker-pull 自动消除
yaml 间等待泡沫。`--eval-concurrency 2`：同 keybuilder yaml 经 BackendPool 共享 backend，
2 路填 yaml 尾部 straggler 空泡且基本不增显存。若 jupyter 10C/32G server 端打爆，
用 `exp/serving_benchmark/autotune_workers.py` 重定标。

### 5.3 聚合 + 分析（client 本地，无 GPU）

```bash
uv run exp/weighted_sum/summarize.py \
  --journal exp/weighted_sum/data/trajectory/journal.jsonl \
  --out     exp/weighted_sum/data/trajectory/results.json
uv run exp/weighted_sum/analysis/plot_trajectory_results.py \
  --results  exp/weighted_sum/data/trajectory/results.json \
  --baseline exp/weighted_sum/data/phase2/all_results.csv
```

### 5.4 Wall-clock 粗估

7200 episode、单 jupyter 2-replica（临时）。weighted_sum 基线 13600 ep 同机跑通；本实验 ~0.53×
episode 量。具体取决于 always_hit 下每 episode 步数与 CP1 key 构建延迟，预计数小时级；
journal 断点续跑，可分批。（精确值在首批 ~200 ep 后据实测吞吐回填。）

---

## 6. 执行步骤

1. **生成 yaml**（client 或本地）：`emit_trajectory_yamls.py` → 72 份到 `config/trajectory/`，
   生成即跑 `validate_cache_config` 自检（全绿才算通过）。
2. **起 server**（jupyter，§5.1）+ `tether expose` 拿公网端口；`tether ps` 确认 PORTS 节。
3. **跑评测**（timan107，§5.2）：conductor 7200 ep，journal 落 `data/trajectory/`。
4. **聚合 + 绘图**（§5.3）。
5. **写分析 md**（`exp/weighted_sum/` 下 trajectory 结果，对照老实验：救弱 Δ / 强基线稳定性 / 最优 depth）。
6. **文档同步**：weighted_sum.md 加 trajectory 节 + docs/README.md 同步。

---

## 7. 测试策略

- **生成器去重断言**（含可复现性自检）：`emit_trajectory_yamls.py` 在生成时显式断言并打印：
  base 总数 = 18、trajectory yaml = 72、与 `config/top10/` 的物理重叠 = 4（spatial_16/max_pool
  各 top1+top2）。18 个 base 的 (keybuilder, weights, 角色) 选择从 `data/phase2/all_results.csv`
  按 §2.1 的 A 口径**程序化重算**（而非手抄清单），与 `config/top10/` 现有 yaml 交叉核对；
  top10 与 per-keybuilder top-k 都可能并列 SR，断言能挡住后续清单漂移。任一断言不过即 fail-fast。
- **生成器 schema 自检**：72 份 yaml 全部过 `validate_cache_config`（重点 trajectory_depth/weights
  合法 + per_field zscore + write_policy=never + in_memory）；任一不过即 fail-fast。
- **§6 Verify**：`uv run pytest` 全绿（本实验零改 src/，应无回归；trajectory 机制已有 src 测试覆盖）。
- **首批 smoke**：正式 7200 前先跑单个 base × depth=5 的 100 ep，确认 server 接受 bundle、
  `__hit_meta__` 正常、防泄漏门生效（0 命中建库 init），再放全量。
- **depth-1 可比性实测记录**（R1 关键假设）：smoke 阶段额外抽 **1–2 个 base 在本实验同一
  jupyter server 上重测 depth-1 的 100 ep**，与 `all_results.csv` 旧 SR 比对，记录差值与 SE
  （n=100 → SE≈±4.5%）。差值落在 ±SE 内即确认「复用旧 depth-1 基线」可比；否则升级为全量
  depth-1 重跑（R1 兜底）。该记录写入最终分析 md。

---

## 8. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **跨机/跨时漂移**：depth-1 基线复用旧 SR，若 jupyter 容器重启换了物理 H200 或路径/batching 变化，base 可比性受损 | §7 实测 jupyter run-to-run ≤1pp（RESULTS §7），同机复用安全；首批 smoke 顺带抽 1-2 个 base 重测 depth-1 与旧 SR 比对（±SE 内即放心） |
| R2 | **query-history 跨 replica 串话**：trajectory 需每 episode query_history 一致；多 replica 下若一 episode 的步分散到不同 replica，历史错乱 | infer 连接级 sticky（replica_proxy least-conn sticky），1 worker = 1 WS = 1 episode 固定 1 replica；episode 内不重连。掉线由 conductor episode 级重试（整 episode 重跑） |
| R3 | **多分支 trajectory 回退**：artifact 若有 `prev_ids>1` 触发 `_MultiBranchSentinel` legacy DAG | libero_spatial artifact 是单链 trajectory_id；smoke 阶段查日志无 "Multi-branch" 警告即确认单链主路径 |
| R4 | **jupyter 32G host-RAM**：2 replica（临时）× 4 keybuilder bundle（spatial_16 pkl 430MB），BackendPool 仅副本内共享 | 2 副本已显著降 RAM 压力；`--eval-concurrency 2` 限同时 active yaml；监控 `free -h`，逼近上限则降并发。若恢复 3 副本则 `--replica-spawn-batch 2` 错峰加载 |
| R5 | **init-state 泄漏** | `run_phase2` held-out init 硬门 + init_map fail-fast（已就位）；smoke 验证 0 命中建库 init |
| R6 | **HOME 环境** | server 命令显式 `export HOME=/home/ziyang10`（devices.md §2.2 gotcha） |

---

## 9. 交付物

- 72 trajectory yaml（`config/trajectory/`）+ journal/results（`data/trajectory/`）。
- `emit_trajectory_yamls.py` + `plot_trajectory_results.py`。
- trajectory 结果分析 md + 图（`exp/weighted_sum/`）。
- weighted_sum.md trajectory 节 + docs/README.md 同步。
- 结论：trajectory 对 weighted_sum 各 base（强/弱、4 keybuilder、top10）的 SR Δ × depth，
  与老实验「救弱不救强、峰值 4-5」结论的异同。
