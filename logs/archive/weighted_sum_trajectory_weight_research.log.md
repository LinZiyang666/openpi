# spatial_16 在 Trajectory regime 下重搜权重（过拟合 vs 本质拖累的分离实验）

> **Status**: `Plan`（G1 owner-APPROVED 2026-05-27 via chat / §4 Code 进行中 / 运行待 owner 下令）
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-05-27
> **关联**: 续 [`weighted_sum_trajectory_search.log.md`](weighted_sum_trajectory_search.log.md)。第一阶段发现「trajectory 救弱不救强、top10 全军 −6.4pp」。owner 指出 top10 在 jupyter 已 3 次重测 ≤1pp（近确定性）→ 排除选择偏差/回归，degradation 是真实效应。本实验分离两个真因。

---

## 1. 动机与假设

第一阶段结论：trajectory 让已调优的强配置（top10）一致变差（−3~−12pp，10/10 全负）。两个候选真因：

- **H-3a 权重对 d1 regime 过拟合**：top10 的权重是在**单步检索**目标上 3 轮搜出来的最优，从未在 trajectory regime 下重调。换检索机制 = 目标错配，旧最优权重不再最优。
- **H-机制**（偏差-方差 + 时间特异性）：trajectory 是时间平滑算子；强配置单步已低方差高精度，无方差可降、只增旧步偏差 → 本质性拖累，与权重无关。

**判据**：在 **depth 3/4/5 各自重新搜权重**，看每个 depth 的**最优重调权重**能否把 SR 追回到接近/达到 d1 天花板（~74%）：

- 若**能追回甚至超过 d1** → H-3a 主导（degradation 主要是权重错配，trajectory 本身不损强，重调即可）。
- 若**重调后仍显著低于 d1** → H-机制 主导（trajectory 对强索引是本质拖累，调权重救不回）。
- 同时观察**最优权重向量是否随 depth 漂移**（d1-best vs d3/d4/d5-best 的 v0/v1/rs 是否系统性不同）——漂移本身就是「权重对 regime 敏感」的直接证据。

**只用 `cp1_spatial_pool_16`**（top10 主力 8/10，最具代表性；省一个数量级算力）。

---

## 2. 设计

### 2.1 权重网格 × 深度

- keybuilder：**仅 cp1_spatial_pool_16**；模态 = {vision_0, vision_1, robot_state}（prompt_emb/vision_2 关，同 weighted_sum）。
- **权重网格**：grid3 在 (v0,v1,rs) 单纯形上加密（比 weighted_sum 基线 step=0.125 更细）。拟用 **step=0.0625**（权重为 1/16 的倍数，三模态各 ≥0.0625、和=1），并以 rs≥0.1875 聚焦有用区。**精确配置数在生成时断言+打印**（目标 ~70）。
- **深度**：**{1, 3, 4, 5}**。**含 d1**——这是关键：新权重配置不在旧 418 里，必须各自跑 d1 才有**无偏的同配置基线**来判定权重是否随 regime 漂移（不能复用旧 d1，旧 d1 只有旧网格点）。depth 6 按第一阶段结论（回落）省略。
- 规模：**~70 weight × 4 depth = ~280 yaml**（与 owner 估计一致）× 100 ep = **~28000 episode**。
- `trajectory_weights`：复用第一阶段递减方案（d3 `[.5,.3,.2]` / d4 `[.4,.3,.2,.1]` / d5 `[.35,.25,.2,.12,.08]`）。

### 2.2 不变量（全继承）

`weighted_score_sum_knn` + `top_k=1` + `step_filter=all`；Layer-1 per-field **zscore**（复用 `calibration_normalizers.json`，零额外校准）；`field_similarity` v0/v1=cosine、rs=l2(exp)；`judge=always_hit`；`backend=in_memory`；`write_policy=never`；held-out init 防泄漏。单一 jupyter H200（同机可比，无跨 GPU 污染）。

---

## 3. 代码改动（极少）

| 路径 | 改动 |
|---|---|
| `exp/weighted_sum/emit_trajectory_weight_sweep.py` | **新生成器（~60 行）**：复用 `emit_yamls.build_eval_config` + `grid3_weight_configs`（已存在），对 spatial_16 笛卡尔 (weight grid) × (depths {1,3,4,5}) 生成 ~280 yaml；d1 不写 trajectory 字段。生成时断言+打印配置数 + 逐 yaml `load_cache_config` 校验。preload_path 相对路径。 |
| `exp/weighted_sum/analysis/plot_weight_sweep_trajectory.py` | **新分析（~80 行）**：每 depth 的权重→SR 热力/散点 + 各 depth 最优权重对比 + 最优 SR vs d1 天花板曲线 + 最优权重向量随 depth 漂移图。 |
| 复用零改动 | `run_phase2.py` / conductor / 并发 server / `summarize.py` / `build_eval_config` / `grid3_weight_configs` |

**不改任何 src/**。生成器与 emit_trajectory_yamls 同构（已验证可行）。

新目录：`config/trajectory_wsweep/`（~280 yaml，gitignore 覆盖）、`data/trajectory_wsweep/`（journal+results）。

---

## 4. 基础设施 / 运行

同第一阶段拓扑：server=jupyter `serve_policy --replicas 2`（mem lock off）+ expose；client=timan107 `run_phase2 --yaml-dir config/trajectory_wsweep --workers 48`。
**预算**：~28000 ep / @~2.1 ep/s ≈ **3.5-4 h wall-clock**。journal 断点续跑 + cron/Monitor 守护同第一阶段。

---

## 5. 分析与结论

1. 每 depth：`argmax_weight SR`，对比 d1 天花板（74%）+ 该权重在 d1 的 SR。
2. **最优权重向量随 depth 的漂移**（v0/v1/rs 轨迹）。
3. 判定 H-3a vs H-机制（见 §1）。
4. 副产物：trajectory regime 下 spatial_16 的最优权重区（若 H-3a 成立，这是 trajectory 部署的新推荐权重）。

---

## 6. 风险

| # | 风险 | 缓解 |
|---|---|---|
| W1 | 28000 ep ~4h，server 长跑中断 | journal 断点续跑 + Monitor/cron 守护 + server 自愈重启（第一阶段已验证流程）|
| W2 | grid 配置数偏离 280 | 生成时打印精确数；step/rs-floor 是旋钮，owner 审核时可调密度 |
| W3 | 含 d1 重测与旧 d1 网格点不重合 | 故意：新网格点本就需各自 d1 基线；旧 74% 仅作天花板参照，不直接配对 |
| W4 | 单 keybuilder 结论是否外推到 max_pool 等 | 明确限定 spatial_16（最具代表性）；如需可后续补 max_pool |

---

## Review Log
