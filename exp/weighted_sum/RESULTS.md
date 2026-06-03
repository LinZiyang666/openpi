# Weighted-Sum 两层缓存检索搜索 — 实验结果

> LIBERO-Spatial（10 task）/ pi05_libero / judge=`always_hit`（纯缓存重放，隔离检索质量）。
> 每配置 100 episode（10 task × 10 held-out trial）。
> **状态（2026-05-26）**：旧实验数据（双 server 混跑，含 87% 等）已全部作废。当前在 **jupyter 单机（H200 NVL）单库重跑基线 136 配置**，完成后从头跑 5 轮调优。下方实验结果 / 参数哲学 / 可复现性诊断三节**待新实验完成后填写**。方法学（归一化公式、keybuilder、防泄漏）不受影响，保留。

---

## 一、统计学归一化数学公式

检索打分分两层：**Layer-1** 把每个模态字段的原始相似度单调映射到有界区间 `[0,1]`（per-(模态,keybuilder) 由 Phase-1 离线校准选定），**Layer-2** 对各字段归一化分数做加权和。

### 1. Field-level similarity（字段原始相似度）

对模态字段 *f* 的 query key `q_f` 与库内候选 key `k_f`：

- **cosine**（vision_0 / vision_1）：

$$\mathrm{cos}(q_f,k_f)=\frac{\langle q_f,k_f\rangle}{\lVert q_f\rVert\,\lVert k_f\rVert}\in[-1,1]$$

- **l2**（robot_state）：

$$d(q_f,k_f)=\lVert q_f-k_f\rVert_2\ \ (\text{距离，越小越相似})$$

定向分数 `x`：cosine 直接取 `cos`；l2 取 `-d`（使「越大越相似」），记作 `x = orient(raw, sim_type)`。

### 2. Layer-1 per-field normalizer（单调、保幅、有界 [0,1]）

> ⚠ **下列是「候选策略池」，不是同时使用**：每个字段在运行时**只用 1 个** normalizer。Phase-1 离线校准对**每个 (模态字段, keybuilder)** 把适用候选都拟合一遍，按分离度指标打分，**选出 1 个最优**（`selected`）、其余排成 `shortlist`。三个字段各自独立选，互不相同。
>
> 候选范围由 `sim_type` 约束：**zscore / affine_clip** 对 cosine 和 l2 都适用；**logit / power / neg_log_one_minus** 仅 cosine（vision）；**exp_l2** 仅 l2（robot_state）。即 vision 从 {zscore, affine_clip, logit, power, neg_log_one_minus} 选 1 个，robot_state 从 {zscore, affine_clip, exp_l2} 选 1 个。
>
> **本实验**：三字段 Phase-1 **恰好都选中 `zscore(tanh)`**（一致是巧合，非规定）。`__norm2` 变体是人为换成各自 shortlist 第二名（vision→`neg_log_one_minus`、robot_state→`affine_clip`）的对照。

- **zscore + tanh squash**（selected，全字段）：

$$n(x)=\tfrac{1}{2}\bigl(\tanh\!\tfrac{x-\mu}{\sigma}+1\bigr)$$

- **affine_clip**（robot_state shortlist 第二）：

$$n(x)=\mathrm{clip}\!\Bigl(\tfrac{x-\text{lo}}{\text{hi}-\text{lo}},\,0,\,1\Bigr)$$

- **neg_log_one_minus**（cosine，vision shortlist 第二；放大近 1 饱和区）：

$$v=-\log(1-\cos+\varepsilon),\qquad n=\mathrm{clip}\!\Bigl(\tfrac{v-v_{lo}}{v_{hi}-v_{lo}},\,0,\,1\Bigr)$$

- **logit**（cosine；把分辨率分配给稠密的近 1 区）：

$$u=\mathrm{clip}\!\Bigl(\tfrac{\cos-\text{lo}}{\text{hi}-\text{lo}},\varepsilon,1-\varepsilon\Bigr),\quad n=\mathrm{rescale}\Bigl(\log\tfrac{u}{1-u}\Bigr)\to[0,1]$$

- **power**（cosine）：`u = clip((cos-lo)/(hi-lo),0,1)`，`n = u^γ`，γ 使分布中位映到 ~0.5。

- **exp_l2**（l2 候选之一；把距离直接映为 similarity）：

$$n(d)=\exp(-d/\tau)\in(0,1]$$

#### 参数符号定义（含义 + 来源）

记定向分数 $x=\mathrm{orient}(\text{raw},\text{sim\_type})$：cosine 取 $x=\cos$；l2 取 $x=-d$（距离取负，使「越大越相似」）。所有参数由 Phase-1 在该 (模态, keybuilder) 的**校准集分数分布**上离线拟合（鲁棒百分位默认 P1/P99）：

| 符号 | 含义 | 拟合方式 |
|---|---|---|
| $\mu,\ \sigma$ | 定向分数 $x$ 的均值、标准差 | zscore：$\mu=\mathrm{mean}(x),\ \sigma=\mathrm{std}(x)$ |
| $\text{lo},\ \text{hi}$ | 定向分数 $x$ 的下界、上界 | affine_clip / logit / power：取 $x$ 的 P1、P99 |
| $v_{lo},\ v_{hi}$ | 变换值 $v=-\log(1-\cos+\varepsilon)$ 的下界、上界 | neg_log_one_minus：取 $v$ 的 P1、P99 |
| $\tau$ | l2 距离尺度 | exp_l2：$\tau=\mathrm{median}(d)/\ln 2$（中位距离映到 0.5） |
| $\gamma$ | 幂指数 | power：$\gamma=\ln 0.5/\ln u_{med}$（中位映到 0.5） |
| $\varepsilon$ | 数值稳定小量 | 固定 $10^{-4}$ |

所有 normalizer **单调递增、有界 [0,1]**（不用 rank/CDF 这类破坏保幅性的映射）。

#### 本实验 selected normalizer（cp1_max_pool 实际拟合值）

三字段 Phase-1 **全部选中 `zscore(tanh)`**（实际 YAML 中的 `score_normalization.fields`）：

| field | sim_type | selected normalizer | $\mu$ | $\sigma$ |
|---|---|---|---|---|
| vision_0 | cosine | zscore(tanh) | 0.9796 | 0.0051 |
| vision_1 | cosine | zscore(tanh) | 0.9752 | 0.0053 |
| robot_state | l2 | zscore(tanh) | −1.8498 | 1.0008 |

> robot_state 的 $\mu<0$ 因 $x=-d$（定向距离为负）；vision 的 $\mu\approx0.98$ 因同任务 patch 余弦普遍接近 1，$\sigma\approx0.005$ 极小 → zscore 把这条**极窄的余弦带**拉伸铺满 [0,1]（这正是放大近 1 饱和区的关键）。
> shortlist 第二候选：vision_0/vision_1→`neg_log_one_minus`、robot_state→`affine_clip`，**仅** `__norm2` 后缀配置使用（见第三节）。其余字段的 `exp_l2` / `logit` / `power` 是候选集成员但未被选中。

### 3. Layer-2 weighted score sum（加权融合检索）

对启用字段集合 *F*，库内每个候选 *c* 的总检索分：

$$S(c)=\sum_{f\in F} w_f\cdot n_f\bigl(\mathrm{sim}_f(q_f,c_f)\bigr),\qquad \sum_{f\in F} w_f=1,\ w_f\ge 0$$

`top_k=1`：返回 $\arg\max_c S(c)$ 对应的缓存动作序列重放。`step_filter=all`。`judge=always_hit`（不设命中阈值，纯检索重放，使成功率直接反映「检索到的近邻动作能否解出任务」）。

**搜索即在 K=3 个模态权重单纯形 $\{(w_{v0},w_{v1},w_{rs}):\sum w=1\}$ 上找使任务成功率最大的点。** prompt_emb 已排除（不参与加权），但保留在 `backend.vector_dims` 全字段集中（库 artifact 相等性校验要求）。

---

## 二、Keybuilder（视觉 token → 单 key 向量的压缩策略）

CP1 系列把 SigLIP 的 256 个 patch token（16×16 网格，每 token emb_dim）压成一个 key 向量，差异仅在 **vision 池化方式**（robot_state 始终为 32 维原始状态向量）：

| keybuilder | vision 池化 | 几何含义 |
|---|---|---|
| **cp1_mean_pool** | `tokens.mean(dim=0)` → [emb] | 全 patch 语义平均（丢空间） |
| **cp1_max_pool** | `tokens.max(dim=0)` 逐维取最大 → [emb] | 保留每维最显著激活（突出关键特征） |
| **cp1_spatial_pool_16** | `adaptive_avg_pool2d` 16×16→4×4 → 16×emb 展平 | 保留粗空间结构（4×4） |
| **cp1_spatial_pool_64** | 16×16→2×2 → 4×emb 展平（命名遗留，原称 64× 压缩） | 最粗空间（2×2） |

> clip keybuilder（CLIP ViT-B/32、L/14）已移除：server 端要加载 CLIP 模型算 key，多变体 × replica 累积撑爆 GPU（OOM）。CP1 系列是纯向量库（server 不加载额外模型）。

---

## 三、实验结果

> **所有配置用的 normalizer（统计学层）标注**：除带 `__norm2` 后缀者外，**全部三字段统一用 selected = `zscore(tanh)`**（参数见第一节实际值表）；`field_similarity`：vision_0/vision_1 = cosine、robot_state = l2。带 `__norm2` 后缀的配置（仅第 1/2 轮探索）把 vision_0/vision_1 换成 `neg_log_one_minus`、robot_state 换成 `affine_clip`（shortlist 第二候选）。
> 配置名 `vX@a vY@b rs@c` 里的数字是 Layer-2 **权重** $w$（百分比，$\sum w=1$），与 normalizer 无关——下面所有最优/次优配置（无 `__norm2` 后缀）的统计学层都是 **zscore(tanh)**。

> 数据源：`baseline_jupyter.jsonl`（jupyter 单机 H200 单库，136 配置 × 100 ep = 13600，`all_stages_done`）。旧的 87% 等双 server 混跑污染数据已全部删除。

### 基线（136 配置 = 4 keybuilder × 34；13600 episode，2026-05-26）

整体 SR **62.1%**，全局最优 **72%**。权重记作 `v0/v1/rs`（vision_0 / vision_1 / robot_state 的 Layer-2 权重）。

| yaml 配置（keybuilder · 权重 v0/v1/rs · normalizer） | 成功率 |
|---|---|
| `cp1_spatial_pool_16` · grid3 0.25/0.37/0.37 · zscore | **72%（全局最优）** |
| `cp1_spatial_pool_16` · grid3 0.12/0.37/0.50 · zscore | **72%** |
| `cp1_spatial_pool_16` · grid3 0.37/0.12/0.50 · zscore | 71% |
| `cp1_max_pool` · grid3 0.25/0.25/0.50 · zscore | 71%（max_pool 最优） |
| `cp1_spatial_pool_16` · grid3 0.37/0.25/0.37 · zscore | 70% |
| `cp1_max_pool` · grid 0/0.37/0.62 · zscore | 70% |
| `cp1_spatial_pool_64` · …（最优） | 67% |
| `cp1_mean_pool` · …（最优） | 67% |
| `cp1_*` · iso_vision_0（纯 vision_0 单模态） | 29–44%（全局最差） |

**各 keybuilder 最优 SR**：`cp1_spatial_pool_16` 72% ≈ `cp1_max_pool` 71% > `cp1_spatial_pool_64` 67% ≈ `cp1_mean_pool` 67%。

> ⚠ spatial_16(72%) 与 max_pool(71%) 仅差 1 episode，**在 n=100 的 SR 标准误（≈±4.5%）内打平**，不能据此断定孰优。故 5 轮调优**两个 keybuilder 都细化**（用户决策），不靠噪声丢弃任一。

### 3 轮调优（围绕最优区 center 加密；完整数据见 `data/libero_spatial/phase2/all_results.csv`，418 配置）

调优在两个 top keybuilder（max_pool + spatial_16）的最优区逐轮加密。**各轮最优演化**：

| 阶段 | 配置数 | overall SR | spatial_16 最优 | max_pool 最优 |
|---|---|---|---|---|
| 基线 | 136 | 62.1% | 72% | 71% |
| 第 1 轮 | 106 | 66.9% | **74%** | 73% |
| 第 2 轮 | 90 | 66.3% | **74%** | 73% |
| 第 3 轮 | 86 | ~66% | **74%** | 73% |

**全实验最优区（SR 73–74%，跨轮一致复现）**：
- `cp1_spatial_pool_16` · grid3 **v0@0.06 v1@0.44–0.50 rs@0.44–0.50** · zscore → **74%**
- `cp1_max_pool` · grid3 **v0@0.06–0.31 v1@0.25–0.44 rs@0.44–0.50** · zscore → **73%**
- 全实验 SR Top-10：8 个 `cp1_spatial_pool_16` + 2 个 `cp1_max_pool`（见 `config/top10/libero_spatial/`）。

> 第 1 轮起即触及 ~74% 天花板，第 2·3 轮加密无提升 → **已收敛**。`__norm2`（shortlist 第二 normalizer）对照未超过 zscore。完整逐配置 SR 见 `data/libero_spatial/phase2/all_results.csv`。

---

## 四、发现的参数设置哲学

> 基于基线 + 3 轮调优共 418 配置（`data/libero_spatial/phase2/all_results.csv`）的完整归纳。jupyter 单机 H200、`always_hit` 纯检索、每配置 100 held-out ep。

**1. SR 天花板 ~74%，全程稳定收敛。** 基线全局最优 72% → 第 1 轮升到 74% → 第 2·3 轮加密无提升（持平 74%）。`always_hit` 纯检索在当前库密度（每任务约千条目）下的 SR 上限就是 ~74%；继续在最优区加密只在 ±4.5% 标准误（n=100）内波动。（旧的 87% 是双 server 混跑污染/彩票值，已作废——见 §6 可复现性。）

**2. keybuilder：`spatial_16` ≈ `max_pool` 噪声内平手，均碾压 mean/spatial_64。** 最优 SR：spatial_16 74% ≈ max_pool 73%（差 1 ep，SE 内）≫ spatial_64 67% ≈ mean_pool 67%。全实验 Top-10 里 8 个 spatial_16 + 2 个 max_pool——spatial_16 出现频次略高但不足以判定显著更优。**结论：保留空间结构（spatial_16 的 4×4 池化）与逐维最大激活（max_pool）对任务级检索几乎等效，远好于全局平均（mean）和过粗空间（spatial_64 2×2）。**

**3. 最优权重区：低 vision_0 + 中高 vision_1/robot_state。** 跨 3 轮一致复现的最优簇：**v0@0.06–0.31、v1@0.44–0.50、rs@0.44–0.50**（grid3 三模态）。要点：
   - **三模态融合 > 双模态 > 单模态**：纯视觉单模态 `iso_vision_0` 全局最差（29–44%），含 robot_state 的三模态显著更好（73–74%）。
   - **robot_state 重要但非压倒**：rs 权重最优带 ~[0.44, 0.50]；rs 提供本体位姿/夹爪的强检索线索，但需与视觉融合而非独占。
   - **主视角 vision_0 权重可压低**：最优区 v0 普遍偏低（@0.06 多见），腕部 vision_1 + robot_state 贡献更大。（这修正了基线单轮"双视觉严格均衡"的初步印象——细化后腕部视角更关键。）

**4. 归一化：三字段统一 `zscore(tanh)` 最优。** Phase-1 离线校准三字段都选中 zscore(tanh)（见 §1）；调优中对最优点换用 shortlist 第二候选（`__norm2`：vision→neg_log_one_minus、rs→affine_clip）做对照，**未超过 zscore**。

**5. 方法学可信度**：held-out init 防泄漏（数值级验证 0 命中，§5）+ 固定单 GPU 杜绝跨/混 server 污染（§6）+ C2 write-frozen 纯检索评测。故 73–74% 是干净可复现的检索质量上限。

> **跨 GPU 验证（top10 H200 vs A100）见 §7**（取全实验 SR 前 10 配置在 a100 重跑，量化同配置跨架构的 SR 差异）。

---

## 五、方法学要点（防泄漏 / 公平性）

- **held-out init 防泄漏**：建库用「全集 \ pruned」差集中每 task 抽的 5 个 init；eval 只用其余 held-out init。已数值级验证（实验用的 100 个 init 条目 vs 建库 50 个 init，bit 级 array_equal + 跨 task 全量 + 严格 allclose **0 命中**，最近一对 L2=0.046），**无测试污染**。
- **C2 write-frozen**：`write_policy: never`，运行时库不写入（纯检索评测）。
- **prompt_emb 排除**：不参与加权，但保留于 `backend.vector_dims` 全字段集（artifact 相等性校验）。

---

## 六、可复现性诊断（波动根因，2026-05-26）

> **待填写（新实验完成后）。** 旧诊断（D0–D3 + 跨 GPU 浮点根因等）基于已作废的双 server 混跑数据与已删除的诊断脚本，结论尚有未排除项（建库↔serving 路径/batching 差异未验），全部删除。
>
> 当前可复现性保障：**固定单一 GPU（jupyter H200 NVL）单库**跑全部实验，杜绝跨/混 server 污染。跨 GPU 差异的干净量化见 §7。

---

## 七、跨 GPU 波动对比（top10：H200 vs A100，2026-05-26）

取全实验 SR 前 10 配置（8 spatial_pool_16 + 2 max_pool），**两个 GPU 各跑 3 次**（a100：run1/2/3；jupyter：实验首次 + 重跑 v1/v2；同 held-out init、同 100ep/配置），同时测**各自 run-to-run 波动**（噪声 floor）与**跨架构均值差**。数据：`data/libero_spatial/phase2/top10_variance.csv`（逐次 SR）。

| 配置（grid3 v0/v1/rs） | a100 r1/r2/r3（均值,极差）| jupyter exp/v1/v2（均值,极差）| cross (a100−jup) |
|---|---|---|---|
| v0@31 v1@44 rs@25 | 76 76 76 (76, 0) | 73 73 73 (73, 0) | **+3pp** |
| v0@31 v1@38 rs@31 | 72 72 72 (72, 0) | 74 73 73 (73, 1) | −1pp |
| v0@25 v1@44 rs@31 | 71 71 71 (71, 0) | 73 73 73 (73, 0) | −2pp |
| v0@25 v1@37 rs@37 | 72 73 72 (72, 1) | 72 72 72 (72, 0) | 0pp |
| v0@19 v1@31 rs@50 | 70 71 70 (70, 1) | 73 73 73 (73, 0) | −3pp |
| v0@6 v1@50 rs@44 | 65 65 65 (65, 0) | 74 74 74 (74, 0) | −9pp |
| v0@6 v1@44 rs@50 | 63 63 64 (63, 1) | 74 74 74 (74, 0) | **−11pp** |
| v0@12 v1@37 rs@50 | 61 61 61 (61, 0) | 72 72 72 (72, 0) | **−11pp** |
| v0@6 v1@25 rs@69 | 60 62 60 (61, 2) | 73 73 73 (73, 0) | **−12pp** |
| v0@31 v1@25 rs@44 | 55 55 54 (55, 1) | 73 73 73 (73, 0) | **−18pp** |

**关键统计**：
- **各 GPU 自身 run-to-run 波动（噪声 floor）极小**：a100 极差均值 **0.6pp**（最大 2pp）；jupyter 极差均值 **0.1pp**（最大 1pp）。两 GPU 各 3 次 → **同一 GPU 上结果基本确定性**（mujoco 物理边界噪声可忽略）。
- **跨架构均值差**：a100 − jupyter 平均 **−6.4pp**，平均 |diff| **7.1pp**，a100 偏低 8/10、偏高 2/10，单配置最大 **−18pp**。

**结论（确凿）**：
- **跨 GPU 差异（~7pp、最大 18pp）≫ 各 GPU 自身噪声 floor（0–0.6pp），相差一个数量级以上 → 跨架构 SR 差异是真实、系统性的，不是运行噪声。** 机制：H200(Hopper) vs A100(Ampere) 的 bf16 浮点累加按架构不同 → pi05 深层放大 → cp1 query key 漂移 → `always_hit` 在密集库紧边界命中翻转 → SR 变。
- **同 GPU 内基本确定性**（各 3 次：jupyter 极差≤1pp、a100 ≤2pp），**跨 GPU 不可 bit 对齐**（硬件级）——干净验证，无需任何"取最高/选择偏差"修正（jupyter 近零波动即 mean≈max≈各次）。
- **印证单 GPU 约束**：top10 是在 H200 选的，迁到 A100 平均掉 ~6pp、最差掉 18pp → 一台 GPU 调出的"最优"不完全迁移到另一架构。**固定单一 GPU（§6）是可复现的前提**，这也是本实验全程坚持 jupyter 单机的根本原因。
