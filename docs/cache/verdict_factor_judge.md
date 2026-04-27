# Verdict Factor Judge 使用指南

> **前置知识**：阅读 [tutorial.md](tutorial.md) §6 了解 Judge 组件基础，§10 了解 YAML 配置；阅读 [../architecture/cache_system.md](../architecture/cache_system.md) §5.6 / §5.11 / §5.12 了解 verdict factor 系统的架构契约。
>
> **设计文档**：完整方案设计与决策见 [`logs/verdict_factor_judge.log.md`](../../logs/verdict_factor_judge.log.md)（Plan，G1 / G2 APPROVED）。
>
> **状态说明（2026-04-26）**：B1 + B2 全部 land。B1 = F1a-A / F1a-T / F2 `extract` body、3 Composer 算法、`PercentileRollingNormalizer`、Orchestrator view+history 注入、`_state_history` / anchor checkpoint policy / `on_task_end` leak fix、`_JUDGE_TYPES` 解禁 `composite`、validator 7 项 composite-specific 校验（含 5a-5d warm_start：CP1-only / canonical timestep / pairwise / tier 排序）、`min_top_k_hint` 反向喂回 strategy。B2 = F1b OnlineExtractor + OfflineWriter、`LibraryStats.compute_from_entries`、Orchestrator `__init__` 收 `offline_writers` / `library_stats` 且 `_build_entry_chain` 落地 merge loop、`build_per_connection_components` 收集 writer / library_stats 并通过 `scripts/serve_policy.py` 两条 production assembly 路径喂给 Orchestrator、Backend `load_artifact` library_stats 加载 / fallback、三 build_*.py `--factors-yaml` CLI、`exp/common/factor_postprocess.py` 写入 helper。端到端 composite judge 与完整离线 build 链路均可跑。

---

## 1. 概述

`CompositeJudge` 把"单阈值 cosine 命中"替换为**"因子向量 → Normalizer → Composer → JudgeResult"** 三段式判定，让多个统计 / 运动学描述子（jerk、direction consistency、curvature radius、cumulative displacement、top-K action variance）共同参与命中决策。

```
SearchResultLite[]  ─┐
PayloadView (view)   ├──► OnlineExtractor*  ──► raw: dict[str, float]
HistoryView          │   （key 必须 = 该 extractor 自己声明的 descriptor_orientations.keys()）
cached_data         ─┘
                                      │
                                      ▼
                              Normalizer (可选，例 PercentileRollingNormalizer)
                                      │
                              norm: dict[str, float]
                              （all-NaN → CompositeJudge 短路 MISS）
                                      │
                                      ▼
                                  Composer (WeightedSum / AndGate / OrGate)
                                      │
                                      ▼
                                JudgeResult(hit_type, winner_id, start_t)
```

**与 ThresholdJudge 选用建议**：

| 情景 | 推荐 |
|---|---|
| 单一 cosine 分布形状已经良好，需要快速上线 | `ThresholdJudge` |
| 想要把 retrieval 的"动作不连续"风险纳入判定 | `composite` + F1a-A |
| 需要 candidate 池一致性约束 | `composite` + F2 |
| 跨 episode 的运动平滑度作为 hit 质量参考 | `composite` + F1b-A / F1b-T（需 build pkl 时算 LibraryStats） |
| 多维度组合（典型 B2 后默认） | `composite` + F1b-A + F1b-T + F2，weighted_sum |

> **批次状态**：B0 = CompositeJudge 类骨架 + 5 因子注册 + describe + capability flags。B1 = F1a / F2 在线 extract、3 Composer 与 PercentileRollingNormalizer 算法、Orchestrator view+history 注入、`composite` YAML 解禁、validator 7 项 composite 校验。B2 = F1b 在线 extract + 离线 OfflineWriter、`LibraryStats.compute_from_entries`、Backend `load_artifact` library_stats 加载/fallback、artifact 写入 helper。

---

## 2. 五个因子总览

| 注册名 | 类 | 数据源 | requires_library_stats | requires_chain_walk | 算法批次 |
|---|---|---|---|---|---|
| `f1a_a` | `RuntimeContinuityAction` | winner `payload.action_chunk[0]` + `history.actions` | True | False | B1 |
| `f1a_t` | `RuntimeContinuityState`  | winner `query_keys["robot_state"]` + `view.walk_next(winner_id, k)` | True | True | B1 |
| `f1b_a` | `SourceWindowSmoothnessAction` | OfflineWriter 读链上 `entries[i].payload.action_chunk[0]` 序列；OnlineExtractor 仅读 `payload.factors` | True | False | B2 |
| `f1b_t` | `SourceWindowSmoothnessState`  | OfflineWriter 读链上 `entries[i].query_keys["robot_state"]` 序列；OnlineExtractor 仅读 `payload.factors` | True | False | B2 |
| `f2`    | `TopKActionConsensus` | top-K 个 `payload.action_chunk` via `view.get_many` | False | False | B1 |

**两类 capability flags 与 source / key_initial 解耦**：

- `source: "action" | "state"` —— 因子消费的数据语义。
- `key_initial: "a" | "t"` —— 写入 `payload.factors` 的 key 命名空间，与 registry 名后缀一致（避免 YAML config 引用 `f1a_t_jerk` 但 extractor 实际产 `f1a_s_jerk` 这种失配）。
- `requires_library_stats=True` 的因子必须搭配 `backend.type=in_memory`（当前唯一暴露 `library_stats` 的 backend）。
- `requires_chain_walk=True`（仅 F1a-T）必须搭配实现 `fetch_entry` capability 的 backend（当前唯一是 InMemoryBackend）。
- 上述 capability vs backend 约束由 `validate_cache_config` 在 config-load 阶段强制（B1+ 6 项静态校验中的 #3 / #4）。

**描述子集（F1a + F1b 共享）**：

| 描述子 | 取向 | 公式 | 含义 |
|---|---|---|---|
| `jerk` | risky | `median \|Δ²a / σ\|`（active-DOF 上） | 加速度变化幅度，高 → 不平滑 |
| `dir`  | safe  | `mean cos(v[t], v[t+1])` | 方向一致性，高 → 平滑 |
| `curv_radius` | non_monotonic | `mean ‖p[t] − centroid‖` | 窗口几何弥散度（中等 = 圆弧，极小 = 停滞，极大 = 大幅直线） |
| `cum_disp`    | non_monotonic | `sum ‖p[t+1] − p[t]‖` | 累积路径长度（小 + 低 jerk = 静止；大 = 快速移动） |

`risky` / `safe` Composer 自动按取向 flip score；`non_monotonic` 必须在 `composer.directions` 显式指定 `"high"` / `"low"` / `"range:[lo,hi]"`，否则 validator 拒收（避免无任务先验 force-fit 进单调聚合）。

> **批次状态**：5 因子的 metadata（capability flags + describe + register）= B0 已 land。`extract` / `compute_for_episode` body：F1a / F2 = B1，F1b = B2。

---

## 3. 因子的精确数学表达式

> 本章定义所有因子的运算契约（B1 / B2 实现 body 时按此公式落地）；上一章只给了直觉描述。GitHub 渲染数学公式：行内 `$...$`、单行 `$$...$$`。本地编辑器若不渲染，源代码中的 LaTeX 仍可读。

### 3.1 通用记号

| 符号 | 含义 | 形状 / 范围 |
|---|---|---|
| $a_t \in \mathbb{R}^A$ | 第 $t$ 步的 action 向量（已 input-transform / pre-output-transform） | Pi0.5: $A = 32$（universal action space） |
| $s_t \in \mathbb{R}^S$ | 第 $t$ 步的 robot_state（post-input-transform） | Pi0.5: $S = 32$（universal state space） |
| $\sigma \in \mathbb{R}^D$ | per-DOF std（$D = A$ 或 $S$） | 来自 `LibraryStats.{action,state}_sigma`（B2 算） |
| $M \in \{0,1\}^D$ | active mask（True iff $\sigma_d \geq \varepsilon_{\text{active}}$） | `LibraryStats.{action,state}_active_mask`，default $\varepsilon_{\text{active}} = 0.01$ |
| $D_{\text{act}} = \sum_{d=1}^{D} M_d$ | active 维数 | LIBERO 实测：action 7 / state 8（$D=32$ 中余下 24 维为 universal-32 零填充） |
| $\varepsilon$ | clamp_min epsilon（避免除零） | $10^{-8}$（实现层硬编码） |

**Per-DOF z-score**（动作侧示例，state 同理用 `state_sigma`）：

$$
\tilde{a}_{t,d} \;=\; \frac{a_{t,d}}{\max(\sigma_d,\;\varepsilon)}
\qquad d = 1, 2, \ldots, D
$$

**Active 子空间限制**：所有后续算子在 $D_{\text{act}}$ 维子空间内做。下面公式中的 $\tilde{a}_t$ 默认已经是 $\tilde{a}_t[M] \in \mathbb{R}^{D_{\text{act}}}$。

**速度与近似 jerk**（z-score 后空间内）：

$$
v_t \;=\; \tilde{a}_{t+1} - \tilde{a}_t \;\in\; \mathbb{R}^{D_{\text{act}}}
\qquad\text{(一阶差分)}
$$

$$
j_t \;=\; \tilde{a}_{t+1} - 2\,\tilde{a}_t + \tilde{a}_{t-1} \;\in\; \mathbb{R}^{D_{\text{act}}}
\qquad\text{(二阶差分，近似 jerk)}
$$

注：物理 jerk 是三阶导数；这里 $j_t$ 是离散二阶差分（在 z-score 空间内），命名沿用 plan / 文献俗称。

**窗口表示**：$W = (p, f)$ 表示窗口跨过 $[k - p,\; k + f]$（共 $|W| = p + f + 1$ 个采样点），其中 $k$ 是 anchor 时刻。F1a 用 `window_k = K`（自动转成 $(K, K)$ 或类似）；F1b 用 YAML 显式给出 `windows: [(p_1, f_1), ...]`。

---

### 3.2 通用描述子的精确数学

设窗口 $W$ 内的 active-子空间序列为

$$
\mathrm{pts} \;=\; \bigl[\tilde{a}_{k-p},\; \tilde{a}_{k-p+1},\; \ldots,\; \tilde{a}_{k+f}\bigr] \;\in\; \mathbb{R}^{|W| \times D_{\text{act}}}
$$

对应的速度与 jerk：$v_W \in \mathbb{R}^{(|W|-1) \times D_{\text{act}}}$、$j_W \in \mathbb{R}^{(|W|-2) \times D_{\text{act}}}$。

#### 3.2.1 `jerk` — 加速度幅值（risky）

$$
\boxed{\;
\mathrm{jerk}(W) \;=\; \frac{1}{D_{\text{act}}} \sum_{d=1}^{D_{\text{act}}} \;\operatorname*{median}_{0 \leq t \leq |W|-3} \bigl|\,j_W[t, d]\,\bigr|
\;}
$$

伪代码（与实际实现一致）：

```python
jerk_factor = j_W.abs().median(dim=0).values.mean()
# j_W: [|W|-2, D_act]
# .median(dim=0).values : [D_act]   per-DOF time-median
# .mean()              : scalar     average over DOF
```

**为什么 median over time, mean over DOF**：median 抑制 gripper 单帧脉冲（双稳态切换瞬间）；mean over DOF 让所有 active 维等权贡献。high jerk → retrieval 拼接处出现了 inference 通常不会产生的速度跳变 → risky。

#### 3.2.2 `dir` — 方向一致性（safe）

把每个时刻的 active 速度向量整体 normalize 后求相邻时刻余弦：

$$
\boxed{\;
\mathrm{dir}(W) \;=\; \frac{1}{|W| - 2} \sum_{i=0}^{|W|-3} \frac{\langle v_W[i],\; v_W[i+1]\rangle}{\lVert v_W[i] \rVert_2 \,\cdot\, \lVert v_W[i+1] \rVert_2}
\;}
$$

伪代码：

```python
v1, v2 = v_W[:-1], v_W[1:]                       # each: [|W|-2, D_act]
cos    = F.cosine_similarity(v1, v2, dim=-1)     # [|W|-2]
dir_factor = cos.mean()
```

**为什么不是 per-DOF cos**：`dir` 测的是 active 子空间内运动方向的连贯性 —— 一个 $D_{\text{act}}$ 维向量整体的转向；per-DOF 余弦会被零分量主导。**值域**：$[-1, 1]$，1 = 严格同向，$-1$ = 反向。

#### 3.2.3 `curv_radius` — 窗口几何弥散度（non-monotonic）

$$
\bar{c}_W \;=\; \frac{1}{|W|} \sum_{t=0}^{|W|-1} \mathrm{pts}[t] \;\in\; \mathbb{R}^{D_{\text{act}}}
\qquad\text{(窗口质心)}
$$

$$
\boxed{\;
\mathrm{curv\_radius}(W) \;=\; \frac{1}{|W|} \sum_{t=0}^{|W|-1} \bigl\lVert \mathrm{pts}[t] - \bar{c}_W \bigr\rVert_2
\;}
$$

伪代码：

```python
centroid = pts.mean(dim=0, keepdim=True)         # [1, D_act]
curv_rad = (pts - centroid).norm(dim=-1).mean()
```

**几何含义**：在 z-score 后的 $D_{\text{act}}$ 维空间里，窗口内点云对中心的平均欧氏距离。停滞 $\to$ 几乎零；圆弧 $\to$ 中等；大幅直线 $\to$ 大。**取向 non-monotonic**：YAML 必须显式 `direction: range:[lo, hi]` 才能进 weighted_sum。

#### 3.2.4 `cum_disp` — 累积路径长度（non-monotonic）

$$
\boxed{\;
\mathrm{cum\_disp}(W) \;=\; \sum_{t=0}^{|W|-2} \bigl\lVert \mathrm{pts}[t+1] - \mathrm{pts}[t] \bigr\rVert_2
\;}
$$

伪代码：

```python
cum_disp = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
```

**几何含义**：active 子空间里相邻点之间步长之和。停滞 $\to$ 接近 0；快速移动 $\to$ 大。与 `curv_radius` 组合可分辨"快速直线 / 缓慢直线 / 停滞 / 圆弧 / 微抖"五种 regime。

---

### 3.3 F1a-A / F1a-T — 拼接窗口（Splice Window）数学

F1a 测的是"如果不用 cache，inference 会给我一个与现在多接近的 action / state？"—— 把**已执行历史**与**候选未来**拼接成一个序列，然后跑 §3.2 的描述子。

#### 3.3.1 F1a-A 序列拼接

记 $K = \mathtt{window\_k}$（YAML 字段），`history.actions` 末尾 $K$ 个 action 已在执行（pre-output-transform domain），winner candidate 的 `payload.action_chunk[0]` 是 cache 给出的下一步建议 action：

$$
\mathrm{splice}_a \;=\; \bigl[\,a_{-K},\; a_{-K+1},\; \ldots,\; a_{-1},\; a_0^{\text{cand}}\,\bigr] \;\in\; \mathbb{R}^{(K+1) \times A}
$$

随后：

1. Per-DOF z-score（用 `library_stats.action_sigma`）：$\mathrm{splice}_a \to \widetilde{\mathrm{splice}}_a$
2. Active 子空间限制（用 `library_stats.action_active_mask`）：$\widetilde{\mathrm{splice}}_a \in \mathbb{R}^{(K+1) \times A_{\text{act}}}$
3. 设 $\mathrm{pts} = \widetilde{\mathrm{splice}}_a$，$v_W = \mathrm{pts}[1:] - \mathrm{pts}[:-1]$，$j_W = \mathrm{pts}[2:] - 2\,\mathrm{pts}[1:-1] + \mathrm{pts}[:-2]$
4. 套 §3.2.{1..4} 计算 4 个描述子，写入 `payload.factors`：

$$
\begin{aligned}
\texttt{f1a\_a\_jerk}        &= \mathrm{jerk}(\mathrm{splice}_a) \\
\texttt{f1a\_a\_dir}         &= \mathrm{dir}(\mathrm{splice}_a) \\
\texttt{f1a\_a\_curv\_radius}&= \mathrm{curv\_radius}(\mathrm{splice}_a) \\
\texttt{f1a\_a\_cum\_disp}   &= \mathrm{cum\_disp}(\mathrm{splice}_a)
\end{aligned}
$$

**边界**：history 长度 $< K$ 时（episode 早期）→ 全部描述子返回 $\mathrm{nan}$；CompositeJudge 的 cold-start sentinel 会让早期 verdict 走 MISS（与 inference 拼接最自然）。

#### 3.3.2 F1a-T 序列拼接

state 侧需要"未来一段"才能算窗口；cache 没有"未来 state"，但可以从下游 entry 链取：

$$
\mathrm{splice}_s \;=\; \bigl[\,\underbrace{s_{-K},\;\ldots,\;s_{-1}}_{\text{history.states 末 } K \text{ 项}},\; \underbrace{s_0^{\text{cand}}}_{\text{winner robot\_state}},\; \underbrace{s_1^{\text{cand}},\;\ldots,\;s_K^{\text{cand}}}_{\texttt{view.walk\_next}(K)}\,\bigr] \;\in\; \mathbb{R}^{(2K+1) \times S}
$$

后续 z-score / active-mask / 描述子流程同 F1a-A，只是用 `state_sigma` / `state_active_mask`，写入 key `f1a_t_*`。

**chain walk 失败时（trajectory boundary / fork）**：`view.walk_next` 返回 list 长度 $< K$，splice 序列变短；F1a-T 实现按"长度 $< (K+1)$ → 描述子置 $\mathrm{nan}$"处理（与 history 不足同规则）。

---

### 3.4 F1b-A / F1b-T — 链上窗口（Chain Window）数学

F1b 在 **offline build pkl** 阶段对每个 episode 的整条 entry 链跑窗口扫描，把每个 entry 在每个窗口的描述子值写入它自己的 `payload.factors`。online verdict 只读不算。

#### 3.4.1 序列与 z-score

设 episode 有 $T$ 个 entries $[e_0, e_1, \ldots, e_{T-1}]$（按 `prev_ids/next_ids` 链单调）。源序列：

- F1b-A: $x_t = \mathtt{entries}[t].\mathtt{payload.action\_chunk}[0] \in \mathbb{R}^A$（pre-output-transform 的 normalized 第一步 action）
- F1b-T: $x_t = \mathtt{entries}[t].\mathtt{query\_keys["robot\_state"]} \in \mathbb{R}^S$（post-input-transform 的 normalized state）

Per-DOF z-score 与 active-mask（用对应的 sigma / mask）：

$$
\tilde{x}_t \;=\; \frac{x_t}{\max(\sigma,\;\varepsilon)}
\qquad
\tilde{x}_t^{\text{act}} \;=\; \tilde{x}_t[M] \;\in\; \mathbb{R}^{D_{\text{act}}}
$$

#### 3.4.2 窗口扫描

对 YAML `windows = [(p_1, f_1), (p_2, f_2), ...]` 中的每个 $(p, f)$，对每个 entry 索引 $k = 0, 1, \ldots, T-1$：

$$
W_{k,(p,f)} \;=\; \bigl[\tilde{x}_{k-p}^{\text{act}},\; \tilde{x}_{k-p+1}^{\text{act}},\; \ldots,\; \tilde{x}_{k+f}^{\text{act}}\bigr] \;\in\; \mathbb{R}^{(p+f+1) \times D_{\text{act}}}
$$

记 $\mathrm{pts} = W_{k,(p,f)}$，套 §3.2 公式：

$$
\begin{aligned}
\mathtt{factors}\bigl[\texttt{f1b\_<key\_initial>\_jerk\_\_p}p\texttt{\_f}f\bigr]        &\leftarrow \mathrm{jerk}(W_{k,(p,f)}) \\
\mathtt{factors}\bigl[\texttt{f1b\_<key\_initial>\_dir\_\_p}p\texttt{\_f}f\bigr]         &\leftarrow \mathrm{dir}(W_{k,(p,f)}) \\
\mathtt{factors}\bigl[\texttt{f1b\_<key\_initial>\_curv\_radius\_\_p}p\texttt{\_f}f\bigr] &\leftarrow \mathrm{curv\_radius}(W_{k,(p,f)}) \\
\mathtt{factors}\bigl[\texttt{f1b\_<key\_initial>\_cum\_disp\_\_p}p\texttt{\_f}f\bigr]    &\leftarrow \mathrm{cum\_disp}(W_{k,(p,f)})
\end{aligned}
$$

`<key_initial>` = `"a"`（F1b-A）或 `"t"`（F1b-T）。

**边界**：$k < p$ 或 $k + f \geq T$ → 该窗口所有描述子写入 $\mathrm{nan}$（不抛错；CompositeJudge 收到 NaN 后按 §11 FAQ 的 NaN 规则处理）。

**单 entry 的 factors 维度**：4 描述子 × $N_{\text{windows}} \approx 12\text{-}24$ 个 float（per side），双侧 $24\text{-}48$ float $\approx 200$ 字节，相对 `action_chunk [10, 32]` 的 $1.3$ KB 可忽略。

#### 3.4.3 在线侧仅读

F1b 的 OnlineExtractor 实现：

```python
def extract(self, results, view, history, cached_data):
    winner = view.get(results[0].id)
    out = {}
    for key in self.descriptor_orientations:                # 类内 describe() 算出
        if winner.factors is None or key not in winner.factors:
            out[key] = float("nan")                         # 老 entry / 缺字段 → NaN
        else:
            out[key] = winner.factors[key]
    return out
```

零计算开销；所有数值在 build pkl 时已固化。

---

### 3.5 F2 — Top-K 候选共识方差

F2 测的是 cache 的 top-K 候选在"下一步该做什么"上是否达成一致；分歧大 → retrieval 在猜 → risky。

#### 3.5.1 取数

设 $\mathrm{ids} = \{r.id : r \in \mathtt{results}[:K]\}$，

$$
X \;=\; \begin{bmatrix} \mathtt{view.get(id_1).action\_chunk[0]} \\ \vdots \\ \mathtt{view.get(id_K).action\_chunk[0]} \end{bmatrix} \;\in\; \mathbb{R}^{K \times A}
$$

**Active mask**：F2 用 **candidate-local** active mask（不是 `library_stats.action_active_mask`），保持 `requires_library_stats=False` 自洽：

$$
M^{\text{local}} \;=\; \bigl\{ d : \mathrm{var}_d > \varepsilon \bigr\}
\qquad (\varepsilon = 10^{-8})
$$

$$
X_{\text{act}} \;=\; X[:,\, M^{\text{local}}] \;\in\; \mathbb{R}^{K \times |M^{\text{local}}|}
$$

理由：pi05 padded DOF 在所有候选上常数 0 → per-DOF 候选方差天然为 0 → candidate-local mask 在 pi05 deployment 下与 library mask 行为等价，但**不依赖** `LibraryStats`，validator 不需要要求 `backend.type=in_memory`。

$K$ 是 YAML `params.K`（默认 5）；F2 通过 `min_top_k_hint` 自动让 SearchStrategy 多取 $K$ 个候选。

#### 3.5.2 方差

per-DOF 方差（$K$ 维上的 population 方差），再聚合为标量：

$$
m_d \;=\; \frac{1}{K} \sum_{k=1}^{K} X_{\text{act}}[k, d]
\qquad
\mathrm{var}_d \;=\; \frac{1}{K} \sum_{k=1}^{K} \bigl(X_{\text{act}}[k, d] - m_d\bigr)^{2}
$$

$$
\boxed{\;
\mathtt{f2\_var} \;=\; \frac{1}{A_{\text{act}}} \sum_{d=1}^{A_{\text{act}}} \mathrm{var}_d
\;}
$$

伪代码（B1 实装）：

```python
chunk0 = stack([view.get(r.id).action_chunk[0] for r in results[:K_eff]])  # [K_eff, A]
var_d  = chunk0.var(dim=0, unbiased=False)                                 # [A] population
active = var_d > 1e-8                                                       # candidate-local
if not active.any():
    return {"f2_var": float("nan")}        # 全候选完美一致 — NaN 而非 0
f2_var = float(var_d[active].mean())
```

**为什么不归一化到 sigma**：F2 在 candidate 池内做相对方差，scale-invariant —— 同一 episode chunk 的几个候选会自然落在 sigma 的同一量级，归一化反而抹掉信息。这就是 `requires_library_stats=False` 的根因。

**写入**：`{"f2_var": float(f2_var)}`，orientation `"risky"`。

**边界**：
- $\mathtt{len(results)} < K$ 时（cache 几乎空 / cold-start）→ $K_{\text{eff}} = \min(K, \mathtt{len(results)})$
- $K_{\text{eff}} < 2$（单候选 / 空池）→ 返回 NaN（无 consensus 信号）
- 所有 DOF 候选完美一致（$M^{\text{local}}$ 为空）→ 返回 NaN（避免与 "高方差 → MISS" 翻转后误判为 hard hit）

---

> **批次状态**：F1a-A / F1a-T / F2 / Composer / Normalizer 算法 body 已在 B1 落地；F1b OnlineExtractor + OfflineWriter 算法 body 在 B2 落地（`docs/architecture/cache_system.md` §5.12）。`describe()` 与 capability flags 在 B0 已 ship。

---

## 4. 全周期流程

```
       ┌─────────────────────────────────────────────────────────────┐
       │  B2 必须，B0/B1 跳过：build pkl + 写 LibraryStats + 写       │
       │  payload.factors                                             │
[Step1]│  exp/common/build_in_memory_cache_artifact.py + factor_     │
       │  postprocess.py                                              │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  写 YAML（type: composite + factors / composer / normalizer）│
[Step2]│  cache_composite_judge.yaml                                  │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  validate_cache_config → 6 项 composite 静态校验             │
[Step3]│  build_per_connection_components → CompositeJudge wiring     │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  scripts/serve_policy.py --cache_config xxx.yaml            │
[Step4]│  在线推理：每个 verdict 走 view+history → extract → norm    │
       │  → compose → JudgeResult                                    │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  跑 experiment（exp/cp1_cache + analysis 脚本），对比         │
[Step5]│  threshold judge 与 composite judge 的 success_rate / hit    │
       │  rate / mean start_t                                         │
       └─────────────────────────────────────────────────────────────┘
```

> **批次状态**：完整 5 步走通 = B2 land 后才有意义。B0 阶段可走 Step 2-3 验证 fail-fast 错误（YAML 在 config-load 被拒收，不会进入 Step 4）。

---

## 5. Step 1: 构建带 factors 的离线 Artifact（B2）

### 5.1 命令

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_mean_pool \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_with_factors.pkl \
    --factors-yaml exp/common/configs/factors/f1b_default.yaml
```

`--factors-yaml` 指向描述哪些 OfflineWriter 应跑 + 用什么参数的 YAML（与在线 YAML 中 composite judge 的 `factors` 字段同结构；只关心实现 `OfflineWriter` Protocol 的因子，即 F1b-A / F1b-T）。`build_clip_cache_artifact.py` / `build_llm_layer_matrix.py` 同样支持该 flag。例：

```yaml
# exp/common/configs/factors/f1b_default.yaml
factors:
  - type: f1b_a
    params:
      windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
  - type: f1b_t
    params:
      windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
```

### 5.2 流程内部步骤

1. `build_in_memory_cache_artifact.py` 用 KeyBuilder 跑完所有 entry 写入 `entries: list[CacheEntry]`
2. `enrich_artifact_with_factors(entries, offline_writers)` helper（位于 `exp/common/factor_postprocess.py`）：
   - `LibraryStats.compute_from_entries(entries, active_eps_action, active_eps_state)`
   - 按 `entry.trajectory_id` 切分 episode，对每个 OfflineWriter 调 `writer.compute_for_episode(per_episode_entries, library_stats)`
   - 把返回的 `list[dict[str, float]]` 合入 `entries[i].payload.factors`
3. 写 artifact pkl：

```python
{
    "key_builder_type": "cp1_mean_pool",
    "checkpoint_id": "CP1",
    "vector_dims": {...},
    "entries": [...],            # entry.payload.factors 已填好
    "library_stats": LibraryStats(...),  # 新增字段
}
```

### 5.3 老 artifact 兼容

未带 `library_stats` 字段的旧 artifact，`InMemoryBackend.load_artifact` 会自动 fallback：

```python
self.library_stats = data.get("library_stats")
if self.library_stats is None:
    logger.info("Artifact missing library_stats; computing from %d entries", len(self._entries))
    self.library_stats = LibraryStats.compute_from_entries(list(self._entries.values()))
```

启动时一次性 compute，后续推理热路径不重复算。`payload.factors is None` 的旧 entry 会让 F1b 的 OnlineExtractor 返回 NaN，Composer 按取向规则跳过。

> **批次状态**：build pkl 工具与 helper、LibraryStats compute 算法、SourceWindowSmoothness compute_for_episode = B2 全套；B0 阶段调 `compute_from_entries` / `compute_for_episode` 都会 raise NotImplementedError。

---

## 6. Step 2: 写 YAML

```yaml
# exp/cp1_cache/configs/composite_judge_demo.yaml
enabled: true

key_builder:
  type: cp1_mean_pool

keys:
  vision_0: { enabled: true, weight: 1.0 }
  vision_1: { enabled: true, weight: 1.0 }
  robot_state: { enabled: true, weight: 0.5 }

backend:
  type: in_memory                      # composite + F1a/F1b 必须 in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_with_factors.pkl

checkpoints:
  cp1:
    enabled: true
    judge:
      type: composite                  # B0 阶段 validator 拒收（fail-fast at load）
      factors:
        - type: f1a_a
          params:
            window_k: 5
            descriptors: [jerk, dir, curv_radius, cum_disp]
        - type: f1b_a
          params:
            windows: [{past: 0, future: 5}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]
            active_eps: 0.01
        - type: f1b_t
          params:
            windows: [{past: 0, future: 5}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]
            active_eps: 0.01
        - type: f2
          params:
            K: 5
      composer:
        type: weighted_sum
        weights:
          f1a_a_jerk:                      0.10
          f1a_a_dir:                       0.10
          f1b_a_jerk__p0_f5:               0.10
          f1b_a_dir__p0_f5:                0.10
          f1b_t_jerk__p0_f5:               0.10
          f1b_t_dir__p0_f5:                0.10
          f1b_a_curv_radius__p5_f5:        0.10
          f1b_a_cum_disp__p5_f5:           0.10
          f1b_t_curv_radius__p5_f5:        0.05
          f1b_t_cum_disp__p5_f5:           0.05
          f2_var:                          0.10
        tier_thresholds:
          full_hit:   0.80
          warm_start: 0.60
        warm_start_t: 0.5            # 必须 ∈ CANONICAL_DENOISE_TIMESTEPS
        directions:
          # 所有 non_monotonic key 在 weight ≠ 0 时必填，否则 validator 拒收
          f1b_a_curv_radius__p5_f5:  range:[0.3, 1.0]
          f1b_a_cum_disp__p5_f5:     high
          f1b_t_curv_radius__p5_f5:  range:[0.3, 1.0]
          f1b_t_cum_disp__p5_f5:     high
      normalizer:
        type: percentile_rolling
        window_size: 200
        cold_start_strategy: force_miss   # 默认；前 200 verdict 强制 MISS（all-NaN sentinel）
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1                           # F2 通过 min_top_k_hint 自动撑到 5（不破坏 top_k 语义）
  cp3:
    enabled: false
```

**字段语义提示**：

- `composer.weights` 的 key 必须是 extractor `describe()` 实际会产生的 key —— 拼错会被 Composer 当 weight=0 的不存在 key 处理（B1+ 加更严格的覆盖率校验）。
- `composer.tier_thresholds.full_hit` 与 `warm_start` 都是 0..1 范围的 normalized score；`warm_start_threshold` 必须 < `full_hit_threshold`，否则 warm_start tier 不可达（validator 5d）。
- `warm_start_t` 必须出现在 `CANONICAL_DENOISE_TIMESTEPS = {0.1, 0.2, ..., 0.9}`（与 `always_warm_start.start_t` 同规则）；CP3 不支持 warm_start。
- `directions` 仅对 `non_monotonic` 因子有效，三种形式：
  - `"high"` —— 数值越高越偏 hit
  - `"low"`  —— 数值越低越偏 hit
  - `"range:[lo,hi]"` —— 落在 [lo, hi] 区间内偏 hit

### 6.1 6 项 composite-specific 静态校验（B1+ 激活）

`validate_cache_config` 在 config-load 阶段会跑下面 6 类校验，全部通过才能进入 builder：

1. `factors` 与 `composer` 必须存在（type=composite 时）
2. 每个 `FactorConfig.type` 必须在 `factors.registry.known()` 中
3. `requires_library_stats=True` 的因子要求 `backend.type == "in_memory"`
4. `requires_chain_walk=True` 的因子（F1a-T）要求 `backend.type == "in_memory"`（exposes `fetch_entry`）
5. composite warm-start 完整校验（4 子规则）：
   - 5a `composer.warm_start_t` 仅 CP1 允许
   - 5b `warm_start_t` 必须 ∈ `CANONICAL_DENOISE_TIMESTEPS`（写回归一化值）
   - 5c pairwise rule：`tier_thresholds.warm_start` 与 `composer.warm_start_t` 必须同时存在或同时缺失（and / or composer 不支持 warm_start，置 → raise）
   - 5d tier ordering：`tier_thresholds.warm_start < tier_thresholds.full_hit`（仅 weighted_sum）
6. `directions` 覆盖率：`cls.describe(params)` 算出的所有 `non_monotonic` key，若 `composer.weights[key] != 0` 则 `composer.directions[key]` 必填且形式合法

> **批次状态**：YAML schema parse + B0 fail-fast（`_JUDGE_TYPES` 不含 `composite` → 在 config-load 被拒收）= B0 已 land；6 项静态校验 + algorithm body = B1。当前 B0 跑此 YAML 会报 `judge.type='composite' is not yet enabled in B0; available in B1+ when CompositeJudge algorithms land.`。

---

## 7. Step 3: 启动推理

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config exp/cp1_cache/configs/composite_judge_demo.yaml
```

启动序列（B1+ 流程）：

```
load_cache_config(yaml)
  ├─ _dict_to_dataclass → CacheConfig (含 JudgeConfig.factors: list[FactorConfig])
  └─ validate_cache_config → 通过 6 项 composite-specific 校验

build_cache_components(config)
  └─ build_per_connection_components(config, storage)
      ├─ _build_backend → InMemoryBackend (.library_stats 从 artifact 加载)
      ├─ per-CP loop:
      │    ├─ library_stats = per_conn_storage.library_stats   # facade duck-types backend
      │    ├─ judges[cp_id] = _build_judge(judge_cfg, library_stats)
      │    │     └─ if type == "composite":
      │    │          ├─ extractors = [
      │    │          │     cls(**dict(f.params),
      │    │          │         library_stats=library_stats if cls.requires_library_stats else ...)
      │    │          │     for f in cfg.factors
      │    │          │  ]
      │    │          ├─ composer  = _build_composer(cfg.composer)
      │    │          ├─ normalizer = _build_normalizer(cfg.normalizer)
      │    │          └─ CompositeJudge(extractors, composer, normalizer)
      │    │              # constructor 自动 collect 全部 descriptor_orientations + 调
      │    │              # composer.bind_orientations + normalizer.bind_keys
      │    ├─ min_hint = judges[cp_id].min_required_top_k
      │    └─ strategies[cp_id] = _build_search_strategy(ss_cfg, ..., min_top_k_hint=min_hint)
      ├─ offline_writers = collect_offline_writers_from_judges(judges)
      └─ Orchestrator(..., offline_writers=ow, library_stats=library_stats)
```

**单次 verdict 路径**（B1+，每次 `Orchestrator.check()` 跑一遍）：

```python
view    = StoragePayloadView(self._storage)              # per-check 生命周期，内部 memo
history = HistoryView(actions=list(self._action_history),
                      states =list(self._state_history))

judge_result = judge(
    results, checkpoint_id, self._key_builder.cached_data,
    view=view, history=history,
)
# CompositeJudge 内部：
#   raw = {}
#   for ext in extractors:
#       out = ext.extract(results, view, history, cached_data)
#       assert out.keys() == ext.descriptor_orientations.keys()  # key contract
#       raw.update(out)
#   norm = normalizer(raw)
#   if norm and all(isnan(v) for v in norm.values()):
#       return JudgeResult(MISS)        # cold-start sentinel
#   return composer.compose(norm, winner_id=results[0].id)

if hit_type in (FULL_HIT, WARM_START):
    payload = view.get(winner_id)        # memo 与 extractor 共享
```

> **批次状态**：Orchestrator view+history 注入 + winner fetch rewire + state_history anchor checkpoint policy = B1。B0 阶段 Orchestrator `check()` 不注入，老 Judge 通过 `**kwargs` 吃掉新参数，行为字节级不变。

---

## 8. Step 4: 跑实验对比 ThresholdJudge vs CompositeJudge

```bash
# Phase 1: 用 ThresholdJudge baseline 跑
uv run python exp/cp1_cache/run_cache_experiments.py \
    --config exp/cp1_cache/configs/threshold_baseline.yaml \
    --env LIBERO --num-episodes 50

# Phase 2: 用 CompositeJudge 跑同一 artifact
uv run python exp/cp1_cache/run_cache_experiments.py \
    --config exp/cp1_cache/configs/composite_judge_demo.yaml \
    --env LIBERO --num-episodes 50

# Phase 3: 分析
uv run python exp/cp1_cache/analysis/compare_judges.py \
    --baseline-dir exp/cp1_cache/runs/threshold_baseline \
    --composite-dir exp/cp1_cache/runs/composite_judge_demo
```

关键观察指标：

| 指标 | 期望对比 |
|---|---|
| `success_rate` | composite ≥ threshold（命中质量 ↑） |
| `cp1_full_hit_rate` | composite 可能 ↓（更保守），由 weights 调） |
| `cp1_warm_start_rate` | composite 提升（多了 tier） |
| `mean_start_t` | composite 应稳定在配置的 `warm_start_t` |
| `composite_factor_log` | 每 verdict 各 factor 值（B1+ 加 logger）—— 用来调权重 |

> **批次状态**：跑 experiment 与 analysis 脚本本身 = B0+（tooling 早就有）；CompositeJudge 真正参与 = B1+；F1b 因子能用 = B2。

---

## 9. 自定义因子：扩展 OnlineExtractor / OfflineWriter

### 9.1 写一个新的 OnlineExtractor

最小实现（无 OfflineWriter，无 library_stats，无 chain walk）：

```python
# src/openpi/cache/components/factors/my_factor.py
from openpi.cache.components.factors.registry import register


@register("my_factor")
class MyFactor:
    # ---- 必填 class-level capability flags ----
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = False

    def __init__(self, threshold: float):
        self._threshold = threshold
        # 实例级 orientation map 必须用 classmethod 算（保证 validator
        # 能在不实例化的情况下拿到同一份 key 列表）
        self.descriptor_orientations = self.__class__.describe(
            {"threshold": threshold}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        # 必须是纯函数：只看 params，不读 library_stats / 不做 I/O
        return {"my_factor_score": "safe"}

    def extract(self, results, view, history, cached_data) -> dict[str, float]:
        # 返回 dict 的 keys 必须 == self.descriptor_orientations.keys()
        # 否则 CompositeJudge 在 verdict 时抛 RuntimeError(key contract violation)
        winner_payload = view.get(results[0].id)
        score = float(winner_payload.action_chunk[0].abs().mean().item())
        return {"my_factor_score": score}
```

注册后立即可在 YAML 用：

```yaml
factors:
  - type: my_factor
    params:
      threshold: 0.5
```

### 9.2 写一个需要 library_stats 的因子

```python
@register("my_normed_factor")
class MyNormedFactor:
    required_top_k: int = 0
    requires_library_stats: bool = True   # 让 builder 自动注入 library_stats kwarg
    requires_chain_walk: bool = False

    def __init__(self, library_stats: "LibraryStats"):
        self._sigma = library_stats.action_sigma
        self.descriptor_orientations = self.__class__.describe({})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"my_normed_score": "risky"}

    def extract(self, results, view, history, cached_data):
        ...
```

`requires_library_stats=True` 自动让 validator 强制 `backend.type=in_memory`（library_stats 仅 InMemoryBackend 暴露），并让 `_build_judge` 在构造时注入 `library_stats=library_stats` kwarg。

### 9.3 写一个走 chain walk 的因子

```python
@register("my_chain_factor")
class MyChainFactor:
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = True   # validator 强制 backend 提供 fetch_entry

    def __init__(self, walk_depth: int):
        self._walk_depth = walk_depth
        self.descriptor_orientations = self.__class__.describe(
            {"walk_depth": walk_depth}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"my_chain_drift": "risky"}

    def extract(self, results, view, history, cached_data):
        winner_id = results[0].id
        # PayloadView 在 B0 仅支持 ForkPolicy.TRAJECTORY；遇 fork raise
        forward_entries = view.walk_next(winner_id, k=self._walk_depth)
        ...
```

### 9.4 写一个 OfflineWriter（B2 模式）

`OfflineWriter` 在 build pkl 时被 `enrich_artifact_with_factors` helper 调，回填 `entry.payload.factors`：

```python
@register("my_offline_factor")
class MyOfflineFactor:
    # ---- OnlineExtractor surface ----
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = False

    def __init__(self, ...):
        self.descriptor_orientations = self.__class__.describe({...})

    @classmethod
    def describe(cls, params):
        return {"my_offline_score": "safe"}

    def extract(self, results, view, history, cached_data):
        # 在线侧从 payload.factors 读 offline 写好的值
        winner_payload = view.get(results[0].id)
        if winner_payload.factors is None or "my_offline_score" not in winner_payload.factors:
            return {"my_offline_score": float("nan")}     # 老 entry / 缺字段都返回 NaN
        return {"my_offline_score": winner_payload.factors["my_offline_score"]}

    # ---- OfflineWriter surface ----
    def required_payload_fields(self) -> set[str]:
        return set()      # 不需要新 raw payload tensor

    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: "LibraryStats",
    ) -> list[dict[str, float]]:
        # 返回 list 长度 == len(entries)，与 entries 平行
        return [{"my_offline_score": ...} for _ in entries]
```

### 9.5 写一个新 Composer

```python
# src/openpi/cache/components/factors/composers/my_composer.py
from openpi.cache.components.judge import HitType, JudgeResult


class MyComposer:
    def __init__(self, threshold: float):
        self._threshold = threshold
        self._orientations: dict[str, str] = {}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        # CompositeJudge.__init__ 调一次；此处校验 / 储存即可
        self._orientations = dict(orientations)

    def compose(self, factors: dict[str, float], *, winner_id: str) -> JudgeResult:
        # factors 已被 Normalizer 处理（NaN 由 Composer 自己决定如何处理）
        if all(v >= self._threshold for v in factors.values() if not (v != v)):
            return JudgeResult(HitType.FULL_HIT, winner_id)
        return JudgeResult(HitType.MISS)
```

注册到 `_build_composer` —— 当前 builder 还是显式 if-elif 树：

```python
# src/openpi/cache/config.py _build_composer
elif cfg.type == "my_composer":
    return MyComposer(threshold=cfg.tier_thresholds["full_hit"])
```

> **未来工作**：Composer 与 Normalizer 也可以做成 registry 模式（与 factor 同），目前 plan 没有这一步。

### 9.6 写一个新 Normalizer

```python
class MyNormalizer:
    def __init__(self, ...):
        self._keys: list[str] = []

    def bind_keys(self, keys: list[str]) -> None:
        self._keys = list(keys)

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        # 返回 dict 必须包含与 raw 相同的 keys
        # 如要触发 cold-start MISS：返回全 NaN dict（CompositeJudge 短路）
        ...

    def on_episode_start(self) -> None:
        # 默认 no-op；想 per-episode reset rolling window 才覆盖
        return None
```

注册到 `_build_normalizer` 的 if-elif 树。

> **批次状态**：扩展自定义因子 / Composer / Normalizer 的 API surface = B0 已 land；但 CompositeJudge 真正能跑要等 B1。B0 阶段也可单独单元测试自定义因子的 `describe` + 类构造（参见 `tests/cache/components/factors/test_registry.py` 的 stub 测试模式）。

---

## 10. 模块文件一览

| 路径 | 作用 |
|---|---|
| `src/openpi/cache/components/factors/base.py` | OnlineExtractor / OfflineWriter Protocol + LibraryStats + HistoryView |
| `src/openpi/cache/components/factors/registry.py` | `register / get_class / build / known` |
| `src/openpi/cache/components/factors/runtime_continuity.py` | F1a-A / F1a-T thin subclass |
| `src/openpi/cache/components/factors/source_window.py` | F1b-A / F1b-T thin subclass + `_DESCRIPTOR_ORIENTATIONS` + `_normalize_windows` |
| `src/openpi/cache/components/factors/consensus.py` | F2 |
| `src/openpi/cache/components/factors/composers/__init__.py` | Composer Protocol + WeightedSum / AndGate / OrGate |
| `src/openpi/cache/components/factors/normalizers/__init__.py` | Normalizer Protocol + PercentileRollingNormalizer |
| `src/openpi/cache/components/payload_view.py` | PayloadView Protocol + StoragePayloadView + ForkPolicy |
| `src/openpi/cache/components/judge.py` | `CompositeJudge`（与 ThresholdJudge / AlwaysHit / AlwaysWarmStart 同文件） |
| `src/openpi/cache/storage_types.py` | `CachePayload.factors` Optional 字段 |
| `src/openpi/cache/cache_storage.py` | `fetch_entry` + `library_stats` duck-typed facade |
| `src/openpi/cache/backends/in_memory_backend.py` | `fetch_entry` 公共方法 + `library_stats` attr |
| `src/openpi/cache/config.py` | `FactorConfig / ComposerConfig / NormalizerConfig` + `_build_composer / _build_normalizer / _build_judge` composite 分支 + B0 `_JUDGE_TYPES` 不含 `composite`（fail-fast at load） |
| `tests/cache/components/factors/` | factor metadata + Composer / Normalizer protocol + CompositeJudge 单元测试 |
| `tests/cache/test_payload_view.py` | StoragePayloadView 单元测试 |
| `tests/cache/test_cache_storage_factor_facade.py` | fetch_entry / library_stats facade 测试 |
| `tests/cache/test_config_factor.py` | FactorConfig / ComposerConfig 解析 + B0 拒收测试 |

---

## 11. 常见问题

### Q: 为什么 `f1a_t` 的 key 是 `f1a_t_jerk` 而不是 `f1a_s_jerk`？

`source` 是因子语义字段（`"action"` / `"state"`），`key_initial` 是 payload.factors 的命名空间（`"a"` / `"t"`），与 registry 名后缀对齐。这样 YAML config 中引用的 `f1a_t_jerk` 就跟 extractor 实际产出的 key 完全一致，避免 weights 静默失配（G2 Round 1 → Round 2 修订）。

### Q: 老 entry（`payload.factors is None`）会怎样？

F1b 的 OnlineExtractor.extract 会返回 NaN，Composer 按取向规则跳过该因子（weighted_sum 不计入和与权重和；and-gate 视为不通过；or-gate 视为通过被忽略）。NaN 是"信号缺失"的合法表达，不会让 verdict 报错。

### Q: 冷启动（rolling window 还没填满）会发生什么？

`PercentileRollingNormalizer(cold_start_strategy="force_miss")` 在 window 未满时返回 **all-NaN dict**；CompositeJudge 检测到 `all(isnan(v))` 后**短路 MISS**，不调 Composer。这保证 cold-start 期间 cache 走 inference 路径，不会基于不可靠 percentile 给出激进 hit。

其它策略：`"passthrough"`（直接用原始 raw）、`"lenient"`（用已积累样本，N<10 同 force_miss）。

### Q: F2 的 `K=5` 会不会破坏 strategy 的 `top_k=1` 语义？

不会。CompositeJudge 收 `min_required_top_k = max(extractor.required_top_k for extractor)` 喂给 SearchStrategy 的新 `min_top_k_hint` kwarg，`SearchStrategy` 内部用 `max(yaml_top_k, min_top_k_hint)` 决定真实 fetch 数量。YAML 里 `top_k: 1` 的语义保留（"我策略需要 1"），F2 在背后把它撑到 5。in-memory backend 实测 topk(5) vs topk(1) 性能差异忽略。

### Q: 如果我把 `f1a_t` 配进 Qdrant backend 会发生什么？

`validate_cache_config` 在 config-load 阶段 raise：`requires_chain_walk=True` 的因子要求 `backend.type=in_memory`（仅 InMemoryBackend 暴露 `fetch_entry`）。Fail-fast at load，不会进入推理路径。

### Q: `non_monotonic` 的 `direction` 怎么选？

- `"high"` —— 数值越大越好（如 `cum_disp` 大 = 大幅运动）
- `"low"`  —— 数值越小越好
- `"range:[lo, hi]"` —— 落在区间内偏 hit（如 `curv_radius` range:[0.3, 1.0] 表示喜欢中等弥散度，停滞 / 直线大幅都不偏 hit）

具体取值由数据定标得到（B2 后可加 calibration 脚本扫 percentile 给推荐值）。

### Q: 现在能跑 composite YAML 吗？

可以（B1 起）。`_JUDGE_TYPES` 已含 `composite`；validator 接受 `judge.type=composite` 并按 §6.1 跑 7 项 composite-specific 静态校验。F1a-A / F1a-T / F2 + 3 Composer + PercentileRollingNormalizer 全部已实现，端到端 verdict 可跑。F1b 因子若配进 YAML，需 backend 加载的 artifact pkl 含 `library_stats` 字段（B2 build pipeline 自动写入；老 artifact 启动时 fallback 现算）。

B0 阶段可以做的：
- 写自定义 factor 的 `describe()` / 构造单元测试
- 用 `_build_judge(cfg, library_stats=None)` 直接构造 CompositeJudge 测它的 collect+bind+key contract+cold-start sentinel
- review schema / config 校验链路是否符合预期

### Q: 如何调试 verdict 失败？

1. CompositeJudge 在 `__init__` 阶段抛 `ValueError("conflicting orientations")` —— 两个 extractor 声明同 key 但 orientation 不同，检查两个 factor 的 `descriptor_orientations` 是否冲突
2. CompositeJudge 在 verdict 时抛 `RuntimeError(... key contract violation)` —— extractor.extract() 返回的 keys 与声明不一致，检查 extract 实现
3. Composer 报 weight key 不存在 —— extractor 没产 / 拼错 key namespace（F1a-T 是 `f1a_t_*`，不是 `f1a_s_*`）
4. validator 报 `directions[K] missing for non_monotonic factor with non-zero weight` —— 在 YAML composer.directions 显式补上

### Q: 我能不能不用 Normalizer 直接用 raw factor 喂 Composer？

可以，YAML 不写 `normalizer:` 字段或 `_build_normalizer` 返回 None，CompositeJudge 把 raw dict 直接传给 Composer。但 Composer 的 `tier_thresholds` 阈值要按 raw scale 调（不再是 [0,1] percentile rank），通常更难调。推荐至少配 PercentileRollingNormalizer 让 score 标准化。
