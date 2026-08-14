# E3 — 高相似条件下的动作分歧率（静态混叠是否存在）

> 数据源：离线库 `exp/common/data/cache_artifacts/<suite>/<key_builder>.pkl`，无需 rollout
> 主口径：`cp1_spatial_pool_16` + 各 suite 的 **E4 base 配置**（与 rollout 臂同一个 key 空间）
> 稳健性：6 个 key builder 在**同一 grid 点**上的谱系
> 产物：`data/e3/primary__e4_base_config.json`、`family__<builder>.json`、`spatial__clip_*.json`

---

## 1. 一句话结论

**在生产的 key 空间里，"key 高度相似但动作分歧"这件事几乎不发生**——top 1% 相似的同任务跨 episode 对中，动作分歧率 spatial 0.24%、libero_10 2.79%，而随机对是 47–48%。两个 suite 都判 `almost_no_aliasing`，并且这个结论在 6 种 key 编码下全部成立。**静态混叠不是 d1 天花板的解释。**

---

## 2. 度量的定义

```
ADR(τ_k, τ_a) = P( ‖a_i,₇ − a_j,₇‖₂ ≥ τ_a │ sim(i, j) ≥ τ_k )
```

三个设计决定，每一个都堵掉一条把结论做成同义反复的路：

1. **条件率而非联合率。** `τ_k` 取分位数（P99），联合率天然被 1% 封顶，"几乎没有混叠"就成了定义的推论。分母必须是"高相似对"本身。
2. **`τ_a` 是物理标定的**，取 `P95(同 episode 相邻 cycle 的 executed-action 距离)`，所以"分歧"的含义是**超过正常时间演化**，而不是某个凭空挑的数字。同一次计算顺带产出 E2 用的阶段容差 `W`。
3. **只数同任务、跨 episode 对。** 同一条轨迹的相邻帧因为平凡的时间连续性而相似，那不是混叠。

动作走完整输出链（`model_transforms.outputs → Unnormalize(quantile) → LiberoOutputs[:, :7]`）取 client-space 的 7 维，与实际下发给机器人的量一致。

bootstrap 重采单位是 **episode**（带重数），每一次抽样内部重建 pair 集合、**重新估计 `τ_a` 与 `τ_k`**，并在同一抽样内算 ADR、随机对照及其差——这样区间才是可比的。

---

## 3. 冻结常量对拍

`τ_a^phys` 与 `W` 在 G1 前由独立批次算定并写入 plan，此处只做对拍、不回填；漂移即测试失败。

| suite | τ_a^phys（冻结 / 复算） | W（冻结 / 复算） | 中位数随 lag 单调 |
|---|---|---|---|
| libero_spatial | 1.9994 / **1.9994** ✓ | 6 / **6** ✓ | 是 |
| libero_10 | 2.0036 / **2.0036** ✓ | 8 / **8** ✓ | 是 |

---

## 4. 主结果

| suite | n_pairs | n_high_sim (P99) | **ADR** | 95% CI | 随机对 ADR | ADR − 随机 的 CI | verdict |
|---|---|---|---|---|---|---|---|
| libero_spatial | 42,499 | 425 | **0.235%** | [0.00%, 1.55%] | 48.00% | [−59.51, −45.43] pp | `almost_no_aliasing` |
| libero_10 | 290,533 | 2,905 | **2.788%** | [1.19%, 4.53%] | 47.25% | [−46.58, −41.96] pp | `almost_no_aliasing` |

判据是**绝对 + 相对双条件**，两条都须成立：ADR 的 CI 上界 ≤ 5%（绝对），且 ADR 与随机对之差的 CI 上界 < 0（相对）。两个 suite 都满足。

跨-suite（draw-wise 差，非两个区间相减）：`ADR_l10 − ADR_spatial` = **+2.55pp**，CI [+0.57, +4.41]。libero_10 的混叠确实略多，但两者都落在"几乎没有"的量级里。

### 4.1 ADR 随时间间隔单调上升

| \|Δcycle\| | libero_spatial | libero_10 |
|---|---|---|
| 0–2 | （n=93，低于 200 对门槛，留空） | **1.23%** |
| 3–5 | （n=97，留空） | **6.81%** |
| >5 | 2.54% | **16.02%** |

**"高相似且时间接近"几乎从不分歧；分歧集中在时间远的对上。** 这条对 H-B 也是负面证据：如果 key 相似度完全丢失了阶段信息，分歧率就不该随 `|Δcycle|` 这样单调；实际是相似度本身已经隐含编码了阶段。

spatial 的前两档低于预注册的 200 对门槛，按规则**只报计数不报率**——不是数据缺失，是拒绝在 93 个样本上给一个看起来精确的百分数。

### 4.2 阈值网格（敏感性）

`τ_k ∈ {P90, P95, P99, P99.5} × τ_a ∈ {phys, P50, P75, P90}`，16 格全部报告。趋势一致且方向正确：放宽 `τ_k`（要求更松的相似）ADR 上升，收紧 `τ_a`（要求更大的分歧）ADR 下降。

| 口径 | libero_spatial | libero_10 |
|---|---|---|
| τ_k=P90, τ_a=phys | 1.46%（n=4,250） | 12.03%（n=29,054） |
| τ_k=P95, τ_a=phys | 0.66%（n=2,125） | 6.11%（n=14,527） |
| τ_k=P99, τ_a=phys（**主口径**） | 0.24%（n=425） | 2.79%（n=2,905） |

在任何一格里 ADR 都远低于同 `τ_a` 下的随机对照（47–48%），结论不依赖阈值的具体取法。

### 4.3 per-task 分解

libero_10 有 8/10 个 task 达到报告门槛，ADR 最高的三个是 `put both the alphabet soup and the tomato sauce in the basket`（6.50%）、`put both the cream cheese box and the butter in the basket`（3.43%）、`turn on the stove and put the moka pot on it`（3.14%）——都是**双物体 / 多阶段**任务，符合"同一视觉状态可能对应不同子目标"的直觉。这个排序供 E5 的高-ADR 子集次要分析使用（不进主比较族）。

libero_spatial **0/10** 个 task 达到门槛（每 task 的高相似对太少），故不报 per-task 率。

---

## 5. Key builder 谱系（稳健性）

6 个 key builder 在**同一 grid 点**（`grid3_vision_0@12_vision_1@12_robot_state@75`）上重跑，故差异只来自 key 编码：

| key builder | libero_spatial ADR | libero_10 ADR | 两 suite verdict |
|---|---|---|---|
| cp1_spatial_pool_16 | 0.47% | 2.34% | `almost_no_aliasing` |
| cp1_spatial_pool_64 | 0.47% | 2.38% | `almost_no_aliasing` |
| cp1_max_pool | 0.24% | 2.03% | `almost_no_aliasing` |
| cp1_mean_pool | 0.24% | 2.38% | `almost_no_aliasing` |
| clip_vit_b_32 | 0.24% | （该 suite 无对应 yaml） | `almost_no_aliasing` |
| clip_vit_l_14 | 0.71% | （同上） | `almost_no_aliasing` |

**12 个 (builder, suite) 组合全部判 `almost_no_aliasing`**，spatial 落在 0.24–0.71%、libero_10 落在 2.03–2.38% 的窄带内。随机对照恒为 47–48%，与 key 编码无关（它只依赖动作距离），这本身是个正确性 sanity check。

代价是：**这 6 种编码在混叠意义上高度同质**，所以本谱系不能用来论证"key 越强混叠越少"——E1 的剂量-反应独立地遇到同一堵墙（见 `e1_residual.md` §8）。

---

## 6. Discussion — 本实验**不**能证明什么

- **不能证明"检索取回的东西是对的"。** ADR 低说明**key 空间里相似的东西动作也相似**；它不保证 top-1 取回的那个 entry 就是当前状态该执行的动作。取回质量由 E4/E5 在闭环上测。
- **不能推广到 key 不相似的区域。** 度量是在 top 1% 相似对上定义的，对"查询落在库覆盖稀疏处"的情形没有任何说法——而 E2 提示失败恰恰可能发生在那里。
- **与 E2 的表面张力必须并列呈现。** E2 在 libero_10 上看到失败 episode 的 winner 更常来自错阶段（+16.5pp）。两者不矛盾：E3 说"相似的 key 对应相似的动作"（静态混叠不存在），E2 说"失败时取回物的时间位置更偏"（可能是查询漂到了库覆盖之外）。**任何把 E2 读成"静态混叠导致失败"的写法都与本报告直接冲突**，合并解读见 `synthesis.md`。
- **库的 `outcome` 全为 `None`**，所以"分歧的那些对里有没有一方来自失败轨迹"无法检验。
- **动作距离的阈值是同一个 τ_a 用于全部维度**，没有按维度加权（夹爪维与位姿维同权）。整 chunk 的 Frobenius 距离作为敏感性口径已在网格中给出，结论不变。
- **spatial 的样本量偏小**：425 个高相似对撑起 0.24% 这个点估计，CI 上界到 1.55%。它足以支持"≤5%"的绝对臂，但**不足以**区分 0.2% 与 1.5%。

---

## 7. 产物清单

```
exp/markov_sufficiency/data/e3/                    # gitignored
├── primary__e4_base_config.json      # 主口径：两 suite × 各自 E4 base 配置 + 跨-suite 差
├── family__cp1_spatial_pool_16.json  # 谱系：统一 grid 点，两 suite
├── family__cp1_spatial_pool_64.json
├── family__cp1_max_pool.json
├── family__cp1_mean_pool.json
├── spatial__clip_vit_b_32.json       # CLIP 系列只有 libero_spatial 有 yaml
└── spatial__clip_vit_l_14.json
```

每份产物含 `calibration`（含冻结对拍）、`analysis`（含 10,000 次 bootstrap 的 per-draw ADR 序列）、`threshold_grid`、`by_cycle_gap`、`by_task`。复现命令见 `logs/markov_sufficiency_plan.log.md` §13.4。
