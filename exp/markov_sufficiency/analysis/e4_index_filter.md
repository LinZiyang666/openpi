# E4 — 名义 inference-index 过滤消融

> 数据：8600 个配对 episode（2 suite × [A0–A3 各 950 + A4 500]），两个互斥 init 池合成
> 判决层级：`A3 − A1` 是 primary（Holm 族，2 个 cell）；interaction 永久估计性；其余描述性 / exploratory
> 产物：`data/e4/e4_family.json`、`data/e4/arms_<suite>/`

---

## 1. 一句话结论

**名义 index 过滤既没有让历史变得有用（两个 suite 的 `A3 − A1` 都未改善），也没有改变 depth 效应的大小（两个 interaction 的 CI 都含 0）。** 过滤本身看起来很有效（`A1 − A0` 高达 +4.8/+7.4pp），但那主要是**它把 20–26% 的搜索步变成 MISS、回退到 teacher 推理**——增益来自更少地使用 cache，而不是用得更准。

---

## 2. 这个实验测的**不是**阶段对齐

生产的 `step_filter=window/exact` 只按 **episode 内 inference-cycle 序号**过滤候选（`QueryFilter.step_range`，`search_strategy.py:393-404`）。两条快慢不同的轨迹在同一序号上未必处于同一阶段，所以它不解决"1:1 时间弹性"问题，`exact` 甚至可能加剧错位。

因此本实验的结论句只能收窄为"**在名义 index 过滤下，历史仍无增量**"，**不能**用它排除 H-B。真正的阶段对齐检验在 E1-O（`e1_residual.md` §4，用归一化进度对齐，8/8 cell 不支持 H-B）。两者合起来才构成对 H-B 的完整检验。

臂矩阵（base 配置固定，只动两个旋钮）：

| 臂 | trajectory_depth | step_filter | 角色 |
|---|---|---|---|
| A0 | 1 | all | anchor（兼 E2-primary 的采集臂） |
| A1 | 1 | window 5 | primary 的对照 |
| A2 | d_best=3 | all | 复现性检查 |
| A3 | d_best=3 | window 5 | primary |
| A4 | d_best=3 | exact | exploratory |

---

## 3. 成功率与 MISS 率（两者必须一起读）

| 臂 | libero_spatial SR | MISS 率 | libero_10 SR | MISS 率 |
|---|---|---|---|---|
| A0（d1 + all） | 0.6642 | **0.0%** | 0.4516 | **0.0%** |
| A1（d1 + window） | 0.7126 | 20.2% | 0.5253 | 25.7% |
| A2（d3 + all） | 0.6295 | **0.0%** | 0.4642 | **0.0%** |
| A3（d3 + window） | 0.6905 | 19.6% | 0.5274 | 24.9% |
| A4（d3 + exact） | 0.7300 | **23.9%** | 0.4760 | **29.1%** |

**plan §5.2 假设"`always_hit` 下每个搜索步都有 winner"，这在加了 `step_filter` 后不成立。** MISS 不是 judge 拒绝，而是 index 过滤把候选集清空，那些步直接回退到 teacher 推理——而 teacher 的 anchor SR（0.83 on spatial）高于任何 cache 臂。

⇒ **任何只看 SR 的跨臂比较都会把"少用 cache"误读成"检索更准"。** 这也是本报告每张表都并列 MISS 率的原因。

---

## 4. Primary：`A3 − A1`（Holm 族，α=0.05）

| suite | 风险差 | Holm 调整 CI | 95% 描述性 CI | McNemar p | Holm 拒绝 | π_d^obs vs proxy Q75 | verdict |
|---|---|---|---|---|---|---|---|
| libero_spatial | **−2.21pp** | [−5.16, +0.74] | [−4.84, +0.42] | 0.108 | 否 | **0.163 > 0.12** | `estimation_only_discordance_above_proxy` |
| libero_10 | **+0.21pp** | [−2.74, +3.26] | [−2.74, +3.26] | 0.944 | 否 | 0.217 < 0.31 | `no_improvement_found_inconclusive` |

两个 suite 都**没有发现改善**。spatial 额外触发了 §3.5.4 的**预注册降级规则**：实测 discordance 0.163 超过无-filter proxy 的 Q75（0.12），故其判决层级预先降为估计性——该规则只依赖 nuisance 参数、不依赖效应方向，不构成 outcome-dependent 判决。

**不得**把这两行写成"过滤无效"或"H-B 被削弱"：`no_improvement_found` 是"未发现改善、证据不足"，不是等价。等价需要 CI 完全落在 ±δ_E4=3pp 内，spatial 的 [−5.16, +0.74] 和 libero_10 的 [−2.74, +3.26] 都不满足（libero_10 差一点，但 plan §5.4 已先验声明该 suite 在 950 ep 下不可能达成 ±3pp 等价判定，此处不临场放宽）。

---

## 5. Interaction（永久估计性，不报 p 值）

`θ = (A3 − A1) − (A2 − A0)`，四臂**联合**结局的 cluster bootstrap（10,000 次，percentile CI）：

| suite | θ̂ | 95% percentile CI | n_pairs |
|---|---|---|---|
| libero_spatial | +1.26pp | [−1.16, +3.68] | 950 |
| libero_10 | −1.05pp | [−4.00, +1.79] | 950 |

**两个 CI 都含 0。** 也就是说，加不加 index 过滤，depth 的效应大小没有可检出的变化——**过滤不是 depth 失效的原因**。这条只作辅助解读，不下任何二元结论。

---

## 6. 描述性对比

### 6.1 `A2 − A0`：depth 效应的复现性检查

| suite | 风险差 | 95% CI | 读法 |
|---|---|---|---|
| libero_spatial | **−3.47pp** | [−6.11, −0.84] | **复现了历史的 depth 退化**（CI 排除 0） |
| libero_10 | +1.26pp | [−1.68, +4.32] | 未复现，CI 含 0 |

spatial 上 depth>1 确实有害，量级比 trajectory 线报告的 −12pp 小（那是不同 base 与不同 ep 数下的数字）。libero_10 上 depth 效应不可检出——与该 suite 历史上 depth 效应本就微弱、且存在 d3-trough 松动信号一致（E5 正是为此设计）。

### 6.2 `A1 − A0`：过滤本身的效应（**受 MISS 混淆**）

| suite | 风险差 | 95% CI | MISS 率变化 |
|---|---|---|---|
| libero_spatial | +4.84pp | [+2.84, +6.84] | 0% → 20.2% |
| libero_10 | +7.37pp | [+4.95, +9.69] | 0% → 25.7% |

两个 CI 都排除 0，但**这不是"过滤提高了检索质量"的证据**：A1 相对 A0 多了 20–26% 的步回退 teacher。要把两者分开，需要一个"同样比例随机放弃 cache"的对照臂，本 plan 没有采集。**该行只能作为现象记录。**

### 6.3 A4（exact）：exploratory

| suite | A4 − A2 | 95% CI | A4 MISS 率 |
|---|---|---|---|
| libero_spatial | **+11.40pp** | [+7.60, +15.20] | 23.9% |
| libero_10 | +0.20pp | [−4.00, +4.20] | 29.1% |

spatial 上 A4 是所有臂中 SR 最高的（0.7300），同时 MISS 率也最高。plan §5.4 事前就警告过 exact 可能导致候选塌陷并据此把 A4 定为 exploratory；实测支持这一判断——**A4 测的与其说是 exact 对齐的价值，不如说是"更频繁地放弃 cache"的价值**。

---

## 7. Discussion — 本实验**不**能证明什么

- **不能排除 H-B。** 见 §2：`step_filter` 是名义 index 过滤，不是阶段对齐。H-B 的检验在 E1-O。
- **不能声称"过滤无效"。** primary 的两个结局都是"未发现改善"，不是等价；等价判定需要 CI 落入 ±3pp，两个 suite 都不满足。
- **`A1 − A0` 不能读作检索质量提升。** MISS 混淆见 §3、§6.2。
- **A4 的 +11.4pp 不是"exact 对齐有效"的证据**，它与最高的 MISS 率同时出现。
- **配对只锚定 init，不锚定 noise。** flow-matching 的采样噪声在 server 侧每次独立，与 ablation 线同口径；这会稀释配对的效率，但不引入偏倚。
- **950 ep 是硬上限**（db_init 45/task + 官方 pruned 50/task 的全部 held-out），不存在扩样空间，所以降级规则是唯一出路，而不是"再多跑一点看看"。

---

## 8. 产物清单

```
exp/markov_sufficiency/data/e4/                     # gitignored
├── e4_family.json                    # Holm 族判决 + interaction + 描述性对比
├── arms_libero_spatial/              # 每臂一个 episode 结局表 + arms_summary.json
├── arms_libero_10/
├── libero_spatial/{official,db_init}/journal.jsonl + per_step__*.jsonl
└── libero_10/{official,db_init}/journal.jsonl + per_step__*.jsonl
```

两个 init 池在 journal 与 per-step 两侧都经池偏移隔离（`journal_to_arms.POOL_OFFSET` / `merge_per_step`），否则 950 个配对 episode 会塌缩成 500 且指标被污染。复现命令见 `logs/markov_sufficiency_plan.log.md` §13。
