# X9b — cache size × 成功率消融：正式分析报告（P8）

> 数据：P7 正式评测 28 臂 × 500 episode = **14,000 episodes**（2026-08-17 12:16 → 08-19 06:10）。
> 代码：`ceb0158`（评测/统计管线）+ 分析期修复（`--outcome-filter`，见 plan §12.2.2）。
> 预注册：`logs/cache_size_ablation_plan.log.md` §3/§8（G1 APPROVED R11 → G2 APPROVED R7，统计构造 G2-R5 冻结）。
> 判读全部由 **Holm 后检验**驱动（每套件 8 检验、独立 family）；CI 仅作效应量展示。

## 0. 一句话结论

**两套件一致落入分支 G**：把 B-train 数据预算内能采到的全部轨迹（每任务 45 条）灌进纯 cache
（`always_hit` + `always_search`，brute-force top-1），其成功率**确证性地落后 teacher 超过 δ = 5 pp**
（l10 gap +35.2 pp，Holm 后 p = 0.039；spatial gap +16.4 pp，Holm 后 p = 0.021）；
**size 轴未分辨出平台**——不得暗示饱和，也不得写"更多数据也没用"。

**作用域（每格义务）**：单 teacher（Pi0.5）/ 单 keybuilder（`cp1_spatial_pool_16`）/
两 LIBERO 套件 / sim-only。任何超出此作用域的引用都是过度引申。

## 1. 主结果（primary family = `all` 口径）

自变量定义（§3.1b 裁定 1）：**每任务采集轨迹数**（不过滤成败）。`success` 口径为次级对照（§4）。

### libero_10（主判别战场，teacher 锚 0.868）

| 档 | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---:|---:|---:|---:|---:|---:|
| 每任务轨迹（实测 x 轴） | 1 | 2 | 5 | 10 | 20 | **39.2** |
| entries | 555 | 1,094 | 2,741 | 5,660 | 11,720 | 26,493 |
| SR | 0.272 | 0.360 | 0.456 | 0.478 | 0.480 | **0.516** |
| SR 95% BCa CI | [.150,.420] | [.232,.492] | [.292,.658] | [.330,.634] | [.346,.626] | [.386,.650] |

**落格 `D-teacher` × `Q-fail` × `P-inconc`，M-yes = False ⇒ 分支 G**：

- **Q（主轴）**：检验 8（劣效，单侧）Holm 后 **p = 0.0390 拒绝** ⇒ `gap > 5 pp` 确立——**不够用**；
  检验 7（非劣）Holm 后 p = 1.0000 未拒。
- **D（并列）**：检验 6（方向，双侧）Holm 后 **p = 0.0478 拒绝**，`mean(d_t) > 0` ⇒ teacher 统计上更优。
- 效应量：`gap = 0.868 − 0.516 = +0.352`，95% BCa CI `[+0.220, +0.502]`。
- **P（descriptive）**：S5–S6 斜率 CI `[−0.4, +7.0] pp`（相对 2 pp 阈不可判）⇒ 平台未分辨出。
  S4→S5 几乎零增益（+0.2 pp，CI `[−3.6, +3.2]`）但 S5→S6 又 +3.6 pp——"缓爬带平段"，只可描述。

### libero_spatial（天花板区验证，teacher 锚 0.974）

| 档 | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---:|---:|---:|---:|---:|---:|
| 每任务轨迹（实测 x 轴） | 1 | 2 | 5 | 10 | 20 | **43.9** |
| entries | 229 | 448 | 1,072 | 2,133 | 4,428 | 9,813 |
| SR | 0.502 | 0.464 | 0.688 | 0.728 | 0.762 | **0.810** |

**落格同为 `D-teacher` × `Q-fail` × `P-inconc` ⇒ 分支 G**：
检验 8 Holm 后 **p = 0.0205 拒绝**；检验 6 Holm 后 **p = 0.0078 拒绝**；
`gap = +0.164`，CI `[+0.106, +0.208]`；S5–S6 斜率 CI `[−0.2, +9.6] pp` ⇒ 平台未分辨出。

**汇总（§8.4.2 规则）**：两套件各自独立 Holm、结论并列，**不做"至少一个套件显著"式推断**；
本轮两套件落格一致，以 l10 为主叙事。spatial 的功效限制（天花板区配对差异稀少、10 cluster）
本轮未成为瓶颈——其 family 反而出了更小的 p。

### 完整 Holm 表

| 检验 | l10/all p (Holm) | spatial/all p (Holm) |
|---|---|---|
| 1 S1–S2 | 0.2020 (0.8078) | 0.5766 (1.0000) |
| 2 S2–S3 | 0.0420 (0.2517) | 0.0302 (0.1815) |
| 3 S3–S4 | 0.4478 (1.0000) | 0.2663 (0.9951) |
| 4 S4–S5 | 0.9337 (1.0000) | 0.2488 (0.9951) |
| 5 S5–S6 | 0.1220 (0.6098) | 0.1005 (0.5024) |
| 6 方向（双侧） | 0.0068 (**0.0478 ✓**) | 0.0010 (**0.0078 ✓**) |
| 7 非劣（单侧） | 0.9961 (1.0000) | 0.9980 (1.0000) |
| 8 劣效（单侧） | 0.0049 (**0.0390 ✓**) | 0.0029 (**0.0205 ✓**) |

无退化档对（R3 未触发）、无退化重采样、无 not-evaluable、primary 两族均无 CI/门禁分歧。

## 2. 判读纪律（必须随结论一起引用）

1. **S1→S2 的下降不是非单调性证据**。两套件的检验 1 Holm 后均为 ~1.0；`M-yes = False`，分支 N 未触发。
   曲线图上那个下降只能作形状描述。
2. **档间检验 1–5 全部未拒**（含推动曲线主升段的 S2–S3）。10 个 cluster 的功效紧张是预注册已披露的
   设计代价（§8.1.1），"未拒绝"不得读成"无差异"。
3. **P-inconc 的含义是"平台与继续上升都未被确认"**。分支 A 式的"payload 上限不是数据量问题"在
   两套件上都没有依据；反过来"再多采就够了"同样没有依据。
4. 统计有效性口径：studentized null-imposed sign-flip（穷举 1,024 模式，确定性）在渐近边际有效性
   条件下支撑 Holm 的 strong FWER；**n = 10 的有限样本可信度由预注册 DGP 模拟支持，不宣称有限样本
   精确**（§8.2.1）。

## 3. 管线自检与数据完整性

- **逐 episode FULL_HIT 见证**：primary 两族 16 臂 × 500 episode 全部齐备（accepted-attempt 联结，
  `stale_rows_ignored = 0`）——每一条计入的 episode 都被证明纯 cache 服务，无 teacher 静默回落。
- **经验噪声底（天然 null 对照）**：l10 的 S1/S2 两口径库逐字节相同 ⇒ 跨族之差是纯运行噪声。
  实测 S1 |Δ| = 0.2 pp、S2 = 0.0 pp——闭环噪声底 ≲ 0.2 pp，远小于所有被解读的效应。
- **证据缺口披露（owner 裁定，plan §12.1）**：次级族 `spatial/success` 的 1/3,000 episode
  （S6 最后一条）无 FULL_HIT 见证，按失败照常计入。① 缺口 1/14,000（仅该次级臂）；
  ② 方向可证伪——该条判失败，若真回落 teacher（SR 0.974）反而更该成功；
  ③ 影响上界 +0.16 pp 且对 cache 不利（δ = 5 pp）。primary 两族与 l10/success **零缺口**。
- **A 池绑定**：launch record 与版本控制内冻结记录的 rollup sha256 逐位一致
  （spatial `0eeece46…`、l10 `52457a37…`）；A ∩ B = ∅ 逐任务断言通过。
- **teacher anchor 协议**：init 集（官方 pruned_init 0..49）/ seed（7）/ replan_steps（5）三轴
  与评测臂实测一致；500/500 唯一键联结。

## 4. 次级对照（`success` 口径，descriptive——不进任何 family）

| 套件 | S1 | S2 | S3 | S4 | S5 | S6 | 落格（descriptive） |
|---|---:|---:|---:|---:|---:|---:|---|
| l10 | 0.274 | 0.360 | 0.464 | 0.470 | 0.498 | 0.522 | G：`D-none` × `Q-fail` × `P-inconc` |
| spatial | 0.532 | 0.488 | 0.688 | 0.698 | 0.748 | 0.816 | 未出分析（FULL_HIT 门因上述 1 条标 FAIL，不弱化门） |

- **过滤失败轨迹在顶档几乎不影响结果**（l10 S6：0.522 vs 0.516；spatial S6：0.816 vs 0.810），
  **在小库上则可分**：spatial/all 的 S1 在 task 8 为 **0/50**（`success` 同档 16/50）——两族 S1 列表
  只差一条，正是 task 8 的失败轨迹 `episode_0`。**一条失败轨迹足以把 `always_hit` 下的一个任务打死**，
  这是裁定 1「库含失败轨迹会照常回放」的最强实证（descriptive）。
- l10/success 出现了预注册的可达格 **`D-none` × `Q-fail`**（检验 8 Holm 后 0.0468 拒、检验 6 Holm 后
  0.0615 未拒）。按冻结读法：幅度确认超 δ、方向未在家族层面确认，两者用不同的尾，**不是不一致**，
  不得转述为彼此印证；其 gap CI `[+0.202, +0.494]` 与 D 判定的分歧由家族门禁裁决（如实披露）。
  primary 族在同一数据形态上把 D 收紧为 `D-teacher`（0.0478 过线）——两族点估计几乎相同，
  差异在 Holm 临界带内，须并列展示。

## 5. 归一化敏感性（§8.3，descriptive）

| 臂 | ΔSR (recal − fixed) | 95% BCa CI | ±3 pp 等价 |
|---|---:|---|---|
| spatial S1 | +0.6 pp | [−2.8, +4.0] pp | ✗（上界越界） |
| spatial S6 | +1.0 pp | [−0.2, +2.2] pp | ✓ |
| l10 S1 | −0.6 pp | [−2.4, +0.8] pp | ✓ |
| l10 S6 | −1.4 pp | [−5.0, +1.6] pp | ✗（下界越界） |

2/4 未通过等价检验，均为**宽度问题**（CI 半宽 > 3 pp）而非方向性效应；四臂方向不一致
（spatial 正、l10 负），支持"归一化失配不是主效应"的定性判断（该判断 descriptive）。
**措辞义务**：主曲线一律限定为「**在生产标定参数下**」；分支 N 未触发，该限定不改变主结论。

## 6. 成本轴（descriptive；⚠ 本节全部数字是 **src 原版 backend** 的账）

> **08-19 补充实验 X9b-L（见 `relatency.md`）**：R1–R4 优化栈（从未上生产）实测检索斜率
> **10–13 µs/桶内条目**（两机、严格线性），l10 顶档检索仅 ~31 ms/call ⇒ **加固定分量后比 teacher
> 快 ~4.4×，本节的"盈亏平衡点"在优化栈下不存在**。以下原版数字保留作生产现状记录。

- 检索延迟：per-call ≈ `126 ms + k × entries`；P6 两点拟合 k ≈ 44 ms/千条，P7 六档反解
  k ≈ 96 ms/千条（两次都含邻居争用，真值待空闲复测）。取 P7 值时与 teacher（~690 ms/call）的
  **盈亏平衡在 ~5,900 entries**（l10 的 S4 与 S5 之间）——size 轴同时是成本轴，S6 的 26.5k entries
  下纯 cache 每 call 比 teacher 慢 ~4×。
- 内存：库常驻 ≈ 1.22× pkl（fp16 落盘 → fp32 常驻），最大组（l10/all 8 臂）~30 G。
- 吞吐旁注：12 worker 对 4 worker 的提速在 S4 档 1.76×、S6 档 ~1.3×——大库档瓶颈在 server 检索侧，
  加 client 并发收益递减。

## 7. 局限

1. **可识别性（§3.3.1）**：`always_hit` 的 SR = f(库内容, 检索质量)，检索质量固定在单 keybuilder +
   top-1 + 一套归一化参数上。本实验**测不出 payload 的绝对上限**；「不够用」的结论限定在
   **本 index 下**。index 上界臂（`sim_state` 检索）已明确推迟（plan §13.5）。
2. n = 10 cluster：档间检验功效低（§2.2）；BCa 实测欠覆盖（0.885 vs 名义 0.95，plan §8.1.2）
   使 CI 偏窄，但主结论由检验而非 CI 驱动。
3. S6 的 x 轴语义是"数据预算耗尽点"（实测 39.2 / 43.9 条/任务，非名义 45）；外推超出 B-train
   450 init 预算不可判定。
4. 延迟斜率测于共享机争用下，绝对值仅供量级参考。

## 8. 产物索引

| 产物 | 位置 |
|---|---|
| **绘图数据（单一数据源）** | 本目录 `plot_data.json`（`emit_plot_data.py` 从分析 JSON 逐字收集，记录来源 sha256；补充实验——如异机延迟——用 `--attach-latency` 挂到已有点上） |
| **X9b-L 延迟重测报告** | 本目录 `relatency.md`（优化 backend、两机、12 pkl） |
| 曲线图（每套件一张，SR 两口径叠加 + 延迟面板） | 本目录 `size_curve_libero_10.png`（all+success 叠加）、`size_curve_libero_spatial.png`（仅 all——success 族 FULL_HIT 门 FAIL 无分析产物，见 §4）；重出：`plot_size.py --data plot_data.json --family <suite>/all [--family <suite>/success] --latency-label …` |
| 分析 JSON/MD（机器可读） | weilandserver `/data/openpi/ablation_study/cache_size/results/p8/`（3 份，spatial/success 无——其 FULL_HIT 门 FAIL） |
| journal / per-step 原始数据 | weilandserver `/data/openpi/ablation_study/cache_size/eval/` |
| 库 pkl（24）与采集 h5（1,000） | 同上 `artifacts/`、`collect_h5/`（sha 台账在 `results/`） |
| 配置 provenance（72 文件） | `exp/ablation_study/cache_size/config/` |
| 执行记录与全部裁定 | `logs/cache_size_ablation_plan.log.md` §12 |
