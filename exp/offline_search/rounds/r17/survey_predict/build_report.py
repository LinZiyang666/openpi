"""Assemble the owner report from already audited, read-only evidence."""
from pathlib import Path

OUT=Path(__file__).resolve().parent
# Keep all output inside OUT. Absolute source links are convenient in the owner workspace.
REPO=Path('/home/weiland/projects/openpi')
SOURCES={
 'P':REPO/'exp/r17_integration/handbacks/HANDBACK_P4.md',
 'B':REPO/'exp/r17_integration/review/REVIEW_B2.md',
 'R':OUT.parent/'horizon/RESULTS_DP_FASTWAM.md',
 'H':OUT.parent/'horizon/SUMMARY.md',
 'C':OUT.parent/'horizon/COSMOS_K2.md',
 'T':OUT.parent/'horizon/STUDY.md',
 'A':OUT.parent/'cache/CACHE_RESULTS.md',
 'S':OUT.parent/'cache/CACHE_STUDY.md',
 'I':OUT.parent/'knob/REPORT.md',
 'M':OUT/'MT50_AUDIT.md', 'L':OUT/'LIBERO_AUDIT.md', 'N':OUT/'MT50_CACHE_AUDIT.md',
 'Q':OUT/'SEARCH_SCOPE.md',
}
def ref(key,line,end=None):
    label=f'{SOURCES[key].name}:{line}'+(f'–{end}' if end else '')
    return f'[{label}]({SOURCES[key]}:{line})'
def code(path,line):return f'[{path}:{line}]({REPO/path}:{line})'

report=f'''# R17：减步与延长执行段能否事前选定？

**结论：支持“当前工作没有建立经验证的事前安全选择器，设置依赖闭环经验评估”；不支持“我们对所有模型都检验过差分预测器并证明其失效”。** DP 有实际离线动作质量探针，但没有经 SR 校准的安全阈值；其余模型的当前证据主要是经验阶梯。非单调或点估计上升说明不能把“更接近满步输出”直接当作成功率保证，本身也不能证明任何预测器必然失败。证据范围见 {ref('Q',5,11)}。

按 owner 指示，全部旧实现结果（R17 之前及 step_diag 全线）已排除，本文不引用其数值。

**读表规则。** K 是名义采样步数；实际网络评估次数另记 NFE。H_exec/C 是复用同一动作块的执行承诺长度，H_pred 是模型输出长度；传输请求周期不一定是重新推理周期。纯策略阶梯与缓存长段消融分列。百分数为 SR，↑仅指相对该表对照的观测上升；没有 ↑ 不等于显著下降。既有非劣分析直接引用；下文标“新增”的计数只读现有 journal，不新做显著性或非劣判决。{code('exp/r17_integration/common/serving.py',329)}；{ref('H',25)}；{ref('M',3,10)}

## 实际尝试过哪些信号？

| 信号 / 尝试 | 模型、数据范围 | 预先判据或实际用途 | 结果及不能支持的结论 |
|---|---|---|---|
| **减步动作与满步动作的 MSE；与示范动作的 MSE；内部 clip 比例** | **DP / LIBERO**，真实训练相机帧，共同初始噪声；实际执行了 CPU 质量探针 | 没有预注册“差异小于某阈值即可保 SR”的规则；修复后明确要求另做 held-out/benchmark quality gate | 能暴露极端低步数的严重问题；没有评估安全分类准确率、SR 损失预测误差或跨任务泛化。训练帧、单次噪声协议不覆盖闭环状态分布；不能把“差异小”升级为安全。{ref('P',287,313)} |
| 精确分布 oracle、网格与原生 sampler 对齐 | DP 数学诊断；实现正确性而非机器人任务预测 | 核验 DDIM 更新式、网格及参考实现输出 | 网格数学修复并不消除低步输出质量问题；也不验证 SR。属于上行探针的排错依据，不是另一个已验证的 SR 预测器。{ref('B',11,20)} |
| 通用离线动作误差、teacher replay / sweep | R17 共享工具；有布局与通道单测 | 代码明确返回 `offline_action_error_only` / `closed_loop_sr=None` | 未找到各模型上用其选择 K/H_exec 并对照 SR 验证的真实实验；工具存在不等于预测器已测试。{code('exp/r17_integration/search/analyze.py',164)}；{code('exp/r17_integration/search/analyze.py',192)}；{code('exp/r17_integration/tests/debug/test_analysis.py',71)} |
| native/staged 动作差异、数值 parity、短 smoke | 模型接入与运行资格检查 | 检查实现和协议是否正确 | 不检验 K/H_exec→SR 的映射，不能算本问题的预测器测试。π parity 明写无 quality claim；Fast qualification 明写无 SR 认证。{code('exp/r17_integration/models/pi05/parity.py',36)}；{code('exp/r17_integration/models/fastwam/qualify.py',20)} |
| 原生块长度、发布配置与执行上限 | 各模型的几何/协议检查 | 排除越界或改变模型几何的配置 | 只能判“能否执行”，不能判“不掉 SR”；没有发现 H_exec 安全预测器的独立验证。{ref('T',78,82)}；{code('exp/r17_integration/models/pi05/adapter.py',35)} |
| 缓存 LOEO 修正、停滞门、IR 库回放标定 | DP / Fast-WAM / Cosmos 的 R17 LIBERO 缓存；MT50 亦有缓存实现/实测 | 拟合缓存动作、触发模型回退或标定调用开销，**不选择纯策略安全 K/H_exec** | 有实做，但没有验证能预测纯策略的安全边界；不同 C 还要重拟合修正器，不能把缓存效果当作纯 horizon 证明。{code('exp/r17_integration/handbacks/HANDBACK_P6.md',1873)}；{ref('S',99)} |
| 开发集 SR 下降选档、测试后提档 | DP / Fast-WAM / Cosmos，LIBERO 的 IR 旋钮 | 每任务每档 30 局，开发容差 5pp；正式每任务 50 局并按实测结果提档 | **已经在跑闭环**，而且选的是模型调用频率；不是运行前的廉价离线预测。该记录支持“经验调参”，不能改称纯 K/H 的先验选择；测试后提档也不能称为独立未触碰的确认集。{ref('I',5,17)} |

DP 探针的当前结果如下。使用 **48 帧 × 前 5 个动作 × 7 通道**，单位为外部 quantile 归一化动作，噪声 seed=0；这是训练帧而非独立 SR 测试。记录还计算了整块动作和 physical-wire MSE，同样没有 SR 判据；下表只是原值四舍五入。{ref('P',289,313)}

| K / sampler | 相对 DDPM 满步 MSE | 相对示范 MSE | 内部 clip 比例 | 来源 |
|---|---:|---:|---:|---|
| DDIM 1 | 1.767130 | 1.948037 | 0.983445 | {ref('P',302)} |
| DDIM 2 | 0.020040 | 0.228040 | 0.056548 | {ref('P',303)} |
| DDIM 4 | 0.015955 | 0.235429 | 0.079241 | {ref('P',304)} |
| DDIM 10 | 0.018994 | 0.239900 | 0.100446 | {ref('P',305)} |
| DDPM 100 | 0 | 0.238635 | 0.106027 | {ref('P',306)} |

这里 **K1 被标为 lossy 并限制准入，K2/K4/K10 的“可加载”不是保 SR 认证**。后续 DDIM10 的闭环结果见下表，但没有预注册“此 MSE 应预测多少 SR”的映射，不能事后编造预测失败率。DP 从 DDPM 改为 DDIM，同时改变了 sampler 家族，严格说不是只改变 K。{ref('P',308,313)}；{ref('P',459,460)}；{ref('B',20)}

## 纯策略 K / H_exec 阶梯

### LIBERO

Spatial / Long 指 `libero_spatial` / `libero_10`。每个 suite、每臂 **500 局（10 任务×50 官方初态）**。DP/Fast/Cosmos 的既有分析逐局核验初态字节 hash；π/GR 的下表为本次新增 journal 汇总，仅另核对计划键、协议与模型身份，尚未补做字节/渲染复核。所有比较均可按任务与原始初态配对，不当独立两样本；同初态不保证全程噪声、batch 或运行负载相同。{ref('R',11,24)}；{ref('C',13,15)}；{ref('L',3,6)}

| 模型 | 固定 H_exec | K：Spatial SR / Long SR | 上升、非单调与限定 | 来源 |
|---|---:|---|---|---|
| π0.5 | 5 | 10：96.2 / 85.4 → 1：**97.4↑** / 83.2 | Spatial 上升、Long 下降；仅两个 K，不能称已测出非单调曲线。新增配对 g/l：Spatial 8/2，Long 34/45 | {ref('L',10,13)} |
| GR00T N1.5 | 5 | 4：90.0 / 86.8 → 1：89.2 / **87.8↑** | Long 上升；新增 g/l：Spatial 12/16，Long 33/28。K1 虽 accepted 已齐，SUMMARY/RECEIPT 仍 PARTIAL，保留此资格限制 | {ref('L',14,17)} |
| DP | 5 | DDPM100：91.2 / 79.6 → DDIM10：90.0 / 76.4 → DDIM5：90.6 / 75.4 | Spatial 在进一步减步后回升，未回到满步；Long 持续下降。包含 sampler 改变 | {ref('R',38,41)}；{ref('H',25)} |
| Fast-WAM | 5 | 10：90.8 / 91.6 → 2：**91.8↑** / 89.8 → 1：90.4 / 88.6 | Spatial 非单调，K2 高于满步；Long 下降 | {ref('R',50,53)} |
| Cosmos | 16 | 5：96.4 / 98.6 → 名义2：**96.6↑** / 96.6 | 实际 NFE 为 5→1；Spatial 上升很小；有渲染节点混杂 | {ref('C',108,109)}；{ref('C',25,28)}；{ref('C',14,15)}；{ref('C',45)} |

Object / Goal **只有满步基线，没有当前 K 或 H 阶梯**：DP **70.0 / 90.8**，Fast-WAM **95.6 / 97.8**，Cosmos **99.6 / 98.4**；各 suite 同样 **500 局**。π/GR 在本次当前正式记录中没有这两个 suite 的完整基线或阶梯。不能把两主 suite 的结果写成四 suite 都测过。{ref('L',19,20)}；{ref('L',23,24)}；{ref('L',27,28)}；{ref('Q',8)}

| 模型 / 固定 K | H_exec：Spatial SR / Long SR | 观测解释 | 来源 |
|---|---|---|---|
| DP / DDPM100 | 5：91.2 / 79.6；10：89.8 / **83.8↑**；15：86.4 / **84.0↑**；16：85.6 / **81.0↑** | Long 延长均高于默认，但非单调；Spatial 越长越低 | {ref('R',32,37)}；{ref('H',25)} |
| Fast-WAM / K10 | 5：90.8 / 91.6；10：**91.8↑ / 94.2↑**；16：**92.6↑** / 90.8；20：**92.0↑** / 91.6；32：90.0 / 85.4 | 两个 suite 均非单调；最长在 Long 明显更低 | {ref('R',42,49)}；{ref('H',25)} |
| Cosmos / K5 | 16：96.4 / 98.6 | 已执行原生完整块，未测更长纯执行段；延长会越出当前块几何 | {ref('H',25)}；{ref('H',37)} |
| π0.5 / GR00T | 仅上述固定 H_exec 基线 | 未找到当前纯策略 H 阶梯，不引用历史 C 对比填空 | {ref('L',10,17)}；{ref('Q',5,8)} |

**预注册结论必须保留。** 主门要求两个 suite 分别满足 **2pp 非劣**，在 **24 个比较**的校正下，配对 CP 差下界和 task 内配对 bootstrap 下界均过门；既有 **12 个候选**无一通过联合采用门。Fast-WAM H10 仅在未校正敏感性下双 suite 通过；其 Spatial/Long 的 family CP 下界为 **−3.23 / −2.81pp**。不能把未通过等同于已证明有害，也不能把点估计上升写成“保证无损”。这里的非劣容差也不等于严格零损失。{ref('T',123,132)}；{ref('H',3,7)}；{ref('R',42,43)}

**后续同设置复跑（新增汇总，不与主阶梯池化）。** DP `hs7dpctl_full` 的 Spatial/Long 为 **90.2/77.2**，匹配 DDIM10 为 **88.8/75.8**，各 **500 对**，g/l 分别 **29/36、59/66**；Fast-WAM `hs7fwk1_4090c` 的 K1 为 **91.6/88.6**，各 **500 局**。后者是服务配置变化后的复跑，不能替换主阶梯控制来挑更好的差值。{ref('L',30,35)}；{ref('I',115)}

### MetaWorld MT50（本次新增描述统计）

下表各臂 **2,500 局＝50 任务×50 初态，seed=7，H_exec=5**，均只计 `accepted:true` 且无非空 `error` 的正常终局。按任务/seed/原始初态可配对；仅此单一评测 seed，不能声称跨 seed 无损。**G/T = 难度组等权均值 / 任务等权均值**；难度组任务数 **28/11/6/5**，两种均值可能给出不同方向。逐任务 SR、全部纳入行号、g/l、配置和 SHA 在附件，未新做推断检验。{ref('M',3,10)}

| 模型 | K 顺序 | G/T SR%，与 K 顺序对应 | 非单调/上升 | 来源 |
|---|---|---|---|---|
| π0.5 | 1，2，3，4，5，满步10 | 38.17/49.96；**48.70/59.64↑**；**53.12/61.28↑**；**50.01/58.92↑**；43.41/53.80；46.75/56.32 | 两种均值均非单调；中间减步设置均可高于满步 | {ref('M',14,19)} |
| GR00T | 1，2，3，满步4 | 60.70/72.16；66.49/75.16；67.00/75.24；67.14/75.76 | 本组均值单调，减步未高于满步；接近不等于已证非劣 | {ref('M',20,23)} |
| Fast-WAM | 1，2，3，4，5，满步10 | 27.57/44.64；41.39/57.00；**44.00/59.08↑**；**44.04/58.52↑**；**44.65/58.72↑**；43.71/57.16 | 两种均值均非单调；部分减步臂高于满步 | {ref('M',24,29)} |
| DP | DDPM100 → DDIM10 | 46.03/61.20 → **47.89↑**/59.40 | 组均值升、任务均值降；同时换 sampler，且这两臂服务主机不同 | {ref('M',30,31)}；{ref('B',20)} |
| Cosmos | — | 不纳入本题的 MT50 范围 | 本题仅 LIBERO | owner 范围 |

配对信息示例：π K3 相对满步新增成功/失败 **255/131**，Fast K3 为 **200/152**；不能把每次成功差当独立新增样本。补充同 H100 主机记录：GR 满步4 **66.83/75.60**、K2 **66.26/75.16**；DP DDPM100 **45.58/61.00**，同表 DDIM10 仍为 **47.89/59.40**。`h1` 命名指 H100 主机，**不是更长 H_exec**；主机相同也不等于负载、batch 或模型噪声完全相同。{ref('M',17)}；{ref('M',27)}；{ref('M',31,34)}；{ref('M',7,9)}

MT50 没有找到纯策略 H_exec 扩展阶梯。π 的现用输出块就是 **5 步**，不能靠只改执行量扩到块外；GR/DP/Fast 的下节缓存 H 对照不能补作纯策略 H 的证据。{code('exp/r17_integration/models/pi05/adapter.py',35)}；{ref('M',5)}；{ref('Q',8,9)}

## 缓存系统内的长执行段（单独的经验消融）

`c/r/h/g` 分别指缓存、修正器、长执行段、停滞门。C 改变缓存更新间隔，也改变修正器/停滞标定；门可触发额外观察，因此 C 是配置承诺长度，不能当每个片段实际执行长度恒定的保证。**这不是纯减步，也不是纯策略 H 的独立复现。** DP/Fast 的当前缓存长段取 **10** 是预注册 fallback，而非通过非劣认证；Cosmos 的长段标签复用同一个 **16** 配置，不能增加样本量。{ref('S',57,68)}；{ref('S',95,99)}；{ref('H',33,39)}；{code('exp/r17_integration/search/cache_recipe.py',349)}

**LIBERO，库大小 500 集，H_exec 配置 5→10**。下表各端点每 suite **500 配对局**；双值为 Spatial/Long，K 保持各自满步。所列全是现有报告数值，不重算其统计。{ref('A',25,30)}；{ref('S',89,99)}

| 模型 | 短→长组 | 短段 SR（S/L） | 长段 SR（S/L） | 方向 | 来源 |
|---|---|---|---|---|---|
| DP | c→ch | 94.6 / 64.8 | 91.4 / **84.8↑** | suite 方向相反 | {ref('A',39,44)} |
| DP | cr→chr | 93.6 / 68.2 | **93.8↑ / 87.2↑** | 均升 | {ref('A',47,52)} |
| DP | cg→chg | 93.0 / 87.4 | 92.0 / 87.4 | 降/平 | {ref('A',41,46)} |
| DP | crg→chrg | 94.0 / 88.8 | 94.0 / 88.6 | 平/降 | {ref('A',49,54)} |
| Fast-WAM | c→ch | 87.8 / 69.0 | **89.2↑ / 75.6↑** | 均升 | {ref('A',55,60)} |
| Fast-WAM | cr→chr | 88.4 / 70.4 | **89.4↑ / 76.8↑** | 均升 | {ref('A',63,68)} |
| Fast-WAM | cg→chg | 89.2 / 81.4 | 89.2 / 78.2 | 平/降 | {ref('A',57,62)} |
| Fast-WAM | crg→chrg | 90.2 / 83.4 | 88.2 / 80.0 | 均降 | {ref('A',65,70)} |

不能只挑某个库大小下的正向结果。下面给出 **crg/chrg** 的全部已登记库大小；每格写“短/长 SR”，仍是 **每端点每 suite 500 局**。{ref('A',25)}；{ref('A',74)}

| 模型 / suite | N50 | N100 | N200 | N300 | N400 | N500 | 来源 |
|---|---|---|---|---|---|---|---|
| DP / Spatial | 85.4/84.0 | 90.2/**92.0↑** | 93.4/**94.0↑** | 93.0/**94.2↑** | 93.8/93.4 | 94.0/94.0 | {ref('A',86,90)}；{ref('A',50,54)} |
| DP / Long | 59.4/**60.6↑** | 76.4/**80.6↑** | 80.4/**83.6↑** | 84.8/**86.2↑** | 87.6/87.6 | 88.8/88.6 | {ref('A',87,91)}；{ref('A',49,53)} |
| Fast / Spatial | 85.6/**87.0↑** | 87.2/**87.4↑** | 87.8/**88.6↑** | 89.2/88.4 | 88.8/**89.2↑** | 90.2/88.2 | {ref('A',102,106)}；{ref('A',66,70)} |
| Fast / Long | 71.8/64.6 | 72.2/69.4 | 79.8/75.0 | 81.2/79.2 | 84.8/81.8 | 83.4/80.0 | {ref('A',103,107)}；{ref('A',65,69)} |

其余 c/cr/cg 的全部库大小亦在既有 {ref('A',76,107)}；这里不将“随库大小变化”误称为 H 阶梯。Cosmos 的 `crg=chrg` 为 **95.8/85.6（S/L）**，只是同一臂别名；π/GR 未找到对应的当前 LIBERO 缓存长段结果。{ref('A',37,38)}；{ref('A',6)}；{ref('Q',5,8)}

**MT50，新增 journal 汇总：crg→chrg，C=5→10。** 每个下列完整端点 **2,500 局**，G/T 含义同上；DP 用 DDIM10、GR 用 K2、Fast 用 K3，比较中不改变 K。表中 g/l 为短→长的配对新增成功/失败。{ref('N',3,9)}；{ref('N',155,195)}；{ref('B',20)}

| 模型 | 库大小 | G：短→长 | T：短→长 | g/l | 来源 |
|---|---:|---|---|---|---|
| DP | 250 | 30.80→25.15 | 46.64→41.08 | 74/213 | {ref('N',155)} |
| DP | 500 | 39.99→33.25 | 52.52→48.00 | 114/227 | {ref('N',156)} |
| DP | 1000 | 42.92→36.79 | 57.28→52.04 | 137/268 | {ref('N',157)} |
| DP | 1500 | 43.38→39.83 | 58.28→55.44 | 162/233 | {ref('N',158)} |
| DP | 2000 | 43.87→40.68 | 58.72→56.24 | 149/211 | {ref('N',159)} |
| DP | 2500 | 45.36→41.49 | 59.36→56.72 | 148/214 | {ref('N',160)} |
| GR00T | 250 | 28.65→26.93 | 42.48→39.80 | 114/181 | {ref('N',179)} |
| GR00T | 500 | 38.07→33.98 | 51.40→45.60 | 108/253 | {ref('N',180)} |
| GR00T | 1000 | 43.11→40.55 | 56.68→53.16 | 118/206 | {ref('N',181)} |
| GR00T | 1500 | 47.12→42.91 | 60.32→56.48 | 111/207 | {ref('N',182)} |
| GR00T | 2000 | 49.22→47.21 | 61.52→59.92 | 146/186 | {ref('N',183)} |
| GR00T | 2500 | 50.51→46.47 | 63.00→59.84 | 128/207 | {ref('N',184)} |
| Fast-WAM | 250 | 24.23→**24.28↑** | 39.40→38.08 | 133/166 | {ref('N',194)} |
| Fast-WAM | 500 | 28.19→27.26 | 43.72→42.12 | 108/148 | {ref('N',195)} |

MT50 缓存的 c→ch、cr→chr、cg→chg 全部完整对照见 {ref('N',133,195)}。例如 GR 的 c→ch、库 **250** 时 G/T 为 **23.04/36.12→24.45/36.76**，均升；Fast 的 c→ch、库 **500** 时为 **25.32/39.80→25.18/40.08**，两种均值方向相反。这些例子再次要求明确统计口径。π 只有原生短块，当前缓存无长段维度；Fast 续跑尚未全部齐局，未齐 cell 不给 SR、也不外推剩余库大小。{ref('N',161)}；{ref('N',186)}；{ref('N',123,131)}；{code('exp/r17_integration/models/pi05/adapter.py',37)}

## 逐模型证据覆盖与论文措辞

| 模型 | 实际 K/H_exec 安全预测器测试？ | LIBERO Spatial/Long | LIBERO Object/Goal | MT50 | RoboCasa |
|---|---|---|---|---|---|
| π0.5 | **未找到**；只有经验 K 对比，parity 不算 | K 有；纯 H/缓存 H 缺 | 当前正式证据缺 | K 有；原生短块，扩 H 缺 | 当前可用证据缺 |
| GR00T N1.5 | **未找到**；只有经验 K/H 对比 | K 有但减步臂收据 PARTIAL；纯 H/缓存 H 缺 | 当前正式证据缺 | K 有；缓存 H 有；纯 H 缺 | 当前可用证据缺 |
| DP | **有离线动作质量探针，仅 LIBERO；没有 SR 校准或 H 安全预测器验证** | K（含 sampler 变更）、纯 H、缓存 H 有 | 仅满步基线，无阶梯 | K（含 sampler 变更）、缓存 H 有；纯 H/预测器缺 | 当前可用证据缺 |
| Fast-WAM | **未找到**；只有经验 K/H 对比 | K、纯 H、缓存 H 有 | 仅满步基线，无阶梯 | K 有；缓存 H 部分完整；纯 H/预测器缺 | 当前可用证据缺 |
| Cosmos | **未找到**；只有经验 K 对比 | K 有；已执行全块；缓存 h 是别名 | 仅满步基线，无阶梯 | 不在本题范围 | 不在本题范围 |

此表“缺”均指本次限定的现行记录；不是断言从未做过未留档工作。依据为上文逐臂来源与 {ref('Q',5,11)}。

| 建议论文表述 | 判定与边界 |
|---|---|
| “We did not establish a validated a-priori method; settings were evaluated/chosen empirically.” | **支持，限定当前实现、已测模型/套件与配置**。实际记录是离线排错、经验阶梯、闭环门和预定 fallback，不是先验安全选择器。{ref('P',308,313)}；{ref('H',27,39)}；{ref('I',5,9)} |
| “We tested difference-based predictors on all five models and found them inaccurate.” | **不支持**。当前只有 DP 的质量代理实测；没有覆盖所有模型的预测准确率检验。{ref('Q',6,7)} |
| “No reliable a-priori method exists.” | **不支持**。未建立方法与不存在方法是两回事；本记录不是穷尽性否定。 |
| “Smaller K / longer H always hurts” 或 “our chosen settings preserve SR.” | **均不支持这种无条件说法**。观测有升有降；非劣未通过不证明必有损失，经验挑选也不保证零损失。{ref('R',32,53)}；{ref('H',29)}；{ref('M',14,31)} |
| “The cache ablation proves a safe pure-policy execution horizon.” | **不支持**。缓存/修正器/门参与决策，C 相关重拟合使因果范围不同；Cosmos 的 h 别名也不是额外实验。{ref('S',97,99)} |

**论文草稿（英文）**

For the fixed policies studied here, we treat the denoising budget and executed action horizon as model- and benchmark-dependent empirical choices. In our current implementation, we did not establish a validated offline criterion that certifies preservation of closed-loop success before rollout evaluation. For Diffusion Policy, offline action-deviation, demonstration-error, and clipping diagnostics exposed severe degradation at an aggressive sampling budget, but did not provide an SR-calibrated safety threshold. The records for the other policies contain empirical rollout comparisons rather than corresponding tests of an a-priori safety predictor, and horizon-extension coverage is incomplete. Observed success was sometimes higher with fewer denoising steps or longer execution segments, and some measured response curves were non-monotonic. We therefore evaluated candidate settings in closed loop, retaining the prespecified non-inferiority criteria where available and distinguishing observed improvements from evidence of no degradation. These conclusions are limited to the tested configurations; the Diffusion Policy comparison also changes the sampler family, and execution-horizon changes within the cache system additionally alter retrieval and correction behavior.

草稿对应证据：质量探针 {ref('P',287,313)}；经验阶梯 {ref('R',32,53)}、{ref('M',14,31)}、{ref('L',10,17)}；非劣及范围 {ref('T',123,132)}、{ref('S',99)}。不在草稿中声称“所有模型都测试过预测器”。

**可用相关文献（已核对原始论文页面；仅作背景，不作为不存在预测器的证明）**

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*，RSS 2023；扩展期刊版本亦在同一 arXiv 页面。与迭代去噪及 receding-horizon 执行直接相关。[原文](https://arxiv.org/abs/2303.04137)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*，2023（ACT）。与 action chunking 和动作序列建模相关；不能把 chunk 输出长度直接等同于本报告 H_exec。[原文](https://arxiv.org/abs/2304.13705)
- Prasad et al., *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation*，2024。可作为少步策略加速相关工作；它涉及蒸馏训练，不是固定权重纯减 K 的安全性证明。[原文](https://arxiv.org/abs/2405.07503)

**审计附件。** {ref('Q',1)}；{ref('M',1)}；{ref('L',1)}；{ref('N',1)}。新增计算脚本为本目录 `audit_existing.py`、`audit_libero.py`、`audit_cache_mt50.py`；全部只读现有输入。原始 journal 不变；JSON 保留输入 SHA 与纳入行号，便于复查。缓存续跑按读入字节快照汇总，具体时间见 `mt50_cache_audit.json` 的 `snapshot_utc`；后来追加的行不属于本表。没有重跑或修改预注册分析。
'''
(OUT/'REPORT.md').write_text(report)
print('Wrote',OUT/'REPORT.md','lines',len(report.splitlines()))
