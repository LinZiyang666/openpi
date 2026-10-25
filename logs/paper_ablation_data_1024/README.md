# 看板全部数据（消融 + 主结果），给论文附录用（看板会话导出，10-24 23:xx）

本目录就是网页看板每张表背后的数据，一张表一个 json；`_dataset_descriptions.json` 是看板上各表的标题和英文说明原文。

- 来源：本地预览页 `openpi_ext/scratch/r17_ops/livepage/live_1023/harness_1023.html` 嵌的 `window.__DATA`（已按 10-24 的新 IR 口径重生）。
- 例外：`cells.json` 换成了 `livepage/dash_src/cells.json`（240 格），它比预览页多了 DP、Fast-WAM 的四个长执行段组。
- 首个种子：LIBERO 库种子 20261002 / 测试集 A；MT50 库 seed 8 / 测试 seed 7。

## 各表口径

| 文件 | 内容 | 维度 | 每格字段 | IR |
|---|---|---|---|---|
| `cells.json` | **LIBERO 缓存部件消融**（DP、Fast-WAM、Cosmos；原库；无旋钮；模型调用用减步） | 模型 × 套件 × 组 × 库大小 50–500。组：c 纯缓存、cr +修正器、cg +停滞门、crg +修正器+停滞门；DP、Fast-WAM 另有 ch、chr、chg、chrg（+长执行段）。共 240 格，每格 500 局（官方测试集 A） | n、sr、baseline_sr（全步纯推理）、diff/diff_pp、hit_rate（新鲜决策命中率）、nfe_per_control | 无正式 IR。计数在 `exp/offline_search/rounds/r17/cache/CACHE_RESULTS.json` 的 rows[].cost：controls、fresh_decisions（决策头）、library（命中）、requests（模型调用）、nfe。cohort=full 是主口径，sensitivity 是去掉 32 个固定起点后的敏感性分析 |
| `cells_vla.json` | **π0.5、GR00T LIBERO 缓存消融** | 模型 × 套件 × 组 × 库大小 50–500（含 300、400）。组：r10a 纯缓存（R10）、r15ac 缓存+修正器（R15） | n、sr、baseline_sr、diff_pp | 原表的 IR 在 `exp/offline_search/rounds/r15/pure_corr/TEST_A.md`（「SR @ IR」，D5 阶段计价，本来就是纯 GPU graph 时间，不受 10-24 改动影响） |
| `mt50_cells.json` | **MT50 缓存部件消融**（四个模型；原库；无旋钮；模型调用用减步：π0.5 K2、GR00T K2、DP DDIM10、Fast-WAM K3） | 模型 × suite（group = 四难度组等权均值，task = 逐任务均值）× 组 × 库大小 250–2500。DP、Fast-WAM、GR00T 有 8 组（c cr ch chr cg crg chg chrg），π0.5 有 4 组（c cr cg crg）。每格 2,500 局 | n、sr、baseline_sr（`mt50_base.json`，同减步纯推理）、diff/diff_pp、miss_steps | 只有 SR。计数要从各 run 的 journal/per_step 读：π0.5 cspi_k2+cspi_k2vc，GR00T csgr_h5+csgr_h6，DP csdp_1504，Fast-WAM csfw_k3+k3w4+k3w5+k3w6（取数脚本 `openpi_ext/scratch/r17_ops/mw/cells.py`） |
| `knob.json` / `mt50_knob.json` / `knob_vla.json` | **完整系统 + 逐任务 IR 旋钮**（长执行段开），原库 `_u` 与成功过滤库 `_f` 并列 | 模型 × 套件 × 全/减步 × 库大小 | sr_u/sr_f、n、diff_pp、ir_u/ir_f、pure_ir、rule（knob 或 full_system） | R17 各线已按只计 GPU 时间重算；knob_vla（π0.5/GR00T LIBERO）用 D5，不变 |
| `onoff.json` / `mt50_onoff.json` | **长执行段开（`_u`） vs 关（`_f`）**，均为成功过滤库 + 修正器 + Q3 停滞门 + 旋钮下降终档 | 模型 × 套件 × 全/减步 × 库大小 | sr、baseline_sr（同卡同步数纯推理）、diff_pp、ir、pure_ir | 已按新口径 |
| `steps.json` / `mt50_steps.json` | 纯推理只减步数 | 模型 × 步数设置 × 套件 | sr、diff、nfe_cut | — |
| `baselines.json` / `baselines_vla.json` / `mt50_base.json` | 纯推理参照 | 模型 × 套件 | sr、n | — |
| `latency.json` / `mt50_latency.json` | 4090 batch-1 分段延迟 | 模型 | stage1/2、全/减步 loop、系数 | ⛔ **旧口径**（含 CPU prepare 和 host 空隙）。以 `exp/r17_integration/profile/` 和 `rounds/r17/longseg_1017/profiles/4090/` 下的 `results_gpu/`、`results_mt50_gpu/`、`results_cached_text_gpu/` 为准 |
| `architecture.json` | 五个模型的结构与每次决策的流程 | — | — | — |

## 「缓存 + 修正器」（组 cr / r15ac）的库大小曲线覆盖

- **LIBERO-10、LIBERO-Spatial**：DP、Fast-WAM、Cosmos 来自 `cells.json`（cr，50–500 齐）；π0.5、GR00T 来自 `cells_vla.json`（r15ac，50–500 齐，含 300、400）。全部是原库、首个种子、默认段长（cr 不含长执行段）。模型调用：DP、Fast-WAM、Cosmos 用减步；π0.5、GR00T 的 R15 是全步流程。
- **MT50**：四个模型都在 `mt50_cells.json`（cr，250–2500 齐，调用为减步）。
- 正在跑的 π0.5/GR00T 300、400 是「完整系统 + 旋钮下降」的正式实验（三个种子、四个变体），不属于这张消融表。

## 作废 / 缺失 / 注意

- 消融各表（cells / cells_vla / mt50_cells）都**没有**正式 IR。要 IR 就按新的 `*_gpu` 价格表和上面的计数自己算：IR = (v·(调用+命中) + m·调用)·C0 / 控制步数。
- `cells.json` 的状态字段都是「完成」（每格 500 局）。
- 10-22 还有一批「缓存变体」（DP、Fast-WAM、GR00T 的 8 组 + chrg_koff；带 decision_heads、policy、library、controls、level_ir），文件是 `exp/offline_search/rounds/r17/longseg_1017/CACHE_VARIANT_RESULTS_1022.json`。它不在看板上，需要时可用。
