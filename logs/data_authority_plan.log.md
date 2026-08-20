# 权威实验数据登记地 `exp/data_authority/`

> **Status**: `Implemented`（§4 Code 完成；§6 Verify 通过 51/51；**G1 与 G2 均由 owner 明令 override**，见 §0 与文末 Review Log）
> **Level**: L3
> **Authority**: Execution
> **Date**: 2026-08-20
> **关联**: 首批种子条目取自 [`cache_size_ablation_plan.log.md`](cache_size_ablation_plan.log.md)（X9b）与 [`archive/weighted_sum_threshold_pareto.log.md`](archive/weighted_sum_threshold_pareto.log.md)。

---

## 0. ⚠ G1 与 G2 两道门均由 owner 明令跳过（记录在案）

本变更定级 **L3**：它修订 `docs/experiments/artifact_layout.md` —— 该文件是 WORKING_AGREEMENT §8 已注册的子系统规则文档，与 WA 同等效力。按 WA §2.4，L2+ 必须先过 G1。

2026-08-20 会话中，执行方已完整陈述定级理由与 G1 义务并停在门前；项目 owner 答复「我同意，你立马创建」，依 WORKING_AGREEMENT 开篇授予 owner 的「Holds absolute authority over this Working Agreement and all project matters. May override any process at will」行使 override。

随后 owner 追加第二次 override：「我寄给予此次超越流程的特列，免除G2审查等任何流程阻塞限制」，据同一条权源豁免 §5 G2 代码评审门。

- **被跳过的**：G1 计划评审门、**G2 代码评审门**。
- **未被跳过的**：§6 Verify（已实跑通过，见 §11）。
- **不在 override 覆盖范围内的**：§7 Commit 与 §8 Push。这两条不是「阻塞门」而是「无明确指令不得动作」，撤销阻塞不等于下达指令；执行方不据本次 override 自行 commit / push / `git add`。
- 本 plan 文件按 §2 要求落盘，但其时序是与 §4 Code 并行而非先于 —— 这是 G1 override 的直接后果，一并记录，不粉饰。
- **本次交付未经任何独立审阅**。这是 owner 的知情决定，不是疏漏；但由此 §10 风险表里所有「靠评审兜住」的假设都失去了那层兜底，尤其是 §3.1 三条新规则条款、`collect.py` 的复制语义、以及 `analysis_task` 悬空校验 —— 它们只经过作者自审与自写的测试。

## 1. 动机

`exp/` 下每个实验各自持有 `data/`，而 `.gitignore` 的 `exp/**/data/**` 默认吞掉全部字节。真实语料落在产出它的机器上（cache_size 的 24 个库 pkl 共约 50 GB 在 weilandserver `/data`），repo 里只剩下**指向它们的散落引用**：报告正文一句、plan 里一行、聊天记录里一段路径。

后果在 X9b 结题时已经具体化：判断「磁盘上这个 pkl 是不是出报告的那一个」，靠的是人肉比对 entry 数。没有单一、可校验、带出处的答案。

本目录提供该答案，且只提供该答案。

## 2. 范围

**做**：登记「哪个副本是权威的」—— 位置、大小、sha256、内容摘要、产出配置、消费方、已知怪毛病。

**不做**：不存字节、不跑实验、不产出结论、不做分析、不阻止任何实验读任何文件。

## 3. 结构（与四槽实验目录刻意不同）

```
exp/data_authority/
├── README.md          # 操作索引（设计依据在本文件，按 WA §4 不放 exp/）
├── __init__.py
├── registry.py        # 数据集台账 + 分析 MANIFEST 的格式与读取，纯标准库
├── verify.py          # 副本 / 分析任务完整性校验
├── collect.py         # 收编分析产物并写 MANIFEST
├── records/           # 数据集台账，一数据集一 JSON
└── analysis/<任务>/   # 收编来的分析产物，任务层强制
```

无 `config/` `data/`：它既不配置实验、也不产出实验数据，这两槽在此处是空壳。

**`analysis/` 是 owner 于 2026-08-20 追加的规则**（见 §3.1）。原 §1.2 条款写的是「registry must not carry `config/`, `data/` or `analysis/`」，该条款已按 owner 指令修订为「禁 `config/`、`data/`，准 `analysis/`」。

### 3.1 `analysis/` 收编规则（owner 2026-08-20 追加）

数据集只登记指针，分析产物则**直接搬进来**：图与报告只有几百 KB，且是实验的最终对外面孔，值得随代码走版本控制（`.gitignore` 只吞 `exp/**/data/**`，`analysis/` 本就 tracked）。

三条约束，全部写进 `artifact_layout.md` §1.2：

1. **`analysis/<任务>/` 这一层强制**。平铺的 `analysis/` 会变成一堆同名 `pareto_combined.png`，归属只能靠猜；任务层是可归属性的唯一保证。
2. **每个任务目录必须有 `MANIFEST.json`**，逐文件记 `sha256` + `source`。没记来源的图是孤儿，而登记地存在的意义就是这里没有孤儿。因此 `collect.py` 把「搬进来」和「写 MANIFEST」做成一个不可分的动作 —— 走这个工具就不可能塞进一个无出处的文件。
3. **搬 = 复制不是移动**（除非同一次改动里一并修订引用方）。实验的 `analysis/*.md` 用相对路径引图，`git mv` 会静默打断已发布报告：`cache_size/analysis/analysis.md` §8 与 `threshold_pareto_results.md` §7 都是这种引用。

数据集记录新增可选字段 `analysis_task`，`validate_record` 会拒绝悬空指针（指向不存在的任务目录即报错）。

`verify_analysis_task` 把失败分三类报：**缺文件**（收编没做完）/ **内容变了**（有人重画了图没重新收编）/ **有未登记的野文件**（正是 MANIFEST 要防的孤儿）。三者修法不同，不能混作一类。

**`records/` 的命名是硬约束**：叫 `data/` 会被 `.gitignore` 第 6 行吞掉，于是台账本身不进版本控制，整个目录失去意义。这条约束写进了 README 与本文件两处。

## 4. 台账格式（schema_version 1）

必填顶层：`schema_version` `dataset_id` `kind` `title` `experiment` `status` `authority` `integrity` `content` `provenance`。

- `dataset_id` = `<实验>/<套件>/<名>`，全小写；文件名 = 把 `/` 换成 `__` 再加 `.json`，`validate` 强制两者一致（否则台账可以有两个身份）。
- `authority` = `{node, path, access}`。`access: tether` 时 `path` 必须绝对 —— 远端相对路径对任何读者都不可解析。
- `integrity` = `{sha256, size_bytes, file_count}`，三者同时记录，用于区分「换了文件」（sha 变、size 不变）与「截断」（两者都变）。
- `caveats` 是本目录存在的一半理由：**一条没写 caveat 的记录，读者会默认它干净**。

`validate_record` 返回问题列表而非抛异常 —— 一次报全部，避免半校验的台账诱使读者以为其余项也查过了。

## 5. 校验判据（`verify.py`）

三个量：`file_count` / `size_bytes` / `sha256`。树形数据集的摘要取**排序后 `(相对路径, sha256)` 清单**的 sha256，因而与创建顺序无关，且单个成员出错可用 `--per-file` 定位。

**全文件无 `du`**，这是刻意的：`du` 对硬链接去重，在本项目的软链语料树上会少报。同理 `os.walk(followlinks=False)` + 只取非符号链接的普通文件 —— 否则 `find -L` 式的跟随会把别的树的字节算进来。两条都写进了 `verify.py` 的模块 docstring。

远端数据集**不拉回本地量**：`verify.py` 打印在 owner 节点上执行的 `tether exec` 命令。按 tether `usage.md` §5.16，`pull` 是固定 5 分钟超时且 `--timeout` 调大无效，GB 级 pkl 拉不回来 —— 让测量发生在字节所在处是唯一可行解。

## 6. 首批种子（4 条，全部实测非转抄）

| dataset_id | 节点 | entries | trajectories | tasks |
|---|---|---:|---:|---:|
| `cache_size/libero_spatial/all_s3` | weilandserver | 1,072 | 50 | 10 |
| `cache_size/libero_10/all_s3` | weilandserver | 2,741 | 50 | 10 |
| `weighted_sum/libero_spatial/cp1_spatial_pool_16` | local | 1,018 | **49** | 10 |
| `weighted_sum/libero_10/cp1_spatial_pool_16` | local | 2,640 | 50 | 10 |

四者 keybuilder 同为 `cp1_spatial_pool_16`、`vector_dims` 逐字段相同，结构上可直接对照。

测量方式：sha256 用 `sha256sum`（远端经 tether，l10 那条 1.14 GB 因超 tether exec 10 分钟上限改走 tmux 解耦）；entries / trajectories 用对 `artifact["entries"]` 的 `trajectory_id` 去重普查。cache_size 两条的 entry 数与 X9b 报告 §1 表格逐位吻合 ⇒ 磁盘上这两个文件即出报告的那两个。

**两条已登记的 caveat**：

1. `weighted_sum/libero_spatial` 实测 **49** 条轨迹而非 50，同管线的 libero_10 兄弟恰为 50。成因未定（候选：建库时一条 episode 被丢或 id 归并）。记录明确禁止把该库转述为「50 条轨迹」。
2. `weighted_sum/libero_10` 是 phase5 因子富化后的**原地覆盖**版本，同目录留有 `cp1_spatial_pool_16.pre_phase5.bak.pkl`（1,096,041,697 B，mtime 2026-05-22）。跨库对照时只有这一侧带 `payload.factors`。

## 7. 执行假设（owner 未逐条答复 6 项裁决，按推荐值执行；每条便宜可逆）

| # | 裁决项 | 取值 | 反悔代价 |
|---|---|---|---|
| 1 | 目录名 | `exp/data_authority/` | `git mv` + 三处索引改名 |
| 2 | 位置 | `exp/` 下（改子系统规则文档，非 WA §4 宪法条款） | 移到 repo 根需改 WA §4 |
| 3 | 存什么 | 台账 + 校验工具，字节留在产出节点 | 加 `store/` 软链槽即可，台账不动 |
| 4 | `.gitignore` | **不改**（`records/` 不在 `data/` 下） | 无 |
| 5 | 首批范围 | 本次实测的 4 个 pkl（原推荐的 executor_substitution 换成有实测数的这四条） | 增量补登 |
| 6 | 效力 | **建议性索引**，非准入门 | 升级为硬规则 = 在 artifact_layout 加约束条款 |

## 8. 文件改动

| 路径 | 改动 |
|---|---|
| `exp/data_authority/{__init__,registry,verify,collect}.py` | 新增 |
| `exp/data_authority/analysis/{cache_size,threshold_pareto}/` | 新增，26 个产物文件 + 2 份 MANIFEST |
| `exp/data_authority/README.md` | 新增（操作索引，非设计文档） |
| `exp/data_authority/records/*.json` | 新增 4 条种子台账 |
| `tests/data_authority/test_data_authority_{registry,verify,analysis}.py` | 新增，51 例 |
| `docs/experiments/artifact_layout.md` | 新增 §1.2 注册「登记目录」这一类 |
| `docs/README.md` | artifact_layout 行描述同步 §1.2 |
| `logs/README.md` | 本文件入索引 |
| `src/` | **零改动** |

## 9. 测试策略

- 台账格式：全部随附记录必过 `validate`；`dataset_id` 唯一且与文件名双向一致；sha256 / 正整数 / 远端绝对路径 / 未知 kind·status 的负例。
- 分析收编：随附 MANIFEST 全过 validate 且逐文件有 source、任务层无散落文件、`collect` 复制后源文件仍在（不是 move）、拒绝无 `--force` 覆盖、`--refresh` 保留手写 description、三类失败（缺失 / 内容变 / 未登记）各自单独可辨、悬空 `analysis_task` 被拒。
- 校验判据：**硬链接族按路径计数**（`du` 会算成一份）、**符号链接不跟随**（`find -L` 会重复计入外部字节）、树摘要顺序无关但内容敏感、`verify_path` 逐轴报告使「换文件」可与「截断」区分、`remote_command` 里不得出现 `du`。

## 9.1 并行改动（非本次执行方所为，如实记录）

会话进行中，`records/` 出现第五条记录 `latency_bench/libero_spatial/executor_costs`（`kind: benchmark_results`，同时 `registry.py` 的 `KNOWN_KINDS` 被追加该 kind）。该条目由另一路改动写入，非本次执行方产出，此处不认领、也不回退，仅记录其存在以免 G2 审阅时把它当成本次交付的一部分。它目前无 `analysis_task`（该字段是可选的，故仍通过校验）。

## 9.2 §6 Verify 记录

- `uv run pytest tests/data_authority/` → **51 passed**（blast radius = 本次唯一改动的测试目录；`src/` 零改动，无其它模块 import 本包）。
- `uv run ruff check` + `ruff format --check` → 全绿，7 文件。
- `registry validate` → 5 条记录 + 2 份 MANIFEST 全 ok。
- `verify threshold_pareto --analysis` → 19/19 逐文件 sha256 相符；`verify weighted_sum/libero_10/cp1_spatial_pool_16` → 1.1 GB 库 sha256 相符。

## 10. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 台账与实际字节漂移（有人原地覆盖了权威副本） | `verify` 是显式动作而非自动；caveat 2 正是一次已发生的原地覆盖，已登记 |
| R2 | `records/` 被后人改名成 `data/` 而静默失track | README + 本文件 + `.gitignore` 语义三处写明；`validate` 找不到记录即空台账，可见 |
| R3 | 建议性索引无人用，退化成过期摆设 | 由 owner 决定是否升级为准入门（裁决项 6，当前 = 否） |
| R4 | 远端节点下线导致权威副本不可达 | `authority.node` 显式记录节点；后续可加 `replicas` 字段（schema 已预留位置，本期未用） |

## Review Log

### G2 Round 0 — Owner

- **G2 WAIVED**（2026-08-20）。owner 原话：「我寄给予此次超越流程的特列，免除G2审查等任何流程阻塞限制」。
- 本条不是 `APPROVED` 判决，**不得转述为「G2 通过」**：本次交付没有任何独立方看过。日后若有人据此认为该目录已受审，应以本条为准 —— 它受的是豁免，不是审查。
- 豁免范围止于流程门。§7 Commit / §8 Push 仍待 owner 明确指令。
