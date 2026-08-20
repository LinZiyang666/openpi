# `data_authority/` — 权威实验数据登记地

本目录**不是实验**，不产生任何实验结论，也不参与任何一次 rollout。它回答且只回答一个问题：

> **一份实验数据，哪个副本是权威的？它在哪台机器上、多大、sha256 是多少、由什么配置产出、谁在消费它、它有什么已知的怪毛病？**

因此这里的结构与 `exp/` 下所有实验目录都不同：**没有 `config/` `data/` `analysis/` 四槽**，因为它既不配置实验、也不产出实验数据、更不做分析。

```
exp/data_authority/
├── README.md          # 本文件：怎么查、怎么登记、怎么校验
├── registry.py        # 台账的读取 / 校验 / 检索（纯标准库，任何节点可跑）
├── verify.py          # 副本完整性校验（file_count / size_bytes / sha256）
├── collect.py         # 把 analysis 产物搬进来并写 MANIFEST
├── records/           # 数据集台账，一个数据集一个 JSON
└── analysis/          # 收编来的分析产物
    └── <任务>/        # 一个研究任务一个子目录 —— 这一层是强制的
        ├── MANIFEST.json   # 每个文件：sha256 + 它是从哪儿搬来的
        └── <图 / 报告 / 绘图数据>
```

## 只登记指针，不搬字节

字节不在这里。本项目的实验语料动辄几十 GB（单是 cache_size 一个实验的 24 个库 pkl 就 50 GB），落在产出它的那台机器上，由 `authority.node` + `authority.path` 指出。台账走 git、字节不走 git，所以：

- `records/` **刻意不叫 `data/`** —— `.gitignore` 的 `exp/**/data/**` 会吞掉它。这个命名是硬约束，不要改。
- 想要字节，按 `verify.py` 打印的 `tether exec` 命令**去那台机器上量**，而不是拉回来量。

## `analysis/` —— 收编分析产物

数据集只登记指针，但**分析产物（图、报告、绘图数据）是直接搬进来的**：它们只有几百 KB，且是实验的最终对外面孔，值得跟着代码一起走版本控制。

两条硬约束：

- **`analysis/<任务>/` 这一层是强制的**。平铺的 `analysis/` 会变成一堆 `pareto_combined.png`，谁都说不清哪张属于哪个实验。
- **每个任务目录必须有 `MANIFEST.json`**，逐文件记 `sha256` + `source`（从哪儿搬来的）。没记来源的图不是权威产物，是孤儿 —— 而登记地存在的意义就是这里没有孤儿。

**搬 = 复制，不是移动**。实验的 `analysis/*.md` 用相对路径引自己的图，移走会静默打断已发布的报告。

```bash
# 搬一批产物进来（注意：源文件在前，flag 在后）
uv run python -m exp.data_authority.collect <任务> <src>[:<目标相对路径>] ... \
    --title "..." --source-experiment exp/<实验>

# 重生成 MANIFEST（不搬任何东西，只重新 hash；手写的 description 会保留）
uv run python -m exp.data_authority.collect <任务> --refresh
```

数据集记录里的可选字段 `analysis_task` 指向对应任务目录，`validate` 会拒绝悬空指针。

## 常用操作

```bash
# 列出全部权威数据集
uv run python -m exp.data_authority.registry ls

# 按实验 / 套件 / 类型过滤
uv run python -m exp.data_authority.registry ls --experiment weighted_sum --suite libero_10

# 看一条的完整记录（含 caveats —— 引用数据前请务必读）
uv run python -m exp.data_authority.registry show cache_size/libero_spatial/all_s3

# 校验全部台账格式
uv run python -m exp.data_authority.registry validate

# 校验本机副本是否还是台账说的那份
uv run python -m exp.data_authority.verify weighted_sum/libero_spatial/cp1_spatial_pool_16

# 远端数据集：不拉回来，打印在owner节点上执行的量法
uv run python -m exp.data_authority.verify cache_size/libero_10/all_s3

# 列出收编的分析任务 / 看某个任务的 MANIFEST
uv run python -m exp.data_authority.registry analysis
uv run python -m exp.data_authority.registry analysis threshold_pareto

# 校验某个分析任务：缺文件 / 内容变了 / 有没登记的野文件，三类分开报
uv run python -m exp.data_authority.verify threshold_pareto --analysis
```

## 登记一条新数据集

1. 在数据所在的机器上量出三个数：`file_count`、`size_bytes`、`sha256`。
   **不要用 `du`** —— 它对硬链接去重，在本项目的软链语料树上会少报。`verify.py` 里没有 `du`，原因写在该文件的模块 docstring 里。
2. 复制 `records/` 下任一条改写，文件名 = `dataset_id` 把 `/` 换成 `__` 再加 `.json`。
3. `registry validate` 必须全绿才算登记完成。
4. 已知的怪毛病写进 `caveats` —— 这个字段是本目录存在的一半理由。一个没写 caveat 的记录，读者会默认它干净。

## 效力

**建议性索引，不是准入门**。它不阻止任何实验读任何文件；它只保证「哪份是权威的」这件事有一个单一、可校验、带出处的答案。是否升级成硬规则由 owner 定。

设计依据与裁定记录见 [`logs/data_authority_plan.log.md`](../../logs/data_authority_plan.log.md)（按 WA §4，设计文档不放 `exp/`）。
