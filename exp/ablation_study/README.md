# `ablation_study/` — 消融实验族

本目录是一个**实验族**（experiment family），不是单个实验。族下每个子目录是一个自成一体的实验，各自持有 `config/` `data/` `analysis/`。

族的共同问题：**cache 系统的价值究竟落在哪里？** 各实验从不同侧面切这同一个问题。

| 实验 | 问的是什么 | 状态 |
|---|---|---|
| `executor_substitution/` | 把 hit / miss 槽的**执行体**换掉，价值是掉在 payload 还是 index 上？ | 已收官（2026-08-14）。结论：payload 可替换（hit→学生 +6.0/+18.4pp 超整个 cache 系统），index 承重（miss 槽换学生 −19~−33pp）。报告见 `executor_substitution/analysis/analysis.md` |
| `cache_size/` | 库**多大**才够用？纯 replay 的成功率随库规模怎么走？ | 设计冻结（G1 APPROVED），实施中。设计见 [`logs/cache_size_ablation_plan.log.md`](../../logs/cache_size_ablation_plan.log.md) |
| `latency_bench/` | 各**执行体**（teacher / ACT / SmolVLA）一次推理到底多贵？瓶颈在算力还是在 kernel 发射？ | 已收官（2026-08-19）。结论：eager 下三者全是 launch-bound（GPU 利用率 6–15%），编译后 3–9 倍加速；命中步换 ACT 稳定省 71–82%，换 SmolVLA 任何档位都不省；拆 stage 在 default 档要付 32%，CUDA Graph 下归零——而生产 interceptor 恰好降级掉了 CUDA Graph。报告见 `latency_bench/analysis/analysis.md` |

---

## ⚠ 过渡态说明

族规则（`docs/experiments/artifact_layout.md` §1）要求族目录**只含** `README.md` 与各实验子目录，不得直接存放代码 / config / data。

本目录当前**尚未满足**该要求：`executor_substitution` 的文件仍平铺在族根下（`sidecar_server.py`、`config/`、`analysis/`、`data/` 等）。这是有意接受的临时偏离，原因与期限如下：

- **原因**：X14（`exp/rl_router/`）在**运行时 import** 这些模块（`run_rl_router.py:110`、`microbench_cost.py:371`、`emit_router_yamls.py:54` 等），把它们收进子目录会立即打断正在跑的实验。
- **期限**：X14 全部 run 结束后的单一静默窗口内完成收编（plan §9.4 的 M-c2），届时同步更新全部引用、软链名与 manifest 路径。
- **在此之前**：新实验（`cache_size/`）已经按族规则落位，纯新增、零破坏。

---

## 大数据落盘

族内实验的大体积产物（h5、checkpoint、pkl、rollout journal）**不在 repo 所在磁盘上**，而是落在 `/data` 并软链回来。详见各实验 `data/` 目录下的 `__READ_ME__…` 说明文件。

校验这类软链树时注意两个反直觉点（都实际踩过）：`du` 对硬链接去重、`find -L` 对软链重复计数，两者在含链接的树上都给不出与"内容"一致的数。只用不跟随链接的 `find -type f` 计数与 `-printf '%s'` 累加字节，再辅以 sha256 抽样。
