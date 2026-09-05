# exp/actioncache_baseline — ActionCache 式 post-backbone（CP2）基线

设计与决策记录：[`logs/actioncache_baseline_plan.log.md`](../../logs/actioncache_baseline_plan.log.md)（v0.6，G1 APPROVED）。
运行手册：[`docs/experiments/actioncache_baseline.md`](../../docs/experiments/actioncache_baseline.md)。
原文本地副本：`docs/papers/actioncache_2607.06370v2.{pdf,txt}`。

| 脚本 | 作用 |
|---|---|
| `build_cp2_artifact.py` | 从 CP1 库逐条复制（同 id / payload / 链边），只换 key（backbone 输出 → 稀疏三值投影）与 `checkpoint_id=CP2` |
| `verify_cp2_artifact.py` | 一一对应 / CP2 标签 / 真实 backend 检索 / 元数据绑定，fail-closed |
| `parity_check.py` | H5 重建 Stage 1 vs 真 `run_stage1`（原始图像）两条路径的 key 余弦 ≥ 0.999 |
| `build_shadow_table.py` | teacher cohort 离线回放 → 每决策 top-1 cosine（全库、无 task 过滤） |
| `export_arms.py` | GST K=1 IR 寻址切点 → 每臂 yaml + `arm_matrix.yaml` + `export_record.json`，逐个 load-and-assert；机械执行 n0 ≤ 8 / n1 ≤ 7 / 每组 ≤ 17 臂预算，省略目标逐条记原因 |
| `bench_cp2_overhead.py` | 真模型 + 真 orchestrator 的 `check(CP2)` 每决策开销（cold/warm median/P95 + 分段） |
| `aggregate.py` | journal + per_step → 每臂 SR / Wilson / IR（CUDA-Graph 与 eager 两档）；fail-closed 完整性门（`stats.audit_run`：0 dup、attempt 集合相等、截断/短 failed、臂集合 == record、server `library_sha256` == record）与档纯度门 |
| `compare_to_reference.py` | 50 库组：对 `exp/rit_pareto` K=2 no-gate 前沿做两侧分层 bootstrap 的 ΔSR 三分裁决 |
| `stats.py` / `libs.py` | 统计函数（上凹包、两侧 bootstrap、审计门）与共享常量 / 契约（成本表、阈值换算、臂命名、H5 索引、`cp2_contract_problems`、`weights_digest`） |

评测用 `exp/gate_threshold_pareto/run_gtp.py --checkpoint cp2 --judge-type threshold --eval-gate always_search --warm-tiers 0.1`。

目录约定按 `docs/experiments/artifact_layout.md`：`config/`（生成的 yaml，gitignored）、`data/`（库、shadow 表、raw，gitignored）、`analysis/`（图与结果文档，入库）。
