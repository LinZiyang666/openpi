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

### GR00T N1.5 × LIBERO（`logs/actioncache_baseline_groot_plan.log.md`）

上表按 teacher profile（`libs.PI05` / `libs.GROOT_LIBERO`，由 `key_builder.type` 推断）参数化：`export_arms.py --teacher groot_libero` 需要 `--cost-record`（实测 teacher 表）、`--encoder-cost-record`（每 suite 的编码单价 E，CUDA-Graph certified）与 `--preflight-record`（决策开销门），只出 10 臂（首选 {45,60,75,90} 或整档 fallback，两路径均固定 `t01`–`t04` + `ref650`；恰 n0/n1 两档）；`aggregate.py` / `compare_to_reference.py` 按 export record 的 `cost` 摘要计价（FULL = P+E，WARM@0.875 = P+E+L/8，MISS = M+E，分母 M）；`verify_cp2_artifact.py --teacher groot_libero` 核对 schedule / teacher / `stage1_path` / (16,32)。岛上的建库、shadow、parity、采集验收与 bench 在 `exp/libero_groot/`（`cp2_reconstruct.py`、`build_cp2_artifact_groot.py`、`build_shadow_table_groot.py`、`groot_cp2_parity.py`、`emit_task_map.py`、`verify_shadow_h5.py`、`bench_cp2_overhead_groot.py`、`ops/{launch_acb_collectors,run_acb_collect_clients,run_cp2_encoder_cost,run_cp2_overhead,run_acb_eval_group}.sh`）；评测用 `run_gtp --checkpoint cp2 --warm-tiers 0.875`。runbook §9。

评测用 `exp/gate_threshold_pareto/run_gtp.py --checkpoint cp2 --judge-type threshold --eval-gate always_search --warm-tiers 0.1`。

目录约定按 `docs/experiments/artifact_layout.md`：`config/`（生成的 yaml，gitignored）、`data/`（库、shadow 表、raw，gitignored）、`analysis/`（图与结果文档，入库）。

GR00T 价格来源分开保留共享 teacher 表的标定 checkpoint 与每 suite 的 E checkpoint；汇总重验模型/layout、采样与价格值。正式导出检查 shadow JSONL 的实际 150 集身份、决策唯一连续性与有限分数，并重验 preflight 数值；所有输入与矩阵规划门均先于写臂。
