# Research Log — cp1 渐进调优 · ROUND 4：build 优化（batched avg_pool2d keybuilder）

> **Authority**: Execution（Owner 工作流 override G1/G2，本会话内审查）· **Type**: 性能调优（exp 内，零改 src）· **Status**: In Progress（Round 4 完成 / 进 Round 5）
> **前序**: Round 3 [`cache_latency_bench_round3_lean.log.md`](cache_latency_bench_round3_lean.log.md)

---

## 背景

search 被 R1-R3 压到 3.7ms 后，build 段（keybuilder spatial_pool）成第二大（1.28ms / 24%）。owner 点名优化 build，6 专家 workflow 实测多方案（batched avg_pool2d / matmul / per-field avg_pool2d）。

## chosen：batched avg_pool2d

`CP1SpatialPool16BatchedKeyBuilder(CP1SpatialPool16KeyBuilder)`（`opt/r4_pool_keybuilder.py`），只 override `build()`：vision_0+vision_1 stack `[2,256,2048]` → 一次 `avg_pool2d(k=4,s=4)` on `[2,2048,16,16]` → 一次 D2H → split。三杠杆：① `adaptive_avg_pool2d`→固定 `avg_pool2d`；② 2 camera 合一 kernel + **合一次 D2H**；③ batched 非连续视图命中更快 CPU pool 路径。

**选 batched 而非 matmul**（实测 6.35× 但 GPU cuBLAS GEMM 不保证 bit-exact，生产硬风险）**或 per-field**（双设备 bit-exact 但仅 1.22×）：**batched 双设备 bit-exact + D2H 合并 portable**，最快。matmul/per-field 降 deferred。

## 结果

| | round3 baseline | **round4 batched** | |
|---|---|---|---|
| build median | 1.28ms | **0.45ms** | **2.85×** |
| build p95 | 2.37ms | **0.74ms** | 3.2× |
| total median | 5.22ms | **4.15ms** | |
| total p95 | 7.66ms | **5.95ms** | |
| hit_rate | 1.0 | **1.0** | bit-exact key → 检索不变 |

**全程 total**：35.49ms（baseline）→ 6.27（R2 prenorm）→ 5.22（R3 lean）→ **4.15ms（R4 build）**。

## 等价（最强：逐 bit）

- 单测 3/3：`avg_pool2d==adaptive`（16→4 整除）、`_batched_spatial_pool==src _spatial_pool_tokens` ×2/×3 camera。
- GATE（`r4_equiv_batched.py`，200 真实步）：`query_key_max_abs_diff=0.0`（逐 bit）、`verdict_flips_3way=0`、`winner_id_mismatch=0`。
- 审查 APPROVE：`BIT EQUAL VALID: YES` / `SHIP CLEAN: YES` / `SRC UNTOUCHED: YES`。响应 MINOR（GATE 用 copy 而非 alias 语义对齐 ship path）。

## portable vs bench（owner 点名要的诚实）

- **PORTABLE（生产 GPU 受益）**：lever1 `adaptive→avg_pool2d`（双设备 bit-exact，去 adaptive grid 计算）；lever2 **合并 D2H**（2 次 GPU→CPU transfer 合 1 次）。
- **BENCH-ONLY artifact**：lever3 batched 非连续视图快路径（torch 2.7.1 CPU dispatch 启发式，是 bench 6.79× kernel-only 大头来源，**生产 GPU 不靠它**）。`_to_cpu_float32` 的 `.cpu()` 在生产是必需 D2H 绝不能省。生产净收益（lever1+2）量级小于 bench 但稳，且双设备 bit-exact（GPU 上线前应复跑 `torch.equal`）。

## ship vs 探索

- **ship（git add）**：`r4_pool_keybuilder.py`、`inject.py`（+`attach_batched_pool_keybuilder`，删 matmul helper）、`r4_equiv_batched.py`、`run_round4_pool_latency.py`、`tests/exp/test_batched_pool_keybuilder.py`。
- **workflow 越界写的 untracked 探索脚本不 ship**（留工作树供参考，owner 决定删否）：`matmul_pool_keybuilder.py`/`r4_build_keybuilder.py`（deferred builder）、`r4_{pool,build}_micro.py`/`r4_batch_{fuse,why}.py`/`r4_layout_check.py`/`r4_equiv_endtoend.py`/`verify_matmul_pool_e2e.py`（一次性实测）。

## 后续

- **Round 5**: 内存释放（692MB `_entries` vision 副本与 `_mat` normed copy 重复，`safe_release_backend` plan 就绪）。
- **Round 6**: Tier 2 fp16+rescore（带宽减半，与释放权衡）/ 综合优化报告。
- search 现 **~87% 主导**（3.6ms，已到带宽底+orchestrator 包装），build 段关闭。进一步压 latency 需 fp16（Tier 2）或降维（Tier 3，owner 暂缓）。
