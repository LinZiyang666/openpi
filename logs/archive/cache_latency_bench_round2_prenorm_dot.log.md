# Research Log — cp1_search 渐进调优 · ROUND 2：prenorm-dot（cosine → GEMV）

> **Authority**: Execution（Owner 独裁令工作流 override G1/G2，本会话内审查，见 [[project_search_tuning_workflow]]）· **Type**: 性能调优（exp 内，零改 src）· **Status**: In Progress（Round 2 完成 / **目标达成** / 进 Round 3）
> **前序**: Round 1 [`cache_latency_bench_round1_stack_elim.log.md`](cache_latency_bench_round1_stack_elim.log.md) · **优化报告**: [`cache_latency_bench_search_optimization_report.log.md`](cache_latency_bench_search_optimization_report.log.md)

---

## 🎯 目标达成

| | baseline | R1 zero-copy | **R2 prenorm-dot** | vs baseline |
|---|---|---|---|---|
| cp1_search median | 33.92ms | 21.05ms | **4.70ms** ✅ <5ms | **7.2×** |
| cp1_search p95 | 97.08ms | 53.76ms | **7.66ms** ✅ <10ms | **12.7×** |

**`<10ms`（最好 `<5ms`）双双达成。** micro-bench 预测的 ~2.76ms 算子 + 其余五段 ≈ 4.7ms，对得上。

---

## 第二刀（chosen_approach）

`PrenormDotBackend(PrebuiltMatrixBackend)`，只 override 两个方法：
- **`_prebuild_task_matrices`**：super() 建 raw 矩阵后，对 cosine 桶（vision_0/vision_1）**in-place L2 行归一化 REPLACE**（normed-only，+0GB；只改 `self._mat`，不污染 `entry.query_keys`）；l2 桶（robot_state）保持 raw。
- **`_compute_field_scores`**：继承 Round 1 的 gather/zero-copy/safety-window；唯一算子改动 = cosine 字段 `qn=q/q.norm().clamp_min(1e-12); mat.mv(qn)`（单次 fp32 GEMV，消除 `F.cosine_similarity`）；l2 字段 `torch.norm` verbatim；非白名单/非安全窗口 → `super()` 回退。

**根因（Round 1 micro-bench 实测）**：`F.cosine_similarity` 比纯点积慢 ~24×（N=399 桶 29ms vs 1.2ms），是其 broadcast/中间张量/未走 BLAS 的实现低效。prenorm-dot 把范数 bake 进矩阵 → cosine 退化成 BLAS GEMV。联网佐证 normalize+matmul ≫ F.cosine_similarity。

---

## 等价验证（非 bit-identical，3-way verdict + 几何安全证明）

Round 2 改了数值路径 → 非 bit 等价（prenorm vs cosine raw ~2e-6，fused 实测 max 9.54e-7）。用 **near-threshold subset**（owner 要求不全量）：band=1e-5（= 10× 实测 err 上界 9.54e-7，覆盖所有 flip-eligible），subset **122 query**（从初版过宽的 1309 收紧 10.7×）。

| config | verdict_flips | winner_mismatch | geometric_safety_max | PASS |
|---|---|---|---|---|
| LOO(index_select) t=4 | 0 | 0 | **-1.83e-7** | ✅ |
| zero-copy(self-incl) t=4 | 0 | 0 | -5.76e-4 | ✅ |
| LOO t=1 | 0 | 0 | -1.83e-7 | ✅ |
| LOO t=8 | 0 | 0 | -1.83e-7 | ✅ |

**几何安全证明**：`geometric_safety_max = max(actual_err - boundary_gap) < 0` → 每个 query 的 prenorm-vs-cosine 误差都小于它到最近阈值的距离 → **数学上不可能翻档**（比"0/2640 经验观察"更强）。审查独立复现 -1.83e-7 over 全 2640。`fp32 强制 + assert`（fp16=15/2640、bf16=194/2640 翻档，故 fp32-ONLY）。`new_prenorm_hits>0` 防假阳性。

---

## 审查（步骤4，本会话 2 agent）

- **等价/精度**：APPROVE。`GEOMETRIC SAFETY VALID: YES`；fp32 even under autocast；clamp_min 真实负载（norms 816-1043）不触发；l2 bit-identical；fallback 正确。`SUBSET CAN SHRINK TO ~13 query`（err 上界 9.54e-7）。
- **工程契约**：APPROVE。`SRC UNTOUCHED: YES`；`RESCORE STUB ACCEPTABLE: YES`（geometric safety<0 证明 rescore 无需，实现它是违反 WA §3.1 的 speculative dead code）。

**响应 MINOR**：subset band 1.07e-4→1e-5（1309→122）；`l2_ok` gate on `_l2_fields`（消除 dead config）；`__init__` 加 cosine/l2 disjoint assert；`attach_prenorm_dot` 加 double-attach guard。

---

## 文件（全 exp/ + tests/，零 src 改动）

| 文件 | 内容 |
|---|---|
| `opt/prenorm_dot_backend.py` | `PrenormDotBackend`：prebuild L2 归一化 cosine 桶 + `_compute_field_scores` mv；rescore 杠杆预留（default off，raise） |
| `opt/inject.py`（扩展）| `attach_prenorm_dot` + `build_components_with_prenorm_dot` |
| `opt/run_round2_latency.py` | latency runner（pin set_num_threads + rescore 透传）|
| `opt/compare_prenorm_equivalence.py` | near-threshold subset 3-way 对比 + geometric safety + multi-thread |
| `config/round2/cp1_libero10.yaml` | round2 锚点（复制 round1，零语义变化）|
| `tests/exp/test_prenorm_dot_backend.py` | 7 单测（unit-norm / mv≈cosine / l2 bit / fallback / clamp / gather-commute / rescore raise）|

---

## rescore 杠杆（deferred，default off）

`rescore_top_k>0 + keep_raw` 预留为 bit-parity-on-winner 保险（top-K 精确 cosine 重算），本轮 `raise NotImplementedError`。理由：geometric safety<0 已证 0 翻档，rescore 无可纠正者；实现它违反最小改动。仅当未来某 box 的 BLAS jitter 让 subset 翻档、或需审计 bit-parity 时启用。

## 后续轮（Round 3+）

- **R3**: per-entry vision 张量释放（回收预建+归一化矩阵的内存翻倍；绕开 `cache_storage.py:289` 隐患）+ 线程数调优（拐点 4-5）。
- robot_state l2 点积化（micro-bench 显示非瓶颈，~0.01ms，低优先）；双 GEMV 合并（专家实测更慢，已否决）。
- 生产 rebench：cpuset 隔离 + serve_policy/pi05 co-tenant p99（带宽争用，§10）。
- depth≥2 / trajectory / rrf 的 prenorm 结构（独立设计）。

---

## 执行记录

- 步骤1 草拟（4 专家 workflow）→ 步骤2 编码（7 文件，零改 src）→ 步骤3 对比（subset 122，4 config PASS，geometric safety<0）→ 步骤4 审查（2 APPROVE，响应 MINOR）→ 步骤5 latency（**4.70ms/7.66ms，达标**）→ 步骤6 → Round 3。
- 单测 14/14（7 prenorm + 7 prebuilt 回归）。
