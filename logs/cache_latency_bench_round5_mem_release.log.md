# Research Log — cp1 渐进调优 · ROUND 5：内存释放（ReleaseVisionBackend）

> **Authority**: Execution（Owner 工作流 override G1/G2，本会话内审查）· **Type**: 性能调优 / 内存（exp 内，零改 src）· **Status**: In Progress（Round 5 完成 / 进 Round 6）
> **前序**: Round 4 [`cache_latency_bench_round4_build.log.md`](cache_latency_bench_round4_build.log.md) · 草拟来源 Round 3 workflow safe_release plan

---

## 背景

R1-R4 把 latency 干到 4.15ms（目标早达成）。Round 5 转**内存维度**：快路径（LEAN/prenorm）只读预建的 `_mat`，从不读 `_entries[*].query_keys` 的 vision 张量——后者是 `_mat` normed copy 的**字节重复**（vision_0/vision_1 ~692MB）+ disabled vision_2/prompt_emb（~368MB dead，entry 存了但永不索引）。这是可回收的常驻内存（生产多 replica × 1GB 是真约束 + 降带宽争用面）。

## chosen：ReleaseVisionBackend + 哨兵 empty(0) + fail-closed 守卫

`ReleaseVisionBackend(LeanSearchBackend)`（`opt/safe_release_backend.py`），继承 R1-R4 全部优化，加：
- **`release_vision()`**：把 entry 的 cosine + disabled vision 字段张量替换为**单个共享 `torch.empty(0)` 哨兵**（保留 dict key 使 `field in query_keys` 仍 True、快路径完全不变）。入口断言 `is_frozen + _prebuilt + _unit_keys + 无 active session`。robot_state（0.3MB）保留（l2 读 `_mat` raw bucket）。
- **fail-closed 守卫**：override `_compute_field_scores`，释放后若走 fallback（会 `torch.stack` entry 张量）→ raise `ReleaseUnsafeError`（用与 PrenormDot 同构的 prenorm_ok/l2_ok 判定检测 fallback）。

**哨兵选择**（Round 3 workflow 实测裁决）：`empty(0)` fail-loud + 可捕获；vs del-key（静默空 valid_indices→全零分，无报警）；vs `resize_(0)`（`.shape` 仍报旧尺寸→静默过 dim check + stack SIGSEGV 不可捕获）。

## 结果

| | 结果 |
|---|---|
| **释放内存** | **1059.7MB**（692MB active vision 重复 + 368MB dead vision_2/prompt_emb）|
| latency total | 4.46ms（与 R4 4.15 噪声内，**释放不碰算子/`_mat`/路由**）|
| hit_rate | **1.0**（与 baseline 同 → 检索完全不变）|
| lean_hits / fallbacks | 2640 / 0（释放后真实回放全走 LEAN 读 `_mat`）|

## 等价 + 安全

- 守卫单测 5/5：释放后快路径 bit-same、entry storage nbytes==0、守卫 fallback raise、freeze/active-session 拒绝。
- 审查 APPROVE：`GUARD COMPLETE: YES`（穷举所有读 entry vision 入口——唯一物理读点 `in_memory_backend.py:386` 被 override 全拦；LEAN fallback / rrf / trajectory(新+legacy) / single / memo 全部汇流到 `_batch_field_scores → _compute_field_scores`，无绕过）、`SRC UNTOUCHED: YES`。
- 单向破坏（哨兵不可逆）与 keep_raw/rescore 互斥——depth_1 在 0/2640 fp32 record 下 rescore lever 本就 default off 未实现，无 raw 需求，取舍合理。

## 文件（全 exp/ + tests/，零 src）

`opt/safe_release_backend.py`、`opt/inject.py`（+`attach_release_vision`）、`opt/run_round5_release_latency.py`、`tests/exp/test_safe_release_backend.py`（5）。

## 后续

- **Round 6**: 综合优化报告（5 轮 latency 演进 + 内存 + portable 分析 + Tier 2(fp16)/3(降维) 评估 + 生产落地建议）。
- Tier 2 fp16+rescore 与释放冲突（rescore 需 keep_raw +692MB，与释放目标对立）——若上 fp16 则不释放，二选一，留报告评估。
