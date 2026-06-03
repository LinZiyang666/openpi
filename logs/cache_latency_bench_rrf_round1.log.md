# Research Log — weighted_rrf 渐进调优 · Round 1：嫁接验证（复用 R1/R2 backend）

> **Authority**: Execution（Owner 独裁令工作流 override G1/G2，本会话内审查，见 [[project_search_tuning_workflow]]）· **Type**: 性能调优（exp 内，零改 src）· **Status**: In Progress（RRF Round 1 完成 / 进 Round 2）
> **weighted_sum 总报告**: [`tuning_final_report`](cache_latency_bench_tuning_final_report.log.md)（已 ship 的 R1-R5 backend）

---

## 背景

weighted_sum fusion 已做完 6 轮（35.49→4.15ms + 释放 1GB）。owner 要对 weighted_rrf 也做 6 轮。**核心发现（亲验）**：weighted_rrf 与 weighted_sum **共享同一热核** `_compute_field_scores`——`_search_weighted_rrf`(in_memory:530) 调 `_batch_field_scores(sid=qid=None)` → `:423-424` 直落 `_compute_field_scores`。故 ship 的 `PrebuiltMatrixBackend`(R1)/`PrenormDotBackend`(R2) override 对 weighted_rrf **自动生效，零新 backend 代码**。

## 第一刀：嫁接验证（全 exp/，零新 backend）

新增：libero_10 RRF config（与 weighted_sum 同库可比）+ `compare_rrf_equivalence.py`（winner-id parity 判据）+ `run_rrf_latency.py`（stock/prebuilt/prenorm 三态）+ 单测。

## 等价（winner-id parity——几何界对 RRF 失效）

RRF 用 `argsort` 排名融合（非 normalize+加权和），top RRF score 是 rank-reciprocal（~0.016），与 cosine 阈值无关 → weighted_sum 的几何安全界**结构性失效**（离散 tie）。判据 = **winner-id parity（HARD）**：

| backend | winner_mismatch | rrf_score_abs_diff_max | 结论 |
|---|---|---|---|
| R1 prebuilt | **0** / 534 | **0.0** | bit-equal（cosine 算子保留→argsort 序逐位一致→RRF 逐 bit 同）|
| R2 prenorm | **0** / 534 | 1.65e-5（仅 2 非零）| winner parity（prenorm 误差只微动 2 个非 winner RRF score，winner 0 翻）|

单测 3/3（R1 bit-equal + R2 winner parity + l2 raw-norm 不做 exp）。

## latency（同 libero_10 库，与 weighted_sum 直接可比）

| backend | RRF cp1_search median | p95 | total |
|---|---|---|---|
| stock | 30.87ms | 82.5 | 32.88 |
| prebuilt (R1) | 22.34ms | 55.5 | 24.61 |
| **prenorm (R2)** | **4.73ms** | **7.16** | 6.56 |

**RRF search 30.87→4.73ms（6.5×），与 weighted_sum 同库 4.70ms 几乎一模一样**（热核相同）——印证 owner "RRF 不差于 weighted_sum"。RRF stock 比 sum stock 还略快（30.87 vs 33.9，无 normalize 层）。kinematic judge 段 0.38ms。

## 审查 APPROVE

`REUSE VALID: YES`（R1/R2 对 RRF 自动生效真成立，无 RRF 特有路径绕过 `_compute_field_scores`）、`RRF EQUIV SOUND: YES`（winner-id parity 正确判据，R1 bit-equal/R2 0-flip 充分）、`SRC UNTOUCHED: YES`。0 finding。防假阳性守卫有效；judge 标定错配（libero_spatial jsonl）对 cp1_search latency + winner 等价无害（judge 段单独，等价直调 backend.search 不经 judge）。

## reuse vs new（诚实）

backend 层 **~90% 复用**（R1/R2 自动生效）；真新增 = config + harness（winner-id 方法学）+ runner + test。"RRF 比 sum 简单"在 latency/复用侧成立，**但等价证明侧更重**（winner-id parity 实测，无几何界可依）。

## 文件（全 exp/ + tests/，零 src）

`config/round_rrf/cp1_libero10_rrf_kin.yaml`、`opt/compare_rrf_equivalence.py`、`opt/run_rrf_latency.py`、`tests/exp/test_compare_rrf_equivalence.py`。

## RRF 6 轮 roadmap

- **R2**: prenorm 在 RRF 等价深挖——near-tie subset（exact-RRF-tie / near-RRF-margin / near-cosine-gap 并集）+ 多线程确认 0-flip 非 schedule-specific + **fp32-ONLY 定档**（dtype sweep 验 fp16/bf16 翻 RRF 档，RRF argsort 比 score_sum 对误差更敏感）。
- **R3**: `LeanRRFBackend(PrenormDotBackend)` override `_search_weighted_rrf`（**父类选 PrenormDot 非 LeanSearch**——后者 override 的是 _search_weighted_score_sum，对 RRF 零触达）。先 roofline 定 framework gap——RRF 无 normalize 层 → 浪费可能 <weighted_sum 的 ~2ms → gap<1ms 则降级"确认无浪费"。
- **R4**: 复用 batched keybuilder（正交，attach_batched_pool_keybuilder 叠加）。
- **R5**: 复用 release vision（改继承链：`RrfReleaseVisionBackend(LeanRRFBackend)` 或 mixin）。
- **R6**: 收口 e2e（winner-id / RRF-score / kinematic 三档逐档相等对 stock）+ 六段 latency 全表 + 同库对标 weighted_sum 35.49→4.15ms。
