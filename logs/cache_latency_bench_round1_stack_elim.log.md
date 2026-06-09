# Research Log — cp1_search 渐进调优 · ROUND 1：stack 消除（PrebuiltMatrixBackend）

> **Authority**: Execution（Owner 独裁令工作流 override G1/G2，见 [[project_search_tuning_workflow]] / WA §7,§10） · **Type**: 性能调优（exp 内，零改 src 老实现） · **Status**: In Progress
> **工作流**: 每轮 6 步 = ①多专家草拟 plan → ②主 agent 编码 → ③对比法验证 → ④审查 agent → ⑤latency bench → ⑥下一轮。本文件 = ROUND 1 的 plan + 执行记录。
> **基础设施**: [`cache_latency_bench_plan.log.md`](../../../logs/cache_latency_bench_plan.log.md) · **终版优化报告**: [`cache_latency_bench_search_optimization_report.log.md`](cache_latency_bench_search_optimization_report.log.md)

---

## 目标与边界

cp1_search 渐进多轮调优，最终目标 <10ms（最好 <5ms）。本轮（ROUND 1）只做**最小、零等价风险**的第一刀，并把「预建矩阵 + id→row + 安全回退 + 对比法基础设施」脚手架立起来供后续轮叠加。

**硬约束**：所有新代码只在 `exp/cache_latency_bench/`（+ `tests/exp/`）；新实现做成 `InMemoryBackend` 子类；**绝不改 src 老实现/框架**；对比法验证（老 vs 新，相同输入，3-way verdict 零翻转）；审查在本会话内 spawn。

---

## 第一刀（chosen_first_cut）

`PrebuiltMatrixBackend(InMemoryBackend)`，**只 override `_compute_field_scores`**：把 `in_memory_backend.py:386-387` 的 per-query `torch.stack([candidates[i].query_keys[field] ...]).float()` 候选矩阵重建，换成「库冻结后一次性预建的 per-(checkpoint_id, task_key, field) 连续 fp32 矩阵 + id→row dict」，search 时按候选 id `index_select` gather 出对应行（`.contiguous()`），**继续用原算子** `F.cosine_similarity`（vision cosine）/ `torch.norm(p=2,dim=1)`（robot_state l2），散射回 `[n]` 的 scores/mask 逻辑（`:398-402`）照搬。**绝不**做 prenorm-dot / torch.mv / GEMV / 张量释放 / 线程绑定。

**安全窗口**（任一不满足即 `return super()._compute_field_scores(...)` 干净回退）：所有「有该 field」的候选 id 都命中**同一** (ckpt,task,field) 桶矩阵、且 sim_type ∈ {cosine, l2}。被 trajectory / memo（sid|qid 已设）/ rrf / step_range 路径调用时——因为只 override `_compute_field_scores`、不碰 `search`/`_search_weighted_score_sum`/`_batch_field_scores`——这些路径天然走父类。

### 为什么先做 stack 消除（why_first）

三刀（stack 消除 / prenorm-dot / GEMV）里，stack 消除是**唯一只删除「重复物化」而不触碰任何归约算子或顺序**的一刀。老路径每 query 把同一批不变的库张量（`write_policy=never` 冻结）重新 stack+`.float()` 拷进新连续 buffer（N≤399、D=32768、两 vision 字段 ≈104MB/query 物化），纯重复工作。预建矩阵 gather 出的 `[V,D]` 与老 stack 出的来自**同一份 fp32 源、同序、同 dtype、同 contiguous**，逐元素 bit-identical；cosine/norm 逐行独立归约（行内对 D 求和、不跨候选行）→ raw score / normalizer / 加权和 / topk / judge 全链路按位一致 → **预期 0/2640 verdict 翻转**。prenorm-dot 改 cosine 归一化浮点路径、GEMV 改归约顺序，等价论证更重（报告已警 fp16=15/2640、bf16=194/2640 翻转），正确留 R2/R3。

**已核实 src**：depth_1 → `search_strategy.py:155` depth≤1 返回 `{}` → 派发单步 `_search_weighted_score_sum`（`:613-616` 调 `_batch_field_scores` 不传 sid/qid → `:423-424` 提前 return `_compute_field_scores`）；`orchestrator.py:453` 把 task_key 写进 QueryFilter → 每 query 候选都来自单一 (checkpoint,task) 桶（「单桶」前置天然成立）。

---

## 文件清单（全 exp/ + tests/，零 src 改动）

| 文件 | 内容 |
|---|---|
| `exp/cache_latency_bench/opt/__init__.py` | 空包 init |
| `exp/cache_latency_bench/opt/prebuilt_matrix_backend.py` | `PrebuiltMatrixBackend(InMemoryBackend)` + `_prebuild_task_matrices()` + override `_compute_field_scores` |
| `exp/cache_latency_bench/opt/inject.py` | `build_components_with_prebuilt(config)`：src build 后过继 `_entries` 引用给子类、prebuild、替换 `storage._backend` |
| `exp/cache_latency_bench/opt/compare_equivalence.py` | 对比法主程序：2640 LOO，老 vs 新，3-way verdict + winner + score-diff → `parity_result.json` |
| `exp/cache_latency_bench/config/round1/cp1_libero10.yaml` | round1 latency 锚点 yaml（复制 depth_study/depth_1.yaml，不引新字段） |
| `exp/cache_latency_bench/opt/run_round1_latency.py` | latency runner：复用 run.py/replay.py 管线 + 注入 + warm-up |
| `tests/exp/test_prebuilt_matrix_backend.py` | 合成数据单测：快路 vs 父类 allclose + 回退 + cosine/l2 + contiguity bit-identical |

---

## 实现顺序（A→G，铁律：等价 E/F 先于 latency G）

- **A** 子类骨架（import InMemoryBackend、`_mat`/`_id2row` 容器）
- **B** `_prebuild_task_matrices()`：遍历 `self._entries.values()`（与 `_filter_entries:343` 同迭代序），按 (ckpt, task_key) 分桶，每 field `torch.stack(...).contiguous()` 存 `_mat[(ckpt,tk,f)]` + `_id2row`
- **C** override `_compute_field_scores`：签名与父类 `:363-369` 一字对齐；`valid_indices` 复算；安全窗口判定→快路 gather+原算子+散射；否则 `super()`
- **D** `inject.py`：过继 `_entries` 引用，assert `storage._backend is new` 且 `new._entries is old._entries`
- **E** `compare_equivalence.py`：用 `build_field_normalizers`（`score_normalizers.py`）按 depth_1.yaml 构造 normalizers；每 LOO query 调两 backend 的 `search(spec)` 收 top1；写 `parity_result.json`
- **F** 单测（合成数据）先过，再跑 compare_equivalence 全量 2640 LOO
- **G** 仅在 F 的 `verdict_flips==0` 通过后跑 `run_round1_latency.py` 量 latency

---

## 对比法 protocol（步骤3 判据）

- **baseline**：src `InMemoryBackend`（老路径 stack+cosine/norm）；**new**：`PrebuiltMatrixBackend`；两者 `_entries` 指向同一批 entry（库严格相同）。
- **inputs**：复用 `data/opt_bench/task_pack.pt`（已从真实 `libero_10/cp1_spatial_pool_16.pkl` 抽 per-task raw vision_0/vision_1[N,32768]+robot_state[N,32]+ids）；每 task bucket LOO（逐 entry 当 query、其余当库）；2640 真实 LOO query，覆盖 10 task（含 N=399 最大桶）。
- **硬判据（blocking）**：(1) 3-way verdict 零翻转（top.score≥0.997697→FULL / ≥0.997403→WARM / else MISS），`verdict_flips==0/2640`；(2) `winner_id_mismatch==0`。
- **软监控**：top fused score abs-diff max≤1e-6（预期 0~1e-7 layout 级）；每字段 raw score max-abs-diff≤1e-6（gather 后 `.contiguous()` 预期 0）；近阈 margin 计数。

## latency bench plan（步骤5）

用既有 `run.py`→`replay.py`（CPU、write_policy=never、与历史 d1 同 repeats）量 cp1_search median：① baseline 现场重测（消 WSL/CPU 漂移，不直接信历史 37.5ms）；② new 测（注入 + 计时前 warm-up 全部桶矩阵）；③ 可比性铁律：同库/同 keybuilder/同 repeats/同 device/同 depth_1 语义，**只换 backend**；④ 报告 median/p95，标注预建矩阵常驻内存翻倍（不释放，留 R3）。本轮线程数**不调**。

## 预期收益

砍掉 per-query「list-comp + torch.stack + .float()」三件套 host 物化（~104MB/query 分配+拷贝），但**保留** F.cosine_similarity 的 per-query 范数重算与 torch.norm 全矩阵读取（带宽主成本，需 R2/R3）。保守估计 cp1_search median：libero_10 ~37.5ms→~15-25ms（~1.5-2.5×）；libero_spatial ~12-16ms→~8-12ms。**不指望本轮 <10ms**。

## 后续轮（deferred）

- **R2**: prenorm-dot（cosine 字段 load 时 L2 预归一化，cosine→单次 torch.mv 点积）；fp32-ONLY 等价复核，真实 backend 重证 0/2640。robot_state l2 点积化 `||q-m||²` 改写。
- **R3**: 双 GEMV 合并 + per-entry vision 张量释放（回收内存翻倍；绕开 `cache_storage.py:289` 隐患）；线程数调优（拐点 4-5）/绑定。
- **后续**: 安全回退矩阵主动对比覆盖 trajectory/sid|qid/rrf/step_range；score-memo / depth>1 路径。

## Open questions 的默认决断（owner 不在场，按合理默认自决）

1. **对比入口** → 用 `backend.search(spec)` 走全链路（覆盖 `_filter_entries` 迭代序、验候选顺序与桶行映射真实一致），不用直调 `_search_weighted_score_sum`。
2. **LOO vs H5** → R1 硬等价只用 LOO（2640，含边界桶）；H5 仅用于 latency。
3. **inject `__dict__` 共享** → assert 关键属性已搬 + `new._entries is old._entries`（`__init__:77-106` 全普通实例属性、无 `__slots__`，已核）。
4. **ReplayHarness 注入 seam** → 先试 build 后替换 `storage._backend`；若时序不便则在 runner 内复刻组装（load_cache_config→build_cache_components→inject→CacheOrchestrator）。
5. **内存** → 预建 ~690MB + 原库未释放，R1 接受翻倍并在 latency 报告标注；run 前确认可用内存。

---

## 执行记录

### 步骤2-4（编码 / 对比 / 审查）
- **编码**：7 新文件（`opt/{__init__,prebuilt_matrix_backend,inject,compare_equivalence,run_round1_latency,micro_bench}.py` + `config/round1/cp1_libero10.yaml` + `tests/exp/test_prebuilt_matrix_backend.py`）+ `replay.py` 加可选 `components_hook`（向后兼容）。**零改 src 框架**。
- **对比（步骤3）**：合成单测 7/7；LOO 全量 2640（index_select 路径）`verdict_flips=0 / winner_mismatch=0 / score_abs_diff_max=0.0 / fast_hits=7920 / fallbacks=0`；self-included 2640（zero-copy 路径）同样 `0/0/0.0`。**两条快路在真实库均逐 bit 相等**。
- **审查（步骤4）**：本会话 spawn 3 审查 agent（等价 / 工程契约 / 对抗），全 APPROVE，0 BLOCKER。对抗 agent 实证 old 走父类 3 次 vs new 快路 0 父类调用（非假阳性）、corrupt winner row → 正确 divergence。响应 RESIDUAL RISK 加 frozen-library 契约 docstring + `inject` 防御 assert。详见 Review Log。

### 步骤5（latency）— 关键发现：index_select 疏漏 → 零拷贝修正
| 实现 | cp1_search median | p95 |
|---|---|---|
| baseline 老 InMemoryBackend | 33.92ms | 97.09ms |
| 新 `index_select(...).contiguous()`（**疏漏**：仍物化拷贝）| 31.67ms (1.07×) | 102.9ms |
| 新 **zero-copy**（整桶有序 → 直接用常驻矩阵）| **21.05ms (1.61×)** | **53.76ms (1.81×)** |

**教训**：`index_select` 仍重新物化 104MB，没达成"消除重建"。修正 = 检测候选==整桶有序（`rows==range(N)`）→ 零拷贝直接喂常驻矩阵；否则才 gather。修正后 1.61× median / 1.81× p95（大桶省物化最多）。两边 hit_counts 均 2640 FULL_HIT。

### 底层 micro-bench（owner 要求：真实实验定位瓶颈，不闭门造车）
真实 N=399 最大桶，CPU fp32，t=4（`opt/micro_bench.py`）：

| impl | t=4 |
|---|---|
| mv_only（纯点积，带宽底）| 1.20ms |
| rownorm_only | 1.29ms |
| **cosine_one（F.cosine_similarity 单字段）** | **29.23ms** |
| cur_full 2cos+l2（Round 1）| 57.68ms |
| **prenorm_full 2mv+l2（prenorm-dot）** | **2.76ms** |

**爆点**：`F.cosine_similarity` 比纯点积慢 **~24×**（29 vs 1.2ms），远超范数重算（rownorm 仅 1.29ms）所能解释 —— 是其 broadcast/中间张量/未走 BLAS 的实现低效。**Round 1 的 21ms 上限正源于此（保留了 cosine_similarity）**。prenorm-dot 同桶 **2.76ms（vs 57.68ms，~21×）**，独立复现报告 2.46ms，联网佐证 normalize+matmul ≫ cosine_similarity。`prenorm vs cosine raw max|diff|=2.03e-6` → 非 bit 等价，Round 2 须 fp32-ONLY 重验 3-way verdict。

**Round 1 结论**：第一刀（stack 消除 + 零拷贝）= **1.61× / 33.9→21ms**，是渐进脚手架（预建矩阵 + id→row + 安全回退 + 对比法基础设施全部立起）。**数量级提速（→~2-3ms，达标 <5ms）在 Round 2 prenorm-dot**。Status: **Round 1 完成 → 进 Round 2**。

---

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-05-31 20:37 CDT

Review angle: engineering contract / zero-src-change / injection safety / backward compatibility. Target = working tree (untracked `exp/cache_latency_bench/opt/*`, `config/round1/cp1_libero10.yaml`, `tests/exp/test_prebuilt_matrix_backend.py`; unstaged `exp/cache_latency_bench/replay.py`).

- [Non-blocking] [Suggestion] `compare_equivalence.py:111,128` LOO `shared.pop(qid)` / `shared[qid] = q_entry` mutates `_entries` directly, bypassing the C2 frozen-guarded `insert`/`delete`. — reasoning: it is comparator-only (not the latency path, not production), restores state in the same loop iteration, and is the only in-process holder of that dict, so there is no concurrent-read hazard. The direct-dict approach is in fact the only way to do LOO without tripping `BackendFrozenError` on a frozen pooled backend. Acceptable for R1; no action required.
- [Non-blocking] [Suggestion] `inject.py:30` relies on `old.__dict__` not containing `_mat/_id2row/_prebuilt/_fast_hits/_fallbacks`. — reasoning: verified — these keys are subclass-only (`InMemoryBackend.__init__` in_memory_backend.py:77-106 never sets them), so `__dict__.update` cannot clobber the containers set by `PrebuiltMatrixBackend.__init__`. Order is correct (subclass `__init__` → update → prebuild). No defensive assert needed, but one could be added in a later round.

Checklist verdict (all PASS):
1. Zero-src — `git status/diff --stat -- src/` empty (staged+unstaged+untracked); whitelist `config.py:1910` & `backend_pool.py:145` still `{"in_memory","qdrant"}`. `replay.py` diff = optional `components_hook=None` only; `run.py` (no hook) path bit-identical to pre-change.
2. `__dict__.update` safety — all parent `__init__` attrs are plain `__dict__` (no `__slots__`); only properties `vector_dims`/`is_frozen` are `_dims`/`_is_frozen`-backed and transplant correctly. Shared `_entries`/`_score_memo`/`_active_search_sessions` refs are read-only on the fast path (override never mutates them).
3. Freeze/C2 — `_prebuild_task_matrices` is not in `_MUTATION_METHODS` and only reads `_entries` + builds new `_mat`/`_id2row`, so running post-freeze is contract-compliant. `__init_subclass__` adds no new guards (subclass defines no mutation method). Timing order: prebuild in hook between build and orchestrator bind.
4. Injection seam — strategies hold the `per_conn_storage` facade object (config.py:1799) and call `self._storage.search` → `self._backend.search` at call time (cache_storage.py:111); swapping `storage._backend` (inject.py:34) is honored on every later search. No strategy caches the backend directly (`grep _backend` on orchestrator/search_strategy = none).
5. Standards — module + public docstrings present; comments English (em-dash only); no dead code; `__pycache__` gitignored. `_fast_hits/_fallbacks` are pure diagnostics, not in the numeric path.
6. Pluggability honesty — R1 only does bench injection; production whitelist untouched (item 1). Honest.

Tests: `uv run pytest tests/exp/test_prebuilt_matrix_backend.py` → 7 passed (cosine/l2 bit-identical, reorder robustness, 3 fallback modes, empty-valid). Override signature/valid-filter/scatter match parent in_memory_backend.py:363-403 verbatim.

APPROVED — proceed to step-5 latency (gated on step-3 `compare_equivalence` showing `verdict_flips==0 && winner_id_mismatch==0` per plan §对比法 protocol).
