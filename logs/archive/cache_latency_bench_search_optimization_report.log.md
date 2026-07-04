# cp1_search 优化 — 最终报告（fp32 双 GEMV CPU Fast Path）

> **Authority**: Execution · **类型**: 研究 / 优化研究（3 轮专家 agent，未向 `src/` 落地改动） · **状态**: 完成 — R1 探索 → R2 实测 → R3 对抗验证 + 报告。
> **目标**: 把 `cp1_search`（~37ms，占 CP1 延迟 ~95%）压到 **<10ms，最好 <5ms**；评估两条方向 —（1）留在 CPU，（2）迁到 GPU。
> **基础设施**: [`cache_latency_bench_plan.log.md`](../../../logs/cache_latency_bench_plan.log.md) · **深度扫描**: [`cache_latency_bench_depth_study.log.md`](cache_latency_bench_depth_study.log.md)
> **方法**: 3 个 `Workflow` 轮次（R1 = 6 探索专家 + 综合；R2 = 5 实测专家 + 综合；R3 = 4 对抗验证专家 + 报告撰写），地基为已核实的 `src` 阅读 + 一份紧凑 per-task tensor pack。所有延迟单位 **ms**；等价性基于 **2640 个真实 libero_10 leave-one-out (LOO) query** over `cp1_spatial_pool_16`。论断标注 **实测（measured）** vs **推断（reasoned）**；R3 复现的数字标 **[R3 复现]**。

## 执行摘要

**<10ms p99** 与 **<5ms median** 目标在 **CPU 上达成，但有条件**。定稿方案 —— 一个 drop-in 的 **fp32 prenorm-dot** 替换 `cp1_search` 热核（per-`(task_key, vision field)` L2-预归一化连续 fp32 矩阵；对 distinct 的 vision_0/vision_1 query 做**两次独立 `torch.mv` GEMV** + 一次 `torch.norm` robot_state L2；随后现有 `ZScoreNormalizer` → 加权和 0.25/0.4375/0.3125 → `topk(1)`）—— 实测 **median 2.46–2.81ms / p99 ~5ms idle**（i7-12700H，N=399，OMP=4），相对当前 `torch.stack` + `F.cosine_similarity` 路径（当前 median 94.8ms）是 **~35–40× 同机加速**。<10ms p99 在 idle 与面对**算力受限（compute-bound）** co-tenant（独占核上）时成立（p99 6.3–6.9ms），但**面对饱和 DRAM 带宽的 co-tenant 即使在独占核上也破功**（p99 17.7ms）：该核是内存带宽受限的（104.6 MB/query @ 48.5 GB/s，逼近 DDR5 上限），核绑定无法切分共享内存控制器。所以目标在专用 / cpuset 隔离、且 co-tenant 非带宽密集的 CPU 主机上达成；否则有风险。

正确性挺过对抗式 **3-way** 检验（这是规范指标，因为生产 `ThresholdJudge` 确为三档：FULL ≥0.997697 / WARM_START ≥0.997403 / MISS）。在全部 2640 个真实 LOO query 上：**fp32 prenorm-dot = 0 winner 翻转、0 三档 verdict 翻转**（对 `F.cosine_similarity` fp32 参考）；分类分布一致（1231 FULL / 698 WARM / 711 MISS）。从原始 `task_pack.pt` 用真实 `torch.mv` 核独立复现（0/0），并在**真实 `InMemoryBackend`** 上对 1.1GB pkl 复现（0/2640 winner/score/verdict）。**fp16 = 15** 三档翻转，**bf16 = 194** → 均不可接受；核必须 **fp32-ONLY**。残余风险真实但未触发：fp32 fused-score 误差（静态 max 2.86e-5 / 多线程 reduction 下 1.07e-4）远大于到阈值的最小间隙（1.83e-7），所以翻转在数学上可能，仅因误差**在 query 内所有候选间相关**而未发生；新路径还引入了 bit-确定的当前路径所没有的小**线程数非确定性**（7.08e-5）。**结论：CONFIRMED with caveats** —— fast path 范围限定 `depth_1` 落地；GPU 仅在专用节点值得；若需 bit 可复现则加 fp32 top-K rescore；生产签收前过一次 live-co-tenant rebench（§10）。

---

# cp1_search 优化 — 最终报告

**范围**：所研究 `depth_1` 配置（`exp/cache_latency_bench/config/depth_study/depth_1.yaml`）的 `cp1_search` 热核，libero_10 / `cp1_spatial_pool_16`。R1–2 收敛出方案；R3 对抗式地尝试推翻它，然后撰写本报告。

---

## 1. 问题与诊断

当前 cp1_search 字段打分热核 `_compute_field_scores`（`src/openpi/cache/backends/in_memory_backend.py:363-403`）每个 query 都通过对 `candidates[i].query_keys[field_name]` 的 Python 列表推导（`:386`）+ `torch.stack(vecs).float()`（`:387`）重建候选矩阵，再对 vision 做 `F.cosine_similarity`（`:392`）、对 l2 做 `torch.norm`（`:394`）。最大 task bucket（N=399）下，当前完整路径实测 **median 94.8ms / p90 104.3ms**（3 字段）。这是主导成本；下游各段可忽略 —— `_filter_entries` 对 2640 entry 0.068ms、normalizer apply 0.056ms、topk 0.007ms。

**根因**：每 query 的 `torch.stack` 每次都重新物化并重转 ~104MB vision 数据（两个 `[N,32768]` fp32 矩阵），且 `F.cosine_similarity` 每次重算候选范数。两者都可消除：depth_1 下候选是**静态、连续的 task bucket**（§6 证），故矩阵及其范数可在 load 时一次性建好。

---

## 2. 方法（3 轮）

- **R1 / R2：** 设计并实测 fast path；在二元 FULL 边界确立 fp32-only（0/2640）、延迟（idle median 2.46ms）、GPU 画像、争用 caveat。
- **R3（对抗）：** 四项独立验证 + 复现：(a) 把规范指标重新推导为**三档**（FULL/WARM/MISS）而非二元，并重跑等价性；(b) 在**真实 `InMemoryBackend`** 里实现 fast path，对 1.1GB pkl 跑 legacy-vs-fast；(c) 对**核绑定**做算力 vs 带宽 co-tenant 压力测试；(d) 探测**线程数非确定性**与"释放张量 / depth≥2"的范围边界。

---

## 3. 两条方向，附实测数

### 方向 A — 留在 CPU（推荐）
**实测。** fp32 双 GEMV fast path，N=399，真实 LOO query（i7-12700H，10C/20T，torch 2.7.1+cu126）：
- Idle：median **2.46ms** / p90 3.19ms（OMP=4）；4 物理核绑定（`taskset -c 0,2,4,6`，t4）median **2.81ms** / p90 3.67 / p99 **5.09** / min 1.92ms。8 线程 p99 5.10ms。所有 4–8 线程 median <5ms；所有 ≥2 线程 p99 <10ms。
- **[R3 复现]** loadavg ~2.8：T=4 median **2.63ms** / p90 4.27 / p99 7.36 / min 1.95ms；T=8 median 2.70 / p99 5.23；T=1 median 4.31 / p99 7.36。佐证 R2 idle 头条（p99 比 R2 的 5.10 高是残余 sibling 负载所致）。
- 对当前路径同机加速 **~35–40×**（当前 median 94.8ms；`C_winner_match: true`，`C_exact_cos_maxdiff: 1.9e-6`）。
- 内存带宽受限：**每 query 读 104.6MB @ 48.5 GB/s**（两个 fp32 `[399,32768]` prenorm 矩阵），已是 DDR5 ~60–76 GB/s 上限的大部分。线程拐点 4–5；10 逻辑线程拖垮 p99（pin8/t8 idle max 18.9ms vs pin4/t4 max 7.9ms）。

### 方向 B — 迁到 GPU（RTX 3060）
**实测。** Idle 0.35–0.75ms。但在与 policy 现实共享下：30% policy 占空比 p99 **20.5ms**；饱和时 median **84–126ms**。GPU 被 pi05 前向占用而争用。

### 决策
**留在 CPU。** CPU fast path 在 idle / 轻争用下已满足 <5ms median 与 <10ms p99，且无 GPU 常驻冲突。除非节点专用（§7），GPU 的亚毫秒 idle 无意义。

---

## 4. 定稿方案（fp32 双 GEMV CPU）

在 `load_artifact` 时，对每个 `(checkpoint_id, task_key, vision field)` 建一个**连续 fp32 矩阵**，cosine 字段 L2-预归一化（使 cosine = 单位向量点积），robot_state（l2）存**原始**，外加 `id->row` dict 与 `row->id` list。每 query：
1. `vision_0` 与 `vision_1` 是**不同的 query 向量**（`base_0_rgb` vs `left_wrist_0_rgb`）→ **两次独立 `torch.mv(unit_mat, unit_q)`**（**不是**融合的 `[2N,D]` GEMV）。
2. `robot_state` → `torch.norm(q - rawmat, dim=1)` L2。
3. 每字段原始分喂给**现有 `ZScoreNormalizer`** `0.5*(tanh((orient(x)-mu)/sigma)+1)`（`score_normalizers.py:174-178`），严格单调；`_orient` 对 l2 取负（`:55-65`）。
4. 加权和 **0.25 / 0.4375 / 0.3125** → `topk(1)`。**仅 fp32。**

这与生产算术 `in_memory_backend.py:591-632`（`_search_weighted_score_sum`：`raw -> normalizers[field](raw) -> += weight*s*mask -> topk`）一致。

---

## 5. 等价性证明 — 三档 verdict（规范指标）

生产 judge 是**三档**，且是**单个 top fused score 的纯函数**（`ThresholdJudge.__call__`，`judge.py:220-237`）：`score >= 0.997697 -> FULL_HIT`；`0.997403 <= score < 0.997697 -> WARM_START`；否则 `MISS`。两个比较均含等号（`>=`）。`start_t=0.5` 是 WARM_START 结果的**输出属性**，**不是 gate** —— 不依赖 step index 或 history。空结果 → MISS。（judge 复刻验证为与真实 `ThresholdJudge` bit-一致：跨精确边界、`math.nextafter` 的 1-ULP 邻点、粗网格，0 失配 / 24 点。）

**结果（2640 真实 LOO query，libero_10 cp1_spatial_pool_16）：**

| 精度 | winner 翻转 | 三档 verdict 翻转 | FULL↔WARM | WARM↔MISS | FULL↔MISS | max fused err |
|---|---|---|---|---|---|---|
| **fp32 prenorm-dot** | **0** | **0** | 0 | 0 | 0 | 2.86e-5（静态）/ 1.07e-4（live mv） |
| fp16 | 32 | **15** | 7 | 8 | 0 | 1.56e-2 |
| bf16 | 244 | **194** | 104 | 90 | 0 | 1.22e-1 |

参考分类分布：**1231 FULL / 698 WARM / 711 MISS** —— 三档全覆盖；那 698 个 WARM query 正是二元（仅 FULL）指标从未测过的面。`FULL↔MISS = 0`（所有精度）：WARM 带宽（2.94e-4）超过连 bf16 的 top-score 扰动，故没有 verdict 跨两档。

**独立复现（R3）：**
- **[R3 复现]** 从原始 `task_pack.pt`，用真实 `torch.mv(unit_mat, unit_q)` 核 + 绑定的 depth_1 normalizer 参数：**0 winner / 0 三档翻转**，分类计数一致（1231/698/711），`fused_err_max = 1.07e-4`，`min_gap_full = 1.83e-7`，`min_gap_warm = 2.64e-7`，`min_top1_top2_margin = 5.96e-8`，`transitions = {}`（无）。
- GEMM-sweep 与 real-GEMV sweep 一致（0/2640 分类失配）→ fp32 下该指标与实现无关。
- **Live-backend parity：** fast path 在真实 `InMemoryBackend` 里实现（纯加 +130 行，legacy 侧逐字一致），对 1.1GB pkl 跑：**0/2640 winner、0 score（>1e-6）、0 verdict** 失配；`max|score_fast - score_legacy| = 9.54e-7`（< 1e-6 门槛）；fast path 确认触发（360/360 命中，0 回退），在故意打乱 `_entries` 顺序下 `id->row` remap 仍正确（0 失配，max_err 0.0）。

**为何挺过（相关性论证）。** zscore normalizer 放大（sigma~0.0062 → ~161× z-scale × tanh' ~0.5 × 权重 0.4375 × 2 vision 字段），把 ~7.7e-7 的原始 cosine 误差放大成 ~2.86e-5 fused。这放大**对 query 内所有候选相同**（同一 query 单位向量、同一 normalizer），故每个候选的分一起平移，保住 argmax/阈值关系。这就是为何 140 个近 FULL + 83 个近 WARM = 223 个 query 落在 fp32 误差带内却无一翻转。两个最接近 FULL 的 query（q=66/67，间隙 1.83e-7）在 cosine 参考与 GEMV 下都稳定为 WARM（均 0.9976968）。

**robot_state caveat（任何下游文档都要写）。** depth_1.yaml 的 `field_similarity.robot_state.to_similarity: {exp, tau:1.0}`（44-46 行）是**死配置**：`build_field_normalizers`（per_field）建 `ZScoreNormalizer.from_params_dict(sim_type='l2')` 并忽略 `to_similarity`。生效的 normalizer 是 **zscore-on-negated-L2**（mu=-1.958，sigma=0.748），**不是** exp-on-L2。优化不动 robot_state（仍是 fp32 `torch.norm` L2），故它在每次 ref-vs-variant 比较中**抵消** —— 翻转计数与该歧义无关。

---

## 6. 范围是单点的 —— 以及 depth_1 为何安全

depth_1 下落地是对 `_compute_field_scores` 里 `torch.stack`+`F.cosine_similarity` 那一处的干净 drop-in：
- **Trajectory off：** `depth<=1` 时 `_build_trajectory_fields` 返回 `{}`（`search_strategy.py:155`），故 QuerySpec trajectory 字段为 None → 派发到单步 `_search_weighted_score_sum`。**源码已核。**
- **Memo 被绕过：** `sid/qid=None` → `_batch_field_scores` 提前返回（`:423-424`），无 per-step memo。
- **step_filter='all'：** 候选 = 整个 task bucket，作为**连续 VIEW，无 gather**。在 yaml 语料里审计：每处都是 `'all'`，**从不是 `step_range`**（一次验证数到 1293、另一次 4149；计数因 glob 不同而异，但实质论断 —— 恒为 `'all'` —— 一致）。

---

## 7. GPU 启用判据

**仅当**节点专用 / 独占于 cp1 推理时采用 GPU：
- **GO-GPU** 若 cp1 有 GPU 且无竞争性 policy 占空比 → idle **0.35–0.75ms**。
- **NO-GPU** 若 pi05 policy 共享该 GPU：30% 占空比 → p99 **20.5ms**（破 <10ms）；饱和 → median **84–126ms**。典型 GPU 服务主机上，把 cp1 留在 CPU；它只与 policy 的 CPU 侧胶水竞争。

---

## 8. 落地方案（具体）

1. **预建（在 `load_artifact` 内）**：对每个 `(checkpoint_id, task_key, field)` 建连续 fp32 矩阵 —— cosine 字段 **L2-归一化**（vision_0, vision_1），l2 字段（robot_state）存**原始** —— 外加 `id->row` dict + `row->id` list。**每字段只存它需要的那种形式**；不要同时存 normed 和 raw 两份，且**跳过 weight-0 / 禁用字段**（vision_2, prompt_emb）。live-parity 原型曾天真地两份都存（+~2.7GB）；生产构建不能这样。
2. **Fast path（替换 `:386-394` 的 stack/cosine/norm 处）**：`full0 = mat0_unit.mv(q0_unit); full1 = mat1_unit.mv(q1_unit); fullr = torch.norm(matr_raw - qr, dim=1)`；然后 `scores = full[row_idx]` 用 `id->row` remap（处理 `_filter_entries` dict 迭代顺序 ≠ 构建顺序 —— 已由打乱顺序 smoke 测验证）。再走现有 ZScoreNormalizer → 加权和 → topk。**仅 fp32。**
3. **回退（强制守卫）**：当出现以下任一时回退到现有 `_compute_field_scores`：trajectory active、`sid/qid` 已设、`rrf` fusion、`step_range` 存在、或某候选 id 不在矩阵中（回退 → None → legacy；已验证正确）。把替换范围限定 `trajectory_depth==1 && fusion==weighted_score_sum && step_filter=='all'`。
4. **内存 / 释放**：per-entry vision 张量**仅在** `write_policy: never`（depth_1.yaml `:79-80`）下可在预建后释放。释放必须发生在**所有 entry 加载完成之后、任何 insert 路径之前**，并断言 write-frozen。**理由**：`CacheStorage.insert`/`batch_insert` → `_check_entry_dims` 读 `entry.query_keys[field_name].shape`（`cache_storage.py:289`）含 vision 字段；若有 write 发生，释放这些张量会抛错。**源码已核。**
5. **线程绑定**：保留 **4 个不同物理核**（每核一个 SMT sibling，如 `taskset -c 0,2,4,6`）并 `torch.set_num_threads(4)`。用 4 不用 8（线程拐点 4–5；8 恶化尾部）。绑定必须是 **cpuset 预留**（cgroup `cpuset.cpus` + `cpuset.cpus.exclusive`），不是裸 taskset —— 见风险 R3。

---

## 9. 风险与缓解

- **R1 — 刀刃式 verdict 间隙（实测-可能，未触发）。** 到阈值的最小间隙（1.83e-7）远**小于** fp32 fused 误差（静态 2.86e-5 / live 1.07e-4）—— 三档翻转数学上可能；在 0/2640 上未发生。**缓解（任何审计/回放用途推荐）**：fp32 **top-K rescore** —— 用 fast GEMV 在全 N 上算，取 top-K（K~8–16），仅对这 K 个用精确 `F.cosine_similarity` 重算，在 rescore 后的 top-1 上判档。这使最终 FULL/WARM/MISS 与 legacy bit-一致（消除系统误差和线程抖动），代价可忽略（K 个 cosine vs N~300）。rescore 的"top-K 必含 legacy top-1"性质**未**实测验证（机器饱和）—— 采用前确认（近乎确定：max per-field cosine 误差仅 2.26e-6）。
- **R2 — 新的线程数非确定性（实测）。** 当前 `F.cosine_similarity` 跨 {1,2,4,8} 线程 bit-确定（0.0）；新 `torch.mv`/`@` 不是（fused 抖动 **7.08e-5**，cos 抖动 2.26e-6），在 40 个最紧 margin 探针上 0 argmax/verdict 翻转。**不要**把新路径在确定性层面说成"不比现状差"。**缓解**：R1 的 top-K rescore 可消除；或 pin `set_num_threads`（代价是 4 线程延迟）。若不要求 bit 可复现，有界残余在 0/2640 证据下可接受 —— 但须作为新的、小的、相关的非确定性来报告。
- **R3 — 带宽争用击穿绑定（实测）。** Pin4 + 3 个算力 co-tenant 在独占核上：p99 **6.88ms**（gate 守住）vs 同负载不绑定 **11.94ms**（gate 破）—— 绑定对算力负载有用。但 Pin4 + 4 个带宽流式 co-tenant 在独占核上：median 7.04 / p99 **17.72ms** / min 升 1.9→4.3ms —— 尽管独占核仍 **gate 破**，因为该核每 query 读 104.6MB @ 48.5 GB/s（逼近 DDR5 上限），核绑定无法切分共享内存控制器。还有：在超订下绑定 consumer 而让 producer 不绑定漫游是**有害**的（live loadavg-20：绑定 p99 61ms > 不绑定 40ms）。**缓解**：独占 cpuset 预留（把 producer 隔出预留核），不是裸 taskset；并在生产 gate 加带宽余量断言（§10）。
- **R4 — Turbo 降频（实测，次要）。** Pin4 + 12 个算力核忙：min 升 2.5→4.1ms（全核 turbo 降频）。与调度和带宽机制不同；忙主机上要预算。
- **R5 — fp16/bf16 诱惑（实测-不可接受）。** fp16 = 15/2640（0.57%）、bf16 = 194/2640（7.3%）三档 verdict 偏差。**缓解**：强制 fp32；拒绝任何降精度。
- **R6 — 蔓延到 depth≥2 / 写（实测-不安全）。** depth≥2 走 `_compute_level_scores`（`:774`），用非连续 `layer_entries`（`:702-705`）的任意祖先子集 + 历史 query（`:711`）—— per-task-bucket 矩阵不适用；且释放张量破坏 per-entry 读。**缓解**：§8.3 回退；depth≥2 用单独设计的 per-(field,query_id) [N] 向量 memo，而非本 drop-in。

---

## 10. 生产 rebench gate（签收前必过；在真实服务主机上跑）

1. cpuset 隔离 **4 个物理核**给 cp1（`cpuset.cpus` + `cpuset.cpus.exclusive`，不是裸 taskset）；`torch.set_num_threads(4)`。
2. 以**目标占空比**跑真实 `serve_policy.py` + pi05 作 co-tenant（非 idle）。
3. 驱动 **≥5000 个真实 cp1_search query**；要求**全程 p99 < 10ms 且 p99.9 < 20ms 且 max < 50ms**，在 policy loop 活跃下测。
4. 同时采样 DRAM 带宽（`pcm-memory` / `perf stat -e mem-loads`）；**断言系统总 DRAM 带宽保持 < 平台上限的 ~70%**。若 policy co-tenant 带宽密集，4 核预留撑不住 —— gate 判失败。
5. 在 policy 的**峰值**占空比（非平均）重复。

**GO** 仅当所有分位界在 live 负载下成立 **且**带宽断言通过。否则 **NO-GO → 要求专用 / 独占推理节点**（那里 idle 2.5–5ms p99 是有效数）。同时在**真实生产 CPU/BLAS 构建**上（非 WSL2 —— 见 §11）重新验证三档等价性与延迟：fp32 reduction 顺序与混合 P/E 拓扑在裸机上不同。

---

## 11. 未测 / 诚实局限

- **R2 的 idle 延迟头条在 R3 重负载下未能再复现**（box 处于 loadavg 26–28，sibling agent 所致；那里测得 ~10–16× 与 ~10–25ms 核 —— 与争用模型一致）。2.46ms idle 最优只在真正低负载窗口可确认（R2 的 pinned run；我的 [R3 复现] median 2.63ms @ loadavg ~2.8）。绝对 idle 数需一台保证独占的主机来锁定。
- **未测真实 pi05 co-tenant。** 所有争用 co-tenant 都是合成的（numpy matmul = 算力；numpy 128MB stream = 带宽）。决定性未知数 —— GPU 主机上 pi05 的 CPU 侧胶水是算力受限（预留守住 ~7ms p99）还是带宽受限（预留失败 ~18ms p99）—— 未解，且 gate 签收（§10.2）。
- **WSL2 把 i7-12700H 混合 P/E 拓扑压平**成 10C/20T SMT 视图（无 per-core 最大频率、无 E-core 区分）。绑定/降频行为须在真实 CPU 上重验；把延迟敏感路径避开 E-core。
- **泛化未测。** 等价性仅在 libero_10 `cp1_spatial_pool_16`（2640 entry，10 task，vision_dim 32768）上确立。其它 key-builder / checkpoint（cp3）/ 大得多的 N 未测。线程抖动随 D（32768，固定）而非 N 增长，故更大 N 不应恶化它 —— 是推理非测量。
- **top-K rescore 保险推荐但未实测**（"top-K=16 必含 legacy top-1" —— 机器饱和）。近乎确定但未确认。
- **depth≥2 / step_range 路径未被本落地或任何 parity 测覆盖**（设计上越界；若这些路径将来激活则是个开口）。
- **fast index 内存代价未压测。** live 原型保留 normed+raw 两份使 vision 内存翻倍（+~2.7GB）；生产构建须每字段只存一种（§8.1）。

---

## 12. 推荐

落地 **fp32-ONLY 双 GEMV CPU fast path**，作为对 `_compute_field_scores`（`in_memory_backend.py:386-394`）中 `torch.stack` + `F.cosine_similarity` 处的范围限定 drop-in 替换，限定 `trajectory_depth==1 && fusion==weighted_score_sum && step_filter=='all'`，其余一切情形（trajectory active、sid/qid 已设、rrf、step_range、或候选 id 不在矩阵中）干净回退。在 `load_artifact` 建 per-`(checkpoint_id, task_key, field)` 连续 fp32 矩阵 —— cosine normed、l2 raw、跳过 weight-0 字段、每字段一种形式 —— 带 `id->row` remap，并仅在 `write_policy:never` 下、所有加载完成后释放 per-entry vision 张量。经**独占 cpuset 预留**（非裸 taskset）绑 4 个不同物理核并 `torch.set_num_threads(4)`。**不要**用 fp16/bf16（15/194 三档翻转），**不要**迁 GPU 除非节点专用。因 fp32 fused 误差（1.07e-4）超过最小阈值 margin（1.83e-7）且新路径引入当前路径没有的线程数非确定性，若需要 bit 可复现（回放/审计/跨机 parity）则加 **fp32 top-K（K=16）rescore** 保险。生产签收以 §10 rebench 为门。

**置信度：高**（方案 + 等价性由实测含 live-backend parity 证明；开放风险是部署环境而非方案本身）。

---

## 产物（gitignore / `/tmp`；未 commit）

- `exp/cache_latency_bench/data/opt_bench/task_pack.pt`（per-task fp32 矩阵，2640 行 / 10 task）。
- `equiv_sweep_3way_result.json`（fp32=0 / fp16=15 / bf16=194）、`equiv_sweep_result.json`（FULL 边界）、`r3_threeway_result.json`（fp32 0/2640，fused_err_max 1.07e-4，223 in-band）、`r3_nondeterminism_result.json`（当前 0.0 vs 新 7.08e-5）、`r2_timing_results.json` / `r2_results.json`。
- Live-parity 原型 + 脚本在隔离 worktree `.claude/worktrees/wf_b9b2a679-c9c-2/...`（`r3_full_2640.py`、`r3_smoke.py`、`r3_edge.py`、`r3_latency_breakdown.py`）；争用 harness 在 `/tmp`（`r3_pinned_bench.py`、`r3_bwload.py`）。

**关键源码**：`in_memory_backend.py:363-403`（目标）、`:591-632`（算术）、`:423-424`（memo 绕过）；`judge.py:220-237`（三档）；`score_normalizers.py:55-65, 156-191`（orient + zscore）；`search_strategy.py:147-177`（depth gate）；`cache_storage.py:277-294`（释放张量隐患）；`depth_study/depth_1.yaml`。
