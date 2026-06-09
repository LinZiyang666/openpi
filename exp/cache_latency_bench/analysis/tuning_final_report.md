# cp1 Latency 渐进调优 — 最终总报告（5 轮 + 评估）

> **Authority**: Execution（Owner 独裁令工作流 override G1/G2，每轮本会话内 spawn 审查，见 [[project_search_tuning_workflow]]）· **Type**: 优化总报告 · **Status**: In Progress（6 轮调优收官）
> **逐轮 log**: R1 [`round1_stack_elim`](../../../logs/cache_latency_bench_round1_stack_elim.log.md) · R2 [`round2_prenorm_dot`](../../../logs/cache_latency_bench_round2_prenorm_dot.log.md) · R3 [`round3_lean`](../../../logs/cache_latency_bench_round3_lean.log.md) · R4 [`round4_build`](../../../logs/cache_latency_bench_round4_build.log.md) · R5 [`round5_mem_release`](../../../logs/cache_latency_bench_round5_mem_release.log.md)
> **前置研究**: [`search_optimization_report`](../../../logs/cache_latency_bench_search_optimization_report.log.md)（R1/R2/R3 专家研究，fp32 GEMV 方案探索）

所有延迟 **ms**，libero_10 / cp1_spatial_pool_16 / 2640 step / CPU t=4 / 静默（无后台抢占）。**全程零改 src 框架**——所有优化是 `InMemoryBackend`/keybuilder 的 exp 层子类，经 `components_hook` 注入 bench。

---

## 执行摘要

cp1 端到端延迟（cp1_total median）从 baseline **35.49ms 压到 4.15ms（8.5×）**，外加释放 **1059.7MB** 常驻内存。目标（<10ms，最好 <5ms）在 **Round 2 就达成**；后续轮把 p95 拉出安全余量并关闭 build 段、回收内存。**所有改动等价**（各轮 0 verdict 翻档，R1/R4/R5 更是 bit-identical），每轮经独立审查 agent APPROVE。

| 轮 | 优化 | cp1_total median | 关键段 | 等价判据 |
|---|---|---|---|---|
| baseline | src InMemoryBackend | **35.49ms** | search 33.9 (95%) | — |
| **R1** | stack 消除 + 零拷贝（预建矩阵 + id→row） | 22.8ms | search 21.0 | bit-identical 0/2640 |
| **R2** | prenorm-dot（cosine→`mat.mv`，消除 F.cosine_similarity） | **6.27ms** ✅ | search 4.70 | 3-way 0 翻档 + geometric safety<0（fp32-ONLY） |
| **R3** | LEAN（消除 wrapper/scatter/topk 框架冗余） | 5.22ms | search 3.7, p95 7.66 | bit-identical vs R2 |
| **R4** | build 优化（batched avg_pool2d keybuilder） | 4.15ms | build 1.28→0.45 | query_key bit-identical (0.0) |
| **R5** | 内存释放（entry vision 副本 → empty(0) 哨兵） | 4.46ms* | freed 1059.7MB | hit_rate 1.0 + 守卫 fail-closed |

\* R5 latency 与 R4 噪声内（释放不碰算子）；R5 的价值是内存，非 latency。

---

## 最终 latency 画像（R4 配置，median）

| 段 | baseline | 最终 | 占比 | 说明 |
|---|---|---|---|---|
| collect | 0.006 | 0.005 | 0.1% | 可忽略 |
| gate | 0.002 | 0.003 | 0.1% | 可忽略 |
| **build** | 0.95 | **0.45** | 11% | R4 batched avg_pool2d |
| **search** | 33.9 | **3.6** | **87%** | R1-R3，已逼近带宽底 + orchestrator 包装 |
| judge | 0.013 | 0.010 | 0.2% | threshold judge 极廉价 |
| fetch | 0.005 | 0.005 | 0.1% | 可忽略 |
| **total** | **35.49** | **4.15** | 100% | |

search 现绝对主导（87%）。其 3.6ms = GEMV 带宽底（~1.7ms 中位）+ orchestrator→strategy→facade 包装（~1ms，src 层，未碰）+ normalize/argmax 残余。

---

## 各轮技术要点 + 等价

- **R1 stack 消除**：`PrebuiltMatrixBackend` 在 load 时预建 per-(ckpt,task,field) 连续 fp32 矩阵 + id→row；search 时整桶有序则**零拷贝直接用常驻矩阵**（关键修正——初版 `index_select` 仍物化 104MB，只 1.07×；零拷贝才 1.61×），否则 gather。**保留原算子** → bit-identical。micro-bench 定位 `F.cosine_similarity` 比纯点积慢 **24×**（实现低效），指向 R2。
- **R2 prenorm-dot**：prebuild 时 L2 归一化 cosine 桶 → cosine 退化成单次 `mat.mv(qn)` BLAS GEMV，消除 cosine_similarity。**非 bit 等价**（fp32 rsqrt ~2e-6）→ 用 near-threshold subset × {LOO,zero-copy} × threads{1,4,8} 验 3-way verdict 0 翻档 + **geometric safety_max=-1.83e-7<0**（数学证明：每 query 误差 < 到阈值距离 → 不可能翻档）。**fp32-ONLY**（fp16=15/bf16=194 翻档）。
- **R3 LEAN**：roofline 实测纯算子 2.55ms vs ReplayHarness search 4.70ms，差 ~2ms 是 backend 框架冗余（`_batch_field_scores` wrapper + 整桶恒等 scatter + topk + dispatch）。override `_search_weighted_score_sum` 稳态直接 GEMV→normalize→加权和→argmax。`_lean_bucket` 首次 O(n) 验证 + 缓存 + O(1) 端点检查（防 reordered 桶错位）。bit-identical vs R2。
- **R4 build**：`adaptive_avg_pool2d(16→4)` == `avg_pool2d(k=4,s=4)`（16%4==0 整除 bit-exact）+ 2 camera stack 合一次 pool/D2H。**query_key 逐 bit 一致**（max_diff=0.0），最强等价。
- **R5 内存**：快路径只读 `_mat`、不读 entry vision（是 `_mat` 字节重复）→ 释放为共享 `empty(0)` 哨兵（保留 key），fail-closed 守卫拦截 fallback。释放 1059.7MB，hit_rate 1.0。

---

## portable vs bench artifact（owner 点名要的诚实区分）

| 优化 | 生产（GPU/裸机）受益？ | 说明 |
|---|---|---|
| R1 零拷贝预建矩阵 | ✅ portable | 消除 per-query 物化，架构无关 |
| R2 prenorm-dot GEMV | ✅ portable | normalize+matmul ≫ cosine_similarity（联网佐证），BLAS/cuBLAS 都受益 |
| R3 LEAN 框架精简 | ✅ portable | 消除 backend 内 Python/算子冗余，生产同走此 backend |
| R4 lever1 fixed avgpool | ✅ portable | 去 adaptive grid 计算，双设备 bit-exact |
| R4 lever2 合并 D2H | ✅ portable（生产更关键）| 2 次 GPU→CPU transfer 合 1 次 |
| R4 lever3 非连续视图快路径 | ❌ **CPU bench artifact** | torch 2.7.1 CPU dispatch 启发式，6.79× kernel-only 大头来源，生产不靠它 |
| R5 内存释放 | ✅ portable | 生产多 replica × 1GB 是真约束 |

**绝对 ms 数是 WSL2 偏差的**（VM 拓扑压平、带宽非裸机、Windows 侧争用），相对加速比 portable。生产裸机（EPYC/E5）需 §10 rebench；R4 GPU bit-exact 需上线前 `torch.equal` 复验。

---

## Tier 2（fp16）/ Tier 3（降维）评估 —— 否决，理由

进一步压 search（87% 主导）需突破带宽底，两条路都**不值得**在已达标 4.15ms 下做：

**Tier 2 — fp16 矩阵 + fp32 top-K rescore（实测否决）**：
- fp16 读字节减半（单 vision 52→26MB）。但**实测 GEMV 仅 1.282→0.821ms（1.56×，非 2×——mv 非纯带宽）**，双 vision 省 ~0.9ms → search 3.6→~2.7ms、total ~3.5ms（~15%）。
- **fp16 翻档**：实测 fp16 vs fp32 cosine max|diff|=**3.24e-4 ≈ FULL band 3e-4** → 必翻档，**必须 top-K fp32 rescore**。
- **rescore 需 keep_raw fp32（+692MB），与 Round 5 释放 1GB 直接冲突**——二选一。
- 结论：边际 ~0.65ms total，代价 = 放弃 1GB 内存 + rescore 复杂度 + 翻档风险 + GPU fp16 行为未知。**不值得**。

**Tier 3 — 降维（PCA/随机投影，owner 暂缓）**：
- D=32768→D'=1024~4096，读字节降 8-32×，理论 search GEMV <0.2ms。
- 但**改变 cosine 语义**（近似检索）→ 必须全维 fp32 rescore 补偿 + 验检索质量/翻档；投影矩阵存储 + 投影成本。
- 且 search 压到 <2ms 后 build/orchestrator 兜底 → total 卡在 ~3ms 量级。
- 风险最高（赌检索质量），owner 已明确暂缓。

**带宽硬底（实测推算）**：fp32 全维 ~1.4ms median search。当前 3.6ms 含 ~1ms orchestrator/facade（src 层，碰它要改框架）。**算子层已基本榨干**。

---

## 生产落地建议

1. **src 落地路径**（本调优全在 exp 层验证，生产启用需一次正式 L2 改 src）：
   - search：把 R1-R3 的预建矩阵 + prenorm-dot + LEAN 落进 `InMemoryBackend`（drop-in 改 `_compute_field_scores`/`_search_weighted_score_sum`，范围守卫 `depth_1 && weighted_score_sum && step_filter=all`，其余回退），或加 backend type 进白名单（`config.py:1910` + `backend_pool.py`）。
   - build：把 R4 batched avg_pool2d 落进 `CP1SpatialPool16KeyBuilder._reduce_vision` 或加 config-selectable 变体。
   - 内存：R5 释放需 `write_policy:never` 冻结契约 + fail-closed 守卫，生产同样适用。
2. **§10 production rebench gate**（前置研究报告定义）：cpuset 隔离 4 物理核 + 真实 pi05 co-tenant，≥5000 query，要求 p99<10ms + DRAM 带宽<70% 上限。**带宽争用是唯一未解尾部风险**（co-tenant 带宽密集时核绑定切不开共享内存控制器，p99 可破 10→18ms）。
3. **GPU bit-exact 复验**：R2 fp32 GEMV / R4 avg_pool2d 在 CUDA 上需 `torch.equal` 复跑（理论 fp32 整除等价，未实测）。
4. **fp32-ONLY 铁律**：任何降精度（fp16/bf16）翻档，除非配 fp32 top-K rescore。

---

## 风险与 caveat

- **WSL2 绝对 ms 偏差**：所有 latency 是 WSL2 CPU 数，相对加速 portable、绝对值需裸机复测。
- **带宽争用**（§10）：决定性未知数——生产 pi05 co-tenant 是算力受限（核绑定守住 ~7ms p99）还是带宽受限（失败 ~18ms p99），需独占机 + 真实 co-tenant rebench。
- **泛化**：仅 libero_10 cp1_spatial_pool_16 / depth_1 / weighted_score_sum 验证；其它 keybuilder/checkpoint/depth≥2 未覆盖（设计上回退 super）。
- **R5 单向破坏**：哨兵释放不可逆，与 keep_raw/rescore 互斥（depth_1 0/2640 fp32 record 下合理）。
- **lever3 / fp16 1.56× 等是 CPU/torch 版本敏感**，升级 torch 可能变。

---

## 结论

6 轮调优把 cp1 latency 从 35.49ms 干到 **4.15ms（8.5×，目标 <5ms 达成）**、释放 **1GB** 内存，**全程零改 src + 每轮独立审查 APPROVE + 0 verdict 翻档**。算子层已基本榨干（search 87% 逼近带宽底）；进一步（fp16/降维）边际收益小且代价大，**不建议**。生产落地以 §10 rebench gate 为门，src 落地是独立的正式 L2 任务。
