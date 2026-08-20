# X9b-L — 优化 backend 的 size×延迟重测（补充实验，2026-08-19）

> **动机**：X9b 主报告 §6 的成本账出自 src 原版 backend；`exp/cache_latency_bench` 的
> R1–R4 优化栈（预建矩阵 / prenorm-dot GEMV / LEAN / batched build）**从未上生产**。
> owner 裁定：性能数据以优化后 backend 为准，用 X9b 不过滤（`all`）族 12 个 pkl 在两台机器上重测。
> **描述性实验**——不进任何 family，不改 X9b 的 SR 结论。

## 协议

- 库：X9b `all` 族 12 pkl（2 套件 × S1–S6，与 P7 评测逐字节同一批）；查询流：April 采集的
  50 h5/套件（与 5–6 月旧 bench 同源，两机同一批文件，字节校验一致）。
- runner：`opt/run_round4_pool_latency.py`（R1–R4 全栈，`components_hook` 注入，零改 src），
  每档独立进程、跑完即退；`torch.set_num_threads(4)` + **绑核**（旧 bench 无绑核，本轮两机都加：
  weilandserver `taskset -c 84-87`（有常驻邻居），WSL `taskset -c 0-3`）；median/p95。
- 检索按 `payload.task_key` 任务分桶（与 server 同路径）；**x 轴 = 单桶实际扫描条目数**（每档取
  10 任务桶均值，真值在 `results/bucket_sizes_all.json`）。
- 完整性：23/24 点 hit_rate=1.0、`lean_fallbacks=0`（LEAN 稳态路径 100% 覆盖）。

## 结果：延迟对桶大小严格线性

最小二乘 `search_ms = a + k·bucket_entries`，**全程最大残差 < 0.5 ms**：

| host / suite | 截距 a | **斜率 k** | 点数 |
|---|---:|---:|---:|
| i7-12700H (WSL) / spatial | 0.55 ms | **10.1 µs/条** | 6 |
| i7-12700H (WSL) / l10 | 0.62 ms | **9.6 µs/条** | 5 |
| Xeon E5-2696 v4 / spatial | 0.26 ms | **13.3 µs/条** | 6 |
| Xeon E5-2696 v4 / l10 | 0.40 ms | **11.8 µs/条** | 6 |

逐档中位数（`cp1_search_ms`，全段数据在 `plot_data.json` 的 `retrieval_latency_ms 下按 CPU 型号命名的标签（字段自描述：search_ms_median 等）`）：

| 档（l10 桶均值条数） | S1 (55) | S2 (109) | S3 (274) | S4 (566) | S5 (1,172) | S6 (2,649) |
|---|---:|---:|---:|---:|---:|---:|
| Xeon E5-2696 v4 | 0.80 | 1.39 | 3.69 | 7.36 | 14.71 | **31.42** |
| i7-12700H | 1.13 | 1.66 | 3.23 | 6.18 | 11.85 | OOM（见下） |

- **与旧 bench 严格吻合**：旧 bench 优化后 3.54 ms 是在 ~264 条/桶上测的 ⇒ 13.4 µs/条，
  与本轮两机 10–13 µs/条同量级同斜率。⚠ 曾有一版口径误认为旧 bench 单桶全扫 2,640 条
  （探针只查了 entry 属性、漏了 `payload.task_key`），据此推出的 "优化斜率 ~1.4 ms/千条" **作废**。
- 量级自洽（roofline）：每条目每 call 触碰 2×32,768×4B ≈ 262 KB ⇒ 10 µs/条 ≈ 26 GB/s，
  正是 4 核内存带宽的合理值——**优化后的 GEMV 已贴带宽底，斜率没有再降的空间（同精度/同核数下）**。
- 两机差异（i7-12700H 快 ~25%）为单核内存带宽差异，非测量噪声（四条拟合互相独立、残差全 <0.5 ms）。

## 对 X9b 成本结论的改写（生产化的账）

以 l10 顶档 S6（桶均值 2,649 条）计，若把优化栈接入 serving：

| | 检索分量/call | + 固定分量（Stage-1 前向+网络 ≈126 ms） | vs teacher ≈690 ms |
|---|---:|---:|---:|
| src 原版（X9b P7 争用实测） | ~2.5 s | ~2.6 s | **慢 ~3.8×** |
| **优化栈（本轮实测）** | **31 ms** | **~157 ms** | **快 ~4.4×** |

**"size 轴同时是成本轴、过盈亏平衡点纯 cache 比 teacher 还慢"的结论只在原版 backend 下成立**；
优化栈下盈亏平衡点在本实验全部 size 范围内不存在（固定分量主导，检索占比 <20%）。
X9b 主报告 §6 已加此限定。

## 局限与披露

1. **WSL 的 l10/S6 点缺失（OOM-kill，anon-rss 21.7 G 撞 23 G 顶）**——这本身是个真发现：
   优化栈把 fp16 库转常驻 fp32 矩阵，**峰值内存 ≈ 3× pkl 字节**（fp16 entries + fp32 矩阵并存于
   prebuild 期），11 G 的 S6 pkl 需要 ~33 G。生产部署内存账要按 3× fp16-pkl 估。
2. 本轮是**进程内直调**（无 server 编排、无并发）；接入 serving 后的固定分量与并发争用另测。
3. R2 等价性记录（0 verdict 翻档 + geometric safety）出自旧 fp32 库；X9b fp16 pkl 经 prebuild
   `.float()` 后走同一 fp32 GEMV，dtype 断言在位，但**等价性未在 X9b 库上重新验证**——
   本轮只主张延迟，不主张检索结果等价。
4. 单 keybuilder（`cp1_spatial_pool_16`，2×32,768-dim vision + 32-dim state）；斜率随 key 维度线性变化。

## 产物

| 产物 | 位置 |
|---|---|
| 逐档 per_step.csv + summary.json | 本目录 `relatency_data/{wsl,weilandserver}/`（+ `wsl_smoke/`；每 host 含 env 快照与 12 份 run.log） |
| 桶大小真值 | weilandserver `/data/.../results/bucket_sizes_all.json` |
| 并入绘图数据 | 本目录 `plot_data.json`（`retrieval_latency_ms` 下的 `i7-12700H` / `Xeon E5-2696 v4`） |
| 带成本面板的图 | 本目录 `size_curve_libero_10.png` / `size_curve_libero_spatial.png`（右面板双 host 曲线，WSL 的 l10 顶档为 OOM 缺口） |
