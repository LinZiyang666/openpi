# Latency Bench — cache 系统各执行体的 batch=1 推理成本

> 状态：**完成**（2026-08-19，weilandserver 独占空闲 4090）。
> 起因：`executor_substitution` 报告 §6 的延迟表被判定不可用（见 §7），本实验重新测量并把口径钉死。
> 原始数据：`../data/*.json`（逐次耗时、GPU 状态、provenance 全部落盘）。
> 表格再生成：`uv run exp/ablation_study/latency_bench/analysis/summarize.py`。

## 1. 结论

1. **两个学生模型与 teacher 在 eager 下全部是 launch-bound，不是算力受限**。batch=1 时 GPU 利用率只有 6–15%、功耗不到额定的四分之一；`torch.compile` 不改变任何计算量，却把三个模型的延迟都压到约三分之一到九分之一，GPU 利用率随之升到 36–94%。
2. **参数量不预测延迟**。eager 下 450M 的 SmolVLA（540.71 ms）比 ~3B 的 pi0.5（448.26 ms）**慢 21%**；编译到 CUDA-Graph 后两者才回到同一量级（60.64 vs 67.48 ms）。
3. **cache 命中步的固定成本（Stage1 建 key）随编译大幅缩小**：63.06 → 10.26 ms。按此重算，命中步换 ACT 在任何编译档下都省 71–82%；换 SmolVLA 在任何档下都**不省**（−5% 到 −35%）。
4. **拆 stage 要花钱，但 CUDA Graph 能把这笔钱抹平——而生产恰好用不上 CUDA Graph**。同一编译模式下，"一团"比"三段"在 default 档快 **32%**（140.56 vs 206.67 ms wall），在 CUDA-Graph 档持平（72.04 vs 72.51 ms）。生产的 interceptor 主动降级到 `max-autotune-no-cudagraphs`（§6），落在前一种情形 ⇒ **cache 系统为了在 Stage1 之后插手拿 key，实际付着约 32% 的推理开销，且这笔钱技术上可以省掉**。
5. CPU 上瓶颈翻转：pi0.5 在 Xeon 上要 **20.4 秒**，且大头是 Stage2 的 LLM 前向（16.3 s），而不是 GPU 上占 78% 的 Stage3 去噪 —— CPU 上是真的算不动，GPU 上是喂不饱。

## 2. 方法

**硬件**：weilandserver — RTX 4090 48 GB（测量期间独占、无其他进程）、Intel Xeon E5-2696 v4（88 线程 @2.2 GHz，governor `schedutil`）。
**软件**：torch 2.7.1+cu126；学生走 lerobot 0.3.3（`~/lerobot_venv`），teacher 走 openpi venv。
**权重**：全部为真实冻结 checkpoint —— ACT = `libero_spatial/act_selected/task_0` @020000（单任务模型，延迟与 ensemble 驻留无关：路由只是一次字典查找）；SmolVLA = `libero_spatial/smolvla/checkpoints/020000`；teacher = `pi05_libero_pytorch`。
**输入**：合成观测（uint8 HWC 双视角 224×224 + 8 维 state + libero_spatial task_0 的真实 prompt）。延迟只取决于张量形状与图结构，与像素/权重数值无关；SmolVLA 的 `pad_language_to: longest` 使 prompt 长度进入负载，故用真实 prompt。
**计时契约**：
- 学生 = 生产 sidecar 的 `policy_fn` 本体（脚本直接 import `sidecar_server.make_act_policy / make_smolvla_policy`），与 `sidecar_server.py:92-96` 的 `forward_ms` 逐行同构：obs dict → lerobot batch → `predict_action_chunk` → `.cpu().numpy()`。
- teacher = 生产 staged 路径 `Policy.infer`（`policy.py:101-131`，每个 stage 后 `cuda.synchronize()`），另用 `exp/rl_router/microbench_cost.py` 的 CUDA-event 口径交叉核对。
- 每次采样前后 `cuda.synchronize()`；预热丢弃（编译档预热含编译时间）。
- **每个格子一个独立进程，串行执行，测量期间该卡无任何其他负载**。

**复现**（在各自 venv、各自 host 上逐条运行，勿并行）：

```bash
# 学生 eager
lerobot_venv/bin/python bench_students.py --policy act     --device cuda --warmup 20 --iters 200 --out data/act_cuda_4090.json
lerobot_venv/bin/python bench_students.py --policy smolvla --device cuda --warmup 10 --iters 100 --out data/smolvla_cuda_4090.json
# 学生 编译档（CUDA Graph 需 --mark-step，见 §5）
lerobot_venv/bin/python bench_students_compile.py --policy act     --mode reduce-overhead --out data/act_cuda_compile_ro.json
lerobot_venv/bin/python bench_students_compile.py --policy smolvla --mode reduce-overhead --mark-step --out data/smolvla_cuda_compile_ro_markstep.json
# teacher 三段（cache 系统的拆分方式）
openpi/.venv/bin/python bench_teacher.py --mode reduce-overhead --mark-step \
    --out data/pi05_compile_ro_3stage.json
# teacher 一团（整图）：--no-builtin-compile 必加，否则会在内置 max-autotune 之上嵌套编译
openpi/.venv/bin/python bench_teacher.py --mode reduce-overhead --mark-step --fused \
    --no-builtin-compile --out data/pi05_fused_clean_ro.json
```

## 3. 学生模型延迟矩阵

| 模型 | 参数 | 设备 | 构建 | 中位 (ms) | p90 (ms) | GPU 利用率 |
|---|---|---|---|---|---|---|
| ACT | 51.6M | RTX 4090 | eager | 17.74 | 18.21 | 14% |
| ACT | 51.6M | RTX 4090 | compile default | 10.16 | 10.44 | 20% |
| ACT | 51.6M | RTX 4090 | **compile CUDA-Graph** | **4.91** | 5.05 | 36% |
| ACT | 51.6M | Xeon CPU | eager | 55.90 | 56.48 | — |
| SmolVLA | 450M | RTX 4090 | eager | 540.71 | 553.16 | 6% |
| SmolVLA | 450M | RTX 4090 | compile default | 178.00 | 178.90 | 22% |
| SmolVLA | 450M | RTX 4090 | **compile CUDA-Graph** | **60.64** | 61.14 | 50% |
| SmolVLA | 450M | Xeon CPU | eager | 1 813.36 | 1 961.53 | — |

ACT 的参数量为加载真权重后实测 **51.55M**（`executor_substitution` plan §2 记的 "~80M" 偏大）；SmolVLA 实测 **450.05M**，与 plan 一致。

## 4. Teacher（pi0.5）延迟与分阶段

`wall` 含输入/输出 transform；`model total` 为 `stage_timing.total_ms`。

| 构建 | wall (ms) | model total (ms) | Stage1 | Stage2 | Stage3 | GPU 利用率 |
|---|---|---|---|---|---|---|
| eager（三段） | 453.45 | 448.26 | 63.06 | 35.27 | 349.81 | 15% |
| compile default（三段） | 206.67 | 200.85 | 47.55 | 30.36 | 122.81 | 36% |
| **compile CUDA-Graph（三段）** | **72.51** | **67.48** | **10.26** | 27.69 | 29.57 | **93%** |
| compile default（一团） | **140.56** | 135.30 | — | — | — | 50% |
| **compile CUDA-Graph（一团）** | **72.04** | n/a（见下） | — | — | — | 94% |
| eager，CPU（三段） | 20 427.26 | 20 418.88 | 3 194.80 | 16 291.78 | 937.69 | — |

- Stage1 = 视觉编码 + token 准备（**每个 cache 命中步都必须付**）；Stage2 = LLM prefix KV；Stage3 = action expert 的 10 步去噪。
- eager 交叉核对：`microbench_cost.py` 的 CUDA-event 口径给出 415.45 ms（p90 429.21，n=50），与 staged wall 453.45 ms 的差额来自 transform 与逐阶段同步点，量级自洽。
- **一团两档必须关掉内置编译才有效**（`--no-builtin-compile`，见 §6）；日志会打印 `sample_actions compiled = False` 作为凭据。首轮未关时测得的 188.28 / 72.11 ms 是"在已编译函数上再套一层"的产物，已作废并从 `data/` 删除（凭据：污染档编译耗时 231 s，干净档只要 39.6 s）。
- **一团 + CUDA-Graph 的 `model total` 不可用**：该路径下 `Policy.infer` 内部计时只测到"提交计算图"（kernel 仍异步），录得 1.75 ms，是测量伪影；该档唯一有效的数字是带同步的 wall 72.04 ms。
- CPU 档 n=10（每次 20 秒），其余 GPU 档 n=30–200。

## 5. 瓶颈定位

**eager 下的 SmolVLA 分解**（`diag_smolvla_breakdown.py`，n=10）：总 554.02 ms = VLM prefix 一次 49.31 ms + 10 步去噪 × 46.68 ms（占 84%）。KV cache 复用正确（prefix 只跑一次），所以慢的不是重复计算，而是每步都要重新发射 16 层专家的大量小 kernel。同期 GPU 利用率 6%、功耗 91.6 W/450 W、SM 2520/3105 MHz。ACT 同样：连续推理下利用率仅 14%、87 W（`diag_act_gpu_state.py`）。

**编译的两层收益**（以 SmolVLA 为例）：`default` 只做算子融合与去 Python 开销 → 3.0×；`reduce-overhead` 再叠加 CUDA Graph，把整段 kernel 序列录一次、之后一次提交重放 → 累计 8.9×，功耗 92 → 207 W。功耗翻倍而计算量不变，是"此前 GPU 在等 CPU 喂活"最直白的证据。

**CUDA Graph 的使用约束（实测踩到）**：SmolVLA 直接用 `reduce-overhead` 会抛 `accessing tensor output of CUDAGraphs that has been overwritten`（`smolvlm_with_expert.py:51 apply_rope` 的输出被下次重放覆盖）。解法是每次调用前 `torch.compiler.cudagraph_mark_step_begin()`（本实验采用）或在图外 clone 输出。生产 sidecar 若要启用，必须一并处理这条，且注意变长 prompt 会触发重新捕获。

**CPU 上瓶颈翻转**：pi0.5 在 Xeon 上 Stage2（3B backbone 单次前向）占 16.29 s / 20.42 s，Stage3 去噪只要 0.94 s —— 与 GPU 上 Stage3 占 78% 完全相反。GPU 上是发射受限，CPU 上是算力受限。

## 6. 一团编译 vs 三段编译，以及 openpi 的内置编译

### 6.1 源码事实（本实验期间发现，影响口径定义）

- `pi0_config.py:35` — `pytorch_compile_mode` **默认 `"max-autotune"`**；`pi0_pytorch.py:234-235` 在模型构造时就把 `sample_actions` 编译掉。因此**"未编译的一团"并不存在**：任何走 `sample_actions` 的测量都会触发一次编译。
- `Policy.infer` 的 staged 分支（`policy.py:101-131`）调用的是 `_stage1_token_prep` / `_stage2_llm_backbone` / `_stage3_action_expert` 三个私有方法，**绕过 `sample_actions`，因而不受内置编译影响**。本实验三段各档的手动编译作用在这三个方法上，与内置编译互不干扰，数据有效。
- `interceptor.py:304-340` — cache 系统自己编译 `run_stage1/2/3`（`_stageN_*` 的薄封装），并把 `max-autotune` **主动降级为 `max-autotune-no-cudagraphs`**，源码注释理由是避免 CUDAGraph 输出复用错误。**生产 cache 因此拿不到 CUDA Graph。**

`sample_actions` 的函数体正是 `_stage1_token_prep → _stage2_llm_backbone → _stage3_action_expert`（`pi0_pytorch.py:764-773`），两种编法计算完全相同，差别只在 stage 边界的 graph break 与中间结果落地。

### 6.2 对比结果

两侧使用完全相同的编译模式；一团侧以 `--no-builtin-compile` 清掉 `pytorch_compile_mode`，避免嵌套编译。

| 编译档 | 三段 (wall) | 一团 (wall) | 一团优势 | GPU 利用率 (三段→一团) |
|---|---|---|---|---|
| compile default | 206.67 ms | **140.56 ms** | **+66.10 ms（32.0%）** | 36% → 50% |
| compile CUDA-Graph | 72.51 ms | **72.04 ms** | +0.47 ms（0.6%，噪声内） | 93% → 94% |

**读法**：段边界的开销是真实的（default 档 32%），但 CUDA Graph 把整串 kernel 录成一张图重放后，边界开销消失，两者持平。结合 §6.1 —— 生产的 interceptor 因为降级而拿不到 CUDA Graph，**cache 系统当前实际承担着约 32% 的拆分开销**。该开销并非不可避免：本实验在 SmolVLA 上用 `torch.compiler.cudagraph_mark_step_begin()` 正是绕过了同一类 CUDAGraph 输出复用错误（§5），同样的手法适用于 stage 拆分。

### 6.3 关于 max-autotune 的成本（未纳入本表的原因）

尝试直接测量生产口径（`max-autotune-no-cudagraphs`）时，编译在 20 分钟后仍未完成，实测特征：主进程单核满载（20 秒墙钟消耗 19.6 秒 CPU），而 `compile_threads=32` 起的 33 个 worker 合计只占 2.5% CPU，autotune benchmark 累计仅 5.75 秒（10 组，日志标记 `SingleProcess`）。**时间几乎全花在单线程的 dynamo tracing + inductor lowering 上，多核帮不上忙**；Stage3 的 10 步去噪循环在编译期被完全展开，图规模因此放大约十倍。

对系统的直接含义：生产 server 每次冷启动都要付这笔编译账（`fx_graph_cache` 命中时才快）。若改为只编译 `denoise_step` 单步、把 10 步循环留在 Python 侧，图规模降一个数量级，运行时收益基本不变——这是一条独立于本报告结论的优化建议。

## 7. 派生量：路由步 vs 教师整步

命中步的真实成本 = Stage1 + 学生前向（建 key 必须跑视觉编码器）。teacher 整步取同档 `model total`。

| 档位 | Stage1 | +ACT | +SmolVLA | teacher 整步 | ACT 节省 | SmolVLA 节省 |
|---|---|---|---|---|---|---|
| eager | 63.06 | 80.79 | 603.77 | 448.26 | **+82%** | **−35%** |
| compile default | 47.55 | 57.71 | 225.55 | 200.85 | **+71%** | **−12%** |
| compile CUDA-Graph | 10.26 | 15.17 | 70.90 | 67.48 | **+78%** | **−5%** |

**ACT 的延迟优势在三个档位下都成立且稳定（71–82%）；SmolVLA 在三个档位下都是负收益** —— 把教师整步换成"Stage1 + SmolVLA"永远更慢。

## 8. 与 `executor_substitution` 报告 §6 的偏差

原表在 4090 上测于与 pi0.5 server 共卡的生产环境，`queue_ms` 中位 0.00 只排除了 sidecar 自身排队，排不掉 GPU 层面的争用。

| 量（RTX 4090） | 原报告 | 本实验（独占空闲卡，eager） | 偏差 |
|---|---|---|---|
| Stage1 | 114.6 ms | 63.06 ms | 原值偏高 1.8× |
| Stage2+3 | 575.4 ms | 385.07 ms | 原值偏高 1.5× |
| teacher 整步 | ~690 ms | 448.26 ms | 原值偏高 1.5× |
| ACT sidecar | 39.33 ms | 17.74 ms | 原值偏高 2.2× |
| SmolVLA sidecar | 412.27 ms | 540.71 ms | 原值偏**低** 1.3× |

原始 sidecar 日志自身也支持"争用"解释：`act7012_lat_sidecar.jsonl`（n=523）的 min 为 19.94 ms、首批调用 21–23 ms，却有 39.33 ms 的中位数 —— 同一进程内快慢差一倍。

**需要更正的原结论**：原报告称 4090 上 hit→SmolVLA 省 24% 延迟、只有 H200 上才倒挂（+36%）。按本实验，4090 上同样倒挂（eager −35%，编译后 −5%）。原报告"SmolVLA 前向不随更快硬件缩放"的解读也不成立 —— 那是 host 侧发射开销的差异，不是显卡差异；编译后 SmolVLA 立刻降到 60.64 ms 即为反证。

## 9. 与公开数字的核对

| 模型 | 公开报道 | 本实验 |
|---|---|---|
| π0.5 @ 4090 | 93.6 ms（TurboVLA Table 1，batch=1）；94.7 ms（第三方）；29.2 ms（Dexmal Triton 定制 kernel，2 views） | eager 448.26 → CUDA-Graph **67.48** |
| π0 @ 4090 | 84.2 ms；baseline 102.3 ms → torch.compile 35.2 ms（2.90×） | — |
| SmolVLA | 203.1 ms（TurboVLA，同表）；101–129 ms（第三方）；官方论文只给任务级时间 | eager 540.71 → CUDA-Graph **60.64** |
| ACT | 45.34 ± 1.03 ms（MSACT）；58 ms @A100；88.1 ± 23.4 ms（多相机大配置） | eager 17.74 → CUDA-Graph **4.91** |

eager 档与公开数字差 4–5 倍，正是"公开数字来自编译/优化后的实现，而我们此前跑的是 eager 路径"；编译后本实验的数字落进甚至优于公开区间。公开数据彼此口径不一（相机数 1/2/3、chunk 10/50/63、是否 compile/Triton、硬件不同），**不可与本表逐格互比**，仅作量级校验。

## 10. 不做的主张

- 本实验测的是**孤立的模型前向成本**，不含检索、gate、判决、websocket 往返、client 侧环境步进。系统端到端延迟必须另测。
- 编译档的数字要求推理形状固定；变长 prompt / 多任务 ACT ensemble 切换在 CUDA Graph 下的行为未测。
- 未测 replay（cache 命中回放）路径本身的开销，故 §7 的表只比较"教师整步 vs Stage1+学生"，不含 cache 基线臂。
- 编译收益不改变任何成功率结论；`executor_substitution` 的 SR 结果不受本实验影响。
- **eager 三段（448.26 ms）不代表生产**：生产带 cache 时走 interceptor 的编译版三段（`max-autotune-no-cudagraphs`）。该口径的实测值因编译成本过高未取得（§6.3），但它必然落在 default 200.85 与 CUDA-Graph 67.48 之间且靠近 default 端——因为它有 autotune 而无 CUDA Graph。§7 的派生表按档位分别给出，不依赖这一未测点。
- 学生模型不受内置编译影响：lerobot 的 ACT / SmolVLA 没有等价机制，eager 档即为 sidecar 现状。

## 11. 产物

```
exp/ablation_study/latency_bench/
├── __init__.py
├── bench_students.py              # ACT / SmolVLA，eager，生产 policy_fn 口径
├── bench_students_compile.py      # 学生 eager vs compile（--mode / --mark-step）
├── bench_teacher.py               # pi0.5 staged，三段或 --fused，含 CPU 档
├── diag_smolvla_breakdown.py      # prefix / 每步去噪 分解 + GPU 状态
├── diag_act_gpu_state.py          # ACT 连续推理下的 GPU 利用率采样
├── data/*.json                    # 15 个结果（逐次耗时 + provenance）
│                                  # 一团档以 pi05_fused_clean_*.json 为准；
│                                  # 首轮嵌套编译的两个文件已作废删除
└── analysis/
    ├── analysis.md                # 本报告
    └── summarize.py               # 从 data/ 再生成全部表格
```
