> Status: Plan
> Date: 2026-04-28
> Level: L1

# verdict_factor_judge Phase 2 Layer 2 (redesign) — 6-server / 6-client 运行教程（spatial16 only）

Plan：[`logs/verdict_factor_judge_experiment_plan.log.md`](verdict_factor_judge_experiment_plan.log.md) §Phase 2 Layer 2。
Layer 1 分析：[`exp/verdict_factor_judge/analysis/phase2_layer1_results.md`](../exp/verdict_factor_judge/analysis/phase2_layer1_results.md)。
yaml 生成器：[`exp/verdict_factor_judge/phase2_layer2_spec.py`](../exp/verdict_factor_judge/phase2_layer2_spec.py)。
inference_ratio 公式（必看）：plan §inf_ratio。

---

## 实验目标

老 Layer 2（80 cell, 2026-04 已废）失败：WeightedSum + 多 factor 在 CLT 下分数集中 0.5，旧固定 `full_hit=0.5` × `warm_start ∈ {0.20, 0.30, 0.40}` 把多 factor cells 全打成 always-WARM / always-MISS 退化解，0 Pareto 信号。

Redesign 把**阈值**升为一等轴，并对多 factor 走 AndGate / OrGate（per-factor thr，无 CLT pool）：

- **22 recipes**（继承自老 Layer 2 + 1 新 cross-window：`f1bt_lr_short_jerk`）
- **Phase A** — 单 factor jerk × `t_full / t_dual_07_w10 / t_dual_05_w10 / t_dual_03_w10` × `full_hit ∈ {0.20, 0.25, 0.30, 0.35}` = **80 cell**
- **Phase B** — 单 factor all-4 desc × T-FULL/T-DUAL × `full_hit ∈ {0.20, 0.25, 0.30, 0.35}` = **48 cell**
- **Phase C** — 多 factor **AndGate**（S2）× `hit_strict / warm_07 / warm_05 / warm_03` × `pf_thr ∈ {0.20, 0.25, 0.35}` = **60 cell**
- **Phase D** — 多 factor **OrGate**（S3）× `hit_strict / warm_07 / warm_05` × `pf_thr ∈ {0.65, 0.75, 0.85, 0.90}` = **36 cell**
- **Phase E** — Cross-window + F2-aug 单 factor × T-FULL/T-DUAL × `full_hit ∈ {0.25, 0.30}` = **16 cell**
- **240 cell × 100 ep = 24,000 ep ≈ 9-10 h wall-clock（6-server 并跑）**

每条 client 命令跑一个 batch dir（**40 yaml**），约 9-10 h。

每个 eval yaml 都带 `judge.export_factor_outputs: true`：CompositeJudge 会把 raw / norm / composer score / cold-start sentinel 注入 `__hit_meta__` → client 端 `per_step_writer` episode-batched 落盘 → 后续可 offline 重扫任意阈值，无需重跑 server。

---

## ⚠ 跑前必做：6 server 全部 git pull + 重启

新 Layer 2 依赖 `factor_outputs` 端到端通道（`src/openpi/cache/{judge,orchestrator,interceptor,components/factors/composers,config}.py` + `examples/libero/main.py` + `exp/verdict_factor_judge/per_step_log_writer.py` 改动）。这些代码**未 push 到老 server**，老 server 上跑会**静默忽略** `export_factor_outputs: true`，per_step jsonl 不会带 factor_outputs，offline 重扫无数据。

每台机器先：

```bash
cd /scratch/zixuans8/openpi  # 或对应 repo path
git pull
# 重启所有 server 进程（kill 旧 serve_policy.py + 重新起 §1 命令）
```

---

## 端口拓扑

**所有 6 server 都跑 spatial16**。

| 批次 | server | machine | 入口 | yaml dir |
|---|---|---|---|---|
| **batch1** | S1 | Mach1 (timan107, frp) | `155.98.36.32:8998` | `config/spatial16/phase2_layer2_b1/` |
| **batch2** | S2 | Mach1 (timan107, frp) | `155.98.36.32:8999` | `config/spatial16/phase2_layer2_b2/` |
| **batch3** | S3 | Mach1 (timan107, frp) | `155.98.36.32:9000` | `config/spatial16/phase2_layer2_b3/` |
| **batch4** | S4 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8001` | `config/spatial16/phase2_layer2_b4/` |
| **batch5** | S5 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8002` | `config/spatial16/phase2_layer2_b5/` |
| **batch6** | S6 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8003` | `config/spatial16/phase2_layer2_b6/` |

> Mach1 frp 沿用 8998/8999/9000；Mach2 直连用 8001/8002/8003。GPU 分配 — server 全用 `CUDA_VISIBLE_DEVICES=0`；client 端 LIBERO sim 渲染前 3 (batch1-3) GPU 0、后 3 (batch4-6) GPU 1。

---

## §0 yaml 生成 + 输出目录

```bash
uv run python -m exp.verdict_factor_judge.phase2_layer2_spec
```

期望输出：

```
Total Layer 2 cells: 240 (single cfg = spatial16_w8_d4)
Wrote 480 yaml files (eval + sibling warmup)
Per-batch counts:
  batch 1: 40 eval + 40 warmup
  batch 2: 40 eval + 40 warmup
  batch 3: 40 eval + 40 warmup
  batch 4: 40 eval + 40 warmup
  batch 5: 40 eval + 40 warmup
  batch 6: 40 eval + 40 warmup
```

每台机器跑前先建数据目录：

```bash
mkdir -p ~/openpi/exp/verdict_factor_judge/data/phase2_layer2/spatial16/{per_step,episode_results,logs}
```

Schema 验证（强烈建议跑前一次）：

```bash
uv run python -c "
from openpi.cache.config import load_cache_config, validate_cache_config
from pathlib import Path
ok = 0; n = 0
for y in Path('exp/verdict_factor_judge/config/spatial16').glob('phase2_layer2_*/*.yaml'):
    n += 1
    cfg = load_cache_config(y); validate_cache_config(cfg); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 480/480 validated
```

---

## §1 服务器命令（6 server，每 server 1 终端）

bootstrap yaml = batch1 第一个 Phase A eval yaml（`f1bt_lr_jerk_t_full_h20`，最简单的 single-factor T-FULL cell，最小 schema 风险）。**6 server 全用 `CUDA_VISIBLE_DEVICES=0`**，必须带 `--warmup-dump-root`。

### Machine 1 — frp（timan107，frp → 155.98.36.32）

#### Server S1 — local port 7998

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s1 \
    --env LIBERO --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S2 — local port 7999

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s2 \
    --env LIBERO --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S3 — local port 8000

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s3 \
    --env LIBERO --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Machine 2 — 直连（149.165.151.106）

#### Server S4 — port 8001

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s4 \
    --env LIBERO --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S5 — port 8002

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s5 \
    --env LIBERO --port 8002 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S6 — port 8003

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full_h20.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s6 \
    --env LIBERO --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## §2 客户端命令（6 client，1:1 配对 server）

server 全部启起来后，6 个 client 终端**同时启动**。每 batch 40 yaml × 100 ep ≈ 9-10 h。

### batch1 → S1 (client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1 \
    --host 155.98.36.32 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch2 → S2 (client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b2 \
    --host 155.98.36.32 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch3 → S3 (client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b3 \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch4 → S4 (直连 149.165.151.106:8001, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b4 \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch5 → S5 (直连 149.165.151.106:8002, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b5 \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch6 → S6 (直连 149.165.151.106:8003, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b6 \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer2/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl \
    --resume
```

> ⚠ 6 个 client 共写一个 `per_yaml_summary.jsonl`。`run_phase.py --resume` 模式是**同 yaml_id 跳过**，6 batch 的 yaml stem 全唯一所以不会冲突；但**多 process append 一个 jsonl 文件**有 PIPE_BUF 撕裂风险。如发现 summary 行错位，跑完后用以下命令验：
>
> ```bash
> wc -l exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl
> # 期望 240 行（240 cells）；如 < 240 说明有 yaml 失败，看 logs/
> jq -c '.yaml_id' exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl | sort -u | wc -l
> # 期望 240（无重复）
> ```
>
> 如出问题 → 把 6 个 batch 各跑各自的 summary 文件（`per_yaml_summary_b{1..6}.jsonl`），跑完 `cat` 合并。

---

## §3 数据回收

实验完成后（client 端 = 跑 client 的机器），打包：

```bash
cd $repo_root  # /scratch/zixuans8/openpi 或类似
tar czf phase2_layer2_data_$(date +%Y%m%d_%H%M%S).tar.gz \
    exp/verdict_factor_judge/data/phase2_layer2/
ls -lh phase2_layer2_data_*.tar.gz
```

下载到 Windows `C:\Users\lzy66\Desktop\fsdownload\` 后告诉 Claude 文件名分析。

> 体积估算：240 yaml × 100 ep × ~25 verdict/ep × ~12 keys avg × `factor_outputs` schema ≈ **2-3 GB compressed**（旧 layer 2 是 ~600 MB；新方案多了 factor_outputs，但 episode-batched 写盘 + jsonl 压缩比好）。

Sanity（跑后一眼）：

```bash
d=exp/verdict_factor_judge/data/phase2_layer2/spatial16
echo "  per_step:        $(ls $d/per_step/*.jsonl 2>/dev/null | wc -l)"
echo "  episode_results: $(ls $d/episode_results/*.json 2>/dev/null | wc -l)"
echo "  summary rows:    $(wc -l < $d/per_yaml_summary.jsonl 2>/dev/null)"
```

期望 240/240/240。

---

## §4 分析协议（跑完后）

1. 用 [`plot_pareto.py`](../exp/verdict_factor_judge/analysis/plot_pareto.py) 复用，但**改 `_load_phase2_layer1` 函数**指向 `phase2_layer2/spatial16/`。出 1 张 spatial16 单 cfg 大图（240 蓝点 + r/p 灰点 + always-WARM 红星），按 phase 着色（A=绿、B=蓝、C=红、D=橙、E=紫）。
2. 计算 strict-Pareto-positive：哪些 cell 不被任何 baseline dominate。
3. 阈值轴热力图：x = `full_hit` / `pf_thr`，y = recipe，颜色 = `(SR, inf)` 一维投影。看哪个 (recipe, threshold) 区间最甜。
4. 对比 4 个 phase 的 inf 分布：单 factor (A/B/E) vs AndGate (C) vs OrGate (D)，验证"AndGate / OrGate 真比 WeightedSum + 多 factor 强"假设。
5. 选 Top-3 winner（不同 inf 段：低 inf / 中 inf / 高 SR 各 1）→ 进 Layer 3 大样本（1000 ep × 1 seed）。
6. **Offline 阈值重扫**：用 `factor_outputs.norm` + composer 算法离线重算任意 (full_hit / pf_thr) 下的 hit / warm / miss 决策——无需重跑 server。脚本骨架见 `analysis/offline_rescore.py`（待写）。

成功标准：**至少 1 个 cell 在 inf < 0.5 段 SR ≥ 0.95**（突破 always-WARM@0.3 的 inf=0.65 / SR=0.94，且优于 layer1 best）。如果做不到 → 进 Phase 5 calibration / 调 percentile_rolling window_size。

---

## §5 失败处理

- **某 yaml 跑挂**：`run_phase.py` 单 yaml 隔离，log + continue，`--resume` 重启自动跳过已 done。
- **server 挂**：那 batch 整体重启 server + client，`--resume` 安全。
- **frp / 直连断**：`tmux capture-pane` 看是否在 `Waiting for server at ws://...`；frp 那边 `frpc reload`，直连那边检查 server 进程是否还在。
- **factor_outputs 缺失**：检查 server 是否 git pull + 重启过；老代码会**静默忽略** `export_factor_outputs: true`，per_step jsonl 不带 factor_outputs 字段。

---

## §6 数据布局

```
exp/verdict_factor_judge/
├── config/spatial16/
│   ├── phase2_layer1_a/                 # Layer 1 已完成（不动）
│   ├── phase2_layer1_b/                 # 同上
│   └── phase2_layer2_b{1..6}/           # Layer 2 redesign batch 切片（40 yaml/batch）
└── data/phase2_layer2/spatial16/
    ├── per_yaml_summary.jsonl           # 240 行 / cfg, 含 success_rate
    ├── per_step/<yaml_id>.jsonl         # 240 个文件，每 verdict 一行（含 factor_outputs）
    └── episode_results/<yaml_id>.json   # 240 个文件，每 episode 成功/失败
```
