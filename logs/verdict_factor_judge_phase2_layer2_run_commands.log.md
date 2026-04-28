> Status: Plan
> Date: 2026-04-28
> Level: L1

# verdict_factor_judge Phase 2 Layer 2 — 6-server / 6-client 运行教程（spatial16 only）

Plan：[`logs/verdict_factor_judge_experiment_plan.log.md`](verdict_factor_judge_experiment_plan.log.md) §Phase 2 Layer 2。
Layer 1 分析：[`exp/verdict_factor_judge/analysis/phase2_layer1_results.md`](../exp/verdict_factor_judge/analysis/phase2_layer1_results.md)。
yaml 生成器：[`exp/verdict_factor_judge/phase2_layer2_spec.py`](../exp/verdict_factor_judge/phase2_layer2_spec.py)。
inference_ratio 公式（必看）：plan §inf_ratio。

---

## 实验目标

Layer 1 跨 cfg 结论 = "F1b-T × jerk × LR/FUT 最强"，但 spatial16 单 cfg 还有 4 个 all-4 desc / curv 类 winner。Layer 2 **不再多 cfg**，只跑 spatial16，把 80 cell 全压上去把它榨干。覆盖：

- 26 recipes（含 cross-cfg + spatial16 单 cfg 两套 winner archetype）
- 7 tier configs（T-FULL + T-DUAL × 3 start_t × 3 阈值组合）
- 5 weight 策略（uniform / jerk_heavy / state_dom / f1bt_3x / f1bt_only）
- 80 cell × 100 ep = **8,000 ep ≈ 3 h wall-clock（6-server 并跑）**

每条 client 命令跑一个 batch dir（13-14 yaml），约 35 min。

---

## 端口拓扑

**所有 6 server 都跑 spatial16**（与 Layer 1 不同：Layer 1 是 2 cfg/server，Layer 2 单 cfg 全员上）。

| 批次 | server | machine | 入口 | yaml dir |
|---|---|---|---|---|
| **batch1** | S1 | Mach1 (timan107, frp) | `155.98.36.13:8998` | `config/spatial16/phase2_layer2_b1/` |
| **batch2** | S2 | Mach1 (timan107, frp) | `155.98.36.13:8999` | `config/spatial16/phase2_layer2_b2/` |
| **batch3** | S3 | Mach1 (timan107, frp) | `155.98.36.13:9000` | `config/spatial16/phase2_layer2_b3/` |
| **batch4** | S4 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8001` | `config/spatial16/phase2_layer2_b4/` |
| **batch5** | S5 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8002` | `config/spatial16/phase2_layer2_b5/` |
| **batch6** | S6 | Mach2 (149.165.151.106, 直连) | `149.165.151.106:8003` | `config/spatial16/phase2_layer2_b6/` |

> Mach1 frp 沿用 Phase 2 Layer 1 的 8998/8999/9000；Mach2 直连用 8001/8002/8003。GPU 分配 — server 全用 `CUDA_VISIBLE_DEVICES=0`；client 端 LIBERO sim 渲染前 3 (batch1-3) GPU 0、后 3 (batch4-6) GPU 1。

---

## §0 yaml 生成 + 输出目录

```bash
# 生成 80 cell × 2 = 160 yaml file
uv run python -m exp.verdict_factor_judge.phase2_layer2_spec
```

期望输出：

```
Total Layer 2 cells: 80 (single cfg = spatial16_w8_d4)
Wrote 160 yaml files (eval + sibling warmup)
Per-batch counts:
  batch 1: 14 eval + 14 warmup
  batch 2: 14 eval + 14 warmup
  batch 3: 13 eval + 13 warmup
  batch 4: 13 eval + 13 warmup
  batch 5: 13 eval + 13 warmup
  batch 6: 13 eval + 13 warmup
```

每台机器跑前先建数据目录（用户 ziyang10 / zixuans8 / root 自行替换 `~`）：

```bash
mkdir -p ~/openpi/exp/verdict_factor_judge/data/phase2_layer2/spatial16/{per_step,episode_results,logs}
```

—

## §1 服务器命令（6 server，每 server 1 终端）

bootstrap yaml 用 phase2_layer2_b1 的第一个 eval yaml（任意 spatial16 phase2 yaml 都行；composite judge 没 dump.path 要求，避免 calibration dir 问题）。**6 server 全用 `CUDA_VISIBLE_DEVICES=0`**，必须带 `--warmup-dump-root`（Layer 1 踩过的坑）。

### Machine 1 — frp（timan107，frp → 155.98.36.13）

#### Server S1 — local port 7998

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1/spatial16_w8_d4_phase2_l2_f1bt_lr_jerk_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s6 \
    --env LIBERO --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## §2 客户端命令（6 client，1:1 配对 server）

server 全部启起来后，6 个 client 终端**同时启动**。

### batch1 → S1 (client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer2_b1 \
    --host 155.98.36.13 --port 8998 \
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
    --host 155.98.36.13 --port 8999 \
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
    --host 155.98.36.13 --port 9000 \
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
> # 期望 80 行（80 cells）；如 < 80 说明有 yaml 失败，看 logs/
> jq -c '.yaml_id' exp/verdict_factor_judge/data/phase2_layer2/spatial16/per_yaml_summary.jsonl | sort -u | wc -l
> # 期望 80（无重复）
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

Sanity（跑前一眼）：

```bash
d=exp/verdict_factor_judge/data/phase2_layer2/spatial16
echo "  per_step:        $(ls $d/per_step/*.jsonl 2>/dev/null | wc -l)"
echo "  episode_results: $(ls $d/episode_results/*.json 2>/dev/null | wc -l)"
echo "  summary rows:    $(wc -l < $d/per_yaml_summary.jsonl 2>/dev/null)"
```

期望 80/80/80。

---

## §4 分析协议（跑完后）

1. 用 [`plot_pareto.py`](../exp/verdict_factor_judge/analysis/plot_pareto.py) 复用，但**改 `_load_phase2_layer1` 函数**指向 `phase2_layer2/spatial16/`。出 1 张 spatial16 单 cfg 大图（80 蓝点 + r/p 灰点 + always-WARM 红星）。
2. 计算 strict-Pareto-positive：哪些 cell 不被任何 baseline dominate。
3. 选 Top-3 winner（不同 inf 段：低 inf / 中 inf / 高 SR 各 1）→ 进 Layer 3 大样本（1000 ep × 1 seed）。

成功标准：**至少 1 个 cell 在 inf < 0.6 段 SR ≥ 0.97**（突破当前 Layer 1 best 的位置）。如果做不到 → 触发兜底权重启发（plan §Layer 2.B）或直接进 Phase 5 calibration。

---

## §5 失败处理

- **某 yaml 跑挂**：`run_phase.py` 单 yaml 隔离，log + continue，`--resume` 重启自动跳过已 done。
- **server 挂**：那 batch 整体重启 server + client，`--resume` 安全。
- **frp / 直连断**：`tmux capture-pane` 看是否在 `Waiting for server at ws://...`；frp 那边 `frpc reload`，直连那边检查 server 进程是否还在。

---

## §6 数据布局

```
exp/verdict_factor_judge/
├── config/spatial16/
│   ├── phase2_layer1_a/                 # Layer 1 已完成（不动）
│   ├── phase2_layer1_b/                 # 同上
│   └── phase2_layer2_b{1..6}/           # Layer 2 batch 切片（13-14 yaml/batch）
└── data/phase2_layer2/spatial16/
    ├── per_yaml_summary.jsonl           # 80 行 / cfg, 含 success_rate
    ├── per_step/<yaml_id>.jsonl         # 80 个文件，每 verdict 一行
    └── episode_results/<yaml_id>.json   # 80 个文件，每 episode 成功/失败
```
