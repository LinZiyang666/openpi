> Status: Plan
> Date: 2026-04-28
> Level: L1

# verdict_factor_judge Phase 2 Layer 1 — 6-server 执行命令清单

Plan：[`logs/verdict_factor_judge_experiment_plan.log.md`](verdict_factor_judge_experiment_plan.log.md)（**注**：原 plan §4 Phase 2 是 4-yaml 跨因子 ablation；本文档对应**重设计后的 Phase 2 v2**，实验目标改为 (SR, inference_ratio) Pareto 前沿外推，分层做 within-factor descriptor / window 探索后再做组合）。

Phase 1 结果分析：[`exp/verdict_factor_judge/analysis/phase0_phase1_results.md`](../exp/verdict_factor_judge/analysis/phase0_phase1_results.md)。

---

## 实验目的

在保证 success rate 的前提下**尽量压低推理压力** = 把 (SR, inference_ratio) 拉到 baseline Pareto 前沿之外。判决优先级 FULL_HIT (0 推理) > WARM_START (部分推理) > MISS (全推理)。

Phase 1 数据已显示：

- F-FULL × T-DUAL_07 SR 0.97-1.00 但 99.5% WARM_START → 等同 always-WARM baseline，**没省推理**
- 单因子 yaml SR 0.87-0.95，inf_ratio 0.46-0.58，**全部贴在 random_periodic Pareto 前沿上下，未突破**

Phase 2 v2 必须让 judge 根据因子信号**三档决策**（hit / warm / miss）。Layer 1 找出每个因子内部哪些 descriptor / window 携带 hit-vs-miss 区分信号，Layer 2 用 Layer 1 winner 做组合 + tier 阈值 + 启发权重 sweep，Layer 3 在 winner 上做 1000 ep 大样本 stat power。

---

## 算力 / 拓扑

**6 GPU server × 1 client/server**。每 cfg 起 2 个 server（A/B），yaml 集合按 stem 字典序切 a/b 半（spec 脚本固定切法 = 13 + 13）。运行时序：

| 阶段 | yaml 数 | episode 数 | server 数 | wall-clock 估算 |
|---|---|---|---|---|
| Layer 1（within-factor 探索）| 78 (3 cfg × 26 yaml) | 7,800 | 6 (2/cfg, batch1-6) | ~2 h |
| Layer 2（待 Layer 1 分析后生成）| TBD | TBD | 6 | TBD |
| Layer 3（winner 1000 ep × 1 seed）| 9 (3 winners × 3 cfg) | 9,000 | 6 | ~1.5 h |

## 端口映射（6 server 拓扑）

S1-S3 沿用 Phase 1 frp（本地端口 + 1000 = `155.98.36.32` 公网入口）。S4-S6 在另一台机器上有**直连公网 IP `149.165.151.106`**，不走 frp，端口直接从 8001 开始。

| 批次 | server 名 | cfg | 公网入口 | yaml dir |
|---|---|---|---|---|
| **batch1** | S1-clip-A | `clip` | `155.98.36.32:8998` (frp) | `config/clip/phase2_layer1_a/` |
| **batch2** | S2-clip-B | `clip` | `155.98.36.32:8999` (frp) | `config/clip/phase2_layer1_b/` |
| **batch3** | S3-mxp-A | `max_pool` | `155.98.36.32:9000` (frp) | `config/max_pool/phase2_layer1_a/` |
| **batch4** | S4-mxp-B | `max_pool` | `149.165.151.106:8001` (直连) | `config/max_pool/phase2_layer1_b/` |
| **batch5** | S5-sp16-A | `spatial16` | `149.165.151.106:8002` (直连) | `config/spatial16/phase2_layer1_a/` |
| **batch6** | S6-sp16-B | `spatial16` | `149.165.151.106:8003` (直连) | `config/spatial16/phase2_layer1_b/` |

> **Frp 检查**（仅 S1-S3）：`~/.config/frp/frpc.toml` 沿用 Phase 1 的 8998/8999/9000 三条 tcp 映射，不需新增。
> **直连检查**（S4-S6）：客户端到 `149.165.151.106:8001-8003` 三个端口需通：
> ```bash
> for p in 8001 8002 8003; do
>   nc -zv 149.165.151.106 $p 2>&1 | head -1
> done
> ```

---

## §0 yaml 生成

```bash
# Layer 1 yaml 一次生成完毕；spec 脚本是纯笛卡尔 + 切半
uv run python -m exp.verdict_factor_judge.phase2_spec
```

期望输出：

```
Wrote 156 Phase 2 Layer 1 yamls under exp/verdict_factor_judge/config/
Per-cfg split: half_a=13 stems, half_b=13 stems
```

每 cfg 每 half = 13 eval + 13 sibling warmup = 26 yaml file。3 cfg × 2 half × 26 = **156 file 总**。

**前置 sanity**：

```bash
# yaml 数齐全
for cfg in clip max_pool spatial16; do
  for half in phase2_layer1_a phase2_layer1_b; do
    n=$(ls exp/verdict_factor_judge/config/$cfg/$half/*.yaml 2>/dev/null | wc -l)
    printf "  %s/%s: %d yaml\n" "$cfg" "$half" "$n"
  done
done
# 期望每行 26
```

```bash
# yaml 校验（schema）
uv run python -c "
from openpi.cache.config import load_cache_config, validate_cache_config
from pathlib import Path
ok = 0; n = 0
for y in Path('exp/verdict_factor_judge/config').glob('*/phase2_layer1_*/*.yaml'):
    n += 1
    cfg = load_cache_config(y); validate_cache_config(cfg); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 156/156 validated
```

**输出目录**（在 server / client 跑前都得存在）：

```bash
mkdir -p exp/verdict_factor_judge/data/phase2_layer1/{clip,max_pool,spatial16}/{per_step,episode_results,logs}
```

---

## §1 Layer 1 — 6 server / 6 client 一对一

Layer 0 (F2 500 ep 复测) **已合并到 Layer 1**：F2 单 yaml 在 inf_ratio≈0.50 处，即便锁噪声到 0.94 也只是贴 random_periodic 前沿，**不可能单独突破 Pareto**；其真值在 Layer 3 winner 1000 ep 复测时一并锁定。

每 server 一个终端，每 client 一个终端，全部 12 个终端**同时启动**。每 batch ~13 yaml × 100 ep ≈ 2 hour wall-clock。

### §1.1 服务器命令（6 server，每 server 1 终端）

服务器 bootstrap 用各自 cfg 的 **phase2 layer1_a 任一 eval yaml**（不是 phase0；phase0 yaml 的 `judge.dump.path` 父目录强校验会要求 `data/calibration` 存在，phase2 eval yaml 用 composite judge 没这个限制）。

**所有 6 个 server 必须带 `--warmup-dump-root`**：B2 wire 的 `fetch_dump` / `unload_warmup_buffer` ctrl + `dump.deferred` 解析全部依赖此 root；不加则 `run_phase.py` 第 1 步切 warmup yaml 时立刻报 `load_cache_config: server was started without --warmup-dump-root`（已踩坑）。每 server 用独立 root 后缀（`_s1` ~ `_s6`），同 host 多 server 也能跑（`.resolve()` allowlist 防 traversal），但隔离更稳。

**所有 6 个 server 都用 `CUDA_VISIBLE_DEVICES=0`** —— 服务进程只占用本机的 GPU 0；client 端的 LIBERO sim 渲染才需要分卡（前 3 client GPU 0、后 3 client GPU 1）。

#### Machine 1 — 走 frp（timan107，frp → 155.98.36.32）

##### Server S1 — clip, local port 7998

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/clip/phase2_layer1_a/clip_w7_d4_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s1 \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S2 — clip, local port 7999

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/clip/phase2_layer1_a/clip_w7_d4_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s2 \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S3 — max_pool, local port 8000

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/max_pool/phase2_layer1_a/max_pool_w3_d5_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s3 \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Machine 2 — 直连（149.165.151.106，port = 公网 port）

##### Server S4 — max_pool, port 8001

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/max_pool/phase2_layer1_a/max_pool_w3_d5_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s4 \
    --env LIBERO \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S5 — spatial16, port 8002

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer1_a/spatial16_w8_d4_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s5 \
    --env LIBERO \
    --port 8002 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S6 — spatial16, port 8003

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase2_layer1_a/spatial16_w8_d4_phase2_f1a_a_d_cum_only_t_full.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s6 \
    --env LIBERO \
    --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### §1.2 客户端命令（6 client，1:1 配对 server）

server 启动完毕后再上 client。每 client 一个终端，6 条**同时启动**。

### batch1 → S1 (clip half-A, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/clip/phase2_layer1_a \
    --host 155.98.36.32 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/clip/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/clip/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/clip/per_yaml_summary.jsonl \
    --resume
```

### batch2 → S2 (clip half-B, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/clip/phase2_layer1_b \
    --host 155.98.36.32 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/clip/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/clip/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/clip/per_yaml_summary.jsonl \
    --resume
```

### batch3 → S3 (max_pool half-A, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/max_pool/phase2_layer1_a \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/max_pool/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/max_pool/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/max_pool/per_yaml_summary.jsonl \
    --resume
```

### batch4 → S4 (max_pool half-B, 直连 149.165.151.106:8001, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/max_pool/phase2_layer1_b \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/max_pool/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/max_pool/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/max_pool/per_yaml_summary.jsonl \
    --resume
```

### batch5 → S5 (spatial16 half-A, 直连 149.165.151.106:8002, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer1_a \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/spatial16/per_yaml_summary.jsonl \
    --resume
```

### batch6 → S6 (spatial16 half-B, 直连 149.165.151.106:8003, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase \
    --phase-dir exp/verdict_factor_judge/config/spatial16/phase2_layer1_b \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir exp/verdict_factor_judge/data/phase2_layer1/spatial16/per_step \
    --episode-results-dir exp/verdict_factor_judge/data/phase2_layer1/spatial16/episode_results \
    --summary-out      exp/verdict_factor_judge/data/phase2_layer1/spatial16/per_yaml_summary.jsonl \
    --resume
```

---

## §2 数据回收 + Layer 1 分析

### 2.1 打包 + 下载

每 cfg 跑完后，**在远端**打包：

```bash
tar czf phase2_layer1_data_$(date +%Y%m%d_%H%M%S).tar.gz exp/verdict_factor_judge/data/phase2_layer1/
ls -lh phase2_layer1_data_*.tar.gz
```

下载到本地 `C:\Users\lzy66\Desktop\fsdownload\` 后告诉 Claude 分析。

### 2.2 Pareto 决策协议

Layer 1 完成后必须算 (SR, inf_ratio) 二元组对照 baseline。**单纯 SR 高不算赢**。

```python
# 每 yaml 的 inf_ratio 公式（warm_start 默认 start_t=0.7）
inf_ratio = (n_full_hit * 0.0 + n_warm_start * 0.3 + n_miss * 1.0) / n_eval_verdicts

# baseline 对照
# - random_periodic Pareto 前沿: exp/random_periodic_gate/analysis/aggregate.csv
# - always-WARM baseline: exp/warm_start/data/state_full_<cfg>.json (warm_t=0.7)
# - Phase 1 best: f_full_d_all_t_dual_07 (inf=0.30, SR=0.97-1.00) / f_f1a_t (inf=0.50, SR=0.94)
```

每 yaml 标注一个 Pareto 类别：

| 类别 | 判定 | 行动 |
|---|---|---|
| **Pareto-positive** | 在某 inf_ratio 段同时 beat random_periodic AND always-WARM | 进 Layer 2 winner 候选 |
| **Pareto-on-frontier** | 在某 inf_ratio 段 match 但不 strictly beat | 备选，看组合后是否能 push |
| **Pareto-below** | 在某 inf_ratio 段被 random_periodic dominated | 淘汰，Layer 2 不再用 |

### 2.3 Layer 1 输出预期

26 个 yaml 跨 3 cfg = 78 数据点。期望发现：

- **F1a-T**：哪个 desc 单独 SR / hit_rate 最强（Phase 1 只测了 all-4 = 0.94）
- **F1b 窗口形状**：W-SHORT / W-PAST / W-FUT / W-SYM-S 哪个 NaN 最少 + 信号最强
- **W-LONG-RISK 是否值得保留**：50%+ NaN 下 yaml 还能不能跑 / 决策合不合理
- **F1a-A close-out**：curv_radius / cum_disp 是否真信号（Phase 1 只测了 jerk / dir / all-4）

---

## §3 Layer 2 + Layer 3（待 Layer 1 后规划）

**Layer 2 yaml 不在本次生成范围内** — 它需要 Layer 1 的 winner descriptors / windows 作输入参数。Layer 1 数据回来后我们再生成 Layer 2 spec：

- **Layer 2.A (cross-factor combo, 8 yaml)**：用 Layer 1 winner 组成 4 套 tuned 因子集 × 2 tier (T-FULL / T-DUAL_07)
- **Layer 2.B (tier threshold sweep, 14 yaml)**：F-FULL-TUNED 上 sweep `full_hit_thr` × `warm_start_thr` × `start_t`
- **Layer 2.C (启发权重, 2 yaml)**：F-FULL-TUNED 加权重对照（uniform vs heuristic from Layer 1 SR）

**Layer 3 (winner 1000 ep × 1 seed, 9 run)**：从 Layer 2 选 3 个不同 inf_ratio 段的 Pareto 候选做大样本复测。

---

## §4 失败处理

- **某 yaml 跑挂**：异常单 yaml 隔离（`run_phase.py` 已捕 → log error → continue）。看 logs/ 里 traceback；用 `--resume` 重启即可，已 done 的 yaml 自动跳过。
- **某 server 挂**：那 batch 整体重启，`--resume` 安全。
- **frp 断**：`tmux capture-pane -pt run3` 看是否在 `Waiting for server at ws://...`；如是，`frpc reload` 后 wait recover。

---

## §5 数据布局

```
exp/verdict_factor_judge/
├── config/{clip,max_pool,spatial16}/
│   ├── phase1/                          # Phase 1 yaml（已跑完）
│   ├── phase2_layer1_a/                 # Layer 1 half-A，13 stem × 2 (eval+warmup) = 26 yaml
│   └── phase2_layer1_b/                 # Layer 1 half-B
└── data/phase2_layer1/{clip,max_pool,spatial16}/
    ├── per_yaml_summary.jsonl           # main summary, --resume 接续
    ├── per_step/<yaml_id>.jsonl         # 每 verdict 一行
    └── episode_results/<yaml_id>.json   # 每 episode 成功/失败
```
