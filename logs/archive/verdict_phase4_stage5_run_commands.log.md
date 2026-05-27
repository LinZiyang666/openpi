> Status: Plan
> Date: 2026-05-08
> Level: L1

# verdict_phase4 Stage 5 — 48-cell × 500ep 真值复测 6-server 执行命令清单

Plan：[`logs/verdict_phase4_weight_sweep.log.md`](verdict_phase4_weight_sweep.log.md) (G1 APPROVED R3 / G2 APPROVED R2)。
Stage 1 / 2 结果：[`phase4_stage1_results.md`](../exp/verdict_factor_judge/analysis/phase4_stage1_results.md) · [`phase4_stage2_results.md`](../exp/verdict_factor_judge/analysis/phase4_stage2_results.md)。
Predecessor 命令模板：[`logs/verdict_phase4_stage2_run_commands.log.md`](verdict_phase4_stage2_run_commands.log.md)。

---

## 实验目的

Stage 1 + Stage 2 数据揭示一个共同问题：**100 ep × 1 seed 的 SR 噪声 ~3-5pp 让多个关键决议处于 borderline 状态**：

- p2 R2 dir-only winner（SR=0.97 vs uniform 0.92，+5pp）—— 但 R1 α=1.0（数学等价 R2 uniform）SR=0.97，意味着 R2 uniform "真值"在 0.92-0.97 之间漂动，dir-only 优势可能完全是 noise。
- phase3 g1/g10 anchor 的"真值 SR"（plan 写 g1=0.95, g10=0.96）也是 100ep × 1 seed 的单点估计。
- p1 R2 path-only 异常 SR=0.83（比相邻 desc-only 低 9-11pp），是真信号还是 noise tail？
- W-FUT 双窗权重（R4，Stage 1+2 都没跑）能否 marginal 提升？
- phase3 vs phase4 解出的 thr 差 +0.05（fh）+ +0.06（ws）—— 是 reproducible bug 还是 sample drift？

Stage 5 用 **libero_spatial 单 seed max init = 500 ep/cell** 把 SR 标准误从 ±5pp 压到 ±2.2pp，**48 cells 同时回答上述全部 5 类未决问题**。

判决 cascade（不变）：
```
s ≥ FH_thr           → FULL_HIT
WS_thr ≤ s < FH_thr  → WARM_START @ start_t=0.5
s < WS_thr           → MISS
```

---

## 算力 / 拓扑

**6 server × 8 cells/server = 48 cells**：

| 阶段 | yaml 数 | episode 数 | server 数 | wall-clock 估算 |
|---|---:|---:|---:|---|
| Stage 5 真值复测 | 48 cell | 48 × 500 = 24,000 ep | 6 (S1-S6) | **~3.3 h**（瓶颈最慢那台 server） |

每 cell 500 ep × 5 worker ≈ 25 min；每 server 8 cell × 25 min = 200 min ≈ 3.3 h。

| Group | Server | n | 子集 | 主目标 |
|---|---|---:|---|---|
| **A** | S1 (frp 8998) | 8 | phase3 g1 / g10 各 4 关键 cell | anchor + 高 SR 真值 |
| **B** | S2 (frp 8999) | 8 | phase4 p1 R2 8 patterns（去 disp-only） | desc 权重在 p1 |
| **C** | S3 (frp 9000) | 8 | phase4 p2 R2 8 patterns（去 disp-only） | desc 权重在 p2，验证 dir-only winner |
| **D** | S4 (直连 8001) | 8 | phase4 R1 α 端点+中点 (4×p1+4×p2) | α 信号 vs noise |
| **E** | S5 (直连 8002) | 8 | phase4 R4 W-FUT 双窗 (4×p1 + 4×p2) | 窗权重首次实测 |
| **F** | S6 (直连 8003) | 8 | phase3 g6 × 4 + g4/g8/g9/g11 × 1 = 8 | 退化 verify + 旁系 anchor + g6 (pure online) baseline |

---

## §0 前置条件

### §0.1 git pull（同步 stage 5 准备）

stage 5 需要 commit `6a64488`（含 phase3 runner `--cell-ids` + 10 个新 phase4 R4 yaml）：

```bash
cd /scratch/zixuans8/openpi
git fetch origin
git log --oneline -3 origin/Ziyang
# 期望最新: 6a64488

# 如有 untracked phase4 yaml 冲突, 先 mv 到 /tmp 再 pull (同 stage1/2 流程)
TS=$(date +%s)
mkdir -p /tmp/phase4_yaml_backup_$TS
mv exp/verdict_factor_judge/config/spatial16/phase4/eval/*.yaml \
   /tmp/phase4_yaml_backup_$TS/ 2>/dev/null

git pull origin Ziyang
```

### §0.2 warmup factor_raw 双源检查

stage 5 phase3 cells 用 `data/phase3/warmup/<recipe>__warmup.jsonl` cache（stage1 phase3 实验的产物）；phase4 cells 用 `data/phase4/warmup_factor_raw/{p1,p2}.jsonl` cache（phase4 stage1 的产物）。

```bash
echo "=== phase3 warmup (Group A + F 用) ==="
ls -lh exp/verdict_factor_judge/data/phase3/warmup/spatial16_w8_d4_phase3_g{1_f1b,4_f1b,6_f1a,8_f1a,9_f1b,10_f1b,11_f1a}_*__warmup.jsonl 2>/dev/null

echo "=== phase4 warmup (Group B/C/D/E + F2 部分用) ==="
ls -lh exp/verdict_factor_judge/data/phase4/warmup_factor_raw/
# 期望: p1_state_fut_online_act.jsonl + p2_action_fut_online_act.jsonl
```

### §0.3 测试 sanity（< 10 s）

```bash
uv run pytest \
    tests/exp/test_phase3_runner.py \
    tests/exp/test_phase4_runner.py \
    tests/exp/test_phase4_spec.py \
    tests/cache/components/factors/test_composer_zero_nan.py \
    tests/exp/test_phase3_threshold_solver.py \
    -q
# 期望: ~180 passed
```

### §0.4 输出目录

```bash
mkdir -p exp/verdict_factor_judge/data/phase5/{per_step,episode_results,thresholds,warmup}
```

> Stage 5 的所有 cells 的 `--summary-out` / `--per-step-log-dir` / `--episode-results-dir` 都指向 `data/phase5/`。phase3 已 cached 数据复用 `data/phase3/warmup/`（read-only），phase4 cached 数据复用 `data/phase4/warmup_factor_raw/`（read-only）。

---

## §1 yaml 准备

### §1.1 phase3 yaml（Group A + F 大部分）

stage 5 phase3 cells 用现有的 phase3 yaml（stage1 时已 commit + 跑过）。无需 emit。

### §1.2 phase4 R4 yaml（Group E）

10 个 R4 yaml 已在 commit `6a64488` 中 pre-emit + commit。**无需本地重 emit**。如果想重 emit（自检验证）：

```bash
ALPHA='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'
OFF='p1_state_fut_online_act=uniform,p2_action_fut_online_act=dir-only'
ON='p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform'
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 4 \
    --alpha-star "$ALPHA" --offline-pattern "$OFF" --online-pattern "$ON"
# 应与 origin 上 yaml 一致（同 factor_raw + 同 weights → 同 thr）
```

### §1.3 yaml 总数 sanity

```bash
# Group A 用 8 phase3 yaml
ls exp/verdict_factor_judge/config/spatial16/phase3/eval/spatial16_w8_d4_phase3_g{1_f1b,10_f1b}*__fh{0.3,0.4,0.5}_ws{0.4,0.5}.yaml | wc -l
# 期望: 8

# Group B/C R2 yaml (16 个全 patterns，stage 5 跑 16/18，跳过 p1+p2 各自 disp-only 总 2)
ls exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r2_a*.yaml | wc -l
# 期望: 18 (commit a0eb43f)

# Group D R1 yaml (14 个, stage 5 跑 8 个: 4 alpha × 2 recipe)
ls exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r1_a*.yaml | wc -l
# 期望: 14 (commit b422e55)

# Group E R4 yaml (10 个，stage 5 跑 8 个: 4 patterns × 2 recipe)
ls exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r4_a*.yaml | wc -l
# 期望: 10 (commit 6a64488)

# Group F phase3 (g6 4 cell + g4/g8/g9/g11 各 1 cell = 8)
ls exp/verdict_factor_judge/config/spatial16/phase3/eval/spatial16_w8_d4_phase3_{g6_f1a_a_d_jerk_curv_pair__fh{0.3,0.4,0.5}_ws{0.4,0.5},g8_f1a_t_d_curv_only__fh0.5_ws0.5,g11_f1a_a_d_curv_only__fh0.5_ws0.5,g4_f1b_t_w_short_d_jerk__fh0.5_ws0.5,g9_f1b_t_w_sym_s_d_all__fh0.5_ws0.5}.yaml 2>/dev/null | wc -l
# 期望: ≥ 8
```

---

## §2 启 6 server（同 stage 1/2 拓扑）

如果 stage 1/2 的 server 还在跑，**直接复用**。否则按下表重启 6 server：

| Server | 机器 | local port | 公网入口 | warmup-dump-root | bootstrap yaml |
|---|---|---:|---|---|---|
| S1 | timan107 (frp) | 7998 | 155.98.36.32:8998 | /tmp/openpi_warmup_phase5_s1 | phase3 g1 warmup |
| S2 | timan107 (frp) | 7999 | 155.98.36.32:8999 | /tmp/openpi_warmup_phase5_s2 | phase4 p1 warmup |
| S3 | timan107 (frp) | 8000 | 155.98.36.32:9000 | /tmp/openpi_warmup_phase5_s3 | phase4 p2 warmup |
| S4 | 直连 | 8001 | 149.165.151.106:8001 | /tmp/openpi_warmup_phase5_s4 | phase4 p1 warmup |
| S5 | 直连 | 8002 | 149.165.151.106:8002 | /tmp/openpi_warmup_phase5_s5 | phase4 p1 warmup |
| S6 | 直连 | 8003 | 149.165.151.106:8003 | /tmp/openpi_warmup_phase5_s6 | phase3 g6 warmup |

> server bootstrap yaml 选谁不影响 stage 5 — 实际 cell 跑前会 `load_cache_config(cell_yaml)` 切换 bundle。任一 phase3 / phase4 warmup yaml 都行。`--warmup-dump-root` 为 stage 5 没用（不 fetch_dump），但 server 启动 require 这个 flag，给个新 path 即可。

模板（每 server 自己改 port + cache-config + dump-root）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config <warmup_yaml> \
    --warmup-dump-root /tmp/openpi_warmup_phase5_s<N> \
    --env LIBERO \
    --port <PORT> \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## §3 客户端命令（6 batch × 8 cell × 500 ep）

每 batch 独立 `--summary-out` 文件避免并发写冲突；最后 §4 合并。

> **resume 语义差异（两 runner 行为不同，命令也不同）**：
>
> - **phase3 batches (batch1, batch6)** 用 `run_phase3.py`，**必须显式传 `--resume`** 否则 main 会 `summary_path.write_text("")` 把已跑数据 truncate。stage 5 首次启动时 summary 为空，`--resume` 为 no-op；中途挂掉重启时 `--resume` 跳过已 done 的 cell — safety net。**所有 phase3 batches 都带 `--resume`**。
> - **phase4 batches (batch2-5)** 用 `run_phase4.py`，**无 `--resume` flag**：runner 总是 `_load_done_yaml_ids(summary_path) if summary_path.exists() else set()`，自动 resume，不需要 flag。如果想强制重跑（fresh start），手动 `rm <summary-out>` 文件后再启动。
>
> 不要 copy-paste 时把 `--resume` 加到 phase4 命令（argparse 会报 unknown argument）；也不要从 phase3 命令删掉 `--resume`（中途挂的话会丢数据）。

### batch1 → S1 (Group A, 8 phase3 cells, frp 8998, GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g1_f1b_t_w_fut_d_all g10_f1b_a_w_fut_d_all \
    --cell-ids fh0.5_ws0.5 fh0.3_ws0.5 fh0.4_ws0.5 fh0.5_ws0.4 \
    --host 155.98.36.32 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 50 \
    --skip-warmup \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase5/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch1.jsonl \
    --resume
```

> `--eval-trials 50` 让每 cell 跑 10 task × 50 trial = **500 ep**（libero_spatial init max）。
> `--skip-warmup` 复用 phase3 stage 1 已写的 `data/phase3/warmup/*.jsonl`（必须存在）。
> `--cell-ids` 4 个 (FH, WS) 子串 × 2 recipe = 8 cells 自动选中。

### batch2 → S2 (Group B, 8 phase4 p1 R2 patterns, frp 8999, GPU 0)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p1_state_fut_online_act \
    --cell-ids off-uniform off-jerk-heavy off-dir-heavy off-disp-heavy off-path-heavy off-jerk-only off-dir-only off-path-only \
    --host 155.98.36.32 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 50 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch2.jsonl
# (no --resume: run_phase4.py auto-reads done set from summary_out file)
```

> `off-disp-only` 故意**不在** `--cell-ids` 内（disp-only 退化 verify 不在 stage 5 这次跑，可作为后续 2-cell follow-up）。

### batch3 → S3 (Group C, 8 phase4 p2 R2 patterns, frp 9000, GPU 0)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p2_action_fut_online_act \
    --cell-ids off-uniform off-jerk-heavy off-dir-heavy off-disp-heavy off-path-heavy off-jerk-only off-dir-only off-path-only \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 50 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch3.jsonl
# (no --resume: run_phase4.py auto-reads done set from summary_out file)
```

### batch4 → S4 (Group D, 8 phase4 R1 α cells, 直连 8001, GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --cell-ids a0.0 a0.4 a0.6 a1.0 \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 50 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch4.jsonl
# (no --resume: run_phase4.py auto-reads done set from summary_out file)
```

> R1 不需要 `--alpha-star`（α 是 sweep 维度）。`--cell-ids` 4 子串 × 2 recipe = 8 cells。

### batch5 → S5 (Group E, 8 phase4 R4 W-FUT 双窗 cells, 直连 8002, GPU 1)

```bash
ALPHA='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'
OFF='p1_state_fut_online_act=uniform,p2_action_fut_online_act=dir-only'
ON='p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 4 \
    --alpha-star "$ALPHA" --offline-pattern "$OFF" --online-pattern "$ON" \
    --cell-ids win-short-heavy win-long-heavy win-short-only win-long-only \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 50 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch5.jsonl
# (no --resume: run_phase4.py auto-reads done set from summary_out file)
```

> R4 `win-uniform` 排除（= R2 uniform 已在 batch2/3 跑过）。`--cell-ids` 4 win 子串 × 2 recipe = 8 cells。

### batch6 → S6 (Group F, 8 phase3 cells, 直连 8003, GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g6_f1a_a_d_jerk_curv_pair g8_f1a_t_d_curv_only g11_f1a_a_d_curv_only g4_f1b_t_w_short_d_jerk g9_f1b_t_w_sym_s_d_all \
    --cell-ids \
        g6_f1a_a_d_jerk_curv_pair__fh0.5_ws0.5 \
        g6_f1a_a_d_jerk_curv_pair__fh0.5_ws0.4 \
        g6_f1a_a_d_jerk_curv_pair__fh0.4_ws0.5 \
        g6_f1a_a_d_jerk_curv_pair__fh0.3_ws0.5 \
        g8_f1a_t_d_curv_only__fh0.5_ws0.5 \
        g11_f1a_a_d_curv_only__fh0.5_ws0.5 \
        g4_f1b_t_w_short_d_jerk__fh0.5_ws0.5 \
        g9_f1b_t_w_sym_s_d_all__fh0.5_ws0.5 \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 50 \
    --skip-warmup \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase5/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch6.jsonl \
    --resume
```

> Batch 6 全 phase3（8 cells，单一 runner）。phase4 R2 disp-only verify 暂不在 stage 5 跑（如需可在 stage 5 完成后单独跑 2 cell follow-up）。

---

## §4 数据回收 + 分析

### §4.1 合并 6 batch summary

```bash
cat exp/verdict_factor_judge/data/phase5/per_yaml_summary_batch{1,2,3,4,5,6}.jsonl \
    > exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl

wc -l exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl
# 期望: 48 行

# Sanity: 每 (phase / round) 都齐
uv run python -c "
import json
rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl')]
print(f'total rows: {len(rows)} (expect 48)')
from collections import Counter
phases = Counter()
for r in rows:
    yid = r.get('yaml_id', '')
    if 'phase3' in yid: phases['phase3'] += 1
    elif 'phase4' in yid:
        if '__r1_a' in yid: phases['phase4_r1'] += 1
        elif '__r2_a' in yid: phases['phase4_r2'] += 1
        elif '__r4_a' in yid: phases['phase4_r4'] += 1
for k, v in sorted(phases.items()):
    print(f'  {k}: {v}')
"
# 期望: phase3 16 (Group A 8 + Group F 8), phase4_r1 8 (D), phase4_r2 16 (B 8 + C 8), phase4_r4 8 (E)
# 合计 16 + 8 + 16 + 8 = 48 ✓
```

### §4.2 关键对照表（自动出报告）

待 stage 5 数据下载后由 Claude 本地处理。预期输出：

1. **anchor 真值锁定**（Group A）：
   - phase3 g1 / g10 各 4 cell 在 500ep 下的 SR + inf
   - 与 stage1 phase3 100ep 数据 +/-2.2pp 内一致 → anchor 锁定
2. **R2 dir-only winner reproducibility**（Group C）：
   - p2 R2 dir-only 500ep SR vs uniform 500ep SR
   - 若仍 +5pp 差 → 真信号；若 ≤ 2.2pp → noise driven
3. **R1 α 中点 noise vs signal**（Group D + Group B/C 含 α=1）：
   - p1/p2 R1 α 端点 + 中点 SR 在 500ep 下是否仍噪声内
4. **W-FUT 双窗权重首次实测**（Group E）：
   - p1/p2 R4 4 patterns 是否比 R2 uniform / R2 dir-only 高
5. **退化 + outlier verify**（Group F + Group B/C path-only）：
   - p1 R2 path-only outlier 0.83 是否 reproducible（Group B 含）
   - phase3 g8/g11 退化 reproducibility（Group F 含）
   - phase3 g6 (pure online) baseline 真值（Group F 4 cells）
   - g4 / g9 旁系 anchor（W-SHORT jerk / W-SYM-S all）
6. **inf 偏移 follow-up**（Group A 真值 vs Group B/C R2 uniform）：
   - phase3 g1 (0.5, 0.5) SR vs phase4 p1 R2 uniform SR：差距是 noise 还是系统性
   - 如果 phase3 SR > phase4 SR + 2pp → phase4 thr +0.05 偏移是真 bug

### §4.3 inference_ratio 公式（同前）

```python
inf_ratio = (n_full_hit * 0.0 + n_warm_start * 0.75 + n_miss * 1.0) / n_eval_verdicts
```

### §4.4 打包 + 下载

```bash
TS=$(date +%Y%m%d_%H%M%S)
tar czf phase5_${TS}.tar.gz \
    exp/verdict_factor_judge/data/phase5/
ls -lh phase5_${TS}.tar.gz
```

下载到本地 `C:\Users\lzy66\Desktop\fsdownload\` 后告诉 Claude，分析 + 写 stage 5 results.md。

---

## §5 失败处理

- **某 batch 跑挂**：`--resume` 重启时跳已 done 的 cell；phase3 用 `--resume` flag，phase4 自动从 `--summary-out` 读 done set。
- **frp 断 / 直连 timeout**：同 stage1/2 修法。
- **某 cell SR=0 或异常**：检查 `episode_results/<yaml_id>.json` 是否生成；若 LIBERO worker 跑挂会导致 `_aggregate_sr_from_episode_json` 返 None。重跑该 batch 即可。

---

## §6 数据布局（执行结束后）

```
exp/verdict_factor_judge/data/phase5/
├── per_step/<yaml_id>.jsonl                              # 48 个，每 cell 500 ep × ~30 verdict/ep
├── episode_results/<yaml_id>.json                        # 48 个
├── thresholds/                                           # phase3 cells 的 thresholds.json (cached)
├── per_yaml_summary_batch{1..6}.jsonl                    # 6 个 batch summary（每 server 一个）
└── per_yaml_summary.jsonl                                # §4.1 merge 后 48 行 master
```

---

## §7 节点检查总单

| # | 步骤 | 命令位置 | 预期产出 |
|---:|---|---|---|
| 1 | git pull | §0.1 | 同步 stage5 commit |
| 2 | warmup factor_raw 双源检查 | §0.2 | phase3 7 个 + phase4 2 个 jsonl |
| 3 | 测试 sanity | §0.3 | ~180 passed |
| 4 | 输出目录 | §0.4 | mkdir 完成 |
| 5 | yaml 总数 sanity | §1.3 | A 8 + B/C/F 各 ≥ 8 + D 8 + E 8 |
| 6 | 启 6 server | §2 | 6 server listen on 7998-8003/8001-8003 |
| 7 | 6 batch run-eval（每 server 1 batch / 8 cells） | §3 | 6 summary file |
| 8 | merge summary | §4.1 | 48 行 |
| 9 | 打包下载 | §4.4 | tar.gz |
| 10 | 本地分析 | §4.2 | stage 5 results.md |
