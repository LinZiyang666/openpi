> Status: Plan
> Date: 2026-05-09
> Level: L1

# verdict_phase5 systematic sweep — 240-cell × 100ep × 6-server execution commands

Plan：[`logs/verdict_phase5_systematic_sweep.log.md`](verdict_phase5_systematic_sweep.log.md) (G1 APPROVED R3 / G2 APPROVED R2)。
Predecessor 模板：[`logs/verdict_phase4_stage5_run_commands.log.md`](verdict_phase4_stage5_run_commands.log.md)。

---

## 0. 实验目的

Phase 4 stage1+2+5 把 **offline factor 维度**几乎扫透；**online factor 维度**（单窗口 / 多窗口 / 权重 / 多因子组合 / threshold）系统性地未扫。Phase 5 用 **5 group × 48 cell × 100ep × 1 seed = 240 cell** 一次扫完，每 group 输出 SR/inf 响应曲线 + 总 Pareto 前沿。

100ep noise floor 已在 phase4 stage5 验明 (~±4.4pp @ SR=0.95)：决议规则降级 — G1-G4 内只声明 SR top1-top2 ≥ 5pp 的为 winner，否则 inconclusive；G5 走 Pareto frontier 不选 winner。

---

## 1. 算力 / 拓扑

**6 server × 100ep × 240 cell**，按 4:4:4:3:3:3 ratio round-robin 分配 → S1=48 / S2=48 / S3=45 / S4=33 / S5=33 / S6=33（验算 48×2+45+33×3=240 ✓）：

| Server | 机器 | local port | 公网入口 | cell 数 | ETA (5min/cell × 5 worker) |
|---|---|---:|---|---:|---|
| S1 | timan107 (frp 8998) | 7998 | 155.98.36.32:8998 | **48** | ~4h |
| S2 | timan107 (frp 8999) | 7999 | 155.98.36.32:8999 | **48** | ~4h |
| S3 | timan107 (frp 9000) | 8000 | 155.98.36.32:9000 | **45** | ~3.75h |
| S4 | 直连 | 8001 | 149.165.151.106:8001 | **33** | ~2.75h |
| S5 | 直连 | 8002 | 149.165.151.106:8002 | **33** | ~2.75h |
| S6 | 直连 | 8003 | 149.165.151.106:8003 | **33** | ~2.75h |

**warmup 预算（瓶颈在 timan107，因为它持有 cell 多）**：约 148 个 phase5 own warmup yaml 一次性跑完（G1: 48, G2: 48, G3: 4, G4: 48；G5 0 — 复用 phase3 g6 + phase4 p1/p2 历史 jsonl）。每 yaml ~0.5 min × 5 worker → 单 server 顺序跑 48 个 ≈ 24 min；6 server 并跑后实际 ~30 min（timan107 的 3 个端口共享 GPU 顺序跑）。

总 wall-clock ≈ **2.5h warmup + 4h eval = 6.5h** 瓶颈。

---

## 2. §0 前置条件

### §0.1 git pull（同步 phase5 准备）

phase5 子包 + tests + analysis 在 commit `<commit hash TBD>` 全部 push：

```bash
cd /scratch/zixuans8/openpi
git fetch origin
git log --oneline -3 origin/Ziyang
# 期望最新含 phase5 commit (待 commit + push 后填 hash)

# 如有 untracked phase5 yaml 冲突, 先 mv 到 /tmp 再 pull
TS=$(date +%s)
mkdir -p /tmp/phase5_yaml_backup_$TS
mv exp/verdict_factor_judge/config/spatial16/phase5/{eval,warmup}/*.yaml \
   /tmp/phase5_yaml_backup_$TS/ 2>/dev/null

git pull origin Ziyang
```

### §0.2 sanity test（< 10 s）

```bash
uv run pytest \
    tests/exp/test_phase5_spec.py \
    tests/exp/test_phase5_runner.py \
    tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py \
    tests/exp/test_phase3_runner.py \
    tests/exp/test_phase4_runner.py \
    -q
# 期望: 154+ passed
```

### §0.3 输出目录

```bash
mkdir -p exp/verdict_factor_judge/data/phase5_systematic/{per_step,episode_results,thresholds,warmup_factor_raw}
```

> Phase 5 全部 cells 的 `--summary-out` / `--per-step-log-dir` / `--episode-results-dir` 都指向 `data/phase5_systematic/`。
> G5 cells 复用 read-only 历史 raw：phase3 g6 → `data/phase3/warmup/`，phase4 p1/p2 → `data/phase4/warmup_factor_raw/`。

---

## 3. §1 Yaml 准备（lazy — run-eval 自动按需 emit + warmup）

> **重大简化**：commit `036eb36` 起 `run-eval` 是 **lazy per-cell pipeline**：
> 每个 cell 在 eval 前会自动检查 (a) warmup factor_raw jsonl 是否存在，
> 不存在就在本 server 跑 warmup yaml + fetch dump + extract raw；
> (b) eval yaml 是否存在，不存在就 solve thresholds + write yaml。
>
> 因此**不需要先单独跑 emit-warmup-yamls / run-warmup / emit-eval-yamls 三个 mode**。每个 server 直接跑 `run-eval` 即可，server 自己处理它分配到的 cells 的 warmup + emit + eval。
>
> 已落盘的 yaml + raw 全部跳过（idempotent）。中途断了重跑命令仍能从 summary 续。
>
> 已 pre-emit（commit `036eb36` 已 push 云端）：148 warmup yaml + 48 G5 eval yaml；剩余 192 G1-G4 eval yaml 由各 server 在跑自己 cell 时 lazy emit。

如需单独提前 emit / run warmup（debug 用）：

```bash
# Optional debug: emit all 148 warmup yamls (no server)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode emit-warmup-yamls

# Optional debug: pre-run all warmups on one server (server required)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-warmup \
    --host 155.98.36.32 --port 9000 ...
```

但**正常流程跳过 §1 整段**，直接进 §2 / §4 / §5。

---

## 4. §2 启 6 server

如 stage5 server 还在跑可直接复用（phase5 用同一拓扑，端口与 frpc 配置不变）。否则：

| Server | 机器 | local port | 公网入口 | warmup-dump-root | bootstrap yaml |
|---|---|---:|---|---|---|
| S1 | timan107 (frp) | 7998 | 155.98.36.32:8998 | /tmp/openpi_warmup_phase5_s1 | 任意 phase5 warmup yaml |
| S2 | timan107 (frp) | 7999 | 155.98.36.32:8999 | /tmp/openpi_warmup_phase5_s2 | 任意 phase5 warmup yaml |
| S3 | timan107 (frp) | 8000 | 155.98.36.32:9000 | /tmp/openpi_warmup_phase5_s3 | 任意 phase5 warmup yaml |
| S4 | 直连 | 8001 | 149.165.151.106:8001 | /tmp/openpi_warmup_phase5_s4 | 任意 phase5 warmup yaml |
| S5 | 直连 | 8002 | 149.165.151.106:8002 | /tmp/openpi_warmup_phase5_s5 | 任意 phase5 warmup yaml |
| S6 | 直连 | 8003 | 149.165.151.106:8003 | /tmp/openpi_warmup_phase5_s6 | 任意 phase5 warmup yaml |

模板（每 server 改 port + dump-root；bootstrap yaml 选谁不影响 cell 跑前会重 load）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase5/warmup/<任一>.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase5_s<N> \
    --env LIBERO \
    --port <PORT> \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## 5. §3 ~~跑 warmup~~（已并入 run-eval lazy pipeline，正常流程跳过）

> 现在每 server 跑 `run-eval` 时会按需自动 warmup 自己分配的 cells（per-cell lazy）。
> 想 debug 单独跑 warmup 见 §1 末尾的 optional 块。

预算变更：每 server warmup 时间分摊到自己 cell 数；瓶颈不再是单 server 串跑全 148。

| Server | cells | est. warmup ep | est. warmup time | est. eval time | total wall-clock |
|---|---:|---:|---:|---:|---:|
| S1 | 48 (G1-4 cells) | ~960 ep | ~1.5h | ~4h | ~5.5h |
| S2 | 48 | ~960 | ~1.5h | ~4h | ~5.5h |
| S3 | 45 | ~900 | ~1.4h | ~3.8h | ~5.2h |
| S4 | 33 | ~660 | ~1h | ~2.75h | ~3.75h |
| S5 | 33 | ~660 | ~1h | ~2.75h | ~3.75h |
| S6 | 33 | ~660 | ~1h | ~2.75h | ~3.75h |

> G5 cell 不触发 warmup (复用历史 raw)，所以 S1-S6 中含 G5 cell 的 server warmup 时间会更短。
> 6 server 并行 → 瓶颈 S1/S2 ≈ 5.5h wall-clock total。

---

## 6. §4 准备 per-server cell-id list（6 个 file）

cell 在 6 server 上的分配是 round-robin（4:4:4:3:3:3 比例），需要把每 server 的 yaml_id 列表导出。在任一 client 节点跑：

```bash
mkdir -p /tmp/phase5_cells

uv run python -c "
from exp.verdict_factor_judge.phase5.spec import generate_all_cells, allocate_to_servers
alloc = allocate_to_servers(generate_all_cells())
import os
for srv, cells in alloc.items():
    path = f'/tmp/phase5_cells/{srv}.txt'
    with open(path, 'w') as f:
        f.write(' '.join(c.yaml_id for c in cells))
    print(f'{srv}: {len(cells)} cells -> {path}')
"
# 期望:
#   S1: 48 cells -> /tmp/phase5_cells/S1.txt
#   S2: 48 cells -> /tmp/phase5_cells/S2.txt
#   S3: 45 cells -> /tmp/phase5_cells/S3.txt
#   S4: 33 cells -> /tmp/phase5_cells/S4.txt
#   S5: 33 cells -> /tmp/phase5_cells/S5.txt
#   S6: 33 cells -> /tmp/phase5_cells/S6.txt
```

把这 6 个 .txt 也 scp / rsync 到运行 cmd 的对应 server（或 source-of-truth 节点上集中跑）。

---

## 7. §5 客户端命令（6 server × per-server cell-id list × 100ep）

每 batch 独立 `--summary-out` 文件避免并发写冲突；最后 §6 合并。

> **resume 语义**：`run_phase5.runner.py` 总是从 `--summary-out` 读 done set 自动 resume，**没有 `--resume` flag**（同 phase4 runner）。中途挂掉重启 = 重跑相同命令；要 fresh start 先 `rm` summary 文件。

### batch1 → S1 (frp 8998, GPU 0, 48 cells)

```bash
CELLS_S1="$(cat /tmp/phase5_cells/S1.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S1 \
    --host 155.98.36.32 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch1.jsonl
```

### batch2 → S2 (frp 8999, GPU 0, 48 cells)

```bash
CELLS_S2="$(cat /tmp/phase5_cells/S2.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S2 \
    --host 155.98.36.32 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch2.jsonl
```

### batch3 → S3 (frp 9000, GPU 0, 45 cells)

```bash
CELLS_S3="$(cat /tmp/phase5_cells/S3.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S3 \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch3.jsonl
```

### batch4 → S4 (直连 8001, GPU 1, 33 cells)

```bash
CELLS_S4="$(cat /tmp/phase5_cells/S4.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S4 \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch4.jsonl
```

### batch5 → S5 (直连 8002, GPU 1, 33 cells)

```bash
CELLS_S5="$(cat /tmp/phase5_cells/S5.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S5 \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch5.jsonl
```

### batch6 → S6 (直连 8003, GPU 1, 33 cells)

```bash
CELLS_S6="$(cat /tmp/phase5_cells/S6.txt)"

uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval \
    --cell-ids $CELLS_S6 \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase5_systematic/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase5_systematic/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir      exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir  exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase5/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch6.jsonl
```

---

## 8. §6 数据回收 + 分析

### §6.1 合并 6 batch summary

```bash
cat exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary_batch{1,2,3,4,5,6}.jsonl \
    > exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl

wc -l exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl
# 期望: 240
```

### §6.2 合并后跑 decision gate

`run-eval` 单 batch 跑完会写自己 batch 的 summary，但 5 个 per-group decision file 需要在 merge 后基于全 240 行跑一次：

```bash
uv run python -c "
from pathlib import Path
from exp.verdict_factor_judge.phase5.runner import _dump_decision_gate_table_phase5
gate = _dump_decision_gate_table_phase5(
    Path('exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl')
)
print({k: len(v) if isinstance(v, dict) else 0 for k, v in gate.items()})
"
# 期望: {'g1': 8 buckets, 'g2': 4, 'g3': 4, 'g4': 4, 'g5': 3 recipes}
ls exp/verdict_factor_judge/data/phase5_systematic/g{1,2,3,4,5}_decision.json
# 期望: 5 个文件全在
```

### §6.3 分组 sanity（合 240 行）

```bash
uv run python -c "
import json
rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl')]
print(f'total rows: {len(rows)} (expect 240)')
from collections import Counter
print('per-group:', Counter(r.get('group') for r in rows))
# expect: g1=48, g2=48, g3=48, g4=48, g5=48
"
```

### §6.4 绘图

```bash
# Pareto 散点 + 5 group 染色 + gold-circle Pareto positives
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.phase5.plot_pareto

# G1 (window × channel) + G5 (FH × WS) heatmaps
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.phase5.plot_heatmaps
```

输出：
- `exp/verdict_factor_judge/analysis/phase5/pareto.png`
- `exp/verdict_factor_judge/analysis/phase5/heatmaps.png`

### §6.5 打包下载

```bash
TS=$(date +%Y%m%d_%H%M%S)
tar czf phase5_systematic_${TS}.tar.gz \
    exp/verdict_factor_judge/data/phase5_systematic/
ls -lh phase5_systematic_${TS}.tar.gz
```

下载到本地 `C:\Users\lzy66\Desktop\fsdownload\` 后告诉 Claude 写 `exp/verdict_factor_judge/analysis/phase5/results.md`。

---

## 9. §7 失败处理

- **某 batch 跑挂**：直接重跑相同命令，runner 自动从 `--summary-out` 读 done 集跳过已完成 cells。
- **frp 断 / 直连 timeout**：同 stage1/2/5 修法。
- **某 cell SR=null 或异常**：检查 `episode_results/<yaml_id>.json` 是否生成；LIBERO worker 跑挂会让 `_aggregate_sr_from_episode_json` 返 None。重跑该 batch 即可（runner 跳过 done cells，只重跑未 done 的）。
- **emit-eval-yamls skip**：通常因为某 G5 cell 的 phase3/phase4 历史 raw 没在指定 dir。补 `--phase3-warmup-dir` / `--phase4-warmup-raw-dir` 到正确位置后重跑该 mode。

---

## 10. §8 数据布局（执行结束后）

```
exp/verdict_factor_judge/data/phase5_systematic/
├── per_step/<yaml_id>.jsonl                     # 240 个，每 cell 100 ep × ~30 verdict/ep
├── episode_results/<yaml_id>.json               # 240 个
├── warmup_factor_raw/<warmup_yaml_id>.jsonl     # 148 个 phase5 own warmup raw
├── per_yaml_summary_batch{1..6}.jsonl           # 6 个 batch summary（每 server 一个）
├── per_yaml_summary.jsonl                       # §6.1 merge 后 240 行 master
├── g1_decision.json                             # 8 buckets per §4.1 / 5pp 阈值
├── g2_decision.json                             # 4 buckets per §4.2
├── g3_decision.json                             # 4 buckets per §4.3
├── g4_decision.json                             # 4 buckets per §4.4
└── g5_decision.json                             # 3 recipes per §4.5 (Pareto frontier)

exp/verdict_factor_judge/config/spatial16/phase5/
├── warmup/<warmup_yaml_id>.yaml                 # 148 个
└── eval/<yaml_id>.yaml                          # 240 个

exp/verdict_factor_judge/analysis/phase5/
├── pareto.png                                   # §6.4 出图
├── heatmaps.png                                 # §6.4 G1+G5 heatmaps
└── results.md                                   # 数据回本地后追写
```

---

## 11. §9 节点检查总单（lazy run-eval pipeline）

| # | 步骤 | 命令位置 | 预期产出 |
|---:|---|---|---|
| 1 | git pull | §0.1 | 同步 phase5 commit (含 196 pre-emit yaml) |
| 2 | sanity test | §0.2 | 100+ passed |
| 3 | mkdir 输出目录 | §0.3 | 4 子目录建好 |
| 4 | 6 server bootstrap | §2 | 6 server listen on 8998-9000 / 8001-8003 |
| 5 | dump per-server cell list | §4 | 6 个 .txt files |
| 6 | 6 batch run-eval（自动 warmup+emit+eval）| §5 (batch1..6) | 6 个 batch summary + lazy-emit 192 G1-G4 eval yaml + 148 raw jsonl |
| 7 | merge summary | §6.1 | 240 行 master |
| 8 | dump decision gate | §6.2 | 5 个 g{N}_decision.json |
| 9 | 分组 sanity | §6.3 | per-group counts |
| 10 | 绘图 | §6.4 | pareto.png + heatmaps.png |
| 11 | 打包下载 | §6.5 | tar.gz |
| 12 | 本地分析 + results.md | (out-of-runbook) | results.md |
