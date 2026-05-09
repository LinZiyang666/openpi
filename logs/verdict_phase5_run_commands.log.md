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

## 3. §1 Yaml 准备（一次性，单 server 即可）

### §1.1 emit warmup yamls（148 个 phase5 own warmup）

任一 client 节点（不一定要 timan107，本地 WSL2 也可）：

```bash
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode emit-warmup-yamls \
    --warmup-yaml-dir exp/verdict_factor_judge/config/spatial16/phase5/warmup
# 期望: "[emit-warmup-yamls] 148 warmup yamls -> ..."
```

### §1.2 emit eval yamls（240 个）

> ⚠️ emit-eval-yamls **必须在 warmup factor_raw 落盘之后**：solver 会读 raw jsonl 算 thresholds。如果 G5 raw 不在 phase3/phase4 的现有路径，后面会 skip 这些 cell 的 yaml emit。

实际两步：先 §3.1 跑完 warmup，所有 phase5 own raw 落盘后，再回这里 emit eval。

```bash
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode emit-eval-yamls \
    --warmup-jsonl-dir exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw \
    --phase3-warmup-dir exp/verdict_factor_judge/data/phase3/warmup \
    --phase4-warmup-raw-dir exp/verdict_factor_judge/data/phase4/warmup_factor_raw \
    --eval-yaml-dir exp/verdict_factor_judge/config/spatial16/phase5/eval
# 期望: "[emit-eval-yamls] 240 ok, 0 skipped -> ..."
```

### §1.3 yaml sanity

```bash
ls exp/verdict_factor_judge/config/spatial16/phase5/warmup/*.yaml | wc -l
# 期望: 148

ls exp/verdict_factor_judge/config/spatial16/phase5/eval/*.yaml | wc -l
# 期望: 240
```

### §1.4 commit + push（pre-emit yamls）

让云端 server pull 时直接拿到 emit 好的 yamls，避免 server 端要 re-emit：

```bash
git add exp/verdict_factor_judge/config/spatial16/phase5/{warmup,eval}/*.yaml
git commit -m "phase5 systematic sweep: pre-emit 148 warmup + 240 eval yamls"
git push origin Ziyang
```

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

## 5. §3 跑 warmup（148 个 phase5 own warmup yamls）

> 注意：G5 cell **不**触发新 warmup（复用 phase3 g6 + phase4 p1/p2 历史 jsonl，0 新 warmup）；G3 共享 warmup（4 个 yaml 给 12×2×2=48 个 cells）；G1/G2/G4 各 cell 独立 warmup。

简化策略：**所有 warmup 由 timan107 (S3 端口 9000) 串行跑**（最快，不用切 server）。runner 在 phase5 own raw jsonl 已存在时自动 skip，所以可以重跑无副作用。

```bash
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-warmup \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --warmup-yaml-dir       exp/verdict_factor_judge/config/spatial16/phase5/warmup \
    --warmup-jsonl-dir      exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw
# 期望: 148 warmup yamls 顺序跑完，每个 ~0.5 min × 5 worker，总 ~25 min。
# G5 不在此运行（数据已在 phase3/phase4 dir）。
```

跑完后 `data/phase5_systematic/warmup_factor_raw/` 含 148 个 jsonl。

跑完后回到 §1.2 emit eval yamls（必须在 warmup raw 全落盘后才能 solver 求 thresholds）。

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

## 11. §9 节点检查总单

| # | 步骤 | 命令位置 | 预期产出 |
|---:|---|---|---|
| 1 | git pull | §0.1 | 同步 phase5 commit |
| 2 | sanity test | §0.2 | 154+ passed |
| 3 | mkdir 输出目录 | §0.3 | 4 子目录建好 |
| 4 | emit warmup yamls | §1.1 | 148 个 yaml |
| 5 | 6 server bootstrap | §2 | 6 server listen on 8998-9000 / 8001-8003 |
| 6 | run warmup（148 个）| §3 | 148 个 raw jsonl |
| 7 | emit eval yamls | §1.2 (post-§3) | 240 个 yaml + 0 skipped |
| 8 | yaml commit + push | §1.4 | 云端 pull 直接拿 |
| 9 | dump per-server cell list | §4 | 6 个 .txt files |
| 10 | 6 batch run-eval | §5 (batch1..6) | 6 个 batch summary |
| 11 | merge summary | §6.1 | 240 行 master |
| 12 | dump decision gate | §6.2 | 5 个 g{N}_decision.json |
| 13 | 分组 sanity | §6.3 | per-group counts |
| 14 | 绘图 | §6.4 | pareto.png + heatmaps.png |
| 15 | 打包下载 | §6.5 | tar.gz |
| 16 | 本地分析 + results.md | (out-of-runbook) | results.md |
