> Status: Plan
> Date: 2026-05-08
> Level: L1

# verdict_phase4 Stage 2 — R2 offline 4-desc 权重 6-server 执行命令清单

Plan：[`logs/verdict_phase4_weight_sweep.log.md`](verdict_phase4_weight_sweep.log.md) (G1 APPROVED R3 / G2 APPROVED R2)。
Stage 1 结果：[`exp/verdict_factor_judge/analysis/phase4_stage1_results.md`](../exp/verdict_factor_judge/analysis/phase4_stage1_results.md)。
Predecessor 命令模板：[`logs/verdict_phase4_stage1_run_commands.log.md`](verdict_phase4_stage1_run_commands.log.md)。

---

## Stage 1 → Stage 2 决议输入

Stage 1 G_R1 winner（per-recipe argmax `SR − 0.5·inf`）：

| recipe | α\* | phase 4 SR | 通过 anchor (anchor − 2pp) |
|---|:---:|:---:|:---:|
| p1_state_fut_online_act | **1.0** | 0.96 | ✓ (0.95 anchor) |
| p2_action_fut_online_act | **1.0** | 0.97 | ✓ (0.96 anchor) |

下一轮（R2）的 alpha-star 输入：

```
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'
```

> **重要**：α\*=1.0 意味着两 recipe 的 R2 都在"纯 offline 配置"下扫 4-desc 权重 — online weight 全 0。**R3 退化**（(1−α\*)=0 让 5 个 online pattern 数学上等价）；R2 决议后**直接条件触发 R4**（跳过 R3，详见 §9）。

---

## 实验目的

Stage 2 = R2 offline 4-desc 相对权重 sweep。R1 锁定 α=1.0（offline 完全主导）后，把 8 个 offline score 的等权（α/8 each）按 9 种 desc-pattern 重新分配，每个 desc 内的两窗 (p0_f3, p0_f5) 仍等分该 desc 的份额（窗内权重留到 R4）：

| pattern | (jerk, direction, dispersion, path_length) shares |
|---|---|
| uniform     | (1, 1, 1, 1) |
| jerk-heavy  | (2, 1, 1, 1) |
| dir-heavy   | (1, 2, 1, 1) |
| disp-heavy  | (1, 1, 2, 1) |
| path-heavy  | (1, 1, 1, 2) |
| jerk-only   | (1, 0, 0, 0) |
| dir-only    | (0, 1, 0, 0) |
| disp-only   | (0, 0, 1, 0) |
| path-only   | (0, 0, 0, 1) |

每 recipe 9 pattern × 2 recipe = **18 eval cell**。Locked cell 同 stage 1：p1 (0.5, 0.5)、p2 (0.5, 0.4)。Composer warm_start_t=0.5（warm cost 0.75）。

---

## 算力 / 拓扑

**6 GPU server × 1 client/server**，沿用 stage 1 + phase3 同款端口拓扑。18 cells 按 `(recipe, pattern 子集)` 切到 6 batch，每 batch 3 cells：

| 阶段 | yaml 数 | episode 数 | server 数 | wall-clock 估算 |
|---|---|---|---|---|
| emit-eval-yamls（无 server） | 18 eval | — | — | < 30 s |
| R2 eval | 18 eval | 18×100 ≈ 1,800 ep | 6 (batch1-6) | ~12-18 min |
| **Stage 2 总** | 18 yaml | ≈ 1,800 ep | 6 | **~15-20 min** |

> Stage 2 **不需要** run-warmup —— stage 1 已经写好 `data/phase4/warmup_factor_raw/{p1, p2}.jsonl`，R2 的 emit-eval-yamls 直接读这个 cache。

按 yaml_id `off-<pattern>` 子串切 6 batch（`--cell-ids` allowlist）：

| 批次 | server | recipe（`--recipe`） | pattern 子集（`--cell-ids`） | cells | 公网入口 | client GPU |
|---|---|---|---|---:|---|---:|
| **batch1** | S1-sp16 | `p1_state_fut_online_act` | `off-uniform off-jerk-heavy off-dir-heavy` | 3 | `155.98.36.13:8998` (frp) | 0 |
| **batch2** | S2-sp16 | `p1_state_fut_online_act` | `off-disp-heavy off-path-heavy off-jerk-only` | 3 | `155.98.36.13:8999` (frp) | 0 |
| **batch3** | S3-sp16 | `p1_state_fut_online_act` | `off-dir-only off-disp-only off-path-only` | 3 | `155.98.36.13:9000` (frp) | 0 |
| **batch4** | S4-sp16 | `p2_action_fut_online_act` | `off-uniform off-jerk-heavy off-dir-heavy` | 3 | `149.165.151.106:8001` (直连) | 1 |
| **batch5** | S5-sp16 | `p2_action_fut_online_act` | `off-disp-heavy off-path-heavy off-jerk-only` | 3 | `149.165.151.106:8002` (直连) | 1 |
| **batch6** | S6-sp16 | `p2_action_fut_online_act` | `off-dir-only off-disp-only off-path-only` | 3 | `149.165.151.106:8003` (直连) | 1 |

> 子串注意：`off-jerk-heavy` ≠ `off-jerk-only`（前者末尾 `heavy` 后者末尾 `only`），其他 desc 同款不冲突；不可写成 `off-jerk` 否则同时 match heavy 与 only。
>
> 6 batch 各自独立 `--summary-out` 文件避免并发写冲突；最后在 §6 merge。

> **Frp / 直连 检查**（与 stage 1 / phase3 同款，已通过则免）：
> ```bash
> for p in 8001 8002 8003; do nc -zv 149.165.151.106 $p 2>&1 | head -1; done
> ```

---

## §0 前置条件

### §0.1 git pull（同步 thr back-fill 修复）

Stage 1 暴露的 `summary.fh_thr / ws_thr` 字段缺失 bug 已 fix（commit `0e0fb6d`）；R2 跑前必须 pull 最新代码：

```bash
cd /scratch/zixuans8/openpi
git fetch origin
git log --oneline -3 origin/Ziyang
# 期望最新: 0e0fb6d (Phase 4 runner: back-fill fh_thr/ws_thr into per-yaml summary)

git checkout Ziyang
git pull origin Ziyang
# 期望: Updating fbadea3..0e0fb6d, Fast-forward
```

### §0.2 stage 1 cache + factor_raw 检查

```bash
# warmup factor_raw 必须存在（stage 1 跑过 run-warmup）
ls -lh exp/verdict_factor_judge/data/phase4/warmup_factor_raw/
# 期望: p1_state_fut_online_act.jsonl + p2_action_fut_online_act.jsonl 各 ~185K
```

### §0.3 测试 sanity（< 10 s）

```bash
uv run pytest \
    tests/cache/components/factors/test_composer_zero_nan.py \
    tests/exp/test_phase3_threshold_solver.py \
    tests/exp/test_phase4_spec.py \
    tests/exp/test_phase4_runner.py \
    -q
# 期望: 178 passed (22 + 20 + 97 + 39, 含 stage1 暴露 bug fix 的 2 新测试)
```

### §0.4 输出目录

```bash
mkdir -p exp/verdict_factor_judge/data/phase4/r2_offline_desc/{per_step,episode_results}
```

---

## §1 emit-eval-yamls（无 server，一次性）

R2 的 emit-eval-yamls 需要 `--alpha-star`（stage 1 决议给的）。`composer_weights` 由 phase4_spec.generate_r2_weights(rid, α\*, offline_pattern) 算出，写入 18 个 cell yaml。

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 2 \
    --alpha-star "$ALPHA_STAR"
```

期望输出（18 行）：
```
[emit-eval-yamls] r2 p1_state_fut_online_act a1.0__off-uniform fh_thr=... ws_thr=... -> .../eval/spatial16_w8_d4_phase4_p1_state_fut_online_act__r2_a1.0_off-uniform.yaml
... (17 more lines)
```

**Sanity**：

```bash
n=$(ls exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r2_a*.yaml 2>/dev/null | wc -l)
printf "phase4 R2 eval yamls: %d (expect 18)\n" "$n"

# 18 yaml schema-valid
uv run python -c "
from openpi.cache.config import load_cache_config
from pathlib import Path
ok = n = 0
for y in sorted(Path('exp/verdict_factor_judge/config/spatial16/phase4/eval').glob('*__r2_a*.yaml')):
    n += 1
    cfg = load_cache_config(y); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 18/18 validated

# 抽检：jerk-only pattern 下 dir/dispersion/path_length offline weight 应全 0
uv run python -c "
import yaml
y = yaml.safe_load(open('exp/verdict_factor_judge/config/spatial16/phase4/eval/spatial16_w8_d4_phase4_p1_state_fut_online_act__r2_a1.0_off-jerk-only.yaml'))
w = y['checkpoints']['cp1']['judge']['composer']['weights']
n_jerk_off = sum(1 for k,v in w.items() if k.startswith('jerk_offline_state') and v != 0)
n_other_off = sum(1 for k,v in w.items() if 'offline' in k and not k.startswith('jerk') and v != 0)
n_online = sum(1 for k,v in w.items() if 'online' in k and v != 0)
print(f'jerk-only offline non-zero: {n_jerk_off} (expect 2: p0_f3 + p0_f5)')
print(f'non-jerk offline non-zero: {n_other_off} (expect 0)')
print(f'online non-zero: {n_online} (expect 0; α*=1.0 zeros online)')
"
```

---

## §2 6-server 启动

如果 stage 1 的 server 还在跑，**直接复用**（同款 warmup yaml + warmup-dump-root，bootstrap state 不变）。否则按 stage 1 §2.1 + §4 启 6 个 server。

如果 server 全部重启过，6 个 server 的 bootstrap 命令与 stage 1 完全相同（用任一 phase4 warmup yaml）：

```bash
# 模板（每 server 自己改 port + warmup-dump-root + cache-config）
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s<N> \
    --env LIBERO \
    --port <PORT> \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

| Server | 机器 | local port | 公网入口 | warmup-dump-root |
|---|---|---|---|---|
| S1 | timan107 (frp) | 7998 | 155.98.36.13:8998 | /tmp/openpi_warmup_phase4_s1 |
| S2 | timan107 (frp) | 7999 | 155.98.36.13:8999 | /tmp/openpi_warmup_phase4_s2 |
| S3 | timan107 (frp) | 8000 | 155.98.36.13:9000 | /tmp/openpi_warmup_phase4_s3 |
| S4 | 直连 149.165.151.106 | 8001 | 同 | /tmp/openpi_warmup_phase4_s4 |
| S5 | 直连 149.165.151.106 | 8002 | 同 | /tmp/openpi_warmup_phase4_s5 |
| S6 | 直连 149.165.151.106 | 8003 | 同 | /tmp/openpi_warmup_phase4_s6 |

---

## §3 客户端命令（6 client，1:1 配对 server）

server 全启动完毕后再上 client。每 client 一个终端，6 条**同时启动**。每 batch 独立 `--summary-out`；§6 合并。

`ALPHA_STAR` 必须传给每个 batch（用 shell 变量提前 export 一次：`export ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'`）。

### batch1 → S1 (p1, off-{uniform, jerk-heavy, dir-heavy}, frp 8998, GPU 0)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p1_state_fut_online_act \
    --cell-ids off-uniform off-jerk-heavy off-dir-heavy \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch1.jsonl
```

### batch2 → S2 (p1, off-{disp-heavy, path-heavy, jerk-only}, frp 8999, GPU 0)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p1_state_fut_online_act \
    --cell-ids off-disp-heavy off-path-heavy off-jerk-only \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch2.jsonl
```

### batch3 → S3 (p1, off-{dir-only, disp-only, path-only}, frp 9000, GPU 0)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p1_state_fut_online_act \
    --cell-ids off-dir-only off-disp-only off-path-only \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch3.jsonl
```

### batch4 → S4 (p2, off-{uniform, jerk-heavy, dir-heavy}, 直连 8001, GPU 1)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p2_action_fut_online_act \
    --cell-ids off-uniform off-jerk-heavy off-dir-heavy \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch4.jsonl
```

### batch5 → S5 (p2, off-{disp-heavy, path-heavy, jerk-only}, 直连 8002, GPU 1)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p2_action_fut_online_act \
    --cell-ids off-disp-heavy off-path-heavy off-jerk-only \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch5.jsonl
```

### batch6 → S6 (p2, off-{dir-only, disp-only, path-only}, 直连 8003, GPU 1)

```bash
ALPHA_STAR='p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 \
    --alpha-star "$ALPHA_STAR" \
    --recipe p2_action_fut_online_act \
    --cell-ids off-dir-only off-disp-only off-path-only \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r2_offline_desc/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch6.jsonl
```

---

## §4 数据回收 + R2 决策门

### §4.1 合并 6 batch summary

```bash
cat exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary_batch{1,2,3,4,5,6}.jsonl \
    > exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl

# Sanity: 18 行
wc -l exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl

# 每 (recipe, pattern) 都齐
uv run python -c "
import json, collections
counts = collections.Counter()
for line in open('exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl'):
    row = json.loads(line)
    counts[(row.get('recipe_id'), row.get('offline_pattern'))] += 1
for (rid, pat), n in sorted(counts.items()):
    print(f'{rid:30s} {pat:15s}  rows={n}')
print(f'unique (recipe, pattern) cells: {len(counts)} (expect 18)')
"

# Sanity: thr 字段已回填（stage 1 bug 修复后）
uv run python -c "
import json
n_thr_filled = 0
for line in open('exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl'):
    row = json.loads(line)
    if row.get('fh_thr') is not None and row.get('ws_thr') is not None:
        n_thr_filled += 1
print(f'rows with both fh_thr / ws_thr filled: {n_thr_filled} (expect 18)')
"
# 期望: 18（commit 0e0fb6d 修复后）
```

### §4.2 R2 决策门

```bash
uv run python -c "
from pathlib import Path
from exp.verdict_factor_judge.run_phase4 import _dump_decision_gate_table
import json
out = _dump_decision_gate_table(
    round_id=2,
    summary_path=Path('exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl'),
)
print('=== R2 winners (per-recipe argmax SR; force uniform if Δ ≤ 2pp) ===')
for rid, w in out['winners'].items():
    print(f'  {rid}: pattern={w.get(\"offline_pattern\")} SR={w[\"success_rate\"]:.3f}')
print('=== baselines (uniform) ===')
for rid, b in out['baselines'].items():
    print(f'  {rid}: uniform SR={b[\"success_rate\"]:.3f}')
print('=== continuation decisions ===')
for rid, d in out['trigger_decisions'].items():
    print(f'  {rid}: continue={d[\"continue\"]} | {d[\"reason\"]}')
print('=== next_args_suggestion ===')
print(json.dumps(out['next_args_suggestion'], indent=2))
"
```

`decision_gate.json` 自动写到 `exp/verdict_factor_judge/data/phase4/r2_offline_desc/decision_gate.json`，含：
- `winners[rid]`：argmax SR 行；若 argmax SR − uniform SR ≤ 2pp → 强制取 uniform 行（plan §3.2）
- `trigger_decisions[rid]["continue"]`：True 表示 winner > uniform + 2pp（pattern 真有用）；False 表示 force uniform
- `next_args_suggestion`：直接可粘贴的下轮 CLI（**只含 continue=True 的 recipe**）

### §4.3 出图

R2 SR 在 9 patterns × 2 recipe 上的对比（待实现 `plot_r2_patterns_phase4.py`）。临时用以下命令出粗图：

```bash
uv run python << 'EOF'
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7))
order = ["uniform", "jerk-heavy", "dir-heavy", "disp-heavy", "path-heavy",
         "jerk-only", "dir-only", "disp-only", "path-only"]
for ax, rid, anchor in [(a1, "p1_state_fut_online_act", 0.95), (a2, "p2_action_fut_online_act", 0.96)]:
    by_pat = {r["offline_pattern"]: r["success_rate"] for r in rows if r["recipe_id"] == rid}
    ys = [by_pat.get(p) for p in order]
    ax.bar(range(9), ys, color=["steelblue" if p == "uniform" else "tab:orange" for p in order])
    ax.axhline(anchor, color="gray", linestyle=":", label=f"phase3 anchor={anchor}")
    ax.set_xticks(range(9)); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("SR")
    ax.set_title(rid); ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("exp/verdict_factor_judge/analysis/phase4_stage2_r2_patterns.png", dpi=160, bbox_inches="tight")
print("saved -> phase4_stage2_r2_patterns.png")
EOF
```

### §4.4 inf 端点 follow-up（stage 1 暴露的差异）

Stage 1 发现 phase4 R1 cell 的 thr 比 phase3 同 cell 高 ~0.05（fh_thr 0.43 vs 0.38），导致 inf 偏高 0.10。R2 同样的 cell 也会受这个影响，确认是否系统性偏移：

```bash
uv run python << 'EOF'
import json
WARM_C = 0.75
def inf_of(r):
    n = r.get("n_eval_verdicts") or 0
    return (r["n_warm_start"]*WARM_C + r["n_miss"])/n if n else 0.0

# 看 R2 uniform pattern 与 stage1 α=1.0 的 thr / inf 是否一致
r2_rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl')]
r1_rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl')]
import yaml
for rid in ("p1_state_fut_online_act", "p2_action_fut_online_act"):
    r2_uni = next(r for r in r2_rows if r["recipe_id"] == rid and r["offline_pattern"] == "uniform")
    r1_a1  = next(r for r in r1_rows if r["recipe_id"] == rid and r["alpha"] == 1.0)
    # R1 thr from yaml (since stage1 summary has None)
    y1 = yaml.safe_load(open(f'exp/verdict_factor_judge/config/spatial16/phase4/eval/spatial16_w8_d4_phase4_{rid}__r1_a1.0.yaml'))
    tt1 = y1["checkpoints"]["cp1"]["judge"]["composer"]["tier_thresholds"]
    print(f'{rid}:')
    print(f'  R1 α=1.0 (yaml thr): fh={tt1["full_hit"]:.4f} ws={tt1["warm_start"]:.4f}  SR={r1_a1["success_rate"]:.3f}  inf={inf_of(r1_a1):.3f}')
    print(f'  R2 uniform (summary): fh={r2_uni["fh_thr"]:.4f} ws={r2_uni["ws_thr"]:.4f}  SR={r2_uni["success_rate"]:.3f}  inf={inf_of(r2_uni):.3f}')
    # R1 α=1 与 R2 uniform 在 weight 数学上等价：(1, 1, 1, 1) × 8 keys uniform = α/8 each
    # SR / thr / inf 应该一致（±噪声）。差异是 phase4 内部一致性 sanity。
EOF
```

> **期望**：R1 α=1.0 与 R2 uniform 在数学上等价（offline 8 keys 等权 + online 0），thr 应一致、SR 应在 100 ep 噪声内一致。若不一致，记到 `phase4_stage2_results.md` 作为 follow-up。

### §4.5 打包 + 下载

```bash
TS=$(date +%Y%m%d_%H%M%S)
tar czf phase4_stage2_${TS}.tar.gz \
    exp/verdict_factor_judge/data/phase4/r2_offline_desc/ \
    exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r2_a*.yaml \
    exp/verdict_factor_judge/analysis/phase4_stage2_r2_patterns.png
ls -lh phase4_stage2_${TS}.tar.gz
```

下载到本地后告诉 Claude 分析。

---

## §5 失败处理

- **某 batch 跑挂**：`run_phase4 --mode run-eval` 已捕异常 → log + continue 下一 cell；下次启同 batch 命令时，`_load_done_yaml_ids` 自动跳已 done 的 cell（按 `--summary-out` 文件 resume）。
- **emit-eval-yamls fail**（factor_raw 缺失）：检查 `data/phase4/warmup_factor_raw/<recipe>.jsonl` 是否存在 + 非空；不在则回 stage 1 §2 重跑 run-warmup。
- **frp 断 / 直连 timeout**：同 stage 1 §7 修法。
- **summary thr 字段仍是 None**：检查 git 是否已 pull `0e0fb6d`；若没 pull，commit `fbadea3` 之前的版本不修这个 bug。

---

## §6 数据布局（执行结束后）

```
exp/verdict_factor_judge/
├── config/spatial16/phase4/
│   └── eval/                                                             # §1 emit (18 yaml)
│       ├── spatial16_w8_d4_phase4_p1_state_fut_online_act__r2_a1.0_off-uniform.yaml
│       ├── spatial16_w8_d4_phase4_p1_state_fut_online_act__r2_a1.0_off-jerk-heavy.yaml
│       ├── ... (16 more)
│       └── spatial16_w8_d4_phase4_p2_action_fut_online_act__r2_a1.0_off-path-only.yaml
├── data/phase4/
│   └── r2_offline_desc/                                                  # §3 run-eval output
│       ├── per_step/<yaml_id>.jsonl                                      # client 写
│       ├── episode_results/<yaml_id>.json                                # client 写
│       ├── per_yaml_summary_batch{1..6}.jsonl                            # 6 batch 独立
│       ├── per_yaml_summary.jsonl                                        # §4.1 merge 后 master, 18 行
│       └── decision_gate.json                                            # §4.2 R2 决议
└── analysis/
    └── phase4_stage2_r2_patterns.png                                     # §4.3
```

---

## §9 下一步：跳过 R3，进 R4 条件触发

> **关键判断**：α\* = 1.0 让 R3 的 (1−α\*) = 0 → online 总权重 0 → 5 个 online pattern 数学上**完全等价**（全 0 weight，跑也是 5 个 identical cells）。**直接跳过 R3**。

R2 决议后（§4.2），按 plan §3.3 的 R4 触发逻辑改写为对比 R2 baseline（uniform）：

| R2 trigger_decisions[rid].continue | 含义 | 行动 |
|---|---|---|
| **True**（pattern\* SR > uniform SR + 2pp） | 该 recipe 的 desc 偏好显著 | 该 recipe **跑 R4**（W-FUT 双窗权重 5 patterns） |
| **False**（force uniform） | desc 偏好不显著 | 该 recipe **跳 R4**，phase 4 终结 |

R4 的 emit-eval-yamls 命令模板（待 R2 决议后填）：

```bash
# 例：仅 p1 触发 R4（next_args_suggestion 给的 cli_command 直接 paste）
ALPHA_STAR='p1_state_fut_online_act=1.0'                        # R2 给
OFF_PAT='p1_state_fut_online_act=jerk-heavy'                    # R2 给
ON_PAT='p1_state_fut_online_act=uniform'                        # R3 跳过 → 默认 uniform

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 4 \
    --alpha-star "$ALPHA_STAR" \
    --offline-pattern "$OFF_PAT" \
    --online-pattern "$ON_PAT" \
    --recipe p1_state_fut_online_act
# 5 cells
```

> Stage 3 (R4) run-commands 在 R2 决议明确后另出 log。

---

## §10 节点检查总单

| # | 步骤 | 命令位置 | 预期产出 |
|---:|---|---|---|
| 1 | git pull (thr fix) | §0.1 | commit `0e0fb6d` |
| 2 | factor_raw cache 检查 | §0.2 | 2 个 jsonl 各 ~185K |
| 3 | 测试 sanity | §0.3 | 178 passed |
| 4 | 输出目录 | §0.4 | mkdir 完成 |
| 5 | emit-eval-yamls | §1 | 18 R2 eval yaml |
| 6 | server S1-S6（如未启） | §2 | 6 server listen |
| 7 | 6 batch run-eval | §3 | 6 summary file |
| 8 | merge summary | §4.1 | 18 行 + thr 字段已回填 |
| 9 | R2 决策门 | §4.2 | decision_gate.json + next_args_suggestion |
| 10 | 9-pattern 对比图 | §4.3 | phase4_stage2_r2_patterns.png |
| 11 | inf 端点 follow-up | §4.4 | R1 α=1 与 R2 uniform thr 一致性 |
| 12 | 打包下载 | §4.5 | tar.gz |
