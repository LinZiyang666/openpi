> Status: Plan
> Date: 2026-05-08
> Level: L1

# verdict_phase4 Stage 1 — R1 α 扫描 6-server 执行命令清单

Plan：[`logs/verdict_phase4_weight_sweep.log.md`](verdict_phase4_weight_sweep.log.md) (G1 APPROVED R3 / G2 APPROVED R2)。
Predecessor 实验运行教程（同款拓扑参考）：[`logs/verdict_phase3_run_commands.log.md`](verdict_phase3_run_commands.log.md)。

---

## 实验目的

Phase 4 把 phase3 winner **g1+g6** 与 **g10+g6** 融合成两个 10-score recipe（**p1**：offline_state W-FUT 4 desc + online_action W-K3 2 desc；**p2**：offline_action W-FUT 4 desc + online_action W-K3 2 desc），在 per-recipe 锁定的 ultra-cheap (FH, WS) 上**只扫权重向量**。

**Stage 1 = R1 α 扫描**：

| 维度 | 值 |
|---|---|
| 可调参数 | α = offline 总权重占比 ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}（7 点） |
| 内部分配 | offline 8 score 各占 α/8；online 2 score 各占 (1-α)/2 |
| Locked cell | p1 → (FH=0.5, WS=0.5)，p2 → (FH=0.5, WS=0.4) |
| Composer | `weighted_sum_zero_nan` **真加权和**（phase4 改造：`Σ w_k · contrib_k / Σ w_k`），warm_start_t = 0.5（warm cost 0.75） |
| 端点 sanity | α=0 → 退化为 g6；α=1 → 退化为 g1 / g10。两个端点必须复现 phase3 同 cell SR ±3pp |

每 recipe 7 α × 2 recipe = **14 eval cell** + 2 warmup = 总 16 个 yaml 跑。

判决 cascade（与 phase3 一致，inclusive）：
```
s ≥ FH_thr           → FULL_HIT
WS_thr ≤ s < FH_thr  → WARM_START @ start_t=0.5
s < WS_thr           → MISS
```

---

## 算力 / 拓扑

**6 GPU server × 1 client/server**，沿用 phase3 同款端口拓扑。14 个 R1 cell 按 `(recipe, α 子集)` 切到 6 batch：

| 阶段 | yaml 数 | episode 数 | server 数 | wall-clock 估算 |
|---|---|---|---|---|
| warmup（一次） | 2 warmup | 2×20 = 40 ep | 1（任一 server） | ~5-10 min |
| R1 eval | 14 eval | 14×100 ≈ 1,400 ep | 6 (batch1-6) | ~12-18 min |
| **Stage 1 总** | 16 yaml | ≈ 1,440 ep | 6 | **~25 min** |

按 yaml_id α 子串切 6 batch（`--cell-ids` 子串过滤，1-OR 语义）：

| 批次 | server | recipe（`--recipe`） | α 子集（`--cell-ids`） | cells | 公网入口 | client GPU |
|---|---|---|---|---:|---|---:|
| **batch1** | S1-sp16 | `p1_state_fut_online_act` | `a0.0 a0.2` | 2 | `155.98.36.13:8998` (frp) | 0 |
| **batch2** | S2-sp16 | `p1_state_fut_online_act` | `a0.4 a0.5` | 2 | `155.98.36.13:8999` (frp) | 0 |
| **batch3** | S3-sp16 | `p1_state_fut_online_act` | `a0.6 a0.8 a1.0` | 3 | `155.98.36.13:9000` (frp) | 0 |
| **batch4** | S4-sp16 | `p2_action_fut_online_act` | `a0.0 a0.2` | 2 | `149.165.151.106:8001` (直连) | 1 |
| **batch5** | S5-sp16 | `p2_action_fut_online_act` | `a0.4 a0.5` | 2 | `149.165.151.106:8002` (直连) | 1 |
| **batch6** | S6-sp16 | `p2_action_fut_online_act` | `a0.6 a0.8 a1.0` | 3 | `149.165.151.106:8003` (直连) | 1 |

> `--cell-ids` 是 phase4 runner 的 yaml_id 子串 allowlist (commit `fbadea3`)：cell 通过当 `any(sub in cell.yaml_id for sub in cell_ids)` 时保留。R1 cell yaml_id 形如 `spatial16_w8_d4_phase4_<recipe>__r1_a<alpha>`，所以 `a0.0 a0.2` 子串恰好 match α=0.0 与 α=0.2 两个 cell。
>
> **6 batch 各自独立 `--summary-out`** 避免并发写冲突（同 phase3 风格）。最后在 §5 merge。

> **Frp 检查**（仅 S1-S3）：`~/.config/frp/frpc.toml` 沿用 8998/8999/9000，与 phase3 完全相同，不需新增。
>
> **直连检查**（S4-S6）：
> ```bash
> for p in 8001 8002 8003; do
>   nc -zv 149.165.151.106 $p 2>&1 | head -1
> done
> ```

---

## §0 前置条件（一次性，必须先验证）

### §0.1 pkl key check

Phase 4 复用 phase3 已 enrich 的 pkl（含 16 个 W-FUT offline key：`{jerk, direction, dispersion, path_length}_offline_{state, action}__{p0_f3, p0_f5}`）。**不需要重新 enrich**。

```bash
uv run python -c "
import pickle
d = pickle.load(open('exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl','rb'))
keys = set(d['entries'][0].payload.factors.keys())
expected = {f'{desc}_offline_{ch}__p{p}_f{f}'
            for desc in ('jerk','direction','dispersion','path_length')
            for ch in ('action','state')
            for (p,f) in [(0,3),(0,5)]}
missing = expected - keys
print(f'phase4 needed offline keys: 16; missing: {missing or \"none\"}')
print(f'total keys: {len(keys)}  (expect 232 from phase3 enrich)')
"
# 期望:
#   phase4 needed offline keys: 16; missing: none
#   total keys: 232
```

> 若 missing 非 none，先按 [`verdict_phase3_run_commands.log.md §0`](verdict_phase3_run_commands.log.md#§0-pkl-enrich) 的 enrich 流程跑一次。

### §0.2 git 检查 + 测试 sanity

```bash
git log --oneline -3
# 期望最新 commit 是 phase4 commit + --cell-ids 扩展（fbadea3）

# phase4 相关测试 sanity（< 10 s）
uv run pytest \
    tests/cache/components/factors/test_composer_zero_nan.py \
    tests/exp/test_phase3_threshold_solver.py \
    tests/exp/test_phase4_spec.py \
    tests/exp/test_phase4_runner.py \
    -q
# 期望: 176 passed (22 + 20 + 97 + 37)
```

### §0.3 输出目录

```bash
mkdir -p exp/verdict_factor_judge/data/phase4/{warmup_factor_raw,warmup_dump,r1_alpha/{per_step,episode_results}}
mkdir -p exp/verdict_factor_judge/config/spatial16/phase4/{warmup,eval}
```

---

## §1 emit-warmup-yaml（无 server，一次性）

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-warmup-yaml --round 1
```

期望输出：
```
[emit-warmup-yaml] p1_state_fut_online_act -> exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml
[emit-warmup-yaml] p2_action_fut_online_act -> exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml
```

**Sanity**：

```bash
n=$(ls exp/verdict_factor_judge/config/spatial16/phase4/warmup/*.yaml 2>/dev/null | wc -l)
printf "phase4 warmup yamls: %d (expect 2)\n" "$n"

# 两个 warmup yaml 全 schema-valid
uv run python -c "
from openpi.cache.config import load_cache_config
from pathlib import Path
ok = 0; n = 0
for y in sorted(Path('exp/verdict_factor_judge/config/spatial16/phase4/warmup').glob('*.yaml')):
    n += 1
    cfg = load_cache_config(y); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 2/2 validated
```

---

## §2 run-warmup（1 server 即可，约 10 min）

Warmup 只需 1 server 跑（生成 `data/phase4/warmup_factor_raw/<recipe>.jsonl` 缓存供后续 6 batch run-eval 共享）。最简：用 S1（同 batch1 用的 server），warmup 跑完后保留这个 server 进 §3 eval。

### §2.1 启 S1 server (timan107，frp 7998 → 155.98.36.13:8998)

bootstrap 用任一 phase4 warmup yaml 即可（`run_phase4.py` 会按 recipe 切片逐个 `load_cache_config`）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s1 \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### §2.2 client 跑 run-warmup（两个 recipe 一起跑）

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-warmup --round 1 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim
```

期望输出：
```
[run-warmup] p1_state_fut_online_act: load + run 2 trials/task
[run-warmup] p1_state_fut_online_act: factor_raw -> exp/verdict_factor_judge/data/phase4/warmup_factor_raw/p1_state_fut_online_act.jsonl
[run-warmup] p2_action_fut_online_act: load + run 2 trials/task
[run-warmup] p2_action_fut_online_act: factor_raw -> exp/verdict_factor_judge/data/phase4/warmup_factor_raw/p2_action_fut_online_act.jsonl
```

**Sanity**：

```bash
ls -lh exp/verdict_factor_judge/data/phase4/warmup_factor_raw/

uv run python -c "
import json
from pathlib import Path
for f in sorted(Path('exp/verdict_factor_judge/data/phase4/warmup_factor_raw').glob('*.jsonl')):
    rows = sum(1 for _ in open(f))
    # peek first row's factor_raw key set
    with open(f) as fh:
        d = json.loads(fh.readline())
    n_keys = len(d.get('factor_raw') or {})
    print(f'{f.name}: {rows} rows, first row {n_keys} keys')
"
# 期望: 每 recipe ~50-200 rows（依 warmup-trials × num_tasks × episode_steps），每行 10 keys
```

> **Stop & verify before §3**：若 factor_raw 不齐，emit-eval-yamls 会 fail-fast 报错。

---

## §3 emit-eval-yamls（无 server，一次性，14 cell 全展开）

R1 不需要 `--alpha-star` / `--offline-pattern` / `--online-pattern`（R1 sweeps α）。

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 1
```

期望输出（14 行）：
```
[emit-eval-yamls] r1 p1_state_fut_online_act a0.0 fh_thr=... ws_thr=... -> exp/verdict_factor_judge/config/spatial16/phase4/eval/spatial16_w8_d4_phase4_p1_state_fut_online_act__r1_a0.0.yaml
... (14 lines: p1 × 7α + p2 × 7α)
```

**Sanity**：

```bash
n=$(ls exp/verdict_factor_judge/config/spatial16/phase4/eval/*__r1_a*.yaml 2>/dev/null | wc -l)
printf "phase4 R1 eval yamls: %d (expect 14)\n" "$n"

# 14 yaml schema-valid
uv run python -c "
from openpi.cache.config import load_cache_config
from pathlib import Path
ok = n = 0
for y in sorted(Path('exp/verdict_factor_judge/config/spatial16/phase4/eval').glob('*__r1_a*.yaml')):
    n += 1
    cfg = load_cache_config(y); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 14/14 validated

# 抽检：α=0 cell 应有 8 个 offline weight = 0，2 个 online weight = 0.5
uv run python -c "
import yaml
y = yaml.safe_load(open('exp/verdict_factor_judge/config/spatial16/phase4/eval/spatial16_w8_d4_phase4_p1_state_fut_online_act__r1_a0.0.yaml'))
w = y['checkpoints']['cp1']['judge']['composer']['weights']
n_off = sum(1 for k,v in w.items() if 'offline' in k and v != 0)
n_on  = sum(1 for k,v in w.items() if 'online'  in k and v != 0)
print(f'α=0 offline non-zero: {n_off} (expect 0); online non-zero: {n_on} (expect 2)')
"
```

---

## §4 6-server run-eval（并行）

S1 已经在跑（§2.1）。再启 S2-S6 五个 server（5 个独立终端），同款 bootstrap 模式。

#### Machine 1 — frp（timan107）

##### Server S2 — local port 7999（frp → 155.98.36.13:8999）

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s2 \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S3 — local port 8000（frp → 155.98.36.13:9000）

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s3 \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Machine 2 — 直连（149.165.151.106）

##### Server S4 — port 8001

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s4 \
    --env LIBERO \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S5 — port 8002

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s5 \
    --env LIBERO \
    --port 8002 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S6 — port 8003

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase4/warmup/spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_phase4_s6 \
    --env LIBERO \
    --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## §5 客户端命令（6 client，1:1 配对 server）

server 全启动完毕后再上 client。每 client 一个终端，6 条**同时启动**。每 batch 独立 `--summary-out` 文件避免并发写冲突；最后在 §6 合并。

### batch1 → S1 (p1, α=0.0/0.2, frp 8998, GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p1_state_fut_online_act \
    --cell-ids a0.0 a0.2 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch1.jsonl
```

### batch2 → S2 (p1, α=0.4/0.5, frp 8999, GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p1_state_fut_online_act \
    --cell-ids a0.4 a0.5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch2.jsonl
```

### batch3 → S3 (p1, α=0.6/0.8/1.0, frp 9000, GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p1_state_fut_online_act \
    --cell-ids a0.6 a0.8 a1.0 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch3.jsonl
```

### batch4 → S4 (p2, α=0.0/0.2, 直连 8001, GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p2_action_fut_online_act \
    --cell-ids a0.0 a0.2 \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch4.jsonl
```

### batch5 → S5 (p2, α=0.4/0.5, 直连 8002, GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p2_action_fut_online_act \
    --cell-ids a0.4 a0.5 \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch5.jsonl
```

### batch6 → S6 (p2, α=0.6/0.8/1.0, 直连 8003, GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --recipe p2_action_fut_online_act \
    --cell-ids a0.6 a0.8 a1.0 \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase4/r1_alpha/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase4/r1_alpha/episode_results \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase4/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase4/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch6.jsonl
```

> **注意**：每 batch 跑完会**各自调一次** `_dump_decision_gate_table`，把自己看到的 summary（仅自己的 cells）写入 `decision_gate.json`。这步在 single-batch 模式下没意义；正确的 R1 决策门要在 §6 merge 后**重跑**。

---

## §6 数据回收 + R1 决策门

### §6.1 合并 6 batch summary

每 batch 跑完写到独立 `per_yaml_summary_batch<N>.jsonl`，全部跑完后合并为 master：

```bash
cat exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary_batch{1,2,3,4,5,6}.jsonl \
    > exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl

# Sanity: 期望 14 行 (2 recipe × 7 alpha)
wc -l exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl

# 每 recipe / α 都齐
uv run python -c "
import json, collections
counts = collections.Counter()
for line in open('exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl'):
    row = json.loads(line)
    counts[(row.get('recipe_id'), row.get('alpha'))] += 1
for (rid, a), n in sorted(counts.items()):
    print(f'{rid:30s} α={a}  rows={n}')
print(f'unique cells: {len(counts)}  (expect 14)')
"
```

### §6.2 R1 决策门（基于 merged summary 重新生成）

每 batch 自己写的 `decision_gate.json` 仅包含该 batch 的 cells，**无意义**。基于 merged summary 重跑决策门：

```bash
uv run python -c "
from pathlib import Path
from exp.verdict_factor_judge.run_phase4 import _dump_decision_gate_table
out = _dump_decision_gate_table(
    round_id=1,
    summary_path=Path('exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl'),
)
import json
print(json.dumps(out['winners'], indent=2))
print('---')
print(json.dumps(out['trigger_decisions'], indent=2))
print('---')
print('next_args_suggestion:', json.dumps(out['next_args_suggestion'], indent=2))
"
```

`decision_gate.json` 自动写到 `exp/verdict_factor_judge/data/phase4/r1_alpha/decision_gate.json`，含：
- `winners[rid]`：每 recipe 的 argmax(SR - 0.5·inf) 行
- `trigger_decisions[rid]`：`continue=True` 表示 SR(α\*) ≥ recipe anchor − 2pp（p1 anchor=0.95、p2 anchor=0.96），可进 R2
- `next_args_suggestion`：直接可粘贴的 R2 CLI 字符串（**仅含 continue=True 的 recipe**）

### §6.3 R1 α 曲线图

```bash
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_alpha_sweep_phase4 \
    --summary exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl \
    --out     exp/verdict_factor_judge/analysis/phase4_alpha_sweep.png
```

输出：双面板（p1, p2），x=α, 蓝实线=SR, 红虚线=inf_ratio，灰虚线=phase3 anchor SR (0.95 / 0.96)。

### §6.4 端点 sanity check（必跑，验 src/composer 改造正确）

```bash
uv run python -c "
import json
rows = [json.loads(l) for l in open('exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl')]
# α=0 应复现 g6 SR 在 (locked cell)；α=1 应复现 g1 / g10 SR
for r in rows:
    rid = r['recipe_id']; a = r['alpha']
    sr = r.get('success_rate')
    if a in (0.0, 1.0):
        print(f'  {rid:30s} α={a}  SR={sr:.3f}  expected_anchor: '
              f'{\"phase3 g6 (online_action only)\" if a == 0.0 else (\"phase3 g1 SR≈0.95\" if rid.startswith(\"p1\") else \"phase3 g10 SR≈0.96\")}')
"
```

期望（±3pp 内）：
- p1 α=1.0 ≈ 0.95（复现 g1 phase3 (0.5, 0.5) cell SR）
- p2 α=1.0 ≈ 0.96（复现 g10 phase3 (0.5, 0.4) cell SR）
- p1 / p2 α=0.0 ≈ g6 phase3 cell SR（参考 phase3_results.md g6 各 cell）

> 端点失败 → composer / solver passthrough 实现有 bug；不要继续 R2。

### §6.5 打包 + 下载（可选）

```bash
tar czf phase4_stage1_$(date +%Y%m%d_%H%M%S).tar.gz \
    exp/verdict_factor_judge/data/phase4/r1_alpha/ \
    exp/verdict_factor_judge/data/phase4/warmup_factor_raw/ \
    exp/verdict_factor_judge/analysis/phase4_alpha_sweep.png
ls -lh phase4_stage1_*.tar.gz
```

下载到本地后告诉 Claude 分析。

### §6.6 inference_ratio 公式（同 phase3）

```python
inf_ratio = (n_full_hit * 0.0 + n_warm_start * 0.75 + n_miss * 1.0) / n_eval_verdicts
# warm cost 0.75 (start_t = 0.5)
```

---

## §7 失败处理

- **某 batch 跑挂**：`run_phase4 --mode run-eval` 已捕异常 → log + continue；下次启同 batch 命令时，`_load_done_yaml_ids` 自动跳已 done 的 cell（按 `--summary-out` 文件 resume）。
- **emit-eval-yamls fail**（factor_raw 缺失）：检查 `data/phase4/warmup_factor_raw/<recipe>.jsonl` 是否存在 + 非空；不在则回 §2 重跑 run-warmup。
- **frp 断**（仅 S1-S3）：`tmux capture-pane -pt run4` 看是否在 `Waiting for server at ws://...`；如是 `frpc reload` 后等恢复。
- **直连 timeout**（S4-S6）：`nc -zv 149.165.151.106 800{1,2,3}`。
- **端点 sanity 失败**（§6.4）：composer / solver 实现 bug。停止 R2 计划，回归 G2 review。

---

## §8 数据布局（执行结束后）

```
exp/verdict_factor_judge/
├── config/spatial16/phase4/
│   ├── warmup/                                                           # §1 emit
│   │   ├── spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup.yaml
│   │   └── spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup.yaml
│   └── eval/                                                             # §3 emit (14 yaml)
│       ├── spatial16_w8_d4_phase4_p1_state_fut_online_act__r1_a0.0.yaml
│       ├── ... (5 more p1 cells)
│       ├── spatial16_w8_d4_phase4_p1_state_fut_online_act__r1_a1.0.yaml
│       ├── spatial16_w8_d4_phase4_p2_action_fut_online_act__r1_a0.0.yaml
│       ├── ... (5 more p2 cells)
│       └── spatial16_w8_d4_phase4_p2_action_fut_online_act__r1_a1.0.yaml
├── data/phase4/
│   ├── warmup_factor_raw/                                                # §2 run-warmup output
│   │   ├── p1_state_fut_online_act.jsonl
│   │   └── p2_action_fut_online_act.jsonl
│   ├── warmup_dump/                                                      # raw fetch_dump bytes
│   │   └── (server-side scratch)
│   └── r1_alpha/                                                         # §5 run-eval output
│       ├── per_step/<yaml_id>.jsonl                                      # client 写
│       ├── episode_results/<yaml_id>.json                                # client 写
│       ├── per_yaml_summary_batch{1..6}.jsonl                            # 6 batch 独立
│       ├── per_yaml_summary.jsonl                                        # §6.1 merge 后 master, 14 行
│       └── decision_gate.json                                            # §6.2 重跑后正确版
└── analysis/
    └── phase4_alpha_sweep.png                                            # §6.3
```

---

## §9 下一步

§6.2 输出的 `next_args_suggestion["alpha-star"]` 字符串是 **Stage 2 (R2 offline desc 9 patterns)** 的输入。例如：

```bash
# Stage 2 启动命令（待 §6 完成后照做）
ALPHA_STAR='p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6'   # 真实值由 §6.2 给出

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 2 --alpha-star "$ALPHA_STAR"
# ... 然后 6 batch run-eval (Stage 2 run-commands log 另写)
```

Stage 2 cell 数 = 18（2 recipe × 9 pattern），可按 (recipe, pattern 子集) 切 6 batch；具体命令在 stage2 log 给出。

---

## §10 节点检查总单（顺序执行）

| # | 步骤 | 命令位置 | 预期产出 |
|---:|---|---|---|
| 1 | pkl key 检查 | §0.1 | "phase4 needed offline keys: 16; missing: none" |
| 2 | 测试 sanity | §0.2 | 176 passed |
| 3 | 输出目录 | §0.3 | mkdir 完成 |
| 4 | emit-warmup-yaml | §1 | 2 yaml |
| 5 | 启 S1 server | §2.1 | server listen on 7998 |
| 6 | run-warmup | §2.2 | 2 factor_raw jsonl |
| 7 | emit-eval-yamls | §3 | 14 eval yaml |
| 8 | 启 S2-S6 server | §4 | 5 server listen on 7999/8000/8001/8002/8003 |
| 9 | 6 batch run-eval | §5 | 6 summary file |
| 10 | merge summary | §6.1 | 14 行 |
| 11 | R1 决策门 | §6.2 | decision_gate.json + next_args_suggestion |
| 12 | α 曲线图 | §6.3 | phase4_alpha_sweep.png |
| 13 | 端点 sanity | §6.4 | α=1 复现 phase3 anchor ±3pp |
