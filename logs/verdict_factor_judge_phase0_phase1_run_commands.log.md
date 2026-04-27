> Status: Plan
> Date: 2026-04-27
> Level: L1

# verdict_factor_judge Phase 0 + Phase 1 执行命令清单

Plan：[`logs/verdict_factor_judge_experiment_plan.log.md`](verdict_factor_judge_experiment_plan.log.md)。

**算力**：3 GPU server × 3 client (5 worker each)。Phase 0 用 **3 client**（每 server 1，避 dump JSONL 多进程并发写撕裂行）；Phase 1 用 **9 client**（每 server 3，`--runs` 切片 8 yaml = 3 / 3 / 2）。

| Phase | yaml 数 | episode 数 | client 数 | 阻断点 |
|---|---|---|---|---|
| 0 | 3 (3 cfg × 1 yaml) | 300 | 3 (1/server) | dump JSONL 单 client per server |
| 1 | 24 (3 cfg × 8 yaml) | 2,400 | 9 (3/server) | 各 client 独立 `--state-path` |

---

## 端口映射

沿用 random_periodic_gate / phase1_libero_spatial_llm 的 frp 约定：server 本地端口 + 1000 = frp 公网端口（`155.98.36.13`）。

| Server | cfg | KeyBuilder run_id | local port | frp 公网入口 | preload pkl |
|---|---|---|---:|---|---|
| **S1** | `clip` | `clip_w7_d4` | `7998` | `155.98.36.13:8998` | `exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl` |
| **S2** | `max_pool` | `max_pool_w3_d5` | `7999` | `155.98.36.13:8999` | `exp/common/data/cache_artifacts/libero_spatial/cp1_max_pool.pkl` |
| **S3** | `spatial16` | `spatial16_w8_d4` | `8000` | `155.98.36.13:9000` | `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` |

> **Frp 检查**：本机 `~/.config/frp/frpc.toml` 需要同时映射 8998 / 8999 / 9000；若 `155.98.36.13` 默认只挂 9000（单 server 部署），先在 frpc.toml 加另两条 tcp 映射并 `frpc reload`。

---

## 前置检查

客户端机器（LIBERO eval 主机）确认 3 入口健康：

```bash
for p in 8998 8999 9000; do
    printf 'port %s: ' "$p"
    curl -s "http://155.98.36.13:${p}/healthz" || echo
done
```

3 个 pkl 齐全（GPU server 侧，repo root）：

```bash
ls exp/common/data/cache_artifacts/libero_spatial/{clip_vit_b_32,cp1_max_pool,cp1_spatial_pool_16}.pkl 2>/dev/null | wc -l
# 期望：3
```

baseline 复用文件存在（client 机器侧，分析阶段需要）：

```bash
ls exp/warm_start/data/baseline_failures.json \
   exp/warm_start/data/{clip,max_pool,spatial16}/cache_eval_results.json
# 期望：4 行均输出
```

yaml 输出目录（client 机器侧 — 跑前为空，phase script 写完后填）：

```bash
ls exp/verdict_factor_judge/config/{clip,max_pool,spatial16}/ 2>/dev/null
mkdir -p exp/verdict_factor_judge/data/{calibration,phase0,phase1}
```

---

## §0 Yaml 生成（执行 Phase 0 / Phase 1 前必做）

`exp/verdict_factor_judge/generate_yamls.py` 是 invariant helper（`write_yaml` + `check_yaml_invariant`），不是 phase generator。需要写两个 phase-spec 脚本，落 27 个 yaml：

| 脚本（待写）| 目标目录 | 输出 yaml 数 |
|---|---|---|
| `exp/verdict_factor_judge/phase0_spec.py` | `exp/verdict_factor_judge/config/{clip,max_pool,spatial16}/` | 3（每 cfg 1） |
| `exp/verdict_factor_judge/phase1_spec.py` | 同上 | 24（每 cfg 8） |

两 spec 脚本笛卡尔展开 yaml dict 后调 `generate_yamls.write_yaml(path, content)`。invariant：Phase 0 yaml 含 `judge.dump.config_id`，由 `write_yaml` pin 成 stem；Phase 1 yaml 无 `dump` block，invariant no-op（`check_yaml_invariant` 直接 return）。

### Phase 0 yaml 列表（3）

| Server | yaml 路径 | 内容要点 |
|---|---|---|
| S1 | `exp/verdict_factor_judge/config/clip/clip_w7_d4_phase0_always_hit_dump.yaml` | judge=`always_hit` + dump（5 因子全开），`search_strategy.top_k=5` |
| S2 | `exp/verdict_factor_judge/config/max_pool/max_pool_w3_d5_phase0_always_hit_dump.yaml` | 同上，KeyBuilder/字段权重换 max_pool |
| S3 | `exp/verdict_factor_judge/config/spatial16/spatial16_w8_d4_phase0_always_hit_dump.yaml` | 同上，KeyBuilder/字段权重换 spatial16 |

3 yaml 公共结构（plan §3.7 + §4 Phase 0）：

```yaml
enabled: true
keys: { ... }                 # 按 plan §3.7 cfg 表填字段权重
key_builder: { type: <cfg_kb> }
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }
    judge:
      type: always_hit
      dump:
        path: exp/verdict_factor_judge/data/calibration/<cfg>.jsonl
        # config_id 由 generate_yamls.write_yaml 自动 pin 成 yaml stem
        factors:
          - { type: f1a_a }
          - { type: f1a_t }
          - { type: f1b_a, windows: [[0,3],[1,1],[3,0],[0,5],[5,0]] }
          - { type: f1b_t, windows: [[0,3],[1,1],[3,0],[0,5],[5,0]] }
          - { type: f2,    K: 5 }
    search_strategy:
      type: weighted_rrf_knn
      top_k: 5                # plan §4 Phase 0：F2 dump 需要 ≥2 候选才能算 variance
      step_filter: all
      rrf_k: 60
      trajectory_depth: <cfg_depth>
      trajectory_weights: [...]
      field_similarity:
        vision_0:    { type: cosine }
        vision_1:    { type: cosine }
        prompt_emb:  { type: cosine }
        robot_state: { type: l2, to_similarity: { type: exp, tau: 0.334717 } }
backend:
  type: in_memory
  vector_dims: { ... }        # 按 cfg 表填
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/<cfg>.pkl
    index_type: brute_force
write_policy: { type: never }
```

### Phase 1 yaml 列表（24，按 lex 序）

每 cfg 8 yaml，stem 形如 `<cfg_run_id>_phase1_<descriptor>.yaml`。lex 排序后的 8 个 descriptor 决定 `--runs` 切片：

| lex idx | descriptor | 因子集 | 描述子启用 | tier | composer |
|---:|---|---|---|---|---|
| 1 | `f_f1a_t_only_d_all_t_full` | `f1a_t` only | D-ALL | T-FULL | C-WS |
| 2 | `f_f1b_a_only_d_all_t_full` | `f1b_a` only | D-ALL | T-FULL | C-WS |
| 3 | `f_f1b_t_only_d_all_t_full` | `f1b_t` only | D-ALL | T-FULL | C-WS |
| 4 | `f_full_d_all_t_dual_07` | F-FULL（5 因子） | D-ALL | **T-DUAL(0.7)** | C-WS |
| 5 | `f_min_a_d_all_t_full` | F-MIN-A (`f1a_a`) | D-ALL | T-FULL | C-WS |
| 6 | `f_min_a_d_dir_t_full` | F-MIN-A | **D-DIR**（仅 `f1a_a_dir`） | T-FULL | C-WS |
| 7 | `f_min_a_d_jerk_t_full` | F-MIN-A | **D-JERK**（仅 `f1a_a_jerk`） | T-FULL | C-WS |
| 8 | `f_min_cons_d_all_t_full` | F-MIN-CONS (`f2`) | n/a | T-FULL | C-WS |

Phase 1 yaml 公共结构（plan §3.4b + §4 Phase 1）：

```yaml
enabled: true
keys: { ... }
key_builder: { type: <cfg_kb> }
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }
    judge:
      type: composite
      composer: { type: weighted_sum, threshold: 0.5 }
      tier:
        - { type: full_hit }                          # T-FULL；T-DUAL_07 yaml 在此后追加 { type: warm_start, start_t: 0.7 }
      normalizer:
        type: percentile_rolling
        window: 200
        cold_start_strategy: force_miss
      factors:
        - { type: f1a_a }                              # 按 yaml descriptor 调整因子集 / windows
      directions:                                      # plan §3.4b N-PCT 表（仅 non_monotonic 描述子需要）
        f1a_a_curv_radius: { type: range, low: 0.3, high: 0.7 }
        f1a_a_cum_disp:    { type: high }
        # F1b 系列每窗口镜像；D-JERK / D-DIR yaml 不含 non_monotonic 描述子，可省 directions
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1                                         # CompositeJudge 由 min_top_k_hint 自动撑到 5
      ...                                              # 同 Phase 0 公共字段
backend: { ... }                                       # 同 Phase 0
write_policy: { type: never }
```

> **F1b 窗口锁定 W-MIX = `(0,3)(1,1)(3,0)(0,5)(5,0)`**（plan §3.3 / §3.3b B-3）。`(5,5)` `(7,7)` 在 Phase 1 默认禁用。

> **directions 配置**：D-JERK / D-DIR 单描述子 yaml 仅含 `jerk` 或 `dir`（orientation = risky / safe），不需要 `directions` block。D-ALL yaml 含 `curv_radius` / `cum_disp` 必须显式给 `range:[0.3,0.7]` / `high`，否则 validator 拒收（plan §3.4b）。

### 生成后验

```bash
# Phase 0：dump.config_id == yaml stem 不变量
uv run python -c "
from exp.verdict_factor_judge.generate_yamls import check_yaml_invariant
import pathlib
for p in pathlib.Path('exp/verdict_factor_judge/config').rglob('*phase0*.yaml'):
    check_yaml_invariant(p)
print('phase0 invariant OK')
"

# Phase 0/1：static composite validator 全过
uv run python -c "
import yaml, pathlib
from openpi.cache.config import build_cache_config
for p in sorted(pathlib.Path('exp/verdict_factor_judge/config').rglob('*.yaml')):
    cfg = build_cache_config(yaml.safe_load(p.read_text()))
    print('OK', p)
"
```

---

## Phase 0 — Calibration Dump（3 yaml × 100 ep = 300 ep）

### §1.1 服务器命令（3 server，每 server 1 终端）

#### Server S1 — clip, local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/clip/clip_w7_d4_phase0_always_hit_dump.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S2 — max_pool, local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/max_pool/max_pool_w3_d5_phase0_always_hit_dump.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server S3 — spatial16, local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/spatial16_w8_d4_phase0_always_hit_dump.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### §1.2 客户端 Runner 命令（3 client，1 per server）

每 client 1 yaml × 100 ep（10 task × 10 trial）。dump JSONL 写到 `data/calibration/<cfg>.jsonl`，文件由 server 端 DumpingJudge 持有 — **同 server 上必须只有 1 client 在跑 Phase 0**，否则单行 JSONL > 4096 字节会被 POSIX 的非原子 append 撕裂。

#### Client A → S1 (clip)

```bash
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/clip \
    --runs 1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase0/clip/run_state.json \
    --log-dir exp/verdict_factor_judge/data/phase0/clip/logs \
    --resume
```

#### Client B → S2 (max_pool)

```bash
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/max_pool \
    --runs 1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase0/max_pool/run_state.json \
    --log-dir exp/verdict_factor_judge/data/phase0/max_pool/logs \
    --resume
```

#### Client C → S3 (spatial16)

```bash
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/spatial16 \
    --runs 1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase0/spatial16/run_state.json \
    --log-dir exp/verdict_factor_judge/data/phase0/spatial16/logs \
    --resume
```

> **`--runs 1`**：phase0 dir 只 1 个 yaml；显式 `--runs 1` 让 `--resume` state 命名稳定。
> **`--episodes-per-run 10`**：每 task 10 trial × 10 task = 100 ep（plan §2.4 单元）。LIBERO 默认 seed 0..9 init —— 与 baseline 复用 join 的 `(task_id, orig_init_state_idx)` 对应。

### §1.3 Phase 0 输出位置

| 文件 | 写入方 | 用途 |
|---|---|---|
| `exp/verdict_factor_judge/data/calibration/clip_w7_d4_phase0_always_hit_dump.jsonl` | DumpingJudge (server) | 每 verdict 1 行，5 因子 raw + nan + (config_id, task_id, orig_init_state_idx, step_idx) |
| `exp/verdict_factor_judge/data/calibration/max_pool_w3_d5_phase0_always_hit_dump.jsonl` | 同上 | 同上 |
| `exp/verdict_factor_judge/data/calibration/spatial16_w8_d4_phase0_always_hit_dump.jsonl` | 同上 | 同上 |
| `exp/verdict_factor_judge/data/phase0/<cfg>/cache_eval_results.json` | runner | per-episode success → join key |
| `exp/verdict_factor_judge/data/phase0/<cfg>/run_state.json` | runner | resume |
| `exp/verdict_factor_judge/data/phase0/<cfg>/logs/<run_id>.log` | runner | stdout/stderr |

### §1.4 Phase 0 完成后的最小验证

```bash
# 行数 = verdict 调用次数。每 ep ≈ 21 step（T median，每 step 触发 1 verdict）
# 100 ep × ~21 verdict ≈ ~2,100 行/cfg；3 cfg ≈ ~6,300 行总
for cfg in clip_w7_d4 max_pool_w3_d5 spatial16_w8_d4; do
    f=exp/verdict_factor_judge/data/calibration/${cfg}_phase0_always_hit_dump.jsonl
    printf "%s: %d lines\n" "$f" "$(wc -l < "$f")"
done

# 100% join 覆盖（join key (config_id, task_id, orig_init_state_idx)）
uv run python -c "
import json, pathlib
for cfg in ['clip', 'max_pool', 'spatial16']:
    out = pathlib.Path(f'exp/verdict_factor_judge/data/phase0/{cfg}/cache_eval_results.json')
    res = json.loads(out.read_text())
    rows = res['rows'] if isinstance(res, dict) else res
    print(cfg, 'cache_eval_results rows:', len(rows))
"
```

---

## Phase 1 — 单因子 Ablation（24 yaml × 100 ep = 2,400 ep）

### §2.1 服务器命令（3 server）

**复用 Phase 0 server 实例**（runner WebSocket `load_cache_config` 会切换到 Phase 1 yaml）。如果 Phase 0 server 已被 ctrl-C，按下方重启（`--cache-config` 用各 cfg 的 phase1 lex-序首份 yaml 作种子）：

```bash
# S1 重启（如需）
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/clip/clip_w7_d4_phase1_f_f1a_t_only_d_all_t_full.yaml \
    --env LIBERO --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# S2 重启（如需）
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/max_pool/max_pool_w3_d5_phase1_f_f1a_t_only_d_all_t_full.yaml \
    --env LIBERO --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# S3 重启（如需）
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/spatial16_w8_d4_phase1_f_f1a_t_only_d_all_t_full.yaml \
    --env LIBERO --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### §2.2 客户端 Runner 命令（9 client，3 per server）

每 cfg 8 yaml 切成 **3 / 3 / 2** 三段（lex 序），每段 1 client 5 worker 跑。`--runs N-M` 1-based 闭区间 → 0-based 索引集。`--state-path` 必须各 client 独立，否则 state file race。

#### S1 (clip) — 3 clients

```bash
# Client A1 — clip yaml 1-3
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/clip \
    --runs 1-3 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/clip/run_state_clientA.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/clip/logs \
    --resume

# Client A2 — clip yaml 4-6
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/clip \
    --runs 4-6 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/clip/run_state_clientB.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/clip/logs \
    --resume

# Client A3 — clip yaml 7-8
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/clip \
    --runs 7-8 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/clip/run_state_clientC.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/clip/logs \
    --resume
```

#### S2 (max_pool) — 3 clients

```bash
# Client B1 — max_pool yaml 1-3
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/max_pool \
    --runs 1-3 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/max_pool/run_state_clientA.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/max_pool/logs \
    --resume

# Client B2 — max_pool yaml 4-6
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/max_pool \
    --runs 4-6 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/max_pool/run_state_clientB.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/max_pool/logs \
    --resume

# Client B3 — max_pool yaml 7-8
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/max_pool \
    --runs 7-8 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/max_pool/run_state_clientC.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/max_pool/logs \
    --resume
```

#### S3 (spatial16) — 3 clients

```bash
# Client C1 — spatial16 yaml 1-3
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/spatial16 \
    --runs 1-3 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/spatial16/run_state_clientA.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/spatial16/logs \
    --resume

# Client C2 — spatial16 yaml 4-6
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/spatial16 \
    --runs 4-6 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/spatial16/run_state_clientB.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/spatial16/logs \
    --resume

# Client C3 — spatial16 yaml 7-8
conda run -p /scratch/zixuans8/libero_sim \
    uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/verdict_factor_judge/config/spatial16 \
    --runs 7-8 \
    --episodes-per-run 10 --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial --seed 42 --conda-env libero_sim \
    --state-path exp/verdict_factor_judge/data/phase1/spatial16/run_state_clientC.json \
    --log-dir   exp/verdict_factor_judge/data/phase1/spatial16/logs \
    --resume
```

### §2.3 Phase 1 输出位置

| 文件 | 写入方 | 用途 |
|---|---|---|
| `exp/verdict_factor_judge/data/phase1/<cfg>/cache_eval_results.json` | runner（合并 3 client）| per-episode success；analysis paired McNemar 主源 |
| `exp/verdict_factor_judge/data/phase1/<cfg>/run_state_client{A,B,C}.json` | runner | resume，3 client 各自独立 |
| `exp/verdict_factor_judge/data/phase1/<cfg>/logs/<run_id>.log` | runner | per-yaml 日志 |

> **Phase 1 不写 dump JSONL**（CompositeJudge 无 `dump` block），因此 9 client 并发安全。

---

## §3 汇总 analysis（Phase 0/1 都跑完后）

待写脚本（plan §6.1 列入）：

| 脚本 | 输入 | 产出 |
|---|---|---|
| `exp/verdict_factor_judge/analysis/phase0_calibration_summary.py` | `data/calibration/*.jsonl` + `data/phase0/*/cache_eval_results.json` | `analysis/phase0_summary.md`：per-cfg 因子分布、winner-conditional NaN%、跨 cfg 一致性 |
| `exp/verdict_factor_judge/analysis/phase1_factor_ablation_summary.py` | `data/phase1/*/cache_eval_results.json` + `exp/warm_start/data/baseline_failures.json` + `exp/warm_start/data/<cfg>/cache_eval_results.json` | `analysis/phase1_summary.md`：每因子 success vs Floor / Ceiling-A / Ceiling-W、`inference_time_saved_ratio`（含 0.5 系数）、Wilson CI、paired McNemar、Pareto 决策表 |

调用骨架（脚本写完后）：

```bash
uv run python -m exp.verdict_factor_judge.analysis.phase0_calibration_summary \
    --calibration-dir exp/verdict_factor_judge/data/calibration \
    --outcome-dir exp/verdict_factor_judge/data/phase0 \
    --out exp/verdict_factor_judge/analysis/phase0_summary.md

uv run python -m exp.verdict_factor_judge.analysis.phase1_factor_ablation_summary \
    --phase1-dir exp/verdict_factor_judge/data/phase1 \
    --warm-start-dir exp/warm_start/data \
    --out exp/verdict_factor_judge/analysis/phase1_summary.md
```

---

## §4 关键 Caveat

### 4.1 Phase 0 dump JSONL 必须 1 client per server

`DumpingJudge` 用 `open(path, "a")` 直接写 JSON 行（`src/openpi/cache/components/judge.py:541`）。POSIX 仅在 write < `PIPE_BUF`（Linux 4096B）时保证 `O_APPEND` 原子；Phase 0 单行含 5 因子 × 多窗口 + factor_nan，可能 > 4 KB → **多进程并发写会撕裂**。Phase 0 严格 1 client per server；Phase 1 无 dump，9 client 并发安全。

### 4.2 `search_strategy.top_k` 语义 — Phase 0 必填 5

`top_k` = search 返回给 judge 的候选池大小，与 judge 的 K 无直接绑定（`src/openpi/cache/config.py:1694` `effective_top_k = max(cfg.top_k, min_top_k_hint)`）。`AlwaysHitJudge` 不暴露 `min_required_top_k` → 默认撑不到 5 → F2 dump 必拿 NaN（`consensus.py:85-89` 的 `K_eff < 2 → return NaN`）。**Phase 0 必须把 yaml 的 `search_strategy.top_k` 显式写成 5**。Phase 1 走 `CompositeJudge` 时由 `min_top_k_hint` 自动撑，yaml 写 1 即可。

### 4.3 `dump.config_id == yaml stem` 不变量

`generate_yamls.write_yaml` 自动 pin `dump.config_id` 成文件 stem。Phase 0 如果 spec dict 显式给了不同的 `config_id`，会 raise `InvariantError`。**Phase 5 的三方 join key (`config_id, task_id, orig_init_state_idx`) 依赖此不变量** —— 直接手编 yaml 改名前必须同步改 `dump.config_id`，或 rename 文件让 stem 跟着变。

### 4.4 多 client 并发命中同 server

Phase 1 每 server 3 client 各自开 ws connection。Server 端 `build_per_connection_components` 给每个 connection 一份独立的 Orchestrator + per-conn judge / cache state；`load_cache_config` 是 per-connection scope。同 cfg 下 3 client 切的 yaml 共用 `backend.in_memory.preload_path` → pkl 只加载一次（server 启动时），ws 切 bundle 不触发 pkl reload。

### 4.5 `--state-path` 必须各 client 独立

默认 `experiment_state.json` 在 `--yaml-dir` 下，3 client 同 dir 会 race。各 client 显式给 `--state-path data/phase1/<cfg>/run_state_client{A,B,C}.json`，且与 `--runs` 切片对应（不同切片用不同 state file，`--resume` 才不会跨切片错乱）。

### 4.6 Phase 1 复用 Phase 0 server，不需重启

runner 通过 ws 控制信号 `load_cache_config` 切 bundle。Phase 0 跑完后 Phase 1 9 client 可以直接命中同一 server 实例。仅当 server 进程被 ctrl-C 后才需要重启（用 §2.1 命令）。

### 4.7 Pkl 路径迁移

warm_start yaml 原指 `libero_spatial_warm/`（已删）。本实验用 `libero_spatial/{clip_vit_b_32, cp1_max_pool, cp1_spatial_pool_16}.pkl`（commit `66a341f` 重建，含 168 keys F1b 因子）。**所有 yaml 的 `backend.in_memory.preload_path` 必须用新路径**，否则 server 加载失败。

### 4.8 端口冲突

`8998 / 8999 / 9000` 与 `random_periodic_gate` / `phase1_libero_spatial_llm` 共享。跑本实验前确认那两个实验的 server 已关。

---

## §5 执行次序（推荐）

1. **生成 yaml**（§0）：写 Phase 0 spec → 跑 spec 脚本 → 验 invariant + composite validator。Phase 1 spec 同步写好（plan 已锁定，可与 Phase 0 同时生成）。
2. **前置检查**（前置检查节）：frp 健康、pkl、baseline 文件、yaml count = 27。
3. **启 3 server**（§1.1）：3 GPU server 各 1 tmux 终端。
4. **跑 Phase 0**（§1.2）：3 client（1/server）。等 calibration JSONL + cache_eval_results.json 完成。验证 §1.4。
5. **Phase 0 analysis**（§3）：跑 `phase0_calibration_summary.py` 出 winner-conditional NaN% + 跨 cfg 一致性。
6. **跑 Phase 1**（§2.2）：复用同 server，9 client（3/server，各 5 worker）。
7. **Phase 1 analysis**（§3）：跑 `phase1_factor_ablation_summary.py` 出 Pareto 决策表 → 决定 Phase 2 因子集 carry-forward 范围。
