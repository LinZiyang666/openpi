> Status: Plan
> Date: 2026-05-07
> Level: L1

# verdict_phase3 — 6-server 执行命令清单

Plan：[`logs/verdict_phase3_threshold_sweep.log.md`](verdict_phase3_threshold_sweep.log.md) (G1 APPROVED R5 / G2 APPROVED R2)。

Predecessor 实验运行教程（同款拓扑参考）：[`logs/archive/verdict_factor_judge_phase2_run_commands.log.md`](archive/verdict_factor_judge_phase2_run_commands.log.md)。

---

## 实验目的

在 spatial16 phase2 layer1 的 11 个金圈点 recipe（因子配置完全冻结）上，**把 thresholds 从「人工 0.5」改成「warmup 数据驱动反推」**，扫 16 个 (FH_ratio, WS_ratio) ∈ {0.2, 0.3, 0.4, 0.5}² 网格 cell，记录 SR / FH / WS / MISS / inference_ratio 与 (FH_thr, WS_thr) 的响应面。

新 composer `weighted_sum_zero_nan`：
- 等权相加（weights 全 1.0）
- **NaN→0 仍计入分母**（与现有 `WeightedSumComposer` 的 NaN-skip 不同）
- 强制双 threshold（FH + WS），不带 all-NaN warm fallback
- WARM_START emit `start_t = 0.5`（warm cost 0.75，不是 phase2 的 0.85）

每 recipe 16 cell × 11 recipe × 1 cfg (spatial16) = **176 eval yaml**。

判决 cascade（inclusive on both bounds）：
```
s ≥ FH_thr        → FULL_HIT
WS_thr ≤ s < FH_thr → WARM_START @ start_t=0.5
s < WS_thr        → MISS
```

---

## 算力 / 拓扑

**6 GPU server × 1 client/server**。Phase 3 是 spatial16-only，11 recipe 按字典序分到 6 batch（前 5 batch 各 2 recipe，batch6 单 g11）：

| 阶段 | yaml 数 | episode 数（warmup + eval） | server 数 | wall-clock 估算 |
|---|---|---|---|---|
| Phase 3 sweep | 11 warmup + 176 eval | 11×20 + 176×100 ≈ 17,820 ep | 6 (1 cfg, batch1-6) | ~3.0 h |

每 batch 工作量 ≈ 2 recipe × (20 + 16×100) = 3240 ep（batch6 单 recipe = 1620 ep）。

## 端口映射（6 server 拓扑，与 Phase 2 同款）

S1-S3 沿用 frp（本地端口 + 1000 = `155.98.36.32` 公网入口）。S4-S6 在另一台机器上有**直连公网 IP `149.165.151.106`**，端口直接从 8001 开始。

| 批次 | server 名 | recipe 切片 | 公网入口 | client GPU |
|---|---|---|---|---|
| **batch1** | S1-sp16 | g1, g2 | `155.98.36.32:8998` (frp) | 0 |
| **batch2** | S2-sp16 | g3, g4 | `155.98.36.32:8999` (frp) | 0 |
| **batch3** | S3-sp16 | g5, g6 | `155.98.36.32:9000` (frp) | 0 |
| **batch4** | S4-sp16 | g7, g8 | `149.165.151.106:8001` (直连) | 1 |
| **batch5** | S5-sp16 | g9, g10 | `149.165.151.106:8002` (直连) | 1 |
| **batch6** | S6-sp16 | g11 | `149.165.151.106:8003` (直连) | 1 |

> 与 Phase 2 区别：Phase 2 是「3 cfg × 2 half」（每 cfg 26 yaml 切 13+13），Phase 3 是「1 cfg × 6 split」（11 recipe 切 2-2-2-2-2-1）。所有 6 server 都加载 spatial16 同款 pkl + key_builder。

> **Frp 检查**（仅 S1-S3）：`~/.config/frp/frpc.toml` 沿用 8998/8999/9000 三条 tcp 映射，不需新增。
>
> **直连检查**（S4-S6）：客户端到 `149.165.151.106:8001-8003` 三个端口需通：
> ```bash
> for p in 8001 8002 8003; do
>   nc -zv 149.165.151.106 $p 2>&1 | head -1
> done
> ```

11 recipe 完整 ID（按 plan §1.1 字典序）：

```
g1_f1b_t_w_fut_d_all          (4 desc × 2 win = 8 keys)
g2_f1b_t_w_long_risk_d_jerk   (1 desc × 2 win = 2 keys)
g3_f1b_t_w_long_risk_d_all    (4 desc × 2 win = 8 keys)
g4_f1b_t_w_short_d_jerk       (1 desc × 3 win = 3 keys)
g5_f1a_t_d_jerk_dir_pair      (2 desc × 1 win = 2 keys)
g6_f1a_a_d_jerk_curv_pair     (2 desc × 1 win = 2 keys)
g7_f1b_a_w_long_risk_d_jerk   (1 desc × 2 win = 2 keys)
g8_f1a_t_d_curv_only          (1 desc × 1 win = 1 key)
g9_f1b_t_w_sym_s_d_all        (4 desc × 3 win = 12 keys)
g10_f1b_a_w_fut_d_all         (4 desc × 2 win = 8 keys)
g11_f1a_a_d_curv_only         (1 desc × 1 win = 1 key)
```

---

## §0 pkl enrich（一次性，必须先做）

Plan §4：现有 `cp1_spatial_pool_16.pkl` 168 个 factor key 全是**老命名** (`f1b_t_jerk__p5_f5` 等)。新 OfflineExtractor 读**新命名** (`jerk_offline_state__p5_f5`)。直接跑会全 NaN。

**必须 enrich**（不需要重 build；保留 168 老 key + 追加 64 新 key = 232 keys）：

```bash
# 备份（防原子写中断）
cp exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl \
   exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pre_phase3.bak.pkl

# 准备 8 offline factor × 8 windows = 64 keys 的 yaml
# 注意 YAML flow-mapping `:` 后必须有空格，否则 PyYAML 把 `past:0` 当 plain scalar，
# 解析为 {"past:0": None}，后续 `int(w["past"])` 报 KeyError: 'past'
cat > /tmp/phase3_offline_factors.yaml <<'YAML'
factors:
  - type: jerk_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: jerk_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: direction_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: direction_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: dispersion_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: dispersion_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: path_length_offline_action
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
  - type: path_length_offline_state
    params:
      windows:
        - {past: 0, future: 3}
        - {past: 0, future: 5}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 5, future: 5}
        - {past: 7, future: 7}
        - {past: 3, future: 0}
YAML

# 跑 enrich (in-place overwrite, ~2-5 min CPU only)
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl \
  --factors-yaml /tmp/phase3_offline_factors.yaml \
  --output exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl
```

期望 log：

```
INFO: Loading source artifact from .../cp1_spatial_pool_16.pkl
INFO: wrote enriched pkl: 1018 entries, 8 offline writers applied → .../cp1_spatial_pool_16.pkl
```

**Smoke gate（必跑）**：

```bash
uv run python -c "
import pickle
d = pickle.load(open('exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl','rb'))
keys = set(d['entries'][0].payload.factors.keys())
expected = {f'{desc}_offline_{ch}__p{p}_f{f}'
            for desc in ('jerk','direction','dispersion','path_length')
            for ch in ('action','state')
            for (p,f) in [(0,3),(0,5),(1,1),(2,2),(3,3),(5,5),(7,7),(3,0)]}
missing = expected - keys
legacy_kept = sum(1 for k in keys if k.startswith('f1b_'))
print(f'new keys missing: {missing or \"none\"}')
print(f'legacy keys kept: {legacy_kept}  (expect 168)')
print(f'total keys: {len(keys)}  (expect 232)')
"
# 期望:
#   new keys missing: none
#   legacy keys kept: 168  (expect 168)
#   total keys: 232  (expect 232)
```

> **重要**：6 server 必须全部读这份 enriched pkl。Phase 3 锁定的 canonical 路径是 `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl`（v2_spec / phase3_spec 都已校准；旧 `exp/warm_start/data/spatial16/...` 路径不存在，prior G1 review B3 已修）。

---

## §1 yaml 生成

```bash
# 生成 11 个 warmup yaml + manifest（eval yaml 由 runner 运行时 patch 生成，不在此步）
uv run python -m exp.verdict_factor_judge.phase3_spec
```

期望输出：

```
wrote 11 warmup yamls + manifest at exp/verdict_factor_judge/config/spatial16/phase3
```

**前置 sanity**：

```bash
# warmup yaml 数齐全
n=$(ls exp/verdict_factor_judge/config/spatial16/phase3/warmup/*.yaml 2>/dev/null | wc -l)
printf "warmup yamls: %d (expect 11)\n" "$n"

# 11 个 warmup yaml 全 schema-valid
uv run python -c "
from openpi.cache.config import load_cache_config
from pathlib import Path
ok = 0; n = 0
for y in sorted(Path('exp/verdict_factor_judge/config/spatial16/phase3/warmup').glob('*.yaml')):
    n += 1
    cfg = load_cache_config(y); ok += 1
print(f'{ok}/{n} validated')
"
# 期望: 11/11 validated
```

**输出目录**（在 server / client 跑前都得存在）：

```bash
mkdir -p exp/verdict_factor_judge/data/phase3/{per_step,episode_results,warmup,thresholds,logs}
mkdir -p exp/verdict_factor_judge/config/spatial16/phase3/eval
```

---

## §2 服务器命令（6 server，每 server 1 终端）

服务器 bootstrap 用 phase3 warmup yaml 中**任一**（`g1` 即可）—— 实际 `run_phase3.py` 会按 recipe 切片逐个 `load_cache_config`。

**所有 6 个 server 必须带 `--warmup-dump-root`**：`run_phase3.py` 跑每个 recipe 的 warmup 时会调 `fetch_dump` / `unload_warmup_buffer`，server 启动时不传 root 立即报错 `load_cache_config: server was started without --warmup-dump-root`。每 server 用独立 root（`_s1` ~ `_s6`）。

**所有 6 个 server 都用 `CUDA_VISIBLE_DEVICES=0`** —— 服务进程只占用本机 GPU 0；client 端 LIBERO sim 渲染才需要分卡（前 3 client GPU 0、后 3 client GPU 1）。

#### Machine 1 — 走 frp（timan107，frp → 155.98.36.32）

##### Server S1 — spatial16, local port 7998

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s1 \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S2 — spatial16, local port 7999

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s2 \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

##### Server S3 — spatial16, local port 8000

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s3 \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Machine 2 — 直连（149.165.151.106，port = 公网 port）

##### Server S4 — spatial16, port 8001

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
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
    --cache-config exp/verdict_factor_judge/config/spatial16/phase3/warmup/spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml \
    --warmup-dump-root /tmp/openpi_warmup_s6 \
    --env LIBERO \
    --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## §3 客户端命令（6 client，1:1 配对 server）

server 启动完毕后再上 client。每 client 一个终端，6 条**同时启动**。

每 batch 一个独立的 `--summary-out` 文件避免并发写冲突，最后在 §4 合并。每 batch 也独立 `--warmup-jsonl-dir / --thresholds-dir / --eval-yaml-dir` 防止 11 recipe 并行时 yaml/JSONL 文件互踩。

### batch1 → S1 (spatial16, recipes g1+g2, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g1_f1b_t_w_fut_d_all g2_f1b_t_w_long_risk_d_jerk \
    --host 155.98.36.32 --port 8998 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch1.jsonl \
    --resume
```

### batch2 → S2 (spatial16, recipes g3+g4, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g3_f1b_t_w_long_risk_d_all g4_f1b_t_w_short_d_jerk \
    --host 155.98.36.32 --port 8999 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch2.jsonl \
    --resume
```

### batch3 → S3 (spatial16, recipes g5+g6, client GPU 0)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g5_f1a_t_d_jerk_dir_pair g6_f1a_a_d_jerk_curv_pair \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 0 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch3.jsonl \
    --resume
```

### batch4 → S4 (spatial16, recipes g7+g8, 直连 149.165.151.106:8001, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g7_f1b_a_w_long_risk_d_jerk g8_f1a_t_d_curv_only \
    --host 149.165.151.106 --port 8001 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch4.jsonl \
    --resume
```

### batch5 → S5 (spatial16, recipes g9+g10, 直连 149.165.151.106:8002, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g9_f1b_t_w_sym_s_d_all g10_f1b_a_w_fut_d_all \
    --host 149.165.151.106 --port 8002 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch5.jsonl \
    --resume
```

### batch6 → S6 (spatial16, recipe g11, 直连 149.165.151.106:8003, client GPU 1)

```bash
uv run python -m exp.verdict_factor_judge.run_phase3 \
    --cfg-id spatial16_w8_d4 \
    --recipe-ids g11_f1a_a_d_curv_only \
    --host 149.165.151.106 --port 8003 \
    --task-suite libero_spatial \
    --num-workers 5 --warmup-trials 2 --eval-trials 10 \
    --cuda-visible-devices 1 \
    --conda-env /scratch/zixuans8/libero_sim \
    --per-step-log-dir       exp/verdict_factor_judge/data/phase3/per_step \
    --episode-results-dir    exp/verdict_factor_judge/data/phase3/episode_results \
    --warmup-jsonl-dir       exp/verdict_factor_judge/data/phase3/warmup \
    --thresholds-dir         exp/verdict_factor_judge/data/phase3/thresholds \
    --eval-yaml-dir          exp/verdict_factor_judge/config/spatial16/phase3/eval \
    --warmup-yaml-dir        exp/verdict_factor_judge/config/spatial16/phase3/warmup \
    --summary-out            exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch6.jsonl \
    --resume
```

---

## §4 数据回收 + 分析

### 4.1 合并 6 batch summary

每 batch 写到独立 `per_yaml_summary_batch<N>.jsonl`，跑完后合并为单一 master 文件：

```bash
cat exp/verdict_factor_judge/data/phase3/per_yaml_summary_batch{1,2,3,4,5,6}.jsonl \
    > exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl

# Sanity: 期望 176 行（11 recipe × 16 cell），加 0 或多个 _NA marker
wc -l exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl

# Sanity: 检查每 recipe 16 cell 都齐
uv run python -c "
import json, collections
counts = collections.Counter()
errors = 0
for line in open('exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl'):
    row = json.loads(line)
    counts[row.get('recipe_id', 'unknown')] += 1
    if 'error' in row:
        errors += 1
for r, n in sorted(counts.items()):
    print(f'{r:35s}  {n:3d} cells')
print(f'_NA / error rows: {errors}')
"
# 期望: 每 recipe 16 cells；_NA = 0（所有 recipe bind_keys 成功）
```

### 4.2 打包 + 下载

```bash
tar czf phase3_data_$(date +%Y%m%d_%H%M%S).tar.gz exp/verdict_factor_judge/data/phase3/
ls -lh phase3_data_*.tar.gz
```

下载到本地 `C:\Users\lzy66\Desktop\fsdownload\` 后告诉 Claude 分析。

### 4.3 inference_ratio 公式（Phase 3 specific）

```python
# Phase 3 warm 全部 fire 在 start_t=0.5（plan §9）
inf_ratio = (n_full_hit * 0.0 + n_warm_start * 0.75 + n_miss * 1.0) / n_eval_verdicts
# 对比 Phase 2 是 0.85（start_t=0.7）
```

baseline 同 Phase 2：
- `random_periodic` 前沿：`exp/random_periodic_gate/analysis/aggregate.csv` (filter `cfg = spatial16_w8_d4`)
- always-WARM @ start_t ∈ {0.30, 0.50, 0.70}：spatial16 SR = {0.942, 0.952, 0.976}, inf = {0.65, 0.75, 0.85}

### 4.4 出图

```bash
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.phase3.plot_pareto
# 输出: exp/verdict_factor_judge/analysis/phase3_pareto.png
```

每个 cell 一个 scatter 点，按 recipe 染色，超过 (random_periodic + always-WARM) Pareto 前沿的 cell 加金圈。同 phase2_layer1_pareto.png 风格。

### 4.5 决策协议（Phase 3 sweep specific）

176 cells × 11 recipe = 17,160 ep。每 cell 标注 (FH_ratio, WS_ratio) 是否 strict-Pareto-positive vs baseline。

| 类别 | 判定 | 行动 |
|---|---|---|
| **Pareto-positive cell** | strict beat (random_periodic + 3 always-WARM) | 候选 winner（特定 recipe / FH / WS 组合） |
| **Pareto-on-frontier** | match 不 beat | 备选 |
| **Pareto-below** | 被 dominate | 淘汰 |

期望发现：
- 每个金圈 recipe 的 (FH_ratio, WS_ratio) 响应面形状：是否有「最佳 cell」单点超过 phase2 layer1 同 recipe 的固定 0.5 threshold
- WS path 的真实价值：phase2 layer1 全 T-FULL（无 WS tier），phase3 第一次给 WS path 充分预算
- NaN-heavy long-window recipes (g2/g3/g7) 在 NaN→0 + 不 fallback 下行为是否劣化

---

## §5 失败处理

- **某 recipe 跑挂**：`run_phase3.py` 已捕异常 → log + continue 下一 recipe；`--resume` 重启时已 done 的 yaml_id（cell 级别）自动跳过。
- **某 recipe `bind_keys` fail-fast** (例如 g2/g3/g7 长窗 + 短 warmup → 某 key < 50 non-NaN sample)：自动写 16 个 `_NA` summary row，不 abort phase。`thresholds.json` 文件含 `error` 字段（无 `cells` 字段），下游聚合按 `_NA` skip。
- **某 server 挂**：那 batch 整体重启，`--resume` 安全。
- **frp 断**（仅 S1-S3）：`tmux capture-pane -pt run3` 看是否在 `Waiting for server at ws://...`；如是，`frpc reload` 后 wait recover。
- **直连 timeout**（S4-S6）：检查 `nc -zv 149.165.151.106 800{1,2,3}`，若不通确认远端机 `serve_policy.py` 还在。

---

## §6 数据布局

```
exp/verdict_factor_judge/
├── config/spatial16/phase3/
│   ├── warmup/                              # 11 个 warmup yaml（spec 一次生成）
│   │   ├── spatial16_w8_d4_phase3_g1_f1b_t_w_fut_d_all__warmup.yaml
│   │   ├── ...
│   │   └── spatial16_w8_d4_phase3_g11_f1a_a_d_curv_only__warmup.yaml
│   ├── eval/                                # 176 个 final eval yaml（runner emit-on-demand 写）
│   │   └── spatial16_w8_d4_phase3_<recipe>__fh<FH>_ws<WS>.yaml
│   └── phase3_manifest.json                 # spec 写的 recipe 索引
├── data/phase3/
│   ├── warmup/<recipe>__warmup.jsonl        # 每 recipe 一个，DumpingJudge 写的 raw factor 序列
│   ├── thresholds/<recipe>__thresholds.json # solver 写的 16 cell 阈值
│   ├── per_step/<yaml_id>.jsonl             # 每 cell 的 verdict-level JSONL（client per_step_log_writer 写）
│   ├── episode_results/<yaml_id>.json       # 每 cell 的 per-episode 成功/失败
│   ├── per_yaml_summary_batch{1..6}.jsonl   # 每 batch 单独 summary，避免并发冲突
│   └── per_yaml_summary.jsonl               # §4.1 合并后 master，176 行（+ 0 个 _NA）
└── analysis/
    └── phase3_pareto.png                    # §4.4 出图
```
