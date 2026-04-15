# Warm Start 成功率扫描实验运行指南

> 基于 [`logs/warm_start_sweep_plan.log.md`](../../logs/warm_start_sweep_plan.log.md)，对应 `configs/cache_runs/warm_start_exp/`、`src/openpi/cache/components/judge.py::AlwaysWarmStartJudge`、`exp/cache_experiment/build_clip_cache_artifact.py`。英文版：[warm_start_sweep.en.md](warm_start_sweep.en.md)。

---

## 实验总览

在 cache 永远命中（`gate: always_search` + `judge: always_warm_start`）的前提下，扫 `start_t ∈ {0.7, 0.5, 0.3}` 三档，验证 warm start（从中间 timestep 起部分 denoise）能否把成功率从 always_hit baseline 拉回到接近纯 inference 的水平。

- 3 个 keybuilder × 3 个 start_t = **9 YAML × 500 ep = 4500 episode**（首批不带 CLIP 时 6 × 500 = 3000）
- **不重跑 baseline**：B0（inference 上界）/ B1（always_hit 下界）直接复用 [trajectory_deviation](trajectory_deviation.md) Step 1a 的 `data/deviation_experiment/cache_eval_results_*.json`。
- 主输出：`data/warm_start_exp/results/cache_eval_results.json`（9 配置合并） + `data/warm_start_exp/timing/<cfg>/*.csv` 延迟指标。

流水线：

```
Step 0   启动 GPU server（一次，带 warm start 初始 bundle）
Step 1   重建 3 份 warm artifact（max_pool / spatial16 / clip → libero_spatial_warm/*.pkl）
Step 2   Smoke pass（单 YAML × 单 task × 5 ep，验 WARM_START 通路）
Step 3   全量（3 server 并发，每 server 跑同一 keybuilder 下 3 档 start_t）
Step 4   分析 + 出图（exp/cache_experiment/analyze_warm_sweep.py）
```

## 网络拓扑

与 [trajectory_deviation.md](trajectory_deviation.md) 完全一致：评估端跑 `exp/cache_experiment/run_cache_experiments.py`，通过 frp 连回 GPU server。server 用 `--concurrent + --cache_config` 启动后，driver 每遍历一份 YAML 都发 `load_cache_config` 控制消息切换 bundle。

---

## 前置条件

1. 两端已 `GIT_LFS_SKIP_SMUDGE=1 uv sync`；评估端装好 `libero_sim` conda 环境。
2. `data/db/libero_cache/libero_spatial/*.h5`（50 episode）就位——Step 1 的 artifact 源数据。
3. **Trajectory Deviation Step 1a 已跑过**，`data/deviation_experiment/cache_eval_results_{clip,maxpool,spatial16}.json` 已落盘。warm start 直接读它们做对照组。
4. `data/warm_start_exp/baseline_failures.json` 已产出（把三份 Step 1a JSON 按 `config_id` split 聚合成 `{cfg: {inference: {fails: [init_idx,...]}, always_hit: {...}}}`）。
5. Pi0.5 LIBERO checkpoint：默认 `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`。

---

## Step 0：启动 GPU server

一份 server 服务一个 keybuilder 的 3 档 start_t。方案 A 共起 3 个 server（端口 8000/7999/7998，frp 映射 9000/8999/8998）。每个 server 的初始 `--cache_config` 用对应 keybuilder 的 **AlwaysSkip inference bundle**（driver 首次跑 warm YAML 时会自动 `load_cache_config` 切到 warm bundle）。

每个 server 用三个终端分别起（三卡分别独占）。local port 与 frp 外网端口映射表：

| Slot | 初始 `--cache_config` | GPU | server `--port` | frp 外网端口 |
|------|----------------------|-----|-----------------|-------------|
| 1 | `deviate_exp/inference_max_pool_w3_d5.yaml` | 0 | 8000 | 9000 |
| 2 | `deviate_exp/inference_spatial16_w8_d4.yaml` | 1 | 7999 | 8999 |
| 3 | `deviate_exp/inference_clip_w7_d4.yaml` | 2 | 7998 | 8998 |

Slot 1（max_pool, GPU 0, port 8000/9000）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Slot 2（spatial16, GPU 1, port 7999/8999）：

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config configs/cache_runs/deviate_exp/inference_spatial16_w8_d4.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Slot 3（clip, GPU 2, port 7998/8998）：

```bash
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> 不加 `--collect-images`：CLIP slot 由 `serve_policy.py` 的 `need_images = ... or key_builder.type == "clip"` 在启动时自动打开；max_pool / spatial16 slot 的 KeyBuilder 不读原图，开着反而每步多一次 `extract_valid_images` 拷贝。前提是每台 server 锁定一种 keybuilder、不会 `load_cache_config` 跨切到 clip——warm_start_sweep 拓扑满足这个约束。

单卡显存不够三 server 共存时退方案 B（仅跑 max_pool + spatial16，CLIP 延后）。

健康检查（frp 外网）：

```bash
curl http://155.98.36.13:9000/healthz   # max_pool
curl http://155.98.36.13:8999/healthz   # spatial16
curl http://155.98.36.13:8998/healthz   # clip
```

三条都返回 `OK` 才能进入 Step 1。

---

## Step 1：构建 warm artifact

旧 `data/cache_artifacts/libero_spatial/*.pkl`（4 月 9 日构建）没写 `payload.intermediates`，warm start 必然被 Orchestrator 降级为 MISS。**必须独立重建到 `libero_spatial_warm/`**（不覆盖 trajectory_deviation 依赖的旧路径）。

### 1.1 max_pool + spatial16（CPU，20 workers 并行）

```bash
mkdir -p data/cache_artifacts/libero_spatial_warm

uv run python exp/cache_experiment/build_in_memory_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --builder-type cp1_max_pool \
    --output data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl

uv run python exp/cache_experiment/build_in_memory_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --builder-type cp1_spatial_pool_16 \
    --output data/cache_artifacts/libero_spatial_warm/cp1_spatial_pool_16.pkl \
    --reducer-type spatial_pool --output-tokens 16
```

50 episode × 1018 step，单机 CPU 约 1 分钟完成；输出 ~49 MB（max_pool）/ ~407 MB（spatial16）。

### 1.2 CLIP（GPU 或 CPU；**必须显式 `--fields`**）

```bash
uv run python exp/cache_experiment/build_clip_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --fields vision_0,vision_1,prompt_emb,robot_state \
    --device cuda \
    --output data/cache_artifacts/libero_spatial_warm/clip_vit_b_32.pkl
```

> `--fields` 必须写全 4 项：YAML `clip_w7_d4_warm_t*.yaml` 的 `keys` 启用了 `vision_0 / vision_1 / prompt_emb / robot_state`，`backend.vector_dims` 相应声明 4 项。builder 默认只含 `vision_0 / robot_state`，缺项会让 `InMemoryBackend.load_artifact()` 启动期报 `vector_dims` 不一致。

GPU 版 batch=64，ViT-B-32 on CUDA 约 1 分钟；产物 ~194 MB。显存 < 2 GB 可退到 `--device cpu`（约 10 分钟）。

### 1.3 intermediates 校验（必跑）

```bash
uv run python - <<'PY'
import pickle
expected = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for name in ["cp1_max_pool","cp1_spatial_pool_16","clip_vit_b_32"]:
    with open(f"data/cache_artifacts/libero_spatial_warm/{name}.pkl","rb") as f:
        obj = pickle.load(f)
    e = obj["entries"][0]
    keys = sorted(e.payload.intermediates.keys())
    assert keys == expected, (name, keys)
    assert e.payload.denoising_num_steps == 10
    full = sum(1 for x in obj["entries"]
               if x.payload.intermediates and len(x.payload.intermediates) == 9)
    print(f"{name}: {len(obj['entries'])} entries, {full} full intermediates")
PY
```

三份都应输出 `1018 entries, 1018 full intermediates`；否则 Step 2 的 smoke 一定失败，不要继续往下跑。

---

## Step 2：Smoke pass（单 YAML × 单 task × 5 ep）

目标：最小代价验 (i) YAML 被 server 正确加载、(ii) WARM_START 路径实际触发、(iii) Orchestrator 未降级。

```bash
uv run python exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/max_pool \
    --runs 2 \
    --task-ids 0 \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 9000 \
    --episodes-per-run 5 \
    --num-workers 1 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_smoke.json
```

参数 & 排序：

- `--runs 2` 按 `sorted(yaml_dir.glob("*.yaml"))`（lexicographic）命中 `max_pool_w3_d5_warm_t0.5.yaml`——`t0.3 → t0.5 → t0.7` 是字典序。
- `--task-ids 0` + `--episodes-per-run 5` + `--num-workers 1` = 5 个串行 episode，最快通路。

**通过条件**（server log 和 driver 产出一起看）：

1. `configs/cache_runs/warm_start_exp/max_pool/cache_eval_results.json` 出现 `config_id == "max_pool_w3_d5_warm_t0.5"`，正好 5 条 record。
2. Server log `grep -cE "judge: WARM_START"` **≥ 1**（WARM_START 路径实际生效）。
3. Server log `grep -c "WARM_START payload incomplete"` **== 0**（Orchestrator 没把 WARM_START 降级为 MISS；若不为 0 回 Step 1.3）。
4. `data/warm_start_exp/timing/max_pool_w3_d5_warm_t0.5/` 下至少一份 `timing_task_*.csv`（timer CSV 通路有效）。

可选：3 档加载验证——`--runs 1-3 --episodes-per-run 1` 跑一遍，server log 出现 3 次 `Cache bundle updated to v*`。

---

## Step 3：全量（3 server 并发）

3 个终端并行，每个 client 绑一个 server，每个 server 服务同一 keybuilder 的 3 档 YAML。

```bash
# Terminal 1 — max_pool (frp 9000)
uv run python exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/max_pool \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 9000 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_max_pool.json

# Terminal 2 — spatial16 (frp 8999)
uv run python exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/spatial16 \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 8999 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_spatial16.json

# Terminal 3 — clip (frp 8998)
uv run python exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/clip \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 8998 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_clip.json
```

> 三个 client 的 `--state-path` **必须不同**（否则 RunState 互相覆盖）。`--num-workers 5` × 3 client = 15 个 libero env 子进程——评估端 `nproc < 16` 时把 num-workers 调到 3。

Driver 在每份 warm YAML 前发 `send_load_cache_config`，server 会重新 `build_shared_storage` → `InMemoryBackend.load_artifact(preload_path)`，等于每份 YAML 都会重新 pickle.load 一次 artifact（max_pool < 1s，spatial16 数秒，CLIP 1–2s；对分钟级的 500 ep 跑量可忽略）。

跑完归档（防止被下一次覆盖）：

```bash
mkdir -p data/warm_start_exp/results
for cfg in max_pool spatial16 clip; do
    src=configs/cache_runs/warm_start_exp/$cfg/cache_eval_results.json
    [ -f "$src" ] && cp "$src" data/warm_start_exp/results/cache_eval_results_${cfg}.json
done
uv run python - <<'PY'
import json, glob
out = []
for fn in sorted(glob.glob("data/warm_start_exp/results/cache_eval_results_*.json")):
    out.extend(json.load(open(fn)))
json.dump(out, open("data/warm_start_exp/results/cache_eval_results.json","w"), indent=2)
print("merged", len(out), "records")
PY
```

9 × 500 = **4500 record**（只跑 max_pool + spatial16 时 3000）。若 `--num-workers` 撞 GPU/CPU 墙，结果里会出现 `attempt > 1` 的 retry record，sanity 时先过滤这些再看 success_rate。

---

## Step 4：分析

`exp/cache_experiment/analyze_warm_sweep.py` 读 `data/warm_start_exp/results/cache_eval_results.json` + `data/warm_start_exp/baseline_failures.json`，产出：

| 产物 | 含义 |
|------|-----|
| `success_rate_sweep.png` | 横轴 start_t，纵轴 success_rate；加 B0 / B1 两条水平虚线 |
| `recovery_on_b1_failure.png` | `|warm_pass ∩ B1_fail| / |B1_fail|`——救回多少 always_hit 失败的 init |
| `incurred_loss.png` | `|warm_fail ∩ B0_pass| / |B0_pass|`——warm 多杀了多少原本 inference 能过的 init |
| `mean_step_latency.png`（副图） | 来自 timer CSV 的 `stage3_warm / stage3_flow / cp1_sum` 均值 |
| `summary.csv` | `(cfg, start_t, n_total, n_success, success_rate, recovery_rate, incurred_loss, p_value)` |
| `failure_intersection.csv` | 每 `(cfg, start_t)` warm-fail 与 B1 三 cfg 交集（51 个"硬骨头"）的重叠数 |

**判读优先级**：先看 `recovery_rate` 单调性（start_t 递增应 → recovery 递增），再对比 `incurred_loss` 曲线——若 `start_t=0.7` 的 `incurred_loss` 过高，说明"过激 warm"本身在伤害原本能过的 init，需要进一步拆延迟曲线。

---

## 关键参数速查

| 位置 | 参数 | 默认 | 何时改 |
|------|------|------|--------|
| YAML `judge.start_t` | 0.3 / 0.5 / 0.7 | — | 必须 ∈ {0.1..0.9}；扩展到 0.1 / 0.9 时 pkl 不用重建（已含 9 档） |
| YAML `timer.output_csv_dir` | `data/warm_start_exp/timing/<cfg>_warm_t<st>` | — | 不能留 null，否则延迟副图没数据 |
| YAML `backend.in_memory.preload_path` | `data/cache_artifacts/libero_spatial_warm/<builder>.pkl` | — | 换数据集（libero_10 / libero_object...）时改路径 |
| `run_cache_experiments.py --runs` | 全部 | — | smoke 用 `--runs 2` 对应字典序 `t0.5` |
| `run_cache_experiments.py --num-workers` | 5 | 5 | 评估端 CPU < 16 时调 3；server 侧 OOM 时调 1 |

## 常见问题

- **WARM_START 降级为 MISS**：server log 出现 `WARM_START payload incomplete`——`orchestrator.py` 已把这条升到 warning，INFO 日志可见。根因永远是 Step 1 的 artifact 没写 `intermediates`，回 Step 1.3 校验。
- **CLIP `vector_dims` 启动报错**：`build_clip_cache_artifact.py` 必须带 `--fields vision_0,vision_1,prompt_emb,robot_state`；少了任何一项都会跟 YAML 的 4-field `vector_dims` 冲突。
- **YAML `--runs N` 命中错文件**：当前 9 份文件名 lexicographic 顺序就是 `t0.3 → t0.5 → t0.7`；不要改名加 `01_` / `02_` 前缀，否则 trajectory_deviation 那套下游 state-path 路径也会跟着漂。
- **3 client 并发 libero env 打爆评估端 CPU**：`--num-workers 5 × 3 = 15` 个 libero 子进程；`nproc < 16` 时调到 3。
