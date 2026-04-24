> Status: Plan
> Date: 2026-04-24
> Level: L1

# Random & Periodic Gate Sweep 执行命令清单

114 个 yaml 分 3 个 batch（每 cfg 一个 batch，每 batch 38 份），三服务器 / 三客户端并行（batch ↔ cfg 一一绑定）：

- `exp/random_periodic_gate/config/batch1/` — 38 份（`clip_w7_d4`，20 periodic + 18 random）
- `exp/random_periodic_gate/config/batch2/` — 38 份（`spatial16_w8_d4`）
- `exp/random_periodic_gate/config/batch3/` — 38 份（`max_pool_w3_d5`）

每 batch 内 runner 按**字典序**扫 yaml；每个 yaml 跑完 libero_spatial 全量 500 ep 再切下一个。Runner 通过 WebSocket `load_cache_config` 为每个 yaml 热切换 cache bundle；server 启动时 `--cache-config` 只是种子。

Runner 绑定到 `exp/random_periodic_gate/run_gate_sweep.py`（plan §5.6）；依赖 `exp/common/data/cache_artifacts/libero_spatial/` 下 `clip_vit_b_32.pkl` / `cp1_spatial_pool_16.pkl` / `cp1_max_pool.pkl` 三份 pkl。

Plan：[`logs/random_periodic_gate_plan.log.md`](random_periodic_gate_plan.log.md)。

## 端口映射

沿用 `phase1_libero_spatial_llm_run_commands.log.md` 的约定：server 本地端口 + 1000 = frp 公网端口（`155.98.36.13`）。**复用前三个 frp tcp 映射 `8998 / 8999 / 9000`**；若 frp 配置已在（phase1 跑过），无须额外申请。

| batch | Cfg | Server 本地端口 | frp 公网端口 | 种子 yaml（字典序首） |
|---|---|---:|---:|---|
| `batch1` | `clip_w7_d4` | `7998` | `8998` | `periodic_k10_n1.yaml` |
| `batch2` | `spatial16_w8_d4` | `7999` | `8999` | `periodic_k10_n1.yaml` |
| `batch3` | `max_pool_w3_d5` | `8000` | `9000` | `periodic_k10_n1.yaml` |

公网 host：`155.98.36.13`。三个 frp 端口与 phase1_libero_spatial_llm 共享，跑此实验前请确认 phase1 对应 server 已关闭，不要端口互撞。

---

## 前置检查

客户端机器确认三个入口健康：

```bash
for p in 8998 8999 9000; do
    printf 'port %s: ' "$p"
    curl -s "http://155.98.36.13:${p}/healthz" || echo
done
```

三个都应返回：`OK`

确认 pkl 齐全（远程 GPU 服务器 home 下）：

```bash
ls exp/common/data/cache_artifacts/libero_spatial/{clip_vit_b_32,cp1_spatial_pool_16,cp1_max_pool}.pkl 2>/dev/null | wc -l
# 期望：3
```

确认 114 YAML 全部存在并通过 schema 校验（repo 内一次性检查）：

```bash
uv run python -m exp.random_periodic_gate.generate_batches --validate
# 期望："validated 114 rendered YAMLs" + "generated 114 YAML files across 3 batches"
```

---

## 1. 服务器命令（GPU 服务器侧，三终端）

每个 batch 一个独立 server；`--cache-config` 传该 batch 的**字典序第一份** yaml 作种子（runner 后续会 WebSocket 热切换到其余 37 份）。三个 batch 的字典序首份均为 `periodic_k10_n1.yaml`。

### Server 1: batch1 (clip_w7_d4), local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch1/periodic_k10_n1.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server 2: batch2 (spatial16_w8_d4), local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch2/periodic_k10_n1.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server 3: batch3 (max_pool_w3_d5), local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch3/periodic_k10_n1.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> 每个 client 只能指向一个独立 server（`load_cache_config` 不能跨 batch 并发），所以三个 batch 的 client 必须各自对应一台 server，不要复用端口。

---

## 2. 客户端 Runner 命令（LIBERO eval 主机，三终端）

`libero` 依赖 MuJoCo 物理引擎，**必须**在 `libero_sim` conda env 中跑；直接 `uv run` 无 libero。使用 `conda run -p /scratch/zixuans8/libero_sim`（绝对路径；按名字装的 env 改成 `-n libero_sim`）。

### Client 1: batch1 via frp port 8998

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch1 \
    --host 155.98.36.13 --port 8998 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

### Client 2: batch2 via frp port 8999

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch2 \
    --host 155.98.36.13 --port 8999 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

### Client 3: batch3 via frp port 9000

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch3 \
    --host 155.98.36.13 --port 9000 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

每个 client 跑 38 yaml × 500 ep = 19,000 episodes；单 episode ≈ 5–10 s × 5 worker 并行 ≈ 6–10 小时 / client。三 client 全并行，总耗时 ≈ 同一值。

---

## 3. 可选：只跑某些 yaml / 调 worker 数

跑更少 worker（MuJoCo EGL 硬上限 5）：

```bash
--num-workers 3
```

仅冒烟某个 batch 的前 2 个 yaml：runner 无 `--runs` 过滤，可直接把 `batch1/` 复制一份到 `/tmp/batch1_smoke/` 只留 2 个 yaml 指进去。

跳过 `load_cache_config` RPC（server 已手动载入时可用，便于单独 dry-run client 路径）：

```bash
--skip-load-cache-config
```

---

## 4. 结果位置

每个 batch 输出写到 `exp/random_periodic_gate/data/batchN/`：

- `results.jsonl` — per-episode JSONL，字段 `cfg / gate_type / gate_params / seed / task_id / init_idx / ep_key / success / total_cycles / num_inference_cycles[(_estimated)] / inference_ratio / inference_ratio_source`（`derived` for PeriodicGate，`expected` for RandomGate）
- `run_state_<slug>.json` — per-yaml BaseRunState 断点续跑状态（`--resume` 读取）

汇总 analysis（所有 client 完成后单机跑）：

```bash
uv run python -m exp.random_periodic_gate.analysis.analyze_gate_sweep \
    --data-root exp/random_periodic_gate/data \
    --baseline-json exp/trajectory_deviation/data/cache_eval_results.json \
    --out-dir exp/random_periodic_gate/analysis
```

产物：
- `aggregate.csv` — per-(cfg, gate_type, slug) 聚合
- `pareto_<cfg>.png` — success_rate vs inference_ratio + baseline 端点（3 张，每 cfg 一张）
- `heatmap_<cfg>_periodic.png` — PeriodicGate cache_len × inference_len 成功率 heatmap（3 张）

---

## 5. 关键 Caveat

### 5.1 YAML 字典序与种子

runner 用 `sorted(batch_dir.glob("*.yaml"))` 迭代；三个 batch 的字典序首份都是 `periodic_k10_n1.yaml`。**server `--cache-config` 必须用该字典序首份**，否则首轮 `send_load_cache_config` 切换时 bundle version 不递增，runner 会 fail-loud。

### 5.2 Server `--concurrent` 必选

`load_cache_config` 控制消息只在 `--concurrent` 模式下被 `websocket_policy_server.py` 识别。漏写 `--concurrent` 则 server 启动时不创建 shared storage，runner 发的 `__ctrl__: load_cache_config` 会被 ignore，cache bundle 始终是启动种子。

### 5.3 端口与 phase1 共用

本实验端口映射 `8998 / 8999 / 9000` 与 `phase1_libero_spatial_llm` 前三条共享。不要在 phase1 还在跑的时候启动本实验 server；反之亦然。若需同时跑，向运维申请新增 tcp 映射并改 §端口映射表。

### 5.4 RandomGate 成本字段语义

JSONL 里 RandomGate 行的 `inference_ratio = p_inference`（字段 `inference_ratio_source="expected"`），不是该 episode 实际触发 inference 的频率（plan §2.3）。analysis 脚本按字段区分 `derived` 与 `expected`，画 Pareto 时 RandomGate 的 x 坐标就是 p，不要误读为实测值。

### 5.5 断点续跑粒度

BaseRunState unit key = `(yaml_basename, task_id, init_idx)`；resume 粒度到单个 episode。`--resume` 应**常开**，避免机器重启后 19,000 ep 全部重跑。state JSON 按 yaml 独立（`run_state_<slug>.json`），不会跨 yaml 污染。

### 5.6 init-state 对齐 baseline

`enumerate_full_suite` 直接读 LIBERO 默认 `task_suite.get_task_init_states(task_id)`（与 `examples/libero/main.py` / Step 1a 相同），`init_idx` 对齐 `exp/trajectory_deviation/data/cache_eval_results.json` 的 `orig_init_state_idx` 列；analysis join 时用 `(cfg, task_id, init_idx)` 三元组即可（见 plan §2.4）。

---

## 6. 执行次序（推荐）

1. 客户端 host 先跑前置 `curl` 健康检查三端口（不通则调 frp）。
2. GPU 服务器开三个 tmux / screen 面板，同时起 3 个 `serve_policy.py`（§1）。
3. 客户端 host 开三个 tmux / screen 面板，同时起 3 个 `run_gate_sweep.py`（§2）。
4. 监控：每个 yaml ~500 ep × 5–10s ÷ 5 worker ≈ 10–17 min；38 yaml × ≈ 6–10 小时 / client，三 client 全并行。
5. 所有 client 结束后，运行 §4 `analyze_gate_sweep.py` 出 Pareto + heatmap。
