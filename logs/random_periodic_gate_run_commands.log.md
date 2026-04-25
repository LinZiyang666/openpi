> Status: Plan
> Date: 2026-04-24
> Level: L1

# Random & Periodic Gate Sweep 执行命令清单

114 个 yaml 分 **6 个 batch**，6 服务器 / 6 客户端并行。每 cfg 38 yaml 中已做 8 个（`periodic_k10_n{1,10,2,3,5}` + `periodic_k1_n{1,10,2}`）保留在原 batch1/2/3；剩余 30 yaml 按字典序切 2 段各 15，前半并入 batch1/2/3，后半落 batch4/5/6。**每 batch 未做工作 = 15 yaml × 500 ep = 7,500 ep**，均匀。

| Batch | Cfg | yaml 组成 | 未做工作 |
|---|---|---|---|
| `batch1` | `clip_w7_d4` | 8 done + 15 前半 = 23 | 15 |
| `batch2` | `spatial16_w8_d4` | 8 done + 15 前半 = 23 | 15 |
| `batch3` | `max_pool_w3_d5` | 8 done + 15 前半 = 23 | 15 |
| `batch4` | `clip_w7_d4` | 15 后半 | 15 |
| `batch5` | `spatial16_w8_d4` | 15 后半 | 15 |
| `batch6` | `max_pool_w3_d5` | 15 后半 | 15 |

- 前半 15（EARLY）：`periodic_k1_n3, k1_n5, k2_n{1,10,2,3,5}, k5_n{1,10,2,3,5}` 共 12 + `random_p0p05_s{0,1,2}` 3
- 后半 15（LATE）：`random_p0p{10,20,30,50,70}_s{0,1,2}`

runner 按字典序扫 batch 内 yaml；`--resume` 读 `data/batchN/run_state_<slug>.json` 跳过已完成 unit（batch1/2/3 里 8 个已做 yaml 的 run_state 留在远程 `data/batch{1,2,3}/` 不要动，runner 扫到秒跳过）。依赖 `exp/common/data/cache_artifacts/libero_spatial/{clip_vit_b_32,cp1_spatial_pool_16,cp1_max_pool}.pkl`。

Plan：[`logs/random_periodic_gate_plan.log.md`](random_periodic_gate_plan.log.md)。

## 端口映射

前 3 batch 沿用 phase1 约定 frp `155.98.36.13:{8998,8999,9000}`（server 本地端口 `7998/7999/8000`）；**batch4/5/6 用新 server 的公网入口直连**，需填 `<SERVER_N_HOST>` / `<SERVER_N_PORT>`。

| batch | Cfg | Server 入口 | Server 本地端口 | 种子 yaml（字典序首） |
|---|---|---|---:|---|
| `batch1` | `clip_w7_d4` | `155.98.36.13:8998` | `7998` | `periodic_k10_n1.yaml` |
| `batch2` | `spatial16_w8_d4` | `155.98.36.13:8999` | `7999` | `periodic_k10_n1.yaml` |
| `batch3` | `max_pool_w3_d5` | `155.98.36.13:9000` | `8000` | `periodic_k10_n1.yaml` |
| `batch4` | `clip_w7_d4` | `149.165.151.106:8001` | `8001` | `random_p0p10_s0.yaml` |
| `batch5` | `spatial16_w8_d4` | `149.165.151.106:8002` | `8002` | `random_p0p10_s0.yaml` |
| `batch6` | `max_pool_w3_d5` | `149.165.151.106:8003` | `8003` | `random_p0p10_s0.yaml` |

> 公网入口假定 3 个 server 实例跑在同一台公网 IP `149.165.151.106` 上（Floating IP → 内网 `10.0.125.135`），用 3 个不同端口区分。若实际是 3 台独立机器，请把 host 改成各自公网 IP；端口同 server `--port`。security group 需允许 8001/8002/8003 入站。

---

## 前置检查

客户端 host 三 frp 端口健康：

```bash
for p in 8998 8999 9000; do
    printf 'port %s: ' "$p"
    curl -s "http://155.98.36.13:${p}/healthz" || echo
done
```

pkl 齐全（远程 GPU 服务器 home 下）：

```bash
ls exp/common/data/cache_artifacts/libero_spatial/{clip_vit_b_32,cp1_spatial_pool_16,cp1_max_pool}.pkl 2>/dev/null | wc -l
# 期望：3
```

batch 布局确认（repo 内）：

```bash
for b in batch1 batch2 batch3 batch4 batch5 batch6; do
    printf "%s: %d\n" "$b" "$(ls exp/random_periodic_gate/config/$b/*.yaml | wc -l)"
done
# 期望：batch1/2/3 各 23, batch4/5/6 各 15
```

---

## 1. 服务器命令（GPU 服务器侧，6 终端）

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

### Server 4: batch4 (clip_w7_d4, 后半)

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch4/random_p0p10_s0.yaml \
    --env LIBERO \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server 5: batch5 (spatial16_w8_d4, 后半)

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch5/random_p0p10_s0.yaml \
    --env LIBERO \
    --port 8002 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server 6: batch6 (max_pool_w3_d5, 后半)

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/random_periodic_gate/config/batch6/random_p0p10_s0.yaml \
    --env LIBERO \
    --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

---

## 2. 客户端 Runner 命令（LIBERO eval 主机，6 终端）

`libero` 依赖 MuJoCo，**必须**在 `libero_sim` conda env 中跑。

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

### Client 4: batch4 via new server

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch4 \
    --host 149.165.151.106 --port 8001 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

### Client 5: batch5 via new server

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch5 \
    --host 149.165.151.106 --port 8002 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

### Client 6: batch6 via new server

```bash
conda run -p /scratch/zixuans8/libero_sim \
    python -m exp.random_periodic_gate.run_gate_sweep \
    --batch-dir exp/random_periodic_gate/config/batch6 \
    --host 149.165.151.106 --port 8003 \
    --task-suite-name libero_spatial \
    --num-workers 5 \
    --resume
```

每 client 跑 15 个未做 yaml × 500 ep = 7,500 ep；5 worker 并行，单 ep ≈ 5–10 s，估时 ~3–5 h / client。6 client 全并行，总耗时 ≈ 同一值。batch1/2/3 runner 额外会扫 8 个已做 yaml，每个 `send_load_cache_config` + 500 unit resume-skip ≈ 1 s，总开销 < 10 s 可忽略。

---

## 3. 结果位置

每个 batch 输出写到 `exp/random_periodic_gate/data/batchN/`：
- `results.jsonl` — per-ep JSONL
- `run_state_<slug>.json` — per-yaml 断点续跑状态

远程 `data/batch{1,2,3}/` 已有 8 个 yaml 的 `run_state_*.json` + 已写入的 `results.jsonl` 行（phase1 轮次产物），**保留不动**；新 15 yaml 的 state 会增量写入同目录。`data/batch{4,5,6}/` 首次运行时自动创建。

汇总 analysis（所有 client 完成后单机跑）：

```bash
uv run python -m exp.random_periodic_gate.analysis.analyze_gate_sweep \
    --data-root exp/random_periodic_gate/data \
    --baseline-json exp/trajectory_deviation/data/cache_eval_results.json \
    --out-dir exp/random_periodic_gate/analysis
```

analysis glob 所有 `data/batch*/results.jsonl`，按 `(cfg, gate_type, slug)` 聚合，跨 batch 合并无痕。

---

## 4. 关键 Caveat

### 4.1 已做 yaml 保留

batch1/2/3 的 8 个已做 yaml（`periodic_k10_n{1,10,2,3,5}` + `periodic_k1_n{1,10,2}`）**保留在 config 目录**；删除它们会让 runner 无法 resume 跳过且 analysis 找不到对应 results 行的 yaml 元数据源。

### 4.2 `generate_batches.py` 幂等性已打破

本次批次布局是**手动 mv 结果**，不等于 `generate_batches.py --validate` 的产物（generator 仍按旧 3 batch × 38 yaml 规则写 `batch{1..3}`）。**不要重跑 generator**，会覆盖当前布局（把 batch1/2/3 填回 38 yaml 并不创建 batch4/5/6）。要改布局请继续手动 mv 或先改 generator 再跑。

### 4.3 Server `--concurrent` 必选 / `--cache-config` 种子

server 启动时 `--cache-config` 的 yaml 是初始 bundle，runner 后续 WebSocket 热切到 batch 内其他 yaml。每 batch 种子必须是字典序首份（batch1/2/3 是 `periodic_k10_n1.yaml`，batch4/5/6 是 `random_p0p10_s0.yaml`），否则首轮 bundle version 不递增、runner fail-loud。

### 4.4 端口与 phase1 共用

`8998 / 8999 / 9000` 与 `phase1_libero_spatial_llm` 共享，跑本实验前确认 phase1 对应 server 已关。batch4/5/6 用独立公网入口，不与 phase1 冲突。

### 4.5 RandomGate / PeriodicGate 指标语义

RandomGate 行 `inference_ratio = p_inference`（`inference_ratio_source="expected"`），为期望值非实测；PeriodicGate 行由 runner 闭式公式推得精确值（`inference_ratio_source="derived"`）。详见 plan §2.3。

### 4.6 断点续跑粒度

`BaseRunState` unit key = `(yaml_basename, task_id, init_idx)`；state JSON 按 yaml 独立（`run_state_<slug>.json`），不会跨 yaml 污染。`--resume` 应常开。

---

## 5. 执行次序（推荐）

1. 填入 batch4/5/6 三台新 server 的公网 host/port（`<SERVER_N_HOST>` / `<SERVER_N_PORT>`）到 §端口映射表 + §1 / §2 所有命令。
2. 客户端 host 先跑前置 `curl` 健康检查（只检 8998/8999/9000 + 新入口）。
3. 6 台 GPU 服务器各开 1 tmux/screen 终端，启 `serve_policy.py`（§1）。
4. 客户端 host 开 6 tmux/screen 面板，启 `run_gate_sweep.py`（§2）。
5. 监控：每个 yaml ≈ 500 ep × 5–10s ÷ 5 worker ≈ 10–17 min；15 yaml × ≈ 3–5 h / client。
6. 所有 client 结束后，运行 §3 analysis。
