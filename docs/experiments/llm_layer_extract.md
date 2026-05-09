# CP1 LLM Layer Extract 实验运行指南

> **本文档**：端到端 experiment runbook —— 数据采集 → artifact build → YAML 编写 → 跑实验 → 分析。
>
> **组件 API 参考**：[`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md)（KeyBuilder 类、reducer 选择、YAML 字段语义）
>
> **设计文档**：[`../../logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/cp1_llm_layer_extract_key_builder_plan.log.md)
>
> **artifact 布局规则**：[`artifact_layout.md`](artifact_layout.md)

---

## 1. 网络拓扑

与其他 cache 实验同构，沿用 frp 隧道远程推理：

```
┌───────────────────────────┐         frp 隧道           ┌───────────────────────────┐
│  GPU 服务器 (无公网IP)      │ ◄─────────────────────── │  LIBERO 评估端              │
│  serve_policy.py            │   155.98.36.32:9000       │  run_cache_experiments.py    │
│  监听 0.0.0.0:8000          │   → localhost:8000         │  examples/libero/main.py     │
└───────────────────────────┘                            └───────────────────────────┘
```

- **GPU 服务器**：跑 PI0Pytorch 推理，必须有 ≥ 16 GB VRAM
- **评估端**：跑 LIBERO benchmark + 实验控制器
- 两端都需本仓库代码 + `uv` 环境
- 公网入口写 `155.98.36.32:9000`，**别默认 localhost**

---

## 2. 前提条件

| 资源 | 检查命令 / 路径 |
|------|----------------|
| Pi0.5 checkpoint（含 `model.safetensors`） | 默认 `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch` |
| `uv sync` 完成（两端） | `GIT_LFS_SKIP_SMUDGE=1 uv sync` |
| LIBERO benchmark 数据（评估端） | 见 [`../deployment/libero.md`](../deployment/libero.md) |
| frp 隧道工作 | `curl http://155.98.36.32:9000/healthz` |

> **GPU VRAM**：服务器需要 5–10 GB 用于 model；artifact build 期间额外 ~3 GB 用于 layer-N forward 激活；推理期间 KeyBuilder 单步层 forward ~1–2 ms（A100/4090 bf16）。

---

## 3. 端到端流程概览

```
┌─────────────────┐   Step 1     ┌─────────────────┐   Step 2     ┌─────────────────┐
│  采集 HDF5 数据   │ ──────────► │  build .pkl    │ ──────────► │  Artifact 文件   │
│ (serve --collect│             │ artifact (cp1_  │             │ /cache_artifacts │
│  + LIBERO env)   │             │  llm_layer_*)   │             │  /<task>/*.pkl  │
└─────────────────┘             └─────────────────┘             └─────────────────┘
                                          │
                                          ▼
┌─────────────────┐   Step 4     ┌─────────────────┐   Step 3     ┌─────────────────┐
│  分析结果        │ ◄────────── │  跑实验           │ ◄────────── │  写 YAML config │
│ (analyze_cache_ │             │ (run_cache_     │             │ (cp1_llm_*.yaml)│
│  results.py)     │             │  experiments.py │             │                 │
└─────────────────┘             └─────────────────┘             └─────────────────┘
   Step 5
```

---

## Step 1 — 数据采集（如已有 HDF5 可跳过）

如果你已经有 `exp/common/data/db/libero_cache/<task_suite>/*.h5`（与 cp1_mean_pool 等其他 builder 共用同一份原始数据），**跳到 Step 2**。否则按下面采集。

### 1.1 GPU 服务器：启动带 `--collect` 的 serve_policy

```bash
uv run scripts/serve_policy.py \
    --collect \
    --collect_dir exp/common/data/db/libero_cache \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

**关键 flag**：
- `--collect`：开启 HDF5 落盘（每 episode 一文件）
- `--collect_dir`：写入根目录；最终路径是 `<root>/<experiment_name>/episode_*.h5`

**采集行为**：每次推理时通过 forward hook 抓取：
- `vision_0/1/2`: SigLIP 256 × 2048 token（per camera）
- `prompt_emb`: lang token 200 × 2048（已 padding）
- `robot_state`: 32-d
- `clean_action / noise_action_*`: action expert 中间结果
- `attrs.task / attrs.success`: episode 元数据

数据采集详细机制见 [`../data_collection/guide.md`](../data_collection/guide.md)。

### 1.2 评估端：跑 LIBERO 任务采集 episodes

参考 [`cp1_cache.md`](cp1_cache.md) §3 的具体命令；典型用法是用 `examples/libero/main.py` 直接驱动或经过 `run_cache_experiments.py` 自动批量。50 个成功 episodes 已经够 baseline 实验。

### 1.3 验证 HDF5 schema

```bash
uv run python -c "
import h5py, os, sys
d = sys.argv[1]
fp = os.path.join(d, sorted(os.listdir(d))[0])
f = h5py.File(fp, 'r')
print('attrs:', dict(f.attrs))
g = f[sorted(k for k in f.keys() if k.startswith('step_'))[0]]
for k in g.keys():
    o = g[k]
    print(' ', k, getattr(o, 'shape', '(group)'),
          getattr(o, 'dtype', ''))
" exp/common/data/db/libero_cache/libero_spatial
```

预期输出（与 G1 plan §6.2 验证一致）：
```
attrs: {'episode_id': 0, 'experiment_name': 'libero_spatial',
         'num_steps': 16, 'success': True, 'task': '...', 'timestamp': '...'}
  clean_action  (10, 32)  float32
  prompt_emb    (200, 2048) float16    ← 已 padding 到 max_token_len=200
  robot_state   (32,)     float32
  vision_0      (256, 2048) float16
  vision_1      (256, 2048) float16
  vision_2      (256, 2048) float16
  ...
```

> **关键**：HDF5 不存 `lang_masks`。Step 2 的 builder 会从 `attrs['task']` + `robot_state` 用 `PaligemmaTokenizer` **重新 tokenize 还原 lang_masks**（确定性算法）。所以采集端不用改任何东西。

---

## Step 2 — 构建 Cache Artifact

### 2.1 命令模板

```bash
mkdir -p exp/common/data/cache_artifacts/<task_suite>

uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/<task_suite> \
    --builder-type cp1_llm_layer_extract \
    --extract-layer 0 \
    --prefix-reducer-type prefix_mean_pool \
    --checkpoint-dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch" \
    --config-name pi05_libero \
    --device cuda \
    --output exp/common/data/cache_artifacts/<task_suite>/cp1_llm_l0_meanpool.pkl
```

### 2.2 关键参数

| 参数 | 取值 | 说明 |
|------|------|------|
| `--builder-type` | `cp1_llm_layer_extract` | 选本 builder（区别于 SigLIP-only 的 cp1_mean_pool 等） |
| `--extract-layer` | `0..17` | gemma_2b layer 索引；首版 baseline 选 0；sweep 多份 yaml |
| `--prefix-reducer-type` | `prefix_mean_pool` / `per_modality_mean_pool` / `per_modality_max_pool` / `per_modality_spatial_pool_16` / `per_modality_spatial_pool_4` | A=单 key 教授原意；B=四模态独立 key (mean / max / 16-token 4×4 spatial / 4-token 2×2 spatial)。两档 spatial 对齐 legacy `cp1_spatial_pool_{16,4}`。详见 [`../cache/llm_layer_extract.md §3.2`](../cache/llm_layer_extract.md)|
| `--checkpoint-dir` | PI0Pytorch 权重目录 | 必须**与采集 HDF5 时同 checkpoint**，否则 self-check fail |
| `--config-name` | `pi05_libero`（或类似） | 用于加载 TrainConfig |
| `--device` | `cuda`（默认） | CPU 模式仅供 smoke test |
| `--workers` | 自动强制 `-1` | 模型 5–10 GB VRAM，无法用 ProcessPool 并行 |

### 2.3 启动 self-check（自动）

构建首步会跑一次 tokenizer self-check：用首 episode 首步重 tokenize → `embed_language_tokens × √2048` → 与 HDF5 `prompt_emb` 在 mask=True 位置 `allclose(rtol=1e-2, atol=1e-2)`。

- ✅ 通过 → 日志 `Tokenizer self-check passed (max abs diff: ...)`，继续构建
- ❌ fail → **立即 abort**，附错误：`Tokenizer/embed self-check failed (max abs diff: ...)`，提示检查 checkpoint / tokenizer 来源

self-check 内部实现见 `exp/common/build_in_memory_cache_artifact.py::_self_check_tokenizer_consistency`。

### 2.4 批量构建（layer sweep + reducer 对比）

```bash
TASK=libero_spatial
CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
ART_DIR=exp/common/data/cache_artifacts/$TASK
mkdir -p $ART_DIR

# Layer sweep × reducer 矩阵
for L in 0 2 5; do
  for R in prefix_mean_pool per_modality_mean_pool; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/$TASK \
        --builder-type cp1_llm_layer_extract \
        --extract-layer $L \
        --prefix-reducer-type $R \
        --checkpoint-dir "$CKPT" \
        --config-name pi05_libero \
        --device cuda \
        --output $ART_DIR/cp1_llm_l${L}_${R}.pkl
    echo "Done: layer=$L reducer=$R"
  done
done
```

构建 50 个 episodes（典型 ~5000 steps）单组 ≈ 5–8 分钟（model load ~30s + 每步 ~1–2 ms）。layer 越大越慢（线性）。

### 2.5 Artifact 元数据查验

```bash
uv run python -c "
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    art = pickle.load(f)
print('builder:', art['key_builder_type'])
print('vector_dims:', art['vector_dims'])
print('entries:', len(art['entries']))
print('reducer_params:', art.get('reducer_params'))
" exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_prefix_mean_pool.pkl
```

预期 `reducer_params` 含 `extract_layer / prefix_reducer_type / apply_final_norm / checkpoint_dir / config_name / tokenizer_class / tokenizer_source / tokenizer_max_len`。Provenance 用于将来排查 online/offline 漂移。

---

## Step 3 — 编写 YAML Config

### 3.1 模板 A — `prefix_mean_pool`（教授原意 baseline，单字段）

```yaml
# exp/<your_exp>/config/cp1_llm_l0_meanpool.yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000

key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0                # 必须与 artifact 一致
  prefix_reducer:
    type: prefix_mean_pool

keys:
  vision_0:    { enabled: true,  weight: 1.0 }    # 唯一允许的 vision 字段
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }    # 必须 false
  robot_state: { enabled: true,  weight: 0.5 }    # 可选

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048                # gemma_2b width，必须 = 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_prefix_mean_pool.pkl

checkpoints:
  cp1:
    enabled: true                 # cp1_llm_layer_extract 要求 cp1 启用
    gate:
      type: always_search
    judge:
      type: threshold
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      step_filter: all
      rrf_k: 60
  cp3:
    enabled: true
    gate:
      type: always_search
    judge:
      type: threshold
      threshold: 0.95
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      step_filter: all

write_policy:
  type: never                     # 实验阶段只读，不让本次推理污染 artifact
```

### 3.2 模板 B — `per_modality_mean_pool`（保留模态消融）

差异只在 `key_builder` / `keys` / `backend.vector_dims` / `preload_path`：

```yaml
key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0
  prefix_reducer:
    type: per_modality_mean_pool

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: true,  weight: 1.0 }
  vision_2:    { enabled: true,  weight: 1.0 }
  prompt_emb:  { enabled: true,  weight: 0.5 }
  robot_state: { enabled: true,  weight: 0.5 }

backend:
  vector_dims:
    vision_0:    2048
    vision_1:    2048
    vision_2:    2048
    prompt_emb:  2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_per_modality_mean_pool.pkl
```

### 3.3 Config 校验规则（启动失败的常见原因）

启动前 `validate_cache_config()` 会做这些 cross-check（详见 `src/openpi/cache/config.py:546+`）：

| 校验 | 错误现象 | 修复 |
|------|---------|------|
| `extract_layer` ∈ [0, 17] | `extract_layer=18 out of range` | 改 layer |
| `prefix_reducer.type` 合法 | `prefix_reducer.type 'X' unknown` | 五选一 |
| `prefix_mean_pool` + 启用 vision_1/2/prompt_emb | `... would never be populated` | 关掉这些字段 |
| `vector_dims.<f>` 与 reducer 输出维不匹配 | `does not match prefix_reducer output dim N` | 按 reducer 修改：mean/max → 2048；spatial_16 vision → 32768；spatial_4 vision → 8192；spatial_* prompt → 2048 |
| `cp1.enabled = true` | `requires checkpoints.cp1.enabled=true` | 启用 cp1 |
| `in_memory.preload_path` 缺 | `requires backend.in_memory.preload_path` | 指向 Step 2 产物 |

---

## Step 4 — 运行实验

### 4.1 GPU 服务器：启动 serve_policy 带 cache config

```bash
uv run scripts/serve_policy.py \
    --cache_config exp/<your_exp>/config/cp1_llm_l0_meanpool.yaml \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

启动日志应见：
```
CP1LLMLayerExtractKeyBuilder: attached to model (depth=18, extract_layer=0)
Cache config loaded: backend=in_memory, key_builder=cp1_llm_layer_extract, ...
```

> 如果你需要并发多 connection 跑多 cache config，改用 `--concurrent` 模式。详见 [`cp1_cache.md`](cp1_cache.md) §5。

### 4.2 评估端：批量驱动 cache 实验

如果你只跑一组 yaml，直接用 `examples/libero/main.py`。如果你要跑一个 yaml 目录里的多组 config（典型 layer × reducer sweep），用 `run_cache_experiments.py`：

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/<your_exp>/config \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --log-dir exp/<your_exp>/data \
    --state-path exp/<your_exp>/data/experiment_state.json \
    --resume                       # 如果上次跑断了
```

**关键 flag**（详见 `exp/common/run_cache_experiments.py::_build_arg_parser`）：
- `--yaml-dir`：每个 `.yaml` 是一组实验配置
- `--episodes-per-run`：每个 task 跑多少 episode
- `--num-workers`：并行 LIBERO instances（每个 worker 一个 GPU process）
- `--task-ids`：可只跑指定 task（默认整个 suite）
- `--resume`：状态文件存在时继续（per-task granularity）
- `--state-path` / `--log-dir`：**显式指向 `data/`**，避免污染 `config/`

### 4.3 单组 YAML 快速 smoke

```bash
uv run examples/libero/main.py \
    --host 155.98.36.32 --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 1 \
    --task-ids 0 \
    --num-workers 1 \
    --seed 42 \
    --cuda-visible-devices 0
```

---

## Step 5 — 结果分析

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/<your_exp>/data/experiment_state.json \
    --output exp/<your_exp>/analysis/cp1_llm_l0_meanpool_summary.json
```

产物 JSON 含 per-task / per-config success rate 和 cache hit rate。

如果你跑了 layer × reducer 对比，建议在 `analysis/` 目录写一个简单 plot 脚本（参考 `exp/common/analysis/<其他实验>/plot_results.py`），从 summary JSON 出 figure。

---

## 6. Verify：在线 / 离线 parity（强烈建议每次新 artifact 后跑一次）

新 artifact 是否真的与在线推理产生相同的 query keys？跑 manual parity test：

```bash
PI05_CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch" \
PI05_CONFIG_NAME=pi05_libero \
uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v
```

测试断言（详见 [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) §6）：
1. 在线 / 离线 `prefix_pad_masks` 完全一致
2. 在线 / 离线 layer-N hidden state 在 mask=True 位置 `allclose(rtol=1e-2, atol=1e-2)`
3. 在线 / 离线 `KeyBuilder.build()` 输出 `allclose`
4. KeyBuilder 单层 replay 与 HF `output_hidden_states[1]` 等价
5. 真实模型 + InMemoryBackend 的 `collect→build→search→fetch` 链路对两条不同 obs 各自命中正确 entry

任何 fail 都说明离线契约被破坏（最常见：`--checkpoint-dir` 与采集时不一致）。**Verify 阶段必须本地跑通才能信任本次实验结果。**

---

## 7. 常见问题

### Q: 启动 self-check fail 了怎么办？

报错形如 `Tokenizer/embed self-check failed (max abs diff: 12.34)`。最常见原因：

1. `--checkpoint-dir` 与采集 HDF5 时用的 checkpoint 不一致
2. PaligemmaTokenizer 下载源被替换或本地缓存损坏
3. HDF5 是用其他模型或旧版 tokenizer 采集的

排查路径：先核对 artifact `reducer_params` 元数据（`tokenizer_source` / `checkpoint_dir`）是否与采集时一致；再查 `gs://big_vision/paligemma_tokenizer.model` 本地缓存。

### Q: VRAM 不够怎么办？

- artifact build：默认 cuda；如果 GPU 紧张，让其他进程让出，或用 `--device cpu`（**不推荐**，每步 forward 几秒）
- 推理服务：Pi0.5 + cache 系统 ≈ 6–8 GB；如果 OOM，降 `--num-workers` 或换更大 GPU

### Q: extract_layer 选多少？

首版起点 `0`（cost 占 Stage 2 的 5.5%）。如果 baseline hit 质量不达标：
- `2`：~16.7% Stage 2，3 轮跨模态 attention
- `5`：~33.3% Stage 2，6 轮（性价比拐点）
- 不建议 ≥ 8（接近 ROI 拐点；不如直接复活 CP2）

### Q: 想跑跨任务实验？

prefix_mean_pool 的设计动机就是多任务 / 跨任务泛化。建议步骤：
1. 用混合多任务 episodes 采集 HDF5
2. build 一份 unified artifact
3. 跑评估时换不同 task suite 看 hit rate / SR transfer

### Q: 在线推理要慢多少？

A100 / 4090 bf16 上单步 0.5–2 ms（layer 0）。Pi0.5 总推理 ~50 ms，相对增加 ~2–4%，可接受。

### Q: 能复用 cp1_temporal_prune 的 reducer 吗？

不能。两边的 reducer 协议输入不同，强行复用会破坏 padding 语义。详见 [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) §9。

### Q: artifact build 能并行加速吗？

不能，`--workers` 被强制设为 `-1`（serial）。每个 worker 都要加载 5–10 GB 的模型，VRAM 装不下。如果你有多块 GPU，可以**手动并行不同 task suite 的 build**（每块 GPU 跑一个 build_in_memory_cache_artifact 进程，加 `CUDA_VISIBLE_DEVICES`）。

---

## 8. 与其他 Cache 实验的关系

| 实验 | 关键差异 |
|------|---------|
| [cp1_cache.md](cp1_cache.md) | SigLIP token pool（无 LLM forward）、4 种 reducer。**最快但 key 表征最弱**。 |
| [temporal_prune.md](temporal_prune.md) | SigLIP + 跨步 token pruning。**有状态**。 |
| **本实验** | LLM layer-N hidden state（**带跨模态融合**）、2 种 reducer。**慢一点但 key 表征最强**。 |
| [warm_start_sweep.md](warm_start_sweep.md) | warm start hit 阈值 sweep。本实验配出 artifact 后可与之组合。 |

---

## 9. 参考文件清单

| 文件 | 作用 |
|------|------|
| [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) | 组件 API / YAML 字段语义 |
| [`../../logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/cp1_llm_layer_extract_key_builder_plan.log.md) | 设计文档（含 G1/G2 review 历史） |
| `exp/common/build_in_memory_cache_artifact.py` | Step 2 builder 入口 |
| `exp/common/run_cache_experiments.py` | Step 4 实验驱动 |
| `exp/common/analyze_cache_results.py` | Step 5 结果聚合 |
| `tests/cache/test_llm_layer_extract_parity.py` | Verify 阶段 parity test |
| [`artifact_layout.md`](artifact_layout.md) | `exp/<exp>/{config,data,analysis}/` 布局规则 |
| [`../data_collection/guide.md`](../data_collection/guide.md) | Step 1 数据采集机制 |
