# Plan: 基于 libero_10 采样 init 批量采集推理轨迹并构建 InMemory Cache Artifact

**Status**: Plan
**Level**: L1（`exp/` 流水线执行，不涉及源码修改）
**Date**: 2026-04-20
**Owner**: Ziyang Lin（执行由 Claude Execution Authority 辅助）

> L1 按 [`WORKING_AGREEMENT.md` §2.1](../WORKING_AGREEMENT.md#21-work-classification) 无 G1/G2 强制 Gate；本 plan 仅用于流程固化与可复现性记录。若后续被验证扩展为实验基线（例如作为某实验的默认缓存 artifact），届时由项目 Owner 决定是否升级为 L2 并补 G1。

---

## 1. 目标与范围

- **输入源**：`exp/common/data/db_init/libero_cache/libero_10/*.init`。共 11 个任务，每任务 5 个采样 init 状态，由 `exp/common/data/db_init/sample_cache.py` 预先产出。
- **中间产物**：针对每个 `(task, init)` 组合驱动 LIBERO 仿真推理，服务端以 `--collect` 模式落 HDF5 到 `exp/common/data/db/libero_cache/libero_10/episode_*.h5`（与既有 `libero_spatial` 的 h5 库同目录惯例，见 `docs/experiments/cp1_cache.md`、`docs/experiments/warm_start_sweep.md`、`exp/common/build_clip_cache_artifact.py`）。
- **最终产物**：基于上述 h5 构建 InMemoryBackend pkl artifact，落到 `exp/common/data/cache_artifacts/libero_10/<builder_type>.pkl`（遵循 [`docs/experiments/artifact_layout.md`](../docs/experiments/artifact_layout.md) §1–§3）。
- **不改动源码**：只调用既有 CLI，不新增/修改 `src/`、`scripts/`、`examples/`、`exp/` 下任何 `.py`。

## 2. 与现有组件的集成点

| 组件 | 路径 | 本计划用法 |
|------|------|-----------|
| Policy server | `scripts/serve_policy.py` | `--collect --collect_dir ./exp/common/data/db/libero_cache` 触发 h5 落盘到 `db/libero_cache/<experiment_name>/` |
| LIBERO 客户端 | `examples/libero/main.py` | `--args.init_states_dir` 加载采样 init；`_load_init_states` (`examples/libero/main.py:822`) 读 `.init` 后交给 `task_suite` |
| 数据采集钩子 | `src/openpi/policy/...`（`--collect` 外层 wrapper，schema 定义见 [`docs/data_collection/guide.md`](../docs/data_collection/guide.md)） | 每 episode 落一个 h5，包含 `vision_0/1/2`、`prompt_emb`、`robot_state`、`noise_action_*`、`clean_action` |
| Artifact builder (pool 系列) | `exp/common/build_in_memory_cache_artifact.py` | 扫描 `--data-dir` 下所有 `.h5`，仅纳入 `success=True` episode，支持 4 种 pool builder + `cp1_temporal_prune`；CPU 多进程 |
| Artifact builder (CLIP) | `exp/common/build_clip_cache_artifact.py` | 读 h5 `input_images/` 用 open_clip 图像编码器产 key；单进程 GPU（模型仅加载一次）；同一 h5 库可产 ViT-B-32 / ViT-L-14 两个 artifact |

## 3. 产物目录与 Tracking 策略

```
exp/common/data/
  db/
    libero_cache/
      libero_spatial/                 # 既有（50 episode，其他实验已依赖）
      libero_10/                      # 本计划新建，中间 h5（.gitignore 屏蔽）
        episode_<id>_<timestamp>.h5
        ...
  cache_artifacts/
    libero_10/                        # 本计划新建，pkl artifact（.gitignore 屏蔽）
      cp1_mean_pool.pkl
      cp1_spatial_pool_16.pkl
      cp1_spatial_pool_64.pkl
      cp1_max_pool.pkl
      clip_vit_b_32.pkl
      clip_vit_l_14.pkl
```

- `db/libero_cache/` 的命名延续 2026-04-17 `experiment_artifact_layout_plan` 从 `data/db/libero_cache/` 迁移而来的约定（见 `logs/archive/experiment_artifact_layout_plan.log.md` §迁移映射表）。

- 符合 `docs/experiments/artifact_layout.md` §1 "canonical tree" 与 §3 "Tracking policy"：`exp/**/data/**` 默认被 `.gitignore` 屏蔽；本计划产物不属于已白名单的 tracked 例外。
- `.init` 输入文件是既有 tracked 资产，本计划只读不改。

## 4. 执行步骤

### 4.1 前置检查

1. `ls exp/common/data/db_init/libero_cache/libero_10/*.init | wc -l` = 11。
2. GPU 机已放置 pi05_libero 权重：`$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`。
3. 仿真机 LIBERO 环境可用（参 [`docs/deployment/libero.md`](../docs/deployment/libero.md)）。
4. `exp/common/data/db/libero_cache/libero_10/` 若已有历史 h5 先备份或清空，避免旧数据混入本次 artifact（对应 Risk §R5）。`libero_spatial/` 同级目录保持不动。

### 4.2 启动 Policy Server（GPU 机）

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  --collect \
  --collect_dir ./exp/common/data/db/libero_cache \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

- Server 绑 `0.0.0.0:8000`（`scripts/serve_policy.py:56, 386-387` — host 固定 `0.0.0.0`，port 本计划显式声明 8000，对齐 frp 内网映射）。
- 外部通过 frp 暴露在 `155.98.36.13:9000` → 内网 `:8000`（见 [`docs/deployment/libero.md`](../docs/deployment/libero.md) 拓扑图）。
- `--collect_dir` 的最终 h5 路径为 `<collect_dir>/<experiment_name>/episode_*.h5`（[`docs/data_collection/guide.md` §Output Location](../docs/data_collection/guide.md)）。本计划里 `experiment_name = args.task_suite_name = "libero_10"`，所以 h5 落到 `exp/common/data/db/libero_cache/libero_10/`。
- 首次推理会触发模型编译，服务端可能长时间无输出（Risk §R1）。

### 4.3 启动 LIBERO 客户端（仿真机）

```bash
uv run examples/libero/main.py \
  --task-suite-name libero_10 \
  --init-states-dir exp/common/data/db_init/libero_cache/libero_10 \
  --num-trials-per-task 5 \
  --host 155.98.36.13 --port 9000 \
  --display
```

- `examples/libero/main.py:871` 用 `tyro.cli(Args)` 把 `Args` 字段直接平铺到顶层，故 CLI flag 是 `--task-suite-name`（横杠）而非 `--args.task_suite_name`；下划线形式会被 tyro 拒绝为 "Unrecognized options"。
- `155.98.36.13:9000` 为本项目常用的 frp 公网入口（`docs/deployment/libero.md` / `docs/deployment/aloha_sim.md` / `exp/common/run_cache_experiments.py:12, 20`）；frp 内部转发到 server 的 `0.0.0.0:8000`。
- 客户端启动前用 `nc -zv 155.98.36.13 9000` 或 `curl http://155.98.36.13:9000/healthz` 确认通路可达（[`docs/deployment/libero.md` §Connectivity Check](../docs/deployment/libero.md)）。

- 预期 11 task × 5 init = 55 episode。
- 客户端必须在 `episode_start` 传 `experiment=args.task_suite_name`、`task=str(task_description)`、`episode_id=global_episode_id`（[`docs/data_collection/guide.md` §Minimal Simulator Changes](../docs/data_collection/guide.md) 强制要求的全局计数器）。执行前完成 Risk §R2 的 pre-check。

### 4.4 Build Artifact

```bash
mkdir -p exp/common/data/cache_artifacts/libero_10

# (a) Pool 系列：CPU 多进程
for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_64 cp1_max_pool; do
  uv run exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_10 \
    --builder-type $bt \
    --output exp/common/data/cache_artifacts/libero_10/${bt}.pkl
done

# (b) CLIP 系列：单进程 GPU，ViT-B-32 + ViT-L-14 共两份
uv run exp/common/build_clip_cache_artifact.py \
  --data-dir exp/common/data/db/libero_cache/libero_10 \
  --clip-model ViT-B-32 --clip-pretrained openai \
  --output exp/common/data/cache_artifacts/libero_10/clip_vit_b_32.pkl

uv run exp/common/build_clip_cache_artifact.py \
  --data-dir exp/common/data/db/libero_cache/libero_10 \
  --clip-model ViT-L-14 --clip-pretrained openai \
  --output exp/common/data/cache_artifacts/libero_10/clip_vit_l_14.pkl
```

- Pool 默认 `--workers 0`（全部 CPU）。若遇到 Risk §R4（OOM）则改 `--workers 4` 单独重跑 `cp1_spatial_pool_16`。
- CLIP 不接受 `--workers`（脚本注释 `exp/common/build_clip_cache_artifact.py:2-4`：模型仅加载一次以避免 GPU OOM）；运行时需 GPU 可用。
- CLIP 读取的是 h5 里的 `input_images/` group（`raw_image_collection_plan` 的产出）；需要确认采集阶段确实落了原图，否则 CLIP build 会因缺字段失败（Risk §R7）。
- 仅 `success=True` 的 episode 入 artifact（`exp/common/build_in_memory_cache_artifact.py:211-213`；CLIP 脚本同约定）。

## 5. 验收与测试策略

L1 无 G1/G2 Gate，按 WA §2.7 执行 Verify：

1. **h5 完整性抽查**：对最新 h5 检查顶层 `attrs` 含 `task/success/num_steps`，至少一个 `step_xxxx` 组包含 5 个 `CACHE_QUERY_FIELDS`（`vision_0/1/2`、`prompt_emb`、`robot_state`）。
2. **成功率统计**：遍历所有 h5，统计 `attrs["success"]=True` 占比，记录到 Verify 结果。低于 50% 触发 Risk §R3 的回退策略。
3. **Artifact 可加载性**（六份 pkl 均需通过）：

   ```python
   import pickle
   from openpi.cache.backends.in_memory_backend import InMemoryBackend

   pool_types = ["cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_max_pool"]
   clip_files = ["clip_vit_b_32", "clip_vit_l_14"]

   for name in pool_types + clip_files:
       path = f"exp/common/data/cache_artifacts/libero_10/{name}.pkl"
       with open(path, "rb") as f:
           art = pickle.load(f)
       assert art["checkpoint_id"] == "CP1"
       assert len(art["entries"]) > 0, f"{name}: empty artifact"
       if name in pool_types:
           assert art["key_builder_type"] == name
       backend = InMemoryBackend(art["vector_dims"])
       backend.load_artifact(art)
       assert backend.count() == len(art["entries"])
       print(f"{name}: {backend.count()} entries OK")
   ```

4. **回归测试**：`uv run pytest tests/cache -q` 应继续通过（本计划未改码，预期绿）。

## 6. Risk Register

| ID | 风险 | 缓解 |
|----|------|------|
| R1 | LIBERO 首次推理因模型编译长时间无输出，易被误判为死锁 | 按 `docs/data_collection/guide.md §Important Notes` 用 `top/htop` 监视 Python 进程 CPU；耐心等待 5–10 min |
| R2 | `examples/libero/main.py` 若未按 guide 约定调用 `client.episode_start(experiment=..., task=..., episode_id=global_counter)`，会导致 h5 落到错误子目录或 `episode_id` 重复 | 执行前 `grep episode_start examples/libero/main.py` 核对实参；必要时由项目 Owner 决定是否改客户端（改则升 L2） |
| R3 | 55 个 episode 成功率偏低，artifact 有效 entry 过少 | 调大 `--args.num_trials_per_task` 或多轮累加；h5 可以多次 append，build 阶段会汇总所有 h5 |
| R4 | `cp1_spatial_pool_16` 向量维 32768，55 × ~500 step × 2 vision 条目的 float32 pkl 在 build 时并发过高可能 OOM | 单独运行该 builder 时 `--workers 4`；监控 `free -h` |
| R5 | `exp/common/data/db/libero_cache/libero_10/` 残留旧 h5 会污染本次 artifact | 4.1 前置检查确认目录干净或先 `mv` 备份；同级 `libero_spatial/` 严禁误删（其他实验依赖） |
| R6 | 采集机与 build 机不同时，h5 需跨机同步 | rsync / 共享存储；同机执行可规避 |
| R7 | CLIP build 依赖 h5 中 `step_xxxx/input_images/` group；若采集时 `--collect` 之外有任何路径绕过了 `CollectionPolicy._extract_obs_fields` 则会缺字段 | build 前用 `h5dump -n exp/common/data/db/libero_cache/libero_10/episode_*.h5 \| grep input_images` 抽查一个 h5；缺就回到 §4.2 + §4.3 重采 |
| R8 | CLIP ViT-L-14 单模型 GPU 常驻 ~1.5 GB，单进程串行约 55 episode × ~100 step = ~5500 张图 encode，需约 10–20 min GPU 时间 | 与 pool 系列错峰跑，或指定 `--clip-device cuda:1` 单独拿一张卡 |

## 7. 回退方案

- **h5 损坏**：删除该 episode 的 h5，重跑对应 `(task, init)`。
- **artifact 缺陷**：删除对应 pkl 重建；`.gitignore` 屏蔽，不污染仓库。
- **`.init` 输入**：本计划只读，不会被修改；无需回退。

## 8. 不在范围内

- 任何 Python 源码修改（新 KeyBuilder、新 backend、`--collect` 钩子微调）。
- `sample_cache.py` 采样策略调整（本次复用既有 11 × 5 采样）。
- CP3 entry 构建（`build_in_memory_cache_artifact.py` 当前仅发 `CP1`）。
- `cp1_temporal_prune` artifact 的本批次产出（12 × 2 = 24 个组合，体量显著大于本次；可后续独立计划）。

## 9. 完成判据（Definition of Done）

- [x] 本 plan 文件提交并同步 `logs/README.md` 索引
- [ ] 55 个 episode 的 h5 全部落盘，且至少一个 h5 含 `step_xxxx/input_images/` group（CLIP 前置）
- [ ] 六份 pkl artifact（4 pool + 2 CLIP）全部加载检查通过（§5.3）
- [ ] `uv run pytest tests/cache -q` 通过

---

## Review Log

_L1 无强制 G1/G2。此段保留以符合 `protocols/execution_authority.md` §2 plan 文件规范；如需 ad-hoc 审阅可在此 append `### Ad-hoc Round N — Reviewer/Executor` 条目。_
