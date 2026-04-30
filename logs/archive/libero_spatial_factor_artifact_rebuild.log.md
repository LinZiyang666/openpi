---
status: Implemented
authority_in_effect: Execution
level: L1
created: 2026-04-26
implemented: 2026-04-26
relates_to:
  - logs/verdict_factor_judge_b1_b2.log.md  # B1+B2 已 land 的 --factors-yaml CLI
  - exp/common/factor_postprocess.py
  - exp/common/build_in_memory_cache_artifact.py
  - exp/common/build_clip_cache_artifact.py
---

# Libero Spatial — F1b 因子 Artifact 重建计划

> **Level**: L1（实验级，无代码改动；仅写 minimal YAML + 跑 6 条 build CLI + 删 1 个目录）。无 G1/G2 gate，user 看完 plan 直接 OK 我开跑。

## 1. 目标

把 `exp/common/data/cache_artifacts/libero_spatial/` 下 **6 个 pkl 全部重建**，让每个 entry 带上 F1b-A + F1b-T 两个离线因子的 4 描述子 × 21 窗口 = 168 floats / entry。同时**删除** `exp/common/data/cache_artifacts/libero_spatial_warm/`。

## 2. 21 个窗口 enumeration

```
purefuture: (0, k) for k=1..7    → 7 个
purepast:   (k, 0) for k=1..7    → 7 个
sym:        (k, k) for k=1..7    → 7 个
total = 21（不含 (0,0)，无信息量）
```

## 3. Minimal YAML

写到 `exp/common/configs/factors/f1b_libero_spatial_w21.yaml`：

```yaml
# F1b-A + F1b-T over 21 windows (purefuture + purepast + symmetric, k=1..7).
# Used by the libero_spatial 6-pkl rebuild — see
# logs/libero_spatial_factor_artifact_rebuild.log.md.
factors:
  - type: f1b_a
    params:
      windows:
        - {past: 0, future: 1}
        - {past: 0, future: 2}
        - {past: 0, future: 3}
        - {past: 0, future: 4}
        - {past: 0, future: 5}
        - {past: 0, future: 6}
        - {past: 0, future: 7}
        - {past: 1, future: 0}
        - {past: 2, future: 0}
        - {past: 3, future: 0}
        - {past: 4, future: 0}
        - {past: 5, future: 0}
        - {past: 6, future: 0}
        - {past: 7, future: 0}
        - {past: 1, future: 1}
        - {past: 2, future: 2}
        - {past: 3, future: 3}
        - {past: 4, future: 4}
        - {past: 5, future: 5}
        - {past: 6, future: 6}
        - {past: 7, future: 7}
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
  - type: f1b_t
    params:
      windows: <same 21 windows as above>
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
```

实际 YAML 把 `<same>` 展开成完整列表（21 行 × 2 因子 = 42 行 windows）。

## 4. 6 个 pkl 的重建命令

数据源 = `exp/common/data/db/libero_cache/libero_spatial/`（50 episodes）。

```bash
# 4 in-memory pkls — build_in_memory_cache_artifact.py
for bt in cp1_mean_pool cp1_max_pool cp1_spatial_pool_16 cp1_spatial_pool_64; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type $bt \
        --output exp/common/data/cache_artifacts/libero_spatial/${bt}.pkl \
        --factors-yaml exp/common/configs/factors/f1b_libero_spatial_w21.yaml
done

# 2 CLIP pkls — build_clip_cache_artifact.py
uv run python exp/common/build_clip_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --clip-model ViT-B-32 --clip-pretrained openai \
    --output exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl \
    --factors-yaml exp/common/configs/factors/f1b_libero_spatial_w21.yaml

uv run python exp/common/build_clip_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --clip-model ViT-L-14 --clip-pretrained openai \
    --output exp/common/data/cache_artifacts/libero_spatial/clip_vit_l_14.pkl \
    --factors-yaml exp/common/configs/factors/f1b_libero_spatial_w21.yaml
```

执行顺序：先跑 `cp1_mean_pool`（最小 35MB → 最快） smoke test，确认产物：
- `pickle.load(...)` 顶层 dict 含 `library_stats` 键（非 None）
- 抽一个 entry 看 `payload.factors` keys 数 = 168（4 desc × 2 因子族 × 21 窗口）
- 边界 entry 含 NaN（per docs §3.4 边界规则）

smoke 通过再批量跑余 5 个。

## 5. 清理 `libero_spatial_warm/`

```bash
rm -rf exp/common/data/cache_artifacts/libero_spatial_warm/
```

**Side effect**：9 个 `exp/warm_start/config/{clip,max_pool,spatial16}/*_warm_t*.yaml` 的 `preload_path` 指向不存在的文件。两条处理：

- (a) **保留 yaml 不动**（默认）：作为实验设计存档；下次想 revive warm_start 就重新 build 这 3 个 warm 版 pkl
- (b) 也 `git rm` 9 个 yaml：彻底切干净

我建议 (a)。需要 (b) 你说。

## 6. 验收

| 项 | 期望 |
|---|---|
| 6 pkl 全部产出，文件大小近原始（factors 才 ~700B/entry） | `ls -lah` 近原值 |
| 每个 pkl `pickle.load` 后顶层 `library_stats` 非 None | smoke script |
| 任意 interior entry `payload.factors` 含 168 keys 全 finite（非 NaN） | smoke script |
| 任意 head/tail entry 部分 keys 为 NaN（边界规则） | smoke script |
| `libero_spatial_warm/` 已删 | `ls` 不存在 |

## 7. 时间估算

- `cp1_*_pool` 4 个：每个 < 1 min build + < 5s enrichment → ~5 min total
- `clip_vit_b_32`：~10 min（GPU）+ enrichment
- `clip_vit_l_14`：~20 min（GPU）+ enrichment
- 总计：~40 min（视 GPU + I/O）

## 8. 风险

| 风险 | 缓解 |
|---|---|
| build script 在新加 `--factors-yaml` 路径上漏 bug（builder lifecycle 改动后没 manual smoke 过） | 先跑最小 pkl smoke，发现问题前不批量铺 |
| YAML 21 窗口写错 → 整批 entry 全 NaN | smoke test 抽 interior entry 看 finite |
| `libero_spatial_warm/` 删除影响 in-flight 实验 | 删之前 `ls -la libero_spatial_warm/*.pkl` 看 mtime；如近期没动过就安全 |
