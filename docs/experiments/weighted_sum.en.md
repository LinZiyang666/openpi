# Weighted-Sum Calibration + Weight-Search Runbook (libero_spatial)

> End-to-end experiment for the two-layer `weighted_score_sum` retrieval: **Phase 1** picks the Layer-1 normalization method+params per (keybuilder, modality) offline; **Phase 2** uses the conductor in pure-eval mode to identify useful modalities and search for optimal weights.
> Design reference: [`logs/weighted_sum_two_layer_refactor.log.md`](../../logs/weighted_sum_two_layer_refactor.log.md); architecture: [`cache_system.md` §5.8.1](../architecture/cache_system.md); orchestration: [`conductor_tutorial.md`](conductor_tutorial.md).

---

## 0. Background

The old `weighted_score_sum` was almost unusable because calibration failed (in-library random-pair distribution + percentile collapsing into a constant on high-baseline cosine). The refactor splits it into a **Layer-1 normalization** (pluggable, monotonicity-preserving, `[0,1]`, rank excluded) + **Layer-2 weighted sum**, and picks the normalization method in a data-driven manner from the real query×whole-library distribution. `prompt_emb` is dropped from the experiment (near-constant within a task); candidate modalities = `{vision_0, vision_1, robot_state}` (CP1 may additionally include `vision_2`).

Data reuses the 6 library artifacts already at `exp/common/data/cache_artifacts/libero_spatial/` (1018 entries / 10 tasks / all-CP1, with `trajectory_id`) — zero extra collection.

## 1. Phase 1 — Offline Calibration (no GPU / no server)

```bash
uv run exp/common/calibrate_score_normalizers.py \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --output exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json \
    --max-queries 300
```

- **LOEO**: each entry is used as a query, scored against the whole library after filtering out its own episode by `trajectory_id` → the real query×whole-library raw distribution (eliminating self-match / intra-chain near-neighbor pollution, faithfully reproducing online conditions).
- For each (stem, field ∈ vector_dims minus prompt_emb), fit all compatible candidate normalizers and rank them by **`J = mag_sep + β·intra_spread − λ·sat`** (magnitude-structure metric; rank metrics are monotone-invariant and hence unusable) to produce a **top-2 shortlist**, with `selected` = top-1.
- Output is grouped by **artifact stem** (both CLIP variants have `builder_type = clip` so they would collide): `{stem: {builder_type, vector_dims, fields: {field: {sim_type, shortlist, selected}}}}`.

Diagnostic plot:

```bash
uv run exp/weighted_sum/analysis/plot_phase1_calibration.py \
    --calibration exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json
```

> The final method is **not** decided by J alone — the shortlist is handed to Phase 2 for real-task success-rate ranking.

## 2. Phase 2 — Weight Search (conductor pure-eval)

### 2.1 Emit eval YAMLs

```bash
uv run exp/weighted_sum/emit_yamls.py \
    --calibration exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json \
    --stem cp1_spatial_pool_16 \
    --preload-path exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl \
    --output-dir exp/weighted_sum/config/phase2/libero_spatial --mode both
```

Each YAML: `weighted_score_sum_knn` + Phase-1-selected `score_normalization.type: per_field`, `judge.type: always_hit` (pure replay to isolate retrieval quality), `keys.prompt_emb.enabled: false`, **`write_policy.type: never`** (C2 write-frozen, otherwise server load fails fast). `--mode`: `isolation` (single-modality → find useful modalities) / `grid` (weight grid over useful modalities) / `both`.

### 2.2 Launch the server (see conductor_tutorial §1)

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=<ckpt> --port 8001 \
    --cache_config exp/weighted_sum/config/phase2/libero_spatial/<any>.yaml
```

### 2.3 Run evaluation (init-state leakage guard)

```bash
uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/phase2/libero_spatial \
    --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal exp/weighted_sum/data/libero_spatial/phase2/journal.jsonl \
    --servers <host>:8001 --task-ids 0-9 --eval-trials 20
```

> ⚠ **Init-state leakage hard gate**: library construction only used ~5/50 inits per task. `run_phase2` reads the used `orig_init_state_idx` from `init_map` via `init_holdout` and **only samples episodes from the remaining held-out inits**; missing `init_map` fails fast — never silently skipped.

### 2.4 Aggregate + analyze

First aggregate the conductor journal into per-yaml success_rate, then plot:

```bash
uv run exp/weighted_sum/summarize.py \
    --journal exp/weighted_sum/data/libero_spatial/phase2/journal.jsonl \
    --out exp/weighted_sum/data/libero_spatial/phase2/results.json

uv run exp/weighted_sum/analysis/plot_phase2_results.py \
    --results exp/weighted_sum/data/libero_spatial/phase2/results.json
```

Group by keybuilder and plot success_rate × weight configuration (aligned with the `exp/common/analysis/phase1/libero_spatial` style). 2a isolation results are cross-validated with the Phase-1 `mag_sep` prior to lock down "the set of useful modalities + per-modality final method"; 2b runs a coarse→fine grid over useful modalities for the optimal weights.

## 3. Trajectory Extension (depth>1 multi-step retrieval)

Layer trajectory search (multi-step query history aggregation) on top of Phase-2's best configuration, methodologically aligned with the old trajectory experiment over Phase 1. Design and decisions: [`logs/weighted_sum_trajectory_search.log.md`](../../logs/weighted_sum_trajectory_search.log.md).

**Base selection (18, deduped)**: ① per-keybuilder (each of the 4 CP1 keybuilders contributes top1+top2+second-to-last; second-to-last taken from the regular weight grid with the same `zscore`, excluding `__norm2`/`iso_`); ② the full-experiment top10. The two sets overlap by 4 (spatial_16/max_pool's top1+top2 are all in top10), union 18. depth ∈ {3,4,5,6}; `trajectory_weights` reuses the old decreasing scheme; the depth-1 baseline reuses the existing SR from `data/libero_spatial/phase2/all_results.csv` (comparable on the same jupyter machine).

```bash
# 1) Emit 72 trajectory yamls (18 base × 4 depth), with built-in dedup assertion + schema self-check
PYTHONPATH=. uv run exp/weighted_sum/emit_trajectory_yamls.py
#    → exp/weighted_sum/config/trajectory/<base_id>__d{depth}.yaml

# 2) Launch server (jupyter, --replicas + HOME — see devices.md / §2.2 above) + tether expose
# 3) Run eval (timan107, each yaml 100 ep = 10 task × --eval-trials 10)
PYTHONPATH=. uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/trajectory/libero_spatial \
    --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal  exp/weighted_sum/data/libero_spatial/trajectory/journal.jsonl \
    --servers <host>:<port> --task-ids 0-9 --eval-trials 10 --workers 48 --gpus 8

# 4) Aggregate + analyze (merge depth-1 baseline + depth 3/4/5/6, compute Δ vs single-step)
uv run exp/weighted_sum/summarize.py \
    --journal exp/weighted_sum/data/libero_spatial/trajectory/journal.jsonl \
    --out     exp/weighted_sum/data/libero_spatial/trajectory/results.json
PYTHONPATH=. uv run exp/weighted_sum/analysis/plot_trajectory_results.py \
    --results  exp/weighted_sum/data/libero_spatial/trajectory/results.json \
    --baseline exp/weighted_sum/data/libero_spatial/phase2/all_results.csv
```

> Trajectory only supports `InMemoryBackend`; `run_phase2` / conductor / concurrent server are transparent to trajectory — zero change. `emit_trajectory_yamls.py` reuses `emit_yamls.build_eval_config` (which already accepts trajectory params).

## 4. Files

| Path | Purpose |
|------|---------|
| `exp/common/calibrate_score_normalizers.py` | Phase 1 offline calibration |
| `exp/weighted_sum/emit_yamls.py` | Phase 2 eval YAML generation (C2 / per_field / prompt_emb mask) |
| `exp/weighted_sum/emit_trajectory_yamls.py` | Trajectory extension: 18 base × depth generation (reuses `build_eval_config`) |
| `exp/weighted_sum/init_holdout.py` | Held-out init leakage guard |
| `exp/weighted_sum/weight_search_strategy.py` | Conductor pure-eval strategy |
| `exp/weighted_sum/run_phase2.py` | Phase 2 / Trajectory eval entrypoint (generic yaml-dir runner) |
| `exp/weighted_sum/summarize.py` | journal → per-yaml success_rate JSON |
| `exp/weighted_sum/analysis/` | Phase 1/2 + trajectory plotting (`plot_trajectory_results.py`) |
