# Verdict Factor Judge — User Guide

> ⚠️ **2026-05-07 — Refactor (G1 APPROVED Round 4 / G2 APPROVED Round 3) has landed**:
>
> - **5 → 17 factors flat layout**: `f1a_a / f1a_t / f1b_a / f1b_t / f2`
>   are **all gone**; new names follow `<descriptor>_<source>_<channel>`:
>   - 4 descriptors: `jerk` / `direction` (was `dir`) / `dispersion`
>     (was `curv_radius`) / `path_length` (was `cum_disp`)
>   - 4 variants = 2 sources × 2 channels: `online | offline` ×
>     `action | state` → 16; plus `topk_action_variance` = 17.
> - **4-layer judge architecture**: Normalization → Factor → Calibration
>   → Composer; each layer is independently pluggable via yaml.
> - **No cold-start**: Layer 1 + Layer 3 require pre-loaded calibration
>   data at startup; the legacy `cold_start_strategy: force_miss /
>   passthrough / lenient` + `all_nan_fallback` +
>   `JudgeResult.factor_outputs.sentinel` fields are removed.
> - **Diagnostic schema_version=2**: `factor_outputs.{raw, calibrated,
>   composer_score}` (legacy `norm` / `score` / `sentinel` removed).
>
> Authoritative reference (Chinese): [`verdict_factor_judge.md`](verdict_factor_judge.md).
> Full design + decision history: [`logs/verdict_factor_judge_refactor.log.md`](../../logs/verdict_factor_judge_refactor.log.md).
> Pre-refactor design docs: `logs/old_verdict_factor_*.log.md` (8 archived files).

> **Prerequisites**: read [tutorial.md](tutorial.md) §6 for the Judge
> component basics and §10 for YAML config; read
> [../architecture/cache_system.md](../architecture/cache_system.md)
> §5.12 / §5.13 for the verdict-factor architecture contract.

---

## 1. Overview (refactored 4-layer architecture)

`CompositeJudge` splits the hit decision into **four orthogonal,
pluggable layers**. Each layer is independently swappable through the
yaml `judge` block; the layers are orthogonal — a layer never holds a
reference to another layer's instance, only to an interface contract.

```
                    ┌──────────────────────────────────────┐
raw action / state  │ Layer 1   Normalization              │
─────────────────►  │   ZScoreNormalization                │
                    │   stats_source: offline (LibraryStats)│
                    └──────────────┬───────────────────────┘
                                   │ normalized data injected via FactorContext
                                   ▼
SearchResultLite[]  ┌──────────────────────────────────────┐
PayloadView      ──►│ Layer 2   17 Factors                  │
HistoryView         │   `<descriptor>_<source>_<channel>`   │
                    │   + topk_action_variance              │
                    └──────────────┬───────────────────────┘
                                   │ raw factor dict[str, float]
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ Layer 3   Calibration (per key)       │
                    │   PercentileRollingCalibration        │
                    │   samples_source: offline | warmup    │
                    │   bind_keys() fail-fast at startup    │
                    └──────────────┬───────────────────────┘
                                   │ calibrated dict in [0, 1]
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ Layer 4   Composer                    │
                    │   declared_dependencies (instance)    │
                    │   compose(calibrated, *, winner_id)   │
                    │   subclass owns NaN handling          │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    JudgeResult(hit_type, winner_id, start_t,
                                factor_outputs={schema_version=2,
                                                raw, calibrated,
                                                composer_score})
```

**Key invariants**:

- Each layer's Protocol is unaware of the others' implementations (layers communicate only through agreed dataclasses / dicts).
- The only legitimate NaN sources are factor physical edges (`history` < P, `walk_next` runs out, missing winner data); **not** Layer 3 cold-start (buffers are full at startup) and **not** missing Layer 1 σ (startup fail-fast).
- Empty `results` → `CompositeJudge` returns `JudgeResult(MISS)` directly (the composer is never invoked with `winner_id=None`).
- Composer subclasses **own** NaN handling: `WeightedSumWithWarmFallbackComposer` returns WARM_START when every non-zero-weight key is NaN (this is how the legacy `all_nan_fallback` semantics are preserved).

---

## 2. The 17 Flat Factors

Each `<descriptor>_<source>_<channel>` is its own registered factor identity. The 4-axis Cartesian gives 16 + 1 topk = 17.

**4 kinematic descriptors (formula unchanged from pre-refactor; some renamed)**:

| New name | Old name | Orientation | Formula | Meaning |
|---|---|---|---|---|
| `jerk` | `jerk` | risky | `median \|Δ²a\|` (z-scored, active-DOF, time-median then DOF-mean) | Acceleration variation; high → not smooth |
| `direction` | `dir` | safe | `mean cos(v[t], v[t+1])` | Velocity-direction consistency; high → smooth |
| `dispersion` | `curv_radius` | non_monotonic | `mean ‖p[t] − centroid‖` | Window geometric spread (medium = arc; very small = stuck; very large = long straight) |
| `path_length` | `cum_disp` | non_monotonic | `sum ‖p[t+1] − p[t]‖` | Cumulative path length (small + low jerk = stationary; large = fast motion) |

**4 variant axes** (each descriptor × 4 = 16 standalone factors):

| Axis | Values | Semantics |
|---|---|---|
| source | `online` | Computed at verdict time (build splice from `[history[-P:], winner, walk_next(F)]` and call `ctx.normalization.normalize_*`) |
| source | `offline` | Computed at artifact build by `OfflineWriter.compute_for_episode`, stored in `payload.factors`; verdict-time reads `winner.payload.factors` |
| channel | `action` | Operates on chain `payload.action_chunk[0]` sequence |
| channel | `state` | Operates on chain `query_keys["robot_state"]` sequence |

**The 17th factor**:

| Registered name | Class | Source | requires_chain_walk | required_top_k |
|---|---|---|---|---|
| `topk_action_variance` | `TopkActionVariance` | per-DOF variance of top-K candidates' `action_chunk[0]` (candidate-local active mask) | False | K |

**Online factors share a unified splice** (`<descriptor>_online_<channel>`, 8 total, `requires_chain_walk=True`):

```
splice = [history[-P:],   winner,   walk_next(F).<channel>]
              ^               ^         ^
              past            anchor    future
              P executed steps  winner   F downstream chain steps
```

- `online_action`: `history.actions[-P:]` + `winner.action_chunk[0]` + `walk_next(F).payload.action_chunk[0]`
- `online_state`:  `history.states[-P:]`  + `winner.query_keys["robot_state"]` + `walk_next(F).query_keys["robot_state"]`
- splice step `t`'s state and action come from the **same inference quote** (entry t's `query_keys` + `payload`).

**Offline factors** (`<descriptor>_offline_<channel>`, 8 total, `requires_chain_walk=False`):

- `OfflineWriter.compute_for_episode(entries, library_stats)` performs a sliding-window descriptor pass at artifact-build time, writing `entry.payload.factors[<key>]`.
- At verdict time, only `winner.payload.factors[<key>]` is read; no chain walk.
- key template: `<descriptor>_offline_<channel>__p<P>_f<F>`.

**Backend capability**: any yaml containing a `requires_chain_walk=True` factor → `backend.type` must == `"in_memory"` (the only backend exposing `fetch_entry`). Static yaml-load validator.

`risky` / `safe` composer auto-flips contributions by orientation; `non_monotonic` keys MUST appear under `composer.directions` with `"high"` / `"low"` / `"range:[lo, hi]"`, otherwise yaml load is rejected.

---

## 3. yaml configuration example (refactored 4-layer schema)

```yaml
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }

    judge:
      type: composite

      # ── Layer 1 ──
      normalization:
        type: zscore                                  # registered Normalization subclass
        params: {}                                    # ZScoreNormalization has no params
        stats_source:
          type: offline                               # only allowed value (warmup deferred)
          # offline: σ + active_mask read from backend.load_artifact's library_stats field

      # ── Layer 2 ── (any subset of the 17 factors)
      factors:
        - type: jerk_online_state
          params:
            windows:
              - { past: 5, future: 5 }
              - { past: 7, future: 7 }
        - type: dispersion_offline_state
          params:
            windows: [{ past: 5, future: 5 }]
        - type: topk_action_variance
          params: { K: 5 }

      # ── Layer 3 ──
      calibration:
        type: percentile_rolling
        params: { window_size: 50 }
        samples_source:
          type: warmup                                 # | offline
          # warmup: read from WarmupPool[eval_yaml_id] (sibling warmup yaml must run first)
          # offline alt:
          # offline: { path: data/calibration/spatial16_v2.jsonl, format: jsonl }

      # ── Layer 4 ──
      # ComposerConfig is FLAT — `weights` / `tier_thresholds` /
      # `warm_start_t` / `warm_fallback_start_t` / `directions` /
      # `per_factor_thresholds` sit directly under `composer:`,
      # **not** nested under `composer.params:`.
      composer:
        type: weighted_sum_with_warm_fallback           # | weighted_sum / and / or
        weights:
          jerk_online_state__p5_f5:        1.0
          jerk_online_state__p7_f7:        1.0
          dispersion_offline_state__p5_f5: 1.0
          topk_action_variance:            0.5
        tier_thresholds:
          full_hit:    0.30
          warm_start:  0.10                             # optional; triggers regular warm tier
        warm_start_t: 0.7                               # paired with tier_thresholds.warm_start
        warm_fallback_start_t: 0.7                      # all-NaN fallback (legacy all_nan_fallback equivalent)
        directions:                                     # required for non_monotonic keys
          dispersion_offline_state__p5_f5: "range:[0.3, 0.7]"

      export_factor_outputs: true                       # default false; true emits schema_version=2
```

**Key validator rules** (`config.py` runs at yaml load time):

1. `judge.type=="composite"` requires all four blocks `normalization` / `factors` / `calibration` / `composer`.
2. Every `factors[].type` must be one of the 17 registered names. Legacy 5 names (`f1a_a` / `f1a_t` / `f1b_a` / `f1b_t` / `f2`) load with `Unknown factor name`.
3. The composer's `declared_dependencies` (instance attribute) must ⊆ Layer 2 union key set.
4. Any `requires_chain_walk=True` factor → `backend.type=="in_memory"` is required.
5. Non_monotonic keys consumed by the composer must appear in `composer.directions`.
6. `normalization.stats_source.type` must be `"offline"`; `calibration.samples_source.type` ∈ `{"offline", "warmup"}`.
7. Legacy fields `judge.normalizer` / `judge.all_nan_fallback` / `judge.cold_start_strategy` in yaml → load fails with "legacy schema, rewrite to 4-layer".

---

## 3.1 Multi-factor / online+offline yaml combinations

The 17 factors are **flat**: every `<descriptor>_<source>_<channel>` is an independent registered name. Any subset can be combined in the same yaml's `factors:` list, as long as each entry occupies its own list element AND the composer's `weights` (or `per_factor_thresholds`) reference all the keys derivable from those factors' `describe(params)`.

**Key contracts**:

- `factors:` is a list; each element is `{type: <registry-name>, params: {...}}`.
- Even the online + offline variants of the same descriptor are **two independent entries** (different registry names, mutually independent).
- Multi-window for one factor type goes under a single entry's `params.windows: [...]` (every one of the 17 factors natively supports multi-window).
- Composer `weights` / `per_factor_thresholds` must reference every key produced by these factors' `describe(params)` (multi-window → multiple keys, one per window).

### 3.1.1 Multiple descriptors (single source × channel)

Example: state channel, online source, two descriptors (jerk + direction), one window each.

```yaml
factors:
  - type: jerk_online_state
    params:
      windows: [{past: 5, future: 5}]
  - type: direction_online_state
    params:
      windows: [{past: 5, future: 5}]

composer:
  type: weighted_sum
  weights:
    jerk_online_state__p5_f5:      1.0      # risky → composer auto 1-v
    direction_online_state__p5_f5: 1.0      # safe  → composer uses v directly
  tier_thresholds: { full_hit: 0.5 }
```

### 3.1.2 Same descriptor, both online and offline

Example: jerk on state channel, simultaneously enabling online (verdict-time splice + walk_next) and offline (artifact-build chain pass) — these are **two independent factor entries** that **cannot be merged** into one.

```yaml
factors:
  - type: jerk_online_state              # verdict-time splice [history[-5:], winner, walk_next(5)]
    params:
      windows: [{past: 5, future: 5}]
  - type: jerk_offline_state             # offline build sliding window into payload.factors
    params:
      windows: [{past: 5, future: 5}]

composer:
  type: weighted_sum_with_warm_fallback
  weights:
    jerk_online_state__p5_f5:  1.0
    jerk_offline_state__p5_f5: 1.0       # same window as online but different source → different key
  tier_thresholds: { full_hit: 0.5 }
  warm_fallback_start_t: 0.7             # all NaN (chain ran out + offline never wrote) → WARM_START
```

> Physical meaning of the two keys: `jerk_online_state__p5_f5` is "jerk of the splice built at verdict time from history + walk_next"; `jerk_offline_state__p5_f5` is "jerk this entry observed in its own chain at the (P=5, F=5) window during artifact build". They are **two estimates of the same physical quantity**; calibration uses independent percentile buckets for each.

### 3.1.3 Same factor multi-window + all 17 factors enabled

Example: 4 desc × 2 source × 2 channel = 16 + topk = 17 enabled, full-space sweep on spatial16.

```yaml
factors:
  # 8 online factors, 2 windows each
  - { type: jerk_online_action,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: jerk_online_state,         params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_online_action,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_online_state,    params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_online_action,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_online_state,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_online_action, params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_online_state,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  # 8 offline factors, same windows
  - { type: jerk_offline_action,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: jerk_offline_state,         params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_offline_action,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_offline_state,    params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_offline_action,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_offline_state,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_offline_action, params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_offline_state,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  # 17th: top-K candidate consensus variance (no windows, single key)
  - { type: topk_action_variance, params: { K: 5 } }

composer:
  type: weighted_sum
  weights:
    # 16 desc factors × 2 windows = 32 keys (uniform 1.0) + topk separate weight
    jerk_online_action__p5_f5:        1.0
    jerk_online_action__p7_f7:        1.0
    jerk_online_state__p5_f5:         1.0
    jerk_online_state__p7_f7:         1.0
    # ... (omitting the remaining 28 desc keys — same pattern, uniform 1.0) ...
    topk_action_variance:             0.5   # different weight is fine
  tier_thresholds: { full_hit: 0.30 }
  directions:                                 # non_monotonic keys (dispersion / path_length) need a direction
    dispersion_online_state__p5_f5:  "range:[0.3, 0.7]"
    path_length_online_state__p5_f5: "high"
    # ... every non_monotonic key must appear ...
```

### 3.1.4 Multi-factor with AndGate / OrGate

`weighted_sum` uses `weights`; `and` / `or` use `per_factor_thresholds`, judging each key independently.

```yaml
factors:
  - { type: jerk_online_state,        params: { windows: [{past: 5, future: 5}] } }
  - { type: dispersion_offline_state, params: { windows: [{past: 5, future: 5}] } }

composer:
  type: and          # both keys must pass their threshold for FULL_HIT
  per_factor_thresholds:
    jerk_online_state__p5_f5:        0.3   # risky → v <= 0.3
    dispersion_offline_state__p5_f5: 0.5   # non_monotonic → directions block decides pass condition
  directions:
    dispersion_offline_state__p5_f5: "range:[0.3, 0.7]"
  warm_start_t: 0.7    # optional: emit WARM_START on pass (CP1-only)
```

### 3.1.5 The warmup yaml MUST over-collect every key above

Each eval yaml needs a sibling warmup yaml to fill the Layer 3 calibration rolling buffers. The warmup's `judge.dump.factors` must **at minimum** cover every key the eval composite consumes (otherwise eval `bind_keys` fails fast):

```yaml
# <eval_yaml_id>__warmup.yaml (the generator usually produces this; manual shape below)
checkpoints:
  cp1:
    judge:
      type: always_warm_start
      start_t: 0.7
      dump:
        deferred: true
        config_id: <eval_yaml_id>__warmup
        factors:
          # ⚠ Must cover every key the eval factors will produce — either
          #    mirror the eval factors list 1-to-1, or just over-collect
          #    the full 17 factors (recommended: switching eval factor mix
          #    later won't require re-running warmup).
          - { type: jerk_online_state,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
          - { type: jerk_offline_state,       params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
          # ... (the other 15 factors) ...
          - { type: topk_action_variance, params: { K: 5 } }
```

In practice you don't write warmup yamls by hand —
`exp/verdict_factor_judge/v2_spec.py:build_warmup_yaml(cfg_id, eval_yaml_id, eval_factors=...)`
takes the eval factor list and auto-unions the dump factor superset
(per `(descriptor, source, channel)` group, eval windows ∪
`_W_UNION_DEFAULT`).

---

## 4. Factor formulas (precise)

> The 4 descriptor formulas are **identical** to the pre-refactor versions; only renamed and flattened. Formula source: `src/openpi/cache/components/factors/_descriptor_kernel.py`.

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| `seq` | Single window's z-scored, active-subspace points; shape `[W, D_act]` |
| `v[t]` | `seq[t+1] - seq[t]` (first difference); shape `[W-1, D_act]` |
| `j[t]` | `seq[t+1] - 2 seq[t] + seq[t-1]` (second difference); shape `[W-2, D_act]` |
| `D_act` | Number of dims True in `LibraryStats` active mask (z-score drops padded DOFs) |

z-score: Layer 1 Normalization computes `seq / sigma.clamp_min(eps=0.01)` then selects the active mask; the factor layer only sees an already-normalized `seq`.

### 4.2 Four descriptor formulas

#### `jerk` (risky)
```
jerk = mean_DOF(   median_t(|j[t]|)   )
```
Time-axis median (absorbs gripper single-frame spikes), then DOF-axis mean. NaN when `j` is empty (W < 3).

#### `direction` (safe)
```
direction = mean_t(  cos(v[t], v[t+1])  )    // only over pairs whose endpoints have positive norm
```
Mean cosine between consecutive velocity vectors. NaN when `v.shape[0] < 2` (W < 3) or every adjacent velocity pair has at least one zero-norm endpoint.

#### `dispersion` (non_monotonic)
```
dispersion = mean_t(  ‖seq[t] - centroid(seq)‖  )
```
Window geometric spread (mean distance from points to centroid). NaN when W < 2.

#### `path_length` (non_monotonic)
```
path_length = sum_t(  ‖seq[t+1] - seq[t]‖  )
```
Cumulative step-length sum. NaN when W < 2.

### 4.3 Splice shape per factor

#### Online (8 factors)
```
seq = [history[-P:], winner, walk_next(F).<channel>]      # length P + 1 + F
```
- channel = `action`: `history.actions[-P:]` + `winner.action_chunk[0]` + chain `walk_next` `payload.action_chunk[0]`
- channel = `state`:  `history.states[-P:]`  + `winner.query_keys["robot_state"]` + chain `walk_next` `query_keys["robot_state"]`

NaN physical edges:
- `len(history) < P` (early in episode)
- `walk_next` returns < F entries (chain end / fork detected)
- winner missing `robot_state` (state channel)

#### Offline (8 factors)
At verdict time, simply read `winner.payload.factors[<key>]` (offline build wrote this).
- Offline build: scan the entire chain, for each entry × each `(P, F)` window run a sliding descriptor pass; result lands in `entry.payload.factors`.
- Boundary: any `(entry, window)` whose window pokes past the chain ends gets NaN.

#### `topk_action_variance` (the 17th)
```
var = mean(  per_DOF_var(  [r.payload.action_chunk[0] for r in results[:K]]  )  )
                                                                ↑
                                                  candidate-local active mask
                                                  (per-DOF var > 1e-8)
```
NaN when `len(results) < K` (search did not return enough candidates) or when every per-DOF variance is below the active-mask epsilon (the candidate pool agrees on every dim).

Does NOT consult Layer 1 normalization: the variable is candidate-pool variance, scale-invariant; Pi0.5 padded DOFs are exactly 0 across the whole library, so candidate-local variance is exactly 0 there and they drop out of the active mask automatically.

---

## 5. End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1   Build artifact (one-off, GPU heavy)                 │
│   build_in_memory_cache_artifact.py: HDF5 → .pkl            │
│   Contains entries + LibraryStats (σ + active_mask)         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 2   Enrich pkl with offline factors (seconds, smoke)    │
│   build_in_memory_cache_artifact.py enrich-existing-pkl     │
│   --input old.pkl --factors-yaml factors.yaml --output new.pkl│
│   ↑ Adds 17-factor keys to payload.factors; does NOT recompute σ│
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 3   Warmup yaml run (collect calibration samples)       │
│   <eval>__warmup.yaml + DumpingJudge                         │
│   → JSONL of factor raw values                               │
│   → preload_normalizer_buffer ctrl → WarmupPool[eval_yaml_id]│
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 4   Eval yaml run                                       │
│   load_cache_config: validator + _build_calibration pulls    │
│      from WarmupPool[eval_yaml_id] / offline jsonl, fills    │
│      buffers → bind_keys fail-fast on undersized samples     │
│   Per verdict: Layer 1 → Factor → Calibration → Composer    │
│   Optional export_factor_outputs writes schema_v=2 jsonl     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Step 1+2: build / enrich artifact

### 6.1 Build from HDF5 (first time)
```bash
uv run python -m exp.common.build_in_memory_cache_artifact \
  --data-dir exp/common/data/db_init/libero_cache/libero_spatial \
  --builder-type cp1_spatial_pool_16 \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --workers -1 \
  --factors-yaml my_factors.yaml             # also enrich 17-factor keys in this pass
```

### 6.2 Incremental factor add to an existing pkl (seconds, smoke)
```bash
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --factors-yaml my_factors.yaml \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16_v2.pkl
```
Key points (plan §16 B6.5):
- Reuses input pkl's `library_stats` (does **NOT** recompute σ); smoke completes in seconds.
- Pre-existing `payload.factors` keys are preserved (additive merge); new 17-factor keys are added.
- The new pkl is a sibling of the input; the input file is untouched (rollback friendly).

### 6.3 `factors.yaml` shape (offline factors only)
```yaml
factors:
  - type: jerk_offline_action
    params:
      windows:
        - {past: 1, future: 1}
        - {past: 5, future: 5}
  - type: dispersion_offline_state
    params:
      windows: [{past: 5, future: 5}]
```
Only the 8 offline factors (`<descriptor>_offline_<channel>`) are accepted; online + topk lack the `compute_for_episode` surface and `_load_offline_writers_from_yaml` rejects them explicitly.

---

## 7. Step 3+4: yaml configuration + run inference

The full 4-layer yaml is shown in §3. This section enumerates per-field details.

### 7.1 Layer 1 Normalization
```yaml
normalization:
  type: zscore                  # only registered type
  params: {}                    # ZScoreNormalization has no params
  stats_source:
    type: offline               # only allowed value; warmup channel deferred
```
Startup check: `backend.library_stats` must be reachable (`backend.type == "in_memory"`).

### 7.2 Layer 2 Factors
One entry per factor, `{type, params}`. Typical params:
- 8 online + 8 offline: `windows: [{past, future}, ...]`
- `topk_action_variance`: `K: int >= 2`

### 7.3 Layer 3 Calibration
```yaml
calibration:
  type: percentile_rolling      # only registered type
  params:
    window_size: 50             # rolling buffer size
  samples_source:
    type: warmup                # | offline
    # warmup: pulled from WarmupPool[eval_yaml_id] (sibling warmup yaml must run first)
    # offline alt:
    # offline:
    #   path: data/calib/spatial16_v2.jsonl
    #   format: jsonl           # | pkl
```
At `bind_keys`: every Layer 2 union key must have ≥ window_size non-NaN samples in `samples`, otherwise yaml load fails.

### 7.4 Layer 4 Composer

> **Schema shape**: `ComposerConfig` is **flat** — every key field (`weights` / `tier_thresholds` / `per_factor_thresholds` / `warm_start_t` / `warm_fallback_start_t` / `directions`) sits directly under `composer:`, **not** under `composer.params:`. The yaml parser silently warns and ignores any `composer.params.*`; the validator then likely rejects the config because a required field (e.g. `tier_thresholds.full_hit`) is missing.

The 4 registered subclasses:

| type | Top-level fields | Hit conditions |
|---|---|---|
| `weighted_sum` | `weights` / `tier_thresholds: {full_hit, warm_start?}` / `warm_start_t` | Orientation-flipped weighted sum ≥ full_hit → FULL_HIT; ≥ warm_start → WARM_START (CP1-only) |
| `weighted_sum_with_warm_fallback` | Above + `warm_fallback_start_t` | Same; when every non-zero-weight key is NaN → WARM_START @ `warm_fallback_start_t` (legacy `all_nan_fallback` equivalent). **`warm_fallback_start_t` is a WARM_START emission path too — like `warm_start_t`, it is CP1-only.** |
| `and` | `per_factor_thresholds` / `warm_start_t?` | Every key must pass its threshold → FULL_HIT (or WARM_START if `warm_start_t` is set; CP1-only) |
| `or` | Same | Any key passing its threshold → FULL_HIT / WARM_START |

`directions`: any `non_monotonic` orientation key must declare a direction (`"high"` / `"low"` / `"range:[lo, hi]"`); otherwise yaml load is rejected.

> **CP3 restriction** (plan §3.6 / validator §13.3 rule 5c): CP3 has no `intermediates` payload to resume from, so **any** WARM_START emission path is rejected by the yaml validator on CP3 — this includes `warm_start_t` (regular warm tier) AND `weighted_sum_with_warm_fallback`'s `warm_fallback_start_t` (all-NaN fallback). On CP3, composite judges must omit both fields.

### 7.5 Diagnostic field `factor_outputs` (schema_version=2)

When `judge.export_factor_outputs: true`, every verdict writes the following on `JudgeResult.factor_outputs`:
```python
{
  "schema_version": 2,
  "raw":             {key: float | None},     # Layer 2 raw
  "calibrated":      {key: float | None},     # Layer 3 output
  "composer_score":  float | None,            # Layer 4 internal aggregate
}
```
NaN is converted to None on the wire (JSON-strict compatible). `hit_type / winner_id / start_t` remain at the top level of `JudgeResult` (not inside `factor_outputs`).

---

## 8. Custom extensions (writing your own factor / Calibration / Composer)

> This section is a step-by-step tutorial: write code → register → reference in yaml → unit test → start server. Each subsection's yaml fragment can be pasted directly into the §3 full 4-layer yaml template.

### 8.1 Adding a Layer 2 Factor

#### Step 1: Decide the factor's capabilities and naming

| Decision | Rule | Effect |
|---|---|---|
| Naming | Recommended to follow `<descriptor>_<source>_<channel>` (see §2); custom factors can use any name but avoid the 17 reserved factor names. | Registered name is the literal value of `factors[].type` in yaml. |
| `requires_chain_walk` | True iff `extract` calls `ctx.view.walk_prev / walk_next`. | Yaml load is rejected if the backend lacks `fetch_entry`. |
| `required_top_k` | Minimum top-K candidates the factor needs; 0 if not applicable. | CompositeJudge takes the max over all factors and feeds search-strategy `min_top_k_hint`. |
| Descriptor key + orientation | `"safe"` (high = good) / `"risky"` (high = bad) / `"non_monotonic"` (must be declared in composer `directions`). | Composer auto-flips by orientation (safe → use v directly; risky → use `1-v`). |

#### Step 2: Implement the class

Place the file in a module that `factors/registry.py` imports (simplest: append to `factors/online.py`, or create `factors/my_pack.py` and add an import to `registry.py`). Full skeleton:

```python
# src/openpi/cache/components/factors/my_pack.py

"""My custom factor — describe what physical signal this captures."""

from __future__ import annotations

import math

import torch

from openpi.cache.components.factors.base import FactorContext
from openpi.cache.components.factors.registry import register


@register("my_action_burstiness")
class MyActionBurstiness:
    """Mean burst-ratio over the last K executed actions.

    Returns a single risky-orientation key:
        my_action_burstiness__k<K>: float in [0, 1]   (or NaN at boundary)
    """

    # ---- class-level capability flags ----
    requires_chain_walk: bool = False     # only reads history.actions
    required_top_k:      int = 0          # not a candidate-pool factor

    # ---- constructor ----
    def __init__(self, *, K: int) -> None:
        if K < 2:
            raise ValueError(f"MyActionBurstiness K must be >= 2, got {K}")
        self.K = int(K)
        self.descriptor_orientations = self.__class__.describe({"K": K})

    # ---- pure metadata classmethod (validator calls this WITHOUT instantiating) ----
    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        K = int(params["K"])
        return {f"my_action_burstiness__k{K}": "risky"}

    # ---- per-verdict extraction ----
    def extract(self, ctx: FactorContext) -> dict[str, float]:
        key = next(iter(self.descriptor_orientations))
        if len(ctx.history.actions) < self.K:
            return {key: float("nan")}                       # boundary

        # Pull last K actions, normalize via Layer 1 (z-score).
        seq = torch.stack(
            [torch.as_tensor(a, dtype=torch.float32) for a in ctx.history.actions[-self.K:]],
            dim=0,
        )                                                    # [K, A]
        normed = ctx.normalization.normalize_action(seq)     # [K, A_active]
        if normed.shape[-1] == 0:
            return {key: float("nan")}                       # empty active mask

        # Burst ratio = fraction of consecutive-step diffs whose magnitude
        # exceeds the median diff magnitude. Risky: high = bursty.
        diffs = (normed[1:] - normed[:-1]).norm(dim=-1)      # [K-1]
        if diffs.numel() < 2:
            return {key: float("nan")}
        median = float(diffs.median())
        ratio = float((diffs > median).float().mean())
        return {key: ratio}
```

#### Step 3: Have registry import the module on load

Append a single line to `factors/registry.py`:

```python
from openpi.cache.components.factors import my_pack  # noqa: F401
```

Any `from openpi.cache.components.factors import registry` triggers this import, and the `@register` side effect immediately registers the factor in the global registry.

#### Step 4: Reference in yaml

```yaml
checkpoints:
  cp1:
    judge:
      type: composite
      normalization: { type: zscore, params: {}, stats_source: { type: offline } }
      factors:
        - type: my_action_burstiness
          params: { K: 5 }
      calibration:
        type: percentile_rolling
        params: { window_size: 50 }
        samples_source:
          type: warmup            # warmup yaml MUST over-collect this key
      composer:
        type: weighted_sum
        weights: { my_action_burstiness__k5: 1.0 }
        tier_thresholds: { full_hit: 0.7 }   # high burst-ratio = miss → high (1-v) = full_hit
```

#### Step 5: Unit test

```python
# tests/cache/components/factors/test_my_action_burstiness.py
import math
import pytest
import torch

from openpi.cache.components.factors.base import FactorContext, HistoryView, LibraryStats
from openpi.cache.components.factors.my_pack import MyActionBurstiness
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID

def _ctx(history_actions):
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    ls = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    return FactorContext(
        results=[SearchResultLite(id="w", score=1.0, checkpoint_id=CheckpointID.CP1)],
        view=None, normalization=ZScoreNormalization(ls),
        history=HistoryView(
            actions=[torch.tensor(a, dtype=torch.float32) for a in history_actions],
            states=[],
        ),
    )

def test_K_below_2_rejected():
    with pytest.raises(ValueError, match=">= 2"):
        MyActionBurstiness(K=1)

def test_history_too_short_emits_nan():
    f = MyActionBurstiness(K=5)
    out = f.extract(_ctx([[0.0, 0.0]] * 3))
    assert math.isnan(out["my_action_burstiness__k5"])

def test_describe_classmethod_is_pure():
    """Validator calls describe() WITHOUT instantiating — the map must be derivable from params alone."""
    out = MyActionBurstiness.describe({"K": 5})
    assert out == {"my_action_burstiness__k5": "risky"}
```

#### 8.1 Key contracts (fail-loud list)

- The `extract` return dict's key set MUST equal `self.descriptor_orientations.keys()` — CompositeJudge `__call__` enforces this with a key contract assertion (it raises rather than silently NaN-ing).
- `describe(params)` is a classmethod the validator calls **without instantiating**, so it MUST NOT rely on `self`, do I/O, or query `library_stats`.
- Any physical edge (history too short / walk runs out / degenerate denominator) → return `float("nan")`; do **NOT** raise. The composer subclass owns NaN handling.
- Do NOT z-score inside the factor; use `ctx.normalization.normalize_action / normalize_state` (Layer 1 owns z-score).

---

### 8.2 Adding a Layer 3 Calibration

Implement the `Calibration` Protocol (`factors/calibrations/base.py`): `__init__(samples) / bind_keys(keys) / __call__(raw) / on_episode_start()`.

```python
# src/openpi/cache/components/factors/calibrations/my_zscore.py

"""Z-score-on-factors calibration: per-key (v - μ) / σ over the warmup
samples. Demonstration of an alternative to PercentileRollingCalibration."""

from __future__ import annotations

import math

from openpi.cache.components.factors.base import CalibrationSamples


class ZScoreOnFactorsCalibration:
    """Each verdict's factor value is mapped to its z-score against the
    bound key's warmup-sample distribution. NaN inputs propagate."""

    def __init__(self, samples: CalibrationSamples) -> None:
        if samples is None:
            raise ValueError("ZScoreOnFactorsCalibration requires CalibrationSamples")
        self._samples = samples
        self._stats: dict[str, tuple[float, float]] = {}   # key -> (mu, sigma)

    def bind_keys(self, keys: list[str]) -> None:
        # bind_keys is the FAIL-FAST hook (plan §6.3): every Layer-2 union
        # key must have enough non-NaN samples here, otherwise raise.
        for k in keys:
            samples = self._samples.samples.get(k)
            if samples is None:
                raise KeyError(f"calibration source missing key {k!r}")
            non_nan = [v for v in samples if not math.isnan(v)]
            if len(non_nan) < 30:
                raise ValueError(
                    f"key {k!r}: only {len(non_nan)} samples, need >= 30"
                )
            n = len(non_nan)
            mu = sum(non_nan) / n
            var = sum((x - mu) ** 2 for x in non_nan) / n
            sigma = max(var ** 0.5, 1e-6)
            self._stats[k] = (mu, sigma)

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in raw.items():
            if math.isnan(v) or k not in self._stats:
                out[k] = float("nan")
                continue
            mu, sigma = self._stats[k]
            out[k] = (float(v) - mu) / sigma
        return out

    def on_episode_start(self) -> None:
        return None     # stats are immutable post-bind_keys
```

Register (top of subclass + add to `calibrations/__init__.py`):

```python
# src/openpi/cache/components/factors/calibrations/__init__.py
from openpi.cache.components.factors.calibrations.my_zscore import (
    ZScoreOnFactorsCalibration,
)
```

`_build_calibration` adds a branch (`config.py`):

```python
if cfg.type == "z_score_on_factors":
    return ZScoreOnFactorsCalibration(samples, **dict(cfg.params))
```

Yaml reference:

```yaml
calibration:
  type: z_score_on_factors
  params: {}                         # forwarded as kwargs (this class has none)
  samples_source:
    type: warmup                     # warmup pool fills CalibrationSamples
```

⚠ Key contracts: `bind_keys` is the **only** fail-fast hook (called at startup); `__call__` receiving an unknown key MUST return NaN (do NOT raise — CompositeJudge has already done key-contract validation upstream). Plan §6.3 / §6.5 rule 3 strictly enforces no cold-start state.

---

### 8.3 Adding a Layer 4 Composer

Implement the `Composer` Protocol (`composers/base.py`): compute `self.declared_dependencies` (instance attribute) at construction, implement `bind_orientations(orientations)` and `compose(calibrated, *, winner_id)`.

```python
# src/openpi/cache/components/factors/composers/my_max.py

"""Max-only composer: max calibrated value across all weighted keys.
Useful when any single high-signal factor should suffice for hit."""

from __future__ import annotations

import math
from typing import Optional

from openpi.cache.components.judge import HitType, JudgeResult


class MaxComposer:
    """JudgeResult based on max(calibrated[k] for non-zero-weight keys).

    safe key: contributes calibrated[k]
    risky key: contributes 1 - calibrated[k]
    non_monotonic: requires `directions[k]` per orientation contract.
    """

    declared_dependencies: set[str]

    def __init__(
        self,
        *,
        weights: dict[str, float],
        full_hit_threshold: float,
        warm_start_threshold: Optional[float] = None,
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        self._weights = dict(weights)
        self._full = float(full_hit_threshold)
        self._warm = warm_start_threshold
        self._warm_t = warm_start_t
        self._directions = dict(directions or {})
        self._orientations: dict[str, str] = {}
        # Layer 4 contract — declare what factor keys we need
        self.declared_dependencies = {k for k, w in self._weights.items() if w != 0.0}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)
        missing = [
            k for k, ori in self._orientations.items()
            if ori == "non_monotonic" and self._weights.get(k, 0.0) != 0.0
            and k not in self._directions
        ]
        if missing:
            raise ValueError(f"non_monotonic keys missing directions: {sorted(missing)}")

    def compose(
        self,
        calibrated: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        contribs = []
        for k, w in self._weights.items():
            if w == 0.0:
                continue
            v = calibrated.get(k, float("nan"))
            if math.isnan(v):
                continue
            ori = self._orientations.get(k)
            if ori == "safe":
                contribs.append(v)
            elif ori == "risky":
                contribs.append(1.0 - v)
            else:
                # non_monotonic — handle per direction (omitted for brevity)
                contribs.append(v)
        if not contribs:
            return JudgeResult(HitType.MISS, composer_score=None)
        score = max(contribs)
        if score >= self._full:
            return JudgeResult(HitType.FULL_HIT, winner_id=winner_id, composer_score=score)
        if self._warm is not None and self._warm_t is not None and score >= self._warm:
            return JudgeResult(
                HitType.WARM_START, winner_id=winner_id,
                start_t=self._warm_t, composer_score=score,
            )
        return JudgeResult(HitType.MISS, composer_score=score)
```

Register in `composers/__init__.py` re-export table + add a branch to `_build_composer`:

```python
# composers/__init__.py
from openpi.cache.components.factors.composers.my_max import MaxComposer

# config.py _build_composer
if cfg.type == "max":
    return MaxComposer(
        weights=cfg.weights,
        full_hit_threshold=cfg.tier_thresholds["full_hit"],
        warm_start_threshold=cfg.tier_thresholds.get("warm_start"),
        warm_start_t=cfg.warm_start_t,
        directions=cfg.directions,
    )
```

Yaml reference:

```yaml
composer:
  type: max
  weights:
    jerk_online_state__p5_f5: 1.0
    direction_online_state__p5_f5: 1.0
  tier_thresholds: { full_hit: 0.7, warm_start: 0.4 }
  warm_start_t: 0.7
```

⚠ Key contracts:
- `declared_dependencies` is an **instance attribute** (not a classmethod), computed at construction. CompositeJudge `__init__` statically asserts `composer.declared_dependencies ⊆ Layer 2 union key set`.
- All WARM_START emission paths are bound by plan §3.6 / §13.3 rule 5c **CP1-only** — any composer subclass that outputs `HitType.WARM_START` cannot be configured under cp3 (validator rejects `warm_start_t` on cp3).
- Empty `calibrated` / all-NaN: it is recommended to return `JudgeResult(HitType.MISS)`. If your composer wants to return WARM_START in that scenario, see `WeightedSumWithWarmFallbackComposer` for the implementation pattern (also gated CP1-only via `composer.warm_fallback_start_t`).

---

### 8.4 Adding an OfflineWriter (so an offline factor can write artifact)

A `Factor` subclass that ALSO implements `required_payload_fields()` + `compute_for_episode(entries, library_stats)` automatically satisfies the `OfflineWriter` Protocol (duck typing). `exp/common/factor_postprocess.py:_load_offline_writers_from_yaml` discovers it via `hasattr(cls, 'compute_for_episode')`.

Full skeleton (a demo writing per-entry mean velocity magnitude over a chain `(P, F)` window):

```python
# src/openpi/cache/components/factors/my_pack.py (extends 8.1)

@register("mean_speed_offline_action")
class MeanSpeedOfflineAction:
    """Per-entry mean velocity magnitude over a (P, F) chain window."""

    requires_chain_walk: bool = False     # online path reads payload.factors
    required_top_k:      int = 0

    def __init__(self, *, windows: list[dict]) -> None:
        from openpi.cache.components.factors.base import normalize_windows
        self._windows = normalize_windows(windows)
        self.descriptor_orientations = self.__class__.describe({"windows": windows})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        from openpi.cache.components.factors.base import normalize_windows
        return {
            f"mean_speed_offline_action__p{p}_f{f}": "non_monotonic"
            for (p, f) in normalize_windows(params["windows"])
        }

    # ---- ONLINE path: read what the OfflineWriter wrote ----
    def extract(self, ctx) -> dict[str, float]:
        keys = list(self.descriptor_orientations)
        if not ctx.results:
            return {k: float("nan") for k in keys}
        winner = ctx.view.get(ctx.results[0].id)
        if winner.factors is None:
            return {k: float("nan") for k in keys}
        return {k: float(winner.factors.get(k, float("nan"))) for k in keys}

    # ---- OFFLINE path: writer surface ----
    def required_payload_fields(self) -> set[str]:
        return set()                                    # uses existing schema

    def compute_for_episode(
        self, entries, library_stats,
    ) -> list[dict[str, float]]:
        import torch
        keys = list(self.descriptor_orientations)
        T = len(entries)
        if T == 0:
            return []
        seq = torch.stack(
            [torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32) for e in entries],
            dim=0,
        )                                               # [T, A]
        sigma = library_stats.action_sigma.clamp_min(0.01)
        seq_norm = seq / sigma                          # [T, A]
        active = library_stats.action_active_mask
        pts = seq_norm[..., active]                     # [T, A_active]
        out: list[dict[str, float]] = []
        for k_idx in range(T):
            row: dict[str, float] = {}
            for (P, F) in self._windows:
                lo, hi = k_idx - P, k_idx + F
                if lo < 0 or hi >= T:
                    row[f"mean_speed_offline_action__p{P}_f{F}"] = float("nan")
                    continue
                w = pts[lo:hi + 1]
                v = (w[1:] - w[:-1]).norm(dim=-1)       # [W-1]
                row[f"mean_speed_offline_action__p{P}_f{F}"] = (
                    float(v.mean()) if v.numel() else float("nan")
                )
            out.append(row)
        return out
```

How `exp/common/factor_postprocess.py` finds this class: through `_load_offline_writers_from_yaml` going to the registry and checking `hasattr(cls, 'compute_for_episode')`, so **no second registration site** is needed.

**Write the artifact**:

```bash
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --factors-yaml my_factors.yaml \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16_v2.pkl

# my_factors.yaml
factors:
  - type: mean_speed_offline_action
    params:
      windows: [{past: 1, future: 1}, {past: 5, future: 5}]
```

The CLI only accepts the 8 offline factors (registry name + `hasattr(compute_for_episode)`); using an online factor or topk fails immediately. Full e2e smoke test: `tests/exp/common/test_build_enrich_existing_pkl.py`.

---

### 8.5 Authoring checklist (the five-step recipe)

1. **Be clear about physical meaning** — what signal does the factor capture? safe / risky / non_monotonic? does it need chain walk? does it need top-K?
2. **Write the class + `@register("name")`** — three methods (`__init__` / `describe(params)` / `extract(ctx)`) + capability flags.
3. **Add the import** — append `from . import my_pack` to `factors/registry.py` so the `@register` side effect runs when the registry module loads.
4. **Write the yaml + smoke** — copy the §3 full yaml template, swap `factors` / `calibration` / `composer` to point at the new names; run `load_cache_config` locally to confirm validator passes.
5. **Write tests** — at least: happy path + physical edge (NaN exit) + the classmethod `describe(params)` returning a value consistent with the instance's `descriptor_orientations`.

---

## 9. Refactor delta cheat-sheet (old → new)

| Dimension | Old | New |
|---|---|---|
| Number of factors | 5 (`f1a_a/f1a_t/f1b_a/f1b_t/f2`) | 17 (`<desc>_<source>_<channel>` × 16 + `topk_action_variance`) |
| Descriptor names | `jerk / dir / curv_radius / cum_disp` | `jerk / direction / dispersion / path_length` |
| Architecture | Factor + Normalizer + Composer + framework-level fallback | 4 orthogonal layers (Normalization / Factor / Calibration / Composer) |
| Cold-start | `cold_start_strategy: force_miss/passthrough/lenient` | **Removed** — startup fail-fast |
| `all_nan_fallback` yaml | `{type: warm_start, start_t}` | **Removed** — handled by `WeightedSumWithWarmFallbackComposer` subclass |
| `factor_outputs` fields | `{raw, norm, score, sentinel}` | `{raw, calibrated, composer_score, schema_version=2}` |
| `OfflineWriter` signature | `compute_for_episode(entries, library_stats)` | Unchanged |
| `OnlineExtractor` Protocol | Multi-arg `extract(results, view, history, cached_data)` | Single-arg `Factor.extract(ctx: FactorContext)` |
| Composer dependency check | classmethod `declared_dependencies(params)` | Instance attribute `composer.declared_dependencies` |
| Wire protocol | `__hit_meta__["factor_outputs"]` same | `schema_version=2`; legacy clients see field-absent = v1 |

Full design + decision history: [`logs/verdict_factor_judge_refactor.log.md`](../../logs/verdict_factor_judge_refactor.log.md) (G1 APPROVED Round 4 / G2 APPROVED Round 3, 2026-05-07).
