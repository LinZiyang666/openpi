# Verdict Factor Judge — User Guide

> **Prerequisites**: Read [tutorial.md](tutorial.md) §6 for the Judge component basics and §10 for YAML configuration; read [../architecture/cache_system.md](../architecture/cache_system.md) §5.6 / §5.11 / §5.12 for the verdict-factor architecture contract.
>
> **Design document**: full design and decision history is in [`logs/verdict_factor_judge.log.md`](../../logs/verdict_factor_judge.log.md) (Plan, G1 / G2 APPROVED).
>
> **Status (2026-04-26)**: B0 has landed — protocol layer, metadata skeletons for the 5 factors, `CompositeJudge` / `Composer` / `Normalizer` class shells, `PayloadView`, the `payload.factors` schema, the duck-typed facade, the config dataclass + fail-fast validator, and the unit tests. **Algorithm bodies and orchestrator wiring are not implemented yet**: B1 lands the `extract` bodies for `RuntimeContinuity` / `TopKActionConsensus` plus orchestrator view+history injection; B2 lands `SourceWindowSmoothness.extract` + `compute_for_episode` + `LibraryStats` + offline-build-pkl tooling. This guide is written against the *target* usable shape; each section ends with a `> **Batch status**: B0/B1/B2` marker calling out what actually executes today.

---

## 1. Overview

`CompositeJudge` replaces the single-threshold cosine match with a **"factor vector → Normalizer → Composer → JudgeResult"** three-stage pipeline. Multiple statistical / kinematic descriptors (jerk, direction consistency, curvature radius, cumulative displacement, top-K action variance) cooperate on the hit decision.

```
SearchResultLite[]  ─┐
PayloadView (view)   ├──► OnlineExtractor*  ──► raw: dict[str, float]
HistoryView          │   (key contract: keys MUST equal that
cached_data         ─┘    extractor's descriptor_orientations.keys())
                                      │
                                      ▼
                              Normalizer (optional, e.g. PercentileRollingNormalizer)
                                      │
                              norm: dict[str, float]
                              (all-NaN → CompositeJudge short-circuits to MISS)
                                      │
                                      ▼
                                  Composer (WeightedSum / AndGate / OrGate)
                                      │
                                      ▼
                                JudgeResult(hit_type, winner_id, start_t)
```

**When to choose `composite` vs `ThresholdJudge`**:

| Situation | Recommended |
|---|---|
| Cosine score distribution is already well-behaved; want a quick deployment | `ThresholdJudge` |
| Want to fold "retrieval-induced action discontinuity" risk into the verdict | `composite` + F1a-A |
| Need candidate-pool consistency as a guard | `composite` + F2 |
| Want cross-episode motion smoothness as a hit-quality signal | `composite` + F1b-A / F1b-T (requires LibraryStats at build time) |
| Multi-dimensional combination (the typical post-B2 default) | `composite` + F1b-A + F1b-T + F2, weighted_sum |

> **Batch status**: CompositeJudge class shell + 5 factor registrations land in B0. Algorithm bodies = B1 / B2.

---

## 2. The five factors at a glance

| Registry name | Class | Data source | requires_library_stats | requires_chain_walk | Algorithm batch |
|---|---|---|---|---|---|
| `f1a_a` | `RuntimeContinuityAction` | winner `payload.action_chunk[0]` + `history.actions` | True | False | B1 |
| `f1a_t` | `RuntimeContinuityState`  | winner `query_keys["robot_state"]` + `view.walk_next(winner_id, k)` | True | True | B1 |
| `f1b_a` | `SourceWindowSmoothnessAction` | OfflineWriter reads `entries[i].payload.action_chunk[0]` along the chain; OnlineExtractor reads `payload.factors` only | True | False | B2 |
| `f1b_t` | `SourceWindowSmoothnessState`  | OfflineWriter reads `entries[i].query_keys["robot_state"]` along the chain; OnlineExtractor reads `payload.factors` only | True | False | B2 |
| `f2`    | `TopKActionConsensus` | top-K `payload.action_chunk` via `view.get_many` | False | False | B1 |

**`source` vs `key_initial` decoupling**:

- `source: "action" | "state"` — semantic name of the data the factor consumes.
- `key_initial: "a" | "t"` — namespace for the keys the factor writes into `payload.factors`. Aligned with the registry name suffix so YAML configs that reference `f1a_t_jerk` find an extractor that actually produces that key (rather than `f1a_s_jerk` from a naive `source[0]` derivation).
- Factors with `requires_library_stats=True` require `backend.type=in_memory` (currently the only backend exposing `library_stats`).
- `requires_chain_walk=True` (only F1a-T today) requires a backend that exposes the `fetch_entry` capability (currently only InMemoryBackend).
- These capability-vs-backend constraints are enforced at config-load time by `validate_cache_config` (rules #3 / #4 of the 6 static composite checks added in B1+).

**Descriptor set (shared by F1a + F1b)**:

| Descriptor | Orientation | Formula | Meaning |
|---|---|---|---|
| `jerk` | risky | `median \|Δ²a / σ\|` (over active DOFs) | Acceleration-change magnitude; high → not smooth |
| `dir`  | safe  | `mean cos(v[t], v[t+1])` | Direction consistency; high → smooth |
| `curv_radius` | non_monotonic | `mean ‖p[t] − centroid‖` | Window geometric dispersion (medium = arc, very small = stuck, very large = long straight) |
| `cum_disp`    | non_monotonic | `sum ‖p[t+1] − p[t]‖` | Cumulative path length (small + low jerk = stationary; large = fast motion) |

`risky` / `safe` orientations let the Composer auto-flip score signs; `non_monotonic` keys MUST get an explicit `composer.directions` entry of `"high"` / `"low"` / `"range:[lo,hi]"` — otherwise the validator rejects the config (preventing accidental force-fitting of non-monotonic descriptors into a monotone aggregation).

> **Batch status**: All 5 factors' metadata (capability flags + describe + register) lands in B0. `extract` / `compute_for_episode` algorithm bodies: F1a / F2 = B1, F1b = B2.

---

## 3. Full lifecycle

```
       ┌─────────────────────────────────────────────────────────────┐
       │  Required for B2; skipped in B0/B1: build pkl + write       │
       │  LibraryStats + write payload.factors                       │
[Step1]│  exp/common/build_in_memory_cache_artifact.py +             │
       │  factor_postprocess.py                                       │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Write YAML (type: composite + factors / composer /          │
[Step2]│  normalizer)                                                 │
       │  cache_composite_judge.yaml                                  │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  validate_cache_config → 6 static composite checks           │
[Step3]│  build_per_connection_components → CompositeJudge wiring     │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  scripts/serve_policy.py --cache_config xxx.yaml            │
[Step4]│  Online inference: each verdict runs view+history → extract │
       │  → norm → compose → JudgeResult                             │
       └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Run experiments (exp/cp1_cache + analysis scripts);         │
[Step5]│  compare threshold judge vs composite judge on               │
       │  success_rate / hit rate / mean start_t                      │
       └─────────────────────────────────────────────────────────────┘
```

> **Batch status**: The full 5-step walk only becomes meaningful after B2 lands. In B0, you can exercise Step 2-3 to verify the fail-fast behavior (the YAML is rejected at config load and never reaches Step 4).

---

## 4. Step 1: Build an offline artifact with factors (B2)

### 4.1 Command

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_mean_pool \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_with_factors.pkl \
    --factor-config-yaml exp/common/configs/factors/f1b_default.yaml
```

`--factor-config-yaml` points at a YAML describing which OfflineWriters to run with what parameters (same shape as the `factors` block in the online judge YAML; only factors that implement the `OfflineWriter` Protocol matter — F1b-A / F1b-T). Example:

```yaml
# exp/common/configs/factors/f1b_default.yaml
factors:
  - type: f1b_a
    params:
      windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
  - type: f1b_t
    params:
      windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
      descriptors: [jerk, dir, curv_radius, cum_disp]
      active_eps: 0.01
```

### 4.2 Internal flow

1. `build_in_memory_cache_artifact.py` runs the KeyBuilder over all entries and produces `entries: list[CacheEntry]`.
2. The `enrich_artifact_with_factors(entries, offline_writers)` helper (lives in `exp/common/factor_postprocess.py`):
   - Calls `LibraryStats.compute_from_entries(entries, active_eps_action, active_eps_state)`.
   - Splits episodes by `entry.trajectory_id`; for each OfflineWriter calls `writer.compute_for_episode(per_episode_entries, library_stats)`.
   - Merges the returned `list[dict[str, float]]` into `entries[i].payload.factors`.
3. Writes the artifact pkl:

```python
{
    "key_builder_type": "cp1_mean_pool",
    "checkpoint_id": "CP1",
    "vector_dims": {...},
    "entries": [...],            # entry.payload.factors filled in
    "library_stats": LibraryStats(...),  # new field
}
```

### 4.3 Backwards compatibility for old artifacts

Artifacts that lack a `library_stats` field still load — `InMemoryBackend.load_artifact` falls back automatically:

```python
self.library_stats = data.get("library_stats")
if self.library_stats is None:
    logger.info("Artifact missing library_stats; computing from %d entries", len(self._entries))
    self.library_stats = LibraryStats.compute_from_entries(list(self._entries.values()))
```

The compute happens once at startup; the verdict hot path never recomputes. Old entries with `payload.factors is None` cause F1b's `OnlineExtractor` to return NaN, and the Composer skips the factor per its orientation rule.

> **Batch status**: The build-pkl tool, helper, `LibraryStats.compute_from_entries` algorithm, and `SourceWindowSmoothness.compute_for_episode` are the full B2 set. In B0, calling either method raises NotImplementedError.

---

## 5. Step 2: Write the YAML

```yaml
# exp/cp1_cache/configs/composite_judge_demo.yaml
enabled: true

key_builder:
  type: cp1_mean_pool

keys:
  vision_0: { enabled: true, weight: 1.0 }
  vision_1: { enabled: true, weight: 1.0 }
  robot_state: { enabled: true, weight: 0.5 }

backend:
  type: in_memory                      # composite + F1a/F1b require in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_with_factors.pkl

checkpoints:
  cp1:
    enabled: true
    judge:
      type: composite                  # In B0 this is rejected at config load (fail-fast)
      factors:
        - type: f1a_a
          params:
            window_k: 5
            descriptors: [jerk, dir, curv_radius, cum_disp]
        - type: f1b_a
          params:
            windows: [{past: 0, future: 5}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]
            active_eps: 0.01
        - type: f1b_t
          params:
            windows: [{past: 0, future: 5}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]
            active_eps: 0.01
        - type: f2
          params:
            K: 5
      composer:
        type: weighted_sum
        weights:
          f1a_a_jerk:                      0.10
          f1a_a_dir:                       0.10
          f1b_a_jerk__p0_f5:               0.10
          f1b_a_dir__p0_f5:                0.10
          f1b_t_jerk__p0_f5:               0.10
          f1b_t_dir__p0_f5:                0.10
          f1b_a_curv_radius__p5_f5:        0.10
          f1b_a_cum_disp__p5_f5:           0.10
          f1b_t_curv_radius__p5_f5:        0.05
          f1b_t_cum_disp__p5_f5:           0.05
          f2_var:                          0.10
        tier_thresholds:
          full_hit:   0.80
          warm_start: 0.60
        warm_start_t: 0.5            # Must be in CANONICAL_DENOISE_TIMESTEPS
        directions:
          # Required for every non_monotonic key with non-zero weight; validator rejects otherwise
          f1b_a_curv_radius__p5_f5:  range:[0.3, 1.0]
          f1b_a_cum_disp__p5_f5:     high
          f1b_t_curv_radius__p5_f5:  range:[0.3, 1.0]
          f1b_t_cum_disp__p5_f5:     high
      normalizer:
        type: percentile_rolling
        window_size: 200
        cold_start_strategy: force_miss   # Default: first 200 verdicts forced to MISS via all-NaN sentinel
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1                           # F2 transparently bumps this to 5 via min_top_k_hint
  cp3:
    enabled: false
```

**Field-level notes**:

- Keys in `composer.weights` MUST match what extractor `describe()` actually produces — typos are silently treated as missing keys (B1+ adds stricter coverage validation).
- `composer.tier_thresholds.full_hit` and `warm_start` are normalized scores in [0, 1]; `warm_start` must be `< full_hit_threshold` (otherwise the warm-start tier is unreachable — caught by validator rule 5d).
- `warm_start_t` must lie in `CANONICAL_DENOISE_TIMESTEPS = {0.1, 0.2, ..., 0.9}` (same rule as `always_warm_start.start_t`); CP3 does not support warm_start.
- `directions` only applies to `non_monotonic` factors. Three forms:
  - `"high"` — higher value is more hit-leaning
  - `"low"`  — lower value is more hit-leaning
  - `"range:[lo,hi]"` — values in [lo, hi] are more hit-leaning

### 5.1 The 6 static composite-specific checks (activated B1+)

`validate_cache_config` runs these 6 rules at config-load time; all must pass before the builder runs:

1. `factors` and `composer` MUST be present when `type=composite`.
2. Every `FactorConfig.type` MUST be in `factors.registry.known()`.
3. Factors with `requires_library_stats=True` require `backend.type == "in_memory"`.
4. Factors with `requires_chain_walk=True` (F1a-T) require `backend.type == "in_memory"` (the only backend exposing `fetch_entry`).
5. Composite warm-start checks (4 sub-rules):
   - 5a `composer.warm_start_t` only allowed on CP1.
   - 5b `warm_start_t` MUST be in `CANONICAL_DENOISE_TIMESTEPS` (the normalized value is written back).
   - 5c Pairwise rule: `tier_thresholds.warm_start` and `composer.warm_start_t` MUST be both present or both absent (and / or composer do not support warm_start; setting the timestep raises).
   - 5d Tier ordering: `tier_thresholds.warm_start < tier_thresholds.full_hit` (only weighted_sum).
6. `directions` coverage: every key with `non_monotonic` orientation reported by `cls.describe(params)` whose `composer.weights[key] != 0` MUST have a valid `composer.directions[key]` entry.

> **Batch status**: YAML schema parsing + B0 fail-fast (`_JUDGE_TYPES` excludes `composite` → rejected at config load) lands in B0; the 6 static checks + algorithm bodies land in B1. Running the YAML above in B0 produces: `judge.type='composite' is not yet enabled in B0; available in B1+ when CompositeJudge algorithms land.`

---

## 6. Step 3: Start inference

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config exp/cp1_cache/configs/composite_judge_demo.yaml
```

Startup sequence (B1+):

```
load_cache_config(yaml)
  ├─ _dict_to_dataclass → CacheConfig (incl. JudgeConfig.factors: list[FactorConfig])
  └─ validate_cache_config → 6 composite-specific checks pass

build_cache_components(config)
  └─ build_per_connection_components(config, storage)
      ├─ _build_backend → InMemoryBackend (.library_stats from artifact)
      ├─ per-CP loop:
      │    ├─ library_stats = per_conn_storage.library_stats   # facade duck-types backend
      │    ├─ judges[cp_id] = _build_judge(judge_cfg, library_stats)
      │    │     └─ if type == "composite":
      │    │          ├─ extractors = [
      │    │          │     cls(**dict(f.params),
      │    │          │         library_stats=library_stats if cls.requires_library_stats else ...)
      │    │          │     for f in cfg.factors
      │    │          │  ]
      │    │          ├─ composer  = _build_composer(cfg.composer)
      │    │          ├─ normalizer = _build_normalizer(cfg.normalizer)
      │    │          └─ CompositeJudge(extractors, composer, normalizer)
      │    │              # constructor auto-collects every descriptor_orientations
      │    │              # and calls composer.bind_orientations + normalizer.bind_keys
      │    ├─ min_hint = judges[cp_id].min_required_top_k
      │    └─ strategies[cp_id] = _build_search_strategy(ss_cfg, ..., min_top_k_hint=min_hint)
      ├─ offline_writers = collect_offline_writers_from_judges(judges)
      └─ Orchestrator(..., offline_writers=ow, library_stats=library_stats)
```

**Single verdict path** (B1+, runs once per `Orchestrator.check()`):

```python
view    = StoragePayloadView(self._storage)              # per-check lifetime, internal memo
history = HistoryView(actions=list(self._action_history),
                      states =list(self._state_history))

judge_result = judge(
    results, checkpoint_id, self._key_builder.cached_data,
    view=view, history=history,
)
# Inside CompositeJudge:
#   raw = {}
#   for ext in extractors:
#       out = ext.extract(results, view, history, cached_data)
#       assert out.keys() == ext.descriptor_orientations.keys()  # key contract
#       raw.update(out)
#   norm = normalizer(raw)
#   if norm and all(isnan(v) for v in norm.values()):
#       return JudgeResult(MISS)        # cold-start sentinel
#   return composer.compose(norm, winner_id=results[0].id)

if hit_type in (FULL_HIT, WARM_START):
    payload = view.get(winner_id)        # memo shared with extractors
```

> **Batch status**: Orchestrator view+history injection + winner fetch rewire + state_history anchor-checkpoint policy = B1. In B0, `Orchestrator.check()` does not inject anything; legacy judges absorb the new kwargs via `**kwargs` and behave bit-identically.

---

## 7. Step 4: Run experiments comparing ThresholdJudge vs CompositeJudge

```bash
# Phase 1: Baseline run with ThresholdJudge
uv run python exp/cp1_cache/run_cache_experiments.py \
    --config exp/cp1_cache/configs/threshold_baseline.yaml \
    --env LIBERO --num-episodes 50

# Phase 2: Run with CompositeJudge on the same artifact
uv run python exp/cp1_cache/run_cache_experiments.py \
    --config exp/cp1_cache/configs/composite_judge_demo.yaml \
    --env LIBERO --num-episodes 50

# Phase 3: Compare
uv run python exp/cp1_cache/analysis/compare_judges.py \
    --baseline-dir exp/cp1_cache/runs/threshold_baseline \
    --composite-dir exp/cp1_cache/runs/composite_judge_demo
```

Key metrics to watch:

| Metric | Expected comparison |
|---|---|
| `success_rate` | composite ≥ threshold (better hit quality) |
| `cp1_full_hit_rate` | composite may decrease (more conservative); tunable via weights |
| `cp1_warm_start_rate` | composite increases (extra tier) |
| `mean_start_t` | composite stays at the configured `warm_start_t` |
| `composite_factor_log` | per-verdict factor values (logger added in B1+) — used to retune weights |

> **Batch status**: The experiment runner and analysis scripts themselves are B0+ (tooling already exists). CompositeJudge actually participating = B1+; F1b factors usable = B2.

---

## 8. Custom factors: extending OnlineExtractor / OfflineWriter

### 8.1 Write a new OnlineExtractor

Minimal implementation (no OfflineWriter, no library_stats, no chain walk):

```python
# src/openpi/cache/components/factors/my_factor.py
from openpi.cache.components.factors.registry import register


@register("my_factor")
class MyFactor:
    # ---- Required class-level capability flags ----
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = False

    def __init__(self, threshold: float):
        self._threshold = threshold
        # Instance-level orientation map MUST come from the classmethod so the
        # validator can fetch the same key list without instantiating.
        self.descriptor_orientations = self.__class__.describe(
            {"threshold": threshold}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        # Pure function: only reads params, no library_stats / no I/O
        return {"my_factor_score": "safe"}

    def extract(self, results, view, history, cached_data) -> dict[str, float]:
        # Returned dict's keys MUST equal self.descriptor_orientations.keys();
        # otherwise CompositeJudge raises RuntimeError("key contract violation")
        # at verdict time.
        winner_payload = view.get(results[0].id)
        score = float(winner_payload.action_chunk[0].abs().mean().item())
        return {"my_factor_score": score}
```

Once registered, you can use it directly in YAML:

```yaml
factors:
  - type: my_factor
    params:
      threshold: 0.5
```

### 8.2 A factor that needs library_stats

```python
@register("my_normed_factor")
class MyNormedFactor:
    required_top_k: int = 0
    requires_library_stats: bool = True   # builder will inject the library_stats kwarg
    requires_chain_walk: bool = False

    def __init__(self, library_stats: "LibraryStats"):
        self._sigma = library_stats.action_sigma
        self.descriptor_orientations = self.__class__.describe({})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"my_normed_score": "risky"}

    def extract(self, results, view, history, cached_data):
        ...
```

`requires_library_stats=True` forces the validator to require `backend.type=in_memory` (only InMemoryBackend exposes `library_stats`) and makes `_build_judge` inject `library_stats=library_stats` at construction time.

### 8.3 A factor that walks the chain

```python
@register("my_chain_factor")
class MyChainFactor:
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = True   # validator forces backend to expose fetch_entry

    def __init__(self, walk_depth: int):
        self._walk_depth = walk_depth
        self.descriptor_orientations = self.__class__.describe(
            {"walk_depth": walk_depth}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"my_chain_drift": "risky"}

    def extract(self, results, view, history, cached_data):
        winner_id = results[0].id
        # In B0, PayloadView only supports ForkPolicy.TRAJECTORY; raises on real forks
        forward_entries = view.walk_next(winner_id, k=self._walk_depth)
        ...
```

### 8.4 An OfflineWriter (the B2 model)

`OfflineWriter` is invoked at build-pkl time by the `enrich_artifact_with_factors` helper to populate `entry.payload.factors`:

```python
@register("my_offline_factor")
class MyOfflineFactor:
    # ---- OnlineExtractor surface ----
    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = False

    def __init__(self, ...):
        self.descriptor_orientations = self.__class__.describe({...})

    @classmethod
    def describe(cls, params):
        return {"my_offline_score": "safe"}

    def extract(self, results, view, history, cached_data):
        # Online side reads the value the offline writer pre-computed
        winner_payload = view.get(results[0].id)
        if winner_payload.factors is None or "my_offline_score" not in winner_payload.factors:
            return {"my_offline_score": float("nan")}     # Old entries / missing keys → NaN
        return {"my_offline_score": winner_payload.factors["my_offline_score"]}

    # ---- OfflineWriter surface ----
    def required_payload_fields(self) -> set[str]:
        return set()      # No extra raw payload tensors needed

    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: "LibraryStats",
    ) -> list[dict[str, float]]:
        # Returned list length MUST equal len(entries); parallel to entries
        return [{"my_offline_score": ...} for _ in entries]
```

### 8.5 Write a new Composer

```python
# src/openpi/cache/components/factors/composers/my_composer.py
from openpi.cache.components.judge import HitType, JudgeResult


class MyComposer:
    def __init__(self, threshold: float):
        self._threshold = threshold
        self._orientations: dict[str, str] = {}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        # Called once by CompositeJudge.__init__; validate / store as needed
        self._orientations = dict(orientations)

    def compose(self, factors: dict[str, float], *, winner_id: str) -> JudgeResult:
        # `factors` is already Normalizer-processed; Composer chooses how to handle NaN
        if all(v >= self._threshold for v in factors.values() if not (v != v)):
            return JudgeResult(HitType.FULL_HIT, winner_id)
        return JudgeResult(HitType.MISS)
```

Register in `_build_composer` — currently an explicit if-elif tree:

```python
# src/openpi/cache/config.py _build_composer
elif cfg.type == "my_composer":
    return MyComposer(threshold=cfg.tier_thresholds["full_hit"])
```

> **Future work**: Composer and Normalizer could also use the registry pattern (like factors do); not in the current plan.

### 8.6 Write a new Normalizer

```python
class MyNormalizer:
    def __init__(self, ...):
        self._keys: list[str] = []

    def bind_keys(self, keys: list[str]) -> None:
        self._keys = list(keys)

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        # Returned dict MUST contain the same keys as `raw`
        # To trigger cold-start MISS: return an all-NaN dict (CompositeJudge short-circuits)
        ...

    def on_episode_start(self) -> None:
        # Default no-op; override only if per-episode rolling-window reset is desired
        return None
```

Register in the `_build_normalizer` if-elif tree.

> **Batch status**: The API surface for adding custom factor / Composer / Normalizer is B0-ready, but CompositeJudge cannot actually run until B1 lands. In B0 you can still unit-test custom factors' `describe` + class construction (see the stub-test pattern in `tests/cache/components/factors/test_registry.py`).

---

## 9. Module file map

| Path | Role |
|---|---|
| `src/openpi/cache/components/factors/base.py` | OnlineExtractor / OfflineWriter Protocol + LibraryStats + HistoryView |
| `src/openpi/cache/components/factors/registry.py` | `register / get_class / build / known` |
| `src/openpi/cache/components/factors/runtime_continuity.py` | F1a-A / F1a-T thin subclasses |
| `src/openpi/cache/components/factors/source_window.py` | F1b-A / F1b-T thin subclasses + `_DESCRIPTOR_ORIENTATIONS` + `_normalize_windows` |
| `src/openpi/cache/components/factors/consensus.py` | F2 |
| `src/openpi/cache/components/factors/composers/__init__.py` | Composer Protocol + WeightedSum / AndGate / OrGate |
| `src/openpi/cache/components/factors/normalizers/__init__.py` | Normalizer Protocol + PercentileRollingNormalizer |
| `src/openpi/cache/components/payload_view.py` | PayloadView Protocol + StoragePayloadView + ForkPolicy |
| `src/openpi/cache/components/judge.py` | `CompositeJudge` (alongside ThresholdJudge / AlwaysHit / AlwaysWarmStart) |
| `src/openpi/cache/storage_types.py` | `CachePayload.factors` Optional field |
| `src/openpi/cache/cache_storage.py` | `fetch_entry` + `library_stats` duck-typed facade |
| `src/openpi/cache/backends/in_memory_backend.py` | `fetch_entry` public method + `library_stats` attribute |
| `src/openpi/cache/config.py` | `FactorConfig / ComposerConfig / NormalizerConfig` + `_build_composer / _build_normalizer / _build_judge` composite branch + B0 `_JUDGE_TYPES` excludes `composite` (fail-fast at load) |
| `tests/cache/components/factors/` | factor metadata + Composer / Normalizer protocol + CompositeJudge unit tests |
| `tests/cache/test_payload_view.py` | StoragePayloadView unit tests |
| `tests/cache/test_cache_storage_factor_facade.py` | fetch_entry / library_stats facade tests |
| `tests/cache/test_config_factor.py` | FactorConfig / ComposerConfig parsing + B0 rejection tests |

---

## 10. FAQ

### Q: Why is the `f1a_t` key namespaced as `f1a_t_jerk` rather than `f1a_s_jerk`?

`source` is the factor's semantic field (`"action"` / `"state"`); `key_initial` is the `payload.factors` namespace (`"a"` / `"t"`), aligned with the registry name suffix. This way YAML configs that reference `f1a_t_jerk` perfectly match the extractor's actual output, avoiding silent weight misalignment (decided in G2 Round 1 → Round 2).

### Q: What happens with old entries (`payload.factors is None`)?

F1b's `OnlineExtractor.extract` returns NaN, and the Composer skips that factor per its orientation rule (weighted_sum: not counted into the sum or weight total; and-gate: treated as a fail; or-gate: treated as pass-but-ignored). NaN is the legal expression of "signal missing"; it never crashes the verdict.

### Q: What happens during cold-start (rolling window not yet full)?

`PercentileRollingNormalizer(cold_start_strategy="force_miss")` returns an **all-NaN dict** while the window is still filling; CompositeJudge detects `all(isnan(v))` and **short-circuits to MISS**, never invoking the Composer. This guarantees that during cold-start the cache routes through the inference path and never issues an aggressive hit based on unreliable percentile.

Other strategies: `"passthrough"` (use raw values directly) and `"lenient"` (compute percentile from partial samples; below 10 falls back to force_miss).

### Q: Doesn't F2's `K=5` break the strategy's `top_k=1` semantics?

No. CompositeJudge collects `min_required_top_k = max(extractor.required_top_k for extractor)` and feeds it into `SearchStrategy`'s new `min_top_k_hint` kwarg; `SearchStrategy` uses `max(yaml_top_k, min_top_k_hint)` as the actual fetch count. The YAML's `top_k: 1` retains its semantics ("the strategy needs 1") while F2 transparently bumps it to 5 in the background. In-memory backend benchmarks show topk(5) vs topk(1) cost is negligible.

### Q: What if I configure `f1a_t` against a Qdrant backend?

`validate_cache_config` raises at config-load time: factors with `requires_chain_walk=True` require `backend.type=in_memory` (only InMemoryBackend exposes `fetch_entry`). Fail-fast at load — never reaches the inference path.

### Q: How do I pick `direction` for a `non_monotonic` factor?

- `"high"` — higher value is better (e.g. large `cum_disp` = strong sustained motion)
- `"low"`  — lower value is better
- `"range:[lo, hi]"` — falls inside [lo, hi] is better (e.g. `curv_radius` range:[0.3, 1.0] expresses "prefer medium dispersion: stationary or long straight motion are both penalized")

Concrete values come from data calibration (B2+ may add a percentile-scanning calibration script suggesting recommended values).

### Q: Can I run a composite YAML in B0?

No. B0 ships the shell, but `_JUDGE_TYPES` excludes `composite`, so the validator raises directly at config load: `judge.type='composite' is not yet enabled in B0; available in B1+ when CompositeJudge algorithms land.` This is intentional fail-fast: it avoids stub composers running until the first verdict before reporting NotImplementedError.

What you can do in B0:
- Write `describe()` / construction unit tests for custom factors.
- Construct `CompositeJudge` directly via `_build_judge(cfg, library_stats=None)` and exercise its collect+bind+key contract+cold-start sentinel paths.
- Review the schema / config validation chain to confirm it matches expectations.

### Q: How do I debug a verdict failure?

1. CompositeJudge raises `ValueError("conflicting orientations")` at `__init__` — two extractors declare the same key with different orientations; check both factors' `descriptor_orientations`.
2. CompositeJudge raises `RuntimeError(... key contract violation)` at verdict time — the keys returned by `extract()` differ from what was declared; check the extract implementation.
3. Composer reports a missing weight key — extractor did not produce it / the key namespace is wrong (F1a-T is `f1a_t_*`, not `f1a_s_*`).
4. Validator reports `directions[K] missing for non_monotonic factor with non-zero weight` — explicitly add it to YAML `composer.directions`.

### Q: Can I skip the Normalizer and feed raw factor values directly to the Composer?

Yes. Omit the `normalizer:` field in YAML or have `_build_normalizer` return None, and CompositeJudge passes the raw dict straight to the Composer. But then `tier_thresholds` must be calibrated to the raw scale (no longer [0, 1] percentile rank), which is usually harder. Recommended: keep at least PercentileRollingNormalizer to standardize scores.
