# Cold-start dead loop in PyTorch + Pi0.5 verdict-factor cache judge

> Status: open question
> Date: 2026-04-27
> Affected: `openpi` cache subsystem, Phase 1 single-factor ablation
> Author: Ziyang Lin (with diagnosis assistance)

## TL;DR

We're running a "verdict-factor judge" experiment on top of the openpi cache
system (PyTorch + Pi0.5 inference path). Phase 0 (AlwaysHit baseline +
factor dump) reproduces the expected success rate (~0.69, the cache's
historical Ceiling-A) and produces a healthy calibration JSONL with
non-trivial factor distributions. Phase 1 (CompositeJudge using those
factors) produces **100% MISS in live LIBERO runs** — verified by direct
visual observation of `serve_policy.py` stdout for a continuous hour.
Offline replay of the same yaml's normalizer + composer pipeline on Phase 0
raw values predicts ~91% WARM_START. The discrepancy is real and we believe
it's a **cold-start bootstrap problem that mutates into a stable dead loop**.
We're looking for advice on whether this is a known issue with verdict-style
cache judges, and what bootstrap mechanism is appropriate.

---

## 1. System under test

- Repo: `openpi` (fork of Physical-Intelligence/openpi); PyTorch path only,
  JAX disabled.
- Inference: Pi0.5, action_chunk = 50 actions per inference call.
- Eval: LIBERO `libero_spatial`, 10 tasks × 10 trials per task = 100 ep per
  yaml, max_steps = 220 per ep, replan_steps = 8.
- Cache library: 1018 entries (49 trajectories), in-memory backend, 4
  query keys (`vision_0`, `vision_1`, `prompt_emb`, `robot_state`),
  `weighted_rrf_knn` retrieval, `top_k=5` (auto-widened from yaml's 1 by
  CompositeJudge's `min_required_top_k` hint to satisfy F2's `K=5`).
- Per inference, the cache emits one of three verdicts:
  - `FULL_HIT`  — execute cached `action_chunk[0]` (saves all of inference)
  - `WARM_START`— hand model the cached intermediate at denoise t=0.7
                  (saves stage-3 flow-matching only ≈ 50% of inference)
  - `MISS`      — pure inference, no cache benefit

## 2. Verdict-factor pipeline

A `CompositeJudge` aggregates 5 factor families per verdict step:

| Factor | What it measures | Source | Failure modes (NaN drivers) |
|---|---|---|---|
| F1a-A | runtime action splice continuity (executed history + candidate's first action) | `history.actions[-K:]` + `winner.action_chunk[0]` | `len(history.actions) < K=3` (boundary, ep start only) |
| F1a-T | runtime state splice continuity (observed history + walked-forward state chain) | `history.states[-K:]` + `view.walk_next(winner_id, k=K=3)` | walk_next runs out of forward entries (winner near trajectory tail) |
| F1b-A | offline action-side smoothness over W-MIX windows on the candidate's source trajectory | per-entry `payload.factors` (precomputed at build time) | window crosses trajectory boundary → NaN per window |
| F1b-T | same as F1b-A but state-side | same | same |
| F2    | top-K candidate-action variance (consensus signal) | `view.get_many(top-K results)` action_chunk[0] variance | `K_eff < 2`, OR all variance dims < eps (candidates too similar) |

Each factor produces 1 to 20 named keys (descriptors × windows). All keys
go through `PercentileRollingNormalizer(window_size=200,
cold_start_strategy=force_miss)` then a `WeightedSumComposer` with uniform
weights and threshold = 0.5 (FULL_HIT) / 0.3 (WARM_START tier under
T-DUAL).

Composer pseudo-formula:

```
contrib(k) = percentile(k)            if orientation == "safe"   (e.g. dir)
           = 1 - percentile(k)        if orientation == "risky"  (e.g. jerk, f2_var)
           = 1 if lo <= percentile <= hi else 0    if non_monotonic range
score = sum(weight × contrib over non-NaN keys) / sum(weight over non-NaN keys)
hit = FULL_HIT  if score >= 0.5
    = WARM_START if score >= 0.3 (T-DUAL only)
    = MISS otherwise
```

Cold-start sentinel: `if all(values are NaN) → MISS` (`judge.py:346`).
This is meant as a safety net during the first ~200 verdicts before the
rolling buffer fills.

## 3. Phase 0 evidence (works as expected)

Phase 0 yaml = `AlwaysHit` judge wrapped in a transparent `DumpingJudge`
that side-channels every verdict's raw factor values to JSONL. AlwaysHit
forces the cache to win every step regardless of factors, so the
trajectory state stays in-distribution. Each cfg ran 100 ep (subset of
the warm_start 500-ep baseline; success rates within Wilson CI of full
baseline):

| cfg | n ep | success | calibration JSONL rows | F1a-A NaN% | F1a-T NaN% | F1b-A mean NaN% | F2 NaN% |
|---|---:|---:|---:|---:|---:|---:|---:|
| clip_w7_d4 | 100 | 0.70 | 2856 | 16% | 41% | 26% (10-44%) | 0% |
| max_pool_w3_d5 | 100 | 0.66 | 2971 | 16% | 44% | 28% (10-44%) | 0% |
| spatial16_w8_d4 | 100 | 0.65 | 2936 | 16% | 43% | 27% (10-44%) | 0% |

All factors produce signal. F1a-A 16% mean NaN is mostly the per-ep
boundary (steps 0-2, `len(history.actions) < K=3`) plus `dir`'s
zero-norm gate. F2 is rock-solid at 0% NaN. Cross-cfg distributions are
nearly identical. Phase 0 success ≈ Ceiling-A (warm_start baseline 0.69)
within Wilson 95% CI — confirms the test loop is sane.

## 4. Phase 1 problem

Phase 1 swaps out AlwaysHit for `CompositeJudge` with the same 5 factors.
The first four yamls in lex order produced (over 100 ep each):

| yaml | factor set | tier | success rate (n=100) |
|---|---|---|---:|
| 1. f_f1a_t_only_d_all_t_full | F1a-T only (4 keys) | T-FULL (threshold 0.5) | 0.97 |
| 2. f_f1b_a_only_d_all_t_full | F1b-A only (20 keys) | T-FULL | 0.99 |
| 3. f_f1b_t_only_d_all_t_full | F1b-T only (20 keys) | T-FULL | 0.99 |
| 4. f_full_d_all_t_dual_07    | all 5 factors (49 keys) | T-DUAL: full_hit 0.5 + warm_start 0.3 (start_t 0.7) | 0.97 |

For reference baselines (clip cfg, n=500 each from Phase 0):

```
Floor (pure inference, no cache):              0.992
Ceiling-A (AlwaysHit, full cache):             0.674
Ceiling-W (AlwaysWarmStart at start_t=0.7):    0.980
```

All four Phase 1 yamls' success ≈ Floor. Critically, **the operator
watched `serve_policy.py` stdout in real time for over an hour and
observed every single verdict line as `judge: MISS` — zero `FULL_HIT`,
zero `WARM_START`, across all four yamls including the F-FULL × T-DUAL
yaml**. Statistical noise can't explain this; it is empirically 100%
MISS at the verdict level.

## 5. Offline replay diverges from live behavior

We replayed each Phase 1 yaml's exact normalizer + composer pipeline (built
from the actual yaml via `openpi.cache.config.load_cache_config` +
`_build_composer`) on Phase 0 calibration JSONL row by row. This bypasses
the model and LIBERO env entirely — pure factor → normalizer → composer
arithmetic.

| yaml | replay FULL_HIT | replay WARM_START | replay MISS |
|---|---:|---:|---:|
| 1. f_f1a_t_only | 22.4% | 0% | 77.6% (47% cold-sentinel + 31% below threshold) |
| 2. f_f1b_a_only | 33.3% | 0% | 66.7% |
| 4. f_full_d_all_t_dual_07 | 0% | **90.9%** | 9.1% (7% cold-sentinel + 2% composer) |
| 8. f_min_cons (F2 only) | 45.8% | 0% | 54.2% |

The replay says yaml 4 should produce ~91% WARM_START with the threshold
0.3 fallback tier doing the work. Live observation says 100% MISS.

## 6. Wiring sanity check (rules out simple bugs)

Loaded yaml 4 locally and inspected the constructed components:

```python
judges.CP1: CompositeJudge
  min_required_top_k = 5          # F2's K=5 propagated correctly
search_strategies.CP1: WeightedRrfKnnStrategy
  _top_k = 5                      # auto-widened from yaml's top_k=1
```

So F2 will receive 5 candidates per verdict — `K_eff >= 2` always
satisfied, F2 ≠ NaN by the K_eff path (consensus.py:86-89).

Earlier in this experiment we found and fixed a separate F1a-A bug where
`Orchestrator.broadcast_action` appended the full `[chunk_len, A]`
action chunk to `_action_history` instead of the per-step
`action_chunk[0]`, causing F1a-A's `_build_action_splice` to silently
produce NaN for every verdict (commit `7a159db`). Phase 0 calibration
above was collected after that fix. Phase 1 also runs against the fixed
orchestrator. So this is a **separate failure mode**, not a regression
of the F1a-A wiring.

## 7. Hypothesis: cold-start dead loop driven by state-distribution drift

What we believe is happening, in order:

1. **Cold start (~steps 0-200, ~7-12 ep)**: PercentileRollingNormalizer
   in `force_miss` mode emits NaN for every key until its rolling buffer
   has 200 valid samples. CompositeJudge's all-NaN sentinel fires →
   100% MISS. This is by design for the cold-start window.

2. **Pure inference during cold start**: every MISS means the eval loop
   executes a fresh inference action, not a cached one. The robot's
   state evolves under the inference policy.

3. **State drift out of training distribution**: the cache library was
   built from ~50 ep of training trajectories where Pi0.5 was the
   teacher. Pure inference runs of the LIBERO eval loop diverge from
   those training trajectories within a few timesteps (Pi0.5 inference
   is non-deterministic; LIBERO's stochastic init states accentuate
   the drift).

4. **Retrieval winner-tail bias**: out-of-distribution query states have
   their nearest neighbor at the *edge* of the training distribution —
   typically the tail of a trajectory (where the trajectory was
   approaching task completion / state space boundary). We measured
   this in Phase 0 already: `f1b_*_p0_f5` (look-ahead-5 windows)
   showed 41-44% winner-conditional NaN against a global-entry NaN of
   24% — retrieval winners are systematically biased toward
   trajectory tails.

5. **Once we leave AlwaysHit territory, the bias intensifies**:
   - F1a-T's `view.walk_next(winner_id, k=3)` returns fewer than 3
     forward entries → `_build_state_splice` returns None → all 4
     F1a-T keys NaN.
   - F1b-A / F1b-T's pre-computed factors at boundary entries are
     NaN by definition (windowed smoothness uses out-of-trajectory
     points).
   - F2 fails via the second NaN path (`consensus.py:101-106`): if
     top-5 candidates are all from the same tail cluster, their
     action_chunk[0]'s are nearly identical, candidate-local
     variance is < eps on every dim, F2 returns NaN.
   - F1a-A is the only factor with a working signal (history.actions
     accumulates regardless of MISS, library_stats.action_sigma is
     loaded from artifact). But out-of-distribution actions have
     extreme jerk → high percentile after warm-up → contrib (risky
     orientation, `1 - percentile`) is *low* → composer score gets
     dragged toward 0.

6. **All-NaN sentinel fires every step OR composer score < 0.3**:
   either way, MISS. State drifts further. Loop closes.

The system gets pinned in an absorbing state where the cache produces
zero useful retrievals because the queries are themselves out of
distribution due to having gotten zero useful retrievals.

The replay in §5 cannot reproduce this because it uses Phase 0 raw
values — those values were collected under AlwaysHit conditions where
every step's cached action keeps state in-distribution. Phase 1's
runtime factor distribution is **categorically different** from Phase
0's, even though both were collected on the same artifact + cfg.

## 8. What's been considered

The plan we wrote
(`logs/verdict_factor_judge_experiment_plan.log.md` §3.4 / §4 Phase 1)
assumes "cold start costs 7-12 ep, then the system stabilizes". That
assumption is empirically wrong on this benchmark — Pi0.5 inference is
unstable enough that 7-12 ep of pure inference puts the trajectory
permanently outside the cache's covering distribution.

Possible mitigations, ranked by intrusiveness:

| Option | What it does | Effort | Why it might / might not help |
|---|---|---|---|
| **P0**: Change cold-start sentinel from MISS to WARM_START at conservative `start_t` (yaml-configurable, e.g. 0.7) | When all factor keys are NaN (cold start window or runtime collapse), still consult the cache and execute a partially-cached action. Keeps state in-distribution. | ~10 lines + yaml field + L2 plan/G1/G2 | Directly breaks the bootstrap death spiral. The same conservative `start_t` (0.7) achieves 0.98 success in Phase 0 baseline as `AlwaysWarmStart` — so falling back to WARM_START is safe. The risk is that "always WARM_START on NaN" may mask real factor failures and inflate success rates under a flawed CompositeJudge. |
| **P1**: N-PRELOAD normalizer buffer from Phase 0 calibration | Skip the 200-sample cold-start window entirely | ~200 lines + L2 | Doesn't address steps 3-6 of the dead loop — preloading percentile estimates doesn't change runtime NaN rates or the F1a-A high-jerk issue. Necessary but not sufficient. |
| **P2**: Pre-warming N episodes under AlwaysWarmStart before each Phase 1 yaml | Force in-distribution state for the first N eval episodes | runner change, awkward because LIBERO ep are independent | Doesn't generalize — what would deployment do? Each "fresh" deployment has cold-start. |
| **P3**: Plan-level redesign | Verdict factor framework needs to assume bootstrap as part of its operating model, not an edge case | large | Right answer if the dead loop is fundamental to retrieval-bound caches under high-variance inference. |

Our current lean: **P0 + P1 together**, then re-run Phase 0/1. P0 alone
should be sufficient to break the death spiral; P1 reduces the
warm-up cost of valid factors that DO work. P2/P3 only if P0 still
underperforms.

## 9. Specific asks

We'd appreciate any of the following:

1. **Prior art**: are there published cache verdict / retrieval gating
   designs that hit this same bootstrap problem? Whatever they did would
   be useful.

2. **Counter-argument to the dead loop hypothesis**: is there a more
   parsimonious explanation that fits both the live observation (100%
   MISS) and the offline replay (91% WARM_START predicted)?

3. **P0 critique**: is "all-NaN → WARM_START at conservative start_t"
   the right fallback semantics, or does it weaken the verdict signal
   in a way we'll regret? Would a *probabilistic* fallback (random WARM
   with prob ε) be better, to preserve some MISS evidence for
   diagnostics?

4. **N-PRELOAD critique** (P1): the calibration was collected under
   AlwaysHit; preloading those raw factor values into the normalizer
   means percentile ranks during Phase 1 reflect *AlwaysHit's
   distribution*, not Phase 1's. Does this mismatch cause its own
   problems?

5. **Pi0.5-specific**: is Pi0.5 inference unusually non-deterministic
   compared to other VLA models? If so, retrieval-bound caches built
   on its trajectories may fundamentally need stronger bootstrap than
   one built on, say, deterministic teacher policies.

## Appendix A: Code references

- Cold-start sentinel: `src/openpi/cache/components/judge.py:346`
- WeightedSum composer scoring: `src/openpi/cache/components/factors/composers/__init__.py:108-157`
- F2 NaN paths: `src/openpi/cache/components/factors/consensus.py:86-106`
- F1a-T splice failure: `src/openpi/cache/components/factors/runtime_continuity.py:230-268`
- F1a-A action history (recently fixed): `src/openpi/cache/orchestrator.py:355-368`
- Plan we're working from: `logs/verdict_factor_judge_experiment_plan.log.md`
- Phase 0/1 runbook: `logs/verdict_factor_judge_phase0_phase1_run_commands.log.md`

## Appendix B: How to reproduce the diagnosis

```bash
# 1. Inspect the offline replay (no model load, no env)
uv run python << 'EOF'
import json, math
from openpi.cache.config import load_cache_config, _build_composer
from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer
from openpi.cache.components.factors import registry

for stem in ['f_f1a_t_only_d_all_t_full', 'f_full_d_all_t_dual_07', 'f_min_cons_d_all_t_full']:
    cfg = load_cache_config(f'exp/verdict_factor_judge/config/clip/phase1/clip_w7_d4_phase1_{stem}.yaml')
    jc = cfg.checkpoints['cp1'].judge
    composer = _build_composer(jc.composer)
    norm = PercentileRollingNormalizer(window_size=jc.normalizer.window_size,
                                        cold_start_strategy=jc.normalizer.cold_start_strategy)
    KEYS, ORI = [], {}
    for fcfg in jc.factors:
        d = registry.get_class(fcfg.type).describe(fcfg.params)
        KEYS.extend(d.keys()); ORI.update(d)
    norm.bind_keys(KEYS); composer.bind_orientations(ORI)

    counts = {'FULL_HIT':0, 'WARM_START':0, 'MISS':0}
    for line in open('exp/verdict_factor_judge/data/calibration/clip_w7_d4_phase0_always_hit_dump.jsonl'):
        row = json.loads(line)
        raw = {k: float('nan') if row['factor_nan'].get(k, False) else float(row['factor_raw'].get(k, float('nan'))) for k in KEYS}
        n = norm(raw)
        if n and all(math.isnan(v) for v in n.values()):
            counts['MISS'] += 1; continue
        counts[composer.compose(n, winner_id='x').hit_type.name] += 1
    total = sum(counts.values())
    print(f"{stem}: FH={counts['FULL_HIT']/total*100:.1f}% WS={counts['WARM_START']/total*100:.1f}% MISS={counts['MISS']/total*100:.1f}%")
EOF

# 2. Inspect the wiring (verifies F2's min_top_k_hint propagates)
uv run python << 'EOF'
from openpi.cache.config import load_cache_config, build_shared_storage, build_per_connection_components

cfg = load_cache_config('exp/verdict_factor_judge/config/clip/phase1/clip_w7_d4_phase1_f_full_d_all_t_dual_07.yaml')
storage = build_shared_storage(cfg)
conn = build_per_connection_components(cfg, storage)
for cp_id, judge in conn['judges'].items():
    print(f"judge.{cp_id}: min_required_top_k = {getattr(judge, 'min_required_top_k', '?')}")
for cp_id, strat in conn['search_strategies'].items():
    print(f"search.{cp_id}: _top_k = {getattr(strat, '_top_k', '?')}")
EOF

# 3. Live observation
# Run any Phase 1 yaml with serve_policy.py stdout captured (|& tee) and
# grep `judge:` lines. The expected output is 100% MISS for every yaml,
# regardless of factor mix or tier configuration.
```
