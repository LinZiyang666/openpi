# R6 P3 superset profiler, schema v1

The only mode switch is `Profile(enabled=True, ...)` in `--os-kwargs`. One
experiment collects every field below. `p=0` still computes every shadow chunk;
`enabled=False` delegates to the original fitted A/B method without a profiler.
There are no new shared-plugin CLI flags. The campaign uses exact deployed A
fits, a separate shadow B guard, and randomized policy replacement.

`dec` rows remain the source of ledger accounting. Additional `p3_startup` and
`p3_anchor` rows use the same JSONL file and its existing atomic write lock.
The startup records p, seed, replicate, strata and extension schema versions.
An anchor joins by `(arm, uid, attempt, step)`; `step` counts five-control
decisions, not physical timesteps. Task/init identify the original A-pool state.

| Group | Contents / units |
|---|---|
| `cache_chunk`, `policy_chunk`, `executed_chunk` | H×32 float32 values serialized losslessly as JSON numbers; normalized **model** actions. H=10 π0.5, H=16 GR00T in these deployed cells. Only columns 0..6 are valid action dimensions. `cache_chunk` is the proposal even on a MISS. `executed_chunk` is the chunk selected for service, not a claim that every row reached the simulator. |
| `served_wire_chunk`, `executed_head` | Output-transformed wire chunk; normalized five-step head respectively. The next dec must serve rows 5..9, unless the episode terminates first. `commit_controls=10` is intended commitment, never actual terminal control count. |
| `distance` | Whole-horizon RMS over valid seven dimensions, commit10 RMS, per-step RMS and L2. No empirical action-sigma rescaling. These are model-normalized units and must not be numerically pooled across models without calibration. |
| `retrieval` | Actual 16 selected rows, raw metric distances, ranking distances (include continuity after MISS), 16 nearest raw distances in sorted order, normalized kernel weights, candidate episodes/steps/progress, phase rows/steps, metric query code, metric name (`early` or `main`), raw d1, empirical LOEO quantile and sample count, regime, median distance, continuity terms, weighted per-step and whole-chunk neighbour RMS dispersion. |
| `guards` | All original B result extras without the plugin's scalar cap; fitted thresholds; min-camera guard's individual endpoint cosines; dense state motion. `shadow_only=true` means B guard advice on A's actual history, not a served B verdict. Missing/nonfinite inputs at the first decision are null/absent, not zeros. Guard outputs can pick a different top-16 tied member from A, as documented by P1. |
| `state` | Complete normalized robot_state and raw wire state. |
| `assignment` | Private deterministic SHA256/random.Random assignment keyed by seed/task/init/replicate/decision-step; arm name, port, connection, scheduling and retry attempt do not enter. Policy noise has an independent hash domain. Includes p, uniform draw, stratum, eligible/baseline hit, assigned CALL, injected, actual policy execution, actual execution propensity. B's mandatory MISS has actual propensity 1 and is ineligible for the randomized contrast. |
| `timing` | Shared stage1, instrumented method query (`retrieval_ms`, includes profiling feature extraction), profiling features, separate stage2 and stage3, full shadow stage2/3 wall interval including noise/prologue, and total request time, all milliseconds. CPU fake replay has stage23_ms instead of two real stages. CUDA synchronization and coordinator queueing make these stage wall times, not isolated GPU kernel times. |
| `cost` | One physical full-policy result per anchor; zero additional forwards for injection. Ledger MISS is 0/1 for actual policy service. Profiling-only calls/time count only shadow chunks discarded at CACHE anchors. |
| `extensions` | Namespaced `{schema, data}` entries, empty by default. |

Stage 1 is reused from the exact observation that produced retrieval. Stage2/3
runs once with full K10/K8 and private seeded noise. On a selected MISS, the real
interceptor still takes its ordinary MISS branch, broadcasts once, and receives
the stored stage outputs through connection-local proxies. This retains plugin
history, wire transforms and policy-tail admission. It does not recursively call
infer, ask a second orchestrator for a decision, or execute the shadow on CACHE.
On B's mandatory MISS the already-computed private-noise result is also reused;
enabled B is consequently a seeded profiling version of B. Disabled B is exact.

LOEO calibration uses every candidate-library row against other episodes of its
own task. It uses float64 Euclidean distances between the fitted float32 metric
codes, separately for the early and main metrics; early full-rank codes are
reconstructed with the deployed affine map. The empirical midrank CDF is
`(searchsorted(left)+searchsorted(right))/(2*N)`. This measures deployment-library
coverage in a fixed fitted representation; the metric itself is not refitted
leaving each episode out. Existing A may have borrowed PCA/metric information;
P3 does not change or hide that provenance. Step-0's early CDF uses all library
rows in the early metric, not a step-0-only population. Float rounding differs
slightly from the serving squared-norm dot-product distance arithmetic.

Optional portable propensity stratification is
`strata={"edges":[0.5,0.9],"p":[0.1,0.2,0.4]}`. The signal is the pre-treatment
LOEO d1 quantile. Campaign defaults to uniform p=.2 to retain simple positivity
and avoid choosing bins from evaluation outcomes.

For ideation requests, add scalar/vector fields inside the appropriate versioned
group, or register a pure read-only extension before connections are created:
`hooks.register_extension("placement", "placement.v1", fn(session, record))`.
The function receives a detached record; it must not mutate the session, draw
from global RNG, or retain GPU tensors. New action/state conventions should
supply adapters instead of changing the existing valid-dimension definitions.
CSV flattening preserves unknown extension values, with lists encoded as JSON.

`read_logs.py` writes:

- `anchors.csv`: one accepted vision anchor, outcome Y, episode N/M, remaining
  decisions/MISSes, assignment, features, timing, cost and metric code columns.
- `neighbours.csv`: one row per anchor × neighbour, with episodes/progress/phase.
- `action_steps.csv`: one row per anchor × chunk step, seven scalar dimensions
  for cache/policy/selected chunk and per-step error/dispersion.
- `episodes.csv`, `accounting.json`: episode totals and separate physical vs
  profiling policy counts. Raw complete H×32 chunks remain in JSONL.

The reader replays assignments, verifies configured propensity, enforces accepted
attempt/outcome joins, contiguous decisions, coverage at exactly every vision
anchor, selected action identity, and the following cache/policy tail. It refuses
incomplete/conflicting evidence. Client journals from CPU tests are synthetic
plumbing fixtures and must never be used as simulator performance evidence.

`estimate.py` reuses K5's `(task, original init)` cluster bootstrap across all
replicates. Its centered HT score is `(A-p)/(p*(1-p))*(Y-.5)`. It estimates an
anchor-weighted CALL-minus-CACHE excursion under **future randomized p
continuation**, conditional on encountered histories. It does not identify
one MISS followed by pure A; that requires branch rollouts or a different trial.
Only pre-treatment subgroups should be selected. Never use error labels or
outcomes to choose subgroups and report unadjusted intervals on the same data.
