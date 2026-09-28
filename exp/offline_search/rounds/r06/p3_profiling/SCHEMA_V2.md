# R6 P3 Profile v2

Use `exp.offline_search.rounds.r06.p3_profiling.v2:Profile`, `enabled=true`.
V1 remains available at `method:Profile`; it is not silently upgraded. All v1
fields retain their meanings unless explicitly noted below. Units remain the
deployed library's normalized action coordinates, valid channels 0..6 for these
LIBERO models. The fitted interface manifest declares the mask, H, R and cameras.
No simulator-only field is sent to the policy, retriever or assignment function.

## Records and joins

| Record | Key | Contents |
|---|---|---|
| `p3_v2_startup` | tag, connection | Profile design, seed/replicate, nominal p, strata, sampling settings, complete immutable catalog and its SHA256 |
| `p3_anchor`, `r6p3.anchor.v2` | uid, attempt, decision step | Full cache/policy/executed chunks, both output-transformed alternative wire chunks, original v1 retrieval/guard/state/timing fields; design assignment, resamples, calibration, histories, provenance |
| `p3_decision`, `r6p3.decision.v2` | same, **every** decision | Source, parent anchor, shadow seed, timing and actual stage invocation counts; lossless input NPZ reference; blind diagnostic retrieval/guard/calibration from an isolated method clone |
| `p3_inputs/<uid-hash>_aN/step_N.npz` | record reference | Raw visual keys, normalized state, raw state, shadow policy, executed normalized chunk and wire chunk. Anchors also retain all 16 neighbour chunks; blind decisions retain the isolated cache proposal. No object pickle required. |
| client `controls.jsonl` | task_uid, attempt, control index | Attempt start, reset/init hash and RNG, each inference, each issued control, physical successor, every physics-substep actuator command where writable, rollout end/error |
| `step_N.npz`, `r6p3.snapshot.v2` | client identity + step | Pre-inference simulator state/time, exposed solver/actuator arrays, numeric controller attributes, explicit skipped attributes, Python/NumPy/environment RNG states, images, wire state, chosen wire chunk, remaining deadline and selection probability; **restore_certified=false** |

The server step is a five-control **decision**, not a simulator tick. A terminal
decision may execute fewer than five controls. The client logs the actual
number; `read_v2` rejects shifted/gapped control sequences and premature `done`.
Neither a full proposed chunk nor `commit_controls` proves it was fully applied.
The reader derives `actual_commit_controls`, partial-commit flags, after-head
and after-commit states, next-anchor links and remaining controls/MISSes.

`read_v2` writes anchors, decisions, controls, neighbours, action_steps,
episodes, all journal attempts, and available unaccepted server prefixes to
CSV. The default full-product path requires `--client-root`. Explicit
`--server-only` accepts standard-client runs with journal outcomes and server
input archives, labels them `collection_mode=server_only`, and leaves actual
control counts, completion masks, physical successors and control-normalized IR
null. Environment seed is unverified in that mode. `--require-stage-counts`
rejects absent/mismatching measured dispatch counters; `--require-snapshots`
checks sampling records and selected snapshot identities/payloads and requires
client telemetry. Neither flag certifies physical restoration. Infrastructure errors
stay in the attempt audit; they do not become ordinary task failures. It does
not silently include retries as independent episodes.

## Design dimensions and precedence

The standard A and B cohorts use a ten-control intended commitment. Policy
duration is independently randomized over `design.durations=[5,10]` when enabled.
Cache commitment remains A's ten controls. Ordinary B's existing no-progress
early-look behavior is preserved; a pre-guard experimental CACHE choice bypasses
that look veto to make the ten-control treatment well-defined.

`design` supports `pre_guard`, `durations`, `holds`, `delays`, `cap`, `cooldown`,
`start_anchor`, `episode_doses`, `cohort`, `cohort_probability`, `split_modulus`,
and a human-readable allocation description. Unknown keys and invalid ranges
are rejected. `episode_doses` and absolute state-stratum propensities are
different designs and cannot be combined. Fixed cohorts are a complete matched
block, with assignment probability **1**, not a fictional Bernoulli 1/9.
`episode_assignment` reports the actual randomized dose/probability in dose_mix.

At an anchor: prefix restriction → exhausted cap → pending delayed obligation →
held policy control → cooldown → mandatory baseline guard (unless pre_guard) →
random coin. A delayed design permits one selected CALL after 0/1/2 anchors
with cap=1 and A afterward. A hold means fresh policy at the next k−1 anchors,
not reusing a chunk beyond its supported horizon. The policy duration/hold/delay
choices have independent named SHA256 RNG domains and logged probabilities.
`p=0` with an A base and no episode-dose override remains A.

`assignment` contains nominal p/uniform/coin, pre-guard verdict, actual source,
effective actual propensity, availability, support, override reason, remaining
credits, previous spending, last CALL, pending delay and hold state, all choice
probabilities, selected commit duration and complete future-controller config.
The delay design's immediate propensity **marginalizes over its delay draw**;
its separate trigger/delay assignment identifies the placement package. Do not
condition on the realized delay and continue using the marginalized propensity.
Cap/hold/cooldown/due decisions have actual propensity 0 or 1 and no local
CACHE/CALL support. A baseline B forced MISS also has propensity 1 unless the
pre-guard design is selected. No infrastructure exception is reclassified as a
randomized CACHE control.

The estimand changes with the design: randomized holds/delays/durations are
intervention **packages** under their named continuation. V1's simple HT reader
must not be used on these logs. In particular, selecting on final realized M,
final episode progress, or eventual failure does not create a valid baseline
stratum. Cluster uncertainty on task/init across seed blocks. The scalar
episode-dose contrasts have whole-episode outcomes; local anchor contrasts do
not estimate A's entire success rate.

## Resampling, isolation and cost

The primary shadow is frozen before serving. With independent probability
`resample_p` (campaign 1/16), take `resample_draws-1` extra action-head draws
(campaign three), reusing stage 2. Restore the primary stage-3 object and noise
before the real MISS interceptor proceeds. Seeds, chunks, dispersion and cost
are logged. Blind shadows compute a new stage 1/2/3 from that actual observation;
they use a copied key builder and cloned method and never broadcast or push
history. The live decision remains blind. Re-observation proposals on blind
states are diagnostic, not executed alternatives.

V2 counts actual stage-function dispatches per request, independently of ledger
MISS. At a normal anchor: s1=1, s2=1, s3=1+extra draws. On an observed blind
decision: s1=s2=s3=1, all profiling-only. Injection adds **zero** forwards.
Coordinator dispatches may share a batched kernel; these are request counts,
not GPU-kernel utilization counts. `timing.key_build_ms`, `verdict_ms`,
`base_query_ms`, `shadow_guard_query_ms`, stage times and isolated blind-query
time have named scopes. Base query on B includes retrieval and guard logic;
feature recomputation is additional profiling work. Do not add overlapping
method/request wall timings as though they were exclusive GPU time.

Deployment cost is still c1×vision + (1−c1)×MISS, divided by decisions or by
actual active controls/5 as explicitly labeled. Physical collection additionally
includes all unused anchor shadows, blind full forwards, resampling and I/O.
An all-shadow trace cannot measure the latency of a profiler-disabled server.

## Calibration and extension contract

`calibrate` is CPU prefit preparation, not another rollout instrument. It uses
the exact deployed guard fit; excludes the source library episode from candidate
rows; replays recorded successful-library policy states/actions with vision
stride two; and emits per-task and pooled episode maxima for coverage/predicted
error, stuck, lag, overtime, progress and terminal guard, plus effective empirical
guard-alert shares and an episode-averaged LOEO chunk residual. PCA, metric and
V7 fits remain frozen. This is **not** fully refitted LOEO or a deployment-level
conformal guarantee. Online per-task empirical p-values use
`(1 + count(maximum >= observed_stat))/(1 + n)`; there is no invented pooled
fallback or claimed common alpha. The raw pooled tables are retained for later
method research. Q1's universal success probability and a new library-state
policy noise floor are not manufactured from these tables.

The immutable catalog references exact A/B fit bytes and the deployed manifest,
stores V7 calibration arrays and distance-LOEO tables, the supplied checkpoint
provenance, interface declarations and explicit unavailable metadata. A model
weight SHA must be supplied through `provenance.checkpoint_sha256` by the
coordinator; the profiler does not guess it from an arm name.

The v1 `hooks.register_extension(namespace, schema, fn)` remains available for
pure diagnostic feature functions; it runs on a detached pre-treatment record.
New fields belong under a versioned namespace, never in the serving result.
Additional client measurements belong in privileged `controls`/snapshot data.
New bank fits can be cells of this same Profile/campaign via base_fit,
guard_fit and `--os-root`; constructing those banks and obtaining their outcomes
is separate scientific scope, not an extra field on the current eight banks.
