# R7 C4 profile toolkit

CPU-only adapters. All writes stay in this directory or `/tmp/r7_C4`; library,
recordings, journals and earlier-round code are read-only. No CLI starts a
server, policy, simulator, worker or chain. Run from the repository root:

```bash
C4PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
C4MOD=exp.offline_search.rounds.r07.c4_profile
```

Every analyzer accepts `--out /tmp/r7_C4/<name>` and `--cells all` or a list
such as `pi05_l10_50 groot_spatial_500`. `spatial` is canonical. JSON has null
for missing measurements. CSVs include denominators and episode identities.
The C1 `StageTable` is the only segmenter: known labels include the unanimous
macro run, learned gripper mode, and event/interior membership; disagreement
of macro runs is `runmixed`, disagreement of modes is `mixed`, and failed
library members yield `unknown`. These are gripper-command stages, not grasp
success labels.

`profile.common` supplies read-only array access, JSON/CSV writers and episode
bootstrap; `profile.runprof.pct` supplies timing distributions. Camera contrasts
use the same paired-episode bootstrap principle as `profile.compare`. The P3
strict products supply decisions/anchors/controls; its `identical` comparator
and `control_join` are reused for ordinary logs and actual-control validation.
Existing `profile.breakdown` requires harness error products and `timeline`
requires harness result NPZs: their CLIs cannot consume P3 products directly.
C4 emits equivalent stage slices and timelines in the recording schema rather
than inventing a harness-format conversion. This is also why call uncertainty
uses an explicit fixed-task/init bootstrap rather than an unstratified row
bootstrap. No multiprocess defaults from `profile/` are invoked.

## follow_audit

```bash
"${C4PY[@]}" -m "$C4MOD.follow_audit" --cells all --campaigns bval p3 --caps 1 2 --out /tmp/r7_C4/follow_all
```

`follow_audit.json`, `follow_summary.csv` and `follow_timeline.csv` report stage
× horizon structural support, SF/UF admission, displacement/absolute residual
distributions, lost kernel mass, observation censoring, row/episode-radius
alerts and alert lead in controls to a library mode boundary. Zero-weight
members still participate in support and stage checks. The C1 `FollowExtension`
plan is used directly. A horizon's endpoint residual is descriptive; valve
checks stop before the cap's unconditional LOOK.

`SF_eligible` means the full requested extension survives all observed checks.
Stage admission and valve-observation counts are separate. Modeled SF cycle
cost includes early valve LOOKs, including a first-blind abort; missing checks
remain censored. Modeled UF/SF IR is a fixed-anchor cycle projection with zero
calls. It is neither a changed-rollout renewal schedule nor realized IR/SR.
The row radius is C1's p95 of LOEO commitment maxima; the episode sensitivity
is p95 of episode maxima, matching the shared API. It is not an inverse-length
weighted quantile of individual checks.

## camera_audit

```bash
"${C4PY[@]}" -m "$C4MOD.camera_audit" --cells all --folds 2 --reps 2000 --out /tmp/r7_C4/camera_all
```

`camera_audit.json` and `camera_rows.csv`: full/wrist/third-person action RMS,
gripper-mode disagreement, missing/extra event rates, and first-event timing
error by frozen stage, with paired episode uncertainty. All three supervised
metrics and action scales refit on disjoint successful source-episode folds;
the existing policy PCA stays frozen. Fold assignment is outcome-independent
episode order modulo `--folds`; aliases stay together. The main metric is
screened at all library rows; deployed early-step fitting is not replayed.

Candidate-episode deletion removes the highest-mass retrieved source episode
and recomputes the same kernel when sufficient candidates remain. B-val
anchors are also compared with same-observation shadow chunks using metrics
fit on successful library rows. `--skip-bval` omits that pass. Missing events
are not converted into arbitrary timing errors. Camera screening on GR00T
uses stored keys; it does not create a deployable GR00T one-camera path or
prove isolated-tower key/policy-input parity. That proof belongs to C2.

## stage_value

```bash
"${C4PY[@]}" -m "$C4MOD.stage_value" --cells all --reps 2000 --out /tmp/r7_C4/value_all
```

Defaults to R6 uniform U18/U30 and frontier risk-lottery arms. Override with
`--run-roots ROOT ... --arms ARM ...`. Accepted-attempt joins reject conflicting
duplicates and gaps. Legacy reused-attempt infrastructure prefixes are selected
by the latest step-zero incarnation before acceptance and counted in the audit.
Full kernels are required. Logged dose probabilities are checked against the
frozen fitted lottery, and CALL is checked against its conditional anchor coin.

Two levels: episode dose `d ~ pi(d|task)`, then `Z ~ Bernoulli(p_d)`. The score
is `g(d)/pi(d) * [Z/p_d - (1-Z)/(1-p_d)]`. The natural-dose target cancels the
first factor; equal-supported-dose analysis uses `g=1/K_task`. Never replace
the CACHE probability with `1-pi*p`. Endpoint doses have no local intervention
support; tasks with no supported dose are explicitly absent from the equal-dose
target. Outcomes are cross-centered on the other init parity within task.

First-entry estimates apply one supported excursion under its original
continuation, with zero contribution for unreached/end-point episodes. Occupancy
results sum per-episode anchor scores: a derivative for a common stage
probability perturbation, not a per-call ATE. Dividing by eventual stage duration
would condition on a post-treatment quantity and is deliberately avoided.
Results include positivity, dose support, CALL/CACHE ESS, task/init bootstrap,
Bonferroni stage intervals within arm/estimand/target, fold/held-task checks,
and downstream call/cost effects. They do not identify an entire CT controller,
an unexecuted camera/cadence package, or an equal-cost routing frontier. No
outcome-fitted serving table is emitted.

Intervals are withheld when a stage has no observed CALL or no observed CACHE;
the HT point score remains a diagnostic. Sparse ESS is reported, not repaired
with pseudo-observations. Cross-centering is held fixed during the bootstrap,
so uncertainty is conditional on that nuisance fit.

## failure_clock

```bash
"${C4PY[@]}" -m "$C4MOD.failure_clock" --cells all --campaigns bval p3 --worst 3 --out /tmp/r7_C4/clock_all
```

`failure_clock.json`, `failure_decisions.csv`, `failure_controls.csv`: actual
control clocks, wire-command transitions, measured aperture/EE motion/contacts/
object positions where available, first frozen-library absolute-residual p75
crossing, first displacement-valve alert, replayed confirmed stall, and terminal
outcome. `--stall-csv PATH ...` replaces the existing Q3 pilot-status files.
Missing stall replay says `not_replayed`. Worst episodes retain their full
decision/control timelines. Physical intent remains unannotated: contact and
object motion do not establish grasp/slip/release. Wire polarity is not named
close/release without the output adapter. Timeout is not failure onset.

## profile_report

```bash
"${C4PY[@]}" -m "$C4MOD.profile_report" --run-root '<RUN>' --references r7_pi05_l10_50_A_profile r7_pi05_l10_50_CU30_profile --legacy-full-camera --out /tmp/r7_C4/PROFILE_report
```

Reads ordinary plugin logs, accepted journals and summaries. Optional P3 client
controls are strictly joined to decision/wire chunks; control-normalized IR is
available only with complete accepted traces. Audited B-val manifests additionally
check actual reset hashes. Request cost is `.152/.148` per full look, `.848/.852`
per call, zero per blind decision, and π0.5 `.055198` per wrist look plus `.049890`
per measured completion. Wrist/completion pricing is the R4 proportional-latency
assumption. Actual `owner_cost` and summary `(N,V,M)` must reconcile.
The recorded completion count is used, including repeated tower forwards.

Outputs per cell/variant: components, descriptive SR, camera mix, grants, valve/
LOOK and wrist-selection reasons by stage, query-time distributions, telemetry
coverage, paired common-episode IR/component contrasts, and worst timelines.
LOOK reasons belong to the previous anchor's stage; fresh grants to the new
anchor. SR is descriptive only. Paired IR resamples task/init clusters and
averages per-episode ratios; run IR is the pooled actual-work ratio. Both are
labelled. `--references` can contain all eight A and four CU base-C arms; only
same-cell pairs are compared. `--arms` restricts variants.

`--keys JSON` overrides logical fields with a dotted path or ordered paths,
e.g. `{"granted":["extras.new_grant","blind_extras.new_grant"]}`. Current
defaults are in `telemetry_keys.json`. Actual camera dispatch is read from
`camera_mode`, not the forecast `os_sw_next_camera`. Wrist MISS without actual
completion telemetry fails. `--legacy-full-camera` explicitly attests that
missing camera fields on ordinary A/SF/UF/CU/CT logs mean full vision; SW's
actual fields still take precedence. No profiler-only forward is charged to
deployment IR.

## Preparation and verification

`prepare_profile`, `render_profile`, `prefit`, and `verify_replay` are CPU CLIs.
Exact coordinator setup/launch/analysis commands are in [PROFILE_RUN.md](PROFILE_RUN.md).
Prepared emit_arms rows cover 44 PROFILE arms (36 new variants plus eight paired
A references), and 32 implemented evaluation arms. The four SA arms are recorded
as pending: C1 explicitly ships a composition hook, not an SA class/budget fit;
SELECTION requires composing after profiling. This toolkit does not invent them.

```bash
"${C4PY[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r07/c4_profile/test_toolkit.py --basetemp /tmp/r7_C4/pytest_green
"${C4PY[@]}" -m "$C4MOD.verify_replay" --cells all --out /tmp/r7_C4/verify_all
```
