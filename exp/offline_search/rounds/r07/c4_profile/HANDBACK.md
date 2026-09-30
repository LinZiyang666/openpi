# R7 C4 profile toolkit hand-back

The five offline analyzers, frozen C1 stage integration, audited non-test B-val
manifests, 44 PROFILE specs, 32 implemented evaluation specs, and exact prefit
recipes are delivered. **22 tests pass; disabled SF and UF match A on 10,569
decisions / 52,449 applied controls / 240 episodes, with zero parity failures.**
No R7 rollout, server, worker, chain, tmux, port, remote operation, GPU process,
or git command was run. Shared files and earlier rounds were not edited.

Four planned SA evaluation arms cannot yet be specified: SELECTION explicitly
assigns composition after PROFILE, and C1/C3 deliver hooks rather than a frozen
SA class plus cadence/camera budget recalibration. They are listed in
[pending_variants.json](prepared/pending_variants.json). SW additionally awaits
C2's real GPU parity and a mixed-judge wrist-origin MISS smoke. The exact guarded
GPU command and the missing smoke-spec dependency are stated in
[PROFILE_RUN.md](PROFILE_RUN.md). These are outstanding coordinator/owner work;
they are not claimed as passing C4 checks.

## Delivered interfaces

Each tool is a CPU CLI with usage, output schema, estimand and limitations in
[README.md](README.md). Output paths are restricted to C4's owned directory or
`/tmp/r7_C4`. All successful library metrics are outcome-independent fits;
test/P3 outcomes are evaluation or forensic evidence only.

| CLI | Delivered analysis | Existing-data output |
|---|---|---|
| `follow_audit` | Frozen stage × cap; exact SF/UF admission; all-member successor support; displacement/absolute residual distributions; row/episode-radius alerts; signed lead to library mode boundary; lost mass; censored modeled IR | `/tmp/r7_C4/follow_all/` |
| `camera_audit` | Full/wrist/third-person metrics, action/gripper-event errors, episode folds and paired uncertainty, candidate-episode deletion; same-observation B-val shadow chunks | `/tmp/r7_C4/camera_all/` |
| `stage_value` | Supported first-entry excursions and occupancy probability-shift derivatives under the recorded R6 lottery continuation; natural/equal-supported-dose targets; correct two-level propensities; downstream work, ESS and fold/task diagnostics | `/tmp/r7_C4/value_all/` |
| `failure_clock` | Strict decision/applied-control chronology, command transitions, measured aperture/contacts/object/EE motion, deviation/valve/stall crossings and terminal failure; intent remains unannotated | `/tmp/r7_C4/clock_all/` |
| `profile_report` | Accepted decisions/journals/summaries; realized owner IR components, actual cameras/completions, stage grants and LOOK reasons, descriptive SR, same-episode A/C contrasts, worst timelines, optional strictly joined control IR/reset attestation | `/tmp/r7_C4/report_existing/` (historical validation) |

`common.py` imports C1's `rounds.r07.stages.stages:StageTable`; no private
segmenter was substituted. It checks frozen A/library identity and stage-table
content/retrieval fingerprints. Existing `profile.common`, `profile.runprof.pct`
and `profile.timeline.moves/FLAG` are reused. The camera bootstrap follows
`profile.compare`'s paired episode convention. `breakdown` requires harness
error products and cannot directly read P3 products; these adapters retain the
strict recording schema. P3's `identical` and `control_join` are reused for
ordinary accepted streams and actual controls. P3 A_r0 is selected explicitly;
existing calibration exclusions are preserved.

## Headline outputs

`follow_audit` screened **5,341 anchors** (1,817 existing B-val, 3,524 P3),
producing **10,682 anchor × cap** records. All eight B-val cells clear the 5%
SF1 eligibility screen. SF eligibility requires the whole requested extension
and all observed pre-cap valve checks; no member is dropped, including a
zero-weight member. Missing future observations are censored separately.

| B-val cell | Anchors | SF1 eligible | UF1 eligible | SF2 eligible | Modeled SF1 IR | Modeled UF1 IR |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 L10 50 | 346 | 44.22% | 72.54% | 36.13% | .062354 | .055771 |
| π0.5 L10 500 | 274 | 27.37% | 94.89% | 23.36% | .066821 | .051545 |
| π0.5 Spatial 50 | 152 | 36.84% | 57.24% | 30.26% | .064178 | .059090 |
| π0.5 Spatial 500 | 135 | 48.15% | 77.04% | 37.78% | .061254 | .054866 |
| GR00T L10 50 | 365 | 30.41% | 85.48% | 25.75% | .064233 | .051843 |
| GR00T L10 500 | 290 | 21.72% | 94.48% | 18.97% | .066854 | .050258 |
| GR00T Spatial 50 | 133 | 44.36% | 65.41% | 39.85% | .060566 | .055762 |
| GR00T Spatial 500 | 122 | 45.08% | 83.61% | 40.16% | .060388 | .052185 |

Modeled baseline A IR is .076/.074. These are fixed-anchor cycle projections,
including an early first-blind valve abort, with zero policy calls; they are
not realized rollout savings. SF1 excludes one censored cycle each in π0.5
L10-50 and L10-500; other B-val cells have complete modeled cycles. Signed
lead time is to a library mode boundary, not to physical failure. The full
stage × horizon valve/lost-support distributions remain in the JSON/CSVs.

`camera_audit` evaluated **72,373 successful library rows** in two disjoint
source-episode metric folds and **1,817 B-val anchors**, each in three camera
modes: **222,570 camera records**. Paired wrist-minus-full head RMS changes:

| Cell | Library delta [95% episode CI] | B-val delta [95% episode CI] |
|---|---|---|
| π0.5 L10 50 | .00571 [−.00211, .01399] | .00791 [−.01055, .02031] |
| π0.5 L10 500 | .01467 [.01299, .01636] | .02377 [.00955, .04335] |
| π0.5 Spatial 50 | −.00171 [−.01234, .00866] | .01023 [−.01073, .03443] |
| π0.5 Spatial 500 | .01129 [.00976, .01291] | .00627 [−.00356, .01567] |
| GR00T L10 50 | .00419 [−.00321, .01109] | .00704 [−.02544, .03357] |
| GR00T L10 500 | .01394 [.01224, .01553] | .01180 [−.00045, .02488] |
| GR00T Spatial 50 | .02023 [.00518, .03350] | −.00517 [−.04247, .02197] |
| GR00T Spatial 500 | .01874 [.01712, .02039] | .02365 [.01742, .03167] |

Action errors are normalized by the fitted action scale. This measures retrieval
error, not policy success or encoder parity. Per-stage full/wrist/third-person
errors, gripper missing/extra events and conditional timing errors are included.
The frozen PCA was not refitted by fold, so this is a supervised-metric screening
experiment, not a fully inductive held-out representation experiment. GR00T
stored-camera screening does not supply a deployable one-camera serving path.

`stage_value` joined **23 R6 arms / 11,500 accepted episodes / 473,553 decisions**:
**187,221 supported anchors**, **52,158 endpoint/forced anchors excluded**, zero
missing full kernels, and **567 discarded earlier infrastructure-prefix rows**.
Of 244 natural-dose first-entry stage comparisons, eight lack an observed
CALL or CACHE branch and have their bootstrap intervals withheld. Of the 236
reported intervals, 235 include zero. The sole exclusion is an `unknown` stage
in π0.5 Spatial-500 risk rho=.105: 49 first entries, four CALLs, CALL ESS=4,
effect .013166, Bonferroni interval [.000183, .030977]. This sparse exploratory
finding does not establish useful stage heterogeneity or a routing rule.

The HT score uses `g(d)/pi(d) * [Z/p_d − (1−Z)/(1−p_d)]`; the natural target
cancels the dose factor. Forced doses cannot identify a local excursion.
Occupancy sums per-episode anchor scores rather than conditioning on eventual
stage duration. Outcome centering uses the other init parity within task and
is held fixed in the cluster bootstrap. No outcome-fitted serving table exists.

`failure_clock` covers **240 episodes / 10,569 decisions / 54,849 physical
controls**; 2,400 warmup controls are explicitly separate from **52,449 active
controls**. There are **43 failures** (17 B-val, 26 P3). All 43 have an observed
absolute-deviation crossing; 25 have a displacement-valve alert; 26 have a
replayed confirmed-stall crossing. Aperture/contact/object-position fields are
present for all controls in these retained recordings. These are measured
chronologies and frozen-library proxies, not labels for grasp/slip/release
intent or the onset of failure.

`profile_report` was checked on four historical R6 arms, 500 episodes each,
with two exact 500-pair contrasts. This is analyzer validation, not R7 PROFILE:

| Historical arm | Decisions | Looks | Calls | Owner IR | SR, descriptive |
|---|---:|---:|---:|---:|---:|
| π0.5 L10-50 U30 | 28,646 | 14,421 | 7,590 | .301205 | .878 |
| GR00T L10-50 U30 | 31,629 | 15,916 | 8,407 | .300937 | .770 |
| π0.5 L10-50 C30 | 29,111 | 15,093 | 7,613 | .300572 | .868 |
| GR00T L10-50 C30 | 31,069 | 16,112 | 8,505 | .309982 | .806 |

All summary N/V/M and outcomes reconcile. Historic logs lack client control
traces/camera counters, so the full-camera assumption was explicit and
control-normalized IR is unavailable. Paired mean episode IR differences U−C:
π0.5 .001890 [−.003185,.007322]; GR00T −.006749 [−.012437,−.001017]. Historical
C30 is not the new CU configuration; new paired A/CU references are rerun on
the same selected B states.

## Arm matrix, exact specs and prefit commands

The complete method strings, kwargs, plugin flags, final-path dependencies and
manifest per variant × cell are literal rows in
[arms_profile.json](prepared/arms_profile.json) and
[arms_eval500.json](prepared/arms_eval500.json). Profile-only SF2 is included
for cap selection; evaluation SF remains a provisional E=1 list.

| Family | Method | PROFILE rows | Implemented evaluation rows |
|---|---|---:|---:|
| SF1/SF2 | `exp.offline_search.rounds.r07.c1_follow.methods:StageFollow` | 16 | 8 SF1 |
| UF1 | `exp.offline_search.rounds.r07.c1_follow.methods:UniformFollow` | 8 | 8 |
| SW / SF+SW | `exp.offline_search.rounds.r07.c2_wrist.method:StageWrist` | 4 SW | 4 SW + 4 SF+SW |
| CU30 / CT30 | `exp.offline_search.rounds.r07.c3_calls.methods:CallController` | 8 | 8 |
| Paired A | `exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM` | 8 | reference only |
| SA30 | C1/C3 composition and budget pending | after PROFILE | 4 pending |

Follow kwargs extend frozen A by `extend_blocks=1|2`, `stage_gate=True`,
`state_valve=True`; UF sets both gates False. SW uses `enabled=True`, frozen
base fit, independently fitted wrist metric and stages. SF+SW's base is SF1.
CU/CT set `rho=.30`, `placement='uniform'`, `tilt=False|True`,
`cooldown_scope='stall'`, literal calibrated CU/CT paths, common randomization
key and seed 26092903. All other kwargs are preserved from the owners.

Common PROFILE flags include `--os-root`, `--os-no-shadow-native`, `--os-blind`,
`--os-log-r4`, `--os-log-inputs`, and exact `--os-fit-artifact`. SW adds
`--os-tokens off --os-request-cameras`; CU/CT add
`--os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`.
`full_model=True` is present for SW/CU/CT. Exact rows are authoritative.

[prefit_profile.sh](prepared/prefit_profile.sh) has 44 literal prefit commands;
[prefit_eval500.sh](prepared/prefit_eval500.sh) has 32. `render_profile` replaces
`<RUN>` in all specs and commands before fitting. `prefit` verifies exact
spec/kwargs on reload and refuses overwrite. Frozen A retrieval is reused,
including C1 fits; C2/C3 dependency paths bind the coordinator's final RUN.
Eight representative A/SF1/SF2/UF1/SW/CU/CT/SF+SW prefit/reload checks passed;
the other generated prefit commands were not executed by C4. Both spec sets
emit successfully: **44/32 arms and 88/64 YAML/matrix files**, respectively.

## Non-test manifest and coordinator work

The four model × suite manifests in `prepared/manifests/` use the same two
frozen B-val indices per task in each suite for both models. Selection uses
the lowest two unused indices, excludes R6's calibration init, checks frozen
split/pool hashes, and verifies zero identical B/test state bytes. Reset-state
hashes are stored and checked by `profile_report` when client traces exist.
There are 20 pairs per manifest, 40 distinct selected B states across suites.
The dense banks were acquired on these B states; PROFILE is allocation/plumbing
screening rather than independent bank generalization.

The exact coordinator-only commands are in [PROFILE_RUN.md](PROFILE_RUN.md):
render, copy dependency banks/calibrations, prefit, emit, install the protected
P3 v2 client bundle, stage audited B pools and the private ordinary-arm wrapper,
start a coordinator-admitted stream receiver, run both-model A/SF1/SF2/UF1
plumbing smokes, satisfy C2 camera gates, then the remaining 44-arm PROFILE and
paired report. No launch command was executed. The wrapper appends B
`--apool-record/--apool-dir` after stock defaults and refuses test-shadow files.
P3 deterministic seed, dispatch fence, reset attestation, accepted-attempt
collection and manifest-specific DONE checks remain intact.

Freeze/prune the evaluation list only after explorers review these PROFILE
outputs. Composition and a separate SA budget re-solve follow that decision.
C1 also reports a literal-spec conflict: an immediate first-blind valve abort
can change A before serving any extension. Off-lever parity and stage/structural
rejection parity are proven; identity on enabled early valve aborts is not.
The modeled costs include those aborts, and the coordinator must resolve the
interpretation rather than treating them as parity passes.

## Telemetry contract

[telemetry_keys.json](telemetry_keys.json) maps logical fields to configurable
dotted paths. Arbitrary added logical keys appear in coverage and timelines.
Known owner keys are:

- C1 anchor: `os_sf_mode0`, `os_sf_mode1`, `os_sf_unanimous`, `os_sf_event_mass`,
  `os_sf_rows_to_event`, `os_sf_unknown`, `os_sf_stage_gate`, `os_sf_state_valve`,
  `os_sf_cap`, `os_sf_structural`, `os_sf_stage_ok`, `os_sf_granted`.
- C1 blind: `os_sf_age`, `os_sf_extension`, `os_sf_valve_checked`,
  `os_sf_valve_fire`, `os_sf_delta`, `os_sf_radius`, `os_sf_look`, `os_sf_source`.
- C2: `os_sw_next_camera`, `os_sw_reason`, `os_sw_mode_mass`, `os_sw_unanimous`,
  `os_sw_event_mass`, `os_sw_unknown_mass`, `os_sw_rows_to_event`, `os_sw_valve`,
  `os_sw_radius`, `os_sw_target_blocks`, `os_sw_camera`, `os_sw_look_cost`,
  `os_sw_completion_cost`, `os_sw_call_cost`, `os_sw_decision_cost`.
- Actual C2 fields: `camera_mode`, `stage1_mode`, `camera_completion`,
  `camera_completion_calls`, `camera_stage1_calls`, `owner_cost`,
  `camera_cost_assumption`. Blind requests are reported as blind even when
  `stage1_mode` contains its full-mode placeholder. Forecasts do not price work.
- C3 CT: `os_c3_weight`, `os_c3_lambda`, `os_c3_event_mass`, `os_c3_h`,
  `os_c3_h_dev`, `os_c3_dev_entry`, `os_c3_dev_latched`, `os_c3_deviation`,
  `os_c3_p75`, `os_c3_unanimous`, `os_c3_unknown`.
- C3 inherited reasons/coins: `os_force_miss`, `os_reason`, `os_c_version`,
  `os_c_stall_call`, `os_c_fresh`, `os_c_R`, `os_c_Ehat`, `os_c_p`,
  `os_c_nominal_p`, `os_c_coin`, `os_c_stall_state`, `os_c_extra_look`,
  `os_c_cooldown`, `os_c_anchor`, `os_c_rho`, `os_c_commit_controls`,
  `os_c_control_index`, `os_c_hold`, `os_c_call`, `os_c_stall_delta_hat`,
  `os_c_stall_e90`, `os_c_stall_a10`, `os_c_stall_window_span`, `os_c_stall_W`,
  `os_c_carried_anchor_score`, `os_c_anchor_p`, `os_c_forced_look`.

Owner IR charges actual full looks .152/.148, calls .848/.852, blinds zero;
π0.5 wrist .055198 and each recorded completion forward .049890 are labelled
R4 proportional-latency assumptions. Missing wrist MISS completion telemetry,
owner-cost mismatch, summary mismatch, conflicting duplicates, malformed control traces
or wrong B reset hashes fail validation. Missing controls yield no nominal
control-IR approximation. C4 emits no new serving extras.

## Verification commands and artifacts

The reported verification commands use this prefix, from the repository root; maximum four
concurrent Python processes, no GPU:

```bash
C4PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
C4MOD=exp.offline_search.rounds.r07.c4_profile
"${C4PY[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r07/c4_profile/test_toolkit.py --basetemp /tmp/r7_C4/pytest_delivery
"${C4PY[@]}" -m "$C4MOD.follow_audit" --cells all --out /tmp/r7_C4/follow_all
"${C4PY[@]}" -m "$C4MOD.camera_audit" --cells all --out /tmp/r7_C4/camera_all
"${C4PY[@]}" -m "$C4MOD.stage_value" --cells all --out /tmp/r7_C4/value_all
"${C4PY[@]}" -m "$C4MOD.failure_clock" --cells all --out /tmp/r7_C4/clock_all
"${C4PY[@]}" -m "$C4MOD.verify_replay" --cells all --out /tmp/r7_C4/verify_all
"${C4PY[@]}" -m "$C4MOD.profile_report" --run-root /home/weiland/trace_runs/os_closed_loop/r06_c_validation --arms r6c_pi05_l10_50_U30 r6c_groot_l10_50_U30 --references r6c_pi05_l10_50_C30 r6c_groot_l10_50_C30 --legacy-full-camera --out /tmp/r7_C4/report_existing
"${C4PY[@]}" -m "$C4MOD.prepare_profile" --call-specs exp/offline_search/rounds/r07/c3_calls/arms_eval500.json
"${C4PY[@]}" -m "$C4MOD.render_profile" --run-root /tmp/r7_C4/validation_run --out /tmp/r7_C4/rendered_validation
"${C4PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root /tmp/r7_C4/validation_run --spec /tmp/r7_C4/rendered_validation/arms_profile.json
"${C4PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root /tmp/r7_C4/eval_validation_run --spec /tmp/r7_C4/rendered_validation/arms_eval500.json
for name in r7_pi05_l10_50_A_profile r7_pi05_l10_50_SF1_profile r7_groot_l10_50_SF2_profile r7_pi05_l10_50_UF1_profile r7_sw_pi05_l10_50_profile r7_pi05_l10_50_CU30_profile r7_groot_l10_50_CT30_profile; do
  "${C4PY[@]}" -m "$C4MOD.prefit" --spec /tmp/r7_C4/validation_run/arms_profile.json --name "$name" --out /tmp/r7_C4/prefit_checks
done
"${C4PY[@]}" -m "$C4MOD.prefit" --spec /tmp/r7_C4/validation_run/arms_eval500.json --name r7_sf_sw_pi05_l10_50 --out /tmp/r7_C4/prefit_checks
```

The prefit checks above have already run and intentionally refuse overwrite.
Final unit log: `/tmp/r7_C4/pytest_delivery.log`, **22 passed in 0.13s**. Tests
cover exact unequal two-level enumeration, endpoint rejection, missing observed
branches, censoring/zero-weight support, displacement, gripper event errors,
actual camera/call/completion costs, configurable telemetry, accepted-incarnation
joins/duplicate-gap rejection, same-episode pairing, cluster repeats, output
ownership, manifest disjointness, complete spec counts, and an end-to-end
summary reconciliation. Failures found during development were fixed and the
affected checks rerun.

`verify_replay` independently checked both disabled C1 classes on all 16
cell × campaign groups, including archived action/vision/verdict parity and
6,612 enabled structural/stage-rejection blind/LOOK comparisons. Action and
manifest source SHA256s were unchanged. Camera/C3 serving parity proofs belong
to their owners' hand-backs; C4 introduces no serving lever and does not claim
to have rerun their GPU or full-policy proofs.

Machine-readable headline results, verification counts and source/output hashes
are retained in `results/` and `delivery_manifest.json`. Large per-anchor/control
products remain at the scratch paths above. No shared-file install times apply.
