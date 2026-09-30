# C3 hand-back: CU / CT at ρ=.30

All requested C3 deliverables are implemented and verified. Code and documentation
are confined to `exp/offline_search/rounds/r07/c3_calls/`; calibration and scratch
are in `/tmp/r7_C3/`. No earlier-round or shared source was edited. No server,
worker, closed loop, chain, remote access, git, GPU, tmux or port operation ran.
Python used repository `.venv/bin/python`, CPUs `30-33,74-77`, one numerical
thread, hidden CUDA and no bytecode. At most five Python processes overlapped.

## Method identities and exact semantics

Both variants use method string
`exp.offline_search.rounds.r07.c3_calls.methods:CallController`.
Per-cell kwargs (complete literal rows are in `arms_profile.json` and
`arms_eval500.json`):

```json
{
  "rho": 0.3,
  "placement": "uniform",
  "tilt": false,
  "cooldown_scope": "stall",
  "calibration_path": "<RUN>/cal/<cell>/CU/calibration.json",
  "stall_model_path": "<RUN>/stall/<cell>",
  "random_seed": 26092903,
  "randomization_key": "R6-C-v2/<cell>"
}
```

CT changes `tilt` to true and the calibration subdirectory to `CT`. The same
keyed coin is used in CU/CT and R6. Both keep R6's calibrated mandatory stall
calls, one free-anchor cooldown after a **stall call only**, ambiguous
lottery-then-LOOK rule, LOOK cap, unchanged A retrieval, cache commitment,
force-MISS judge and unchanged C10 policy-tail lifecycle. No policy chunk is
extended. `placement='R'` remains usable only with tilt off, for R6 identity.

CT's event factor is the kernel mean of member weights:
`1 + event_mass*(1/h - 1)` when h>0, otherwise 1. Known event-near members have
weight 1/h; interiors and unknown members have weight 1. No event-mass cutoff
is fitted. The deviation factor is 1/h_dev on the first **fresh anchor** whose
current-state residual exceeds C1's per-task library-LOEO p75, otherwise 1.
These independent factors multiply at an overlapping event/deviation entry.
Repeated high anchors have no second entry factor; a value at/below p75 clears
the latch; invalid/missing evidence preserves the latch. Zero occupancy
disables that factor. There is no zero-call interior.

C1's StageTable supplies segmentation, state scales, valid dimensions, p75 and
occupancies. C3 fits no competing stage table or threshold. `StageTable.fit`
receives R6's frozen A through `manifest['_retrieval']`; its retrieval
fingerprint must match the bank. CT artifacts bind the table payload SHA256,
content fingerprint and retrieval fingerprint. The table's h is successful-row
occupancy and h_dev is the strictly-above-p75 LOEO occupancy, as defined in
`stages/STAGES_API.md`.

CT runs the original R6 query once; after unchanged retrieval and before the
original lottery it supplies `min(1, λ*weight)` as the inherited uniform
nominal probability. The original stall rules can still override that p.
The budget model uses R6's capped-score solver slot called `R`, with
`Ehat=call_weight`, **not** predicted cache-policy disagreement. Its public
method placement remains uniform. A CT calibration cannot be used with tilt
off: use CU's re-solved uniform artifact when disabling the tilt. On the same
calibration, placement and coins, tilt-off CU is bit-identical to R6.

## Calibration and feasibility

The original `fit_calibration.fit` was executed unchanged on the existing
non-test B-val recordings, with only its output-path guard scoped to C3's
scratch destination. It rechecks the frozen selection, catalog, exclusions,
table/library hashes and actual simulator-reset attestation. No outcomes are
loaded. Each cell contributes ten recordings, one per task; total **40 episodes
/ 996 recorded anchors**. No R6 artifact was overwritten.

CU's .30 solution dictionary is exactly equal to R6's retained uniform-stall
solution in all four cells. CT lifts the original cadence DAG with a
high-deviation-entry latch, preserving the original stall history, cooldown and
LOOK-cap states. It then re-solves λ on that product DAG. This handles a LOOK
changing which anchor first enters high deviation. It does not precompute
first entry along a single fixed cadence.

| cell | floor (CU=CT) | ceiling (CU=CT) | CU λ | CT λ | feasible CU / CT |
|---|---:|---:|---:|---:|---|
| pi05_l10_50 | 0.133324939828 | 0.461342451012 | 0.500026588464 | 0.360534667465 | yes / yes |
| pi05_spatial_50 | 0.125752969697 | 0.466940744449 | 0.494777615807 | 0.294508532222 | yes / yes |
| groot_l10_50 | 0.153875017851 | 0.440704194944 | 0.502167763104 | 0.325783539972 | yes / yes |
| groot_spatial_50 | 0.119061009895 | 0.475642604002 | 0.498874679981 | 0.357746992232 | yes / yes |

All eight modeled IRs are .30, maximum absolute error ≤5.56e-17. All solver
grids are monotone. Full precision, modeled calls/anchors/extra LOOKs and
StageTable calibration counts are in `calibration_feasibility.json` and
`/tmp/r7_C3/cal/<cell>/feasibility.json`. Per-cell original/lifted DAG sizes are
725/725, 255/255, 851/851, 208/208 in the table order above. The unit fixture
also exercises a real latch split after a missing observation.

The cost model remains R6's equal-episode mean of owner decision cost:
`[c_vision*anchors + (1-c_vision)*calls]/nominal_blocks`. Inputs are .152/.848
for π0.5 and .148/.852 for GR00T; blind requests cost zero. Mandatory stall
calls, cooldowns, ambiguous extra LOOKs, terminal odd blocks and coin clipping
are integrated exactly. Observations, proposals and episode lengths remain
fixed during calibration. Calls can change them in a real rollout; profiling
must validate realized cost. This is neither a hard spending cap nor SR/OPE.

Output layout (copy the **whole cell directory**, including relative links):

```text
/tmp/r7_C3/cal/<cell>/
  CU/calibration.json, cost_replay.json, r_bank.json, r_bank.npz,
     calibration_anchor_labels.csv
  CT/calibration.json, cost_replay.json
  stages.pkl, stage_features.json, feasibility.json, r6_refit.log
```

CT's bank path is `../CU/r_bank.json` and its table path is `../stages.pkl`.
The stall models were copied for packaging checks into
`/tmp/r7_C3/stall/<cell>/`; their original read-only source is
`r06_c_cal/stall/<cell with spatial replaced by sp>/`.

## Arm specs, prefit and coordinator smoke recipe

`arms_profile.json` contains eight `emit_arms` rows: CU/CT × four sparse cells,
20 non-test B-val episodes per arm. Names are
`r7_<cell>_CU30_profile` and `r7_<cell>_CT30_profile`. Manifests are C4's
`<RUN>/manifests/<model>_<suite>_bval20.json`; these select two unused B-val
states per task, excluding the R6 calibration state, without outcomes.
`arms_eval500.json` contains the eight full-eval rows named
`r7_<cell>_CU30` / `r7_<cell>_CT30`, with
`<RUN>/manifests/eval500.json`. Prune only after profiling, per SELECTION §4.

Every row carries `full_model=true`, `cost_ledger=true`, 5-control client
requests, normal model-specific image size, and these plugin flags:

```text
--os-root /home/weiland/trace_runs/offline_search_store
--os-no-shadow-native --os-blind --os-policy-tail --os-policy-tail-blocks 1
--os-judge guard_only --os-fit-artifact <RUN>/fits/<arm>.pkl
```

Profile rows additionally carry `--os-log-inputs --os-log-r4`. No native shadow
or new neural model is installed. The actual `emit_arms` implementation was
exercised in scratch for both eight-arm spec files.

Coordinator sequence (not executed by C3): copy complete calibration cell
directories to `<RUN>/cal/<cell>`, copy the four calibrated stall directories
to `<RUN>/stall/<cell>` with canonical full `spatial` names, and install C4's
manifests/pool provenance and the frozen R6 eval500 pairing. Render final paths
**before** prefit; loading scratch-path test prefits under different kwargs is
not valid. From the repository root, with `R7_C3_RUN` set to the final run path:

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package emit --run-root "$R7_C3_RUN" --out /tmp/r7_C3/final_specs
bash /tmp/r7_C3/final_specs/prefit_commands.sh
```

The emitted script contains **16 exact, individually named prefit commands**,
one for every planned variant × cell × phase. The checked template commands
are also in `prefit_commands.sh` here, but its placeholder specs deliberately
refuse prefit until rendered. Copy the resulting `/tmp/r7_C3/prefits/<arm>.pkl`
files into the final `<RUN>/fits/` directory.

For the coordinator's ordinary offline config emission:

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$R7_C3_RUN" --spec /tmp/r7_C3/final_specs/arms_profile.json
```

C4 `prepare_profile --call-specs .../c3_calls/arms_eval500.json` imports these
literal call rows and makes independently copied profile rows; the separate
C3 JSON files can also be combined directly with C4's arm lists. Preserve the
20-state non-test manifests and the C4 B-pool/reset fence.

Before full profiling, coordinator smoke should use the first two selected
non-test pairs for CU/CT on one cell of each model, then expand to all eight
20-episode profile arms. Confirm p/coin/force-MISS equality, cache/policy-tail
provenance, stall-call-only cooldown, ordinary LOOK→retrieval→judge handling,
CT first-entry latch and logged weight/p75, and actual stage invocation counts.
Price actual vision/MISS dispatches. Apply the SELECTION CT/CU ±.03 realized
IR gate on the complete profile, not on a two-episode smoke. Freeze surviving
full-run arms before any test episode. This hand-back launches no experiment.

All **16 current-name prefits** have been built and reloaded under checked
scratch kwargs; hashes/bytes are in `prefit_manifest.json`, files are in
`/tmp/r7_C3/prefits_checked/`. Historical preliminary profile-name prefits
remain in that scratch directory and are not included in the manifest.

## Tests and exact counts

Final aggregated verification is `verification.json`; the retained final unit
log is `/tmp/r7_C3/unit_final.log`. All checks pass:

- **13 unit tests**: independent branch enumeration over 21 λ values versus
  the lifted DAG (atol 2e-14), true latch splitting, unit/zero occupancy,
  overlap, high/re-entry/missing-state latching, unknown task, input gates,
  exact-target/infeasible solve, extension reset/MISS clearing and stall LOOK
  priority.
- **26,992 raw-input decision pairs / 14,013 fresh-anchor pairs / 560 episode
  passes**: four cells × uniform and R placement × all 10 B-val and 60 P3 A
  streams. Actions, scores, rows, confidence, all extras including key order,
  cache/policy tails, controller logs/status, verdicts and vision flags are
  bit-identical to the real current R6 controller. Differences: **zero**.
- **1,978 additional decision pairs / 40 B-val episodes**: the real C1
  FollowExtension composition with E=0 is identical to R6.
- **2,057 deployed-controller anchor agreements / 80 fixed-proposal paths**:
  CU and CT match the exact calibration DAG path, including call, stall state,
  cooldown, extra LOOK and CT weight/entry behavior.
- **80,000 Monte Carlo episode paths** (1,000 keyed seeds per arm, ten episodes
  per cell): expected IR, anchors, calls and LOOKs checked in **32 comparisons**;
  maximum deviation **2.206295367 standard errors**.
- **12 real plugin/orchestrator CPU replays / 5,934 decisions**. R6 and CU's
  served arrays, sources, verdicts and vision flags are bit-identical on all
  **1,978 B-val decisions**. All eleven CT telemetry keys survive the 40-scalar
  mixed-mode log cap on every request, including carried tail requests.
- **16 current-name prefits/reloads**, **16 actual emitted config rows**, and
  **16 loader rejections** (dry-run calibration, old ambiguous rule, altered
  stage payload hash, controller/calibration tilt mismatch across four cells).
- **C4 integration**: its imported call rows exactly equal C3's eight profile
  and eight eval500 rows; four NONTEST_BVAL manifests have 20 unique pairs each.
  This check captures C4's output in memory and writes no external files.

Plugin replay decision-basis IR (fake policy, recorded observations; these are
not new closed-loop profile results or SR estimates):

| cell | CU replay IR | CT replay IR |
|---|---:|---:|
| pi05_l10_50 | 0.307116279 | 0.303767442 |
| pi05_spatial_50 | 0.305674419 | 0.327707641 |
| groot_l10_50 | 0.281072902 | 0.308385144 |
| groot_spatial_50 | 0.288763359 | 0.292015267 |

Exact commands used, from the repository root:

```bash
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell "$cell"
done
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m unittest exp.offline_search.rounds.r07.c3_calls.test_calls -v
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell "$cell"
  taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell "$cell" --seeds 1000
  for variant in R6 CU CT; do
    taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.plugin_replay --cell "$cell" --variant "$variant"
  done
done
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package emit
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package emit --run-root /tmp/r7_C3 --out /tmp/r7_C3/specs_filled
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.integration
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.verify
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.build_handback
```

The cell jobs ran independently in parallel, each with the displayed prefix.
Packaging used `package.prefit` sequentially for every literal rendered row;
the equivalent per-row CLI commands are emitted by `package emit`.
Calibration refuses an existing calibration artifact; plugin replay refuses an
existing output directory; prefit refuses an existing pickle. For a rerun,
use `--out` with a fresh C3 scratch destination. Replay evidence `verify`
expects the retained standard locations.

One initial unit fixture asserted a latch split while retaining complete
history, which already encoded that latch; it was corrected to exercise the
bounded-history merge with missing state. The first plugin fixture lacked the
fake adapter's required `stage1` hook; that hook was supplied and all twelve
replays then passed. Its unsuccessful scratch output is retained separately
in `/tmp/r7_C3/plugin/`. No failed result is counted as passing evidence.

## Telemetry

CU retains R6's original log keys exactly (uniform weight is identically one;
no new weighting decision). CT adds these scalar keys:

```text
os_c3_weight os_c3_lambda os_c3_event_mass os_c3_h os_c3_h_dev
os_c3_dev_entry os_c3_dev_latched os_c3_deviation os_c3_p75
os_c3_unanimous os_c3_unknown
```

Deviation/p75 are omitted when invalid or unsupported. They are available on
every validated request in these four cells. Carried tail scores keep the
last anchor's stage values; `os_c_fresh=0` distinguishes them from new decisions.
New stage keys are prioritized before legacy diagnostics for the scalar cap.
Required force-MISS inputs also remain inside the cap.

Inherited decision/reason keys:

```text
os_force_miss os_reason os_c_version os_c_stall_call os_c_fresh
os_c_R os_c_Ehat os_c_p os_c_nominal_p os_c_coin os_c_stall_state
os_c_extra_look os_c_cooldown os_c_anchor os_c_rho os_c_commit_controls
os_c_control_index os_c_hold os_c_call
os_c_stall_delta_hat os_c_stall_e90 os_c_stall_a10
os_c_stall_window_span os_c_stall_W
os_c_carried_anchor_score os_c_anchor_p os_c_forced_look
```

`os_reason`: 63 stall call, 62 lottery call, 0 cache; pending ambiguous LOOK is
`LookReason(8,'c_slow_ambiguous')`. Stall states inactive/ok/confirmed/ambiguous
map to 0/1/2/3. The plugin's `vision`, `src`, `hit`, `look_reason` and actual
invocation counts remain the cost ledger's source of truth. R6 R/Ehat are
diagnostics in CT and do not influence its call weight.

## SA composition and limits

`COMPOSITION.md` documents the generic extension hook and the ready
`install_follow_extension(FollowExtension(...))` bridge. It preserves R6's
stall LOOK priority, separates cache from policy provenance, leaves policy
tails unchanged, and clears extension plans on MISS/reset/invalidation.
Disabled composed parity is proven on all four B-val streams. Install a
fitted component before reset. C1 can add SF here without replacing the call
controller. SA still needs its own changed-cadence/camera cost replay, λ solve
and combined scalar-cap audit after profiling, as SELECTION explicitly plans;
CU/CT calibration must not be reused as an SA .30 artifact. No SA rollout or
uncalibrated SA arm is claimed in this C3 deliverable.

The inherited R6 transport explicitly supports 5-control requests and
10-control commitments. This lever uses manifest geometry and C1's valid
dimensions; port the underlying R6 tail transport/control clock before using
another request geometry. No task/suite names, phase fractions, robot-unit
thresholds, camera count or state/action width are introduced in method logic.
The table's statistical rules and the owner cost shares are documented inputs.
P3 test streams were used only for identity verification, never calibration.
No profile/test closed loop or SR estimate was run: those are coordinator work.

## Owned files and SHA256

All files below are newly added in C3's owned directory. There are no shared
file install times. `delivery_manifest.json` also hashes this hand-back and
records read-only source dependencies; `artifact_manifest.json` hashes the
calibration/stall artifacts. Large replay/prefit arrays stay in scratch.

| file | SHA256 |
|---|---|
| `COMPOSITION.md` | `caf74811ff320763aa3ba9b0426103d1242b0c8d6052bc3c4fc70d9a04a5cf0c` |
| `__init__.py` | `7c6c4ee77eb8c20a15f17410fac56381b5fedfe304acd602a5aeaed9a85ce73e` |
| `arms_eval500.json` | `a1128b258c955c5419e6d444daa11f85b28425562431de2d452d675905739d43` |
| `arms_profile.json` | `cfc788d870645529ac14ddba30984e438a52328664b68061bacec082b1c6d74c` |
| `build_handback.py` | `0de137ba4dc523ba0f9370d2cfcf09f5b228c711688c8a0a43027a43b5f6368a` |
| `calibrate.py` | `7afb06f56890f99ea93e289d4604884c124e99377398eddb0bd352eabd9473a0` |
| `calibration_feasibility.json` | `9517dfb252566033cc7baac283af2bfdae93ea6c6534ae80a836c43d6e1342c3` |
| `common.py` | `c0d3f7e6e2e6177361af0d0b50cbbac5a97ea90c116d87407271177d26336942` |
| `composition.py` | `893081ae8c650dd8c020c3536ef166248ac839ea511ed3767b81e128a82ee563` |
| `integration.json` | `dc5814a3e83ed54f64ad53876bb1476e2ccbcc1f591f5b2344150bd082f23213` |
| `integration.py` | `8f948bbff8dee4069269c09370b606cf86a915d3bf4cf4abdd04dd25d301d3d0` |
| `methods.py` | `91d3b43b9a3083b83f0af44e3a2bb49bcf39c786db83f04ff4be65e88a7f3f70` |
| `package.py` | `7a881d37ad84eeb36b3d5ae1eed587d9f6a07da69084dadf5e6789ff5fcb7fda` |
| `package_manifest.json` | `5924fa74365e9e62fe32d2a9038531d8508e6a4ad69a56c0894ff45e29c1abe9` |
| `plugin_replay.py` | `e0b48bfe98110aafd3ad1d488c9763ebc23493cc4c7547b6fc06afb850ceb2c2` |
| `prefit_commands.sh` | `19bfd61ae5426436004aa3a6f2510d7dd33107b2be9e181a55465855bdd55ca0` |
| `prefit_manifest.json` | `6f5fb19e7860751e8ea9194c6c09ea0c95264833555cdbc80d1473e1aa2c5bb5` |
| `recordings.py` | `67414bbd1d4e278636bae69c2e6be33f8faf16183a19190126e6b7706c5e9b8c` |
| `replay.py` | `426c032fec3a392c3b20fcd1a23c6b4058e0ec7bd09c4baf0921ecdb4512683d` |
| `test_calls.py` | `bb553fce6a92fc4cc17b5f90be313812c11b8c9ba5718bbc03c40a5926c0c2f2` |
| `tilt.py` | `3913801f2920e6616a9ac5d1bcd556359955d0e2d4b08f0cb5fc1fce09d15d24` |
| `verification.json` | `bf02a823ace211993439625eb6d2357f5197906ef7af1366739a621c878046e9` |
| `verify.py` | `5c2f36e069430e217393d49ecbfa818f0af85a7f6b3b95d9b55a1ad57d7f5090` |

## Dense follow-up

Completed SELECTION §9.3 (preregistered 2026-09-30 06:2x CDT): CU18 and CT18
on the four dense libraries. The preceding sections record the original sparse
delivery. This section supersedes their source hashes for the five updated
files (`common.py`, `calibrate.py`, `package.py`, `replay.py`, `integration.py`).
Sparse arm JSON, package/prefit recipes, calibrations, stage tables, 16 checked
fits and serving/composition sources remain byte-identical: **75 of 75** baseline
SHA256 checks passed (`/tmp/r7_C3/dense_sparse_baseline.json`). `CELLS` still means
the four sparse cells; dense selection is explicit. All Python ran on the assigned
CPUs with the prefix below; at most eight Python processes overlapped. No shared
or earlier-round file was edited and no closed loop, server, worker, GPU, git,
remote host, chain, tmux or port was used.

### Dense feasibility at ρ=.18

| cell | CU floor | CT floor | CU ceiling | CT ceiling | CU λ | CT λ | CU modeled IR | CT modeled IR | feasible CU / CT |
|---|---|---|---|---|---|---|---|---|---|
| `pi05_l10_500` | 0.091724254623273493 | 0.091724254623273493 | 0.48918527656829541 | 0.48918527656829541 | 0.21851634866786762 | 0.11286299490007476 | 0.17999999999999999 | 0.17999999999999997 | yes / yes |
| `pi05_spatial_500` | 0.1099149212425002 | 0.1099149212425002 | 0.49520026834763681 | 0.49520026834763681 | 0.18159946138661776 | 0.093216977373562299 | 0.17999999999999997 | 0.17999999999999999 | yes / yes |
| `groot_l10_500` | 0.10648277226643935 | 0.10648277226643935 | 0.47672149384477297 | 0.47672149384477297 | 0.19416521254971775 | 0.10293144502009377 | 0.17999999999999997 | 0.17999999999999999 | yes / yes |
| `groot_spatial_500` | 0.088943942338990944 | 0.088943942338990944 | 0.49776963496861193 | 0.49776963496861193 | 0.22354798693257266 | 0.11622089590807172 | 0.18000000000000002 | 0.17999999999999994 | yes / yes |

All eight modeled IR errors are ≤5.56e-17. Floors and ceilings include mandatory
stall calls, the one-fresh-anchor stall-call cooldown, lottery-then-LOOK behavior,
and extra LOOKs. λ is re-solved over the inherited R6 cadence DAG, including the
high-deviation entry latch for CT; no target is clipped. The tilt and controller
serving code are unchanged from the sparse package. CT's internal budget solver
uses `placement='R'` with its score replaced by the stage weight, then exports the
solution under `solutions.stall.uniform`; both live arms use `placement='uniform'`.

There are **40 non-test calibration episodes / 821 recorded fresh anchors**
(274, 135, 290, 122 respectively). π0.5 loads its frozen deployed `bpool_cs`
fit; GR00T loads `bpool_all`; all use kref 8. The historical fit kwargs say
`lib='big'`; the loaded candidate names and bank paths are checked against the
actual deployed libraries. Original R6 coefficients, source fits, both tested
placement solutions, stall fingerprints, reset attestations, table/catalog
hashes, and non-test manifest provenance are verified unchanged. The original
fitter validates each acquisition/source-episode exclusions digest in the
accepted recording catalogs against its per-cell B-val manifest; exclusions
are preserved rather than reconstructing or enlarging the calibration set.

Each `/tmp/r7_C3/cal/<dense cell>/` contains `CU/`, `CT/`, `stages.pkl`,
`stage_features.json`, `feasibility.json` and `r6_refit.log`. CU owns the copied
R bank; CT references `../CU/r_bank.json` and `../stages.pkl`. The stage artifact
is a byte-identical copy of C1's `/tmp/r7_C1/stages/<cell>.pkl`, checked against
the deployed library arrays/geometry and frozen-A retrieval fingerprint. No dense
stage statistics were fitted by C3. `artifact_manifest_dense.json` records
**52** calibration/stall file hashes. `calibration_feasibility_dense.json`
contains full-precision solutions and provenance. Source R6 artifacts remain
read-only and their recorded checksums still match.

Exact calibration commands used (redirected to the per-cell
`/tmp/r7_C3/dense_fit_<cell>.log` during verification):

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell pi05_l10_500 --rho .18 --out /tmp/r7_C3/cal/pi05_l10_500
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell pi05_spatial_500 --rho .18 --out /tmp/r7_C3/cal/pi05_spatial_500
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell groot_l10_500 --rho .18 --out /tmp/r7_C3/cal/groot_l10_500
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell groot_spatial_500 --rho .18 --out /tmp/r7_C3/cal/groot_spatial_500
```

These commands refuse to overwrite completed calibration artifacts. To reproduce
without replacing them, supply a new owned `--out` directory. Omitting `--rho`
selects .18 for dense cells and .30 for sparse cells.

### Dense arms and exact prefits

`arms_dense_profile.json` and `arms_dense_eval500.json` contain eight rows each
in the verified `emit_arms` input format, with literal `<RUN>` placeholders.
Names are `r7_<cell>_{CU18,CT18}_profile` and `r7_<cell>_{CU18,CT18}`.
Both use `exp.offline_search.rounds.r07.c3_calls.methods:CallController`,
`rho=.18`, uniform placement, calibrated stall, `cooldown_scope='stall'`,
`random_seed=26092903`, `randomization_key='R6-C-v2/<cell>'`; CU has `tilt=false`
and `cal/<cell>/CU/calibration.json`, CT has `tilt=true` and `cal/<cell>/CT/calibration.json`.
Both have `stall_model_path='<RUN>/stall/<cell>'`, `full_model=true`, `cost_ledger=true`,
5-control client replanning and the same model-specific resize as the sparse arms.
Plugin flags remain `--os-root /home/weiland/trace_runs/offline_search_store`,
`--os-no-shadow-native --os-blind --os-policy-tail --os-policy-tail-blocks 1
--os-judge guard_only --os-fit-artifact <RUN>/fits/<arm>.pkl`.
Profile adds `--os-log-inputs --os-log-r4` and binds
`<RUN>/manifests/<model>_<suite>_bval20.json`; evaluation binds
`<RUN>/manifests/eval500.json`. Final manifests are coordinator inputs.

Copy the **complete** four dense calibration directories into `<RUN>/cal/`,
and copy the four R6 stall directories into `<RUN>/stall/<canonical cell>/`
(the source directory uses `sp` where the canonical cell says `spatial`).
Replace `<RUN>` below with the concrete run root before executing; render
final paths first so fit metadata matches the deployed arm rows exactly:

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package emit --dense --out /tmp/r7_C3/specs_dense_final --run-root '<RUN>'
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_pi05_l10_500_CU18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_pi05_l10_500_CT18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_pi05_spatial_500_CU18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_pi05_spatial_500_CT18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_groot_l10_500_CU18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_groot_l10_500_CT18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_groot_spatial_500_CU18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_profile.json --name r7_groot_spatial_500_CT18_profile --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_pi05_l10_500_CU18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_pi05_l10_500_CT18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_pi05_spatial_500_CU18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_pi05_spatial_500_CT18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_groot_l10_500_CU18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_groot_l10_500_CT18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_groot_spatial_500_CU18 --out /tmp/r7_C3/prefits_dense
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_final/arms_dense_eval500.json --name r7_groot_spatial_500_CT18 --out /tmp/r7_C3/prefits_dense
```

There is one command per planned variant × cell × phase, **16 prefits**.
`prefit_commands_dense.sh` is also emitted alongside the specs; regenerate it
with the final `--run-root` and `--out` above. Prefitting unresolved `<RUN>` rows
is refused. Copy the resulting `/tmp/r7_C3/prefits_dense/<arm>.pkl` into
`<RUN>/fits/` only after final-path prefitting. Checked scratch artifacts use
`/tmp/r7_C3` as their run root and are validation examples, not final-run fits.
The 16 checked artifacts and their hashes are listed in `prefit_manifest_dense.json`.

Exact scratch preparation and prefit commands actually used:

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.verify_dense prepare
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_pi05_l10_500_CU18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_pi05_l10_500_CT18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_pi05_spatial_500_CU18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_pi05_spatial_500_CT18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_groot_l10_500_CU18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_groot_l10_500_CT18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_groot_spatial_500_CU18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_profile.json --name r7_groot_spatial_500_CT18_profile --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_pi05_l10_500_CU18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_pi05_l10_500_CT18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_pi05_spatial_500_CU18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_pi05_spatial_500_CT18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_groot_l10_500_CU18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_groot_l10_500_CT18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_groot_spatial_500_CU18 --out /tmp/r7_C3/prefits_dense_checked
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.package prefit --spec /tmp/r7_C3/specs_dense_filled/arms_dense_eval500.json --name r7_groot_spatial_500_CT18 --out /tmp/r7_C3/prefits_dense_checked
```

### Verification and telemetry

All four dense cells passed raw archived-input R6-vs-C3 identity at both uniform
and R placement on 10 B-val + 60 P3 episodes per placement: **22,546 decision
pairs / 11,605 fresh-anchor pairs / 560 episode passes**, with zero differences
in actions, verdicts, vision flags, full result fields, log ordering and scalar
or array bytes. The CU calibration supplied to the identity harness is checked
equal to the retained R6 calibration's coefficients, frozen base and both
tested placement solutions, so the comparison establishes retained R6 behavior.
The unchanged disabled C1 composition also passed **1,628 B-val decision pairs**.

The real CU/CT controllers agreed with their fixed-proposal DAG paths at
**1,678 anchors / 80 episode paths**. Independent keyed-lottery replay used
1,000 seeds per arm: **80,000 episode paths / 32 statistics**, maximum deviation
**2.73416269039779 standard errors**, all below 5. The budget
expectation includes call count, fresh anchors and extra LOOKs, with owner prices
.152/.848 for π0.5 and .148/.852 for GR00T. **16** dense prefits/reloads,
**16** intentional loader mismatch rejections and **16** real `emit_arms` config
rows passed. Emitter verification writes configs only.

The complete existing sparse verifier passed again: **13 unit tests**, zero
failures; 16 sparse artifact-loader rejection checks; 16 original prefits;
16 emitted config rows; C4 exact import of 8 profile + 8 evaluation rows and
four 20-episode non-test manifests; retained raw-stream, DAG and 12 plugin replay
evidence checked. Serving source SHA256 and all sparse fit/calibration/arm hashes
are unchanged. Exact test/replay commands (logs under `/tmp/r7_C3/`):

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell pi05_l10_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell pi05_l10_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell pi05_spatial_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell pi05_spatial_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell groot_l10_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell groot_l10_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell groot_spatial_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell groot_spatial_500 --seeds 1000
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.verify
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.verify_dense check
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c3_calls.build_dense_handback
```

Two initial verification failures were resolved and rerun: the dense GR00T
spatial Monte Carlo anchor count is exactly constant (12.3), and naive float64
summation gave an artificial standard error and z≈31.6. Replay statistics now
accumulate in extended precision and check constant statistics against an
explicit 64-epsilon roundoff bound; no controller, calibration or acceptance
threshold changed. C4 had added a recipe-writing helper after the original
integration test; the read-only test now suppresses that helper along with its
writer. The original failed logs are retained as `dense_cost_groot_spatial_500.log`
and `dense_sparse_verification.log`; corrected runs passed, including replays
of all four dense cost cells and the complete sparse verifier.

Dense telemetry is identical to the sparse controller schema. CT adds:

```text
os_c3_weight os_c3_lambda os_c3_event_mass os_c3_h os_c3_h_dev
os_c3_dev_entry os_c3_dev_latched os_c3_deviation os_c3_p75
os_c3_unanimous os_c3_unknown
```

Inherited call/cadence keys include `os_c_p`, `os_c_nominal_p`, `os_c_coin`,
`os_c_call`, `os_c_stall_state`, `os_c_extra_look`, `os_c_cooldown`, `os_c_anchor`,
`os_c_rho`, `os_c_control_index` and `os_c_commit_controls`; keys and scalar-budget
placement are unchanged from the original delivery. CU's complete result extras
remain byte-identical to R6 when tilt is off and placement matches.

### Coordinator next steps and caveats

Bind verified, fresh-reset 20-episode non-test B-val manifests for each dense
model/suite and profile all eight arms with the supplied input/R4 logs and cost
ledger. SELECTION §9.3 gates each arm's realized owner IR at **.18 ± .03**.
Only then run the eight eval arms on 500 test pairs each. Report the dense
four-cell pooled CT−CU bootstrap separately from sparse results; acceptance
requires its lower bound >0 and every cell's **pooled-decision |ΔIR|≤.015**.
For cost matching, compute per-cell `.152 V/N + .848 M/N` (π0.5) or
`.148 V/N + .852 M/N` (GR00T) from summed decisions, and also report per-episode
IR; label cells that miss the matched-IR condition. Do not pool library sizes.
No profile/eval outcome or success-rate improvement is claimed by this package.

Calibration is the existing equal-episode fixed-recording owner-IR model;
hypothetical calls do not change its recorded observations or proposals. Acquisition
episodes were excluded while collecting B-val inputs; deployed fits retain the
full dense bank, as in R6. Raw identity replay uses archived same-observation
policy chunks without model inference. Live calibration feasibility and CT−CU
matched cost still require the coordinator's profiles and actual ledger.
`COMPOSITION.md` and `install_follow_extension()` remain the dense SA composition
hook too; any changed extension/camera cadence needs its own budget replay and λ
solve, and CU18/CT18 calibration must not be treated as an SA calibration.

### Dense files and current SHA256

No shared files were installed. The following hashes supersede earlier sparse
report hashes only where sources were updated; original sparse output files
and artifacts retain their prior hashes. `delivery_manifest_dense.json` also
hashes this appended hand-back, all listed files and read-only dense inputs.

| file | SHA256 |
|---|---|
| `common.py` | `d29e66201ab6f65ed37511f0490f0e0da58a7a442da6c40984fc1ca05840844a` |
| `calibrate.py` | `9a971b542b8ae2f5b4c137b655e407d6be3cd9749594d8005699a7cc16a90967` |
| `package.py` | `6b7a3fd9343dd2770ad3370a4e6306431f29c303426a7f08733e08b51163c062` |
| `replay.py` | `6bdc7df11456a7f6c7dbcf25b96884480fa67e8b0c72dd65cd0f0f9c241768e0` |
| `integration.py` | `002e3c97ece2f6c77f2f353631f28a59699fd5e950e413e1f0ef461870f6c11e` |
| `verify_dense.py` | `9992d3fbe17adbd38b070fbf26f8561ba9255574d8bf67bb102345e55ec421e1` |
| `build_dense_handback.py` | `528161029e2c5c71cfe287606d3e1e78e8055226fd226aba0eb8a6898f461060` |
| `arms_dense_profile.json` | `1959daef658bf3389065db47ab69a737f829022db9c3d125104d0b5f96a09205` |
| `arms_dense_eval500.json` | `9969adbc31f0bada643fafaee255e28d182537fb99c4e56e44556e24add8a8ea` |
| `prefit_commands_dense.sh` | `3972666997f6fc4ed444db285b18e7afae5c701ea07cad6536bbc83f5cc2cff3` |
| `package_manifest_dense.json` | `7c9e8613b2df7b2e9f69f3e5c88193fbaf4b9ce5b2378b7486c0fdaa403e11a0` |
| `calibration_feasibility_dense.json` | `7a92331efa2a1e2f844529068efdacedb8ad5c8fb0c8c23425b2ac07e5c75d31` |
| `prefit_manifest_dense.json` | `9dc7bf8b8229eef4f980bc3e6eb1bdc2314241e6efbdcc3b14534061d6cb1f83` |
| `verification_dense.json` | `beef9692475616afd188cc49057384b29a5fa5c0ae27e16b053be79c8197f27e` |
| `artifact_manifest_dense.json` | `3f941a814a5259bd78595e48ab1120ff424b6353c7a54d26247c62acf604f255` |
