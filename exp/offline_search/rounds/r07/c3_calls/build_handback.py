"""Build C3's hand-back and content manifests from the verified artifacts."""
import json
from pathlib import Path

from .common import CELLS, HERE, SCRATCH, RECORDINGS, sha, write_json


def main():
    report = json.loads((HERE / 'verification.json').read_text())
    assert report['status'] == 'PASS'
    bounds = '\n'.join(
        f'| {f["cell"]} | {f["CU"]["floor"]:.12f} | {f["CU"]["ceiling"]:.12f} | '
        f'{f["CU"]["parameter"]:.12f} | {f["CT"]["parameter"]:.12f} | yes / yes |'
        for f in report['feasibility'])
    plugin_table = '\n'.join(
        f'| {cell} | ' + ' | '.join(f'{next(r for r in report["plugin_reports"] if r["cell"] == cell and r["variant"] == v)["IR"]:.9f}'
                                   for v in ('CU', 'CT')) + ' |'
        for cell in CELLS)
    source_files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in
                          ('HANDBACK.md', 'delivery_manifest.json', 'artifact_manifest.json'))
    files = '\n'.join(f'| `{p.name}` | `{sha(p)}` |' for p in source_files)
    prefix = ("taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
              "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python")
    body = f'''# C3 hand-back: CU / CT at ρ=.30

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
{{
  "rho": 0.3,
  "placement": "uniform",
  "tilt": false,
  "cooldown_scope": "stall",
  "calibration_path": "<RUN>/cal/<cell>/CU/calibration.json",
  "stall_model_path": "<RUN>/stall/<cell>",
  "random_seed": 26092903,
  "randomization_key": "R6-C-v2/<cell>"
}}
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
{bounds}

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
{prefix} -m exp.offline_search.rounds.r07.c3_calls.package emit --run-root "$R7_C3_RUN" --out /tmp/r7_C3/final_specs
bash /tmp/r7_C3/final_specs/prefit_commands.sh
```

The emitted script contains **16 exact, individually named prefit commands**,
one for every planned variant × cell × phase. The checked template commands
are also in `prefit_commands.sh` here, but its placeholder specs deliberately
refuse prefit until rendered. Copy the resulting `/tmp/r7_C3/prefits/<arm>.pkl`
files into the final `<RUN>/fits/` directory.

For the coordinator's ordinary offline config emission:

```bash
{prefix} -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$R7_C3_RUN" --spec /tmp/r7_C3/final_specs/arms_profile.json
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
  maximum deviation **{report['max_standard_errors']:.9f} standard errors**.
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
{plugin_table}

Exact commands used, from the repository root:

```bash
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  {prefix} -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell "$cell"
done
{prefix} -m unittest exp.offline_search.rounds.r07.c3_calls.test_calls -v
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  {prefix} -m exp.offline_search.rounds.r07.c3_calls.replay identity --cell "$cell"
  {prefix} -m exp.offline_search.rounds.r07.c3_calls.replay cost --cell "$cell" --seeds 1000
  for variant in R6 CU CT; do
    {prefix} -m exp.offline_search.rounds.r07.c3_calls.plugin_replay --cell "$cell" --variant "$variant"
  done
done
{prefix} -m exp.offline_search.rounds.r07.c3_calls.package emit
{prefix} -m exp.offline_search.rounds.r07.c3_calls.package emit --run-root /tmp/r7_C3 --out /tmp/r7_C3/specs_filled
{prefix} -m exp.offline_search.rounds.r07.c3_calls.integration
{prefix} -m exp.offline_search.rounds.r07.c3_calls.verify
{prefix} -m exp.offline_search.rounds.r07.c3_calls.build_handback
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
{files}
'''
    (HERE / 'HANDBACK.md').write_text(body)
    dependencies = [HERE.parent / 'stages' / 'stages.py', HERE.parent / 'stages' / 'STAGES_API.md',
                    HERE.parent / 'c1_follow' / 'methods.py']
    r6 = HERE.parent.parent / 'r06' / 'ideation_Q1' / 'method_c'
    dependencies += [r6 / name for name in ['methods.py', 'budget.py', 'fit_calibration.py', 'stall_bridge.py']]
    artifacts = [p for root in (SCRATCH / 'cal', SCRATCH / 'stall') for p in sorted(root.rglob('*')) if p.is_file()]
    write_json(HERE / 'artifact_manifest.json', dict(root=str(SCRATCH),
        artifacts=[dict(path=str(p.relative_to(SCRATCH)), sha256=sha(p), bytes=p.stat().st_size) for p in artifacts]))
    delivered = source_files + [HERE / 'HANDBACK.md', HERE / 'artifact_manifest.json']
    write_json(HERE / 'delivery_manifest.json', dict(
        status='PASS', owned_root=str(HERE), scratch=str(SCRATCH), shared_file_installs=[],
        files=[dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in delivered],
        read_only_dependencies=[dict(path=str(p), sha256=sha(p)) for p in dependencies],
        original_calibrations=[dict(cell=c, path=str(RECORDINGS / 'cal' / c / 'calibrated/calibration.json'),
            sha256=sha(RECORDINGS / 'cal' / c / 'calibrated/calibration.json')) for c in CELLS]))
    print(json.dumps(dict(handback=str(HERE / 'HANDBACK.md'), status='PASS', files=len(delivered))), flush=True)


if __name__ == '__main__':
    main()
