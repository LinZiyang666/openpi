"""Append the dense completion evidence, preserving the original sparse report."""
import hashlib
import json

from .common import DENSE_CELLS, HERE, SCRATCH, sha, write_json
from .package import specs


PREFIX = ("taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
          "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
          "PYTHONPATH=.:src .venv/bin/python")
MODULE = 'exp.offline_search.rounds.r07.c3_calls'


def main():
    result = json.loads((HERE / 'verification_dense.json').read_text())
    assert result['status'] == 'PASS'
    baseline = json.loads((SCRATCH / 'dense_sparse_baseline.json').read_text())
    for path, digest in baseline.items():
        assert sha(path) == digest
    paths = []
    for cell in DENSE_CELLS:
        for folder in (SCRATCH / 'cal' / cell, SCRATCH / 'stall' / cell):
            paths.extend(p for p in folder.rglob('*') if p.is_file())
    artifacts = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(paths)}
    write_json(HERE / 'artifact_manifest_dense.json', artifacts)
    commands = []
    for phase in ('profile', 'eval500'):
        for row in specs(phase, dense=True):
            commands.append(f'{PREFIX} -m {MODULE}.package prefit '
                            f'--spec /tmp/r7_C3/specs_dense_final/arms_dense_{phase}.json '
                            f'--name {row["name"]} --out /tmp/r7_C3/prefits_dense')
    rows = []
    for report in result['feasibility']:
        cu, ct = report['CU'], report['CT']
        rows.append(f'| `{report["cell"]}` | {cu["floor"]:.17g} | {ct["floor"]:.17g} '
                    f'| {cu["ceiling"]:.17g} | {ct["ceiling"]:.17g} '
                    f'| {cu["parameter"]:.17g} | {ct["parameter"]:.17g} '
                    f'| {cu["predicted_IR"]:.17g} | {ct["predicted_IR"]:.17g} | yes / yes |')
    calibration_commands = '\n'.join(
        f'{PREFIX} -m {MODULE}.calibrate --cell {cell} --rho .18 --out /tmp/r7_C3/cal/{cell}'
        for cell in DENSE_CELLS)
    replay_commands = '\n'.join(
        f'{PREFIX} -m {MODULE}.replay {case} --cell {cell} --seeds 1000'
        for cell in DENSE_CELLS for case in ('identity', 'cost'))
    files = ['common.py', 'calibrate.py', 'package.py', 'replay.py', 'integration.py',
             'verify_dense.py', 'build_dense_handback.py', 'arms_dense_profile.json',
             'arms_dense_eval500.json', 'prefit_commands_dense.sh', 'package_manifest_dense.json',
             'calibration_feasibility_dense.json', 'prefit_manifest_dense.json',
             'verification_dense.json', 'artifact_manifest_dense.json']
    hashes = '\n'.join(f'| `{name}` | `{sha(HERE / name)}` |' for name in files)
    text = f'''
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
{chr(10).join(rows)}

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
**{len(artifacts)}** calibration/stall file hashes. `calibration_feasibility_dense.json`
contains full-precision solutions and provenance. Source R6 artifacts remain
read-only and their recorded checksums still match.

Exact calibration commands used (redirected to the per-cell
`/tmp/r7_C3/dense_fit_<cell>.log` during verification):

```bash
{calibration_commands}
```

These commands refuse to overwrite completed calibration artifacts. To reproduce
without replacing them, supply a new owned `--out` directory. Omitting `--rho`
selects .18 for dense cells and .30 for sparse cells.

### Dense arms and exact prefits

`arms_dense_profile.json` and `arms_dense_eval500.json` contain eight rows each
in the verified `emit_arms` input format, with literal `<RUN>` placeholders.
Names are `r7_<cell>_{{CU18,CT18}}_profile` and `r7_<cell>_{{CU18,CT18}}`.
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
{PREFIX} -m {MODULE}.package emit --dense --out /tmp/r7_C3/specs_dense_final --run-root '<RUN>'
{chr(10).join(commands)}
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
{PREFIX} -m {MODULE}.verify_dense prepare
'''
    checked_commands = []
    for phase in ('profile', 'eval500'):
        for row in specs(phase, dense=True):
            checked_commands.append(f'{PREFIX} -m {MODULE}.package prefit '
                f'--spec /tmp/r7_C3/specs_dense_filled/arms_dense_{phase}.json '
                f'--name {row["name"]} --out /tmp/r7_C3/prefits_dense_checked')
    text += '\n'.join(checked_commands) + f'''
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
**{result['max_standard_errors']:.15g} standard errors**, all below 5. The budget
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
{replay_commands}
{PREFIX} -m {MODULE}.verify
{PREFIX} -m {MODULE}.verify_dense check
{PREFIX} -m {MODULE}.build_dense_handback
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
{hashes}
'''
    handback = HERE / 'HANDBACK.md'
    prior = handback.read_text().split('\n## Dense follow-up\n', 1)[0]
    handback.write_text(prior + text)
    dependencies = {}
    for report in result['feasibility']:
        cu = json.loads((SCRATCH / 'cal' / report['cell'] / 'CU' / 'calibration.json').read_text())
        source_paths = [report['source_calibration'], report['stage_source'], cu['base_source']['source_artifact']]
        source_paths.extend(cu['tables'])
        source_paths.extend(cu['catalog_paths'])
        for path in source_paths:
            dependencies[str(path)] = sha(path)
    write_json(HERE / 'delivery_manifest_dense.json', dict(status='PASS',
        owned_files={name: sha(HERE / name) for name in files + ['HANDBACK.md']},
        dense_artifacts=len(artifacts), read_only_inputs=dependencies,
        sparse_baseline=baseline, sparse_report_prefix_sha256=hashlib.sha256(prior.encode()).hexdigest()))
    print(json.dumps(dict(status='PASS', handback=str(handback), dense_artifacts=len(artifacts),
                          delivery_files=len(files) + 1)), flush=True)


if __name__ == '__main__':
    main()
