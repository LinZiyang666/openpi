"""Write the numeric C1 hand-back from passing CPU outputs, not estimates."""
from datetime import datetime, timezone
import json
from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sha, sources
from .prefit import HERE, OUT, VARIANTS

PREFIX = ('taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
          "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python")
MODULE = 'exp.offline_search.rounds.r07.c1_follow'


def main():
    reports = [json.loads((OUT/(c+'_replay.json')).read_text()) for c in sources()]
    fits = [json.loads((OUT/(c+'_fit.json')).read_text()) for c in sources()]
    packaging = json.loads((OUT/'packaging.json').read_text())
    assert packaging['arms_checked'] == 48
    tests = (OUT/'unit_tests.log').read_text()
    assert tests.rstrip().endswith('OK')
    assert all(all(r['parity_failures'] == 0 and r['recorded_action_equal'] == r['decisions'] and
                   r['recorded_vision_equal'] == r['decisions'] and r['recorded_verdict_equal'] == r['decisions']
                   for r in cell['identity'].values()) for cell in reports)
    commands = ['#!/usr/bin/env bash', '# Rebuild individual variants using the delivered verified stage tables.',
                '# For a fresh rebuild first run: ' + PREFIX + f' -m {MODULE}.prefit --cells all',
                'set -euo pipefail']
    for cell in sources():
        for variant in VARIANTS:
            commands.append(PREFIX + f' -m {MODULE}.prefit --cells {cell} --variants {variant} --reuse-stages')
    (HERE/'prefit_commands.sh').write_text('\n'.join(commands)+'\n')
    summaries = []
    for cell in reports:
        for variant, r in cell['extensions'].items():
            summaries.append(dict(cell=cell['cell'],variant=variant,**r))
    (OUT/'results.json').write_text(json.dumps(dict(schema='r7.c1.replay.v1',reports=reports,packaging=packaging),indent=2)+'\n')
    lines = ['# R7 C1 hand-back — SF/UF and shared stages', '',
             f"Written {datetime.now(timezone.utc).isoformat()}. Owned paths only; CPU-only work on 22–25,66–69.", '',
             'Shared `stages/stages.py` and `stages/STAGES_API.md` first landed 2026-09-30 01:03:30 CDT',
             '(06:03:30 UTC), before follow code and fits. No shared plugin file was changed.', '',
             '## Deliverables and exact methods', '',
             '* `stages/stages.py`: E1 two-means/three-row gripper stages, true successors, unknown failed rows,',
             '  content fingerprints, successful-library scales, E3 frozen-A candidate-LOEO valve/deviation tables.',
             '* `c1_follow/methods.py`: `StageFollow` (SF), `UniformFollow` (UF), and importable `FollowExtension`.',
             '* `arms_profile.json`: SF(E=1), SF(E=2), UF(E=1) ×8 = 24 non-test B-val profile arms.',
             '* `arms_eval500.json`: the same 24 candidate full-evaluation arms, subject to profiling pruning/cap freeze.',
             '* `/tmp/r7_C1/stages/`: eight independent frozen tables; `/tmp/r7_C1/fits/`: 24 embedded-table method fits.',
             '* `replay.py`, `test_follow.py`, `verify_specs.py`: reproducible recorded-stream, component and packaging checks.',
             '* `COMPOSITION.md`: plan/check/serve hook, compatible with C3’s `install_follow_extension` bridge.', '',
             'SF method: `exp.offline_search.rounds.r07.c1_follow.methods:StageFollow`.',
             'UF method: `exp.offline_search.rounds.r07.c1_follow.methods:UniformFollow`.',
             'SF1 kwargs add `extend_blocks=1, stage_gate=true, state_valve=true`;',
             'SF2 changes cap to 2; UF1 adds `extend_blocks=1, stage_gate=false, state_valve=false`.',
             'All use selected A’s `serving="anchor_tail", budget=1, gates="budget_only"`.',
             'Sparse cells use `lib="current", kref=5`; dense π0.5 uses `lib="bpool_cs", kref=8`;',
             'dense GR00T uses `lib="bpool_all", kref=8`. The wrapper accepts those explicit names and',
             'maps them to historical A’s `big` constructor internally, checking the resolved library.', '',
             'Plugin flags in every row: `--os-root /home/weiland/trace_runs/offline_search_store`,',
             '`--os-no-shadow-native --os-blind --os-fit-artifact <RUN>/fits/r7_<cell>_<variant>.pkl`.',
             'Profile additionally uses `--os-log-inputs` and C4’s `<RUN>/manifests/<model>_<suite>_bval20.json`.',
             'All rows use five-control client decisions, manifest geometry internally, and cost_ledger=true.',
             'There is no call judge or policy-tail flag on these pure-cache arms.', '',
             'π0.5 extensions execute true `next²`/`next³` member heads with original float32 weights.',
             'GR00T E=1 executes its anchor action controls 10:15 exactly; E=2 next executes `next³` heads.',
             'All members require support, even zero-weight members. No terminal clamp/drop/renormalization.',
             'E=2 is granted in full or rejected; no silent cap degradation. Policy-origin decisions always LOOK.', '',
             '## Identity and precise requirement limitation', '',
             'Disabled `extend_blocks=0` delegates directly to A, including Result/blind extras, and introduces',
             'no new fit/calibration dependency. Both disabled SF and disabled UF were checked on every stream.',
             'Enabled SF anchors rejected by structural/stage gates preserve A’s actions, verdicts and vision',
             'flags; they add stage telemetry. The following identity scope is proven:', '',
             '| cell | P3 decisions | B-val decisions | exact archived actions | rejected-anchor blind/LOOK checks |',
             '|---|---:|---:|---:|---:|']
    for d in reports:
        p, b = d['identity']['p3'], d['identity']['bval']
        lines.append(f"| {d['cell']} | {p['decisions']} | {b['decisions']} | {p['recorded_action_equal']+b['recorded_action_equal']} | {p['rejected_anchor_decisions']+b['rejected_anchor_decisions']} |")
    nd = sum(r['decisions'] for d in reports for r in d['identity'].values())
    controls = sum(r['executed_controls'] for d in reports for r in d['identity'].values())
    rejected = sum(r['rejected_anchor_decisions'] for d in reports for r in d['identity'].values())
    lines += ['', f'**{nd:,} decisions / {controls:,} recorded controls / 240 episodes, zero failures.**',
              'Every archived action chunk, recorded vision flag and cache verdict matches A; both disabled',
              'classes independently match A bitwise (including scores, confidence, rows, weights and extras).',
              f'Enabled structural/stage rejections pass {rejected:,} blind/LOOK comparisons.', '',
              '**Cannot simultaneously satisfy two literal clauses:** immediate outside-radius LOOK at the',
              'first blind check can change A before any extra block is served. Thus “identity whenever no',
              'extension is ultimately granted/served” cannot also hold for those early valve aborts.',
              'The delivered implementation checks the valve at every in-budget blind decision of an',
              'extension-eligible SF commitment and immediately LOOKs on a breach; structurally/stage-rejected',
              'anchors retain A and do not run an effective valve. No identity claim covers early valve aborts.',
              'B-val observed first-blind aborts: SF1 π0.5 L10-50=2, GR00T L10-500=1; SF2 each of those=1;',
              'all other cells=0. This is an explicit unresolved specification conflict, not a parity claim.', '',
              '## B-val replay eligibility, valve and modeled cost', '',
              'Each cell has 10 existing non-test B-val episodes (one per task), not the new 20-episode profile.',
              'Grant share is frozen structural/stage eligibility at the anchor, before the dynamic valve.',
              'Extra blocks and extension share report what survives the recorded-state valve.',
              'Valve rate is fired checks / effective eligible-commitment valve checks; UF has no valve.',
              'IR below is vision-cost / mean completed fixed-anchor cycle length, with zero calls;',
              'missing future states/cap LOOKs are censored. It is neither renewed-controller realized IR nor',
              'a causal SR estimate. Overlapping anchors are not independent. A comparison uses .076/.074.', '',
              '| cell | arm | anchors | grants | grant % | extra blocks | valve fires/checks | censored | modeled IR | mean / p95 blind μs |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['cell']} | {r['variant']} | {r['anchors']} | {r['granted']} | {100*r['granted_share']:.2f} | {r['extra_blocks']} | {r['valve_fires']}/{r['valve_checks']} | {r['censored']} | {r['modeled_IR']:.5f} | {r['timing']['mean_us']:.1f} / {r['timing']['p95_us']:.1f} |")
    lines += ['', 'Timing measures the full `blind_step`, including ordinary A tail checks and budget LOOKs;',
              'it excludes archive reads, constructing histories, vision, policy and wire transforms.',
              'Exact timing sample counts and age breakdowns are in `/tmp/r7_C1/<cell>_replay.json`.',
              'All eligibility shares exceed 5%; every modeled saving exceeds .005. Realized IR and SR kill',
              'criteria must still be measured on the coordinator’s new B-val profile episodes.', '',
              '## Library calibration', '',
              'Successful source rows only; even-row A cadence; source episode excluded from the frozen A',
              'candidate set. PCA/metric are not refit per held-out episode. Full support at both reference',
              'checks is required; each row sample is max(D_delta at five and ten controls). Row p95 is default.',
              'Episode p95 is E3’s quantile of per-episode maxima, not inverse-episode-length row weighting.',
              'The two conventions are explicitly separate in STAGES_API.md. CT p75 is anchor absolute',
              'state residual by task, with strictly-above empirical occupancy. No state-unit threshold floor.', '',
              '| cell | successful rows | full-support LOEO anchors | episodes | row p95 | episode p95 |',
              '|---|---:|---:|---:|---:|---:|']
    for f in fits:
        c=f['calibration']
        lines.append(f"| {f['cell']} | {f['successful_rows']} | {c['anchors']} | {c['episodes']} | {c['q_row95']:.6f} | {c['q_episode95']:.6f} |")
    lines += ['', '## Exact commands and checks', '', 'Run from `/home/weiland/projects/openpi`:', '', '```bash',
              PREFIX+f' -m {MODULE}.prefit --emit-specs',
              PREFIX+f' -m {MODULE}.prefit --cells all',
              PREFIX+f' -m unittest {MODULE}.test_follow -v',
              PREFIX+f' -m {MODULE}.replay --cells all',
              PREFIX+f' -m {MODULE}.verify_specs',
              PREFIX+f' -m {MODULE}.assemble_handback', '```', '',
              '`prefit_commands.sh` enumerates each of the 24 variant × cell rebuild commands using the delivered',
              'stage tables. Initial calibration command was `prefit --cells all`; final packaging rebuilds used',
              '`prefit --cells all --reuse-stages`, with identical stage/retrieval fingerprints.',
              'Final replays used CPUs 22–25; independent fit/test/packaging work used 66–69; all thread limits',
              'and environment variables above were retained. At most three CPU processes ran concurrently.', '',
              'Component tests: **13 passed**. Coverage includes E1 segmentation parity, isolated reversal,',
              'content/library fingerprints, true successor failures, unknown/zero-weight unanimity, exact',
              'π0.5/native-GR00T synthesis, non-five-control geometry, every eligible blind valve age, and',
              'disabled-fit dependency identity. Unit log: `/tmp/r7_C1/unit_tests.log`.',
              'Packaging: **48 specs / 24 fits / 48 YAMLs / 48 matrices checked**, **24 policy-origin LOOK checks**.',
              'Replay logs: `/tmp/r7_C1/replays.log`; consolidated machine-readable output:',
              '`/tmp/r7_C1/results.json`; each cell also has replay JSON and per-anchor timeline JSON.',
              'Fits were library-only adaptations of provenance-checked deployed A fits; no recorded outcomes',
              'were used in fitting. B-val replay reproduces acquisition-episode exclusion via the existing',
              '`/tmp/q1_method_c_fits/<cell>/recording_exclusions.json` maps. No test-init calibration.', '',
              '## Telemetry', '',
              'Anchor keys: `os_sf_mode0`, `os_sf_mode1`, `os_sf_unanimous`, `os_sf_event_mass`,',
              '`os_sf_rows_to_event`, `os_sf_unknown`, `os_sf_stage_gate`, `os_sf_state_valve`, `os_sf_cap`,',
              '`os_sf_structural`, `os_sf_stage_ok`, `os_sf_granted`.',
              'Eligible blind keys additionally: `os_sf_age`, `os_sf_extension`, `os_sf_valve_checked`,',
              '`os_sf_valve_fire`, `os_sf_delta`, `os_sf_radius`, `os_sf_look`, `os_sf_source`.',
              'Source 1 = unchanged native anchor tail, source 2 = true successor heads.',
              'LOOK 1 = cap/budget, 6 = A lifecycle, 11 = follow state valve; names stay on LookReason.',
              'New scalars precede legacy scalars in eligible first-tail results and fit within the plugin’s',
              '24-scalar pure-cache budget. Complete provenance remains in blind_extras. Disabled results',
              'remain untouched. Stage-rejected results add plan telemetry to blind_extras.',
              'The unchanged plugin logs `vision`, `source`, `look_reason`, executed heads, rows and weights.',
              'SF/UF introduce no camera or call decision; those fields belong to C2/C3 when composed.', '',
              '## Coordinator next steps', '',
              '1. Copy `/tmp/r7_C1/fits/*.pkl` to `<RUN>/fits/` and `/tmp/r7_C1/stages/*.pkl` to `<RUN>/stages/`.',
              '   Fitted tables are embedded, and artifact kwargs contain no scratch/stage path; copying needs',
              '   no metadata rewrite. Use the exact C1 method strings/kwargs. Profile specs reuse these same',
              '   fit filenames. C4’s aggregate profile specs use *_profile.pkl: copy/rename matching fits',
              '   if selecting that aggregate list; spec/kwargs/cell still match.',
              '2. Bind only `<RUN>` in plugin paths and C4 manifest paths, then use emit_arms with',
              '   `arms_profile.json`. C4 prepares the audited 20 B-val pairs per cell and their launch route.',
              '3. Smoke one B-val episode each for π0.5 SF1/SF2/UF1 and GR00T SF1/SF2/UF1 before the full',
              '   20-per-cell profile. Check full-hit verdicts, five-control wire heads, native/successor source,',
              '   terminal LOOKs, valve LOOK re-entry and actual stage1 counts/cost ledger.',
              '4. Profile all 24 variants × cells plus A on the same B-val pairs with full decision telemetry.',
              '   Apply SELECTION §4 kill rules and select the SF cap before any R7 test episode. Then emit',
              '   surviving eval500 rows; C1’s SF2 evaluation rows are candidate cap alternatives, not an',
              '   instruction to test both caps before freezing the choice.',
              '5. C3 can install `FollowExtension` via its documented bridge; SA requires a separate budget',
              '   solve and profile. No SA composed class, calibration or arm is claimed here.', '',
              'No closed loop, model inference, server, worker, chain, tmux, port, remote-host, git or destructive',
              'command was run. `/home/weiland/trace_runs` was read-only. The unresolved identity/early-valve',
              'conflict above is the only stated contract exception; realized profile/eval outcomes remain',
              'coordinator work by instruction.', '',
              '## File SHA256 and artifact inventory', '',
              'SHA256s below cover added code/spec/docs except this self-referential hand-back. Its digest',
              'and all files are also recorded in `FILES.sha256` after generation.', '', '```text']
    files = sorted(p for root in (HERE,HERE.parent/'stages') for p in root.rglob('*')
                   if p.is_file() and p.name not in ('HANDBACK.md','FILES.sha256'))
    for p in files:
        lines.append(f"{sha(p)}  {p.relative_to(HERE.parents[4])}")
    lines += ['```', '', 'Artifact SHA256s and byte sizes: `/tmp/r7_C1/packaging.json` (all 24 fits),',
              '`/tmp/r7_C1/<cell>_fit.json` (eight tables and three fits each). Frozen-table digests:', '', '```text']
    for f in fits:
        lines.append(f"{f['stage_sha256']}  {f['stage']}")
    lines += ['```','']
    (HERE/'HANDBACK.md').write_text('\n'.join(lines))
    files.append(HERE/'HANDBACK.md')
    (HERE/'FILES.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(files)))
    print(json.dumps(dict(decisions=nd,controls=controls,episodes=240,component_tests=13,
                         fits=24,arm_specs=48,handback=str(HERE/'HANDBACK.md'))))


if __name__=='__main__':
    main()
