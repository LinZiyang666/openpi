"""Render the coordinator handback from the completed CPU inventories."""
from collections import Counter,defaultdict
from pathlib import Path
from .audit import HERE
from exp.offline_search.rounds.r17.filter_failed_knob.common import read

def main():
    catalog=read(HERE/'CAMPAIGNS.json');jobs=catalog['jobs'];queues=read(HERE/'QUEUES.json')
    audit=read(HERE/'AUDIT.json');affected=[r for r in audit['rows'] if r['affected']]
    trash=Path('/home/weiland/projects/_trash_longseg_20261017');v=read(trash/'VERIFICATION.json')
    lines=['# Decision-head clocks: withdrawal and fresh reruns (2026-10-17)', '',
      'CPU-only handback. No new server, worker, driver, chain or GPU job was launched. Existing active/queued campaigns and chain scripts were not rebuilt or edited. The new namespace is `fa_ls_*`; `NAMES.json` maps withdrawn lanes to fresh definitions. Do not launch the withdrawn NAMEs or resume their old continuations.', '',
      'The source model runtimes, worker archives, all fitted/calibrated arrays, correctors, Q3 artifacts and periodic gaps stay unchanged. New bootstraps pin the corrected server methods; new driver overlays preserve their decision metadata. Every new job requests timan1 **GPU 2** and timan107, with the original total W. The existing capacity allocator records any launch-time redistribution. timan108 is absent. No worker source/archive changes or worker rsync update is needed.', '',
      '## Corrected execution and calibration', '',
      '`ConnectionPolicy` counts segment heads separately from deliveries. A pending segment delivers all remaining tails; a tail extracts no new state and evaluates no retrieval/recipe/gate/knob/stall/corrector/model logic. Histories, semantic RNG index, corrector step, periodic schedule and Q3 schedules advance on heads. Frozen worker delivery IDs remain the ACK fence. `decision_index`, `delivery_index`, `env_step_index`, `segment_head`, and the actual/wire seeds survive frozen-worker transport in `server_timing`; new drivers promote the clock fields.', '',
      'Q3 artifacts were calibrated on 5-control observations. Their numeric calibration quantum is retained while the index now counts heads. For C10, a Q3 sample and one-anchor cooldown now span 10 environment controls instead of 5; a W-observation window and extra-look interval span twice as many environment controls. Extra LOOK coalesces with the next segment head; it cannot create a midpoint decision. The frozen Q3 false-alarm/coverage calibration is therefore not a cadence-matched claim. The corrector sees head-level state differences. The periodic gaps now count heads and may miss their former nominal IR targets. **No stall, metric, corrector or knob artifact was recalibrated/refit.** A cadence-matched Q3 calibration or target-IR recalibration would be required to re-establish those old claims; neither is part of these reruns.', '',
      'The same-step native-length pure result remains the SR target. The fresh `pure + long` controls measure the only admissible pure fallback inside the complete system. `RESULTS.json` stitches those C10 outcomes for tasks selecting pure, leaves them missing until measured, and reports whether their actual per-task tolerance holds. It never silently inserts C5 episodes into the C10 system. The native-helper stitch is retained under `native_length_fallback_diagnostic`, not as the final result.', '',
      '## Audit and backup', '',
      f'{len(audit["rows"])} definitions inventoried; {sum(bool(r.get("file")) for r in audit["rows"])} logs have empirical evidence. {sum(r.get("cuts",0)>0 for r in affected)} have sampled midpoint cuts; four additional uninterrupted ch/chr lanes used delivery-count recipe clocks and are withdrawn for the decoupling. “0%” means no cuts observed in the stated sample, not a full-file proof.', '',
      'The final sampler uses one reader, up to eight windows totaling 16 MiB / 16,000 lines per file, 4 MiB/s, five seconds between files, and `/proc/pressure/io` full avg10 <=20. Already completed bounded-prefix scans (up to 32 MiB) and full audits from before the I/O correction were reused and labeled; no full log was reread. `AUDIT.tsv` lists all affected and unaffected definitions and sample sizes.', '',
      '| Withdrawn run | Sample rows | Sample MiB | Midpoint cut % | Reason |',
      '|---|---:|---:|---:|---|']
    if (HERE/'RECALIBRATION.md').exists():
        lines[2:2]=['**Owner recalibration hold (2026-10-18):** dependent runs must stay behind the coordinator’s `[recal] released` gate. The old-artifact calibration paragraph below is historical and superseded. See [RECALIBRATION.md](RECALIBRATION.md) for source contradictions awaiting definitions, and the six independent gated cohorts. No recalibration pass or release is claimed.', '']
    for r in affected:lines.append(f'| `{r["run"]}` | {r.get("sample_rows",0):,} | {r.get("sample_bytes",0)/2**20:.2f} | {r.get("cut_percent",0):.3f} | {r.get("withdrawal_reason")} |')
    count=v['entries'] if isinstance(v['entries'],int) else len(v['entries'])
    lines+=['',f'Backup verification: **{v["status"]}; {count} rename-only entries; at least {v["bytes"]:,} bytes ({v["bytes"]/2**30:.3f} GiB)**. Sizes of 133 directory subtrees are shallow lower bounds, avoiding a cold inode walk. Same-filesystem device/inode checks verified every move. The manifest and dry run are `{trash}/MANIFEST.json` and `DRY_RUN.tsv`; `MOVE_JOURNAL.jsonl` and `VERIFICATION.json` record execution/verification. This covers local collected outcomes; no remote outcome directory was moved through tether.', '',
      'Definitions, plans, archives, assets, libraries/fits/stall artifacts, pure references and ready/admission records were kept. Moved derived artifacts include the GR00T/DP/Fast-WAM MT50 descent outputs, affected full-system/filtered outputs, DP/Fast-WAM formal_A rows/off decisions, cache rows and website blobs. Unaffected JSON rows were retained. Invalid continuation registrations were withdrawn; Cosmos and pure-Cache registrations stay intact. `knobsmokedp1355` is **trash only, no replacement and no queue item**, by the owner ruling.', '',
      '## Fresh definitions', '',
      f'{len(jobs)} built campaigns: {sum(j["episodes"] for j in jobs):,} episodes. Eight deferred LIBERO level units add at most 240,000; total fresh rerun bound **{sum(j["episodes"] for j in jobs)+240000:,}** before the off-cell skips. Of this, 19,000 are ten library-independent pure+long controls and 10,500 are six new native-length pure references. The historical resume cohorts contain no old outcomes; canonical task/init/seed/bundle/recipe/YAML signatures cover each full source lane with zero overlap (`source_cohorts` in `CAMPAIGNS.json`). Scheduler UID namespaces can change for a cohort; scientific identities and initial states do not.', '',
      '**Fast-WAM MT50 cache variants:** `fa_ls_v02_s`: R3, sizes 250/500/1000, 60,000 episodes. `fa_ls_v02_b`: R2, sizes 1500/2000/2500, 60,000. This is the owner-selected RUN_CONFIGS memory-cap split, recorded in each campaign and FFK definition. `OWNER_SPLIT_PROOF.log` verifies the canonical source plans.', '',
      '| NAME | Episodes | GPU | R/B | MPS | Requested workers | Stage |', '|---|---:|---|---|---|---|---|']
    for j in jobs:
        workers=', '.join(('timan1 GPU2' if n=='timan1' else n)+'×'+str(w) for n,w in j['workers'].items())
        c=read(Path(j['run'])/'campaign.json');mps=c.get('mps',{})
        enabled=mps.get('enabled',False) if isinstance(mps,dict) else bool(mps)
        lines.append(f'| `{j["name"]}` | {j["episodes"]:,} | {j["gpu"]} | {j["replicas"]}/{j["batch"]} | {"on" if enabled else "off"} | {workers} | {j["stage"]} |')
    totals=Counter()
    for j in jobs:totals[j['model'],j['benchmark'],j['mode']]+=j['episodes']
    for u in read(HERE/'FORMAL_QUEUE.json')['units']:
        if u['stage']=='levels':totals[u['model'],'libero',u['mode']]+=u['maximum_episodes']
    lines+=['','| Model | Benchmark | Mode | Fresh episode bound incl. controls/studies |', '|---|---|---|---:|']
    for key,n in sorted(totals.items()):lines.append('| '+' | '.join(key)+f' | {n:,} |')
    lines+=['','## Same-GPU pure references and timing', '',
      '`REFERENCES.json` inventories both GPUs for every affected model/benchmark/mode, with counts, R/B, descriptor equality and measured rates. `REFERENCE_SELECTION.json` is authoritative for the fresh main grids; `REFERENCE_ACTUAL_IDENTITY.json` also verifies actual ready.json model identities for all five reused references. The new native pure jobs are `fa_ls_dp_{r,f}_b_m` (2,500 each), `fa_ls_dp_{r,f}_b_l` (1,000 each), `fa_ls_fw_f_b_l` (1,000), and `fa_ls_dp_r_b_v` (2,500 for the migrated cache study). Each LIBERO 1,000 is 500 per suite. The pure+long jobs end `_p_m` / `_p_l` and use the same serving as their grids.', '',
      '| Model / benchmark / mode | H100 existing native pure | RTX 4090 existing native pure |', '|---|---|---|']
    refs=read(HERE/'REFERENCES.json')['cells'];combos=sorted({(r['model'],r['domain'],r['mode']) for r in refs})
    for model,bench,mode in combos:
        columns=[]
        for gpu in ('H100','RTX 4090'):
            r=next(v for v in refs if (v['model'],v['domain'],v['mode'],v['gpu'])==(model,bench,mode,gpu))
            text=', '.join('`'+v['run']+'`'+(' (identity matches)' if v['descriptor_equal'] else ' (different sealed identity)') for v in r['references'])
            if not text:text='Missing; fresh cost '+str(r['episodes_needed'])+' episodes'
            columns.append(text)
        lines.append(f'| {model} / {bench} / {mode} | '+ ' | '.join(columns)+' |')
    lines+=['',
      'Pure references for the unaffected legacy VLA and Cosmos queue units are listed in `UNAFFECTED_REFERENCES.json`; their unchanged serving/ref selection is reused. Existing GR00T/Fast-WAM MT50 pure references have fewer replicas (R8 vs R11, R3 vs R2 respectively), as in the approved source pipeline; B and GPU match. Their published references are reused, with this provenance explicit.', '',
      '`VARIANT_REFERENCES.json` separately inventories exact-descriptor references for the older cache studies. Their within-study corrector/long/stall comparisons remain paired; the analyzer reports no incompatible absolute pure comparison. The five missing historical LIBERO pure reference shapes cost **5,000 episodes** total at matching R/B (1,000 each, shared across repeated sizes/cohorts); these extra references are inventoried, not queued. Their absence does not affect fresh original-versus-filtered main-grid pairing. No historical pure result or differing-version latency receipt is silently substituted. Historical C10/C15 pure horizon controls are preserved by the keep-pure rule but are not reused as decoupled fallback controls.', '',
      'The existing cost receipts are 4090 profiles; GR00T/Fast-WAM MT50 receipts were not valid exclusive profiles. `PROFILES.json` defines seven existing-profiler jobs, including the unaffected H100 Cosmos pipeline, shared by full/reduced modes whose actual model identities agree. Their standalone bundles contain exact load.json copies and recorded RGB/state; no experiment is restaged or modified for profiling. Profile outputs are accepted only with valid=true, matching actual loader model identity and matching GPU. Until measured, formal IR is null (the model-call proxy and SR remain available). This measures costs only; it does not recalibrate gaps or Q3. Legacy R15/R16 keeps its native per-decision cost ledger; no new legacy timing qualification is claimed.', '',
      '## Commands and scheduling', '',
      'Run from `/home/weiland/projects/openpi`. Every fresh/deferred unit has the following exact pair (substitute a NAME from the tables/lists):', '',
      '```bash', 'bash exp/offline_search/rounds/r17/longseg_1017/commands/prepare_<NAME>.sh', 'bash exp/offline_search/rounds/r17/longseg_1017/commands/launch_<NAME>.sh', '```', '',
      '`H100_COMMANDS.txt` / `4090_COMMANDS.txt` contain every expanded prepare/launch pair; `H100_NAMES.txt` / `4090_NAMES.txt` are chain-ready ordered NAME lists. Additive fa_ls wrappers also exist in `filter_failed_knob/commands`, where the current chain runner looks; no existing command or chain file was replaced. Do not rebuild the catalog or invoke `filter_failed.build` / `filter_failed_knob` global builders. Existing active runs `fa_co_r_u_l` and `ffk1015_pi05_mt50_reduced_descent` are preserved and omitted from the new remaining-work lists. The current old chains still mention withdrawn ffk DP/Fast-WAM jobs; replace their remaining items with these lists before those jobs are dispatched.', '',
      'LIBERO OFF runs first in both modes/arms. Its levels prepare invokes the existing cell rule through `longseg_1017.workflow`: new OFF reaching the same-step pure SR skips the entire cell; only failing cells get the unchanged five-level grid. Build explicitly after the corresponding OFF and pure references finish:', '', '```bash']
    for u in read(HERE/'FORMAL_QUEUE.json')['units']:
        if u['stage']=='levels':lines.append('.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name '+u['name'])
    lines+=['```', '', 'Analysis and CPU checks:', '', '```bash',
      'env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.check',
      '.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.analyze --benchmark all',
      '.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.analyze --benchmark all --include-unaffected',
      '.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.cache_variants', '```', '',
      'The analyses and off classification use the single-reader I/O governor. New prepares use 4 MiB/s Python archive/hash I/O and public rsync, PSI pauses, and reuse installed immutable input seals. H100 stages the small serving metadata tree and any missing original input roots; it does not duplicate already installed fitted arrays. Transport only: new local servers use public 23130, driver 23139; H100 servers keep their original serving ports and use local driver 23199. `TRANSPORT.json` and each campaign record the changes; task UIDs and scientific metadata are unchanged.', '',
      'Measure the seven timing units **after both experiment queues are closed**, on idle GPUs with MPS off, so another lane’s local renderers cannot invalidate exclusivity. Use the same prepare/launch pair with NAMEs in `H100_TIMING_NAMES.txt` / `4090_TIMING_NAMES.txt`. The profiler already enforces its own exclusivity/validity rules. Re-run analysis after profiles finish. `--include-unaffected` adds Cosmos and legacy VLA formal-A rows through the existing analyzer, writing its auxiliary output only under this package; it leaves the existing formal_A output intact. Pi0.5 MT50 remains with its unchanged existing analysis.', '',
      '## Ordered queues and estimates', '',
      'One lane per GPU; reduced steps first, OFF before dependent levels. New DP MetaWorld grids and its cache study are on 4090 with unchanged R/B/MPS/archive and fresh matching pure controls. The migrated cache study keeps its qualified R4/B8/24-GiB shape (`BALANCE_BUILD.log`, `BALANCE_CPU.log`). GR00T R11 and Fast-WAM R2 MT50 must stay on H100 under the unchanged resident-memory limits. Other cache studies preserve their source GPU/configuration, including the owner’s R3/R2 split. The source-attested local DP R4 admission presentation is reused only where the source preserved CACHE_CAPACITY_ADMISSION.json, exactly as in the standard ffk builder; actual R/B, sampler and all loader identity checks remain intact. The lists are a near-balanced feasible parallel plan, not a measured optimal makespan.', '',
      'Rates are previous accepted-episode spans (not withdrawn SR). DP MT50 on its new GPU uses the measured 4090 native-pure rate as a conservative proxy; the new R8/B32 and R4/B16 grid throughputs are unmeasured. Corrected clocks, level mixtures, shared worker capacity and infrastructure losses can change these estimates. Queue hours exclude active-run remaining time, prepare/startup/cleanup and use all-cell level upper bounds. Legacy pure test-A spans supply the VLA estimates; historical source metadata supplies R17 estimates. `QUEUES.json` records every rate and source.', '']
    for gpu,items in queues['gpus'].items():
        hours=sum(r['hours'] or 0 for r in items)
        lines += [f'### {gpu}: {sum(r["episodes"] for r in items):,} episodes; approximately {hours:.1f} GPU-hours', '', '```text',*[r['name'] for r in items],'```','']
    cpu=read(HERE/'CPU_PROOF.json');deferred=read(HERE/'DEFERRED_CPU_PROOF.json')
    lines += ['Timing after both queues: '+', '.join(f'{g}: {sum(r["hours"] for r in rs):.2f} h proxy' for g,rs in queues['profiles_after_both_queues'].items())+'.','',
      '## CPU evidence', '',
      f'**22 targeted tests passed; {len(cpu["rows"])} frozen startup/planner checks passed; {len(deferred["rows"])} deferred level-template checks passed.** CUDA was initialized in zero probes; the longest computed MPS pipe is {max(r["mps_pipe_bytes"] for r in cpu["rows"]+deferred["rows"])} bytes (limit 85). `PROFILE_INPUT_CPU.json` also checks seven independent profiler bundles / eleven observation inputs and exact load.json bytes. `ANALYSIS_SMOKE.log` exercises all 84 expected affected grid cells; every unmeasured SR and paired difference remains null.', '',
      '`CPU_TESTS.log` records the real fitted-recipe replay, forced periodic/stall calls, same-observation short/long clock invariant, exact tails, frozen-worker metadata, disabled-cache controls, staged-input reuse, transport disjointness and I/O archive roundtrip tests. No simulator/server/socket was started by these tests.', '',
      '```bash',
      'env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q \\',
      '  exp/r17_integration/tests/common/test_serving.py::test_forced_knob_or_stall_look_waits_for_committed_tail \\',
      '  exp/r17_integration/tests/ops/test_cache_study.py::test_sealed_fit_real_layout_cache_corrector_and_tail \\',
      '  exp/r17_integration/tests/ops/test_cache_study.py::test_original_q3_confirmed_ambiguous_and_one_fresh_anchor_cooldown \\',
      '  exp/offline_search/rounds/r17/longseg_1017/test_boundary.py', '```', '',
      '`CPU_PROOF.json` binds each new campaign’s exact descriptor bytes to the existing strict frozen-loader CPU proof, reruns the new pinned bootstrap in its actual sealed-version interpreter with CUDA disabled, checks native DP admission, unchanged input inodes and native planner/UID coverage, and checks actual GPU-hash MPS paths <=85 bytes. We reused immutable loader proofs rather than rehash/reload multi-GB weights/fits on the busy data disk; the receipt says so. `DEFERRED_CPU_PROOF.json` checks all potential level tasks and ties the deferred templates’ identical descriptors/runtime/interpreter to their checked OFF campaign. Actual selected cells are instantiated only after the new OFF results.', '',
      'Broader pre-existing tests contained six unrelated model-binding failures in code with existing pass branches; they were not changed or used to weaken any identity check. Only the targeted required tests are claimed here.', '']
    recal=read(HERE/'RECALIBRATION.json') if (HERE/'RECALIBRATION.json').exists() else {}
    if recal.get('status','').startswith('APPLIED'):
        lines[2:4]=['**Owner decision-clock recalibration (2026-10-18):** '+
            f'{recal["recalibrated_levels"]} potential level definitions and {recal["recalibrated_Q3_artifacts"]} distinct C10 Q3 artifacts have been recalibrated. '+
            'Level IR is M/N segment heads (absolute prediction tolerance .005); latency-weighted IR remains separate. '+
            'C5 artifacts and independent queued units are unchanged. '+
            f'{len(recal["unattainable_levels"])} tolerance exceptions are explicitly listed in [RECALIBRATION.md](RECALIBRATION.md). '+
            'The coordinator alone reviews and writes `[recal] released`; this agent has not released or launched anything.', '']
        lines=[line.replace('The source model runtimes, worker archives, all fitted/calibrated arrays, correctors, Q3 artifacts and periodic gaps stay unchanged.',
            'The source model runtimes, worker archives, retrieval metric/whitening/PCA and correctors stay unchanged. Only C10 Q3 artifacts and periodic gaps are recalibrated under the owner ruling.') for line in lines]
        lines=[line.replace('Existing active/queued campaigns and chain scripts were not rebuilt or edited.',
            'Only recalibration-dependent, never-prepared gated definitions were regenerated under the owner ruling. Active/independent campaigns and chain scripts were not edited.') for line in lines]
        lines=[('C10 Q3 is refitted on successful-library segment-head sequences using the original empirical LOEO q90 residual/q10 advance procedure. '
            'Its temporal units are exec_steps=commit_controls=1 decision, with the original window fraction, minimum windows, K and reference requirements. '
            'This removes the former quantum conversion. No new online FAR or conformal-coverage guarantee is claimed. '
            'All C5 artifacts remain unchanged. See RECALIBRATION.md for actual per-task parameters and achieved empirical metrics.'
            if line.startswith('Q3 artifacts were calibrated on 5-control') else line) for line in lines]
        lines=[line.replace('Until measured, formal IR is null (the model-call proxy and SR remain available).',
            'Until timing is measured, latency-weighted IR is null. Level IR is the independently reported head-count M/N, matching the gap fitter.') for line in lines]
        lines=[line.replace('## CPU evidence','## Initial CPU evidence (before recalibration)') for line in lines]
        lines+=['## Recalibration evidence and current scheduling','',
            'The queue lists above are the original balancing proposal. The coordinator has since placed six independent cohorts before the existing recalibration gate; the live chain order is authoritative and was not edited here. '+
            'See RECALIBRATION_DEPENDENCIES.json for every gated item. Deferred levels automatically consume the new input map when built after OFF.','',
            'Regenerated definitions preserve task UID/init/seed and serving topology; their new recipe/YAML identities remain in actual evidence. '+
            'RECALIBRATION_APPLIED.json beside each changed campaign records the comparison. RECALIBRATION_CPU_PROOF.json and '+
            'RECALIBRATION_DEFERRED_CPU_PROOF.json are the new startup/planner receipts; RECALIBRATION_TESTS.final.log is the final targeted test result.','',
            '```bash',
            'env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.check --out exp/offline_search/rounds/r17/longseg_1017/RECALIBRATION_CPU_PROOF.json',
            '```','']
        if recal.get('cpu_verification'):
            proof=recal['cpu_verification']
            lines += [f'Final recalibrated checks: **{proof["startup_planner_passed"]}/60 startup/planner, '
                f'{proof["deferred_templates_passed"]}/8 potential level templates, '
                f'{proof["targeted_tests_passed"]} targeted tests passed**. '
                f'CUDA initialization count {proof["cuda_initialized"]}; maximum MPS pipe '
                f'{proof["maximum_mps_pipe_bytes"]}/85 bytes. '
                'The level map has 838/840 entries within .005; the two original-library LIBERO spatial '
                'size-100 DP k10 entries (full/reduced) have a confirmed-Q3 call floor .1059288538. '
                'They are explicitly marked outside tolerance, with no silent target or stall-rule relaxation.','']
    (HERE/'HANDBACK.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
