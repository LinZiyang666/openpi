"""Build the handback from completed checks, hashes and actual fit payloads."""
import datetime
import hashlib
import json
from pathlib import Path
import pickle
import re
import shlex
import numpy as np
from exp.offline_search.rounds.r04.k7_guard.prepare import HERE, PREFIX, RUN
from exp.offline_search.rounds.r04.k7_guard.evidence import ROOT


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
                    + ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def pct(n,d):return f'{100*n/d:.4f}%'


def main():
    out=HERE/'results'
    fits=read(out/'fits.json'); arms=read(HERE/'arms_k7.json')
    parity=[r for p in sorted(out.glob('parity_*.json')) for r in read(p)['rows']]
    rates=[r for p in sorted(out.glob('rates_*.json')) for r in read(p)]
    hist=read(out/'historical.json');edges=read(out/'edges.json')
    for check in ('parity','plugin','smoke'):
        commands=read(out/'final'/f'{check}_commands.json')
        assert all(r['returncode']==0 for r in commands)
    plugins=[]
    for arm in arms:
        r=read(out/'final'/('plugin_'+arm['name'])/'selftest_report.json')
        assert r['PASS'];plugins.append(dict(arm=arm['name'],**r))
    concurrent=read('/tmp/k7_guard_concurrency_final/summary.json')
    assert len(concurrent)==5 and all(r['PASS'] for r in concurrent)
    (out/'concurrency_summary.json').write_text(json.dumps(concurrent,indent=2)+'\n')
    smoke=[]
    for p in sorted((out/'final').glob('smoke_*.log')):
        text=p.read_text();assert 'SMOKE PASS' in text
        n=int(re.search(r'fit\+query: (\d+) decisions',text)[1]);smoke.append(dict(tag=p.stem,decisions=n))
    footprint=[]
    for fit in fits:
        path=Path(fit['artifact']);assert path.stat().st_size==fit['bytes'] and sha(path)==fit['sha256']
        with path.open('rb') as f:b=pickle.load(f)
        m=b['method'];key=b['cell'].rsplit('_',1)[0]
        folder=ROOT/'library'/key/m.base.cand_name
        footprint.append(dict(name=path.stem,key=key,library=m.base.cand_name,rows=m.C.L,
            episodes=len(np.unique(m.C.ep)),representation_bytes=int(m.bytes_per_entry()*m.C.L),
            bytes_per_entry=m.bytes_per_entry(),valid_action_bytes=int(m.C.L*5*7*4*2),
            fit_s=b['fit_s'],pickle_bytes=fit['bytes'],m_thr=m.m_thr,c_thr=m.c_thr,
            stored_top_level_npy_bytes=sum(p.stat().st_size for p in folder.glob('*.npy'))))
    (out/'footprint.json').write_text(json.dumps(footprint,indent=2)+'\n')
    files=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('HANDBACK.md','HANDBACK.sha256'):
            files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p),
                modified_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()))
    (out/'deliverable_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
    external=[HERE.parent/'k1_blind/judge.py',HERE.parent/'k1_blind/blind_awm.py',
              HERE.parents[1]/'r03/h3_judge/judge.py',HERE.parents[2]/'closed_loop/plugin.py']
    (out/'dependency_hashes.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p)) for p in external],indent=2)+'\n')
    lines=['# K7 handback: vision-confirmed stuck guard', '',
        'Completed verification: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'.', '',
        '**Implementation, evidence, arm validation and final checks passed. Deployment placement is blocked:** '
        'the requested `/home/weiland/trace_runs/os_closed_loop/r04_k7/fits/` is mounted read-only in this session. '
        'The plugin completed the first fit but failed opening its `.pkl.tmp` with `OSError: [Errno 30] Read-only file system` '
        '(`results/prefit.log`). All five deployment artifacts were then fitted successfully with the installed plugin into '
        '`/tmp/k7_guard_fits/`, plus a spatial-50 verification fit. No artifact was installed at the requested destination. '
        'The coordinator must copy the verified files before launch.', '',
        'Only the new K7 directory and K7-owned temporary artifacts were written. Shared K1, K6, R3, plugin, src, '
        'harness and profile files were not edited. No GPU, server, port, simulator/LIBERO worker, chain, remote host, '
        'git operation or review-test access was used. All Python processes used CPUs 26–29,70–73 with BLAS threads 1 '
        'and CUDA hidden; at most eight Python processes overlapped. The concurrency worker uses eight connections/threads.', '',
        '## Rule and implementation', '',
        '`judge.py:VisionConfirmedBlindMixedJudge` subclasses K1 `BlindMixedJudge`; '
        '`judge.py:BlindMixedJudge` is an alias. Enable `stuck_guard="vision_confirmed"` (the default). '
        '`stuck_guard="dense"` explicitly delegates to K1. `progress_guard="noprog_span"`, '
        '`memo_reset_after_miss=False`, `events="none"` are the deployment settings. GR00T is refused with `SkipCell` '
        'before fitting because stock and K1 have different terminal-closed gripper signs. A memo-reset option is '
        'refused with vision confirmation because it would violate stock no-progress parity.', '',
        'For each pair of consecutive real vision anchors a < b, compute the minimum of the two task-centred '
        'camera cosines, using stock `core.centred_cos`, `M0`, `M1`, and the stock library 95th-percentile `c_thr`. '
        'Every dense transition j in (a,b] is confirmed only if that anchor cosine is ≥ c_thr AND its own '
        'float32 valid-state L2 motion `norm(rs[j,:8]-rs[j-1,:8])` is strictly below the stock pooled library '
        '10th-percentile `m_thr`. `stuck_n` is the length of the trailing consecutive confirmed run at the '
        'current vision decision. A high-motion transition breaks the run; later low-motion transitions within '
        'the same confirmed anchor interval can form a new suffix. Prefix transitions without a left vision '
        'anchor are never counted. No keys are read from blind rows, copied forward or synthesized.', '',
        'This is causal: confirmation arrives only at the closing anchor. No retrospective verdict is applied to '
        'already served blind HITs, and no right anchor is anticipated. Endpoint agreement is evidence of stillness, '
        'not proof that nothing moved out and back inside the gap. Requiring endpoint agreement preserves the '
        'stock visual veto and the raw-motion threshold while allowing dense state evidence to span a short gap.', '',
        'The same corrected count, clipped at 5, feeds V7’s `stuck` feature. Its stock LOEO library calibration '
        'and fitted thresholds are unchanged and were compared bit for bit against K1’s stock-calibrated frozen fits. '
        'Overtime intentionally consumes the corrected count too, since stock guard 3 requires `stuck_n >= 1`. '
        'All other guard/event/burst code on gaps is copied unchanged from K1; the motion diagnostic uses raw L2 '
        'and `dense_motion_guard=0` identifies the change. K1’s `blind_step`, phase advancement, budget, gripper, '
        'terminal, dense-motion look gate, residual gate, lifecycle gate and span no-progress logic are inherited.', '',
        'When all history decisions have vision, the implementation calls stock `MixedJudge.query` directly '
        'and synchronizes K1’s progress span afterward. All stock extras remain exact, including `noprog_n` '
        '(the all-vision branch does not add `noprog_span` to the result). Internally the span equals stock '
        '`noprog_n`, so subsequent blind look decisions retain K1 behavior. Step zero starts at count 0. '
        'After MISS, blind stepping requires a new real anchor; the stuck history itself is not reset, matching '
        'stock. Episode/task resets clear counters, progress and anchors; implicit query identity changes also reset.', '',
        '## Final parity and offline evidence', '',
        f'Final rerun: **{sum(r["decisions"] for r in parity):,} full scalar result comparisons, zero mismatches**, '
        '20 evenly spaced episodes per cell/scale, in guard-only and events+burst configurations. The reference is '
        'the stock MixedJudge class with the stock AWM class and frozen stock calibration, not K7 in another mode. '
        'Compared complete top-k, scores, actions, library, confidence, every extras key/value and `_s` state. '
        'This includes all `os_*` verdict inputs, flags/reasons/phases, `stuck_n`, `motion`, `vself`, `pred_err`, '
        '`zsum` and raw/final V7 confidence. Fresh, stale and step-zero regimes are all covered. '
        'The independent count computation also matched stock on all 189,904 decision/library pairs.', '',
        table(['Cell','Library','Guard-only comparisons','Events+burst comparisons','Mismatches'],
              [[r['cell'],r['scale'],r['decisions'],r['decisions'],0] for r in parity if r['mode']=='guard_only']),
        'Source: `evidence.py`, `results/parity_*.json`, `results/final/parity_*.log`. '
        'Final command lists are `results/final/parity_commands.json`. All calibration arrays, task means and '
        'thresholds are identical at each deployed scale; no 50-library result borrows 500-library information.', '',
        'Every all-vision row below uses all 500 recorded episodes of its query cell. These are **stuck flag '
        'counts / all decisions**, not total MISS rates or SR. Stock predicates and K1 dense predicates are '
        'evaluated on the exact same online QueryViews; K7’s independent history function is asserted equal '
        'to the stock scalar recurrence at every decision. Full retrieval/guard/V7 parity is the separate '
        'scalar check above.', '',
        table(['Cell','Library','N','Stock count (rate)','K1 dense count (rate)','K7 count (rate)'],
              [[r['cell'],r['scale'],r['decisions']]+[f'{r[k]["stuck_fires"]} ({pct(r[k]["stuck_fires"],r["decisions"])})'
                  for k in ('stock','k1_dense','k7')] for r in rates if r['budget']==0]), '',
        'For B=2, the frozen scheduler uses K1’s real blind-step/gate machinery and span progress, with '
        'ideation A’s saved 16-member anchors and K1 replay helpers. Robot observations, executed chunks and '
        'HIT/MISS history stay recorded; hypothetical judge verdicts do not alter the stream. Thus all '
        'comparators see the same vision mask. Stock is an **omnivision reference projected onto those vision '
        'decisions**, not a deployable stock judge with missing keys. K7 receives NaN key rows plus the '
        'vision mask for blind history; the K1 dense predicate reads only robot states. Inf cells have no blind decisions because every recorded previous '
        'action is a MISS; their B=2 rates equal B=0.', '',
        table(['Cache cell','Library','Vision / all','Stock fires','K1 dense fires','K7 fires','Stock / dense / K7 rates per vision'],
              [[r['cell'],r['scale'],f'{r["vision"]}/{r["decisions"]}']+[r[k]['stuck_fires'] for k in ('stock','k1_dense','k7')]
               +[' / '.join(pct(r[k]['stuck_fires'],r['vision']) for k in ('stock','k1_dense','k7'))]
               for r in rates if r['budget']==2 and r['cell'].endswith('cache')]), '',
        'Source: `results/rates_*.json` and `results/counts_*.npz` (full count arrays and B=2 mask). '
        'The B=2 schedule inherits ideation A’s documented batched-versus-scalar anchor limitation '
        '(maximum action difference 0.000298366 in its validation); no B=0 bit-parity claim uses those batched anchors. '
        'No counterfactual state evolution, live SR or IR improvement is inferred from these measurements.', '',
        '## Historical closed-loop audit', '',
        'Read all accepted attempts, including journal status `failed`, from the three existing 500-episode arms. '
        'The supplied reason counts are reproduced exactly. Stock R3 has no robot-state field here, but its '
        'logged raw `motion` and `vself` suffice for its stuck recurrence. K1 B=0 has full robot state and '
        'adjacent `vself`, allowing exact float32 raw-motion reconstruction. K1’s original dense count was '
        'also reconstructed with zero mismatches on both R4 paths. All three have no keys and `log_inputs=false`.', '',
        table(['Arm','N','Vision','Observed MISS','Observed stuck','Stock predicate on same path'],
              [[r['arm'],r['counts']['decisions'],r['counts']['vision'],r['counts']['miss'],r['counts']['stuck_flag'],
                r['stock_stuck_rederived'] if r['stock_stuck_rederived'] is not None else 'Unavailable across gaps'] for r in hist]), '',
        'On K1 B=0, holding every observation, proposal and other logged flag fixed, corrected reason counts '
        'are **stuck 670, terminal 259, overtime 119, no-progress 2,088**, for **3,136 / 29,615 '
        '(10.5892%)** projected forced MISSes, versus 4,144 (13.9929%) observed. Terminal/no-progress '
        'reason totals may rise when the higher-priority stuck bit disappears; their underlying flag bits '
        'were preserved. This is not an executed new arm. R3’s 699 stuck counts reproduce with zero '
        'count mismatches over all 29,340 decisions. For B=2, `vself` is absent after blind gaps, and no '
        'anchor-pair cosine/keys are logged: **K7 gap guards and bitwise V7 confidence cannot be reconstructed '
        'from those logs**. Full V7 bit parity is established only with the offline keys above.', '',
        'Source: `historical.py`, `results/historical.json`; it records all exact input file paths and sizes at read.', '',
        '## Plugin and edge verification', '',
        f'Final rerun of installed plugin blind selftest: **{len(plugins)}/{len(plugins)} PASS**, '
        f'{sum(r["decisions"] for r in plugins)} decisions, {sum(r["blind"] for r in plugins)} blind, '
        f'{sum(r["miss"] for r in plugins)} MISS, {sum(r["stage1_calls"] for r in plugins)} stage-1 calls, '
        f'{sum(r["broadcasts"] for r in plugins)} broadcasts. Each uses two interleaved connections, '
        'four episodes, duplicate rejection and offline exact log replay. The command’s `--blind --judge guard_only` '
        'is translated by the existing selftest into plugin `--os-blind --os-judge guard_only`.', '',
        table(['Arm','Decisions','Vision','Blind','MISS'],[[r['arm'],r['decisions'],r['vision'],r['blind'],r['miss']] for r in plugins]), '',
        'Eight-connection concurrent execution matched a fresh serialized process of the **same installed plugin** '
        'in observed reservation order. Comparisons include every non-timing decision field, wire action bytes, '
        'verdict, dense state/key/action/HIT/vision history, guard state, progress span and the complete blind anchor '
        '(rows, weights, phase, action, state and lifecycle). Native shadow search and the existing offline log '
        'verifier run too. Each configuration has 16 episodes and 183 decisions; all reached eight overlapping '
        'fake stage-1 calls. Delays are 60 ms for stage 1 and 25 ms on MISS. This is CPU fake-policy correctness, '
        'not GPU batching or live latency validation.', '',
        table(['Arm','Connections','Decisions','Vision','Blind','MISS','Exact match'],
              [[r['config'],r['connections'],r['decisions'],r['vision'],r['blind'],r['miss'],r['all_rows_actions_verdicts_histories_equal']] for r in concurrent]), '',
        'Evidence: `results/final/plugin_*/{selftest_report.json,verify_blind.json}`; '
        '`/tmp/k7_guard_concurrency_final/<arm>/{threaded,serialized}/{concurrency.json,verify.json}`; '
        'compact `results/concurrency_summary.json`. `concurrency_test.py` is an owned adaptation of K6’s '
        'driver. It adds guard/anchor state snapshots, uses the exact ncal=3000 arm prefits, and treats timing '
        'as informational under shared CPU load.', '',
        f'**{len(smoke)}/8 unchanged harness smokes PASS**, {sum(r["decisions"] for r in smoke)} decisions, '
        'both suites/regimes/scales, fresh default-ncal fits and reversed-episode determinism. '
        f'**{len(edges["synthetic_checks"])} focused edge checks**, {edges["gap_queries"]} real gap queries, '
        'and four lifecycle configurations passed. These cover step zero, missing left anchor, blind-key sentinel '
        'independence, camera conjunction, strict/inclusive thresholds, visual/motion run breaks, MISS-to-new-anchor, '
        'task reset, unchanged flags other than stuck/overtime, unchanged proposals, repeated gap queries, V7’s corrected count, '
        'and explicit GR00T refusal. An initial edge fixture expected a blind HIT while K1’s no-progress gate '
        'was active; the lifecycle-only fixture was corrected to disable guards for that isolated check. '
        'The judge required no correction. Evidence: `results/edges.json`, `results/final/smoke_*.log`.', '',
        '## Arms, fits and footprint', '',
        '`arms_k7.json` is emit_arms format. All five rows specify full model, `cost_ledger: true`, '
        '`--os-blind`, `--os-judge guard_only`, `--os-no-shadow-native`, the explicit cold-store `--os-root`, '
        'and `<RUN>/fits/<arm>.pkl`. '
        'Base kwargs are copied from their exact K1 counterparts: phase_particles, gates all; current/kref 5 '
        'at 50, big/kref 8 at 500. B=0 is the adapter control, expected to match stock g500 on identical '
        'inputs; its rollout SR is unverified. Five emitted CacheConfigs and artifact metadata all validated '
        'through the unchanged emitter/parser (`results/arms_validation.json`).', '',
        table(['Arm','Suite','Library','B'],[[r['name'],r['suite'],r['kwargs']['base_kwargs']['lib'],r['kwargs']['base_kwargs']['budget']] for r in arms]), '',
        'All listed artifacts were produced by the installed plugin `prefit_main` on CPU, protocol 4, '
        'default ncal=3000, loaded successfully in verification, and hashed after the final checks. '
        'Each artifact path currently starts with `/tmp/k7_guard_fits/`.', '',
        table(['Artifact basename','Bytes','SHA256'],[[Path(r['artifact']).name,r['bytes'],'`'+r['sha256']+'`'] for r in fits]), '',
        'The extra `verification_pi05_spatial_50.pkl` is evidence-only, not a sixth arm. Fits differ in '
        'their method/budget metadata even when byte sizes coincide. `results/fits.json` records every '
        'actual command and requested destination.', '',
        table(['Artifact','Library eps / rows','Representation bytes','Fit pickle MB','Fit s','Stock m_thr / c_thr'],
              [[r['name'],f'{r["episodes"]} / {r["rows"]}',r['representation_bytes'],f'{r["pickle_bytes"]/1e6:.6f}',
                f'{r["fit_s"]:.3f}',f'{r["m_thr"]:.12g} / {r["c_thr"]:.12g}'] for r in footprint]), '',
        'Representation is **632 bytes/row**, excluding shared actions and fixed calibration/projection arrays; '
        'valid π0.5 full-chunk action payload is another 280 bytes/row. The actual pickle sizes above include '
        'padded actions and auxiliary/fixed arrays. The supplied deployed-pickle comparison is 431 MB for '
        'π0.5 spatial and 1,103 MB for π0.5 l10. Spatial “50” actually contains 49 library episodes. '
        '`results/footprint.json` also records top-level stored NPY bytes (including full-resolution keys, '
        'excluding token subdirectories) to distinguish the cold store from the compact deployment fit. '
        'No borrowed big-library information or external models were used.', '',
        '## Exact commands and coordinator next steps', '',
        'All commands run from `/home/weiland/projects/openpi`. The exact per-arm prefit commands for the '
        '**requested final destination** are in `prefit.sh` and reproduced here. The first failed at the '
        'read-only destination; the remaining final-destination invocations were not run there. '
        'The successful commands differ only in artifact prefix, `/tmp/k7_guard_fits/`; '
        '`prefit_staged.sh` and `results/prefit_staged_commands.json` contain all six executed commands.', '',
        '```bash', (HERE/'prefit.sh').read_text().strip(), '```', '',
        'The checks actually run (initial and final labels use separate plugin log directories):', '',
        '```bash', 'PY=('+shlex.join(PREFIX)+')', 'K=exp.offline_search.rounds.r04.k7_guard',
        '"${PY[@]}" -m "$K.prepare"', 'bash exp/offline_search/rounds/r04/k7_guard/prefit.sh  # failed: read-only destination',
        '"${PY[@]}" -m "$K.run_prefits"',
        '"${PY[@]}" -m "$K.run_checks" parity --tag initial',
        '"${PY[@]}" -m "$K.run_checks" rates --tag full',
        '"${PY[@]}" -m "$K.edge_checks"',
        '"${PY[@]}" -m "$K.historical"',
        '"${PY[@]}" -m "$K.validate_arms"',
        '"${PY[@]}" -m "$K.run_checks" plugin --tag initial',
        '"${PY[@]}" -m "$K.run_checks" smoke --tag final',
        '"${PY[@]}" -m "$K.run_checks" parity --tag final',
        '"${PY[@]}" -m "$K.run_checks" plugin --tag final',
        '"${PY[@]}" -m "$K.concurrency_test" --source installed --config all --out /tmp/k7_guard_concurrency_final',
        '"${PY[@]}" -m "$K.edge_checks"  # final edge rerun',
        '"${PY[@]}" -m "$K.write_handback"', '```', '',
        'Coordinator: copy the five staged arm pickles to the requested directory, check the hashes above, '
        'resolve `<RUN>` to `/home/weiland/trace_runs/os_closed_loop/r04_k7`, then use normal emit_arms '
        'and the normal next server start. No shared plugin change or refit is required. Run the B=0 control '
        'before interpreting B=1/B=2, then measure paired 500-init SR, vision share, MISS share and the cost '
        'ledger on the requested arms. Existing servers do not import the new class until their next startup.', '',
        '```bash',
        *[f"cp --no-clobber {shlex.quote(r['artifact'])} {shlex.quote(r['requested'])}" for r in fits if not Path(r['artifact']).name.startswith('verification_')],
        '```', '',
        'Copying/launching was **not performed** here. No claim is made that the projected MISS reduction '
        'preserves success, that endpoint agreement observes hidden visual motion, or that CPU fake-policy '
        'concurrency predicts GPU/SR performance.', '',
        '## Deliverables and SHA256', '',
        'Modification times and exact source/script sizes are in `results/deliverable_hashes.json`. '
        'Read-only dependency hashes are in `results/dependency_hashes.json`. The handback itself is '
        'hashed separately in `HANDBACK.sha256` after generation.', '',
        table(['Owned file','SHA256'],[[r['path'],'`'+r['sha256']+'`'] for r in files])]
    (HERE/'HANDBACK.md').write_text('\n'.join(lines)+'\n')
    (HERE/'HANDBACK.sha256').write_text(sha(HERE/'HANDBACK.md')+'  HANDBACK.md\n')
    summary=dict(PASS=True,parity_comparisons=sum(r['decisions'] for r in parity),full_count_comparisons=189904,
        plugin_decisions=sum(r['decisions'] for r in plugins),plugin_blind=sum(r['blind'] for r in plugins),
        concurrency_decisions=sum(r['decisions'] for r in concurrent),smoke_decisions=sum(r['decisions'] for r in smoke),
        destination_blocked=True,staged_fits=len(fits))
    (out/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
