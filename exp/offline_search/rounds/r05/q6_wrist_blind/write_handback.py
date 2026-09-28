"""Audit final evidence, hash deliverables, and write the factual hand-back."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import shlex
import subprocess
from .common import HERE, FITS, PREFIX, write_json

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def rate(r,k): return f"{r[k]['fires']:,} ({100*r[k]['per_vision']:.4f}%)"

def main():
    unit=read(HERE/'results/unit.json'); assert unit['PASS']
    arms=read(HERE/'results/arms_validation.json'); assert arms['PASS']
    evidence=[read(HERE/f'results/evidence_{s}_{n}.json') for s in ('l10','spatial') for n in (50,500)]
    assert all(x['PASS'] for x in evidence)
    for group in ('plugin','existing','smoke','evidence'):
        commands=read(HERE/f'results/final/{group}_commands.json')
        assert all(x['returncode']==0 for x in commands)
    plugins=[]
    for p in sorted((HERE/'results/final').glob('plugin_*/selftest_report.json')):
        r=read(p); assert r['PASS']; plugins.append(dict(name=p.parent.name.removeprefix('plugin_'),**r))
    assert len(plugins)==8
    existing=[]
    for p in sorted((HERE/'results/final').glob('*/selftest_report.json')):
        if p.parent.name.startswith('plugin_'): continue
        r=read(p); assert r['PASS']; existing.append(dict(name=p.parent.name,**r))
    assert len(existing)==8
    concurrency=read('/tmp/q6_concurrency_final/summary.json')
    assert len(concurrency)==8 and all(x['PASS'] and x['peak']==8 for x in concurrency)
    write_json(HERE/'results/concurrency_summary.json',concurrency)
    smokes=[]
    for p in sorted((HERE/'results/final').glob('smoke_*/*/pi05_*.json')):
        r=read(p)
        if 'n_decisions' in r: smokes.append(dict(path=str(p),**r))
    assert len(smokes)==8, [(r['path']) for r in smokes]
    blobs=[]
    for p in sorted(FITS.glob('*.pkl')):
        with p.open('rb') as f: b=pickle.load(f)
        m=b['method']; assert m.fitname == m.libname and m.camera_mode=='wrist_only'
        blobs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),fit_s=b['fit_s'],
            m_thr=m.m_thr,c_thr=m.c_thr,library=m.libname,rows=m.C.L,
            representation_bytes=int(m.bytes_per_entry()*m.C.L),bytes_per_entry=m.bytes_per_entry(),
            kwargs=b['kwargs'],cell=b['cell'],spec=b['spec'],
            modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()))
    assert len(blobs)==8
    write_json(HERE/'results/fits.json',blobs)
    dependencies=read(HERE/'results/dependencies_before_final.json')
    after={p:sha(p) for p in dependencies}
    write_json(HERE/'results/dependencies_after_final.json',after)
    assert dependencies==after, 'a read-only dependency changed during final verification'
    for p in HERE.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    for p in HERE.glob('*.sh'): subprocess.run(['bash','-n',str(p)],check=True)
    manifest=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('HANDBACK.md','HANDBACK.sha256'):
            manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
                modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()))
    write_json(HERE/'results/deliverables.json',manifest)
    summary=dict(PASS=True,verified_utc=datetime.now(timezone.utc).isoformat(),
        unit_checks=len(unit['synthetic_checks']),parity_queries=sum(r['parity_queries'] for r in unit['real']),
        gap_queries=sum(r['gap_queries'] for r in unit['real']),
        plugin_decisions=sum(r['decisions'] for r in plugins),plugin_blind=sum(r['blind'] for r in plugins),
        plugin_miss=sum(r['miss'] for r in plugins),concurrency_decisions=sum(r['decisions'] for r in concurrency),
        concurrency_blind=sum(r['blind'] for r in concurrency),
        concurrency_miss=sum(r['miss'] for r in concurrency),
        smoke_decisions=sum(r['n_decisions'] for r in smokes),
        unchanged_dependency_hashes=True,primary_arms=4,variant_arms=4,fit_count=8)
    write_json(HERE/'results/final_audit.json',summary)
    names=('stock_two_camera','k1_dense','k7_two_camera','q6_wrist')
    report=[f'# Q6 wrist-only blind guard hand-back\n\nFinal audit: **PASS**, {summary["verified_utc"]}.',
    '''
Only `exp/offline_search/rounds/r05/q6_wrist_blind/` and `/tmp/q6_*` were written. No shared plugin, stage override,
K1/K3/K7, harness or src file was edited. No GPU, server, port, LIBERO worker, chain, remote host, git command,
or review-test read was used. Python commands used CPUs **10–13,54–57**, BLAS/OMP threads 1 and CUDA hidden;
driver subprocesses are explicitly affinity-prefixed, with at most eight Python processes.

## Rule and justification

`judge.py:WristVisionConfirmedBlindJudge` subclasses K7 `VisionConfirmedBlindMixedJudge`, fixes its base to
K1 `BlindWristAWM` (K1 blind machinery + K3 `WristAWM`), and supplies K3 `WristView` during threshold fitting,
V7 calibration, queries and `confirmed_stuck`. Both legacy visual slots reference the wrist key, so
`min(c_wrist,c_wrist) = c_wrist`. The guard consumes no base-camera placeholder or blind-row visual key. The retrieval
representation contains wrist once: 64 wrist PCs + 8 state dimensions. This is a method composition, with no
plugin change required.

For consecutive real vision anchors a < b, compute `core.centred_cos(w[b], w[a], M_wrist[task])`. Confirm
transitions in `(a,b]` only when this is **>= the deployed library wrist p95**. Each dense transition must also
have **float32 valid-state L2 motion < the pooled deployed-library p10**. Count the trailing run exactly as K7:
high motion or failed visual confirmation breaks it; a prefix without a left visual anchor is unconfirmed.
The stuck bit fires at count >= 2; the corrected count also feeds overtime and V7 (clipped at 5).
The complete motion/gap implementation is inherited from K7, with no normalized K1 dense-motion substitution.
At all-vision/B=0 the stock MixedJudge path is retained and agrees bitwise with K3's wrist judge on the tested
streams, including all extras and confidence. Q6 does not claim two-camera stock retrieval or verdict parity.

The available wrist view supplies a real visual veto while fitting its own percentile keeps the same nominal
upper-tail calibration convention. Reusing the two-camera-min threshold would calibrate a different statistic.
Percentile calibration does not make guard firing probabilities equal: motion conjunction, serial runs, and
task distribution matter. Wrist-only confirmation can miss changes visible exclusively to the base camera;
endpoint agreement also cannot exclude out-and-back motion during a blind gap. These are limitations, not
offline evidence of an SR improvement.

Defaults: `base_kwargs={"serving":"anchor_tail","budget":1,"gates":"budget_only"}`, `events="none"`,
`progress_guard="noprog_span"`, `stuck_guard="vision_confirmed"`, `memo_reset_after_miss=False`.
`budget_only` is K1's base gate setting: K1's judge-level no-progress look rule and lifecycle checks remain active.
First decision, after MISS, exhausted budget, invalid anchor, reset and task change require vision. The tail
uses exactly steps 5–9 of the accepted vision chunk. The separate phase variant explicitly uses
`serving="phase_particles", budget=2, gates="all"`; it is not the default or a measured winner.
GR00T, dense-stuck substitution, legacy `noprog_n`, altered percentiles, and memo reset are refused.

## Calibration and library composition

Each fit uses its own deployed library (`current` at 50, `bpool_cs` at 500), including task means, wrist p95,
motion p10, AWM metric and V7 LOEO calibration. No 50-scale fit borrows 500-scale rows. The deployed 500 pools
are **not entirely successful**: l10 has 436 successful / 64 failed episodes; spatial has 487 / 13.
Calibration uses all deployed rows, as requested. Success labels are evaluator-only for the separate
successful-demo rates below. Spatial “50” has 49 actual episodes.
''',table(['Suite','Scale','Episodes / rows','Successful eps / rows','Raw motion p10','Two-camera min p95','Wrist p95'],
        [[s['suite'],s['scale'],f"{e['footprint']['library_episodes']} / {e['footprint']['library_rows']}",
          f"{e['footprint']['successful_episodes']} / {e['footprint']['successful_rows']}",
          f"{s['m_thr']:.12g}",f"{s['stock_c_thr']:.12g}",f"{s['c_thr']:.12g}"]
         for s,e in zip(unit['real'],evidence)]),
    '''
## Final stuck firing rates

These are **stuck-bit counts / vision decisions**, with percentages, including episode-start decisions in the
denominator. They are not overall guard/MISS rates or SR. All-vision library/stream counts use the exact online
float32 cosine and motion predicates; independent K7/Q6 history computations are asserted equal to scalar
recurrences at every row. Library percentile fitting retains stock/K3 arithmetic (float64 reduction after
float32 centering), while online comparison uses stock `centred_cos`.
The K1 dense comparator retains its own per-task state normalization (std floor .05), RMS motion and
task-specific p10. Stock/K7/Q6 use the pooled raw-L2 p10. Every comparator uses stuck count >= 2.

### Successful demos and all deployed library rows
''']
    for source in ('successful_library','deployed_library_all'):
        rows=[r for e in evidence for r in e['rows'] if r['source']==source]
        report.append(source+'\n\n'+table(['Suite','Scale','Episodes','N','Stock 2cam','K1 dense','K7 2cam','Q6 wrist'],
            [[r['suite'],r['scale'],r['episodes'],r['decisions']]+[rate(r,k) for k in names] for r in rows]))
    report.append('### Replayed decision streams, all vision\n\nAll 500 recorded episodes in each cell.\n')
    rows=[r for e in evidence for r in e['rows'] if r['source'].startswith('pi05') and r['schedule']=='all_vision']
    report.append(table(['Cell','Scale','N','Stock 2cam','K1 dense','K7 2cam','Q6 wrist'],
        [[r['source'],r['scale'],r['decisions']]+[rate(r,k) for k in names] for r in rows]))
    report.append('''
### Same-mask replay with blind gaps

The actual Q6 `blind_step` and scalar `query` select each common vision mask using recorded states, actions,
HIT/MISS history and its real wrist retrieval/progress. Hypothetical guard verdicts do not change that recorded
path. Stock is an **omnivision reference projected onto the common vision mask**; it is not deployable with
missing keys. K7 and Q6 see only real anchor keys, with NaN blind history rows. K1 sees dense states. This isolates
the guard predicates, not four alternative closed-loop trajectories. Inf cells contain only recorded MISSes,
so lifecycle prohibits blindness and both gap schedules have exactly their all-vision rates above.
''')
    for schedule in ('tail','phase2'):
        rows=[r for e in evidence for r in e['rows'] if r['source'].endswith('_cache') and r['schedule']==schedule]
        report.append(schedule+'\n\n'+table(['Cell','Scale','Vision / all','Stock 2cam','K1 dense','K7 2cam','Q6 wrist'],
            [[r['source'],r['scale'],f"{r['vision']} / {r['decisions']}"]+[rate(r,k) for k in names] for r in rows]))
    report.append('''
Raw evidence: `results/evidence_{l10,spatial}_{50,500}.json`, per-row count/vision-mask arrays
`results/counts_*.npz`, exact commands `results/final/evidence_commands.json`. Inputs are the cold store's
`library/pi05_{l10,spatial}/{current,bpool_cs}/` and `queries/pi05_{l10,spatial}_{inf,cache}/`.
Historical closed-loop logs without endpoint keys cannot establish the wrist predicate across gaps; no
historical wrist SR or observed closed-loop guard reduction is claimed here.

## Tests rerun against final files
''')
    report.append(f"**{summary['unit_checks']} synthetic/configuration checks**, **{summary['parity_queries']:,} full bitwise K3 wrist comparisons**, "
        f"**{summary['gap_queries']} real gap queries**; four same-library threshold recalibration checks, exact tail slices, "
        "missing-camera access traps, NaN/sentinel independence, strict/inclusive boundaries, task centering, lifecycle/reset, "
        "V7 corrected count, unchanged proposal and non-stuck/non-overtime flag checks all passed. `results/unit.json`.\n")
    report.append(f"Eight installed-plugin blind selftests passed: **{summary['plugin_decisions']} decisions, "
        f"{summary['plugin_blind']} blind, {summary['plugin_miss']} MISS**. Each includes two interleaved connections, four episodes, "
        "duplicate rejection, once-only broadcast/commit and exact logged-input replay.\n")
    report.append(table(['Arm','Decisions','Vision','Blind','MISS'],
        [[r['name'],r['decisions'],r['vision'],r['blind'],r['miss']] for r in plugins]))
    report.append('''
`plugin_selftest.py` invokes the **unchanged installed** selftest in its own process. It passes
`--os-blind --os-stage1-mode wrist_only --os-tokens off --os-judge guard_only` through the production stage
parser/validator and startup hook; its fake key builder emits zero camera 0 plus exact stored wrist/state,
and token access raises. Startup asserts wrist_only and K10. This is CPU serving/history verification, not
a GPU vision-tower/MISS-completion test; K3's existing GPU evidence is a dependency, not rerun or credited here.
''')
    report.append(f"Eight connections per configuration, **{summary['concurrency_decisions']} total decisions**, "
        f"including **{summary['concurrency_blind']} blind and {summary['concurrency_miss']} MISS**, "
        "all eight configurations matched a fresh serialized process in reservation order. All non-timing decision "
        "fields, wire action bytes, verdicts, dense key/state/action/HIT/vision histories, guard/progress state and "
        "full blind anchors matched exactly; each threaded run reached eight overlapping fake stage calls.\n")
    report.append(table(['Arm','Decisions','Vision','Blind','MISS','Peak concurrent'],
        [[r['config'],r['decisions'],r['vision'],r['blind'],r['miss'],r['peak']] for r in concurrency]))
    report.append('Raw concurrency evidence: `/tmp/q6_concurrency_final/<arm>/{threaded,serialized}/`; compact '
        '`results/concurrency_summary.json`. Sleeping fake-policy timing is not a throughput claim.\n')
    report.append('Eight existing selftest recipes passed (the standard test source was not edited):\n'+
        table(['Recipe','Decisions','PASS'],[[r['name'],r.get('decisions',r.get('n_decisions','see report')),r['PASS']] for r in existing]))
    report.append(f"Eight fresh-fit harness smokes passed, **{summary['smoke_decisions']} decisions**, both suites/regimes/scales, "
        "including reversed-episode determinism, valid online inputs and finite/shape checks. Default ncal=3000 for "
        "Q6 fits, smokes, plugin and concurrency tests; only the inherited K1/K3/K7 regression recipes use ncal=64. "
        "Full smoke metrics/timing are in `results/final/smoke_*/`; runtime timing was measured under shared CPU load.\n")
    report.append(table(['Smoke','Decisions','Mean action error','AURC','ms/query'],
        [[Path(r['path']).parent.parent.name,r['n_decisions'],f"{r['metrics']['err_mean']:.6f}",
          f"{r['metrics']['aurc']:.6f}",f"{r['timing']['ms_per_query']:.3f}"] for r in smokes]))
    report.append('''
Initial evidence-driver checks exposed two evaluator issues: its direct wrist `_self_change` call initially
omitted `WristView`, and its initial assertion incorrectly assumed the 500 pools were all-success. The driver
was corrected; no judge change was needed. Final evidence reran all four cells/scales after those corrections.
Initial failed logs are retained. Every final command group returned zero.

## Arms, prefits and bytes

`arms_q6.json` has exactly **four primary arms**, π0.5 `{l10,spatial}` × `{50,500}`, all wrist + anchor_tail B1 +
wrist-confirmed guard. `arms_phase2.json` is a separate four-arm prespecified variant, not part of the requested
four-arm primary set. Both are emit_arms format, with `<RUN>` fit/evidence placeholders, `full_model:true`,
`miss.num_steps:10`, `write_policy.type:never`, and `cost_ledger:true`. All eight emitted CacheConfigs, production
stage-method validation and exact artifact spec/kwargs/cell metadata passed. No policy-tail flag is enabled.

All eight prefits were actually executed, sequentially, using the installed CPU plugin into `/tmp/q6_fits/`.
`prefit.sh` is the exact complete command list; `results/prefit_commands.json` has argv arrays. Stage mode is a
server-wrapper flag and is therefore omitted from the plugin-only prefit command; the fitted method already
fixes the wrist metric. Serve with every flag from the arm JSON.
''')
    report.append(table(['Artifact','Bytes','Fit seconds','SHA256'],
        [[Path(r['path']).name,r['bytes'],f"{r['fit_s']:.3f}",f"`{r['sha256']}`"] for r in blobs]))
    report.append(table(['Suite / scale','Stored library NPY bytes','Representation bytes','Tail artifact bytes'],
        [[f"{r['suite']} / {r['scale']}",e['footprint']['library_npy_bytes'],
          next(b['representation_bytes'] for b in blobs if Path(b['path']).name==f"r5q6_p_{r['suite']}_{r['scale']}_tail.pkl"),
          next(b['bytes'] for b in blobs if Path(b['path']).name==f"r5q6_p_{r['suite']}_{r['scale']}_tail.pkl")]
         for r,e in zip(unit['real'],evidence)]))
    report.append('''
Representation is **376 bytes/row**, excluding shared action payload and fixed PCA/calibration arrays.
Valid π0.5 full-chunk actions add 280 bytes/row. Actual fit pickle sizes above include padded action chunks and
auxiliary arrays. Stored library bytes count top-level NPY files (full-resolution keys included, token
subdirectories excluded). Owner-supplied deployed-pickle reference is **431 MB spatial / 1,103 MB l10**; those
reference figures were not remeasured here. Detailed fit kwargs, timestamps and thresholds: `results/fits.json`.

Exact prefit commands, already run (execute on a fresh destination; existing files are intentionally refused):

```bash
'''+(HERE/'prefit.sh').read_text().strip()+'\n```\n')
    report.append('''
## Final reproduction and coordinator handoff

From `/home/weiland/projects/openpi`, `bash exp/offline_search/rounds/r05/q6_wrist_blind/verify_final.sh`
reruns unit/parity, the eight Q6 plugin selftests, eight existing recipes, eight harness smokes, arm validation,
eight-connection concurrency and full-library/full-stream evidence. It contains the exact CPU/thread prefix;
per-command argv and return codes are in `results/final/*_commands.json`. Use fresh output paths on a repeat:
change the test groups' `--tag final` to a new tag and `/tmp/q6_concurrency_final` to another `/tmp/q6_*` path
(plugin logs append and the concurrency driver deliberately refuses existing directories). The command to audit/write
this report is:

```bash
'''+shlex.join(PREFIX+['-m','exp.offline_search.rounds.r05.q6_wrist_blind.write_handback'])+'\n```\n')
    report.append('''
Coordinator: copy the four `*_tail.pkl` artifacts into `<RUN>/fits/`, check the SHA256s above, replace `<RUN>`
in `arms_q6.json`, and emit using the installed `closed_loop.ops.emit_arms`. Start with the normal coordinator
short smoke on each suite/scale: verify startup `stage1_mode=wrist_only`, `miss_steps=10`, first/after-MISS
vision, no more than one consecutive blind decision, and `src=cache_blind` action equality to anchor steps5–9.
Then run the paired 500-init evaluation if the smoke passes. The phase2 file is optional and needs its own
matching prefits. Copying into the run store, model serving and rollouts were not performed by Q6.

The exact CPU integration smoke can be repeated before coordinator serving with
`run_checks plugin --tag coordinator_smoke` using the module and CPU prefix in `verify_final.sh`;
`results/final/plugin_commands.json` lists each complete per-arm command and all required wrist/blind flags.

Report SR, realized vision share v and MISS share m. Primary owner cost remains `.152*v + .848*m` per five
controls, with MISS charged as full inference. If reporting the K3 measured wrist-cost basis separately,
include base-camera completion on MISS; do not mix those eager measurements with owner constants. No new
hardware-cost measurement or IR/SR improvement is claimed here.

Decomposition: Q6 preserves K3 wrist selection/synthesis at fixed library; the method change is the visual
confirmation used by blind-gap stuck detection. Blind control completes an accepted chunk or advances
phase particles in the separately named variant. Library size and kref differ by the established 50/500
presets. These offline rates establish predicate behavior, not causal success or error-based arm selection.

## Owned files and read-only dependencies

`results/deliverables.json` records source/script/arm sizes, modification timestamps (installation times for
these newly created, previously unused paths), and SHA256. Shared dependency hashes before/after final
verification match in `results/dependencies_{before,after}_final.json`. No shared installation step was needed.
''')
    report.append(table(['Owned file','Modified UTC','SHA256'],
        [[Path(r['path']).name,r['modified_utc'],f"`{r['sha256']}`"] for r in manifest]))
    (HERE/'HANDBACK.md').write_text('\n\n'.join(report)+'\n')
    (HERE/'HANDBACK.sha256').write_text(sha(HERE/'HANDBACK.md')+'  HANDBACK.md\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
