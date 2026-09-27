"""Build the factual handback only after all final verification records exist and pass."""
from __future__ import annotations
import datetime, hashlib, json, pathlib
HERE=pathlib.Path(__file__).resolve().parent
OUT=HERE/'results'
read=lambda p:json.loads(p.read_text())
parity=[read(p) for p in sorted(OUT.glob('parity_*_*.json'))]
replays=[r for p in sorted(OUT.glob('replay_*_*.json')) for r in read(p)]
smokes=read(OUT/'smokes.json');fits=read(OUT/'fits.json')
contract=read(OUT/'contract.json');variants=read(OUT/'variant_parity.json')
geometry=read(OUT/'control_geometry.json');arms=read(OUT/'arms_validation.json')
plugins={p.parent.name:read(p) for p in sorted(OUT.glob('plugin_*/selftest_report.json'))}
legacy={p.parent.name:read(p) for p in sorted(OUT.glob('selftest_*/selftest_report.json'))}
assert len(parity)==8 and all(r['bit_equal'] for r in parity)
assert len(replays)==64 and max(r['absolute_delta'] for r in replays)<2e-6
assert len(smokes)==60 and all(r['returncode']==0 for r in smokes)
assert len(fits)==36 and len(plugins)==10 and all(r['PASS'] for r in plugins.values())
assert len(legacy)==6 and all(r['PASS'] for r in legacy.values())
assert contract['edge_case_checks']==86 and arms['PASS']
# The final analytic control-step implementation must have finished all 32 reruns.
final_jobs=[json.loads(line) for line in (OUT/'smokes_control_final.log').read_text().splitlines() if line.startswith('{')]
assert len(final_jobs)==32 and all(r['returncode']==0 for r in final_jobs)
measure=[]
for job in smokes:
    tag=job['tag'];method=job['command'][job['command'].index('--method')+1]
    paths=list((OUT/'smokes'/tag).glob('*/*_inf.json'))+list((OUT/'smokes'/tag).glob('*/*_cache.json'))
    assert len(paths)==1,(tag,paths)
    v=read(paths[0]);measure.append(dict(tag=tag,cell=v['cell'],method=method,decisions=v['n_decisions'],
       fit_s=v['fit']['fit_s'],ms_per_query=v['timing']['ms_per_query'],err=v['metrics']['err_mean'],
       aurc=v['metrics']['aurc'],grip_mis=v['metrics']['grip_mis']))
summary=dict(vision_parity_comparisons=sum(x['comparisons'] for x in parity),parity_variants=120,
             smoke_runs=len(smokes),smoke_decisions=sum(x['decisions'] for x in measure),
             replay_comparisons=sum(x['n'] for x in replays),replay_anchor_target_windows=sum(x['n'] for x in replays)//2,
             replay_max_abs_error=max(x['absolute_delta'] for x in replays),
             plugin_runs=len(plugins),plugin_decisions=sum(x['decisions'] for x in plugins.values()),
             plugin_blind=sum(x['blind'] for x in plugins.values()),plugin_miss=sum(x['miss'] for x in plugins.values()),
             plugin_stage1_calls=sum(x['stage1_calls'] for x in plugins.values()),
             legacy_decisions=sum(x['decisions'] for x in legacy.values()),contract=contract,variants=variants,
             arms=arms,smoke_metrics=measure,plugins=plugins,legacy=legacy)
(OUT/'verification_summary.json').write_text(json.dumps(summary,indent=2))
lines=['# K1 R4 handback',f'\nCompleted verification: {datetime.datetime.now(datetime.timezone.utc).isoformat()}.',
'\nAll implementation, tests, arm specs, and documentation are under `exp/offline_search/rounds/r04/k1_blind/`. No existing R1–R3, harness, src, profile, shared plugin, or shared operations file was edited. No GPU, server, LIBERO worker, network operation, git command, or closed-loop chain was run. The RAM store was absent; all real-store tests used `/home/weiland/trace_runs/offline_search_store`.',
'\n## Files and installation',
'\nMethods: `__init__.py`, `blind_awm.py`, `judge.py`, `control_step.py`, `wrist.py`. Each method was developed in `dev/` and copied to a sibling temporary file before one rename. The installed methods and development copies match.',
'\n```text\n'+(HERE/'install_times.txt').read_text().rstrip()+'\n```',
'\nHandoff: `arms_r4.json`, `arm_plan.json`, `batch.json`, `prefit.sh`, `README.md`, `INTEGRATION.md`, `HANDBACK.md`. `make_arms.py` and `validate_arms.py` regenerate/validate these specs. `checks.py`, `run_checks.py`, `test_contract.py`, `test_variants.py`, `test_control_geometry.py`, `run_smokes.py`, `run_existing_checks.sh`, `run_plugin_blind.sh`, `measure_fits.py`, and `make_handback.py` reproduce verification. Detailed logs, metric JSON/NPZ files, fit measurements and command lists are in `results/`. Large pickles are outside the repo at `/tmp/k1_blind_fits/`.',
'\n## Switches and behavior',
'\nModule prefix: `exp.offline_search.rounds.r04.k1_blind.`',
'\n| Method | Exact switches |\n|---|---|',
'| `blind_awm:BlindAWM` | `lib="current", kref=5` for 50, `lib="big", kref=8` for 500; `serving="phase_particles", "kernel_clock", "top1_clock", "anchor_tail"`; `budget=0..4`; `gates="all", "budget_only"`; `residual_threshold=.25, .5, 1` |',
'| `blind_awm:BlindAWM3` | Same blind switches plus the inherited AWM3 switches; ordinary query keeps AWM3 behavior |',
'| `judge:BlindMixedJudge` | `base_kwargs={...above...}`, `progress_guard="noprog_span"` or `"noprog_n"`, `memo_reset_after_miss=false or true`; ordinary MixedJudge kwargs including `noprog_n`, `guards`, `events`, `burst` pass through |',
'| `judge:MemoResetMixedJudge` | Defaults to `progress_guard="noprog_n", memo_reset_after_miss=true`; only progress comparisons reset after executed MISS |',
'| `control_step:ControlStepLibrary` | Same library kwargs; `ablation="G", "GS"`, `offsets=[0,1,2,3,4]` (coarse `[0,2,4]` optional); pure-cache only |',
'| `wrist:BlindWristAWM`, `wrist:BlindWristMixedJudge` | K3 wrist metric composed with the blind / gap guard adapter; the judge accepts `base_kwargs` |',
'\nEnable blind serving through K2 with `plugin_args: ["--os-blind", ...]`. Without that flag the plugin uses ordinary queries. The method captures every anchor row/weight in both `query()` and `os_synth()`, resets per episode, rejects first/after-MISS/invalid/task-change/discontinuous requests, and optionally supports immediate `invalidate_anchor()` feedback. Look codes 1–6 follow the brief; the span guard additionally requests code 8 (`noprog_span`). The unchanged `noprog_n` guard explicitly requests code 8 (`noprog_n_requires_vision`) on otherwise eligible blind requests because its adjacent-vision assumption is unchanged.',
'\nThe phase kernel keeps 16 members and fixed weights; each advances only within its own episode using offsets h−1/h/h+1, penalty .05, monotone phase and at most two rows per blind decision. Gates use nominal clock advance. Tail eligibility is one blind block for H=10 and two for H=16; deterministic wire padding never extends eligibility. State std (floor .05) and normalized motion percentiles come only from the deployed library.',
'\nIntentional differences are explicit: after a blind gap, AWM’s unavailable adjacent-camera `still` diagnostic is omitted; retrieval/action/confidence are unchanged. The span judge uses dense normalized proprioception and elapsed anchor intervals, so its guards/confidence can differ from stock MixedJudge; `dense_motion_guard=1` labels that change. It corrects the terminal-closed sign for GR00T; the legacy passthrough retains stock semantics. G/GS intentionally change stale ranking and (GS) action alignment; their library-LOEO confidence scales are recalculated, so fresh/step-zero confidence can change even though selection/actions remain exact AWM. Wrist parity is against K3 WristAWM because deleting a camera intentionally changes the metric. No threshold or phase-rule change from A §1.1–§1.3 was needed. Control-step projection uses C’s analytic chord formula and lower-offset rounding on ties.',
'\n## Final verification',
f'\n- **{summary["vision_parity_comparisons"]:,} vision comparisons passed bit for bit**, covering all 120 combinations of serving × B=0..4 × gates × residual threshold on both regimes, both models/suites and both library scales. Compared top-k, scores, full action, confidence and extras. `results/parity_*.json`.',
f'- **{variants["stock_guard"]} stock guard comparisons**, **{variants["memo_reset"]} memo-reset checks**, **{variants["gap_actions"]} span-guard action comparisons**, **{variants["awm3"]} AWM3 comparisons** (plain, ridge, gripper options), **{variants["wrist"]} wrist comparisons** passed. `results/variant_parity.json`.',
f'- **60/60 harness smokes passed**, {summary["smoke_decisions"]:,} decisions, two episodes each, with fresh-fit/reversed-episode determinism checks: BlindAWM and G/GS on all 16 cells/scales; span/stock/memo judges on π0.5 l10 inf/cache at both scales. All 32 G/GS smokes were repeated after final analytic-offset installation. Full err/AURC/gripper metrics and timing: `results/verification_summary.json`, `results/smokes.json`, `results/smokes/`.',
f'- **{contract["edge_case_checks"]} edge checks** and four fake-driver sequences passed (π0.5/GR00T × 50/500): vision → blind → blind → vision → MISS → vision, anchors, lifecycle reasons `[6,1,6]`, dense commits, fresh policy-tail action equality, no-progress spans, memo reset, pickle/reset, tail limits, phase bounds and splice/tie behavior. `results/contract.json`.',
f'- **{summary["replay_anchor_target_windows"]:,} store anchor/target windows**, **{summary["replay_comparisons"]:,} blind serving comparisons**, h=1/h=2 and phase/clock, passed against ideation A. Maximum mean-error difference **{summary["replay_max_abs_error"]:.3g}**, tolerance 2e-6. Saved full 16-member ideation anchors are the replay input; independent scalar query parity is tested separately. Their documented batched-anchor versus scalar maximum action reconstruction difference remains 0.000298366. Inf replay retains the ideation fixed-path hypothetical-HIT assumption; these are not counterfactual rollouts. `results/replay_*.json`.',
f'- **10/10 installed plugin blind selftests passed**, {summary["plugin_decisions"]} decisions, {summary["plugin_blind"]} blind, {summary["plugin_miss"]} MISS, {summary["plugin_stage1_calls"]} stage-1 calls, exactly {summary["plugin_decisions"]} broadcasts. These use real CPU orchestrators, interleaved connections, four episodes/run, duplicate rejection and exact log replay. Mixed integration uses ncal=64 for its fake-driver calibration; the smoke/fit tests use default ncal=3000. `results/plugin_*/selftest_report.json` and `verify_blind.json`.',
f'- **6/6 existing plugin selftests passed**, {summary["legacy_decisions"]} decisions: old ProbeB0, B=0 adapter, and every-decision MISS/inf replay on both models. Top-k/scores/confidence/action/extras equality 100%. **4/4 existing G3 contract checks passed** (both models × both libraries), 100% kernel/confidence/action agreement. `results/selftest_*/selftest_report.json`, `results/g3_*.log`.',
f'- **{sum(r["queries"] for r in geometry)} G/GS queries** checked: {sum(r["unchanged_branch"] for r in geometry)} step-zero/fresh queries retained AWM rows/scores/actions exactly; all {sum(r["stale"] for r in geometry)} sampled stale queries selected nonzero offsets and changed the splice. G/GS rows and scores were identical. This verifies the intervention, not SR. `results/control_geometry.json`.',
f'- **{arms["arms"]} arm specs / {arms["configs"]} CacheConfigs validated**, including plugin argument parsing, MISS K=2 YAML and L=10 client overrides. `results/arms_validation.json`.',
'\n### Replay numbers (cache l10; normalized RMS action error)',
'\n| Model | Library | h=1 phase / clock | h=2 phase / clock | h=2 phase−clock |\n|---|---:|---:|---:|---:|']
for model in ('pi05','groot'):
    for scale in (50,500):
        def val(serv,h):return next(r['error'] for r in replays if r['cell']==model+'_l10_cache' and r['scale']==scale and r['serving']==serv and r['h']==h)
        p1,c1,p2,c2=val('phase_particles',1),val('kernel_clock',1),val('phase_particles',2),val('kernel_clock',2)
        lines.append(f'| {model} | {scale} | {p1:.9f} / {c1:.9f} | {p2:.9f} / {c2:.9f} | {p2-c2:+.9f} |')
lines += ['\n## Measured fresh fits and deployed footprint',
'\nProtocol-4 plugin payloads, decimal MB. These are fresh fits, including serialization round trips; fit seconds exclude serialization and were measured while other assigned CPU tests ran. They are not throughput-under-load conclusions. Full detail and paths: `results/fits.json`. The same fitted arrays support serving/gate/budget variants; per-arm metadata must match exact kwargs. Compact retrieval bytes/entry: BlindAWM 586, span judge 632, G/GS 588, wrist blind 330, wrist span judge 376. Valid action payload is another 280 B/row for π0.5 or 448 B/row for GR00T; measured pickle bytes include padded chunks and all auxiliary arrays.',
'\n| Cell | Scale | Episodes / rows | Blind representation bytes | Blind pickle bytes (MB) | Fit s | Owner deployed pkl MB |\n|---|---:|---:|---:|---:|---:|---:|']
for r in sorted((r for r in fits if r['tag']=='blind'),key=lambda r:(r['key'],r['scale'])):
    dep={"pi05_spatial":431,"pi05_l10":1103,"groot_spatial":429,"groot_l10":1068}[r['key']]
    lines.append(f'| {r["key"]} | {r["scale"]} | {r["episodes"]} / {r["rows"]} | {int(r["rows"]*r["bytes_per_entry"]):,} | {r["pickle_bytes"]:,} ({r["pickle_bytes"]/1e6:.3f}) | {r["fit_s"]:.3f} | {dep} |')
lines+=['\n| Method | Cell | Scale | Pickle bytes (MB) | Fit s |\n|---|---|---:|---:|---:|']
for r in sorted((r for r in fits if r['tag']!='blind'),key=lambda r:(r['tag'],r['key'],r['scale'])):
    lines.append(f'| {r["tag"]} | {r["key"]} | {r["scale"]} | {r["pickle_bytes"]:,} ({r["pickle_bytes"]/1e6:.3f}) | {r["fit_s"]:.3f} |')
lines += ['\n## Exact reproduction commands',
'\nRun from `/home/weiland/projects/openpi`. All Python invocations (including scripts’ child processes) used the following prefix; scripts never exceeded eight Python processes in the assigned eight-CPU mask.',
'\n```bash\nPY=(taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python)\nK=exp.offline_search.rounds.r04.k1_blind\n"${PY[@]}" -m "$K.checks" parity --key pi05_l10 --scale 50\n"${PY[@]}" -m "$K.checks" replay --key pi05_l10 --scale 50\n"${PY[@]}" -m "$K.run_checks"\n"${PY[@]}" -m "$K.run_smokes"\n"${PY[@]}" -m "$K.measure_fits"\n# Final implementation checks after analytic chord installation:\n"${PY[@]}" -m "$K.measure_fits" --only-control\n"${PY[@]}" -m "$K.run_smokes" --only-control\n"${PY[@]}" -m "$K.test_contract"\n"${PY[@]}" -m "$K.test_variants"\n"${PY[@]}" -m "$K.test_control_geometry"\nbash exp/offline_search/rounds/r04/k1_blind/run_existing_checks.sh\nbash exp/offline_search/rounds/r04/k1_blind/run_plugin_blind.sh\n"${PY[@]}" -m "$K.make_arms"\n"${PY[@]}" -m "$K.validate_arms"\n"${PY[@]}" -m "$K.make_handback"\n```',
'\n`results/smokes.json` records every exact smoke command; the two shell scripts contain every exact existing/G3/plugin test command. All runs were CPU-only. `results/verification_summary.json` contains final counters and all per-smoke metrics. The final method files are SHA-256 listed in `results/source_manifest.json`.',
'\n## Coordinator next steps and caveats',
'\n1. `arms_r4.json` is an ordered candidate set: 88 batch-three rows and 28 batch-four rows (116 total), with π0.5 l10 first, then spatial; GR00T rows are pure-cache phase/tail controls. `arm_plan.json` labels matched controls and conditional combinations. It includes B=0, phase B=1/2, clock B=1, gated/ungated tail, periodic k=5 at 50 and k=8 at 500, and policy L=10 controls. Batch four includes B=1/2 plus MISS K2, then dummy-camera caching/prefix packing, then wrist-only/prefix packing, plus both G and GS l10 arms at both scales. This grid makes alternatives reviewable; it does not claim all candidates should be run or that a winner is known.',
'2. Replace every `<RUN>` in the JSON with the chosen coordinator run directory, then use the installed K4 emitter. Mixed rows set `full_model:true`. K2 stacks use `yaml_patch.miss.num_steps=2`, arm-local `miss.evidence_dir`, and `write_policy.type=never`. Redundancy flags are `--os-stage1-mode dummy_cached --os-pack-prefix`; wrist rows use `--os-stage1-mode wrist_only --os-pack-prefix --os-tokens off`. All use `--os-no-shadow-native`. L=10 rows use `replan_steps:10` and `--os-judge periodic:1`; normalize their IR per five controls.',
'3. Prefit exact arm artifacts with `RUN=/path/to/run bash exp/offline_search/rounds/r04/k1_blind/prefit.sh`. Each generated command uses the CPU prefix above and `--os-fit-artifact "$RUN/fits/<arm>.pkl"`. The actual measured representative artifacts are in `/tmp/k1_blind_fits`; they are useful for review, but cannot be renamed blindly into per-arm artifacts because the plugin validates spec/kwargs/cell metadata. `batch.json` provides deduplicated offline method/cell rows.',
'4. Only the coordinator should start full-model servers, run stratified collapse screens / the 500 paired episodes, push remote scripts, or make SR/IR decisions. No closed-loop SR, GPU transform parity, prefix-packing numerical parity, cheap-vision savings, or hardware latency claim is made by K1. K3’s validated stage implementation and K2’s real-model transform/bypass acceptance remain prerequisites for those combined arms. The CPU plugin tests validate serving/history mechanics with recorded normalized state and actions.',
'5. Interpretation: phase versus clock isolates continuation at fixed anchor synthesis/library; top1 removes kernel mixing; gated versus ungated tail changes look control. G versus AWM changes index geometry and GS versus G changes aligned synthesis. 50 versus 500 changes the actual deployed library. No fit here borrows larger-library data for the 50-episode arm. The small π0.5 spatial library has 49 episodes. Static MixedJudge cannot judge spliced heads consistently, so G/GS explicitly refuse G3 mixed wrapping. Full online success/failure and control-cost effects remain unmeasured by this agent.']
(HERE/'HANDBACK.md').write_text('\n'.join(lines)+'\n')
manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                      mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat())
          for p in HERE.iterdir() if p.is_file() and p.name!='HANDBACK.md'}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('smoke_metrics','plugins','legacy','contract','variants','arms')},indent=2))
