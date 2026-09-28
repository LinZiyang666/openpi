"""Render the hand-back from observed final evidence (no rollout inference)."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
s=json.loads((B/'results/final_summary.json').read_text())
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',*['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])
inst=table(['Shared file','Installed UTC','SHA256'],[(Path(r['path']).name,r['installed_utc'],f"`{r['sha256']}`") for r in s['installed']])
parity=table(['Mode','Bytes','JSONL rows'],[(r.get('mode',r.get('judge')),r['bytes'],r['rows']) for r in s['parity_off']+s['parity_pi05_tail']])
fit=table(['Artifact in /tmp/q2_fits/','Episodes / rows','Library top-level NPY bytes','CycleTail pickle bytes','Native source pickle bytes'],[(Path(r['path']).name,f"{r['episodes']} / {r['rows']}",r['library_top_level_npy_bytes'],r['bytes'],r['deployed_pkl_bytes']) for r in s['arms']['fits']])
fitsha=table(['Artifact','SHA256'],[(Path(r['path']).name,f"`{r['sha256']}`") for r in s['arms']['fits']])
prefit_commands='\n'.join(r['shell'] for r in json.loads((B/'prefit_commands.json').read_text()))
newcon=table(['Cell / scale / executed controls','Decisions per side','Vision','MISS','Policy tails','Exact parity'],[(r['config'],r['decisions'],r['vision'],r['miss'],r['policy_tails'],r['all_rows_actions_verdicts_histories_equal']) for r in s['new_concurrency']])
existing=[]
for group in ('k2','k1','k4'):
    rows=[r for r in s['existing']['rows'] if r['group']==group]
    existing.append((group.upper()+' original plugin matrix',len(rows),sum(r['decisions'] for r in rows),'PASS'))
checks=existing+[
('K5 real randomized replays',len(s['k5_audit']['replay_audits']),sum(r['decisions'] for r in s['k5_audit']['replay_audits']),'PASS'),
('K5 overlay',1,s['k5_overlay']['decisions'],'PASS'),
('K6 concurrency',len(s['k6_concurrency']),sum(r['decisions'] for r in s['k6_concurrency']),'PASS, plus serialized side'),
('K6 edges',1,s['k6_edges']['decisions'],'PASS'),
('K7 plugin arms',5,sum(r['decisions'] for r in s['k7_plugins']),'PASS'),
('K7 scalar comparisons',4,s['totals']['k7_scalar'],'PASS'),
('K7 full-cell count parity',4,s['totals']['k7_fullcell'],'PASS'),
('K10 plugin matrix',9,sum(r['decisions'] for r in s['k10_selftests']),'PASS'),
('K10 concurrency',9,sum(r['decisions'] for r in s['k10_concurrency']),'PASS, plus serialized side'),
('K10 original transform/lifecycle/ledger',len(s['k10_edges']['checks']),s['k10_edges']['decisions'],'PASS'),
('Q2 GR00T selftests',8,sum(r['decisions'] for r in s['new_selftests']),'PASS'),
('Q2 lifecycle/wire integration',8,sum(r['decisions'] for r in s['new_edges']),'PASS'),
('Q2 concurrency',8,sum(r['decisions'] for r in s['new_concurrency']),'PASS, plus serialized side'),
('Q2 invalid options/horizon contracts',s['new_contracts']['count'],'—','PASS'),
('Six arms / four exact prefits',6,4,'PASS')]
ct=table(['Check','Configurations/checks','Decisions/comparisons','Result'],checks)
text=f'''# Q2 hand-back: GR00T CycleTail with bounded chunk tails

Final verification UTC: {s['verified_utc']}.

Implemented and atomically installed the five owned shared files below. CycleTail, six arm specs, four exact prefits, CPU wire/lifecycle tests and eight-connection parity are complete. The full K10-installed regression recipe (K2/K1/K4/K5/K6/K7/K10) passed against the final shared files. No shared source changed after installation. No server, port, GPU workload, LIBERO worker, closed-loop chain, remote command, git command, subagent or review-test read was used. All Python work used repository `.venv/bin/python`, CPUs `30-33,74-77`, BLAS/OMP threads 1 and CUDA hidden. The store was read-only.

## Installed files and new deliverables

{inst}

`results/install.json` records preimage hashes, byte sizes and the five atomic rename times. Installation parsed each candidate, checked every shared preimage, wrote/fsynced a same-directory temporary file, used one `os.replace` per shared file (dependency `blind.py` first), and fsynced the directory. `before/` and `dev/` retain the preimages and installed candidates. K1, K7, K10 method files, `src/`, `harness/`, `profile/`, `ops/`, and other owners' files were not modified.

New deliverables under this directory: `judge.py`, `__init__.py`, `arms_q2.json`, `prefit_commands.json`, `prefit.sh`, `prepare_coordinator.py`, `make_arms.py`, `new_tests.py`, `cpu_transforms.py`, `contract_tests.py`, `concurrency_test.py`, `make_concurrency.py`, `tail_parity.py`, `run_final.py`, `validate_arms.py`, `summarize.py`, `write_handback.py`, `REPRODUCE.md`, preparation/install helpers, relocated original regression recipes in `regression/`, and evidence in `results/`. `results/deliverable_hashes.json` gives final SHA256 and UTC modification times of primary new source/spec/documentation files, including this hand-back; `results/file_inventory.json` covers all added files except itself and active report stdout logs. The unchanged original K10 assertions run from relocated copies; original round-4 files were not edited.

## Switches and semantics

Method: `exp.offline_search.rounds.r05.q2_groot.judge:CycleTail`.

G10 kwargs at 50 episodes: `{{"lib":"current","kref":5,"cycle_k":4,"tail_blocks":1}}`; at 500: `{{"lib":"big","kref":8,"cycle_k":4,"tail_blocks":1}}`.

Required serving switches: `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native`, with a full-model server and a five-control client. G15 uses method `tail_blocks=2` and `--os-policy-tail-blocks 2`. G15 is implemented/tested, but is not added to the requested six-arm pilot. `--os-policy-tail-blocks` accepts only 1/2, requires `--os-policy-tail`, and refuses two blocks for π0.5's H10. `--os-policy-tail` alone retains its one-tail default and the observed exact π0.5 log bytes. With flags absent all six observed legacy log modes remain byte-identical.

CycleTail subclasses K1 BlindAWM and fixes `serving="anchor_tail", gates="budget_only", budget=tail_blocks`. Real visual anchors are numbered from zero per internal episode: anchors 0, 4, 8, ... force a MISS, others HIT. Blind requests never advance that counter. Both sources execute their own original chunk: head 0–4, tail 5–9, and optionally 10–14. Offset 15 cannot provide a complete five-control response. After the configured tail count, real vision is required. K1 already implements both H16 cache-tail offsets; no K1 edit or new cache-anchor mechanism was needed.

There is no MixedJudge, progress, terminal or stuck guard in CycleTail. Blind rows retain NaN visual-key histories; no visual observation is fabricated. GR00T normalized gripper <0 means closed. The existing GR00T wire adapter converts openness to LIBERO positive-close exactly once. The method only adds a correctly signed previous-gripper diagnostic and never alters action signs. It uses K4's `seed_process()` hook on real queries, including after unpickling. This seeds the serving process; request ordering can still change stochastic model sampling.

The plugin retains copies of normalized and already transformed wire policy chunks with per-connection internal episode identity, original step and five-control cursor. Valid length is H minus that cursor. A second policy tail uses the original MISS chunk at offset 10, even though the preceding request was itself a HIT. It verifies the method's full normalized H×32 chunk against the exact original slice/padding, then returns the original wire slice/padding without reapplying transforms. Cache tails use the existing GR00T CPU adapter. All tail selection/capture/consumption, method state, histories and lifecycle calls remain under the K6 connection lock. No runtime lock surrounds method work or policy inference; the pre-existing GR00T model lock is unchanged.

Step 0, reset (even with repeated external UID), task change, stale episode snapshots, non-five execution audits, invalid state, missing/invalid method output, exhausted tails, burst/cap, or globally due periodic MISS require vision. Duplicate last request IDs are rejected before reservation/commit and preserve the cursor; an admitted failure invalidates it. GR00T mixed implicit task changes may reset the request ID to zero. K6's duplicate contract remains an immediate-last-ID check, not a replay cache of arbitrary historical IDs. A MISS at the last episode decision cannot leak its tail into the next episode.

An absent `__extra__.executed_steps` means the configured five-control protocol, as in K10. The server cannot infer unreported actual execution counts. Keep mixed arms at client L5; only the library-independent pure-policy controls use client L10. Terminal partial chunks end the episode and have no continuation.

## Logging and cost

Policy-tail rows have `src=source="policy_tail"`, `vision=false`, `hit=true`, null stage timings/`miss_k`, and no search/shadow. The normalized full chunk, original wire actions and audit fields are in `--os-log-inputs` NPZs. GR00T/explicit-block startup and NPZ metadata add `policy_tail_blocks`. CycleTail extras identify source anchor step, tail offset and remaining valid rows; policy-tail rows' library rows/weights are proposal provenance, not their action source. `verify_logs.py`, the fake selftest and wire replay verifier now resolve the original anchor across either tail length.

The owner cost is `(0.152*vision_decisions + 0.848*MISSes) / five_control_slots`, with every MISS fully priced (GR00T K8 is a full policy call, not 8/10 of a π0.5 call). Blind policy/cache tails cost zero. K4's existing explicit vision/hit ledger independently reports zero vision, zero MISS and zero total cost for every tested policy-tail-only set. K4's automatic GR00T measured stage table is a separate basis; do not label it as the owner .152/.848 basis. L10 pure-policy controls use two five-control slots per request. No closed-loop SR, realized rollout IR, GPU transform parity or model throughput claim is made here.

## Final verification

{ct}

Q2's eight lifecycle runs contain 128 checks, 420 decisions and 68 policy-tail ledger rows, all at zero charged tail cost. Its fake selftests contain 384 decisions, 224 blind decisions and 64 policy tails. The original K10 edge run contains 20 checks, 114 decisions, 42 policy tails and 400 exact L10 control comparisons. The original K10 method test contains 72 vision comparisons, 66 blind comparisons, six vetoes and 21 lifecycle checks. K10's concurrency matrix contains {sum(r['policy_tails'] for r in s['k10_concurrency'])} policy tails per side; its nine plugin selftests contain {sum(r['policy_tail'] for r in s['k10_selftests'])} policy tails. K7 has 13 synthetic edge checks and 150 blind-gap queries.

K7 edges and the K5 planted/null estimator, forged-log, incomplete/cross-scale, export and cost-ledger checks also passed through the original recipes. K10's inherited HIT-path/method lifecycle comparisons passed unchanged. K7's last two plugin arms reused its exact existing prefits after method/kwargs/cell validation (paths/hashes in `results/k7_fit_reuse.json`); independent replay fitting remained unchanged. The final structured counters are in `results/final_summary.json`; exact Q2 subprocess commands and return codes are in `results/final_commands.json`. `regression/run_installed.sh` and its constituent recipes give every legacy command. See `REPRODUCE.md` for the exact top-level commands and fresh-output requirements.

Byte comparisons freeze time/PID and use identical invocation/output paths, including startup records:

{parity}

Q2 eight-connection parity (each threaded run replayed serially in a fresh process using its observed reservation order):

{newcon}

Every compared non-timing decision row, returned action, verdict, dense history, mutable anchor and cursor matched. All new concurrency configurations reached eight overlapping fake stage-1 calls. These are fixed-input CPU correctness checks; timings are not a live GR00T throughput forecast.

Wire tests use 256 evenly spaced actual full inference chunks from each of `queries/groot_spatial_inf/a_inf.npy` and `queries/groot_l10_inf/a_inf.npy`: 512 unique stored chunks. Each compares exact action bytes against both L10 and L15 deque consumption. The eight suite/scale/tail tests repeat these same samples, totaling {s['totals']['actual_queue_controls_repeated']} control comparisons; repetitions are not independent evidence. They use actual GR00T inverse action components and `/data/ckpt/n15_libero_{{spatial,10}}/experiment_cfg/metadata.json`, plus the installed production output adapter. The production CPU state adapter's masked float32/bfloat16 casts and changing-mask rejection also pass. The installed K10 test separately retains actual π0.5 output-transform and state-dependent transform checks.

Initial full GR00T transform imports under repository Python failed first on missing pytorch3d, then on a transformers VideoInput incompatibility. The final test path loads the actual action inverse components via the installed GR00T dependency paths and omits its identity model inverse and unrelated image transforms; no VLM processor or model is constructed. Two new fixture errors (required empty video concat order and required 256×256 image shape) were corrected before successful final runs. Failed development logs are preserved. Full image-transform/model execution remains unverified here.

## Arms, exact prefits and footprint

`arms_q2.json` is emit_arms format with literal `<RUN>` placeholders: four G10 arms at {{spatial,l10}}×{{50,500}}, plus `r5q2_g_spatial_policy_L10` and `r5q2_g_l10_policy_L10`. All are full model, cost_ledger=true, full ordinary MISS schedule, resize_size=256. Mixed clients replan every five controls; pure controls every ten. Pure controls use `pure_inference=true` and server_seed=5101. Mixed CycleTail uses the existing per-server `--os-seed` and seeds through its query hook; the emitter restricts the separate server_seed field to pure_inference.

Exact coordinator prefit commands, including all kwargs/flags and CPU/environment prefixes, are in `prefit_commands.json`. The following actual commands generated the four final artifacts, which were subsequently loaded by the four final G10 plugin selftests:

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r05/q2_groot/prefit.sh
```

Expanded coordinator templates (replace literal `<RUN>` with the same run root):

```bash
{prefit_commands}
```

{fit}

Library NPY bytes above include top-level arrays (including raw full-resolution keys), excluding token subdirectories; recursive NPY byte totals and full native pickle paths are in `results/arms_validation.json`. CycleTail's compact representation is 586 bytes/entry, plus action/fixed arrays included in its actual pickle size. Each 50-library fit uses its own current library; no borrowed big-library fitting or outcome information is used.

{fitsha}

## Coordinator next step (unexecuted live smoke recipe)

Use the installed plugin on the next normal server start; existing live processes retain their imported code. Copy the four exact artifacts and emit all six specs with:

```bash
cd /home/weiland/projects/openpi
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/q2_groot/prepare_coordinator.py --run-root <RUN>
```

This preparation script validates exact method/kwargs/cell metadata, copies by temporary-file rename, and invokes emit_arms; it does not launch a server. Its local validation target was `/tmp/q2_coordinator_prepare`. Do not rename a BlindAWM or K10 pickle as CycleTail. If temporary fits are unavailable, expand each `prefit_commands.json` template with the intended `<RUN>` and fit again.

Coordinator only: run the ordinary full-model GR00T L10 controls and G10 arms on ten tasks × inits 0–9 in both suites and both library scales; retain five-control mixed clients and capture `--os-log-inputs` on a short instrumented subset. If using the existing pilot helper, explicitly set both `PILOT_TASKS=0,1,2,3,4,5,6,7,8,9` and `PILOT_EPISODES=0,1,2,3,4,5,6,7,8,9`: its defaults select five trap tasks. Use a separate pilot run root/name from the eventual 500-init run, since DONE markers otherwise skip the completed pilot arm. Confirm first-anchor MISS, subsequent offsets, actual MISS/vision counts and no stale response after reset; compare the L10 policy control before attributing an effect to cache control. Promote surviving G10 arms to the established 500 paired initializations. G15 is available for a later screen, not requested in this pilot spec. No smoke, rollout, remote sync, server or chain was run by Q2; live SR/IR and actual GR00T GPU behavior remain unverified.
'''
(B/'HANDBACK.md').write_text(text)
files=[]
for path in sorted(B.rglob('*')):
    if path.is_file() and path.suffix in ('.py','.sh','.md','.json') and not any(x in path.relative_to(B).parts for x in ('results','before','dev')):
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
(B/'results/deliverable_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
inventory=[]
for path in sorted(B.rglob('*')):
    if path.is_file() and path.name not in ('file_inventory.json','handback.log','summarize.log'):
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        inventory.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
(B/'results/file_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(B/'HANDBACK.md')
