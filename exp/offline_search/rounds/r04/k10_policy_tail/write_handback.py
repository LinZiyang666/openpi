"""Assemble handback only from completed installed verification artifacts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
B=Path(__file__).resolve().parent
R=B/'results';I=R/'installed'
def read(p):return json.loads(Path(p).read_text())
install=read(R/'install.json');existing=read(I/'existing_summary.json');parity=read(I/'parity.json')
concurrency=read('/tmp/k10_installed_concurrency/summary.json')
tails=read('/tmp/k10_installed_tail_concurrency/summary.json')
edges=read('/tmp/k10_installed_edges/report.json');matrix=read(I/'tail_matrix.json');methods=read(I/'method_test.json')
for f in install:
    assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
assert existing['checks']==35 and existing['decisions']==1599
assert len(concurrency)==12 and len(tails)==9 and len(matrix)==9
assert all(r['PASS'] for r in concurrency+tails+matrix)
assert edges['PASS'] and methods['PASS']
commands=read(I/'k7_commands.json');assert len(commands)==14 and all(r['returncode']==0 for r in commands)
k7plugin=[read(Path('/tmp/k10_installed/k7')/r['name']/'selftest_report.json') for r in read(B.parent/'k7_guard/arms_k7.json')]
k7unit=I/'k7_unit/results';k7edges=read(k7unit/'edges.json')
k7parity=[read(p) for p in sorted(k7unit.glob('parity_*.json'))]
k7rates=[read(p) for p in sorted(k7unit.glob('rates_*.json'))]
assert len(k7parity)==4 and len(k7rates)==4
for r in k7plugin+k7parity:assert r['PASS']
k5=read(I/'final_audit.json');assert k5['PASS']
k6edges=read('/tmp/k10_installed_k6_edges/edge.json');assert k6edges['PASS']
assert read(I/'overlay_installed.json')['PASS']
assert read(I/'estimator_validation.json')['PASS']
summary=dict(existing=existing,parity=parity,k6_concurrency=concurrency,k10_concurrency=tails,k10_edges=edges,k10_matrix=matrix,k10_method=methods,k7_plugin=k7plugin,k7_edges=k7edges,k7_parity=k7parity,k7_rates=k7rates,k5_audit=k5,k6_edges=k6edges)
(R/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for src,dst in [('/tmp/k10_installed_concurrency/summary.json','concurrency_summary.json'),('/tmp/k10_installed_tail_concurrency/summary.json','tail_concurrency_summary.json'),('/tmp/k10_installed_edges/report.json','policy_tail_edges.json')]:
    (I/dst).write_bytes(Path(src).read_bytes())
lines=['# K10 hand-back: opt-in policy tail','',f'Completed UTC: {datetime.now(timezone.utc).isoformat()}.', '',
'Implemented and atomically installed the plugin path, K7 subclass, fake-policy/selftest and replay verification support. No K1/K7, src, harness, profile, ops, cost-table or other owner files were edited. No GPU, server, port, LIBERO worker, chain, remote host, git command or review_tests content was used. Python ran on CPUs 26–29,70–73, CUDA disabled, OMP/OpenBLAS/MKL threads 1. No agent delegation was used.','',
'## Design and contract','',
'`--os-policy-tail` requires `--os-blind` and π0.5. Without the new switch, no policy response is retained for tail reuse and legacy logs keep their exact schema and values (fixed-clock/PID comparisons below). Existing methods are unaffected. The optional method hook is `policy_tail_step(BlindQueryView) -> BlindResult | LookReason`; an absent hook requests vision. It receives the actual previous MISS via `prev_hit=False`, `prev_a_exec`, and dense histories, with no visual fields.','',
'The plugin saves copies of both the normalized policy chunk and the original returned wire actions, tagged with the actual per-connection EpisodeView identity and decision step. Only the immediately following decision can consume them. Step 0, reset/task change, a snapshot from another episode, invalid current state, non-five execution audit, a second blind after a policy tail, judge burst/cap or globally due periodic MISS all require vision. A rejected duplicate reserves no index and preserves the saved response; an admitted failure clears it. The same per-connection K6 lock covers selection, response capture, commit and lifecycle. No runtime lock was added around method or policy work.','',
'For H=10, serve normalized `policy_chunk[5:10]` followed by five repeats of its last row, preserving every one of the 32 columns and float32 dtype. The wire chunk separately uses `previous_response.actions[5:10]` followed by five repeats of its final wire row, preserving its dtype and columns. The plugin enforces exact normalized equality against the saved MISS regardless of hook output. No output transform is reapplied to the policy tail. This preserves the same controls as the L=10 client queue even if transforms depend on the old observation state. Only five controls of the tail response may execute. For H>10 the helper retains all remaining rows but still allows only one five-control blind decision before vision.','',
'`PolicyTailJudge` subclasses K7 `VisionConfirmedBlindMixedJudge`. HIT `blind_step` is inherited unchanged. It retains the last vision proposal separately for gate evaluation while the ordinary K1 anchor is still invalidated on MISS. The policy hook temporarily passes that proposal and a `prev_hit=True` gate facade through the inherited budget, span guard and base gates, then restores the invalid anchor. Actual `hist_hit`, action history, vision mask and guard memos are never changed. The candidate library rows are gate provenance only; the returned action is the policy tail. Supplied arms use anchor_tail, budget 1, budget_only. Budget 0 is available for tests; budgets >1 and non-tail serving are rejected. K7 stuck confirmation and no-progress span consume the policy-tail row as a blind gap; visual keys remain NaN.','',
'### Execution-count contract','',
'The existing LIBERO client sends no per-request execution count and requests again when its action queue empties. As in the existing blind protocol, absent `__extra__.executed_steps` means the configured five-control client contract. A supplied value other than 5 requests vision; terminal partial blocks end the episode and cannot supply a tail in the next episode. K10 rows expose `previous_executed_steps` and `execution_audit` so a protocol assumption is distinguishable from an explicit audit. The supplied arms retain L=5. Do not use the switch with an L=10 or otherwise non-five client that omits this audit: the server cannot infer unreported physical execution.','',
'### Logs and costs','',
'Policy-tail rows are `src=source="policy_tail"`, `vision=false`, `hit=true`, `judge="blind"`, `look_reason=null`, `blind_age=0` (consecutive blind decisions BEFORE this request), `miss_k=null`, both stage timings null, searched/shadow_available false, and `exec_ok=true`. The next vision row has blind_age 1. `hit=true` means a served response requiring no new policy inference, not a library retrieval. Wire hit_type is FULL_HIT. Normalized served_head is exact, and NPZ records retain full normalized and wire chunks, source and audit fields. Policy-tail extras identify `policy_tail=1` and the source decision.','',
'The existing K4 ledger reads the explicit vision and hit flags: a tail costs zero and adds neither a vision nor a MISS. Member rows/weights/lib describe the last vision proposal used for gates, consistent with proposal diagnostics on MISSes; they must not be interpreted as the action source. Existing KPI action metrics use served_head correctly; library spell/terminal diagnostics on these rows remain proposal-based. No cost-table or ops change was needed.','',
'## Atomic installation','',
'Every shared file was developed in dev/, preimage-checked, parsed, written/fsynced to a same-directory temporary file and installed with one `os.replace` (atomic rename/mv), then directory-fsynced. Dependency order was blind.py before plugin.py. No shared edits followed these installs.','',
'| File | Installed UTC | SHA256 |','|---|---|---|']
for r in install:lines.append(f"| `{Path(r['path']).relative_to(B.parents[4])}` | {r['installed_utc']} | `{r['sha256']}` |")
lines += ['', 'Primary new deliverables (UTC file availability / final modification time):', '',
          '| File | UTC | SHA256 |', '|---|---|---|']
for name in ('judge.py','__init__.py','arms_k10.json','prefit.sh','prefit_commands.json'):
    path=B/name
    lines.append(f"| `{name}` | {datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")
lines+=['','`results/install.json` also records preimage hashes and sizes. `results/deliverable_hashes.json` records the added top-level source/spec/report deliverables and their UTC modification times/hashes (excluding the checksum sidecar). `before/` contains the captured shared preimages; `dev/` the installed candidates and adapted recipes. Primary new files are judge.py, arms_k10.json, prefit.sh, prefit_commands.json, policy_tail_test.py, method_test.py, tail_concurrency_test.py, run_tail_matrix.py, validation/installation runners and this handback.','',
'## Installed verification results','',
'All reported final checks used the installed shared files. Compact evidence is in results/installed/ and results/final_summary.json; full logs/NPZs are in the /tmp paths listed below.','',
'| Check | Runs / checks | Decisions / comparisons | Result |','|---|---:|---:|---|']
for group in ('k2','k1','k4'):
    rr=[r for r in existing['rows'] if r['group']==group];lines.append(f"| {group.upper()} original plugin matrix | {len(rr)} | {sum(r['decisions'] for r in rr)} decisions | PASS |")
lines+= [f"| K5 four real MixedJudge randomized replays | 4 | {sum(r['decisions'] for r in k5['replay_audits'])} decisions | PASS |",
'| K5 injected overlay lifecycle/history | 1 | 104 decisions | PASS |',
f"| K6 eight-connection serialized/concurrent parity | {len(concurrency)} configurations | {sum(r['decisions'] for r in concurrency)} threaded + same serialized | PASS |",
f"| K7 existing plugin arms | {len(k7plugin)} | {sum(r['decisions'] for r in k7plugin)} decisions | PASS |",
f"| K7 scalar parity | 4 cell/scales | {sum(r['decisions'] for p in k7parity for r in p['rows'])} complete query comparisons | PASS |",
f"| K7 full-cell count parity | 4 cell/scales | {sum(r['decisions'] for p in k7rates for r in p if r['budget']==0)} decisions | PASS |",
f"| K7 edges | {len(k7edges['synthetic_checks'])} synthetic checks | {k7edges['gap_queries']} gap queries | PASS |",
f"| K6 edges | {k6edges['connections']} connections | {k6edges['decisions']} decisions, {k6edges['reserved']} reservations, {k6edges['quantile_transactions']} quantile transactions | PASS |",
f"| K10 existing blind driver, three cells × three judges | {len(matrix)} | {sum(r['decisions'] for r in matrix)} decisions, {sum(r['policy_tail'] for r in matrix)} policy tails | PASS |",
f"| K10 eight-connection serialized/concurrent parity | {len(tails)} configurations | {sum(r['decisions'] for r in tails)} threaded + same serialized; {sum(r['policy_tails'] for r in tails)} tails per side | PASS |",
f"| K10 transform/L10/lifecycle/guard/ledger | {len(edges['checks'])} checks | {edges['decisions']} decisions, {edges['policy_tails']} tails, {edges['l10_controls']} L10 controls | PASS |",
f"| K10 vs K7 HIT path | 3 cell/scales | {sum(r['vision'] for r in methods['rows'])} exact query results, {sum(r['blind'] for r in methods['rows'])} exact blind results, {sum(r['veto'] for r in methods['rows'])} identical vetoes | PASS |",
'| Arm emitter and parser | 3 | exact kwargs / full model / cost ledger / L5 | PASS |','',
'K6 edge_test reran the original deterministic race/failure/partial-write/quantile/clone checks; see results/installed/k6_edges.log and /tmp/k10_installed_k6_edges. K7 original edges, scalar parity and full-cell rate checks reran through k7_unit.py with only output paths relocated; their complete original counters are in results/installed/k7_unit/results/ and copied in final_summary.json. K5 estimator validation reran 200 planted + 200 null datasets, 500 init clusters each, 499 bootstrap draws; exact effects, forged logs, incomplete/cross-scale pairs, raw/collect/KPI/export consistency, historical layouts and four-arm cost ledger passed. These are regression checks, not rollout outcomes.','',
'Flag-off JSONL comparisons use identical invocation/output paths, frozen clock and PID, against before/; all bytes, including startup rows, were equal:','',
'| Mode | Bytes | Rows |','|---|---:|---:|']
for r in parity:lines.append(f"| {r['mode']} | {r['bytes']:,} | {r['rows']} |")
lines+=['',
f"The new edge test recorded {edges['tail_ledger']['decisions']} policy-tail decisions with ledger vision_decisions={edges['tail_ledger']['vision_decisions']}, misses={edges['tail_ledger']['misses']}, total_cost={edges['tail_ledger']['total_cost']} and IR per five controls={edges['tail_ledger']['ir_per_five_controls']}. It used the installed pi05 output transform stack and normalization assets at `{edges['checkpoint']}`. The client comparison extended/popped action queues exactly as examples/libero/main.py does; a separate state-dependent AbsoluteActions case demonstrated why retaining the original wire response matters.",
'', 'Development findings: the first transform fixture tried the checkpoint-local assets path (absent); it was corrected to the existing repository normalization-asset fallback. The all-MISS duplicate test exposed loss of a cached tail on duplicate rejection; this was fixed before installation by invalidating only admitted requests. A method-test fixture initially used dataclasses.replace on EpisodeView, which is not a dataclass; corrected to an explicit test identity. The initial transform and duplicate failures are retained in results/dev/. Final results above passed.','',
'Guard-only concurrency replays served zero policy tails at all three cell/scales: guards vetoed reuse after their 7, 4 and 10 MISSes respectively. Forced-MISS and periodic configurations exercised 347 tails per threaded/serialized side. This is an implementation/gate check, not evidence that the feature improves rollout SR or IR; realized policy-tail share must be measured by the coordinator.', '', '## Arms and exact coordinator prefits','',
'`arms_k10.json` is emit_arms format with literal `<RUN>` placeholders. Exact base kwargs match K7 tail1ug: l10 500 and spatial 500 use lib=big/kref=8; l10 50 uses lib=current/kref=5. All use anchor_tail/budget=1/gates=budget_only, noprog_span, events=none, vision_confirmed. All are full-model, cost_ledger=true, with `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native`. MISS remains the ordinary K=10 policy path; no YAML miss patch. K10 deployment prefits were not run here and are intentionally left to the coordinator. Do not relabel K7 pickles as K10: the artifact validates exact method/kwargs/cell metadata.','',
'Run `RUN=/absolute/run/root bash exp/offline_search/rounds/r04/k10_policy_tail/prefit.sh` from the repository. The exact expanded template command for each arm (replace literal `<RUN>` with the same run root) is:','', '```bash']
lines += [shlex.join(c) for c in read(B/'prefit_commands.json')]
lines+=['```','',
'Emitter/parser validation wrote /tmp/k10_emitted and results/arms_validation.json. It did not launch servers or run prefits. Test ncal=64 configurations are verification-only; supplied arms retain K7 default calibration kwargs. The separate HIT parity test reads existing K7 fits as references, never as deployment K10 artifacts. No new learned representation or borrowed cross-scale controller information is introduced.','',
'## Exact verification commands','',
'Commands ran from /home/weiland/projects/openpi. The shell recipes preserve original owner test logic, changing affinity/loaders and owned output paths. Their expanded command records are retained alongside reports.','',
'```bash',
'P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)',
'B=exp/offline_search/rounds/r04/k10_policy_tail',
'"${P[@]}" "$B/install.py"',
'bash "$B/run_installed.sh" > "$B/results/installed_run.log" 2>&1',
'"${P[@]}" "$B/method_test.py" > "$B/results/installed/method_test.log" 2>&1',
'"${P[@]}" "$B/validate_arms.py" > "$B/results/arms_validation.log" 2>&1',
'"${P[@]}" "$B/write_handback.py"',
'```','',
'For a fresh rerun, choose fresh /tmp outputs in the recipes; blind logs append and concurrency/edge drivers refuse existing outputs. Do not rerun install.py just to repeat tests: it deliberately requires the original preimages. Full test commands are in dev/installed_{k2,k1,k4}.sh, run_installed.sh, results/installed/replay_commands_installed.json, k7_commands.json and tail_matrix.json.','',
'## Coordinator next steps and limits','',
'1. Run the three exact prefits, resolve <RUN> in arms_k10.json and emit the arms with the existing emitter. Keep the client at five controls/request and the full model loaded. Use K=10 MISS, no miss YAML patch.','2. Let current arms finish. New servers import the installed plugin; already running servers retain their imported code. Any separate checkout needs all five shared files and the K10 method module. The plugin never restarts running services.','3. Run the coordinator-owned live smoke, then paired evaluation at both library scales and spatial. Audit src=policy_tail, non-vision/non-MISS counts, actual client L=5, and the ledger. No live GPU, simulator, closed-loop SR/IR, stochastic rollout equality or hardware throughput claim is made here.','4. L10 equality is verified for supplied normalized policy chunks and the installed CPU output transforms, not for newly sampled GPU inference or simulator dynamics. Endpoint-confirmed K7 guards retain their documented blind-gap limitations. A method may request vision after a MISS; policy-tail counts therefore need not equal MISS counts.','']
(B/'HANDBACK.md').write_text('\n'.join(lines))
manifest=[]
for p in sorted(B.iterdir()):
    if p.is_file() and p.name not in ('HANDBACK.sha256',):
        manifest.append(dict(path=str(p),modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(R/'deliverable_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
(B/'HANDBACK.sha256').write_text(hashlib.sha256((B/'HANDBACK.md').read_bytes()).hexdigest()+'  HANDBACK.md\n')
print(json.dumps(dict(PASS=True,existing=existing['decisions'],k6=sum(r['decisions'] for r in concurrency),k10=sum(r['decisions'] for r in matrix+tails)+edges['decisions'],handback=str(B/'HANDBACK.md'))))
