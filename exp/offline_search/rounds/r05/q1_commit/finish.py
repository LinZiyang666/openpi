"""Audit final evidence and write the coordinator hand-back; no experiments launched."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np

from prepare import HERE, PREFIX, write

R = HERE / 'results'
REG = HERE / 'regression/results/installed'
SHARED = HERE.parents[2] / 'closed_loop'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def info(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path),
                available_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())


def main():
    shared = []
    for name in ('blind.py', 'plugin.py', 'selftest.py', 'verify_logs.py', 'replay_client.py'):
        path = SHARED / name
        assert path.read_bytes() == (HERE / 'regression/dev' / name).read_bytes(), f'{name} changed during verification'
        shared.append(info(path))
    paths = dict(methods=R/'methods.json', delivery=R/'delivery.json',
                 matrix=Path('/tmp/q1_final_matrix/summary.json'),
                 concurrency=Path('/tmp/q1_final_concurrency/summary.json'),
                 monitor=Path('/tmp/q1_final_monitor/summary.json'),
                 edges=Path('/tmp/q1_edges_final/report.json'),
                 grasp_plugin=Path('/tmp/q1_grasp_plugin_final/report.json'),
                 existing=REG/'existing_summary.json', parity=REG/'parity.json',
                 k5_overlay=REG/'overlay_installed.json', k5_audit=REG/'final_audit.json',
                 k5_estimator=REG/'estimator_validation.json', k5_ledger=REG/'ledger_check.json',
                 k6_concurrency=Path('/tmp/q1_reg_installed_concurrency/summary.json'),
                 k6_edges=Path('/tmp/q1_reg_installed_k6_edges/edge.json'),
                 k7_commands=REG/'k7_commands.json', k10_matrix=REG/'tail_matrix.json',
                 k10_method=REG/'method_test.json',
                 k10_concurrency=Path('/tmp/q1_reg_installed_tail_concurrency/summary.json'),
                 k10_edges=Path('/tmp/q1_reg_installed_edges/report.json'))
    results = {k: read(p) for k, p in paths.items()}
    assert all(results[k]['PASS'] for k in ('methods', 'delivery', 'edges', 'grasp_plugin', 'k6_edges', 'k10_edges', 'k10_method'))
    for key, count in [('matrix', 14), ('concurrency', 6), ('monitor', 2), ('k6_concurrency', 12),
                       ('k10_matrix', 9), ('k10_concurrency', 9)]:
        assert len(results[key]) == count and all(x['PASS'] for x in results[key]), key
    assert len(results['k7_commands']) == 14 and all(x['returncode'] == 0 for x in results['k7_commands'])
    assert len(results['parity']) == 6 and all(r['byte_identical'] for r in results['parity'])
    assert all(r['PASS'] for r in results['existing']['rows'])
    assert sum(x.get('inherited_exact_all_npz_fields', False) for x in results['matrix']) == 4
    byte_audit = []
    for row in results['matrix']:
        if not row['tag'].endswith('_inherited'):
            continue
        root = Path('/tmp/q1_final_matrix')
        a = sorted((root/row['tag']/'inputs').glob('*.npz'))
        b = sorted((root/row['tag'].replace('_inherited', '_k10')/'inputs').glob('*.npz'))
        assert len(a) == len(b) == 4
        fields = 0
        for x, y in zip(a, b):
            with np.load(x, allow_pickle=False) as zx, np.load(y, allow_pickle=False) as zy:
                keys = [k for k in zx.files if k != 'meta' and not k.endswith(('_ms', '_us'))]
                for key in keys:
                    va, vb = zx[key], zy[key]
                    assert va.dtype == vb.dtype and va.shape == vb.shape and va.tobytes() == vb.tobytes(), (row['tag'], key)
                    fields += 1
        byte_audit.append(dict(arm=row['tag'], episodes=4, array_fields=fields, all_bytes_equal=True))
    results['inherited_npz_byte_audit'] = byte_audit
    for item in results['delivery']['fits']:
        assert digest(Path(item['path'])) == item['sha256'], item['path']
    for item in results['monitor']:
        item['artifact'] = info(Path(item['fit']))
    results['k7_unit'] = {p.name: read(p) for p in sorted((REG/'k7_unit/results').glob('*.json'))}
    for key, source in paths.items():
        if source.parent != R:
            write(R / (key + '.json'), json.dumps(results[key], indent=2) + '\n')
    write(R/'final_summary.json', json.dumps(dict(PASS=True, shared_imports=shared, evidence_paths={
        k: str(v) for k, v in paths.items()}, results=results), indent=2) + '\n')

    methods, delivery = results['methods'], results['delivery']
    matrix, concurrency = results['matrix'], results['concurrency']
    deployed = [r for r in matrix if r['tag'].endswith('_deployed')]
    cells = methods['parity_lifecycle']
    groups = {}
    for family in ('k2', 'k1', 'k4'):
        rows = [r for r in results['existing']['rows'] if r['group'] == family]
        groups[family] = (len(rows), sum(r['decisions'] for r in rows))
    k7 = results['k7_unit']
    counts7 = dict(parity=sum(sum(r['decisions'] for r in v['rows']) for k,v in k7.items() if k.startswith('parity_')),
                   rates=sum(sum(r['decisions'] for r in v if r['budget'] == 0) for k,v in k7.items() if k.startswith('rates_')))
    now = datetime.now(timezone.utc).isoformat()
    primary = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ('HANDBACK.md', 'HANDBACK.sha256')]
    source_rows = '\n'.join(f"| `{p.name}` | {info(p)['available_utc']} | `{digest(p)}` |" for p in primary)
    fit_rows = '\n'.join(f"| `{r['arm']}` | {r['bytes']:,} | `{r['sha256']}` |" for r in delivery['fits'])
    library_rows = '\n'.join(f"| `{r['arm']}` | {r['library_rows']:,} | {r['library_disk_bytes']:,} | {r['bytes']:,} | {r['owner_deployed_pkl_MB']} |" for r in delivery['fits'])
    arm_rows = '\n'.join(f"| `{r['tag'].removesuffix('_deployed')}` | {r['decisions']} | {r['vision']} | {r['miss']} | {r.get('policy_tail',0)} | {r['eligible_misses'] if 'c10' in r['tag'] else 'n/a'} |" for r in deployed)
    concurrent_rows = '\n'.join(f"| `{r['config']}` | {r['decisions']} | {r['vision']} | {r['miss']} | {r['policy_tails']} | {r['tail_eligible'] if 'c10' in r['config'] else 'n/a'} |" for r in concurrency)
    commands = read(HERE/'prefit_commands.json')
    exact = '\n'.join(shlex.join(c) for c in commands['executed'])
    monitor_rows = '\n'.join(f"| {r['suite']} | {r['scale']} | {r['q99']:.15g} |" for r in methods['monitor'])
    text = f'''# Q1 hand-back: C10 plan completion and D1 grasp inspection

Final audit UTC: {now}. All numbers below come from completed CPU checks on the final files. No server, port,
GPU, simulator, LIBERO worker, chain, remote host, git command, review_tests read or subagent was used.
Only `exp/offline_search/rounds/r05/q1_commit/**` and `/tmp/q1_*` were written. All Python commands used CPUs
26–29,70–73, OMP/OpenBLAS/MKL threads 1, CUDA disabled, and bytecode writes disabled.

## Implementation and switches

`exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge` subclasses K10 `PolicyTailJudge`.
`policy_tail_gate="inherited"` (default), `monitor="off"` delegates the policy-tail hook directly to K10;
ordinary query and HIT blind handling are inherited. `policy_tail_gate="lifecycle"` consumes the separate
one-use vision anchor, requires the immediately preceding real-vision MISS in the same episode/task, dense
history and finite state/chunk, and returns the original policy chunk shifted by five. It bypasses rejected-cache
budget, span and base gates. It does not call the inherited blind gate, fabricate HIT histories, or clear progress
memos. K7's vision-confirmed stuck and span guards still run at the next real anchor; the tail remains a blind gap.
Tail rows/weights retain the last cache proposal as diagnostic provenance; `src=policy_tail` and the saved policy
chunk identify the executed action source.

Enable C10 with `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native` and the lifecycle kwarg.
The plugin enforces execution count, original normalized/wire-chunk equality, identity, repeated last decision IDs and one
complete tail. An absent `executed_steps` audit follows the existing five-control client contract; actual unreported
physical execution cannot be inferred. Explicit 0/4/6/10 controls reject reuse. Terminal MISSes cannot cross reset.

Optional `monitor="loeo_xyz99"` requires lifecycle mode. Each deployed library independently fits the ideation-A
16-feature xyz ridge, episode-held-out p99, and final float32 B/scales (208 parameter bytes). It predicts from the
actual policy head and prior state, never the rejected cache action. Residual above p99 requests LookReason 5,
without independently forcing a policy call. Defaults and all six requested arms keep the monitor off.

| Suite | Library episodes | Own-library LOEO p99 |
|---|---:|---:|
{monitor_rows}

`exp.offline_search.rounds.r05.q1_commit.judge:GraspCheckJudge` uses K7 anchor_tail / budget 1 / budget_only.
On an otherwise eligible blind HIT it applies D's command-matched aperture rule with exact strict/inclusive
thresholds: two closed executed heads; observed normalized aperture <.05; matched original mass >=.5; matched
successor mean >.25 and mass above .25 >=.75. Quantiles and compact tables use only the deployed library.
An alarm returns `LookReason(9,"grasp_aperture_contradiction")`; the associated vision query sets
`os_force_miss=1`, `os_reason=9`, `grasp_check=1`, and retains the original guard reason as `grasp_base_reason`.
The allowance is marked consumed only when a later dense history confirms this issued query executed as a
real-vision MISS. Repeated or abandoned proposals do not consume it. Episode/task reset clears pending and consumed
state. The existing per-connection lock protects these fields; fitted arrays are treated as read-only and shared by cloning.
D1 clearly refuses GR00T before fitting. Enable with `--os-blind --os-judge guard_only --os-no-shadow-native`,
without `--os-policy-tail`.

## Final verification

Compact evidence: [results/final_summary.json](results/final_summary.json). Full logs/NPZs are at its exact
`evidence_paths` and in `/tmp/q1_final_matrix`, `/tmp/q1_final_concurrency`, `/tmp/q1_final_monitor`,
`/tmp/q1_edges_final`, `/tmp/q1_grasp_plugin_final`, and `/tmp/q1_reg_*`.

- Method streams: {sum(r['vision'] for r in cells)} bit-exact K10/inherited vision results,
  {sum(r['hit_blind'] for r in cells)} ordinary blind-result comparisons and
  {sum(r['inherited_policy'] for r in cells)} inherited policy-hook comparisons, over both suites/scales.
  Lifecycle served **{sum(r['served'] for r in cells)}/{sum(r['eligible'] for r in cells)}** eligible MISS tails,
  with all 32 columns of `policy_chunk[5:10]` byte-exact even with positive no-progress span, zero cache budget and
  active base gates. {sum(r['lifecycle_veto'] for r in cells)} lifecycle rejection/reset cases passed.
- Real fitted next-anchor guard check: `stuck_n=2`, `noprog_span=2`, flags `9`, `os_force_miss=1`; blind keys stayed
  NaN and actual history stayed `[MISS,HIT]`. Existing K7 parity/edge/rate checks below also passed.
- D1 historical helper and method eligible-blind replay: **223/223** alarms at 50 and **47/47** at 500, zero
  disagreements across 31,186 / 28,888 recorded decisions. There are 12,620 / 12,341 blind rows (12,120 / 11,841
  with step>=2). With execution-confirmed once-per-episode caps: **73 / 24** interventions. Repeated vision
  proposals were checked 146 / 48 times. Historical JSONL lacks full camera keys, so the pending-MISS unit test
  mocks new vision retrieval. The capped unit replay injects a committed vision/MISS history marker at the first
  alarm to test allowance consumption while retaining recorded heads/states; it does not measure a new policy
  execution. Real library tables, recorded heads/states/rows/weights and inherited K7 blind selection determine
  the alarm. The plugin fixture separately commits actual fake-policy MISS responses. These are fixed-history
  replays and synthetic execution checks, not counterfactual rollouts.
- New installed plugin matrix: **14 runs / {sum(r['decisions'] for r in matrix)} decisions**, all passed with
  guard_only; four inherited-vs-K10 pairs match every non-timing NPZ field exactly. Each run rejects four duplicates.
- C10 transformed-action/lifecycle integration: **{results['edges']['decisions']} decisions,
  {results['edges']['policy_tails']} tails**, 21 checks, eight connections, **400** real-transform L10-equivalent
  controls byte-exact. Includes partial execution, step 0, task/reset, final MISS, duplicate preservation,
  wrong-action substitution, finite state, burst/periodic precedence and state-dependent output transforms.
  Its {results['edges']['policy_tails']} tail ledger rows have vision=0, MISS=0, cost=0, IR5=0.
- Six-arm eight-connection replay: **{sum(r['decisions'] for r in concurrency)} threaded + the same serialized
  decisions**, exact actions/verdicts/history/log fields under the recorded reservation schedule. Every eligible C10
  MISS served its exact tail. All configurations reached eight simultaneous fake-policy calls; this is a correctness
  stress test, not a hardware throughput claim.
- D1 synthetic-contact plugin stress: **{results['grasp_plugin']['decisions']} decisions / 8 connections / 24
  episodes**, exactly 16 interventions in 16 eligible episodes, 16 duplicate rejections, no second intervention,
  and resets after a final forced MISS. Synthetic successor apertures and current aperture deliberately guarantee
  exposure after actual closed heads; the historical test above separately validates real-table alarm agreement.
- Optional-monitor own-library calibration matched ideation A at all four cells/scales. Two actual monitor prefits
  and installed plugin runs at l10 50/500 passed ({sum(r['decisions'] for r in results['monitor'])} decisions), plus
  controlled zero-residual and over-threshold policy-head tests. Extra verification fits are named `_monitor_final.pkl`
  in `/tmp/q1_fits`; they are not requested arms.
- Existing K2: {groups['k2'][0]} runs / {groups['k2'][1]} decisions; K1: {groups['k1'][0]} /
  {groups['k1'][1]}; K4: {groups['k4'][0]} / {groups['k4'][1]}. Six old flag-off fixed-clock/PID modes remained
  byte-identical. K5 overlay, four 342-decision randomized replays, 200 planted + 200 null estimator datasets,
  log-forgery/estimator and ledger checks passed. K6: 12 eight-connection configurations, 2,196 threaded + the same
  serialized decisions; 23-decision / 25-reservation edge test and 64 quantile transactions passed.
- Existing K7: five plugin arms / 240 decisions; {counts7['parity']:,} scalar query comparisons and
  {counts7['rates']:,} full-cell diagnostic decisions; original edges passed. Existing K10: nine plugin runs /
  {sum(r['decisions'] for r in results['k10_matrix'])} decisions and
  {sum(r['policy_tail'] for r in results['k10_matrix'])} tails; nine eight-connection configurations /
  {sum(r['decisions'] for r in results['k10_concurrency'])} threaded decisions with exact serialized parity;
  original policy-tail edge/transform/ledger and K10-vs-K7 method lifecycle/parity checks passed.
- Six arm emit/parser/CacheConfig checks passed: full model, ordinary K=10 MISS, L=5 and explicit cost ledger.

Final production-fit plugin selftests (48 decisions each):

| Arm | Decisions | Vision | MISS | Policy tails | Eligible C10 MISSes with follow-up |
|---|---:|---:|---:|---:|---:|
{arm_rows}

Final production-fit concurrency, per threaded side (serialized side exactly matches):

| Arm | Decisions | Vision | MISS | Policy tails | Eligible C10 policy tails |
|---|---:|---:|---:|---:|---:|
{concurrent_rows}

## Fits, bytes and exact commands

[arms_q1.json](arms_q1.json) has four C10 lifecycle arms (pi05 l10/spatial x 50/500) and two D1 arms (l10 x 50/500),
literal `<RUN>` fit placeholders, full model, five-control clients, full K=10 MISS, `cost_ledger: true`, and the
requested flags/root. Library 50 uses current/kref 5; 500 uses big/kref 8 (resolved bpool_cs). All requested prefits
were run afresh with the final installed plugin into `/tmp/q1_fits/`; none is a relabeled K7 pickle.

| Fit basename (under `/tmp/q1_fits/`, suffix `.pkl`) | Bytes | SHA256 |
|---|---:|---|
{fit_rows}

Raw library disk bytes count all regular files below the selected store-library directory; these include the
offline representation and are not deployment-pickle bytes. Owner deployed-pkl reference MB is borrowed from the
binding findings, not remeasured. No big-library table or monitor information enters a 50-library fit. Spatial's
nominal 50 library contains 49 episodes. D1 adds 23,768 / 265,256 compact table bytes (including eight quantile
bytes), excluding Python/pickle overhead; actual serialized fit bytes are above.

| Arm | Library rows | Raw library disk bytes | Q1 fit bytes | Owner deployed pkl MB |
|---|---:|---:|---:|---:|
{library_rows}

The exact six executed commands (repository root):

```bash
{exact}
```

[prefit_commands.json](prefit_commands.json) additionally contains the exact `<RUN>/fits` / `<RUN>/prefit_logs`
template per arm. [prefit.sh](prefit.sh) runs the six `/tmp/q1_fits` commands. Both use the required CPU/env prefix.

## Reproduction and installation provenance

Final verification commands from `/home/weiland/projects/openpi`:

```bash
P=({shlex.join(PREFIX)})
B=exp/offline_search/rounds/r05/q1_commit
"${{P[@]}}" "$B/prepare.py"
"${{P[@]}}" "$B/prepare_checks.py"
"${{P[@]}}" "$B/prepare_integration.py"
bash "$B/prefit.sh" > /tmp/q1_prefits_final.log 2>&1
bash "$B/run_regression.sh" > /tmp/q1_regression.log 2>&1
"${{P[@]}}" "$B/regression/method_test.py" > /tmp/q1_k10_method.log 2>&1
"${{P[@]}}" "$B/policy_tail_test.py" --source installed --out /tmp/q1_edges_final > /tmp/q1_edges_final.log 2>&1
"${{P[@]}}" "$B/grasp_plugin_test.py" --out /tmp/q1_grasp_plugin_final > /tmp/q1_grasp_plugin_final.log 2>&1
bash "$B/run_new_checks.sh"
"${{P[@]}}" "$B/finish.py"
```

The new-check wrapper runs method tests, delivery validation, the 14-run plugin matrix, six-arm concurrency and
two monitor fits/replays. Expanded per-job commands are in the matrix/monitor reports and relocated regression
scripts. Choose fresh `/tmp/q1_*` roots before rerunning: test logs append and concurrency fixtures refuse existing
directories. Run the parallel new-check wrapper only after the regression K2 multi-process shell matrix finishes;
thereafter six new-check driver/worker processes plus two regression processes stay within the eight-process cap.

The shared plugin/selftest/verifier changed during parallel Q2 work at 2026-09-28 01:36 UTC. Interrupted earlier
evidence was archived under `regression/results_beforeq2/` and `/tmp/q1_beforeq2_*`; none supplies the final counts.
The full final suite was rerun against the updated installed imports. The final audit compares all five shared
source files with its captured pre-check snapshot and records their hashes in results/final_summary.json.
No shared file was edited by Q1. New method files were developed only inside the owned Q1 directory.

Initial fixture-only failures were fixed before final reruns: K10's hook needs a dataclass facade; NPZ historical
arrays must be materialized once; wall-time fields cannot enter exact replay equality; a committed plugin decision
clears `_dec`; and the synthetic aperture must match the fake key builder. No failed run is counted as final evidence.

All added top-level source/spec files with final availability times and SHA256:

| File | UTC | SHA256 |
|---|---|---|
{source_rows}

`results/file_inventory.json` contains the complete owned-file inventory, including relocated recipes and evidence
(excluding the inventory and self-referential hand-back files). `HANDBACK.sha256` hashes this hand-back.

## Coordinator smoke and remaining limits

Coordinator-only recipe; Q1 did not execute it. Choose a new run root and coordinator-authorized GPU/ports/CPUs.
Copy the six arm fits with verified hashes, resolve placeholders, and emit the ordinary arm YAMLs:

```bash
RUN=/home/weiland/trace_runs/os_closed_loop/r05_q1_smoke
Q=exp/offline_search/rounds/r05/q1_commit
mkdir -p "$RUN/fits"
for arm in r5q1_c10_p_l10_50 r5q1_c10_p_l10_500 r5q1_c10_p_sp_50 r5q1_c10_p_sp_500 r5q1_d1_p_l10_50 r5q1_d1_p_l10_500; do
  cp "/tmp/q1_fits/$arm.pkl" "$RUN/fits/$arm.pkl"
done
sed "s|<RUN>|$RUN|g" "$Q/arms_q1.json" > "$RUN/arms_q1_resolved.json"
"${{P[@]}}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$RUN" --spec "$RUN/arms_q1_resolved.json"
# Coordinator supplies PORTS and SERVER_CPUS already authorized for its live work.
bash exp/offline_search/closed_loop/ops/sync_remote.sh "$RUN" r5q1_c10_p_l10_50 r5q1_d1_p_l10_50
unset OSCL_MANIFEST
PORTS="$PORTS" SERVER_CPUS="$SERVER_CPUS" OSCL_TASKS=0,1 OSCL_EPISODES=0,1 WPS=2 bash exp/offline_search/closed_loop/ops/chain.sh "$RUN" r5q1_c10_p_l10_50 r5q1_d1_p_l10_50
```

Use a separate full-run root after the smoke. Retain the full model, K=10 MISS and L=5 client, import the Q1 module
and current plugin dependencies, and preserve exact method/kwargs/cell artifact metadata. Audit policy-tail source,
zero tail vision/MISS cost, anchor guards, D1 reason 9 and <=1 D1 intervention/episode. Vision costs .152, each
full MISS adds .848, and each blind/policy-tail decision costs zero on the owner basis. D1's forced vision MISS
costs 1.0 at that decision; no fixed-path calculation here establishes realized rollout savings or success.

Unverified: real GPU policy inference, websocket/simulator integration, LIBERO closed-loop success/cost effects and
physical grasp recovery. Fake-policy controls and historical fixed observations establish implementation behavior,
not causal SR improvements. Endpoint-confirmed K7 guards retain their documented blind-gap limitations; the optional
xyz monitor observes arm displacement, not object contact or task success. No new policy, representation, borrowed
big-library calibration, or external model was introduced.
'''
    write(HERE/'HANDBACK.md', text)
    write(HERE/'HANDBACK.sha256', digest(HERE/'HANDBACK.md') + '  HANDBACK.md\n')
    inventory = [info(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in
                 ('file_inventory.json', 'HANDBACK.md', 'HANDBACK.sha256')]
    write(R/'file_inventory.json', json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(PASS=True, handback=str(HERE/'HANDBACK.md'), files=len(inventory),
                         production_fits=len(delivery['fits']), new_matrix_decisions=sum(r['decisions'] for r in matrix),
                         new_concurrency_per_side=sum(r['decisions'] for r in concurrency))))


if __name__ == '__main__':
    main()
