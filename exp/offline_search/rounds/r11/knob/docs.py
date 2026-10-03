"""Write the reviewable handback from completed CPU evidence."""
from pathlib import Path
import json
from .build import HERE,REPO,R10
from exp.offline_search.rounds.r10.data import sha

PREFIX='''cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL'''
LOCAL='''"${P[@]}" -m exp.offline_search.rounds.r11.knob.input_audit
"${P[@]}" -m exp.offline_search.rounds.r11.knob.build prepare --workers 4
"${P[@]}" -m exp.offline_search.rounds.r11.knob.rebuild_all
"${P[@]}" -m exp.offline_search.rounds.r11.knob.tests
"${P[@]}" -m exp.offline_search.rounds.r11.knob.verification all --workers 4
"${P[@]}" -m exp.offline_search.rounds.r11.knob.faults
"${P[@]}" -m exp.offline_search.rounds.r11.knob.finalize
"${P[@]}" -m exp.offline_search.rounds.r11.knob.docs'''
COORD='''# Install the new serving closure once, with both fleets idle.
WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only
WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker

run_root() {
  local knob_host=$1 knob_tag=$2 knob_ports=$3 knob_sync_port=$4
  local knob_run=/home/weiland/trace_runs/os_closed_loop/$knob_tag
  local -a knob_arms
  mapfile -t knob_arms < <("${P[@]}" -c 'import json,sys; print("\\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$knob_run/arms.json")
  WORKER_HOST=$knob_host "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host SYNC_PORT=$knob_sync_port "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host PORTS=$knob_ports WPS=4 MAX_ATTEMPTS=3 POLL_SECONDS=60 \\
    "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$knob_run" "${knob_arms[@]}"
}

# First: 20-arm B pilot. Keep all fitted settings fixed.
run_root timan107 r11_devknob_50 23230,23231 23197 || exit 1

# Two fleets, two sequential roots apiece; distinct forwarded/sync ports.
(run_root timan108 r11_knob_1 23220,23221 23198 && \\
 run_root timan108 r11_knob_2 23220,23221 23198) > exp/offline_search/rounds/r11/knob/coordinator_timan108.log 2>&1 &
knob_fleet108=$!
(run_root timan107 r11_knob_3 23230,23231 23197 && \\
 run_root timan107 r11_knob_4 23230,23231 23197) > exp/offline_search/rounds/r11/knob/coordinator_timan107.log 2>&1 &
knob_fleet107=$!
wait "$knob_fleet108"
wait "$knob_fleet107"'''

def main():
    final=json.loads((HERE/'final_audit.json').read_text())
    assert final['PASS']
    roots=json.loads((HERE/'roots.json').read_text())
    cal=json.loads((HERE/'calibration_audit.json').read_text())
    dep=json.loads((HERE/'deployment.json').read_text())
    keys=json.loads((HERE/'key_discrepancy.json').read_text())
    unit=(HERE/'tests.log').read_text().split('----------------------------------------------------------------------')[-1].strip()
    manifest_sha=sha(Path(roots[0]['root'])/'eval500.json')
    root_table='| Root | Fleet | Arms | Σ predicted IR | Plan files / bytes | Relocated fit bytes |\n|---|---|---:|---:|---:|---:|\n'
    for i,r in enumerate(roots):
        d=next(d for d in dep['roots'] if d['root']==r['root'])
        host='timan108' if i<2 else 'timan107'
        root_table+=f"| {Path(r['root']).name} | {host} | {r['arm_count']} | {r['predicted_IR_sum']:.9f} | {d['plan_files']} / {d['plan_bytes']:,} | {d['relocated_artifact_bytes']:,} |\n"
    common='''Serving entry: `exp.offline_search.rounds.r11.knob.recipe:R11Knob`.
The nine choices are `off`, `random`, `periodic`, `periodic_pgt1`,
`random_tail2`, `distance`, `disagreement`, `error_hybrid`, and
`adaptive_error_hybrid`. The default is off. Guard decisions retain their
original reason and flags. Knob-only reasons are 61–68 in that order after off.
Layer 4 has no task-indexed fitted parameters. Task/init identify the schedule
coin only. Astra’s controller uses its reference episode-UID seeded RNG.

One pickle embeds the unchanged library-fitted R10 cache/guard/corrector plus
the cell-wide knob settings, pooled predictor and sorted CDF reference. Serving
imports no R11 exploration or fitting module and opens no exploration path.
The only external numerical payload is the parent action array through R10
`IndexedRows`. The ordinary miss commits one policy chunk and one five-control
policy tail, then performs fresh retrieval. No guard/corrector memo is rewritten.
Periodic history and adaptive costs come from accepted histories; proposal
retries neither redraw nor advance state. Tail provenance is per connection,
reset per episode, and checked against actual committed MISS history.
'''
    readme='# R11Knob\n\n'+common+'\n'+root_table+'''
All five roots are under `/home/weiland/trace_runs/os_closed_loop/`.
The four A roots contain 94 arms at 500 pairs each. The dev root contains 20
arms at 50 B pairs each, size-50 subsets, K=5/task, seed=20261003, and exactly
the `r11_dev_size50` manifests. The B contracts attest disjointness against the
actual prefit. Dev and A arms share the same fit bytes; the pilot cannot tune them.

`build fit` always fits lower layers afresh from the selected nested B library:

```bash
'''+PREFIX+'''
"${P[@]}" -m exp.offline_search.rounds.r11.knob.build fit --model pi05 --suite l10 --r10-size 50 --method periodic --target .32 --output exp/offline_search/rounds/r11/knob/one_pi05_l10_50_periodic_ir32.pkl
```

The frozen calibration caches are SHA-bound B-only intermediates. Rates,
thresholds and pooled heads are recalculated and compared exactly with both
explorer freezes. `build rebuild --model MODEL --suite SUITE --r10-size N`
regenerates raw-library nested astra features and opus fold-PCA anchor tables
under `knob/rebuilt/` and asserts every numeric array equals the references.
`prepare` reuses the already library-fitted R10 lower-layer artifacts, retaining
their byte-exact arrays; it recalculates every original and added knob setting.
The fresh-fit smoke check independently verifies identical lower-layer arrays.
Existing fit output paths are refused. Identical prediction reruns preserve the
original timestamp; altered predictions are refused.

Full CPU reproduction:

```bash
'''+PREFIX+'\n'+LOCAL+'''
```

Evidence: `calibration_audit.json`, `rebuild_audit.json`, `selftests.json`,
`lifecycle_faults.json`, `fresh_fit_audit.json`, `runtime_audit.json`,
`deployment.json`, and `final_audit.json`. Read `PREDICTION_ADDENDUM.md` for
prospective library IR predictions and `HANDBACK.md` for measured discrepancies.
`H100_SOURCES.sha256` lists the complete serving/controller source closure;
`ALL_SOURCES.sha256` lists this implementation and verification code.
No extra H100 store arrays are required. Standard plans relocate all 94 fits
and the 20 dev-arm references without external head/threshold dependencies.

Coordinator deployment and launch commands (provided, never executed here).
The full setup installs the source closure and requires idle fleets under the
standard controller’s maintenance lock. Verify that the chosen forwarded and
sync ports are free before launch. If the exact closure is already installed,
skip setup and retain the verification/plan/sync/chain sequence:

```bash
'''+PREFIX+'\n'+COORD+'''
```

The coordinator owns remote compatibility, actual policy calls, realized IR,
SR and live ledger/cadence audits. CPU uses recorded B keys/actions and fake
stage-2/3 outputs through the real plugin/orchestrator; it makes no causal SR
claim and does not verify GPU memory or throughput.
'''
    (HERE/'README.md').write_text(readme)
    hand='# R11 layer-4 handback\n\n**COMPLETE on CPU.** No sync, server, worker, chain, neural policy inference,\nrobotics simulator, closed-loop evaluation, GPU action or git command was\nlaunched. All writes are under `knob/` and the five new run roots. Existing\nR10/explorer inputs remain SHA-identical.\n\n'+common+'\n'+root_table
    hand+='\nFleet sums: timan108 **'+f"{roots[0]['predicted_IR_sum']+roots[1]['predicted_IR_sum']:.9f}"+'**, timan107 **'+f"{roots[2]['predicted_IR_sum']+roots[3]['predicted_IR_sum']:.9f}"+'''**.
The A manifest is byte-identical to both R10 size roots: SHA256
`'''+manifest_sha+'''`.
Each B arm has the same seed/pairs as the size-50 dev prerequisite and carries
`dev=true, init_pool=B`. The 20 dev arms reference the exact A-grid fit bytes.

Key verification outputs, produced by this agent:

```text
'''+(HERE/'input_audit.json').read_text()[:0]+'INPUTS_READ '+str(final['protected_inputs_unchanged'])+' protected inputs; SHA unchanged\nCALIBRATION '+json.dumps({k:v for k,v in cal.items() if k!='records'})+'\n'+unit+'\n'+(HERE/'selftests.log').read_text().splitlines()[-1]+'\n'+(HERE/'faults.log').read_text().strip()+'\n'+(HERE/'rebuild_all.log').read_text().strip()+'\nFRESH_FIT '+json.dumps(json.loads((HERE/'fresh_fit_audit.json').read_text()))+'\nRELEASE_PASS '+json.dumps(final)+'''\n```

Each of the 114 actual-prefit CPU plugin selftests has 360 decisions over ten
tasks, two simultaneous connections and 30 episode resets. They verify action
bytes, correction arithmetic, provenance, no knob calls on guard anchors,
actual N/V/M, ordinary miss → policy tail → fresh anchor, periodic bounds and
keyed coins. The off comparisons cover all eight cell-sizes, 2,880 decisions
and 480 forced guards, with zero differences in full actions, synthesis,
retrieval, confidence, flags and extras; only timing/runtime identity is omitted.
The two extra adaptive lifecycle tests deliberately reject a policy tail via
an execution-count mismatch, covering 720 decisions and 20 real fallback looks;
the next proposal charges the actual accepted ledger. Unit tests cover every
method, deterministic coins, binomial random share, retries, reset, independent
clones, stale rejected tail proposals, nonfinite fallback, endpoints/ties,
exact uncorrected feature arithmetic, and controller parity with astra.

All 38 opus settings and 44 astra settings equal their freezes exactly.
All eight raw-library rebuilds reproduce the nested astra arrays/pooled heads
and opus anchor arrays exactly. The fast schedule solver matches the explorer
simulator at 8 and 16 replicates with maximum absolute IR deviation **0**.
Static astra calibration and adaptive q0/validation are recomputed with the
reference library procedure; frozen dose/threshold/tie/beta/eta equality is exact.
The pooled-head fit is regenerated and every coefficient/mean/std equals the
frozen head. Added targets and all final library forecasts were written to
`PREDICTION_ADDENDUM.md` before any R11 evaluation, retaining its timestamp.

Discrepancies and selected reference behavior:

1. Opus’s offline `_u` uses `('r11-random'/'r11-gap', seed, replicate,
   parent_episode_id, anchor index/step)`. Its serving SPEC explicitly requires
   the R6 `uniform` function with R11 keys, original task/init, decision step,
   domain, and periodic `last_call_step`. Serving follows that documented rule,
   preserving R6’s complete `Q2-deploy-v1` payload prefix. Calibration follows
   the frozen offline solver so its four-decimal settings remain exact.
   `key_discrepancy.json` replays the required serving keys on the same B states
   with original B init identities. Maximum absolute IR change versus the
   eight-replicate rounded-setting reference is **'''+f"{keys['max_absolute_IR_effect']:.12f}"+'''**;
   the largest is GR00T Spatial-50 random @.25: reference .250054954205,
   serving-seed library replay .262114904246. This is a one-seed sampling/key
   effect, not a changed setting or measured closed-loop error. No tuning was
   applied. Four-decimal setting rounding itself changes modeled IR by at most
   **0.000177352206495**. Our prospective schedule predictions use the
   eight-replicate solve convention; opus’s original arm-grid forecasts use
   sixteen replicates of the unrounded solve. Both count simulators agree exactly.
2. Astra retains fixed selected-library PCA; opus refits fold PCA below 500.
   No score/guard tables are numerically mixed. Largest base-IR difference is
   GR00T L10-50: opus .199270389171, astra .209650084602, Δ .010379695431.
   Other nonzero deltas (astra−opus): GR00T Spatial-50 +.007094088260,
   GR00T L10-200 +.001893009742, pi05 L10-200 +.000148290636,
   pi05 L10-50 −.000869743590; both L10-500 and pi05 Spatial-50 have zero delta.
3. The task requires a unique reason per method. Astra’s proposed shared 11 is
   replaced by 65/66/67/68. Random-tail-2 uses external 64 for both its trigger
   and single follow-up; its internal trigger-step memo plays the role of opus’s
   reason-61 marker. Periodic-pgt1 uses 63 for its ordinary and post-guard calls.
   These logging changes have **zero action/IR effect** and preserve guard reasons.
4. Astra’s post-freeze static endpoint handling takes precedence over its general
   nonfinite rule: at q=0 even a nonfinite score is off, at q=1 it calls. All
   frozen static doses are interior, so endpoint precedence has **zero frozen-arm
   effect**. Interior nonfinite scores always call and are ledger-charged.
5. The plugin marks an executed policy tail as HIT. Thus the fresh look after
   a valid tail uses the inherited stale metric (`prev_hit=True`), as astra
   explicitly documents; cache-anchor invalidation still forces fresh retrieval.
   The opus phrase `prev_hit=False` after a miss does not describe the look after
   the intervening committed tail. No lower-layer history or metric was altered;
   the numerical effect on knob-off decisions is **zero** (proved by replays).

Artifacts and deployment:

- 94 distinct local fit artifacts: **'''+f"{sum(r['bytes'] for r in cal['records']):,}"+''' bytes** total,
  range **'''+f"{min(r['bytes'] for r in cal['records']):,}–{max(r['bytes'] for r in cal['records']):,}"+''' bytes**.
  The 20 dev arms reuse these fit files. `calibration_audit.json` binds every SHA.
- Extra H100 store bytes: **0**. No new payload arrays, cached keys, predictor
  files, CDF files, G fits or calibration files are loaded by serving.
- Every root has a passing standard `h100_sync/plan.json`, with relocated action
  paths under `/data/oscl_h100/store/library/`. The table above gives complete
  plan bytes (including shared existing assets) and relocated artifact bytes.
- `H100_SOURCES.sha256` and `source_list.json` describe the serving/controller
  closure. Only `knob/{__init__,recipe,controller}.py` are new serving files;
  no `closed_loop/` changes were needed. Runtime unpickling was checked in a
  clean process and imports neither explorer nor fit code.

Exact local reproduction:

```bash
'''+PREFIX+'\n'+LOCAL+'''
```

Exact coordinator commands (provided only):

```bash
'''+PREFIX+'\n'+COORD+'''
```

Coordinator work still requiring real deployment: install/verify the H100 code
closure and both worker islands; verify full pi05/GR00T model and checkpoint
compatibility, GPU memory/throughput and live stage-1/stage-2/3 execution; run the
fixed-setting B pilot, then the A sweep; audit actual ledgers, lifecycle, guard
and knob reason counts, realized IR vs prospective predictions, and SR. This
CPU handback verifies plugin mechanics with recorded B keys/actions and fake
policy outputs. It does not claim live IR or success. No setting may be changed
using the pilot or A results within this frozen experiment.

Exact arms per root:
'''
    for root in roots:
        hand+='\n`'+Path(root['root']).name+'`:\n\n```text\n'+'\n'.join(root['arms'])+'\n```\n'
    (HERE/'HANDBACK.md').write_text(hand)
    (HERE/'DOCUMENTS.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in (HERE/'README.md',HERE/'HANDBACK.md',HERE/'PREDICTION_ADDENDUM.md')))
    print('DOCS_READY',HERE/'README.md',HERE/'HANDBACK.md')

if __name__=='__main__':main()
