# R11Knob

Serving entry: `exp.offline_search.rounds.r11.knob.recipe:R11Knob`.
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

| Root | Fleet | Arms | Σ predicted IR | Plan files / bytes | Relocated fit bytes |
|---|---|---:|---:|---:|---:|
| r11_knob_1 | timan108 | 23 | 7.000685492 | 135 / 24,155,933,167 | 803,274,675 |
| r11_knob_2 | timan108 | 24 | 7.126871767 | 138 / 24,186,850,023 | 834,190,142 |
| r11_knob_3 | timan107 | 23 | 6.997343000 | 135 / 24,191,538,684 | 838,880,714 |
| r11_knob_4 | timan107 | 24 | 7.127705120 | 138 / 24,321,490,160 | 968,829,878 |
| r11_devknob_50 | timan107 | 20 | 5.789788376 | 126 / 23,950,916,194 | 598,263,466 |

All five roots are under `/home/weiland/trace_runs/os_closed_loop/`.
The four A roots contain 94 arms at 500 pairs each. The dev root contains 20
arms at 50 B pairs each, size-50 subsets, K=5/task, seed=20261003, and exactly
the `r11_dev_size50` manifests. The B contracts attest disjointness against the
actual prefit. Dev and A arms share the same fit bytes; the pilot cannot tune them.

`build fit` always fits lower layers afresh from the selected nested B library:

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL
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
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL
"${P[@]}" -m exp.offline_search.rounds.r11.knob.input_audit
"${P[@]}" -m exp.offline_search.rounds.r11.knob.build prepare --workers 4
"${P[@]}" -m exp.offline_search.rounds.r11.knob.rebuild_all
"${P[@]}" -m exp.offline_search.rounds.r11.knob.tests
"${P[@]}" -m exp.offline_search.rounds.r11.knob.verification all --workers 4
"${P[@]}" -m exp.offline_search.rounds.r11.knob.faults
"${P[@]}" -m exp.offline_search.rounds.r11.knob.finalize
"${P[@]}" -m exp.offline_search.rounds.r11.knob.docs
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
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL
# Install the new serving closure once, with both fleets idle.
WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only
WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker

run_root() {
  local knob_host=$1 knob_tag=$2 knob_ports=$3 knob_sync_port=$4
  local knob_run=/home/weiland/trace_runs/os_closed_loop/$knob_tag
  local -a knob_arms
  mapfile -t knob_arms < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$knob_run/arms.json")
  WORKER_HOST=$knob_host "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host SYNC_PORT=$knob_sync_port "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host PORTS=$knob_ports WPS=4 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
    "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$knob_run" "${knob_arms[@]}" || return
  WORKER_HOST=$knob_host "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$knob_run" "${knob_arms[@]}"
}

# First: 20-arm B pilot. Keep all fitted settings fixed.
run_root timan107 r11_devknob_50 23230,23231 23197 || exit 1

# Two fleets, two sequential roots apiece; distinct forwarded/sync ports.
(run_root timan108 r11_knob_1 23220,23221 23198 && \
 run_root timan108 r11_knob_2 23220,23221 23198) > exp/offline_search/rounds/r11/knob/coordinator_timan108.log 2>&1 &
knob_fleet108=$!
(run_root timan107 r11_knob_3 23230,23231 23197 && \
 run_root timan107 r11_knob_4 23230,23231 23197) > exp/offline_search/rounds/r11/knob/coordinator_timan107.log 2>&1 &
knob_fleet107=$!
wait "$knob_fleet108"
wait "$knob_fleet107"
```

The coordinator owns remote compatibility, actual policy calls, realized IR,
SR and live ledger/cadence audits. CPU uses recorded B keys/actions and fake
stage-2/3 outputs through the real plugin/orchestrator; it makes no causal SR
claim and does not verify GPU memory or throughput.
