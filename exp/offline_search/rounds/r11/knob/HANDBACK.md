# R11 layer-4 handback

**COMPLETE on CPU.** No sync, server, worker, chain, neural policy inference,
robotics simulator, closed-loop evaluation, GPU action or git command was
launched. All writes are under `knob/` and the five new run roots. Existing
R10/explorer inputs remain SHA-identical.

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

Fleet sums: timan108 **14.127557259**, timan107 **14.125048121**.
The A manifest is byte-identical to both R10 size roots: SHA256
`2c3b0477794c04b9f9925100346f0d0adbca4dcfe42f3c2eb26278f18b4d9a29`.
Each B arm has the same seed/pairs as the size-50 dev prerequisite and carries
`dev=true, init_pool=B`. The 20 dev arms reference the exact A-grid fit bytes.

Key verification outputs, produced by this agent:

```text
INPUTS_READ 132 protected inputs; SHA unchanged
CALIBRATION {"PASS": true, "cells": 8, "frozen_schedule_arms": 38, "frozen_state_arms": 44, "exact_settings": true, "max_simulator_deviation": 0.0}
Ran 16 tests in 0.284s

OK
ALL_SELFTESTS {"PASS": true, "test_arms": 94, "dev_arms": 20, "decisions": 41040, "equivalence_cell_sizes": 8, "equivalence_decisions": 2880, "equivalence_forced_guards": 480, "differing_decisions": 0}
LIFECYCLE_FAULT_PASS pi05 10 360
LIFECYCLE_FAULT_PASS groot 10 360
REBUILD_EXACT pi05 l10 50
REBUILD_EXACT pi05 l10 500
REBUILD_EXACT pi05 spatial 50
REBUILD_EXACT groot l10 50
REBUILD_EXACT groot l10 500
REBUILD_EXACT groot spatial 50
REBUILD_EXACT pi05 l10 200
REBUILD_EXACT groot l10 200
FRESH_FIT {"PASS": true, "base_arrays_exact": true, "setting_exact": true, "cache_arrays_byte_identical": true, "head_arrays_byte_identical": true, "guard_calibration_byte_identical": true, "distance_scale_identical": true, "distance_scale": 11.805561065673828}
RELEASE_PASS {"PASS": true, "protected_inputs_unchanged": 132, "test_arms": 94, "dev_arms": 20, "methods": 9, "frozen_settings_exact": 82, "raw_rebuild_cells_exact": 8, "off_equivalence_decisions": 2880, "off_forced_guards": 480, "selftest_decisions": 41040, "lifecycle_fallbacks": 20, "extra_h100_store_bytes": 0, "standard_plans": 5, "no_launch": true, "no_gpu": true}
```

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
   eight-replicate rounded-setting reference is **0.012059950042**;
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

- 94 distinct local fit artifacts: **3,445,199,813 bytes** total,
  range **28,525,358–72,454,715 bytes**.
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

Exact coordinator commands (provided only):

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

Coordinator work still requiring real deployment: install/verify the H100 code
closure and both worker islands; verify full pi05/GR00T model and checkpoint
compatibility, GPU memory/throughput and live stage-1/stage-2/3 execution; run the
fixed-setting B pilot, then the A sweep; audit actual ledgers, lifecycle, guard
and knob reason counts, realized IR vs prospective predictions, and SR. This
CPU handback verifies plugin mechanics with recorded B keys/actions and fake
policy outputs. It does not claim live IR or success. No setting may be changed
using the pilot or A results within this frozen experiment.

Exact arms per root:

`r11_knob_1`:

```text
r11_pi05_l10_50_adaptive_error_hybrid_ir40
r11_pi05_l10_50_error_hybrid_ir40
r11_groot_l10_50_distance_ir40
r11_pi05_spatial_50_random_ir40
r11_groot_spatial_50_adaptive_error_hybrid_ir40
r11_pi05_l10_50_random_tail2_ir32
r11_groot_l10_200_random_ir32
r11_pi05_l10_50_distance_ir32
r11_pi05_spatial_50_error_hybrid_ir32
r11_groot_spatial_50_disagreement_ir32
r11_pi05_l10_50_periodic_pgt1_ir32
r11_groot_l10_50_random_ir32
r11_groot_spatial_50_random_ir32
r11_pi05_spatial_50_periodic_ir25
r11_groot_l10_50_periodic_ir25
r11_groot_l10_200_random_ir25
r11_pi05_l10_200_error_hybrid_ir25
r11_groot_l10_500_error_hybrid_ir25
r11_groot_spatial_50_error_hybrid_ir25
r11_groot_spatial_50_distance_ir25
r11_groot_l10_200_periodic_ir25
r11_groot_spatial_50_random_ir25
r11_groot_l10_200_off
```

`r11_knob_2`:

```text
r11_pi05_spatial_50_periodic_ir40
r11_pi05_spatial_50_distance_ir40
r11_groot_spatial_50_error_hybrid_ir40
r11_pi05_l10_50_random_ir40
r11_groot_spatial_50_random_ir40
r11_pi05_l10_50_random_ir32
r11_groot_l10_50_random_tail2_ir32
r11_pi05_spatial_50_distance_ir32
r11_groot_l10_50_error_hybrid_ir32
r11_groot_l10_50_distance_ir32
r11_pi05_l10_50_periodic_ir32
r11_groot_l10_50_periodic_pgt1_ir32
r11_groot_spatial_50_periodic_pgt1_ir32
r11_pi05_spatial_50_adaptive_error_hybrid_ir32
r11_pi05_l10_200_periodic_ir25
r11_pi05_l10_500_distance_ir25
r11_pi05_l10_50_distance_ir25
r11_pi05_spatial_50_error_hybrid_ir25
r11_groot_l10_50_error_hybrid_ir25
r11_pi05_l10_500_random_ir25
r11_groot_l10_50_random_ir25
r11_groot_l10_50_off
r11_pi05_l10_200_off
r11_groot_spatial_50_off
```

`r11_knob_3`:

```text
r11_groot_l10_50_periodic_ir40
r11_pi05_l10_50_periodic_ir40
r11_groot_l10_50_error_hybrid_ir40
r11_pi05_spatial_50_error_hybrid_ir40
r11_pi05_spatial_50_adaptive_error_hybrid_ir40
r11_pi05_l10_50_adaptive_error_hybrid_ir32
r11_groot_spatial_50_distance_ir32
r11_pi05_spatial_50_disagreement_ir32
r11_groot_l10_200_error_hybrid_ir32
r11_pi05_l10_50_disagreement_ir32
r11_groot_l10_200_periodic_ir32
r11_pi05_spatial_50_periodic_pgt1_ir32
r11_groot_l10_50_adaptive_error_hybrid_ir32
r11_pi05_l10_50_random_ir25
r11_groot_l10_500_periodic_ir25
r11_groot_l10_50_distance_ir25
r11_pi05_l10_50_error_hybrid_ir25
r11_pi05_l10_200_distance_ir25
r11_groot_l10_200_distance_ir25
r11_groot_l10_500_random_ir25
r11_pi05_l10_50_periodic_ir25
r11_pi05_spatial_50_random_ir25
r11_pi05_l10_50_off
```

`r11_knob_4`:

```text
r11_groot_l10_50_random_ir40
r11_groot_spatial_50_periodic_ir40
r11_groot_spatial_50_distance_ir40
r11_pi05_l10_50_distance_ir40
r11_groot_l10_50_adaptive_error_hybrid_ir40
r11_pi05_spatial_50_periodic_ir32
r11_groot_spatial_50_periodic_ir32
r11_groot_l10_50_disagreement_ir32
r11_groot_spatial_50_error_hybrid_ir32
r11_pi05_l10_50_error_hybrid_ir32
r11_groot_l10_200_distance_ir32
r11_groot_l10_50_periodic_ir32
r11_pi05_spatial_50_random_ir32
r11_groot_spatial_50_adaptive_error_hybrid_ir32
r11_pi05_l10_500_periodic_ir25
r11_groot_l10_200_error_hybrid_ir25
r11_pi05_l10_500_error_hybrid_ir25
r11_groot_l10_500_distance_ir25
r11_pi05_spatial_50_distance_ir25
r11_pi05_l10_200_random_ir25
r11_groot_spatial_50_periodic_ir25
r11_groot_l10_500_off
r11_pi05_l10_500_off
r11_pi05_spatial_50_off
```

`r11_devknob_50`:

```text
r11_devknob_pi05_l10_50_off
r11_devknob_pi05_l10_50_random_ir32
r11_devknob_pi05_l10_50_periodic_ir32
r11_devknob_pi05_l10_50_distance_ir32
r11_devknob_pi05_l10_50_error_hybrid_ir32
r11_devknob_pi05_spatial_50_off
r11_devknob_pi05_spatial_50_random_ir32
r11_devknob_pi05_spatial_50_periodic_ir32
r11_devknob_pi05_spatial_50_distance_ir32
r11_devknob_pi05_spatial_50_error_hybrid_ir32
r11_devknob_groot_l10_50_off
r11_devknob_groot_l10_50_random_ir32
r11_devknob_groot_l10_50_periodic_ir32
r11_devknob_groot_l10_50_distance_ir32
r11_devknob_groot_l10_50_error_hybrid_ir32
r11_devknob_groot_spatial_50_off
r11_devknob_groot_spatial_50_random_ir32
r11_devknob_groot_spatial_50_periodic_ir32
r11_devknob_groot_spatial_50_distance_ir32
r11_devknob_groot_spatial_50_error_hybrid_ir32
```
