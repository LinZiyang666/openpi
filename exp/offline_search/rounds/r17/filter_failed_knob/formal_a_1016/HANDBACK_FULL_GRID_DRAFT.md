# Formal LIBERO test-A knob grids — 2026-10-16

This is the active LIBERO knob procedure. Earlier dev-selected LIBERO levels,
A retests and raises are withdrawn. MT50 stays on its existing direct-test
procedure. No dev outcomes are read and no dev-chosen levels select tasks or levels.
Immutable campaign/fit templates and level calibrations are reused.

The authoritative launch inventory is `CAMPAIGNS.json` (48 launch units,
544,450 new episodes); `TOTALS.json` aggregates the 20 model/mode/arm profiles.
Each model/mode has both an original and a success-filtered arm. R17 has one
campaign per profile; R15 full-step profiles have eight cell campaigns of 2,500
episodes each. Use only the command files named by this inventory or the
`*_profile_*` scripts below. Short cell-named folders/command files under this
new directory are unpublished CPU build drafts, superseded by the unique
prefixed run basenames in the inventory; do not glob them for scheduling.

## Design and counts

Test A only: tasks 0–9, inits 0–49, 50 episodes/task/level. R17 levels are
k50→k40→k30→k20→k10→koff; VLA levels are ir45→ir40→ir32→ir25→off.
The same-step same-GPU pure A outcome is the top level. A level passes iff its
deficit is ≤2 successes out of 50. The first failure selects the preceding
level, with pure selected if the first non-pure level fails. Lower-level
recoveries do not change the choice. For a complete 500-episode cell whose
stall-only system reaches or exceeds the pure cell SR, select off directly and
report `tuned=false`. No raise rounds or separate chosen-level tests are needed.

The definitions contain the full requested non-pure grids; the off cell rule
is applied by analysis, which never selects a knob for a qualifying cell.
These are fixed full-grid counts, before any optional off-first dispatch
pruning. No dispatch pruning has been applied or claimed.

Filtering only slices retrievable candidate rows. No library top-up, fitting,
metric/PCA/whitening/corrector/stall changes or periodic-gap/cap recalibration.
R17 imports `filter_failed` row selection and its existing views; VLA uses the
existing `candidates` slicer. Pure references remain unfiltered. Empty tasks
are run through the full system's existing MISS path.

| Model | Mode | New episodes original / filtered | GPU h original / filtered | GPU | R/B/W | Workers |
|---|---|---:|---:|---|---|---|
| dp | reduced | 30,000 / 30,000 | 7.12 / 7.12 | RTX 4090 | 4/8/48 | local×48 |
| dp | full | 36,000 / 36,000 | 10.73 / 10.73 | RTX 4090 | 4/8/48 | local×48 |
| fastwam | reduced | 30,000 / 30,000 | 9.13 / 9.13 | RTX 4090 | 2/16/70 | timan107×30, timan108×40 |
| fastwam | full | 36,000 / 36,000 | pending / pending | H100 | 2/16/85 | timan107×45, timan108×40 |
| cosmos | reduced | 30,000 / 30,000 | 9.67 / 9.67 | H100 | 8/1/85 | timan107×45, timan108×40 |
| cosmos | full | 36,000 / 36,000 | 16.02 / 16.02 | H100 | 8/1/85 | timan107×45, timan108×40 |
| pi05 | reduced | 18,500 / 16,000 | pending / pending | RTX 4090 | 3/1/64 | timan107×64 |
| pi05 | full | 20,000 / 20,000 | pending / pending | H100 | 3/1/39 | timan108×39 |
| groot | reduced | 17,950 / 16,000 | pending / pending | RTX 4090 | 4/1/64 | timan107×64 |
| groot | full | 20,000 / 20,000 | pending / pending | RTX 4090 | 4/1/64 | timan107×64 |

New episodes: **544,450**. Existing/queued standalone off plans supply **47,550**
more level episodes, making **592,000** grid observations including reused off
(the library-independent pure references are reused separately).
Counts are simulation slots, not claims that pending reused off plans finished.

The reduced R17 grids omit 6,000 off episodes/profile. Original reduced VLA off
plans cover only 30 of 80 task/cells for π0.5 and 41 of 80 for GR00T; the other
89 task/cells' old mixed-run outcomes are withdrawn. The new original grids
therefore add only the missing off cohorts (π0.5: 2,500; GR00T: 1,950). Existing
filtered VLA off plans cover all 4,000/model. Full-step grids include off.

Hour estimates use source official-A accepted episodes divided by wall time
from LAUNCH.started to RECEIPT.sealed_at, including startup/cleanup. Those five
count/time records were captured before the parallel cleanup and are preserved
in `THROUGHPUT.json`; they contain no SR, journals or per-step results. They
estimate the original and filtered arms at the same source rate. Actual
throughput can change with level mixture, filtering and execution nodes.
Fast-WAM full and both VLA modes have no surviving source-test timing metadata
available here. Their hours are null, not invented. Supply timing-only
accepted-count/elapsed-seconds for those five model/modes to fill that file;
`formal_a inventory` recalculates `episodes / source_episodes_per_hour`.

R17 source serving: DP 4090 R4/B8/MPS100; Fast-WAM reduced 4090 R2/B16/MPS100,
full H100 R2/B16/MPS100; Cosmos H100 R8/B1/MPS100 both modes. R15 VLA full:
π0.5 H100 R3/B1/W39 on timan108; GR00T 4090 R4/B1/W64 on timan107.
R16 reduced: π0.5 4090 R3/B1/W64, GR00T 4090 R4/B1/W64, timan107.
VLA retains the native source environment; no new MPS setting is introduced.
R17 execution topology is recorded through the existing worker-count knob:
timan1 workers are reassigned to timan108, preserving total W, server R/B/MPS,
sealed canonical topology, serving identities, plans and filtered rows.

## Allowed references and dependencies

R17 pure A references (native loader follows continuation ancestry):

| Model | Reduced | Full |
|---|---|---|
| DP | hs7dpctl_ddim10 | hs7dpctl_full |
| Fast-WAM | hs7fwk1_4090c | hs6fw_res_152931 |
| Cosmos | hs2_cosmos_h100_k2incl_c16_t107_20261010_115120 | formal_cosmos_libero_official2000_20261010_130639_a8ca2043 |

R17 off: original DP fsreddp2, FW fsredfw; original Cosmos is the already queued
ffk1015_cosmos_libero_reduced_off_u. Filtered off reuses the already existing
ffk1015_<model>_libero_reduced_off. Do not rebuild/reprepare any of these.
VLA reduced off: the standalone group.json files under r16/k1ref/runs/off and
filter_failed_knob/vla/runs/off. Reduced pure: R16 design.json phase-1 pure-A
records. GR00T full pure: the R14 pure A records explicitly named by the R15
static c10_proof.json, on the matching 4090 environment.

**π0.5 full has an unresolved same-GPU reference.** R15 official A used H100;
its R14 pure A reference actually used local RTX 4090. Metadata-only searches
found no existing full-step H100 pure A reference. No new pure campaign has
been defined or staged. The π0.5 full grids are defined and CPU checked, but
hold their scheduling until the coordinator supplies an existing matching H100
pure reference or authorizes two library-independent 500-episode pure-A jobs.
Analysis leaves those eight paired cell rows pending; it never substitutes
4090 or B-pool outcomes. An existing approved reference can be registered in
`PURE_A_OVERRIDES.json`, keyed `pi05:full:libero_10` and
`pi05:full:libero_spatial`, with `{ "root": "...", "arm": "..." }` values.

## Exact coordinator commands

[COMMANDS.md](COMMANDS.md) lists the literal prepare/launch pair for every profile.

All paths are absolute in the scripts. Run from /home/weiland/projects/openpi.
Definitions are already built. **Do not run filter_failed.build, the historical
filter_failed_knob build/workflow, or formal_a build again.** Existing queues
and launched runs were left in place. Prepare and launch only these fresh runs.

```bash
cd /home/weiland/projects/openpi
FFKA=exp/offline_search/rounds/r17/filter_failed_knob/formal_a_1016

# Choose one row of the table above. The scripts serialize its cell campaigns.
# Models: dp fastwam cosmos pi05 groot. Modes: reduced full.
# Arms: original filtered. π0.5 full is on reference hold as described above.
MODEL=dp
MODE=reduced
ARM=original
bash "$FFKA/commands/prepare_profile_${MODEL}_${MODE}_${ARM}.sh"
bash "$FFKA/commands/launch_profile_${MODEL}_${MODE}_${ARM}.sh"

# Or one launch unit, using its exact name from CAMPAIGNS.json:
JOB=ffk1015_dp_libero_reduced_formal_A_filtered
bash "$FFKA/commands/prepare_${JOB}.sh"
bash "$FFKA/commands/launch_${JOB}.sh"

# One analysis entry point for all models/modes/both arms (CPU only):
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src   .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze

# Refresh estimates after adding authorized source timing metadata:
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src   .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a inventory
```

R17 prepare/launch invokes the existing ffk stage/launcher and frozen bootstrap.
R15 invokes the existing legacy sync/chain with exact source server/worker env.
R16 invokes the existing generic resident conductor in test phase (every fixed
level independent), from the immutable openpi_r16_k1ref_server tree, with its
existing public-network rsync transport wrapper. It does not bootstrap an old
inventory or consult dev selections. Its prepare script performs the CPU check;
native conductor staging happens at coordinator launch. No worker source edit
or deployment rsync is needed from this change.

`RESULTS.json` contains one row per 104 model/mode/suite/size cells: original and
filtered kept counts, empty tasks, per-task levels and first failure, final SR,
measured IR, per-level measurements, completeness, and paired
filtered−original differences overall, by initial state and by task/init.
A missing cohort/reference stays pending. R17 continuation tips are resolved
through the existing filter_failed continuation registry, so ancestors are
merged without double counting. Off data and pure inference are never filtered.
IR is measured with the standard R17 latency coefficients or native VLA D5
ledger; nominal ladder targets are never recalibrated.

## Recommended scheduling

Finish existing queues first. Check the current chain processes and
openpi_ext/scratch/r17_ops/ff/run_*.sh before adding work. Full ordered launch-unit
names, episode counts and resources are in commands/4090_ORDER.txt and
commands/H100_ORDER.txt; LAUNCH_ORDER.json enumerates node conflicts.
Within a profile run original then filtered, retaining identical serving env.

- RTX 4090: DP reduced → Fast-WAM reduced → π0.5 reduced → GR00T reduced →
  DP full → GR00T full.
- H100: Cosmos reduced → Fast-WAM full → Cosmos full → π0.5 full (reference hold).
- One campaign per server GPU. Jobs sharing timan107 or timan108 must not overlap
  their frozen EGL leases. DP's local workers allow overlap with an H100 fleet;
  4090 VLA timan107 fleets can overlap π0.5 H100 timan108 fleets once its reference
  issue is resolved. Never use timan1. Worker relocation is owner-authorized;
  do not interpret a changed worker machine as an experimental arm.

## CPU evidence

`CPU_CHECKS.json`, per-job cpu_checks/, PLAN_IDENTITY.json and native plans/
record startup/descriptor, planner and original-table equivalence. All 48
launch units pass; all 24 original arms have equal serving tables.
PLAN_MATRIX.json proves 544,450 new + 47,550 reused off = 592,000 unique
semantic level/task/init slots, zero overlaps, all 1,184 level/cells balanced
at 10 tasks × 50 inits, and identical canonical coverage across the arms. R17 probes
use the exact chosen sealed-version server interpreters, strict native loader
identity paths and recipe construction for every new grid group, with no GPU.
H100 uses in-memory YAML path substitution onto verified source input bytes;
no remote staging/write is performed. All 12 match their source actual
ready.json identity. FW full uses installed knobtestfw2 for the CPU probe only,
after proving its descriptors, all required inputs, runtime, R/B/MPS equal the
uninstalled final knobtestfw3 definition. FW reduced uses the published official
A runtime archive a7a387... rather than its campaign's later mutable fwtextcache
pointer; its native identity now equals the source ready.json. No check was
weakened. Both corrections affect only fresh formal_A definitions/probes.

VLA checks run the native PluginRuntime artifact identity path and source-A
model/checkpoint/step argv equality. GR00T checks also run under its actual
serving interpreter. All 144 reduced-VLA grid entries additionally pass the
native plugin path from the immutable R16 source serving tree; FROZEN_R16_*.json
record the interpreter, source module and each fit identity. Native R15 sync plans are exercised with process/network
launch operations blocked; R16 native TaskGraph checks verify unique episodes.
Original R15 fit bytes equal the source; original R16 inner retrieval/fitted
objects serialize byte-identically, with only the authorized uniform level
settings changed. All four source R15 RUNTIME_SOURCES.sha256 entries match current local
and H100 code (read-only inspection), recorded in R15_RUNTIME_MANIFEST_CPU.json. Model weights are not loaded on a GPU; these are startup-input checks,
not simulated outcome claims.

The targeted test covers the integer 2-success boundary, first-level pure
fallback, ignored lower recovery, cell off override, missing init and VLA's
terminal off name. Exact rerun:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src   .venv/bin/python -m pytest -q -p no:cacheprovider   exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a.py
```

No servers, workers, drivers, chains, GPU work, package installs, commits,
deletions or data moves were performed. The first argv-only VLA check called a
native helper that created empty runtime directories in two withdrawn source
R16 roots (`runs/resident_cpu_identity_0/runtime`, no files). Subsequent checks
redirect helper outputs into this new CPU directory. No source definition,
asset, journal or outcome was changed; the empty directories were left in place
under the no-delete rule.
