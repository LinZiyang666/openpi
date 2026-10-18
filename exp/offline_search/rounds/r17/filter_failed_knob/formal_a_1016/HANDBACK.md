# Formal test-A knob queue — coordinator review, 2026-10-16

2026-10-17 18:5x: [Prepare repair and capacity-checked legacy sharing](prepare_1017_185x/HANDBACK.md).
Continuation closure no longer falsely enters the GPU2 runtime bootstrap.
Native closure passed for both parents (Cosmos 3,015 remaining; FW 13,190).
R15/R16 future formal-A fleets retain native plans/serving and use scoped
worker locks. The handback records capacity snapshots, read-only deployed-helper
verification, exact existing commands and the H100 replacement queue order.

2026-10-17 18:3x: [Control-probe repair and continuation handback](probe_1017/HANDBACK.md).
The new UNKNOWN policy prevents single-probe retirement. The replacement
Fast-WAM continuation has 13,190 episodes; Cosmos has a live-source draft that
refreshes at prepare after closure. Exact stop PIDs/order and commands are linked.

2026-10-17 15:3x: [4090 capacity and infra-error repair](infra_1017/HANDBACK.md)
records the whole-server reservation, queued local-slot table, six exact
`fa_fw_r_u_lc1` episodes and automatic continuation merge. Infra-error outcomes
remain missing until repaired; source runs and running chains are unchanged.

2026-10-17 09:4x: [H100 asset staging repair](h100_assets_1017/HANDBACK.md) fixes
the missing short-name `assets/full.yaml` on H100 while preserving the old
recipe addresses and all scientific bytes. See its reset status, read-only
proof, exact commands and remaining H100 queue order before launching.

2026-10-17: all active and deferred launch units now use short `fa_` names to
satisfy the frozen MPS pipe limit. The [current handback](short_names_1017/HANDBACK.md)
and [NAME_MAP.json](NAME_MAP.json) supersede the old naming/count snapshots below.
There are 34 units, 48 built native runs (163,950 episodes), and 42 possible
deferred native runs (upper bound 322,000 episodes). All 90 native/template
startup and path checks pass. The failed Cosmos run is UNSTAGED; its short
replacement is `fa_co_r_u_l`. π0.5 pure_l10 is `fa_pi_f_p10`, a whole 500-episode
replacement whose old outcomes must not be merged. R15/R16 host selection is
retained. Existing running chains and old command files remain unchanged.

This replaces the full-grid draft. Both library arms use the accepted formal
per-task first-failure rule on A inits 0–49 (50/task/level, deficit ≤2 successes
versus same-step same-GPU pure). `choose()` is unchanged. The whole-cell off
rule now prevents dispatch: complete off SR ≥ pure SR omits every non-off level
of that cell. An incomplete off cohort is pending, never a non-qualifying cell.
No dev outcome or dev-chosen level informs dispatch or selection.

`QUEUE.json` contains **34 queue NAMEs**. Nineteen units are built, with 47 native
campaigns underneath. Fifteen level units are deferred until off is complete.
`CAMPAIGNS.json` lists active native children; `QUEUE_DRAFTS.json` preserves the
old, never-launched full-grid definitions as immutable inputs. Do not launch
old full-grid/profile scripts. Use only the NAMEs in the two ordered lists.

## Episode counts

| Model | Mode | Built original / filtered | Deferred maximum original / filtered |
|---|---|---:|---:|
| dp | reduced | 5,000 / 5,000 | 0 / 0 |
| dp | full | 6,000 / 6,000 | 30,000 / 30,000 |
| fastwam | reduced | 30,000 / 22,500 | 0 / 0 |
| fastwam | full | 6,000 / 6,000 | 30,000 / 30,000 |
| cosmos | reduced | 0 / 0 | 30,000 / 30,000 |
| cosmos | full | 6,000 / 6,000 | 30,000 / 30,000 |
| pi05 | reduced | 2,500 / 14,000 | 16,000 / 0 |
| pi05 | full | 4,000 / 4,000 | 16,000 / 16,000 |
| groot | reduced | 1,950 / 0 | 16,000 / 16,000 |
| groot | full | 4,000 / 4,000 | 16,000 / 16,000 |

Built total: **133,950**, including **1,000 shared pure-reference episodes**.
Deferred upper bound: **352,000**. Grand upper bound: **485,950**; final totals
fall as off cells qualify. Exactly **59,500 non-off episodes** were removed
now, from 24 completed reduced-mode cells. No source off episode is rerun.

Pruned: both DP arms omit sizes 100/200/300/400/500 in both suites (ten cells
per arm); filtered Fast-WAM omits Spatial 100/200/300; filtered π0.5 omits
L10 200. `off_decisions/*.json` gives the native-loader success counts, pure
counts, completeness and decision. The just-measured review audit is retained
with hashes. At later builds, `evaluate()` reads the permitted current off/pure
lanes afresh, including R17 registered continuation tips.

A source fact differs from “reduced off already measured for both arms”:
Cosmos original/filtered have no launch/receipt; filtered GR00T has no DONE.
Those existing jobs are still in the coordinator's old queues and are untouched.
Original reduced VLA off covers only part of every cell: π0.5 1,500/4,000,
GR00T 2,050/4,000. New off_complete units run only the missing 2,500 / 1,950
initial states, using the existing R16 fits/pipeline. Their non-off grids wait
until those completed cells can be classified. Counts above report ready
work and honest upper bounds, rather than assuming missing off outcomes fail.
Filtered Fast-WAM's receipt is FAIL after 6,000 accepted episodes; the standard
loader supplies complete, error-free per-cell outcomes, which are reused.

Each full mode has two NAMEs per arm: `_off` first, then `_levels`. R17 off has
6,000 episodes/profile; VLA off has 4,000/profile. R15 VLA units serialize the
eight unchanged native cell campaigns (500 each). After off analysis, only
non-qualifying cells enter the second stage: 2,500/cell for R17 (five levels),
2,000/cell for VLA (four levels). No off episode is repeated in stage two.
An all-qualifying level unit returns success without starting any processes.

## Authorized π0.5 H100 references

- ffk1015_pi05_libero_full_formal_A_pure_l10 — 500 episodes.
- ffk1015_pi05_libero_full_formal_A_pure_spatial — 500 episodes.

Both are ordinary R15 pure-C10 arms on the full grid's H100 R3/B1, timan108×39,
source interpreter/checkpoint/environment, A task/init manifest and rollout
seed. They use the unchanged original pure fit (R13 periodic K=0, the native
pure-C10 implementation); no candidate filtering or fitting is applied.
`PURE_A_OVERRIDES.json` registers them for both arms and every size. π0.5 full
has **no authorization hold**. Its level build naturally waits for these
reference outcomes and its own off outcomes, like every formal second stage.

## Workers and serving

| Model/mode | GPU | R/B/W | Execution workers |
|---|---|---|---|
| DP both | RTX 4090 | 4/8/48 | local×24 + timan1 GPU2×24 |
| Fast-WAM reduced | RTX 4090 | 2/16/70 | timan1 GPU2×30 + timan107×20 + timan108×20 |
| Fast-WAM full | H100 | 2/16/85 | timan1 GPU2×35 + timan107×25 + timan108×25 |
| Cosmos both | H100 | 8/1/85 | timan1 GPU2×35 + timan107×25 + timan108×25 |
| π0.5 reduced | RTX 4090 | 3/1/64 | timan107×64 |
| π0.5 full / pure | H100 | 3/1/39 | timan108×39 |
| GR00T both | RTX 4090 | 4/1/64 | timan107×64 |

R17 uses the existing `resume.topology` / `worker_topology.redistribute` knob
with the merged timan1_gpu2 coordinator. Every R17 unit has at least two worker
machines, all timan1 leases specify device 2, cap 83, excluded GPUs 0/1/3, and
original total W is retained. Native R15 permits exactly one WORKER_HOST per
cell; native R16 permits its timan107 agent island. Those pipelines retain the
source single-host fleet; no multifleet VLA method/launcher was invented.
Worker resilience may redistribute at actual launch and records the allocation.

The original local-only DP endpoint 23400 is outside the public direct-port
range needed by remote workers. Only fresh DP units use public 23160 and
backends 23161–23164, driver 23110. This is recorded in
`formal_A_endpoint_override`; R/B/W, runtime, batch treatment, model identity,
plans' semantic task/init/seed identities, fits, corrector and stall remain
unchanged. All other source serving settings and GPUs remain unchanged.

## Exact commands and ordering

The [literal command index](COMMANDS.md) contains every NAME's prepare/launch
pair and all ten full second-stage build commands. Parent aliases are at
`exp/offline_search/rounds/r17/filter_failed_knob/commands/`, the directory
already hardcoded in chain_ffk.run.sh. Native cell campaigns also have standard
prepare/launch aliases, for targeted recovery; do not schedule both a profile
controller and its native children. Each profile alias calls the queue dispatcher,
which invokes the existing native stage/launcher, R15 sync/chain or R16 resident
conductor. R15 cell children run sequentially under their source topology;
its native chain already skips completed arms on retry.

Prepare of a deferred `_levels` NAME evaluates off, builds its selected cells,
then prepares the native campaigns. It cannot dispatch a cell with incomplete
off or reference data. Explicit CPU-only build for any full profile:

```bash
cd /home/weiland/projects/openpi
MODEL=dp
ARM=original
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels \
  --model "$MODEL" --mode full --arm "$ARM"
```

Models: dp, fastwam, cosmos, pi05, groot; arms: original, filtered. The same
command with `--mode reduced` finalizes the five reduced profiles that still
lack complete off evidence. `publish` refreshes totals and ordered lists,
without rewriting identical commands or any prepared/launched native run.
Do not run init again, filter_failed.build or the old knob builders.

After existing queues finish, use the exact ordered NAMEs in
[H100_ORDER.txt](commands/H100_ORDER.txt) and
[4090_ORDER.txt](commands/4090_ORDER.txt). Copies live beside the parent aliases
as `formal_A_1016_H100_ORDER.txt` and `formal_A_1016_4090_ORDER.txt`.
Pure H100 references come first; reduced precedes full; every off/off_complete
precedes its levels. Units sharing a worker node must respect its EGL leases;
worker resilience and native admission determine actual concurrent capacity.
No chain, server, worker, driver, prepare or launch was executed here.

Analysis remains one CPU entry point, now merging active stage children and
permitted standalone off references, while excluding the superseded full grids:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

It writes the same 104 cell rows with per-task levels, SR, measured IR and
paired filtered−original differences by initial state. Off and levels merge
by task/init without double counting; standard R17 continuation tips include
off lanes. Selection and tolerance are unchanged. A cell intentionally pruned
at off can be scientifically complete while `grid_complete=false` (unused
levels are absent).

## CPU evidence

CPU checks exercise each built native plan, frozen startup inputs/identity,
source model/checkpoint/step argv, original serving tables and actual GR00T
interpreter. The pure fits/plan match the source pipeline and the grid's H100
serving settings. R17 startup reads all actual new group recipes. The targeted
off-dispatch test distinguishes whole-cell equality from per-task tolerance
and leaves partial cohorts pending. Script syntax and chain alias existence
are checked. Updated records are CPU_CHECKS.json and QUEUE_CPU_PROOF.json.

No source library, fit, calibration, existing ff/ffk run or running/queued
command was rebuilt/re-staged. No GPU work, package installs, commits,
credential access, data deletion/move or remote write was performed.

Final CPU result: **47/47 startup/plan checks PASS**, 23 original serving-table
checks PASS, two targeted tests PASS, analysis smoke **104 rows**. Plan coverage
is **181,500 unique slots** (133,950 new including pure; 47,550 planned reused
off), zero overlaps; **59,500 pruned non-off slots are absent**. All 66 distinct
controller/native NAMEs have standard parent and local prepare/launch aliases.
No new campaign has PREPARED/LAUNCH/RECEIPT/DONE claims.
