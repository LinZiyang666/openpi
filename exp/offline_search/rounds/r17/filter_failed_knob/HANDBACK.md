# Filter-failed full-system knob ablation — 2026-10-15

2026-10-17 18:5x: [Continuation prepare and shared legacy fleets](formal_a_1016/prepare_1017_185x/HANDBACK.md).
Both native parent-closure probes now pass through stage's GPU2 wrappers.
Cosmos prepare will freeze 3,015 remaining UIDs; Fast-WAM retains 13,190.
Future R15/R16 formal-A launches use scoped worker locks and measured capacity,
so an unrelated timan107 fleet's host lock does not block them. The linked
handback gives the unchanged command pairs and H100 re-queue order.

2026-10-17 18:3x: [Control-probe UNKNOWN policy and replacement continuations](formal_a_1016/probe_1017/HANDBACK.md).
Worker status uses 90 seconds; UNKNOWN is retained for five minutes and reset
on successful status. Fast-WAM source closure is proved; `fa_fw_r_u_lc1` now
contains 13,190 episodes and supersedes the archived six-episode draft.
`fa_co_f_f_oc1` has a 3,333-episode draft; prepare freezes its final remaining
UIDs after the coordinator stops the still-running source through native cleanup.

2026-10-17 15:3x: [4090 server reservation and infra-error continuation](formal_a_1016/infra_1017/HANDBACK.md).
The local probe now reserves the sealed whole-server budget. The six-episode
`fa_fw_r_u_lc1` draft is registered and awaits its parent's closure before
prepare. Infra errors remain missing until valid continuation outcomes arrive;
25 other completed formal-A lanes/cells have none of these errors. See the
dated per-campaign capacity table and exact commands in that handback.

2026-10-17 09:4x: [H100 asset staging repair and remaining queue](formal_a_1016/h100_assets_1017/HANDBACK.md).
H100 preparation now mirrors the sealed assets at both the pinned recipe address
and the short `local_assets` address. Original Fast-WAM full off was reset and
prepared successfully; read-only H100 checks pass for 334 asset hashes and 77
referenced inputs. Filtered full off still needs the remote rename-only reset
exception described in that handback. No scientific inputs were rebuilt.

2026-10-17: all formal-A launch units now use short `fa_` names. The
[current handback](formal_a_1016/short_names_1017/HANDBACK.md),
[NAME_MAP.json](formal_a_1016/NAME_MAP.json), and regenerated GPU order lists
supersede earlier naming/count snapshots below. All 90 built/potential native
paths and startup identities pass CPU checks. The failed Cosmos run is
UNSTAGED and rebuilt as `fa_co_r_u_l`; π0.5's 500-episode pure_l10 replacement
is `fa_pi_f_p10` (never merge the unreachable old outcomes). Running chains
and old command files remain unchanged; R15/R16 host selection is retained.

## Queue revision — coordinator review, 2026-10-16

The active [queue handback](formal_a_1016/HANDBACK.md) supersedes the earlier
full-grid instructions. Two 500-episode π0.5 H100 pure-A references are authorized
and registered; the hold is released. Off-qualified cells are removed before
dispatch. Full modes run off first, then build only non-qualifying cells.
There are 34 chain NAMEs with parent-directory prepare/launch aliases: 133,950
episodes built and at most 352,000 deferred. Source off data is still incomplete
for Cosmos, filtered GR00T and the original reduced VLA partial cells; exact
dependencies and off-completion jobs are in the handback. R17 fleets now include
timan1 GPU2 (cap83) and multiple worker machines. Existing queues are untouched.


CPU preparation only. The coordinator stages and launches every campaign. No
server, worker, driver, chain, GPU operation, package installation, commit, or
remote write was performed here. Existing `ff1015_*` runs, their definitions,
views/fits/studies/campaigns, and the coordinator's extended `filter_failed/reset.py`
were left alone. No worker source files changed; no worker rsync is needed.

The coordinator launched `ffk1015_dp_libero_reduced_off` at 21:44:36 while this
handback was being completed. **Do not prepare or launch that run again.** A
later bootstrap-freeze pass here added `assets/ffk_bootstrap.py` and its campaign
hash after that launch; it should have refused a launched run. Candidate
selection and model inputs are unchanged. The freeze function now refuses
launched runs, with a CPU regression test. The run was not reset, stopped or
restored. `LAUNCHED_RUN_AUDIT.json` records the change. Its last inspected receipt
was `PENDING_EVIDENCE`, with owned processes closed and GPU released; that is
not a claim of scientific completion. The native launcher itself updated
`EPISODE_PLAN.json` at 21:44:53, before the later bootstrap freeze.

## Formal design (owner ruling)

- No development set for IR knob tuning on any benchmark. The cache library is
  built from the development pool (LIBERO B pool, MetaWorld seed 8), so tuning
  on it means tuning on the library's own data.
- The per-task IR knob is descended directly on the test set: LIBERO uses the
  whole test set A (500 episodes per suite = 10 tasks × 50 initial states,
  indices 0–49); MetaWorld MT50 uses seed 7 (50 tasks × 50 initial states).
- For every task, measure the levels on the test set from the most expensive
  level downwards, 50 episodes per task per level. A level passes when it has at
  most 2 fewer successes than same-step-count pure inference on the same GPU and
  the same initial states (integer comparison). The first failing level freezes
  the previous level. The reference is always same-step-count pure inference
  (reduced-step knob vs reduced-step pure inference, full-step vs full-step).
- Library cells whose complete system with the knob off (stall gate only)
  already reaches the same-step pure-inference success rate are not tuned; they
  use that knob-off result.
- All earlier LIBERO knob results that chose levels on a development set (then
  tested or raised on test set A) are withdrawn and are being deleted and re-run
  under this design: original and success-filtered libraries, all models (DP,
  Fast-WAM, Cosmos, π0.5, GR00T), full and reduced steps. MetaWorld knob results
  already followed this design and stay valid.

In this handback the MT50 path already follows the formal design. Every LIBERO
dev-set step below (R17 B inits 0–29 dev campaigns and `workflow descent`, the
A `build-test`/`test` raise generations, the R16 `vla.sh dev` waves, `freeze`
and `test` raise loop, and the R15 full-step dev roots) is withdrawn: do not
prepare or launch it. Knob-off A campaigns remain valid, because they decide
which cells need no tuning under the formal design.

## Method and serving inputs

`frozen_fit_candidate_rows_only`, without top-up or refitting. R17 residents
load the original size-N fit, then call the existing
`filter_failed.row_filter.r17_candidates`. Original view, metric/PCA/whitening,
corrector, Q3 files and every calibrated periodic gap remain byte-identical.
The original complete fit attests the original Q3 metric; retrieval uses the
candidate subset. The filtered views already produced by `filter_failed/` are
referenced in each campaign's `filter_failed_knob.cells` metadata. The recipe's
store descriptor continues to identify the original size-N fitted store.

The small row-filter bootstrap is frozen in each new run's
`assets/ffk_bootstrap.py`, included in standard preparation seals, and passed to
the original resident entry point. Neither the native runtime nor workers are
replaced. The original DP admission bridge is reused only where the source
actually recorded that bridge. Per-job `server_python` selects the source
interpreter; there are no global environment changes.

Empty R17 candidate sets, including a singleton removed by B LOEO, return an
explicit model MISS. MT50 empty tasks run all 50 test episodes. The serving
change permits precisely that empty-candidate MISS and forces a new model
decision instead of taking a cache tail. It does not invent a donor, correction,
stall observation or nominal-target calibration. The real serving CPU test
checks two successive decisions both execute the model with CUDA uninitialized.

The frozen `assets/ffk_evidence.py` extends only the native CPU post-run audit's
ordinary-MISS condition. It reconstructs empty eligibility from original
episode labels, the sealed candidate selection and the planned LOEO init; those
policy calls retain their actual NFE/control counts. All other native audit
conditions remain. The same small extension runs in either inline or background
CPU finalization. No worker evidence field or worker archive changes.

R16 uses the original pickled classes. Only candidate arrays in
`method.inner.base.tasks` and its LOEO `_full_tasks` candidate snapshot are
sliced. Corrector, guard tables, normalizations and frozen per-task caps stay
unchanged. Pure B references and phase-1 pure A references are reused; previously
measured pure-C10 cohorts can also be reused. No old cache/off outcome enters
the new filtered inventory. R16 has no zero-success task in these selected sizes.

Measured IR uses the original `knob/ir_formal.py` profiles for R17, including
LIBERO cached-text pricing and the separate MT50 profiles. R16 retains its D5
pricing. No target is recalibrated after filtering.

## Source audit and decisions still needed

* MT50 is seed-7 A50, library seed 8, no LOEO, six sizes and six non-pure levels.
  There is no new dev set. π0.5 is crg/C5/K10 or K2; GR00T chrg/C10/K4 or K2;
  DP chrg/C10/DDPM100 or DDIM10; Fast-WAM chrg/C10/K10 or K3.
* π0.5 full MT50 is split by **actual accepted episode origin**, retaining the
  oldest execution on equal-timestamp copied resume rows: `pimk1_ful` 4,006;
  `pimk1_fulb` 1,761; `pimk1_fulc4` 502; `pimk1_fuld` 83,731. Total 90,000.
  See `MT50_PI05_FULL_ORIGINS.json`. The superseded whole-lane draft was renamed
  with `.DRAFT_WHOLE_LANE_NOT_FOR_LAUNCH`; it is absent from all command lists.
* MT50 pure-reference overrides come from the original result headers:
  DP full `tdp7h1_full2`; GR00T full/reduced `tgr7h1_full` / `tgr7h1_k2`.
  The workflow supplies these to the unchanged `descent.py --control` path.
  Other references retain the source defaults. Fast-WAM reduced's unfiltered
  result can remain pending until `fwmk1_redx` finishes.
* R17 LIBERO was prepared here with B inits 0–29, LOEO, the 5 pp first-failure
  ladder and the A raise rule; that dev-set procedure is withdrawn. Under the
  formal design the filtered LIBERO knob is descended directly on test set A,
  50 episodes per task per level, with at most 2 fewer successes than
  same-step pure inference on the same GPU and A initial states.
  Cosmos keeps the source's literal `crg_*` names: `cache_study.full_group`
  selects `crg` because its native horizon is already 16; `chrg` is the alias
  with the same corrector, Q3 gate and C16 settings.
* Reduced knob-off A is defined for DP/Fast-WAM/Cosmos. Cosmos includes both a
  filtered run and `_u` unfiltered control, each 6,000 episodes. Each arm's
  whole-cell no-knob decision is evaluated independently against the same pure
  reference. Qualifying cells use their measured off results and need no knob A
  test. Fast-WAM reduced A uses `hs7fwk1_4090c`; pure references are not filtered.
* R16 k1ref requires completed knob-off A **before** emitting the missing dev
  wave (`workflow.operational_dev_selection`). Therefore off A is ready now;
  the exact first-wave task set is emitted after those outcomes, rather than
  guessing which cells satisfy the no-knob rule. A fresh filtered dev inventory
  needs at most five waves (ir45, ir40, ir32, ir25, off), whereas the old
  supplement needed at most four because it reused already measured cache levels.
  These dev waves are withdrawn under the formal design; only the knob-off A
  result is kept.
* **Decided 2026-10-16: LIBERO π0.5/GR00T full steps use R15.** Filtered arms
  cover 50/100/200/500; no unfiltered knob arms are rerun. Size 50 compares
  with existing R14, explicitly crossing R14→R15; sizes 100/200/500 compare
  with existing R15 results. Size 50 uses existing
  R13 catalog artifacts through R15's unchanged size-agnostic serving code.
  The R16 full-step alternative has not been built. Commands, counts and
  CPU evidence: [R15 full-step handback](vla_full_r15/HANDBACK.md).
  Withdrawn under the formal design: R14/R15 chose levels on B-pool dev
  inits, so their results are withdrawn and the R15 full-step dev roots are
  not to be launched.
* **Pending owner interpretation: terminal pure deficits.** The brief says
  “until all pass”; the source `test.raised_level` and k1ref `raise_level` stop
  raising an already-pure task even when it remains below tolerance. The source
  report records such residual violations. Current analysis preserves the source
  rule and reports `all_tasks_within_tolerance` separately. No repeated pure
  retest or different terminal rule was introduced; that decision is pending.

## Library provenance

For R17, success is `READY.json#metadata.episodes[].success`, selected by the
original size view's `episode_ids`. The exact original and filtered view paths,
labels and kept IDs are imported from `filter_failed/CELLS.json`; nothing there
is rewritten.

For **every R16 cell**, the active library is the pickled method's `library`
under `/home/weiland/trace_runs/offline_search_store/library/<model>_<suite>/`:
π0.5 `bpool_cs`, GR00T `bpool_all`. Success is `episodes.json[].success`.
The legacy arm row's SP16 `library` field is not the active retrieval library.
`vla/CELLS.json` records each exact off fit, original episode set, parent and
labels. The audit checks the fitted `lib_ep` equals that parent's `episode.npy`
indexed by the fit's `row_subset`; its lazy action payload is the same parent's
`action.npy`. Dev-level/A source definitions remain in the original sealed
k1ref inventory and are used directly by the new emitter.

## CPU validation

`VERIFICATION.json` checks all R17 copied source artifacts and archives, and
no-filter serving-input equality in R17 MT50, R17 LIBERO and R16 VLA. R16's NaN
reset sentinel is compared by scalar bytes. `cpu_tests_final.log` includes the real
empty-MISS serving test and local/remote launcher argument tests.
`STUDY_PROOF.json` records all 21 native study validations;
`FINAL_PLAN_PROOF.json` records episode counts, direct routes, frozen resident
bootstrap bytes and the source's policy RNG namespace. The four π0.5 full
studies each list a configuration once while their dispatch seals retain the
distinct task/init cohorts. `EMPTY_MISS_EVIDENCE_PROOF.json` counts eligible
empty-MISS episodes without simulating them.

`startup_checks/` and `STARTUP.json` record frozen-loader descriptor identity,
interpreter package versions, native DP startup admission, all full-system
recipe/fit/Q3 loads, and equality to the source's actual serving specification.
Remote probes read source assets whose model descriptor bytes equal the new
local copies; they do not stage new remote files. The source `knobdevdpred2`
never reached a server; `knobdevdpred3` has the same runtime and descriptor bytes
and provides that lane's actual ready evidence.

Two source metadata differences are preserved rather than repaired:
`pimk1_fuld` records the updated code/preprocess identity in `ready.json`, and
Fast-WAM's local text-cache runtime has one changed `formal_batch_server.py`
hash relative to its old expected descriptor. Their computed specifications
are compared to the **actual source ready specifications**. The frozen native
loader's existing checks are retained, including any existing record-only
content check; this work does not disable an identity check.

R16's actual off registries load on CPU for both models in the sealed snapshot
(`vla/cpu/*/PROOF.json`), without constructing a policy/model backend or a
server. The timan107 island's 2,882 worker files match its SHA manifest:
`R16_WORKER_PROOF.json`. No worker deployment or rsync was performed.

All 21 R17 startup probes passed. Nineteen equal their source's actual ready
specification; the two Cosmos off arms have no launched source and are paired
with one another. Eight focused CPU tests pass, including native audit
acceptance of an empty MISS and rejection of that MISS with a donor present.
Private future-stage fixtures
also pass: six R17 official-A template builds (100 planned episodes each) and
one R16 next-wave build (120 planned episodes). These fixtures are excluded from
launch inventories and were never simulated; see `cpu_future_r17.log` and
`cpu_future_r16.log`.

## Exact prepare / launch / analyze commands

Run from `/home/weiland/projects/openpi`:

```bash
D=exp/offline_search/rounds/r17/filter_failed_knob
M=exp.offline_search.rounds.r17.filter_failed_knob
CPU=(env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python)

# Definitions and commands already exist; use them directly.
# Do not rebuild/replan launched runs. Do not run filter_failed.build,
# k1ref bootstrap, or vla initial again.

# CPU verification (no staging):
"${CPU[@]}" -m "$M.verify"
"${CPU[@]}" -m "$M.startup"
"${CPU[@]}" -m pytest -q "$D/test_runtime.py" -p no:cacheprovider
bash "$D/vla.sh" verify

# A single R17 campaign: repeat using an exact name from ALL_CAMPAIGNS.json.
RUN=ffk1015_fastwam_libero_reduced_off
bash "$D/commands/prepare_${RUN}.sh"
# Coordinator only, after the GPU and worker nodes are available:
bash "$D/commands/launch_${RUN}.sh"
```

Prepare scripts use the existing standard CPU stage code. Launch scripts clear
the CPU-only CUDA/JAX masks and enter the standard prepared launcher. The
native source replicas, batch/memory settings, MPS and worker archives are
preserved, with the required current direct routing: obsolete full-dev broker
addresses are replaced by standard `formal_eval.open_tunnels` direct endpoints
(that function performs no expose operation). LIBERO driver ports are the
RUN_CONFIGS knob allocation: DP 23110, H100 Fast-WAM 23111, Cosmos 23112, local
Fast-WAM 23113. Native server/backend ports, model settings and R/B/W stay at
their source values. No launch script was run during preparation.

MT50: after **all source-origin jobs for that model/mode** complete, run the
unchanged direct-test descent through the wrapper:

```bash
for MODEL in pi05 groot dp fastwam; do
  "${CPU[@]}" -m "$M.workflow" descent --benchmark mt50 --model "$MODEL" --mode reduced
done
# Repeat with --mode full after those campaigns complete.
```

R17 LIBERO, **withdrawn under the formal design** (record only; do not run the
dev campaigns, B descent or raise generations below; the off A campaigns stay
valid): run the new off A campaigns and the dev campaigns in the resource
order below. After a model/mode's dev and any required off A finish:

```bash
MODEL=dp                       # dp | fastwam | cosmos
MODE=reduced                   # reduced | full
"${CPU[@]}" -m "$M.workflow" descent --model "$MODEL" --mode "$MODE"
"${CPU[@]}" -m "$M.workflow" build-test --model "$MODEL" --mode "$MODE" --generation 0
# If every cell uses measured off A, build-test prints that fact and emits no run.

GEN=0
RUN=$(printf 'ffk1015_%s_libero_%s_test%02d' "$MODEL" "$MODE" "$GEN")
"${CPU[@]}" -m "$M.startup" --run "/data/mw_finetune/eval/$RUN"
"${CPU[@]}" -m "$M.stage" --run "/data/mw_finetune/eval/$RUN"
# Coordinator launch, with CPU-only masks removed:
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m "$M.launch" --run "/data/mw_finetune/eval/$RUN"
"${CPU[@]}" -m "$M.workflow" test --model "$MODEL" --mode "$MODE" --generation "$GEN"

# Only when raise.json contains tasks: increment GEN, then build -> prepare ->
# launch -> analyze the next generation. It contains only raised 50-episode cohorts.
GEN=$((GEN + 1))
"${CPU[@]}" -m "$M.workflow" build-test --model "$MODEL" --mode "$MODE" --generation "$GEN"
# Repeat the RUN/startup/stage/launch/test commands above for this GEN.
```

The official-A template uses the source test campaign's runtime/topology, with
the original dev ladder's fit/Q3/gap artifacts. Existing source raise releases
are reused where that round exists; subsequent rounds retain the source test
template. Reduced Fast-WAM keeps its 4090 text-cache lane. DP full additionally
used the same-topology `hs7dpctl_full` reference after its original-control
raise loop; reproduce that source phase with a fresh analysis generation:

```bash
# Choose the next unused GEN after the original-control raises finish.
"${CPU[@]}" -m "$M.workflow" test --model dp --mode full --generation "$GEN" --control DP=hs7dpctl_full
# If it raises tasks, build the following generation and follow stage/launch/test.
```

R16 π0.5/GR00T reduced (the off A commands stay valid; the dev waves, `freeze`
and the `test` raise loop are **withdrawn under the formal design** and must not
be run):

```bash
# Off A is already defined: 4,000 per model, not the old supplement's 3,550 total.
bash "$D/commands/prepare_ffk1015_pi05_libero_reduced_off.sh"
bash "$D/commands/launch_ffk1015_pi05_libero_reduced_off.sh"
bash "$D/commands/prepare_ffk1015_groot_libero_reduced_off.sh"
bash "$D/commands/launch_ffk1015_groot_libero_reduced_off.sh"

# Coordinator-only queue consumer (each call waits for the native run lifecycle):
run_r16_queue() {
  while IFS= read -r LINE; do
    bash "$D/vla.sh" run --group "${LINE#@local }" --workers 64 --port 23159
  done < "$1"
}
# After BOTH off groups are DONE, emit and finish each exact next dev wave:
for GEN in 1 2 3 4 5; do
  bash "$D/vla.sh" dev --generation "$GEN"
  QUEUE=$(printf '%s/vla/runs/dev%02d/queue.txt' "$D" "$GEN")
  if test ! -f "$QUEUE"; then break; fi
  run_r16_queue "$QUEUE"
done
bash "$D/vla.sh" freeze
GEN=0
bash "$D/vla.sh" test --generation "$GEN"
while test ! -f "$D/vla/FINAL.json"; do
  QUEUE=$(printf '%s/vla/runs/test%02d/queue.txt' "$D" "$GEN")
  run_r16_queue "$QUEUE"
  PREVIOUS=$(printf '%s/%s/vla/analyses/g%02d/analysis.json' "$PWD" "$D" "$GEN")
  GEN=$((GEN + 1))
  bash "$D/vla.sh" test --generation "$GEN" --previous "$PREVIOUS"
done
# FINAL.json contains the selected levels and paired episode/cost arrays.
```

Final cell JSON (152 cells including explicit pending full-step VLA rows):

```bash
"${CPU[@]}" -m "$M.analyze" --output "$D/analysis/cells.json"
# Before new results exist, counts/provenance without parsing live lanes:
"${CPU[@]}" -m "$M.analyze" --inventory-only --output "$D/analysis/inventory.json"
```

Each row gives collected/kept/removed episodes, empty tasks, selected task
levels, final SR and measured IR for both arms, and paired differences by init.
MT50 includes equal difficulty-group means and task means. No synthetic empty
task failures are inserted. Missing/unfinished cells remain pending.

## Resource schedule and episode inventory

`ALL_CAMPAIGNS.json` is the initial definition inventory; `LAUNCH_ORDER.json`
contains a concrete reduced-first serial order and a worker-node conflict map.
`commands/{4090,h100}_{reduced,full}.txt` gives each GPU's exact lane order.
These are schedules, not automatically started chains.

One campaign at a time is a valid conflict-free schedule. In particular, a 4090
lane reserving timan107 or timan1 cannot overlap an H100 lane reserving that same
node: the frozen workers' EGL VRAM admission owns those reservations. A local-only
4090 DP lane can overlap a remote-only H100 lane if their listed nodes and ports
are disjoint. All four π0.5 full MT50 origin jobs together form one 90,000-episode
lane and must all finish before its descent analysis.
Apply the same node/port conflict rule to the existing live `ff1015_*` and
unfiltered lanes; this preparation does not reserve or stop any fleet.

The LIBERO dev, frozen-A and raise counts in this section belong to the
withdrawn dev-set procedure; formal-design LIBERO runs are counted afresh.
Future stages: each R17 LIBERO model/mode has at most 6,000 initial A simulations,
less whole-cell no-knob reuse; each raise is 50 times its raised-task count.
R16 has at most 2,400 dev episodes per model per wave (five waves), at most 4,000
frozen A episodes per model, less off/pure reuse, and 50 per newly raised cohort.
π0.5 uses three and GR00T four 4090 resident processes; both use timan107×64.
The future full-step VLA alternative is excluded from these prepared totals.

Later LIBERO A/raise definitions retain these source resources. `NN=00` has at
most 6,000 episodes per R17 model/mode, minus whole-cell off reuse; each later
generation has exactly 50 times the raised-task count. New full-step VLA
campaigns remain undefined.

| Future campaign pattern (prefix `ffk1015_`) | Source A lane | GPU | R / B | Workers |
|---|---|---|---|---|
| `dp_libero_reduced_testNN` | `knobtestdp_red` | RTX 4090 | 4 / 8 | local×48 |
| `dp_libero_full_testNN` | `knobtestdp` | RTX 4090 | 4 / 8 | local×48 |
| `fastwam_libero_reduced_testNN` | `knobtestfw_red` | RTX 4090 | 2 / 16 | timan1×40, timan107×30 |
| `fastwam_libero_full_testNN` | `knobtestfw3` | H100 | 2 / 16 | timan1×40, timan107×45 |
| `cosmos_libero_reduced_testNN` | `knobtestcos_red` | H100 | 8 / 1 | timan1×40, timan107×45 |
| `cosmos_libero_full_testNN` | `knobtestcos2` | H100 | 8 / 1 | timan1×40, timan107×45 |
| `pi05_libero_reduced_devNN` / `testNN` | R16 k1ref | RTX 4090 | 3 / 1 | timan107×64 |
| `groot_libero_reduced_devNN` / `testNN` | R16 k1ref | RTX 4090 | 4 / 1 | timan107×64 |

The existing source raise campaigns inspected have the same resources as their
row above. Finish the reduced dev/off lanes, their A tests and raises before
the full-step lanes when following the recommended serial schedule. The per-GPU
order files identify the already-launched DP off run and must not be treated as
a request to relaunch it.

<!-- GENERATED_TABLES -->

### Initial campaign definitions

| Campaign (all start `ffk1015_`) | Episodes | GPU | R / B | Workers per node |
|---|---:|---|---|---|
| `pi05_mt50_reduced_descent` | 90,000 | RTX 4090 | 4 / 8 | local×32, timan107×80 |
| `pi05_mt50_full_descent_01` | 4,006 | RTX 4090 | 4 / 1 | local×40 |
| `pi05_mt50_full_descent_02` | 1,761 | RTX 4090 | 3 / 8 | local×96 |
| `pi05_mt50_full_descent_03` | 502 | RTX 4090 | 3 / 8 | timan107×40 |
| `pi05_mt50_full_descent_04` | 83,731 | RTX 4090 | 4 / 8 | local×32, timan107×80 |
| `groot_mt50_reduced_descent` | 90,000 | H100 | 11 / 1 | timan1×40 |
| `groot_mt50_full_descent` | 90,000 | H100 | 11 / 1 | timan1×40 |
| `dp_mt50_reduced_descent` | 90,000 | H100 | 4 / 16 | timan1×40, timan107×40, timan108×40 |
| `dp_mt50_full_descent` | 90,000 | H100 | 8 / 32 | timan1×40, timan107×40, timan108×40 |
| `fastwam_mt50_reduced_descent` | 90,000 | H100 | 2 / 16 | timan1×40, timan107×40, timan108×40 |
| `fastwam_mt50_full_descent` | 90,000 | H100 | 2 / 16 | timan1×40, timan107×40, timan108×40 |
| `dp_libero_reduced_dev` | 21,600 | RTX 4090 | 4 / 8 | local×48 |
| `dp_libero_full_dev` | 21,600 | RTX 4090 | 4 / 8 | local×48 |
| `fastwam_libero_reduced_dev` | 21,600 | RTX 4090 | 2 / 16 | timan1×40, timan107×30 |
| `fastwam_libero_full_dev` | 21,600 | H100 | 2 / 16 | timan1×40, timan107×45 |
| `cosmos_libero_reduced_dev` | 21,600 | H100 | 8 / 1 | timan1×40, timan107×45 |
| `cosmos_libero_full_dev` | 21,600 | H100 | 8 / 1 | timan1×40, timan107×45 |
| `dp_libero_reduced_off` | 6,000 | RTX 4090 | 4 / 8 | local×48 |
| `fastwam_libero_reduced_off` | 6,000 | RTX 4090 | 1 / 8 | local×22 |
| `cosmos_libero_reduced_off` | 6,000 | H100 | 8 / 1 | timan1×40, timan107×45 |
| `cosmos_libero_reduced_off_u` | 6,000 | H100 | 8 / 1 | timan1×40, timan107×45 |
| `pi05_libero_reduced_off` | 4,000 | RTX 4090 | 3 / 1 | timan107×64 |
| `groot_libero_reduced_off` | 4,000 | RTX 4090 | 4 / 1 | timan107×64 |

Initial total: **881,600 episodes across 23 definitions**. These totals exclude outcome-dependent A/raise stages and pending full-step VLA alternatives.

### Failures removed (kept episodes in parentheses)

The two R17 step modes share the same episode sets. R16 counts apply to reduced k1; full VLA remains awaiting a decision. MT50 columns are 250/500/1000/1500/2000/2500; R17 LIBERO columns are 50/100/200/300/400/500; R16 columns are 50/100/200/500.

| Model | Benchmark / suite | Failed (kept), in size order |
|---|---|---|
| pi05 | MT50 / mt50 | 113 (137) / 218 (282) / 438 (562) / 651 (849) / 878 (1122) / 1075 (1425) |
| groot | MT50 / mt50 | 64 (186) / 121 (379) / 238 (762) / 360 (1140) / 484 (1516) / 601 (1899) |
| dp | MT50 / mt50 | 97 (153) / 188 (312) / 377 (623) / 557 (943) / 738 (1262) / 923 (1577) |
| fastwam | MT50 / mt50 | 106 (144) / 205 (295) / 410 (590) / 626 (874) / 841 (1159) / 1043 (1457) |
| dp | LIBERO / libero_spatial | 8 (42) / 16 (84) / 23 (177) / 35 (265) / 51 (349) / 64 (436) |
| dp | LIBERO / libero_10 | 8 (42) / 18 (82) / 36 (164) / 55 (245) / 70 (330) / 95 (405) |
| fastwam | LIBERO / libero_spatial | 7 (43) / 15 (85) / 25 (175) / 41 (259) / 55 (345) / 68 (432) |
| fastwam | LIBERO / libero_10 | 1 (49) / 5 (95) / 12 (188) / 21 (279) / 29 (371) / 38 (462) |
| cosmos | LIBERO / libero_spatial | 5 (45) / 8 (92) / 11 (189) / 19 (281) / 27 (373) / 35 (465) |
| cosmos | LIBERO / libero_10 | 1 (49) / 2 (98) / 6 (194) / 13 (287) / 17 (383) / 21 (479) |
| pi05 | LIBERO / libero_10 | 6 (44) / 11 (89) / 19 (181) / 64 (436) |
| pi05 | LIBERO / libero_spatial | 4 (46) / 6 (94) / 7 (193) / 13 (487) |
| groot | LIBERO / libero_10 | 6 (44) / 10 (90) / 25 (175) / 73 (427) |
| groot | LIBERO / libero_spatial | 5 (45) / 9 (91) / 14 (186) / 44 (456) |

### Episode totals per model

Parent episodes count each maximum-size collected library once. View memberships count overlaps across nested sizes once per view; neither total is doubled for the two step modes.

| Model | Parent collected / kept | View memberships collected / kept |
|---|---:|---:|
| pi05 | 3,500 / 2,348 | 9,450 / 5,947 |
| groot | 3,500 / 2,782 | 9,450 / 7,396 |
| dp | 3,500 / 2,418 | 10,850 / 7,491 |
| fastwam | 3,500 / 2,351 | 10,850 / 7,302 |
| cosmos | 1,000 / 944 | 3,100 / 2,935 |

### Frozen startup matrix

| Campaign | CPU startup | Actual source identity | Interpreter |
|---|---|---|---|
| `ffk1015_pi05_mt50_reduced_descent` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_pi05_mt50_full_descent_01` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_pi05_mt50_full_descent_02` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_pi05_mt50_full_descent_03` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_pi05_mt50_full_descent_04` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_groot_mt50_reduced_descent` | PASS | yes | `/home/exouser/gr00t_n15_venv/.venv/bin/python` |
| `ffk1015_groot_mt50_full_descent` | PASS | yes | `/home/exouser/gr00t_n15_venv/.venv/bin/python` |
| `ffk1015_dp_mt50_reduced_descent` | PASS | yes | `/data/mw_finetune/dp/env/bin/python` |
| `ffk1015_dp_mt50_full_descent` | PASS | yes | `/data/mw_finetune/dp/env/bin/python` |
| `ffk1015_fastwam_mt50_reduced_descent` | PASS | yes | `/data/mw_finetune/envs/fastwam/bin/python` |
| `ffk1015_fastwam_mt50_full_descent` | PASS | yes | `/data/mw_finetune/envs/fastwam/bin/python` |
| `ffk1015_dp_libero_reduced_dev` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_dp_libero_full_dev` | PASS | yes | `/home/weiland/projects/openpi/.venv/bin/python` |
| `ffk1015_fastwam_libero_reduced_dev` | PASS | yes | `/data/mw_finetune/envs/fastwam/bin/python` |
| `ffk1015_fastwam_libero_full_dev` | PASS | yes | `/data/mw_finetune/envs/fastwam/bin/python` |
| `ffk1015_cosmos_libero_reduced_dev` | PASS | yes | `/data/cosmos/cosmos-policy/.venv/bin/python` |
| `ffk1015_cosmos_libero_full_dev` | PASS | yes | `/data/cosmos/cosmos-policy/.venv/bin/python` |
| `ffk1015_dp_libero_reduced_off` | PASS | yes | `/data/mw_finetune/dp/env/bin/python` |
| `ffk1015_fastwam_libero_reduced_off` | PASS | yes | `/data/mw_finetune/envs/fastwam/bin/python` |
| `ffk1015_cosmos_libero_reduced_off` | PASS | N/A: source off not launched | `/data/cosmos/cosmos-policy/.venv/bin/python` |
| `ffk1015_cosmos_libero_reduced_off_u` | PASS | N/A: source off not launched | `/data/cosmos/cosmos-policy/.venv/bin/python` |
| `ffk1015_pi05_libero_reduced_off` | PASS: sealed snapshot + actual CPU registry | sealed R16 inputs | original R16 interpreter choices |
| `ffk1015_groot_libero_reduced_off` | PASS: sealed snapshot + actual CPU registry | sealed R16 inputs | original R16 interpreter choices |

## 2026-10-16 DP MT50 reduced continuation

`ffk1015_dp_mt50_reduced_descentc1` retains 1,059 accepted outcomes and plans
88,941 missing episodes on H100 R4/B16, timan107 × 40 + timan108 × 40.
Read-only native closure checks passed on the coordinator, H100, timan107 and
timan108; the original FAIL receipt is preserved. Workflow and analysis resolve
the registered continuation automatically. Exact prepare/launch/descent commands
and CPU coverage proofs: [continuation handback](continuation_checks_1016/HANDBACK.md).

## 2026-10-16 LIBERO VLA full steps on R15

The owner's corrected R15 decision is implemented under `vla_full_r15/`: only
16 filtered knob cells at 50/100/200/500. The four never-launched size-50 control
definitions, fit copies and script entries were removed before staging.
The original R15 100/200/500 results remain the baselines for those sizes.
Size 50 compares with the existing R14 result; JSON and reporting explicitly
label that comparison **R14→R15, across pipelines**. Size 50 is
expressed by a configuration/catalog extension using existing fits; serving
method code is unchanged. No R16 full-step alternative was built.

There are 24,000 B dev episodes in five waves (2,400/model/wave), then 16,000 A episodes through
R15's uniform/per-task emitter. R15 has no A raise loop. Two additional π0.5
pure-A arms add 1,000 episodes: the archived R15 π0.5 A pure references actually
ran on the 4090, while its cache cells ran on H100. Matching GR00T A references
and all four B references are reused unchanged. These two pure-inference
references are separate from knob arms. Total new simulations: **41,000**
(π0.5 21,000; GR00T 20,000), 10,000 fewer after removing the controls.

π0.5 uses H100/timan108 (39–40 workers); GR00T uses 4090/timan107 (48–64), with
the actual source arm's server/worker counts retained. No timan1 is used.
CPU checks pass: 114 native startup identity rows, 34 native plans, 24 keep-all
source-fit replays, actual serving-interpreter probes, and five targeted tests.
Exact stage commands, counts and provenance: [R15 handback](vla_full_r15/HANDBACK.md).

## 2026-10-16 raise-template cache input completion

`workflow build-test` now passes its model/mode's official-A `TEST_SOURCE` to
the builder. For a separate raise template, the builder verifies the actual
bytes and manifest pins of all overlapping cache inputs, then copies only
missing inputs needed by the selected cells (view, fit, stall and parent
metadata). It extends the existing input inventory before the normal manifest
and study sealing steps. Runtime, worker archive, serving release, topology,
level caps and the in-memory candidate filter are unchanged. A missing or
different official-A input stops the build.

All 18 existing source raise templates were inspected. All six model/modes
have sparse templates; actual overlapping bytes match their official-A source.
First-round omissions, out of 12 official-A cells:

| Model | Full R1 omitted cells | Reduced R1 omitted cells |
|---|---:|---:|
| DP | 5 | 6 |
| Fast-WAM | 3 | 3 |
| Cosmos | 1 | 1 |

Later raise templates are sparse too; per-template counts and byte checks are
in [SOURCE_AUDIT.json](raise_assets_1016/SOURCE_AUDIT.json).

Only `ffk1015_dp_libero_full_test01` was rebuilt, from the same
`results/dp_libero_full/test00/raise.json`: **700 episodes / 14 groups / 10 cells**,
RTX 4090, R4, local × 48 workers. Its frozen CPU startup passes, including native
DP admission and all 14 recipes, with identity equal to the source's actual
ready records and no CUDA initialization. Four missing cells were completed:
libero_spatial 100/300/500 and libero_10 300. Exactly 32 input files were added;
only `assets/cache_inputs/manifest.json` and `assets/CACHE_STUDY.json` changed
among existing files. The campaign and episode plan remain byte-identical.

The old directory was renamed to
`/data/mw_finetune/eval/bad_assets_1016_ffk1015_dp_libero_full_test01` (outside
the workflow's test-lane glob). The failed startup receipt was renamed to
`startup_checks/ffk1015_dp_libero_full_test01.bad_assets_1016.json`. Neither was
deleted. [RENAMED.json](raise_assets_1016/RENAMED.json) and
[REBUILD.json](raise_assets_1016/REBUILD.json) retain the hashes and comparisons.

Queued `ffk1015_dp_libero_reduced_test01` was not rebuilt, prepared or modified.
The current raise table matches its selected tasks. With writes blocked, the
new completion step is an exact no-op for its two size-50 cells; reconstructing
the native input manifest and study seal produces the existing bytes. All 78
queued run files and its startup receipt retain their original hashes. See
[REDUCED_UNCHANGED.json](raise_assets_1016/REDUCED_UNCHANGED.json).

One targeted regression test passes:

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed_knob/test_raise_assets.py
```

The rebuilt full run is ready for coordinator preparation and launch. No
preparation, server, worker, driver or chain was started for this repair:

```bash
cd /home/weiland/projects/openpi
RUN=/data/mw_finetune/eval/ffk1015_dp_libero_full_test01
M=exp.offline_search.rounds.r17.filter_failed_knob
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m "$M.stage" --run "$RUN"
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m "$M.launch" --run "$RUN"
```

## Superseded full-grid draft — 2026-10-16

Both library arms now have independent, fixed-level test-A grids; no dev outcomes
are read and no dev-chosen levels select tasks or levels. Per-task first failure uses a ≤2-success deficit versus same-step
pure A on the same GPU; a qualifying complete off cell is reported directly.
The active [handback](../filter_failed_knob/formal_a_1016/HANDBACK.md) and
[48-job inventory](../filter_failed_knob/formal_a_1016/CAMPAIGNS.json) supersede
the historical LIBERO dev/test/raise commands below or above. They define
544,450 new episodes and reuse standalone off/pure references. π0.5 full is
on hold for a matching existing H100 pure-A reference; five source timing
profiles remain unavailable after outcome cleanup. Do not rebuild queued or
launched runs. Entry point:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

## Decision-head clock withdrawal and rerun — 2026-10-18

The [long-segment handback](../longseg_1017/HANDBACK.md) supersedes the
withdrawn R17 DP/Fast-WAM LIBERO and DP/GR00T/Fast-WAM MetaWorld outcomes
and queue entries. A decision now consists of one segment head and all its
tails; recipe, knob, stall, corrector and RNG clocks advance only on heads.
The initial no-recalibration handback is superseded by the owner ruling below.
Cosmos, π0.5 and legacy VLA pipelines remain unchanged. See the handback for the
complete sampled audit, rename-only backup manifest, fresh definitions,
matching pure references and the two replacement queue lists.

`knobsmokedp1355` is backup only, with no rerun. The Fast-WAM MetaWorld
cache-variant replacement is `fa_ls_v02_s` (R3, sizes 250/500/1000) plus
`fa_ls_v02_b` (R2, sizes 1500/2000/2500), 60,000 episodes each. Additive
`fa_ls_*` prepare/launch wrappers exist in `filter_failed_knob/commands`.
Do not dispatch withdrawn names still present in old queues. New LIBERO
levels are built only from the new off results and same-step test-A pure
references; neither old off results nor development-chosen levels are used.

## Owner decision-clock recalibration — 2026-10-18

[RECALIBRATION.md](../longseg_1017/RECALIBRATION.md) is the current calibration
handback: 840 potential level definitions and 42 C10 Q3 artifacts. Level IR is
M/N over segment heads, absolute tolerance .005; latency-weighted IR remains
separate. Both prediction and level-result analysis exclude tails. Two
original-library DP spatial size-100 k10 entries (full/reduced) have the
confirmed-Q3 call floor 134/1265 = .105928854 and remain explicitly outside
tolerance; no target or Q3 rule is relaxed.

Q3 reproduces the original successful-library LOEO q90 residual/q10 advance
procedure on decision sequences with the same reference/window constants.
C5 artifacts stay unchanged. New C10 artifacts end `_decision1018` beside the
old files, with per-campaign aliases under `assets/q3_decision1018` for H100
metadata staging. Retrieval metrics/PCA/whitening and correctors are unchanged.
Only never-prepared dependent definitions are regenerated in place; UIDs,
inits, seeds and serving topology are preserved. Eight deferred level units
apply the new inputs after their new OFF results. Prepare/launch NAMEs are
unchanged. The coordinator's live queues and recalibration release gate are
authoritative; this work does not release or launch anything.
