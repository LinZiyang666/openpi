# P3 R6 handback: one superset profiler

Implemented entirely under `exp/offline_search/rounds/r06/p3_profiling/`.
The updated owner direction supersedes separate shadow/injection campaigns:
one enabled mode computes one full policy result at every vision anchor,
records all profiling features, and optionally executes that **same** result
with the existing ten-control policy commit. A shadow B guard is recorded even
when running A. No shared plugin, src, ops, or other-round file was edited.

## Files, installation and switches

- `method.py`: exact deployed A/B delegation; deployed-fit provenance checks;
  early/main metric LOEO CDFs; retrieval/guard/state feature capture; A's policy
  tail lifecycle hook. Defaults disabled.
- `hooks.py`: enabled-method-only, connection-local stage taps and output reuse,
  profiling records, independent RNG, explicit `__p3__` wire marker, and extension
  registry. Imports install one guarded `_ConnPolicy.__init__` wrapper **inside
  the importing process**, not on disk. All mutable stage state is per connection.
- `assignment.py`: K5's SHA256/private-Random design extended to every anchor,
  with independent assignment/policy-noise domains and optional quantile strata.
- `snapshots.py`, `worker.py`, `run_gtp.py`, `run_arm.sh`: opt-in client snapshots
  and smallest stock-runner wrappers. See `BRANCHES.md` for exact source paths,
  integration and remaining branch-restore blockers.
- `read_logs.py`, `estimate.py`: strict accepted-attempt CSV readers, optional
  input/snapshot joins, and a K5-derived task/init-cluster HT/bootstrap estimator.
- `campaign.py`, `arms.json`, `prefit.py`: one 32-arm emit_arms campaign, `<RUN>`
  placeholders, sequential CPU prefits using the exact existing source fits.
- `smoke_specs.py`, `arms_smoke.json`: 24 small smoke specs (p=0/.2/1 in eight
  cells), all using the same superset mode; these are validation, not three
  field-collection campaigns.
- `replay.py`, `test_matrix.py`, `test_units.py`, `test_reader.py`, `results/`:
  reproducible CPU checks and numerical evidence.
- `SCHEMA.md`: units, estimand, extensibility and the full field dictionary.

`results/install_manifest.json` records UTC file modification/install times and
SHA256s, including the unchanged shared plugin and all 16 source A/B fits.
No shared file was installed, so there was no shared-file atomic-replacement
step. Shared plugin observed SHA256:
`758a8f061faa9db75477fad4e66a13cb5cade402e858b159854d9fa5ee187a3b`.
New module files are already in the workspace; servers load them only if their
method is explicitly selected. Client snapshots use temp files in their final
directory plus `os.replace`.

Method: `exp.offline_search.rounds.r06.p3_profiling.method:Profile`.
`--os-kwargs` contains `enabled`, `base_spec`, `base_kwargs`, `base_fit`, `p`,
`seed`, `replicate`, and optionally `guard_spec/guard_kwargs/guard_fit` and
`strata`. The supplied campaign sets enabled=true, p=.2, seed=603, replicates
0..3. Optional B uses `campaign.make_kwargs(..., baseline="B")`; no B campaign
is proposed. `enabled=false` delegates without feature computation or hooks on
that connection. `p=0` computes all labels and serves A exactly in the replay
tests. B's baseline guard MISSes remain mandatory; randomization only replaces
baseline CACHE choices, with that eligibility and actual propensity recorded.

Required existing flags: full model; `--os-blind --os-policy-tail
--os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native`.
The campaign also sets `--os-log-inputs` to preserve histories for deferred
branch reconstruction. The client remains five controls/request. Full K10 π0.5
and K8 GR00T are required. Stage-1 variants, K5 landmark overlay, GPU retrieval,
trace/warm-reset modes, caps/bursts, larger commits and non-A/B base settings are
refused. Every shadow uses the already-computed exact stage1 for its observation,
then full stage2/3 once. Executed MISSes take the real interceptor MISS path,
with cached stage outputs returned locally. No extra inference is charged for
injection. The selected chunk is validated before its history broadcast.

## Observed tests

All Python commands used repository `.venv/bin/python`, `taskset -c 2-5,46-49`,
OMP/OPENBLAS/MKL threads=1, CUDA hidden and no bytecode writes. At most two replay
children were active; no server, LIBERO worker, chain, tmux, port, remote host,
GPU inference, git or src edit was used.

| Check | Observed result |
|---|---|
| Eight cells × A/off-A/p0/p.3/p1/B/off-B/on-B | 64 jobs; 5,632 decisions; 2,848 anchors; PASS |
| Exact full served-action, vision and MISS comparisons | 24 pairs: A=off-A, A=p0, B=off-B in every cell; PASS |
| Profile coverage and reuse | 1,424 profile calls = 1,424 enabled vision anchors; 703 total MISSes across matrix; p1 MISSes at every anchor; PASS |
| Strict table extraction | 22,784 neighbour rows; 18,464 action-step rows; every nonterminal anchor's following tail checked; PASS |
| Input-history archives | 32 enabled configurations, 1,424 anchors matched to archived decisions/hits/vision/executed heads; PASS |
| Staged production-shaped CPU fixtures | π0.5 direct/coordinator and GR00T × p0/p1: six cases, exactly 1 stage1 + 1 stage2 + 1 stage3 each; global Torch RNG unchanged; stage tensors released; PASS |
| Final broadcast/log/wire-marker follow-up | 16 jobs, 1,536 decisions, 384 profile anchors, six exact A/off-A/p0 and B/off-B comparisons; six staged wrong-chunk rejections before broadcast; PASS |
| Deterministic assignment | 10,000 keys, order reversal exact; 1,970 CALL at p=.2; invalid p refused; PASS |
| Library LOEO exclusion | Analytic four-row fixture: main nearest distances [9,9,10,10], early affine metric [18,18,20,20], excluding the entire same episode; PASS |
| Estimator synthetic checks | Null: −.01475, 95% CI [−.04838,.01890]; planted .20: .21625, CI [.18223,.24629]; 1,000 clusters retained in each 5,000-anchor fixture; PASS |
| Reader negative cases | Six refusals: missing anchor, forged p, wrong action, missing tail, mismatched outcome, conflicting duplicate; PASS |
| Simulator snapshot fixture | One atomic NPZ, exact pre-inference fake physics state, identity and returned actions; PASS; no real simulator |
| Arm emission | 32 campaign arms → 64 YAML files in `/tmp/p3_emit_validation_v1`; PASS; final campaign including input archives revalidated separately |

The first development run `/tmp/p3_first_v1` refused the deployed early metric.
That exposed the necessary separate early-code CDF; the implementation now
calibrates both early/main metrics and `/tmp/p3_first_v2` passed 48 decisions.
Source inspection also found that production interceptors omit factor outputs
from wire metadata; the final hook explicitly adds `__p3__`, and the final
follow-up replay checks assert that marker on both vision and blind decisions.
The final follow-up covers π0.5 l10/50 and GR00T l10/500 after logging/broadcast
hardening; its separate report is under `results/`.

The complete prefitted Profile was also written under `/tmp/p3_prefit_complete`
and reloaded through the real plugin for GR00T l10/50, p=.2: 48 decisions,
24 shadow anchors, 5 injected MISSes; `results/artifact_reload.json` contains
the exact command and report. All 16 A/B wrapper constructors and Python ASTs
were checked, invalid non-boolean activation was refused, and `bash -n` passed
for the owned launcher. Reader/estimator CLIs were exercised on CPU fixtures;
one p=.3 fixture reported ledger .182 + profiling .318 = physical .5.

Raw matrices: `/tmp/p3_matrix_v1`, `/tmp/p3_matrix_final` (the latter includes
input archives); final focused checks: `/tmp/p3_matrix_hardened`.
Exact subprocess commands and stdout are retained per arm as `command.json`
and `replay.log`. CPU observations are fixed recorded streams; fake policy
labels come from recorded `a_inf`. Truncated replay outcome journals are
synthetic reader fixtures, **not closed-loop success or causal outcomes**.

The Q5/P1 legacy shared-plugin matrices were read but not rerun: their conditional
requirement applies to a shared-file change, and none was made. The P3 matrix
uses the actual installed plugin and deployed A/B fits in all eight cells.
No real-model parity, batched/concurrent real-GPU parity, websocket integration,
real simulator restore or closed-loop SR has been verified.

## One proposed campaign and cost

`arms.json`: π0.5/GR00T × l10/spatial × 50/500 libraries, all A with shadow B
guard features, p=.2 uniform at every anchor, four independent randomization
replicates of the same 500 paired A-pool inits. This is 2,000 episodes per cell,
16,000 total, 32 arms. Each episode collects shadows, interventions, retrieval,
guards, state, stage timings, input histories and (with the client wrapper)
simulator snapshots together. Source-fit paths are absolute read-only paths;
only new profiler-fit paths use `<RUN>`.

With 30 anchors/episode assumed, this yields 60,000 labels and about 12,000
injected MISSes per cell. p=.2 retains cache execution at most anchors while
giving both treatment arms useful support. Expected owner ledger IR for A is
`.5*c1 + .5*p*(1-c1)` = .1608 π0.5 / .1592 GR00T, ignoring terminal partial
commits. Physical inference-equivalent cost is about .5 because every anchor
computes policy; profiling-only cost is .3392 / .3408. Ledger IR continues to
use only actual `dec.vision`/`dec.hit`; profiling time/calls are separate.

Power calculation: fixed-centered HT score variance bound `.25/[p(1-p)]`;
95% half-width `1.96*SE`, 80% two-sided MDE `(1.96+.841621)*SE`. Four replicates
share each init: cluster size is 120 anchors, **not 30**, and there are only
500 unique task/init clusters per cell. Sensitivity design effect is
`1+(120-1)*rho`, where rho is within-init correlation of the HT score.

| Assumed score rho | 95% half-width | 80% MDE |
|---|---|---|
| 0 | 1.00 pp | 1.43 pp |
| .01 | 1.48 pp | 2.12 pp |
| .05 | 2.64 pp | 3.77 pp |
| .10 | 3.59 pp | 5.13 pp |
| .20 | 4.98 pp | 7.12 pp |
| 1 | 10.96 pp | 15.66 pp |

Thus 2,000/cell is a useful exploratory precision target for pooled or broad
state strata, approximately a five-point effect at rho≈.1. It does not promise
five-point per-task or fine-bin precision: a tenth-sized stratum inflates MDE
roughly √10. Correlation and state occupancy are unknown; estimate them from
the first replicate, retaining all fields, and report cluster-bootstrap CIs.
The reported `episodes_for_5pp` sensitivity assumes adding equally sized init
clusters; it cannot justify arbitrarily many repeats of the same 500 inits.

Cost estimate uses the read-only eager batch-1 table
`closed_loop/ops/cost_table.json`: full-policy 463.394 ms π0.5, 250.131 ms GR00T.
At 30 anchors × 8,000 episodes/model: **30.89 + 16.68 = 47.57 GPU compute-hours**.
Budget **95.14 GPU-hours** at 2× for shared-load/queueing/collection overhead;
this is a proposal, not a measured rollout runtime. If mean anchors are 60,
double both figures. Simulator GPU-hours, fit CPU-hours, restarts and artifact
I/O are not included. Profiling stage timers include queueing/synchronization;
do not mislabel their sum as hardware utilization.

Storage planning: input archives alone contain ~252 GB of uncompressed pooled
keys at 60 decisions/episode ×16,000 episodes, plus actions/metadata. Two raw
256² RGB views ×480,000 snapshots would be ~189 GB before compression; snapshots
are compressed, but actual ratio is unknown. Add JSONL and fitted artifacts.
Reserve roughly **500 GB** for the campaign or set snapshot stride>1 explicitly;
this is a planning estimate. Library/fit bytes were not changed or pruned. Large
library information already borrowed by deployed A/B remains in those fits.

## Coordinator recipes (not run here)

CPU preparation from repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_profile
mkdir -p "$RUN"
sed "s|<RUN>|$RUN|g" "$D/arms.json" > "$RUN/arms_in.json"
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms \
  --run-root "$RUN" --spec "$RUN/arms_in.json"
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.prefit \
  --run-root "$RUN" --spec "$RUN/arms_in.json"
```

For smoke, resolve `arms_smoke.json` to a fresh smoke run instead and prefit only
the selected smoke arms with `--arms NAME ...`. Select tasks 0,1 and inits 0,1
using the existing `OSCL_TASKS=0,1 OSCL_EPISODES=0,1` coordinator controls. Begin
with π0.5 l10/50 and GR00T l10/500, p0 and p1, then p.2 and the remaining cells.
Use full models, ordinary five-control clients, the P3 snapshot worker wrapper,
and coordinator-assigned resources. Do not inherit the CPU-preparation empty
CUDA mask into full-model servers. No launch command, port or host was used here.

After standard remote sync, install/select the **owned** P3 client launcher
as described in `BRANCHES.md`; merely emitting server arms does not enable
snapshots. Preserve its stock launcher input and archive snapshots separately.
Before full collection, require smoke checks: every vision anchor has one
`p3_anchor`; no blind probe; p0 has zero MISS; p1 has one policy tail after every
nonterminal anchor; shadow normalized chunk equals the injected chunk; stage
counts show no second stage2/3 forward; every snapshot joins the correct
uid/attempt/step and captures the state before action; input archives exist.
For real p0 parity use the same observation history and deployed A fit, compare
full normalized chunks and wire heads; closed-loop timing alone cannot prove
bit parity under different GPU batching schedules.

Read collected logs (choose actual arm names), requiring saved input histories:

```bash
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.read_logs \
  --run-root "$RUN" --arms r6p3_pi05_l10_50_r0 r6p3_pi05_l10_50_r1 \
  --require-inputs --snapshot-root "$RUN/client_snapshots" --out "$RUN/tables_pi05_l10_50"
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.estimate \
  "$RUN/tables_pi05_l10_50/anchors.csv" --arms r6p3_pi05_l10_50_r0 r6p3_pi05_l10_50_r1 \
  --out "$RUN/pi05_l10_50_effect.json"
```

The estimate is a sequential randomized excursion under continued randomization,
not a single-MISS-then-A counterfactual. Snapshots are implemented but exact
paired restore remains uncertified; see `BRANCHES.md` for the restore protocol
and blockers. No user questions were needed: the chosen interpretation is to
collect branch-ready evidence now and label restoration guarantees honestly.

Reproduce CPU checks with the same environment exports and fresh `/tmp` roots:

```bash
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.test_matrix --out /tmp/p3_recheck --workers 2
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.test_units --out /tmp/p3_units_recheck
taskset -c 2-5,46-49 .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.test_reader --matrix /tmp/p3_recheck --out /tmp/p3_reader_recheck
```
