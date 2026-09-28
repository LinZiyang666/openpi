# P2 paper ablations: identity metric and B trigger leave-one-out

Delivered 8 metric arms and 16 trigger arms with 24 fresh prefits (1,106,853,913 bytes total). The metric check has
zero ordered top-16 mismatches on 23,522 recorded queries. Same-history B audits have zero mismatches over 188,844
decisions / 95,151 vision anchors. New all-off equals original B-off exactly on 108,237 recorded decisions; it
equals stable-tie A exactly, while deployed A differs at one pre-existing GR00T l10/50 tied anchor and its tail.

Repository: `/home/weiland/projects/openpi`. Implementation and evidence files are confined to
`exp/offline_search/rounds/r06/p2_ablations/`; generated fits, replay logs and emitted validation configs are under
`/home/weiland/trace_runs/os_closed_loop/r06_abl/`. No shared implementation was edited. No git commands,
servers, LIBERO workers, experiment chains, tmux sessions or remote hosts were used. All Python commands used
CPUs `26-29,70-73`, the repo `.venv/bin/python`, OMP/OpenBLAS/MKL threads 1 and an empty CUDA_VISIBLE_DEVICES.
The plugin itself records its ordinary read-only HEAD metadata. No `tests/review_tests/` access.

## Files and switches

- `arms_metric.json`: 8 arms, models pi05/GR00T × l10/spatial × 50/500. Method
  `exp.offline_search.rounds.r06.p2_ablations.metric:IdentityBlindAWM`.
- `arms_trigger_loo.json`: 16 arms, models pi05/GR00T × l10/spatial × four disabled guards, all 50-library.
  Methods `...p2_ablations.judge:TriggerCommitJudge` (pi05) and `...:TriggerGrootCommitJudge` (GR00T).
  Added kwarg `disabled_guard`: `stuck`, `terminal`, `overtime`, or `no_progress`.
  Test/control values `none` and `all` are also supported; they are not extra paper arms.
- `make_arms.py`, `results/provenance.json`: exact source rows and arm derivation.
- `prefit.py`, `run_jobs.py`: fresh prefit and bounded CPU subprocess runners.
- `metric_check.py`, `replay_audit.py`, `allvision_check.py`, `nesting_check.py`, `logic_check.py`, `validate.py`,
  `summarize.py`: executable checks and aggregation.
- `results/summary.json`, `results/TABLES.md`: final replay numbers and checks.
- `results/validation.json`, `results/fit_sha256.txt`: every final prefit's size, SHA256, library rows, class and metadata.
- `results/initial_float32/`: preserved initial metric-check failures and original prefit command record.
  Superseded float32 artifacts are outside the final fit directory, at `r06_abl/initial_float32/`.

The two arm specs are emit_arms input lists. Only name, method, fit-artifact pathname, and the trigger switch differ
from their authoritative A/B source rows. All other fields (including their absence), kwargs, plugin-arg ordering,
full_model, cost_ledger, client settings, kref and libraries are copied exactly. The pi05 spatial/500 A source's
absent `cost_ledger` is intentionally preserved. GR00T B explicitly retains `--os-policy-tail-blocks 1`, resize 256,
and replan_steps 5; pi05 B retains its implicit one-block setting. Artifact paths are `<RUN>/fits/<arm>.pkl`.
`validate.py` restores the changed fields and asserts whole-row equality with the original spec.

## Metric ablation: exact identity, not a large-lambda approximation

`codes=0` means full Cholesky metric; `codes>0` selects generalized-eigen directions after metric whitening.
Neither means identity. Finite `lam` retains the action-similarity covariance in
`(S_w + lam tr(S_w)/d I)^-1`; `lam=inf` is not a usable exact identity setting. No large lambda is used.

`identity_metric` returns the stock float64 mean, `std + 1e-6`, and `eye(136)` for BOTH normal and early fits.
AWM.fit has no injectable metric hook, so the subclass binds its unchanged function bytecode to a private globals
dictionary containing the replacement fitter. It never assigns to the earlier module or monkeypatches a shared
function. The local fitter captures the features/statistics for the explicit Euclidean evaluator; no evaluation
query is used in fitting. Stock PCA, action scales, pseudo-query calibration, blind statistics and synthesis run
unchanged. Every delivered prefit was freshly generated using the plugin prefit entry point.

Representation and branches:

- Same PCA-64 per camera, plus the same eight valid state features (136 total), with the same current/big PCA bases.
- Per-task main normalization uses all that task's fit-library rows. Early normalization uses step <= 2 rows of that
  task. Step 0 uses the early metric against ALL candidate rows; subsequent pure-cache anchors use the main metric.
- Euclidean `norm(Z_candidates - (x_query - mean)/std)` in float64. Identity is exact; standard deviation's existing
  `+1e-6` is retained. The query-side PCA projection remains stock float32.
- Same top-16 selection routine, kernel, kref 5/8, full-chunk synthesis, anchor_tail, budget 1, budget_only gates.
  No policy calls. Post-MISS continuity code is preserved but unreachable under these A specs.

Numerical issue found and resolved: simply returning I from the metric fitter while keeping the stock float32
quadratic/early-affine distance arithmetic produced reordered near-tied neighbors against a direct float64
Euclidean reference. The initial check failed in five cells; the per-query evidence is preserved in
`results/initial_float32/`. The final evaluator uses direct differences in float64, while retaining the same early
fit/step-0 decision branch. This is an explicit numerical implementation difference from stock AWM, needed to make
the requested Euclidean ranking exact on the tested queries. It retains two additional 136-d float64 candidate-code
arrays (2,176 bytes/entry beyond stock); this is a behavior ablation, not a memory/latency claim. Stock distance work
is still performed by the inherited `_dist` before replacement; no search-latency benefit is claimed.

The independent test re-reads the library PCA projections and raw valid state, recomputes each task's main/early
mean and std, constructs direct float64 Euclidean distances, and compares ORDERED top-16 row IDs. It also checks
PCA/baseline array equality and exact kernel synthesis. Query selection is all 500 episode starts plus 1,000 evenly
spaced rows in EACH of the recorded cache and inf cells (duplicates removed), for each of the eight fits. The inf
stream is used as recorded observations with A's no-MISS regime; no recorded policy history enters retrieval.

## Trigger semantics

The mask runs AFTER the complete original CommitJudge/GrootCommitJudge query (including P1's GR00T terminal-sign
fix). It clears the selected os_flags bit and re-derives lowest-bit os_reason, os_force_miss, and `_s['flag'][-1]`.
Everything that produced the flags is still computed. Non-verdict diagnostics and fitted thresholds are untouched.

| disabled_guard | bit removed | retained computation |
|---|---:|---|
| stuck | 1 | vision-confirmed stuck counter, V7 stuck feature, overtime's `stuck_n >= 1` prerequisite |
| terminal | 2 | terminal-row diagnostic, executed gripper and model-specific closed sign |
| overtime | 4 | overtime, lag and stuck diagnostics |
| no_progress | 8 | progress history and span; only no-progress MISS and its blind LOOK veto are disabled |

Thus `stuck` does NOT set the counter to zero or change its threshold. An observation with flags 1|4 becomes 4 and
still MISSes. Disabling a lower-priority bit can change the logged reason while retaining MISS; only a singleton
firing set can lose its MISS verdict. This is what "equals B except where the disabled guard was the only firing
reason" means for the decision: payload/confidence are always unchanged on equal histories, diagnostic flag/reason
fields necessarily record the remaining firing set.

Disabling no-progress also bypasses K1's `LookReason(8, 'noprog_span')` blind veto; progress computation is retained.
This is the complete leave-one-out of that guard, and makes `all` use the same blind gate as A. Lifecycle,
anchor-tail budget, stuck/terminal/overtime logic, and the committed policy tail remain inherited. The explicit
blind-veto tests cover a positive span even where normal committed B replays do not encounter a separate veto.
Only deployed B settings are accepted: guards enabled before masking, events none, burst 0, monitor off,
policy_tail_gate lifecycle, vision-confirmed stuck, noprog_span, no memo reset, anchor_tail/budget1/budget_only.

## Replay proof and its limits

`replay_audit.py` uses P1's real installed-plugin replay driver and deployed B artifact as the reference. Each
freshly fitted LOO artifact plus none/all controls receives the SAME online QueryView at each reference B vision
anchor. Separate per-episode state is maintained; cache blind and policy-tail calls are checked too. B supplies
served action/HIT/vision histories, so the proof isolates the removed trigger instead of conflating it with later
history divergence. Every recorded cache and inf episode is replayed (500 per stream × four 50-library cells).
Assertions cover exact top-k, scores, action, confidence, every non-verdict extra, unchanged stuck/progress state,
correct flag memo, independently recomputed B predicates, and removed MISS iff only the disabled guard fired.
The table reports direct removals on that fixed B history. It is NOT an estimate of removal totals in independent
LOO rollouts: changed cache/policy histories can affect later selection and guards.

`allvision_check.py` additionally queries 30 evenly spaced recorded episodes from each cache/inf stream per cell,
using recorded execution histories. This exercises nontrivial all-vision stuck/terminal/overtime/no-progress logic
and the GR00T terminal correction. The regular B schedule mostly uses this branch at step 0 only.

`nesting_check.py` runs each new all-off path through the installed plugin on all 500 cache-stream episodes for
each 50-library cell. It compares original B with guards=False, deployed A, and P1's test-only stable-tie A control.
The test loads the fresh `stuck` artifact, sets its runtime disabled mask to 15, and adds comparison hooks; its
startup metadata therefore still names that artifact. This avoids fitting four redundant all-off artifacts.
The known pre-existing AWM argpartition vs G3 stable-top-k tie discrepancy is audited explicitly; exact deployed-A
nesting must not be claimed where the table reports tied anchors/tails. No deployed tie rule was changed.

The 24 separate plugin smoke replays use EACH actual deliverable artifact and spec on three evenly spaced cache
stream episodes (including first/last). Their own histories evolve according to that variant. The final aggregator
checks cache rows 5..9, each nonfinal MISS's single policy tail, no metric MISS, stage-1/vision count agreement,
and at most one blind decision in a row.

All replay observations are fixed recordings. No simulated action-dependent state evolution, success-rate result,
GPU latency, hardware control or actual policy inference was measured. Policy MISSes in the CPU replay receive the
recorded `a_inf` chunk. No closed-loop experiment was launched; that remains the coordinator's work.

## Commands and reproducibility

Run from the repository root. Every Python invocation, including children of the runners, uses this prefix:

```bash
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
D=exp/offline_search/rounds/r06/p2_ablations
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.make_arms
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.prefit
```

`prefit.py` defaults to `/home/weiland/trace_runs/os_closed_loop/r06_abl`, refuses existing artifacts unless
`--resume`, and permits at most seven children plus its coordinator. For a full fresh reproduction use
`--run /absolute/fresh/run/root`; the supplied specs' `<RUN>` must resolve to that root. Existing final prefits
should be loaded rather than refitted. Hashes identify delivered bytes; refitting changes recorded fit timing.
Initial full prefit commands are in `results/initial_float32/prefit_commands.json`; final metric-refit commands are
in `results/prefit_commands.json`. The initial metric artifacts were moved aside, then the final eight were fitted
with `prefit --parallel 2 --resume` (log `results/prefit_metric_final.log`).

The executed test matrices are fully specified by the `args` arrays in `results/audit_jobs.json` and
`results/verification_jobs.json`:

```bash
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.run_jobs "$D/results/audit_jobs.json" --parallel 4 --logs /home/weiland/trace_runs/os_closed_loop/r06_abl/test_logs/audit
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.run_jobs "$D/results/verification_jobs.json" --parallel 2 --logs /home/weiland/trace_runs/os_closed_loop/r06_abl/test_logs/verification
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.logic_check
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.validate
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.summarize
```

The two matrix runners were concurrent: 4+2 children and two coordinators = eight Python processes maximum.
Other Python commands ran only after enough slots were free. On rerun choose fresh replay `--out` directories
in the job files and a fresh validation output directory; replay/validation refuse existing directories. Result
JSONs and text logs persist in the new directory and `r06_abl/test_logs/`; raw replay decisions and normalized
served chunks persist in `r06_abl/replay/`. Tests are read-only toward the source fits and store.

A minimal CPU smoke recipe (fresh output directory; no server or worker):

```bash
"${P[@]}" -m exp.offline_search.rounds.r06.p1_groot_commit.replay \
  --method exp.offline_search.rounds.r06.p2_ablations.judge:TriggerGrootCommitJudge \
  --kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off","disabled_guard":"stuck"}' \
  --cell groot_l10_cache --replay-cell groot_l10_cache --episodes 3 \
  --fit-artifact /home/weiland/trace_runs/os_closed_loop/r06_abl/fits/r6p2_stuck_g_l10_50.pkl \
  --judge guard_only --policy-tail --blocks 1 --out /absolute/fresh/smoke/output
```

For coordinator emission, resolve `<RUN>` in BOTH JSON specs before invoking
`exp.offline_search.closed_loop.ops.emit_arms --run-root <RUN> --spec <resolved-spec>` (same Python prefix).
`validate.py` executed this exact emitter/config/parser path into `r06_abl/arm_validation/` without launching
anything. The two source arm lists remain unresolved as requested.

## Observed numbers and artifact hashes

See [results/TABLES.md](results/TABLES.md) for all eight metric checks, every trigger variant's removed MISS count
in both replay streams, all-off nesting exceptions, smoke counts, and all 24 prefit byte sizes/SHA256 hashes.
The machine-readable complete report is [results/summary.json](results/summary.json).
