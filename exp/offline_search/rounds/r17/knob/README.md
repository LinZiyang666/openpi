# R17 per-task knob analysis

LIBERO formal-A launch units now use short `fa_<model>_<mode>_<arm>_<stage>`
names, preserving plans and serving while respecting the frozen MPS pipe limit.
The [name map](../filter_failed_knob/formal_a_1016/NAME_MAP.json) is authoritative;
analysis resolves old aliases through it. See the
[current CPU handback and GPU orders](../filter_failed_knob/formal_a_1016/short_names_1017/HANDBACK.md).
This is a naming repair; the test-A selection and off-cell rule below are unchanged.

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

The LIBERO path described below (`descent.py` on B inits 0–29 at 30 episodes
per level with a 5 pp rule, then `test.py` A tests and `raise.json` rounds) is
that withdrawn procedure; do not use it to choose LIBERO levels. The MT50 path
(`descent.py` on seed-7 lanes, two-success tolerance, direct selection) follows
the formal design.

Run from the repository root with `.venv/bin/python`. These scripts only read
input lanes and write the specified output directory; all outcome and campaign
cost parsing uses the unchanged cache loader, including resume ancestors.
Receipt status does not restrict reads. No model, simulator, or GPU is imported.

`descent.py` pairs B inits 0–29 with `READY.json` teacher episodes, applies the
integer first-failure rule at 5/0/2 percentage points, and writes `levels.json`,
`levels_tol0pp.json`, `levels_tol2pp.json`, `descent.json`, and `DESCENT.md`.
All six candidate levels must have 30 outcomes to freeze a task. Incomplete
tasks remain in the report with their observed counts and are omitted from the
level mappings. Stitched summaries state the number of included tasks and
compare exactly those tasks with pure inference. Each cell also reports the
full 300-episode teacher baseline. Failures and subsequent recoveries are
recorded separately for every tolerance.

`test.py` reads the supplied frozen A lanes and the fixed official controls,
automatically following both Cosmos segments and the Fast-WAM resume. It writes
`test.json`, `TEST_A.md`, and `raise.json` (changed tasks only). The 5,000-draw,
seed-20261005 interval resamples paired inits within each task, sharing suite
draws across models and sizes. Partial cells report observed pairs and cannot
trigger a raise for an incomplete task. A task with two planned levels is an
input error.

IR proxy aggregates actual `policy` calls and executed `controls`, using C0=5
for DP/Fast-WAM and C0=16 for Cosmos. `library` calls and `policy_tail` requests
do not add model calls. Pure-only inference is 1.0 by the owner's convention.
For mixed stitching, teacher capture rows provide actual pure-task counters;
missing counters produce a null IR rather than an estimate. Nominal knob
targets never substitute for measured IR. `--library-root` and `--control-root`
allow temporary fixtures; defaults are the fixed `/data` locations.

The following LIBERO dev/test commands run the withdrawn dev-set procedure and
are kept only as a record of how the withdrawn results were produced. Set the
following six variables to the eventual campaign directories (data
does not exist yet), then run:

```bash
.venv/bin/python exp/offline_search/rounds/r17/knob/descent.py \
  --lane "$DP_DEV_LANE" --lane "$FASTWAM_DEV_LANE" --lane "$COSMOS_DEV_LANE" \
  --out exp/offline_search/rounds/r17/knob/dev_results

.venv/bin/python exp/offline_search/rounds/r17/knob/test.py \
  --lane "$DP_TEST_LANE" --lane "$FASTWAM_TEST_LANE" --lane "$COSMOS_TEST_LANE" \
  --out exp/offline_search/rounds/r17/knob/test_results
```

Additional `--lane` arguments can include other segments; accepted outcomes
are merged by canonical task/init and latest timestamp, without double counting.

Verification:

```bash
.venv/bin/python -m unittest \
  exp/offline_search/rounds/r17/knob/test_knob.py \
  exp/offline_search/rounds/r17/cache/test_analyze_cache.py -v
```

Synthetic fixtures mimic `campaign.json`, `assets/CACHE_STUDY.json`,
`EPISODE_PLAN.json`, and all three driver JSONL files. They exercise all-pass
descent, failure at k30, recovery, missing/partial levels, paired gains/losses,
tail controls, missing cost, all three model groups, control resumes, bootstrap
sampling, and retest boundaries through both CLIs.

Real loader and reference checks (the cache read takes about six minutes):

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from exp.offline_search.rounds.r17.knob import common as c

lane = Path('/data/mw_finetune/eval/csztd_r3_l48')
snapshot = c.cache.receipt_snapshot(lane)
cells = c.cache.load_lane(lane, snapshot, c.cache.Sources(), c.cache.SIZES)
print(len(cells), snapshot['accepted_episodes'], snapshot['cost_episodes'])
for model in c.C0:
    for suite in c.cache.SUITES:
        reference = c.teacher_reference(c.LIBRARY_ROOT, model, suite)
        print(model, suite, c.stats({k: e for k, e in reference.items() if k[2] < 30}, model),
              c.stats(reference, model))
PY
```

The real read-only cache loader check on `/data/mw_finetune/eval/csztd_r3_l48`
loaded 80 cells and 40,000 accepted episodes with 40,000 complete cost records
(369 seconds). The official A controls each loaded 500 episodes per suite with
complete costs. All six B references passed the exact 10-task × 50-init check,
including complete capture counters:

| Model | Suite | B inits 0–29 | B inits 0–49 |
|---|---|---:|---:|
| DP | libero_spatial | 255/300 (85.00%) | 436/500 (87.20%) |
| DP | libero_10 | 241/300 (80.33%) | 405/500 (81.00%) |
| Fast-WAM | libero_spatial | 256/300 (85.33%) | 432/500 (86.40%) |
| Fast-WAM | libero_10 | 282/300 (94.00%) | 462/500 (92.40%) |
| Cosmos | libero_spatial | 269/300 (89.67%) | 465/500 (93.00%) |
| Cosmos | libero_10 | 290/300 (96.67%) | 479/500 (95.80%) |

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

## Off-first queue revision — coordinator review

The active [formal queue handback](../filter_failed_knob/formal_a_1016/HANDBACK.md)
implements dispatch pruning: complete off SR ≥ same-step pure SR removes all
non-off levels of that cell, independently per arm. Full modes run an off-only
stage first and build the non-off stage from analysed off results. Incomplete
cohorts wait. The two authorized shared π0.5 full H100 pure references release
the earlier reference hold. Use the new ordered NAME lists and parent-directory
prepare/launch aliases; earlier full-grid commands are superseded.

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

[RECALIBRATION.md](../longseg_1017/RECALIBRATION.md) records 840 potential
level fits and 42 C10 Q3 refits. Level IR is M/N over segment heads; tails
count in neither. Predicted and reported level IR use that definition;
latency-weighted IR remains a separate measured quantity. The tolerance is
absolute .005. Two original-library DP spatial size-100 k10 entries
(full/reduced) have an irreducible confirmed-Q3 call floor 134/1265 = .105928854;
they are explicitly marked outside tolerance. No target or gate rule is relaxed.

C10 Q3 uses the original successful-library LOEO q90 residual/q10 advance
procedure on decision sequences, preserving the minimum references, window
rules and metric bytes. C5 artifacts are unchanged. New Q3 files live beside
the originals, ending `_decision1018`; retrieval fits and correctors are
unchanged. Regenerated gated definitions preserve task UIDs/inits/seeds and
carry new recipe identities. Deferred LIBERO levels consume the same map after
the new OFF cell rule. The coordinator controls `[recal] released`; chain
scripts and running/independent campaigns are not changed by recalibration.
