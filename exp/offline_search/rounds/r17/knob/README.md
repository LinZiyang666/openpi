# R17 per-task knob analysis

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

Set the following six variables to the eventual campaign directories (data
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
