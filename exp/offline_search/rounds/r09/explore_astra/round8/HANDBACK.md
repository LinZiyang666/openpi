# Handback — R9 astra round 8

Diagnosis only. **Zero arms frozen.** No `/home/weiland/trace_runs/os_closed_loop/r09_astra_r8/` run package was created, no standard-emitter mutation was needed, and no sync or evaluation launch was performed. PREDICTION.md is intentionally absent because there is no proposed arm. The coordinator's GR00T escalation experiment remains separate.

Read [REPORT.md](REPORT.md) first (plain-Chinese owner explanation), then [DATA_ANALYSIS.md](DATA_ANALYSIS.md). The main conclusion is recurrent difficult-state recovery, not a missed no-progress call or demonstrated phase-wide corrector damage. On evaluation the success gap is only −1 pp (88 vs 89 wins); the −6 pp pooled gap is driven by 0–19. Guard first-call recovery is about 92% for both models, while repeated calls and gripper cycles accumulate in GR00T losses.

## Reviewable evidence

- `results/lost_episodes.csv`: all 41 GR00T and 22 π0.5 lost pairs, with timing, events, distance, correction, and pure-policy duration.
- `results/episode_events.json`: exact event indices and all calls for the 600 admitted recipe episodes.
- `results/lost_eval_timelines.svg`: all eight GR00T evaluation losses; gripper commands and real calls, plus paired pure completion time.
- `results/{decisions,episodes,pairs,calls,calls_annotated,corrector}.parquet`: admitted analysis tables.
- `results/{summary,diagnostics,corrector_summary}.json`: complete split-specific summaries, denominators, and ablations.
- `SCREEN_PROTOCOL.json`: four observation-only diagnostic gates, recorded before their evaluation. Every gate fails to improve training and evaluation; no new arm selected.
- `PROVENANCE.json`, `results/artifact_compare_*.json`: input lineage and source/deployed recipe comparison. All 602 arrays and shared scalar serving state agree; earlier deployed pickles omit eight later optional pace-wrist fields each.
- `results/tests.log`: 18 passing admission, split, phase-sign, episode-weighting, recovery-censoring, join and actual-output checks.

## Reproduce locally, analysis only

From `/home/weiland/projects/openpi`:

```bash
taskset -c 10-21,54-65 bash exp/offline_search/rounds/r09/explore_astra/round8/tools/reproduce.sh
```

The script sets `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.:src`, and OMP/OpenBLAS/MKL/NumExpr threads to one. It runs only offline parsers, numerical audits, plots and tests. Every output and plotting cache stays in round8. It never calls a controller, emitter, serving API, remote synchronization, git, or process-control utility.

Read-only inputs are the four named allowed outcome arms (only parsed inits 0–29), eight existing all-0–29 debug compact archives (A/CU/IP/P10 for both models), library/model artifacts, and ten admitted init-0 episode metadata files. Root-prefix checks occur before opening files; compact identity members are checked before payloads. Full-population summary files and prohibited run roots are unnecessary. Thresholds are fitted on 0–19 only. The existing recipe task heads are replayed unchanged for diagnosis; no new task table, task gate, or task threshold exists.

## Coordinator implication

Use the listed failures to examine the separately planned GR00T escalation experiment, while retaining its same-batch R9Recipe control. Report first-call and repeated-call recovery separately, alongside success and calls; an improvement in demo-progress clearing alone is insufficient. For a later non-escalation investigation, the missing evidence is same-state policy shadow and object/contact evidence on recipe trajectories, or a predeclared randomized corrector ablation. That is a proposed evidence requirement, not an authorized or emitted experiment in this handback.

Do not turn the task loss ranking, retrospective phase labels, or the 20–29 results into serving selectors. Cross-batch variation, π0.5 escalation, sparse causal labels, and reuse of the development evaluation set limit the diagnosis. No new deployment success claim is made.
