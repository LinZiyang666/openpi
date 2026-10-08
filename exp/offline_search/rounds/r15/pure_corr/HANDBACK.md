# Pure cache + frozen size-specific GC_dist corrector

**Ready, CPU preparation only; evaluation not launched.** 24 arms, four roots,
six arms per root (sizes 50/100/200/300/400/500), 500 official A episodes per arm,
recording off. All 24 queue lines in [queue.txt](queue.txt) are `@h100`, globally
ordered by increasing size. Runner environment: `SERVER_HOST=h100 WORKER_HOST=timan107`.
One fleet; no fleet is reserved or bound inside a root. Both models retain R10's H100 host.

- `/home/weiland/trace_runs/os_closed_loop/r15_purecorr_pi05_l10`
- `/home/weiland/trace_runs/os_closed_loop/r15_purecorr_pi05_spatial`
- `/home/weiland/trace_runs/os_closed_loop/r15_purecorr_groot_l10`
- `/home/weiland/trace_runs/os_closed_loop/r15_purecorr_groot_spatial`

Arm pattern: `r15_purecorr_<model>_<suite>_<size>_AC_dist`.
Entry point: `exp.offline_search.rounds.r15.pure_corr.runtime:PureCorrector`.
A's frozen cache, retrieval, synthesis, anchor-tail commit, client settings and YAML
bytes are preserved. The guard object and judge/policy-tail flags are absent.
Each artifact embeds that exact cell/size's GC_dist heads, sigma and distance rule/scale.
A's cache/PCA/metric arrays are checked byte-for-byte against that GC_dist fit.
Strength is `.5 * clip((2-r)/1.25, 0, 1)`; step 0 is uncorrected. No refit or selection.

The 24-row [corrector_bindings.csv](corrector_bindings.csv) binds every arm to
its R10 GC_dist fit, head, distance calibration, metric fit and subset/row paths
with SHA-256 hashes. [audit.json](audit.json) additionally records exact episode
IDs and all R10 A/G/GC_dist reference arm paths and fit/config/manifest/summary hashes.
The audit fails on changed hashes, wrong cell/size, differing embedded heads/scales,
cache metrics, subset rows or episode IDs. R10 GR00T GC_dist l10 sizes 50–300 are
in `r10_corr3_groot`; its other eight completed references are in `r10_corr3_groot_b`.
The official 500-pair template is copied byte-for-byte. Rollout seed 7 matches
R10's omitted worker seed (`worker_entry --seed` defaults to 7); it is explicit here.

CPU proofs: **15 synthetic tests pass** ([tests.log](tests.log)); all 24 plugin
replays pass ([proofs.json](proofs.json)). Corrector-off equals R10 A and guard-on
equals R10 GC_dist: **7,680 compared decisions, zero differences**, including
480 forced no-progress MISS comparisons. Replay drives ten tasks, two interleaved
connections and episode resets on B-library inputs; full action bytes, histories,
retrieval/scores/confidence/verdicts/tails and semantic decision logs are checked.
All 3,840 production replay decisions are checked; all 24 arms show changed actions
and logged engagement. No test-A trajectories/journals/outcomes were used for preparation;
R10 summary files supplied only completion/pairing metadata and hashes.

Engagement lives in decision-log `extras`: `os_pc_applied=1` for a fresh nonzero
correction, `os_pc_inherited=1` for its blind tail, `os_pc_strength` for the actual
blend, `os_pc_rms` for correction RMS, and `os_pc_count` for cumulative fresh
engagements in the episode (resets at step 0). The first-arm CPU example is in
[proofs/r15_purecorr_pi05_l10_50_AC_dist.json](proofs/r15_purecorr_pi05_l10_50_AC_dist.json).
After its real run, inspect `runs/<arm>/server_*/decisions_*.jsonl` under the root.

After all runs, from the repository root:

```sh
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.pure_corr.analyze
```

This prints SR @ standard stage-ledger IR for A/A+C/G/GC_dist, paired gains/losses
and two-sided exact binomial p for A+C vs A, and six-size pooled rows per cell.
Only synthetic outcome fixtures tested this analysis; it has not read real results.
Pooled pairing uses `(size, task, init)` identities as in R10; sizes reuse inits.
Recheck frozen bindings with the same CPU prefix and `-m exp.offline_search.rounds.r15.pure_corr.build audit`.

All four standard plans passed with `SERVER_HOST=h100 WORKER_HOST=timan107`:
[plans.json](plans.json); each root has `h100_sync/plan.json`.
All 24 relocated source/served artifact hashes and frozen arrays pass
[relocation.json](relocation.json). New run dependencies total **909,912,530 bytes**;
extra store bytes **0**. Plans and receipts are local evidence; nothing was synced.

Deploy these **two new runtime files to h100**, at the same repository-relative paths:

| File | SHA-256 |
|---|---|
| `exp/offline_search/rounds/r15/pure_corr/__init__.py` | `b5f132be2f449283078e96cfc2dcaedabad661c92bd467f7a8c5a06de42d526a` |
| `exp/offline_search/rounds/r15/pure_corr/runtime.py` | `d55c53aa02482d4c3d824537dbef69c313eac6d4b4bc9a4943cd8dc2fb312dc4` |

No new runtime file is needed on timan107; its unchanged worker prerequisites
(including seed forwarding and the default-7 worker) and all 43 unchanged H100
import prerequisites have paths/hashes in [deployment.json](deployment.json) and
[DEPLOY.sha256](DEPLOY.sha256). The fit/config/manifest bytes and destination hashes
are in each standard plan. **timan108 is excluded; deploy nothing there.**
No shared runtime edits, servers, GPU, remote operations or evaluation launches.
