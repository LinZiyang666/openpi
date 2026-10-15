<task>
Owner ablation (2026-10-15): with the same number of collected episodes, does dropping the failed episodes from the cache library improve the pure Cache group? Repo /home/weiland/projects/openpi. You write and CPU-test code and hand back exact commands; the coordinator launches every server, worker and chain.

Design (owner, binding):
- For every existing library and every library size, remove the failed episodes from that size's episode set. Do NOT top up with successes: the filtered library is smaller (e.g. size 500 with 40 failures -> 460 episodes), so both arms come from the same collected episodes.
- Refit everything the Cache group fits on the library (metric/whitening/PCA, anything else the standard build computes) on the filtered set, with the standard fit pipeline, exactly as if the library had been built from those episodes.
- Pure Cache group only (group `c`: no corrector, no long horizon, no stagnation guard, no knob, no model calls). Everything else identical to the existing unfiltered Cache cells: snapshot/runtime, serving config, horizon, test identities and seeds, GPU type.
- Scope: every model, every size.
  - MT50 (seed 7 test, 2,500 episodes per cell): π0.5, GR00T, DP, Fast-WAM; sizes 250/500/1000/1500/2000/2500. Libraries `/data/r17_library/mt50/<model>/bpool` (success labels in `READY.json` metadata.episodes[].success; size views `subsets/size_*/VIEW.json` episode_ids; fits `fits/size_*`); studies `/data/r17_library/mt50/studies/<model>_k*/STUDY.json`; unfiltered Cache results in the campaigns listed in `/home/weiland/projects/openpi_ext/scratch/r17_ops/livepage/build_1015/build.py` (CACHE dict).
  - LIBERO DP / Fast-WAM / Cosmos (official test A, 500 per cell, libero_spatial + libero_10): sizes 50..500; libraries `/data/r17_library/libero/<model>/<suite>/bpool`; study `exp/r17_integration/ops/cache_study/sizes/STUDY.json`.
  - LIBERO π0.5 / GR00T: the unfiltered Cache rows on the dashboard are R10 (`exp/offline_search/rounds/r10/REPORT.md`: π0.5 `bpool_cs`, GR00T `bpool_all`, nested 5..50 per task, failures kept). Find exactly which library, subsets and fit each R10 Cache arm used, and where per-episode success labels live. Note: `/data/r17_library/libero/groot/<suite>/*_sp16_manifest.json` reports 36–49 trajectories per task, which may already be success-only; establish what the R10 arms actually used. If the R10 arms cannot be reproduced on code that still runs, run the unfiltered Cache arms again alongside the filtered ones in the same pipeline (say so and give counts).
- My CPU count of failures per R17 library size (check it): MT50 π0.5 43–45%, GR00T 24–26%, DP 37–39%, Fast-WAM 41–42%; LIBERO DP 12–19%, Fast-WAM 2–15%, Cosmos 2–10%.

Deliver:
1. Filtered size views + refit artifacts written next to the originals under new names (never modify existing views, fits, studies or campaigns).
2. Cache-only study/campaign definitions per model, prepared like the existing cache studies (same launcher, same R/B/W topology as the unfiltered run), ready to launch.
3. An analysis script producing, per model × suite × size: filtered episodes kept, SR filtered vs unfiltered (MT50: group mean and per-task mean), paired difference by init, and a JSON with one row per cell.
4. CPU tests; a handback `exp/offline_search/rounds/r17/filter_failed/HANDBACK.md` with the failure counts table, episode totals per model, the GPU each campaign must run on (same as its unfiltered Cache cells), and exact prepare/launch/analyze commands.
</task>

<rules>
- CPU only: no servers, GPU, remote commands, git commits, pkill, rm -rf. Never touch credentials.
- tether: never push/pull/expose. If worker files must change, list them with an rsync command and leave it to me.
- Keep it simple: reuse the existing build, fit, study and launcher code; add the smallest new code; no new validation gates.
- If a source fact contradicts this brief, stop and report it instead of guessing.
</rules>

<output>
Final message: what was built, the failure-count table, per-model episode totals, and the handback path.
</output>
