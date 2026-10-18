<task>
Coordinator, 2026-10-17 21:3x. Owner ruling plus the defect I verified. CPU only; I launch everything.

Owner rule: in the complete system every decision executes the whole long execution segment, whether it is a library/cache hit or a model MISS.

Defect:
- In R17 lanes where `commit_horizon` (10) > `layout.replan_steps` (5), i.e. GR00T/DP/Fast-WAM on MetaWorld and DP/Fast-WAM on LIBERO, `exp/r17_integration/common/serving.py` evaluates `force_look` at the step-5 midpoint: `can_tail = ... and not force_look`. `force_look` (`search/cache_recipe.py:312`) is true when the IR knob's `periodic_gap` clock fires or the stall gate's `look_due` arrives. The remaining 5 steps are dropped and a new decision is made.
- Evidence from `grmk1_ful/driver/per_step.jsonl`, first 400k lines, 9,008 episodes: segments cut at step 5 were koff 6%, k10 20%, k20 34%, k50 62%.
- Not affected: Cosmos (commit 16 = replan 16); π0.5 MetaWorld (chunk 5); legacy R16/R15 π0.5/GR00T LIBERO (`decisions_*.jsonl`: policy→policy_tail, cache→cache_blind always); pure-cache groups (commit 5, no stall gate); pure references.
- The owner ordered: move all affected data to a trash directory (backup, never delete), rerun everything affected, balanced across both GPUs.

Do, in this order:

1. Fix. A forced look (knob or stall gate) may only take effect at a segment boundary (cursor 0). A committed segment always runs to `commit_horizon` unless the episode ends. Make the smallest change in serving/recipe; keep the knob and stall clocks otherwise unchanged and record any look that was deferred. Add a test that fails on the old behaviour. Then prove it on the real path with a CPU/replay or minimal dry check that every head is followed by its `_tail`.

2. Audit, with an empirical criterion. For every R17 run under `/data/mw_finetune/eval` (MetaWorld and LIBERO, original and filtered, knob levels, knob-off and cache-variant studies, including #133's cache-variant lanes), use `driver/per_step.jsonl` to count heads whose next decision is a new head at +`replan_steps` (a midpoint cut) rather than its `_tail`. Any run with cuts > 0 is affected. Also list every result/analysis derived from affected runs, e.g.:
   - `rounds/r17/knob/mt50/{groot,dp,fastwam}_{full,reduced}`, `knob/full_system_*`;
   - `filter_failed_knob/results/*_mt50_*`;
   - the DP/Fast-WAM rows of `formal_a_1016/RESULTS.json`;
   - continuation drafts and `CONTINUATIONS.json` entries;
   - website data built from them.
   Read the per_step files by streaming; they are up to 10 GB.

3. Trash, mv only. Use the same layout as `/home/weiland/projects/_trash_devknob_20261016/` (sol_trash_devknob.md): trash roots on the same filesystems, named `_trash_longseg_20261017`. Write MANIFEST.json (source, trash path, size, cut statistics) and a dry run first, then move, then verify. Keep definitions, runtime/worker archives, assets, libraries, fits, stall artifacts and pure references in place; move outcome data and derived results. Do not touch running runs (`fa_co_r_u_l`, `fa_pi_r_u_l`) or anything Cosmos/π0.5/legacy-VLA.

4. Rebuild every affected campaign fresh with the fixed code: the same plans, seeds, libraries, sizes, levels and serving as before, with new short names. Also add, per affected model × step mode × benchmark, one "pure + long segment" lane: model call at every decision, full `commit_horizon` execution, no cache. That is 2,500 episodes on MetaWorld seed 7 and 500 per LIBERO suite. Analysis can then decide the fallback level. Register the formal_A LIBERO units so that `formal_a_queue` builds levels from the new off results.

5. Balance. Comparisons need same-GPU, same-step pure references and same-GPU IR timing. List which pure references exist on H100 and on the RTX 4090 for each model × step mode × benchmark, and the cost of any missing one. Propose the split of all rebuilt work plus the unaffected queued work into two ordered queue lists (H100, 4090) that finishes earliest. Use measured rates from the previous runs; RUN_CONFIGS has the per-GPU configurations. Currently queued unaffected work: 4090 after `fa_pi_r_u_l`: fa_pi_r_f_l fa_gr_r_u_l fa_gr_r_f_l fa_gr_f_u_l fa_gr_f_f_l ffk1015_pi05_mt50_reduced_descent; H100 after `fa_co_r_u_l` (~02:00): fa_co_f_f_oc1 fa_pi_f_u_o fa_pi_f_f_o fa_co_r_f_l fa_co_f_u_l fa_co_f_f_l fa_pi_f_u_l fa_pi_f_f_l.

Give me prepare/launch command pairs and the two ordered lists. Do not launch anything.
</task>

<rules>
- CPU only locally. tether exec only for read-only checks. Never push, pull or expose.
- Never start or kill servers, workers, drivers, chains or GPUs.
- Do not modify running runs or running chains (ffk-4090d, ffk-4090fa, ffk-h100fa, ffk-h100fc, h100_disk_guard2).
- mv only; no rm, no pkill, no git commits. Never touch credentials. Do not edit `logs/session_handoff.md`.
</rules>

<output>
Final message:
- the fix and tests;
- the audit table: run, cut %, affected yes/no;
- trash totals;
- rebuilt campaigns with episode counts;
- the reference inventory per GPU;
- the two queue lists with estimated hours.
</output>
