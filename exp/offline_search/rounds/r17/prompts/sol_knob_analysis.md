<task>
Write the two analysis scripts for the R17 per-task IR-knob study (design layer 5, "逐任务旋钮下降"). CPU only. Data does not exist yet; build and test against synthetic lanes plus the real loader on existing cache lanes.

Study (fixed by the coordinator):
- Models DP, Fast-WAM, Cosmos; suites libero_spatial, libero_10; library sizes 50/100/200/300/400/500; 10 tasks per suite.
- Levels from highest to lowest inference: `pure` (every decision calls the model), `k50`, `k40`, `k30`, `k20`, `k10` (periodic knob calibrated to that miss ratio), `koff` (complete group, stall gate only). Campaign `cache_group` names are `<base>_<level>`, base `chrg` for DP/Fast-WAM and `crg` for Cosmos (e.g. `chrg_k30`, `crg_koff`); `library_size` is in task extra as today.
- IR proxy (owner: no cost profile yet, use the miss ratio) = model calls / (executed controls / C0), C0 = 5 for DP/Fast-WAM, 16 for Cosmos. Pure inference = 1.0.
- DEV = B pool, init indices 0..29, every task × level × size (one lane per model, `init_pool='B'` in task extra). The pure-inference reference on B is NOT run: use the teacher library episodes in `/data/r17_library/<model>_libero_<suite>/bpool/READY.json` (`metadata.episodes`: `task_name`, `orig_init_idx`, `success`; check you get exactly one episode per task/init, 500 per suite).
- TEST = official A, inits 0..49, one frozen level per task; pure-inference controls on A: DP `hs5dp_ctl_142812`, Fast-WAM `hs6fw_res_152931`, Cosmos `formal_cosmos_libero_official2000_20261010_100533_5f558380` + its resume `formal_cosmos_libero_official2000_20261010_130639_a8ca2043` (all under `/data/mw_finetune/eval/`).
</task>

<scripts>
Location `exp/offline_search/rounds/r17/knob/`. Reuse the loader in `exp/offline_search/rounds/r17/cache/analyze_cache.py` (`load_lane`, `accepted_terminal`, `canonical`, the per-episode `cost` counters `controls`/`policy`/`library`) by import or the smallest refactor; do not copy it. Teach it the knob group names if needed without changing existing results.

1. `descent.py --lane <dev lane> [--lane ...] --out <dir>`
   - Per model × suite × size × task, paired by init against the B teacher reference (n = 30): start at `pure` (passes by construction), visit k50 → k40 → k30 → k20 → k10 → koff; a level passes iff `100*(cand_successes - pure_successes) >= -tol*n` (integer arithmetic, equality passes); stop at the first failure and select its predecessor; if all pass select koff. Primary tol = 5 pp; also emit 0 and 2 pp tables. This is the frozen R14/R16 rule (`exp/offline_search/rounds/r16/PROTOCOL.md`).
   - Outputs: `levels.json` = `{model: {suite: {size: {task_name: level}}}}` (5 pp), `levels_tol0pp.json`, `levels_tol2pp.json`, `descent.json` (per task: every level's successes, n, IR proxy, pass/fail, first-failure, non-monotone recoveries), and `DESCENT.md` in plain Chinese for the owner (no agent codenames, no abbreviations in tables): per cell the 10 task levels, the stitched selected SR and IR proxy on B vs B pure SR, and a short totals table. Missing or partial levels: report what exists, mark the task as incomplete, never impute.
2. `test.py --lane <test lane> [--lane ...] --out <dir>`
   - Per model × suite × size: n, SR, IR proxy, paired difference vs the pure-inference control on A (gains/losses, the paired bootstrap 95% interval with 5,000 draws seed 20261005 as in R16), plus per-task SR vs control.
   - Raise list for a retest (owner's R16 rule, integer comparisons): task SR below control by ≥ 10 pp → `pure`; by 6–9 pp → one level up, but if already `k50` → `pure`; `pure` stays. Write `raise.json` (`{model:{suite:{size:{task:new_level}}}}`) and `TEST_A.md` (Chinese, same style as `exp/offline_search/rounds/r16/REPORT.md`).
</scripts>

<keep_it_simple>
Owner rule: main function plus a few necessary checks. No gates, seals, fingerprints, hash pins or validation layers; read lanes whatever their receipt status. Short code.
</keep_it_simple>

<verification>
Synthetic lanes (temp dirs mimicking campaign layout: `campaign.json`, `assets/CACHE_STUDY.json`, `EPISODE_PLAN.json`, `driver/journal.jsonl`, `driver/attempts.jsonl`, `driver/per_step.jsonl`) covering: a task passing all levels (→ koff), a first failure at k30 (→ k40), a non-monotone recovery, a missing level. Also run the real loader on `/data/mw_finetune/eval/csztd_r3_l48` (read-only) to show it still loads. Check the B teacher reference counts for all six model/suite libraries.
</verification>

<action_safety>
Edit only `exp/offline_search/rounds/r17/knob/*` and, minimally, `exp/offline_search/rounds/r17/cache/analyze_cache.py` (its outputs for existing lanes must not change; rerun its tests). Read-only on /data. Do not start/stop anything, no GPU, no remote nodes, no git commit. Two other agents are editing `exp/r17_integration/*`; do not touch it.
</action_safety>

<handback>
Reply with: what each script does (a few lines), test commands and results, the B reference SR per model/suite (pure inference on B inits 0..29 and 0..49), and the exact commands I will run on the real lanes.
</handback>
