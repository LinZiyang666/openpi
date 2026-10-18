<task>
Coordinator, 2026-10-16 16:3x. Owner ruling (formal design from now on): the per-task IR knob is never tuned on a development set. It is descended directly on the test set (LIBERO test A, all 500 episodes per suite; MetaWorld seed 7). The cache libraries are built from the development pool, so every LIBERO knob result that chose levels on a development set is invalid. The owner ordered: move all affected data into an abandoned trash directory (backup, never delete).

Affected = LIBERO IR-knob data whose levels came from a development-set descent, for both the original library and the success-filtered library, all models (DP, Fast-WAM, Cosmos, π0.5, GR00T), full and reduced steps:
- development-set descent runs (including their development-set pure-reference arms),
- test-set runs whose levels were frozen from those descents (official-A tests, raise/retest rounds, uniform-level comparisons),
- every analysis/result table derived from them.
Known locations (verify, and find any others):
- /data/mw_finetune/eval/knobdev*, knobtest*;
- /data/mw_finetune/eval/ffk1015_{dp,fastwam,cosmos}_libero_{reduced,full}_dev, ffk1015_dp_libero_{reduced,full}_test00/test01 (and the renamed bad-assets copy of full test01);
- /home/weiland/trace_runs/os_closed_loop/r14_descent_*, r14_pilot_*, r15_descent_*, r16_dev_*, r16_test_*, r16_conductor_dev_*, r16_conductor_test_*, r16_conductor_retest_*;
- /home/weiland/projects/openpi/exp/offline_search/rounds/r16/k1ref/runs/{dev*,test*} and its results/analyses;
- repo result folders: rounds/r17/knob/dev_results_*, test_results_*; rounds/r14/descent, descent_astra, knob; rounds/r15/descent; rounds/r16/dev_results_conductor, retest_v1*, retest_v1_v2_results, retest_v1_v2_v3_results, phase/retest result tables; rounds/r17/filter_failed_knob/results/dp_libero_*; rounds/r17/filter_failed_knob/vla_full_r15 (never-launched development-set design).

NOT affected — keep in place untouched:
- all MetaWorld knob data (descended directly on seed 7), all pure-Cache studies (original and filtered), pure-inference baselines and control reruns (hs7*ctl*, tpi7*, tgr7*, tdp7*, tfw7*, r15_purecorr_*, r15_qk_pilot_*, r16_phase1_*, smoke runs), libraries, and every test-set knob-off run (complete system, stall gate only): fsred*, ffk1015_*_libero_reduced_off*, rounds/r16/k1ref/runs/off, rounds/r17/filter_failed_knob/vla/runs/off.
- Anything running or queued: check `ps -eo pid,args | grep "[c]hain_"` and the queue scripts under /home/weiland/projects/openpi_ext/scratch/r17_ops/ff/run_*.sh; every item they will prepare must still prepare after your move (in particular ffk1015_pi05_libero_reduced_off and ffk1015_groot_libero_reduced_off read rounds/r16/artifacts/*_task_descent.pkl and rounds/r16/k1ref definitions; ffk1015_cosmos_libero_reduced_off_u/off; ff1015_cosmos_li_04; ffk1015_dp_mt50_reduced_descentc1 and the MetaWorld lanes).
- Definitions, runtime/worker archives, assets, ready.json/CACHE_CAPACITY_ADMISSION.json and level/calibration artifacts that a new direct-test-set descent would build from. Move outcome data (journals, per_step, episode/result files, server batching/serving logs, worker logs, receipts, summaries, descent/test tables, analyses); where a run directory is only outcome data or a never-launched development design, move the whole directory.

How:
- Trash roots on the same filesystem as the source (rename only, no copy): /data/mw_finetune/_trash_devknob_20261016/, /home/weiland/trace_runs/_trash_devknob_20261016/, /home/weiland/projects/_trash_devknob_20261016/ (repo files: keep the repo-relative path under it). Preserve relative paths.
- First write the full manifest (source path, trash path, size, reason) to /home/weiland/projects/_trash_devknob_20261016/MANIFEST.json and a dry-run listing; then move; then verify every manifest entry moved and every kept dependency of running/queued items still exists.
</task>

<rules>
- mv only. No rm, no rm -rf, no copying large data. Never touch credentials. No git add/commit (moved tracked files will show as deleted; leave them).
- No servers, GPU, workers, chains, pkill. tether: none needed.
- Do not edit code or queue scripts. Do not touch /home/weiland/projects/openpi/logs/session_handoff.md.
</rules>

<output>
Final message: totals moved per category (count, GiB), anything you judged ambiguous and left in place, and the dependency check result for running/queued items.
</output>
