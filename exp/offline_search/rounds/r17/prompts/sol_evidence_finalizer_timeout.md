<task>
Two consecutive H100 Cosmos LIBERO formal runs finished with complete data but RECEIPT.status=FAIL because of
finalization_error "TimeoutError('CPU evidence finalizer readiness deadline exceeded')":
- /data/mw_finetune/eval/fa_co_r_u_l (original library · Cosmos · LIBERO · reduced · test-set levels): 30,000 accepted, 0 errors, verified_completion, all_owned_closed, GPU and ports released.
- /data/mw_finetune/eval/fa_co_f_f_oc1 (filtered library · Cosmos · LIBERO · full · knob-off continuation): 3,015 accepted, 0 errors, verified_completion; driver returncode 2 only because I retired the local and timan107 workers mid-run (recovered_worker_shutdown=true).

Do, CPU only (no GPU, no launches, no edits to chain scripts or running campaigns):
1. Find why the CPU evidence finalizer misses its readiness deadline (look at the finalizer code, its deadline, what it waits for, and the timings in each run's PHASE_TIMINGS.jsonl / receipts). Fix the cause or size the deadline from measured needs, so future runs do not hit it. Every new prepare picks up the fix; do not change frozen releases of running campaigns.
2. Re-run the finalization for the two runs above so their receipts reflect the real outcome (keep the old RECEIPT.json as a backup next to it). The retirement-induced returncode 2 with verified_completion=true must not count as a failure.
3. Check that `formal_a analyze` (exp.offline_search.rounds.r17.filter_failed_knob.formal_a) and the longseg analyzer accept both runs afterwards, and list any other closed formal runs with the same finalizer timeout so I can re-finalize them too.
Keep the /data I/O discipline (single reader, check /proc/pressure/io).
</task>
<compact_output_contract>Reply in Chinese, short: cause, fix, the two receipts' new status, other affected runs, tests.</compact_output_contract>
