URGENT (I interrupted your fairness work; resume it after this):

The first real dynamic campaign failed in 3 minutes with 0 accepted episodes:
/data/mw_finetune/eval/fa_ls_fw_r_p_m (H100, Fast-WAM MT50 reduced pure+long control, maxima timan1 60 / timan107 60), launched 05:19:47, ended 05:22:46 rc=2, RECEIPT FAIL "driver incomplete".
Driver traceback: exp/r17_integration/conductor/driver.py:56 _record_pull -> ValueError("worker probe changed during the campaign").
driver/rejected_worker_probes.json: worker_id fa_ls_fw_r_p_m_timan1_0, the only differing probe field is renderer.pid (accepted 638357, new 638473). So a recycled/respawned generation reuses the same worker_id with a new renderer process, and the R17 probe validator treats it as a changed worker.

Do now, CPU only (no launches):
1. Fix it so every new generation is either a distinct worker identity or an explicitly attested new generation of the slot, without weakening the validator for anything else (endpoint, model, runtime, GPU identity must still be pinned). Reproduce the failure first in a CPU test with the real driver + your fleet spawn/recycle path, then show the test passing. Also check the R16/R15 legacy paths for the same pattern (fa_pi_f_u_o, R15 pi05 full knob-off on H100, is preparing right now in dynamic mode; tell me immediately if it will hit the same thing).
2. fa_ls_fw_r_p_m now has LAUNCH/RECEIPT files with 0 accepted episodes. Give me the exact command to reset and re-prepare it (keep its evidence as a backup; same plan/UIDs/seeds).
3. When the fix is in the tree and verified, tell me; I will release the H100 chain (it waits for "[autoscale] fixed"). The 4090 chain starts around 11:00 with R16 items and reaches R17 items later; the fix must be in before then.
Also: your maxima raise and fairness changes are still in the candidate area; do not switch them in until this fix is verified, then switch everything together with the before/after regression table.
