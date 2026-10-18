<task>
Coordinator, 2026-10-17 18:5x. Two prepare failures from the H100 queue (ffk-h100fa). CPU only.

1. `fa_co_f_f_oc1` (your Cosmos continuation; the parent `fa_co_f_f_o` closed at 18:45:05 with PARTIAL, all_owned_closed=True, 2,985 accepted) failed prepare in 2 s. Log: `r17_ops/ff/fa_co_f_f_oc1.prep.log`. The failure comes from `worker_resilience.coordinator().call` → `tether exec --safe --timeout 10s timan1 -- env ...`: `File "<string>", line 184, in run` → `line 94, in install_runtime` → `ModuleNotFoundError: No module named 'exp'`, i.e. the timan1 GPU2 bootstrap path. A normal prepare through the same timan1 path works: `fa_co_r_u_l` prepared at 18:46–18:49 and started. So the problem is specific to the continuation's prepare. Fix it, verify with a CPU or read-only dry run, and tell me when `fa_co_f_f_oc1` can be prepared. The 4090 continuation `fa_fw_r_u_lc1` has the same shape, so check it too.

2. `fa_pi_f_u_o` / `fa_pi_f_f_o` (π0.5 full, H100 servers, legacy R15 workers) failed prepare with `RuntimeError: H100_CHAIN_BUSY lock=/home/weiland/trace_runs/os_closed_loop/h100_fleet_timan107.lock`. The 4090 is running R16 π0.5 levels (`fa_pi_r_u_l`) whose workers hold that timan107 lock. Its queue continues with more R16/R15 lanes on timan107 (fa_pi_r_f_l, fa_gr_r_*_l, fa_gr_f_*_l). So the H100 π0.5 full items (offs and later levels) would be blocked or skipped for many hours. Owner rule: workers are machine-agnostic and one busy or dead worker machine must never block experiments. Make the R15/R16 formal_A items run when timan107's lock is held by another fleet, with the smallest change. Either let two fleets share timan107 when measured GPU/RAM capacity allows (the frozen packages already apply EGL VRAM admission), or fall back to timan1 GPU2 through the existing `timan1_gpu2` bootstrap. Keep the same serving and plans. Tell me which you chose and the measured capacity.

Give me the exact queue names and order to re-queue on the H100 after `fa_co_r_u_l` (currently running, 30,000 episodes). Do not start anything.
</task>

<rules>
- CPU only locally. tether exec only for read-only checks. Never push, pull or expose.
- Never start or kill servers, workers, drivers, chains or GPUs.
- Do not modify running runs (`fa_co_r_u_l`, `fa_pi_r_u_l`) or running chains (ffk-h100fa, ffk-4090d, ffk-4090e, h100_disk_guard2).
- No pkill, rm or git commits. Never touch credentials.
</rules>

<output>
Final message: root cause and change for each, verification, and the H100 re-queue list.
</output>
