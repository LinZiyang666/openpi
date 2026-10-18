<task>
Coordinator, 2026-10-17 15:3x. Two fixes, CPU only.

Evidence. `fa_fw_r_u_l` (original Fast-WAM LIBERO reduced, test-set levels, 30,000 episodes, 4090 server R2) launched 14:45:34. The launch probe in `filter_failed/worker_resilience.py:measure()` measured `local` capacity 44 (see `/data/mw_finetune/eval/fa_fw_r_u_l/WORKER_NODE_STATE.before_local_retire_1525.json`). The probe ran while the two Fast-WAM replicas were still loading (4090 at 438 MiB). `growth` is `local_worker_server_vram_headroom_mib`, default 16384. timan108 is offline, so `allocate()` spread its 20 over spare capacity and gave `local` 22 (plus timan1 12, timan107 36).

Once loaded, the replicas used 16.7 + 16.5 GB, the 22 local EGL renderers 13.6 GB, and another lane's (H100 fa_co_f_u_o) 2 local renderers 1 GB. That exceeds the 4090's 47.4 GB. Six episodes then failed with `FatalEpisodeError: retry budget exhausted; RuntimeError('LIBERO inference failed')`; the replica log shows `torch.OutOfMemoryError ... 111 MiB free`. These are k40 rows of `r17_0a398b0fbdcabda79f72_n100_chrg_k40` (eval:34:21, 34:28, 34:49, 36:46 and two more; grep `"error"` in `driver/journal.jsonl`).

At 15:25 I retired `local` for this run by rewriting `WORKER_NODE_STATE.json` (nodes timan1 12 + timan107 36, retired ['local']; backup beside it). The driver shut the local workers down. The 4090 is at 33 GB, and there have been no new errors since 15:24.

1. Launch capacity. For a campaign whose server is on the 4090 (`server_node == 'local'`), the `local` capacity probe must reserve the lane's admitted server memory: replicas × per-replica budget from the campaign/admission records the launcher already has, or the measured steady replica size from RUN_CONFIGS. It must not use the flat 16 GiB default, and it must not depend on whether the servers have loaded yet. Make the smallest change in `measure()`/`capacity()`. Show, for every queued 4090 campaign in `filter_failed_knob/formal_a_1016/CAMPAIGNS.json` and `ffk1015_pi05_mt50_reduced_descent`, the local capacity the new rule would allow on an otherwise idle 4090 with the other lanes' current renderer usage. The next queued 4090 Fast-WAM run is `fa_fw_r_f_l` (requested timan1 30, timan107 20, timan108 20). Add a CPU test.

2. Infra-error episodes must not count as task failures. The formal design decides each task level from 50 episodes with a 2-success tolerance, so an OOM failure can flip a decision. Today `cache/analyze_cache.py:accepted_terminal` (via `formal_a_analysis.load_r17`) takes these rows as success=False. Provide:
   (a) an exact re-run of only the errored task_uids (same YAML, init, seed, serving), runnable on the 4090 after `fa_fw_r_u_l` finishes, built the way earlier continuations were (e.g. `ffk1015_dp_mt50_reduced_descentc1`) so that `continued_lanes` picks it up; and
   (b) analysis treatment so that an accepted row whose `error` is an infra failure (`FatalEpisodeError ... retry budget exhausted`) is replaced by the continuation's accepted row for the same task_uid, and is reported as missing (never imputed) until then. Check that no other completed formal_A run has such rows. If one does, list it.
   Give me the prepare/launch commands for the continuation. Do not start it.
</task>

<rules>
- CPU only locally. tether exec only for read-only checks. Never push, pull or expose.
- Never start or kill servers, workers, drivers, chains or GPUs. Do not modify running runs (`fa_fw_r_u_l`, `fa_co_f_u_o`) or running chains (ffk-4090b, ffk-h100a, h100_disk_guard2).
- No pkill, rm or git commits. Never touch credentials.
</rules>

<output>
Final message: root cause confirmation, the two changes with file/function, test results, the per-campaign local capacity table, and the continuation commands.
</output>
