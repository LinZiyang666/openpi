<task>
The Fast-WAM official baseline failed at 391/2000 accepted episodes:
run dir /data/mw_finetune/eval/formal_fastwam_libero_official2000_20261010_101044_b0c4e68b,
RECEIPT.json status FAIL, error `RuntimeError('remote worker fleet failed: timan107')`.
Topology: server h100 (2 replicas), 24 workers on timan107, launched with formal_eval.sh before your
RAM-admission / bounded-restart fix (no --worker-memory-mib / --worker-restart-limit).

Right now the same topology is running the first horizon arm WITH your new flags:
`hs2_fast_h100_k10_c10_20261010_112043` (h100, 2 replicas, timan107 x24, --worker-memory-mib 6144
--worker-restart-limit 2, log /home/weiland/projects/openpi_ext/scratch/r17_ops/hs_fast_k10_c10.log).
Also running and NOT yours to touch: DP campaign hs2_dp_k100_c5_default_* (4090 + timan108 x20) and the
Cosmos baseline/K2 on h100 + timan108.

Do:
1. Root-cause the timan107 fleet failure from the run dir, the worker logs on timan107 (tether exec/pull),
   and node RAM/OOM evidence (dmesg/journal if readable). Say exactly which worker died first and why.
2. Say whether the running C10 arm is exposed to the same failure; if it is a launcher bug, fix it in
   exp/r17_integration/ops/ (CPU tests) so the next arms pick it up. Do not stop or restart any campaign.
3. Recovery for the baseline: can we keep the 391 accepted episodes and run only the missing
   (suite, task, init) cells with an identical config, producing one merged receipt that the paired
   analysis can treat exactly like a complete 2000-episode baseline? If formal_eval lacks this, implement
   a `--resume-from <run_dir>` (or equivalent) that refuses on any config/bundle/checkpoint mismatch.
   If merging is not sound, say so and we rerun from scratch.
</task>
<compact_output_contract>
End with: root cause (2-3 lines, with evidence path), C10 risk (yes/no + why), fix made (files, tests),
and the exact command to resume or rerun the Fast-WAM baseline on h100 + timan107 (ports must not
collide with 23173-23177 used by C10; use 23180/23189 if free).
</compact_output_contract>
<action_safety>
Kill only by PID, never pkill -f with a bare pattern, never rm -rf. Do not touch /data/i018, /dev/shm/i018,
or other users' processes. Do not touch codex credentials. GPU use: none needed.
</action_safety>
