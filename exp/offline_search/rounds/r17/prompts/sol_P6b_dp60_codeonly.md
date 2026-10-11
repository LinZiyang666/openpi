<task>
CODE + CPU PREPARATION ONLY. Do NOT launch, arm, nohup, restart or stop any campaign/lane — the coordinator runs and guards all
experiments. (The coordinator stopped your two armed waiters 1130348/1130349 and re-armed the H100 size lane csznh itself.)
Change for the DP size lane (/data/mw_finetune/eval/cszmdp): use ~60 local workers (owner: local workers serve the local 4090).
It failed your renderer guard only because each worker reserves 1 GiB VRAM; measured renderer use is ~0.54 GiB (T1/Q). Set the
renderer reservation to 0.7 GiB, RAM reservation 3.5 GiB, global RAM headroom 8 GiB (not 10%), and count running fleets as
max(reservation, measured RSS). Keep the 4 DP replicas. Re-prepare cszmdp with local=60 (CPU), validate, tests.
End by giving the coordinator the exact command to arm it (it must wait for csl10dp PASS), and the same reservation numbers for
any other lane. Then stop.
</task>
<compact_output_contract>
End with: files/tests, cszmdp fleet/replicas, the exact arm command. No launches.
</compact_output_contract>
<action_safety>
CPU only, no process control of campaigns. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
