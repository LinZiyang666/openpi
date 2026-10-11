<task>
Fast-WAM arm /data/mw_finetune/eval/hs6fw_c20 accepted 1000/1000 (one LIBERO inference error on task 34 init 48 resolved by
retry) but closed FAIL: `RuntimeError('remote worker close/snapshot incomplete')`, driver_completion recovered_worker_shutdown
=True. The coordinator retry queue continued to c32 (first attempt hit `Address already in use` right after c20's close; a
re-prepared hs6fw_c32s1 is retrying). Do (CPU only):
1. Recover hs6fw_c20 to a sealed PASS if the evidence supports it (preserve the original receipt), or say exactly what is
   missing and whether the 1000 cells are still valid for the paired analysis.
2. Fix the close path so this case ("all cells accepted, worker close/snapshot incomplete but fleet verified gone") is
   handled like your rc2 ACK recovery for all models, and so the next campaign cannot hit "Address already in use" from
   the previous campaign's ports (wait for port release or pick free ports within the model's window).
3. Then run the CPU seal barrier you documented (Q_to_R.md:99) for every finished hs5dp_*/hs6fw_* arm that is not PASS yet.
</task>
<compact_output_contract>
End with: hs6fw_c20 status, files/tests, port-release fix, and the seal-barrier result per arm.
</compact_output_contract>
<action_safety>
CPU only; do not stop/signal running campaigns or queues. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
