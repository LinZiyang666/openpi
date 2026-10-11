<task>
The Cosmos baseline resume finished its cells but closed FAIL:
run /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043 (resume of
formal_cosmos_libero_official2000_20261010_100533_5f558380, waiver timan108), journal has 274 accepted / 268 success,
RECEIPT error `RuntimeError('driver incomplete: 2 is not a completed campaign with resolved retry errors')`.
1. Determine which attempts errored and whether every errored cell was retried to an accepted result with the same
   seed/init (so the 274 cells are valid). If any cell is invalid, say which and give the minimal rerun command.
2. If valid, fix the closure logic so a driver exit with fully resolved retried errors is accepted (record the retries),
   CPU tests, and produce the sealed merged result for the full official-50 Cosmos K5 baseline (1726 retained + 274
   resumed = 2000) with per-suite / per-task SR and the provenance of both segments (use formal_eval aggregate/seal or
   the resume merge path). Do not launch GPU work.
3. Report the merged per-suite SR next to the Cosmos K2 arm hs2_cosmos_h100_k2incl_c16_t107_20261010_115120
   (libero_spatial and libero_10), noting which worker node each segment used (timan108 vs timan107).
</task>
<compact_output_contract>
End with: validity verdict for the 274 cells, files changed/tests, merged SUMMARY path, and a table: suite x
(K5 baseline n/SR, K2 n/SR).
</compact_output_contract>
<action_safety>
CPU only; do not launch, stop or signal campaigns. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018,
codex credentials. timan1 forbidden.
</action_safety>
