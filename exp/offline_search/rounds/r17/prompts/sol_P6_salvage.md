<task>
Node-failure salvage for library campaigns.
- Campaign: `/data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3`
- What happened: cosmos_libero × libero_10 failed at 484/500 attempts when the timan108 tether agent briefly went OFFLINE ("node_offline: status=STALE", about 01:36). The receipt says all_owned_closed=false. 23 attempts errored at n_steps=0 ("LIBERO inference failed").
- Cleanup so far: the coordinator killed the orphan timan108 worker group by PGID. The H100 GPU lock is free.
- Leftovers: H100 still holds about 446 MB under `/data/r17_library/cosmos_libero_libero_10` (un-transferred shards); H100 `/tmp/r17_formal/<campaign>` is about 8 MB.
- Repair refused: `repair/library requires a closed collection receipt`.

Implement a salvage path (e.g. `--salvage FAILED_CAMPAIGN`, CPU + H100 transfer only, no GPU):
1. Drain and verify any remaining H100 shards for that campaign into the local campaign (SHA-checked, bounded), then clean the verified H100 copies.
2. Close the collection by reconstructing a closed receipt from what is provably complete: episodes whose decisions/acks are complete and accepted. Drop incomplete or torn episodes explicitly, with evidence.
3. Make the result acceptable to `--repair`, so repair re-runs every planned episode lacking a complete accepted non-error result (not attempted, errored, or dropped as incomplete).
Never fabricate or partially accept an episode. Keep the original campaign read-only apart from the documented salvage outputs.
CPU tests. Then give the exact salvage + repair commands.
- H100 is about to run the Fast-WAM library chain (ports 23340–23359, its own campaign dirs). Do not touch it.
- The 4090 runs DP libero_10. Do not touch it.
</task>
<compact_output_contract>Append "## Node-failure salvage" to HANDBACK_P6.md. Final message: 4 lines.</compact_output_contract>
