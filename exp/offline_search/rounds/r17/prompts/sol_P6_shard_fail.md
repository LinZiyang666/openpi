<task>
The first real H100 library collection failed at about 100 episodes with only `RuntimeError('H100 bounded shard transfer failed')`. The log has no cause.
- Campaign: `/data/r17_library/cosmos_libero_libero_spatial/campaigns/teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be` (cosmos_libero, 2 replicas, 24 workers on timan108).
- `H100_SHARD_TRANSFERS.json` lists 90 shards, all verified and cleaned, with high_water_reserved_bytes 17.7e9 vs limit 20e9.
- Coordinator log: `/home/weiland/projects/openpi_ext/scratch/r17_ops/lib_cosmos_full_libero_spatial.log`.

1. Find the real cause from the campaign evidence, H100 `/tmp/r17_formal/<campaign>/...`, and the code. Candidates: quota reservation accounting (per-shard reserve too large vs 20 GB with two replicas streaming), a tether pull error, a race on an open shard, or a partial.
2. Fix it. Also make every transfer failure record and print the concrete cause: shard id, step, stderr.
3. Add a regression CPU test.
4. Check whether any H100 staging is left over for this campaign. Remove only this campaign's verified-or-orphaned stage files, and report what you removed.
Do not start GPU work; the coordinator relaunches. Do not touch the DP library collection on the 4090 (`dp_libero_libero_spatial` campaign running).
</task>
<compact_output_contract>Append "## Shard transfer failure fix" to HANDBACK_P6.md with the cause and the exact relaunch command. Final message: 4 lines.</compact_output_contract>
