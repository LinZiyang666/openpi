<task>
Diagnose and fix: two formal campaigns sharing one worker node.
- DP official campaign `/data/mw_finetune/eval/formal_dp_libero_official2000_20261010_100532_bda5ad98` (server: local 4090; 24 workers on timan107) failed at 167/2000 with `RuntimeError('remote worker fleet failed: timan107')` at about 10:18.
- At 10:10 a second campaign, `formal_fastwam_libero_official2000_20261010_101044_b0c4e68b` (server: H100; `--worker-node timan107`, 24 workers), deployed its fleet to the same timan107. That campaign is still running fine.
- Suspicion: the second deployment overwrote or locked something shared, e.g. the shared worker root/dependency dir (`install_worker_dependencies`, WORKER_ROOT `.../libero311_v1`), the B/A input installs, a shared lock, or a pattern-based cleanup or health check that misattributed processes.
Find the actual cause from the DP campaign evidence (worker receipts on timan107 under `/tmp/r17_formal/<campaign>/`, `TETHER_CALLS.jsonl`, `driver.log`) and the code. Fix it so concurrent campaigns on one node are isolated: per-campaign paths, immutable shared installs, and no cross-campaign kill or health logic. Add a CPU regression test.
Do NOT touch the running Fast-WAM (timan107) or Cosmos (timan108) campaigns. No GPU work.
</task>
<compact_output_contract>Append "## Concurrent fleets on one node" to HANDBACK_Q.md: cause, fix, test. Final message: 3 lines.</compact_output_contract>
