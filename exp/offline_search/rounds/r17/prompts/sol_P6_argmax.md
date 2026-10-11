<task>
H100 repair launch failed immediately: `OSError(7, 'Argument list too long')`.
- Campaign: `/data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_repair_20261010_023755_54360429`
- Command: the exact repair command from your "Node-failure salvage" section, `--repair .../teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3/salvage`, server h100, workers timan108.
The local DP repair worked, so the H100 path is passing something large (episode list / campaign JSON / capture bindings) as argv to `tether exec` or a subprocess. Move it to a file staged via the existing SHA-checked transfer and pass the path instead. Add a CPU test with a large repair cohort.
Do not start GPU work. H100 is running the Fast-WAM libero_10 library campaign (ports 23340–23359); do not touch it. Give the exact relaunch command for the Cosmos repair.
</task>
<compact_output_contract>Append "## H100 repair argv fix" to HANDBACK_P6.md. Final message: 3 lines.</compact_output_contract>

<task_2>
Extend salvage to LOCAL campaigns and to "missing worker result" episodes.
- Campaign: `/data/r17_library/dp_libero_libero_10/campaigns/teacher_dp_libero_libero_10_500_20261010_005446_9bd136a9` (local 4090 + timan107; receipt FAIL, all_owned_closed=true).
- State: 497/500 accepted with 0 error attempts. Three planned episodes (suite task 30, inits 15/18/20) never returned a worker result; their workers apparently hung.
- `--repair` refuses: "missing worker result cannot attest zero progress: …eval:30:15".
- `--salvage` refuses: "salvage requires matching FAILED full H100 library campaign".
Allow salvage for local campaigns. For missing-result episodes, drop any partial capture rows for those UIDs into quarantine with evidence, never into the library, so repair can re-run them as fresh attempts. Same strictness otherwise. CPU tests. Give the exact salvage + repair commands for this campaign; the 4090 is idle and waiting for them.
</task_2>

<task_3>
A third STALE event hit at about 03:02, this time on the H100 agent: `tether exec --safe --timeout 10s h100 -- env … node_offline: status=STALE`. It aborted `/data/r17_library/fastwam_libero_libero_10/campaigns/teacher_fastwam_libero_libero_10_500_20261010_023853_9d2cd40c` at about 35/500 (all_owned_closed=true). Different nodes, recurring, always while our campaigns run. Hypothesis: our own tether traffic (frequent lease/liveness probes on several nodes plus per-shard `tether pull`/exec) starves agent heartbeats at the broker.
Measure the tether call rate of a running library campaign from code/receipts, then reduce it:
- lease/liveness probe interval ≥ 30 s;
- batch shard transfers (several shards per pull);
- reuse one remote session where possible.
Keep the new transient-offline tolerance. Report calls/minute before and after.
</task_3>
