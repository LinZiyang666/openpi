<task>
Root cause of the 20:2x-20:31 failures (DP continuation dpc8513584592 fleet death "receipt unavailable", Fast-WAM resume
`mkdir /tmp/r17_formal/fastwamc4456346641: No space left on device`): timan107's root filesystem (/, /tmp and /scratch share it,
212 GB) hit 100%. Our per-campaign remote run dirs + worker tars under /scratch/zixuans8/mw_r17/formal/runs had grown to 12 GB
(~0.25-1 GB per campaign) and a 9 GB timan1 staging copy by sol T1 (/scratch/zixuans8/mw_r17/timan1_provision) filled the rest.
The coordinator removed 24 remote copies of PASS campaigns whose timan107.tar is verified locally (freed 6 GB; now 98%) and
relaunched both lanes (cache_lane h100 csl10h and 4090 csl10dp, resume-aware).
Do (CPU; do not disturb the running lanes):
1. Add automatic remote cleanup: after a campaign's evidence tar is pulled AND its local SHA/size is verified AND the receipt is
   sealed, remove that campaign's remote run dir and tar on the worker node (rm without -f, path-confined to our runs dir,
   logged in the receipt). Never remove anything still needed for resume/finalize.
2. Add a disk guard to launch/recycle: refuse to start (or pause dispatch) when a worker node's free space on the run filesystem
   is below a threshold (e.g. 8 GB, configurable), with a clear error, and report free space in receipts.
3. Do the same for the local machine and timan1 when they become worker nodes (Q is adding local workers, T1 provisioned timan1).
4. Append the pitfall to REQUESTS (for RUN_CONFIGS): shared root FS on timan107; never stage bulk copies there.
Tests.
</task>
<compact_output_contract>
End with: files/tests, cleanup rule, guard threshold, and current free space on timan107.
</compact_output_contract>
<action_safety>
Delete only our campaign run dirs/tars that meet the rule; never other users' or other projects' data (Isaac-GR00T,
openpi_trace*, libero_sim, etc.). No rm -rf, no pkill -f bare patterns, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
