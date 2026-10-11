<task>
CODE/CONFIG ONLY — do NOT launch, restart, stop, drain or arm any campaign or lane; give the coordinator exact commands.
Deadline ~22:25 CDT for item 1.
1. Reorder the H100 work to minimise model switches (owner: YAML hot-switch exists, so finish one model completely before
   switching): Fast-WAM size-500 continuation (fastwamc5481985003, running on lane csl10h) -> Fast-WAM sizes 50,100,200,300,400
   -> Cosmos size 500 (your cosmos500w85_220813) -> Cosmos sizes 50..400. Today the csl10h lane will dispatch Cosmos 500 right
   after Fast-WAM finishes, and the coordinator armed P6b's size lane (/data/mw_finetune/eval/csznh, cache_lane --after-lane
   csl10h --wait-for-pass, coordinator background task) which runs Fast sizes then Cosmos sizes. Make the csl10h lane stop after
   the Fast-WAM continuation (or repoint its next slot) WITHOUT interrupting Fast-WAM, split/repoint csznh so Fast-WAM sizes run
   next on the same resident model if possible, and put Cosmos 500 + Cosmos sizes into one Cosmos lane afterwards. Prefer keeping
   one resident server per model across sizes (library VIEW hot-switch). Fleets: Fast-WAM sizes timan107 ~45 + timan1 ~40; Cosmos
   the same. Tell the coordinator exactly which process to stop (if any) and which commands to run, in order.
2. Disk: timan nodes are short on disk. Implement periodic incremental sync of each running campaign's remote run directory to
   the local machine (/data, via the ziyanglin.com rsync daemon path, bandwidth-limited, not tether) every few minutes, verify, and
   delete synced remote chunks immediately (keep only what resume/finalize still needs remotely). Tests. Give the coordinator the
   command to run the sync daemon (the coordinator will run it).
</task>
<compact_output_contract>
End with: the ordered command list for the coordinator (with any stop step), files/tests, and the sync daemon command.
</compact_output_contract>
<action_safety>
No process control of campaigns or lanes. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
