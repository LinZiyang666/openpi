<task>
CODE ONLY — do NOT launch/stop/resize any live campaign; the coordinator runs and guards experiments and will use your command.
Owner: the conductor should be able to take more workers while a campaign runs (we keep getting more devices: local, timan1 GPU 3;
VRAM on timan1 can be taken by others if we wait). Today fleets are frozen at prepare (--worker-counts) and adding workers needs a
stop + resume segment (~5 min idle). Implement live fleet expansion/shrink for running formal/cache campaigns:
- an "attach fleet" command that, given a running campaign dir (e.g. /data/mw_finetune/eval/<run>), starts an additional worker
  fleet on a node (timan107 / local / timan1 GPU 3 guard) that registers with the live driver and receives dispatches like the
  original fleet (same pinned runtime/worker payload, same recycle/admission/disk guards, per-worker node recorded in receipts);
- a matching "detach/drain fleet" command (finish current episodes, no replacements);
- the campaign's final seal must include every attached fleet's evidence; resume semantics unchanged.
CPU tests with a fake driver + real launcher paths. Give the coordinator exact commands and a 2-minute live-safe check procedure.
Coordinate with Q (editing ops/ for lane reorder + periodic sync right now): check the diff first, keep edits narrow.
</task>
<compact_output_contract>
End with: files/tests, attach/detach commands with an example for timan1=24 on a running H100 campaign, limitations.
</compact_output_contract>
<action_safety>
No process control of campaigns. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
