<task>
Coordinator decision: unblock the Cosmos baseline resume (274 cells) now on timan107; timan108 may stay offline
for hours. The closure gate exists to prevent double-counted or stale episodes. With the source driver/proxy dead
(RECEIPT FAIL, driver closed), the h100 replicas gone, and the resume campaign using a new run id, new driver
port and new policy ports, a surviving timan108 worker cannot deliver anything into the resumed campaign.
1. Add an explicit, narrow waiver to the resume gate, e.g. `--resume-waive-unreachable-source-node timan108`,
   allowed only when: the source receipt is terminal, the source driver/proxy/replicas are verified closed, the
   named node is the only unverified one and is unreachable via tether, and the new campaign's ports/run id differ.
   Record the waiver, the evidence and the time in the resume receipt and the merged SUMMARY. CPU tests.
2. Do not launch the campaign; return the exact final command (h100 Cosmos replicas, timan107 workers, ports
   23243-23248 as you staged, your recycling/VRAM flags). The coordinator launches it.
Others editing ops/ now: Q (prestaging/queue helper), P3 (batch flags). Check diffs, keep edits narrow.
</task>
<compact_output_contract>
End with: files changed, tests, and the final command.
</compact_output_contract>
<action_safety>
No GPU, no campaign launch, no signals to any process. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
