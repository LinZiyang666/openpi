<task>
Owner: the 4090 lane has lots of spare capacity. Right now the DP size-500 continuation dpc3464789869 (lane csl10dp, ~1440/3999
of the remaining l10 cells done at 21:47) runs 32 workers ALL on timan107, zero local; DP server replicas ~36% CPU each, 4090
~10% util. Owner rule: local workers serve the local 4090 lane first (no extra transport); the 4090 lane may also use external
workers. Local machine: 88 cores, ~237 GiB free RAM.
Do:
1. Restart the DP continuation as a new resume segment with a bigger fleet: local ~60 workers (localhost, Q's local worker node,
   4 GiB each, renderer on the local 4090), timan107 0 (free it for the H100 lane's Cosmos), DP replicas 3-4 so each cache server
   replica stays < ~70% CPU. Cached actions do not depend on worker count/replicas; for gated MISS calls the batch composition
   changes — record fleet/replicas per segment in the merged receipt. Relax the continuation "same total workers" check for cache
   campaigns accordingly (Q implemented same-total redistribution; coordinate via REQUESTS/P6_to_Q.md; Q is busy re-preparing the
   Cosmos campaign right now).
2. Stop the current segment cleanly at an episode boundary (accepted cells retained), start the new one, verify it accepts at a
   much higher rate, and keep it running. Do not touch the H100 lane.
3. Report the new run name, fleet, replicas, accepted/min before vs after.
</task>
<compact_output_contract>
End with: new segment name, fleet/replicas, rate before/after, files/tests.
</compact_output_contract>
<action_safety>
Kill only verified dpc3464789869-owned PIDs. Do not touch the H100 lane or other users' processes. No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
