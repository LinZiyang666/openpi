<task>
The armed size lanes use the old fleet sizes; the owner wants MORE workers now that local and timan1 are available, and local
workers reserved for the local 4090 lane (no extra transport). See exp/r17_integration/REQUESTS/COORD_to_P6b.md (21:1x section).
Measured capacity: timan107 ~45 workers (48 cores, ~123 GiB free), local ~60 (88 cores, ~237 GiB free), timan1 GPU 3 ~40
(48 cores, ~468 GiB free; other users present).
Do:
1. Stop your two waiting arm processes (PIDs 834339, 834494) cleanly (they have not dispatched).
2. Re-prepare the size lanes:
   - 4090 lane (DP 50..400): local ~60 workers; raise DP replicas to 3-4 if needed so each cache server replica stays < ~70% CPU
     (cache groups take actions from the library; replica count does not change cached actions — record it); add timan107
     workers only if the server keeps up.
   - H100 lane (Fast-WAM 50..400 then Cosmos 50..400): timan107 ~45 + timan1 ~40, no local workers; Cosmos the same split
     (scaled if its server cannot keep up).
   Respect RAM/renderer/disk admission (4 GiB per worker, 6 GiB on timan1, disk guard). NOTE: Q is fixing a timan1 bug right now
   (planned-recycle exit 75 through T1's namespace wrapper counted as failures -> restart budget exhausted; see
   REQUESTS/Q_*.md / sol_Q_t1fix). Do not dispatch timan1 fleets until Q's fix is in; if it is not in by the time the H100 lane
   would dispatch, fall back to timan107 ~45 + local 0 and re-add timan1 later.
3. Validate (CPU) and re-ARM both lanes (nohup; wait for csl10dp / csl10h predecessor PASS). Report PIDs, fleets, durations.
</task>
<compact_output_contract>
End with: fleets and replicas per lane, validation result, armed PIDs and wait conditions, updated duration estimates.
</compact_output_contract>
<action_safety>
CPU only until predecessors PASS. Do not touch running lanes (dpc3464789869 on 4090; csl10h Fast-WAM continuation on H100).
timan1 GPU 3 only. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
