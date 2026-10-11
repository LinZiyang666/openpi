<task>
Owner: local has 251 GiB RAM and only ~61 GiB is used; 33 DP workers is far too few. The admission is too conservative.
Measured cache-worker peak RSS is ~3.1-3.5 GB (timan107 OOM records). Do now:
1. For the local node and cache campaigns: per-worker reservation 3.5 GiB (not 4), global headroom 8 GiB (not 10% = 25 GiB),
   keep the 8 GiB server-growth reserve; reservations of running fleets should count max(reservation, actual RSS) not
   reservation alone. Tests.
2. Grow the DP continuation on the 4090 to ~45 local workers immediately (new segment at an episode boundary, as you just did),
   then, when the H100 Fast-WAM continuation fastwamc5481985003 finishes (~22:30-22:40) and its 16 local workers exit (the next
   Cosmos campaign uses timan107 + timan1 only), grow DP again to ~60 local workers. Make this automatic if the lane can do it
   (a watcher that re-segments when admission headroom appears), otherwise do it by hand at that time.
3. Apply the same reservation/headroom numbers to P6b's size-grid lanes (coordinate via REQUESTS/P6_to_P6b.md) so the DP size
   lane uses ~60 local workers and the H100 size lanes don't take local workers.
Report segments, worker counts and accepted/min.
</task>
<compact_output_contract>
End with: new reservation/headroom rule, DP segment names with worker counts, accepted/min, files/tests.
</compact_output_contract>
<action_safety>
Do not touch the H100 lane's running fleet. Kill only your own verified DP PIDs. No pkill -f bare patterns, no rm -rf,
no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
