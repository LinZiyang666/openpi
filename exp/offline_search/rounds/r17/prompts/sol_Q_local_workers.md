<task>
Owner exception (today only for this study): LIBERO workers may also run on the local machine weilandserver (88 CPUs, 251 GB RAM,
RTX 4090 48 GB with lots of free VRAM; the DP cache server also lives on this 4090). timan107 is CPU-saturated (load ~88/48), so
the cache study is worker-bound; timan108 is offline; timan1 stays forbidden.
Do:
1. Add a worker node option for the local machine to formal_eval / formal_workers / cache_lane (e.g. --worker-node local, and
   mixed fleets timan107 + local with a per-node worker count), without tether for local execution: LIBERO/robosuite env ready
   locally (verify; install into a private env under /data if missing, no system changes), EGL rendering on the local 4090
   (respect the server's VRAM; renderer VRAM admission), RAM admission with headroom for the local servers, worker recycling,
   per-worker device/node recorded in receipts.
2. Routing: local workers -> local DP server via localhost; local workers -> h100 servers via the same tether-exposed endpoints
   timan107 uses (or a better path), with the startup/port checks from today's fixes.
3. CPU tests, then a <=10 min smoke: 4 local workers on a short campaign against an existing endpoint or a local no-op policy,
   verifying observations match timan107-rendered ones for a few fixed init states (rendering GPU differs: report pixel
   deltas) and that nothing disturbs the running lanes (DP continuation dpc8513584592 on 4090, H100 lane csl10h).
4. Tell the coordinator (and write REQUESTS/Q_to_P6b.md for sol P6b, thread 01a12882-35d3-7201-85b3-3b62a4b417d2, who is
   extending the cache lanes to sizes 50-400) how to add N local workers to upcoming campaigns, and the recommended N per lane
   given local CPU/RAM/VRAM next to the DP server. Running campaigns cannot change fleets; apply from the next campaign.
</task>
<compact_output_contract>
End with: files/tests, local env status, smoke result (pixel deltas, throughput per local worker), recommended N per lane, and
exact flags.
</compact_output_contract>
<action_safety>
Do not stop/signal running lanes or fleets. No system package changes, no sudo. No pkill -f bare patterns, no rm -rf, no
/data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
