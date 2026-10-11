<task>
Urgent diagnosis + mitigation: kernel-side memory growth on timan107 kills LIBERO worker fleets.

Evidence (coordinator, 11:4x CDT):
- timan107: MemTotal 220 GiB. Running: 24 LIBERO worker_entry processes of Fast-WAM campaign
  hs2_fast_h100_k10_c10_20261010_112043 (server h100), EGL rendering on GPU 0 (~7.8 GB VRAM).
- Process AnonPages flat at ~55 GiB (each worker RSS ~2.6-3.0 GiB), Slab ~3 GiB, Cached ~3.4 GiB, no tmpfs use,
  no ZFS, no D/Z processes, but ~95 GiB of MemTotal is unaccounted and MemAvailable fell 76.2 -> 69.5 GiB
  between ~11:45 and 11:48:50 while AnonPages stayed flat. Per-worker: 60 fds, 44 /dev/nvidia* fds, ~133 nvidia maps.
- The Fast-WAM baseline (same node, 24 workers) died at 11:17 from a node-wide OOM (Q: worker 18 PID 423719 OOM-killed,
  then the launcher stopped the fleet). At 11:30, with a fresh fleet, node "used" was only 49 GiB -> the hidden
  memory was released when the old workers exited, so it is per-process.
- timan108 earlier today: all our LIBERO workers hung in D state (rwsem_down_write_slowpath) and the tether agent went
  offline; may be the same mechanism.
At the current rate C10 hits OOM around 12:30-12:40.

Do (read-only on the running campaign; do not stop it):
1. Identify where the memory goes (e.g. /proc/meminfo deltas over time incl. KReclaimable, VmallocUsed, Percpu,
   PageTables, nvidia driver allocations / pinned host memory, /proc/<pid>/smaps_rollup incl. Pss/locked, per-episode
   growth vs env reset / EGL context creation in examples/libero worker path). Quantify GiB per worker per episode.
2. Mitigation that keeps results identical: e.g. reuse the EGL context / env across episodes, or recycle each worker
   process after N episodes (episode-level dispatch already exists; a recycled worker must not change any episode's
   seed or init). Implement it behind a flag with CPU tests, default ON for formal_eval LIBERO fleets, and give the
   per-worker memory to use for --worker-memory-mib so admission matches reality.
3. Capacity: with the fix, how many LIBERO workers fit on timan107 (owner expects 80-90) and on timan108.
4. If C10 will OOM before it finishes (~1000 episodes, at 374 at 11:47), say so early in an AGENT message
   with the projected time; the coordinator decides.
Coordinate: Q is editing exp/r17_integration/ops/ (formal_resume, worker_memory); P3/P4 edit Fast-WAM/DP serving.
If you need changes in their files, write exp/r17_integration/REQUESTS/M_to_Q.md.
</task>
<compact_output_contract>
End with: root cause + evidence, GiB/worker/episode, the fix (files, flag, tests), recommended --worker-memory-mib,
capacity per node, and C10 OOM projection.
</compact_output_contract>
<action_safety>
Do not kill or signal any campaign process. No GPU use. timan1 is forbidden. timan108 may be offline; do not
hammer tether (batch probes, >=30 s apart). No pkill -f bare patterns, no rm -rf, do not touch other users'
processes, /data/i018, /dev/shm/i018, or codex credentials.
</action_safety>
