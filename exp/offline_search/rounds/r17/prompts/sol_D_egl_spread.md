<task>
Strong lead on the LIBERO worker "kernel memory leak": on timan107 right now (Cosmos K2 fleet, 24 workers) GPU0
(GTX 1080, 8 GB) is at 8085/8192 MiB with ALL 24 workers rendering on it, GPUs 1-7 are empty (2 MiB each), and
node MemAvailable falls ~5.6 GiB/min while process RSS stays flat; the memory comes back when workers exit. Same
pattern killed the Fast-WAM baseline (node OOM) and probably hung timan108 (DP + Cosmos fleets both on EGL GPU0,
as you found). Hypothesis: VRAM exhaustion on one GPU makes the NVIDIA driver back EGL/GL allocations with host
memory that no process RSS shows.

Do:
1. Make every formal LIBERO fleet spread its workers round-robin over all usable GPUs of the worker node
   (per-worker MUJOCO_EGL_DEVICE_ID / EGL device selection, whatever the LIBERO/robosuite path honors), with a
   per-GPU VRAM admission (measure per-worker VRAM) that also accounts for other R17 fleets and other users'
   usage on that GPU (timan108 GPUs 1-2 are xy51's). Extend your renderer-device guard accordingly. Record the
   device per worker in the fleet receipt. Results must not change (rendering device must not alter observations;
   verify byte-identical observations for a few fixed init states rendered on two different GPUs of timan107).
2. Prove the hypothesis on timan107 with a short smoke (<=10 min, e.g. 16 workers spread over 8 GPUs against an
   existing policy endpoint or a local no-op policy if the framework has one): node MemAvailable must stay flat.
   Do not disturb the running Cosmos K2 fleet (hs2_cosmos_h100_k2incl_c16_t107_*, ports 23240-23249).
3. Give per-worker RAM and VRAM so --worker-memory-mib is realistic, and the capacity of timan107 (owner expects
   80-90 workers) and timan108.
sol M (thread 01a126b8-3457-7f30-a120-0da1edf582ac) is investigating the same leak and may be adding worker
recycling; read exp/r17_integration/REQUESTS/M_to_*.md and its diff, and write findings to
exp/r17_integration/REQUESTS/D_to_M.md so you do not duplicate or clobber each other. Q edits ops/formal_resume.
</task>
<compact_output_contract>
End with: hypothesis confirmed or refuted (MemAvailable trace numbers), files changed, tests, observation identity
check, per-worker RAM/VRAM, node capacities, and whether the Cosmos baseline resume (274 cells) can now run on
timan107 (give the command; ports must not collide with K2).
</compact_output_contract>
<action_safety>
Do not stop or signal running campaigns. Do not touch xy51's processes or GPUs. timan1 forbidden. Kill only your
own PIDs; no pkill -f bare patterns; no rm -rf; no /data/i018, /dev/shm/i018, or codex credentials.
</action_safety>
