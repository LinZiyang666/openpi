<task>
Owner exception (this cache study only): LIBERO workers may run on timan1 using ONLY GPU 3 (RTX A6000 48 GB, currently free)
for EGL rendering. GPUs 0-2 belong to other users — never touch them. timan1: 48 CPUs (load ~5), ~440 GB available RAM,
our dir /scratch/zixuans8, tether node `timan1`.
Provision timan1 as a worker node exactly like timan107 (see how formal_workers deploys to timan107: payload/runtime paths under
/scratch/zixuans8/..., venv, LIBERO/robosuite assets, MUJOCO_EGL/EGL device selection, tether allow_roots and /tmp staging):
1. Bring the worker runtime/venv and LIBERO assets to timan1 (copy from timan107 or rebuild; verify hashes match the formal
   worker payload expectations); confirm EGL rendering works on GPU 3 only (CUDA_VISIBLE_DEVICES / EGL device pinned to the
   physical GPU 3) and fails closed if pointed elsewhere.
2. Measure: per-worker RAM/VRAM and sim throughput on timan1 with 4 test workers rendering a few fixed A init states; compare
   observation bytes/pixels with timan107 renders for the same states.
3. Write the node facts the launcher needs (paths, python, env vars, allowlisted GPU {3}, recommended max workers given CPU/RAM
   and that other users use the machine) to exp/r17_integration/REQUESTS/T1_to_Q.md. Do NOT edit launcher code
   (exp/r17_integration/ops/*): sol Q is changing it now (adding a local worker node) and will add timan1 next from your notes.
</task>
<compact_output_contract>
End with: provisioning status, EGL GPU-3 pin proof, per-worker resources/throughput, pixel comparison, recommended worker count.
</compact_output_contract>
<action_safety>
Only GPU 3 on timan1; do not touch other users' processes, GPUs 0-2 or other users' dirs. No sudo/system packages. Keep tether
calls modest. No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
