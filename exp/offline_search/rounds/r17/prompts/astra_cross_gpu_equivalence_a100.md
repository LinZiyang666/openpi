Coordinator follow-up, 2026-10-17 23:0x, from the owner:

1. The new A100 may now be used for THIS equivalence work only. It is not yet a formal experiment device.
   - It is the tether node `a100` (ONLINE; public 149.165.152.232). Use `tether exec` only; never push, pull or expose. Move files by public-network rsync with `--bwlimit`.
   - It has passwordless sudo. Use sudo only when the work needs it (e.g. reading kernel logs, installing a missing system library for the serving runtime), and record every sudo command in your report.
   - Its disk is very limited. Check `df -h -l` first and keep the footprint minimal:
     - copy only the checkpoints/runtimes/recorded inputs a test needs;
     - no duplicate copies, no large caches or datasets;
     - delete only files you created there, and only after the test that needed them;
     - report the space used.
   - Short diagnostic replays on the A100 (≤15 minutes each) you may run yourself. Hand longer campaigns, and anything on the 4090 or H100, to me as exact commands.
2. If a 4090 replay needs GPU memory, say how much and for how long. I will free it, e.g. by temporarily dropping one replica of the running 4090 lane, and restore it after you finish. Do not touch the 4090 or H100 lanes yourself.

Continue the original task with this in scope: add the A100 to the layer (a)/(b) checks and to the knob-isolation order (TF32 off, highest matmul precision, compile off or fixed, same attention kernel, deterministic flags).
