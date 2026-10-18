<task>
Owner directive, 2026-10-17 22:3x. Build a dynamic worker autoscaler for our conductor/runtime. A configured worker count becomes only the desired maximum. During the whole run, the runtime decides how many workers are active, by node memory and by measured throughput, and keeps changing that number.

Context: your report `rounds/r17/ops_incidents/WORKER_MACHINE_CRASHES.md` found 15 global OOMs on timan107 today. The owner judged static caps such as 32 per node too conservative. Tonight, Cosmos with 85 workers ran slower than with 44, because workers queued at the H100 server: round trip p50 0.95 s / p90 4.1 s against server cost 0.2–0.37 s.

Design. Refine it and implement it:
1. Signals, collected node-locally with no high-frequency tether polling:
   - node MemAvailable and `/proc/pressure/memory`;
   - per-worker RSS and lifetime peak, from the node fleet supervisor over its own children;
   - per-episode worker time split, sim/render vs waiting for the server;
   - global accepted rate and server queue depth or in-flight counts, from the driver.
   Report through existing channels (supervisor status / heartbeat, worker pull payloads).
2. Memory guard: fast, seconds, hard.
   - Admit one more worker only if MemAvailable − floor (default 20 GiB) − projected remaining growth of live workers ≥ projected peak of a new worker. Take the projection from the observed p90 peak of recent generations of the same campaign, not a fixed reservation.
   - Below the floor, stop spawning: do not refill recycled slots.
   - Below a critical line (default 10 GiB), shrink: retire the newest workers at their next episode boundary.
   - Add back gradually after a hold time above floor + hysteresis.
3. Throughput tuner: slower, minutes, within the memory envelope.
   - Hill-climb / AIMD on the active count using windowed accepted rate.
   - Stop adding when the marginal gain is below a threshold, or when workers spend most of their time waiting for the server.
   - Shrink toward N ≈ rate × unqueued cycle time when queueing dominates.
   - Probe upward periodically, so it recovers when the network or server gets faster.
   - Never exceed the configured maximum.
4. Actuation:
   - Scale down gracefully. The driver stops giving new episodes to a selected worker, which exits after its current episode, and the supervisor does not refill the slot. This generalizes the existing node-retire path (`WORKER_NODE_STATE.json` + MSG_SHUTDOWN in `filter_failed/worker_resilience.py install_driver`) to individual workers.
   - Scale up through the existing spawn path. Stagger it, at most one environment creation per GPU at a time.
   - Place new workers on nodes in proportion to memory headroom. No tiny remnant slices (e.g. local×2).
5. Invariance:
   - Episode plans, task UIDs, inits and seeds must not depend on which worker or how many workers run. Prove it with tests.
   - Log every scaling decision to an append-only JSONL: time, node, ±n, reason, signal values, resulting counts.
6. Scope:
   - R17 path first: `exp/r17_integration/ops/formal_eval.py` launch/probe loop, `formal_process.py` supervise, `worker_lifetime.py`, `local_workers.py`, `formal_workers.py`, `filter_failed/worker_resilience.py`, `filter_failed/timan1_gpu2.py`.
   - Legacy R15/R16 (`rounds/r16/conductor.py`, `src/openpi/conductor/{driver,agent}.py`, `closed_loop/ops/h100/*`): at least the memory guard. Also remove the "64 workers on one host" requirement, by allowing fewer or splitting across hosts.
   - Nodes: timan107, timan1 GPU2 and local. timan108 is dead: never probe it, and drop it from probes.
   - MetaWorld workers never recycle today (N0). Add a periodic recycle, e.g. every 32 episodes or 30 minutes, at an episode boundary.

Read the frontline code before designing. Cite file:line for the integration points.

Deliverables:
- the design doc, in Chinese;
- new module(s) with CPU tests driven by simulated memory and throughput curves: memory ramp to the floor, recovery, a server-bound plateau, a later improvement;
- an integration patch for the shared files, prepared as a patch file and NOT applied (see rules);
- a live trial plan with exact commands for one small lane, comparing fixed vs dynamic counts on memory and throughput. Do not launch it.
</task>

<rules>
- CPU only locally. tether exec only for read-only checks. Never push, pull or expose. No launches; no killing processes; do not touch running runs or chains.
- Another agent (sol, thread 01a140c4) is editing `exp/r17_integration/common/serving.py`, `exp/offline_search/rounds/r17/longseg_1017/*`, `filter_failed_knob/formal_a_1016/*`, `filter_failed/CONTINUATIONS.json` and rebuild/launch workflows. Do not edit those, or any file with uncommitted changes from that work. Put new code in new files, and integration changes in `rounds/r17/autoscale_1017/INTEGRATION.patch`. I apply it after sol's turn.
- Do not saturate /data: no parallel bulk reads; sample big logs; check `/proc/pressure/io` (wait while full avg10 > 20).
- No rm, no git commits. Never touch credentials.
</rules>

<compact_output_contract>
Write everything under `exp/offline_search/rounds/r17/autoscale_1017/`: DESIGN.md (Chinese), modules, tests, INTEGRATION.patch, TRIAL.md. Final message: what was built, test results, the files the patch touches, and the trial command.
</compact_output_contract>
