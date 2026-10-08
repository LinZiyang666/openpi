<task>
R16 conductor follow-up (coordinator, 2026-10-08 03:0x). The conductor dev run works: π0.5 groups finished with worker utilization 0.994 (10,260 and 10,080 new episodes). The GR00T k1 group is running now, but throughput is ~38 episodes/min: each of the 4 GR00T resident servers sits at exactly 100% of one CPU core (Python single-thread bound), GPU utilization is 29%, and the timan107 workers mostly wait. More GR00T server processes would scale throughput. Make two small changes, register them honestly, and keep everything else frozen.
</task>

<changes>
1. Configurable server count: `conductor_runner.py` hard-codes π0.5 = 3 and GR00T = 4 (two per suite). Add an explicit option (e.g. `--groot-servers-per-suite N`, default 2; optionally `--pi05-servers N`, default 3) that only changes how many identical resident processes serve each suite; ports from `--ports` or `local_server.choose_ports`. Resident method sets, per-connection hot switch, dispatch, gates, identities, seeds and logs must stay exactly as now. A group already started must be resumable with a different server count (completed episodes kept; in-flight attempts fenced and re-dispatched as today).
2. Stop-loop bug: when a group finishes, `local_server.stop` raises LOCAL_PORT_RELEASE_TIMEOUT for the first server (π0.5 shutdown takes longer than the 60 s budget) and the remaining servers of that group are never stopped; the next group then fails with LOCAL_PORT_BUSY. Make the runner stop every recorded server even when one stop fails, wait longer for release (or poll until released within a bounded time), and only then report. Do this in R16 code, not by editing the shared `closed_loop/ops/h100/local_server.py` unless unavoidable.
3. Register both as an operational amendment in `CONDUCTOR_LOCK.json` (or a new lock entry) so that `analyze freeze` and later verification accept the updated runner while still rejecting any change to measurement-relevant code (methods, gates, protocol, analysis, pairing). State clearly that the server count does not enter any decision or estimate.
4. CPU tests: server_specs for 3 and 4 processes per suite (suite assignment, unique ports, both checkpoints), resume with a changed count, and the stop-all-servers path with one simulated timeout. Re-run the existing R16 conductor test suite.
5. Short `HANDBACK_GROOT_SERVERS.md`: what changed, the exact command line I should use to resume the running GR00T k1 group with 3 servers per suite (6 processes; ~6.2–6.8 GB each on the 4090), and how to fall back to 2 per suite quickly if the other session needs its GPU memory back.
</changes>

<constraints>
CPU only: taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python. The GR00T k1 dev group is running right now (tmux r16_cond_dev, 4 local resident servers, timan107 workers): do not touch its processes, journals, group directories or logs; do not start servers, GPU work, remote operations, git, pkill or rm -rf. Do not change frozen design/protocol/amendment files or any measurement-relevant code. Test A is measure-only.
</constraints>

<default_follow_through_policy>
Do not stop for routine questions; keep the change minimal and record choices in the handback. Stop and report only if the server count cannot be changed without affecting measurement or resumability.
</default_follow_through_policy>
