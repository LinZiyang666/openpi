<task>
Continuation of P3 (you own `exp/offline_search/rounds/r06/p3_profiling/`). Status: v2 client smoke on timan107 has run
4/8 arms (π0.5 l10-50 A / P10 / factorial / window, SR .5 / 1.0 / .5 / .5), `read_v2 --require-stage-counts
--require-snapshots` passed on the first two; the GR00T four are running now. The pilot run root
`/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot` is emitted (216 arms) and prefit is running.

Measured problem: client telemetry is ~35 MB per episode (almost all `controls.jsonl`, 18–38 MB per episode attempt;
`step_N.npz` snapshots are small). It is written to `/scratch/zixuans8/openpi_trace/os_cl/p3_client/<run>/<arm>/
p3_telemetry` on timan107, then `collect_client.py` tars it into `/tmp/p3_telemetry_<run>_<arm>.tar` (same filesystem)
and pulls it; nothing is ever deleted. timan107 `/` (holds /scratch and /tmp) has only 40 GB free; the pilot is
4,320 episodes (~150 GB), the full campaign 36,000.

**Owner's instruction: do not store the data on timan107 — stream it back to weilandserver (this machine) in real
time.** Build this as an opt-in transport, default behaviour unchanged:

1. **Receiver** (runs on weilandserver, launched by the coordinator in tmux): listens on a TCP port given by the
   coordinator (it will be in 23100–23199; timan107 reaches weilandserver ports as `ziyanglin.com:<port>`, the same
   way the workers reach the policy servers — see `chain_p3.sh` `servers=`). It writes into
   `<RUN>/runs/<arm>/client_telemetry/` with **exactly the same relative layout and bytes** that file mode +
   `collect_client.py` produce today, so `read_v2` is unchanged. One receiver per run root, many concurrent workers
   and arms. Validate every path component (no traversal, arm/run names match the existing safe-identifier rule),
   write via temp + atomic rename per finished file, and write a per-file / per-episode-attempt completion record with
   sha256 and byte count.
2. **Client sink** (py3.8, stdlib only; the timan107 client is Python 3.8.20, numpy 1.22.4): when e.g.
   `P3_STREAM=<host>:<port>` is set, telemetry lines and snapshot files go to the receiver instead of local files.
   Must never block or slow the control loop noticeably (background sender thread, bounded queue); resumable after a
   dropped connection or receiver restart (sequence / offset acknowledgements, idempotent re-sends, no duplicated or
   lost lines); on prolonged failure spill to the current local directory so no data is lost, and mark the spill so the
   coordinator can pull just the spill. Optional per-frame zlib on the wire is fine; storage stays in today's format.
3. **Coordinator side**: `chain_p3.sh` (your owned chain) gets an opt-in `P3_STREAM_PORT` mode: check the receiver is
   healthy before launching an arm, forward `P3_STREAM` to the client, and after ARM_DONE verify completeness against
   the journal (every accepted episode attempt has its complete record) before writing the DONE marker; pull only spills
   if any. **In both modes**, after a verified collect, delete that arm's remote telemetry directory and its
   `/tmp/p3_telemetry_<run>_<arm>.tar` on timan107 (`rm` / `rm -r` without `-f`, only those exact paths, only after
   local sha verification) — today nothing is cleaned. Keep all stock admission checks.
4. **Tests** (CPU, loopback on 127.0.0.1 with an ephemeral port — never ports 23100–23199): stream mode vs file mode
   produce byte-identical `client_telemetry` trees on recorded or synthetic episodes; fault injection (receiver
   killed and restarted mid-episode, connection drops, duplicate frames, slow receiver, spill-and-recover); py3.8
   compatibility check of the client files (`check_python38.py`); throughput numbers (MB/s, per-line latency, sender
   queue high-water) for ~20 concurrent synthetic workers at the pilot's data rate (~35 MB per episode, ~3 episodes
   per minute per line, two lines).
5. Rebuild the client bundle in a new directory, and give me the exact coordinator commands: receiver launch
   (tmux, CPU set, port placeholder), redeploy, a **stream smoke** (2 arms × 4 episodes in a fresh run root) with the
   read_v2 check, and how to switch the pilot's remaining arms from file mode to stream mode between arms.

Update HANDBACK.md (new section: files, sha256, switches, tests with numbers, commands, caveats). Final message: short
summary.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 6-9,50-53`, at most 8 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Never start servers, receivers on real ports, LIBERO workers or chains; never touch tmux sessions, ports
  23100-23199, timan107 or other hosts (no tether) — the coordinator runs every experiment, and a smoke plus two
  closed-loop chains are running right now. Files currently deployed on timan107 and used by the running smoke must
  keep working: do not change the behaviour of existing client files without the new switch. No git. No `rm -rf`.
  Never `pkill -f`. Do not edit `src/`. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it in HANDBACK.md, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
