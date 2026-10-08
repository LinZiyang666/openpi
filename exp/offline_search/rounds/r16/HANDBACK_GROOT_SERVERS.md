# R16 GR00T server-count follow-up — 2026-10-08

`conductor_runner.py` now accepts `--groot-servers-per-suite N`. Ordinary dev/test defaults stay at two per checkpoint; isolated smoke defaults stay at one. π0.5 stays at three processes. Each GR00T process holds the same original suite-specific resident methods and live K. Count changes only endpoint topology and throughput; it is not an input to a decision, gate, SR/IR estimate, pairing or CI.

Resume retains surviving per-suite processes with their exact receipts and log tags. Increasing 2→3 adds one process per suite. Decreasing 3→2 stops only those two extra processes. Retirement receipts are saved before shutdown, so interruption cannot orphan them. Completed episodes and persisted attempt fences use the unchanged conductor resume path; unfinished episodes are re-dispatched.

Shutdown attempts every recorded server even if one stop fails. After all initial stop attempts, owned-process/port release is polled for up to 240 additional seconds across the timed-out servers. A recovered timeout completes its stop receipt. Unresolved failures are reported together as `R16_STOP_INCOMPLETE`; the next group does not start. The shared `local_server.py` is unchanged. The worst-case bound is its existing 60 seconds per initial stop, plus 240 seconds of additional polling.

Registration: `CONDUCTOR_LOCK.json` has an operational entry, mirrored in `GROOT_SERVERS_OPERATIONAL.json`, retaining the prior lock/preimage under `before/`. Only the runner hash changes and `test_groot_servers.py` is added. Methods, resident serving, dispatch/gates, design/protocol/amendment, analysis/pairing/pricing and shared code retain their original hashes. Future `analyze freeze` binds this updated lock. No frozen table or input was rewritten. All 195 R16/conductor regression and operational tests pass (`groot_servers_tests.xml`), including measurement-source tamper rejection.

Coordinator commands only; none were run here. In tmux `r16_cond_dev`, interrupt the current runner to reach `INTERRUPTED`, and stop its enclosing queue loop as well so it does not advance to k2. Wait for its worker lease to release (up to 90 seconds after a hard controller loss). The fleet locks refuse a concurrent runner. Existing model processes remain available for resume.

Run this at the tmux shell to resume GR00T k1 with **three servers per suite / six total**:

```bash
cd /home/weiland/projects/openpi
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m exp.offline_search.rounds.r16.conductor_runner run \
  --group /home/weiland/trace_runs/os_closed_loop/r16_conductor_dev_groot_k1_v1/group.json \
  --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 \
  --workers 64 --port 23179 --groot-servers-per-suite 3
```

For a quick memory return, interrupt this runner, wait for its agent release, then use the **same command with `--groot-servers-per-suite 2`**. It retires one process per checkpoint before resuming with the four surviving originals. No worker sync or new worker archive is needed: the agent and resident implementation are unchanged. For k2, use its group path and the same explicit count. `--ports`, if supplied, is the complete target list: first N LIBERO-10 ports, then N Spatial ports; omission chooses free ports and preserves existing live assignments.

The supplied measurement implies six processes need **37.2–40.8 GB**, versus **24.8–27.2 GB** for four. Six exceeds the original 28 GB planning allocation and needs the coordinator's temporary larger allocation; returning to two per suite releases about 12.4–13.6 GB. Startup keeps the existing free-memory checks. CPU-limited ideal scaling from four to six would be ~38→57 episodes/min; real throughput is unmeasured here.

No live group, process, journal or log was accessed or modified. No GPU/server/remote operation, refit or test-A outcome read occurred. `GROOT_SERVERS_OPERATIONAL.sha256` is the local follow-up checksum list; the previous deployment's `local.sha256` predates this explicit runner/lock update.
