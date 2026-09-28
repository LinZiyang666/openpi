# P3 opt-in telemetry streaming

This delivery was prepared and tested locally. No remote deployment, remote
cleanup, policy server, LIBERO worker, chain, tmux session or port 23100–23199
was touched. Tests used 127.0.0.1 with OS-assigned ephemeral ports only. Python
commands were pinned to **6–9,50–53**, with OMP/OpenBLAS/MKL=1 and CUDA hidden.
The twenty synthetic workers are threads, not twenty processes.

## Transport and data contract

`P3_STREAM` selects a background sender. Without it, the existing client file
path and lifecycle are preserved. With it, `controls.jsonl` lines are queued as
their existing UTF-8 bytes; snapshots use the same NumPy NPZ serialization into
memory. Healthy clients create **no local telemetry files or tar** on timan107.
Telemetry goes to the dedicated receiver, never the policy endpoint.

The receiver writes:

```
<RUN>/runs/<arm>/client_telemetry/<sha256(uid)[:24]>_a<attempt>/controls.jsonl
<RUN>/runs/<arm>/client_telemetry/<sha256(uid)[:24]>_a<attempt>/step_NNNNNN.npz
<RUN>/runs/<arm>/p3_stream/<key>/<filename>.json   # per-file SHA256/byte receipt
<RUN>/runs/<arm>/p3_stream/<key>/complete.json    # identity, files, status/outcome
<RUN>/runs/<arm>/p3_stream_verified.json         # journal + local-file verification
<RUN>/runs/<arm>/p3_stream_collect.json          # spill collection/replay report
<RUN>/runs/<arm>/p3_cleanup.json                # successful cleanup proof/command
```

Final telemetry files have exactly the existing names and byte format.
`read_v2.py` is unchanged. A hidden `.FILE.p3part` contains an unfinished file;
finished files publish with same-directory `os.replace`. Receipts also publish
atomically. Attempt receipts distinguish complete/error/prefix; an accepted
journal attempt must have a complete receipt with the same outcome. Its files
are re-hashed locally before cleanup. Numeric simulator/control semantics,
server Profile behavior and ledger accounting were not changed.
An attempt receipt also has total `bytes` and a `sha256` of its canonical sorted
file manifest (names, sizes and content hashes), with the scope named explicitly.

The protocol uses length-framed JSON headers and raw byte payloads, capped at
256 KiB each. Data is addressed by file and byte offset. ACK follows file fsync;
after receiver restart the partial file length is the durable offset. Retries
compare overlapping bytes, append only missing bytes, and reject conflicts or
offset gaps. Finish validates whole-file SHA/size before rename. Repeated
completion requests use the existing receipt instead of reparsing a large log.
Per-attempt process locks serialize receiver writes and coordinator spill
replay. One receiver serves the run's known arms and concurrent workers; it
validates run, arm, UID, hashed attempt directory, filename and path components,
rejecting symlinks/traversal. A persistent per-run token prevents accidental or
unauthorized cross-run writes. The wire is not TLS-encrypted.

The sender holds a bounded queue (8 MiB by default). Network waits, reconnects,
ACK waits and receiver disk operations run in its background thread. Queue
overflow, persistent transport failure, or an episode-close deadline switches
that attempt to a **sticky spill**: unacknowledged frames plus subsequent frames
are saved in the current local telemetry directory as `stream.p3spill` and a
SHA/byte-count `stream.p3spill.json`. Only a suffix may be present: keep the
receiver's already-acknowledged prefix. An unfinished spill is visibly `.part`.
Spill replay is idempotent and works with a restarted or live receiver. It does
not rerun an episode or policy call.

At an episode boundary, close waits for completion ACK or a durable spill before
the stock runner can report an accepted outcome. This wait is outside the
control loop. Normal enqueue never waits for network IO. During an outage,
spill writes necessarily use local disk synchronously: bounded memory, a
nonblocking producer, and losslessness under arbitrary outages cannot all be
guaranteed without that fallback. Abrupt client termination or power loss can lose its
unacknowledged RAM queue; the missing receipt then prevents verification/DONE.
Do not treat this as a client-crash WAL or a hard real-time guarantee.

## Files, package and switches

New: `stream_protocol.py`, `stream_sink.py`, `stream_receiver.py`,
`stream_collect.py`, `prepare_stream_smoke.py`, `test_stream.py`,
`test_stream_reader.py`, `test_stream_coordinator.py`.

Updated: `telemetry.py` (guarded sink), `run_gtp_v2.py` (worker environment),
`client_plan.py`, owned `chain_p3.sh`, `collect_client.py`, bundle/preflight/audit
helpers and tests. Shared chain/run_arm/plugin and `src/` were not edited.
The complete owned-chain diff is `chain_p3.diff`. Serving engines and the reader
are unchanged; no shared-file regression/install condition was triggered.

Use **`client_bundle_stream_release/`**, not the intermediate
`client_bundle_stream/`. Its manifest maps local→remote paths; its 100 KiB tar
installs fourteen files: eight Python modules, five package markers and the
owned launcher. The receiver/collector stay on weilandserver. Existing package
markers and the island's stock launcher remain protected by the prior installer.
Run-specific bundles additionally contain selected arm YAMLs. The minimal tar:

```
fc225d41d27d1939d415ba00f2e0e82423610c20bfb9139b74ef6fc1fd7fbd04
```

Per-file source/bundle hashes and local publication time are recorded in
`results/install_manifest_stream.json`. No remote installation time is claimed.

| Switch | Meaning / default |
|---|---|
| `P3_STREAM` | Client `host:port`; unset preserves file mode. |
| `P3_STREAM_TOKEN` | Run receiver token; automatically forwarded by the owned chain. |
| `P3_STREAM_RUN`, `P3_STREAM_ARM` | Explicit identities forwarded by chain; otherwise derived from the standard telemetry directory. |
| `P3_STREAM_QUEUE_BYTES` | Queue plus in-flight frame budget; default 8,388,608 bytes per worker. |
| `P3_STREAM_FAIL_S` | Persistent-error threshold, default 5 s; evaluated after a failed request. |
| `P3_STREAM_TIMEOUT_S` | Connect/data request timeout, default 1 s. Finish/attempt receipt waits allow at least 10 s for hashing/validation. |
| `P3_STREAM_CLOSE_S` | Episode-boundary drain deadline, default 10 s; then durable spill and sender shutdown. |
| `P3_STREAM_PORT` | Owned-chain opt-in: receiver health check at 127.0.0.1, client endpoint `ziyanglin.com:<port>`. Policy ports must be distinct. |
| `<RUN>/state/P3_STREAM_PORT` | Optional port file reread at each arm boundary by this new chain version. Enables an existing new-version file-mode chain's remaining arms. |
| `P3_STREAM_TOKEN_FILE` | Coordinator token path; default `<RUN>/state/p3_stream.token`. Token is redacted from the saved client plan. |
| `stream_collect --collect` | Pull only marked spills, verify archive and replay, verify every accepted journal attempt, then cleanup. |
| `collect_client --cleanup` | File mode: existing full collect, verify archive/extracted bytes and accepted attempts, then cleanup. |

In **both modes**, the owned chain's DONE marker follows ordinary collect,
telemetry verification and successful cleanup. A failure prevents DONE.
Cleanup re-hashes the current remote file tree against the verified local
archive and checks the remote tar SHA immediately before removal; changed or
new files and symlinks cause refusal. Commands use only:

```
rm -r -- /scratch/zixuans8/openpi_trace/os_cl/p3_client/<run>/<arm>/p3_telemetry
rm -- /tmp/p3_telemetry_<run>_<arm>.tar
```

Both are existence-guarded, have no glob and no `-f`. No parent arm/run directory
is removed. Cleanup requires a quiescent, completed arm; this is why it is after
the driver and collection checks, never a periodic background purge. Error
prefixes on incomplete arms are retained. Unfinished spill files are preserved
locally in the spill archive; they are not falsely marked complete.

## Local tests and limits

All results below are synthetic CPU or packaging checks, not new LIBERO SR.

| Check | Observed result |
|---|---|
| Old file / new file / stream parity | Identical two-file trees, 41,230 bytes, including a real synthetic NPZ and all control lines with fixed test clocks; issued controls unchanged. |
| Receiver killed/restarted mid-episode | A separately spawned test receiver was SIGKILLed and restarted on its prior ephemeral port. 1,010,079 control bytes + 600,423 snapshot bytes recovered exactly; two connections; receipt verified. |
| Lost ACK / duplicate frames / validation | Forced connection drop after durable write; two duplicate frames accepted idempotently; seven invalid/conflicting frame cases rejected. |
| Slow receiver / bounded spill | 829,197 control bytes recovered exactly; 64 KiB test queue high-water 60,075 bytes. Replaying the spill twice is safe. |
| Prolonged outage / close timeout | Both spill paths passed, including sender shutdown and replay with a live receiver. |
| Concurrent arms | Two arms, two accepted attempts, 100 lines routed/verified separately. |
| Cleanup | Real cleanup command exercised only on substituted owned temporary paths. Mutation blocked deletion; verified bytes allowed exact-path deletion. File-mode tether calls were mocked; bad archive SHA prevented cleanup. No real tether call. |
| Unchanged strict reader | Four streamed episodes, 96 decisions, 48 anchors, 468 controls, 48 snapshots (52 files, 1,211,608 bytes): `--require-stage-counts --require-snapshots` PASS. |
| 20-worker burst | 39.45 MB in 1.80 s, **21.88 MB/s**, no spills; queue high-water 1.99 MB. Enqueue p50/p95/p99/max: 2.78/8.07/10.97/19.07 ms. |
| 20-worker pilot-rate test | 698.91 MB (about 35 MB/worker), offered at 3.5 MB/s. Including finish/verification: **3.42 MB/s**, 204.13 s, no spills, queue high-water **33,541 bytes**. 21,300 enqueue samples: p50/p95/p99/max **0.591/2.118/3.434/8.434 ms**. |
| Real Python 3.8.20 | 67 files compile; nine owned Python files, six isolated dependency-free imports; actual stdlib sender streamed two files on loopback without spill. NumPy/simulator runtime uses the coordinator's smoke, not this isolated cached interpreter. |
| Bundle / dispatch / v2 units | 18-file smoke bundle, eight isolated imports, five dispatch paths, stock/marker preservation and corruption rejection PASS. Six staged + six legacy coexistence cases, 12,000 assignment units, seven synthetic controls/21 substeps/one snapshot PASS. |
| Smoke preparation | Two arms emitted and prefitted at `/tmp/p3_stream_smoke_fixture`; no model/server/episode launched. |

Benchmark workers, sender threads and receiver threads share one Python process
and its GIL on the assigned eight CPUs; this is a contention test. Enqueue
timings exclude pre-existing JSON construction and NPZ serialization. The test
demonstrates local capacity, not WAN latency or real simulator timing parity.
The coordinator must check those in the stream smoke.

An initial paced test failed because simultaneous terminal validation exceeded
the short ACK deadline and duplicates reparsed the log. The fix uses fast
completion receipts, longer background finalization waits and process locks.
All twenty tiny completion-frame spills from that failed test (6,699 bytes)
were recovered, certifying all 698,914,730 underlying telemetry bytes. It is
recorded as a failed development test plus successful recovery, not a passing
throughput result (`results/stream_development_recovery.json`).

Evidence: `/tmp/p3_stream_tests_certified`, `/tmp/p3_stream_rate_certified`,
`/tmp/p3_stream_reader_certified`, `/tmp/p3_stream_coord_certified`,
`/tmp/p3_stream_deploy_tests`, `/tmp/p3_stream_v2_units`; copied numeric reports
are under `results/stream_*.json`. The isolated Python 3.8 check is
`results/python38_stream.json`, with source inventory `CLIENT_IMPORTS_STREAM.md`.

## Exact coordinator commands: fresh two-arm stream smoke

These commands are **prepared, not executed here**. Replace the two port
placeholders with coordinator-selected free ports; neither may be a live
smoke/chain port, and they must differ. Use a fresh run root. P10 in each model
exercises executed policy tails: π0.5 l10/50 and GR00T l10/500, tasks0–1 ×
inits0–1 = eight validation episodes. All anchors snapshot; K4 smoke resampling
is retained. Production pilot arm parameters are unchanged.

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_stream_smoke
export P3_STREAM_PORT='<RECEIVER_PORT>'
POLICY_PORT='<POLICY_PORT>'
P3PY=(taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)

"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.prepare_stream_smoke --run-root "$RUN"
mkdir -p "$RUN/calibration" "$RUN/state"
cp "$D/calibration_v2/pi05_l10_50.json" "$D/calibration_v2/groot_l10_500.json" "$RUN/calibration/"
"${P3PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$RUN" --spec "$RUN/arms_in.json"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.prefit --run-root "$RUN" --spec "$RUN/arms_in.json"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.build_client_bundle --run-root "$RUN" --out "$RUN/client_bundle_stream"
bash "$D/deploy_client.sh" "$RUN/client_bundle_stream" 2>&1 | tee "$RUN/client_deploy_stream.log"
test "${PIPESTATUS[0]}" -eq 0 || exit 1

tmux new-session -d -s "p3rx_$(basename "$RUN")" "cd /home/weiland/projects/openpi && taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.stream_receiver --run-root '$RUN' --port '$P3_STREAM_PORT' --ready-file '$RUN/state/p3_stream.ready.json' >> '$RUN/receiver.log' 2>&1"
for i in $(seq 1 30); do
  "${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" >/dev/null 2>&1 && break
  sleep 1
done
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" || exit 1
mapfile -t ARMS < "$RUN/arm_names.txt"
PORTS="$POLICY_PORT" WPS=2 SERVER_CPUS=6-9,50-53 STAGE1_ONLY=0 P3_PHASE=smoke \
  bash "$D/chain_p3.sh" "$RUN" "${ARMS[@]}"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 \
  --run-root "$RUN" --arms "${ARMS[@]}" --client-root "$RUN/runs" \
  --require-stage-counts --require-snapshots --out "$RUN/tables_stream_smoke"
```

Acceptance: eight accepted attempt receipts, strict reader PASS, no unexplained
spill, and `p3_cleanup.json` for each arm. Inspect enqueue/close behavior and
receiver IO/CPU on the real hosts. Existing predicate/actuator missingness and
exact simulator restoration remain separate, previously documented limitations.
Keep the receiver run directory and token file across receiver restarts.

## Switch the existing pilot's remaining arms

The pilot was inspected read-only: **216 arms**, schedule present, first pilot
manifest exists, seed603/snapshot-every1/p1 resolve. It has no `arm_names.txt`;
derive names from `arms.json` below. Do not emit or prefit it again. Allow its
existing prefit to finish before starting the remaining arms.

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
export P3_STREAM_PORT='<PILOT_RECEIVER_PORT>'
P3PY=(taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.build_client_bundle --run-root "$RUN" --out "$RUN/client_bundle_stream"
bash "$D/deploy_client.sh" "$RUN/client_bundle_stream" 2>&1 | tee "$RUN/client_deploy_stream.log"
test "${PIPESTATUS[0]}" -eq 0 || exit 1
mkdir -p "$RUN/state"
tmux new-session -d -s "p3rx_$(basename "$RUN")" "cd /home/weiland/projects/openpi && taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.stream_receiver --run-root '$RUN' --port '$P3_STREAM_PORT' --ready-file '$RUN/state/p3_stream.ready.json' >> '$RUN/receiver.log' 2>&1"
for i in $(seq 1 30); do
  "${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" >/dev/null 2>&1 && break
  sleep 1
done
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" || exit 1
printf '%s\n' "$P3_STREAM_PORT" > "$RUN/state/P3_STREAM_PORT.tmp"
mv "$RUN/state/P3_STREAM_PORT.tmp" "$RUN/state/P3_STREAM_PORT"

# If no new-version pilot chain is already active, launch/resume it:
mapfile -t ARMS < <("${P3PY[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
: "${PORTS:?set coordinator-selected policy ports distinct from receiver}"
: "${WPS:?set the existing pilot worker setting}"
SERVER_CPUS=6-9,50-53 STAGE1_ONLY=0 P3_PHASE=pilot \
  bash "$D/chain_p3.sh" "$RUN" "${ARMS[@]}"
```

For a chain **already running this new version**, create the port file only
after receiver health succeeds; its next arm uses streaming. An in-flight
file-mode arm finishes with file-mode verified collect/cleanup. Do not launch
a second chain on that run root. An already-running **older** shell cannot
gain the new boundary check by replacing its script on disk or exporting an
environment variable elsewhere. The coordinator must finish its current arm
and hand over the launch loop to this version between arms; then the resume
command above skips existing manifest-hash DONE markers. This delivery does not
interrupt any running chain. It atomically replaced only the owned chain file.

To clean telemetry retained by a completed old file-mode arm, use the new
verified collector explicitly, only when that arm has no active producers:

```bash
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.collect_client \
  --run-root "$RUN" --arm '<COMPLETED_ARM>' --cleanup
```

No blanket historical cleanup is performed. Never use shared `sync_remote.sh`
for this delivery; the new bundles carry configs and preserve shared launchers.

## Capacity and outstanding validation

At the owner's measured 35 MB/episode, the pilot's client telemetry alone is
approximately **151.2 GB** and the full campaign's **1.26 TB** (decimal), plus
snapshots/receipts; server-side input archives are additional. Normal streaming
stores one copy of each telemetry file on weilandserver and no full telemetry
tar on either host. Twenty default sender queues cap queued/in-flight payloads
at about 160 MiB total plus frame/serialization working memory. An outage can
consume local spill space for the current arm; the next arm cannot start with
an unhealthy receiver. Spills and their pulled archives are retained locally
for audit, so failure storage includes those extra copies.

There is no extra policy forward or GPU-hour charge from this transport. Actual
WAN throughput, client enqueue/serialization overhead, receiver disk capacity
and end-to-end smoke behavior remain **unverified** until coordinator execution.
The owner-reported four completed π0.5 file-client smoke arms and running GR00T
arms were not rerun, modified or reinterpreted as streaming evidence here.
