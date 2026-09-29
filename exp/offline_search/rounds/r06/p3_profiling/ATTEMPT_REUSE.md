# Pilot attempt-key collision: diagnosis, repair and prevention

Prepared 2026-09-29, locally on CPUs 6–9,50–53, CUDA hidden, thread libraries=1.
No remote command, network connection, service start/stop, tmux operation, or
live dataset modification was performed. Under the pilot root, all new files
are inside the failed arm's fresh `repair_*` directories. The repair is **ready,
not installed on the live arm**.

## Exact cause and authoritative bytes

Arm `r6p3v2_groot_l10_50_window_r2`, uid ending `:eval:4:1`, key
`322d18e7918f5eef8d5c7971_a1` was reused across two driver processes. The stock
scheduler resets `_dispatch_gen` on construction; its driver source explicitly
documents this and assigns a new `run_id`. The transport used only uid/attempt.

The first driver (`a957aab75885`) left a **748,614-byte, 15-line error trace**:
ten settling controls, zero successful policy decisions, `profile_error`, and
`rollout_error`. Its receiver receipt says `status=error`, not complete. The
collected `client/driver.log` records a policy WebSocket keepalive ping timeout.
The journal contains no accepted row for that incarnation. After 19/20 outcomes,
the chain resumed at 23:48:24 CDT with driver `f6ae5e413713`; its accepted journal
row is again `attempt=1`, success=true, duration 42.629 s.

The new sender's 1,048-byte `attempt_start` line is identical to the old one, so
it received one ACK. Its reset frame at offset **1,048** then conflicts: first
differing byte **23,130**, in `rng.python_global`. The old RNG array begins
`[2147483648,1958101769,3299301836,...]`; the new one begins
`[2147483648,2471535094,895644600,...]`. Different serialized lengths shift later
line boundaries. There are **12 conflicting data frames, all in controls.jsonl**.
No snapshot conflicts exist: the old incarnation never completed an inference.
This is not a partially acknowledged frame or an episode-close race.

The finished spill is 20,452,050 bytes, SHA256
`889e57c2cde165238722e19375410db77badb18570c07411514e2ca07cd7e925`;
the pulled tar is 20,459,520 bytes, SHA256
`1b4e15fed29bbbeb0bedd91cfb067c2a999d25fedec5943a59cc9eb25482c8db`.
Its descriptor records one ACK, 36 connections, and repeated conflicting-byte
rejections. The spill contains 307 data, 24 finish and one attempt-manifest frame.
The reported earlier 23:12 outage was not independently established as the
cause of this later policy connection failure.

**Authority:** retain the receiver bytes for the other 19 accepted attempts.
For this uid, take the receiver's identical 1,048-byte acknowledged prefix and
the spill's subsequent bytes. Every reconstructed file must match the sender's
final complete-file SHA256 and byte count; gaps or checksum mismatches abort.
All 24 files match: **20,376,706 bytes**, including 23 NPZ snapshots and
15,908,972 control bytes. The latter SHA256 is
`69260b10ad1353e94b927c1742d2a1dd120e434d9e2eb247f52267eea53b2a5c`.

This selection is independently tied to the accepted driver: all ten terminal
timing fields exactly match its `client/per_step.jsonl` row with run_id
`f6ae5e413713`; its 46 unique winners join server connection 24 and all decision
markers. It has 285 telemetry lines and 236 controls. Outcome alone is never
used to choose between conflicting incarnations. Details and every conflicting
offset are in `repair.json` and `results/attempt_reuse_repair.json`.

## Repair tool and verification

`repair_stream.py --prepare` builds a separate complete arm view and verifies it
with the unchanged `stream_collect.verify_arm`. It does not change source data.
The successful prepared directory is:

```
/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/runs/r6p3v2_groot_l10_50_window_r2/repair_20260929T051500Z/
```

It holds `repair.json`, `candidates/`, `originals/`, a reconstructed run tree,
and strict-reader `tables/`. All **20 accepted episodes / 753 files /
580,503,832 bytes** verify. `read_v2 --require-stage-counts --require-snapshots`
passes on the real server logs plus repaired client copy: 1,460 decisions,
733 anchors, 7,478 controls; one unaccepted server attempt is retained.

`--install` is an explicit coordinator action. It checks unchanged source
journal/per-step hashes and attempt inventories, preserves original evidence,
takes the receiver's existing per-attempt flock, publishes exact verified files
using same-directory temp + `os.replace`, and publishes the complete receipt
last. It neither cleans the remote host nor writes DONE. The receiver caches no
file contents or receipts and needs **no restart**. Running the original spill
through the normal collector is then idempotent.

An initial prepare in `repair_20260929_attempt_reuse/` failed the decision-marker
proof because generic `dec.source` differs from the authoritative `p3_decision`
marker on blind steps. The proof now joins the latter by tag/connection/step.
That incomplete diagnostic directory is not an install candidate; no source
data was altered by either prepare.

## Prevention and client bundle

New `dispatch_fence.py` reserves and fsyncs a monotonically increasing dispatch
generation **before returning a task to a worker**. The owned `run_gtp_v2.py`
installs this wrapper only for `P3_STREAM` drivers. It updates the scheduler's
generation fence together with the dispatched task, preserving stale-result
rejection. Retry budgets, ordering, uid, task inputs and action logic are
unchanged. A fresh arm starts at attempt 1, as before; a resumed driver continues
the counter instead of reusing 1. File mode remains unchanged.

The tiny counter files live at
`.../p3_client/<run>/<arm>/.p3_dispatch/`, outside telemetry cleanup. Preserve
them across resumes. The persistent bootstrap marker chooses a reserved high
range starting at **100001** when upgrading an arm with an existing legacy
journal but no counters; the pilot's stock retry limit is three (generations
1–4 per old driver). Fresh arms keep their original numbers. The protocol limit
of 1,000,000 fails closed; there is no wrap or counter reset. Counter write
failure cannot return a task. This metadata is retained locally on timan107;
telemetry still streams as before.

No change was needed to `stream_sink.py`, `stream_receiver.py`,
`stream_collect.py`, `telemetry.py`, `read_v2.py` or either chain. Their hashes
match the preceding delivery. All replacements of existing owned client files
were installed with same-directory temp + `os.replace`. The remote installer
also replaces owned files atomically. Existing imported driver processes stay
on their old code; redeploy applies to subsequent driver launches. Until then,
another legacy resume can hit the same collision and must fail verification.

New config-free bundle: `client_bundle_attempt_fence_20260929_release/`, **15 files**
(nine modules, five package markers, launcher). SHA256:

```
29ddcfb4984742db5661d05f741dbd23bbdbeb23d12b2965b2f88e68319e61d7
```

Use this release directory, not the intermediate bundle without `_release`.
The manifest installs the counter dependency before the driver entrypoint.
It preserves existing configs and stock launchers. There is no reason to emit,
prefit, redeploy a policy plugin, change a receiver port or restart the receiver.
Per-file hashes/publication time are in `results/install_manifest_attempt_reuse.json`.

## Exact coordinator commands (not executed here)

Keep the affected arm quiescent and its existing claim held. Install the already
verified repair, then let its **existing frozen chain** perform ordinary collect,
spill replay, SHA/journal verification, cleanup and the manifest-hash DONE marker.
Choose a free policy port; the receiver remains on 23171. The chain retains its
usual GPU/port admission checks. This finalization invocation starts its ordinary
policy server/driver, but the complete journal should dispatch no new episodes.

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
ARM=r6p3v2_groot_l10_50_window_r2
REPAIR="$RUN/runs/$ARM/repair_20260929T051500Z"
P3PY=(taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)

# At an arm boundary: deploy the new client for subsequent driver launches.
bash "$D/deploy_client.sh" "$D/client_bundle_attempt_fence_20260929_release" \
  > "$REPAIR/coordinator_deploy.log" 2>&1

"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.repair_stream --install "$REPAIR"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect \
  --run-root "$RUN" --arm "$ARM" --verify

FINALIZE_PORT='<FREE_POLICY_PORT>'
P3_STREAM_PORT=23171 PORTS="$FINALIZE_PORT" WPS=20 SERVER_CPUS=6-9,50-53 \
  STAGE1_ONLY=0 P3_PHASE=pilot \
  bash "$RUN/ops/chain_p3.stream.frozen.sh" "$RUN" "$ARM"
```

Do not manually touch DONE, remove the claim to requeue the arm, or clear its
receiver prefix. If source-change guards reject installation, prepare a new
copy and review its result before installing it:

```bash
REPAIR="$RUN/runs/$ARM/repair_$(date -u +%Y%m%dT%H%M%SZ)"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.repair_stream \
  --run-root "$RUN" --arm "$ARM" --prepare "$REPAIR"
```

## Tests and verified-arm scan

| Executed CPU check | Result |
|---|---|
| Actual failed arm, isolated prepare | 20 accepted attempts, 753 files, 580,503,832 bytes; every hash passes. |
| Real-arm disposable clone: install → normal `collect(..., no_remote=True)` → repeat install | PASS; same 20 attempts, one spill replay, no remote calls. |
| Strict reader on repaired real data | 20 episodes, 1,460 decisions, 733 anchors, 7,478 controls; stage counts and snapshots PASS. |
| New regression: identical prefix ACK, new RNG/reset, reused key | Reproduces conflicting duplicate bytes; exact reconstruction and idempotent installation/replay PASS; missing prefix and wrong driver rejected. |
| Real in-process scheduler | First dispatch equals stock; retries/resume generations 1,2,3; stale attempt rejected; 40 concurrent reservations unique; reopened counter returns 41; storage errors fail closed; stream-only driver wiring PASS. |
| Existing transport assertions through direct Store delivery, no sockets | 41,230-byte old-file/new-file/stream parity; duplicates, seven invalid frames, 829,197-byte bounded spill/replay, outage and close deadline, two arms PASS. Strict synthetic reader: 4 episodes / 48 snapshots PASS. |
| Packaging / mocked coordinator | 19-file smoke bundle, nine isolated module imports, five dispatch paths, marker/stock preservation and corrupt-archive guards PASS; remote calls=0. |
| Real Python 3.8.20 | 68-file compile, ten owned files, seven dependency-free imports, runtime persistent counter reopen PASS. No third-party simulator launch. |

Current constraints prohibit starting receivers, so network and process-kill
tests were **not** rerun; direct Store delivery reuses the existing assertions.
There is no new WAN or live-client performance claim.

`audit_stream_incarnations.py --hash-files` read all **99 verified streaming
arms / 1,980 accepted attempts / 44,770 files / 38,147,280,848 bytes** in its
inventory and checked journal hashes, outcomes, receipt aggregates, sizes and
every file SHA: **zero failures** (report completed 2026-09-29 05:16:17 UTC).
Eleven earlier file-mode arms have collect records but no stream verification
markers; they are outside that count. Locally preserved spill markers show:

- `groot_l10_50_dose50_r0`: a timeout spill, already replayed and cleaned by the
  coordinator. Its 20 outcomes / 687 files verify; no conflict or damaged bytes.
- This `groot_l10_50_window_r2` arm: the single unverified collision described above.

Thus no already-verified streaming arm in the scan has evidence of damage from
this bug. This is an inventory snapshot, not a claim about future running arms
or uncollected remote markers. The receiver's refusal and the cleanup guard
worked: incompatible incarnations were not silently merged or deleted.

Numeric evidence: `results/attempt_reuse_{repair,install_clone,regression,transport,coordinator,deploy,python38,scan}.json`.
Local test roots: `/tmp/p3_attempt_reuse_test_final_20260929`,
`/tmp/p3_real_repair_install_20260929`, `/tmp/p3_attempt_reuse_transport_20260929`,
`/tmp/p3_attempt_reuse_deploy_final_20260929`, `/tmp/p3_attempt_reuse_coordinator_20260929`.
The coordinator deploy recipe was syntax-checked and run with `--dry-run` only.
