<task>
Continuation of P3 (streaming transport). The pilot has been running in stream mode since 17:45 (lanes L3/L4, receiver
`p3rx_r06_p3_pilot` on 23171, frozen chain `r06_p3_pilot/ops/chain_p3.stream.frozen.sh`); ~107 arms verified cleanly,
0 spills until now. First failure, reproducible on every retry:

- arm `r6p3v2_groot_l10_50_window_r2` (run root `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot`): the driver
  completed 20/20 accepted episodes (journal complete, SR .60); the chain then ran
  `stream_collect --run-root <RUN> --arm <arm> --collect`, which pulled the remote spill tar (20,459,520 bytes) into
  `runs/<arm>/client_spills.tar` / `client_spills/` and failed at `stream_collect.py` line ~222 with
  `ValueError: conflicting duplicate bytes`. Four attempts, same error (see `runs/chain.log`, `lane_L4.log`).
- Around 23:12 CDT two other chains (different run roots) saw simultaneous ARM_INCOMPLETE (497–498/500), i.e. a
  probable short timan107 network/tether hiccup; this arm ran 23:37–23:49.
- Everything is preserved: receiver-side `runs/<arm>/client_telemetry` (550 MB) + `p3_stream/` receipts/partials, the
  pulled spill archive, and the remote spill / telemetry on timan107 (not cleaned — cleanup only happens after
  verification). The arm's claim is held; lane L4 was stopped and a replacement lane continues with other arms.

Do, read-only on the live data until you have a verified repair:
1. Find the root cause: which file(s)/attempt(s) conflict, at which offsets, what the two byte ranges contain (e.g. the
   same attempt restarted and re-sent different bytes, a second attempt with the same key, a sender retry after a
   partial ACK, an episode-boundary close-deadline spill racing with late ACKs, clock/sequence reuse, …). Use the
   receipts, the spill manifest `stream.p3spill.json`, the journal (attempts / accepted rows), and the server decision
   logs for that uid/attempt.
2. Decide which bytes are authoritative for every accepted attempt, and write a repair tool (new file in your
   directory) that reconstructs the exact telemetry tree for this arm without guessing, verifies it against the journal
   the way `stream_collect --collect` does, and only then lets the chain's normal verified cleanup run. If the data of an
   accepted attempt cannot be reconstructed exactly, say so; do not paper over it.
3. Fix the underlying bug so it cannot recur (client sink, receiver, or collector), keeping byte-identical behaviour for
   the healthy path and the existing tests passing; add a regression test that reproduces this failure mode. Any change
   to a file the running pilot lanes execute (`stream_collect.py`, `stream_receiver.py`, `stream_sink.py`,
   `telemetry.py`, chain) must be installed atomically (temp file in the same directory + os.replace) and must not break
   the running receiver (it keeps running; do not restart it — tell me if a restart is required and why). Client-side
   files on timan107 change only via a redeploy that I run; give me the bundle and the exact commands if needed.
4. Tell me the exact commands to repair and finalize this arm (so its DONE marker is written by the normal verification
   path), and whether any already-verified arm could be affected by the same bug (scan all `runs/*/p3_stream_verified.json`
   / spill markers).
Update HANDBACK.md (short section). Final message: short summary.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 6-9,50-53`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python`.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Never start or stop servers, receivers, chains, lanes or workers; never touch tmux sessions or ports 23100-23199;
  no tether writes to timan107 (read-only `tether exec` inspection of the remote spill is allowed). Do not modify or
  delete anything under the run root except new files in a fresh subdirectory `runs/<arm>/repair_<timestamp>/`.
  No git. No `rm -rf`. Never `pkill -f`. Do not edit `src/`. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
