<task>
You are R4 coding agent K6: remove the server-wide serialization of R4 serving without changing any decision.
Problem (measured by the coordinator): in exp/offline_search/closed_loop/plugin.py, `_ConnPolicy.infer` enables the
R4 stage adapter whenever the runtime is in R4 mode (`--os-blind`, `--os-log-r4`, or the K5 randomization flags), and
then holds `s.rt.decision_lock` — ONE runtime-wide RLock shared by every websocket connection — around the whole
decision, including `self._osp_inner.infer(obs)` (full stage 1/2/3 policy inference on the GPU). All connections of a
server are therefore served one at a time. On a live l10 blind arm (2 servers × 24 LIBERO workers) the per-decision
server latency median is 2.9 s (queueing; stage 1 alone ~97 ms, K10 stage 2+3 ~560 ms), versus 0.42 s for legacy
(non-R4) arms where connections run concurrently; an arm takes ~40 min instead of ~13. Outcomes are unaffected
(LIBERO is synchronous), only wall-clock is.
Goal: R4-mode decisions from different connections run concurrently exactly as legacy decisions do, while every
per-connection decision (method query, blind step, judge verdict, served action, history, K5 randomization, log rows
apart from timing fields) stays identical to serialized execution.
1. Find every piece of state a decision touches that is shared across connections in R4 mode (runtime counters such as
   `decision_count` / the server-wide periodic clock used by `--os-blind` periodic judges, log writers, the fitted
   method object(s), the stage adapter, orchestrator/interceptor state, K5 overlay state, ProbeBlind counters, etc.) and
   decide for each whether it is per-connection, needs a short critical section, or is read-only. The decision index for
   the server-wide periodic clock must stay unique and monotonic (reserve it atomically, e.g. under a short lock), but
   must not be held across GPU inference. Legacy (non-R4) behaviour must stay byte-identical.
2. Implement the narrowest correct locking (per-connection locks where state is per connection; short critical
   sections for shared counters/writers). Do not hold any runtime-wide lock across `_osp_inner.infer`, stage 1, or
   stage 2/3.
3. Verify (a) every existing selftest and verifier still passes (K2 matrix, K1 blind matrix, K4 plugin matrix, K5
   randomized replays: see rounds/r04/k5_rand/HANDBACK.md and run_all_checks.sh for the commands; adapt output paths to
   your own scratch directory); (b) a concurrency test: N ≥ 8 interleaved connections driven concurrently (threads)
   with a fake policy that sleeps (e.g. 50–100 ms) inside inference: per-connection decision rows / served actions /
   verdicts / history equal the serialized run (ignore timing fields and the arbitrary interleaving of the global
   periodic index, but check uniqueness and monotonicity), and wall time drops roughly by the concurrency factor;
   include blind (`--os-blind`, phase_particles B=2, guard-only span judge), `--os-log-r4` guard-only MixedJudge,
   periodic:k with `--os-blind`, and K5 randomization configurations; (c) optionally, if GPU memory allows, one live
   STAGE1_ONLY server smoke with the repository's replay client driving ≥ 4 concurrent connections.
4. Install atomically (temp file in the same directory + mv) and re-run all checks against the installed files.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 of an action-cache exploration for VLA policies on
LIBERO. Read first: exp/offline_search/rounds/r04/CODING_BRIEF.md (binding: live-system rule, method<->plugin contract,
log fields), rounds/r04/k2_serving/HANDBACK.md (who wrote the R4 serving path and why the lock exists), 
rounds/r04/k5_rand/HANDBACK.md (the latest plugin owner and its verification recipes), closed_loop/README.md.
The coordinator runs closed-loop chains on this code RIGHT NOW: every 15–40 minutes a new server process imports
closed_loop/plugin.py. You own closed_loop/plugin.py, closed_loop/selftest.py, closed_loop/verify_logs.py,
closed_loop/replay_client.py, closed_loop/blind.py and the new directory rounds/r04/k6_concurrency/. Develop in
rounds/r04/k6_concurrency/dev/, prove byte-identical legacy behaviour and decision-identical R4 behaviour, and install
each shared file in ONE atomic step. Never leave a shared file half-edited.
Store: /dev/shm/offline_search_store (hot) or /home/weiland/trace_runs/offline_search_store (cold copy).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: optional and only for step 3(c): at most one STAGE1_ONLY=1 server (~2.5 GB), only if `nvidia-smi` shows
  ≥ 8 GB free; ports only in 23180–23189 (scan `ss -ltnH` first); stop it and release the GPU right after. Never touch
  ports 23150–23169 or any process / tmux session you did not start. Stop at once on any OOM.
- Edit only the files you own (listed above). Do not modify src/, harness/, profile/, rounds/r01..r03/, other agents'
  rounds/r04/k1..k5 directories, ops/* or the cost table.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve the task fully: analysis of shared state, implementation, verification (existing + new concurrency tests),
atomic installation, hand-back. Check edge cases: episode reset and task change while other connections are mid-
episode, a MISS on one connection concurrent with blind decisions on others, duplicate decision ids, exceptions in one
connection not poisoning others, log file integrity under concurrent writes (no interleaved partial lines).
</completeness_contract>

<verification_loop>
Before finishing, re-run every existing plugin selftest and your new tests against the INSTALLED files and report the
final numbers (including the measured concurrency speed-up). If a check fails, fix and re-run.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k6_concurrency/HANDBACK.md (the shared-state analysis table; files changed with install times and
sha256; tests and numbers; residual risks; what the coordinator must do next), then print a one-paragraph summary as
your final message.
</structured_output_contract>
