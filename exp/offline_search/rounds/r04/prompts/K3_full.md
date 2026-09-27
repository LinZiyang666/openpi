<task>
You are R4 coding agent K3: the cost engine (ideation B proposals B1, B3 and the single-camera part of B4).
Implement:
1. Reduced-step MISS: π0.5 via the existing yaml `miss: {num_steps: K}` path (read src/openpi/cache/config.py around
   `_miss_errors` / MissConfig and the interceptor's `_miss_steps`), GR00T via a K parameter exposed in
   closed_loop/ops/start_server.sh (today it hardcodes the denoising steps; keep the old default when unset) matched to
   the bundle's K. Log `miss_steps` in the plugin startup row via a server-side hook you own (coordinate through the
   documented field; do not edit plugin.py — if a field must be added there, describe it in HANDBACK for K2/the
   coordinator). Specify for K4 the exact yaml patch an arm needs (K4 implements generic yaml patching in emit_arms).
2. Exact redundancy elimination, as an opt-in stage override installed at server start (new
   closed_loop/stage_overrides.py, enabled from serve_pi05.py by a flag): (a) cache the embedding of π0.5's always-masked
   dummy third camera once (the exact preprocessed constant, per model/dtype/device) and skip its tower on every
   decision; (b) optional: after a MISS verdict, pack the fully masked prefix positions before stage 2 (ideation B §B3).
   Parity is mandatory: same inputs + same noise → identical keys (bitwise or documented tolerance) and identical actions
   with/without each override, on stored tok-subsample observations, including CUDA-graph paths if the server uses them.
3. Single-camera ("wrist-only") π0.5 key: on a HIT decision run only the wrist camera's tower (plus the cached dummy),
   build the key from it; on a MISS complete the other camera's tower exactly and continue the normal stage 1 → 2 → 3,
   so MISS actions are unchanged. Provide a camera-aware AWM adapter method (in rounds/r04/k3_cost/) refitted on
   wrist-only keys of the same 50- and 500-episode libraries (library keys rebuilt from the stored pooled keys, which is
   exact for camera deletion), with V7/guard tables recalibrated in that space, and the needed key-builder/plugin
   interplay documented (you do not own plugin.py).
4. Measure on the 4090 (batch 1, warmed up, the serving dtype/backend; CUDA graphs if the server uses them) the stage
   costs: full stage 1, dummy-cached stage 1, wrist-only stage 1, stage 2 unpacked/packed, stage 3 per denoising step,
   for π0.5, and GR00T's current split; write closed_loop/ops/cost_table.json (ms and shares, with provenance) for K4's
   ledger. This is a cost-accounting microbenchmark, not a system-throughput study.
Deliver arm specs `k3_cost/arms_r4.json` for the second- and fourth-batch arms of SELECTION.md (MISS K2 on the existing
g500 / g / perk5 / spatial guard-only configurations, K2-only pure inference, dummy-cached and wrist-only variants),
with prefit commands.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 (R4) of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. You are one of four coding agents working in parallel. Read first:
exp/offline_search/rounds/r04/CODING_BRIEF.md (binding: file ownership, live-system rule, the method<->plugin contract,
log fields, cost ledger, rules), then exp/offline_search/rounds/r04/SELECTION.md, the ideation reports in
exp/offline_search/rounds/r04/ideation_{A,B,C}/ (REPORT.md; C also REPORT_2.md), exp/offline_search/closed_loop/README.md,
exp/offline_search/harness/README.md, and the code you build on (named in your task).
Store: /dev/shm/offline_search_store (hot) or /home/weiland/trace_runs/offline_search_store (cold copy, same content).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root (GR00T servers use their own venv, see
closed_loop/ops/start_server.sh).
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: allowed for parity tests and the stage-cost microbenchmark: at most one GPU process at a time; stage-1-only work up to 4 GB when at least 8 GB are free; full-model work (stage 2/3 parity, packing) up to 10 GB only when at least 14 GB are free — otherwise poll every few minutes (time is not a constraint) and do CPU-side work meanwhile; stop GPU work at once on any out-of-memory error; release the GPU after each measurement.
- Edit only the files you own (CODING_BRIEF.md "File ownership"). Develop in a copy, keep existing behaviour
  byte-identical when your new options are absent, install each shared file atomically (temp file + mv), and re-run the
  existing selftests after installing.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains. Never touch ports, processes or tmux sessions you did not start. Do not read tests/review_tests/.
  Do not modify src/, harness/, profile/ or rounds/r01..r03/.
</hard_constraints>

<completeness_contract>
Resolve your task fully before stopping: implementation, tests, and the hand-back file. Do not stop at the first
working version; check edge cases (episode reset, step 0, after a MISS, both models where required, both library sizes
50 and 500 where required).
</completeness_contract>

<verification_loop>
Before finishing, verify every deliverable against the task and the brief with real runs (selftests, parity checks,
smoke runs as allowed). If a check fails, fix and re-run; report the final numbers, not the first attempt.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k3_cost/HANDBACK.md (files changed/added with install times; exact switches/arm-spec fields; tests
and their numbers; what the coordinator must do next; caveats), then print a one-paragraph summary as your final
message.
</structured_output_contract>
