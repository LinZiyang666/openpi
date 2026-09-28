# R5 coding brief (common to the codex coding agents Q1–Q3)

Read first: `rounds/r05/SELECTION.md` (Chinese; this brief and your prompt carry everything in English),
`rounds/r05/FINDINGS.md`, the ideation report your family comes from (`rounds/r05/ideation_{A,B,C,D}/REPORT.md`),
`rounds/r04/CODING_BRIEF.md` (method <-> plugin contract, log fields, cost ledger — still binding), and the R4 hand-backs
you build on: `rounds/r04/k10_policy_tail/HANDBACK.md` (policy tail serving path), `k7_guard/HANDBACK.md`,
`k6_concurrency/HANDBACK.md` (per-connection locking contract), `k1_blind/HANDBACK.md`, `k5_rand/HANDBACK.md`.

## File ownership (parallel agents: never edit a file you do not own)
| owner | files |
|---|---|
| Q1 | `exp/offline_search/rounds/r05/q1_commit/**` (new; method side only) |
| Q2 | `closed_loop/plugin.py`, `closed_loop/blind.py`, `closed_loop/selftest.py`, `closed_loop/verify_logs.py`, `closed_loop/replay_client.py`, `rounds/r05/q2_groot/**` |
| Q3 | (dispatched later, after Q2 hands the plugin back) |
Nobody edits `src/`, `harness/`, `profile/`, `closed_loop/ops/*`, `rounds/r01..r04/` (subclass or wrap instead).

## Live-system rule
The coordinator runs closed-loop chains on this code while you work: every 10–25 minutes a new server imports
`closed_loop/plugin.py` and the method modules. Develop in a copy under your own directory, keep existing behaviour
byte-identical when your new flags/kwargs are absent, prove it (re-run the existing selftest matrices: K2, K1, K4, K5,
K6, K7, K10 recipes — see `rounds/r04/k10_policy_tail/run_installed.sh` and `results/`), and install each shared file in
ONE atomic step (temp file in the same directory + mv). Codex agents never run closed loops, servers, LIBERO workers or
chains; the coordinator does all experiments.

## Cost and accounting (owner basis, unchanged)
Per five-control decision: vision .152, MISS +.848 (full inference, K10 π0.5 / K8 GR00T — MISS step reduction is not
part of the system), blind / policy-tail decisions 0. The K4 ledger reads the explicit `vision` / `hit` flags.

## Rules
- Vision anchor mandatory in every episode; blind stretches bounded (owner ruling). No external models.
- Every conclusion at the 50- and the 500-episode library; library bytes vs deployed pkl; label borrowed information.
- CPU: prefix every python command with `taskset -c <your range>`, BLAS threads 1, processes ≤ threads in range. Never
  CPUs 38-43, 82-87. GPU: none unless your prompt says otherwise.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no ports 23150-23169.
  Do not read `tests/review_tests/`. `/home/weiland/trace_runs` is read-only in the sandbox: write fits to `/tmp/<you>_fits/`.

## Hand-back (final message and `rounds/r05/<your dir>/HANDBACK.md`)
Files changed/added (install times, sha256), how each feature is switched on (exact method strings / kwargs / flags),
tests and their numbers, arm specs (emit_arms format with `<RUN>` placeholders) and exact prefit commands, what the
coordinator must do next (smoke recipe), caveats.
