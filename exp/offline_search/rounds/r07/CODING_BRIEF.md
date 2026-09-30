# R7 coding brief (common to coders C1–C4, gpt-6.1-sol)

Read first: `rounds/r07/SELECTION.md` (what to build and why), `rounds/r07/IDEATION_BRIEF.md` (facts, terminology,
generality rules), the exploration report(s) your family builds on (`rounds/r07/ideation/E*/PROPOSAL.md`), and the code
you extend:
- A (the deployed pure cache): `exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM` with kwargs
  `{"lib": "current"|"bpool_cs"(π0.5 500)|"bpool_all"(GR00T 500), "kref": 5 (50) | 8 (500), "serving": "anchor_tail",
  "budget": 1, "gates": "budget_only"}` and plugin args `--os-no-shadow-native --os-blind --os-fit-artifact <RUN>/fits/<arm>.pkl`
  (+ `--os-root /home/weiland/trace_runs/offline_search_store`). Check exact arm rows in
  `/home/weiland/trace_runs/os_closed_loop/{r05_ptail,r05_x,r04_blind,r06_paper}/arms.json`.
- Blind decisions and look requests: `closed_loop/blind.py`, `closed_loop/README.md`, `rounds/r04/k1_blind/`.
- Policy calls, policy tails, calibrated stall, budget: `rounds/r05/q1_commit/`, `rounds/r06/p1_groot_commit/`,
  `rounds/r06/ideation_Q1/method_c/{methods.py,stall_bridge.py,budget.py,fit_calibration.py,HANDBACK.md}`,
  `rounds/r06/ideation_Q3/stall/{stall.py,INTERFACE.md,HANDBACK_FAST.md}`. R6 C arm rows:
  `/home/weiland/trace_runs/os_closed_loop/r06_c_validation/arms_in.json`.
- One-camera cost path (π0.5): `closed_loop/stage_overrides.py`, `rounds/r04/k3_cost/`, `rounds/r05/q6_wrist_blind/`.
- Arm emission / prefit: `closed_loop/ops/emit_arms.py`, README "Launching full arms".

## File ownership (four agents work in parallel; never edit a file you do not own)
| owner | files |
|---|---|
| C1 | `rounds/r07/stages/**` (land `stages.py` + `STAGES_API.md` FIRST, within your first work block), `rounds/r07/c1_follow/**` |
| C2 | `rounds/r07/c2_wrist/**`, `closed_loop/plugin.py`, `closed_loop/stage_overrides.py` |
| C3 | `rounds/r07/c3_calls/**` |
| C4 | `rounds/r07/c4_profile/**` |
Nobody edits `src/`, `harness/`, `profile/`, `closed_loop/ops/*`, earlier rounds, or another coder's files. Import, subclass
or wrap instead. If you need `stages.py` before C1 lands it, code against SELECTION §3's API and integrate once it
exists; do not write your own copy of the stage table.

## Hard requirements
- **Identity when off:** with your new lever disabled, executed actions / verdicts / vision flags must equal the base
  method bit for bit on recorded streams (prove it; the R6 P3 and B-val strict tables and input archives under
  `/home/weiland/trace_runs/os_closed_loop/{r06_p3_pilot,r06_c_cal}/` are available for CPU replay).
- **Generality (owner):** no task/suite names, no LIBERO-unit thresholds, no literal camera count / action or state
  width / 5-control offset in method logic where the manifest provides it; every number from a documented library rule.
  Both models where the lever exists.
- **Shared-file installs** (C2 only): develop in a copy, keep behaviour byte-identical when new flags are absent, rerun
  the existing plugin selftests, install each shared file in ONE atomic step (temp file in same dir + `mv`).
- **Telemetry:** every new decision must log what it decided and why (stage mass/unanimity, granted extension, valve
  statistic and radius, LOOK reason, camera mode, call weight/p/coin) in the method's `Result.extras` / blind extras,
  within the plugin's scalar log budget; list the keys in your hand-back.
- **Cost accounting** stays the owner basis: vision .152/.148 per two-camera look, call .848/.852, blind 0; wrist look
  .055198 and missing-camera completion .049890 (π0.5, labelled assumption). The cost ledger must count what actually ran.

## Rules
- CPU: prefix every Python command with `taskset -c <your range> env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src` and use `.venv/bin/python` from
  `/home/weiland/projects/openpi`; at most as many processes as CPUs in your range. Never CPUs 38–43, 82–87.
- GPU: none, except C2 as stated in its prompt.
- Codex agents never run closed loops, servers, LIBERO workers, chains, tmux, ports or remote hosts; the coordinator runs
  all experiments. No git. No `rm -rf`. Never `pkill -f`. Do not read `tests/review_tests/`.
- `/home/weiland/trace_runs` is read-only: write fits and scratch under `/tmp/r7_<your id>/`; the coordinator copies them.

## Hand-back (`rounds/r07/<your dir>/HANDBACK.md` and a ≤15-line final message)
Files added/changed (sha256; install times for shared files), exact method strings / kwargs / plugin flags per variant,
arm specs in `emit_arms` format with `<RUN>` placeholders for every planned variant × cell (SELECTION §4–5), exact prefit
commands, tests and their numbers, telemetry keys, what the coordinator must run next (smoke recipe), caveats.
