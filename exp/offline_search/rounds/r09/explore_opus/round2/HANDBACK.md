# Opus round 2 — hand-back (what ran, where things are, exact runs requested)

## What ran (all local CPU)

- CPUs 2-9,46-53 only (taskset), OMP/OPENBLAS/MKL threads = 1. No GPU, no policy inference, no simulator, no remote
  command, no sync, no server, no worker, no tmux session, no chain. No owned process is left running.
- No git state change, no package install, no edit outside `exp/offline_search/rounds/r09/explore_opus/`,
  `derived/r09_opus/` and the new run root `r09_opus_escalation`.
- Rule 1: every reader drops init ≥ 30 at parse time (tested); the two holdout roots are refused (tested). The R8
  summary file `rounds/r08/abl/RESULTS.md` (500-pair aggregates) was read once as background before the round-2
  rules were applied; none of its numbers is used — every number in my documents is recomputed on inits 0–29.
  Constants were chosen on inits 0–19; the screen manifest is inits 20–29 only.
- Rule 2: nothing task-indexed; the trigger uses only the cache's own retrieval and the decision index.
- Rule 3 / coordinator instruction: `DIRECTION.md` was written before substantive work after reading both other
  researchers' round-1 and in-progress round-2 outputs; both round-2 directories were re-read before `PROPOSALS.md`.
- `REPORT.md` could not be written: the agent harness refuses files named like reports for subagents. Its full text
  (plain-Chinese owner section first) is in my final hand-back message; please save it as
  `exp/offline_search/rounds/r09/explore_opus/round2/REPORT.md`.

## Files

| path (under `exp/offline_search/rounds/r09/explore_opus/round2/`) | content |
|---|---|
| `DIRECTION.md` | Covered directions of astra/fable and my non-overlapping levers |
| `DATA_ANALYSIS.md` | All evidence, tables, commands |
| `PROPOSALS.md` | Ranked proposals, frozen specs, exact arms, adoption bars |
| `methods.py` | Serving classes `EscalateCalls`, `EscalateOnlyNP`, `EscalateOnlyNPGroot` |
| `tools/` | catalog, episodes, r8ledger, serverledger, forensics, replicates, anatomy, triggers, preregister, prepare_arms, deploy_sources, screen_analysis, reproduce.sh, tests/ (14 tests) |
| `artifacts/r9o_*.pkl` | 8 frozen prefits (escalation arms) |
| `arms_in.json` | Emitter input (16 arms) |
| `out/replicates/`, `out/anatomy/`, `out/triggers/` | Analysis outputs (JSON) |
| `out/preregistration.json` | Frozen screen predictions |
| `out/selftest_frozen_arms.log` | CPU plugin selftest: 8/8 escalation arms PASS |

Derived data: `/home/weiland/trace_runs/offline_search_store/derived/r09_opus/` (49 MB; catalog, per-decision ledgers,
forensic extracts; discovery inits only). Run root: `/home/weiland/trace_runs/os_closed_loop/r09_opus_escalation/`
(`arms.json`, `config/`, `manifests/eval_inits20_29.json`, `h100_sync/plan.json` = local dry plan, 81 files,
16.3 GiB logical; nothing synced).

Reproduce everything offline: `bash exp/offline_search/rounds/r09/explore_opus/round2/tools/reproduce.sh` (≈ 3 min).

## Runs requested (coordinator executes; I launched nothing)

16 arms × 100 pairs (tasks 0–9 × inits 20–29), three batches, one chain at a time. Fleet, ports and sync port are the
coordinator's choice (I was assigned none). Estimated wall time: 25–40 min per batch on 32–48 workers (escalation
arms are slower than pure cache because escalated episodes call the policy every 10 controls).

| batch | arms |
|---|---|
| 1 π0.5 long-50 | `r9o_pi05_l10_50_cache r9o_pi05_l10_50_esc r9o_pi05_l10_50_esc_w24 r9o_pi05_l10_50_onlynp r9o_pi05_l10_50_onlynp_esc` (+ optional `r9o_pi05_l10_P10`) |
| 2 GR00T long-50 | `r9o_groot_l10_50_cache r9o_groot_l10_50_esc r9o_groot_l10_50_esc_w24 r9o_groot_l10_50_onlynp r9o_groot_l10_50_onlynp_esc` (+ optional `r9o_groot_l10_P10`) |
| 3 500-demo long | `r9o_pi05_l10_500_cache r9o_pi05_l10_500_esc r9o_groot_l10_500_cache r9o_groot_l10_500_esc` |

If only one batch can run, run batch 1 minus the optional P10; if two, add batch 3 (largest expected cost saving).

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R=/home/weiland/trace_runs/os_closed_loop/r09_opus_escalation
OPS=$PWD/exp/offline_search/closed_loop/ops/h100
ARMS=(r9o_pi05_l10_50_cache r9o_pi05_l10_50_esc r9o_pi05_l10_50_esc_w24 r9o_pi05_l10_50_onlynp r9o_pi05_l10_50_onlynp_esc)

# 1. install the serving module in the isolated h100 tree (new files only; refuses differing existing files)
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round2.tools.deploy_sources            # prints the plan
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round2.tools.deploy_sources --execute  # coordinator
#    files: exp/offline_search/rounds/r09/explore_opus/{__init__.py, round2/__init__.py, round2/methods.py}
#    methods.py sha256 2ede2b5f16a59700bd62b4927aaf763d54da566f5a38abcb2980ceb44d03ef0b

# 2. plan + sync (coordinator-reserved SYNC_PORT in 23100..23197; WORKER_HOST = the fleet you assign)
WORKER_HOST=<fleet> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R" "${ARMS[@]}"
WORKER_HOST=<fleet> SYNC_PORT=<port> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R" "${ARMS[@]}"

# 3. chain (full-model arms; ports/WPS per your assignment)
WORKER_HOST=<fleet> PORTS=<p1,p2,p3,p4> WPS=<n> MAX_ATTEMPTS=2 POLL_SECONDS=30 \
  OSCL_MANIFEST="$R/manifests/eval_inits20_29.json" \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R" "${ARMS[@]}"

# 4. readout (paired vs same-topology controls, measured rescue rate r, IR, escalation timing)
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round2.tools.screen_analysis --run-root "$R"
#    -> exp/offline_search/rounds/r09/explore_opus/round2/out/screen_report.json
```

All escalation arms are `full_model: true` with `--os-blind --os-policy-tail [--os-policy-tail-blocks 1]
--os-judge guard_only` (copied from the R8 pure-policy / only-no-progress arm specs); the `_cache` and `_onlynp`
controls are byte-for-byte the R8 arm specs with the new manifest.

## Pre-registered predictions and adoption bars

See `PROPOSALS.md` (P1–P3) and `out/preregistration.json`. Main bar per cell: measured rescue rate r ≥ .50 and paired
SR gain over the same-topology pure cache ≥ +6 pp (50-demo) / +3 pp (500-demo) at IR ≤ .20 / ≤ .13. Keep and report
unfavourable results; do not re-tune lag 12 / deadline 80 / window 24 on inits 20–29 before the holdout.

## Known limitations

- The one unknown is the takeover's rescue rate r (prior .45–.65); everything before the trigger is measured.
- GR00T statistics use the standard *server* decision logs (uid, step, top-k rows; validated against the R8 debug
  rows); the escalation arms additionally log `r9o_lag`, `r9o_escalated`, `r9o_esc_step` in every server decision
  record's extras.
- The test-only lag-1 artifacts used to exercise the MISS path live in `/tmp/opus_test_*.pkl` (scratch, not used
  by any arm).
