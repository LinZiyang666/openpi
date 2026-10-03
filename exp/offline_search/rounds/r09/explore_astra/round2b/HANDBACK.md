# Round 2b hand-back

**Ready:** four shared observation-only late-progress detectors, four fixed response variants per cell, 16 fitted candidates, 22 emitted standard-mode arms, and a 100-pair manifest containing only inits 20–29. Fit and calibration use 0–19. No new task-indexed switch or online simulator predicate is used.

**Not run:** no round-2b chain, remote sync/source install, simulator, GPU job, H100 server, worker, tmux session, or network operation. The coordinator's active `r09_astra_round2_eval` run was only read through its eval-only manifest and filtered journals. No process or file in that run was changed. No owned experiment process requires cleanup.

The strongest candidate is bounded policy takeover after an observation-based alert, first on π0.5 long-50. Expected fixed-path IR≈.117 there; SR is unmeasured. A three-call variant, a single-call variant, and training-budget-matched random three-call placement are frozen controls. Do not attach a success-preservation claim to the offline detector metrics.

## Material new result from the running round-2 experiment

Snapshot 2026-10-02 05:16:56 UTC, all five π0.5 long-50 arms complete on **20–29 only**:

| Cache | Motion | Motion + gripper | Gripper only | P10 |
|---:|---:|---:|---:|---:|
| .72 | .63 | .63 | .74 | .92 |

Motion versus cache: +7/−16 pairs; joint: +11/−20; gripper: +3/−1. These are new disjoint-fit results, not the old overlapping-fit corrector numbers. They directly warn against promoting another correction on action MSE alone. The new round-2b methods therefore use unmodified cache actions.

Other ongoing arms may finish later. Refresh only the allowlisted reader below; absent/incomplete arms are status, not failures. The report snapshot is intentionally dated rather than represented as final fleet status.

## Files and outputs

Owned package:

```text
/home/weiland/projects/openpi/exp/offline_search/rounds/r09/explore_astra/round2b/
```

| File/location | Purpose |
|---|---|
| `REPORT.md` | Plain-Chinese owner conclusion, technical interpretation |
| `PROPOSALS.md` | Ranked, frozen responses and confirmation protocol |
| `DATA_ANALYSIS.md` | Populations, causal labels, metrics, negative results, limitations |
| `tools/data.py`, `labels.py` | Identity firewall and strictly pre-action predicate states |
| `tools/screen.py`, `ablation.py` | Init-disjoint detector fit/eval and clock controls |
| `tools/causal.py` | Randomized first-alert progress and terminal effects |
| `tools/reanchor.py` | Cheap response falsification using teacher shadows only as labels |
| `tools/responses.py`, `bounds.py` | Exact random-budget calibration, fixed-path cost, SR sensitivity/bounds |
| `tools/live_results.py` | Read-only coordinator eval-root journal snapshot |
| `methods.py`, `inference.py` | NumPy serving monitor, recovery state machines, policy-tail adapter |
| `tools/prepare_confirmation.py` | Real-library verification, prefits and arm emission |
| `tools/test_tools.py` | 13 unit tests, no existing test directory accessed |
| `artifacts/*_monitor.npz/json` | Four frozen detector heads and training choices |
| `artifacts/*_phase.npy`, `*_response.json` | Library phase and train-calibrated response parameters |
| `artifacts/r9r2b_astra_*.pkl` | 16 self-contained fitted candidate methods |
| `FROZEN_CANDIDATES.json` | Artifact/head/phase hashes and exact kwargs |
| `confirmation_specs.json` | 22-arm emitter input |
| `results/` | All numeric records, paired running-result snapshot, verification logs |
| `PROVENANCE.json` | Input, dependency and output integrity inventory |

New derived output directory:

```text
/home/weiland/trace_runs/offline_search_store/derived/r09_astra/round2b/
  <cell>_<variant>_labels.npz
  <cell>_<variant>_eval_scores.npz
```

New configuration-only run root:

```text
/home/weiland/trace_runs/os_closed_loop/r09_astra_round2b/
  arms.json
  config/
  manifests/eval100.json
  h100_sync/plan.json
```

The local first six-arm dependency plan passed: **44 files, 11.078 GiB logical assets**. Existing remote assets can be reused after hashes are verified; this is not a claimed new transfer size. Source installation is separate from asset sync.

## Reproduce locally

From `/home/weiland/projects/openpi`:

```bash
bash exp/offline_search/rounds/r09/explore_astra/round2b/tools/reproduce.sh
```

This re-labels permitted episodes, re-fits on 0–19, evaluates 20–29, freezes prefits and emits configuration. It does not launch or sync. It overwrites only this owned package's generated artifacts and new owned run root. Do not regenerate a frozen package while the coordinator is using it; artifact container hashes may change on regeneration. Reports are authored interpretation, not generated prose.

Verify the existing freeze without refitting:

```bash
R9B_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9B_MOD=exp.offline_search.rounds.r09.explore_astra.round2b.tools
"${R9B_PY[@]}" -m unittest "$R9B_MOD.test_tools" -v
"${R9B_PY[@]}" -m "$R9B_MOD.audit"

# Optional read-only update of the coordinator's round2 eval-only journals.
"${R9B_PY[@]}" -m "$R9B_MOD.live_results"
```

All computation was restricted to CPUs 10–21,54–65 with OMP/OPENBLAS/MKL threads=1. No new package dependency was installed. No git state changes or edits outside the assigned directory/new output roots were performed.

## Deferred coordinator commands — do not run while timan108 is busy

The current user instruction forbids launching chains in this turn. These commands are ready for the coordinator **after the active round-2 chain and its workers/servers exit**. Do not stop or modify that chain to make room.

```bash
cd /home/weiland/projects/openpi
R9B_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9B_MOD=exp.offline_search.rounds.r09.explore_astra.round2b.tools
R9B_RUN=/home/weiland/trace_runs/os_closed_loop/r09_astra_round2b
R9B_OPS=$PWD/exp/offline_search/closed_loop/ops/h100
R9B_ARMS=(r9r2b_astra_pi05_l10_50_cache r9r2b_astra_pi05_l10_50_burst1 r9r2b_astra_pi05_l10_50_burst3 r9r2b_astra_pi05_l10_50_latch r9r2b_astra_pi05_l10_50_random3 r9r2b_astra_pi05_l10_P10)

# First prints the exact sources; --execute takes the fleet lock and refuses
# differing existing source files. Round2 inference.py is an explicit dependency.
"${R9B_PY[@]}" -m "$R9B_MOD.deploy_sources"
WORKER_HOST=timan108 "${R9B_PY[@]}" -m "$R9B_MOD.deploy_sources" --execute
WORKER_HOST=timan108 "${R9B_PY[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R9B_RUN" "${R9B_ARMS[@]}"
WORKER_HOST=timan108 SYNC_PORT=23196 \
  bash "$R9B_OPS/sync_assets.sh" --concurrent "$R9B_RUN" "${R9B_ARMS[@]}"

# Runner admission checks must pass, including GPU free memory and asset hashes.
WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=8 MAX_ATTEMPTS=3 \
  OSCL_MANIFEST="$R9B_RUN/manifests/eval100.json" \
  bash "$R9B_OPS/chain_h100.sh" "$R9B_RUN" "${R9B_ARMS[@]}"
```

Second fixed sparse batch after the first fully exits:

```bash
R9B_ARMS=(r9r2b_astra_groot_l10_50_cache r9r2b_astra_groot_l10_50_burst1 r9r2b_astra_groot_l10_50_burst3 r9r2b_astra_groot_l10_50_latch r9r2b_astra_groot_l10_50_random3 r9r2b_astra_groot_l10_P10)
```

Re-plan/sync the new list and run the same command. Dense batches substitute `_500_` in the five library-specific names; retain the shared P10. Never use a 300/500-pair manifest for these fitted methods. Maximum four servers and 32 workers, assigned ports/fleet only, one chain at a time. If a future **owned** round-2b launch requires stopping, use the runner's root-scoped abort for `r09_astra_round2b`; never pattern-kill or abort the coordinator's other root.

## Remaining uncertainty

Local checks cover 1,120 queries/tails, including 164 exact policy-tail checks, zero score-parity error on the probes, lifecycle reset, clone isolation, and budget caps. They do not establish remote renderer/encoder parity, actual cost after state-distribution changes, end-to-end latency, recovery efficacy, or SR preservation. Standard-mode confirmation has no new privileged progress trace; final success and actual cost are the primary outcomes, with monitor/call diagnostics as supporting evidence.

Final locked evaluation remains entirely coordinator-owned. No reader in this package accepts a holdout root. Select/freeze the final candidate family before the coordinator opens its locked outcomes, and preserve the predeclared simultaneous two-point criterion from `PROPOSALS.md`.
