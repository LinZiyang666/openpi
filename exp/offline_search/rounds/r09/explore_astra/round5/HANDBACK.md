# Ready for coordinator review; no launch performed

Package: `/home/weiland/projects/openpi/exp/offline_search/rounds/r09/explore_astra/round5/`

New run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_r5/`

Six frozen long-50 arms, three per model. The control artifacts are byte-identical copies of the leading r3c artifacts. All candidate heads/thresholds retain fitting inits 0–19; the evaluation manifest contains exactly 100 pairs with inits 20–29. Predictions were written at 2026-10-02T08:59:55.754479+00:00, before arm emission. There is no new closed-loop result.

| Arm | Role |
|---|---|
| `r9a5_pi05_l10_50_control` | Guard + corrected cache + persistent pace escalation |
| `r9a5_pi05_l10_50_latch` | Replace escalation with frozen detector/latch |
| `r9a5_pi05_l10_50_pace12` | Keep pace entry, cap takeover at 12 fresh calls |
| `r9a5_groot_l10_50_control` | Guard + corrected cache |
| `r9a5_groot_l10_50_latch` | Add frozen detector/latch |
| `r9a5_groot_l10_50_pace12` | Add pace-entry 12-call takeover |

No guard threshold, guard call cap, corrector coefficient, library or look schedule changes. The prescribed corrector's existing task-specific heads remain frozen; no new task-indexed behavior is introduced. All production forced-trigger lists are empty. The extra variant family is pace12 only.

## Local evidence

Nine unit tests pass. Twelve real CPU plugin selftests pass, including forced starts in all four reset episodes for each new takeover arm. They cover 576 decisions and 120 policy tails. Frozen guard source/asset hashes are asserted at fit time; `judge.burst=0`; both models' correction changes motion on the `os_synth` path and updates the anchor.

The local **control plan passed**, 59 assets / 15.951 GiB logical. `H100_SOURCES.sha256` lists 69 files with full hashes; `H100_SOURCES.json` includes local and H100 destinations. Existing unchanged remote sources can be reused. The inventory is the static Python import closure of the serving methods; the pre-existing plugin/runtime environment is still required.

Use the provided control adapter, because the stock planner otherwise opens/hashes the raw multi-init query metadata. The adapter calls the existing control implementation with an already filtered identity-only store, and places that store under `/data/oscl_h100/runs/r09_astra_r5/serving_store` on H100. It does not replace the fleet's shared query metadata. This also avoids a concurrent-sync hash conflict with that metadata.

## Verify locally (no network or jobs)

Run from the repository; these commands stay within the CPU/thread limits:

```bash
cd /home/weiland/projects/openpi
R9A5_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
R9A5_MOD=exp.offline_search.rounds.r09.explore_astra.round5.tools
R9A5_RUN=/home/weiland/trace_runs/os_closed_loop/r09_astra_r5
R9A5_ARMS=(r9a5_pi05_l10_50_control r9a5_pi05_l10_50_latch r9a5_pi05_l10_50_pace12 r9a5_groot_l10_50_control r9a5_groot_l10_50_latch r9a5_groot_l10_50_pace12)

"${R9A5_PY[@]}" -m "$R9A5_MOD.verify"
"${R9A5_PY[@]}" -m unittest "$R9A5_MOD.test_tools" -v
```

`tools.selftest` checks/reports existing successful probe evidence on a repeated invocation; original captures and admission receipts live under `r09_astra_r5/selftest/`. `tools.analyze` reproduces the allowed-data analysis. `tools.build --emit` is the original build command and deliberately refuses an already frozen run root; do not refit or regenerate the freeze during execution.

A repeat local dependency plan is:

```bash
WORKER_HOST=timan108 "${R9A5_PY[@]}" -m "$R9A5_MOD.control" plan "$R9A5_RUN" "${R9A5_ARMS[@]}"
```

The integrity seal covers the current plan and analysis output too. Reproducing timestamped outputs or changing a plan's contents can invalidate that seal; verify the delivered freeze before such reproduction.

## Coordinator-only remote commands — not executed in this round

After the coordinator has assigned an idle fleet and ports, the exact sequence below runs the six arms sequentially on timan108, so every model's comparison includes its control in the same batch. The displayed ports are the four-port block used in the preceding π0.5 run; use them only after the coordinator releases that block. This package did not inspect, stop or modify any running process. To use timan107, change `WORKER_HOST` consistently for source install, plan, sync and chain; do not split one model's control/candidates across fleets.

```bash
# Assumes the local variables above are set in this shell.
WORKER_HOST=timan108 "${R9A5_PY[@]}" -m "$R9A5_MOD.sources" --execute
WORKER_HOST=timan108 "${R9A5_PY[@]}" -m "$R9A5_MOD.control" plan "$R9A5_RUN" "${R9A5_ARMS[@]}"
WORKER_HOST=timan108 SYNC_PORT=23196 \
  "${R9A5_PY[@]}" -m "$R9A5_MOD.control" sync --concurrent "$R9A5_RUN" "${R9A5_ARMS[@]}"
WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=8 MAX_ATTEMPTS=3 \
  OSCL_MANIFEST="$R9A5_RUN/manifests/eval100.json" \
  "${R9A5_PY[@]}" -m "$R9A5_MOD.control" chain "$R9A5_RUN" "${R9A5_ARMS[@]}"
```

`sources --execute` takes the existing fleet lock and refuses to replace differing source files. A hash mismatch means resolve the frozen dependency discrepancy; do not silently overwrite the guard/corrector. Existing runner admission, asset-hash and fleet checks remain in force. Nothing in these commands is part of the work executed here.

## How to judge the new run

Primary: paired final success and measured pooled owner IR against `control` within each model. Report +/− pairs, exact paired p and init-cluster intervals; use the prediction file's predeclared screen criteria and Holm adjustment if claiming significance across four candidate comparisons. Do not interpret <=5 percentage points on this screen as established.

Fresh decision extras on new arms are `r9a5_score`, `r9a5_start`, `r9a5_call`, `r9a5_used`, `r9a5_guard`, `r9a5_added`; `r9a5_added` identifies takeover requests not already covered by the guard. `pace12` has score −1 (no learned score). Additional takeover uses reason 95 and bit 12 when the guard did not already force a miss; existing guard reasons/flags are retained on overlap. Only fresh decisions count alerts/calls; blind extras carry the preceding diagnostics and must not be double-counted.

Count actual post-alert paired rescues separately from all paired wins, first-alert timing, cap exhaustion, and guard overlap. Ordinary guard calls after takeover budget exhaustion remain valid. In the old experiment six of eleven π0.5 paired gains occurred without takeover; that is why this distinction matters. Standard logs cannot establish camera-based detector counterfactuals, post-divergence SR, or an exact offline latch comparison on the corrected stack.

Future locked evaluation remains exclusively coordinator-owned. No tool here accepts a locked-root name or opens outcomes for inits 30–49. No owned remote processes were created, so no process cleanup is needed.
