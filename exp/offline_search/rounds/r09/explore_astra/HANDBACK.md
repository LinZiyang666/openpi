# Astra hand-back

## Result and coordinator decision requested

Main new candidate: a small motion-residual head trained on cache-path policy shadows. It preserves the ten-control commitment and gripper commands, costs approximately the existing cache IR, and improves held-out-in-fold local action error across all eight cells and transfer paths. **New closed-loop SR is unknown.** The main simpler comparator is a library-only ten-step retrieval metric. A separate whole-episode task router has a promising π0.5 long-50 offline SR/cost point but unstable selection.

**Request to the coordinator, as required by the brief for runs with uncertain/~30-minute duration:** schedule the first four-arm discovery batch below, then the three-arm π0.5 long-50 episode-routing comparison if resources permit. Also reserve a local **23100–23197 synchronization port** before invoking asset sync. Astra was assigned only H100 server ports 23230–23239; the runner rejects those as sync ports, so I did not take an unassigned one. The exact code, fits, manifest, arms and local dependency plan are ready for review.

No additional scientific claim depends on approving a speculative analysis. What remains is the actual closed-loop test and remote environment validation.

## What ran

CPU-only work, with affinity inside **10–21,54–65** and OMP/OPENBLAS/MKL threads=1:

- Discovery extractor: 70 arms, 21,000 accepted episodes, 952,842 decisions.
- Same-observation synthesis, horizon-metric, camera and teacher-proxy calibration analyses.
- Five-fold residual learning, transfer/negative-control audits and a matched-label shadow-memory baseline.
- Whole-episode cost/SR LP routing, 216 sensitivity configurations, and 1,000 refit bootstrap samples for each of eight binary routers.
- Audited randomized-call effects on all four available IP cells.
- Eleven unit/deployment tests; additional 80 real-library residual query checks; 40 task-level metric tests exercising both early and ordinary retrieval and blind tails.
- Prepared 34 exact discovery-only arms with fitted artifacts. Local four-arm dependency planning succeeded (37 files, about 4.374 GiB logical assets).

**Did not run:** GPUs, images through a policy, simulator branches, closed loop, remote source deployment, asset synchronization, H100 servers, timan108 workers or tmux sessions. No owned process remains running. No git state changes, package installations or existing-source edits were made. All writes are in the assigned research directory, `derived/r09_astra`, or `r09_astra_confirmation`.

## Files and outputs

| Location | Contents |
|---|---|
| `REPORT.md` | Chinese owner section, findings and limits |
| `DATA_ANALYSIS.md` | Estimands, counts, numbers, commands, negative results |
| `PROPOSALS.md` | Ranking, per-cell IR/evidence, exact arms and confirmation criteria |
| `tools/` | Extractor, estimators, benchmarks, tests, artifact preparation, source-deployment utility |
| `methods.py`, `inference.py` | Isolated NumPy-serving prototypes; no existing serving source edits |
| `artifacts/*_student.npz` | Eight compact fitted residual heads |
| `artifacts/r9_astra_*.pkl` | Eight residual, eight episode-lottery and four sparse metric prefits |
| `confirmation_specs.json` | 34-arm emitter input |
| `results/*.json` | Complete numeric results, including unfavorable cells/configurations |
| `*.log` | Tool runs, tests, serving checks and local dependency-plan logs |
| `PROVENANCE.json` | Final output audit and owned-file SHA256/size inventory |

Absolute compact-store directory:
`/home/weiland/trace_runs/offline_search_store/derived/r09_astra/compact/`.
All compact files enforce discovery membership on load. Each of the 70 original compact arm NPZs has adjacent source/augmentation provenance JSON. Additional student-error NPZs contain only discovery evaluation arrays.

Absolute confirmation root:
`/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation/`.
It contains `arms.json`, `config/`, `manifests/discovery300.json`, and `h100_sync/plan.json` / prepared local relocation files. It contains no accepted experiment outcome. No remote synced receipt or DONE marker is claimed.

## Reproduce offline

From `/home/weiland/projects/openpi`:

```bash
bash exp/offline_search/rounds/r09/explore_astra/tools/reproduce.sh
```

This writes only owned outputs and the new confirmation root. Existing compact extraction resumes; add `--overwrite` to the extractor when deliberately changing the capture generation. The script does not run remote commands. The head needs NumPy/SciPy, not scikit-learn. Exact interpreter/affinity prefixes and individual commands are in `DATA_ANALYSIS.md`.

To verify the final inventory after any intentional regeneration:

```bash
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r09.explore_astra.tools.audit_outputs
```

## Coordinator launch steps — not executed

Run only after assigning the sync port and scheduling the batch. New Python modules are **not** installed by asset sync alone. `deploy_sources` installs only `methods.py` and `inference.py` under the new owned server package, with content verification and refusal to overwrite differing existing bytes. Default mode prints a plan; `--execute` performs the installation under the fleet lock. No existing shared serving source needs replacement.

```bash
cd /home/weiland/projects/openpi
R9PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
R9_RUN=/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation
R9_OPS=$PWD/exp/offline_search/closed_loop/ops/h100
R9_ARMS=(r9_astra_pi05_spatial_50_cache r9_astra_pi05_spatial_50_residual_half r9_astra_pi05_spatial_50_metric10 r9_astra_pi05_spatial_P10)

WORKER_HOST=timan108 "${R9PY[@]}" -m exp.offline_search.rounds.r09.explore_astra.tools.deploy_sources --execute
WORKER_HOST=timan108 "${R9PY[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R9_RUN" "${R9_ARMS[@]}"

# Set R9_SYNC_PORT to a coordinator-reserved 23100..23197 port first.
WORKER_HOST=timan108 SYNC_PORT="${R9_SYNC_PORT:?coordinator-reserved sync port required}" \
  bash "$R9_OPS/sync_assets.sh" --concurrent "$R9_RUN" "${R9_ARMS[@]}"

WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=8 MAX_ATTEMPTS=3 \
  OSCL_MANIFEST="$R9_RUN/manifests/discovery300.json" \
  bash "$R9_OPS/chain_h100.sh" "$R9_RUN" "${R9_ARMS[@]}"
```

The wrapper's local hard-coded affinity 14–17,58–61 is within Astra's allocation. Follow the runner's memory/admission checks, GPU free-memory checks and owned abort/cleanup protocol; no other processes or ports should be touched. One chain, at most four H100 servers and 32 workers. Remote interpreters/checkpoints/render parity are the runner's existing setup, but the new candidates have not been end-to-end validated there.

Second optional batch, after stopping/collecting the first chain:

```bash
R9_ARMS=(r9_astra_pi05_l10_50_episode r9_astra_pi05_l10_50_cache r9_astra_pi05_l10_P10)
```

Re-plan/sync that exact list before launching the same discovery manifest. The residual candidates for long-50 can be evaluated as a later fixed batch if the first result supports them. All 34 names and parameters are in the emitter input and emitted `arms.json`.

If interruption is needed, use only the owned root:

```bash
WORKER_HOST=timan108 bash "$R9_OPS/abort_h100.sh" "$R9_RUN"
```

## Remaining limitations

- Action shadows do not identify counterfactual SR; the observed cadence counterexample is included in both reports.
- Student training consumes discovery trajectories and extra policy labels; old library-size labels alone do not describe its data budget.
- Capture-based array identity/coverage and local serving tests are not a remote simulator parity certificate.
- The task router's bootstrap is a finite-sample sensitivity analysis for a selected discrete learner, not a final NI certificate.
- Existing wrist and no-call designs remain worth confirming, but even favorable discovery points fail the conservative simultaneous two-point bar.
- The coordinator owns opening inits 30–49, freezing the final family of claims, and deciding whether 200 locked pairs provide enough precision. This hand-back does not authorize tuning on those pairs.
