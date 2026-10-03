# Round-2 hand-back

## What is ready

Independent task-agnostic correction and wrist-distillation candidates; no per-task routing or strength adaptation. New heads fit **0–19 only** and are ready for closed-loop evaluation on **20–29 only**. The package contains 28 fitted candidate pickles, 44 emitted arms including controls, and an explicit 100-pair manifest. No locked outcome or other researcher's round-2 output was used.

**Coordinator scheduling request:** run the two fixed long-task five-arm batches and the four-arm short dense wrist batch below. This follows the user's explicit instruction to hand back exact arms when duration is uncertain or could exceed roughly 30 minutes. Their duration is not established locally; I did not start a chain. This is a concrete runnable hand-back, not a request to approve unfinished implementation.

No candidate has new closed-loop SR. Do not advertise preservation of P10 SR or compare the old overlapping-fit corrector result as if it validated these newly fitted heads.

## Work completed

- 24 discovery-only compact source populations, identity firewall before payloads.
- Four fixed shared-head configurations in all eight cells; half/full motion and gripper controls; disjoint-path evaluation.
- Task-agnostic randomized-call value/risk moderators, with first-entry and local-derivative estimands. Mostly negative evidence; no promoted gate.
- Eight hundred explicitly addressed 20–29 physical episodes; initial predicates separated from newly achieved progress.
- Four actual wrist-input heads, fitted with 0–19 wrist paths and pure-cache wrist shadows.
- Eleven unit tests; 560 real-library serving queries, including both wrist camera modes and corrected blind tails. Online/standalone predictor parity was exact in these checks.
- Frozen artifact SHA256s, 44 emitted standard-mode arms, local five-arm dependency plan.

All computation used CPUs 10–21,54–65 with OMP/OPENBLAS/MKL threads=1. No GPU, policy inference, simulator, remote source deployment, synchronization, H100 server, timan108 worker, or tmux session was started. No git state change, package install, existing source edit, or changes to other research lines were made. No owned experiment process needs cleanup.

## Locations

Owned source/output directory:

```text
/home/weiland/projects/openpi/exp/offline_search/rounds/r09/explore_astra/round2/
```

| Path | Purpose |
|---|---|
| `REPORT.md` | Chinese owner section and conclusions |
| `PROPOSALS.md` | Ranking, exact frozen arms, evidence limits, final-claim protocol |
| `DATA_ANALYSIS.md` | Split, populations, estimands, measurements, negative results |
| `tools/data.py` | Discovery-only loader, temporal features, weighting, clustered intervals |
| `tools/shared_head.py`, `learn.py` | Fixed task-agnostic correction experiments |
| `tools/wrist_head.py` | True wrist-input distillation |
| `tools/call_value.py` | Randomized-call causal estimands |
| `tools/physical_audit.py` | Safe episode-addressed predicate audit |
| `tools/statistics.py` | Conservative simultaneous paired noninferiority bound |
| `tools/test_tools.py` | Unit tests |
| `methods.py`, `inference.py` | NumPy-only serving wrapper and shared head |
| `artifacts/*_temporal.npz`, `*_wrist.npz` | Frozen serving head coefficients |
| `artifacts/r9r2_astra_*.pkl` | 28 fitted candidates |
| `results/*.json`, `*.log` | All analyses, deployment checks and logs |
| `FROZEN_CANDIDATES.json` | Exact artifact/head hashes and split |
| `confirmation_specs.json` | 44-arm emitter input |
| `PROVENANCE.json` | Source/output integrity inventory |

New owned experiment root (configuration only, no episode outcomes):

```text
/home/weiland/trace_runs/os_closed_loop/r09_astra_round2_eval/
  arms.json
  config/
  manifests/eval100.json
  h100_sync/plan.json
```

Existing inputs remain read-only under `offline_search_store/derived/r09_astra/compact`. No new raw corpus was copied. Additional outputs stayed within the assigned round2 directory and new owned run root.

## Reproduce offline

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r09/explore_astra/round2/tools/reproduce.sh
```

The script re-fits only on 0–19, evaluates only 20–29, re-emits the owned configuration root, runs tests, and regenerates integrity hashes. It does not launch any remote work. NPZ/PKL container hashes can change on regeneration; use the newly generated freeze consistently, not a mixture of old and new artifacts. Reports are authored interpretations, not automatically regenerated prose.

To verify without refitting:

```bash
R9R2_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9R2_MOD=exp.offline_search.rounds.r09.explore_astra.round2.tools
"${R9R2_PY[@]}" -m unittest "$R9R2_MOD.test_tools" -v
"${R9R2_PY[@]}" -m "$R9R2_MOD.audit"
```

## Exact coordinator launch sequence — not executed

Do not substitute discovery300 or eval500. The head has already trained on 0–19. The manifest below contains only 20–29.

```bash
cd /home/weiland/projects/openpi
R9R2_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9R2_MOD=exp.offline_search.rounds.r09.explore_astra.round2.tools
R9R2_RUN=/home/weiland/trace_runs/os_closed_loop/r09_astra_round2_eval
R9R2_OPS=$PWD/exp/offline_search/closed_loop/ops/h100
R9R2_ARMS=(r9r2_astra_pi05_l10_50_cache r9r2_astra_pi05_l10_50_motion r9r2_astra_pi05_l10_50_joint r9r2_astra_pi05_l10_50_grip r9r2_astra_pi05_l10_P10)

# Requires the assigned fleet to be idle. Installs only the new round2 package
# sources on h100, with SHA verification and refusal to overwrite differing bytes.
WORKER_HOST=timan108 "${R9R2_PY[@]}" -m "$R9R2_MOD.deploy_sources" --execute
WORKER_HOST=timan108 "${R9R2_PY[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R9R2_RUN" "${R9R2_ARMS[@]}"
WORKER_HOST=timan108 SYNC_PORT=23196 \
  bash "$R9R2_OPS/sync_assets.sh" --concurrent "$R9R2_RUN" "${R9R2_ARMS[@]}"

# Use the runner's GPU-memory/admission checks before launching servers.
WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=8 MAX_ATTEMPTS=3 \
  OSCL_MANIFEST="$R9R2_RUN/manifests/eval100.json" \
  bash "$R9R2_OPS/chain_h100.sh" "$R9R2_RUN" "${R9R2_ARMS[@]}"
```

The local first-batch plan succeeded: **40 dependency files, 11.057 GiB logical assets**. This is not new transfer size; historical assets may already exist. Concurrent sync must verify/reuse them. If new bytes exceed the runner's batch limit or any ownership/hash/free-space guard refuses, stop that substep and have the coordinator resolve it; do not bypass guards or edit shared assets. Source deployment is separate from asset synchronization.

After the first chain fully exits and cleans up, replace the arm array, re-plan and sync, and use the same manifest/host/ports for the fixed GR00T batch:

```bash
R9R2_ARMS=(r9r2_astra_groot_l10_50_cache r9r2_astra_groot_l10_50_motion r9r2_astra_groot_l10_50_joint r9r2_astra_groot_l10_50_grip r9r2_astra_groot_l10_P10)
```

Then the wrist batch:

```bash
R9R2_ARMS=(r9r2_astra_pi05_spatial_500_cache r9r2_astra_pi05_spatial_500_wrist_cache r9r2_astra_pi05_spatial_500_wrist r9r2_astra_pi05_spatial_P10)
```

All remaining emitted candidates follow `r9r2_astra_<model>_<suite>_<size>_{motion,joint,grip}`; π0.5 also has `_wrist` and `_wrist_cache`. Their availability is not a recommendation to sweep until finding significance. The three fixed batches are the proposed initial work.

Only one chain at a time, at most four servers and 32 workers. If an owned launch needs stopping, use the runner's owned-root abort, not process-pattern killing:

```bash
WORKER_HOST=timan108 bash "$R9R2_OPS/abort_h100.sh" "$R9R2_RUN"
```

## What to measure and what remains unresolved

Report paired wins/losses, candidate−cache and candidate−P10 SR differences, pooled owner IR, actual encoder/call counts, and end-to-end latency. `_grip` isolates the proposed new mechanism. `_motion` controls for task pooling and changed supervision. The wrist baseline separates correction gains from camera savings.

The package has passed local inference/tail contracts, not remote renderer/policy-server parity. Exact sensor-feature numeric parity, closed-loop feedback behavior, latency, and SR are unresolved. The call-value tool does not estimate deterministic-gate SR. The privileged predicate audit does not establish causal failure onset.

Final family and untouched evaluation remain coordinator-owned. Freeze the selected candidates before opening new 30–49 results; use the conservative paired bound and family correction described in `PROPOSALS.md`. Do not feed locked outcomes back into these tools or tune against them. A failure to establish two-point noninferiority must be reported plainly, even if the point estimate looks favorable.
