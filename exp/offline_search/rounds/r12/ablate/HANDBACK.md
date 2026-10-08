Implemented the owner’s PURE CACHE camera ablation. There are **25 B-screen arms and 25 staged Test-A arms**, using 25 frozen metric/cache artifacts. Nothing was launched or transferred; no GPU, remote operation, git command, server, smoke, real plan/sync/chain, process signal, or deletion was used. Planning and sync in tests used mocked external transport.

Every cell has `both`, `wrist_sub`, `wrist_refit`, `agent_sub`, `agent_refit`. `R12CacheAblate` retains the selected R10 library and layer-1 AWM kernel retrieval plus `anchor_tail`, budget-one, `budget_only` blind reuse. It removes the guard wrapper and correction heads. There is no `--os-judge`, knob, policy tail, look-half planner, completion, shadow policy, or policy call. The plugin always returns FULL_HIT on the method’s pick.

The deployed features are `[agent PCA64, wrist PCA64, state8]`. Let `F` be the stored deployed affine distance transform (`T.Wf`), or `T.W0f` for step zero. **Restricted** means the principal block `Q[I,I]` of the joint Mahalanobis matrix `Q = F Fᵀ`, with `I = [selected camera64, state8]`: `d² = Δx_I Q[I,I] Δx_Iᵀ`. The omitted camera’s difference is zero; camera/state cross terms remain. This is not a separate camera term of an additive metric. No parameters are fitted. Distances use the stored float32 transform promoted to float64 for the norm, then return float32.

**Refit** retains the deployed independent per-view B PCA and state features and freshly fits a 72-dimensional AWM metric per task on the same selected B episodes: action-similar cross-episode neighbors `nn=3`, trace-scaled ridge `.1`, full rank, original float32 action sigma promoted to float64, plus the deployed early fit on steps `<=2` used only at step zero. It preserves the deployed float32 candidate codes and affine early representation. Opus’s wrist metric has different sigma/serving arithmetic, so these artifacts use a fresh deployed-recipe fit. `build.py` enforces the selected-library read boundary; no A/dev query or outcome is a fit input.

One-camera serving encodes only its selected real camera from the initial look onward. Pi05 uses a zero masked dummy without an extra encoder call in the single-camera path; `both` retains the stock dummy work. GR00T uses the same genuine single-image encoder path as R12 look-half, with the third-person equivalent added separately. Its approximately 2% single-camera versus joint-encoding difference is retained, including comparison against joint-encoded library keys. No GPU measurement was performed here.

Per-decision `cache_ablate_cost`, `camera_encoded_count`, `camera_work`, and B look-shard camera counts come from request-owned dispatch receipts. Prices are pi05 `.152`/`.076`, GR00T `.148`/`.074` for two/one cameras; blind decisions, policy calls, and completions cost zero. Collection’s authoritative `cost_ledger` and `cache_ablate_cost_ledger` report `one_camera_looks`, `two_camera_looks`, `encoded_cameras`, and `calls`. Engagement requires zero two-camera looks in single-camera arms and zero one-camera looks in `both`; missing camera receipts or a MISS are errors. All arms select `closed_loop.execution_worker` on **both pools**, and missing/malformed execution receipts fail before encoding. `--os-cache-ablate-sha256` binds the served metric artifact bytes; relocation updates that hash while retaining the original source SHA in sync receipts.

B roots live under `exp/offline_search/rounds/r12/ablate/pilots/`; A definitions under `ablate/test_a_staged/`, installing to `/home/weiland/trace_runs/os_closed_loop/`:

| Cell | B root | A root |
|---|---|---|
| pi05_l10_50 | r12_ablate_pi05_l10_50 | r12_ablate_test_pi05_l10_50 |
| pi05_spatial_50 | r12_ablate_pi05_spatial_50 | r12_ablate_test_pi05_spatial_50 |
| pi05_spatial_200 | r12_ablate_pi05_spatial_200 | r12_ablate_test_pi05_spatial_200 |
| groot_spatial_50 | r12_ablate_groot_spatial_50 | r12_ablate_test_groot_spatial_50 |
| groot_l10_50 | r12_ablate_groot_l10_50 | r12_ablate_test_groot_l10_50 |

Each arm name is `<root>_<method>` for the five methods above. B has 100 episodes/arm, K10 per task, exact byte copies from `r11_dev_k10_size50`/`size200`, with look/image recording on. A has 500 episodes/arm, exact byte copies of the R11 `eval500.json` templates, with all recording off. `pilot_roots.json`, `test_a_roots.json`, each `ROOT.json`, and `artifacts/<cell>/build.json` list exact paths, selections, and artifact hashes. Installation refuses existing destinations.

Files/functions added or changed, relative to `exp/offline_search/`:

| Files | Work / deployment |
|---|---|
| `closed_loop/cache_ablate.py`, `cache_ablate_config.py` | New method, reduced retrieval and frozen metric, explicit selection validation. Deploy to the policy-server tree, including h100 before remote serving. |
| `closed_loop/cache_ablate_cameras.py`, `cache_ablate_ledger.py` | Separate pi05/GR00T acquisition hooks, receipts, work and engagement aggregation. Deploy to server/collection trees. |
| `closed_loop/plugin.py`, `look_recording.py` | New opt-in parsing/hash check, initial camera dispatch, receipt checks and decision/look records. Deploy to policy servers; timan also needs the current `look_recording.py` for B stamping. |
| `closed_loop/ops/h100/assets.py`, `control.py`, `node.py`; `ops/collect.py` | Plan validation, relocated SHA binding, worker selection, final argv validation, authoritative cost collection. Controller/collector use local copies; deploy the `node.py` helper to h100 and the selected timan island. |
| `rounds/r12/ablate/build.py`, `prepare.py`, `coordinator_commands.sh`; `closed_loop/test_cache_ablate.py` | B-only builder, root emitter/installer, coordinator commands, CPU tests. Builder/emitter stay local; source may accompany the deployment snapshot. |

The selected timan island must have the current R12 `execution_worker.py`, `worker_pool.py`, `lookhalf_config.py`, and `look_recording.py`; the first three were not changed. Existing deployed/R12LookHalf/R12Guard behavior passed regression checks. No remote snapshot was updated here.

**440 distinct CPU checks passed, zero failures:** 128 existing R12 checks, 71 new ablation checks, 233 coordinator/devset checks, and 8 camera-dispatch regressions. The broad run passed 429; subsequent additions and affected collection checks also passed. Evidence: `VERIFICATION.json`, `CPU_ALL.log/.xml`, `CPU_FINAL_ADDITIONS.log/.xml`, `CPU_DEPLOYED.log/.xml`, `CPU_COLLECTION.log/.xml`, and `SOURCES_SHA256.json`. Three launch-dependent tests were deselected: detached fake-process lifecycle, supervisor natural completion, and rsync-daemon pull. They would start real processes/listeners. One existing pynvml deprecation warning. Checks used cores `22-37,66-81`, CUDA hidden, and one OMP/BLAS thread. Tests verify full final local and h100 argv for every arm on both pools, source/served hashes, real worker pool attestation, byte-equivalent two-camera layer-1 actions/tails (also through the deployed wrapper with correction removed), B-only metric reproduction, installed encoder invocation counts, execution refusal, and accepted-attempt ledger aggregation.

Coordinator commands below were **not executed**. Use available fleet/ports after the current Test-A jobs release them. Local defaults are four servers on `23150-23153`, timan107, WPS16. Override `PORTS`, `LOCAL_SERVER_COUNT`, `WORKER_HOST` as needed. Do not use `23100-23103` while the live runs occupy them.

```bash
cd /home/weiland/projects/openpi
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1
export OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SERVER_HOST=local WORKER_HOST=timan107
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B)
CTL=exp.offline_search.closed_loop.ops.h100.control
S=exp/offline_search/rounds/r12/ablate/coordinator_commands.sh

# Deploy the selected worker snapshot once, when that fleet is available.
"${P[@]}" -m "$CTL" setup --worker-only

# Install all five staged A definitions; this launches nothing.
"${P[@]}" -m exp.offline_search.rounds.r12.ablate.prepare --install-test-a

# Five B roots, all five arms per root, sequentially.
for cell in pi05_l10_50 pi05_spatial_50 pi05_spatial_200 groot_spatial_50 groot_l10_50; do
  bash "$S" plan B "$cell"
  bash "$S" sync B "$cell"
  bash "$S" chain B "$cell"
done

# After screening, define each SELECTED_* array with only coordinator-chosen
# methods; skip any unselected cell. A refuses an empty method list.
run_selected_A() {
  local cell=$1; shift
  bash "$S" plan A "$cell" "$@"
  bash "$S" sync A "$cell" "$@"
  bash "$S" chain A "$cell" "$@"
}
run_selected_A pi05_l10_50 "${SELECTED_PI05_L10_50[@]}"
run_selected_A pi05_spatial_50 "${SELECTED_PI05_SPATIAL_50[@]}"
run_selected_A pi05_spatial_200 "${SELECTED_PI05_SPATIAL_200[@]}"
run_selected_A groot_spatial_50 "${SELECTED_GROOT_SPATIAL_50[@]}"
run_selected_A groot_l10_50 "${SELECTED_GROOT_L10_50[@]}"
```

For h100 serving, deploy the changed server sources first using coordinator-authorized setup when the active jobs permit it; set `SERVER_HOST=h100`, the selected timan island, suitable h100 `PORTS`, and its worker count before using the same per-root commands. Sync transfers the bound artifacts and exact configs; the A installer does not copy run/state directories.
