<task>
Continuation of P2 (you own `exp/offline_search/rounds/r06/p2_ablations/`). Your metric ablation (identity vs learned
per-task metric) ran closed-loop in all 8 cells (π0.5 −12.6 / −9.4 / −6.2 / −2.6 pp, GR00T −10.2 / −11.2 / −1.0 /
−1.2 pp for l10-50 / l10-500 / sp-50 / sp-500). The owner now asks for a **second paper ablation of A's visual key**:
is it better to spatially pool the stage-1 image tokens before PCA (current A) or to run PCA directly on the tokens?

- Current A: per camera, the 16×16 grid of stage-1 image tokens (256 × 2048) is pooled to 4×4 (`key_v0/key_v1`,
  f32[32768], key builder `cp1_spatial_pool_16` / `cp1_groot_libero_spatial_pool_16`), then PCA-64 per camera, plus
  the 8-d state → per-task z-score → per-task learned metric M_τ → top-16 kernel synthesis (kref 5 / 8) → 10-control
  commit (`rounds/r04/k1_blind/blind_awm.py:BlindAWM` on `rounds/r02/g1_awm/awm.py:AWM`, specs in
  `r05_ptail/arms_in.json` `r5t_p_*_tail1uc`, `r04_blind/arms_in.json` `r4b3_p_sp_500_tail1uc`,
  `r05_x/arms_in.json` `r5x_g_*_tail1u`).
- Ablation "direct PCA": per camera, PCA-64 fitted directly on the flattened full token grid (256 × 2048 = 524,288
  dims), no spatial pooling; everything else identical (same 8-d state, same z-scoring, same per-task metric learning
  recipe re-fitted on the new features, same early step-0 branch structure, same kernel / kref / anchor_tail budget /
  commit). Make the pooling grid a parameter (full = 16×16, current = 4×4, also 1×1 / 2×2 / 8×8) so that other points
  of the resolution ladder need only a fit, but produce specs only for "direct" (8 arms: both models × {l10,
  spatial} × {50, 500} libraries, the same library names the A specs use).

Facts to use: every stored library has full tokens (`<store>/library/<model>_<suite>/<lib>/tok/{v0,v1,rows}.npy`,
f16[256,2048] per row, `manifest.json` `tok_complete: true`; e.g. π0.5 l10 bpool_cs 29,472 rows ≈ 31 GB per camera,
GR00T l10 bpool_all 29,631 rows). Online, the plugin exposes `tok_v0 / tok_v1` (stage-1 prefix tokens) on every decision
(`closed_loop/plugin.py` docstring and `tokens()`, `--os-tokens on` default) — verify that the online tokens equal the
stored library tokens' definition (dtype, layout, which layer / prefix, camera order) for BOTH models before relying
on it, e.g. on a recorded query from the tok subsample. Fit with a streaming / randomized PCA (memmap chunks) that
fits the CPU and RAM budget; report fit time, memory, explained variance, and projection matrix size / online
projection latency on CPU (and on GPU if the plugin can use it without extra servers).

Deliverables: the method subclass in your directory; `arms_pca.json` (8, emit_arms format, `<RUN>` placeholders,
every other field copied exactly from the existing A specs); prefits into
`/home/weiland/trace_runs/os_closed_loop/r06_abl/fits/` (sizes, sha256) — if your sandbox cannot write there, write
to `/tmp/p2_pca_fits/` and say so; replay tests on recorded queries proving the new arm equals A except for the visual
feature block (identical state block, same code path), a numeric sanity check (top-16 overlap with A, offline action
error on held-out recorded queries — descriptive only, closed-loop decides), and the online-token equality check;
smoke recipe (2 arms × 4 episodes); `HANDBACK.md` section. Final message: short summary.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 18-25,62-69`, at most 8 processes, OMP_NUM_THREADS / OPENBLAS /
  MKL at most 4 threads per process and at most 16 threads in total, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python`
  from the repo root. CPUs 38-43 and 82-87 belong to another project: never use them. Keep RAM under 64 GB.
- Do not edit `src/`, `harness/`, `closed_loop/*`, `ops/*` or earlier rounds' files; subclass or wrap. If a shared
  plugin change is unavoidable, stop and say so instead of making it.
- No git. No `rm -rf`. Never `pkill -f`. Never start servers, LIBERO workers, chains, or touch tmux sessions, ports
  23100-23199 or timan107 — closed-loop experiments (pilot lanes, receiver) are running on this machine; the coordinator
  runs all experiments. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it in HANDBACK.md, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
