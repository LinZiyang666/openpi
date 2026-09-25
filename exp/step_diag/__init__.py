"""Step reduction vs warm start: closed-loop shadow diagnostics and equal-NFE arms.

Plan: ``logs/step_vs_warmstart_diagnostics_plan.log.md`` (v3.1). The package is an
experiment-layer composition over the staged inference API and the cache
orchestrator; it never modifies ``src/``.

Modules
-------
metrics        the frozen deviation kernel (``weighted_chunk_deviation`` over the executed
               window and executed dims), per-decision quantities (``d_k``, ``r_k``,
               ``disp_K``) and the PCA-2D / GMM fit diagnostic
recorder       ``DiagRecorder`` (one per served arm process) + ``DiagSession`` (one per client
               connection): policy-agnostic per-decision shadow sampler + evidence writer
               (JSONL metadata + float32 ``.npz`` arrays, finalize / error rows)
pi05           ``Pi05DiagInterceptor``: ``InferenceInterceptor`` subclass that runs the
               recorder on the teacher-executed path and counts stage-3 calls
groot          ``GrootDiagPolicy`` (teacher / plain-k / full) and ``GrootEvidencePolicy``
               (warm): GR00T served objects recording the same evidence via the staged runner
serve_diag_*   server entry points composing the above into ``serve_policy`` /
               ``serve_groot_n15`` / ``serve_groot_libero`` (single arm per process)
envs           the frozen environment identity table (§3.0), Q-B groups and manifest checks
emit_arms      shadow / forced warm-start yamls (per teacher) + arm index
run_diag       RoboCasa conductor driver for the shadow / plain / full / warm arms
run_libero_diag  LIBERO conductor driver: the shadow, or any arm on a task subset (self-start round)
worker_entry   ``StepDiagEpisodeRunner``: the production RoboCasa runner + client proxy
               (identity stamp, infer count), counting env and one summary row per episode
analysis/      admission + metrics tables (``analyze_shadow``), equal-NFE pairing and the
               pre-registered verdicts (``aggregate_arms``)
ops/           server / worker launchers, evidence pull + sha check, weight freeze, analysis
config/        base RIT cells, emitted arms + ``index.json``, worker-island env file
"""
