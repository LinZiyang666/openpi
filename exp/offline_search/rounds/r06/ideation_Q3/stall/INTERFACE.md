# P5 stall interface for P4

`StallModel.fit(library, metric=fitted_A, commit_controls=min(H,2R))`, `.save(directory)`, and `.load(directory)` are implemented in `stall.py`. The artifact is self-contained; source-bank and source-fit fingerprints are in `stall.json`. One `StallTracker(model, task_id_or_instruction)` belongs to one episode. No call budget, randomness, policy invocation, or LOOK scheduling occurs inside the tracker. C owns those decisions and the shared rho budget.

Preferred integration after A retrieval, avoiding duplicate projection:

```python
tracker.observe({"metric_code": z, "metric": "early" if step == 0 and awm.early else "main"},
                control_index=actual_executed_controls)
status = tracker.status()
```

`z` is the deployed A query metric code, exactly as logged in P3 `retrieval.metric_code`. In the main regime it is `x @ T.Wf - T.shift`; in the early regime it is `x0 @ T.W0f - T.c0`. A raw `QueryView` (or mapping with `key_v*`, `rs`, `step`) also works and uses saved projections. An ndarray means an already encoded **main** vector, never a raw visual key. Mixed-regime windows preserve each observation's metric; no early code is compared to main template coordinates. Distances are Euclidean in those fixed codes, computed in float64; action-tail continuity reranking is not a metric and is not added to alignment costs.

Every fresh vision observation, including extra LOOKs, must be supplied with strictly increasing **actual control indices**, not request indices. Duplicate/decreasing times raise an error; create a new tracker at reset. Status is cached in `observe`; `status` has no inference or alignment side effect. Invalid/nonfinite keys clear the window and yield inactive; missing references/unknown tasks yield inactive. No missing value is treated as zero motion.

States: `inactive`, `ok`, `slow_confirmed`, `slow_ambiguous`. Diagnostics include `delta_hat`, `e90`, `a10`, `window_span`, `W`, `K`, reference episode count; active statuses also include phase, distance, spread, and the selected reference window identifiers. Before W+1 fresh observations the tracker is inactive. The adjustment is `W*L / actual_window_span`. C must rate-limit an ambiguous extra LOOK to at most one per `W*L` controls and apply its own call cooldown/budget. The tracker may repeatedly return ambiguous without authorizing repeated LOOKs.

Portable explicit metric mapping (for a fitted representation other than the legacy A class):

```python
metric = {"tasks": {str(task_id): {"rows": row_ids,
          "codes": {"main": main_codes, "early": optional_early_codes}}},
          "projection": None, "provenance": {"source_fit_sha256": "..."}}
```

Codes are 2-D arrays in the exact fitted metric, aligned to `rows`. A generic library mapping supplies `manifest` (`H`, `exec_steps`, optional `task_map`), `task_id`, `episode`, `step`, `success`, and optionally actual `control_index`; otherwise `step*manifest.exec_steps` supplies timestamps. Successful episodes supply both templates and references. No test trajectories enter `fit`.

Artifacts are expected at `/tmp/q3_stall_fits/<cell>/stall.pkl` and `.json`, with `stall.loeo.csv`. Load either the directory or the `.pkl` path. Pickle artifacts must be trusted; hashes detect corruption, not a malicious replacement of both payload and sidecar.
