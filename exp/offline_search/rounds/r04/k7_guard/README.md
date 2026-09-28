# K7 vision-confirmed stuck guard

Use `exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge`
with `stuck_guard="vision_confirmed"`, `progress_guard="noprog_span"`, and K1
`base_kwargs`. Deployment rows are in `arms_k7.json`; each requires a full model,
`--os-blind --os-judge guard_only`, and its exact matching prefit artifact.

The all-vision path calls stock MixedJudge and preserves every stock result field.
Across blind gaps, raw valid-state L2 motion must be below stock `m_thr`, and each
interval must be bounded by actual vision anchors whose minimum camera,
task-centred cosine meets stock `c_thr`. The confirmed trailing transition count
also feeds V7's stuck feature, without changing library calibration. No blind key
is invented and no guard is evaluated while serving a blind HIT. All K1 look
gates, phase stepping, and span progress logic remain in force.

`stuck_guard="dense"` retains K1 as an explicit comparison switch. This module
refuses GR00T at fit time; its terminal sign differs between stock and K1.

See `HANDBACK.md` for measured results, caveats, SHA256s and exact commands.
`prefit.sh` has the requested final destination. That destination is read-only in
this session; `prefit_staged.sh` records the executed fits under
`/tmp/k7_guard_fits`. Copy the verified files before emitting/running the arms.

Verification drivers (run from repository root with the prefix below):

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
K=exp.offline_search.rounds.r04.k7_guard
"${PY[@]}" -m "$K.run_checks" parity --tag rerun
"${PY[@]}" -m "$K.run_checks" rates --tag rerun
"${PY[@]}" -m "$K.run_checks" edges --tag rerun
"${PY[@]}" -m "$K.run_checks" smoke --tag rerun
"${PY[@]}" -m "$K.run_checks" plugin --tag rerun
"${PY[@]}" -m "$K.concurrency_test" --source installed --config all --out /tmp/k7_guard_concurrency_rerun
"${PY[@]}" -m "$K.historical"
"${PY[@]}" -m "$K.validate_arms"
```

Use a fresh tag/output directory for plugin and concurrency reruns. The plugin
appends logs, and the concurrency driver refuses existing worker directories.
Concurrency uses eight connections in one worker process and compares against a
fresh serialized worker in the observed reservation order. All non-timing decision
fields, actions, guards, progress memos, anchors and histories must match.
