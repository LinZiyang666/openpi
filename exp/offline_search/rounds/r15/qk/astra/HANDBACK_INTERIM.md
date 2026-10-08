# Interim handback — R15 track B

2026-10-05. DATA and first CPU prototypes are complete. Full design, neural comparison, timing, and frozen gates follow in this directory. No new-method closed-loop outcomes have been read or generated.

**First result: light linear image keys are feasible to fit, but accuracy is not yet close enough to recommend deployment.** On L10-50, 10-control normalized chunk RMS is .4396 with the deployed key geometry, .5145 with symmetric spatial RGB PCA+AWM, and .4945 with the best tested asymmetric ridge (alpha .1). On L10-200, corresponding values are .3851, .4534, and .4347 (alpha .01). These are direct outputs of `prototype.py`, saved in `pi05_l10_{50,200}/results.json`; no SR claim follows.

The reference and candidates all use the same early/stale regime, 16-neighbor kernel and library chunks. This isolates key geometry; it does not replay continuity, insurance, guard, or correction. The real served size-50 HIT chunks have RMS10 .3885 on their 2,444 labeled HIT looks; that different subset/full-system statistic is contextual, not the controlled baseline.

The symmetric spatial encoder retains a 14×14 grid per camera before learned PCA. The asymmetric linear encoder predicts the cell's existing PCA code from both images and free state, then searches exact offline teacher keys. Three ridge strengths were compared on B-dev only. Nearest-neighbor agreement is only .31–.33 for the better small-library prototypes, so mean error alone is not reassuring. A small spatial CNN and action-aware objective are the next prototypes. No external pretrained weights are proposed for the primary design.

Data: usable π0.5 L10 image+shadow recordings exist (3,198 looks/100 unseen episodes at size 50; 2,289/73 after the size-200 exclusion). R13 shadows lack images. GR00T's original library HDF5s are present but lack RGB; September traces are different trajectories and must not become training data. Details and counts are in `DATA.md`.

Open issues recorded rather than sent as questions: (1) faithful GR00T frame recovery needs the original simulator/build mapping and row-by-row verification; (2) spatial recordings have policy labels only on MISS; (3) H100 latency must be measured by the coordinator on the eventual π0.5 deployment host; local work is CPU and capped local-GPU prototypes only.

Reproduction prefix (repo root):

```bash
taskset -c 10-21,54-65 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= .venv/bin/python exp/offline_search/rounds/r15/qk/astra/prototype.py --size 50
```

Preparation commands use the same prefix with `prepare.py train --size 50`, `train --size 200`, and `dev`. All writes are under this directory or `/home/weiland/trace_runs/offline_search_store/derived/r15_qk/`. No git, server, chain, remote operation, or test-A root/queries was used.
