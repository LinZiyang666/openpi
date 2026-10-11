# B → coordinator: ready for training

All deliverables are in exp/mw_finetune/groot/HANDBACK_B.md and RECIPE.md.
Full 2,500-episode / 204,806-frame MT50 data, exact stats, modality metadata,
base weights and matching pinned GR00T islands are ready on both hosts.
Full CPU data verification and 12 CPU tests pass on both; source RGB equality
was checked locally, and SHA256 protects the H100 transfer.

**Default: full action head (projectors + DiT), frozen VLM/vision, 60k steps,
LR 1e-4, official optimizer/schedule, save at 30k and 60k.**

| GPU | Batch / accumulation | it/s | s/update | Allocated / reserved GiB | 60k projection |
|---|---|---:|---:|---:|---:|
|4090|64 /1|0.7909|1.2643|30.50 /32.63|21.07h|
|H100|128 /1|0.8437|1.1853|44.56 /51.79|19.75h|

These are actual 200-update full-MT50/60k-schedule probes; timing includes data
wait and excludes first 10 updates. Projections exclude saves/validation/queue.
Both standard 4090 checkpoint save/reload and actual 2→4 optimizer resume passed,
returning finite 16x4 physical actions. No B GPU job or server remains running.
4090 batch128 native allocator OOMed on update3; its log is retained. A later
expandable-allocator retry was canceled before acquiring A's formal GPU lane.
The optional retry command is documented, with no claim that it fits.

Each full snapshot is ~16.14 GB. H100 shared tree measured85.31 GB; two snapshots
plus margin need ~34.3 GB, totaling ~119.6 GB against the 120 GB quota. Coordinate C's growth.
No deletion/pruning/commit/push occurred, and no other agent's process was touched.
Transfer staging is retained: local /tmp/r17_groot_data.XPsdy6 and H100
/tmp/r17_groot_http_recovery (~8.8 GB each), plus small code archives.

A's push-back prompt differs from the source rewrite; B preserves source
annotations and passes the canonical serving prompt unchanged. See B_to_A_prompts.
