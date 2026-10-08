# Interim handback: round-1 CPU path ready

Built opt-in `--os-light-key`, frozen CPU TorchScript students, same-cell R10 layers 1–3 refits, and unchanged R11 periodic caps for π0.5 L10-50/200. Artifacts and SHA256 records: `artifacts/pi05_l10_{50,200}/manifest.json`. Encoder/library feature arrays: `/home/weiland/trace_runs/offline_search_store/derived/r15_qk/impl/pi05_l10_{50,200}/library.npz`.

`plugin_replay.json`: PASS for real plugin/CacheOrchestrator CPU replay. Light HIT = zero tower; MISS = exactly one full-policy/tower; blind and policy tail = neither encoder. Duplicate, failed-receipt, reset, single-broadcast tests pass. Updated default teacher actions and audited decision fields equal the saved pre-edit plugin byte for byte.

Generic encoder factory consumes both raw images, normalized/raw state, task/instruction/step, past look images/features, executed actions, last policy output and HIT history. Swap with `build --size SIZE --encoder-config candidate.json`, then refit and re-emit roots. Candidate must bind its exact cell/subset and frozen export. Astra's PILOT.md is pending; these are explicitly round-1 starter artifacts.

Pending: all-row mathematical/LOEO and accepted unseen B-dev replay, pilot emission/baseline equivalence, blocked local/H100 plans, final runtime deployment hash list. H100 needs CPU PyTorch with TorchScript support (local version recorded in final verification), NumPy, existing plugin dependencies, exported student files and fitted pickle payloads. No new GPU work, servers, remote operations, deployment or online runs have occurred.
