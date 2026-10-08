# Deployment during R15 descent chains

The final candidate swap changed **no serving code**. All six QK serving modules and the three shared edits below have the same bytes as the first QK handoff. The shared edits are relative to the saved pre-QK source. They are the only shared files this implementation changes.

| Source path relative to repository | Role | Required SHA256 |
|---|---|---|
| exp/offline_search/closed_loop/plugin.py | server runtime: both serve_pi05 and serve_groot | d5b00e4ef58ec140f0a714bf809b3d0357c4848fd0a7bad74b97560b5910766b |
| exp/offline_search/closed_loop/ops/h100/assets.py | shared dependency planner; also in conservative server import closure | c692a7cc9888b910cd1483c737c9bf582d69b14fc4910411a2ed18e8fb82d7df |
| exp/offline_search/rounds/r04/k4_eval/cost_ledger.py | collector cost accounting; not policy server runtime | 7a367891863f17d15db9cc635a4cff1052fe3297f6f35fabd60b9d5d5f757bc3 |

`plugin.py` is directly imported by both π0.5 and GR00T server wrappers. `assets.py` is shared planner tooling and appears in the conservative server dependency closure; treat it as potentially shared by a server as well. `cost_ledger.py` is loaded by collection tooling and is absent from that server closure. Other descent dependencies are already shared and unchanged by this work.

[descent_safety.json](descent_safety.json) provides source hashes, all 348 conservative server-dependency files and 64 configured B descent arms. No arm enables `--os-light-key`; all recorded descent fit hashes are unchanged. With that flag false, QK-private `_qk_full` absent/false and historical rows without `qk_work`, the specialized AST of all three shared files equals the saved pre-QK AST. Normalization removes only the new parser option, disabled branches and two identity aliases (`audit_view=view`, blind `kd=KEY_DIM`). Default replay independently confirms identical actions and audited decision fields. Timing fields are not claimed byte-identical. This is a source/configuration proof, not an inspection of live remote process memory.

For the bounded code transfer, use `required_code_updates` in [deployment_manifest.json](deployment_manifest.json): six QK-only runtime modules plus these three shared files. Use complete atomic replacements at each target path, preserving working file permissions. Copying source does not reload modules already imported by a running interpreter; future descent starts keep the equivalent default-off path. The six QK runtime modules are not imported with the flag disabled. Keep matching existing descent runtime dependencies in place; the larger `code_files` array is a conservative snapshot/verification inventory, not a requirement to overwrite every unchanged source.

For H100 data, select `payload_files` with `host=h100`: transfer each `staged` source to its `target` relative to `/data/oscl_h100`, checking `served_sha256`. Relocated fitted pickles have different hashes from local source pickles; frozen TorchScript bytes retain their published hashes. New students and fits occupy QK-specific paths. Existing native/library payloads are shared: reuse them when the listed hashes match. Leave descent roots, frozen fits, process code in memory and ongoing execution untouched.

`definitions` lists every source root/contract file/hash for the coordinator; the four `plans` carry prepared arm definitions and B contracts. `worker_B_inputs` and `worker_B_pool_configs` list the existing B-only worker pool and staged configuration hashes; reuse matching worker inputs. No A pool is needed. The six-line queue adds 1,800 arm-episodes, with 600 size-200 baseline outcomes reused only after completion and identity checks.

No deployment, reload, restart, process intervention, remote operation or live environment check was performed. The manifest records local CPU dependency versions and the H100 CPU-export/affinity/thread checks required before QK startup.
