# H100 π0.5 / GR00T formal launcher mapping

Completed 2026-10-12. Both model/server import probes pass on H100 with
`CUDA_VISIBLE_DEVICES=''` and the launcher's runtime environment. No policy
weights were loaded, CUDA was not initialized, and no campaign/server/worker
or file transfer was started. No commit was made.

## Diff summary

Production changes are confined to `host_python()` and `server_env()` in
`ops/formal_eval.py`. Package A's transfer/worker/library regions were preserved.
No production model-side change was required.

| Model on H100 | Interpreter | Extra model source / cache |
| --- | --- | --- |
| `pi05_mw` | `/home/exouser/openpi/.venv/bin/python` | `OPENPI_DATA_HOME=/home/exouser/.cache/openpi` |
| `groot_mw` | `/home/exouser/gr00t_n15_venv/.venv/bin/python` | `/home/exouser/gr00t_n15` appended to `PYTHONPATH` |

`host_python()` also normalizes the H100 GR00T campaign's `vendor` field to
`/home/exouser/gr00t_n15`. The runtime preparation path calls it before writing
the server campaign, so CPU asset discovery/sealing receives the correct
vendor/tokenizer path as well as the correct interpreter.

Both H100 environments use Python3.11; their `PYTHONPATH` excludes the existing
`python_deps312` overlay. The deployed runtime remains first in the normal
client/src/root order. Both use `HF_HOME=/home/exouser/.cache/huggingface`.
GR00T additionally gets `NO_ALBUMENTATIONS_UPDATE=1`. Local model mappings and
the DP/Fast-WAM/Cosmos mappings are unchanged.

CPU mapping tests added at `models/pi05/test_host_runtime.py`.

## H100 probe outputs

Probes used only `tether exec --safe --timeout 50s h100 -- env ...`.
The interpreter and environment arguments were produced locally by the
updated `host_python(campaign)` and `server_env(campaign, runtime_root)`.
Probe-only additions: `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`,
`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

An already-deployed R17 release supplied the runtime modules, without copying
anything to H100:

```text
runtime_root=/data/mw_finetune/qualification/formal_runtime_a7a387ed6d45ce9cf5ece2628f32577ee5a55ab55bbcb2a771e25ebc227b4f07
pi05 PYTHONPATH=<runtime_root>/packages/openpi-client/src:<runtime_root>/src:<runtime_root>
groot PYTHONPATH=<runtime_root>/packages/openpi-client/src:<runtime_root>/src:<runtime_root>:/home/exouser/gr00t_n15
```

π0.5 output (exit0):

```text
executable=/home/exouser/openpi/.venv/bin/python
python=3.11.16
torch=2.7.1+cu126
CUDA_VISIBLE_DEVICES=''
cuda_initialized=false
policy_loaded=false
8/8 imports PASS
```

Passing imports: `exp.r17_integration.ops.formal_server`,
`exp.r17_integration.qualification.model_runtime`,
`exp.r17_integration.models.pi05.{serve,adapter}`,
`openpi.policies.policy_config`, `openpi.training.config`,
`openpi.models.tokenizer`, `openpi.models_pytorch.pi0_pytorch`.

GR00T output (exit0):

```text
executable=/home/exouser/gr00t_n15_venv/.venv/bin/python
python=3.11.16
torch=2.5.1+cu124
transformers=4.51.3
CUDA_VISIBLE_DEVICES=''
cuda_initialized=false
policy_loaded=false
7/7 imports PASS
gr00t.model.policy=/home/exouser/gr00t_n15/gr00t/model/policy.py
```

Passing imports: shared `formal_server` and `qualification.model_runtime`,
`exp.r17_integration.models.groot.{serve,loader}`,
`exp.mw_finetune.groot.{policy,data_config}`, `gr00t.model.policy`.
The GR00T environment retains its installed native stack; it is distinct
from the π0.5 environment. The processor's slow-version fallback warning
did not prevent import.

Full outputs are saved locally:
`/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/pi05_h100_import_probe.log`
and `groot_h100_import_probe.log` in the same directory.

## Tests

**11 passed**: ten host/runtime mapping cases plus the existing Cosmos NVRTC
environment-scope test. Coverage includes both local/H100 interpreters, GR00T
campaign vendor normalization, runtime path ordering, cache paths, removal of
the Python3.12 overlay for these Python3.11 islands, and preserved other-model
paths. Both command templates below passed argument parsing; neither was run.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. .venv/bin/python -m pytest -q exp/r17_integration/models/pi05/test_host_runtime.py exp/r17_integration/ops/tests/test_formal_eval.py::FormalCampaignTests::test_cosmos_nvrtc_paths_are_scoped_to_its_resident
```

## Remaining blockers

- π0.5: no identified CPU import/asset blocker. H100 has
  `/data/ckpt/pi05_metaworld_rlinf/model.safetensors` (7,473,091,464 bytes),
  checkpoint-local `assets/metaworld_mt50/norm_stats.json`, and the local
  PaliGemma tokenizer (4,264,023 bytes). The coordinator's prior weight hash
  verification is unchanged; this package did not rehash the weights.
- GR00T: `/data/mw_finetune/groot/4090_mt50/checkpoint-60000` is still absent
  on H100. The coordinator must copy that complete checkpoint using package
  A's direct rsync helper before preparation/loading. The vendor's local
  Eagle processor/config directory is present and imports successfully.
- The coordinator must deploy the updated runtime through the normal
  launcher/package A path. Probes used an existing release to avoid transfers.
  First H100 GPU model loading, numerical behavior, memory usage and closed-loop
  evaluation remain untested under this package's CPU-only authorization.

## Exact formal command templates

From `/home/weiland/projects/openpi`, after coordinator resource admission
and GR00T weight transfer. Run these sequentially or choose separate vacant
ports within the approved window. One replica uses proxy23160/backend23161;
the driver is reached through `ziyanglin.com:23169`. The policy endpoint is
`149.165.153.233:23160`. Commands below are templates, not executed jobs.
Add `--smoke` for the existing four-episode MT50 smoke (two tasks × two idx).

```bash
exp/r17_integration/ops/formal_eval.sh --model pi05_mw --checkpoint /data/ckpt/pi05_metaworld_rlinf --server-node h100 --worker-node timan107 --replicas 1 --workers 16 --gpu cuda:0 --memory-budget-gib 20 --port 23160 --driver-port 23169 --seconds 86400 --episode-seconds 1800 --retries 2 --run
exp/r17_integration/ops/formal_eval.sh --model groot_mw --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-60000 --server-node h100 --worker-node timan107 --replicas 1 --workers 16 --gpu cuda:0 --memory-budget-gib 20 --port 23160 --driver-port 23169 --seconds 86400 --episode-seconds 1800 --retries 2 --run
```
