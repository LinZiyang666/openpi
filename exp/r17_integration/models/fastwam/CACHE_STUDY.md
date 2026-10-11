Fast-WAM cache study admission (2026-10-10)

The published B500 libraries for `libero_spatial` and `libero_10` now enter the
normal R17 serving cache through the model-owned library validator. The CPU
sealer and live provider registration use the same callback. It returns a frozen
provider replacement with the original ID/ModelSpec; neither teacher binding
nor resident identity is rewritten.

| P6 request | Implementation / evidence | Status |
|---|---|---|
| 1: resident B16 hits/tails and YAML reuse | Existing full.yaml startup, shared factory/BatchRuntime; native socket test switches all8 YAMLs over16 persistent connections,192 decisions, one shared model/core | CPU PASS; startup recipe rejection retained per P6 clarification |
| 2: content-bound teacher compatibility | provider.configure_library_provider; ops/library_compatibility and library_evidence pin every weight, norm/config/tokenizer/vendor file and exact audited adapter generations | CPU PASS locally and H100 prepare-only |
| 3: native accounting and smoke | Native tests: HIT `(1,0,0)`/NFE0; tail `(0,0,0)`/NFE0; confirmed stall `(1,1,10)`/NFE10, followed by a fresh-anchor cooldown HIT | CPU PASS; GPU smoke coordinator-only |

Teacher checkpoint `9f8304f7d001eec42e5c1774b424020f98049fc8e8e62449d4ce09686f807030`
and accepted hs6 control `fa3f174e2301a675c5f3afab04114af5c359e49f66fa23c3601ae2877a62d929`
have identical physical weights, normalization, vendor code, key builder,
wire/output transform, loader, manifest and protocol sources. All staged native
bodies are AST-identical; only constructor/batch admission differ. The exact
complete staged file hashes are pinned too. Newly added batching/launch files
are exact approved hashes, not a wildcard code exemption.

Fresh native content identity:
`f436df1f43a096f67b668e7154865b22543ca6fd6c7ef356bca7a0bbca891743`.
Compatibility source receipt:
`f0cba12ff45e85923da7bd119bab21b7ca325c504c2a5c7ae5e74ddcfbfe6565`.
The key is **49152D** (3072 channels ×16 pooled bins), plus8 raw state values;
64D is the downstream fitted projection. Identical width alone never admits a
library. Unknown source/weight generations, changed normalization, preprocessing,
pooling/grid, output transform, teacher policy digest, non-K10 study bindings
or changed native benchmark semantics fail closed.

Source AST serialization preserves all semantic fields and normalizes only
absent/empty Python3.12 `type_params`. The teacher's archive SHA and staged
SHA are checked in the CPU regression. Every executing root adapter file must
also match the reviewed source pins. A future native edit needs a new reviewed
attestation and fresh CPU sealing; it cannot reuse this evidence by relabelling.

The ordinary disabled recipe does no library I/O, adds no inference stages and
preserves normalized action bytes, manifest and serving metadata. B1 stays
default-off. Cache study uses hs6 topology: H100 R2/B16/5ms/private MPS,
group-commit driver, timan10748 with2048MiB RAM/768MiB VRAM/N4/EGL0..7.
Control/arms keep these configuration fields identical; actual batch membership
remains recorded for each decision. Connection wrappers are replaced on bundle
selection, while resident weights and S07 core stay resident.

CPU checks executed:

- Full owned Fast-WAM suite:235passed,0skip (cache_study_fastwam_cpu.xml).
- Broader Fast/common/ops run:588passed,2 initially stale Cosmos expectations;
  the owner updated those tests and the targeted rerun passed2/2. The357
  common/ops cases therefore passed across the full run and targeted refresh.
- Legacy/default-off stock serving/MW checks:71passed,0skip,1 real EGL test
  deliberately deselected (cache_study_legacy_cpu.xml).
- Real library CPU preflight:16 YAMLs,640 queries covering all10 tasks in each
  suite, both C5/C10, corrector on/off, gate on/off; official-A smoke plan32
  episodes. Receipt: measurements/cache_cpu_preflight_20261010_v4/PREFLIGHT.json.
- P6's exact `smoke.sh fastwam --prepare-only`: PASS, physical H100 CPU seal,
  no server/worker/dispatch/MPS daemon start. Receipt:
  `/data/mw_finetune/eval/mps_fastwam_libero_1010_165002_1cc26649/PREPARED.json`.
  That helper currently uses6144/default1024MiB worker reservations; a separate
  matched2048/768 preparation also PASSED, including timan107 memory admission:
  `/data/mw_finetune/eval/cfwp31010/PREPARED.json`. No server, worker, dispatch
  or MPS daemon was started. Do not launch the mismatched helper receipt.

Local CPU preflight command (one-thread BLAS matches P6's server_env; a different
thread count can change early metric bytes by a few ulps and correctly fails
the artifact fence):

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m \
  exp.r17_integration.models.fastwam.ops.cache_preflight \
  --output exp/r17_integration/models/fastwam/measurements/cache_cpu_preflight_NEW
```

Exact coordinator GPU smoke, **not executed by P3**. Both suites ×8 groups ×2
episodes =32episodes; reserve **10–20 GPU minutes**, an unmeasured planning
estimate including two resident loads and small smoke batches. CPU preparation
and asset transport can take additional wall time. The shell helper needs P6's
resource correction; this normal-launcher command already supplies matched
values and group-commit is selected by cache-study mode:

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --official-50 \
  --cache-study exp/r17_integration/ops/cache_study/yaml/STUDY.json \
  --server-node h100 --worker-node timan107 --workers 48 --replicas 2 \
  --max-batch 16 --batch-wait-ms 5 --mps \
  --worker-memory-mib 2048 --worker-vram-mib 768 --worker-max-episodes 4 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 --memory-budget-gib 70 \
  --port 23420 --driver-port 23439 --seconds 7200 --episode-seconds 1800 \
  --smoke --smoke-episodes 2 --run-name cfwgpu1010 --run
```

Alternatively, the coordinator can launch the already sealed, matched CPU
preparation (its exact32-episode plan and all treatment flags are frozen):

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --prepared-run /data/mw_finetune/eval/cfwp31010 --run
```

GPU cache success/throughput/numerical parity have not been measured. C10 is
currently the preregistered fallback long horizon, pending coordinator/P6's NI
selection freeze. If that selection changes, regenerate/refit/reseal the YAML
study before the formal8000-episode Fast-WAM run. GPU smoke must finish with
correct accounting, batch receipts and unchanged resident generations before
promotion; no running campaign or its frozen runtime was edited.
