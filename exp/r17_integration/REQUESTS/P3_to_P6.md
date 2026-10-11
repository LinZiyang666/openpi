# P3 → P6: Fix round 1 B9/B10 qualification pricing

Fast-WAM production B>1 is disabled until GPU batch-composition numerical
parity is qualified. Ordinary stage2 capture no longer computes CP2 means;
CP2 is an explicit capture_cp2=True model hook, still implemented_unverified.
Captured flow/snapshot rows incur D2H synchronization and must be excluded
from pricing. Please have qualification/model_runtime use separate capture-off
pricing runs for EVERY arm, and preserve capture_enabled/pricing_eligible in
analysis. P3 provides a synchronized capture-off phase measurement command
models.fastwam.measure with separate VAE/patchify/text_context/prefill/head.

Please migrate any P3-test links to test_fastwam_serving.py and
test_fastwam_batcher.py (P4 requested unique module names for aggregate pytest).
All P3 tests now use qualified fixture imports; native/loader test files are
also renamed test_fastwam_native.py / test_fastwam_loader_wire.py to avoid
module name collisions. Capture NPZ/H5 bridge regression is in the native file.

## Fix round 2 — B2 N1 qualification assets and N2/N3 seams

Please fix owned ops qualification inputs: Fast-WAM LIBERO needs two REAL256 RGB
views; 480 LIBERO or a fabricated wrist is invalid. Keep production B>1 disabled
in qualification/capability rows. Reviewer N1's DP probe/count correction is P4/P6.
P3 adds --debug-directory via make_capture_sink_factory(provider,directory),
and closes the per-connection factory on server exit. Formal descriptors now need
benchmark_specs containing P2 manifest/asdict documents for each allowlisted
identity (see P3_to_P2). Please propagate these through qualification descriptors.
The raw-Wan/released-sidecar GPU parity command will be published in P3's handback;
no GPU parity or real-model cost claim is made before that command passes.

Final: P3 handback Fix round2 ready; own105/common composition214 CPU passes.
serve --debug-directory now closes actual per-connection K2/K1 P6 bindings in
CPU regressions. README/HANDBACK contain the executable loader_parity command:
requires local released Diffusers vae/text_encoder and an independent released
UMT5 tokenizer (--released-tokenizer), real paired256 frames, P2 released spec
and fresh code seals. Canonical serialization shim is explicitly recorded:
published checkpoint has1,620 old block aliases; current native key sets differ.
Config/1,651 canonical header shapes match on meta, but frozen/output parity is
not GPU qualified. Please preserve that status and the pending N1 input/B1 slot
corrections before qualification; P3 has not edited your assets/specs.
# 2026-10-10 Fast-WAM batching formal-launcher seam (owner-authorized GPU measurement)

Update after measurement: the local GPU probe closed normally and released the lock at approximately 11:36 CDT. All B1/2/4/8/16 native and real S07 concurrent-connection sweeps passed; B16 direct throughput 11.07 decisions/s versus B1 1.10 (10.06x). P3-owned local formal shim is `models.fastwam.formal_launch --max-batch N --batch-wait-ms 5 <existing P6 launch args>`; it redirects only the local resident entry through `formal_python.sh` to `formal_batch_server`, preserving the P6 seal/deployment/worker/conductor/process paths. B1 delegates byte-for-byte to the existing launcher. This shim supports local pure-inference Fast-WAM LIBERO; H100 needs the P6 seam and its own qualification.

P4 incoming request now asks for the same default-off launcher seam for `dp_libero`, with `models.dp.batch_runtime.BatchConfig.metadata()` under campaign `dp_batching` and owned `models.dp.formal_batch_server`; keep DP graph self-check/sampler/commit attestation and omit the resident lock only on the explicit batched path. P4 owns all DP implementation and measurement; P3 has not edited DP. Cosmos still lacks StageBatcher and is not a cheap extension.

Important binding detail: `qualification.model_runtime.load` already registers the sealed provider. At the opt-in formal entry preserve that provider and sealed YAML unchanged; call `make_connection_factory(provider, config, storage, adapter_factory=lambda: FastWAMBatchedAdapter(provider.load_adapter(), runtime.core))`. This works with worker bundle hot-switches. Re-registering under a new provider ID would make sealed worker YAML validation fail. P3 CPU formal-resident regression exercises this exact path.

Owner now requests default-off Fast-WAM LIBERO batching and a measured B1/2/4/8/16 sweep on the local 4090 under `/tmp/r17_gpu.lock`; H100 campaigns stay untouched. P3 is measuring and publishing model hooks/receipts under `models/fastwam/`. Your `formal_server.serve` currently wraps every inference in `resident_lock`, which prevents cross-connection batches even if a P3 provider uses S07 queues.

Please add a default-off `--max-batch` (default 1), `--batch-wait-ms` (default 5) to `ops/formal_eval.py`, serialize them in the campaign and attestation, and opt in **only fastwam_libero**. At server startup use P3's `batch_runtime.create_runtime(adapter, max_batch, batch_wait_ms, ledger_path)` and register the provider with `coordinator=runtime.core`; use `runtime.wrap_connection_factory(base_factory)` to export per-decision batch composition. When this explicit runtime is active, omit the resident inference lock; S07 stages own execution. Stop the core in `finally`. B1 should execute the old path unchanged. P3 will retain native vector hooks and measured tolerance policy separately from the test-only `allow_unqualified_batch` flag. S07 one worker per stage is required because native Wan VAE has mutable caches.

Batch composition is recorded in P3's stage ledger (semantic request members/row/stage/batch ID) and in each response's `__hit_meta__.batching`. Fixed max/window/worker counts and policy label must match across control/arms. Existing H100 B1 baseline cannot serve as the matched control for B>1 arms; leave it running and label future batched campaigns separately. DP graph and Cosmos do not inherit this change: their batching math/graph workspaces require their owners' qualification.

Until the shared launcher seam is published, the P3-owned `serve --max-batch` entry is the working opt-in server path. Exact ready-to-run commands and measured safe B limit will be in `models/fastwam/BATCHING.md`; coordinator can use that directly without editing shared files.

# Normal launcher seam resolved by owner-directed P3 work, 2026-10-10

Owner explicitly requested first-class normal formal launcher integration after
accepting B8/5ms. P3 narrowly added flags/campaign/control/receipt hooks in
formal_eval.py, comments in formal_eval.sh, and six dispatch lines at the start
of formal_server.serve for Fast-WAM/DP owned entries. No B1 path/body change.
H100 Fast-WAM smoke: native replay4.95dec/s (3.78x queuedB1), maxdiff0.015625;
four physical suites4/4 successes/no errors, all owned closed, GPU287sec.
Missing P3 parent server manifest caused initial receipt PARTIAL after completion;
owned entry now writes it, CPU snapshot/install regression passes, CPU-only
closed-run recovery PASS with original receipt retained. No extra GPU episodes.

Final CPU413passed+29subtests,62 acceptance passed; legacy89pass/6skips/1EGL
deselected followed by all6 explicit socket skips passing. BATCHING.md has exact
eight commands and projected durations. Q added batch treatment keys to resume;
P3 treatmentv2 also binds worker allowlist/reservations/recycling. No P6 action
required for this seam; historical advanced library/quality requests retain scope.


## Cache study admission — CPU only, 2026-10-10

P3 now exports frozen-provider `configure_library_provider` and attaches the
same content/source-bound callback at native registration. Audited teacher
archive proves every stage body unchanged; all weight/norm/vendor/config/key/
wire source pins are enforced, with exact approved resident generations.
CPU tests and real B500 all-16-YAML preflight are in progress.

Please correct your owned `ops/cache_study/smoke.sh` Fast-WAM worker resource
settings: RUN_CONFIGS hs6 is timan10748, RAM2048 MiB, VRAM768 MiB, N4, EGL0..7.
Current smoke uses shared RAM6144 and default VRAM1024, so it does not match
the accepted baseline admission. Use per-model memory flags rather than shared
6144; Fast stays 2xB16/5ms/MPS/group-commit on H100. No existing campaigns or
frozen runtime have been changed. I will run your --prepare-only CPU path;
GPU smoke remains coordinator-only,32 episodes total (2/group/suite).

Cache CPU status update: model-owned Fast-WAM suite235passed,0skip; legacy
71passed,0skip,1realEGLdeselected. Local real-B500 preflight16YAMLs/640queries
PASS (`models/fastwam/measurements/cache_cpu_preflight_20261010_v4/PREFLIGHT.json`).
Your exact `smoke.sh fastwam --prepare-only` also PASS on H100, physically
sealing weights and all16recipes at
`/data/mw_finetune/eval/mps_fastwam_libero_1010_165002_1cc26649/PREPARED.json`.
No GPU/native model load/server/worker/MPS daemon was started. Source proof
now normalizes only absent/empty Python3.12 AST type_params metadata; exact
source SHA pins remain enforced.

Broader Fast/common/ops run:588passed,2failures,0skip. Both are your
`test_metadata_fit_transport_and_cosmos_admission_are_cpu_only[local|h100]`,
line501: it still expects Cosmos admission to raise after D supplied its
valid compatibility hook. Please update that owned regression to assert the
new correctly sealed path, retaining a changed-math/source negative case.
Fast-WAM and all other tests passed. Raw result:
`tests/models/fastwam/receipts/cache_study_full_cpu.xml`.

Prepare-only used the current smoke helper's6144/default1024 worker reservations;
this CPU proof is not a matched deployable fleet receipt. GPU smoke should use
RUN_CONFIGS Fast hs6 RAM2048/VRAM768/N4/EGL0..7. Exact safe manual command and
32-episode estimate are in P3 handback/FAST cache docs. Long C10 is still marked
pending horizon NI selection by STUDY.json; coordinator/P6 must freeze it
before the formal8000-episode cache run. GPU smoke not executed.

Refresh: the two Cosmos transport tests were already updated in the shared
checkout by their owner while the broad run finished. The targeted current
rerun passed2/2 (`cache_study_cosmos_refresh_cpu.xml`), so no further test edit
is requested. Final Fast suite235passed includes both the cross-Python source
attestation and accepted-control-v-fresh-runtime tests.
A matched CPU prepare is now running as `/data/mw_finetune/eval/cfwp31010`,
RAM2048/VRAM768/N4; all GPU/server/worker start flags remain false.
The model handback is `models/fastwam/CACHE_STUDY.md`; exact normal launcher
GPU smoke32episodes is there (10–20GPUmin planning estimate, unmeasured).

Final matched CPU preparation PASS: `/data/mw_finetune/eval/cfwp31010/PREPARED.json`;
all16YAMLs physically sealed on H100, source/native manifests match owner
attestation, timan107_MEMORY_ADMISSION.admitted=true at RAM2048/VRAM768/N4.
No CUDA model/server/worker/MPS/dispatch started. Coordinator-ready command:
`bash exp/r17_integration/ops/formal_eval.sh --prepared-run /data/mw_finetune/eval/cfwp31010 --run`.
32episodes,10–20GPUmin planning estimate; exact fresh flags in Fast CACHE_STUDY.md.
Please fix helper6144/default1024 before another helper launch; the prepared-run
above already matches hs6. Handback P3 cache section and CPU qualification JSON
are final. Long-horizon NI freeze and actual GPU/SR validation remain yours/
coordinator's; P3 has not altered preregistration or study artifacts.
