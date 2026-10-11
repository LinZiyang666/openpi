# P4 → P3: aggregate pytest collection names

Read-only file audit found `tests/common/test_serving.py` and
`tests/models/fastwam/test_serving.py` share a module basename without package
__init__ files. Default pytest collection can import one as the other when
coordinator collects the whole R17 test tree. Please use a unique Fast-WAM test
basename (or a package-safe import scheme) in your owned directory.

P4 is renaming its colliding test_serving/test_batcher modules to test_dp_serving
and test_dp_batcher. No P3/common files were edited.

## 2026-10-10 DP-LIBERO dynamic batching

Owner now accepts small batch-composition arithmetic differences with matched
batch configurations for baseline/arms; requests DP DDPM100/DDIM10/DDIM5 and
1/2 replicas × max1/4/8/16/24/32 GPU measurement under /tmp/r17_gpu.lock.
I read your P3_to_P6 seam request and Fast-WAM batch_runtime/batcher/formal entry.
Your runtime is Fast-WAM/16-specific; I will reuse the same model-agnostic stock
S07 BatchingCore with owned DP native hooks/ledger/runtime (no common edits).
DP scheduler posterior noise will stay per-request B1 draws, independent of
batch members/order. DP graph workspace will belong to the one stage3 worker.

Please coordinate the P6 formal launcher seam to also accept dp_libero:
--max-batch(default1), --batch-wait-ms(default5), campaign dp_batching from
models.dp.batch_runtime.BatchConfig.metadata(). For opt-in use owned
models.dp.formal_batch_server and omit resident_lock; max1 retains old entry
unless an explicit DP runtime is selected. DP batched entry will attest its own
CUDA graph self-check plus execution sampler/commit. No H100 or other campaigns.
I will implement the DP side and provide the exact formal launch command; do not
edit DP model files. GPU run waits for your flock, at most20 GPU minutes.
# 2026-10-10 DP batching handback

DP owned implementation/launch shim now exists; common/P3 files untouched.
models/dp/BATCHING.md has actual35/36 matrix points, recommendedone/B8/window5ms
with12 timan108 workers (observedB8), plus18calls/s two/B4 alternative (actualB3).
CPU182/0skip and legacy49/0skip pass. GPU allowance consumed1199.589s total,
including failed lazy graph capture/restarts; all own PIDs closed, memory0MiB.
DP pre-captures all legal graph signatures before stage threads and rejects
live capture; native B32 private RNG states352requests equal B1 exactly.
Max serving committed-action drift0.000345707. SR untested; fresh matched control.
K5/r2/B32 timing remainsunmeasured; final boundedactivation expired before ready.
P6 request updated for seam/RAM/combined launch-override admission. DP local
shim delegates to the current launcher with its batch arguments removed, so
P6's new Fast-WAM-only batch parser does not block it. Default1 delegates unchanged.

## Normal launcher publication consumed — final handback

Consumed your12:07 update in P3_to_P4. Preferred official50 command now uses
ops/formal_eval.sh directly with DP max8/window5ms, one local4090 resident and
12 timan108 workers. Its exact arguments passed the normal parser without
calling main/deploy. P4 native-seam regressions cover max8/32, graph argument
forwarding and one-worker/stage streams. Final full DP185passed/0skip and
legacy49passed/0skip after publication. No shared-file edits by P4.

The CLI/dispatch request is resolved; owned shim is fallback only. DP emits
per-decision semantic membership/config metadata and JSONL. Shared paired-
control enforcement remains Fast-WAM-specific and is a separate DP study gate.
Fresh read-only timan108 admission still exits64/OFFLINE; guarded command stops
before GPU launch. GPU timing remains35/36, budget1199.589s exhausted, no SR claim.

