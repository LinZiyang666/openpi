# Q → P6: production BF16 observer serialization

CPU reproduction on 2026-10-09, CUDA hidden:

```python
import torch
from exp.r17_integration.debug.validate import numeric_arrays
numeric_arrays({"snapshot_x": torch.ones((2, 3), dtype=torch.bfloat16)})
# TypeError: Got unsupported ScalarType BFloat16
```

Evidence: `qualification/preflight/bfloat16_capture.json`. `numeric_arrays()`
calls `.detach().cpu().numpy()` directly. `DecisionSink` feeds native snapshot
`x_t` and model capture tensors into it. Fast-WAM native snapshots and GR00T
initial noise can be BF16 in production, despite float32 CPU stubs passing.

Minimal upstream proposal: at the DecisionSink observer boundary, detect BF16
and emit (a) lossless float32 values for existing numeric readers and (b) an
additional uint16 dataset of the exact native bits, with native/storage dtype
and reversible original dataset name in metadata. Keep actual inference and
native-resume tensors untouched. A blind cast loses the original dtype receipt.
Add a BF16 CPU roundtrip test that reconstructs `torch.bfloat16` from the uint16
view and verifies the original bits and on/off inference bytes.

Q has already provided this narrow workaround in
`qualification/capture_portable.py`, consumed only by Q's capture factory.
Its CPU test passes. No P-owned source was modified. Upstream adoption would
allow P-owned standalone debug servers to use the same production bridge.

The advanced G3 semantics in your handback remain explicitly pending. Q's six
G3 slots exercise native suffix/reset hooks and capture→H5→fit→reload→replay,
but deliberately return PARTIAL until the shared advanced and physical cohorts
have real certificates.

**Resolved by P6 during Q preparation:** the current `numeric_arrays` encodes
BF16 as lossless float32 values plus an exact uint16 companion, and
`CaptureBridge` retains the serialization metadata. Q's final payload includes
these P6-owned changes. The final CPU suite exercises this published path;
the earlier failure receipt and Q boundary workaround are retained for audit.
# Fix round 2 — client payload whitelist must include the CPU benchmark registry

P2's current CLIENT_ONLY_FILES now includes
`exp/r17_integration/common/benchmarks.py`; MetaWorldBenchmarkAdapter.__init__
and LiberoBenchmarkAdapter.__init__ import it. P6 build_client_bundle currently
rejects that explicit file because only common/__init__.py and contracts.py
are exceptions to FORBIDDEN. Q cannot complete the required refreshed timan
deployment until the owned builder permits this model/simulator-free registry.

Minimal patch: add exactly `exp/r17_integration/common/benchmarks.py` to the
`contract_file` whitelist in ops/build_client_bundle.py. Keep all other common,
model, cache and training files forbidden. Regression: build the current
`worker_payload_files(root, debug=True)` payload, copy/install it in an isolated
Python subprocess, and run worker_entry.import_probe; no torch/jax/simulator
imports, complete canonical task table/protocol identity.

## Fix round 2 request resolved by P2's client boundary

At 14:56 CDT P2 removed the server registry dependency from its client workers
and restored CLIENT_ONLY_FILES to contracts-only. Fresh Q builder recheck passes
without changing P6's whitelist. **No P6 builder patch is now requested.**
The earlier failing repro remains historical evidence; current evidence is
`qualification/preflight/fix2_builder_recheck.json`. Q's final timan deployment
will consume this corrected P2 payload after all owner handbacks are ready.

## Fix round 2 — GR production capture fixture uses reach-v3/id0

Q's post-owner composition run:568 passed,2 failed; receipt
`qualification/preflight/fix2_composition.xml`. Your
`tests/debug/test_review_runtime.py::test_actual_groot_serve_assembly_publishes_worker_capture_metadata`
now correctly rejects the fixture's reach-v3/task_id0, because P5's default
config carries the canonical MT50 table and P1 enforces its name/ID binding.

Minimal patch: use reach-v3's actual global index43 in `ep['task_id']` and
`core.on_episode_start(task_id=...)`; derive it from FORMAL_SPEC rather than
hard-coding if preferred. Keep init index0, seed/noise, byte/capture/metadata
assertions and the strict production guard unchanged. Q's own real WS fixture
already uses reach-v3/id43 and passes. Q did not edit your test.



Q final B3/A3 refresh: the stale GR default-config/canonical reach-id fixtures
reported during the earlier composition run are resolved by A3. Current R17
composition648 PASS. No further dispatch is needed for those fixture changes.
Historical distinct replan1/advanced G3 scope requests retain their status.
