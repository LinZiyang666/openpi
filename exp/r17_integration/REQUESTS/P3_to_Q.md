# P3 → Q: private campaign MPS launcher seam (2026-10-10)

Owner explicitly requests opt-in --mps on formal_eval. I am keeping daemon,
private pipe/log ownership, inspection/accounting, and tests in P3-owned
models/fastwam/ops/mps.py; narrow edits only to formal_eval.py/sh (parser, campaign
field, a leased guard registered before replicas, replica-only environment,
MPS-aware owned VRAM accounting, final receipt). I inspected/snapshotted your
current prestaging/startup-wave code; no formal_resume/prestage/worker edits.

Please include `mps` in formal_resume SEMANTICS: resumed cells must not silently
mix scheduling treatments. Local shim checks both enabled/disabled transitions
before attach meanwhile. Separate non-resumed arms may use a different MPS
setting for SR/counts only after parity, with wall-time cost marked incomparable;
STUDY.md §6.2 requires service-time as well as call/NFE reductions.

H100 currently has four live model contexts and only ~22GiB free, so additional
two-Fast-WAM mixed probes cannot fit. I will neither toggle their environments
nor attach them to MPS. Local DP shared probes do not acquire the exclusive lock.

Final note: helpers are in the owned `fastwam/ops/` subdirectory, so the native
root-Python code seal and existing B8 control ModelSpec stay identical. Their
integrity is covered by the normal full runtime archive. Do not combine --mps
with --runtime-from pointing to a release that predates these helpers; prepare
a fresh content-addressed release. Q's new runtime_from/cwd changes preserved.
The main() shim already rejects resumed on/off mixing; please also enforce mps
in direct formal_resume attach/same_campaign paths. MPS runtime exec strips all
inherited CUDA_MPS_* options and restores only this campaign's pipe/log/100%.
The actual real guard/exec check passed, including removal of an inherited1MiB
memory limit before a16MiB CUDA allocation. No compute-mode write exists.

## Exclusive H100 follow-up —14:24 CDT

Owner stopped the old H100 Fast baseline and authorized this bounded probe.
Recommended new baseline: R2/B16/5ms, private MPS on,48 timan107 workers,
owner-selected hs4 RAM2048/VRAM768/N4/EGL0..7. Measured28.0 native/S07 calls/s
at48 closed-loop clients/50ms, vs24.8 same topology off; drift<=0.015625.
All probe replicas/private daemon closed, H1000contexts/0MiB, Default unchanged.
DP local GPU/workers untouched; read-only48-worker RAM admission passed.

P3 BatchConfig now permits opt-in1..32, CPU/native-row isolation tested16/24/32.
This changes the native code seal: prepare a **fresh** official-50 control;
do not inherit an old B8 runtime/ModelSpec. All six arms then use that control's
frozen runtime with identicalR2/B16/5ms/MPS/W48/RAM2048/VRAM768/N4/EGLlist.
Exact baseline/arm commands, table and receipts are in models/fastwam/BATCHING.md
and P3_to_R.md. No ops, prestaging, worker or resume files edited in this turn.
MPS must be explicit on every arm; runtime-from alone does not turn it on.
