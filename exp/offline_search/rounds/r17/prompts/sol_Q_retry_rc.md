<task>
Small fix to `exp/r17_integration/ops/formal_eval.py`. Campaign `/data/mw_finetune/eval/formal_dp_libero_formal400_20261009_220843_8dca4316` completed 400/400 accepted episodes. Two attempts hit a transient EGL error ("Offscreen framebuffer is not complete, error 0x8cdd", n_steps 0) and were retried successfully by the stock driver. The driver exited with rc=2 and the launcher raised `driver incomplete: 2`, so the receipt is FAIL and no SUMMARY was written until the coordinator ran `aggregate` manually.

Change: when the driver exits with that "completed with retried error attempts" code AND the journal has every planned episode accepted exactly once with no unresolved errors, run aggregate automatically and mark the receipt PASS, with error/retry attempts disclosed (they already appear in SUMMARY). Any other nonzero rc, or a missing or unaccepted episode, stays FAIL.
- First confirm from the stock driver source what rc=2 means.
- Add a CPU test.
- Do not touch running campaigns (Fast-WAM MT50 and DP-MW on H100). sol D and P6 may also edit this file concurrently, so make a minimal, merge-friendly edit.
</task>
<compact_output_contract>Append a short note to HANDBACK_Q.md "## Retried-error completion". Final message: 3 lines.</compact_output_contract>
