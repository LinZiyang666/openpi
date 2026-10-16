<task>
Coordinator answers and one live-run constraint.

1. LIVE RUN: I prepared (your filter_failed_knob.stage) and launched (your filter_failed_knob.launch) `ffk1015_dp_libero_reduced_off` at 21:44 on the 4090; it is running. Never rebuild, rewrite or re-stage any ffk1015_* directory that has LAUNCH.json or RECEIPT.json. Add that guard to build.py (refuse, like filter_failed.stage does) before any further build run.
2. Raise rule at the pure level: preserve the source behavior of rounds/r17/knob/test.py exactly (an already-pure task stays pure; report terminal violations). No extra pure retests.
3. π0.5/GR00T full steps: do not build the R16 full-step alternative yet; I am asking the owner. Put the cost (episodes per arm, unfiltered + filtered, dev/off/test/raise estimate) and the exact reason R14 cannot run (the four hash mismatches) in HANDBACK.md.
4. Finish the rest of the original brief and hand back.
</task>
