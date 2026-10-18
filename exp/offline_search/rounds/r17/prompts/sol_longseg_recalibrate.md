<task>
Owner ruling on your handback (2026-10-18 01:2x): after moving every clock to decision steps, the knob levels' IR and the Q3 stall gate must be recalibrated for the new clock. "There must be no mismatch": every knob level must again hit its nominal IR target, and Q3 must again meet its original calibration claim, under the decision-step clock. Running the reruns with the old periodic gaps / Q3 artifacts and checking IR afterwards is not acceptable.

Queue state (I changed it; do not edit chain scripts): both chains now run only recalibration-independent items first (Cosmos, R16/R15 VLA, native pure references *_b_*, pure+long controls *_p_*), then block on the line `[recal] released` in openpi_ext/scratch/r17_ops/mw/chain.log, which I write after reviewing your work. Everything else (knob descents *_m, LIBERO OFF *_o_l and levels *_l_l, cache-variant studies v01–v14) waits behind that gate. H100 chain ffk-h100ls starts about 04:00, 4090 chain ffk-4090ls about 11:00; the gated part is reached many hours later, but finish as soon as you can.

Do, CPU only, single-reader /data discipline as before:
1. Find exactly how the original periodic gaps were fitted to their nominal IR targets for every level (R17 levels k50..k10 and any ir-named levels) per model × benchmark × step mode × library arm, and how the IR estimate behind that fit is computed. Redo the fit with the decision-step clock (heads only; long-segment tails cost nothing and are not looks), so each level's predicted IR equals its nominal target under the new clock. Update the IR estimator/analysis code so the predicted and the reported IR use the same decision-step definition.
2. Recalibrate Q3 (stall_only + stall_path) on decision-step sequences for every C10 (commit > replan) setting, reproducing the original calibration procedure and its target false-alarm/coverage, instead of reusing 5-control artifacts with a quantum conversion. C5 settings keep their artifacts unchanged (old and new clocks coincide there); say so explicitly per setting.
3. Regenerate the affected fa_ls_* definitions in place with the new gaps/Q3 artifacts (same names, same task UIDs, inits and seeds; only the recalibrated recipe inputs change), rerun your 60 startup/planner checks and targeted tests, and prove for each level that predicted IR under the new clock equals the nominal target.
4. Classify every gated item: which ones truly depend on the recalibration and which do not (e.g. C5-only cache-variant studies). List the independent ones so I can move them ahead of the gate.
5. Keep the old artifacts in place (new files beside them); write RECALIBRATION.md with the per-level old gap, new gap, nominal IR, predicted IR, and the Q3 old/new parameters and achieved calibration metrics.
</task>

<compact_output_contract>
Reply in Chinese, short: what was recalibrated (counts), a compact table of nominal vs predicted IR per model × mode, the Q3 result, which gated items are independent, test results. Do not launch anything.
</compact_output_contract>
