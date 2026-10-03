# Preregistered R9 round-5 predictions

Frozen at **2026-10-02T08:59:55.754479+00:00**, before any round-5 arms were emitted. No round-5 closed loop has run. Fit remains 0–19; screens are 20–29 only. These are judgment forecasts, not offline estimates or calibrated intervals.

| Model | Frozen arm | Success forecast | Owner IR forecast | Plausible SR / IR range |
|---|---|---:|---:|---|
| π0.5 | control: frozen guard + half corrector + persistent pace escalation | .91 | .165 | .86–.96 / .14–.20 |
| π0.5 | latch: guard + half corrector + frozen 2b detector/latch; replaces escalation | .89 | .155 | .83–.95 / .12–.20 |
| π0.5 | pace12: same pace entry, capped at 12 fresh policy calls | .91 | .155 | .85–.96 / .13–.19 |
| GR00T | control: frozen guard + half corrector | .88 | .183 | .82–.94 / .15–.22 |
| GR00T | latch: same stack + frozen 2b detector/latch | .88 | .195 | .81–.94 / .16–.24 |
| GR00T | pace12: same stack + pace-entry 12-call takeover | .89 | .205 | .82–.95 / .17–.25 |

Exactly three arms per model, one extra variant family (pace12). No threshold search, earlier-alert sweep, task routing, guard-call tuning, or look-schedule change. Use imported 2b Monitor/LatchedGate and heads; use imported r3c guard/CorrectedCacheJ. The prescribed frozen corrector retains its original task-conditioned fitted heads; **all new takeover parameters and state transitions are shared across tasks**, with no task difficulty input or switch.

Reason for the extra variant: the learned latch alerts at median decision 58 (π0.5) / 62 (GR00T); pace alerts on the leading traces at median 48 / 45. The populations differ, so this is not an exact earlier-alert effect. Keeping pace entry fixed and limiting persistence to 12 fresh calls isolates takeover duration in π0.5, and tests earlier observation-based entry in GR00T. No guard calls are suppressed after the takeover cap.

I do **not** predict a demonstrated latch gain over the leading stack. In the 2b π0.5 +11/−0 comparison, six wins received no takeover at all; measured post-alert paired rescue is only 5/25. GR00T rescue is 8/32, and latch ties random3 in total success at higher cost. Late intervention and overlap with the no-progress guard limit headroom. Transfer of the detector to corrected proposed actions has not been calibrated.

Primary comparison: paired final success and measured aggregate owner IR against the same-model, same-batch control. Also report +/− discordant pairs, exact two-sided McNemar p (exploratory), init-cluster intervals, actual additional calls, call overlap with the guard, first alert and cap exhaustion. Do not call a <=5-point screen gain established. A promising screen retains success within 2 points and reduces IR by >=.01, or improves success by >=3 points for <=.015 IR; these are prioritization thresholds, not statistical proof. Four candidate-versus-control comparisons: report Holm-adjusted p if making significance claims. No promotion from fixed-path replay or these forecasts. Final locked evaluation remains coordinator-owned.
