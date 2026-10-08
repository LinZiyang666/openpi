# Pre-pilot prediction

Frozen UTC timestamp: **2026-10-03T17:01:37.136881+00:00**. This precedes any closed-loop result of this design. Only B library replay and old, explicitly allowed B-dev baselines were inspected. No changed-controller SR has been measured.

Positive saving means lower total IR. The central SR changes below are **subjective forecasts**, not outputs inferred from action MSE. Every central SR forecast has the broad plausibility interval **[-2, 1] pp**. These are not confidence intervals. The illustrative buyback slope is **25 pp per unit IR**; at **10 pp per unit IR** the buyback costs are larger (`predictions.json`). Existing B dev cannot identify reliable local slopes for every cell.

| cell / size | knob | half/look % | IR saved current path | IR saved stock-full path | SR delta pp | matched saving current @25 pp/IR | matched saving stock @25 pp/IR |
|---|---|---|---|---|---|---|---|
| groot_l10_200 | off | 13.3 | 0.0036 | 0.0036 | -0.3 | -0.0084 | -0.0084 |
| groot_l10_200 | periodic32 | 11.6 | 0.0039 | 0.0039 | -0.3 | -0.0081 | -0.0081 |
| groot_l10_50 | off | 15.3 | 0.0040 | 0.0040 | -0.5 | -0.0160 | -0.0160 |
| groot_l10_50 | periodic32 | 14.4 | 0.0037 | 0.0037 | -0.5 | -0.0163 | -0.0163 |
| groot_l10_500 | off | 12.4 | 0.0041 | 0.0041 | -0.3 | -0.0079 | -0.0079 |
| groot_l10_500 | periodic32 | 10.9 | 0.0036 | 0.0036 | -0.3 | -0.0084 | -0.0084 |
| groot_spatial_200 | off | 21.1 | 0.0071 | 0.0071 | -0.2 | -0.0009 | -0.0009 |
| groot_spatial_200 | periodic32 | 17.8 | 0.0059 | 0.0059 | -0.2 | -0.0021 | -0.0021 |
| groot_spatial_50 | off | 22.8 | 0.0076 | 0.0076 | -0.3 | -0.0044 | -0.0044 |
| groot_spatial_50 | periodic32 | 19.2 | 0.0062 | 0.0062 | -0.3 | -0.0058 | -0.0058 |
| groot_spatial_500 | off | 17.7 | 0.0059 | 0.0059 | -0.2 | -0.0021 | -0.0021 |
| groot_spatial_500 | periodic32 | 15.5 | 0.0052 | 0.0052 | -0.2 | -0.0028 | -0.0028 |
| pi05_l10_200 | off | 26.2 | 0.0053 | 0.0104 | -0.2 | -0.0027 | 0.0024 |
| pi05_l10_200 | periodic32 | 21.5 | 0.0035 | 0.0089 | -0.2 | -0.0045 | 0.0009 |
| pi05_l10_50 | off | 23.1 | 0.0039 | 0.0091 | -0.3 | -0.0081 | -0.0029 |
| pi05_l10_50 | periodic32 | 20.5 | 0.0024 | 0.0079 | -0.3 | -0.0096 | -0.0041 |
| pi05_l10_500 | off | 24.5 | 0.0050 | 0.0101 | -0.2 | -0.0030 | 0.0021 |
| pi05_l10_500 | periodic32 | 21.4 | 0.0035 | 0.0088 | -0.2 | -0.0045 | 0.0008 |
| pi05_spatial_200 | off | 29.2 | 0.0075 | 0.0124 | -0.1 | 0.0035 | 0.0084 |
| pi05_spatial_200 | periodic32 | 24.5 | 0.0050 | 0.0101 | -0.1 | 0.0010 | 0.0061 |
| pi05_spatial_50 | off | 21.3 | 0.0032 | 0.0086 | -0.2 | -0.0048 | 0.0006 |
| pi05_spatial_50 | periodic32 | 20.4 | 0.0029 | 0.0084 | -0.2 | -0.0051 | 0.0004 |
| pi05_spatial_500 | off | 29.7 | 0.0075 | 0.0122 | -0.1 | 0.0035 | 0.0082 |
| pi05_spatial_500 | periodic32 | 26.0 | 0.0057 | 0.0107 | -0.1 | 0.0017 | 0.0067 |


GR00T rows are **conditional arithmetic only**, assuming half/completion each cost 0.074; neither price nor an actual one-camera path is established. Do not call those deployable savings. Pi05 current-path columns use measured request-full/wrist/completion prices; stock-full columns require a separately measured dispatch improvement applied fairly to controls.

Exposure is the B-library safety-atlas half fraction, attenuated by valid-tail B-dev eligibility relative to its ideal alternating limit. Rows without matching B-dev observations use the explicit scenario constants in `forecast_assumptions.json`, not invented closed-loop measurements. Baseline half-MISS probabilities are transported only as a forecast; guard feedback can invalidate them. The stored SR loss priors do not vary with knob setting because no evidence supports a more precise interaction model.

My directional prediction: this is most likely a small raw-IR saving on pi05; **matched-SR savings on the existing request path are uncertain and several central scenarios lose**. Recovering stock-speed full dispatch improves the economics. GR00T is not ready for a cost claim. Small offline MSE changes do not justify predicting lossless SR with confidence.

The exact break-even condition is `IR_saved_raw > SR_loss / local_buyback_slope`. If the slope is zero or unavailable, goal (b) is not established. Goal (a) requires a paired closed-loop noninferiority result with uncertainty, not rounded equal SR.

Maximum pi05 savings without calls, at ideal look rate 0.5: all half 0.0437; this design's alternating limit 0.0218 with stock-speed full looks, or 0.0184 with current request-full pricing. All-policy/all-half theoretical saving is 0.0186, but the primary deliberately chooses full on known periodic calls. Layer-4 calls therefore shrink the attainable cache-only opportunity.

Per-cell optimistic ceilings below retain baseline call rates and stock-speed full looks; they ignore warm-up, safety rejection and guard changes. “Primary upper” is `(full-half)*min(v/2, v-m)`, assuming every half avoids a policy call. B-dev rates are used where a matching baseline exists; the other rows are explicit scenarios as above. GR00T remains conditional on hypothetical prices.

| cell / size | no-call all-half | no-call alternating | all-half with guards | all-half periodic32 | primary upper off | primary upper periodic32 |
|---|---|---|---|---|---|---|
| groot_l10_50 | 0.0370 | 0.0185 | 0.0260 | 0.0164 | 0.0186 | 0.0164 |
| groot_l10_200 | 0.0370 | 0.0185 | 0.0278 | 0.0156 | 0.0187 | 0.0156 |
| groot_l10_500 | 0.0370 | 0.0185 | 0.0296 | 0.0156 | 0.0185 | 0.0156 |
| groot_spatial_50 | 0.0370 | 0.0185 | 0.0330 | 0.0163 | 0.0188 | 0.0163 |
| groot_spatial_200 | 0.0370 | 0.0185 | 0.0342 | 0.0156 | 0.0189 | 0.0156 |
| groot_spatial_500 | 0.0370 | 0.0185 | 0.0296 | 0.0156 | 0.0185 | 0.0156 |
| pi05_l10_50 | 0.0437 | 0.0218 | 0.0389 | 0.0301 | 0.0220 | 0.0198 |
| pi05_l10_200 | 0.0437 | 0.0218 | 0.0397 | 0.0293 | 0.0220 | 0.0186 |
| pi05_l10_500 | 0.0437 | 0.0218 | 0.0387 | 0.0293 | 0.0219 | 0.0186 |
| pi05_spatial_50 | 0.0437 | 0.0218 | 0.0394 | 0.0306 | 0.0222 | 0.0204 |
| pi05_spatial_200 | 0.0437 | 0.0218 | 0.0431 | 0.0293 | 0.0225 | 0.0186 |
| pi05_spatial_500 | 0.0437 | 0.0218 | 0.0387 | 0.0293 | 0.0219 | 0.0186 |


No thresholds or cell choices may be revised from A outcomes. Fine settings and go/no-go decisions belong to the B pilot in EXPERIMENTS.md.
