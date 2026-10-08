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


---

## DIRECTIVE_2 forecast — 2026-10-03T18:22:46.742338+00:00

This is a **new post-freeze B-dev-calibrated forecast**, before any changed-controller outcome of the revised design. The preceding 6301 bytes remain unchanged (SHA256 `d633ebc419f84b6e78bc4029dfa9e8fb0fc28a1f282535046dcb9677cba1353c`). Earlier latency prices and GR00T cost exclusion are superseded. Both models are in scope at exactly half the full vision compute share; request latency tricks earn no savings.

Selected pilot rule: wrist-first two-witness completion gate at `.1`, with `.05` as the strict diagnostic. Metric/head/table fits are selected-library-only; the structure/threshold selection used post-freeze B dev. Raw savings transport the measured fixed-state camera acceptance to the observed full-controller schedule and charge completion on baseline calls. They are conditional forecasts, not measured IR of a new controller. The measured full-controller SRs in OFFLINE.md are not look-half SR.

I cannot infer an honest measured SR change from these recordings. To make the forecast falsifiable, the central **subjective SR-loss scenario** is −.10 pp for π0.5 and −.25 pp for GR00T at `.1`; use the broad subjective range [−1,+.5] pp, not a confidence interval. For `.05`, the central scenario is −.02 pp with range [−.5,+.25] pp. These are explicit assumptions generated by `d2_evidence.py`, never regressions from action MSE. A lossless outcome remains the goal, and then the matched-SR saving equals the raw saving.

For assumed positive local slope b in pp per unit IR and loss L in pp, matched saving is `S-L/b`. The table reports b=25 and b=10 sensitivity scenarios. Sparse paired knob evidence does not estimate these slopes, and no positive entry is a demonstrated matched-SR result.

| cell / knob | raw IR saved | SR Δ scenario, pp | matched saving @25 pp/IR | matched saving @10 pp/IR | max buyable loss @25, pp |
| --- | --- | --- | --- | --- | --- |
| π0.5 l10-200 / off | 0.003408 | -0.10 | -0.000592 | -0.006592 | 0.0852 |
| π0.5 spatial-200 / off | 0.004772 | -0.10 | +0.000772 | -0.005228 | 0.1193 |
| GR00T l10-400 / off | 0.000854 | -0.25 | -0.009146 | -0.024146 | 0.0213 |
| π0.5 l10-400 / off | 0.003491 | -0.10 | -0.000509 | -0.006509 | 0.0873 |
| GR00T l10-50 / off | 0.000736 | -0.25 | -0.009264 | -0.024264 | 0.0184 |
| π0.5 l10-50 / off | 0.002309 | -0.10 | -0.001691 | -0.007691 | 0.0577 |
| GR00T spatial-50 / off | 0.001512 | -0.25 | -0.008488 | -0.023488 | 0.0378 |
| π0.5 spatial-50 / off | 0.002712 | -0.10 | -0.001288 | -0.007288 | 0.0678 |
| GR00T l10-50 / periodic .32 | 0.000205 | -0.25 | -0.009795 | -0.024795 | 0.0051 |
| π0.5 l10-50 / periodic .32 | 0.001564 | -0.10 | -0.002436 | -0.008436 | 0.0391 |


Strict alternative:

| cell / knob | raw IR saved | SR Δ scenario, pp | matched saving @25 pp/IR | matched saving @10 pp/IR | max buyable loss @25, pp |
| --- | --- | --- | --- | --- | --- |
| π0.5 l10-200 / off | 0.000495 | -0.02 | -0.000305 | -0.001505 | 0.0124 |
| π0.5 spatial-200 / off | 0.000424 | -0.02 | -0.000376 | -0.001576 | 0.0106 |
| GR00T l10-400 / off | 0.000052 | -0.02 | -0.000748 | -0.001948 | 0.0013 |
| π0.5 l10-400 / off | 0.000323 | -0.02 | -0.000477 | -0.001677 | 0.0081 |
| GR00T l10-50 / off | 0.000011 | -0.02 | -0.000789 | -0.001989 | 0.0003 |
| π0.5 l10-50 / off | 0.000239 | -0.02 | -0.000561 | -0.001761 | 0.0060 |
| GR00T spatial-50 / off | 0.000093 | -0.02 | -0.000707 | -0.001907 | 0.0023 |
| π0.5 spatial-50 / off | 0.000249 | -0.02 | -0.000551 | -0.001751 | 0.0062 |
| GR00T l10-50 / periodic .32 | 0.000000 | -0.02 | -0.000800 | -0.002000 | 0.0000 |
| π0.5 l10-50 / periodic .32 | 0.000057 | -0.02 | -0.000743 | -0.001943 | 0.0014 |


GR00T's primary table is conditional on the recorded joint-path representation. The numerical-stability stress filter lowers the L10 camera opportunity further:

| GR00T L10 cell | proposals stressed | top changed under stress | robust half looks | IR saved | missed / extra / grip |
| --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 93 | 13 | 28 | 0.000336 | 2 / 0 / 0 |
| GR00T l10-50 / off | 86 | 5 | 35 | 0.000374 | 1 / 0 / 1 |
| GR00T l10-50 / periodic .32 | 25 | 1 | 1 | 0.000026 | 0 / 0 / 0 |


That filter does not repair the missing original pixels or certify separate-camera policy prefixes. For a real single-camera GR00T deployment, raw saving, SR change and matched-SR total IR remain **unidentified** until the exact numerical path passes B validation. Spatial GR00T is likewise conditional. This is an accuracy qualification, not the superseded claim that GR00T has no half-look compute saving.

The all-half and alternating upper bounds under current baseline calls are:

| cell / knob | all-half ceiling | alternating ceiling |
| --- | --- | --- |
| π0.5 l10-200 / off | 0.032558 | 0.019151 |
| π0.5 spatial-200 / off | 0.036480 | 0.019477 |
| GR00T l10-400 / off | 0.027727 | 0.018623 |
| π0.5 l10-400 / off | 0.033585 | 0.019176 |
| GR00T l10-50 / off | 0.025264 | 0.018593 |
| π0.5 l10-50 / off | 0.029242 | 0.019132 |
| GR00T spatial-50 / off | 0.032772 | 0.018824 |
| π0.5 spatial-50 / off | 0.032171 | 0.019390 |
| GR00T l10-50 / periodic .32 | 0.016171 | 0.016171 |
| π0.5 l10-50 / periodic .32 | 0.017799 | 0.017799 |


My directional expectation is that the stricter method is most plausible for goal (a), with very small raw savings; goal (b) is fragile because even a small SR loss can exceed the buyback budgets above. The `.1` method is a B-pilot candidate, not a claim of success. Do not spend missing guard calls as savings. The existing knob roots overlap on only 9 B pairs per model, so no reliable local buyback curve is available. Pending cells and size 500 get no invented forecast. Freeze any further changes in a new timestamped section before changed-controller or test-A outcomes; never alter the earlier sections.


---

## Source correction — 2026-10-03T18:24:51.655018+00:00

The coordinator's new 13:0x CDT note, discovered during final handback review, retires `r12_devlooks_50S` and `r12_devlooks_200` and requests their `_local` replacements. **Withdraw the Spatial-50 and size-200 rows in the 18:22:46 forecast above.** Their source roots are no longer admissible validation evidence. L10-50, L10-400 and knob rows are unaffected. All preceding bytes are retained for the audit trail, not endorsed as current forecasts.

The candidate structure/thresholds remain frozen; no further selection will use the retired roots. The completed Spatial-50 local replacement will be evaluated with those same settings. Size-200 local data is pending and gets no forecast until available. Any new local-root forecast will be appended with its own timestamp. The original-root screens remain only as explicitly superseded historical artifacts in this directory.


---

## DIRECTIVE_2 valid-root forecast — 2026-10-03T18:28:25.598035+00:00

This is a **new post-freeze B-dev-calibrated forecast**, before any changed-controller outcome of the revised design. The original 6301-byte prefix remains unchanged (SHA256 `d633ebc419f84b6e78bc4029dfa9e8fb0fc28a1f282535046dcb9677cba1353c`), as do every earlier forecast/correction section. This section uses only the coordinator's currently accepted roots, including completed `_local` replacements; it supersedes the 18:22 source-dependent forecast. Retired non-local Spatial-50/200 results are excluded. The gate parameters are unchanged. Pending replacement cells get no new forecast. Earlier latency prices and GR00T cost exclusion are superseded. Both models are in scope at exactly half the full vision compute share; request latency tricks earn no savings.

Selected pilot rule: wrist-first two-witness completion gate at `.1`, with `.05` as the strict diagnostic. Metric/head/table fits are selected-library-only; the structure/threshold selection used post-freeze B dev. Raw savings transport the measured fixed-state camera acceptance to the observed full-controller schedule and charge completion on baseline calls. They are conditional forecasts, not measured IR of a new controller. The measured full-controller SRs in OFFLINE.md are not look-half SR.

I cannot infer an honest measured SR change from these recordings. To make the forecast falsifiable, the central **subjective SR-loss scenario** is −.10 pp for π0.5 and −.25 pp for GR00T at `.1`; use the broad subjective range [−1,+.5] pp, not a confidence interval. For `.05`, the central scenario is −.02 pp with range [−.5,+.25] pp. These are explicit assumptions generated by `d2_evidence.py`, never regressions from action MSE. A lossless outcome remains the goal, and then the matched-SR saving equals the raw saving.

For assumed positive local slope b in pp per unit IR and loss L in pp, matched saving is `S-L/b`. The table reports b=25 and b=10 sensitivity scenarios. Sparse paired knob evidence does not estimate these slopes, and no positive entry is a demonstrated matched-SR result.

| cell / knob | raw IR saved | SR Δ scenario, pp | matched saving @25 pp/IR | matched saving @10 pp/IR | max buyable loss @25, pp |
| --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 0.000854 | -0.25 | -0.009146 | -0.024146 | 0.0213 |
| π0.5 l10-400 / off | 0.003491 | -0.10 | -0.000509 | -0.006509 | 0.0873 |
| GR00T l10-50 / off | 0.000736 | -0.25 | -0.009264 | -0.024264 | 0.0184 |
| π0.5 l10-50 / off | 0.002309 | -0.10 | -0.001691 | -0.007691 | 0.0577 |
| GR00T spatial-50 / off | 0.001867 | -0.25 | -0.008133 | -0.023133 | 0.0467 |
| π0.5 spatial-50 / off | 0.002543 | -0.10 | -0.001457 | -0.007457 | 0.0636 |
| GR00T l10-50 / periodic .32 | 0.000205 | -0.25 | -0.009795 | -0.024795 | 0.0051 |
| π0.5 l10-50 / periodic .32 | 0.001564 | -0.10 | -0.002436 | -0.008436 | 0.0391 |


Strict alternative:

| cell / knob | raw IR saved | SR Δ scenario, pp | matched saving @25 pp/IR | matched saving @10 pp/IR | max buyable loss @25, pp |
| --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 0.000052 | -0.02 | -0.000748 | -0.001948 | 0.0013 |
| π0.5 l10-400 / off | 0.000323 | -0.02 | -0.000477 | -0.001677 | 0.0081 |
| GR00T l10-50 / off | 0.000011 | -0.02 | -0.000789 | -0.001989 | 0.0003 |
| π0.5 l10-50 / off | 0.000239 | -0.02 | -0.000561 | -0.001761 | 0.0060 |
| GR00T spatial-50 / off | 0.000119 | -0.02 | -0.000681 | -0.001881 | 0.0030 |
| π0.5 spatial-50 / off | 0.000286 | -0.02 | -0.000514 | -0.001714 | 0.0072 |
| GR00T l10-50 / periodic .32 | 0.000000 | -0.02 | -0.000800 | -0.002000 | 0.0000 |
| π0.5 l10-50 / periodic .32 | 0.000057 | -0.02 | -0.000743 | -0.001943 | 0.0014 |


GR00T's primary table is conditional on the recorded joint-path representation. The numerical-stability stress filter lowers the L10 camera opportunity further:

| GR00T L10 cell | proposals stressed | top changed under stress | robust half looks | IR saved | missed / extra / grip |
| --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 93 | 13 | 28 | 0.000336 | 2 / 0 / 0 |
| GR00T l10-50 / off | 86 | 5 | 35 | 0.000374 | 1 / 0 / 1 |
| GR00T l10-50 / periodic .32 | 25 | 1 | 1 | 0.000026 | 0 / 0 / 0 |


That filter does not repair the missing original pixels or certify separate-camera policy prefixes. For a real single-camera GR00T deployment, raw saving, SR change and matched-SR total IR remain **unidentified** until the exact numerical path passes B validation. Spatial GR00T is likewise conditional. This is an accuracy qualification, not the superseded claim that GR00T has no half-look compute saving.

The all-half and alternating upper bounds under current baseline calls are:

| cell / knob | all-half ceiling | alternating ceiling |
| --- | --- | --- |
| GR00T l10-400 / off | 0.027727 | 0.018623 |
| π0.5 l10-400 / off | 0.033585 | 0.019176 |
| GR00T l10-50 / off | 0.025264 | 0.018593 |
| π0.5 l10-50 / off | 0.029242 | 0.019132 |
| GR00T spatial-50 / off | 0.031414 | 0.018819 |
| π0.5 spatial-50 / off | 0.032803 | 0.019405 |
| GR00T l10-50 / periodic .32 | 0.016171 | 0.016171 |
| π0.5 l10-50 / periodic .32 | 0.017799 | 0.017799 |


My directional expectation is that the stricter method is most plausible for goal (a), with very small raw savings; goal (b) is fragile because even a small SR loss can exceed the buyback budgets above. The `.1` method is a B-pilot candidate, not a claim of success. Do not spend missing guard calls as savings. The existing knob roots overlap on only 9 B pairs per model, so no reliable local buyback curve is available. Pending cells and size 500 get no invented forecast. Freeze any further changes in a new timestamped section before changed-controller or test-A outcomes; never alter the earlier sections.
