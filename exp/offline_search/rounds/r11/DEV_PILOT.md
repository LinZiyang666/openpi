# R11 dev scoring — 20 complete arms


## groot_l10_50

knob off: 0.760 @ 0.203

| target | R | P | D | E | AE | Dis | P+tail | R2 |
|---|---|---|---|---|---|---|---|---|
| 0.32 | 0.820 @ 0.319 (+8/-5) | 0.880 @ 0.314 (+10/-4) | 0.760 @ 0.265 (+8/-8) | 0.820 @ 0.300 (+10/-7) |  |  |  |  |

## groot_spatial_50

knob off: 0.880 @ 0.128

| target | R | P | D | E | AE | Dis | P+tail | R2 |
|---|---|---|---|---|---|---|---|---|
| 0.32 | 0.840 @ 0.324 (+3/-5) | 0.860 @ 0.321 (+4/-5) | 0.840 @ 0.192 (+2/-4) | 0.840 @ 0.294 (+3/-5) |  |  |  |  |

## pi05_l10_50

knob off: 0.900 @ 0.163

| target | R | P | D | E | AE | Dis | P+tail | R2 |
|---|---|---|---|---|---|---|---|---|
| 0.32 | 0.880 @ 0.315 (+1/-2) | 0.880 @ 0.312 (+5/-6) | 0.840 @ 0.213 (+1/-4) | 0.920 @ 0.278 (+5/-4) |  |  |  |  |

## pi05_spatial_50

knob off: 0.820 @ 0.164

| target | R | P | D | E | AE | Dis | P+tail | R2 |
|---|---|---|---|---|---|---|---|---|
| 0.32 | 0.960 @ 0.305 (+7/-0) | 0.960 @ 0.309 (+7/-0) | 0.940 @ 0.216 (+6/-0) | 1.000 @ 0.286 (+9/-0) |  |  |  |  |

## Pooled paired comparisons at matched (cell-size, target)

| comparison | pairs | +b / -c | pp | p |
|---|---|---|---|---|
| P − R | 200 | +15/-11 | +2.00 | 0.56 |
| D − R | 200 | +12/-18 | -3.00 | 0.36 |
| E − R | 200 | +17/-13 | +2.00 | 0.58 |
| E − D | 200 | +21/-11 | +5.00 | 0.11 |
| R − off | 200 | +19/-12 | +3.50 | 0.28 |
| P − off | 200 | +26/-15 | +5.50 | 0.12 |
| D − off | 200 | +17/-16 | +0.50 | 1 |
| E − off | 200 | +27/-16 | +5.50 | 0.13 |

## Realized IR vs target / library prediction

| method | arms | mean(IR − target) | max abs(IR − target) | mean(IR − pred) | within ±.02 of target |
|---|---|---|---|---|---|
| R | 4 | -0.0043 | 0.0152 | -0.0042 | 4/4 |
| P | 4 | -0.0059 | 0.0105 | -0.0059 | 4/4 |
| D | 4 | -0.0986 | 0.1281 | -0.0986 | 0/4 |
| E | 4 | -0.0306 | 0.0420 | -0.0306 | 0/4 |

## Efficiency: (SR − SR_off) / (IR − IR_off), per method, pooled over arms (sum of gains / sum of spend)

- R: 4 arms, ΔSR sum +0.140 over ΔIR sum 0.605 → +0.231 SR per unit IR
- P: 4 arms, ΔSR sum +0.220 over ΔIR sum 0.598 → +0.368 SR per unit IR
- D: 4 arms, ΔSR sum +0.020 over ΔIR sum 0.228 → +0.088 SR per unit IR
- E: 4 arms, ΔSR sum +0.220 over ΔIR sum 0.500 → +0.440 SR per unit IR

## Coordinator note (2026-10-02 23:5x CDT)
- Non-test B-pool inits outside each size-50 subset, 50 episodes per arm: SR differences here are within noise (±7 pp per arm). This pilot was declared a calibration / lifecycle check only; no setting was changed.
- **IR calibration (main purpose):** random and periodic land on target (mean −.004 / −.006, 8/8 within ±.02). The offline-calibrated state thresholds spend less than targeted in closed loop: distance −.099 on average (realized .19–.27 for target .32), error-hybrid −.031.
- Diagnosis (π0.5 L10 distance arm, server decision logs): the knob fires exactly when score > frozen threshold. Closed-loop query distances are smaller than the whole-episode-out calibration distances (median 9.4 vs threshold 12.46; 19% vs 48% of looks above threshold), because closed-loop states follow the cached demos and calibration used a 4/5-library donor bank. The half-random floor of the error hybrid limits the miss.
- Consequence for the test sweep: compare methods on the realized-IR frontier, not by target label.
