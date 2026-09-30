## Arms

| cell | cfg | SR | owner IR | target ρ | IR − ρ | within ±.02 |
|---|---|---|---|---|---|---|
| pi05_l10_50 | R30 | 0.866 | 0.295 | 0.300 | -0.005 | yes |
| pi05_l10_50 | U30 | 0.878 | 0.301 | 0.300 | +0.001 | yes |
| pi05_l10_50 | C30 | 0.868 | 0.301 | 0.300 | +0.001 | yes |
| pi05_l10_50 | C45 | 0.894 | 0.458 | 0.450 | +0.008 | yes |
| pi05_spatial_50 | R30 | 0.944 | 0.278 | 0.300 | -0.022 | **no** |
| pi05_spatial_50 | U30 | 0.940 | 0.292 | 0.300 | -0.008 | yes |
| pi05_spatial_50 | C30 | 0.938 | 0.288 | 0.300 | -0.012 | yes |
| pi05_spatial_50 | C45 | 0.966 | 0.470 | 0.450 | +0.020 | **no** |
| groot_l10_50 | R30 | 0.784 | 0.303 | 0.300 | +0.003 | yes |
| groot_l10_50 | U30 | 0.770 | 0.301 | 0.300 | +0.001 | yes |
| groot_l10_50 | C30 | 0.806 | 0.310 | 0.300 | +0.010 | yes |
| groot_l10_50 | Cmax | 0.822 | 0.450 | 0.441 | +0.009 | yes |
| groot_spatial_50 | R30 | 0.912 | 0.295 | 0.300 | -0.005 | yes |
| groot_spatial_50 | U30 | 0.928 | 0.298 | 0.300 | -0.002 | yes |
| groot_spatial_50 | C30 | 0.940 | 0.298 | 0.300 | -0.002 | yes |
| groot_spatial_50 | C45 | 0.936 | 0.454 | 0.450 | +0.004 | yes |
| pi05_l10_500 | R18 | 0.884 | 0.185 | 0.180 | +0.005 | yes |
| pi05_l10_500 | U18 | 0.874 | 0.179 | 0.180 | -0.001 | yes |
| pi05_l10_500 | C18 | 0.902 | 0.191 | 0.180 | +0.011 | yes |
| pi05_spatial_500 | R18 | 0.966 | 0.175 | 0.180 | -0.005 | yes |
| pi05_spatial_500 | U18 | 0.968 | 0.179 | 0.180 | -0.001 | yes |
| pi05_spatial_500 | C18 | 0.986 | 0.162 | 0.180 | -0.018 | yes |
| groot_l10_500 | R18 | 0.852 | 0.180 | 0.180 | +0.000 | yes |
| groot_l10_500 | U18 | 0.848 | 0.180 | 0.180 | +0.000 | yes |
| groot_l10_500 | C18 | 0.866 | 0.184 | 0.180 | +0.004 | yes |
| groot_spatial_500 | R18 | 0.940 | 0.179 | 0.180 | -0.001 | yes |
| groot_spatial_500 | U18 | 0.954 | 0.181 | 0.180 | +0.001 | yes |
| groot_spatial_500 | C18 | 0.934 | 0.184 | 0.180 | +0.004 | yes |

## 1. Placement: R.30 − U.30 (sparse cells)

| cell | R30 SR | U30 SR | Δ pp [95%] | ΔIR |
|---|---|---|---|---|
| pi05_l10_50 | 0.866 | 0.878 | -1.2 [-4.4, +2.0] | -0.006 |
| pi05_spatial_50 | 0.944 | 0.940 | +0.4 [-2.0, +2.8] | -0.014 |
| groot_l10_50 | 0.784 | 0.770 | +1.4 [-3.0, +5.8] | +0.002 |
| groot_spatial_50 | 0.912 | 0.928 | -1.6 [-4.4, +1.2] | -0.003 |

Pooled R30 − U30 = -0.2 pp [-1.9, +1.4]; all |ΔIR| ≤ .01: False. Verdict (§4): inconclusive (IR not matched within .01: frontier comparison only).

## 2. Stall component: C.30 vs R.30, B and B-without-no-progress (sparse cells)

| cell | C30 SR | R30 SR | C30 − R30 pp [95%] | B (3-run) | B−noNP | ½ of B's NP benefit kept? | C30 − B−noNP pp [95%] |
|---|---|---|---|---|---|---|---|
| pi05_l10_50 | 0.868 | 0.866 | +0.2 [-3.4, +3.8] | 0.827 | 0.712 | yes | +15.6 [+11.2, +19.8] |
| pi05_spatial_50 | 0.938 | 0.944 | -0.6 [-3.2, +2.0] | 0.910 | 0.858 | yes | +8.0 [+4.8, +11.2] |
| groot_l10_50 | 0.806 | 0.784 | +2.2 [-1.8, +6.2] | 0.725 | 0.620 | yes | +18.6 [+14.0, +23.2] |
| groot_spatial_50 | 0.940 | 0.912 | +2.8 [+0.2, +5.6] | 0.876 | 0.906 | n/a (NP harmful here) | +3.4 [+0.4, +6.4] |

## 3. Reach: C.45 (Cmax on GR00T L10-50) vs pure inference, 2 pp NI (conservative CP bound)

| cell | C SR @ IR | L10 SR | NI lower vs L10 (pp) | L5 SR | NI lower vs L5 (pp, mean of refs) |
|---|---|---|---|---|---|
| pi05_l10_50 | 0.894 @ 0.458 | 0.904 | -5.81 (not shown) | 0.854 | -1.87 |
| pi05_spatial_50 | 0.966 @ 0.470 | 0.986 | -4.82 (not shown) | 0.993 | -5.56 |
| groot_l10_50 | 0.822 @ 0.450 | 0.866 | -9.38 (not shown) | DUAL ref (outside os_closed_loop) | see frontier_repaired |
| groot_spatial_50 | 0.936 @ 0.454 | 0.938 | -4.19 (not shown) | DUAL ref (outside os_closed_loop) | see frontier_repaired |

## 4. Dense libraries: C.18 vs B at B's cost

| cell | C18 SR @ IR | B SR @ IR (3-run) | C18 − B pp [95%] | lower > −2 pp | IR ≤ B |
|---|---|---|---|---|---|
| pi05_l10_500 | 0.902 @ 0.191 | 0.879 @ 0.159 | +2.3 [-0.5, +5.1] | yes | no |
| pi05_spatial_500 | 0.986 @ 0.162 | 0.982 @ 0.119 | +0.4 [-0.9, +1.6] | yes | no |
| groot_l10_500 | 0.866 @ 0.184 | 0.872 @ 0.188 | -0.6 [-3.7, +2.4] | no | yes |
| groot_spatial_500 | 0.934 @ 0.184 | 0.959 @ 0.127 | -2.5 [-4.7, -0.4] | no | no |
