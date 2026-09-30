| Cell | A mean before → after | B mean before → after | B − A before → after (pp) | max replicate range before → after (pp) |
|---|---|---|---|---|
| π0.5 L10-50 | 0.714 → 0.714 | 0.817 → 0.827 | +10.3 → +11.3 | 3.6 → 2.0 |
| π0.5 L10-500 | 0.827 → 0.827 | 0.873 → 0.879 | +4.6 → +5.1 | 2.6 → 2.4 |
| GR00T L10-50 | 0.609 → 0.611 | 0.719 → 0.725 | +10.9 → +11.4 | 2.6 → 1.2 |
| GR00T L10-500 | 0.830 → 0.830 | 0.867 → 0.872 | +3.7 → +4.2 | 0.4 → 2.0 |
| GR00T Sp-50 | 0.867 → 0.867 | 0.875 → 0.876 | +0.8 → +0.9 | 0.4 → 0.6 |

| Ablation arm | SR before → after | Δ vs reference mean before → after (pp) [after 95% CI] | reference |
|---|---|---|---|
| `r6p2_direct_g_l10_50` | 0.604 → 0.610 | -0.5 → -0.1 [-4.3, +4.1] | A mean (after repair of the references too) |
| `r6p2_direct_g_l10_500` | 0.816 → 0.822 | -1.4 → -0.8 [-4.3, +2.7] | A mean (after repair of the references too) |
| `r6p2_direct_g_sp_500` | 0.928 → 0.954 | -3.6 → -1.0 [-3.2, +1.0] | A mean (after repair of the references too) |
| `r6p2_direct_p_l10_500` | 0.838 → 0.846 | +1.1 → +1.9 [-0.4, +4.1] | A mean (after repair of the references too) |
| `r6p2_direct_p_sp_50` | 0.826 → 0.828 | -1.1 → -0.9 [-3.7, +1.9] | A mean (after repair of the references too) |
| `r6p2_direct_p_sp_500` | 0.946 → 0.968 | -3.0 → -0.8 [-2.3, +0.7] | A mean (after repair of the references too) |
| `r6p2_identity_g_l10_500` | 0.718 → 0.752 | -11.2 → -7.8 [-11.5, -4.1] | A mean (after repair of the references too) |
| `r6p2_identity_g_sp_50` | 0.858 → 0.866 | -0.9 → -0.1 [-3.9, +3.7] | A mean (after repair of the references too) |
| `r6p2_no_progress_p_l10_50` | 0.704 → 0.712 | -11.3 → -11.5 [-14.8, -8.3] | B mean (after repair of the references too) |
| `r6p2_overtime_p_l10_50` | 0.802 → 0.806 | -1.5 → -2.1 [-4.9, +0.7] | B mean (after repair of the references too) |
| `r6p2_terminal_p_l10_50` | 0.826 → 0.828 | +0.9 → +0.1 [-2.8, +2.9] | B mean (after repair of the references too) |

| Frontier arm | SR before → after | NI lower vs pure L10 before → after (pp) | − B mean after (pp) [95% CI] |
|---|---|---|---|
| `r6q2_groot_l10_500_A_dose0p125` @ IR 0.127 | 0.808 → 0.842 | -11.8 → -8.2 | -3.0 [-6.3, +0.2] |
| `r6q2_groot_l10_50_B_dose0p5` @ IR 0.358 | 0.808 → 0.818 | -11.8 → -10.7 | +9.3 [+5.1, +13.5] |
| `r6q2_groot_l10_50_B_dose0p75` @ IR 0.430 | 0.850 → 0.852 | -6.5 → -6.3 | +12.7 [+8.7, +16.9] |
| `r6q2_groot_l10_50_risk_rho0p35` @ IR 0.346 | 0.832 → 0.840 | -8.9 → -8.0 | +11.5 [+7.5, +15.5] |
| `r6q2_groot_l10_50_risk_rho0p45` @ IR 0.453 | 0.838 → 0.868 | -7.9 → -4.6 | +14.3 [+10.5, +18.2] |
| `r6q2_groot_spatial_50_risk_rho0p12` @ IR 0.118 | 0.854 → 0.876 | -13.5 → -11.1 | +0.0 [-2.6, +2.6] |
| `r6q2_pi05_l10_500_B_dose0p125` @ IR 0.205 | 0.842 → 0.892 | -11.7 → -6.2 | +1.3 [-1.2, +3.9] |
| `r6q2_pi05_l10_500_B_dose0p25` @ IR 0.241 | 0.882 → 0.916 | -7.4 → -3.5 | +3.7 [+1.2, +6.3] |
| `r6q2_pi05_l10_500_risk_rho0p18` @ IR 0.179 | 0.856 → 0.868 | -9.7 → -8.4 | -1.1 [-3.9, +1.7] |
| `r6q2_pi05_l10_500_risk_rho0p24` @ IR 0.241 | 0.884 → 0.910 | -7.1 → -4.2 | +3.1 [+0.3, +5.9] |
| `r6q2_pi05_l10_50_B_dose0p75` @ IR 0.423 | 0.874 → 0.886 | -7.7 → -6.4 | +5.9 [+2.3, +9.4] |
| `r6q2_pi05_l10_50_risk_rho0p35` @ IR 0.341 | 0.862 → 0.870 | -9.2 → -8.3 | +4.3 [+0.8, +7.7] |
| `r6q2_pi05_spatial_500_A_dose0p03125` @ IR 0.093 | 0.946 → 0.976 | -7.2 → -3.6 | -0.6 [-2.0, +0.7] |
| `r6q2_pi05_spatial_50_B_dose0p25` @ IR 0.227 | 0.922 → 0.934 | -9.8 → -8.5 | +2.4 [-0.1, +4.8] |
| `r6q2_pi05_spatial_50_risk_rho0p35` @ IR 0.350 | 0.934 → 0.952 | -8.4 → -6.4 | +4.2 [+1.6, +6.7] |
