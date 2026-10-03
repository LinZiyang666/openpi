# r08_abl results (standard mode, 500 official pairs per arm, h100 servers)

Δ and p are paired exact McNemar against the **same-topology** control of the same cell (B control for
guard arms, A control for CLIP; only-no-progress against both). "old A/B" = mean SR of the three R6
replicates (weilandserver + timan107): cross-topology reference only. Owner IR = .152v+.848m (π0.5),
.148v+.852m (GR00T).

## π0.5 LIBERO-10-50  (old A 0.714 / old B 0.827)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_p_l10_50 | timan107 | 500 | 0.834 | 0.179 |  | +11.0 pp (+76/−21, p=1.72e-08) |
| r8abl_ctrlA_p_l10_50 | timan107 | 500 | 0.724 | 0.076 | -11.0 pp (+21/−76, p=1.72e-08) |  |
| r8abl_onlynp_p_l10_50 | timan107 | 500 | 0.832 | 0.171 | -0.2 pp (+33/−34, p=1) | +10.8 pp (+76/−22, p=3.9e-08) |
| r8abl_clip_p_l10_50 | timan107 | 500 | 0.424 | 0.076 | -41.0 pp (+19/−224, p=1.31e-45) | -30.0 pp (+30/−180, p=2.89e-27) |

## π0.5 LIBERO-10-500  (old A 0.827 / old B 0.879)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_p_l10_500 | timan108 | 500 | 0.894 | 0.155 |  | +6.8 pp (+50/−16, p=3.33e-05) |
| r8abl_ctrlA_p_l10_500 | timan108 | 500 | 0.826 | 0.077 | -6.8 pp (+16/−50, p=3.33e-05) |  |
| r8abl_stuck_p_l10_500 | timan108 | 500 | 0.898 | 0.157 | +0.4 pp (+20/−18, p=0.871) | +7.2 pp (+51/−15, p=1.01e-05) |
| r8abl_terminal_p_l10_500 | timan108 | 500 | 0.890 | 0.156 | -0.4 pp (+20/−22, p=0.878) | +6.4 pp (+53/−21, p=0.000256) |
| r8abl_overtime_p_l10_500 | timan108 | 500 | 0.880 | 0.158 | -1.4 pp (+25/−32, p=0.427) | +5.4 pp (+52/−25, p=0.0028) |
| r8abl_no_progress_p_l10_500 | timan108 | 500 | 0.844 | 0.085 | -5.0 pp (+17/−42, p=0.00155) | +1.8 pp (+24/−15, p=0.2) |
| r8abl_onlynp_p_l10_500 | timan108 | 500 | 0.876 | 0.154 | -1.8 pp (+20/−29, p=0.253) | +5.0 pp (+49/−24, p=0.00463) |
| r8abl_clip_p_l10_500 | timan108 | 500 | 0.644 | 0.076 | -25.0 pp (+15/−140, p=1.33e-26) | -18.2 pp (+30/−121, p=3.72e-14) |

## π0.5 Spatial-50  (old A 0.837 / old B 0.910)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_p_sp_50 | timan107 | 500 | 0.892 | 0.145 |  | +6.4 pp (+38/−6, p=9.43e-07) |
| r8abl_ctrlA_p_sp_50 | timan107 | 500 | 0.828 | 0.078 | -6.4 pp (+6/−38, p=9.43e-07) |  |
| r8abl_onlynp_p_sp_50 | timan107 | 500 | 0.892 | 0.128 | +0.0 pp (+11/−11, p=1) | +6.4 pp (+38/−6, p=9.43e-07) |
| r8abl_clip_p_sp_50 | timan107 | 500 | 0.556 | 0.077 | -33.6 pp (+15/−183, p=6.78e-38) | -27.2 pp (+22/−158, p=1.46e-26) |

## π0.5 Spatial-500  (old A 0.976 / old B 0.982)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_p_sp_500 | timan108 | 500 | 0.978 | 0.119 |  | +0.0 pp (+6/−6, p=1) |
| r8abl_ctrlA_p_sp_500 | timan108 | 500 | 0.978 | 0.078 | +0.0 pp (+6/−6, p=1) |  |
| r8abl_stuck_p_sp_500 | timan108 | 500 | 0.982 | 0.118 | +0.4 pp (+6/−4, p=0.754) | +0.4 pp (+7/−5, p=0.774) |
| r8abl_terminal_p_sp_500 | timan108 | 500 | 0.982 | 0.096 | +0.4 pp (+6/−4, p=0.754) | +0.4 pp (+7/−5, p=0.774) |
| r8abl_overtime_p_sp_500 | timan108 | 500 | 0.980 | 0.118 | +0.2 pp (+4/−3, p=1) | +0.2 pp (+7/−6, p=1) |
| r8abl_no_progress_p_sp_500 | timan108 | 500 | 0.980 | 0.103 | +0.2 pp (+6/−5, p=1) | +0.2 pp (+5/−4, p=1) |
| r8abl_onlynp_p_sp_500 | timan108 | 500 | 0.988 | 0.094 | +1.0 pp (+6/−1, p=0.125) | +1.0 pp (+9/−4, p=0.267) |
| r8abl_clip_p_sp_500 | timan108 | 500 | 0.948 | 0.078 | -3.0 pp (+8/−23, p=0.0107) | -3.0 pp (+10/−25, p=0.0167) |

## GR00T LIBERO-10-50  (old A 0.611 / old B 0.725)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_g_l10_50 | timan107 | 500 | 0.730 | 0.221 |  | +12.0 pp (+98/−38, p=2.73e-07) |
| r8abl_ctrlA_g_l10_50 | timan107 | 500 | 0.610 | 0.074 | -12.0 pp (+38/−98, p=2.73e-07) |  |
| r8abl_onlynp_g_l10_50 | timan107 | 500 | 0.716 | 0.216 | -1.4 pp (+47/−54, p=0.551) | +10.6 pp (+92/−39, p=4.15e-06) |

## GR00T LIBERO-10-500  (old A 0.830 / old B 0.872)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_g_l10_500 | timan108 | 500 | 0.858 | 0.190 |  | +5.2 pp (+54/−28, p=0.00544) |
| r8abl_ctrlA_g_l10_500 | timan108 | 500 | 0.806 | 0.074 | -5.2 pp (+28/−54, p=0.00544) |  |
| r8abl_stuck_g_l10_500 | timan108 | 500 | 0.858 | 0.190 | +0.0 pp (+28/−28, p=1) | +5.2 pp (+52/−26, p=0.00433) |
| r8abl_terminal_g_l10_500 | timan108 | 500 | 0.866 | 0.184 | +0.8 pp (+31/−27, p=0.694) | +6.0 pp (+56/−26, p=0.00122) |
| r8abl_overtime_g_l10_500 | timan108 | 500 | 0.874 | 0.188 | +1.6 pp (+39/−31, p=0.403) | +6.8 pp (+57/−23, p=0.000183) |
| r8abl_no_progress_g_l10_500 | timan108 | 500 | 0.814 | 0.082 | -4.4 pp (+30/−52, p=0.0198) | +0.8 pp (+5/−1, p=0.219) |
| r8abl_onlynp_g_l10_500 | timan108 | 500 | 0.870 | 0.182 | +1.2 pp (+35/−29, p=0.532) | +6.4 pp (+53/−21, p=0.000256) |

## GR00T Spatial-50  (old A 0.867 / old B 0.876)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_g_sp_50 | timan108 | 500 | 0.888 | 0.146 |  | +2.6 pp (+30/−17, p=0.0789) |
| r8abl_ctrlA_g_sp_50 | timan108 | 500 | 0.862 | 0.075 | -2.6 pp (+17/−30, p=0.0789) |  |
| r8abl_onlynp_g_sp_50 | timan108 | 500 | 0.864 | 0.133 | -2.4 pp (+8/−20, p=0.0357) | +0.2 pp (+21/−20, p=1) |

## GR00T Spatial-500  (old A 0.964 / old B 0.959)

| arm | fleet | n | SR | owner IR | vs B control Δ (+/−, p) | vs A control Δ (+/−, p) |
|---|---|---|---|---|---|---|
| r8abl_ctrlB_g_sp_500 | timan108 | 500 | 0.940 | 0.129 |  | -1.6 pp (+3/−11, p=0.0574) |
| r8abl_ctrlA_g_sp_500 | timan108 | 500 | 0.956 | 0.076 | +1.6 pp (+11/−3, p=0.0574) |  |
| r8abl_stuck_g_sp_500 | timan108 | 500 | 0.942 | 0.129 | +0.2 pp (+8/−7, p=1) | -1.4 pp (+3/−10, p=0.0923) |
| r8abl_terminal_g_sp_500 | timan108 | 500 | 0.948 | 0.110 | +0.8 pp (+6/−2, p=0.289) | -0.8 pp (+4/−8, p=0.388) |
| r8abl_overtime_g_sp_500 | timan108 | 500 | 0.942 | 0.129 | +0.2 pp (+6/−5, p=1) | -1.4 pp (+3/−10, p=0.0923) |
| r8abl_no_progress_g_sp_500 | timan108 | 500 | 0.960 | 0.096 | +2.0 pp (+11/−1, p=0.00635) | +0.4 pp (+3/−1, p=0.625) |
| r8abl_onlynp_g_sp_500 | timan108 | 500 | 0.942 | 0.112 | +0.2 pp (+5/−4, p=1) | -1.4 pp (+2/−9, p=0.0654) |

