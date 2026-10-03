# r08_abl supplementary guard ablations

Standard mode; no debug capture. Every arm uses the official 500 test pairs (tasks 0–9 × inits 0–49).
Each arm is paired separately against all three A and all three B references in its cell, as in R6 PAPER_AB.md.
Only name, method, disabled_guards and the fit path change from the frozen deployed B input row.
Fits are fresh through the same plugin prefit entry point as R6 P2. Fit SHAs identify the complete pickle;
the source B SHAs and exact fitted-state comparisons are in the run's validation/validation.json.

| Arm | B source row (replicate 1) | Disabled guards | Fit SHA256 | A×3 / B×3 references |
|---|---|---|---|---|
| `r8abl_onlynp_p_l10_50` | `r05_q1:r5q1_c10_p_l10_50` | stuck, terminal, overtime | `d68bdfdd3aa4ef14ac440de53941996d2bf1cc1ef515c6f1a63950fe14463e88` | `pi05_l10_50` (below) |
| `r8abl_stuck_p_l10_500` | `r05_q1:r5q1_c10_p_l10_500` | stuck | `d08155214a565ec23be8820ccab7c84e7deba93e0eb5d452d2f16d07e3c550ca` | `pi05_l10_500` (below) |
| `r8abl_terminal_p_l10_500` | `r05_q1:r5q1_c10_p_l10_500` | terminal | `07fab231c8dacb2e314c4ad90ca8fa96be93174a13a6e740bd6251c1868996bb` | `pi05_l10_500` (below) |
| `r8abl_overtime_p_l10_500` | `r05_q1:r5q1_c10_p_l10_500` | overtime | `b00c3281585ef2d7e0241e2e5286b16a12ebfefeb5fa792b55422ebb3e8fe6e2` | `pi05_l10_500` (below) |
| `r8abl_no_progress_p_l10_500` | `r05_q1:r5q1_c10_p_l10_500` | no_progress | `c116ea83c25e1ead0183947d9b8a44dc1650182d22031a3efb1cedabebbed7dc` | `pi05_l10_500` (below) |
| `r8abl_onlynp_p_l10_500` | `r05_q1:r5q1_c10_p_l10_500` | stuck, terminal, overtime | `90953ead34afc02a68908018280a44ba7805c53703bd6f8ad41e41140cf24f4c` | `pi05_l10_500` (below) |
| `r8abl_onlynp_p_sp_50` | `r05_q1:r5q1_c10_p_sp_50` | stuck, terminal, overtime | `4b5469639b4be564d48deccb158e4a6d80ab2767abe7ef052aad5d85ce1f8a29` | `pi05_spatial_50` (below) |
| `r8abl_stuck_p_sp_500` | `r05_q1:r5q1_c10_p_sp_500` | stuck | `d15b117064122826bd0448fbad31c68582b336ddb60ba1834143f900f3bdbaa5` | `pi05_spatial_500` (below) |
| `r8abl_terminal_p_sp_500` | `r05_q1:r5q1_c10_p_sp_500` | terminal | `8f75fe794b8bf348514e977aff237870343a92cc0c4bb1d046fa23ce43971e17` | `pi05_spatial_500` (below) |
| `r8abl_overtime_p_sp_500` | `r05_q1:r5q1_c10_p_sp_500` | overtime | `7eb388fec0cb07f2864e95add6b087beb1b018c4d21e59319c077df087486aac` | `pi05_spatial_500` (below) |
| `r8abl_no_progress_p_sp_500` | `r05_q1:r5q1_c10_p_sp_500` | no_progress | `04e8fe522ec5fd92f4bdfbac752bb8b8cef8515f5ad28012c9792ca3bbc81d70` | `pi05_spatial_500` (below) |
| `r8abl_onlynp_p_sp_500` | `r05_q1:r5q1_c10_p_sp_500` | stuck, terminal, overtime | `306f25204027d9ae4759d75735907c100ae2b69853d13a268f0b5cb77a0d9da1` | `pi05_spatial_500` (below) |
| `r8abl_onlynp_g_l10_50` | `r06_paper:r6p1_c10_g_l10_50` | stuck, terminal, overtime | `f6ff08e9d449646c9bbbbe37b6cac6cce74744f6869c89cfe5e84d55d7f69df8` | `groot_l10_50` (below) |
| `r8abl_stuck_g_l10_500` | `r06_paper:r6p1_c10_g_l10_500` | stuck | `6e085f6ccfaf05e6471772e73410933080005bf4b0fe5503d1e8b97d9c4bdb81` | `groot_l10_500` (below) |
| `r8abl_terminal_g_l10_500` | `r06_paper:r6p1_c10_g_l10_500` | terminal | `a90004f9e6e28d6ff0e17edf2b7add08be420ef38ae2afb3654f2930abccdc1a` | `groot_l10_500` (below) |
| `r8abl_overtime_g_l10_500` | `r06_paper:r6p1_c10_g_l10_500` | overtime | `d49c67aa48ccc10067fae0a3151c6db732df3e1df871af09da7a14f841a4e0c5` | `groot_l10_500` (below) |
| `r8abl_no_progress_g_l10_500` | `r06_paper:r6p1_c10_g_l10_500` | no_progress | `e37a4bd64d3673932bd7a291eda74f47c2166eaebfb1aa1552c5bff61c10e053` | `groot_l10_500` (below) |
| `r8abl_onlynp_g_l10_500` | `r06_paper:r6p1_c10_g_l10_500` | stuck, terminal, overtime | `c58f64cb3e47614600c30a38f1bf481a1cd7481607b128705b3dcfd4d5d38358` | `groot_l10_500` (below) |
| `r8abl_onlynp_g_sp_50` | `r06_paper:r6p1_c10_g_sp_50` | stuck, terminal, overtime | `e0057e26134b7039f43cc91b388de9778d0b5f08d49b62219fa2d2f686c6934c` | `groot_spatial_50` (below) |
| `r8abl_stuck_g_sp_500` | `r06_paper:r6p1_c10_g_sp_500` | stuck | `406fc73b3e6bfbeb8761672dbcb53b8fcf9ba0a119839e227be11631216dacf6` | `groot_spatial_500` (below) |
| `r8abl_terminal_g_sp_500` | `r06_paper:r6p1_c10_g_sp_500` | terminal | `43c571386f1212e80f614adc86a130e4c5f7a950b547301513fe6988bcacab5d` | `groot_spatial_500` (below) |
| `r8abl_overtime_g_sp_500` | `r06_paper:r6p1_c10_g_sp_500` | overtime | `1a47d9bf7000615a5b1fbb4e605de8213b71e8b5168c45fb5377956cb7ecdc2a` | `groot_spatial_500` (below) |
| `r8abl_no_progress_g_sp_500` | `r06_paper:r6p1_c10_g_sp_500` | no_progress | `ee8a9cff84b2bfeaaeacc8a38f749ceb9afc5f8bd2f30e87617479fc12e9be3a` | `groot_spatial_500` (below) |
| `r8abl_onlynp_g_sp_500` | `r06_paper:r6p1_c10_g_sp_500` | stuck, terminal, overtime | `3f8ec60f7b67314f7d8270a8a9bc38b5de66881ef10be9a317c339d0fe842514` | `groot_spatial_500` (below) |

References are the exact run:arm names from r06/ops/paper_ab.py (the generator of PAPER_AB.md).

| Cell | A replicates 1 / 2 / 3 | B replicates 1 / 2 / 3 |
|---|---|---|
| `pi05_l10_50` | `r05_ptail:r5t_p_l10_50_tail1uc`<br>`r06_paper:r5t_p_l10_50_tail1uc_rep2`<br>`r06_paper:r5t_p_l10_50_tail1uc_rep3` | `r05_q1:r5q1_c10_p_l10_50`<br>`r06_paper:r5q1_c10_p_l10_50_rep2`<br>`r06_paper:r5q1_c10_p_l10_50_rep3` |
| `pi05_l10_500` | `r05_ptail:r5t_p_l10_500_tail1uc`<br>`r06_paper:r5t_p_l10_500_tail1uc_rep2`<br>`r06_paper:r5t_p_l10_500_tail1uc_rep3` | `r05_q1:r5q1_c10_p_l10_500`<br>`r06_paper:r5q1_c10_p_l10_500_rep2`<br>`r06_paper:r5q1_c10_p_l10_500_rep3` |
| `pi05_spatial_50` | `r05_ptail:r5t_p_sp_50_tail1uc`<br>`r06_paper:r5t_p_sp_50_tail1uc_rep2`<br>`r06_paper:r5t_p_sp_50_tail1uc_rep3` | `r05_q1:r5q1_c10_p_sp_50`<br>`r06_paper:r5q1_c10_p_sp_50_rep2`<br>`r06_paper:r5q1_c10_p_sp_50_rep3` |
| `pi05_spatial_500` | `r04_blind:r4b3_p_sp_500_tail1uc`<br>`r06_paper:r4b3_p_sp_500_tail1uc_rep2`<br>`r06_paper:r4b3_p_sp_500_tail1uc_rep3` | `r05_q1:r5q1_c10_p_sp_500`<br>`r06_paper:r5q1_c10_p_sp_500_rep2`<br>`r06_paper:r5q1_c10_p_sp_500_rep3` |
| `groot_l10_50` | `r05_x:r5x_g_l10_50_tail1u`<br>`r06_paper:r5x_g_l10_50_tail1u_rep2`<br>`r06_paper:r5x_g_l10_50_tail1u_rep3` | `r06_paper:r6p1_c10_g_l10_50`<br>`r06_paper:r6p1_c10_g_l10_50_rep2`<br>`r06_paper:r6p1_c10_g_l10_50_rep3` |
| `groot_l10_500` | `r05_x:r5x_g_l10_500_tail1u`<br>`r06_paper:r5x_g_l10_500_tail1u_rep2`<br>`r06_paper:r5x_g_l10_500_tail1u_rep3` | `r06_paper:r6p1_c10_g_l10_500`<br>`r06_paper:r6p1_c10_g_l10_500_rep2`<br>`r06_paper:r6p1_c10_g_l10_500_rep3` |
| `groot_spatial_50` | `r05_x:r5x_g_sp_50_tail1u`<br>`r06_paper:r5x_g_sp_50_tail1u_rep2`<br>`r06_paper:r5x_g_sp_50_tail1u_rep3` | `r06_paper:r6p1_c10_g_sp_50`<br>`r06_paper:r6p1_c10_g_sp_50_rep2`<br>`r06_paper:r6p1_c10_g_sp_50_rep3` |
| `groot_spatial_500` | `r05_x:r5x_g_sp_500_tail1u`<br>`r06_paper:r5x_g_sp_500_tail1u_rep2`<br>`r06_paper:r5x_g_sp_500_tail1u_rep3` | `r06_paper:r6p1_c10_g_sp_500`<br>`r06_paper:r6p1_c10_g_sp_500_rep2`<br>`r06_paper:r6p1_c10_g_sp_500_rep3` |

Source-row locations: π0.5 = `/home/weiland/trace_runs/os_closed_loop/r05_q1/arms_in.json`;
GR00T = `/home/weiland/trace_runs/os_closed_loop/r06_paper/arms_in.json`.
Run root: `/home/weiland/trace_runs/os_closed_loop/r08_abl`.
The frozen source rows/deployed rows and hashes are in `provenance.json`; the permitted-field diff is
in `validation/source_diff.json`. No manifest field was added to the B specs; the coordinator should
use `manifests/eval500.json` for an explicit selection, or the harness's full official test default.
