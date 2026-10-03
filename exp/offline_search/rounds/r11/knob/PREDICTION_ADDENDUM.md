# Library-only prediction addendum

Frozen UTC: **2026-10-03T02:26:24.797667+00:00**.

Written before any R11 evaluation launch or result. No closed-loop outputs were read. Settings use each explorer’s B-only whole-episode-held-out calibration. All arms below retain fixed settings for the B pilot. IR is conditional on exogenous library states; closed-loop state/guard changes remain unmeasured.

| Cell | Method | Target | Predicted library IR | Setting |
|---|---|---:|---:|---|
| pi05_l10_50 | random | 0.25 | 0.250018462 | 0.1950 |
| pi05_l10_50 | periodic | 0.25 | 0.249982222 | 2.6125 |
| pi05_l10_50 | distance | 0.25 | 0.250000000 | q=0.25946540897712111, t=15.690658569335938, tie=0.9363212580792606 |
| pi05_l10_50 | error_hybrid | 0.25 | 0.250000000 | q=0.22478802362456918, t=0.25465290990677286, tie=0.5049691018648446 |
| pi05_l10_50 | random | 0.32 | 0.320032821 | 0.4177 |
| pi05_l10_50 | periodic | 0.32 | 0.319996581 | 1.0657 |
| pi05_l10_50 | distance | 0.32 | 0.320000000 | q=0.48423480102792382, t=12.455995559692383, tie=0.38679300667718053 |
| pi05_l10_50 | error_hybrid | 0.32 | 0.320000000 | q=0.44858830748125911, t=0.18505643733099134, tie=0.1207993826828897 |
| pi05_l10_50 | random | 0.40 | 0.399976752 | 0.6610 |
| pi05_l10_50 | periodic | 0.40 | 0.400012991 | 0.4386 |
| pi05_l10_50 | distance | 0.40 | 0.400000000 | q=0.71840348327532411, t=10.099003791809082, tie=0.3301885803230107 |
| pi05_l10_50 | error_hybrid | 0.40 | 0.400000000 | q=0.68879890022799373, t=0.12963218211540048, tie=0.7367831668816507 |
| pi05_l10_50 | adaptive_error_hybrid | 0.32 | 0.320569615 | q=0.233154296875, t=None, tie=None |
| pi05_l10_50 | adaptive_error_hybrid | 0.40 | 0.400422949 | q=0.570068359375, t=None, tie=None |
| pi05_l10_50 | disagreement | 0.32 | 0.320000000 | q=0.45893565518781543, t=0.16429209533544947, tie=0.38679142436012626 |
| pi05_l10_50 | periodic_pgt1 | 0.32 | 0.319996581 | 1.8009 |
| pi05_l10_50 | random_tail2 | 0.32 | 0.320032821 | 0.2839 |
| pi05_l10_200 | random | 0.25 | 0.249992306 | 0.2663 |
| pi05_l10_200 | periodic | 0.25 | 0.250001574 | 2.0000 |
| pi05_l10_200 | distance | 0.25 | 0.250000000 | q=0.30184740526601672, t=8.817926406860352, tie=0.22877402743324637 |
| pi05_l10_200 | error_hybrid | 0.25 | 0.250000000 | q=0.28066154895350337, t=0.15442089405885837, tie=0.926135381218046 |
| pi05_l10_500 | random | 0.25 | 0.249997489 | 0.2537 |
| pi05_l10_500 | periodic | 0.25 | 0.250001086 | 2.0924 |
| pi05_l10_500 | distance | 0.25 | 0.250000000 | q=0.29190290020778775, t=7.667969226837158, tie=0.9622749239206314 |
| pi05_l10_500 | error_hybrid | 0.25 | 0.250000000 | q=0.26775972032919526, t=0.1341186409513071, tie=0.41447754204273224 |
| pi05_spatial_50 | random | 0.25 | 0.249932817 | 0.2900 |
| pi05_spatial_50 | periodic | 0.25 | 0.250024117 | 1.7790 |
| pi05_spatial_50 | distance | 0.25 | 0.250000000 | q=0.30262013850733638, t=21.114843368530273, tie=0.34198080701753497 |
| pi05_spatial_50 | error_hybrid | 0.25 | 0.250000000 | q=0.32008757395669818, t=0.2984256253920063, tie=0.6216733637265861 |
| pi05_spatial_50 | random | 0.32 | 0.319960379 | 0.4826 |
| pi05_spatial_50 | periodic | 0.32 | 0.320051680 | 0.8707 |
| pi05_spatial_50 | distance | 0.32 | 0.320000000 | q=0.49886239925399423, t=18.197093963623047, tie=0.17924553388729692 |
| pi05_spatial_50 | error_hybrid | 0.32 | 0.320000000 | q=0.52461850689724088, t=0.22548681447407615, tie=0.08208650769665837 |
| pi05_spatial_50 | random | 0.40 | 0.399939707 | 0.6885 |
| pi05_spatial_50 | periodic | 0.40 | 0.400031008 | 0.3931 |
| pi05_spatial_50 | distance | 0.40 | 0.400000000 | q=0.72412364138290286, t=15.067574501037598, tie=0.7075476455502212 |
| pi05_spatial_50 | error_hybrid | 0.40 | 0.400000000 | q=0.72424238966777921, t=0.16775496143505556, tie=0.8454144042916596 |
| pi05_spatial_50 | adaptive_error_hybrid | 0.32 | 0.319429694 | q=0.424072265625, t=None, tie=None |
| pi05_spatial_50 | adaptive_error_hybrid | 0.40 | 0.399631568 | q=0.65283203125, t=None, tie=None |
| pi05_spatial_50 | disagreement | 0.32 | 0.320000000 | q=0.52211821312084794, t=0.19347305412688093, tie=0.17924543330445886 |
| pi05_spatial_50 | periodic_pgt1 | 0.32 | 0.319960379 | 1.0739 |
| groot_l10_50 | random | 0.25 | 0.249979526 | 0.1707 |
| groot_l10_50 | periodic | 0.25 | 0.250015567 | 2.7237 |
| groot_l10_50 | distance | 0.25 | 0.250000000 | q=0.19625922525301576, t=14.727766990661621, tie=0.9460106226615608 |
| groot_l10_50 | error_hybrid | 0.25 | 0.250000000 | q=0.13894562376663089, t=0.381468330471366, tie=0.584318230394274 |
| groot_l10_50 | random | 0.32 | 0.319970389 | 0.3998 |
| groot_l10_50 | periodic | 0.32 | 0.319970389 | 1.0252 |
| groot_l10_50 | distance | 0.32 | 0.320000000 | q=0.44085539737716317, t=11.571511268615723, tie=0.7276992495171726 |
| groot_l10_50 | error_hybrid | 0.32 | 0.320000000 | q=0.37311248714104295, t=0.2659830009222694, tie=0.5473995017819107 |
| groot_l10_50 | random | 0.40 | 0.400016582 | 0.6555 |
| groot_l10_50 | periodic | 0.40 | 0.400016582 | 0.4512 |
| groot_l10_50 | distance | 0.40 | 0.400000000 | q=0.6961734308861196, t=9.88180160522461, tie=0.19248826848343015 |
| groot_l10_50 | error_hybrid | 0.40 | 0.400000000 | q=0.6493460931815207, t=0.19772293686853637, tie=0.8177053513936698 |
| groot_l10_50 | adaptive_error_hybrid | 0.32 | 0.319821722 | q=0.15576171875, t=None, tie=None |
| groot_l10_50 | adaptive_error_hybrid | 0.40 | 0.399656176 | q=0.53704833984375, t=None, tie=None |
| groot_l10_50 | disagreement | 0.32 | 0.320000000 | q=0.36166758043691516, t=0.24624925624152214, tie=0.7277001910842955 |
| groot_l10_50 | periodic_pgt1 | 0.32 | 0.319970389 | 2.3165 |
| groot_l10_50 | random_tail2 | 0.32 | 0.320006430 | 0.2777 |
| groot_l10_200 | random | 0.25 | 0.250000641 | 0.1910 |
| groot_l10_200 | periodic | 0.25 | 0.249991540 | 2.4585 |
| groot_l10_200 | distance | 0.25 | 0.250000000 | q=0.20888448087498546, t=9.593710899353027, tie=0.3661951990798116 |
| groot_l10_200 | error_hybrid | 0.25 | 0.250000000 | q=0.18871211027726531, t=0.26793641181183364, tie=0.309114464558661 |
| groot_l10_200 | random | 0.32 | 0.320005597 | 0.4221 |
| groot_l10_200 | periodic | 0.32 | 0.319996496 | 1.0069 |
| groot_l10_200 | distance | 0.32 | 0.320000000 | q=0.43948026979342103, t=8.304780006408691, tie=0.7981171226128936 |
| groot_l10_200 | error_hybrid | 0.32 | 0.320000000 | q=0.41415094723924994, t=0.1904106745860522, tie=0.3943845937028527 |
| groot_l10_500 | random | 0.25 | 0.249999814 | 0.1869 |
| groot_l10_500 | periodic | 0.25 | 0.250003409 | 2.4699 |
| groot_l10_500 | distance | 0.25 | 0.250000000 | q=0.21287271613255143, t=8.496006965637207, tie=0.6314517236314714 |
| groot_l10_500 | error_hybrid | 0.25 | 0.250000000 | q=0.1938377576880157, t=0.2419102207272607, tie=0.6065980535931885 |
| groot_spatial_50 | random | 0.25 | 0.249966278 | 0.2933 |
| groot_spatial_50 | periodic | 0.25 | 0.249966278 | 1.7146 |
| groot_spatial_50 | distance | 0.25 | 0.250000000 | q=0.30301626538857818, t=19.323528289794922, tie=0.92253473168239 |
| groot_spatial_50 | error_hybrid | 0.25 | 0.250000000 | q=0.31182539137080312, t=0.27292450435604615, tie=0.5022950363345444 |
| groot_spatial_50 | random | 0.32 | 0.319931724 | 0.4843 |
| groot_spatial_50 | periodic | 0.32 | 0.320020400 | 0.8576 |
| groot_spatial_50 | distance | 0.32 | 0.320000000 | q=0.4767662319354713, t=17.282115936279297, tie=0.5962445545010269 |
| groot_spatial_50 | error_hybrid | 0.32 | 0.320000000 | q=0.50918295187875628, t=0.21449241888400405, tie=0.5287252063862979 |
| groot_spatial_50 | random | 0.40 | 0.399917569 | 0.6902 |
| groot_spatial_50 | periodic | 0.40 | 0.400006245 | 0.3931 |
| groot_spatial_50 | distance | 0.40 | 0.400000000 | q=0.70138734159991145, t=15.50002670288086, tie=0.36619726149365306 |
| groot_spatial_50 | error_hybrid | 0.40 | 0.400000000 | q=0.70895583787932992, t=0.16953316587169148, tie=0.4559612930752337 |
| groot_spatial_50 | adaptive_error_hybrid | 0.32 | 0.319698949 | q=0.4046630859375, t=None, tie=None |
| groot_spatial_50 | adaptive_error_hybrid | 0.40 | 0.399607202 | q=0.615478515625, t=None, tie=None |
| groot_spatial_50 | disagreement | 0.32 | 0.320000000 | q=0.52422668086364865, t=0.18991143306159494, tie=0.5962437172420323 |
| groot_spatial_50 | periodic_pgt1 | 0.32 | 0.319931724 | 1.1020 |

Added targets include both Spatial schedules at .25/.32/.40, Spatial post-guard schedules at .32, and GR00T L10-200 state methods at .32. The complete final grid is listed to make the additions reviewable.
No new SR advantage is assigned to state placement at matched IR. Sparse cells may improve as IR rises; dense L10 cells are expected near neutral. The pure-policy library cannot measure recovery utility.
