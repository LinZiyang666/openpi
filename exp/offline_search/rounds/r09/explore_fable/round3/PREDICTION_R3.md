# Round-3 stack predictions (fable) — written 2026-10-02 01:37:45 CDT, before any r09_fable_r3 arm ran

Evaluation: 100 pairs, tasks 0–9 × inits 20–29, h100 + timan107 (same fleet as the round-2 components). Components on these pairs
(round 2): π0.5 L10-50 cache .740 @ .0765, half corrector .810 @ .076, only-no-progress + half corrector .860 @ .177, empty-grasp
recovery on cache .790 @ .086 (+5 pp for +.010), pure policy .930; GR00T L10-50 cache .560 @ .074, half corrector .770 @ .074,
sign-corrected recovery on cache .630 @ .086 (+7 pp), pure policy .870; opus (timan108): π0.5 only-no-progress .770 @ .187,
+ escalation .850 @ .195; GR00T only-no-progress .780 @ .206, + escalation .730 @ .255.

| arm | stack | predicted SR (range) | predicted owner IR | reasoning |
|---|---|---|---|---|
| r9f3_groot_l10_50_np_corr05 | only-no-progress + half corrector | **.83** (.79–.87) | **.17** (.15–.19) | corrector fixes approach drift (+21 pp alone), the guard rescues stalls (+22 pp alone); gains overlap (both act on the same failing episodes), so less than additive; the corrector reduces stalls → fewer guard calls than onlynp's .206 |
| r9f3_groot_l10_50_corr05_gmS | half corrector + empty-grasp recovery (burst 2, cap 2) | **.81** (.77–.85) | **.082** (.078–.088) | recovery added +7 pp on the plain cache; on the corrector fewer empty grasps remain → +3–5 pp; ~0.6 calls/episode |
| r9f3_groot_l10_50_np_corr05_gmS | only-no-progress + half corrector + recovery | **.85** (.80–.89) | **.18** (.16–.20) | recovery fires earlier than the no-progress guard on the same episodes (30–70 controls lead) and replaces some guard calls; small net gain over np_corr05 |
| r9f3_pi05_l10_50_np_corr05_esc | only-no-progress + half corrector + opus escalation (lag 12, deadline 80) | **.88** (.85–.91) | **.20** (.19–.22) | escalation gave +8 pp over onlynp alone; on np_corr05 (.860) the corrector already removes part of the lagging episodes → +2–4 pp, +.02–.04 IR |
| r9f3_pi05_l10_50_np_corr05_gm | only-no-progress + half corrector + recovery | **.875** (.84–.90) | **.185** (.175–.195) | recovery adds +5 pp on the plain cache; the guard already catches many empty grasps late → +1–3 pp |
| r9f3_pi05_l10_50_np_corr05_gm_esc | full stack (guard + corrector + recovery + escalation) | **.89** (.86–.92) | **.21** (.20–.23) | the three call mechanisms overlap; best SR of the set, highest IR; still ≥ 3 pp below pure policy |

Falsifiable: (a) every stack beats its strongest component on SR; (b) np_corr05 (GR00T) ≥ .80; (c) IR of every np-stack stays below the
onlynp arm's IR of the same model (.187 π0.5 / .206 GR00T) because the corrector reduces stalls; (d) recovery adds ≤ +.012 IR; (e) no stack
reaches pure policy − 2 pp on 100 pairs.
