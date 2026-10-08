# R15 coordinator decisions

All recorded before any R15 outcome exists.

## D1 (2026-10-05 ~10:0x CDT): track A uses the frozen R14 protocol unchanged
`rounds/r14/descent_astra/PROTOCOL.md` applies verbatim: 30 paired B inits per task, primary 5 pp point-estimate rule, 0 / 2 pp alternates kept, first-failure stop, order c10 → ir45 → ir40 → ir32 → ir25 → off, uniform level by the same rule on the cell's 300 episodes, test A measure-only. Only the cell list (12 cells: both models × both suites × sizes 100 / 200 / 500) and the arm set change (D3). The R14 library-allocation `task_periodic` arms are not repeated. Reason: the owner asked to fill in the other sizes, so the rule must stay comparable with R14's size-50 cells.

## D2: dev inits and the size-500 cells
- Per model × suite, the dev list is the B inits outside that model's 200-subset (30 per task). Subsets are nested, so these inits are unseen at 100 and 200.
- Size 500 contains every B init. Its cells use the same dev list with leave-own-episode-out serving: the query init's own library episode is removed from every neighbour lookup. Fits (metric, corrector, guard, caps) still include that episode; this is disclosed as a mild optimistic bias. If leave-own-episode-out cannot be built safely, sol reports and the coordinator decides before any 500 run.

## D3: arms
Per cell: off, periodic ir25 / ir32 / ir40 / ir45. Pure C10 runs once per model × suite on the dev list and serves as the reference for all three sizes, **only if** sol proves on CPU that its decisions and actions do not depend on the library. Otherwise C10 runs per cell. The same test applies to reusing R14's test-A pure C10 arms as the test reference.

## D4: hosts
GR00T roots run on the local 4090 (host effect; R14's GR00T references ran there). π0.5 roots run on h100 (no host effect).

## D5 (2026-10-05 ~14:0x CDT): track B goes to ablation / appendix
Owner, after the size-50 pilot (light key off SR .76 vs current .85, IR .141 vs .165; ir25 .813 vs .860, IR .206 vs .235): "所以还是视觉塔最准，这个又可以作为一个对比设计进ablation或者appendix". The light image key is not part of the main design; it is reported as a comparison design in the ablation or appendix. The size-200 pilot finishes with its reused current-system baselines and is reported alongside.
