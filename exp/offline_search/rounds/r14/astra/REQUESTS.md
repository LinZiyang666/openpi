# Early B-dev recording request

No launch is requested during this offline work. Queue the paired pilot after the R13 pilot has run; the exact arms, counts and pass rules will be frozen in `pilot_grid.csv` and `SPEC.md` here.

Please retain the already requested R13 knob-off shadow recordings for both models and suites. They supply policy chunks on cache-driven states, needed to check pooled retrieval and correction away from policy trajectories. Preserve the corrected cache proposal on misses, both camera keys, robot state, donor IDs/weights, guard histories, complete accepted decision logs, pool contracts and per-episode outcomes. A recording with actions alone cannot validate policy-versus-cache error on cache hits.

For R14, the anticipated additional work is a compact paired pooled-component pilot plus periodic/task-allocation pilot on the same B complement. No new pure-policy recordings are required. Use the existing R11 development identities for this pilot wherever their served-subset contracts permit; this avoids using the reserved R13 pilot list. At the middle library size, use identities from the already consumed R12 development roots, disjoint from the actual served subset. If these are unavailable, defer that cell to the coordinator after R13 releases the reservation; do not read the reserved list to fill the gap.

The pilot is an implementation and budget check, not independent evidence after B-dev design. All completed arms must be retained. Size-full-library allocation is explicitly library-only transfer because there is no B complement. Open issue: availability of same-host paired recordings after R13; resolved operationally by the coordinator, not by moving one arm to a different host.

## Exact queued pilot

| cell | arms | episode runs | host |
| --- | --- | --- | --- |
| pi05_l10_50 | 14 | 700 | h100 |
| pi05_l10_200 | 6 | 300 | h100 |
| pi05_spatial_50 | 14 | 700 | h100 |
| groot_l10_50 | 14 | 700 | local_4090 |
| groot_l10_200 | 6 | 300 | local_4090 |
| groot_spatial_50 | 14 | 700 | local_4090 |

Total: 68 arms / 3400 episode runs; paired 5 existing B inits per task, exact rows in `pilot_grid.csv`. The repeat at the lower dose estimates implementation stability, not additional independent initial states. Please record all pilot real looks; off/base arms need shadow policy chunks. No extra shadow evaluation is required in periodic arms. Run all GR00T arms on the local 4090 and all pi05 arms on H100. Do not split a cell across hosts. This is a queued request, not a launch.
