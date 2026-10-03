# Coordination with the shared IR model and schedule grid

The cost arithmetic and no-progress guard agree exactly on identical input tables; see `crosscheck.json`. The separately fitted score/guard tables have different PCA conventions and must not be combined numerically.

At handback, the proposed totals are 44 astra arms + 38 opus arms + 8 same-batch knob-off controls. For matched-budget random comparisons, the current opus grid lacks these state-knob targets:

- `groot_spatial_50` at target owner IR `0.25`
- `groot_spatial_50` at target owner IR `0.32`
- `groot_spatial_50` at target owner IR `0.40`
- `pi05_spatial_50` at target owner IR `0.25`
- `pi05_spatial_50` at target owner IR `0.32`

Adding these 5 random controls gives 95 total arms, within the brief’s approximate total budget. These additions are a coordinator recommendation, not launches or mutations to opus’s grid. If the coordinator chooses a smaller grid, retain exact matched targets for every state-dependent method under comparison.

No frozen astra parameter or forecast was changed after reading the schedule grid. Source version and exact missing pairs are stored in `coordination.json`.
