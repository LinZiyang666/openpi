<task>
Coordinator, 2026-10-16 14:4x. Defect in `filter_failed_knob.workflow build-test` for raise generations (generation >= 1).

`next_campaign` builds generation N from the source raise template `knobtest<model>[_red]_r<N>` when it exists. `build.build` copies that template's `assets/` (so `assets/cache_inputs` contains only the cells the *unfiltered* arm raised in round N), but the groups come from the dev ladder for every cell the *filtered* raise table names. When the filtered raise touches a cell the template lacks, the recipe's store/fit/stall paths point at files that do not exist.

Observed: `ffk1015_dp_libero_full_test01` (built 14:3x, 700 episodes, source knobtestdp_r1; never prepared or launched). Its raise table (`results/dp_libero_full/test00/raise.json`) needs libero_spatial 50/100/300/400/500 and libero_10 50/100/200/300/400, but the copied assets only have libero_spatial n50/n200/n400 and libero_10 n50/n100/n200/n400. The startup check fails: `FileNotFoundError .../ffk1015_dp_libero_full_test01/assets/cache_inputs/libero_spatial/parent/subsets/size_100/READY.json` (`startup_checks/ffk1015_dp_libero_full_test01.json`). `ffk1015_dp_libero_reduced_test01` happens to need only size-50 cells, which its template has; it passed startup.

I checked that for overlapping cells the data bytes are identical between the official-A source and the raise template (knobtestdp vs knobtestdp_r1: fit.npz, stall.pkl, parent READY.json, subsets row_indices.npy all equal). The row filter is applied in memory through `campaign.filter_failed_knob.selection`; asset bytes are never filtered.

Fix it with the smallest change: keep the raise template's runtime/topology/serving release, and complete any cache_inputs cell the raise needs but the template lacks from the model/mode's official-A source (`TEST_SOURCE`), only after verifying byte equality for the cells both contain and that the A source has the missing cell. Keep the input manifest/seal consistent the same way build already does. Check whether Fast-WAM and Cosmos raise templates (reduced and full) have the same gap and make the fix general.

Then rebuild `ffk1015_dp_libero_full_test01` from the same raise table. It was never prepared/launched: move the old directory aside by rename (e.g. `ffk1015_dp_libero_full_test01.bad_assets_1016`), do not delete it. Also move aside `startup_checks/ffk1015_dp_libero_full_test01.json` by rename. Run the startup check on the rebuilt run, and confirm `ffk1015_dp_libero_reduced_test01` would be built identically by the fixed code (do not rebuild it: it is queued; just compare on CPU).
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm -rf. Never touch credentials.
- Do not modify, rebuild or re-stage any launched or queued run other than ffk1015_dp_libero_full_test01. Queued and must stay byte-identical: ffk1015_dp_libero_reduced_test01.
- tether: exec read-only checks only; never push/pull/expose.
- Reuse existing code; smallest change; add one targeted test.
</rules>

<output>
Final message: the diff summary, which models/modes had the gap, the rebuilt run's episode count and startup result, the reduced test01 equality check, and the exact prepare/launch commands.
</output>
