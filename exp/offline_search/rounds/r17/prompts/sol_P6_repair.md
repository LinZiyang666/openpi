<task>
The DP libero_spatial library campaign collected 500/500 but FAILED at completion: "driver incomplete: journal must accept every planned episode exactly once without errors".
- Campaign: `/data/r17_library/dp_libero_libero_spatial/campaigns/teacher_dp_libero_libero_spatial_500_20261010_000311_bdd4f22f`
- Cause: 4 attempts (task 6, inits 41–44) hit the transient timan107 EGL error `FatalError('Offscreen framebuffer is not complete, error 0x8cdd')` at n_steps=0, i.e. before any decision. Library campaigns run with retries=0, and `build_teacher_parent` requires attempt==1.
The same EGL error appeared in the formal DP-LIBERO run, where the driver retried it fine. These are infrastructure failures before the episode starts, not teacher outcomes. Re-running the same B init is legitimate: one trial per init still holds, and outcomes are not cherry-picked.

Implement:
1. Pre-decision infrastructure retry. Allow the driver to retry an attempt only when it errored with zero executed steps and zero captured decisions (bounded, e.g. 2). Accept attempt>1 in `build_teacher_parent` only when every prior attempt for that uid is such a pre-decision error with no capture rows; record them in the parent provenance. Any error after ≥1 decision or step, and any ordinary failed outcome, is never retried.
2. Repair mode: `library_campaign.sh ... --repair <FAILED_CAMPAIGN>`. Re-run only the planned episodes lacking an accepted non-error result, using the identical seal/bundle/B inputs, and merge their closed captures with the original campaign's verified captures. Then run the normal finish (parent, subsets, fits). It must refuse if the seal, checkpoint or alignment differs, or if any original episode has an error after ≥1 decision.
3. CPU tests for both.
4. Give the exact repair command for the DP spatial campaign above.
- Do not start GPU work.
- Running now: Cosmos finish on weilandserver (CPU build) and DP libero_10 collection on the 4090. Do not touch their processes or files.
</task>
<compact_output_contract>Append "## Pre-decision retry + repair mode" to HANDBACK_P6.md. Final message: 4 lines.</compact_output_contract>
