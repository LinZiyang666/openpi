<task>
Owner decision after your edit (verbatim): 「那就只测k=2算上清理步」. Add exactly ONE Cosmos arm back to the pre-registration: **Cosmos K2 incl. clean step** (official upstream num_steps=2 semantics, real measured NFE recorded), C16, on **H100**, paired with the running H100 official baseline `formal_cosmos_libero_official2000_20261010_100533_5f558380` (K5/C16, same GPU type; pair on libero_spatial + libero_10 by canonical task/init identity).
- Workers: timan108 ×12.
- No 4090 Cosmos control and no K1/K3.
- sol D is enabling K2 in the launcher right now: HANDBACK_COSMOS § "K2 incl. clean step".
Update sections (6) and (7) and the multiplicity count (22 → 24 comparisons). Keep your notes on the old study's NFE/GPU caveats.
Also add the owner's open question on batching: P3 is measuring Fast-WAM server-side batching (`models/fastwam/BATCHING.md`). If batching is adopted, the control and the arms must share the batch configuration; pre-register that rule now.
</task>
<compact_output_contract>Edit STUDY.md in place. Final message: 2 lines.</compact_output_contract>
