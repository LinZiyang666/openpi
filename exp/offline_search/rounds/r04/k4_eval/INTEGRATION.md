K4 installed integration contract (see HANDBACK.md for commands and evidence):
- emit_arms: yaml_patch recursively merged; miss.num_steps gets default evidence_dir=<RUN>/evidence/<arm> if omitted (K3 config requirement).
- client_overrides={replan_steps:L,resize_size:N} copied to arms.json and matrix; direct replan_steps alias. Default L=5. run_arm translates matrix settings into CLI.
- pure_inference=true -> plain proposal + periodic:1; server_seed base uses SeededInference (B0 proposal) to seed torch/random/numpy once before first MISS. chain passes --os-seed base*65536+port; server prints OSCL_POLICY_SEED. No K2/K3 edits needed. seeds=[...] expands distinct arm names.
- server_env forwarded by chain; GR00T miss.num_steps automatically sets GROOT_DENOISING_STEPS.
- cost_ledger=true adds ledger for an arm even without new log fields; --ledger adds it for historical data. Old default outputs remain unchanged.
- Cost table: models.MODEL.full={s1,s2,s3,k}, models.MODEL.modes.MODE={s1,s2,s3,k,miss_s1_extra}, full_cost_ms. K3's installed table and GR00T by_suite override tested. Wrist MISS adds the completion cost. Historical logs retain owner pricing unless explicitly repriced with --cost-table.
- OSCL_MANIFEST / --manifest exact pairs; fingerprinted DONE; accepted pair counting; partial weighted designs report no estimate. chain uploads manifest snapshots; coordinator pushes the three remote scripts.
- Arms use --os-log-r4 (K2) to request all decision fields. First-batch and pure controls are in arms_frontier.json (17 rows, <RUN> placeholders).
