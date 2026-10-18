# R15 generated-asset collision, 2026-10-18

Current H100 chain `h100ly` contains exactly two remaining R15 units: `fa_pi_f_u_l` (5 cells / 20 fits) and `fa_pi_f_f_l` (6 cells / 24 fits). Other entries are R17. Read-only remote inspection found six conflicts in the original arm and zero in the filtered arm. The full relative paths and SHA256 values follow. Missing remote files are new inputs, not conflicts; every one of the 44 fit paths is recorded in CONFLICTS.json.

Root cause: `assets.build_plan()` relocates path metadata in external fits and serializes generated PKLs, but previously addressed those bytes by the original shared `mirror/...` filename. Distinct generated payloads therefore occupied one remote pathname. The unchanged ir25 original SHA256 is `1263ca850946e0ef456b2847ee1c0ab2d188026276cac3bf91b64b0bd9ffc293`; its prepared SHA256 is `4c3acf89ca61bf3917df3f7f003e75ad9691a40eb9a86ac7fa5f5587ed5d71b3`, and the existing H100 mirror is `6b3ab93f26c451c184ca36137981239d5dfa74beda53999e82b858a2035e0c16`.

Fix: legacy formal-A R15 prepare opts into `OSCL_ISOLATE_GENERATED_ASSETS=1`. Rewritten external inputs receive `generated/sha256/<generated-content-SHA256>/<historical-rel>` beneath `/data/oscl_h100`; the prepared arm arguments point at that address. Source PKLs, generated PKL bytes, numeric arrays, shared payload tables, runtime, plans/seeds, server topology and workers are unchanged. Already run-scoped generated YAMLs and deterministic store manifests keep their historical locations. The source hash, generated hash, original relative path and scoped relative path are all retained in the plan. `pull-new` still refuses any mismatch; no identity/seal/admission check is bypassed.

The R15 served-fit verifier derives the new address from its independently recomputed relocation digest and checks both source and generated plan pins. Historical plans retain the historical verification branch. No original/shared H100 file was written, renamed or removed. No stale-file mv is needed.

| Subcampaign | Conflicting relative path | Expected SHA256 | H100 SHA256 |
|---|---|---|---|
| fa_pi_f_u_l_l100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_l10_100_ir25.pkl` | `4c3acf89ca61bf3917df3f7f003e75ad9691a40eb9a86ac7fa5f5587ed5d71b3` | `6b3ab93f26c451c184ca36137981239d5dfa74beda53999e82b858a2035e0c16` |
| fa_pi_f_u_l_l100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_l10_100_ir32.pkl` | `7c259784cbdcd7972c3925eab96e02234d78d3d8a9587facf67cf376c5a0121b` | `969e3f0fd07deb667da3e93a091954649a87db8b7e88aa0ec4166d9ce9dfc257` |
| fa_pi_f_u_l_l100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_l10_100_ir40.pkl` | `337a2b36d25be83e4bbf05f7459a329a52448674dff4c5e629c3fb24ab3c91c6` | `a5e0ae67295576d18d3baf29324a1c4a82421d1ac04f11d407927e0310a8fec7` |
| fa_pi_f_u_l_s100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_spatial_100_ir25.pkl` | `216d857c8107d4f756026be0e315f6c6503143eb36b75c04fe82756e76170a3d` | `9961a708e09f779d255bf7bdde2de775f10e551dd503ea79ea3cddde1c2fa38b` |
| fa_pi_f_u_l_s100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_spatial_100_ir32.pkl` | `42a5a2b140c65f941f4caa8d947bed0826d3bbf2dbea7d82c1f9c106c67ec005` | `5435a2932a71282f8dc0214b1d12f0fe2df2b4a87cb76d4b2acd9ceabe5d17fb` |
| fa_pi_f_u_l_s100 | `mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_spatial_100_ir40.pkl` | `92b3f413823f3a0ea4ee3396f827ab12140a138e0dcc8f1f0f4b34a3a7b8ec67` | `11451d7b923610a1d921dcac36b558998430e72b5b821bb34257c018f6dce4d1` |

CPU verification: 11 fresh native `filter_failed.legacy plan --work-dir` processes / 44 generated external fits. Each work directory is isolated under `pristine_plans/`; no campaign directory was replanned or re-staged by this check. Large immutable store/payload source hashes were reused from their recorded plans with size checks, avoiding a multi-GB /data rehash. All changed/generated fits were actually hashed. Each cell uses a fresh Python process, matching real prepare and the existing verifier: NumPy pickle memo state affects the first serialization in a process. Earlier multi-cell-in-one-process exploratory files are explicitly marked superseded.

Remote verification: tether exec only, read-only; one reader, 32 MiB/s cap, pause between files, /proc/pressure/io full avg10 >20 wait. Read 1,051,954,564 bytes in 31.65 s. This audit covers all rewritten external fit dependencies; it does not claim to have rehashed the multi-GB native store. Normal native prepare continues to verify the complete dependency set.

Tests: 50 passed (new asset addressing / historical local behavior / R15 served-fit verification), plus 132 passed (legacy placement / native H100 and concurrent-sync robustness). Tests exercise two distinct campaigns referencing one source fit, repeated prepare stability, unchanged fit bytes and stale evidence preservation, and rejection of tampered scoped SHA/path bindings. Test placement expectations follow the owner-controlled autoscale switch without changing it.

Re-prepare, coordinator only:

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_pi_f_u_l.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_pi_f_f_l.sh
```

At the final read-only process check, the coordinator chain had already entered `vla_workers launch --job fa_pi_f_f_l_l50`. Do not re-prepare that unit while it runs. Its existing early cell syncs use historical identical mirror bytes; later cell syncs (s100/s200/s500) already record `sha256-v1`. No running directory or chain was edited. The original arm failed prepare before launch and can be re-prepared with the first command. The second command is supplied for a future unlaunched/closed eligible re-prepare only; it is unnecessary for the currently launched filtered unit.
