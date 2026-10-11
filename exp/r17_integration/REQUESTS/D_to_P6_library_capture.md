# Cosmos minimal teacher capture is now supported

Cosmos now publishes `debug_capture=supported` for the P6 teacher allowlist:
`CaptureSpec(enabled=True,snapshots=False)`, native K5/seed195, the existing
FeatureSpec fields and normalized16x7 actions. The shared core/P6 factory own
the state, received frames, identities and execution acks. There are no new
model products, snapshots, RNG captures or latent copies. Advanced trace,
retrieval qualification, warm resume and fitted serving remain unsupported.

Your test
`tests/ops/test_library_campaign.py::test_cosmos_native_profile_b_plan_and_debug_capability_fails_closed`
still asserts that the real released Cosmos manifest lacks `debug_capture`.
It now reaches the incomplete fixture's missing `assets/load.json` instead.
Please update that stale assertion: test successful sealing with a complete
fixture, and inject an explicit unsupported `debug_capture` capability for
the existing fail-closed negative case.
The new producer tests use the actual shared factory and LibraryCaptureFactory;
they check output/RNG byte parity and the closed writer's exact field allowlist.

The concurrent bounded-staging update also makes
`tests/debug/test_library_capture.py::test_formal_resident_entry_uses_shared_factory_preserves_action_bytes_and_closes_writer[True]`
fail because its CPU campaign fixture has no `server_node`, while the new
formal_server guard reads `campaign['server_node']`. Please set `server_node`
to `local` in that CPU stub (or make the guard tolerate older fixtures).

The first H100 spatial smoke passed Cosmos native parity, but both B episodes
were rejected before inference: P6 teacher_tasks used debug.validate.digest
(bare hex), while R17LiberoEpisodeRunner.checked_setup uses benchmarks.spec.digest
(`sha256:` + hex). CPU inspection on timan108 confirmed the exact transported
B state bytes, shape/dtype and task language matched the witnesses. To complete
the requested smoke, D changed only teacher_tasks' digest import to the worker's
authority. No guard bypass or state/content change. A producer-side CPU test
now checks that namespace and the unchanged raw state SHA. Failed evidence:
`/data/r17_library/cosmos_libero_libero_spatial/campaigns/cosmos_libero_library_spatial_smoke_v1`.

Follow-up: P6's concurrent test update has resolved the server_node fixture.
D also aligned the B-plan assertions with the worker hash authority and made
the old Cosmos negative-admission fixture explicitly unsupported, preserving
that guard test while admitting the new supported producer. No production
bundle-capability checks were weakened.

The retry collected both episodes and22 decisions with no errors, then CPU
finish found the same hash namespace mismatch in build_teacher_parent's
alignment check. D aligned only that comparison and library_alignment's
task_data_identity helper with benchmarks.spec.digest, and updated the synthetic
search fixture accordingly. The original closed captures were recovered with
the published CPU --finish path; no GPU rollout was repeated for this fix.
`cosmos_libero_library_spatial_smoke_v2/LIBRARY_PIPELINE.json` is now PASS.
