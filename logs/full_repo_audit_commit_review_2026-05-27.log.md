# HEAD commit 审查报告 2026-05-27

> Status: Fix Applied
> Time: 2026-05-27 10:10 CDT
> Authority: Review
> Review type: Audit / G2-style code review
> Target: `6a2a2c0100dbf9dc4fb05dcfe04b1f01e43bb914` (`Full-repo audit fixes: serving/conductor concurrency, cache, compliance, docs`)
> Scope: review the last commit's code, tests, docs, and audit-log claims; no source-code fixes applied in this review round.

## Verdict

Initial review verdict: NEEDS REVISION.

Post-approval status: FIX APPLIED.

The commit fixes several real defects, and the related targeted test subset passes. However, three issues should be fixed before I would approve the commit as the final state of the full-repo audit:

1. Multi-replica public ingress still has unbounded WebSocket frame size.
2. Stage 3 bucket-first metrics can still kill the stage worker thread.
3. Several docs/index entries still claim the old "bit-identical pre-Phase-5" baseline despite the commit's accepted sdpa-baseline drift.

## Post-Approval Fixes

After owner approval, the three blocking findings and the manual parity-test drift were fixed:

- `src/openpi/serving/replica_proxy.py`: public multi-replica router now uses the same 256 MB WebSocket frame cap as `WebsocketPolicyServer`.
- `src/openpi/serving/batching_coordinator.py`: bucket-first stage 3 metrics and generic stage 3 sub-bucket metrics are guarded so recorder failures are logged as non-fatal.
- `docs/deployment/libero.md`, `docs/deployment/aloha_sim.md`, `docs/experiments/conductor_tutorial.md`, `docs/README.md`, `logs/README.md`: stale C1 wording now says the path preserves raw single-connection structure but uses current sdpa numerics, not historical eager bit-identical semantics.
- `tests/cache/test_llm_layer_extract_parity.py`: manual parity test no longer forces eager and now mirrors the current model backend.
- Regression tests were added to `tests/serving/test_replica_proxy.py` and `tests/cache/test_serving_optimization.py`.

## Findings

### Blocking 1 — Replica proxy public ingress still allows unbounded frames

- Evidence:
  - `src/openpi/serving/websocket_policy_server.py:493-500` now caps single-server frames at `256 * 1024 * 1024`.
  - `scripts/serve_policy.py:668-672` starts `run_proxy(public_port=args.port, backend_ports=internal_ports)` when `--replicas > 1`.
  - `src/openpi/serving/replica_proxy.py:510-513` still starts the public router with `max_size=None`.
- Why it matters:
  - In multi-replica mode the public ingress is the router, not a child `WebsocketPolicyServer`.
  - A large unauthenticated client frame can be buffered by the proxy before it is classified or forwarded to a capped backend.
  - This leaves the commit's stated DoS fix incomplete for the scale-out path.
- Suggested fix:
  - Share or duplicate the same 256 MB cap in `ReplicaProxy.serve()`.
  - Add a focused test that asserts proxy `ws_serve` receives a finite `max_size`.

### Blocking 2 — Stage 3 bucket-first instrumentation can still kill the worker

- Evidence:
  - Generic `_stage_loop_inner()` now protects the outer batch metrics block at `src/openpi/serving/batching_coordinator.py:685-739`.
  - The opt-in bucket-first path calls `_stage3_dispatch_loop()` at `src/openpi/serving/batching_coordinator.py:640-643`.
  - `_dispatch_stage3_bucket()` records two metrics events at `src/openpi/serving/batching_coordinator.py:597-620` without a guard.
- Independent repro:
  - I ran a read-only Python probe with `OPENPI_STAGE3_BUCKET_FIRST=1`, monitor level `BASIC`, and a recorder whose `record_batch()` raises.
  - Output showed `Exception in thread BatchingCoordinator-stage3-w0`, then `stage3_alive_after_first [False]`, and the next stage 3 submit raised `TimeoutError`.
- Why it matters:
  - The commit's liveness goal is that instrumentation failure must not kill stage worker threads.
  - That invariant is still false for the bucket-first stage 3 scheduler.
- Suggested fix:
  - Guard all recorder calls in `_dispatch_stage3_bucket()`.
  - Consider also guarding the internal stage 3 sub-bucket recorder call in `_run_batch()` so instrumentation cannot race with request success/error state.
  - Add a regression test with `OPENPI_STAGE3_BUCKET_FIRST=1` and a failing recorder.

### Blocking 3 — Baseline documentation still contradicts the accepted sdpa drift

- Evidence:
  - The commit records accepted risk M11 in `logs/full_repo_audit_2026-05-26.log.md`: the old eager baseline is not directly comparable to the current sdpa model.
  - `scripts/serve_policy.py` was updated to say non-concurrent numerics match the current sdpa model.
  - Stale statements remain:
    - `docs/deployment/libero.md:451-454`: non-concurrent "stays bit-identical to the pre-Phase-5 behaviour".
    - `docs/deployment/aloha_sim.md:140-145`: same stale claim.
    - `docs/experiments/conductor_tutorial.md:45`, `:294`, `:371`: same stale claim.
    - `logs/README.md:56`: index still says the full-repo audit verified `1717 pass` and `27 文件改动`, while the audit log later records `1721 passed` and additional files/rounds.
- Why it matters:
  - Operators can use these docs to compare new sdpa runs against old eager baselines, which the audit explicitly says is invalid.
  - The stale index weakens the traceability of the audit commit itself.
- Suggested fix:
  - Replace "bit-identical to pre-Phase-5" with "structurally same raw single-connection path; numerics match the current sdpa model, not the historical eager baseline".
  - Sync `logs/README.md` to the final audit-log numbers and changed-file count/scope.

### Non-blocking 1 — Manual LLM-layer parity test still forces eager

- Evidence:
  - `tests/cache/test_llm_layer_extract_parity.py:311` sets `language_model.config._attn_implementation = "eager"  # mirror Stage 2`.
  - The reviewed commit intentionally changed the keybuilder and offline matrix path to use current sdpa semantics.
- Risk:
  - This is marked manual, so it does not break the targeted CI subset, but anyone running it will validate the old contract.
- Suggested fix:
  - Update the manual parity test to compare under the model's current backend, or explicitly document it as a historical eager-only probe.

## Checklist

Initial checklist assessment:

- Architecture consistency: NEEDS REVISION. The single-server ingress cap is not propagated to the multi-replica public ingress.
- Interface compatibility: NEEDS REVISION. Stage 3 bucket-first remains opt-in, but when enabled it violates the coordinator liveness contract under metrics failure.
- Test coverage and passing: NEEDS REVISION. The targeted subset passes, but there is no regression coverage for proxy `max_size` or bucket-first metrics failure.
- Docs and indexes: NEEDS REVISION. Several user-facing docs and the log index still contradict the accepted sdpa baseline drift / final audit state.
- No regressions: NEEDS REVISION. The identified proxy and bucket-first paths remain vulnerable to the same classes of issue the commit claims to close.

Post-fix checklist assessment:

- Architecture consistency: FIXED. Multi-replica router now uses the same finite frame cap as the backend server.
- Interface compatibility: FIXED. Bucket-first stage 3 metrics failures are non-fatal and preserve worker liveness.
- Test coverage and passing: FIXED. Normal regression tests cover the proxy cap and bucket-first liveness.
- Docs and indexes: FIXED. C1 wording and index state now match the sdpa accepted-risk record.
- No regressions: FIXED within the reviewed scope; targeted regression and lint checks pass.

## Verification Run

- `PYTHONPATH=. uv run pytest tests/cache/test_serving_optimization.py tests/conductor/test_driver.py tests/conductor/test_agent.py tests/cache/test_config.py tests/cache/components/test_llm_layer_key_builder.py -q`
  - Result: `182 passed, 11 warnings`.
- Review-only evidence tests:
  - File: `tests/review_tests/test_full_repo_audit_commit_review_2026_05_27.py`
  - `.gitignore` now excludes `tests/review_tests/` so these probes do not enter the normal shared test suite before fixes are approved.
  - Command: `PYTHONPATH=. uv run pytest tests/review_tests/test_full_repo_audit_commit_review_2026_05_27.py -q`
  - Result on current HEAD: `2 failed, 2 warnings`.
  - Blocking 1 failure: `assert captured["max_size"] == 256 * 1024 * 1024` failed because `captured["max_size"] is None`.
  - Blocking 2 failure: `assert all(t.is_alive() for t in stage3_threads)` failed after `RuntimeError("recorder boom")` escaped from `_dispatch_stage3_bucket()` and killed `BatchingCoordinator-stage3-w0`.
- Post-fix verification:
  - `PYTHONPATH=. uv run pytest tests/serving/test_replica_proxy.py tests/cache/test_serving_optimization.py tests/review_tests/test_full_repo_audit_commit_review_2026_05_27.py -q`
    - Result: `72 passed, 4 warnings`.
  - `PYTHONPATH=. uv run pytest tests/serving/test_replica_proxy.py tests/cache/test_serving_optimization.py tests/conductor/test_driver.py tests/conductor/test_agent.py tests/cache/test_config.py tests/cache/components/test_llm_layer_key_builder.py tests/review_tests/test_full_repo_audit_commit_review_2026_05_27.py -q`
    - Result: `214 passed, 11 warnings`.
  - `uv run ruff check src/openpi/serving/replica_proxy.py src/openpi/serving/batching_coordinator.py tests/serving/test_replica_proxy.py tests/cache/test_serving_optimization.py tests/cache/test_llm_layer_extract_parity.py`
    - Result: `All checks passed!`

## Recommended Patch Set After Approval

1. Add a finite `max_size` to `ReplicaProxy.serve()` and test it.
2. Wrap bucket-first stage 3 recorder calls, add a regression test, and consider guarding `_run_batch()` sub-bucket metrics too.
3. Update stale baseline docs and `logs/README.md`.
4. Update the manual LLM-layer parity test comment/behavior.

Business-code and documentation fixes were applied after owner approval.
