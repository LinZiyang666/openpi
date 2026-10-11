# Q → P2: separate π-MW replan1 protocol for a physical G1 tail certificate

The current formal model/benchmark bind is replan5, native H5, commit≤5. It has
no formal free tail. Q's `slot_G1_pi05_mw_C_tail.sh` proves native controller
cursor/suffix bytes with five synthetic one-control acknowledgments, reports
the separate `mw_commit_h5_replan1` diagnostic controller protocol, and returns
PARTIAL. It does not invent applied controls or count this as formal SR.

Minimal additive proposal: supply a distinct `BenchmarkSpec`/protocol identity
with replan1, retaining the original MW kernel, seed/task data/camera identity,
and register that explicit protocol in the π model provider's allowlist. Emit
a separate episode manifest and control-count receipt. Reuse the existing
runner's original `run_episode(..., replan_steps=1)` support. Keep formal
replan5 unchanged. Q can then bind the C slot to that explicit protocol and
join actual execution/terminal receipts, as DESIGN's G1 demands.

Also, Q's generic cross-model MW driver adapts canonical observation keys into
P5's native `image/state` keys at a Q adapter boundary. No P-owned code changed.
This avoids changing P2's authoritative kernel for other models.
## Fix round 2 — stale GR emitter bridge assertion after P5 adoption

Q's post-owner full CPU composition run:568 passed,2 failed; receipt
`qualification/preflight/fix2_composition.xml`. One failure is your
`tests/conductor/test_p2_fix_round2.py::test_gr_default_yaml_emission_bridge_carries_table_and_preserves_options`.
P5's default_config now emits the correct canonical table, so line264's
`assert not original['recipe']['ordered_tasks']` is obsolete.

Minimal patch: assert original ordered_tasks equals FORMAL_SPEC's canonical
table and original task_table_identity equals FORMAL_SPEC.task_table_identity.
Retain the bridge preservation assertions; it should remain idempotent for
an already canonical default_config. No production guard should be weakened.
Q did not edit your fixture or the P5 emitter.



Q final B3/A3 refresh: the stale GR default-config/canonical reach-id fixtures
reported during the earlier composition run are resolved by A3. Current R17
composition648 PASS. No further dispatch is needed for those fixture changes.
Historical distinct replan1/advanced G3 scope requests retain their status.


## GPU fix 1 — major: subsequent probe-less pulls are rejected

The Q4090 GR B/current+60000 runs accepted their workers' full first probes,
then rejected subsequent pulls as package mismatch. WorkerLoop._pull sends
probe ONLY on the first pull per TCP connection (src/openpi/conductor/worker.py
~111–120); MetaWorldConductorDriver._record_pull unconditionally validates
payload.get("probe") or {} on EVERY pull (conductor/driver.py ~30).
Missing subsequent probe becomes {}, so package validation fails incorrectly.
This also blocks a final shutdown pull and produces repeated reconnects until
Q's phase deadline. Q did not modify P2 or frozen shared code.

Both deployed nodes use /scratch/zixuans8/mw_r17/envs/sim/bin/python 3.11.15
and packages metaworld3.0.0/mujoco3.14.0/gymnasium1.3.0 exactly. The deployed
MetaWorldBenchmarkAdapter.worker_probe() passes validate_worker_probe with
renderer/task-data checks explicitly omitted for the CPU-only diagnostic.
Historical run census.json has complete valid renderer/task-data probes for
both workers; rejected rows subsequently omit what was received. Evidence:
qualification/preflight/gpu_fix1_timan{107,108}_{import.log,packages.json,actual_probe.json};
/data/mw_finetune/qualification/runs/slot_G2_groot_mw_B_20261009_185017_419337/census.json.

Minimal patch proposal, owned by P2 (no validator version relaxation):

```python
# inside MetaWorldConductorDriver._record_pull, before validation:
with self._census_lock:
    entry = self._census.get(payload.get("worker_id"), {})
    previous = entry.get("probe")
    peer_text = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) and len(peer) >= 2 else None
    if "probe" in payload:
        probe = payload["probe"]
    else:
        if (previous is None or peer_text is None or peer_text not in entry.get("peers", [])
                or entry.get("server_key") != payload.get("server_host")):
            raise ValueError("worker first pull on this connection lacks probe")
        probe = previous
self._worker_probe_validator(probe or {})
if "probe" in payload and previous is not None and previous != probe:
    raise ValueError("worker probe changed during the campaign")
# retain super()._record_pull(payload, peer); no shared WorkerLoop changes
```

Keep rejected-probe capture including raw payload/probe, accepted previous
probe, peer and reason. Q's driver adds that logging in owned code now.
Regression: full valid first pull→probe-less same-peer second/terminal pull
works; first pull missing probe fails; a new peer/reconnect without probe
fails; explicit empty or modified probe fails; changed endpoint fails.
Use socket-pair/real stock WorkerLoop with CPU fake runner where possible.
Please dispatch before rerunning GR B/current+60000. Native GR A_k2/A_k1
can retry after Q's bounded-port fix independently. π parity diagnosis remains
with P2; Q has not changed pi_native.py.

Q follow-up: P2's separate REQUESTS/P2_to_Q.md parity patch is now adopted in
Q-owned pi_native.py and CPU-qualified. G1 A is ready to retry; this does not
resolve the subsequent probe-less MW pull admission bug above. The strict
connection-aware patch and socket regression remain requested from P2.

## GPU fix 2 — Q consumption / resolved

Q consumed P2's connection-aware driver and refreshed the worker/runtime
deployments. Fresh timan107/108 probes each pass first/full, second/absent and
terminal/absent pulls; seven strict rejection cases per node pass. Q33 and
conductor73 CPU tests pass. All34 preflights rerun,30 PASS/four expected PARTIAL.
This admission seam is resolved; prior real B PARTIALs require fresh GPU retries.
See qualification/receipts/GPU_FIX2.json and HANDBACK_Q.md GPU fix2.
