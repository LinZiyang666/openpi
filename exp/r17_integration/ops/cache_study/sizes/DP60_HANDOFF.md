# DP size lane: CPU preparation for coordinator dispatch

`/data/mw_finetune/eval/cszmdp` now selects the fresh sealed campaign
`/data/mw_finetune/eval/cszpd`: local=60, DP replicas=4, sizes
50, 100, 200, 300, 400. The earlier sealed `cszmd` copy remains available.
The coordinator owns campaign process control. This preparation did not arm,
launch, restart, stop, or signal a campaign or lane. The coordinator reported
that the old waiter PIDs 1130348/1130349 were stopped and that it re-armed H100;
the earlier shared ARMED/CAPACITY_HANDOFF records are historical.

Reservations for this copy, and the same requested values for any other new
lane: RAM 3.5 GiB = 3584 MiB per worker; renderer 0.7 GiB, rounded up to
717 MiB per worker; global RAM headroom 8 GiB = 8192 MiB. RAM accounting takes
the maximum of reservation and measured RSS for each live worker and retains
reservations for owner-held slots between recycled generations. MemAvailable
already includes RSS, so only the unconsumed reservation is additionally
charged. Existing larger declared reservations remain accounted for. The
explicit `worker_ram_reservation_mib` in the sealed worker manifest controls
admission even when the unchanged Q fleet helper advertises its legacy 4 GiB
floor in a receipt. The runtime admission report records the effective policy.

Scientific worker counts and batching metadata remain canonical for comparison
to the paired control; `execution_workers=60` and `execution_worker_nodes`
control the actual driver/fleet. Replica count changes service capacity without
changing cached actions. Native runtime, simulator, client, official-A episode
plan, all B library views and fits remain pinned. The 70% replica CPU target
requires the coordinator's eventual live measurement.

The current workers make the live RAM/renderer previews fail, so preparation
defers admission until predecessor PASS and resource release. There is also
an independent GPU limit: the existing guard still reserves 8192 MiB server
growth and 4914 MiB (10%) GPU headroom on the 49140 MiB card. Sixty renderer
reservations require 43020 MiB; with those reserves the total is 56126 MiB
before any other use. Even an otherwise empty card admits at most 50 workers.
The 60-worker lane therefore continues waiting under the current GPU guard.
Changing those additional GPU reserves requires a separate coordinator decision.

Exact coordinator command (not executed by this preparation):

```bash
nohup bash /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/sizes/run_4090.sh > /data/mw_finetune/eval/cszmdp/COORD_ARMED_60.log 2>&1 < /dev/null &
```

The script requires `csl10dp` PASS, closed predecessor campaign receipts,
ownership/lock release and predecessor process-generation exit. It then waits
for disk, RAM and renderer admission before dispatch. It preserves reservations
of older H100 local workers while they remain active. The existing >=8 GiB disk
guard and cleanup after verified local sealing remain in force.

CPU validation and tests are recorded in `evidence/dp60_cpu_validation.json`,
`evidence/dp60_tests.log`, and `/data/mw_finetune/eval/cszmdp/CPU_PREPARED_60.json`.
The unchanged real-library audit covers 240 configurations and 960 queries.
The sealed CPU driver reproduced all 40000 original-A episode cells exactly;
233 native/scientific files and 54 unchanged worker payload files were verified.
Regression checks: 182 tests and 29 subtests passed, with Python compilation and
shell syntax checks also passing. Code changes are in `worker_memory.py`,
`local_workers.py`, `formal_eval.py`, `formal_prestage.py`, `queue_retry.py`, and
`cache_capacity_lanes.py`; reservation regressions are in
`tests/test_size_worker_reservations.py` with existing fleet fixtures updated.
