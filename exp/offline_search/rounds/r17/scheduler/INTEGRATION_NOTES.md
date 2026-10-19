# Where autoscale_1017 is wired in today (read-only survey, 2026-10-19)

Prefixes: `AS=exp/offline_search/rounds/r17/autoscale_1017`, `OPS=exp/r17_integration/ops`,
`FF=exp/offline_search/rounds/r17/filter_failed`, `FFK=exp/offline_search/rounds/r17/filter_failed_knob`,
`LS=exp/offline_search/rounds/r17/longseg_1017`. Line numbers are from commit 99227452.

## R17 formal path

Call chain:
- new formal batch: `OPS/formal_eval.py` `main` writes `autoscale`/`autoscale_maximum` (1991-2008) -> `prepare` (795) ->
  `defaults.apply` (797) -> `prepare_workers` (892; `apply` again at 440) -> per-node loop (511-668).
- ff/ffk continuation: `FF/resume.py` `build` (100-131, `apply` at 129) -> `FF/stage.py` `_stage` (104-203; `apply` 135;
  `driver_release.prepare` 176/180; `worker_resilience.prepare` 187) -> per-node `prepare_workers` with `payload_override`.
- longseg / re-prepare: `AS/formal_trial.py` `prepare` (82-161, `configure` or `apply(force)` at 107); `LS/vfix.py:300-305`
  `retry`, `LS/build.py:96,160` (`resume.topology`).
- launch: `FF/launch.py:53-67` (ffk: `FFK/launch.py:41-61`) -> `OPS/formal_prestage.py` `launch_prepared` (304):
  `resilience.select` (315) -> `worker_preflight` (379-425) / `launch_workers` (427) -> write `campaign.json` (430) ->
  `formal_eval.run_campaign` (1166) -> workers (1459/1571) -> driver (1566) -> main loop (1574-1584).

Fields (writer -> readers):
- `autoscale`: `formal_eval:2006`, `defaults.apply:88-101` (switch OFF deletes at 71-72; continuations inherit 84-97),
  `formal_trial.configure:32` -> `driver_release:61` (AS overlay into the driver runtime), `prepare_workers`
  442/494/598/626/635, `stage` 177/194, `spawn_local:1226`, `spawn_remote:1243`, `formal_workers:127`,
  `local_workers` 163/186, `timan1_gpu2` 81/102, `worker_resilience.measure:133`, `formal_prestage:417`,
  `DriverController:18`, `LocalGovernor:163`.
- `autoscale_node_maxima`, `autoscale_maximum`: `defaults.apply:134-137` (`desired_maxima` 35-45), `formal_eval:2008`,
  `formal_trial:33` -> `measure:155,168`, `DriverController:21`, `release.refresh_worker` (`AUTOSCALE_RELEASE.json`).
- `execution_worker_nodes` / `execution_workers`: `defaults.apply:135`; `worker_topology.redistribute:24-25` (called by
  `worker_resilience.record:197`, `Running.retire:318`, `resume.local_clone:40` / `topology:198`); `select:218`;
  `prepare:236` / `launch_workers:270` -> `prepare_workers:452,511`, `formal_prestage:388-389`, `open_tunnels:186`,
  `DriverController:20`, `timan1_gpu2.authorize:73`, `local_workers.authorize:68`.
- `autoscale_node_config`, `autoscale_external_legacy_nodes`, `autoscale_reactive_nodes`, `autoscale_mature_s`:
  `defaults.apply:105-111` (from `AS/formal_dynamic.py:29-31` `NODE_CONFIG`), `formal_trial:108-110` -> `LocalGovernor`.
- `worker_node`, `node_workers`, `autoscale_initial`: `prepare_workers:595-601` writes `<node>.json` (658), pushed as
  `stage/campaign.json` (668); `formal_prestage:386-391` sets `node_workers=execution_worker_nodes[node]`, writes
  `<node>_launch.json`, pushed as `launch_campaign.json` (420-425) -> `formal_workers` 60/112/146/149, `LocalGovernor`
  162/167/201, `fleet:164`, `timan1_gpu2.route:103-105`.
- env `OPENPI_AUTOSCALE_NODE` / `OPENPI_AUTOSCALE_CONTROL`: `prepare_workers:599-600` (into `WORKER_DEPLOYMENT.json`
  env, then the spawn request env) -> `formal_process.manage` (`driver.status_result:253`), `controls`
  (`driver.control_entry:260`).
- `worker_vram_mib`, `worker_server_vram_headroom_mib`, `worker_server_vram_floor_mib`: `defaults:141`,
  `local_workers.route:154-162`, `h100_workers.route:72-79`, `worker_resilience.local_route:519-521` ->
  `LocalGovernor._state:249-251`, `fleet.Slots` 23/75.

Driver side:
- `formal_eval.spawn_local:1226-1228` injects `OPENPI_AUTOSCALE_CONFIG=<dir>/campaign.json` and
  `OPENPI_AUTOSCALE_DRIVER_MODULE=...autoscale_1017.driver` for the driver only. `src/openpi/conductor/driver.py:186-189`
  imports it and calls `from_environment(self)`; 190-192 is the R15/R16 legacy branch (`OPENPI_WORKER_MEMORY_GUARD`).
  Hook calls: `_stage_loop` (306-307), `handle_progress` (415-424), pulls (456-478). Driver runtime overlay:
  `driver_release.prepare:61-72` + `AS/release.driver_files`.
- Control to nodes: (a) worker pull replies `_autoscale_control` (`src driver:478`) -> `Observer.response`
  (`AS/worker.py:107-117`); (b) launcher `formal_eval.refresh_remote_owned` (1113-1161, every `PROBE_SECONDS=30`) ->
  `status_entries` (`AS/driver.py:246-249`) -> remote `formal_process.controls` (145-154) -> `control_entry`, renew,
  `manage(status)` -> `status_result` reads `AUTOSCALE_NODE.json` -> `collect_status` (`formal_eval:1159-1160`) writes
  `AUTOSCALE_NODES.json`.
- Lease: `formal_process.supervise:83-87` stops a fleet whose lease is older than 600 s.
- `WORKER_NODE_STATE.json`: written by `worker_resilience.record:207`, `Running.retire:310-322`; read by
  `DriverController.tick:98`, `worker_resilience.install_driver:389-410` (driver patch: expected workers, shutdown of
  retired nodes' sockets, writes `WORKER_DRIVER_STATE.json` when all done; only when wrapped by `FF/launch.py:61` /
  `FFK/launch.py:51`), `Running.__init__:304`, `formal_finalize.py:430` (hash only).
- Done signal: today the controller publishes `done` when `all_done`. Without it, a plain formal_eval run only stops
  fleets via `drain_completed_workers` (exit 2 + `snapshot_incomplete`, `formal_eval:1596`, 985-994). Keep publishing
  `done` from BatchClient.
- `qualification/driver.py:83`: without an autoscaler the shutdown waits for `len(workers)==expected_workers`; with a
  dynamic worker count it would wait out `shutdown_seconds`. BatchClient must remain the autoscaler.
- Node failure handling (`worker_resilience` measure/allocate/retire/select/launch_workers/Running.check/cleanup) stays.

Node side (formal_workers):
- `formal_workers:126-129` builds `LocalGovernor(campaign, run)`; loop is `AS/fleet.run_fleet` (197-200).
- `fleet.py:113-175` every 2 s: `governor.tick()`, reap exited workers (`planned_recycle`, timan1 `wrapped`),
  `governor.finished`, `slots.release`, receipt every 30 s, done/drain exit, up to 16 spawns per round via
  `governor.permit()` -> disk admission -> free index in `range(node_workers)` -> `slots.acquire` -> `launch` ->
  `governor.register`.
- `worker_env` (`node.py:477-482`) + `formal_workers:163-165,171-173` (`OPENPI_AUTOSCALE_GPU_UUID`,
  `OPENPI_AUTOSCALE_PARENT_PID`, `AS/bootstrap.py` wrapper).
- GPU slots (`fleet.Slots` 15-98): inventory every 10 s via `worker_gpu.inventory`; `acquire` needs
  `total - used - pending - (max(512, 10%) + server_headroom) >= worker_vram_mib`; timan1 fills GPU2 first (82-84).
- Recycling: LIBERO `worker_max_episodes` 4 (`prepare_workers:623`), MetaWorld 32 / 1800 s (`defaults:142-143`,
  `local_workers:163-164`); `worker_lifetime --max-episodes` exits 75 with `RECYCLE.json`.

Routing:
- local vs remote: `remote('local')` executes directly (`formal_eval:66-67`); others via tether (h100 alias via
  `physical_node` 153-155). `worker_paths:404-419`; `worker_command:77-79`, `launch:149-154` (local and h100 use
  `local_worker_entry` with loopback policy). worker_id = `run_name_node_index` (60).
- 4090 `local`: `local_workers.route:136-167` (driver 127.0.0.1; loopback when server is local); headroom 5120 MiB, but
  `worker_resilience.local_route:517-522` overrides to 16384 when the server is local and floor 28672 when elsewhere
  (both obsolete: local workers now only serve a local server; 5 GiB server line per owner).
- `h100`: `h100_workers.route:66-80` (loopback to `127.0.0.1:public_port`, headroom 4096 MiB, EGL 0, cap 40).
- `a100`: same route, public endpoint, headroom 0, cap 80.
- `timan1`: `FF/timan1_gpu2.coordinator` (440-462), `authorize` (72-95), `route` (98-113: LIBERO 530 MiB VRAM / 2048 MiB
  RAM per worker, devices [2,1]); fleet runs inside bwrap (`formal_eval:1243-1244`, `FF/launch.py:42`, `bootstrap`
  399-406; `run('fleet')` 360-386; `install_runtime` 193-222 guards GPU0/GPU3); remote calls via `auxiliary` (409-432).

## R15 / R16 legacy paths

Both use the adapter `AS/legacy.py` (`LegacyDriver` driver hook, `AgentGovernor` node governor) via
`OPENPI_WORKER_GOVERNOR`; `AS/ENABLED` is `on`.
- R15 (pi05/GR00T full; `FFK/formal_a_queue.py:172-173`): controller on the 4090 (`vla_workers launch` -> `r15()`
  `FFK/vla_workers.py:114-133` -> `filter_failed.legacy chain` -> `closed_loop/ops/h100/control.main`; or
  `formal_dynamic.legacy_job` 127-157 -> `control.chain` 694) starts on timan107 per arm: tmux `h100/node.py supervise`
  (245-286) -> `run_arm.sh` (fleet lock 19-22; `vla_shared.configure_node` 43-51) -> `run_gtp_subset.py` (driver and agent
  in one process: `ConductorDriver`+`LegacyDriver`, `WorkerAgent`+`AgentGovernor`, `run_gtp.py:610-665`,
  `legacy.py:70-82`) -> workers `examples.libero.worker_entry`. Max workers: `legacy_release.stage` (53, 63-65,
  `LEGACY_AUTOSCALE.json`, called by `control._sync` 464-471), split per port as `--server-workers` (`control.py:848`) ->
  `run_arm --workers W` (`run_arm.sh:72,93`). `LegacyDriver` builds a `DriverController` after the first node snapshot
  (`legacy.py:26-36`); R15 driver root is `$OUT/autoscale` = node `OPENPI_AUTOSCALE_RUN` (`run_arm.sh:28`).
- R16 (reduced, cache study): controller and driver on the 4090: `formal_a.run_r16` (189-197) or `study.serve`
  (214-230) -> `legacy_release.coordinator` re-exec with overlay (88-106) -> `vla_workers.run_r16_selected` (192-241) ->
  `conductor_runner._run` (local model server, then `control.remote` nohup agents on timan107, 192-221) -> `run_driver`
  (`R16Driver`+`LegacyDriver`, `conductor.py:469-473,496-512`). On timan107 one `conductor agent` per group
  (`WorkerAgent`+`AgentGovernor`, `conductor.py:522-569`), lease renewed every 15 s (`conductor_runner.py:211-227`).
  Max: `run_r16_selected` sets `--workers` from `LEGACY_AUTOSCALE.json` (222-227), written by `prepare_r16_job` (304-314).
  Targets reach nodes via `AgentGovernor.exchange` supervisor pulls every 15 s (`legacy.py:84-100`, `src driver:456-460`)
  and worker pull replies.
- 4090 `attach_local.sh` (`/data/mw_finetune/qualification/r16_local_island/attach_local.sh`) is an unmanaged spawner
  (free VRAM > 6 GiB, free RAM > 40 GiB), used by `launch_fa_cs_gr_r.sh` and `*_gr_*_dyn.sh`: remove it from those.
- Legacy-only code: `AS/legacy.py`, `AS/legacy_release.py`, `formal_dynamic` `legacy_plan` (73-88), `proof` R15 branch
  (106-109), `legacy_job` (127-157), `dispatch` R15 branch (164); `node.external_legacy` / `WORKER_MODULES`
  (118-155, 339-353); hooks in `control.py` 120-123, 464-471, 829-834, 848; `h100/node.py` 268-269, 284-286;
  `run_arm.sh` 25-35; `conductor.py` 470-473, 531-541; `conductor_runner.py` 149-153, 171-174, 196-210;
  `vla_workers.py` 105, 192-241, 304-314; `vla_shared.py` 44; `study.py` 205-230.
- Batch keys today: R15 `root.name:arm` (changes per arm; use `root.name`), R16 `group_sha256`.
- R15 spec order: `run_gtp.py:571` groups specs by server and `agent.py:228-238` fills the lowest spec indices first,
  so a target below W fills the first server; the new governor must pick specs round-robin across servers.
- Queue: R15 cells fa_pi_f_u_l {l50,l100,l200,s50,s100}, fa_pi_f_f_l {l50,l100,s50,s100,s200,s500}, fa_gr_f_f_l
  {l50,l100,l200,s50,s100} (staged with old digests: must be re-staged), fa_gr_f_u_l (not built). R16 groups
  fa_cs_pi_r, fa_cs_gr_r (not prepared). `prepare_*_gr_*_dyn.sh` call `vla_workers prepare --job <unit>` which
  `lookup()` (vla_workers.py:30-32) does not accept (expects cell names).
