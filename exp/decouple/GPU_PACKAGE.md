# GPU qualification and benchmark package

The owner's 17:20 smoke reused all four strict B=1 gates and completed GR00T
current/1 end to end at **23.12 episodes/min**. Split/8 failed before front
startup because its Unix socket address exceeded Linux's 107-byte pathname
limit. This CPU-only follow-up fixes address allocation and tests fresh-exec
startup for both models. No GPU, TCP model server or remote launch was performed;
only CPU model stubs and private local Unix IPC were used. Main-tree files and
trace inputs remain read only.

## Next-window commands

Run the end-to-end smoke first, once the whole 48 GB card and worker fleet are free:

```bash
bash /home/weiland/projects/openpi_decouple/exp/decouple/run_gpu_package.sh --smoke --new --deadline "$(TZ=America/Chicago date -Iseconds -d '+10 minutes')"
```

Expected **5–8 minutes**, hard cap **10 minutes**, including two minutes of
shutdown reserve. It reuses all four unchanged passed gates, then runs GR00T
**current/1 → split/8**, 20 accepted B episodes each, ten separate warmup episodes
per arm, 64 original timan107 workers and the frozen spatial/100/k1 method.
Both arms share a two-per-task cohort drawn from the full manifest. It exercises
model startup, worker transport, relay attribution, accepted-attempt filtering,
durable sampling, collection and `results/gpu_smoke/SUMMARY.md`. A missing cached
gate fails before model/worker launch; smoke never requalifies silently. Slow
startup/worker behavior can still cause a deadline stop with a partial summary.
Current/1 took 151 seconds total in the real smoke; split throughput and its
real front startup remain unmeasured. Source hashes changed, so use `--new`
after this fix; the four cached numerical gates still skip.

## Split startup repair

The supervisor allocates a unique **0700** directory named `opi-<runtime-hash>-<random>`
under a usable private `$XDG_RUNTIME_DIR`, falling back explicitly to `/tmp`
(ignoring long `TMPDIR` values). The socket pathname is bounded to 107 encoded
bytes and has mode **0600**. GPU bind and every front connect read the same
0600 `stage_endpoint.json` pointer in the launch runtime. The pointer is also
retained in the arm's `split_logs/stage_endpoint.json` for postmortems. Owned
children stop before the private socket directory is removed, on normal exit,
startup failure or TERM. The persistent pointer is historical after shutdown.

Readiness checks require a blueprint and an actual connectable Unix socket;
file existence alone no longer releases fronts. Websocket readiness/metadata
waits are bounded, port ranges and positive startup timeouts are checked, and
the children's TERM grace is ten seconds for the group. Package backend
23400 and front ports 23401–23416 explicitly bind **127.0.0.1**; the public
relay/driver remain **23100/23179**, with occupied-port preflight retained.

Fresh fronts add the GPU's shared `HF_MODULES_CACHE` to their import search path
before blueprint loading, so `trust_remote_code` processor classes are available
without first loading a model. Stale inherited front IDs are removed from the
GPU environment. The existing vendored cloudpickle and GR00T-only imports are
checked against GR00T's actual site-packages with JAX imports forbidden. Dry-run
affinity now remains exactly `10-17,54-61`; CUDA stays hidden.

The CPU startup check uses one fresh stage child and eight fresh front children
per model beneath a runtime parent longer than 240 bytes. It executes the real
launcher, role CLI, adapters, blueprint export/load, StageService bind and
StageClient/memfd tensor exchange, then front/proxy readiness and child cleanup.
Model loaders and websocket listeners/peers are stubs; no TCP ports are bound.
The GR00T blueprint includes an import-by-name dynamic processor to test cache
visibility. This proves startup plumbing; the next real smoke still qualifies
actual model/processor construction, worker traffic and split actions.

For the full comparison in the next two-hour reservation:

```bash
bash /home/weiland/projects/openpi_decouple/exp/decouple/run_gpu_package.sh --new --deadline "$(TZ=America/Chicago date -Iseconds -d '+2 hours')"
```

Budget **about 100–116 minutes** conservatively with cached gates; the recovered
GR00T one-copy rate could bring it nearer 90 minutes if it repeats. Screening
order is **GR00T current/1 → current/max → split/8 → split/4 → split/16**, then
π0.5 in the same order, then two additional alternating best-versus-best pairs
per model (split→current, then current→split). This is 18 arms, 5,400 measured
episodes and 180 warmups. The original 116-minute plan assumes GR00T rates
10/65/65/50/65 and π0.5 rates 40/100/100/80/100 episodes/min plus 15 minutes of
overhead. Those remain conservative planning assumptions; only GR00T current/1
has been measured in this package. Startup/warmup and worker teardown are excluded.

Replace the generated deadline with the owner's actual handoff time if earlier.
`--deadline` overrides `OPENPI_DECOUPLE_DEADLINE`. Absent both, the original
**2026-10-08 15:15 America/Chicago (CDT, 20:15 UTC)** remains the default; it must
be overridden for another window. ISO offsets/Z work; zone-less times use Chicago.

## Qualification and reuse

Four `qualify_gpu` proofs cover π0.5 k=1/2 and GR00T k=1/2 at B=1, 16 rows,
explicit noise and strict raw-bit equality for three keys and actions plus equal
donor ordering. π0.5 uses `pi05_libero`, `pi05_libero_pytorch`; GR00T uses
`/data/ckpt/n15_libero_spatial` and its original interpreter/PYTHONPATH. These
are model-stage proofs, with adapted π0.5 B camera fixtures for GR00T; they do
not claim full split launcher or closed-loop equivalence qualification.

Shared proofs live in `exp/decouple/results/gate_cache`, independently of
package state and directories archived by `--new`. `PKG_GATE_SKIP <gate>` means
an unchanged proof was reused. Fingerprints cover model/stage/key/transport
code, checkpoint weights/configs, norm stats, recorded donor keys and actual 16
prepared observations, interpreter/environment/dependency versions, k, B and
noise seeds. Benchmark/driver/relay edits and episode-cohort changes do not
invalidate numerical proofs. Supervisor/address-allocation code is likewise
outside the qualifier's import path. A CPU migration compares every other old
and new descriptor field before reindexing existing proofs; any stage, adapter,
transport, weight, fixture or environment change refuses migration. Original
entries and receipts are kept under `gate_cache/startup_migrations`, preserving
the real smoke's bound GPU UUID/driver. Full mode qualifies missing/changed proofs; smoke
fails instead. Device reuse checks host, slot, name, UUID and driver.

The owner's pre-fingerprint proofs lacked UUID/driver. They were explicitly
migrated using this owner report, the preserved window source hashes and
checkpoint paths/mtimes; they bind to UUID/driver on first reuse. Their JSON is
copied into the shared cache, so archiving original outputs cannot lose it.
Weights were SHA256-hashed during CPU preparation. Digest reuse requires
unchanged inode/size/mtime/ctime; small or recently modified files are rehashed.
`results/gpu_window_original_config.json` preserves the window configuration.

## Assets, worker and allocation

`assets/pi05_libero/physical-intelligence/libero/norm_stats.json` is now a regular
local file, SHA256 `c0ee3c1a6ca5f1414d31180af3bc333dca1d78cf912f341943191b393fe68707`.
Generation stages it from the main tree read only when absent; preflight checks
structure/hash before GPU work. The owner's symlink is retained under results;
no writes followed it into the main tree. GR00T keeps π0.5 batching/JAX imports
inside the π0.5 qualifier branch. Missing cloudpickle is still bundled as the
unmodified BSD-licensed 3.1.2 package under `_vendor`, only on GR00T PYTHONPATH.

**No worker bytes need deployment.** Reuse the original R16 island
`/scratch/zixuans8/openpi_r16_conductor_v1`. Live island/file SHA and fleet-lock
checks run only in the owner GPU window. Launch retains R16's default four
30-second tether attempts; its fleet lock refuses duplicates after lost replies.
The existing detached agent uses a 15-second heartbeat and 90-second expiry;
only the package's own lease is revoked. Local orchestration locks also remain.

Every arm uses a common **46,000 MiB** budget and B=1, wait=0. Current copies
start sequentially and prime stages on a recorded B fixture. Admission checks
warmed occupancy plus the upper R16 per-copy footprint (6.5 GiB GR00T, 10.5 GiB
π0.5); expect 6–7 GR00T / 4 π0.5 copies, with actual counts in `fleet.json` and
pinned on repeats. Split holds one weight copy with 4/8/16 CPU fronts. CPU mask
is `10-17,54-61`. Both models use frozen R16 spatial/100/k1 task levels; no refit.
The full B manifest is seed 2026100801, 30 non-fit inits per task, identical
across each model's arms/repeats. Ten disjoint warmups exercise every frozen task
branch; task 0 exercises all GPU stages. No R16 outcome/A journal is reused.

Public/driver ports are **23100 / 23179**; local backend/children **23400–23416**.
Occupied-port checks refuse existing listeners without touching them. Ports
23180/23198/23199 are never allocated by the package.

## Deadline, stop and resume

Hard end is the earlier of the explicit absolute deadline and start + two hours
(full) / ten minutes (smoke). Active work stops **120 seconds earlier**. The
wall-clock watchdog interrupts blocked gates/preflights/arms. Forward clock
adjustments shorten time; backward adjustments never extend the original window.
All newly created local model/relay/driver groups receive TERM together. GPU/relay
groups receive KILL within ten seconds; driver lease cleanup gets at most 100
seconds, bounded by hard end with five seconds left for reaping/summary writes.

Admission estimates a full cohort at the slower of planned/observed rates, plus
60 seconds setup and 100 seconds cleanup, also considering observed total wall
time and adding 15% headroom. The separate 120-second reserve stays protected.
An arm that cannot fit prints `PKG_ARM_SKIPPED_DEADLINE <arm>` without starting
processes or an attempt root. Repeat admission budgets its whole missing pair.
No cohort is shortened. The watchdog stops optimistic estimates that overrun.

`PKG_DEADLINE_STOP` (exit 2) retains completed-arm summaries and skipped-arm
reasons. Ordinary failures use `PKG_GATE_FAIL` / `PKG_ARM_FAIL <arm>` (exit 1);
complete success uses `PKG_DONE` (exit 0). Press Ctrl-C, wait for owned shutdown,
then rerun **without --new** to resume. Completed arms skip; incomplete arms
restart the full cohort in fresh `tryNN` roots. Use a new explicit deadline for
another reservation. Source/config hashes guard arm resume; `--new` archives
local package outputs but keeps all trace roots and the shared numerical cache.

## Durable collection and re-analysis

New arms write `arm.json`, `startup.json` and flushed `resource_samples.jsonl`
before collection. Relay rows learn UID/attempt from the worker's actual
`episode_start.__extra__`, carry connection and monotonic ingress/completion
identity, and clear episode identity at episode-end. The accepted journal
attempt and measurement window fence latencies; warmup, controls and rejected
attempts are excluded. The runtime and offline command share the same collector.

Re-analyse the owner's completed arm without rerunning it (already executed in
this follow-up; the input root is read only):

```bash
cd /home/weiland/projects/openpi_decouple
taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -B -m exp.decouple.collect_gpu --arm-root /home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_132449_a32ab1_groot_initial_current_1_try01 --allow-legacy
```

`results/gpu_reanalysis/{metrics.json,SUMMARY.md}` reports **300 episodes in
580.202 s = 31.0237 episodes/min**, 23 rollout failures, zero accepted execution
errors. The 6,897 accepted-cohort resident `after_infer` policy timings give
**1,829.333 ms p50 / 7,838.896 ms p99**. This is a different latency boundary
from the relay, explicitly labeled as legacy recovery. All 7,135 legacy relay
rows lack UID/attempt/timestamps; their unattributed p50/p99 is 3,893.239 /
9,667.037 ms, including warmup/rejected attempts. Interleaving and a retry prevent
a defensible order-based join. GPU/host CPU/serving CPU means and full-run VRAM
peak are **unavailable**: the old memory-only sampler was lost on collector
failure. `fleet.json` gives only a **6,618.188 MiB startup peak lower bound**.
Missing values remain null; new normal collection requires durable telemetry.

Normal full results go to `results/gpu_comparison/SUMMARY.md`: episodes/min,
GPU/host CPU/serving CPU utilization, peak VRAM, decision p50/p99, rollout
failures/execution errors and paired ratio/spread. Smoke uses `results/gpu_smoke`.
Fresh `decouple_bench_*` trace roots hold sealed groups/manifests, journals,
latencies, samples, logs, census and leases. Existing R16 roots remain read only.
Initial best-pair selection is screening-biased; alternating repeats show
spread, not a confidence interval. Partial screening is labeled explicitly.

## CPU preparation evidence

```bash
bash exp/decouple/run_gpu_package.sh --dry-run
bash exp/decouple/run_gpu_package.sh --smoke --dry-run
```

Full/smoke reports are `results/package_validation.json` and
`results/package_validation_smoke.json`; argv and replay artifacts are separate
under `package_dry_run/` / `package_dry_run_smoke/`. Both validate bash syntax,
12 module helps, actual serving parsers, norm assets, frozen B identity, hashes
of deployed worker metadata/client serializer code, and the qualifier's actual
GR00T imports with JAX absent (using the required main interpreter with GR00T
site-packages; no loader/GPU calls). Captured `worker_frames.msgpack` uses recorded
B observations through the actual deployed serializer. Two interleaved memory
transports replay those bytes through relay, collector and SUMMARY, fencing
warmup and rejected attempts. Both dry-runs also perform the fresh-exec CPU split
startup check above. Evidence is under each `package_dry_run*/cpu_startup/`
model directory: `report.json`, `launcher.log`, front/GPU logs and retained
socket pointers. Only private Unix IPC listeners with CPU model stubs are created.

**65 package tests pass** (`results/package_tests.xml`), including 15 startup,
socket permission/fallback/uniqueness/cleanup, long-path, port and cache-scope
proofs, plus collection/recovery/deadline/resume coverage. The complete decoupling
suite also passes **100 tests** (`results/decouple_tests.xml`). To regenerate after
source edits, use the CPU prefix and main interpreter with
`-m exp.decouple.package_config` and again with `--smoke`. Qualifier input hashes
remain separate from package hashes. The shared gate-cache inspection command is
`-m exp.decouple.gate_cache`; ordinary summary is `-m exp.decouple.summarize_gpu`.

Prepared package SHA256: full
`cdd4220df01878e9835e1b9fcc43d16eb74bf5835fb3847d0a8d832678bc000f`, smoke
`9dd396df3b7e7513d20015c91386e6d3e1ab748214ff6a203626fbca24ad8ee1`.
Python serving/package sources and the shell launcher are sealed; numerical
proof fingerprints remain independent of startup-only source changes.
