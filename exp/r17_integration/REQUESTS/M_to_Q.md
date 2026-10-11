# M → Q: urgent LIBERO kernel memory diagnosis (2026-10-10)

AGENT alert: C10 is running unchanged. At 11:50:14 CDT timan107 MemAvailable
was 55.89 GiB (down from 69.5 GiB at 11:48:50), AnonPages 53.75 GiB.
Linear extrapolation currently puts exhaustion at approximately 12:15–12:30
CDT, potentially before C10 completes. Coordinator decides campaign action;
M will not signal or stop anything. timan108 tether is OFFLINE (8 minutes).

I found your existing `worker_lifetime` implementation and eight-episode
formal_eval default. I will avoid editing ops files. I am collecting paired
kernel/process/episode snapshots and reviewing renderer cleanup; will append
the measured per-episode allocation, safe recycle interval and admission budget.

Needed from Q: retain the episode-boundary recycle option/default in formal_eval,
and use the measured host-memory budget including kernel allocations (RSS alone
misses the growth). Ensure updated worker payload includes any M cleanup files.

11:51 update: all 24 C10 worker PIDs are gone. The existing campaign receipt says
PARTIAL, `TimeoutError('campaign stopped or hard deadline exceeded')`, all owned
closed. The supervisor receipt records forced cleanup of the launcher (exit -9).
M sent no signals. Do NOT describe this as a confirmed C10 node OOM.
MemAvailable recovered from 55.89 to 216.27 GiB; AnonPages from 53.75 to 0.20 GiB.
Therefore ~105 GiB of the released 160 GiB was outside AnonPages.
Slab dropped only 1.51 GiB, VmallocUsed 0.92 GiB, PageTables 0.17 GiB.
NVIDIA kernel module: 535.183.01. /proc/vmallocinfo permission denied; NVIDIA
client allocation counters absent. Driver host allocations are the leading
inference, not a directly measured pinned-host byte counter.

Correction to the initial projection: supplied 11:45→11:48:50 slope gives
~12:28 CDT; 11:48:50→11:50:14 accelerated to ~9.7 GiB/min, which would give
~11:56 CDT if sustained. C10 ended independently by ~11:51 before this forecast.

Existing C10 attempts contain `mujoco.FatalError: Offscreen framebuffer is not
complete, error 0x8cdd`, starting on early episodes, and partially constructed
`MjRenderContextOffscreen.__del__` errors (`con` missing). Retain all good
episodes and exclude actual infrastructure-error attempts in resume.

GPU capacity finding (read-only /proc/driver/nvidia/gpus/*/information):
timan107 has EIGHT GTX 1080 cards, each 8 GB; all C10 workers used device 0.
User's ~7.8 GB GPU0 occupancy for 24 workers plus immediate 0x8cdd errors is
consistent with exhausting one card and NVIDIA host backing/retention. Do not
increase beyond 24 on GPU0. Your new worker_gpu changes should spread future
workers across devices, with a conservative per-card admission (~16 initially).
Host allocation still grows per episode, so retain recycling even when spread.

Observed post-C10 fleet (already running independently; M did not launch):
Cosmos 24 workers, --max-episodes 8, at 11:54:40; RSS ~2.6 GiB, Pss ~2.2–2.4
GiB per process, VmLck/VmPin/Locked all zero, NVIDIA FDs still 44/maps ~130.
These zero counters do not account for driver-owned host pages. timan108's
prior admission files establish 251.63 GiB MemTotal and 25.16 GiB headroom.

Initial recommendation (superseded by the task9 burst measurement below):
--worker-max-episodes8 --worker-memory-mib6144. A 4 GiB reservation is smaller
than Pss2.2–2.4 + eight episodes × fleet-average0.2–0.26 GiB plus baseline
driver/context and transient environment setup. N=4 with6 GiB gives
host-only clean-node capacity timan10732, timan10837 (~4 GiB OS footprint and
10% headroom). Actual timan108 capacity now0: offline/D-state status unverified.
80–90 workers exceed present measurements even with recycling: Pss alone
uses ~176–216 GiB before driver overhead/reserves. GPU0 is already full at24;
recommend16/device initially, distribute across identical GTX1080s for >16.

CPU validation added outside your files:
`exp/r17_integration/tests/benchmarks/test_libero_worker_recycle.py` verifies
native-kernel request/image/action bytes, original0..49 inits, seeds7/42/195,
paired/fixed seed rules and retry attempts across N=1 vs resident N=4 loops.
Together with your wire/lifetime/fleet tests and the read-only accounting tests:
10 passed. No native source or benchmark hash files were changed by M.

Read-only confirmation from your already-running Cosmos fleet: at11:54:40
MemAvailable144.11 GiB, AnonPages51.12; at11:57:41 MemAvailable149.43,
AnonPages49.59, residual18.02. All24 PIDs now new generations, --max-episodes8
still active. The memory recovery across generations supports the mitigation.
M did not launch/stop/deploy anything.

11:59:13 live receipt confirms24 planned recycles,0 worker failures; every
generation0 marker says completed_episodes8/maximum8, all24 workers generation1.
This validates actual process recycling/reclamation without M GPU work.

FINAL RECOMMENDATION after joining memory samples to dispatch timestamps:
--worker-max-episodes4 --worker-memory-mib6144. Please set formal LIBERO defaults
to4 and6144 (keep0 as opt-out), including the default assertion in your lifetime
test. If retaining N=8, reserve9216 MiB (9 GiB), yielding21/24 host workers.

Per-worker-per-episode allocation ESTIMATES, strongly task-dependent:
- 11:45:00→11:48:50: 74 new dispatches,6.7 GiB depletion =>0.091 GiB/dispatch;
  mostly tasks7/8 (31/43 dispatches).
- 11:48:50→11:50:14:25 new dispatches,13.61 GiB depletion with flat AnonPages
  =>0.544 GiB/dispatch; mostly task9 (18 task9/7 task8 dispatches).
- Whole-fleet residual106.2 GiB/468 dispatches at11:50 =>~0.227 GiB/dispatch,
  including fixed driver/probe/OS allocation. Average residual4.43 GiB/worker.
These are node-depletion estimates, not direct individual-driver counters.
Use0.6 GiB/episode for a conservative bound: Pss2.4 +4×0.6 +0.8 transient/
driver allowance =5.6 GiB, rounded up to6144 MiB. The burst makes6 GiB/N=8
too optimistic despite successful recycling in the earlier Cosmos tasks.

Final CPU check:21 passed including the existing native LIBERO bridge tests.
New read-only collector is `exp/r17_integration/diagnostics/libero_memory.py`;
it requires exact campaign/module IDs, records meminfo/status/smaps/PID start
ticks/FDs/maps, and refuses per-episode delta attribution across generations.
