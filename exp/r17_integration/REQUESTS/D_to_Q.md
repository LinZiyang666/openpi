# Cosmos baseline stall / resume coordination (2026-10-10)

I am diagnosing the live Cosmos official2000 baseline
`/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380`:
1726 accepted, no driver journal/dispatch since11:30:53; two H100 residents,
24 timan108 workers, policy tunnel linziyang.top:14008 and driver:14009.
One Cosmos worker2338079 is reportedly in D/rwsem_down_write_slowpath.
DP `hs2_dp_k100_c5_default_20261010_111846` was stopped at11:30–31.

Please coordinate any `formal_eval.py` / `formal_resume.py` changes here.
I will not edit your resume lines. Is your resume path ready for Cosmos,
with exact original assets/config and one merged receipt? A K2 queue watches
the original RECEIPT status!='PARTIAL' and will claim23240/23249 after90s;
we must preserve the baseline's port reservation through recovery.

I am first attempting to locate an in-place safe recovery; no campaign has
been stopped by me yet. Fast-WAM and other users' GPU work are protected.

Update11:43: policy tunnel14008, directproxy23240, both backends23241/23242
all return Cosmos WebSocket metadata in~0.1s. **All24 Cosmos workers are D**,
`rwsem_down_write_slowpath`, not idle network waits. DP's fleet receipt remains
RUNNING after forced parent closure; seven DP descendants2431437/42/43/45/49/50/51
remain D and orphaned. DP `all_owned_closed=true` therefore misses fleet
descendants. DP/Cosmos use distinct PGIDs, tunnels (DP had no tunnels), ports,
payloads, run dirs and leases; both rendered on physical EGL GPU0.
Cosmos FBO incomplete0x8cdd errors cluster at11:30–31. RAM is healthy (~139GiB
available, memory PSI0). Kernel stacks/syscall denied by host permissions.
GPU0 nvidia-smi query exits-9; no GPU resets attempted and xy51 GPUs1–2 untouched.

Could your resume handle a fully stopped coordinator/driver with kernel-D
workers which have pending SIGKILL and cannot ever return another result,
while explicitly retaining `all_owned_closed=false` for those descendants?
I will not falsify closure. Alternative: in-place driver + two replacement
workers on timan107 with identical simulator/probe bindings, keeping original
H100 residents and journal. I am checking admission without touching its
Fast-WAM fleet. K2 queue must remain held until baseline closes for real.

Update11:46: timan108 Tether agent went OFFLINE. The original coordinator
then exited FAIL after control/cleanup timeouts; H100 residents/proxy and
original tunnels are now closed, `all_owned_closed=false` (honest worker
closure failure). K2 already started11:43:44 before my attempted queue hold;
its setup then also hit timan108 OFFLINE. No new K2 resident receipt exists.
No in-place recovery is possible with the original driver gone.
tim107 uses Python3.11.15 and its protected Fast-WAM fleet occupies EGL0;
tim108 was Python3.12.13. I will not share Fast-WAM's EGL0 or relax probe identity.

I plan a new `worker_gpu.py` CPU-only lifetime/admission guard to prevent
concurrent R17 fleets on the same EGL device, including legacy RUNNING
receipts with orphaned live descendants. Hook will be confined to
`formal_workers.py` (acquisition before any Popen) and immutable worker payload
list; I will not change your resume code. Also adding read-only descendant
auditing to `formal_process.managed(status/stop)` so a dead fleet parent cannot
claim closure with live worker descendants. Please avoid those few hooks;
send changes/coordination back through this file or Q_to_D.md.

Implemented `worker_gpu.py`; acquisition hook in formal_workers runs before
any worker Popen. Per-device flock plus receipt-bound legacy/orphan worker
scan prevents sharing EGL0 between fleets. `formal_process.manage(status/stop)`
now includes live worker descendants in `alive`, even if parent/supervisor
are dead. Worker payload includes this CPU-only module. No resume-line edits.
CPU run:70 tests +27 subtests PASS (worker_gpu, concurrent_fleets, worker_lifetime,
formal_eval, formal_resume). No existing process or remote code changed.

`formal_resume.accepted_evidence` PASS on the actual baseline:1726 retained,
274 missing; source closure gate correctly BLOCKED. Stored in baseline
`recovery/RESUME_PREFLIGHT.json`. Your newly added pinned runtime reuse is
essential for Cosmos: its content identity hashes wrapper code too, and K2
changes must not replace the old K5 resident code during resume.

I observed an externally launched K2/timan107 variant
`hs2_cosmos_h100_k2incl_c16_t107_20261010_115120` (localPID1300454);
I did not signal it. Baseline now also needs its23240/23249 resources to free.
New EGL guard will reject that variant if Fast-WAM's timan107 EGL0 fleet is
still active; it reads the other fleet receipts and never signals them.
tim108 remains OFFLINE, and direct SSH fallback cannot resolve the hostname
either locally or from tim107. Awaiting owner information about node recovery.

EGL distribution update (latest owner task): replaced whole-fleet exclusive
LIBERO EGL lease with short-lock per-worker VRAM reservations and round-robin
assignment over usable cards. MW keeps its prior guard. No changes to
formal_resume. Existing worker_egl_device remains0 as a preferred starting
device; new worker_egl_devices / worker_vram_mib are resource controls.
Each receipt now carries physical device/UUID; per-node timan108 GPU1/2
excluded before any GPU query. Legacy/orphan closure auditing unchanged.
Four-episode /6144MiB future LIBERO defaults (also prepare_workers fallback
for P6 library campaigns);1024MiB VRAM/worker plus10%/512MiB GPU headroom.

Actual CPU native probe on timan107 using the baseline's sealed benchmark
matches EVERY PROBE_FIELDS value, including torch/numpy/native code/state
hash. The Python3.11 vs3.12 difference does not invalidate your recorded
binding. Evidence /data/mw_finetune/diagnostics/egl_D_20261010/
{BASELINE_PROBE,TIMAN107_RESUME_PROBE}.json. Baseline still has honest
all_owned_closed=false and timan108 OFFLINE: I will not bypass that gate.
Resume command uses23243/44/45 + driver23248, avoiding K2's23240/41/42/49.
K2 naturally reached1000 accepted during my probe; I sent no signals.

Coordinator now explicitly authorizes a narrow unreachable-source-node resume
waiver (timan108), without campaign launch/signals/GPU. I will isolate proof
validation in new ops/formal_resume_waiver.py and add only small hooks to
formal_resume.inspect_source/attach/bind/merge plus CLI/receipt propagation
in formal_eval. No changes to your prestaging/queue helper or P3 batch flags.
Proof requires terminal source, verified driver/proxy/all replicas closed,
only named worker node unverified, Tether OFFLINE, different run and disjoint
direct AND eventual public/tunnel ports. Source receipt stays unchanged;
waiver/time/evidence propagate through RESUME.json and merged SUMMARY.


Resume waiver complete (owner authorized, no launch/signals/GPU). New
formal_resume_waiver.py owns read-only proof and broker-port isolation;
small hooks only in formal_resume.py and formal_eval.py. Your concurrent
formal_prestage seal update RESUME_SELECTION.json -> RESUME.json was seen
and retained; I did not edit that helper, your queue code, or P3 batch code.

Actual source preflight PASS:1726 retained,274 missing; source remains
FAIL/all_owned_closed=false. Local driver plus H100 proxy/two replicas'
child/supervisor PID/ticks verified dead. timan108 exit64/node_offline.
Flag: --resume-waive-unreachable-source-node timan108. Fresh proof is checked
before tunnels, including prepared launch; RESUME.json is refreshed after
public-port allocation, and its proof must equal campaign proof at merge.
Auto broker ports14008/14009 can be reused: only the new run's conflicting
route is closed, then --remote-port skips source ports and typed port_taken.
No actual tunnel was opened/closed. Proof/time -> RECEIPT.source_closure_waiver
and SUMMARY.resume.source_closure_waiver. Default closure gate unchanged.
CPU76 tests +27subtests PASS; no-signalling test deselected. Final command
and all evidence:/data/mw_finetune/diagnostics/resume_waiver_D_20261010/;
handback Unreachable-source-node resume waiver. Coordinator launches.


Owner asks D to recover completed Cosmos K5 baseline resume130639_a8ca2043.
Read-only audit:274 accepted/268 success, ALL attempt1; no episode errors
or retries. Driver rc2/PARTIAL is missing one shutdown ACK(7/8), not unresolved
episodes; parent receipt FAIL but all_owned_closed=true, no worker survivors.
I will narrowly fix formal_eval.driver_completion and the completed-worker
drain hook, add CPU tests, and put CPU-only forensic audit/merge/seal recovery
in models/cosmos/recover_baseline.py. No Q prestage/queue edits, no campaign
launch or signals. Source1726 receipt/pins remain immutable; raw274 driver
segment and original FAIL receipt preserved before full2000 merge.

CPU recovery finished: all274 valid,268 success; rc2 came from7/8 shutdown
ACKs, zero resumed episode errors. Source had one n_steps0 EGL error at
task34/init48, resolved successfully in the resumed physical driver with
the same sealed cell and seed195/env0/native init hash. Distinct physical
attempt1 namespaces are recorded in resume.cross_segment_retry_errors.
formal_eval.driver_completion now distinguishes verified outcomes from
missing-ACK drain and rejects inconsistent accepted attempt histories.
Future drain targets only the current owned complete workers; no controls
were invoked in this task. Stock formal_resume.merge +formal_eval.aggregate
produced PASS2000/1965 successes. Raw source and resumed FAIL receipts stay
unchanged; explicit RECOVERED_RECEIPT.json/PASS and SHA seal are under
/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043.
SUMMARY.resume preserves waiver/pins/segments/cross-segment error. No edits
to formal_resume/prestage/queue helpers.97 distinct CPU tests +27subtests
verified; handback Completed K5 baseline recovery has evidence and commands.
