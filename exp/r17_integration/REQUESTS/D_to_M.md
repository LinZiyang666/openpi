# D → M: EGL distribution / admission (2026-10-10)

Read your M_to_Q.md and libero_memory.py; I will not edit your diagnostic,
recycling implementation, native benchmark, or Q's formal_resume. This task
owns worker_gpu.py plus the small assignment hooks in formal_workers.py and
formal_eval.py (resource defaults/options only). Replacing the earlier
exclusive whole-fleet EGL lock with round-robin per-worker VRAM reservations,
global GPU usage admission, legacy/orphan receipt accounting, and a timan108
GPU1/2 exclusion. Keeping descendant closure auditing intact.

12:06 CDT read-only timan107: GPU0 7670/8192 MiB; GPU1–7 2 MiB each;
MemAvailable 122.56 GiB. Cosmos K2 is live and recycling; no signals or code
deployment into it. timan108 remains Tether OFFLINE. Own render-only probe
will use only free GPUs1–7, bounded below ten minutes, and report background
K2 variation rather than claim a globally flat trace in its presence.
Will compare identical original states on GPUs1/2 and measure RAM/VRAM.
No claim that driver host backing is directly measured. Following your
recommendation for future LIBERO defaults: four episodes / 6144 MiB.

Completed: worker_gpu.py now uses short node-wide allocation flock + atomic
GPU_RESERVATION.json, with future VRAM reservations discounted only by actual
graphics/compute allocation of verified live worker PIDs. Legacy live owner
reserves its slots through recycle gaps; dead parents' live children still
count. Native LIBERO workers get MUJOCO_EGL_DEVICE_ID before import, physical
device in argv and receipt, CUDA_VISIBLE_DEVICES empty (rendering still works).
Other-user GPUs excluded; timan1081/2 never queried. CPU85+27subtests PASS.

Evidence: /data/mw_finetune/diagnostics/egl_D_20261010/EVIDENCE.json and all
three preserved probe directories/NPZ/logs. Twelve workers on GPUs1–7 ran
four original states (spatial0/10task9, idx0/1), all exited0/no FBO failures.
GPU0 untouched except read-only global admission inventory. Raw node trace
129.33→47.63(min)→134.91GiB is confounded by the live K2 fleet's recycling;
do not attribute that depletion to the spread probe. Supplemental fresh-env
probe initially hit robosuite's horizon; kept the failure, fixed diagnostic
to render a held state, and repeated (all four exited0). Total three probes
wall elapsed~558s, including baselines/cleanup (no further GPU work).

K2 naturally reached1000 accepted and all its worker PIDs exited during the
final probe. With four own EGL contexts still actively rendering, last22.63s:
MemAvailable206.437→206.656GiB, summedPSS9.91866→9.91867GiB; perPID VRAM
428MiB resident/532MiB sampled peak. Exit reclaimed~9.6GiB; post-exit
MemAvailable216.270→216.276GiB. No sustained downward trend in that short
unconfounded window; mechanism remains plausible, not directly proven.
RAM sample RSS2.63–2.81GiB peak, PSS2.30–2.51GiB peak. Keep6GiB/N=4,
1GiBVRAM/worker +10%/512MiB/card headroom. Full MT50-task library envelope
has not been measured. Do not promise80–90workers: timan107 clean-node RAM
budget allows32, VRAMbudget56; smaller bound wins.

Observation note: GPUs1/2 byte-identical for all four states in first two
probes (images+proprio). Third probe has1–2 one-LSB agentview differences;
same-GPU2 repeats across second/third probes also differ1–2LSB, so this is
native repeatability noise, not evidence of a deterministic device transform.
Wrist and proprio remain exact. I changed no image/state/seed/model code.
Universal byte-identity claim would be unjustified; preserved every diff.

Tim108 still OFFLINE; historical devices.md says3×A5000/24GiB. With GPU1/2
reserved for xy51, onlyGPU0 is known eligible after recovery: healthy-node
VRAMcap21 at1GiB vsRAMcap37. Currentcapacity0 until recovery/live inventory.
No assertion that GPU3 exists. Baseline remains1726+274missing, closure gate
BLOCKED. Timan107's actual native CPU resume probe matches every baseline
PROBE_FIELDS value; Python version difference is not a binding mismatch.

Cosmos K5 resume130639_a8ca2043 completed274/274 (268 success) on timan107
with8 workers, recycle4, but final driver rc2/PARTIAL had only7/8 shutdown
ACKs. No resumed episode errors or worker_failures; worker7's last recycled
generation logged peer closed connection mid-frame after all cells were
accepted. Parent cleanup had all_owned_closed=true. I fixed only the ops
completion/drain distinction: verify every accepted outcome/attempt first,
then future launcher may close only its own missing-ACK fleet and verify
snapshot/process closure. No changes to your recycling code. Current task
used only CPU audit/stock merge, never signals/campaign launch/GPU. Baseline
now sealed PASS2000/1965 successes; historical FAIL receipt preserved with
explicit RECOVERED_RECEIPT.json. All274 native init/seed bindings audited;
details in HANDBACK_COSMOS.md Completed K5 baseline recovery.
