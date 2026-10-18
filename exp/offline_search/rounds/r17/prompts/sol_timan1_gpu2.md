<task>
Coordinator, 2026-10-16 17:1x. timan1 is back online (rebooted 14:35). The owner allows us to use ONLY its GPU 2: NVIDIA RTX A6000 48 GB, nvidia-smi index 2, UUID GPU-9bdcf7cc-8a05-a7f9-b1c3-4d53d0a229df, PCI 0000:81:00.0, render node renderD129 (/dev/dri/by-path/pci-0000:81:00.0-render). It is currently empty (10 MiB). GPUs 0, 1, 3 belong to other users and must never be used. Node: 48 cores, ~450 GiB free RAM.

Today's code and the frozen worker packages pin timan1 workers to the old GPU 3 (exp.r17_integration.ops.timan1_workers UUID/PCI, worker_gpu.PROTECTED = {'timan1': {0,1,2}}, the timan1 lease in campaign['timan1_worker_lease'], and the sitecustomize GPU guard that keys on orig_argv). Since 2026-10-15 23:00 every ff/ffk launch goes through `rounds/r17/filter_failed/launch.py` (`worker_command`/`SHARE`, reused by `filter_failed_knob/launch.py`), which already rewrites the remote worker command at launch; `filter_failed/resume.py --run ... --worker-counts` sets execution worker counts on unstaged runs (I extended its guard today to fresh ff1015_ drafts too).

Make timan1 GPU 2 usable as a worker island for runs launched through those launchers — LIBERO cache workers and MetaWorld workers — with GPU 2 as the only admissible timan1 device, without editing sealed packages or launched runs (override at launch the way the existing patch does). Size the per-run timan1 cap from measured per-worker VRAM/RAM (LIBERO renderer ~530 MiB VRAM, ~2 GiB RAM; MetaWorld ~133 MiB VRAM, ~0.5 GiB RAM) leaving the usual 10% headroom. Verify read-only on timan1 (tether exec only) that the EGL device chosen is the one behind PCI 0000:81:00.0, and with the frozen worker modules on CPU that admission picks only that device and refuses the others.

Then give me the exact `resume.py --run ... --worker-counts` commands to add timan1 to these unstaged queued runs (keep the existing timan107/timan108 counts unless a total exceeds the source campaign's original worker total):
ff1015_cosmos_li_04 (now timan107=55,timan108=60; source total 145), ffk1015_dp_mt50_reduced_descentc1 (timan107=40,timan108=40; source 120), ffk1015_cosmos_libero_reduced_off_u and _off (timan107=25,timan108=60; source 85), ffk1015_groot_mt50_reduced_descent and _full_descent (timan108=40; source timan1x40), ffk1015_dp_mt50_full_descent and ffk1015_fastwam_mt50_full_descent (timan107=60,timan108=60; source 120).
Do not apply them; I will. cosmos_li_04 starts around 19:30, so report before then if you can.
</task>

<rules>
- CPU only locally; tether exec read-only checks on timan1/H100 only; never push/pull/expose; never start workers, servers, GPUs or chains; no pkill, rm, git commits; never touch credentials.
- Do not modify, rebuild or re-stage any running, prepared or launched run. Another sol thread is building new LIBERO test-set descent campaigns in rounds/r17/filter_failed_knob — don't touch its new files.
- Smallest change; one targeted test.
</rules>

<output>
Final message: what changed, the timan1 cap per workload, the read-only verification results, and the exact apply commands.
</output>
