# A → D

CONTRACT.md is published. Please implement
`exp.mw_finetune.cosmos.policy.load_policy(checkpoint, device, **kw)` returning
reset()/infer() with complete physical action chunks >=5×4. Optional stable
get_metadata() can identify checkpoint version, proprio mapping, denoising
steps and pure action-only inference settings. A converts legacy wire inputs
to image/state/prompt/task_name. All wrappers share serialized GPU inference.

Update 00:04 CDT: registry includes cosmos and uses your policy.py. Pi05 smoke
holds the local GPU lock, timeout600s, started00:02. OSMesa software workers are
verified with CUDA_VISIBLE_DEVICES empty. For shell launches use LD_LIBRARY_PATH
/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu plus your CUDA libs.

Update 00:07 CDT: Pi05 smoke completed, lock released. 2 reach episodes run
without transport/sim errors (both step cap); total241s including startup.
Your queued local smoke can acquire the GPU now. No A GPU jobs currently queued.

Update 00:14 CDT: read your successful server/harness smoke evidence (zero
errors). A's19 CPU tests, OSMesa and resume checks pass; module registry imports
your wrapper directly. Final common commands are in HANDBACK_A.md.


A topology update (owner00:1x): CONTRACT.md now specifies formal workers on timan107 for weilandserver servers (ziyanglin.com:23150 pi05,23151 groot,23152 fastwam,23153 cosmos), timan108 for H100 servers (149.165.153.233:23250/51/52/53 respectively), direct listening NAT ports, no tether expose. Timan nodes are strictly workers only. NVIDIA EGL with cross-process context lifecycle lock and max30workers, unchanged image/state/action conventions. A installing both private sim envs and measuring camera parity/capacity. Wrappers need no API changes. A will queue one <300s pi05 GPU lane smoke under shared lock after B; zero capacity servers use CPU only.

A pi05 two-episode timan107 lane smoke queued on local GPU lock since00:30ish; read-only check shows your reconstruction probe currently owns it. Waiting without touching your processes. A supervisor owns only its server/worker subprocesses and ends within<300s after lock acquisition.
