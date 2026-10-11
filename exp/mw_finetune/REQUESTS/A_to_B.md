# A → B: GR00T handoff

Implement `exp.mw_finetune.groot.policy.load_policy` as CONTRACT.md specifies.
I created an initial `exp/mw_finetune/groot_config.py` before the scope split;
please **copy** it into your directory if useful and own the final version.
I will remove the top-level draft only after your module is available. It uses
new_embodiment / projector slot 31, one video.image, four scalar state/action
keys, horizon 16, min_max normalization and the official LIBERO image transforms.
I have not changed upstream GR00T files.

Base download is already in progress in local exec session 2193:
`/data/mw_finetune/models/groot_base` (HF revision
869830fc749c35f34771aa5209f923ac57e4564e, 5.45 GB). Its base portion completed;
the same process is now downloading C's LIBERO model. Do not duplicate it.

I started the full pinned dataset download (exec session 13087) into
`/data/mw_finetune/datasets/metaworld_mt50`, v3 revision
b46a1c185a2368660650a82ac5a5083a56722f48, 45.86 GB. Metadata is complete
(separate small download); data continues. I will monitor this existing process
to completion, but **please own all conversion/preparation from here**.
Source images are embedded PNG bytes in Parquet, no MP4; v2.1 also embeds images
and reports 49 text tasks due to a duplicate. v3 has 50 task entries. A planned
lossless RGB H264 conversion (libx264rgb -crf 0) could satisfy GR00T MP4 and
preserve exact image round trips; inspect decoder compatibility.

Use the shared GPU lock; authorization was updated to permit smoke checks now.
H100 is currently ~188 GB free after coordinator cleanup, max120 GB new budget.

Update 00:04 CDT: module-based serving is in place and imports your policy.py.
The top-level groot_config.py now reexports your data_config.py, with no duplicate
configuration. Pi05 smoke is holding the local GPU lock, timeout600s; D's smoke
is queued behind it. CPU OSMesa workers work with CUDA_VISIBLE_DEVICES empty.

Update 00:06 CDT: combined model download session2193 completed successfully;
GR00T base and Fast-WAM LIBERO are both complete on weilandserver.

Update 00:07 CDT: Pi05 smoke finished, local lock released; D queued smoke next.

Update 00:14 CDT: A's full harness/commands are in HANDBACK_A.md;19 A CPU tests
pass; registry imports all4 load_policy functions. OSMesa/resume integration
passed. Data download is roughly50% complete and continues. Model-specific
serving checkpoint metadata is consumed unchanged; no inline GR00T loader remains.


A topology update (owner00:1x): CONTRACT.md now specifies formal workers on timan107 for weilandserver servers (ziyanglin.com:23150 pi05,23151 groot,23152 fastwam,23153 cosmos), timan108 for H100 servers (149.165.153.233:23250/51/52/53 respectively), direct listening NAT ports, no tether expose. Timan nodes are strictly workers only. NVIDIA EGL with cross-process context lifecycle lock and max30workers, unchanged image/state/action conventions. A installing both private sim envs and measuring camera parity/capacity. Wrappers need no API changes. A will queue one <300s pi05 GPU lane smoke under shared lock after B; zero capacity servers use CPU only.
