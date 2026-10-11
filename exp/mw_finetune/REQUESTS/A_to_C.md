# A → C: Fast-WAM handoff

Implement `exp.mw_finetune.fastwam.policy.load_policy` as CONTRACT.md specifies.
Pre-split setup already in progress, please reuse rather than duplicate:

- Upstream official repo: `/data/mw_finetune/vendor/FastWAM`, SHA
  7faa71108368fbb3b6885649f112af607427a2d4.
- LeRobot repo: `/data/mw_finetune/vendor/lerobot`, SHA
  b9cb121cb7d4e3e26ec5c906d08dda68d151ba52. Current main needs Python3.12.
- New local Python3.12 venv: `/data/mw_finetune/envs/fastwam`; uv install is
  running in exec session11222: editable lerobot[fastwam,training], torch2.7.1,
  torchvision0.22.1, torchcodec0.3.0, websockets,msgpack,pytest. I will monitor.
- HF LIBERO model download running in exec session2193 into
  `/data/mw_finetune/models/fastwam_libero`, revision
  53983e1249b4eb4d89ab42b40b8678a882d331ad, 12.04 GB. Metadata already present.
- Full dataset download session13087, root
  `/data/mw_finetune/datasets/metaworld_mt50`, v3 revision
  b46a1c185a2368660650a82ac5a5083a56722f48, 45.86 GB. A monitors to completion.
  B owns conversion; coordinate compact prepared data if needed on H100.

Both `/data/xwam/wan22_5b` roots have Wan2.2_VAE.pth (2.7GiB), DiT shards,
**models_t5_umt5-xxl-enc-bf16.pth (11GiB)** and `google/umt5-xxl` tokenizer.
Do not download them again. Official FastWAM has raw-weight compatible
WanVideoVAE38 and WanTextEncoder in models/wan22; VAE converter prefixes model.
LeRobot normally loads diffusers VAE/T5; an opt-in local shim can substitute
official raw loaders (no upstream editing needed).

Critical inspection: LeRobot `loss.lambda_video=0` currently still computes
full video training loss/MoT, so it does **not** skip future frames automatically.
Video blocks are reparented into mot.layers; LeRobot freeze_video_expert handles
both paths. Proprio embeddings can still require gradients through frozen
video context. Verify actual memory and gradients before promising batches.
The released LIBERO config has model_id='.', dimensions7/8 and two cameras.
The shape-aware loader supports cross-embodiment reinit with strict=False;
new MW checkpoints must serve with strict=True. Reset LIBERO gripper toggles.

I asked an async optional recipe choice before scope split: recommended frozen
video expert vs video-backbone LoRA retaining co-training. No answer received
yet; please coordinate the final materially different recipe with coordinator.

Update 00:04 CDT: LIBERO download still active (8/9 files, large weight incomplete)
and local uv install still active (torch downloaded, remaining dependencies in
progress). Dataset about63/498 files after11min; metadata complete. Do not wait
on the entire dataset for smoke: early parquet files already available.

Update 00:06 CDT: model download session2193 **completed successfully**.
`/data/mw_finetune/models/fastwam_libero/model.safetensors` is now ready.
Local environment install session11222 still active; A monitors.

Update 00:06 CDT: local uv install11222 **completed successfully**. Torch
resolved2.7.1+cu128, Transformers5.5.4, LeRobot0.6.2 at pinned checkout,
diffusers0.39.0, huggingface_hub1.33.0, Python3.12.13. Environment ready.
See A command logs for full versions; I will save a freeze under reports/.

Full environment freeze saved `/data/mw_finetune/reports/A_fastwam_env_freeze.txt`.
Pi05 local smoke finished at00:07; D queued next. A has no pending GPU jobs.

Update 00:14 CDT: A handback/contract are ready,19 A CPU tests pass; all4
registry entry points import successfully. Download roughly50% complete,
continues in original local session13087/PID2988390 (bash parent; no GPU).
H100 shared root measured25GiB,164GiB free at00:10. Shared runtime.disk_gate now
checks the120GB root budget as well as free-space floor if you wish to reuse it.


A topology update (owner00:1x): CONTRACT.md now specifies formal workers on timan107 for weilandserver servers (ziyanglin.com:23150 pi05,23151 groot,23152 fastwam,23153 cosmos), timan108 for H100 servers (149.165.153.233:23250/51/52/53 respectively), direct listening NAT ports, no tether expose. Timan nodes are strictly workers only. NVIDIA EGL with cross-process context lifecycle lock and max30workers, unchanged image/state/action conventions. A installing both private sim envs and measuring camera parity/capacity. Wrappers need no API changes. A will queue one <300s pi05 GPU lane smoke under shared lock after B; zero capacity servers use CPU only.

A updated H100 /home/exouser/openpi top-level serving modules and copied exp/metaworld/tasks.py, solving an exp.metaworld import gap. Verified serve.ZeroPolicy and WebsocketPolicyServer import from /home/exouser/openpi/.venv with PYTHONPATH=/home/exouser/openpi:/home/exouser/openpi/src:/home/exouser/openpi/packages/openpi-client/src. CPU-only zero server temporarily listening23259 for timan108 capacity; no GPU touched. Your FastWAM serving needs the same repo/client PYTHONPATH.

Combined root-venv CPU suite now has51passes and1environment failure in your new test_loss.py: root .venv exposes only a lerobot namespace, no lerobot.policies. A did not edit your tests. Running your suite with /data/mw_finetune/envs/fastwam/bin/python and CUDA_VISIBLE_DEVICES=empty to use the pinned policy implementation. Recommend documenting its island interpreter for the causal-equivalence test.

Final dependency check: CUDA_VISIBLE_DEVICES=empty, /data/mw_finetune/envs/fastwam/bin/python, H100 repo/client PYTHONPATH successfully imports A serve.load_model and WebsocketPolicyServer (no model loaded). C7CPU tests passed in local pinnedFastWAMvenv. Both worker lanes installed and SHA256sync/camera proof/realpi05lane smoke complete; see HANDBACK_A.
