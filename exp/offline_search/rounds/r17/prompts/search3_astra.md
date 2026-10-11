# Find ONE more model that we can conveniently fine-tune on MetaWorld

You are one of two independent researchers with the same brief. Do not open other `exp/offline_search/rounds/r17/survey*` directories, other files in `exp/offline_search/rounds/r17/prompts/`, or `/home/weiland/projects/openpi_ext/scratch/r17_ops/`.

## Decided by the owner (2026-10-08 23:xx)
- Benchmarks: LIBERO (kept) + **MetaWorld MT50**. π0.5 already has a MetaWorld checkpoint.
- We will **fine-tune on MetaWorld ourselves**: **GR00T N1.5** (a VLA; existing LIBERO weights) and **Diffusion Policy** (non-VLA diffusion). Owner: 「我们要在metaworld上方便微调的，一个是groot，一个diffusion，你再找一个」 — find a **third** model.
- Cosmos Policy stays a LIBERO special case.
- MetaWorld MT50 data: `lerobot/metaworld_mt50` — 2,500 episodes, 204,806 frames, one 480² camera, 4-D actions, 50 tasks.

## Requirements for the third model
1. **Released LIBERO weights** (we must not train on LIBERO), verified by file listing; reported LIBERO success rates.
2. **Convenient to fine-tune on MetaWorld**: public training code (ideally LeRobot-compatible data loading), fits our hardware (one RTX 4090 48 GB, or an H100 80 GB for a limited slot), and a fine-tune budget of roughly ≤ 20–30 GPU-hours on MT50. Estimate it concretely (model size, trainable parts — full / action-head-only / LoRA — steps × batch, measured or published throughput). Say whether partial fine-tuning (e.g. freezing a video backbone) is supported by the official code.
3. **Non-VLA** (no VLM/LLM backbone), and **a different family** from GR00T (VLA with flow DiT head) and Diffusion Policy (image-conditioned diffusion U-Net/Transformer). Preferred: world-action models (WAM) / video-diffusion / video-prediction policies.
4. **Stateless**: output depends only on the current observation (+ language/task). No cross-call KV/recurrent state, no observation history, no goal images. Read the inference code to confirm.
5. **Cache must be profitable**: a full call must be expensive and the part after extracting the vision features used as retrieval key must dominate. Give per-call cost breakdown (published/measured latency or FLOPs), full-call : hit ratio, ms saved per hit; compare with π0.5 (68 ms full on a 4090, ~6.6× full:hit).

Candidates to consider (not exhaustive; verify everything, find others): Fast-WAM (6B; check action-expert-only or LoRA fine-tuning), UVA (Unified Video Action), UWM (Unified World Models), VPP (Video Prediction Policy), PAD, Seer, MoDE, Octo, MDT, Genie Envisioner (GE-Act), and any 2025–2026 WAM/video policy with LIBERO weights. Our system and earlier step-reduction logs: `exp/offline_search/rounds/r16/REPORT.md`, `logs/cosmos_nfe_plan.log.md`, `logs/xwam_dp_nfe_plan.log.md`.

## Deliverable (keep it short)
In your output directory: `DECISION.md` (Chinese, **tables only, minimal prose**): a candidate table with pass/fail per requirement 1–5 and the fine-tune cost estimate, then your single pick (plus one fallback) with links. `EVIDENCE.md`: links, file listings, code references for statelessness and fine-tuning.

## Constraints
Web + local reading; CPU-only probes; no GPU, servers, remote/tether, git commit, pkill, rm -rf; do not touch /data/i018, /data/idea_sprint, /home/weiland/idea-sprint, /archive, running processes. No multi-GB downloads. Write only in your output directory. Aim to finish within ~40 minutes.

## Your output directory
`/home/weiland/projects/openpi/exp/offline_search/rounds/r17/survey3_astra/`
CPU-only probes: `taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES=`.
