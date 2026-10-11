<task>
The owner rejected your recommendation and wants the selection redone from scratch. Owner's words (2026-10-08 20:2x):
- 「robocasa365不要我准备废弃这个benchmark，这个上面有很多场景变换，以及物体变换，不利于我们构建cache」
- 「pi0就不要了，vla我们都有两个了groot和pi0.5都有了，我都说了非vla的其他模型，这能体现我们系统的广泛适用性」
Write a new decision; keep the old one only as `DECISION_v1_rejected.md`.
</task>

<binding_constraints>
Models:
- **Only non-VLA policies** — no VLM/LLM backbone (no π0, OpenVLA/OFT, X-VLA, GR00T, SmolVLA, etc.). We already have two VLAs (π0.5, GR00T N1.5); the point now is to show the five-layer system works across policy families. Pick at least 3, ideally of different families (e.g. 2D-image diffusion/flow policy, 3D point-cloud policy, transformer/CVAE chunking policy, VQ/tokenized policy, generalist non-VLM transformer such as Octo-class), each with a vision encoder whose features can serve as the retrieval key. Iterative heads additionally get step reduction; chunking policies get extended commit horizons; say for each which layers apply.
- Prefer public LIBERO checkpoints (see the owner follow-up below on self-fine-tuning as a last resort).
Benchmark (one, plus LIBERO which we keep):
- Recent and influential, **light** (per-episode CPU cost close to LIBERO; MuJoCo/PyBullet-class, CPU workers, no GPU rendering), and **cache-friendly: little scene/layout/object variation within a task** (fixed scene and object set per task, small initial-state perturbations), so that demonstration/rollout libraries cover test states.
- **π0.5 support is mandatory** (a usable checkpoint or official/community fine-tune with reported numbers).
- Owner (20:3x): 「你挑benchmark一定要那种适合我们的benchmark的，又有一定影响力，比较新一点，且这些模型都要有权重」 — the chosen non-VLA models should have released weights on the chosen benchmark (official or credible community checkpoints with reported numbers). Owner follow-up (20:3x): 「我们可以尝试自己微调，但是能不我们自己来就不我们自己来」 — self-fine-tuning is allowed only as a last resort: choose the model set and benchmark jointly to minimize how much we must train ourselves; for any model that would need our own training, say why no released-weight alternative works, the training cost, and the reproduction risk vs published numbers. Verify each checkpoint link exists (files/configs listed), not just paper numbers. Same preference applies to LIBERO checkpoints.
- Rejected outright: RoboCasa365/RoboCasa (scene & object variation), RoboTwin 2.0 and BEHAVIOR (heavy), SimplerEnv/ManiSkill3 (GPU rendering). Re-evaluate everything else (e.g. MetaWorld/MT50 variants, CALVIN, robomimic, LIBERO-PRO/Plus, VLABench, Adroit/DexArt, newer 2025–2026 suites) against these criteria, and verify π0.5 support claims.
</binding_constraints>

<deliverables>
Overwrite `exp/offline_search/rounds/r17/survey/DECISION.md` (Chinese, owner-facing, concise, no codenames) and update `EVIDENCE.md`: recommended non-VLA models with verified facts (encoder → retrieval key, head type/steps, chunk, per-call cost, checkpoints per benchmark or training cost), the benchmark with per-episode CPU cost (measure locally where cheap, CPU-only) and a scene-variation assessment, π0.5 checkpoint for it, layer-by-layer expected value, risks, and a first diagnostic plan with budget. One-line reasons for rejected candidates.
</deliverables>

<constraints>
Same as before: web + local reading; CPU-only measurements (`taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES=`); no GPU, servers, remote/tether, git commit, pkill, rm -rf; do not touch /data/i018, /data/idea_sprint, /home/weiland/idea-sprint, running processes, or /archive. No multi-GB downloads; small configs/metadata fine; a light benchmark may be installed for a CPU timing probe only under a new dir in `/home/weiland/projects/openpi_ext/lines/` (say what you installed).
</constraints>
