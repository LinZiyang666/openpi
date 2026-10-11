# Find robot policies with released weights on BOTH LIBERO and MetaWorld

You are one of two independent researchers with the same brief. Work independently: do not open `exp/offline_search/rounds/r17/survey*/` directories other than your own output directory, nor `exp/offline_search/rounds/r17/prompts/` (except this brief), nor `/home/weiland/projects/openpi_ext/scratch/r17_ops/`.

## Context
We build an inference-cost reduction system for robot policies (MLSys paper): a retrieval cache keyed on the policy's own vision features that replays cached action chunks, plus a no-progress guard, a library corrector, a periodic cost knob and per-task knob descent; iterative heads also get plain step reduction, chunking policies extended commit horizons. Validated on π0.5 and GR00T N1.5 (VLAs) on LIBERO. System details: `exp/offline_search/rounds/r16/REPORT.md`.

Decided by the owner (2026-10-08 22:5x):
- Second benchmark: **MetaWorld** (MT50 or its standard variants). π0.5 has the RLinf MetaWorld SFT checkpoint.
- **Cosmos Policy** is kept as a LIBERO-only special case. **GR00T** will be fine-tuned on MetaWorld by us.
- Now find additional models: 「现在用这个标准找模型，在liebro和metaworld上都有权重，最好是扩散，wam之类的」.

## Hard requirements for candidates
1. **Released weights on both LIBERO and MetaWorld** (official or credible community checkpoints). Verify by file listing (weights + config present), and record reported success rates where they exist; say explicitly when a checkpoint has no reported numbers.
2. **Non-VLA** robot policy (no VLM/LLM backbone). World models / world-action models (WAM) / video-diffusion policies count. Preferred: diffusion / flow / WAM families.
3. **Output depends only on the current observation** (plus language/task). Owner: 「你要填充kv的就不适合我们，我们的cache只适合基于当前observation给输出的」. Reject policies with cross-call state (KV caches to refill, recurrent state) or that need observation history or goal images.
4. **Cache must be profitable**: a full call must be expensive, and the part after extracting the vision features used as retrieval key must dominate. Owner rejected cheap policies (e.g. ACT: 「他cpu都可以算了」). Give a per-call cost breakdown (published latency/FLOPs or code-level estimate), full-call : hit-path ratio, and time saved per hit.
5. Newer members of a family preferred; no two picks from the same family.

## Method expectations
- Be systematic, not anecdotal: enumerate Hugging Face models via the API (e.g. `https://huggingface.co/api/models?search=metaworld&limit=1000`, also `meta-world`, `meta_world`, `mt50`, `mt10`, `libero`), group by author/model family, and intersect families that have both LIBERO and MetaWorld checkpoints. Also search papers (2024–2026) that report results on both LIBERO and MetaWorld and check whether their weights are released (HF, Google Drive, GitHub releases).
- For every surviving candidate: family, date, architecture (vision encoder → retrieval key, backbone, action head and default steps, chunk length), statelessness check (read the inference code), checkpoint links + file listings for both benchmarks, reported success rates, license/gating, profitability numbers, which of our layers apply, risks.

## Deliverables
In your output directory: `DECISION.md` (Chinese, **tables first, minimal prose** — the owner wants lists, not paragraphs) and `EVIDENCE.md` (links, file listings, the HF enumeration you ran). Include a full table of every LIBERO∩MetaWorld candidate you found with pass/fail per requirement, then your ranked picks (or say clearly that none passes and what comes closest).

## Constraints
Web and local reading; CPU-only probes; no GPU, servers, remote/tether, git commit, pkill, rm -rf; do not touch /data/i018, /data/idea_sprint, /home/weiland/idea-sprint, /archive, or running processes. No multi-GB downloads (configs/metadata fine). Write only in your output directory.
