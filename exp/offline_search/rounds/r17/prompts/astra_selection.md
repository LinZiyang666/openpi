<task>
Decide which robot policies and which additional benchmark we should port our inference-acceleration system to next, and justify it. You own the research: method, what to verify, what to measure are your call. Write the decision so the owner can approve it and engineering can start.
</task>

<context>
Read first: `exp/offline_search/rounds/r17/survey/INPUT.md` (owner requirements, two prior surveys: our own history and a web survey), then as needed `exp/offline_search/rounds/r16/REPORT.md`, `rounds/r15/REPORT.md`, `rounds/r16/COORDINATOR_DECISIONS.md`, and the logs the internal survey cites (`logs/cosmos_nfe_plan.log.md`, `logs/xwam_dp_nfe_plan.log.md`, `logs/nfe_baseline_rc365_plan.log.md`, `logs/cache_transfer/`, `logs/session_handoff.md` appendices). The two surveys were done by a weaker model; treat their claims as leads to verify, especially checkpoint availability, step counts, chunk lengths, and benchmark π0.5 support.
</context>

<requirements>
Owner's constraints (binding):
- Port the whole five-layer system (vision-feature retrieval cache, no-progress guard, library corrector, periodic cost knob, per-task knob descent), plus plain step reduction where the action head is iterative and extended commit horizons where the policy chunks actions. Any policy with a vision encoder qualifies; include non-VLA policies (and say whether a policy without a usable vision head could still work).
- Models: strongly prefer public LIBERO checkpoints.
- One additional benchmark: recent and influential, suited to this kind of system, supported by all or most chosen models; **π0.5 support is mandatory**; **must not be heavy** — we run thousands to tens of thousands of episodes per experiment with ~64 CPU worker processes on timan107 (MuJoCo-class sims, offscreen EGL/OSMesa) and one GPU server (4090 48 GB or h100). RoboTwin 2.0 measured ~160 episodes/h on 7 clients and is considered too heavy.
</requirements>

<deliverables>
`exp/offline_search/rounds/r17/survey/DECISION.md` (Chinese, owner-facing, no project codenames): recommended 3–4 models (≥1 non-VLA) and 1 benchmark (+ an optional lightweight robustness add-on if worthwhile), each with the verified facts that matter for our layers (where the retrieval key comes from, head type and default steps, chunk length, inference cost per call, checkpoint links per suite/benchmark, license/gating), per-episode simulator cost estimate for the benchmark (measure on local installs where cheap, CPU-only), expected value of each layer on that model (where is there headroom: e.g. models whose k=1 already loses vs models where it's lossless), main risks, and a first concrete experiment plan with rough GPU/CPU-hour budget. Also `survey/EVIDENCE.md` with links and what you checked. Keep it decision-oriented; list rejected candidates with one-line reasons.
</deliverables>

<constraints>
Web and local reading are fine; CPU-only measurements are fine (`taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES=` …). No GPU, no servers on public ports, no remote/tether operations, no git commit, no pkill, no rm -rf; do not touch /data/i018, /data/idea_sprint, /home/weiland/idea-sprint, running processes, or `/archive` (a migration is writing there). Do not download multi-GB checkpoints; small metadata/config files from HF are fine. Installing a benchmark for a CPU timing probe is fine only in a new directory under `/home/weiland/projects/openpi_ext/lines/` and only if it is light; say what you installed.
</constraints>
