# Workflow 脚本学习材料 — cp1_search 优化研究（3 轮专家 agent）

> 这是为「研究 Workflow 编排机制」准备的学习文件，收录了 cp1_search 延迟优化任务实际跑的
> workflow 脚本（逐字）+ 它依赖的 prep 脚本 + 设计标注。脚本本身是纯 JavaScript（不是 TS）。
> 不是交付物；研究完可删。

---

## 0. Workflow 原语速查（脚本里能用的钩子）

| 原语 | 作用 | 关键点 |
|---|---|---|
| `export const meta = {...}` | 脚本第一行，必须是**纯字面量** | `name`/`description` 必填；`phases[].title` 要和 `phase()` 调用对齐 |
| `phase(title)` | 起一个进度分组 | 之后的 `agent()` 归到这个组；可被 `agent(opts.phase)` 覆盖 |
| `agent(prompt, opts?)` | 派一个子 agent | 带 `schema` → 返回校验过的 JS 对象；不带 → 返回文本；被跳过 → `null` |
| `parallel(thunks)` | **栅栏**：并发跑完所有 thunk 再返回 | 收 `()=>agent(...)` 函数数组；抛错 → 该位 `null`，记得 `.filter(Boolean)` |
| `pipeline(items, s1, s2…)` | **流水线**：每个 item 独立穿过各 stage，无 stage 间栅栏 | 默认优先用它；只有「stage N 需要 stage N-1 的全部结果」才用 `parallel` 栅栏 |
| `log(msg)` | 给用户发一行旁白 | 显示在进度树上方 |
| `return {...}` | workflow 返回值 | 就是主循环拿到的 tool result |
| `args` / `budget` | 入参 / token 预算 | 略 |

并发上限 `min(16, 核数-2)`，多的排队；子 agent 自带 Read/Bash/Grep 等工具，能自己读代码、跑基准。
脚本里 `Date.now()` / `Math.random()` / 无参 `new Date()` 会抛错（为了可 resume）。

---

## 1. Round 1 —— 广度探索（6 专家并行 → 1 综合扇入）

设计意图：6 个不同角度（3 CPU + 3 GPU）并行提方案；共享一份我亲手核实的 `FACTS` 地基防止幻觉；
每个 agent 用 `schema` 产出结构化提案；最后一个综合 agent 把 6 份 JSON 扇入，排名 + 选 finalist + 定 R2 问题。

```javascript
export const meta = {
  name: 'cp1-search-opt-round1',
  description: 'Round 1: 6 experts explore CPU+GPU optimizations to cut cp1_search 37ms -> <10ms (ideally <5ms)',
  phases: [
    { title: 'Explore', detail: '6 domain experts propose optimizations (3 CPU angles, 3 GPU angles)' },
    { title: 'Synthesize', detail: 'rank proposals, pick best CPU + best GPU finalists, set Round 2 questions' },
  ],
}

const FACTS = `
VERIFIED GROUNDING (from src reads + a library probe — treat as ground truth, but re-read the cited code yourself to confirm before relying on any line):

TARGET: cp1_search segment of the OpenPI cache CP1 check() path. Currently ~37 ms median, = ~95% of total CP1 latency (other 5 segments < 1.2 ms combined). GOAL: bring cp1_search to <10 ms, ideally <5 ms.

LIBRARY (libero_10, exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl, 1.1 GB):
- 2640 total entries, single-chain trajectories (50 roots / episodes; 2590 entries have exactly 1 parent — NO multi-branch DAG in this library).
- Per-query candidates AFTER the task_key filter: N = 166..399, mean ~264 (10 tasks). So each search scores only a few hundred candidates, NOT thousands.
- Active scoring fields per query: vision_0 (32768-dim float32, cosine), vision_1 (32768-dim float32, cosine), robot_state (32-dim, L2 -> exp similarity). The two 32768-dim vision fields dominate. prompt_emb/vision_2 are present in the artifact but NOT weighted in the studied config.
- Every entry stores all fields as float32 numpy, converted to torch.float32 on load.

ARITHMETIC: useful work per query ~= 2 vision fields * N(~264) * 32768 dims ~= 1.7e7 multiply-adds. At 37 ms that is ~0.5 GFLOP/s effective — roughly 100x below a single modern CPU core's GEMV peak. CONCLUSION: this segment is OVERHEAD-bound, not arithmetic-bound. Any latency model must explain where the ~37 ms goes (memcpy / allocation / Python loop / framework overhead), not just count FLOPs.

CODE PATH (src/openpi/cache/backends/in_memory_backend.py):
- search() @289: filters entries (_filter_entries @340, linear scan over all 2640 by task_key), then dispatches: depth>=2 trajectory_weights -> _search_with_trajectory @638; else single-step _search_weighted_score_sum @591.
- _search_weighted_score_sum @591: for each active field calls _batch_field_scores, normalizes (per-field zscore->tanh), accumulates weight*score*mask, then topk(top_k=1).
- _batch_field_scores @405: score-memo wrapper. Keyed (session_id, field, query_id, sim_type) in _score_memo. Partitions candidates into memo-hits / misses via a PYTHON per-candidate loop @448, computes only misses via _compute_field_scores.
- _compute_field_scores @363 (the hot kernel, NO cache): @381 list-comprehension to find valid indices; @386-387 vecs=[candidates[i].query_keys[field] for i in valid] then torch.stack(vecs).float() -> rebuilds an [N,32768] matrix FROM N SEPARATE PER-ENTRY TENSORS ON EVERY QUERY (~N*32768*4 = up to 52 MB memcpy per field per query, x2 vision fields); @392 F.cosine_similarity(q, mat) (allocates norm intermediates).
- _search_with_trajectory @638: walks the single chain (_walk_chain @739), per-layer fusion (_compute_level_scores @774 -> _batch_field_scores per field). score-memo means only the CURRENT step's query is a fresh full compute; historical layers are memo hits (this is why latency is flat across depth 1..5).
- search_strategy.py is at src/openpi/cache/components/search_strategy.py (depth dispatch logic).

CONSTRAINTS / CONTRACT (must hold for any proposal):
- Library is FROZEN at serve time (write_policy=never, server C2 contract). => You MAY pre-build / pre-stack / pre-normalize / quantize / index the library ONCE at load time (load_artifact @200). Per-query work is what must shrink.
- Ranking semantics MUST be preserved: final winner = argmax over weighted_score_sum of per-field zscore->tanh normalized similarities, top_k=1, then a downstream threshold judge compares the top score to a fixed threshold (0.997697). Any approximation (ANN, quantization, dim-reduction) must be assessed for whether it changes the winner OR the absolute top score the threshold sees.
- Deployment reality: current target is CPU-only inference hosts. A GPU exists on some serving nodes (a100 exclusive-util node; a shared 'jupyter' node) but GPU residency competes with the policy model's own GPU use. Multi-replica serving exists.
- This is a RESEARCH/DESIGN task: you propose and quantify. Do NOT edit src/. You MAY write throwaway scripts under /tmp or exp/cache_latency_bench/data/ to measure micro-benchmarks if helpful.
`

const PROPOSAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['proposals', 'angle_summary'],
  properties: {
    angle_summary: { type: 'string', description: 'one-paragraph framing of your angle and its headline conclusion' },
    proposals: {
      type: 'array',
      minItems: 1,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'direction', 'mechanism', 'code_changes', 'expected_latency_ms', 'expected_speedup', 'feasibility', 'equivalence_risk', 'dependencies', 'confidence', 'risks'],
        properties: {
          title: { type: 'string' },
          direction: { type: 'string', enum: ['CPU', 'GPU', 'hybrid'] },
          mechanism: { type: 'string', description: 'how it works, concretely' },
          code_changes: { type: 'string', description: 'where it changes the code, with file:line anchors' },
          expected_latency_ms: { type: 'string', description: 'estimated per-query cp1_search ms WITH the arithmetic/roofline that justifies it' },
          expected_speedup: { type: 'string' },
          feasibility: { type: 'string', description: 'integration cost, load-time prebuild cost, memory budget' },
          equivalence_risk: { type: 'string', description: 'does it change the winner or the absolute top score the threshold judge sees? exact vs approximate?' },
          dependencies: { type: 'string', description: 'new libs (faiss, cupy, ...), hardware, portability' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          risks: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
}

const EXPERTS = [
  { label: 'cpu-memlayout', angle: 'CPU memory layout & BLAS', brief: 'Pre-stacked contiguous per-task [N,D] matrices + pre-normalized library vectors at load time, so each query is ONE BLAS GEMV (normalized cosine = dot product) instead of re-stacking N separate tensors per query. Quantify how much of the 37 ms is the per-query torch.stack memcpy/alloc + F.cosine_similarity overhead vs actual GEMV. Address fusing the two vision fields and the score-memo interaction.' },
  { label: 'cpu-quant-thread', angle: 'CPU quantization, dtype & threading', brief: 'fp32->fp16/bf16/int8 quantization of the 32768-dim library, MKL/OpenBLAS GEMM tuning, thread count (OMP/MKL), fusing all candidates+fields into one GEMM, cache-blocking. Quantify memory-bandwidth roofline (52 MB/field touched per query) and how lower precision shrinks it. Assess accuracy/equivalence impact on the top score the threshold sees.' },
  { label: 'cpu-ann-dimreduce', angle: 'CPU ANN, indexing & dimensionality reduction', brief: 'faiss CPU (IVF/HNSW/PQ), product quantization, PCA / random projection to shrink 32768-dim, two-stage coarse-to-fine retrieval. CRITICAL: N is only ~264 per task — assess honestly whether ANN/indexing is even justified vs a plain dense GEMV, or whether the real win is just dim-reduction of the 32768-dim vectors. Quantify recall/equivalence cost since the downstream threshold needs the true top score.' },
  { label: 'gpu-resident-bruteforce', angle: 'GPU resident brute-force', brief: 'Load the per-task library matrices onto GPU ONCE (resident), per query transfer only the 128 KB query vector H2D and run a GEMV / cosine on GPU, D2H the top score. Quantify: H2D(query)+kernel-launch+D2H latency floor vs the 37 ms CPU cost; whether kernel-launch + sync overhead dominates given the tiny per-query FLOPs; GPU memory budget for 2640*2*32768*4 bytes resident; contention with the policy model on the same GPU.' },
  { label: 'gpu-faiss-batch', angle: 'GPU faiss / batching / CUDA graphs', brief: 'faiss-gpu, CUDA graphs to kill launch overhead, batching multiple concurrent sessions queries into one GPU call, fp16 tensor-core GEMM. Analyze the latency-vs-throughput tradeoff: a single tiny query may be launch-bound on GPU; batching helps throughput but not single-request latency. State clearly when GPU wins for single-request <5 ms vs only for aggregate throughput.' },
  { label: 'systems-deploy-critic', angle: 'Systems / deployment / is-GPU-justified critic', brief: 'Step back: given N~264, ~1.7e7 FLOPs/query, and an overhead-bound 37 ms, decide which direction is actually warranted. Weigh CPU-only deploy reality, GPU contention with the policy model, the frozen-library prebuild opportunity, multi-replica serving, and operational complexity (new deps, portability). Propose the decision criterion (when CPU suffices vs when GPU is needed) and the single highest-leverage change. You MAY propose a CPU or GPU or hybrid approach as your recommendation.' },
]

phase('Explore')
const proposals = await parallel(
  EXPERTS.map((e) => () =>
    agent(
      `${FACTS}\n\nYOUR ROLE: ${e.angle}.\nFOCUS BRIEF: ${e.brief}\n\nTASK: Read the cited code (in_memory_backend.py @289-483, @591-737; search_strategy.py) to confirm the facts, then propose one or more CONCRETE, arithmetic-backed optimizations within your angle to bring cp1_search to <10 ms (ideally <5 ms). For each proposal give a latency estimate justified by a roofline/arithmetic argument (explain where the current 37 ms actually goes and how much your change removes), the exact code locations it touches, and an honest equivalence assessment against the threshold-judge semantics. Be concrete and quantitative; flag any claim you could not ground in code. If you can cheaply validate a number with a throwaway micro-benchmark (e.g. time torch.stack vs a pre-stacked GEMV on random [264,32768] data), do so and report the measured ms.`,
      { label: e.label, phase: 'Explore', schema: PROPOSAL_SCHEMA }
    )
  )
)

const valid = proposals.filter(Boolean)
log(`Round 1 explore: ${valid.length}/${EXPERTS.length} experts returned`)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ranked', 'best_cpu', 'best_gpu', 'round2_questions', 'dropped', 'headline'],
  properties: {
    headline: { type: 'string', description: 'one-paragraph verdict: is <5ms reachable on CPU alone? is GPU warranted? what is the single biggest lever?' },
    ranked: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'direction', 'projected_latency_ms', 'score', 'rationale'],
        properties: {
          title: { type: 'string' },
          direction: { type: 'string', enum: ['CPU', 'GPU', 'hybrid'] },
          projected_latency_ms: { type: 'string' },
          score: { type: 'number', description: '0-100 promise score (latency win x feasibility x equivalence-safety)' },
          rationale: { type: 'string' },
        },
      },
    },
    best_cpu: { type: 'array', items: { type: 'string' }, description: 'titles of the 1-2 strongest CPU finalists to deepen in Round 2' },
    best_gpu: { type: 'array', items: { type: 'string' }, description: 'titles of the 1-2 strongest GPU finalists to deepen in Round 2' },
    round2_questions: { type: 'array', items: { type: 'string' }, description: 'specific quantitative questions Round 2 must answer / benchmarks to run' },
    dropped: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'reason'],
        properties: { title: { type: 'string' }, reason: { type: 'string' } },
      },
    },
  },
}

const synthesis = await agent(
  `${FACTS}\n\nYou are the Round 1 synthesis lead. Below are ${valid.length} expert proposal sets (JSON). Rank ALL proposals by promise (latency win x feasibility x equivalence-safety), pick the 1-2 strongest CPU finalists and 1-2 strongest GPU finalists to deepen in Round 2, drop dominated/unjustified ones with reasons, and write the specific quantitative questions + micro-benchmarks Round 2 must run to settle the CPU-vs-GPU decision. Be decisive about whether <5 ms is reachable on CPU alone (given the overhead-bound diagnosis).\n\nEXPERT PROPOSALS:\n${JSON.stringify(valid, null, 2)}`,
  { label: 'synth-r1', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { proposals: valid, synthesis }
```

---

## 2. prep 脚本 —— 「把昂贵的一次性准备放在 workflow 外」模式

R2 的 5 个 agent 都要读真实库，但 `parallel` 会让它们同时加载 1.1G pickle（每个 ~3GB RSS）→ 撑爆 WSL。
所以我**先在 workflow 外自己跑一次** prep，把库抽成 693MB 的 `task_pack.pt`，agent 只读紧凑产物。
这是通用模式：**共享 + 昂贵 + 一次性** 的准备，放在编排之外的 barrier 做。

```python
"""把 1.1G 库抽成 per-task 紧凑张量包，避免 N 个 agent 各自 reload。"""
import collections, json, os, pickle
import numpy as np, torch, torch.nn.functional as F

SRC = "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl"
OUT_DIR = "exp/cache_latency_bench/data/opt_bench"
FIELDS = ["vision_0", "vision_1", "robot_state"]
os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC, "rb") as f:
    data = pickle.load(f)
entries = data["entries"]

def as_t(v):
    return v.float() if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)).float()

buckets = collections.defaultdict(list)
for e in entries:
    buckets[e.payload.task_key].append(e)

pack, meta = {}, {"src": SRC, "n_total": len(entries), "fields": FIELDS, "tasks": {}}
for tk, es in buckets.items():
    rec = {"ids": [e.id for e in es]}
    for fld in FIELDS:
        rec[fld] = torch.stack([as_t(e.query_keys[fld]) for e in es]).contiguous()
    pack[tk] = rec
    meta["tasks"][tk] = {"N": len(es), "vision_dim": rec["vision_0"].shape[1]}

torch.save(pack, os.path.join(OUT_DIR, "task_pack.pt"))

# near-1 cosine 判别带探针（裁决 fp16 争议的关键先验）：最大 task 上 leave-one-out
big_tk = max(meta["tasks"], key=lambda k: meta["tasks"][k]["N"])
Mn = F.normalize(pack[big_tk]["vision_0"], dim=1)
S = Mn @ Mn.T
S.fill_diagonal_(-2.0)
top2 = S.topk(2, dim=1).values
gap = top2[:, 0] - top2[:, 1]
meta["loo_probe_vision_0"] = {
    "loo_top1_cos_min": float(top2[:, 0].min()),
    "loo_top1_top2_gap_min": float(gap.min()),  # 实测 5.36e-7 —— 比 fp16 误差还小
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
```

> 跑法：`PYTHONPATH=$PWD uv run python exp/cache_latency_bench/_prep_opt_bench.py`
> （`run.py`/脚本在 pytest 外没有 conftest 注入路径，必须前缀 `PYTHONPATH=$PWD`。）

---

## 3. Round 2 —— 实测裁决（5 实测专家并行 → 1 综合）

和 R1 同构（`phase → parallel → phase → return`），但 agent 的任务从「提方案」变成「在真实数据上跑基准 / 审代码」，
prompt 里塞的是 R1 的结论 + 紧凑 pack 的路径与 schema + 归一化器配置。

```javascript
export const meta = {
  name: 'cp1-search-opt-round2',
  description: 'Round 2: measure on the REAL libero_10 library — settle fp16 equivalence, confirm CPU <5ms, deepen GPU, design integration',
  phases: [
    { title: 'Measure', detail: '5 experts run real parity/latency/GPU sweeps on the compact task_pack + audit the code' },
    { title: 'Synthesize', detail: 'consolidate measurements, finalize CPU design + GPU decision + fp16 verdict, set R3 questions' },
  ],
}

const GROUND = `
ROUND 2 GROUNDING — this is a MEASUREMENT round. The Round 1 design converged; R2 must turn disputes into hard numbers measured on the REAL libero_10 library data.

TARGET: cp1_search segment of the OpenPI cache CP1 path, currently ~37 ms median (~95% of CP1 total). GOAL <10 ms, ideally <5 ms. Two directions: (1) stay on CPU, (2) move to GPU.

ROUND 1 VERDICT (converged):
- The 37 ms is OVERHEAD, not arithmetic. Per query, in_memory_backend.py _compute_field_scores @386-387 rebuilds an [N,32768] matrix via torch.stack(vecs).float() from N scattered per-entry tensors, then @392 F.cosine_similarity recomputes operand norms + allocates an [N,D] intermediate. _filter_entries @340 cuts candidates to the per-task slice (N=166..399) BEFORE scoring, so the real working set is 34-69 MB, not the full 692 MB.
- WINNING CPU DESIGN (5/6 experts, micro-bench 2-8 ms): at load_artifact build a contiguous, L2-PRE-NORMALIZED per-task matrix per vision field; per query do ONE torch.mv (cosine = dot of unit vectors), then the existing zscore->tanh normalizer + weighted sum + topk(1). EXACT in fp32 (dot == cosine to ~1e-7; ZScoreNormalizer 0.5*(tanh((x-mu)/sigma)+1), fixed mu/sigma, strictly monotone -> winner + absolute top score preserved).
- GPU resident GEMV: idle 0.3-1.9 ms on the local RTX 3060, BUT measured 48 ms under concurrent policy-model load (34x regression). No-op on CPU-only hosts.
- DROPPED: int8 (stock torch 10-40x slower), faiss/ANN (N too small), PCA/random-proj (query projection costs more than exact GEMV), CUDA graphs (zero benefit).
- UNRESOLVED DISPUTE R2 MUST SETTLE: fp16/bf16 equivalence. Random-vector benches showed ~6e-6 error (safe); real near-1 cosine band showed ~1.3e-4-3.4e-4 (can flip winner/verdict). The threshold judge compares the absolute top FUSED score to a fixed 0.997697.

CRITICAL NEW DATA (leave-one-out probe on the real largest task, vision_0, N=399): top1 cosine min 0.99702 / median 0.99932 / max 0.99998; top1-to-top2 gap min 5.36e-7 / median 1.85e-4. So the winner margin can be as small as 5e-7 — fp16 error (~1e-4) at this band is ~200x larger than the tightest margin. This must be quantified across ALL tasks.

COMPACT DATA PACK (already built — do NOT reload the 1.1 GB pickle; load this instead):
- exp/cache_latency_bench/data/opt_bench/task_pack.pt  (torch.load -> dict: task_key -> {"vision_0": Tensor[N,32768] float32 RAW (un-normalized), "vision_1": Tensor[N,32768] float32, "robot_state": Tensor[N,32] float32, "ids": list[str] len N}). 10 tasks, N=166..399.
- exp/cache_latency_bench/data/opt_bench/meta.json  (per-task N + the loo probe).

FUSION / NORMALIZER CONFIG (from the studied yaml exp/cache_latency_bench/config/depth_study/depth_1.yaml):
- active fields + fusion weights: vision_0=0.25, vision_1=0.4375, robot_state=0.3125 (these are the cache config 'keys[].weight'; confirm in code how they map to QuerySpec.fusion_weights).
- field_similarity: vision_0/vision_1 = cosine; robot_state = l2 with to_similarity {type: exp, tau: 1.0}.
- score_normalization: per_field zscore->tanh. vision_0: mu=0.9739899923664463 sigma=0.0061831533438692935; vision_1: mu=0.9659078322399228 sigma=0.006527797454113087; robot_state: mu=-1.9584325681212513 sigma=0.7484941685797242; all squash=tanh.
- For an EXACT fused score, reuse the real normalizer: import build_field_normalizers / ScoreNormalizer from the cache code (grep 'build_field_normalizers' in src/openpi/cache) rather than re-deriving tanh by hand; verify your replication matches on a few samples.
- threshold judge: hit iff top fused normalized score >= 0.997697 (verify the exact comparison in src/openpi/cache/components/judge.py).

ENV: this is a WSL2 host with torch 2.7.1+cu126 and an RTX 3060 laptop GPU (all R1 GPU numbers were measured here). You MAY run real benchmarks. Write throwaway scripts under exp/cache_latency_bench/data/opt_bench/ or /tmp. Do NOT edit src/.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['headline', 'measurements', 'conclusions', 'code_refs', 'risks', 'recommendation'],
  properties: {
    headline: { type: 'string', description: 'one-paragraph bottom line from your measurements' },
    measurements: {
      type: 'array',
      description: 'concrete numbers you actually measured or computed',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'value', 'notes'],
        properties: {
          name: { type: 'string' },
          value: { type: 'string', description: 'the measured value with unit' },
          notes: { type: 'string', description: 'conditions: N, threads, dtype, host, etc.' },
        },
      },
    },
    conclusions: { type: 'array', items: { type: 'string' } },
    code_refs: { type: 'array', items: { type: 'string' }, description: 'file:line anchors you verified' },
    risks: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string' },
  },
}

const TASKS = [
  { label: 'equiv-sweep', brief: `EQUIVALENCE ADJUDICATION (highest priority). Load task_pack.pt. For EVERY task, run leave-one-out queries: take each row i as the query, candidates = all OTHER rows of that task (exclude self). For each query compute the per-field raw similarity FOUR ways for the vision fields: (a) F.cosine_similarity fp32 [REFERENCE = current behavior], (b) pre-normalized-dot fp32, (c) pre-normalized-dot fp16, (d) pre-normalized-dot bf16. robot_state stays fp32 l2->exp in ALL variants. Then build the EXACT fused normalized score using the real ScoreNormalizer (import from src) + the depth_1.yaml weights, and apply the 0.997697 threshold. REPORT per variant vs (a): argmax-winner flips, hit/miss VERDICT flips, raw per-field cosine error (max/mean) AT THE near-1 band, and the distribution of |top_fused_score - 0.997697| with the MIN gap. Deliver a clear verdict on fp32/fp16/bf16.` },
  { label: 'latency-bench', brief: `CPU LATENCY + ROOFLINE + THREAD + TAIL. Load task_pack.pt. Build a standalone prototype of the winning design: per-task L2-pre-normalized matrices, per-query scorer = normalize(q) -> torch.mv(M, qn) per field -> zscore->tanh -> weighted sum -> topk(1), PREALLOCATED reused buffer. Measure median/p90/p99 over >=300 trials at N=166/264/399 fp32. ALSO measure the CURRENT path (torch.stack+F.cosine_similarity) on the SAME data for the real speedup ratio. Sweep torch threads {1,2,4,5,8,all}, report the knee. Report fp16/bf16 latency delta. Confirm <5 ms median / <10 ms p99 on THIS host + the bandwidth roofline.` },
  { label: 'code-audit', brief: `CODE AUDIT (read source). (Q3) Are spec.query_keys['vision_0'] and ['vision_1'] the SAME vector or DISTINCT? (decides fuse-into-one-GEMV validity). (Q4) Is spec.filters.step_range EVER active -> can candidates be a STRICT SUBSET of a task bucket (forcing index_select gather)? (Q7) Can per-entry query_keys['vision_*'] be FREED after prebuild? Grep ALL readers (verdict factors, LibraryStats, _walk_chain, insert, factor judges). (Q-memo) Does the prebuilt matrix break _score_memo cross-step reuse in _compute_level_scores @774? Keep/bypass/replace?` },
  { label: 'gpu-deepen', brief: `GPU FINALIST + CONTENTION (local RTX 3060). Move per-task pre-normalized vision matrices to cuda. Measure idle per-query GPU latency (H2D 128KB + 2-field GEMV + D2H + topk) at N=166/264/399 fp32. THEN reproduce contention: run the search while a heavy background [4096,4096] matmul saturates the GPU on another stream; report latency under load. Model A100 by bandwidth scaling. Decide the PRECISE condition GPU is warranted vs CPU + the device-flag/CPU-fallback shape. Keep fp32.` },
  { label: 'integration-critic', brief: `INTEGRATION DESIGN + ADVERSARIAL CRITIC. Write the file:line integration design against in_memory_backend.py: (1) load_artifact @200-261 prebuild, (2) _compute_field_scores @363-403 fast path (view vs gather), (3) per-task matrix keyed off task_key, (4) _score_memo interaction, (5) free per-entry tensors or not. THEN adversarially critique R1's 2-8 ms: scrutinize _filter_entries @340 linear scan over 2640 (runs every query), the Python per-candidate loop @448, the trajectory path @638 building normalizers per call @678 + _walk_chain, topk jitter, gather fallback. List every place real integration adds latency the prototype omitted + how to neutralize.` },
]

phase('Measure')
const findings = await parallel(
  TASKS.map((t) => () =>
    agent(
      `${GROUND}\n\nYOUR TASK (${t.label}):\n${t.brief}\n\nDo the real work (run scripts / read code), then return your findings via the schema. Ground every number in an actual measurement or a code line; flag anything you could not measure.`,
      { label: t.label, phase: 'Measure', schema: FINDINGS_SCHEMA }
    )
  )
)
const valid = findings.filter(Boolean)
log(`Round 2 measure: ${valid.length}/${TASKS.length} experts returned`)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'fp16_bf16_verdict', 'cpu_design', 'cpu_confirmed_latency', 'gpu_decision', 'integration_risks', 'open_questions_for_r3'],
  properties: {
    verdict: { type: 'string' },
    fp16_bf16_verdict: { type: 'string' },
    cpu_design: { type: 'string' },
    cpu_confirmed_latency: { type: 'string' },
    gpu_decision: { type: 'string' },
    integration_risks: { type: 'array', items: { type: 'string' } },
    open_questions_for_r3: { type: 'array', items: { type: 'string' } },
  },
}
const synthesis = await agent(
  `${GROUND}\n\nYou are the Round 2 synthesis lead. Below are ${valid.length} expert findings (JSON) with REAL measurements. Consolidate: settle fp16/bf16 with measured flip counts, state the confirmed CPU design + measured latency, give the GPU decision criterion with idle-vs-contention numbers, list real-call-path integration risks + mitigations, define what R3 must close.\n\nEXPERT FINDINGS:\n${JSON.stringify(valid, null, 2)}`,
  { label: 'synth-r2', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { findings: valid, synthesis }
```

---

## 4. 三个值得带走的设计要点

1. **共享地基防幻觉**：先自己 `Read`/`grep`/探针把 file:line、库 N、维度、约束核实，写成一个 `FACTS`/`GROUND`
   常量喂给所有 agent。多 agent 最大的风险是各自幻觉，统一 ground truth 是最便宜的护栏。

2. **`parallel` 栅栏 vs `pipeline` 流水线**：这里两阶段有真依赖（综合必须见全部），栅栏对。若是「每个发现→各自验证」
   这种 item 独立穿 stage 的形态，应该用 `pipeline`，省掉栅栏空等的 wall-clock。

3. **昂贵的一次性准备放到编排外**：把「加载 1.1G 库」从 workflow 里拎出来，先生成紧凑 pack，
   让并行 agent 只消费产物——否则 N 个 agent 同时 reload 会 OOM。这是 hybrid（先内联侦察/准备，再扇出）的典型。

> 真正跑过的持久化原件：
> `~/.claude/projects/-home-weiland-projects-openpi/<session>/workflows/scripts/cp1-search-opt-round{1,2}-*.js`
