"""Generate tables only from successful final fresh-process artifacts."""
from common import *
import collections

ledger=json.loads((OUT/'final_commands.json').read_text())
final_code_time=float((OUT/'FINAL_CODE_TIME.txt').read_text())
rows=[]
for ent in ledger:
 if ent['exit'] or ent['started']<final_code_time:continue
 j=ent['job'];r=ent['run']
 path=OUT/(f"bench_{j['cell']}_{j['precision']}_r{r}.json" if j['kind']=='bench' else f"stage_{j['cell']}_r{r}.json")
 rows.append(json.loads(path.read_text())|dict(kind=j['kind'],rep=j['rep'],artifact=path.name))
assert len(rows)==48,(len(rows),'successful final jobs, expected 48')
text=[]
def line(x=''):text.append(x)
def table(headers,records):
 escape=lambda x:str(x).replace('|',r'\|')
 line()
 line('| '+' | '.join(escape(x) for x in headers)+' |');line('| '+' | '.join(['---']*len(headers))+' |')
 for rec in records:line('| '+' | '.join(escape(x) for x in rec)+' |')
 line()
def fmt(x):return f'{x:.3f}'
def sci(x):return f'{x:.3g}'
ids=[c['id'] for c in configs()]
def name(cid):return cid.removeprefix('pi05_').replace('_',' / ')
def group(kind,cid,precision='float32'):
 return sorted([r for r in rows if r['kind']==kind and r['config']['id']==cid and (kind=='stage' or r['precision']==precision)],key=lambda x:x['rep'])
def rng(seq):return f'{min(seq):.1f}–{max(seq):.1f}'
def pair(vals):return ' → '.join(fmt(v) for v in vals)
def times(rr,key,domain='wall'):
 return [r['latency'][key][domain]['p50'] for r in rr]
def ms_cell(rr,key):
 ev=times(rr,key,'event');wa=times(rr,key,'wall')
 return pair(ev)+' / '+pair(wa)+'; min '+fmt(min(r['latency'][key]['event']['min'] for r in rr))+'/'+fmt(min(r['latency'][key]['wall']['min'] for r in rr))

line('**K9 parity: real stored queries, batch 1, two fresh processes per cell and precision.** Counts below are **per process**; repeated inputs are not counted as new independent queries. R1/R2 agreement counts are shown explicitly. Strict diagnostic gates: action atol 1e-4; other real-valued fields atol 1e-3, except `motion`, `vself` and `vis` atol 1e-5. Discrete fields require equality. The top-1 target is ≥99.9%. These are empirical replay checks, not universal error bounds.')
line()
for precision in ('float32','float64'):
 line(f'**{precision}:**')
 tab=[]
 for cid in ids:
  rr=group('bench',cid,precision);p=[r['parity'] for r in rr];n=p[0]['n'];assert len(rr)==2 and p[1]['n']==n
  agree=lambda key:'/'.join(str(v['agreement'][key]) for v in p)
  tab.append([name(cid),n,agree('top1_agree'),agree('top16_set_agree'),agree('top16_order_agree'),
              sci(max(v['max_abs']['action'] for v in p)),sci(max(v['max_abs']['confidence'] for v in p)),
              '/'.join(str(v['agreement'].get('action_over_tol',0)) for v in p)])
 table(['Suite / episodes / method','N','Top-1 agree R1/R2','Top-16 set R1/R2','Top-16 order R1/R2','Max |Δchunk|','Max |Δconfidence|','Chunk >1e-4 R1/R2'],tab)

line('**Float32 maximum absolute differences by output, across both final processes.** AWM confidence is also independently audited for each MixedJudge base. The complete per-field distributions, worst UID/step and every strict mismatch are in the raw JSON. Integer ID differences are reported as mismatches, not excused by a numeric tolerance.')
line()
tab=[]
for cid in ids:
 rr=group('bench',cid);fields=['d1','disp5','dst','w_eff','scores','stuck_n','vself','disp','vis','zsum']
 values=[max(r['parity']['max_abs'].get(k,0.) for r in rr) if any(k in r['parity']['max_abs'] for r in rr) else None for k in fields]
 tab.append([name(cid)]+['—' if v is None else sci(v) for v in values])
table(['Cell']+fields,tab)
line('**Remaining audited float32 scalar/weight differences (maximum across both processes).** Dashes mean the deployed result does not expose that field for this method; additional implemented intermediate values are not claimed as independently audited extras.')
line()
fields=['awm_confidence','weights','d1_rel','c0','lag','pred_err','still','vote','lib_ep','lib_step','term1','top1_prog']
tab=[]
for cid in ids:
 rr=group('bench',cid)
 values=[max(r['parity']['max_abs'].get(k,0.) for r in rr) if any(k in r['parity']['max_abs'] for r in rr) else None for k in fields]
 tab.append([name(cid)]+['—' if v is None else sci(v) for v in values])
table(['Cell']+fields,tab)
line('**Float32 action error by regime (maximum over both processes).** Regimes 0/1/2 mean step 0 / fresh after MISS / stale after HIT or unknown, respectively.')
line()
table(['Cell','N per process 0/1/2','Δchunk step 0','Δchunk fresh','Δchunk stale'],[
 [name(cid),'/'.join(str(group('bench',cid)[0]['parity']['regimes'].get(str(i),0)) for i in (0,1,2))]+
 [sci(max(r['parity']['max_abs_by_regime'][str(i)].get('action',0) for r in group('bench',cid))) for i in (0,1,2)] for cid in ids])
noninitial=max(r['parity']['max_abs_by_regime'][str(i)]['action'] for cid in ids for r in group('bench',cid) for i in (1,2))
line(f'Every tested float32 non-step-0 chunk satisfies atol 1e-4 (maximum {noninitial:.9g}); the chunk failures are at step 0. This does not imply that all other outputs satisfy their own strict gates.')
line()

line('**Retrieval alone, float32, milliseconds per call under concurrent load.** Each eager and captured path has 10 warmups + 80 timed calls/process. Resident inputs; transfers, fit/loading and capture setup excluded. Every entry is event p50 R1→R2 / synchronized wall p50 R1→R2. Batch 8 is total batch latency, not per-query latency.')
line()
tab=[]
for cid in ids:
 rr=group('bench',cid)
 cell=lambda batch,mode:pair([r['latency'][str(batch)][mode]['event']['p50'] for r in rr])+' / '+pair([r['latency'][str(batch)][mode]['wall']['p50'] for r in rr])+'; min '+fmt(min(r['latency'][str(batch)][mode]['event']['min'] for r in rr))+'/'+fmt(min(r['latency'][str(batch)][mode]['wall']['min'] for r in rr))
 mins=lambda batch:sci(min(r['latency'][str(batch)]['graph']['event']['min'] for r in rr))+' / '+sci(min(r['latency'][str(batch)]['graph']['wall']['min'] for r in rr))
 tab.append([name(cid),cell(1,'eager'),cell(1,'graph'),mins(1),cell(8,'eager'),cell(8,'graph'),mins(8)])
table(['Cell','B1 eager event / wall','B1 graph event / wall','B1 graph min event / wall','B8 eager event / wall','B8 graph event / wall','B8 graph min event / wall'],tab)
line('**Float64 retrieval graph tradeoff, milliseconds per call.**')
line()
table(['Cell','B1 graph event p50 R1→R2','B1 graph wall p50 R1→R2','B1 minimum event / wall','Resident MiB f32 / f64'],[
 [name(cid),pair([r['latency']['1']['graph']['event']['p50'] for r in group('bench',cid,'float64')]),
  pair([r['latency']['1']['graph']['wall']['p50'] for r in group('bench',cid,'float64')]),
  fmt(min(r['latency']['1']['graph']['event']['min'] for r in group('bench',cid,'float64')))+' / '+fmt(min(r['latency']['1']['graph']['wall']['min'] for r in group('bench',cid,'float64'))),
  fmt(group('bench',cid)[0]['resident_bytes']/2**20)+' / '+fmt(group('bench',cid,'float64')[0]['resident_bytes']/2**20)] for cid in ids])

line('**Real π0.5 stage 1 → deployed 4×4 pooling → retrieval/synthesis, milliseconds, batch 1, step-0 observation.** Every entry is event p50 R1→R2 / wall p50 R1→R2; 5 warmups + 40 timed calls/process. CPU path ends with the host chunk and verdict data; GPU path includes pinned-host copies of only the final chunk and verdict packet. CUDA-event spans around the CPU path include host gaps: they are not a pure kernel sum. Input transforms and common observation H2D are excluded from both paths.')
line()
table(['Cell','Stock eager S1 + CPU','All eager GPU','Graph S1 + CPU','Combined graph + final D2H','Combined min event / wall'],[
 [name(cid),ms_cell(group('stage',cid),'today_eager_stage_cpu'),ms_cell(group('stage',cid),'gpu_all_eager'),
  ms_cell(group('stage',cid),'today_graph_stage_cpu'),ms_cell(group('stage',cid),'gpu_combined_graph'),
  fmt(min(r['latency']['gpu_combined_graph']['event']['min'] for r in group('stage',cid)))+' / '+fmt(min(r['latency']['gpu_combined_graph']['wall']['min'] for r in group('stage',cid)))] for cid in ids])
line('**Stage-1 graph and increment.** Combined paired samples exclude final D2H in both arms and alternate execution order. Positive/negative samples are retained: load fluctuations can exceed the retrieval increment. These are manual CUDA graphs of the current kernels, not the owner’s historical compiled 10.26-ms stage-1 cost basis.')
line()
table(['Cell','S1 eager event / wall','S1 graph event / wall','Separate graphs + D2H event / wall','Paired Δ event p50 R1→R2','Paired Δ wall p50 R1→R2','Paired Δ event mean R1→R2','Paired Δ event min / p90 (pooled)'],[
 [name(cid),ms_cell(group('stage',cid),'stage1_eager'),ms_cell(group('stage',cid),'stage1_graph'),ms_cell(group('stage',cid),'gpu_separate_graphs'),
  pair([r['paired_no_d2h']['event_increment']['p50'] for r in group('stage',cid)]),
  pair([r['paired_no_d2h']['wall_increment']['p50'] for r in group('stage',cid)]),
  pair([r['paired_no_d2h']['event_increment']['mean'] for r in group('stage',cid)]),
  fmt(min(s['combined']['event']-s['stage']['event'] for r in group('stage',cid) for s in r['paired_no_d2h']['samples']))+' / '+fmt(float(np.percentile([s['combined']['event']-s['stage']['event'] for r in group('stage',cid) for s in r['paired_no_d2h']['samples']],90)))] for cid in ids])

line('**PCIe bytes per decision at the measured retrieval boundary.** Common observation upload is listed separately. No raw tokens or pooled camera keys cross D2H in the GPU design; no native shadow/logging is included.')
line()
table(['Cell','Actual pooled dtype','Current keys D2H','Host f32 key bytes','GPU chunk + verdict D2H','Common observation H2D','Fixed candidate capacity/task'],[
 [name(cid),group('stage',cid)[0]['pool_dtype'],group('stage',cid)[0]['bytes']['current_keys_d2h'],group('stage',cid)[0]['bytes']['host_keys_materialized'],
  group('stage',cid)[0]['bytes']['gpu_output_d2h'],group('stage',cid)[0]['bytes']['image_state_prompt_h2d_common'],group('bench',cid)[0]['capacity']] for cid in ids])
line('The current D2H total is 131,072 bytes of bf16 cameras plus 256 bytes of observed float64 stage state; the builder materializes float32 keys/state on CPU. A float32-on-device pooled-key design with float32 state would transfer 262,272 bytes. New per-decision control inputs are three int64 values (task/step/previous-HIT: 24 H2D bytes) if updated from host; histories/actions should remain resident. This benchmark preloads those controls/history and does not time their upload. The current interceptor can additionally upload the synthesized 1,280-byte CPU chunk beyond the measured boundary; that and eventual wire/output transforms are not charged here.')
line()
line('**Observed load and memory.** Utilization is device-wide, including K9 during measurement; it cannot isolate another process’s utilization. Timings are under load, and all quoted minima are observed loaded-machine minima.')
line()
tab=[]
for cid in ids:
 rr=group('bench',cid)+group('stage',cid)
 samples=[]
 for r in rr:
  if r['kind']=='bench':samples.extend(s for b in r['latency'].values() for v in b.values() for s in v['gpu_samples'])
  else:samples.extend(s for v in r['latency'].values() for s in v['gpu_samples'])
 tab.append([name(cid),rng([r['gpu']['admission']['util_pct'] for r in rr]),rng([s['util_pct'] for s in samples]),
             fmt(min(s['free_mib'] for s in samples)/1024),fmt(max(s.get('own_mib',0) for r in rr for s in r['gpu']['samples'])/1024),
             fmt(max(r['gpu']['peak_reserved_mib'] for r in rr)/1024)])
table(['Cell','Before K9 CUDA util %','During timing util %','Min free GiB during timing','Max observed own GiB','Peak Torch reserved GiB'],tab)
table(['Cell','Read-only fitted artifact'],[[name(c['id']),c['path']] for c in configs()])

line('**Design note.**')
line();line((OUT/'DESIGN.md').read_text())
line('**Numerical findings and caveats.**')
line()
# Exact mismatch catalog, including every field that exceeds its stated strict gate.
catalog=[]
for precision in ('float32','float64'):
 for cid in ids:
  rr=group('bench',cid,precision)
  for r in rr:
   p=r['parity'];m=p['mismatches']
   catalog.append(dict(cell=cid,precision=precision,rep=r['rep'],n=p['n'],agreement=p['agreement'],
                       rank_mismatches=[x for x in m if not x['order']],max_abs=p['max_abs'],max_abs_by_regime=p['max_abs_by_regime']))
dump(OUT/'mismatch_catalog.json',catalog)
for precision in ('float32','float64'):
 rr=[r for r in rows if r['kind']=='bench' and r['precision']==precision]
 rate=min(r['parity']['agreement']['top1_agree']/r['parity']['n'] for r in rr)
 line(f"{precision}: lowest cell top-1 agreement {rate*100:.6f}%; largest full-chunk absolute difference {max(r['parity']['max_abs']['action'] for r in rr):.9g}. The strict 1e-4 chunk gate fails. This prototype is not a demonstrated strict numerical replacement for the CPU path. The supplied float64 variant is a diagnostic option, not a promise of reproducing CPU float32 rounding.")
line()
line('The mismatch classes are: near ties changing top-1 identity and its episode metadata; order changes, including rank-5 boundary swaps within an unchanged top-16 set; exact/near rank-16 boundary ties changing membership; continuous distance/kernel-weight/synthesis drift with identical selected rows; confidence/extras drift propagated through dispersion and calibration; and large absolute transformed-score drift for strongly downweighted members. All eight selected fits are loaded without refitting. `mismatch_catalog.json` groups every observed discrete ranking mismatch and every strict per-field count; each raw benchmark JSON gives every offending UID/step, both 16-row selections, deltas, CPU top-1 and rank-16 gaps, and CPU distances at the GPU-selected rows. The action maximum is over all 320 chunk values, not only the executed 35.')
line()
line('The CPU query uses float32 PCA GEMV, whitening and expanded squared distances, then float64 score/kernel calculations and float32 synthesis. GPU reductions/FMA and GEMM/GEMV execution order differ even with TF32 disabled. Early distances subtract large, nearly equal terms using fitted float32 norm arrays; resulting ties or small gaps can change order/membership. Float64 runtime math still begins with those quantized fitted buffers and need not agree more closely with deployed float32 decisions. Stable sorting only resolves exact ties in the computed GPU distances; it cannot recover a tie created by CPU rounding. Full-chunk synthesis amplifies a rank-16 swap when the displaced row retains non-negligible kernel weight. These mechanisms follow the code and the logged gap/delta evidence; no success-rate effect is measured.')
line()
probes=json.loads((OUT/'numerical_probe.json').read_text())
ratios=[z['cancellation_ratio'] for p in probes for z in p['selection']]
line(f"A separate CPU-only diagnosis (`numerical_probe.py`, `numerical_probe.json`) replays {len(probes)} real step-0 cases. Across their first 18 neighbors, the sum of absolute expanded-distance terms divided by the final squared distance ranges {min(ratios):.1f}–{max(ratios):.1f}. It records PCA f32/f64 differences, each norm/cross term, and exact CPU boundary ties. This diagnoses cancellation; the f64 recomputation is not ground truth for the deployed f32 method.")
line()
line('Coverage: π0.5 spatial/l10 × 50/500 × pure CL2 AWM/guard-only MixedJudge, both float32 and float64 retrieval; 40 real recorded episodes per suite (first two episodes per task in cache and inference arms, up to 64 decisions/episode), preserving executed histories. Spatial has 1,037 and l10 2,200 query calls per process. This bounded replay limits shared-GPU occupation; remaining store episodes and later decisions were not tested. Identical episode-start observations can appear in both arms; repeats and duplicates do not increase independent coverage. Batch-8 latency repeats one stored stale query and records its comparison to batch 1; the full selected-query parity sweep is batch 1. No synthetic rollout was substituted for the real query inputs.')
line()
line('Stage 1 uses the same checkpoint and mixed bf16/fp32 weights as K3, with unused stages 2/3 placed on meta. It loads an exact CPU stage-1-only scratch cache after its first normal checkpoint load; no serving weights/configuration change. Direct stock manual capture failed at `embed_prefix`’s Python-list `torch.tensor` construction (retained in `stage_dev1.log`). The isolated adapter replaces that all-zero attention-mask allocation with `zeros_like`; it runs all three image towers. The final raw stage files test all five stage-output fields on six changed real observations across tasks 0/4/9, and compare graph/eager retrieval/chunk/verdict packet outputs. Changing a test image at a later recorded step while holding the timing regime at step 0 tests graph/input reuse, not a complete chronological episode. The existing stage compile/graph options and K3 adapter were inspected; manual graph capture supplies the requested combined graph, so no additional torch.compile experiment was run and the historical compiled cost basis was not reproduced. No stages 2/3 inference, online guard/controller, R4 blind GPU implementation, K7 GPU implementation, GR00T, simulator, SR, RPC, logging, native-shadow retrieval or serving integration was performed. GR00T was optional; the other omitted items are outside this retrieval/feature experiment or explicitly design-only deliverables.')
line()
line('The GPU timing endpoint is the host chunk plus confidence/guard inputs. Final host guard/progress updates, controller thresholding and the HIT/MISS branch are excluded. The deployed CPU method call computes its own guards as part of the baseline. These timings therefore compare the requested stage/retrieval/synthesis boundary; they do not establish latency of a complete GPU serving replacement. All eager paths retain Python launch overhead. The separately captured path also packs its verdict outside the retrieval graph, whereas the combined graph includes that packing.')
line()
# Validate and report replay/repeat evidence from facts.
stage=[r for r in rows if r['kind']=='stage'];checks=[x for r in stage for x in r['parity'] if x['kind']=='changed_image_task_replay']
line(f"Final stage adapter checks: {sum(c['all_stage_fields_exact'] for c in checks)}/{len(checks)} all-field bitwise matches; maximum graph/eager chunk difference {max(c['graph_eager_action_max_abs'] for c in checks):.9g}, packet difference {max(c['graph_eager_packet_max_abs'] for c in checks):.9g}. Same-eager-stage CPU/GPU chunk error is separately retained for every stage run. All {len([r for r in rows if r['kind']=='bench'])} final append/address checks completed successfully.")
line()
variations=[]
for cid in ids:
 for precision in ('float32','float64'):
  rr=group('bench',cid,precision)
  for batch in ('1','8'):
   for mode in ('eager','graph'):
    vals=[r['latency'][batch][mode]['wall']['p50'] for r in rr]
    variations.append(dict(cell=cid,precision=precision,batch=int(batch),mode=mode,p50_ms=vals,variation_pct=(max(vals)/min(vals)-1)*100))
dump(OUT/'variation.json',variations)
line(f"Module between-process wall-median variation ranges {min(v['variation_pct'] for v in variations):.2f}%–{max(v['variation_pct'] for v in variations):.2f}% (`variation.json`). Paired stage increments and all raw event/wall samples remain available; two fresh processes do not establish quiet-machine latency or multi-client throughput. The historical owner s1/s2/s3=10.26/27.69/29.57-ms and full=67.5-ms constants are reference context only; no IR or SR projection is claimed from this shared-load manual-graph experiment.")
line()
line('Every Python command is pinned to CPUs 26–29,70–73, with OMP/OpenBLAS/MKL threads=1 and bytecode writes disabled. `GPUWatch` invokes nvidia-smi before GPU phases, admits at ≥12 GiB free, polls free/own memory about every 0.2 s, exits its own process below 4 GiB or above 8 GiB own memory, and caps the Torch allocator at 7 GiB to leave context/library allowance. The serial scheduler waits for additional headroom, preserves rejected attempts, and each measurement process exits to release its GPU state. GPU timings include other active server workloads; MPS and other process PIDs are recorded in admission snapshots. Admission failures, including later capture/timing phase checks, are retained in logs; no OOM recovery was attempted. No servers, ports, remote machines, LIBERO workers, git commands, review tests, or running code paths were touched. Writes are confined to this directory and `/tmp/k9_*`. The store fallback is `/home/weiland/trace_runs/offline_search_store`; `/dev/shm/offline_search_store` was absent.')
line()
line('**Exact commands and artifacts.** Working directory `/home/weiland/projects/openpi`; requested `rounds/` is under `exp/offline_search/`. `final_commands.json` contains every expanded taskset/env/Python invocation, attempt, timing, exit code and log path. Successful final jobs alone populate these tables. Earlier dev/smoke artifacts are retained and excluded.')
line()
line('Environment: Python 3.11.15, NumPy 1.26.4, PyTorch 2.7.1+cu126 / CUDA runtime 12.6. nvidia-smi identifies the shared device as NVIDIA GeForce RTX 4090, 49,140 MiB total memory, driver 595.71.05; those are observed device-reported values. `environment.json`, per-attempt admission snapshots and `reference_hashes.txt` retain the environment and reference-source evidence. The final audit checks unchanged retrieval, stage-adapter and measurement-code hashes across all 48 successful processes, ≥99.9% top-1 per cell, graph/input/append checks and memory limits; `audit.json` separately reports the failed strict chunk gate. The CPU references and deployed stage/key-builder sources still match `reference_hashes.txt` at completion (`reference_hashes_final_check.txt`).')
line()
line('```bash\ntaskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/run_jobs.py\ntaskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/audit.py\ntaskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/report.py\n```')
line()
dump(OUT/'final_artifacts.json',[dict(kind=r['kind'],cell=r['config']['id'],precision=r.get('precision','float32'),rep=r['rep'],artifact=r['artifact']) for r in rows])
(OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
print('Wrote REPORT.md from',len(rows),'successful final processes')
