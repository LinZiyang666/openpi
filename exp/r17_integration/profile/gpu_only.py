"""GPU-only stage costs from a batch-1 latency profile's nsys trace (owner 10-24).

The profiler's CUDA-event stages include host work: `prepare` (observation resize/normalize on the CPU,
e.g. 480x480 MetaWorld frames for pi0.5: ~30 of its 38.7 ms "stage1") and host dispatch gaps. The R15/R16
pi0.5/GR00T LIBERO prices (D5) are CUDA-graph times only, so pricing now counts GPU time only on every line:
per measured iteration, the stage's compiled graph(s) plus the non-graph kernels in its window, CPU time on
neither side (as the CPU text-encoder rule already did for DP).

Windows per iteration (single stream, in time order): stage1 graph -> stage2 graph -> k stage3 graphs.
Kernels before the stage2 graph belong to stage1 (prepare kernels included), kernels from the stage2 graph to
the first stage3 graph to stage2, the rest up to the largest idle gap before the next stage1 graph to stage3.

    python -m exp.r17_integration.profile.gpu_only PROFILE.json OUT.json
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path


def stage_times(db):
    c = sqlite3.connect(db)
    graphs = c.execute('select graphId, start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start').fetchall()
    kernels = c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where graphId is null or graphId = 0').fetchall()
    counts = {}
    for g, _, _ in graphs:
        counts[g] = counts.get(g, 0) + 1
    step = max(counts, key=counts.get)  # the denoising step graph launches k times per iteration
    first = {}
    for g, s, _ in graphs:
        first.setdefault(g, s)
    s1, s2 = sorted((g for g in counts if g != step), key=first.get)
    events = sorted([(s, e, g) for g, s, e in graphs] + [(s, e, None) for s, e in kernels])
    starts = [i for i, ev in enumerate(events) if ev[2] == s1]
    def busy(a, b):  # union of the intervals: kernels on other streams can overlap
        total, reach = 0, None
        for s, e, _ in sorted(events[a:b]):
            if reach is None or s > reach:
                total += e - s
                reach = e
            elif e > reach:
                total += e - reach
                reach = e
        return total / 1e6
    rows, lo = [], 0
    for n, i in enumerate(starts):
        nxt = starts[n + 1] if n + 1 < len(starts) else len(events)
        j2 = next(j for j in range(i, nxt) if events[j][2] == s2)
        j3 = next(j for j in range(j2, nxt) if events[j][2] == step)
        last = max(j for j in range(j3, nxt) if events[j][2] == step)
        hi = len(events)
        if n + 1 < len(starts):  # split the stage3 tail from the next iteration's prepare at the largest idle gap
            hi = max((events[j + 1][0] - events[j][1], j + 1) for j in range(last, nxt))[1]
        rows.append(dict(k=sum(1 for j in range(j3, hi) if events[j][2] == step),
                         stage1=busy(lo, j2), stage2=busy(j2, j3), stage3=busy(j3, hi)))
        lo = hi
    return rows


def convert(src, out):
    r = json.loads(Path(src).read_text())
    db = Path(r['cuda_trace_path'].replace('.trace.csv', '.sqlite'))
    if not db.exists():
        db = Path(src).with_suffix('.sqlite')
    rows = stage_times(db)
    # cells by the step launches actually run (Cosmos's nominal 2 reduced steps compute 1)
    full, reduced = max(x['k'] for x in rows), min(x['k'] for x in rows)
    med = lambda k, key: statistics.median(x[key] for x in rows if x['k'] == k)
    before = {key: r.get(key) for key in ('stage1_ms', 'stage2_ms', 'stage3_full_loop_ms', 'stage3_reduced_loop_ms')}
    r.update(stage1_ms=med(full, 'stage1'), stage2_ms=med(full, 'stage2'),
             stage3_full_loop_ms=med(full, 'stage3'), stage3_reduced_loop_ms=med(reduced, 'stage3'))
    r['gpu_only'] = dict(source=str(db), iterations={str(k): sum(1 for x in rows if x['k'] == k) for k in (full, reduced)},
                         harness_ms=before, method=__doc__.split('\n\n')[1].replace('\n', ' '))
    r['timing_scope'] = 'GPU time only: stage graphs + non-graph kernels per stage window (no CPU prepare, no host gaps)'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(r, indent=1) + '\n')
    return before, {k: r[k] for k in before}


if __name__ == '__main__':
    print(json.dumps(convert(sys.argv[1], sys.argv[2])))
