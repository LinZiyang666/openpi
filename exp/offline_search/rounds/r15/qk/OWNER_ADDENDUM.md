# Owner addendum to R15 track B (2026-10-05 ~10:1x CDT)

Owner: "你新方法的探索可以先离线在我们profile上的数据上离线探索，不要一来就线上跑" — explore the new method offline first, on the data we have already profiled/recorded; do not go to online (closed-loop) runs straight away.

For astra:
- Offline evaluation data, in this order: (1) library pseudo-queries (each library row queried against the other episodes of its task, leave-one-episode-out, as the deployed metric's confidence scales already do); (2) existing B-dev recordings with keys / images / policy actions (R12 `r12_devlooks_*`, R13 `r13_devrec_*`, their decision logs; `closed_loop/look_recording.py`). `offline_search_store/queries/*` is test A and stays forbidden.
- The closed-loop pilot is only a plan in SPEC/GATES. The coordinator runs nothing online until the offline results clear the offline gates and the coordinator approves.
- Owner data rule unchanged: each cell's model is trained only on that cell's library.

## Second owner directive (2026-10-05 ~10:3x CDT)
Owner: "可以用gpu但是能不用就不用，最好cpu包办一切" — the GPU may be used, but avoid it whenever possible; ideally the CPU does everything (training, evaluation, latency measurement). Use the GPU only where a CPU path is infeasible, and record why. The earlier 4 GB cap still applies to any GPU use.

## Third owner directive (2026-10-05 ~11:1x CDT)
Owner: "只用robotstate的直接否掉，不能当瞎子" — a key that uses only robot state is rejected outright; retrieval must not be blind. Every candidate must be genuinely image-driven: robot state may be part of the key, but the images must carry real retrieval information, shown by a blindness audit.

## Fourth owner directive (2026-10-05 ~11:2x CDT)
Owner: "它可以使用当步可以获得的所有信息，这些信息我们线上运行的时候也要可以获得，可以上online测试" — the model may use every piece of information available at that step, provided the same information is available when running online (current images of both cameras, robot state, previous looks' images and features, previously executed actions, the last policy output, step index, task id / instruction, HIT/MISS history, ...). Online (closed-loop) testing is allowed: the offline gates no longer block a B-dev closed-loop pilot of the best image-driven candidate (the blindness audit and the training-data rule still apply; test A stays measure-only).
