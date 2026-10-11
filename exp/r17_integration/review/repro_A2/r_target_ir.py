"""Round-2 repro: recipe.target_ir -> periodic gap is (1-t)/t, i.e. a MISS fraction, not R16/R11 owner IR."""
from exp.r17_integration.common.recipe import target_ir_to_gap
from exp.r17_integration.common.layers.periodic import periodic
from exp.r17_integration.common.contracts import RequestContext
for t in (.45, .40, .32, .25):
    gap = target_ir_to_gap(t)
    vision, hit, misses = [], [], 0
    N = 160 // 5                                      # MW decisions per full episode (no tails)
    for seq in range(N):
        force, _ = periodic(RequestContext("c", "p", "reach-v3", 3, seq, task_id=41), vision, hit, gap)
        vision.append(True); hit.append(0 if force else 1); misses += force
    m = misses / N
    print(f"target_ir={t:.2f} gap={gap:.3f} MISS/N={m:.3f}  owner IR pi K10 (.152V+.848M)/N={.152 + .848*m:.3f}"
          f"  pi K2 (.152V+.498M)/N={.152 + .498*m:.3f}  (no guard MISS counted)")
