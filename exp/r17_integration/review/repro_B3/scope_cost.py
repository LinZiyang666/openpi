import sys, time
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import torch
from types import SimpleNamespace
from exp.r17_integration.models.fastwam.staged import conditioning_scope
for name, (h, w, s) in {"libero": (224, 448, 8), "metaworld": (224, 224, 4)}.items():
    p = SimpleNamespace(data={"prompt": "A video recorded from a robot's point of view executing the following instruction: pick up the black bowl",
                              "input_image": torch.rand(1, 3, h, w), "proprio": torch.rand(1, s)})
    conditioning_scope(p)
    t = time.perf_counter(); n = 50
    for _ in range(n): conditioning_scope(p)
    print(name, f"{(time.perf_counter()-t)/n*1000:.2f} ms per stage1 call")
