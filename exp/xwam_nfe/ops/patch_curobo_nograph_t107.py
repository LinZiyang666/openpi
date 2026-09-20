"""timan107 (GTX 1080, 8 GB): build RoboTwin's four MotionGen instances without CUDA graphs (use_cuda_graph=False) -- an
execution-only optimisation whose buffers do not fit next to the RT scene on an 8 GB card; planning results are unchanged."""
import sys
p = sys.argv[1]; s = open(p).read()
if "use_cuda_graph=False" in s:
    print("already patched"); sys.exit(0)
old = "                interpolation_dt=1 / 250,\n                num_trajopt_seeds=1,\n"
n = s.count(old); assert n == 2, n
s = s.replace(old, old + "                use_cuda_graph=False,  # 8 GB Pascal: no CUDA-graph buffers (results unchanged)\n")
open(p, "w").write(s); print("patched", p, "x2")
