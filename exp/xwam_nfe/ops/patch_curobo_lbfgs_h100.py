"""h100 (sm_90): curobo's fused lbfgs_step CUDA kernel hits 'illegal instruction'; force the torch LBFGS path (same algorithm)."""
import sys
p = sys.argv[1]; s = open(p).read()
old = "        NewtonOptBase.__init__(self)\n"
new = old + "        self.use_cuda_kernel = False  # H100 (sm_90): fused lbfgs_step kernel -> 'illegal instruction'; torch path instead\n"
if "H100 (sm_90)" in s:
    print("already patched")
else:
    assert old in s
    open(p, "w").write(s.replace(old, new, 1)); print("patched", p)
